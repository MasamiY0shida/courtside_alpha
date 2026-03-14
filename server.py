"""
server.py — FastAPI Live Inference Server
==========================================
Polls NBA live API, runs predictions, exposes signals.

Start: uvicorn server:app --reload --port 8000
"""

import asyncio
import time
import json
from datetime import datetime
from contextlib import asynccontextmanager

import numpy as np
import xgboost as xgb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from nba_api.live.nba.endpoints import scoreboard as live_scoreboard
from nba_api.live.nba.endpoints import playbyplay as live_pbp
from nba_api.live.nba.endpoints import boxscore as live_boxscore

from features import FeatureEngine, DATA_DIR
from market_data import fetch_polymarket_game_odds


# ══════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════

class ModelSuite:
    """Loads and holds all trained models."""

    def __init__(self):
        print("Loading models...")
        self.win_model = xgb.XGBClassifier()
        self.win_model.load_model(f"{DATA_DIR}/v2_win_probability.json")

        self.margin_model = xgb.XGBRegressor()
        self.margin_model.load_model(f"{DATA_DIR}/v2_margin.json")

        self.proxy_model = xgb.XGBClassifier()
        self.proxy_model.load_model(f"{DATA_DIR}/v2_market_proxy.json")

        self.edge_model = xgb.XGBClassifier()
        self.edge_model.load_model(f"{DATA_DIR}/v2_edge_model.json")

        print("All models loaded.")

    def predict(self, features, feature_engine):
        """Run all models and return predictions."""
        live_arr = feature_engine.to_live_array(features)
        pregame_arr = feature_engine.to_pregame_array(features)

        win_prob = float(self.win_model.predict_proba(live_arr)[0][1])
        proxy_prob = float(self.proxy_model.predict_proba(pregame_arr)[0][1])
        margin_pred = float(self.margin_model.predict(live_arr)[0])

        edge = win_prob - proxy_prob
        abs_edge = abs(edge)

        # Edge model confidence
        features["EDGE"] = edge
        features["ABS_EDGE"] = abs_edge
        features["OOF_PROXY_PROB"] = proxy_prob
        features["OOF_LIVE_PROB"] = win_prob
        edge_arr = feature_engine.to_edge_array(features)
        edge_confidence = float(self.edge_model.predict_proba(edge_arr)[0][1])

        # Kelly sizing
        kelly_fraction = 0.25
        kelly_size = max(0, min(0.2, edge_confidence * 2 - 1)) * kelly_fraction

        return {
            "win_probability": round(win_prob, 4),
            "proxy_probability": round(proxy_prob, 4),
            "predicted_margin": round(margin_pred, 1),
            "edge": round(edge, 4),
            "abs_edge": round(abs_edge, 4),
            "edge_confidence": round(edge_confidence, 4),
            "kelly_size": round(kelly_size, 4),
        }


# ══════════════════════════════════════════════
# LIVE GAME TRACKER
# ══════════════════════════════════════════════

class GameTracker:
    """
    Tracks live game state and play history
    for feature construction.
    """

    def __init__(self):
        self.games = {}  # game_id -> state dict
        self.last_poll = None

    def update_from_scoreboard(self, scoreboard_data):
        """Parse live scoreboard and update tracked games."""
        try:
            games_list = scoreboard_data.get("scoreboard", {}).get("games", [])
        except Exception:
            return []

        active_game_ids = []

        for game in games_list:
            game_id = game.get("gameId", "")
            status = game.get("gameStatus", 0)

            # status: 1=pregame, 2=live, 3=final
            if status != 2:
                continue

            active_game_ids.append(game_id)

            home_team = game.get("homeTeam", {})
            away_team = game.get("awayTeam", {})

            home_score = int(home_team.get("score", 0))
            away_score = int(away_team.get("score", 0))
            period = int(game.get("period", 1))

            # Parse game clock
            clock_str = game.get("gameClock", "PT00M00.00S")
            secs_remaining = self._parse_iso_clock(clock_str)

            # Total game seconds left
            remaining_periods = max(0, 4 - period)
            game_secs_left = remaining_periods * 720 + secs_remaining

            home_id = int(home_team.get("teamId", 0))
            away_id = int(away_team.get("teamId", 0))
            home_tricode = home_team.get("teamTricode", "???")
            away_tricode = away_team.get("teamTricode", "???")

            # Initialize or update game state
            if game_id not in self.games:
                self.games[game_id] = {
                    "home_team_id": home_id,
                    "away_team_id": away_id,
                    "home_tricode": home_tricode,
                    "away_tricode": away_tricode,
                    "play_history": [],
                    "prev_snapshot": {},
                }

            state = self.games[game_id]

            # Save previous for lag features
            state["prev_snapshot"] = {
                "margin": state.get("home_score", 0) - state.get("away_score", 0),
                "scoring_pace": state.get("scoring_pace", 0),
            }

            state["home_score"] = home_score
            state["away_score"] = away_score
            state["period"] = period
            state["game_seconds_left"] = game_secs_left
            state["game_clock"] = clock_str

            # Append to play history
            state["play_history"].append((game_secs_left, home_score, away_score))

            # Compute scoring pace for lag
            elapsed = max(2880 - game_secs_left, 1)
            state["scoring_pace"] = (home_score + away_score) / (elapsed / 60) if elapsed > 60 else 0

        self.last_poll = datetime.now()
        return active_game_ids

    def get_game_state(self, game_id):
        """Return current state for a game."""
        return self.games.get(game_id)

    def _parse_iso_clock(self, clock_str):
        """Parse 'PT05M30.00S' format to seconds."""
        try:
            clock_str = clock_str.replace("PT", "").replace("S", "")
            parts = clock_str.split("M")
            minutes = int(parts[0]) if parts[0] else 0
            seconds = float(parts[1]) if len(parts) > 1 and parts[1] else 0
            return int(minutes * 60 + seconds)
        except Exception:
            return 0


# ══════════════════════════════════════════════
# SIGNAL GENERATOR
# ══════════════════════════════════════════════

class SignalGenerator:
    """Generates trading signals from model predictions."""

    def __init__(self, edge_threshold=0.05, confidence_threshold=0.60):
        self.edge_threshold = edge_threshold
        self.confidence_threshold = confidence_threshold

    def generate(self, predictions, game_state, market_prob=None):
        """
        Produce buy/sell signals from predictions.
        When market_prob is provided, use real market odds for edge;
        otherwise fall back to proxy model edge.
        Returns list of signal dicts.
        """
        signals = []
        confidence = predictions["edge_confidence"]
        kelly = predictions["kelly_size"]

        # Use real market edge when available, else proxy edge
        if market_prob is not None:
            edge = predictions["win_probability"] - market_prob
            abs_edge = abs(edge)
            baseline_label = "market"
            baseline_prob = market_prob
        else:
            edge = predictions["edge"]
            abs_edge = predictions["abs_edge"]
            baseline_label = "proxy"
            baseline_prob = predictions["proxy_probability"]

        # Only signal if edge exceeds threshold AND model is confident
        if abs_edge < self.edge_threshold:
            return signals
        if confidence < self.confidence_threshold:
            return signals

        home = game_state["home_tricode"]
        away = game_state["away_tricode"]

        # Moneyline signal
        if edge > 0:
            signals.append({
                "market": f"moneyline_{home}",
                "direction": "BUY",
                "team": home,
                "edge": round(edge, 4),
                "market_edge": round(edge, 4) if market_prob is not None else None,
                "confidence": round(confidence, 4),
                "kelly_size": round(kelly, 4),
                "reasoning": f"Live model sees {home} at {predictions['win_probability']:.1%} "
                             f"vs {baseline_label} {baseline_prob:.1%}",
            })
        else:
            signals.append({
                "market": f"moneyline_{away}",
                "direction": "BUY",
                "team": away,
                "edge": round(abs_edge, 4),
                "market_edge": round(abs_edge, 4) if market_prob is not None else None,
                "confidence": round(confidence, 4),
                "kelly_size": round(kelly, 4),
                "reasoning": f"Live model sees {away} at {1-predictions['win_probability']:.1%} "
                             f"vs {baseline_label} {1-baseline_prob:.1%}",
            })

        # Spread signal
        margin = predictions["predicted_margin"]
        if abs(margin) > 3:
            spread_team = home if margin > 0 else away
            signals.append({
                "market": f"spread_{spread_team}",
                "direction": "BUY",
                "team": spread_team,
                "predicted_margin": round(margin, 1),
                "confidence": round(confidence, 4),
                "kelly_size": round(kelly * 0.7, 4),
                "reasoning": f"Model predicts {home} by {margin:+.1f} points",
            })

        return signals


# ══════════════════════════════════════════════
# BACKGROUND POLLING TASK
# ══════════════════════════════════════════════

# Global state
feature_engine = None
models = None
tracker = GameTracker()
signal_gen = SignalGenerator()
latest_predictions = {}  # game_id -> full prediction payload
latest_market_odds = {}  # "{TRI}_vs_{TRI}" -> market odds dict
polling_active = False


async def poll_market_odds():
    """Background task: poll Polymarket for real odds every 30 seconds."""
    global latest_market_odds

    print("Market odds polling started...")

    while polling_active:
        try:
            odds = await asyncio.to_thread(fetch_polymarket_game_odds)
            latest_market_odds = odds
            if odds:
                print(f"  [{datetime.now().strftime('%H:%M:%S')}] "
                      f"Fetched {len(odds)} Polymarket game odds")
        except Exception as e:
            print(f"  Market odds poll error: {e}")

        await asyncio.sleep(30)


def _match_market_prob(state):
    """
    Match a live game to its Polymarket odds.
    Returns the market's P(home wins) or None if no match.
    """
    home_id = state["home_team_id"]
    home_tri = state["home_tricode"]
    away_tri = state["away_tricode"]

    # Try both key orderings
    for key in [f"{home_tri}_vs_{away_tri}", f"{away_tri}_vs_{home_tri}"]:
        market = latest_market_odds.get(key)
        if market:
            # Align probability to home team's perspective
            if market.get("home_team_id") == home_id:
                return market["home_win_prob"], market
            else:
                return market["away_win_prob"], market

    return None, None


async def poll_live_games():
    """Background task: poll NBA API every 30 seconds."""
    global latest_predictions

    print("Live polling started...")

    while polling_active:
        try:
            # Fetch live scoreboard
            sb = live_scoreboard.ScoreBoard()
            sb_data = sb.get_dict()
            active_ids = tracker.update_from_scoreboard(sb_data)

            if active_ids:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] "
                      f"{len(active_ids)} live games")

            for game_id in active_ids:
                state = tracker.get_game_state(game_id)
                if not state:
                    continue

                # Build features
                features = feature_engine.build_feature_vector(state)

                # Run models
                predictions = models.predict(features, feature_engine)

                # Match to real market odds
                market_prob, market = _match_market_prob(state)

                # Generate signals using real market odds when available
                signals = signal_gen.generate(predictions, state, market_prob=market_prob)

                # Package everything
                market_edge = (predictions["win_probability"] - market_prob) if market_prob is not None else None
                payload = {
                    "game_id": game_id,
                    "timestamp": datetime.now().isoformat(),
                    "home_team": state["home_tricode"],
                    "away_team": state["away_tricode"],
                    "home_team_id": state["home_team_id"],
                    "away_team_id": state["away_team_id"],
                    "score": {
                        "home": state["home_score"],
                        "away": state["away_score"],
                    },
                    "period": state["period"],
                    "game_clock": state.get("game_clock", ""),
                    "predictions": predictions,
                    "market_odds": {
                        "polymarket_prob": round(market_prob, 4) if market_prob is not None else None,
                        "market_edge": round(market_edge, 4) if market_edge is not None else None,
                        "market_abs_edge": round(abs(market_edge), 4) if market_edge is not None else None,
                        "source": "polymarket" if market else None,
                        "volume": market.get("volume") if market else None,
                        "spread": market.get("spread") if market else None,
                        "total": market.get("total") if market else None,
                    },
                    "signals": signals,
                    "signal_count": len(signals),
                }

                latest_predictions[game_id] = payload

                # Log signals
                home = state["home_tricode"]
                away = state["away_tricode"]
                score = f"{state['home_score']}-{state['away_score']}"

                if market_prob is not None:
                    mkt_edge_pct = abs(market_edge) * 100
                    print(f"  {home} vs {away} ({score} Q{state['period']}) | "
                          f"Model: {predictions['win_probability']:.1%} vs "
                          f"Market: {market_prob:.1%} | "
                          f"Edge: {mkt_edge_pct:.1f}% | "
                          f"Signals: {len(signals)}")
                elif signals:
                    edge_pct = predictions['abs_edge'] * 100
                    conf = predictions['edge_confidence'] * 100
                    print(f"  {home} vs {away} ({score} Q{state['period']}) | "
                          f"Edge: {edge_pct:.1f}% (proxy) | Conf: {conf:.1f}% | "
                          f"Signals: {len(signals)}")

        except Exception as e:
            print(f"  Polling error: {e}")

        await asyncio.sleep(30)


# ══════════════════════════════════════════════
# FASTAPI APP
# ══════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start background polling on startup."""
    global feature_engine, models, polling_active

    feature_engine = FeatureEngine()
    models = ModelSuite()

    polling_active = True
    game_task = asyncio.create_task(poll_live_games())
    market_task = asyncio.create_task(poll_market_odds())
    yield

    polling_active = False
    game_task.cancel()
    market_task.cancel()


app = FastAPI(
    title="NBA Live Betting Alpha Engine",
    version="2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ──

@app.get("/")
async def root():
    return {
        "status": "running",
        "live_games": len(latest_predictions),
        "last_poll": tracker.last_poll.isoformat() if tracker.last_poll else None,
    }


@app.get("/games")
async def get_all_games():
    """All currently tracked live games with predictions."""
    return {
        "count": len(latest_predictions),
        "games": list(latest_predictions.values()),
    }


@app.get("/games/{game_id}")
async def get_game(game_id: str):
    """Predictions for a specific game."""
    if game_id in latest_predictions:
        return latest_predictions[game_id]
    return {"error": "Game not found", "available": list(latest_predictions.keys())}


@app.get("/signals")
async def get_all_signals():
    """Only games with active trading signals."""
    active = {
        gid: data for gid, data in latest_predictions.items()
        if data["signal_count"] > 0
    }
    return {
        "count": len(active),
        "total_signals": sum(d["signal_count"] for d in active.values()),
        "games": list(active.values()),
    }


@app.get("/signals/{game_id}")
async def get_game_signals(game_id: str):
    """Signals for a specific game."""
    if game_id in latest_predictions:
        data = latest_predictions[game_id]
        return {
            "game_id": game_id,
            "signals": data["signals"],
            "predictions": data["predictions"],
        }
    return {"error": "Game not found"}


@app.get("/markets")
async def get_market_comparison():
    """Real market odds vs model predictions for all tracked games."""
    comparisons = []
    for game_id, pred in latest_predictions.items():
        comparisons.append({
            "game_id": game_id,
            "home_team": pred["home_team"],
            "away_team": pred["away_team"],
            "score": pred["score"],
            "period": pred["period"],
            "model_win_prob": pred["predictions"]["win_probability"],
            "proxy_prob": pred["predictions"]["proxy_probability"],
            "market_odds": pred.get("market_odds"),
        })
    return {
        "count": len(comparisons),
        "market_odds_available": len(latest_market_odds),
        "games": comparisons,
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "models_loaded": models is not None,
        "feature_engine_ready": feature_engine is not None,
        "polling_active": polling_active,
        "games_tracked": len(latest_predictions),
        "market_odds_loaded": len(latest_market_odds),
    }


@app.post("/predict")
async def manual_predict(
    home_team_id: int,
    away_team_id: int,
    home_score: int = 0,
    away_score: int = 0,
    period: int = 1,
    game_seconds_left: int = 2880,
):
    """
    Manual prediction endpoint for testing.
    Pass in a game state and get predictions back.
    """
    state = {
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
        "home_score": home_score,
        "away_score": away_score,
        "period": period,
        "game_seconds_left": game_seconds_left,
        "play_history": [(game_seconds_left, home_score, away_score)],
        "prev_snapshot": {"margin": 0, "scoring_pace": 0},
        "home_tricode": feature_engine.team_id_to_name.get(home_team_id, "HOM"),
        "away_tricode": feature_engine.team_id_to_name.get(away_team_id, "AWY"),
    }

    features = feature_engine.build_feature_vector(state)
    predictions = models.predict(features, feature_engine)
    signals = signal_gen.generate(predictions, state)

    return {
        "predictions": predictions,
        "signals": signals,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)