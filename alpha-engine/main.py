"""
Alpha Engine — FastAPI v2 inference server.
Loads the trained v2 XGBoost ensemble and FeatureEngine,
exposes full predictions over HTTP for the Rust execution engine.

Run:
    cd /path/to/unihack && uvicorn alpha-engine.main:app --host 0.0.0.0 --port 8001 --reload

Or from the alpha-engine directory:
    cd /path/to/unihack/alpha-engine && python main.py
"""

import os
import sys

import numpy as np
import xgboost as xgb
from fastapi import FastAPI
from pydantic import BaseModel

# ── Ensure parent dir is importable (for features.py) ────────────────────────
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJECT_ROOT)

from features import FeatureEngine

DATA_DIR = os.path.join(PROJECT_ROOT, "data")


# ══════════════════════════════════════════════
# MODEL SUITE (v2 — 4 models)
# ══════════════════════════════════════════════

class ModelSuite:
    """Loads and holds all v2 trained models."""

    def __init__(self):
        print("[alpha-engine] Loading v3 models...")
        self.win_model = xgb.XGBClassifier()
        self.win_model.load_model(f"{DATA_DIR}/v3_win_probability.json")

        self.margin_model = xgb.XGBRegressor()
        self.margin_model.load_model(f"{DATA_DIR}/v3_margin.json")

        self.proxy_model = xgb.XGBClassifier()
        self.proxy_model.load_model(f"{DATA_DIR}/v3_market_proxy.json")

        self.edge_model = xgb.XGBClassifier()
        self.edge_model.load_model(f"{DATA_DIR}/v3_edge_model.json")

        print("[alpha-engine] All v3 models loaded.")

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
# APP SETUP
# ══════════════════════════════════════════════

app = FastAPI(title="NBA Alpha Engine", version="2.0.0")

feature_engine: FeatureEngine | None = None
models: ModelSuite | None = None


@app.on_event("startup")
def load_models():
    global feature_engine, models
    # Change cwd to project root so FeatureEngine finds data/ via relative path
    os.chdir(PROJECT_ROOT)
    feature_engine = FeatureEngine()
    models = ModelSuite()


# ── Request / response schemas ───────────────────────────────────────────────

class GameState(BaseModel):
    """
    Live game state snapshot sent by the Rust engine.
    Now accepts team IDs so FeatureEngine can look up
    team profiles, rolling stats, and fatigue.
    """
    home_team_id: int        = 0
    away_team_id: int        = 0
    period: int              = 1
    game_seconds_left: float = 2880.0
    home_score: float        = 0.0
    away_score: float        = 0.0


class PredictResponse(BaseModel):
    win_probability: float
    proxy_probability: float
    predicted_margin: float
    edge: float
    abs_edge: float
    edge_confidence: float
    kelly_size: float
    model_loaded: bool


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": models is not None,
        "feature_engine_ready": feature_engine is not None,
        "version": "2.0",
    }


@app.post("/predict", response_model=PredictResponse)
def predict(state: GameState):
    if models is None or feature_engine is None:
        margin = state.home_score - state.away_score
        naive_prob = float(np.clip(0.5 + margin * 0.01, 0.01, 0.99))
        return PredictResponse(
            win_probability=naive_prob,
            proxy_probability=0.5,
            predicted_margin=0.0,
            edge=0.0,
            abs_edge=0.0,
            edge_confidence=0.0,
            kelly_size=0.0,
            model_loaded=False,
        )

    # Build full game state dict for FeatureEngine
    game_state = {
        "home_team_id": state.home_team_id,
        "away_team_id": state.away_team_id,
        "home_score": int(state.home_score),
        "away_score": int(state.away_score),
        "period": state.period,
        "game_seconds_left": int(state.game_seconds_left),
        "play_history": [(int(state.game_seconds_left),
                          int(state.home_score),
                          int(state.away_score))],
        "prev_snapshot": {"margin": 0, "scoring_pace": 0},
        "home_tricode": feature_engine.team_id_to_name.get(state.home_team_id, "HOM"),
        "away_tricode": feature_engine.team_id_to_name.get(state.away_team_id, "AWY"),
    }

    # Build 188-feature vector and run all 4 models
    features = feature_engine.build_feature_vector(game_state)
    predictions = models.predict(features, feature_engine)

    return PredictResponse(model_loaded=True, **predictions)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
