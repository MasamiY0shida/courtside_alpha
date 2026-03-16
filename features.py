"""
features.py — Shared feature construction
==========================================
Used by both model training and live inference server.
Constructs the 188-feature vector from live game state.
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta

DATA_DIR = "data"


class FeatureEngine:
    """
    Loads team profiles and model artifacts once,
    then constructs feature vectors on demand for live games.
    """

    def __init__(self):
        # Load team profiles
        self.pace = pd.read_parquet(f"{DATA_DIR}/pace_profiles.parquet")
        self.clutch = pd.read_parquet(f"{DATA_DIR}/clutch_stats.parquet")
        self.games = pd.read_parquet(f"{DATA_DIR}/season_games.parquet")
        self.fatigue = pd.read_parquet(f"{DATA_DIR}/fatigue.parquet")

        try:
            self.on_court = pd.read_parquet(f"{DATA_DIR}/player_on_court.parquet")
            self.off_court = pd.read_parquet(f"{DATA_DIR}/player_off_court.parquet")
        except FileNotFoundError:
            self.on_court = pd.DataFrame()
            self.off_court = pd.DataFrame()

        # Load feature column lists
        with open(f"{DATA_DIR}/v3_live_features.json") as f:
            self.live_features = json.load(f)
        with open(f"{DATA_DIR}/v3_pregame_features.json") as f:
            self.pregame_features = json.load(f)
        with open(f"{DATA_DIR}/v3_edge_features.json") as f:
            self.edge_features = json.load(f)

        # Build lookups
        self.team_profiles = self._build_team_profiles()
        self.rolling_cache = self._build_rolling_cache()
        self.fatigue_cache = self._build_fatigue_cache()

        # Team name/abbreviation to ID mapping
        self.team_name_to_id = dict(zip(
            self.games["TEAM_ABBREVIATION"], self.games["TEAM_ID"]
        ))
        self.team_id_to_name = dict(zip(
            self.games["TEAM_ID"], self.games["TEAM_ABBREVIATION"]
        ))

        print(f"FeatureEngine initialized:")
        print(f"  {len(self.team_profiles)} team profiles")
        print(f"  {len(self.live_features)} live features expected")
        print(f"  {len(self.pregame_features)} pregame features expected")

    def _build_team_profiles(self):
        """Static season-level team profiles."""
        pace = self.pace.set_index("TEAM_ID")[[
            "PACE", "OFF_RATING", "DEF_RATING", "NET_RATING",
            "AST_PCT", "AST_TO", "REB_PCT", "TS_PCT", "EFG_PCT"
        ]].copy()

        if not self.clutch.empty and "TEAM_ID" in self.clutch.columns:
            clutch_cols = ["TEAM_ID"]
            for col in ["NET_RATING", "W_PCT"]:
                if col in self.clutch.columns:
                    clutch_cols.append(col)
            clutch_feats = self.clutch[clutch_cols].set_index("TEAM_ID")
            clutch_feats.columns = ["CLUTCH_" + c for c in clutch_feats.columns]
            pace = pace.join(clutch_feats, how="left")

        if not self.on_court.empty and not self.off_court.empty:
            on = self.on_court.rename(columns={"NET_RATING": "ON_NET"})
            off = self.off_court.rename(columns={"NET_RATING": "OFF_NET"})
            if "ON_NET" in on.columns and "OFF_NET" in off.columns:
                merged = on[["TEAM_ID", "ON_NET"]].merge(
                    off[["TEAM_ID", "OFF_NET"]],
                    left_index=True, right_index=True, suffixes=("", "_off")
                )
                merged["IMPACT"] = merged["ON_NET"] - merged["OFF_NET"]
                team_impact = merged.groupby("TEAM_ID").agg(
                    MAX_PLAYER_IMPACT=("IMPACT", "max"),
                    STAR_DEPENDENCY=("IMPACT", lambda x: x.max() - x.mean()),
                )
                pace = pace.join(team_impact, how="left")

        return pace.fillna(0)

    def _build_rolling_cache(self):
        """Pre-compute most recent rolling stats per team."""
        games = self.games.copy()
        games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])
        games = games.sort_values("GAME_DATE")

        stat_cols = ["PTS", "FG_PCT", "FG3_PCT", "FT_PCT",
                     "REB", "AST", "STL", "BLK", "TOV", "PLUS_MINUS"]
        stat_cols = [c for c in stat_cols if c in games.columns]

        cache = {}
        for team_id in games["TEAM_ID"].unique():
            team = games[games["TEAM_ID"] == team_id].sort_values("GAME_DATE")
            team_cache = {}

            for window in [5, 10, 20]:
                for col in stat_cols:
                    vals = team[col].astype(float).shift(1)
                    rolled = vals.rolling(window, min_periods=3).mean()
                    team_cache[f"ROLL_{window}_{col}"] = rolled.iloc[-1] if len(rolled) > 0 else 0

                if "WL" in team.columns:
                    wl = (team["WL"] == "W").astype(float).shift(1)
                    rolled = wl.rolling(window, min_periods=3).mean()
                    team_cache[f"ROLL_{window}_WIN_PCT"] = rolled.iloc[-1] if len(rolled) > 0 else 0.5

            # Trajectory
            team_cache["FORM_TRAJECTORY"] = (
                team_cache.get("ROLL_5_PLUS_MINUS", 0) -
                team_cache.get("ROLL_20_PLUS_MINUS", 0)
            )
            team_cache["WIN_TRAJECTORY"] = (
                team_cache.get("ROLL_5_WIN_PCT", 0.5) -
                team_cache.get("ROLL_20_WIN_PCT", 0.5)
            )

            # Streak (shifted)
            if "WL" in team.columns:
                streak = 0
                for wl in team["WL"].values[:-1]:  # exclude current
                    if wl == "W":
                        streak = max(1, streak + 1)
                    else:
                        streak = min(-1, streak - 1)
                team_cache["STREAK"] = streak
            else:
                team_cache["STREAK"] = 0

            cache[team_id] = team_cache

        return cache

    def _build_fatigue_cache(self):
        """Most recent fatigue info per team."""
        fat = self.fatigue.copy()
        fat_latest = fat.sort_values("GAME_DATE" if "GAME_DATE" in fat.columns else "GAME_ID")
        cache = {}
        for team_id in fat["TEAM_ID"].unique():
            team_fat = fat_latest[fat_latest["TEAM_ID"] == team_id].iloc[-1]
            cache[team_id] = {
                "REST_DAYS": team_fat.get("REST_DAYS", 1),
                "IS_B2B": team_fat.get("IS_B2B", 0),
                "GAMES_LAST_7D": team_fat.get("GAMES_LAST_7D", 2),
            }
        return cache

    def build_feature_vector(self, game_state):
        """
        Construct the full feature vector from a live game state dict.

        game_state should contain:
            home_team_id, away_team_id,
            home_score, away_score,
            period, game_seconds_left,
            play_history: list of (seconds_left, home_score, away_score)
        """
        home_id = game_state["home_team_id"]
        away_id = game_state["away_team_id"]
        home_score = game_state["home_score"]
        away_score = game_state["away_score"]
        period = game_state["period"]
        secs_left = game_state["game_seconds_left"]
        history = game_state.get("play_history", [])

        margin = home_score - away_score
        total_points = home_score + away_score
        game_progress = 1 - (secs_left / 2880)
        elapsed = max(2880 - secs_left, 1)
        scoring_pace = total_points / (elapsed / 60) if elapsed > 60 else 0
        quarter_secs_left = secs_left % 720 if period <= 4 else secs_left
        quarter_progress = 1 - (quarter_secs_left / 720)

        # Momentum from play history
        momentum = {}
        for window in [60, 120, 300]:
            h_pts, a_pts = 0, 0
            for sl, hs, as_ in history:
                if secs_left <= sl <= secs_left + window:
                    pass  # within window
            # Simpler: use last N entries
            recent = [p for p in history if p[0] >= secs_left and p[0] <= secs_left + window]
            if len(recent) >= 2:
                h_pts = recent[-1][1] - recent[0][1]
                a_pts = recent[-1][2] - recent[0][2]
            momentum[f"HOME_MOM_{window}s"] = h_pts
            momentum[f"AWAY_MOM_{window}s"] = a_pts
            momentum[f"SWING_{window}s"] = h_pts - a_pts

        # Lead history
        margins_hist = [h - a for _, h, a in history if _ >= secs_left]
        max_home_lead = max(margins_hist) if margins_hist else max(margin, 0)
        max_away_lead = -min(margins_hist) if margins_hist else max(-margin, 0)

        if margins_hist:
            sign_changes = int(np.sum(np.diff(np.sign(margins_hist)) != 0))
        else:
            sign_changes = 0

        # Lag features (from previous poll)
        prev = game_state.get("prev_snapshot", {})
        lag_margin = prev.get("margin", 0)
        lag_pace = prev.get("scoring_pace", 0)

        # Build the feature dict
        features = {
            # Core state
            "PERIOD": period,
            "GAME_SECONDS_LEFT": secs_left,
            "GAME_PROGRESS": game_progress,
            "QUARTER_PROGRESS": quarter_progress,
            "HOME_SCORE": home_score,
            "AWAY_SCORE": away_score,
            "MARGIN": margin,
            "ABS_MARGIN": abs(margin),
            "TOTAL_POINTS": total_points,
            "SCORING_PACE": scoring_pace,
            # Time interactions (v3: force temporal awareness)
            "MARGIN_X_PROGRESS": margin * game_progress,
            "ABS_MARGIN_X_PROGRESS": abs(margin) * game_progress,
            "MARGIN_OVER_SQRT_TIME": margin / max(np.sqrt(secs_left / 60), 1.0),
            "ABS_MARGIN_OVER_SQRT_TIME": abs(margin) / max(np.sqrt(secs_left / 60), 1.0),
            "TIME_REMAINING_FRAC": secs_left / 2880,
            "STAT_RELIABILITY": min(total_points / 30.0, 1.0),
            "PERIOD_ORDINAL": min(period, 5),
            "IS_Q4": int(period == 4),
            "IS_CLOSE_LATE": int((abs(margin) <= 5) and (secs_left <= 300)),
            "IS_BLOWOUT": int(abs(margin) >= 20),
            "IS_CLUTCH": int((abs(margin) <= 5) and (secs_left <= 300) and (period == 4)),
            "IS_FIRST_HALF": int(period <= 2),
            # Lead history
            "MAX_HOME_LEAD": max_home_lead,
            "MAX_AWAY_LEAD": max_away_lead,
            "LEAD_VOLATILITY": max_home_lead + max_away_lead,
            "LEAD_CHANGES": sign_changes,
            # Lag
            "LAG_MARGIN": lag_margin,
            "MARGIN_CHANGE": margin - lag_margin,
            "LAG_SCORING_PACE": lag_pace,
            "PACE_CHANGE": scoring_pace - lag_pace,
        }
        features.update(momentum)

        # ── Team profiles (static) ──
        for prefix, team_id in [("HOME_STATIC", home_id), ("AWAY_STATIC", away_id)]:
            if team_id in self.team_profiles.index:
                for col in self.team_profiles.columns:
                    features[f"{prefix}_{col}"] = self.team_profiles.loc[team_id, col]
            else:
                for col in self.team_profiles.columns:
                    features[f"{prefix}_{col}"] = 0

        # Static differentials
        for col in self.team_profiles.columns:
            features[f"DIFF_STATIC_{col}"] = (
                features.get(f"HOME_STATIC_{col}", 0) -
                features.get(f"AWAY_STATIC_{col}", 0)
            )

        # ── Rolling features ──
        for prefix, team_id in [("HOME", home_id), ("AWAY", away_id)]:
            cache = self.rolling_cache.get(team_id, {})
            for key, val in cache.items():
                features[f"{prefix}_{key}"] = val if not pd.isna(val) else 0

        # Rolling differentials
        home_cache = self.rolling_cache.get(home_id, {})
        away_cache = self.rolling_cache.get(away_id, {})
        for key in home_cache:
            h = home_cache.get(key, 0)
            a = away_cache.get(key, 0)
            h = 0 if pd.isna(h) else h
            a = 0 if pd.isna(a) else a
            features[f"DIFF_{key}"] = h - a

        # ── Fatigue ──
        for prefix, team_id in [("HOME", home_id), ("AWAY", away_id)]:
            fat = self.fatigue_cache.get(team_id, {})
            features[f"{prefix}_REST_DAYS"] = fat.get("REST_DAYS", 1)
            features[f"{prefix}_IS_B2B"] = fat.get("IS_B2B", 0)
            features[f"{prefix}_GAMES_LAST_7D"] = fat.get("GAMES_LAST_7D", 2)

        features["DIFF_REST_DAYS"] = features["HOME_REST_DAYS"] - features["AWAY_REST_DAYS"]
        features["DIFF_FATIGUE"] = features["AWAY_GAMES_LAST_7D"] - features["HOME_GAMES_LAST_7D"]

        # ── Live boxscore features (populated by server.py enrich_from_boxscore) ──
        hb = "home_box"
        ab = "away_box"

        # Team shooting (live)
        features["LIVE_HOME_FG_PCT"] = game_state.get(f"{hb}_fg_pct", 0)
        features["LIVE_AWAY_FG_PCT"] = game_state.get(f"{ab}_fg_pct", 0)
        features["LIVE_DIFF_FG_PCT"] = features["LIVE_HOME_FG_PCT"] - features["LIVE_AWAY_FG_PCT"]

        features["LIVE_HOME_FG3_PCT"] = game_state.get(f"{hb}_fg3_pct", 0)
        features["LIVE_AWAY_FG3_PCT"] = game_state.get(f"{ab}_fg3_pct", 0)
        features["LIVE_DIFF_FG3_PCT"] = features["LIVE_HOME_FG3_PCT"] - features["LIVE_AWAY_FG3_PCT"]

        features["LIVE_HOME_EFG_PCT"] = game_state.get(f"{hb}_efg_pct", 0)
        features["LIVE_AWAY_EFG_PCT"] = game_state.get(f"{ab}_efg_pct", 0)
        features["LIVE_DIFF_EFG_PCT"] = features["LIVE_HOME_EFG_PCT"] - features["LIVE_AWAY_EFG_PCT"]

        features["LIVE_HOME_TS_PCT"] = game_state.get(f"{hb}_ts_pct", 0)
        features["LIVE_AWAY_TS_PCT"] = game_state.get(f"{ab}_ts_pct", 0)
        features["LIVE_DIFF_TS_PCT"] = features["LIVE_HOME_TS_PCT"] - features["LIVE_AWAY_TS_PCT"]

        features["LIVE_HOME_FT_PCT"] = game_state.get(f"{hb}_ft_pct", 0)
        features["LIVE_AWAY_FT_PCT"] = game_state.get(f"{ab}_ft_pct", 0)

        # Rebounding (live)
        features["LIVE_HOME_REB_OFF"] = game_state.get(f"{hb}_reb_off", 0)
        features["LIVE_AWAY_REB_OFF"] = game_state.get(f"{ab}_reb_off", 0)
        features["LIVE_DIFF_REB_OFF"] = features["LIVE_HOME_REB_OFF"] - features["LIVE_AWAY_REB_OFF"]

        features["LIVE_HOME_REB_TOTAL"] = game_state.get(f"{hb}_reb_total", 0)
        features["LIVE_AWAY_REB_TOTAL"] = game_state.get(f"{ab}_reb_total", 0)
        features["LIVE_DIFF_REB_TOTAL"] = features["LIVE_HOME_REB_TOTAL"] - features["LIVE_AWAY_REB_TOTAL"]

        # Turnovers & assists (live)
        features["LIVE_HOME_AST"] = game_state.get(f"{hb}_assists", 0)
        features["LIVE_AWAY_AST"] = game_state.get(f"{ab}_assists", 0)
        features["LIVE_HOME_TOV"] = game_state.get(f"{hb}_turnovers", 0)
        features["LIVE_AWAY_TOV"] = game_state.get(f"{ab}_turnovers", 0)
        features["LIVE_DIFF_AST_TO"] = (
            game_state.get(f"{hb}_ast_to_ratio", 0) -
            game_state.get(f"{ab}_ast_to_ratio", 0)
        )
        features["LIVE_DIFF_TOV"] = features["LIVE_HOME_TOV"] - features["LIVE_AWAY_TOV"]

        # Defense (live)
        features["LIVE_HOME_STL"] = game_state.get(f"{hb}_steals", 0)
        features["LIVE_AWAY_STL"] = game_state.get(f"{ab}_steals", 0)
        features["LIVE_HOME_BLK"] = game_state.get(f"{hb}_blocks", 0)
        features["LIVE_AWAY_BLK"] = game_state.get(f"{ab}_blocks", 0)

        # Fouls & bonus (live) — critical for late-game
        features["LIVE_HOME_FOULS"] = game_state.get(f"{hb}_fouls", 0)
        features["LIVE_AWAY_FOULS"] = game_state.get(f"{ab}_fouls", 0)
        features["LIVE_DIFF_FOULS"] = features["LIVE_HOME_FOULS"] - features["LIVE_AWAY_FOULS"]
        features["LIVE_HOME_IN_BONUS"] = game_state.get(f"{hb}_in_bonus", 0)
        features["LIVE_AWAY_IN_BONUS"] = game_state.get(f"{ab}_in_bonus", 0)
        features["LIVE_HOME_FOUL_TROUBLE"] = game_state.get(f"{hb}_foul_trouble", 0)
        features["LIVE_AWAY_FOUL_TROUBLE"] = game_state.get(f"{ab}_foul_trouble", 0)

        # Timeouts remaining
        features["LIVE_HOME_TIMEOUTS"] = game_state.get(f"{hb}_timeouts_remaining", 0)
        features["LIVE_AWAY_TIMEOUTS"] = game_state.get(f"{ab}_timeouts_remaining", 0)

        # Scoring quality (live)
        features["LIVE_HOME_PTS_PAINT"] = game_state.get(f"{hb}_pts_paint", 0)
        features["LIVE_AWAY_PTS_PAINT"] = game_state.get(f"{ab}_pts_paint", 0)
        features["LIVE_DIFF_PTS_PAINT"] = features["LIVE_HOME_PTS_PAINT"] - features["LIVE_AWAY_PTS_PAINT"]

        features["LIVE_HOME_PTS_FASTBREAK"] = game_state.get(f"{hb}_pts_fastbreak", 0)
        features["LIVE_AWAY_PTS_FASTBREAK"] = game_state.get(f"{ab}_pts_fastbreak", 0)

        features["LIVE_HOME_PTS_2ND"] = game_state.get(f"{hb}_pts_2nd_chance", 0)
        features["LIVE_AWAY_PTS_2ND"] = game_state.get(f"{ab}_pts_2nd_chance", 0)
        features["LIVE_DIFF_PTS_2ND"] = features["LIVE_HOME_PTS_2ND"] - features["LIVE_AWAY_PTS_2ND"]

        features["LIVE_HOME_PTS_OFF_TO"] = game_state.get(f"{hb}_pts_off_to", 0)
        features["LIVE_AWAY_PTS_OFF_TO"] = game_state.get(f"{ab}_pts_off_to", 0)

        features["LIVE_HOME_BENCH_PTS"] = game_state.get(f"{hb}_bench_pts", 0)
        features["LIVE_AWAY_BENCH_PTS"] = game_state.get(f"{ab}_bench_pts", 0)
        features["LIVE_DIFF_BENCH_PTS"] = features["LIVE_HOME_BENCH_PTS"] - features["LIVE_AWAY_BENCH_PTS"]

        # Game flow
        features["LIVE_LEAD_CHANGES"] = game_state.get(f"{hb}_lead_changes", 0)
        features["LIVE_TIMES_TIED"] = game_state.get(f"{hb}_times_tied", 0)
        features["LIVE_HOME_BIGGEST_LEAD"] = game_state.get(f"{hb}_biggest_lead", 0)
        features["LIVE_AWAY_BIGGEST_LEAD"] = game_state.get(f"{ab}_biggest_lead", 0)
        features["LIVE_HOME_BIGGEST_RUN"] = game_state.get(f"{hb}_biggest_run", 0)
        features["LIVE_AWAY_BIGGEST_RUN"] = game_state.get(f"{ab}_biggest_run", 0)

        # Free throw rate (FTA / FGA) — proxy for aggressiveness / drawing fouls
        home_fga = game_state.get(f"{hb}_fga", 0)
        away_fga = game_state.get(f"{ab}_fga", 0)
        features["LIVE_HOME_FT_RATE"] = (
            game_state.get(f"{hb}_fta", 0) / home_fga if home_fga > 0 else 0
        )
        features["LIVE_AWAY_FT_RATE"] = (
            game_state.get(f"{ab}_fta", 0) / away_fga if away_fga > 0 else 0
        )

        # Star player tracking
        features["LIVE_HOME_STAR_PTS"] = game_state.get(f"{hb}_star_pts", 0)
        features["LIVE_AWAY_STAR_PTS"] = game_state.get(f"{ab}_star_pts", 0)
        features["LIVE_HOME_STAR_FOULS"] = game_state.get(f"{hb}_star_fouls", 0)
        features["LIVE_AWAY_STAR_FOULS"] = game_state.get(f"{ab}_star_fouls", 0)
        features["LIVE_HOME_STAR_PM"] = game_state.get(f"{hb}_star_pm", 0)
        features["LIVE_AWAY_STAR_PM"] = game_state.get(f"{ab}_star_pm", 0)
        features["LIVE_HOME_STAR_MINS"] = game_state.get(f"{hb}_star_mins", 0)
        features["LIVE_AWAY_STAR_MINS"] = game_state.get(f"{ab}_star_mins", 0)

        # Lineup quality
        features["LIVE_HOME_LINEUP_PM"] = game_state.get(f"{hb}_lineup_pm", 0)
        features["LIVE_AWAY_LINEUP_PM"] = game_state.get(f"{ab}_lineup_pm", 0)
        features["LIVE_DIFF_LINEUP_PM"] = features["LIVE_HOME_LINEUP_PM"] - features["LIVE_AWAY_LINEUP_PM"]

        # Hot/cold shooting
        features["LIVE_HOME_HOT_SHOOTERS"] = game_state.get(f"{hb}_hot_shooters", 0)
        features["LIVE_AWAY_HOT_SHOOTERS"] = game_state.get(f"{ab}_hot_shooters", 0)
        features["LIVE_HOME_COLD_SHOOTERS"] = game_state.get(f"{hb}_cold_shooters", 0)
        features["LIVE_AWAY_COLD_SHOOTERS"] = game_state.get(f"{ab}_cold_shooters", 0)

        # Composite signals
        features["LIVE_STAR_FOUL_DANGER"] = max(
            features["LIVE_HOME_STAR_FOULS"], features["LIVE_AWAY_STAR_FOULS"]
        )
        features["LIVE_FOUL_TROUBLE_DIFF"] = (
            features["LIVE_AWAY_FOUL_TROUBLE"] - features["LIVE_HOME_FOUL_TROUBLE"]
        )  # positive = away team in more trouble = good for home

        # v3: Time-damp all LIVE_ features so model trusts them less in Q1
        damp = max(game_progress, 0.05)
        for key in list(features.keys()):
            if key.startswith("LIVE_"):
                features[key + "_DAMPED"] = features[key] * damp

        return features

    def to_live_array(self, features):
        """Convert feature dict to array matching model's expected column order."""
        return np.array([[features.get(col, 0) for col in self.live_features]])

    def to_pregame_array(self, features):
        """Convert feature dict to array matching proxy model's column order."""
        return np.array([[features.get(col, 0) for col in self.pregame_features]])

    def to_edge_array(self, features):
        """Convert feature dict to array matching edge model's column order."""
        return np.array([[features.get(col, 0) for col in self.edge_features]])