"""
NBA Live Betting Model
======================
Feature engineering + training for multi-market predictions.
Run this while nba_data.py pulls PBP in another tab.
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import brier_score_loss, mean_absolute_error, log_loss
from sklearn.calibration import calibration_curve
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = "data"

# ──────────────────────────────────────────────
# 1. LOAD ALL DATA
# ──────────────────────────────────────────────

def load_all_data():
    """Load everything we've scraped into memory."""
    data = {}
    files = {
        "games": "season_games.parquet",
        "fatigue": "fatigue.parquet",
        "pace": "pace_profiles.parquet",
        "clutch": "clutch_stats.parquet",
        "player_clutch": "player_clutch_stats.parquet",
        "lineups": "lineup_stats.parquet",
        "on_court": "player_on_court.parquet",
        "off_court": "player_off_court.parquet",
        "pbp": "play_by_play.parquet",
        "comebacks": "comeback_profiles.parquet",
    }
    for key, filename in files.items():
        try:
            data[key] = pd.read_parquet(f"{DATA_DIR}/{filename}")
            print(f"  Loaded {key}: {len(data[key])} rows")
        except FileNotFoundError:
            print(f"  WARNING: {filename} not found, skipping")
            data[key] = pd.DataFrame()
    return data


# ──────────────────────────────────────────────
# 2. BUILD TEAM PROFILE LOOKUP
#    Pre-computes season-level stats per team
#    that we'll merge into every game snapshot
# ──────────────────────────────────────────────

def build_team_profiles(data):
    """
    Combine pace, clutch, comeback tendencies, and on/off
    into a single team-level feature dictionary.
    """
    pace = data["pace"].copy()
    clutch = data["clutch"].copy()
    comebacks = data["comebacks"].copy()
    games = data["games"].copy()

    # ── Pace & efficiency features ──
    team_features = pace.set_index("TEAM_ID")[[
        "PACE", "OFF_RATING", "DEF_RATING", "NET_RATING",
        "AST_PCT", "AST_TO", "REB_PCT", "TS_PCT", "EFG_PCT"
    ]].copy()

    # ── Clutch features ──
    if not clutch.empty and "TEAM_ID" in clutch.columns:
        clutch_cols = ["TEAM_ID"]
        for col in ["NET_RATING", "W", "L", "W_PCT"]:
            if col in clutch.columns:
                clutch_cols.append(col)
        clutch_feats = clutch[clutch_cols].set_index("TEAM_ID")
        clutch_feats.columns = ["CLUTCH_" + c for c in clutch_feats.columns]
        team_features = team_features.join(clutch_feats, how="left")

    # ── Comeback / blown lead tendencies ──
    if not comebacks.empty and not games.empty:
        # Merge comeback data with game info to get team-level stats
        game_info = games[["GAME_ID", "TEAM_ID", "WL"]].drop_duplicates()
        cb = comebacks.merge(game_info, on="GAME_ID", how="left")
        if not cb.empty:
            team_cb = cb.groupby("TEAM_ID").agg(
                BLOWN_LEAD_RATE=("HOME_BLEW_LEAD", "mean"),
                COMEBACK_RATE=("AWAY_BLEW_LEAD", "mean"),  # opponent blew lead = our comeback
                AVG_MAX_LEAD=("MAX_HOME_LEAD", "mean"),
            ).fillna(0)
            team_features = team_features.join(team_cb, how="left")

    # ── Player impact concentration ──
    # How dependent is each team on their best player?
    if not data["on_court"].empty and not data["off_court"].empty:
        on = data["on_court"].copy()
        off = data["off_court"].copy()

        if "NET_RATING" in on.columns and "NET_RATING" in off.columns:
            on = on.rename(columns={"NET_RATING": "ON_NET"})
            off = off.rename(columns={"NET_RATING": "OFF_NET"})

            merged = on[["TEAM_ID", "ON_NET"]].merge(
                off[["TEAM_ID", "OFF_NET"]],
                left_index=True, right_index=True, suffixes=("", "_off")
            )
            merged["IMPACT"] = merged["ON_NET"] - merged["OFF_NET"]

            team_impact = merged.groupby("TEAM_ID").agg(
                MAX_PLAYER_IMPACT=("IMPACT", "max"),
                AVG_PLAYER_IMPACT=("IMPACT", "mean"),
                STAR_DEPENDENCY=("IMPACT", lambda x: x.max() - x.mean()),
            )
            team_features = team_features.join(team_impact, how="left")

    team_features = team_features.fillna(0)
    print(f"  Built profiles for {len(team_features)} teams, {len(team_features.columns)} features each")
    return team_features


# ──────────────────────────────────────────────
# 3. EXTRACT GAME-STATE SNAPSHOTS FROM PBP
#    Each row = one moment in a game with features
# ──────────────────────────────────────────────

def extract_game_snapshots(pbp, snapshot_interval_seconds=120):
    """
    Sample game states every N seconds of game time.
    Each snapshot becomes one training row.
    """
    snapshots = []

    for game_id in pbp["GAME_ID"].unique():
        game = pbp[pbp["GAME_ID"] == game_id].copy()
        game = game.sort_values("GAME_SECONDS_LEFT", ascending=False)

        if "SCOREHOME" not in game.columns or "SCOREAWAY" not in game.columns:
            continue

        game["SCOREHOME"] = pd.to_numeric(game["SCOREHOME"], errors="coerce")
        game["SCOREAWAY"] = pd.to_numeric(game["SCOREAWAY"], errors="coerce")
        game = game.dropna(subset=["SCOREHOME", "SCOREAWAY"])

        if game.empty:
            continue

        # Final result for labels
        final = game.loc[game["GAME_SECONDS_LEFT"].idxmin()]
        home_won = int(final["SCOREHOME"] > final["SCOREAWAY"])
        final_margin = final["SCOREHOME"] - final["SCOREAWAY"]

        # Sample at regular intervals
        max_time = game["GAME_SECONDS_LEFT"].max()
        sample_times = np.arange(0, max_time, snapshot_interval_seconds)

        for t in sample_times:
            # Get most recent state at or before this game time
            state = game[game["GAME_SECONDS_LEFT"] >= t].tail(1)
            if state.empty:
                continue

            row = state.iloc[0]
            home_score = row["SCOREHOME"]
            away_score = row["SCOREAWAY"]
            margin = home_score - away_score
            period = row.get("PERIOD", 1)
            secs_left = row["GAME_SECONDS_LEFT"]

            # ── In-game features ──
            # Scoring pace (points per minute elapsed so far)
            elapsed = max(max_time - secs_left, 1)
            total_points = home_score + away_score
            scoring_pace = total_points / (elapsed / 60)

            # Recent momentum: score change in last 2 minutes of game data
            window = game[
                (game["GAME_SECONDS_LEFT"] <= secs_left + 120) &
                (game["GAME_SECONDS_LEFT"] >= secs_left)
            ]
            if len(window) >= 2:
                momentum_home = window["SCOREHOME"].iloc[-1] - window["SCOREHOME"].iloc[0]
                momentum_away = window["SCOREAWAY"].iloc[-1] - window["SCOREAWAY"].iloc[0]
            else:
                momentum_home = 0
                momentum_away = 0

            # Lead changes and max leads up to this point
            history = game[game["GAME_SECONDS_LEFT"] >= secs_left]
            if "HOME_MARGIN" in history.columns:
                margins_so_far = history["HOME_MARGIN"].dropna()
            else:
                margins_so_far = history["SCOREHOME"] - history["SCOREAWAY"]

            max_home_lead = margins_so_far.max() if not margins_so_far.empty else 0
            max_away_lead = -margins_so_far.min() if not margins_so_far.empty else 0

            snapshots.append({
                "GAME_ID": game_id,
                # ── State features ──
                "PERIOD": period,
                "GAME_SECONDS_LEFT": secs_left,
                "HOME_SCORE": home_score,
                "AWAY_SCORE": away_score,
                "MARGIN": margin,
                "ABS_MARGIN": abs(margin),
                "TOTAL_POINTS": total_points,
                "SCORING_PACE": scoring_pace,
                # ── Time interaction features ──
                "MARGIN_X_TIME": margin * (secs_left / 2880),  # normalized
                "ABS_MARGIN_X_TIME": abs(margin) * (secs_left / 2880),
                "IS_Q4": int(period == 4),
                "IS_CLOSE_LATE": int((abs(margin) <= 5) and (secs_left <= 300)),
                "IS_BLOWOUT": int(abs(margin) >= 20),
                # ── Momentum features ──
                "HOME_MOMENTUM_2MIN": momentum_home,
                "AWAY_MOMENTUM_2MIN": momentum_away,
                "MOMENTUM_SWING": momentum_home - momentum_away,
                # ── Lead history ──
                "MAX_HOME_LEAD": max_home_lead,
                "MAX_AWAY_LEAD": max_away_lead,
                "LEAD_VOLATILITY": max_home_lead + max_away_lead,
                # ── Labels ──
                "HOME_WON": home_won,
                "FINAL_MARGIN": final_margin,
            })

    df = pd.DataFrame(snapshots)
    print(f"  Extracted {len(df)} snapshots from {df['GAME_ID'].nunique()} games")
    return df


# ──────────────────────────────────────────────
# 4. MERGE TEAM PROFILES INTO SNAPSHOTS
# ──────────────────────────────────────────────

def merge_features(snapshots, team_profiles, games):
    """
    Attach team-level season stats to each game snapshot.
    Creates HOME_ and AWAY_ prefixed features + differentials.
    """
    # Get home/away team IDs for each game
    home = games[games["MATCHUP"].str.contains("vs.")][["GAME_ID", "TEAM_ID"]].rename(
        columns={"TEAM_ID": "HOME_TEAM_ID"}
    ).drop_duplicates(subset=["GAME_ID"])

    away = games[~games["MATCHUP"].str.contains("vs.")][["GAME_ID", "TEAM_ID"]].rename(
        columns={"TEAM_ID": "AWAY_TEAM_ID"}
    ).drop_duplicates(subset=["GAME_ID"])

    df = snapshots.merge(home, on="GAME_ID", how="left")
    df = df.merge(away, on="GAME_ID", how="left")

    # Merge team profiles for home team
    home_profiles = team_profiles.add_prefix("HOME_")
    df = df.merge(home_profiles, left_on="HOME_TEAM_ID", right_index=True, how="left")

    # Merge team profiles for away team
    away_profiles = team_profiles.add_prefix("AWAY_")
    df = df.merge(away_profiles, left_on="AWAY_TEAM_ID", right_index=True, how="left")

    # ── Create differential features ──
    # These are often more predictive than raw values
    for col in team_profiles.columns:
        df[f"DIFF_{col}"] = df[f"HOME_{col}"] - df[f"AWAY_{col}"]

    df = df.fillna(0)
    print(f"  Final feature matrix: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


# ──────────────────────────────────────────────
# 5. ADD FATIGUE FEATURES
# ──────────────────────────────────────────────

def merge_fatigue(df, fatigue):
    """Add rest/fatigue features for both teams."""
    fat = fatigue[["GAME_ID", "TEAM_ID", "REST_DAYS", "IS_B2B", "GAMES_LAST_7D"]].copy()

    home_fat = fat.rename(columns={
        "TEAM_ID": "HOME_TEAM_ID",
        "REST_DAYS": "HOME_REST_DAYS",
        "IS_B2B": "HOME_IS_B2B",
        "GAMES_LAST_7D": "HOME_GAMES_LAST_7D",
    })
    df = df.merge(home_fat, on=["GAME_ID", "HOME_TEAM_ID"], how="left")

    away_fat = fat.rename(columns={
        "TEAM_ID": "AWAY_TEAM_ID",
        "REST_DAYS": "AWAY_REST_DAYS",
        "IS_B2B": "AWAY_IS_B2B",
        "GAMES_LAST_7D": "AWAY_GAMES_LAST_7D",
    })
    df = df.merge(away_fat, on=["GAME_ID", "AWAY_TEAM_ID"], how="left")

    # Differentials
    df["DIFF_REST_DAYS"] = df["HOME_REST_DAYS"] - df["AWAY_REST_DAYS"]
    df["DIFF_FATIGUE"] = df["AWAY_GAMES_LAST_7D"] - df["HOME_GAMES_LAST_7D"]  # higher = opponent more tired

    df = df.fillna(0)
    return df


# ──────────────────────────────────────────────
# 6. RECENCY WEIGHTING
# ──────────────────────────────────────────────

def compute_sample_weights(df, games, lambda_decay=0.03):
    """
    Exponential decay: recent games weighted higher.
    Returns array of weights aligned with df rows.
    """
    game_dates = games[["GAME_ID", "GAME_DATE"]].drop_duplicates()
    game_dates["GAME_DATE"] = pd.to_datetime(game_dates["GAME_DATE"])

    df = df.merge(game_dates, on="GAME_ID", how="left")
    max_date = df["GAME_DATE"].max()
    df["DAYS_AGO"] = (max_date - df["GAME_DATE"]).dt.days
    df["SAMPLE_WEIGHT"] = np.exp(-lambda_decay * df["DAYS_AGO"])

    # Boost close games — they're more informative for live betting
    df.loc[df["IS_CLOSE_LATE"] == 1, "SAMPLE_WEIGHT"] *= 1.3

    return df


# ──────────────────────────────────────────────
# 7. DEFINE FEATURE COLUMNS
# ──────────────────────────────────────────────

def get_feature_columns(df):
    """
    Return list of feature columns (everything except
    labels, IDs, metadata).
    """
    exclude = {
        "GAME_ID", "HOME_TEAM_ID", "AWAY_TEAM_ID",
        "HOME_WON", "FINAL_MARGIN",  # labels
        "GAME_DATE", "DAYS_AGO", "SAMPLE_WEIGHT",  # metadata
    }
    features = [c for c in df.columns if c not in exclude and df[c].dtype in ["float64", "int64", "float32", "int32"]]
    return sorted(features)


# ──────────────────────────────────────────────
# 8. TRAIN MODELS
# ──────────────────────────────────────────────

def train_win_probability_model(df, feature_cols):
    """
    XGBoost classifier: predicts P(home team wins).
    Uses time-series split to avoid leakage.
    """
    print("\n" + "=" * 50)
    print("TRAINING: Win Probability Model")
    print("=" * 50)

    X = df[feature_cols].values
    y = df["HOME_WON"].values
    weights = df["SAMPLE_WEIGHT"].values

    tscv = TimeSeriesSplit(n_splits=5)
    brier_scores = []
    models = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        w_train = weights[train_idx]

        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=10,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            early_stopping_rounds=30,
            random_state=42,
            verbosity=0,
        )
        model.fit(
            X_train, y_train,
            sample_weight=w_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        preds = model.predict_proba(X_val)[:, 1]
        brier = brier_score_loss(y_val, preds)
        ll = log_loss(y_val, preds)
        brier_scores.append(brier)
        models.append(model)

        print(f"  Fold {fold+1}: Brier={brier:.4f}, LogLoss={ll:.4f}")

    print(f"  Avg Brier Score: {np.mean(brier_scores):.4f}")

    # Train final model on all data
    final_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="binary:logistic",
        random_state=42,
        verbosity=0,
    )
    final_model.fit(X, y, sample_weight=weights)

    return final_model


def train_margin_model(df, feature_cols):
    """
    XGBoost regressor: predicts final margin (home - away).
    Used for spread markets.
    """
    print("\n" + "=" * 50)
    print("TRAINING: Margin Prediction Model")
    print("=" * 50)

    X = df[feature_cols].values
    y = df["FINAL_MARGIN"].values
    weights = df["SAMPLE_WEIGHT"].values

    tscv = TimeSeriesSplit(n_splits=5)
    mae_scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        w_train = weights[train_idx]

        model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=10,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective="reg:squarederror",
            early_stopping_rounds=30,
            random_state=42,
            verbosity=0,
        )
        model.fit(
            X_train, y_train,
            sample_weight=w_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        mae_scores.append(mae)

        print(f"  Fold {fold+1}: MAE={mae:.2f} points")

    print(f"  Avg MAE: {np.mean(mae_scores):.2f} points")

    # Final model
    final_model = xgb.XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, verbosity=0,
    )
    final_model.fit(X, y, sample_weight=weights)

    return final_model


# ──────────────────────────────────────────────
# 9. FEATURE IMPORTANCE
# ──────────────────────────────────────────────

def print_feature_importance(model, feature_cols, top_n=20):
    """Print top features by importance."""
    importance = model.feature_importances_
    feat_imp = sorted(zip(feature_cols, importance), key=lambda x: -x[1])

    print(f"\n  Top {top_n} features:")
    for name, imp in feat_imp[:top_n]:
        bar = "█" * int(imp * 200)
        print(f"    {name:35s} {imp:.4f} {bar}")


# ──────────────────────────────────────────────
# 10. SAVE MODELS
# ──────────────────────────────────────────────

def save_models(win_model, margin_model, feature_cols):
    """Save models and feature list for live inference."""
    import json

    win_model.save_model(f"{DATA_DIR}/win_probability_model.json")
    margin_model.save_model(f"{DATA_DIR}/margin_model.json")

    with open(f"{DATA_DIR}/feature_columns.json", "w") as f:
        json.dump(feature_cols, f)

    print(f"\n  Models saved to {DATA_DIR}/")
    print(f"    win_probability_model.json")
    print(f"    margin_model.json")
    print(f"    feature_columns.json")


# ──────────────────────────────────────────────
# 11. RUN EVERYTHING
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading data...")
    data = load_all_data()

    print("\nBuilding team profiles...")
    team_profiles = build_team_profiles(data)

    print("\nExtracting game snapshots from play-by-play...")
    snapshots = extract_game_snapshots(data["pbp"], snapshot_interval_seconds=120)

    print("\nMerging team profiles...")
    df = merge_features(snapshots, team_profiles, data["games"])

    print("\nMerging fatigue features...")
    df = merge_fatigue(df, data["fatigue"])

    print("\nComputing recency weights...")
    df = compute_sample_weights(df, data["games"])

    feature_cols = get_feature_columns(df)
    print(f"\nFeature columns ({len(feature_cols)}):")
    for c in feature_cols:
        print(f"  {c}")

    # ── Train ──
    win_model = train_win_probability_model(df, feature_cols)
    print_feature_importance(win_model, feature_cols)

    margin_model = train_margin_model(df, feature_cols)
    print_feature_importance(margin_model, feature_cols)

    # ── Save ──
    save_models(win_model, margin_model, feature_cols)

    print("\n" + "=" * 50)
    print("ALL DONE — models ready for live inference!")
    print("=" * 50)