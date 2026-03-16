"""
Live Observation Recorder
=========================
Records game state, model predictions, market odds, and full feature vectors
during live games for iterative model improvement.

Data flow:
  poll_live_games() --> record_snapshot() --> SQLite
  game ends         --> finalize_game()   --> backfill outcomes
  retrain           --> export_for_training() --> DataFrame for model_v2.py

Key advantage over historical training data:
  Live observations capture REAL Polymarket/Kalshi odds paired with model
  predictions and known outcomes. The edge model can train on actual market
  pricing instead of proxy model estimates.
"""

import sqlite3
import json
import os
from datetime import datetime, timezone

DB_PATH = "live_observations.sqlite"


def _get_db():
    """Get SQLite connection with WAL mode for concurrent reads."""
    db = sqlite3.connect(DB_PATH)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    return db


def init_db():
    """Create tables if they don't exist."""
    db = _get_db()
    db.executescript("""
        CREATE TABLE IF NOT EXISTS snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            recorded_at TEXT NOT NULL,
            game_id TEXT NOT NULL,
            home_team_id INTEGER NOT NULL,
            away_team_id INTEGER NOT NULL,
            home_tricode TEXT,
            away_tricode TEXT,
            period INTEGER,
            game_seconds_left REAL,
            home_score INTEGER,
            away_score INTEGER,

            -- Market odds (real market pricing)
            polymarket_home_prob REAL,
            polymarket_volume REAL,
            polymarket_spread REAL,
            polymarket_total REAL,
            kalshi_home_prob REAL,

            -- Model predictions
            model_win_prob REAL,
            model_proxy_prob REAL,
            model_margin REAL,
            model_edge REAL,
            model_edge_confidence REAL,
            model_kelly_size REAL,

            -- Full feature vector (JSON blob of all features)
            feature_vector TEXT,

            -- Outcome labels (backfilled when game ends)
            final_home_score INTEGER,
            final_away_score INTEGER,
            home_won INTEGER
        );

        CREATE TABLE IF NOT EXISTS game_outcomes (
            game_id TEXT PRIMARY KEY,
            home_team_id INTEGER,
            away_team_id INTEGER,
            home_tricode TEXT,
            away_tricode TEXT,
            final_home_score INTEGER,
            final_away_score INTEGER,
            home_won INTEGER,
            completed_at TEXT,
            total_snapshots INTEGER
        );

        CREATE INDEX IF NOT EXISTS idx_snap_game
            ON snapshots(game_id);
        CREATE INDEX IF NOT EXISTS idx_snap_time
            ON snapshots(recorded_at);
    """)

    # ── Schema migrations (safe to re-run) ──
    for col, dtype in [
        ("boxscore_json",    "TEXT"),
        ("lineup_json",      "TEXT"),
        ("pbp_recent_json",  "TEXT"),
        ("polymarket_bid",   "REAL"),
        ("polymarket_ask",   "REAL"),
    ]:
        try:
            db.execute(f"ALTER TABLE snapshots ADD COLUMN {col} {dtype}")
        except sqlite3.OperationalError:
            pass  # column already exists

    db.close()


def record_snapshot(
    game_id, game_state, predictions, market_odds, feature_vector,
    boxscore_json=None, lineup_json=None, pbp_recent_json=None,
):
    """
    Record a single observation point during a live game.

    Called every ~30 seconds per active game from server.py's poll loop.
    """
    # Extract bid/ask if available in market_odds
    bid = market_odds.get("bid") if market_odds else None
    ask = market_odds.get("ask") if market_odds else None

    db = _get_db()
    db.execute("""
        INSERT INTO snapshots (
            recorded_at, game_id,
            home_team_id, away_team_id, home_tricode, away_tricode,
            period, game_seconds_left, home_score, away_score,
            polymarket_home_prob, polymarket_volume,
            polymarket_spread, polymarket_total,
            polymarket_bid, polymarket_ask,
            model_win_prob, model_proxy_prob, model_margin,
            model_edge, model_edge_confidence, model_kelly_size,
            feature_vector,
            boxscore_json, lineup_json, pbp_recent_json
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        datetime.now(timezone.utc).isoformat(),
        game_id,
        game_state.get("home_team_id", 0),
        game_state.get("away_team_id", 0),
        game_state.get("home_tricode", ""),
        game_state.get("away_tricode", ""),
        game_state.get("period", 0),
        game_state.get("game_seconds_left", 0),
        game_state.get("home_score", 0),
        game_state.get("away_score", 0),
        market_odds.get("polymarket_prob") if market_odds else None,
        market_odds.get("volume") if market_odds else None,
        market_odds.get("spread") if market_odds else None,
        market_odds.get("total") if market_odds else None,
        bid, ask,
        predictions.get("win_probability"),
        predictions.get("proxy_probability"),
        predictions.get("predicted_margin"),
        predictions.get("edge"),
        predictions.get("edge_confidence"),
        predictions.get("kelly_size"),
        json.dumps(feature_vector, default=lambda x: float(x) if hasattr(x, 'item') else str(x)) if feature_vector else None,
        boxscore_json,
        lineup_json,
        pbp_recent_json,
    ))
    db.commit()
    db.close()


def finalize_game(game_id, final_home_score, final_away_score):
    """
    Record final outcome and backfill snapshot labels.

    Called when a tracked game transitions to status=3 (final).
    """
    home_won = 1 if final_home_score > final_away_score else 0
    db = _get_db()

    # Backfill all snapshots for this game
    db.execute("""
        UPDATE snapshots
        SET final_home_score = ?, final_away_score = ?, home_won = ?
        WHERE game_id = ?
    """, (final_home_score, final_away_score, home_won, game_id))

    # Record in game_outcomes summary
    row = db.execute("""
        SELECT home_team_id, away_team_id, home_tricode, away_tricode, COUNT(*)
        FROM snapshots WHERE game_id = ?
    """, (game_id,)).fetchone()

    if row and row[0]:
        db.execute("""
            INSERT OR REPLACE INTO game_outcomes
            (game_id, home_team_id, away_team_id, home_tricode, away_tricode,
             final_home_score, final_away_score, home_won, completed_at,
             total_snapshots)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (
            game_id, row[0], row[1], row[2], row[3],
            final_home_score, final_away_score, home_won,
            datetime.now(timezone.utc).isoformat(), row[4],
        ))

    db.commit()
    db.close()
    return home_won


# ── Query helpers ────────────────────────────────────────────────────────────

def get_pending_game_ids():
    """Return game_ids that have snapshots but no outcome (home_won IS NULL)."""
    if not os.path.exists(DB_PATH):
        return []
    db = _get_db()
    rows = db.execute("""
        SELECT DISTINCT game_id
        FROM snapshots
        WHERE home_won IS NULL
    """).fetchall()
    db.close()
    return [r[0] for r in rows]


def get_completed_game_ids():
    """Return game_ids that already have outcomes backfilled."""
    if not os.path.exists(DB_PATH):
        return set()
    db = _get_db()
    rows = db.execute("""
        SELECT DISTINCT game_id
        FROM snapshots
        WHERE home_won IS NOT NULL
    """).fetchall()
    db.close()
    return {r[0] for r in rows}


def get_pending_games_with_teams():
    """Return pending games with team info for tracker restoration."""
    if not os.path.exists(DB_PATH):
        return []
    db = _get_db()
    rows = db.execute("""
        SELECT DISTINCT game_id, home_team_id, away_team_id,
               home_tricode, away_tricode
        FROM snapshots
        WHERE home_won IS NULL
    """).fetchall()
    db.close()
    return [
        {
            "game_id": r[0],
            "home_team_id": r[1],
            "away_team_id": r[2],
            "home_tricode": r[3],
            "away_tricode": r[4],
        }
        for r in rows
    ]


def export_for_training(db_path=None):
    """
    Export completed-game snapshots as a training-ready DataFrame.

    Returns a DataFrame with:
    - All feature columns (matching model_v2.py column names exactly)
    - GAME_ID, HOME_TEAM_ID, AWAY_TEAM_ID (metadata)
    - HOME_WON, FINAL_MARGIN (labels)
    - MARKET_PROB (real Polymarket odds)
    - RECORDED_MODEL_PROB, RECORDED_MARGIN, RECORDED_EDGE (model state)
    - RECORDED_AT (timestamp for recency weighting)
    """
    import pandas as pd

    path = db_path or DB_PATH
    if not os.path.exists(path):
        return pd.DataFrame()

    db = sqlite3.connect(path)
    snapshots = pd.read_sql_query(
        "SELECT * FROM snapshots WHERE home_won IS NOT NULL", db
    )
    db.close()

    if snapshots.empty:
        return pd.DataFrame()

    # Parse feature vectors from JSON into individual columns
    features_list = []
    for fv_json in snapshots["feature_vector"]:
        features_list.append(json.loads(fv_json) if fv_json else {})
    features_df = pd.DataFrame(features_list)

    # Build training-ready DataFrame
    result = pd.DataFrame()
    result["GAME_ID"] = snapshots["game_id"].values
    result["HOME_TEAM_ID"] = snapshots["home_team_id"].values
    result["AWAY_TEAM_ID"] = snapshots["away_team_id"].values

    # All feature columns (names match training pipeline exactly)
    for col in features_df.columns:
        result[col] = features_df[col].values

    # Labels
    result["HOME_WON"] = snapshots["home_won"].values
    result["FINAL_MARGIN"] = (
        snapshots["final_home_score"] - snapshots["final_away_score"]
    ).values

    # Real market probability
    result["MARKET_PROB"] = snapshots["polymarket_home_prob"].values

    # Model state at time of snapshot (for analysis and OOF substitution)
    result["RECORDED_MODEL_PROB"] = snapshots["model_win_prob"].values
    result["RECORDED_MARGIN"] = snapshots["model_margin"].values
    result["RECORDED_EDGE"] = snapshots["model_edge"].values
    result["RECORDED_AT"] = snapshots["recorded_at"].values

    return result


def get_stats():
    """Return recording statistics for monitoring."""
    if not os.path.exists(DB_PATH):
        return {
            "status": "no database",
            "total_snapshots": 0,
            "completed_games": 0,
        }

    db = _get_db()
    total = db.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0]
    completed = db.execute("SELECT COUNT(*) FROM game_outcomes").fetchone()[0]
    pending = db.execute(
        "SELECT COUNT(DISTINCT game_id) FROM snapshots WHERE home_won IS NULL"
    ).fetchone()[0]
    with_market = db.execute(
        "SELECT COUNT(*) FROM snapshots WHERE polymarket_home_prob IS NOT NULL"
    ).fetchone()[0]

    recent = db.execute("""
        SELECT game_id, home_tricode, away_tricode,
               MIN(recorded_at) as first_snap, MAX(recorded_at) as last_snap,
               COUNT(*) as snap_count, home_won
        FROM snapshots
        GROUP BY game_id
        ORDER BY first_snap DESC
        LIMIT 5
    """).fetchall()
    db.close()

    recent_games = []
    for row in recent:
        recent_games.append({
            "game_id": row[0],
            "matchup": f"{row[1]} vs {row[2]}",
            "snapshots": row[5],
            "status": "completed" if row[6] is not None else "in_progress",
        })

    return {
        "total_snapshots": total,
        "completed_games": completed,
        "pending_games": pending,
        "snapshots_with_market_odds": with_market,
        "market_coverage_pct": (
            round(with_market / total * 100, 1) if total > 0 else 0
        ),
        "recent_games": recent_games,
    }
