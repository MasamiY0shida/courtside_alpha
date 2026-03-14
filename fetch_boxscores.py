"""
Fetch Historical Advanced Boxscore Data
========================================
Pulls per-game advanced team/player stats from the NBA Stats API for all
games in season_games.parquet. Produces data/boxscore_advanced.parquet which
model_v2.py uses to populate the ~15 features that were previously zero
during training (pts_paint, bench_pts, star_pm, star_mins, lineup_pm, etc.).

Usage:
    ./venv/bin/python fetch_boxscores.py

Takes ~30-50 minutes for ~989 games (rate-limited API calls).
Safe to re-run — skips games already fetched.
"""

import pandas as pd
import numpy as np
import time
import os
import sys

DATA_DIR = "data"
OUTPUT_PATH = f"{DATA_DIR}/boxscore_advanced.parquet"


def parse_minutes(mins_str):
    """Parse NBA minutes string like '33:14' to float minutes."""
    if not mins_str or mins_str == "":
        return 0.0
    try:
        parts = str(mins_str).split(":")
        return float(parts[0]) + float(parts[1]) / 60 if len(parts) == 2 else float(parts[0])
    except (ValueError, IndexError):
        return 0.0


def fetch_game_boxscore(game_id):
    """Fetch advanced boxscore for a single game. Returns a dict or None."""
    from nba_api.stats.endpoints import boxscoretraditionalv3, boxscoremiscv3
    import warnings
    warnings.filterwarnings("ignore")

    result = {"GAME_ID": game_id}

    # ── Misc stats (scoring breakdown) ──
    try:
        misc = boxscoremiscv3.BoxScoreMiscV3(game_id=game_id)
        md = misc.get_dict()["boxScoreMisc"]

        for side, key in [("HOME", "homeTeam"), ("AWAY", "awayTeam")]:
            team = md.get(key, {})
            result[f"{side}_TEAM_ID"] = team.get("teamId", 0)
            stats = team.get("statistics", {})
            result[f"{side}_PTS_PAINT"] = stats.get("pointsPaint", 0)
            result[f"{side}_PTS_FASTBREAK"] = stats.get("pointsFastBreak", 0)
            result[f"{side}_PTS_2ND"] = stats.get("pointsSecondChance", 0)
            result[f"{side}_PTS_OFF_TO"] = stats.get("pointsOffTurnovers", 0)
    except Exception as e:
        print(f"  Misc fetch failed for {game_id}: {e}")
        return None

    time.sleep(0.6)

    # ── Traditional stats (player-level for star/bench/lineup) ──
    try:
        trad = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id)
        td = trad.get_dict()["boxScoreTraditional"]

        for side, key in [("HOME", "homeTeam"), ("AWAY", "awayTeam")]:
            team = td.get(key, {})
            players = team.get("players", [])

            # Parse player data
            pdata = []
            for p in players:
                ps = p.get("statistics", {})
                mins = parse_minutes(ps.get("minutes", ""))
                pdata.append({
                    "name": f"{p.get('firstName', '')} {p.get('familyName', '')}",
                    "mins": mins,
                    "pm": float(ps.get("plusMinusPoints", 0) or 0),
                    "pts": int(ps.get("points", 0) or 0),
                    "fga": int(ps.get("fieldGoalsAttempted", 0) or 0),
                })

            if not pdata:
                result[f"{side}_STAR_PM"] = 0
                result[f"{side}_STAR_MINS_TOTAL"] = 0
                result[f"{side}_LINEUP_PM"] = 0
                result[f"{side}_BENCH_PTS"] = 0
                continue

            # Star player = highest minutes (matches inference definition in server.py)
            star = max(pdata, key=lambda x: x["mins"])
            result[f"{side}_STAR_PM"] = star["pm"]
            result[f"{side}_STAR_MINS_TOTAL"] = star["mins"]

            # Lineup PM = sum of top-5-by-minutes players' plus/minus
            by_mins = sorted(pdata, key=lambda x: x["mins"], reverse=True)
            starters = by_mins[:5]
            bench = by_mins[5:]
            result[f"{side}_LINEUP_PM"] = sum(p["pm"] for p in starters)

            # Bench points
            result[f"{side}_BENCH_PTS"] = sum(p["pts"] for p in bench)

    except Exception as e:
        print(f"  Traditional fetch failed for {game_id}: {e}")
        return None

    return result


def main():
    # Load existing games
    games = pd.read_parquet(f"{DATA_DIR}/season_games.parquet")
    game_ids = sorted(games["GAME_ID"].unique())
    print(f"Total games in training data: {len(game_ids)}")

    # Load any previously fetched data (resume support)
    existing = set()
    rows = []
    if os.path.exists(OUTPUT_PATH):
        prev = pd.read_parquet(OUTPUT_PATH)
        existing = set(prev["GAME_ID"].values)
        rows = prev.to_dict("records")
        print(f"Resuming: {len(existing)} games already fetched")

    remaining = [gid for gid in game_ids if gid not in existing]
    print(f"Games to fetch: {len(remaining)}")

    if not remaining:
        print("All games already fetched!")
        return

    for i, game_id in enumerate(remaining):
        if i > 0 and i % 50 == 0:
            # Save progress every 50 games
            df = pd.DataFrame(rows)
            df.to_parquet(OUTPUT_PATH, index=False)
            print(f"  Progress saved: {len(rows)} games total")

        result = fetch_game_boxscore(game_id)
        if result:
            rows.append(result)
            if i % 20 == 0:
                print(f"  [{i+1}/{len(remaining)}] {game_id} OK")
        else:
            print(f"  [{i+1}/{len(remaining)}] {game_id} FAILED")

        time.sleep(0.6)  # Rate limiting

    # Final save
    df = pd.DataFrame(rows)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nDone! Saved {len(df)} games to {OUTPUT_PATH}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
