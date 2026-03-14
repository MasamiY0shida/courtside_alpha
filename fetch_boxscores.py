"""
Fetch Historical Advanced Boxscore Data
========================================
Pulls per-game advanced team/player stats from the NBA live data endpoint
for all games in season_games.parquet. Produces data/boxscore_advanced.parquet
which model_v2.py uses to populate the ~15 features that were previously zero
during training (pts_paint, bench_pts, star_pm, star_mins, lineup_pm, etc.).

Usage:
    python fetch_boxscores.py

Takes ~20-30 minutes for ~989 games (rate-limited).
Safe to re-run — skips games already fetched.
"""

import pandas as pd
import numpy as np
import requests
import time
import os
import sys

DATA_DIR = "data"
OUTPUT_PATH = f"{DATA_DIR}/boxscore_advanced.parquet"

# NBA live data endpoint — no auth/headers needed, no blocking
BOXSCORE_URL = (
    "https://nba-prod-us-east-1-mediaops-stats.s3.amazonaws.com"
    "/NBA/liveData/boxscore/boxscore_{game_id}.json"
)

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
})


def parse_minutes(mins_str):
    """Parse NBA minutes string like 'PT33M14.00S' to float minutes."""
    if not mins_str:
        return 0.0
    s = str(mins_str)
    # Format: PT33M14.00S
    if s.startswith("PT"):
        s = s[2:]  # Remove PT
        minutes = 0.0
        if "M" in s:
            m_part, s = s.split("M", 1)
            minutes += float(m_part)
        if "S" in s:
            s_part = s.replace("S", "")
            if s_part:
                minutes += float(s_part) / 60.0
        return minutes
    # Fallback: "33:14" format
    try:
        parts = s.split(":")
        return float(parts[0]) + float(parts[1]) / 60 if len(parts) == 2 else float(parts[0])
    except (ValueError, IndexError):
        return 0.0


def fetch_game_boxscore(game_id):
    """Fetch boxscore for a single game from NBA S3 endpoint. Returns a dict or None."""
    url = BOXSCORE_URL.format(game_id=game_id)

    try:
        resp = SESSION.get(url, timeout=30)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        game = resp.json()["game"]
    except Exception as e:
        print(f"  Fetch failed for {game_id}: {e}")
        return None

    result = {"GAME_ID": game_id}

    for side, key in [("HOME", "homeTeam"), ("AWAY", "awayTeam")]:
        team = game.get(key, {})
        result[f"{side}_TEAM_ID"] = team.get("teamId", 0)

        # Team-level stats
        stats = team.get("statistics", {})
        result[f"{side}_PTS_PAINT"] = stats.get("pointsInThePaint", 0)
        result[f"{side}_PTS_FASTBREAK"] = stats.get("pointsFastBreak", 0)
        result[f"{side}_PTS_2ND"] = stats.get("pointsSecondChance", 0)
        result[f"{side}_PTS_OFF_TO"] = stats.get("pointsFromTurnovers", 0)
        result[f"{side}_BENCH_PTS"] = stats.get("benchPoints", 0)

        # Player-level stats for star/lineup
        players = team.get("players", [])
        pdata = []
        for p in players:
            ps = p.get("statistics", {})
            mins = parse_minutes(ps.get("minutes", ""))
            pdata.append({
                "mins": mins,
                "pm": float(ps.get("plusMinusPoints", 0) or 0),
                "pts": float(ps.get("points", 0) or 0),
                "fouls": float(ps.get("foulsPersonal", 0) or 0),
            })

        if not pdata:
            result[f"{side}_STAR_PM"] = 0
            result[f"{side}_STAR_MINS_TOTAL"] = 0
            result[f"{side}_STAR_PTS"] = 0
            result[f"{side}_STAR_FOULS"] = 0
            result[f"{side}_LINEUP_PM"] = 0
            continue

        # Star player = highest minutes (matches inference definition in server.py)
        star = max(pdata, key=lambda x: x["mins"])
        result[f"{side}_STAR_PM"] = star["pm"]
        result[f"{side}_STAR_MINS_TOTAL"] = star["mins"]
        result[f"{side}_STAR_PTS"] = star["pts"]
        result[f"{side}_STAR_FOULS"] = star["fouls"]

        # Lineup PM = sum of top-5-by-minutes players' plus/minus
        by_mins = sorted(pdata, key=lambda x: x["mins"], reverse=True)
        starters = by_mins[:5]
        result[f"{side}_LINEUP_PM"] = sum(p["pm"] for p in starters)

    return result


def main():
    # Load existing games
    games = pd.read_parquet(f"{DATA_DIR}/season_games.parquet")
    game_ids = sorted(games["GAME_ID"].unique())
    print(f"Total games in training data: {len(game_ids)}")

    # Load any previously fetched data (resume support)
    REQUIRED_COLS = {"HOME_STAR_PTS", "HOME_STAR_FOULS"}
    existing = set()
    rows = []
    if os.path.exists(OUTPUT_PATH):
        prev = pd.read_parquet(OUTPUT_PATH)
        if not REQUIRED_COLS.issubset(prev.columns):
            print("Existing parquet missing new columns (STAR_PTS/STAR_FOULS) — re-fetching all games.")
            os.remove(OUTPUT_PATH)
        else:
            existing = set(prev["GAME_ID"].values)
            rows = prev.to_dict("records")
            print(f"Resuming: {len(existing)} games already fetched")

    remaining = [gid for gid in game_ids if gid not in existing]
    print(f"Games to fetch: {len(remaining)}")

    if not remaining:
        print("All games already fetched!")
        return

    # Quick connectivity test
    print("Testing API connectivity...")
    test = fetch_game_boxscore(remaining[0])
    if test:
        rows.append(test)
        print(f"  OK: {remaining[0]} — home paint pts: {test.get('HOME_PTS_PAINT')}")
        remaining = remaining[1:]
    else:
        print("  FAILED — check network connectivity")
        sys.exit(1)

    failed = []
    for i, game_id in enumerate(remaining):
        if (i + 1) % 50 == 0:
            df = pd.DataFrame(rows)
            df.to_parquet(OUTPUT_PATH, index=False)
            print(f"  Progress saved: {len(rows)} games total")

        result = None
        for attempt in range(3):
            result = fetch_game_boxscore(game_id)
            if result:
                break
            wait = 5 * (attempt + 1)
            print(f"  Retry {attempt+1}/3 for {game_id} in {wait}s...")
            time.sleep(wait)

        if result:
            rows.append(result)
            if (i + 1) % 50 == 0:
                print(f"  [{i+2}/{len(remaining)+1}] {len(rows)} fetched, {len(failed)} failed")
        else:
            failed.append(game_id)

        time.sleep(0.5)  # Light rate limiting (S3 is generous)

    # Final save
    df = pd.DataFrame(rows)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nDone! Saved {len(df)} games to {OUTPUT_PATH}")
    print(f"Columns: {list(df.columns)}")
    if failed:
        print(f"Failed games ({len(failed)}): {failed[:20]}{'...' if len(failed) > 20 else ''}")


if __name__ == "__main__":
    main()
