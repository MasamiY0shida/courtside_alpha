"""
Backfill game outcomes for live observation snapshots.

Checks the NBA S3 boxscore endpoint for final scores of games
that have snapshots but no recorded outcome (home_won IS NULL).

Usage:
    python backfill_outcomes.py
"""

import requests
import time
import recorder

NBA_BOXSCORE_URL = (
    "https://nba-prod-us-east-1-mediaops-stats.s3.amazonaws.com"
    "/NBA/liveData/boxscore/boxscore_{game_id}.json"
)


def fetch_final_score(game_id):
    """Fetch final score from NBA S3 boxscore endpoint. Returns (home, away) or None."""
    url = NBA_BOXSCORE_URL.format(game_id=game_id)
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return None

        game = resp.json().get("game", {})
        status = game.get("gameStatus", 0)

        if status != 3:  # Not final yet
            return None

        home_score = int(game.get("homeTeam", {}).get("score", 0))
        away_score = int(game.get("awayTeam", {}).get("score", 0))
        home_tri = game.get("homeTeam", {}).get("teamTricode", "?")
        away_tri = game.get("awayTeam", {}).get("teamTricode", "?")

        return home_score, away_score, home_tri, away_tri
    except Exception as e:
        print(f"  Error fetching {game_id}: {e}")
        return None


def backfill():
    """Backfill all pending game outcomes."""
    recorder.init_db()
    pending = recorder.get_pending_game_ids()

    if not pending:
        print("No pending games to backfill.")
        return 0

    print(f"Found {len(pending)} games with missing outcomes.\n")

    filled = 0
    for game_id in pending:
        result = fetch_final_score(game_id)
        if result:
            home_score, away_score, home_tri, away_tri = result
            winner = home_tri if home_score > away_score else away_tri
            recorder.finalize_game(game_id, home_score, away_score)
            print(f"  {home_tri} {home_score} - {away_tri} {away_score} "
                  f"({winner} wins) — backfilled {game_id}")
            filled += 1
        else:
            print(f"  {game_id} — not final yet or unavailable")

        time.sleep(0.5)  # Rate limit

    print(f"\nBackfilled {filled}/{len(pending)} games.")
    return filled


if __name__ == "__main__":
    backfill()
