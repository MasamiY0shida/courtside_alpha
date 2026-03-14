"""
market_data.py — Polymarket + Kalshi Market Data Collector
============================================================
Collects real-time odds from prediction markets and records
historical price movements for model training.
"""

import requests
import time
import json
import os
import re
from datetime import datetime, date
import pandas as pd
import numpy as np

DATA_DIR = "data"
MARKETS_DIR = f"{DATA_DIR}/markets"
os.makedirs(MARKETS_DIR, exist_ok=True)


# ══════════════════════════════════════════════
# 1. POLYMARKET — Gamma API (free, no auth)
# ══════════════════════════════════════════════

GAMMA_BASE = "https://gamma-api.polymarket.com"
GAMMA_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
}

# NBA team names for filtering
NBA_TEAMS = [
    "76ers", "Bucks", "Bulls", "Cavaliers", "Celtics", "Clippers",
    "Grizzlies", "Hawks", "Heat", "Hornets", "Jazz", "Kings",
    "Knicks", "Lakers", "Magic", "Mavericks", "Nets", "Nuggets",
    "Pacers", "Pelicans", "Pistons", "Raptors", "Rockets", "Spurs",
    "Suns", "Thunder", "Timberwolves", "Trail Blazers", "Warriors", "Wizards",
]

NBA_TRICODES = [
    "ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW",
    "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK",
    "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS",
]


def fetch_polymarket_nba_markets():
    """
    Fetch all active NBA game markets from Polymarket.
    Returns list of market dicts with prices and metadata.
    """
    print("Fetching Polymarket NBA markets...")
    all_markets = []
    offset = 0
    limit = 100

    while True:
        try:
            # Use tag-based filtering for NBA
            resp = requests.get(
                f"{GAMMA_BASE}/markets",
                params={
                    "active": True,
                    "closed": False,
                    "limit": limit,
                    "offset": offset,
                },
                headers=GAMMA_HEADERS,
                timeout=30,
            )
            if resp.status_code != 200:
                print(f"  Polymarket API error: {resp.status_code}")
                break

            markets = resp.json()
            if not markets:
                break

            # Filter for NBA game markets
            for m in markets:
                question = (m.get("question", "") + " " + m.get("description", "")).lower()
                # Check if this is an NBA game market
                is_nba = any(team.lower() in question for team in NBA_TEAMS)
                is_nba = is_nba or any(code.lower() in question.split() for code in NBA_TRICODES)
                is_nba = is_nba or "nba" in question

                if is_nba:
                    # Extract prices
                    outcomes = m.get("outcomes", [])
                    outcome_prices = m.get("outcomePrices", [])

                    if isinstance(outcome_prices, str):
                        try:
                            outcome_prices = json.loads(outcome_prices)
                        except:
                            outcome_prices = []

                    market_data = {
                        "market_id": m.get("id", ""),
                        "condition_id": m.get("conditionId", ""),
                        "question": m.get("question", ""),
                        "slug": m.get("slug", ""),
                        "platform": "polymarket",
                        "outcomes": outcomes if isinstance(outcomes, list) else json.loads(outcomes) if isinstance(outcomes, str) else [],
                        "outcome_prices": [float(p) for p in outcome_prices] if outcome_prices else [],
                        "volume": float(m.get("volume", 0) or 0),
                        "liquidity": float(m.get("liquidity", 0) or 0),
                        "clob_token_ids": m.get("clobTokenIds", []),
                        "end_date": m.get("endDate", ""),
                        "active": m.get("active", False),
                        "closed": m.get("closed", False),
                        "fetched_at": datetime.now().isoformat(),
                    }

                    # Parse implied probabilities
                    if market_data["outcome_prices"]:
                        market_data["implied_prob_yes"] = market_data["outcome_prices"][0]
                        if len(market_data["outcome_prices"]) > 1:
                            market_data["implied_prob_no"] = market_data["outcome_prices"][1]

                    all_markets.append(market_data)

            offset += limit
            time.sleep(0.5)

            # Safety limit
            if offset > 1000:
                break

        except Exception as e:
            print(f"  Error: {e}")
            break

    print(f"  Found {len(all_markets)} NBA markets on Polymarket")
    return all_markets


def fetch_polymarket_events_nba():
    """
    Alternative: fetch via events endpoint which groups
    related markets (moneyline, spread, total) together.
    """
    print("Fetching Polymarket NBA events...")
    all_events = []
    offset = 0

    while True:
        try:
            resp = requests.get(
                f"{GAMMA_BASE}/events",
                params={
                    "active": True,
                    "closed": False,
                    "limit": 50,
                    "offset": offset,
                    "order": "id",
                    "ascending": False,
                },
                headers=GAMMA_HEADERS,
                timeout=30,
            )
            if resp.status_code != 200:
                break

            events = resp.json()
            if not events:
                break

            for event in events:
                title = (event.get("title", "") + " " + event.get("description", "")).lower()
                is_nba = any(team.lower() in title for team in NBA_TEAMS)
                is_nba = is_nba or "nba" in title

                if is_nba:
                    # Extract nested markets
                    nested_markets = event.get("markets", [])
                    event_data = {
                        "event_id": event.get("id", ""),
                        "title": event.get("title", ""),
                        "slug": event.get("slug", ""),
                        "platform": "polymarket",
                        "fetched_at": datetime.now().isoformat(),
                        "markets": [],
                    }

                    for m in nested_markets:
                        outcome_prices = m.get("outcomePrices", [])
                        if isinstance(outcome_prices, str):
                            try:
                                outcome_prices = json.loads(outcome_prices)
                            except:
                                outcome_prices = []

                        event_data["markets"].append({
                            "market_id": m.get("id", ""),
                            "question": m.get("question", ""),
                            "outcome_prices": [float(p) for p in outcome_prices] if outcome_prices else [],
                            "volume": float(m.get("volume", 0) or 0),
                            "liquidity": float(m.get("liquidity", 0) or 0),
                        })

                    all_events.append(event_data)

            offset += 50
            time.sleep(0.5)
            if offset > 500:
                break

        except Exception as e:
            print(f"  Error: {e}")
            break

    print(f"  Found {len(all_events)} NBA events on Polymarket")
    return all_events

def fetch_polymarket_game_odds():
    """
    Fetch individual NBA game moneylines from Polymarket.
    These replace the proxy model as our market baseline.
    """
    resp = requests.get(
        f"{GAMMA_BASE}/events",
        params={
            "series_id": 10345,       # NBA
            "tag_id": 100639,         # Individual games
            "active": True,
            "closed": False,
            "order": "startTime",
            "ascending": True,
            "limit": 50,
        },
        headers=GAMMA_HEADERS,
        timeout=30,
    )

    if resp.status_code != 200:
        print(f"Polymarket API error: {resp.status_code}")
        return {}

    events = resp.json()
    game_odds = {}

    for event in events:
        title = event.get("title", "")
        markets = event.get("markets", [])

        for m in markets:
            question = m.get("question", "")
            prices = m.get("outcomePrices", "[]")
            if isinstance(prices, str):
                try:
                    prices = json.loads(prices)
                except:
                    prices = []

            volume = float(m.get("volume", 0) or 0)

            # Moneyline = market where question is just "Team vs Team"
            # (no "Spread:", "O/U", "Points", etc.)
            is_moneyline = (
                "spread" not in question.lower()
                and "o/u" not in question.lower()
                and "points" not in question.lower()
                and "rebounds" not in question.lower()
                and "assists" not in question.lower()
                and "1h" not in question.lower()
                and "moneyline" not in question.lower()
                and prices
            )

            if is_moneyline and volume > 1000:
                team1, team2 = parse_teams_from_question(question)
                if team1 and team2:
                    # Use outcomes array to match each team name to its price.
                    # Polymarket orders outcomes as listed in the question ("Away vs. Home"),
                    # so prices[0]/prices[1] don't reliably map to team1/team2 (which are
                    # returned in TEAM_ALIASES dict order, not question order).
                    outcomes_raw = m.get("outcomes", "[]")
                    if isinstance(outcomes_raw, str):
                        try:
                            outcomes_raw = json.loads(outcomes_raw)
                        except Exception:
                            outcomes_raw = []
                    outcomes_list = outcomes_raw if isinstance(outcomes_raw, list) else []

                    # Default fallback (pre-fix behaviour)
                    team1_prob = float(prices[0])
                    team2_prob = float(prices[1]) if len(prices) > 1 else 1 - float(prices[0])

                    # Override with name-matched probabilities when outcomes are available
                    for i, outcome_name in enumerate(outcomes_list):
                        if i >= len(prices):
                            break
                        oname = outcome_name.lower()
                        for alias, info in TEAM_ALIASES.items():
                            if alias in oname:
                                if info["team_id"] == team1["team_id"]:
                                    team1_prob = float(prices[i])
                                elif info["team_id"] == team2["team_id"]:
                                    team2_prob = float(prices[i])
                                break

                    game_key = f"{team1['tricode']}_vs_{team2['tricode']}"
                    game_odds[game_key] = {
                        "home_team": team1["tricode"],
                        "away_team": team2["tricode"],
                        "home_team_id": team1["team_id"],
                        "away_team_id": team2["team_id"],
                        "home_win_prob": team1_prob,
                        "away_win_prob": team2_prob,
                        "volume": volume,
                        "market_id": m.get("id", ""),
                        "event_title": title,
                        "fetched_at": datetime.now().isoformat(),
                    }

                    # Also look for spread/total in sibling markets
                    for sibling in markets:
                        sq = sibling.get("question", "")
                        sp = sibling.get("outcomePrices", "[]")
                        if isinstance(sp, str):
                            try:
                                sp = json.loads(sp)
                            except:
                                sp = []

                        if "spread" in sq.lower() and sp:
                            # Extract spread number
                            import re
                            spread_match = re.search(r'[-+]?\d+\.?\d*', sq.split("Spread:")[-1] if "Spread:" in sq else sq)
                            if spread_match:
                                game_odds[game_key]["spread"] = float(spread_match.group())
                                game_odds[game_key]["spread_price"] = float(sp[0])

                        if "o/u" in sq.lower() and sp and "1h" not in sq.lower():
                            total_match = re.search(r'\d+\.?\d*', sq.split("O/U")[-1])
                            if total_match:
                                game_odds[game_key]["total"] = float(total_match.group())
                                game_odds[game_key]["over_price"] = float(sp[0])

    print(f"Found {len(game_odds)} NBA game moneylines on Polymarket")
    return game_odds


# ══════════════════════════════════════════════
# 2. KALSHI — Public REST API
# ══════════════════════════════════════════════

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"


def fetch_kalshi_nba_markets():
    """
    Fetch NBA markets from Kalshi via events endpoint.
    Much faster than scanning all markets.
    """
    print("Fetching Kalshi NBA markets...")
    all_markets = []
    cursor = None

    while True:
        try:
            params = {
                "limit": 200,
                "status": "open",
                "with_nested_markets": True,
            }
            if cursor:
                params["cursor"] = cursor

            resp = requests.get(
                f"{KALSHI_BASE}/events",
                params=params,
                timeout=15,
            )

            if resp.status_code != 200:
                print(f"  Kalshi API error: {resp.status_code}")
                break

            data = resp.json()
            events = data.get("events", [])
            cursor = data.get("cursor", "")

            for event in events:
                title = event.get("title", "").lower()
                category = event.get("category", "").lower()
                
                is_nba = "nba" in title or "nba" in category
                is_nba = is_nba or any(t.lower() in title for t in NBA_TEAMS)

                if not is_nba:
                    continue

                for m in event.get("markets", []):
                    yes_price = m.get("yes_bid", 0) or 0
                    last_price = m.get("last_price", 0) or 0

                    all_markets.append({
                        "market_id": m.get("ticker", ""),
                        "event_ticker": event.get("ticker", ""),
                        "title": m.get("title", "") or event.get("title", ""),
                        "subtitle": m.get("subtitle", ""),
                        "platform": "kalshi",
                        "yes_price": yes_price / 100 if yes_price > 1 else yes_price,
                        "last_price": last_price / 100 if last_price > 1 else last_price,
                        "implied_prob_yes": last_price / 100 if last_price > 1 else last_price,
                        "volume": m.get("volume", 0),
                        "open_interest": m.get("open_interest", 0),
                        "status": m.get("status", ""),
                        "fetched_at": datetime.now().isoformat(),
                    })

            if not cursor:
                break
            time.sleep(0.3)

        except Exception as e:
            print(f"  Error: {e}")
            break

    print(f"  Found {len(all_markets)} NBA markets on Kalshi")
    return all_markets


# ══════════════════════════════════════════════
# 3. TEAM NAME MATCHING
#    Match market questions to NBA team IDs
# ══════════════════════════════════════════════

TEAM_ALIASES = {
    "lakers": {"tricode": "LAL", "team_id": 1610612747},
    "celtics": {"tricode": "BOS", "team_id": 1610612738},
    "warriors": {"tricode": "GSW", "team_id": 1610612744},
    "nuggets": {"tricode": "DEN", "team_id": 1610612743},
    "bucks": {"tricode": "MIL", "team_id": 1610612749},
    "76ers": {"tricode": "PHI", "team_id": 1610612755},
    "sixers": {"tricode": "PHI", "team_id": 1610612755},
    "suns": {"tricode": "PHX", "team_id": 1610612756},
    "heat": {"tricode": "MIA", "team_id": 1610612748},
    "knicks": {"tricode": "NYK", "team_id": 1610612752},
    "mavericks": {"tricode": "DAL", "team_id": 1610612742},
    "mavs": {"tricode": "DAL", "team_id": 1610612742},
    "cavaliers": {"tricode": "CLE", "team_id": 1610612739},
    "cavs": {"tricode": "CLE", "team_id": 1610612739},
    "thunder": {"tricode": "OKC", "team_id": 1610612760},
    "timberwolves": {"tricode": "MIN", "team_id": 1610612750},
    "wolves": {"tricode": "MIN", "team_id": 1610612750},
    "clippers": {"tricode": "LAC", "team_id": 1610612746},
    "hawks": {"tricode": "ATL", "team_id": 1610612737},
    "nets": {"tricode": "BKN", "team_id": 1610612751},
    "hornets": {"tricode": "CHA", "team_id": 1610612766},
    "bulls": {"tricode": "CHI", "team_id": 1610612741},
    "pistons": {"tricode": "DET", "team_id": 1610612765},
    "pacers": {"tricode": "IND", "team_id": 1610612754},
    "rockets": {"tricode": "HOU", "team_id": 1610612745},
    "grizzlies": {"tricode": "MEM", "team_id": 1610612763},
    "pelicans": {"tricode": "NOP", "team_id": 1610612740},
    "magic": {"tricode": "ORL", "team_id": 1610612753},
    "trail blazers": {"tricode": "POR", "team_id": 1610612757},
    "blazers": {"tricode": "POR", "team_id": 1610612757},
    "kings": {"tricode": "SAC", "team_id": 1610612758},
    "spurs": {"tricode": "SAS", "team_id": 1610612759},
    "raptors": {"tricode": "TOR", "team_id": 1610612761},
    "jazz": {"tricode": "UTA", "team_id": 1610612762},
    "wizards": {"tricode": "WAS", "team_id": 1610612764},
}


def parse_teams_from_question(question):
    """
    Extract home and away teams from a market question.
    e.g. 'Will the Lakers beat the Celtics?' -> (LAL, BOS)
    e.g. 'Clippers vs. Bulls' -> (LAC, CHI)  (first team = home typically)
    """
    q = question.lower()
    found_teams = []

    for alias, info in TEAM_ALIASES.items():
        if alias in q:
            found_teams.append(info)

    if len(found_teams) >= 2:
        return found_teams[0], found_teams[1]
    elif len(found_teams) == 1:
        return found_teams[0], None
    return None, None


# ══════════════════════════════════════════════
# 4. LIVE ODDS SNAPSHOT
#    Combines both platforms into one view
# ══════════════════════════════════════════════

def get_combined_nba_odds():
    """
    Fetch from both platforms and combine into a
    single view of current NBA market odds.
    """
    print("\n" + "=" * 50)
    print("FETCHING COMBINED NBA ODDS")
    print("=" * 50)

    poly_markets = fetch_polymarket_nba_markets()
    kalshi_markets = fetch_kalshi_nba_markets()

    combined = []

    # Process Polymarket
    for m in poly_markets:
        team1, team2 = parse_teams_from_question(m.get("question", ""))
        combined.append({
            "platform": "polymarket",
            "question": m.get("question", ""),
            "team1": team1["tricode"] if team1 else "???",
            "team2": team2["tricode"] if team2 else "???",
            "team1_id": team1["team_id"] if team1 else None,
            "team2_id": team2["team_id"] if team2 else None,
            "implied_prob": m.get("implied_prob_yes", 0),
            "volume": m.get("volume", 0),
            "liquidity": m.get("liquidity", 0),
            "market_id": m.get("market_id", ""),
            "fetched_at": m.get("fetched_at", ""),
        })

    # Process Kalshi
    for m in kalshi_markets:
        team1, team2 = parse_teams_from_question(m.get("title", ""))
        combined.append({
            "platform": "kalshi",
            "question": m.get("title", ""),
            "team1": team1["tricode"] if team1 else "???",
            "team2": team2["tricode"] if team2 else "???",
            "team1_id": team1["team_id"] if team1 else None,
            "team2_id": team2["team_id"] if team2 else None,
            "implied_prob": m.get("implied_prob_yes", 0),
            "volume": m.get("volume", 0),
            "liquidity": 0,
            "market_id": m.get("market_id", ""),
            "fetched_at": m.get("fetched_at", ""),
        })

    df = pd.DataFrame(combined)
    print(f"\n  Combined: {len(df)} markets total")
    print(f"    Polymarket: {len([m for m in combined if m['platform'] == 'polymarket'])}")
    print(f"    Kalshi: {len([m for m in combined if m['platform'] == 'kalshi'])}")

    return df


# ══════════════════════════════════════════════
# 5. LIVE ODDS RECORDER
#    Polls both platforms periodically and saves
#    a time-series of odds for each game
# ══════════════════════════════════════════════

def record_odds_snapshot():
    """
    Take one snapshot of all current NBA odds.
    Appends to a running CSV file.
    """
    odds = get_combined_nba_odds()
    if odds.empty:
        return odds

    output_file = f"{MARKETS_DIR}/odds_history.csv"
    if os.path.exists(output_file):
        existing = pd.read_csv(output_file)
        odds = pd.concat([existing, odds], ignore_index=True)

    odds.to_csv(output_file, index=False)
    print(f"  Saved {len(odds)} total odds records to {output_file}")
    return odds


def run_odds_recorder(interval_seconds=60, duration_minutes=None):
    """
    Continuously record odds at regular intervals.
    Run this during live games to build historical odds data.

    Usage:
        # Record every 60 seconds indefinitely
        python market_data.py

        # Record every 30 seconds for 3 hours
        python market_data.py --interval 30 --duration 180
    """
    print("\n" + "=" * 50)
    print("STARTING ODDS RECORDER")
    print(f"  Interval: {interval_seconds}s")
    print(f"  Duration: {'indefinite' if not duration_minutes else f'{duration_minutes} min'}")
    print("  Press Ctrl+C to stop")
    print("=" * 50)

    start_time = time.time()
    snapshot_count = 0

    try:
        while True:
            snapshot_count += 1
            print(f"\n[Snapshot #{snapshot_count} — {datetime.now().strftime('%H:%M:%S')}]")
            record_odds_snapshot()

            if duration_minutes:
                elapsed = (time.time() - start_time) / 60
                if elapsed >= duration_minutes:
                    print(f"\n  Duration reached ({duration_minutes} min). Stopping.")
                    break

            time.sleep(interval_seconds)

    except KeyboardInterrupt:
        print(f"\n\nStopped after {snapshot_count} snapshots.")


# ══════════════════════════════════════════════
# 6. ANALYZE CROSS-PLATFORM DIFFERENCES
# ══════════════════════════════════════════════

def analyze_platform_gaps():
    """
    Find games where Polymarket and Kalshi disagree
    on probability — these are arbitrage/edge opportunities.
    """
    odds_file = f"{MARKETS_DIR}/odds_history.csv"
    if not os.path.exists(odds_file):
        print("No odds history yet. Run the recorder first.")
        return

    df = pd.read_csv(odds_file)
    print(f"\nAnalyzing {len(df)} odds records...")

    # Find matching games across platforms
    # Group by question similarity (team matchup)
    poly = df[df["platform"] == "polymarket"].copy()
    kalshi = df[df["platform"] == "kalshi"].copy()

    print(f"  Polymarket: {len(poly)} records")
    print(f"  Kalshi: {len(kalshi)} records")

    # Match by team pairs
    matches = []
    for _, p in poly.iterrows():
        if p["team1"] == "???" or p["team2"] == "???":
            continue
        k_match = kalshi[
            ((kalshi["team1"] == p["team1"]) & (kalshi["team2"] == p["team2"])) |
            ((kalshi["team1"] == p["team2"]) & (kalshi["team2"] == p["team1"]))
        ]
        if not k_match.empty:
            k = k_match.iloc[0]
            gap = abs(p["implied_prob"] - k["implied_prob"])
            matches.append({
                "game": f"{p['team1']} vs {p['team2']}",
                "poly_prob": p["implied_prob"],
                "kalshi_prob": k["implied_prob"],
                "gap": gap,
                "poly_volume": p["volume"],
                "kalshi_volume": k["volume"],
            })

    if matches:
        match_df = pd.DataFrame(matches).sort_values("gap", ascending=False)
        print(f"\n  Found {len(match_df)} cross-platform matches:")
        print(f"\n  {'Game':>25s} {'Poly':>8s} {'Kalshi':>8s} {'Gap':>8s}")
        print(f"  {'-'*51}")
        for _, row in match_df.head(10).iterrows():
            print(f"  {row['game']:>25s} {row['poly_prob']:>7.1%} {row['kalshi_prob']:>7.1%} {row['gap']:>7.1%}")

        match_df.to_csv(f"{MARKETS_DIR}/platform_gaps.csv", index=False)
    else:
        print("  No cross-platform matches found yet.")


# ══════════════════════════════════════════════
# 7. COMMAND-LINE INTERFACE
# ══════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NBA Market Data Collector")
    parser.add_argument("--mode", choices=["snapshot", "record", "analyze"],
                        default="snapshot",
                        help="snapshot=one-time pull, record=continuous, analyze=compare platforms")
    parser.add_argument("--interval", type=int, default=60,
                        help="Seconds between snapshots (record mode)")
    parser.add_argument("--duration", type=int, default=None,
                        help="Minutes to record (record mode, default=indefinite)")
    args = parser.parse_args()

    if args.mode == "snapshot":
        odds = get_combined_nba_odds()
        if not odds.empty:
            print("\n  Current NBA Markets:")
            for _, row in odds.iterrows():
                print(f"    [{row['platform']:>12s}] {row['question'][:60]:60s} "
                      f"Prob: {row['implied_prob']:.1%}  Vol: ${row['volume']:,.0f}")
            odds.to_csv(f"{MARKETS_DIR}/latest_snapshot.csv", index=False)

    elif args.mode == "record":
        run_odds_recorder(
            interval_seconds=args.interval,
            duration_minutes=args.duration,
        )

    elif args.mode == "analyze":
        analyze_platform_gaps()