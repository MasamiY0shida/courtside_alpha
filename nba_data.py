"""
NBA Hackathon Data Toolkit
==========================
Install: pip install nba_api pandas numpy
"""

import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ──────────────────────────────────────────────
# 0. HELPERS & CONFIG
# ──────────────────────────────────────────────

CURRENT_SEASON = "2025-26"
SLEEP = 0.6  # respect NBA rate limits

def safe_get(endpoint_class, **kwargs):
    """Wrapper to handle rate limits and retries."""
    for attempt in range(3):
        try:
            time.sleep(SLEEP)
            ep = endpoint_class(**kwargs)
            return ep.get_data_frames()
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            time.sleep(2 ** attempt)
    return None


# ──────────────────────────────────────────────
# 1. GAME FINDER — Get all game IDs this season
# ──────────────────────────────────────────────

from nba_api.stats.endpoints import leaguegamefinder

def get_season_games(season=CURRENT_SEASON):
    """Pull every game this season with basic metadata."""
    dfs = safe_get(
        leaguegamefinder.LeagueGameFinder,
        season_nullable=season,
        league_id_nullable="00",
        season_type_nullable="Regular Season"
    )
    games = dfs[0]
    # Add useful columns
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])
    games = games.sort_values("GAME_DATE", ascending=False)
    print(f"Found {games['GAME_ID'].nunique()} unique games")
    return games

# Usage:
# season_games = get_season_games()
# game_ids = season_games["GAME_ID"].unique().tolist()


# ──────────────────────────────────────────────
# 2. PLAY-BY-PLAY — Core data for feature extraction
# ──────────────────────────────────────────────


from nba_api.stats.endpoints import playbyplayv3


def get_play_by_play(game_id):
    """Full play-by-play for a single game (V3 endpoint)."""
    try:
        time.sleep(SLEEP)
        pbp = playbyplayv3.PlayByPlayV3(game_id=game_id)
        dfs = pbp.get_data_frames()
        if not dfs or dfs[0].empty:
            return None
        df = dfs[0]

        # V3 has different column names — normalize them
        # Common V3 columns: period, clock, timeActual, actionType,
        # scoreHome, scoreAway, description, personId, teamId, etc.
        
        # Parse game clock into seconds remaining in period
        if "clock" in df.columns:
            df["PCTIMESTRING"] = df["clock"].apply(
                lambda x: str(x).replace("PT", "").replace("M", ":").replace("S", "")
                if pd.notna(x) else "0:00"
            )
        elif "pctimestring" in df.columns.str.lower():
            # fallback if column name varies
            pass

        df["SECONDS_REMAINING"] = df["PCTIMESTRING"].apply(parse_clock)
        
        # Get period number
        period_col = [c for c in df.columns if c.lower() == "period"]
        if period_col:
            df["PERIOD"] = df[period_col[0]].astype(int)
        else:
            df["PERIOD"] = 1

        # Total seconds left in regulation
        df["GAME_SECONDS_LEFT"] = (
            (4 - df["PERIOD"]).clip(lower=0) * 720
            + df["SECONDS_REMAINING"]
        )

        # Normalize score columns
        for old, new in [("scoreHome", "SCOREHOME"), ("scoreAway", "SCOREAWAY")]:
            if old in df.columns:
                df[new] = pd.to_numeric(df[old], errors="coerce")

        # Home margin for downstream use
        if "SCOREHOME" in df.columns and "SCOREAWAY" in df.columns:
            df["HOME_MARGIN"] = df["SCOREHOME"] - df["SCOREAWAY"]

        return df

    except Exception as e:
        print(f"  PBP failed for {game_id}: {e}")
        return None

def parse_clock(clock_str):
    """Convert 'MM:SS' to seconds."""
    try:
        parts = str(clock_str).split(":")
        return int(parts[0]) * 60 + int(parts[1])
    except:
        return 0

# Usage:
# pbp = get_play_by_play("0022500001")


# ──────────────────────────────────────────────
# 3. BOX SCORES — Team & player level per game
# ──────────────────────────────────────────────

from nba_api.stats.endpoints import boxscoretraditionalv2, boxscoreadvancedv2

def get_box_score(game_id):
    """Traditional + advanced box score for a game."""
    trad = safe_get(
        boxscoretraditionalv2.BoxScoreTraditionalV2,
        game_id=game_id
    )
    adv = safe_get(
        boxscoreadvancedv2.BoxScoreAdvancedV2,
        game_id=game_id
    )
    return {
        "player_traditional": trad[0],
        "team_traditional": trad[1],
        "player_advanced": adv[0],
        "team_advanced": adv[1],
    }

# Usage:
# box = get_box_score("0022500001")
# box["player_advanced"]  # has OFF_RATING, DEF_RATING, NET_RATING, PACE, etc.


# ──────────────────────────────────────────────
# 4. QUARTER-BY-QUARTER PERFORMANCE
#    (teams that fade in Q3, surge in Q4, etc.)
# ──────────────────────────────────────────────

from nba_api.stats.endpoints import teamdashboardbygeneralsplits

def get_team_quarter_splits(team_id, season=CURRENT_SEASON):
    """Per-quarter scoring splits for a team."""
    dfs = safe_get(
        teamdashboardbygeneralsplits.TeamDashboardByGeneralSplits,
        team_id=team_id,
        season=season,
        measure_type_detailed_defense="Base",
        per_mode_detailed="PerGame"
    )
    # Index reference: varies by endpoint, but typically
    # dfs will contain overall + split tables
    # Print keys to explore:
    # for i, df in enumerate(dfs): print(f"[{i}] cols: {df.columns.tolist()[:5]}")
    return dfs

# Usage:
# quarters = get_team_quarter_splits(1610612747)  # Lakers


# ──────────────────────────────────────────────
# 5. CLUTCH TIME PERFORMANCE
#    (last 5 min, score within 5 points)
# ──────────────────────────────────────────────

from nba_api.stats.endpoints import leaguedashteamclutch

def get_clutch_stats(season=CURRENT_SEASON):
    """League-wide clutch stats (last 5 min, margin <= 5)."""
    dfs = safe_get(
        leaguedashteamclutch.LeagueDashTeamClutch,
        season=season,
        measure_type_detailed_defense="Advanced",
        per_mode_detailed="PerGame",
        clutch_time="Last 5 Minutes",
        point_diff=5,
    )
    return dfs[0]  # Team-level clutch performance

# Also available: player-level clutch
from nba_api.stats.endpoints import leaguedashplayerclutch

def get_player_clutch_stats(season=CURRENT_SEASON):
    dfs = safe_get(
        leaguedashplayerclutch.LeagueDashPlayerClutch,
        season=season,
        measure_type_detailed_defense="Advanced",
        per_mode_detailed="PerGame",
        clutch_time="Last 5 Minutes",
        point_diff=5,
    )
    return dfs[0]

# Usage:
# clutch = get_clutch_stats()
# clutch[["TEAM_NAME", "W", "L", "NET_RATING"]]


# ──────────────────────────────────────────────
# 6. LINEUP DATA — Net ratings by 5-man unit
# ──────────────────────────────────────────────

from nba_api.stats.endpoints import leaguedashlineups

def get_lineup_stats(season=CURRENT_SEASON, group_quantity=5):
    """5-man lineup stats across the league."""
    dfs = safe_get(
        leaguedashlineups.LeagueDashLineups,
        season=season,
        measure_type_detailed_defense="Advanced",
        per_mode_detailed="PerGame",
        group_quantity=group_quantity,
        # min_filter for meaningful sample size
    )
    lineups = dfs[0]
    lineups = lineups.sort_values("NET_RATING", ascending=False)
    return lineups

# For team-specific lineups:
def get_team_lineups(team_id, season=CURRENT_SEASON):
    dfs = safe_get(
        leaguedashlineups.LeagueDashLineups,
        season=season,
        team_id_nullable=team_id,
        measure_type_detailed_defense="Advanced",
        group_quantity=5,
    )
    return dfs[0]

# Usage:
# lineups = get_team_lineups(1610612747)
# lineups[["GROUP_NAME", "MIN", "NET_RATING", "OFF_RATING", "DEF_RATING"]]


# ──────────────────────────────────────────────
# 7. REST DAYS & SCHEDULE FATIGUE
# ──────────────────────────────────────────────

def compute_rest_and_fatigue(season_games):
    """
    From season_games df, compute rest days and back-to-back flags.
    Run this on the output of get_season_games().
    """
    fatigue = []
    for team_id in season_games["TEAM_ID"].unique():
        team_games = (
            season_games[season_games["TEAM_ID"] == team_id]
            .sort_values("GAME_DATE")
            .copy()
        )
        team_games["PREV_GAME_DATE"] = team_games["GAME_DATE"].shift(1)
        team_games["REST_DAYS"] = (
            (team_games["GAME_DATE"] - team_games["PREV_GAME_DATE"]).dt.days - 1
        ).fillna(3)  # default for season opener
        team_games["IS_B2B"] = (team_games["REST_DAYS"] == 0).astype(int)

        # Rolling fatigue: games in last 7 days
        team_games["GAMES_LAST_7D"] = team_games["GAME_DATE"].apply(
            lambda d: ((team_games["GAME_DATE"] >= d - timedelta(days=7))
                       & (team_games["GAME_DATE"] < d)).sum()
        )

        # Home or away
        team_games["IS_HOME"] = team_games["MATCHUP"].str.contains("vs.").astype(int)

        fatigue.append(team_games[
            ["GAME_ID", "TEAM_ID", "TEAM_NAME", "GAME_DATE",
             "REST_DAYS", "IS_B2B", "GAMES_LAST_7D", "IS_HOME"]
        ])

    return pd.concat(fatigue, ignore_index=True)

# Usage:
# season_games = get_season_games()
# fatigue = compute_rest_and_fatigue(season_games)


# ──────────────────────────────────────────────
# 8. PLAYER ON/OFF IMPACT
#    (how much does each player move the needle)
# ──────────────────────────────────────────────

from nba_api.stats.endpoints import playerdashboardbyteamperformance

def get_player_on_off(player_id, season=CURRENT_SEASON):
    """Player on-court vs off-court splits."""
    dfs = safe_get(
        playerdashboardbyteamperformance.PlayerDashboardByTeamPerformance,
        player_id=player_id,
        season=season,
        measure_type_detailed="Advanced",
        per_mode_detailed="Per100Possessions"
    )
    return dfs

# Alternative: team-level on/off for all players
from nba_api.stats.endpoints import teamplayeronoffdetails

def get_team_on_off(team_id, season=CURRENT_SEASON):
    """Every player's on/off net rating for a team."""
    dfs = safe_get(
        teamplayeronoffdetails.TeamPlayerOnOffDetails,
        team_id=team_id,
        season=season,
        measure_type_detailed_defense="Advanced",
        per_mode_detailed="Per100Possessions"
    )
    return {
        "on_court": dfs[0],   # stats when player is ON
        "off_court": dfs[1],  # stats when player is OFF
    }

# Usage:
# on_off = get_team_on_off(1610612747)
# Calculate impact: on_off["on_court"]["NET_RATING"] - on_off["off_court"]["NET_RATING"]


# ──────────────────────────────────────────────
# 9. SHOOTING SPLITS & 3PT VARIANCE
#    (detect hot/cold regression candidates)
# ──────────────────────────────────────────────

from nba_api.stats.endpoints import teamdashboardbyshootingsplits

def get_team_shooting_profile(team_id, season=CURRENT_SEASON):
    """Season-long shooting profile for regression modeling."""
    dfs = safe_get(
        teamdashboardbyshootingsplits.TeamDashboardByShootingSplits,
        team_id=team_id,
        season=season,
        measure_type_detailed_defense="Base",
        per_mode_detailed="PerGame"
    )
    return dfs  # Multiple split tables (by distance, area, etc.)

# For in-game use: compare live shooting % to these baselines
# Signal = (current_3pt_pct - season_3pt_pct) → positive = regression likely

# Usage:
# shooting = get_team_shooting_profile(1610612747)


# ──────────────────────────────────────────────
# 10. PACE & STYLE TRACKING
# ──────────────────────────────────────────────

from nba_api.stats.endpoints import leaguedashteamstats

def get_team_pace_profiles(season=CURRENT_SEASON):
    """Pace and efficiency for every team."""
    dfs = safe_get(
        leaguedashteamstats.LeagueDashTeamStats,
        season=season,
        measure_type_detailed_defense="Advanced",
        per_mode_detailed="PerGame"
    )
    teams = dfs[0]
    return teams[[
        "TEAM_ID", "TEAM_NAME",
        "PACE", "OFF_RATING", "DEF_RATING", "NET_RATING",
        "AST_PCT", "AST_TO", "REB_PCT", "TS_PCT", "EFG_PCT"
    ]]

# Usage:
# pace = get_team_pace_profiles()
# In-game: compare live pace (from pbp timestamps) to each team's season avg


# ──────────────────────────────────────────────
# 11. REFEREE TENDENCIES
# ──────────────────────────────────────────────

# NBA API doesn't have a direct ref stats endpoint,
# but ref assignments are in box score summaries

from nba_api.stats.endpoints import boxscoresummaryv3

def get_game_refs(game_id):
    """Get referee names for a game (V3 endpoint)."""
    try:
        time.sleep(SLEEP)
        bs = boxscoresummaryv3.BoxScoreSummaryV3(
            game_id=game_id,
            headers=CUSTOM_HEADERS,
            timeout=120
        )
        data = bs.get_dict()

        # V3 returns nested JSON — navigate to officials
        # Structure varies, so let's handle it flexibly
        result = data.get("boxScoreSummary", data)

        # Try common paths for officials
        officials = None
        if "officials" in result:
            officials = result["officials"]
        elif "boxScoreSummary" in data:
            officials = data["boxScoreSummary"].get("officials")

        if officials:
            ref_df = pd.DataFrame(officials)
            return ref_df
        else:
            # Fallback: dig through the response
            for key, val in result.items():
                if isinstance(val, list) and len(val) > 0:
                    if any(k in str(val[0]).lower() for k in ["official", "referee"]):
                        return pd.DataFrame(val)
            return pd.DataFrame()

    except Exception as e:
        print(f"  Refs failed for {game_id}: {e}")
        return pd.DataFrame()

def build_ref_tendency_db(game_ids, season_games):
    """
    Iterate over all games to build ref → game outcome profiles.
    Cross-reference with pace, foul counts, FTA.
    """
    ref_games = []
    for gid in game_ids:
        refs = get_game_refs(gid)
        refs["GAME_ID"] = gid
        ref_games.append(refs)

    ref_df = pd.concat(ref_games, ignore_index=True)

    # Merge with game-level stats (from box scores) to get:
    # - total fouls called
    # - total FTA
    # - pace
    # - home team win (potential home whistle bias)
    # Then group by OFFICIAL_ID to get tendencies
    return ref_df

# Usage:
# ref_data = build_ref_tendency_db(game_ids[:50], season_games)  # start small


# ──────────────────────────────────────────────
# 12. COMEBACK / BLOWOUT TENDENCIES
# ──────────────────────────────────────────────

def compute_comeback_profiles(game_ids):
    """
    For each game, track max lead and whether the leading
    team won. Builds team-level blown-lead and comeback rates.
    """
    records = []
    for gid in game_ids:
        pbp = get_play_by_play(gid)
        if pbp is None or pbp.empty:
            continue

        # Track score margin from home team's perspective
        if "HOME_MARGIN" not in pbp.columns:
            pbp["HOME_MARGIN"] = (
                pd.to_numeric(pbp.get("SCOREHOME", pbp.get("scoreHome")), errors="coerce")
                - pd.to_numeric(pbp.get("SCOREAWAY", pbp.get("scoreAway")), errors="coerce")
            )
        pbp = pbp.dropna(subset=["HOME_MARGIN"])
        if pbp.empty:
            continue

        max_home_lead = pbp["HOME_MARGIN"].max()
        max_away_lead = -pbp["HOME_MARGIN"].min()
        final_margin = pbp["HOME_MARGIN"].iloc[-1]

        records.append({
            "GAME_ID": gid,
            "MAX_HOME_LEAD": max_home_lead,
            "MAX_AWAY_LEAD": max_away_lead,
            "FINAL_MARGIN": final_margin,
            "HOME_WON": final_margin > 0,
            "HOME_BLEW_LEAD": (max_home_lead >= 15) and (final_margin < 0),
            "AWAY_BLEW_LEAD": (max_away_lead >= 15) and (final_margin > 0),
        })

    return pd.DataFrame(records)

# Usage:
# comebacks = compute_comeback_profiles(game_ids[:100])


# ──────────────────────────────────────────────
# 13. RECENCY WEIGHTING UTILITY
# ──────────────────────────────────────────────

def add_recency_weights(df, date_col="GAME_DATE", lambda_decay=0.03,
                        opponent_quality_col=None, quality_boost=1.5):
    """
    Exponential decay weighting with optional opponent quality multiplier.
    More recent games → higher weight.
    Games vs strong opponents → higher weight.
    """
    max_date = df[date_col].max()
    df["DAYS_AGO"] = (max_date - df[date_col]).dt.days
    df["RECENCY_WEIGHT"] = np.exp(-lambda_decay * df["DAYS_AGO"])

    if opponent_quality_col and opponent_quality_col in df.columns:
        # Normalize opponent quality to [1.0, quality_boost]
        oq = df[opponent_quality_col]
        normalized = 1.0 + (quality_boost - 1.0) * (
            (oq - oq.min()) / (oq.max() - oq.min() + 1e-8)
        )
        df["SAMPLE_WEIGHT"] = df["RECENCY_WEIGHT"] * normalized
    else:
        df["SAMPLE_WEIGHT"] = df["RECENCY_WEIGHT"]

    return df

# Usage:
# weighted_games = add_recency_weights(season_games)
# Pass SAMPLE_WEIGHT to XGBoost via `sample_weight` param


# ──────────────────────────────────────────────
# 14. GARBAGE TIME DETECTOR
# ──────────────────────────────────────────────

def flag_garbage_time(pbp_df, margin_threshold=25, time_threshold=360):
    """
    Flag plays that occur during garbage time.
    Default: 25+ point margin with < 6 min left (360 seconds).
    Adjust thresholds based on your analysis.
    """
    pbp_df["ABS_MARGIN"] = pbp_df["HOME_MARGIN"].abs()
    pbp_df["IS_GARBAGE_TIME"] = (
        (pbp_df["ABS_MARGIN"] >= margin_threshold)
        & (pbp_df["GAME_SECONDS_LEFT"] <= time_threshold)
        & (pbp_df["PERIOD"] == 4)
    ).astype(int)
    return pbp_df

# Usage:
# pbp = flag_garbage_time(pbp)
# train on pbp[pbp["IS_GARBAGE_TIME"] == 0] for cleaner signal


# ──────────────────────────────────────────────
# 15. MOMENTUM / RUN DETECTION
# ──────────────────────────────────────────────

def detect_runs(pbp_df, window_possessions=5, run_threshold=10):
    """
    Detect scoring runs using a rolling window over scoring events.
    A 'run' = one team outscoring the other by run_threshold
    over the last window_possessions scoring events.
    """
    scoring = pbp_df[pbp_df["SCOREHOME"].notna()].copy()
    scoring["HOME_SCORE"] = scoring["SCOREHOME"].astype(float)
    scoring["AWAY_SCORE"] = scoring["SCOREAWAY"].astype(float)
    scoring["HOME_PTS_DELTA"] = scoring["HOME_SCORE"].diff().fillna(0)
    scoring["AWAY_PTS_DELTA"] = scoring["AWAY_SCORE"].diff().fillna(0)

    # Rolling margin change over last N scoring events
    scoring["ROLLING_HOME_PTS"] = scoring["HOME_PTS_DELTA"].rolling(window_possessions).sum()
    scoring["ROLLING_AWAY_PTS"] = scoring["AWAY_PTS_DELTA"].rolling(window_possessions).sum()
    scoring["ROLLING_SWING"] = scoring["ROLLING_HOME_PTS"] - scoring["ROLLING_AWAY_PTS"]

    scoring["HOME_ON_RUN"] = (scoring["ROLLING_SWING"] >= run_threshold).astype(int)
    scoring["AWAY_ON_RUN"] = (scoring["ROLLING_SWING"] <= -run_threshold).astype(int)

    return scoring

# Usage:
# runs = detect_runs(pbp)
# runs[runs["HOME_ON_RUN"] == 1]  # moments where home team is surging


# ──────────────────────────────────────────────
# 16. LIVE GAME DATA (for real-time inference)
# ──────────────────────────────────────────────

from nba_api.live.nba.endpoints import scoreboard as live_scoreboard
from nba_api.live.nba.endpoints import playbyplay as live_pbp
from nba_api.live.nba.endpoints import boxscore as live_boxscore

def get_live_scoreboard():
    """Current live scoreboard — all games in progress."""
    sb = live_scoreboard.ScoreBoard()
    return sb.get_dict()

def get_live_pbp(game_id):
    """Live play-by-play for an in-progress game."""
    pbp = live_pbp.PlayByPlay(game_id)
    return pbp.get_dict()

def get_live_boxscore(game_id):
    """Live box score — current player/team stats."""
    bs = live_boxscore.BoxScore(game_id)
    return bs.get_dict()

# Usage (during live games):
# scoreboard = get_live_scoreboard()
# live_plays = get_live_pbp("0022500500")
# live_box = get_live_boxscore("0022500500")


# ──────────────────────────────────────────────
# 17. MASTER DATA PULL — Run this first
# ──────────────────────────────────────────────

def pull_all_historical_data(max_games=None):
    """
    Master function: pulls everything you need for training.
    Run once, save to parquet, never re-scrape.
    """
    print("Step 1: Getting season games...")
    season_games = get_season_games()
    game_ids = season_games["GAME_ID"].unique().tolist()
    if max_games:
        game_ids = game_ids[:max_games]

    print("Step 2: Computing fatigue features...")
    fatigue = compute_rest_and_fatigue(season_games)

    print("Step 3: Pulling team profiles...")
    pace_profiles = get_team_pace_profiles()
    clutch = get_clutch_stats()

    print(f"Step 4: Pulling play-by-play for {len(game_ids)} games...")
    all_pbp = []
    for i, gid in enumerate(game_ids):
        if i % 25 == 0:
            print(f"  ...game {i}/{len(game_ids)}")
        pbp = get_play_by_play(gid)
        if pbp is not None:
            all_pbp.append(pbp)

    pbp_full = pd.concat(all_pbp, ignore_index=True)

    # Save everything
    season_games.to_parquet("season_games.parquet")
    fatigue.to_parquet("fatigue.parquet")
    pace_profiles.to_parquet("pace_profiles.parquet")
    clutch.to_parquet("clutch_stats.parquet")
    pbp_full.to_parquet("play_by_play.parquet")
    print("Done! All data saved to parquet files.")

    return {
        "season_games": season_games,
        "fatigue": fatigue,
        "pace_profiles": pace_profiles,
        "clutch": clutch,
        "play_by_play": pbp_full,
    }

# Usage:
# data = pull_all_historical_data(max_games=100)  # start small to test
# data = pull_all_historical_data()                # full season when ready

# ──────────────────────────────────────────────
# ACTUALLY RUN EVERYTHING — add this at the bottom
# ──────────────────────────────────────────────

import os

OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def already_done(filename):
    """Skip steps where output file already exists."""
    path = f"{OUTPUT_DIR}/{filename}"
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  SKIPPING — {filename} already exists ({size_mb:.2f} MB)")
        return True
    return False


if __name__ == "__main__":

    # ── Step 1 ──
    print("=" * 50)
    print("STEP 1: Season games...")
    print("=" * 50)
    if already_done("season_games.parquet"):
        season_games = pd.read_parquet(f"{OUTPUT_DIR}/season_games.parquet")
    else:
        season_games = get_season_games()
        season_games.to_parquet(f"{OUTPUT_DIR}/season_games.parquet")
        season_games.to_csv(f"{OUTPUT_DIR}/season_games.csv", index=False)
        print(f"  Saved {len(season_games)} rows")

    game_ids = season_games["GAME_ID"].unique().tolist()
    team_ids = season_games["TEAM_ID"].unique().tolist()
    print(f"  {len(game_ids)} games, {len(team_ids)} teams\n")

    # ── Step 2 ──
    print("=" * 50)
    print("STEP 2: Rest & fatigue...")
    print("=" * 50)
    if already_done("fatigue.parquet"):
        fatigue = pd.read_parquet(f"{OUTPUT_DIR}/fatigue.parquet")
    else:
        fatigue = compute_rest_and_fatigue(season_games)
        fatigue.to_parquet(f"{OUTPUT_DIR}/fatigue.parquet")
        fatigue.to_csv(f"{OUTPUT_DIR}/fatigue.csv", index=False)
        print(f"  Saved {len(fatigue)} rows\n")

    # ── Step 3 ──
    print("=" * 50)
    print("STEP 3: Pace profiles...")
    print("=" * 50)
    if not already_done("pace_profiles.parquet"):
        pace = get_team_pace_profiles()
        pace.to_parquet(f"{OUTPUT_DIR}/pace_profiles.parquet")
        pace.to_csv(f"{OUTPUT_DIR}/pace_profiles.csv", index=False)
        print(f"  Saved {len(pace)} teams\n")

    # ── Step 4 ──
    print("=" * 50)
    print("STEP 4: Clutch stats...")
    print("=" * 50)
    if not already_done("clutch_stats.parquet"):
        clutch = get_clutch_stats()
        clutch.to_parquet(f"{OUTPUT_DIR}/clutch_stats.parquet")
        clutch.to_csv(f"{OUTPUT_DIR}/clutch_stats.csv", index=False)
        print(f"  Saved {len(clutch)} teams")

    if not already_done("player_clutch_stats.parquet"):
        player_clutch = get_player_clutch_stats()
        player_clutch.to_parquet(f"{OUTPUT_DIR}/player_clutch_stats.parquet")
        player_clutch.to_csv(f"{OUTPUT_DIR}/player_clutch_stats.csv", index=False)
        print(f"  Saved {len(player_clutch)} player clutch rows\n")

    # ── Step 5: Play-by-play (incremental) ──
    MAX_GAMES = None  # change to None for full season
    pull_ids = game_ids[:MAX_GAMES] if MAX_GAMES else game_ids

    print("=" * 50)
    print(f"STEP 5: Play-by-play ({len(pull_ids)} games)...")
    print("=" * 50)

    # Load existing PBP and only pull missing games
    if os.path.exists(f"{OUTPUT_DIR}/play_by_play.parquet"):
        existing_pbp = pd.read_parquet(f"{OUTPUT_DIR}/play_by_play.parquet")
        done_ids = set(existing_pbp["GAME_ID"].unique())
        remaining = [gid for gid in pull_ids if gid not in done_ids]
        print(f"  Already have {len(done_ids)} games, {len(remaining)} remaining")
    else:
        existing_pbp = pd.DataFrame()
        remaining = pull_ids

    if remaining:
        new_pbp = []
        for i, gid in enumerate(remaining):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(remaining)}...")
            pbp = get_play_by_play(gid)
            if pbp is not None and not pbp.empty:
                pbp["GAME_ID"] = gid
                new_pbp.append(pbp)

            # Checkpoint: save every 50 games so nothing is lost on interrupt
            if len(new_pbp) > 0 and len(new_pbp) % 50 == 0:
                print(f"  Checkpoint: saving {len(new_pbp)} new games...")
                new_df = pd.concat(new_pbp, ignore_index=True)
                pbp_full = pd.concat([existing_pbp, new_df], ignore_index=True)
                pbp_full.to_parquet(f"{OUTPUT_DIR}/play_by_play.parquet")
                print(f"  Saved {pbp_full['GAME_ID'].nunique()} total games to disk")

        # Final save after loop completes
        if new_pbp:
            new_df = pd.concat(new_pbp, ignore_index=True)
            pbp_full = pd.concat([existing_pbp, new_df], ignore_index=True)
            pbp_full.to_parquet(f"{OUTPUT_DIR}/play_by_play.parquet")
            print(f"  Saved {pbp_full['GAME_ID'].nunique()} total games")
    else:
        print("  All PBP already pulled")

    # ── Step 6: Lineup stats (incremental by team) ──
    print("\n" + "=" * 50)
    print("STEP 6: Lineup stats...")
    print("=" * 50)

    if os.path.exists(f"{OUTPUT_DIR}/lineup_stats.parquet"):
        existing_lineups = pd.read_parquet(f"{OUTPUT_DIR}/lineup_stats.parquet")
        done_teams = set(existing_lineups["TEAM_ID"].unique()) if "TEAM_ID" in existing_lineups.columns else set()
        remaining_teams = [t for t in team_ids if t not in done_teams]
        print(f"  Already have {len(done_teams)} teams, {len(remaining_teams)} remaining")
    else:
        existing_lineups = pd.DataFrame()
        remaining_teams = team_ids

    if remaining_teams:
        new_lineups = []
        for i, tid in enumerate(remaining_teams):
            if i % 5 == 0:
                print(f"  Progress: {i}/{len(remaining_teams)} teams...")
            lu = get_team_lineups(tid)
            if lu is not None and not lu.empty:
                lu["TEAM_ID"] = tid
                new_lineups.append(lu)

        if new_lineups:
            new_lu_df = pd.concat(new_lineups, ignore_index=True)
            lineups_full = pd.concat([existing_lineups, new_lu_df], ignore_index=True)
            lineups_full.to_parquet(f"{OUTPUT_DIR}/lineup_stats.parquet")
            lineups_full.to_csv(f"{OUTPUT_DIR}/lineup_stats.csv", index=False)
            print(f"  Saved {len(lineups_full)} total lineup rows")
    else:
        print("  All lineups already pulled")

    # ── Step 7: On/Off impact (incremental by team) ──
    print("\n" + "=" * 50)
    print("STEP 7: Player on/off impact...")
    print("=" * 50)

    if os.path.exists(f"{OUTPUT_DIR}/player_on_court.parquet"):
        existing_on = pd.read_parquet(f"{OUTPUT_DIR}/player_on_court.parquet")
        done_teams_onoff = set(existing_on["TEAM_ID"].unique())
        remaining_onoff = [t for t in team_ids if t not in done_teams_onoff]
        print(f"  Already have {len(done_teams_onoff)} teams, {len(remaining_onoff)} remaining")
    else:
        existing_on = pd.DataFrame()
        existing_off = pd.DataFrame()
        remaining_onoff = team_ids

    if remaining_onoff:
        new_on, new_off = [], []
        for i, tid in enumerate(remaining_onoff):
            if i % 5 == 0:
                print(f"  Progress: {i}/{len(remaining_onoff)} teams...")
            result = get_team_on_off(tid)
            if result and result["on_court"] is not None:
                result["on_court"]["TEAM_ID"] = tid
                result["off_court"]["TEAM_ID"] = tid
                new_on.append(result["on_court"])
                new_off.append(result["off_court"])

        if new_on:
            if not existing_on.empty:
                existing_off = pd.read_parquet(f"{OUTPUT_DIR}/player_off_court.parquet")
            on_df = pd.concat([existing_on] + new_on, ignore_index=True)
            off_df = pd.concat([existing_off] + new_off, ignore_index=True)
            on_df.to_parquet(f"{OUTPUT_DIR}/player_on_court.parquet")
            off_df.to_parquet(f"{OUTPUT_DIR}/player_off_court.parquet")
            print(f"  Saved {len(on_df)} on-court, {len(off_df)} off-court rows")
    else:
        print("  All on/off already pulled")

    # ── Step 8: Refs (skip for now — low priority) ──
    print("\n" + "=" * 50)
    print("STEP 8: Referee assignments — SKIPPED (low priority)")
    print("         Re-enable later if needed")
    print("=" * 50)

    # ── Step 9: Comeback profiles ──
    print("\n" + "=" * 50)
    print("STEP 9: Comeback profiles...")
    print("=" * 50)
    if not already_done("comeback_profiles.parquet"):
        if os.path.exists(f"{OUTPUT_DIR}/play_by_play.parquet"):
            comebacks = compute_comeback_profiles(pull_ids)
            comebacks.to_parquet(f"{OUTPUT_DIR}/comeback_profiles.parquet")
            comebacks.to_csv(f"{OUTPUT_DIR}/comeback_profiles.csv", index=False)
            print(f"  Saved {len(comebacks)} profiles")
        else:
            print("  Skipped — no PBP data yet")

    # ── Summary ──
    print("\n" + "=" * 50)
    print("DONE!")
    print("=" * 50)
    print(f"\nFiles in {os.path.abspath(OUTPUT_DIR)}/:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        size_mb = os.path.getsize(f"{OUTPUT_DIR}/{f}") / (1024 * 1024)
        print(f"  {f:40s} {size_mb:.2f} MB")