"""
NBA Live Betting Model v3
=========================
Fixes critical v2 weaknesses:
  1. Time-damped live features (suppress Q1/Q2 noise)
  2. Removes training/inference mismatched features (interpolated LINEUP_PM, etc.)
  3. Temporal interaction features (model learns that time matters)
  4. Stronger regularization (max_depth 7→5, higher min_child_weight)
  5. Game-stage sample weighting (Q3/Q4 weighted higher than Q1/Q2)
  6. Stat reliability signal (explicit "enough data?" feature)
  7. Prunes 39 dead features from v2
"""

import os
import pandas as pd
import numpy as np
import json
from datetime import timedelta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = "data"


# ══════════════════════════════════════════════
# SECTION 1: DATA LOADING
# ══════════════════════════════════════════════

def load_all_data():
    """Load everything we've scraped."""
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
        "boxscore_adv": "boxscore_advanced.parquet",
    }
    for key, filename in files.items():
        try:
            data[key] = pd.read_parquet(f"{DATA_DIR}/{filename}")
            print(f"  Loaded {key}: {len(data[key])} rows")
        except FileNotFoundError:
            if key == "boxscore_adv":
                print(f"  INFO: {filename} not found — run fetch_boxscores.py to populate")
            else:
                print(f"  WARNING: {filename} not found")
            data[key] = pd.DataFrame()
    return data


# ══════════════════════════════════════════════
# SECTION 2: ROLLING TEAM FORM
#   Instead of static season averages, compute
#   rolling windows so the model sees trajectory
# ══════════════════════════════════════════════

def build_rolling_team_features(games):
    games = games.copy()
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])
    games = games.sort_values("GAME_DATE")

    stat_cols = ["PTS", "FG_PCT", "FG3_PCT", "FT_PCT",
                 "REB", "AST", "STL", "BLK", "TOV", "PLUS_MINUS"]
    stat_cols = [c for c in stat_cols if c in games.columns]

    all_rolling = []

    for team_id in games["TEAM_ID"].unique():
        team = games[games["TEAM_ID"] == team_id].sort_values("GAME_DATE").copy()

        for window in [5, 10, 20]:
            for col in stat_cols:
                # .shift(1) = only use PRIOR games, never the current one
                team[f"ROLL_{window}_{col}"] = (
                    team[col].astype(float)
                    .shift(1)
                    .rolling(window, min_periods=3)
                    .mean()
                )

            if "WL" in team.columns:
                team[f"ROLL_{window}_WIN_PCT"] = (
                    (team["WL"] == "W").astype(float)
                    .shift(1)
                    .rolling(window, min_periods=3)
                    .mean()
                )

        # Trajectory: compare recent to longer-term (both shifted)
        if "PLUS_MINUS" in stat_cols:
            team["FORM_TRAJECTORY"] = (
                team["ROLL_5_PLUS_MINUS"] - team["ROLL_20_PLUS_MINUS"]
            )
        if "WL" in team.columns:
            team["WIN_TRAJECTORY"] = (
                team["ROLL_5_WIN_PCT"] - team["ROLL_20_WIN_PCT"]
            )

        # Streak: only count games BEFORE this one
        if "WL" in team.columns:
            streaks = []
            current_streak = 0
            prev_streak = 0
            for wl in team["WL"].values:
                # Record the streak AS OF before this game
                streaks.append(prev_streak)
                # Then update for next iteration
                if wl == "W":
                    current_streak = max(1, current_streak + 1)
                else:
                    current_streak = min(-1, current_streak - 1)
                prev_streak = current_streak
            team["STREAK"] = streaks

        all_rolling.append(team)

    result = pd.concat(all_rolling, ignore_index=True)
    print(f"  Built rolling features for {result['TEAM_ID'].nunique()} teams")
    print(f"  Rolling columns added: {len([c for c in result.columns if 'ROLL_' in c or c in ['FORM_TRAJECTORY', 'WIN_TRAJECTORY', 'STREAK']])}")
    return result


# ══════════════════════════════════════════════
# SECTION 3: STATIC TEAM PROFILES
#   (kept for pre-game baseline model)
# ══════════════════════════════════════════════

def build_team_profiles(data):
    """Season-level team profiles (used for the market proxy model)."""
    pace = data["pace"].copy()
    clutch = data["clutch"].copy()

    team_features = pace.set_index("TEAM_ID")[[
        "PACE", "OFF_RATING", "DEF_RATING", "NET_RATING",
        "AST_PCT", "AST_TO", "REB_PCT", "TS_PCT", "EFG_PCT"
    ]].copy()

    if not clutch.empty and "TEAM_ID" in clutch.columns:
        clutch_cols = ["TEAM_ID"]
        for col in ["NET_RATING", "W_PCT"]:
            if col in clutch.columns:
                clutch_cols.append(col)
        clutch_feats = clutch[clutch_cols].set_index("TEAM_ID")
        clutch_feats.columns = ["CLUTCH_" + c for c in clutch_feats.columns]
        team_features = team_features.join(clutch_feats, how="left")

    # Player impact
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
                STAR_DEPENDENCY=("IMPACT", lambda x: x.max() - x.mean()),
            )
            team_features = team_features.join(team_impact, how="left")

    team_features = team_features.fillna(0)
    return team_features


# ══════════════════════════════════════════════
# SECTION 4: GAME SNAPSHOT EXTRACTION
#   Higher resolution + richer event features
# ══════════════════════════════════════════════

def parse_clock(clock_str):
    try:
        parts = str(clock_str).split(":")
        return int(float(parts[0])) * 60 + int(float(parts[1]))
    except:
        return 0


def extract_game_snapshots(pbp, games_with_rolling, snapshot_interval=90):
    """
    Sample game states every 90 seconds (up from 120 in v1).
    Adds sequence context: previous snapshot's features as lag features.
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

        # Final result
        final = game.loc[game["GAME_SECONDS_LEFT"].idxmin()]
        home_won = int(final["SCOREHOME"] > final["SCOREAWAY"])
        final_margin = final["SCOREHOME"] - final["SCOREAWAY"]

        # Quarter-level results for quarter prop trading
        quarter_margins = {}
        for q in [1, 2, 3, 4]:
            q_data = game[game["PERIOD"] == q]
            if not q_data.empty:
                q_end = q_data.loc[q_data["GAME_SECONDS_LEFT"].idxmin()]
                q_start_margin = 0 if q == 1 else quarter_margins.get(q - 1, {}).get("end_margin", 0)
                end_margin = q_end["SCOREHOME"] - q_end["SCOREAWAY"]
                quarter_margins[q] = {
                    "end_margin": end_margin,
                    "quarter_margin": end_margin - q_start_margin,
                }

        # Sample at intervals
        max_time = game["GAME_SECONDS_LEFT"].max()
        sample_times = np.arange(0, max_time, snapshot_interval)

        prev_snapshot = None  # for lag features

        for t in sample_times:
            state = game[game["GAME_SECONDS_LEFT"] >= t].tail(1)
            if state.empty:
                continue

            row = state.iloc[0]
            home_score = row["SCOREHOME"]
            away_score = row["SCOREAWAY"]
            margin = home_score - away_score
            period = int(row.get("PERIOD", 1))
            secs_left = row["GAME_SECONDS_LEFT"]

            elapsed = max(max_time - secs_left, 1)
            total_points = home_score + away_score
            scoring_pace = total_points / (elapsed / 60) if elapsed > 60 else 0

            # ── Multi-window momentum ──
            momentum = {}
            for window in [60, 120, 300]:  # 1min, 2min, 5min
                w = game[
                    (game["GAME_SECONDS_LEFT"] <= secs_left + window) &
                    (game["GAME_SECONDS_LEFT"] >= secs_left)
                ]
                if len(w) >= 2:
                    h_pts = w["SCOREHOME"].iloc[-1] - w["SCOREHOME"].iloc[0]
                    a_pts = w["SCOREAWAY"].iloc[-1] - w["SCOREAWAY"].iloc[0]
                else:
                    h_pts, a_pts = 0, 0
                momentum[f"HOME_MOM_{window}s"] = h_pts
                momentum[f"AWAY_MOM_{window}s"] = a_pts
                momentum[f"SWING_{window}s"] = h_pts - a_pts

            # ── Lead history ──
            history = game[game["GAME_SECONDS_LEFT"] >= secs_left]
            margins_so_far = history["SCOREHOME"] - history["SCOREAWAY"]
            max_home_lead = margins_so_far.max() if not margins_so_far.empty else 0
            max_away_lead = -margins_so_far.min() if not margins_so_far.empty else 0

            # Lead changes count
            if not margins_so_far.empty:
                sign_changes = (np.diff(np.sign(margins_so_far.dropna().values)) != 0).sum()
            else:
                sign_changes = 0

            # ── Fraction of game completed ──
            game_progress = 1 - (secs_left / 2880)

            # ── Quarter-level context ──
            current_quarter_secs_left = secs_left % 720 if period <= 4 else secs_left
            quarter_progress = 1 - (current_quarter_secs_left / 720)

            # ── Quarter prop labels ──
            current_q_margin = quarter_margins.get(period, {}).get("quarter_margin", np.nan)
            remaining_q_home_win = sum(
                1 for q in range(period, 5)
                if quarter_margins.get(q, {}).get("quarter_margin", 0) > 0
            )

            snap = {
                "GAME_ID": game_id,
                # ── Core state ──
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
                # ── Time interactions (v3: force temporal awareness) ──
                "MARGIN_X_PROGRESS": margin * game_progress,
                "ABS_MARGIN_X_PROGRESS": abs(margin) * game_progress,
                # Normalized margin: a 5-pt lead with 5 min left is much
                # more significant than the same lead with 40 min left
                "MARGIN_OVER_SQRT_TIME": margin / max(np.sqrt(secs_left / 60), 1.0),
                "ABS_MARGIN_OVER_SQRT_TIME": abs(margin) / max(np.sqrt(secs_left / 60), 1.0),
                # Time remaining as fraction (0=game over, 1=game start)
                "TIME_REMAINING_FRAC": secs_left / 2880,
                # Stat reliability: how many points have been scored?
                # Below 30 total points the live stats are very noisy.
                "STAT_RELIABILITY": min(total_points / 30.0, 1.0),
                # Period as ordinal (1-4+) for tree splits
                "PERIOD_ORDINAL": min(period, 5),
                # Binary flags
                "IS_Q4": int(period == 4),
                "IS_CLOSE_LATE": int((abs(margin) <= 5) and (secs_left <= 300)),
                "IS_BLOWOUT": int(abs(margin) >= 20),
                "IS_CLUTCH": int((abs(margin) <= 5) and (secs_left <= 300) and (period == 4)),
                "IS_FIRST_HALF": int(period <= 2),
                # ── Lead history ──
                "MAX_HOME_LEAD": max_home_lead,
                "MAX_AWAY_LEAD": max_away_lead,
                "LEAD_VOLATILITY": max_home_lead + max_away_lead,
                "LEAD_CHANGES": sign_changes,
                # ── Labels ──
                "HOME_WON": home_won,
                "FINAL_MARGIN": final_margin,
                "CURRENT_Q_MARGIN": current_q_margin,
                "REMAINING_Q_HOME_WINS": remaining_q_home_win,
            }
            snap.update(momentum)

            # ── Lag features from previous snapshot ──
            if prev_snapshot is not None:
                snap["LAG_MARGIN"] = prev_snapshot["MARGIN"]
                snap["MARGIN_CHANGE"] = margin - prev_snapshot["MARGIN"]
                snap["LAG_SCORING_PACE"] = prev_snapshot["SCORING_PACE"]
                snap["PACE_CHANGE"] = scoring_pace - prev_snapshot["SCORING_PACE"]
            else:
                snap["LAG_MARGIN"] = 0
                snap["MARGIN_CHANGE"] = 0
                snap["LAG_SCORING_PACE"] = 0
                snap["PACE_CHANGE"] = 0

            snapshots.append(snap)
            prev_snapshot = snap

    df = pd.DataFrame(snapshots)
    print(f"  Extracted {len(df)} snapshots from {df['GAME_ID'].nunique()} games")
    return df


# ══════════════════════════════════════════════
# SECTION 4b: RUNNING BOXSCORE FROM PBP
#   Reconstruct live boxscore stats at each
#   snapshot time using play-by-play events.
# ══════════════════════════════════════════════

def _compute_game_boxscore(game_pbp, home_id, away_id, snap_times):
    """
    Walk through a single game's PBP chronologically, maintaining
    running boxscore stats. Emit LIVE_* features at each snapshot time.

    Returns: list of dicts (one per snap_time, same order as snap_times).
    """
    if game_pbp.empty or len(snap_times) == 0:
        return [{}] * len(snap_times)

    # Sort chronologically: high secs_left (start) → low (end), then by action order
    events = game_pbp.sort_values(
        ["GAME_SECONDS_LEFT", "actionNumber"], ascending=[False, True]
    )

    # Sort snapshot times descending (earliest game moment first)
    snap_order = np.argsort(-snap_times)
    sorted_times = snap_times[snap_order]

    # ── Running team stats ──
    ts = {}
    for tid in [home_id, away_id]:
        ts[tid] = {
            "fgm": 0, "fga": 0, "fg3m": 0, "fg3a": 0,
            "ftm": 0, "fta": 0, "reb": 0, "oreb": 0,
            "ast": 0, "tov": 0, "stl": 0, "blk": 0, "pf": 0,
        }

    # Per-player tracking for foul trouble + hot/cold shooters
    player_stats = {}   # (tid, pid) → {"fgm", "fga", "fg3m", "ftm", "pf"}
    quarter_fouls = {home_id: 0, away_id: 0}
    timeouts_used = {home_id: 0, away_id: 0}
    last_missed_team = None
    current_period = 1

    def _get_player(tid, pid):
        key = (tid, pid)
        if key not in player_stats:
            player_stats[key] = {"fgm": 0, "fga": 0, "fg3m": 0, "ftm": 0, "pf": 0}
        return player_stats[key]

    def _emit():
        """Build LIVE_* feature dict from current running state."""
        feats = {}
        for side, tid in [("HOME", home_id), ("AWAY", away_id)]:
            s = ts[tid]
            fgm, fga = s["fgm"], s["fga"]
            fg3m, fg3a = s["fg3m"], s["fg3a"]
            ftm, fta = s["ftm"], s["fta"]
            pts = fgm * 2 + fg3m + ftm   # fgm counts all FG (2s as 2, 3s as 2) + extra pt per 3 + FT

            feats[f"LIVE_{side}_FG_PCT"] = fgm / fga if fga > 0 else 0
            feats[f"LIVE_{side}_FG3_PCT"] = fg3m / fg3a if fg3a > 0 else 0
            feats[f"LIVE_{side}_EFG_PCT"] = (fgm + 0.5 * fg3m) / fga if fga > 0 else 0
            tsa = fga + 0.44 * fta
            feats[f"LIVE_{side}_TS_PCT"] = pts / (2 * tsa) if tsa > 0 else 0
            feats[f"LIVE_{side}_FT_PCT"] = ftm / fta if fta > 0 else 0

            feats[f"LIVE_{side}_REB_OFF"] = s["oreb"]
            feats[f"LIVE_{side}_REB_TOTAL"] = s["reb"]
            feats[f"LIVE_{side}_AST"] = s["ast"]
            feats[f"LIVE_{side}_TOV"] = s["tov"]
            feats[f"LIVE_{side}_STL"] = s["stl"]
            feats[f"LIVE_{side}_BLK"] = s["blk"]
            feats[f"LIVE_{side}_FOULS"] = s["pf"]
            feats[f"LIVE_{side}_FT_RATE"] = fta / fga if fga > 0 else 0

            # In-bonus: 5+ team fouls in current quarter
            feats[f"LIVE_{side}_IN_BONUS"] = int(quarter_fouls.get(tid, 0) >= 5)

            # Foul trouble: players with 4+ personal fouls
            trouble = sum(
                1 for (t, _), ps in player_stats.items()
                if t == tid and ps["pf"] >= 4
            )
            feats[f"LIVE_{side}_FOUL_TROUBLE"] = trouble

            # Timeouts remaining (NBA: 7 per game)
            feats[f"LIVE_{side}_TIMEOUTS"] = max(0, 7 - timeouts_used.get(tid, 0))

            # Hot/cold shooters (players with 5+ FGA)
            hot = cold = 0
            for (t, _), ps in player_stats.items():
                if t == tid and ps["fga"] >= 5:
                    pct = ps["fgm"] / ps["fga"]
                    if pct >= 0.6:
                        hot += 1
                    elif pct <= 0.3:
                        cold += 1
            feats[f"LIVE_{side}_HOT_SHOOTERS"] = hot
            feats[f"LIVE_{side}_COLD_SHOOTERS"] = cold

            # Star player fallback from PBP: highest-FGA player.
            # For games with boxscore data, this is overwritten by the
            # boxscore overlay (highest-minutes, matching server.py inference).
            star_key = None
            max_fga = 0
            for (t, p), ps in player_stats.items():
                if t == tid and ps["fga"] > max_fga:
                    max_fga = ps["fga"]
                    star_key = (t, p)
            if star_key:
                sp = player_stats[star_key]
                feats[f"LIVE_{side}_STAR_PTS"] = sp["fgm"] * 2 + sp["fg3m"] + sp["ftm"]
                feats[f"LIVE_{side}_STAR_FOULS"] = sp["pf"]
            else:
                feats[f"LIVE_{side}_STAR_PTS"] = 0
                feats[f"LIVE_{side}_STAR_FOULS"] = 0

            # Can't compute from PBP alone
            feats[f"LIVE_{side}_STAR_PM"] = 0
            feats[f"LIVE_{side}_STAR_MINS"] = 0
            feats[f"LIVE_{side}_LINEUP_PM"] = 0
            feats[f"LIVE_{side}_PTS_PAINT"] = 0
            feats[f"LIVE_{side}_PTS_FASTBREAK"] = 0
            feats[f"LIVE_{side}_PTS_2ND"] = 0
            feats[f"LIVE_{side}_PTS_OFF_TO"] = 0
            feats[f"LIVE_{side}_BENCH_PTS"] = 0
            feats[f"LIVE_{side}_BIGGEST_LEAD"] = 0
            feats[f"LIVE_{side}_BIGGEST_RUN"] = 0

        # Differentials
        feats["LIVE_DIFF_FG_PCT"] = feats["LIVE_HOME_FG_PCT"] - feats["LIVE_AWAY_FG_PCT"]
        feats["LIVE_DIFF_FG3_PCT"] = feats["LIVE_HOME_FG3_PCT"] - feats["LIVE_AWAY_FG3_PCT"]
        feats["LIVE_DIFF_EFG_PCT"] = feats["LIVE_HOME_EFG_PCT"] - feats["LIVE_AWAY_EFG_PCT"]
        feats["LIVE_DIFF_TS_PCT"] = feats["LIVE_HOME_TS_PCT"] - feats["LIVE_AWAY_TS_PCT"]
        feats["LIVE_DIFF_REB_OFF"] = feats["LIVE_HOME_REB_OFF"] - feats["LIVE_AWAY_REB_OFF"]
        feats["LIVE_DIFF_REB_TOTAL"] = feats["LIVE_HOME_REB_TOTAL"] - feats["LIVE_AWAY_REB_TOTAL"]
        h_ast_to = feats["LIVE_HOME_AST"] / max(feats["LIVE_HOME_TOV"], 1)
        a_ast_to = feats["LIVE_AWAY_AST"] / max(feats["LIVE_AWAY_TOV"], 1)
        feats["LIVE_DIFF_AST_TO"] = h_ast_to - a_ast_to
        feats["LIVE_DIFF_TOV"] = feats["LIVE_HOME_TOV"] - feats["LIVE_AWAY_TOV"]
        feats["LIVE_DIFF_FOULS"] = feats["LIVE_HOME_FOULS"] - feats["LIVE_AWAY_FOULS"]
        feats["LIVE_DIFF_PTS_PAINT"] = 0
        feats["LIVE_DIFF_PTS_2ND"] = 0
        feats["LIVE_DIFF_BENCH_PTS"] = 0
        feats["LIVE_DIFF_LINEUP_PM"] = 0
        feats["LIVE_LEAD_CHANGES"] = 0   # already computed in core snapshot features
        feats["LIVE_TIMES_TIED"] = 0

        # Composites
        feats["LIVE_STAR_FOUL_DANGER"] = max(
            feats["LIVE_HOME_STAR_FOULS"], feats["LIVE_AWAY_STAR_FOULS"]
        )
        feats["LIVE_FOUL_TROUBLE_DIFF"] = (
            feats["LIVE_AWAY_FOUL_TROUBLE"] - feats["LIVE_HOME_FOUL_TROUBLE"]
        )
        return feats

    # ── Walk through events, emitting snapshots at boundaries ──
    results = [None] * len(snap_times)
    snap_idx = 0

    for row in events.itertuples():
        secs_left = row.GAME_SECONDS_LEFT

        # Emit snapshots for all times we've reached (secs_left dropped below them)
        while snap_idx < len(sorted_times) and sorted_times[snap_idx] >= secs_left:
            results[snap_order[snap_idx]] = _emit()
            snap_idx += 1

        # ── Process this event ──
        tid = row.teamId
        if tid not in ts:
            # Period start, jump ball, etc. — check for period change
            period = int(getattr(row, "PERIOD", current_period))
            if period != current_period:
                quarter_fouls = {home_id: 0, away_id: 0}
                current_period = period
            continue

        action = row.actionType
        desc = str(getattr(row, "description", ""))
        period = int(getattr(row, "PERIOD", current_period))
        pid = int(getattr(row, "personId", 0))

        if period != current_period:
            quarter_fouls = {home_id: 0, away_id: 0}
            current_period = period

        s = ts[tid]

        if action == "Made Shot":
            sv = int(getattr(row, "shotValue", 2))
            s["fgm"] += 1
            s["fga"] += 1
            if sv == 3:
                s["fg3m"] += 1
                s["fg3a"] += 1
            if "AST" in desc:
                s["ast"] += 1
            ps = _get_player(tid, pid)
            ps["fgm"] += 1
            ps["fga"] += 1
            if sv == 3:
                ps["fg3m"] += 1
            last_missed_team = None

        elif action == "Missed Shot":
            s["fga"] += 1
            sv = int(getattr(row, "shotValue", 2))
            if sv == 3:
                s["fg3a"] += 1
            ps = _get_player(tid, pid)
            ps["fga"] += 1
            last_missed_team = tid

        elif action == "Free Throw":
            s["fta"] += 1
            if "MISS" not in desc:
                s["ftm"] += 1
                ps = _get_player(tid, pid)
                ps["ftm"] += 1

        elif action == "Rebound":
            s["reb"] += 1
            if last_missed_team == tid:
                s["oreb"] += 1
            last_missed_team = None

        elif action == "Turnover":
            s["tov"] += 1

        elif action == "Foul":
            s["pf"] += 1
            quarter_fouls[tid] = quarter_fouls.get(tid, 0) + 1
            ps = _get_player(tid, pid)
            ps["pf"] += 1

        elif action == "Timeout":
            timeouts_used[tid] = timeouts_used.get(tid, 0) + 1

        elif action == "":
            if "STEAL" in desc:
                s["stl"] += 1
            elif "BLOCK" in desc:
                s["blk"] += 1

    # Emit remaining snapshots (after last event)
    while snap_idx < len(sorted_times):
        results[snap_order[snap_idx]] = _emit()
        snap_idx += 1

    return [r if r is not None else {} for r in results]


def enrich_snapshots_with_boxscore(snapshots, pbp, games):
    """
    Reconstruct running boxscore stats from play-by-play data
    and add LIVE_* features to each snapshot.
    """
    print("\n  Enriching snapshots with live boxscore features from PBP...")

    # Map GAME_ID → home/away team IDs
    home_map = (
        games[games["MATCHUP"].str.contains("vs.", na=False)]
        [["GAME_ID", "TEAM_ID"]]
        .drop_duplicates("GAME_ID")
        .set_index("GAME_ID")["TEAM_ID"]
        .to_dict()
    )
    away_map = (
        games[~games["MATCHUP"].str.contains("vs.", na=False)]
        [["GAME_ID", "TEAM_ID"]]
        .drop_duplicates("GAME_ID")
        .set_index("GAME_ID")["TEAM_ID"]
        .to_dict()
    )

    # Process each game
    all_live = []
    game_ids = snapshots["GAME_ID"].unique()

    for i, game_id in enumerate(game_ids):
        home_id = home_map.get(game_id)
        away_id = away_map.get(game_id)

        snap_mask = snapshots["GAME_ID"] == game_id
        n_snaps = snap_mask.sum()

        if home_id is None or away_id is None:
            all_live.extend([{}] * n_snaps)
            continue

        game_pbp = pbp[pbp["GAME_ID"] == game_id]
        snap_times = snapshots.loc[snap_mask, "GAME_SECONDS_LEFT"].values

        feats_list = _compute_game_boxscore(game_pbp, home_id, away_id, snap_times)
        all_live.extend(feats_list)

        if (i + 1) % 100 == 0:
            print(f"    Processed {i + 1}/{len(game_ids)} games...")

    # Add LIVE columns to snapshots
    live_df = pd.DataFrame(all_live, index=snapshots.index)
    for col in live_df.columns:
        snapshots[col] = live_df[col].fillna(0)

    n_live = len([c for c in snapshots.columns if c.startswith("LIVE_")])
    print(f"  Added {n_live} LIVE boxscore features to {len(snapshots)} snapshots")
    return snapshots


def damp_live_features(snapshots):
    """
    v3 FIX: Multiply all LIVE_* features by game_progress so the model
    naturally trusts them less in Q1 (where small-sample noise is extreme)
    and more in Q4 (where stats are stable).

    This creates _DAMPED versions that replace the raw LIVE_ columns.
    The raw columns are kept but the damped ones are what the model trains on.
    """
    progress = snapshots["GAME_PROGRESS"].clip(0.05, 1.0).values

    live_cols = [c for c in snapshots.columns if c.startswith("LIVE_")]
    for col in live_cols:
        snapshots[col + "_DAMPED"] = snapshots[col].values * progress

    n_damped = len(live_cols)
    print(f"  Created {n_damped} time-damped LIVE features")
    return snapshots


def drop_mismatched_features(snapshots):
    """
    v3 FIX: Remove features that are always zero from PBP during training
    but non-zero at inference time (from real boxscore API). This
    training/inference mismatch was the #1 cause of v2's instability.

    The interpolated overlay in v2 tried to fix this with linear scaling,
    but the result didn't match real inference-time distributions.
    """
    # Features that PBP can't compute — always zero in training, real at inference
    mismatched = [
        "LIVE_HOME_LINEUP_PM", "LIVE_AWAY_LINEUP_PM", "LIVE_DIFF_LINEUP_PM",
        "LIVE_HOME_STAR_PM", "LIVE_AWAY_STAR_PM",
        "LIVE_HOME_STAR_MINS", "LIVE_AWAY_STAR_MINS",
        "LIVE_HOME_PTS_PAINT", "LIVE_AWAY_PTS_PAINT", "LIVE_DIFF_PTS_PAINT",
        "LIVE_HOME_PTS_FASTBREAK", "LIVE_AWAY_PTS_FASTBREAK",
        "LIVE_HOME_PTS_2ND", "LIVE_AWAY_PTS_2ND", "LIVE_DIFF_PTS_2ND",
        "LIVE_HOME_PTS_OFF_TO", "LIVE_AWAY_PTS_OFF_TO",
        "LIVE_HOME_BENCH_PTS", "LIVE_AWAY_BENCH_PTS", "LIVE_DIFF_BENCH_PTS",
        "LIVE_HOME_BIGGEST_LEAD", "LIVE_AWAY_BIGGEST_LEAD",
        "LIVE_HOME_BIGGEST_RUN", "LIVE_AWAY_BIGGEST_RUN",
        "LIVE_LEAD_CHANGES", "LIVE_TIMES_TIED",
    ]
    # Also remove their damped versions
    to_drop = []
    for col in mismatched:
        if col in snapshots.columns:
            to_drop.append(col)
        damped = col + "_DAMPED"
        if damped in snapshots.columns:
            to_drop.append(damped)

    snapshots = snapshots.drop(columns=to_drop, errors="ignore")
    print(f"  Dropped {len(to_drop)} mismatched features (PBP≠boxscore)")
    return snapshots


# ══════════════════════════════════════════════
# SECTION 4c: OVERLAY HISTORICAL BOXSCORE DATA
#   v3: SKIP the overlay — it causes training/inference
#   mismatch. Instead we use time-damped PBP features only.
# ══════════════════════════════════════════════

def overlay_historical_boxscore(snapshots, boxscore_adv, games):
    """
    Replace the ~15 LIVE_* features that were zero (because PBP can't
    compute them) with real historical boxscore data scaled by game progress.

    Cumulative stats (pts_paint, bench_pts, etc.) are linearly interpolated:
        mid_game_value ≈ final_game_value × game_progress

    This is approximate but much better than zero — the model can learn
    the relationship between these features and outcomes.
    """
    if boxscore_adv.empty:
        print("  No boxscore_advanced data — skipping overlay")
        return snapshots

    # Build home/away team maps from games data
    home_map = (
        games[games["MATCHUP"].str.contains("vs.", na=False)]
        [["GAME_ID", "TEAM_ID"]]
        .drop_duplicates("GAME_ID")
        .set_index("GAME_ID")["TEAM_ID"]
        .to_dict()
    )

    # Build lookup: (GAME_ID) → boxscore row
    bs_lookup = {}
    for _, row in boxscore_adv.iterrows():
        bs_lookup[row["GAME_ID"]] = row

    # Column mappings: (boxscore_col_prefix, live_feature_name)
    # These are cumulative stats that scale with game progress
    cumulative_maps = [
        ("PTS_PAINT",     "PTS_PAINT"),
        ("PTS_FASTBREAK", "PTS_FASTBREAK"),
        ("PTS_2ND",       "PTS_2ND"),
        ("PTS_OFF_TO",    "PTS_OFF_TO"),
        ("BENCH_PTS",     "BENCH_PTS"),
        # Star player (highest-minutes) — aligns training with server.py inference
        ("STAR_PTS",      "STAR_PTS"),
        ("STAR_FOULS",    "STAR_FOULS"),
    ]
    # These scale with game progress but are rate-like (noisier)
    rate_maps = [
        ("STAR_PM",         "STAR_PM"),
        ("STAR_MINS_TOTAL", "STAR_MINS"),
        ("LINEUP_PM",       "LINEUP_PM"),
    ]

    filled = 0
    total_games = snapshots["GAME_ID"].nunique()

    for game_id in snapshots["GAME_ID"].unique():
        bs = bs_lookup.get(game_id)
        if bs is None:
            continue

        home_id = home_map.get(game_id)
        if home_id is None:
            continue

        mask = snapshots["GAME_ID"] == game_id
        progress = snapshots.loc[mask, "GAME_PROGRESS"].values
        # Clamp progress to [0.01, 1.0] to avoid multiplication by 0
        progress = np.clip(progress, 0.01, 1.0)

        # Determine which side is home/away in the boxscore data
        bs_home_id = bs.get("HOME_TEAM_ID", 0)
        if bs_home_id == home_id:
            home_prefix, away_prefix = "HOME", "AWAY"
        else:
            home_prefix, away_prefix = "AWAY", "HOME"

        # Fill cumulative features (scaled by game progress)
        for bs_suffix, live_suffix in cumulative_maps:
            for side, bs_side in [("HOME", home_prefix), ("AWAY", away_prefix)]:
                col = f"LIVE_{side}_{live_suffix}"
                bs_col = f"{bs_side}_{bs_suffix}"
                if col in snapshots.columns and bs_col in bs.index:
                    final_val = float(bs[bs_col])
                    snapshots.loc[mask, col] = final_val * progress

        # Fill rate/PM features (also scaled, noisier but better than 0)
        for bs_suffix, live_suffix in rate_maps:
            for side, bs_side in [("HOME", home_prefix), ("AWAY", away_prefix)]:
                col = f"LIVE_{side}_{live_suffix}"
                bs_col = f"{bs_side}_{bs_suffix}"
                if col in snapshots.columns and bs_col in bs.index:
                    final_val = float(bs[bs_col])
                    snapshots.loc[mask, col] = final_val * progress

        # Recompute differentials that were hardcoded to 0
        snapshots.loc[mask, "LIVE_DIFF_PTS_PAINT"] = (
            snapshots.loc[mask, "LIVE_HOME_PTS_PAINT"].values
            - snapshots.loc[mask, "LIVE_AWAY_PTS_PAINT"].values
        )
        snapshots.loc[mask, "LIVE_DIFF_PTS_2ND"] = (
            snapshots.loc[mask, "LIVE_HOME_PTS_2ND"].values
            - snapshots.loc[mask, "LIVE_AWAY_PTS_2ND"].values
        )
        snapshots.loc[mask, "LIVE_DIFF_BENCH_PTS"] = (
            snapshots.loc[mask, "LIVE_HOME_BENCH_PTS"].values
            - snapshots.loc[mask, "LIVE_AWAY_BENCH_PTS"].values
        )
        snapshots.loc[mask, "LIVE_DIFF_LINEUP_PM"] = (
            snapshots.loc[mask, "LIVE_HOME_LINEUP_PM"].values
            - snapshots.loc[mask, "LIVE_AWAY_LINEUP_PM"].values
        )

        filled += 1

    print(f"  Overlaid historical boxscore data for {filled}/{total_games} games")
    return snapshots


# ══════════════════════════════════════════════
# SECTION 5: MERGE ALL FEATURES
# ══════════════════════════════════════════════

def merge_all_features(snapshots, games_rolling, team_profiles, fatigue):
    """
    Attach rolling team form + static profiles + fatigue
    to each game snapshot.
    """
    games = games_rolling.copy()

    # ── Identify home/away teams ──
    home = games[games["MATCHUP"].str.contains("vs.")][["GAME_ID", "TEAM_ID"]].rename(
        columns={"TEAM_ID": "HOME_TEAM_ID"}
    ).drop_duplicates(subset=["GAME_ID"])

    away = games[~games["MATCHUP"].str.contains("vs.")][["GAME_ID", "TEAM_ID"]].rename(
        columns={"TEAM_ID": "AWAY_TEAM_ID"}
    ).drop_duplicates(subset=["GAME_ID"])

    df = snapshots.merge(home, on="GAME_ID", how="left")
    df = df.merge(away, on="GAME_ID", how="left")

    # ── Rolling features for home team ──
    rolling_cols = [c for c in games.columns if "ROLL_" in c or c in [
        "FORM_TRAJECTORY", "WIN_TRAJECTORY", "STREAK"
    ]]
    
    if rolling_cols:
        home_rolling = games[["GAME_ID", "TEAM_ID"] + rolling_cols].copy()
        home_rolling = home_rolling.rename(columns={"TEAM_ID": "HOME_TEAM_ID"})
        home_rolling.columns = [
            f"HOME_{c}" if c not in ["GAME_ID", "HOME_TEAM_ID"] else c
            for c in home_rolling.columns
        ]
        df = df.merge(home_rolling, on=["GAME_ID", "HOME_TEAM_ID"], how="left")

        away_rolling = games[["GAME_ID", "TEAM_ID"] + rolling_cols].copy()
        away_rolling = away_rolling.rename(columns={"TEAM_ID": "AWAY_TEAM_ID"})
        away_rolling.columns = [
            f"AWAY_{c}" if c not in ["GAME_ID", "AWAY_TEAM_ID"] else c
            for c in away_rolling.columns
        ]
        df = df.merge(away_rolling, on=["GAME_ID", "AWAY_TEAM_ID"], how="left")

        # Rolling differentials
        for col in rolling_cols:
            if f"HOME_{col}" in df.columns and f"AWAY_{col}" in df.columns:
                df[f"DIFF_{col}"] = df[f"HOME_{col}"] - df[f"AWAY_{col}"]

    # ── Static team profiles ──
    home_profiles = team_profiles.add_prefix("HOME_STATIC_")
    df = df.merge(home_profiles, left_on="HOME_TEAM_ID", right_index=True, how="left")

    away_profiles = team_profiles.add_prefix("AWAY_STATIC_")
    df = df.merge(away_profiles, left_on="AWAY_TEAM_ID", right_index=True, how="left")

    for col in team_profiles.columns:
        df[f"DIFF_STATIC_{col}"] = df[f"HOME_STATIC_{col}"] - df[f"AWAY_STATIC_{col}"]

    # ── Fatigue ──
    if not fatigue.empty:
        fat = fatigue[["GAME_ID", "TEAM_ID", "REST_DAYS", "IS_B2B", "GAMES_LAST_7D"]].copy()
        for prefix, id_col in [("HOME", "HOME_TEAM_ID"), ("AWAY", "AWAY_TEAM_ID")]:
            f = fat.rename(columns={
                "TEAM_ID": id_col,
                "REST_DAYS": f"{prefix}_REST_DAYS",
                "IS_B2B": f"{prefix}_IS_B2B",
                "GAMES_LAST_7D": f"{prefix}_GAMES_LAST_7D",
            })
            df = df.merge(f, on=["GAME_ID", id_col], how="left")

        df["DIFF_REST_DAYS"] = df["HOME_REST_DAYS"] - df["AWAY_REST_DAYS"]
        df["DIFF_FATIGUE"] = df["AWAY_GAMES_LAST_7D"] - df["HOME_GAMES_LAST_7D"]

    df = df.fillna(0)
    print(f"  Final feature matrix: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


# ══════════════════════════════════════════════
# SECTION 6: COMPUTE SAMPLE WEIGHTS
# ══════════════════════════════════════════════

def compute_sample_weights(df, games):
    game_dates = games[["GAME_ID", "GAME_DATE"]].drop_duplicates()
    game_dates["GAME_DATE"] = pd.to_datetime(game_dates["GAME_DATE"])
    df = df.merge(game_dates, on="GAME_ID", how="left")
    max_date = df["GAME_DATE"].max()
    df["DAYS_AGO"] = (max_date - df["GAME_DATE"]).dt.days
    df["SAMPLE_WEIGHT"] = np.exp(-0.03 * df["DAYS_AGO"])
    # v3: game stage weighting — Q3/Q4 snapshots weighted 2x Q1
    # Rationale: late-game predictions matter more for live trading,
    # and early-game stats are too noisy to train good patterns on
    df["SAMPLE_WEIGHT"] *= (0.5 + 0.5 * df["GAME_PROGRESS"].clip(0, 1))
    # Clutch/close-late boost
    df.loc[df["IS_CLUTCH"] == 1, "SAMPLE_WEIGHT"] *= 1.5
    df.loc[df["IS_CLOSE_LATE"] == 1, "SAMPLE_WEIGHT"] *= 1.2
    return df


# ══════════════════════════════════════════════
# SECTION 7: OUT-OF-FOLD PREDICTION PIPELINE
#   This is the critical fix. Every probability
#   used for edge detection is truly out-of-sample.
# ══════════════════════════════════════════════

def get_pregame_features(df):
    """
    Features available BEFORE the game starts.
    The market proxy model only sees these.
    """
    pregame_cols = [c for c in df.columns if any(
        tag in c for tag in [
            "STATIC_", "ROLL_", "TRAJECTORY", "STREAK",
            "REST_DAYS", "IS_B2B", "GAMES_LAST_7D", "DIFF_FATIGUE", "DIFF_REST_DAYS"
        ]
    )]
    return [c for c in pregame_cols if df[c].dtype in ["float64", "int64", "float32", "int32"]]


def get_live_features(df):
    """
    All features including in-game state.
    v3: Prefer _DAMPED versions of LIVE_ features over raw.
    """
    exclude = {
        "GAME_ID", "HOME_TEAM_ID", "AWAY_TEAM_ID",
        "HOME_WON", "FINAL_MARGIN", "CURRENT_Q_MARGIN", "REMAINING_Q_HOME_WINS",
        "GAME_DATE", "DAYS_AGO", "SAMPLE_WEIGHT",
        "OOF_PROXY_PROB", "OOF_LIVE_PROB", "OOF_MARGIN_PRED",
        "EDGE", "ABS_EDGE", "EDGE_PROFITABLE", "BET_EV",
    }
    # Exclude raw LIVE_ columns if their _DAMPED version exists
    damped_cols = {c for c in df.columns if c.endswith("_DAMPED")}
    raw_with_damped = {c.replace("_DAMPED", "") for c in damped_cols}
    exclude = exclude | raw_with_damped

    return sorted([
        c for c in df.columns
        if c not in exclude and df[c].dtype in ["float64", "int64", "float32", "int32"]
    ])

def generate_oof_predictions(df, pregame_features, live_features):
    """
    Generate out-of-fold predictions for BOTH the market proxy
    and live model. Each game's predictions come from a model
    that never saw that game during training.
    
    Uses game-level splits (not row-level) to prevent
    snapshots from the same game leaking across folds.
    """
    print("\n" + "=" * 50)
    print("GENERATING OUT-OF-FOLD PREDICTIONS")
    print("=" * 50)

    # Split by game, not by row — all snapshots from one game
    # must be in the same fold
    game_ids = df["GAME_ID"].unique()
    game_dates = df.groupby("GAME_ID")["GAME_DATE"].first().sort_values()
    sorted_game_ids = game_dates.index.tolist()

    # Time-based game splits (5 folds)
    n_games = len(sorted_game_ids)
    fold_size = n_games // 5

    df["OOF_PROXY_PROB"] = np.nan
    df["OOF_LIVE_PROB"] = np.nan
    df["OOF_MARGIN_PRED"] = np.nan

    for fold in range(5):
        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size if fold < 4 else n_games
        val_games = set(sorted_game_ids[val_start:val_end])
        train_games = set(sorted_game_ids) - val_games

        train_mask = df["GAME_ID"].isin(train_games)
        val_mask = df["GAME_ID"].isin(val_games)

        X_train_pre = df.loc[train_mask, pregame_features].values
        X_val_pre = df.loc[val_mask, pregame_features].values
        X_train_live = df.loc[train_mask, live_features].values
        X_val_live = df.loc[val_mask, live_features].values
        y_train = df.loc[train_mask, "HOME_WON"].values
        y_train_margin = df.loc[train_mask, "FINAL_MARGIN"].values
        w_train = df.loc[train_mask, "SAMPLE_WEIGHT"].values

        # ── Market proxy (pre-game only, heavily regularized) ──
        proxy = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7,
            min_child_weight=20, reg_alpha=0.5, reg_lambda=2.0,
            random_state=42, verbosity=0,
        )
        proxy.fit(X_train_pre, y_train)
        proxy_probs = proxy.predict_proba(X_val_pre)[:, 1]
        df.loc[val_mask, "OOF_PROXY_PROB"] = proxy_probs

        # ── Live win probability model (v3: stronger regularization) ──
        live_cls = xgb.XGBClassifier(
            n_estimators=600, max_depth=5, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.6,
            min_child_weight=15, reg_alpha=0.3, reg_lambda=2.0, gamma=0.2,
            random_state=42, verbosity=0,
        )
        live_cls.fit(X_train_live, y_train, sample_weight=w_train)
        live_probs = live_cls.predict_proba(X_val_live)[:, 1]
        df.loc[val_mask, "OOF_LIVE_PROB"] = live_probs

        # ── Live margin model (v3: stronger regularization) ──
        live_reg = xgb.XGBRegressor(
            n_estimators=600, max_depth=5, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.6,
            min_child_weight=15, reg_alpha=0.3, reg_lambda=2.0, gamma=0.2,
            random_state=42, verbosity=0,
        )
        live_reg.fit(X_train_live, y_train_margin, sample_weight=w_train)
        margin_preds = live_reg.predict(X_val_live)
        df.loc[val_mask, "OOF_MARGIN_PRED"] = margin_preds

        val_y = df.loc[val_mask, "HOME_WON"].values
        proxy_brier = brier_score_loss(val_y, proxy_probs)
        live_brier = brier_score_loss(val_y, live_probs)
        margin_mae = mean_absolute_error(
            df.loc[val_mask, "FINAL_MARGIN"].values, margin_preds
        )

        print(f"  Fold {fold+1}: Proxy Brier={proxy_brier:.4f}, "
              f"Live Brier={live_brier:.4f}, Margin MAE={margin_mae:.2f}")

    # Drop any rows that didn't get predictions (shouldn't happen)
    df = df.dropna(subset=["OOF_PROXY_PROB", "OOF_LIVE_PROB"])

    # Overall OOF metrics
    print(f"\n  Overall OOF Market Proxy Brier: "
          f"{brier_score_loss(df['HOME_WON'], df['OOF_PROXY_PROB']):.4f}")
    print(f"  Overall OOF Live Model Brier: "
          f"{brier_score_loss(df['HOME_WON'], df['OOF_LIVE_PROB']):.4f}")
    print(f"  Overall OOF Margin MAE: "
          f"{mean_absolute_error(df['FINAL_MARGIN'], df['OOF_MARGIN_PRED']):.2f}")

    return df


# ══════════════════════════════════════════════
# SECTION 8: EDGE COMPUTATION (using OOF preds)
# ══════════════════════════════════════════════

def compute_edges(df):
    """
    Compute edges using strictly out-of-fold predictions.
    No model ever sees its own training data here.
    """
    print("\n" + "=" * 50)
    print("COMPUTING EDGES (out-of-fold)")
    print("=" * 50)

    df["EDGE"] = df["OOF_LIVE_PROB"] - df["OOF_PROXY_PROB"]
    df["ABS_EDGE"] = df["EDGE"].abs()

    # Was the edge profitable?
    df["EDGE_PROFITABLE"] = np.where(
        df["EDGE"] > 0,
        df["HOME_WON"],
        1 - df["HOME_WON"]
    ).astype(int)

    # Expected value
    df["BET_EV"] = np.where(
        df["EDGE"] > 0,
        df["HOME_WON"] * (1 / df["OOF_PROXY_PROB"].clip(0.05, 0.95) - 1)
        - (1 - df["HOME_WON"]),
        (1 - df["HOME_WON"]) * (1 / (1 - df["OOF_PROXY_PROB"]).clip(0.05, 0.95) - 1)
        - df["HOME_WON"]
    )

    print(f"  Mean absolute edge: {df['ABS_EDGE'].mean():.4f}")
    print(f"  Edges > 5%: {(df['ABS_EDGE'] > 0.05).sum()} ({(df['ABS_EDGE'] > 0.05).mean()*100:.1f}%)")
    print(f"  Edges > 10%: {(df['ABS_EDGE'] > 0.10).sum()} ({(df['ABS_EDGE'] > 0.10).mean()*100:.1f}%)")
    print(f"  Edge profitable rate: {df['EDGE_PROFITABLE'].mean()*100:.1f}%")
    print(f"  Mean bet EV: {df['BET_EV'].mean():.4f}")

    # Breakdown by edge direction
    home_bets = df[df["EDGE"] > 0.03]
    away_bets = df[df["EDGE"] < -0.03]
    print(f"\n  Home-side edges (>{3}%): {len(home_bets)}, "
          f"win rate {home_bets['EDGE_PROFITABLE'].mean()*100:.1f}%")
    print(f"  Away-side edges (>{3}%): {len(away_bets)}, "
          f"win rate {away_bets['EDGE_PROFITABLE'].mean()*100:.1f}%")

    return df


# ══════════════════════════════════════════════
# SECTION 9: EDGE MODEL + BACKTEST (using OOF)
# ══════════════════════════════════════════════

def train_edge_model(df, live_features):
    """
    Train edge quality model on out-of-fold edges.
    Still uses cross-validation internally.
    """
    print("\n" + "=" * 50)
    print("TRAINING: Edge Quality Model (the money maker)")
    print("=" * 50)

    edge_df = df[df["ABS_EDGE"] > 0.03].copy()
    print(f"  Training on {len(edge_df)} snapshots with |edge| > 3%")
    print(f"  Class balance: {edge_df['EDGE_PROFITABLE'].mean()*100:.1f}% profitable")

    edge_features = live_features + [
        "EDGE", "ABS_EDGE", "OOF_PROXY_PROB", "OOF_LIVE_PROB"
    ]
    edge_features = [c for c in edge_features if c in edge_df.columns]

    X = edge_df[edge_features].values
    y = edge_df["EDGE_PROFITABLE"].values
    weights = edge_df["SAMPLE_WEIGHT"].values

    # Game-level time splits
    game_dates = edge_df.groupby("GAME_ID")["GAME_DATE"].first().sort_values()
    sorted_games = game_dates.index.tolist()
    n = len(sorted_games)
    fold_size = n // 3

    scores = []
    for fold in range(3):
        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size if fold < 2 else n
        val_games = set(sorted_games[val_start:val_end])

        train_mask = ~edge_df["GAME_ID"].isin(val_games)
        val_mask = edge_df["GAME_ID"].isin(val_games)

        y_train = y[train_mask.values]
        y_val = y[val_mask.values]

        if len(np.unique(y_val)) < 2 or len(np.unique(y_train)) < 2:
            print(f"  Fold {fold+1}: SKIPPED (single class)")
            continue

        X_train = X[train_mask.values]
        X_val = X[val_mask.values]
        w_train = weights[train_mask.values]

        model = xgb.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.7,
            min_child_weight=15, reg_alpha=0.3, reg_lambda=2.0, gamma=0.2,
            random_state=42, verbosity=0,
        )
        model.fit(X_train, y_train, sample_weight=w_train,
                  eval_set=[(X_val, y_val)], verbose=False)

        preds = model.predict_proba(X_val)[:, 1]
        score = brier_score_loss(y_val, preds)
        scores.append(score)
        print(f"  Fold {fold+1}: Brier={score:.4f}")

    if scores:
        print(f"  Avg Brier: {np.mean(scores):.4f}")

    # Final model
    final = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7,
        min_child_weight=15, reg_alpha=0.3, reg_lambda=2.0, gamma=0.2,
        random_state=42, verbosity=0,
    )
    final.fit(X, y, sample_weight=weights)
    return final, edge_features


def backtest_strategy(df, edge_model, edge_features,
                      edge_threshold=0.05, kelly_fraction=0.25):
    """Simulate trading using OOF edges only."""
    print("\n" + "=" * 50)
    print("BACKTESTING STRATEGY (out-of-fold)")
    print("=" * 50)

    trade_df = df[df["ABS_EDGE"] > edge_threshold].copy()
    if trade_df.empty:
        print("  No trades above threshold")
        return None

    edge_feats = [c for c in edge_features if c in trade_df.columns]
    trade_df["EDGE_CONFIDENCE"] = edge_model.predict_proba(
        trade_df[edge_feats].values
    )[:, 1]

    # Sweep confidence thresholds to find optimal
    print("\n  Confidence threshold sweep:")
    print(f"  {'Threshold':>10s} {'Trades':>8s} {'Win%':>8s} {'PnL':>10s} {'PnL/Trade':>10s} {'Sharpe':>8s}")
    print(f"  {'-'*56}")

    best_sharpe = -999
    best_threshold = 0.5

    for thresh in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        ct = trade_df[trade_df["EDGE_CONFIDENCE"] > thresh].copy()
        if len(ct) < 10:
            continue

        ct["KELLY_SIZE"] = (ct["EDGE_CONFIDENCE"] * 2 - 1).clip(0, 0.2) * kelly_fraction
        ct["PNL"] = ct["BET_EV"] * ct["KELLY_SIZE"]

        sharpe = ct["PNL"].mean() / (ct["PNL"].std() + 1e-8)
        print(f"  {thresh:>10.2f} {len(ct):>8d} {ct['EDGE_PROFITABLE'].mean()*100:>7.1f}% "
              f"{ct['PNL'].sum():>10.3f} {ct['PNL'].mean():>10.4f} {sharpe:>8.2f}")

        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_threshold = thresh

    print(f"\n  Best threshold: {best_threshold:.2f} (Sharpe={best_sharpe:.2f})")

    # Detailed results at best threshold
    confident_trades = trade_df[trade_df["EDGE_CONFIDENCE"] > best_threshold].copy()
    if confident_trades.empty:
        print("  No confident trades at best threshold")
        return None

    confident_trades["KELLY_SIZE"] = (
        confident_trades["EDGE_CONFIDENCE"] * 2 - 1
    ).clip(0, 0.2) * kelly_fraction
    confident_trades["PNL"] = confident_trades["BET_EV"] * confident_trades["KELLY_SIZE"]

    print(f"\n  === RESULTS AT THRESHOLD {best_threshold:.2f} ===")
    print(f"  Total trades: {len(confident_trades)}")
    print(f"  Win rate: {confident_trades['EDGE_PROFITABLE'].mean()*100:.1f}%")
    print(f"  Avg edge: {confident_trades['ABS_EDGE'].mean()*100:.1f}%")
    print(f"  Total PnL: {confident_trades['PNL'].sum():.3f}")
    print(f"  Avg PnL/trade: {confident_trades['PNL'].mean():.4f}")
    print(f"  Sharpe: {best_sharpe:.2f}")

    print("\n  By quarter:")
    for q in sorted(confident_trades["PERIOD"].unique()):
        q_trades = confident_trades[confident_trades["PERIOD"] == q]
        print(f"    Q{int(q)}: {len(q_trades)} trades, "
              f"win rate {q_trades['EDGE_PROFITABLE'].mean()*100:.1f}%, "
              f"PnL {q_trades['PNL'].sum():.3f}")

    print("\n  By edge size:")
    for lo, hi, label in [(0.05, 0.10, "5-10%"), (0.10, 0.15, "10-15%"), (0.15, 1.0, "15%+")]:
        bucket = confident_trades[
            (confident_trades["ABS_EDGE"] >= lo) & (confident_trades["ABS_EDGE"] < hi)
        ]
        if not bucket.empty:
            print(f"    {label}: {len(bucket)} trades, "
                  f"win rate {bucket['EDGE_PROFITABLE'].mean()*100:.1f}%, "
                  f"PnL {bucket['PNL'].sum():.3f}")

    return confident_trades


# ══════════════════════════════════════════════
# SECTION 10: FEATURE IMPORTANCE
# ══════════════════════════════════════════════

def print_feature_importance(model, feature_cols, top_n=20, title=""):
    importance = model.feature_importances_
    feat_imp = sorted(zip(feature_cols, importance), key=lambda x: -x[1])
    print(f"\n  Top {top_n} features{' — ' + title if title else ''}:")
    for name, imp in feat_imp[:top_n]:
        bar = "█" * int(imp * 200)
        print(f"    {name:40s} {imp:.4f} {bar}")


# ══════════════════════════════════════════════
# SECTION 11: SAVE EVERYTHING
# ══════════════════════════════════════════════

def save_all_models(models, feature_sets):
    """Save all models and feature lists."""
    for name, model in models.items():
        model.save_model(f"{DATA_DIR}/{name}.json")
        print(f"  Saved {name}.json")

    for name, features in feature_sets.items():
        with open(f"{DATA_DIR}/{name}.json", "w") as f:
            json.dump(features, f)
        print(f"  Saved {name}.json")


# ══════════════════════════════════════════════
# SECTION 12: MERGE LIVE OBSERVATIONS
# ══════════════════════════════════════════════

def merge_live_observations(df, db_path="live_observations.sqlite"):
    """
    Load live observations and merge with historical OOF data.

    Key advantage: live snapshots have real Polymarket odds (MARKET_PROB)
    instead of proxy model estimates. For these rows we set:
      OOF_PROXY_PROB = MARKET_PROB   (real market pricing)
      OOF_LIVE_PROB  = model's actual inference-time prediction
      OOF_MARGIN_PRED = model's actual margin prediction

    These are genuinely out-of-sample because the model was trained
    before these games happened.
    """
    from recorder import export_for_training

    live = export_for_training(db_path)
    if live.empty:
        return df

    # Only use snapshots with real market odds (that's the whole point)
    live = live[live["MARKET_PROB"].notna()].copy()
    if live.empty:
        print("  No live observations with market odds — skipping")
        return df

    n_games = live["GAME_ID"].nunique()
    print(f"\n  Merging {len(live)} live observations from {n_games} games "
          f"(with real market odds)")

    # Set OOF columns from real values
    live["OOF_PROXY_PROB"] = live["MARKET_PROB"]
    live["OOF_LIVE_PROB"] = live["RECORDED_MODEL_PROB"]
    live["OOF_MARGIN_PRED"] = live["RECORDED_MARGIN"]

    # Compute sample weights (live games are most recent -> highest weight)
    live["GAME_DATE"] = pd.to_datetime(live["RECORDED_AT"]).dt.tz_localize(None).dt.normalize()
    max_date = max(df["GAME_DATE"].max(), live["GAME_DATE"].max())
    live["DAYS_AGO"] = (max_date - live["GAME_DATE"]).dt.days
    live["SAMPLE_WEIGHT"] = np.exp(-0.03 * live["DAYS_AGO"])

    # Apply clutch/close-late boosts
    if "IS_CLUTCH" in live.columns:
        live.loc[live["IS_CLUTCH"] == 1, "SAMPLE_WEIGHT"] *= 1.5
    if "IS_CLOSE_LATE" in live.columns:
        live.loc[live["IS_CLOSE_LATE"] == 1, "SAMPLE_WEIGHT"] *= 1.2

    # Align columns: ensure all historical columns exist in live data
    for col in df.columns:
        if col not in live.columns:
            live[col] = 0

    # Keep only columns from historical data (maintains column order)
    live = live[df.columns].copy()

    result = pd.concat([df, live], ignore_index=True)
    print(f"  Total training data: {len(result)} snapshots "
          f"({len(df)} historical + {len(live)} live)")
    return result


# ══════════════════════════════════════════════
# SECTION 13: RUN EVERYTHING
# ══════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("NBA BETTING MODEL v3 — Time-Aware, Regularized")
    print("=" * 60)

    # ── Load ──
    print("\n[1/12] Loading data...")
    data = load_all_data()

    # ── Rolling features ──
    print("\n[2/12] Building rolling team form features...")
    games_rolling = build_rolling_team_features(data["games"])

    # ── Static profiles ──
    print("\n[3/12] Building static team profiles...")
    team_profiles = build_team_profiles(data)

    # ── Game snapshots (with v3 temporal features) ──
    print("\n[4/12] Extracting game snapshots (90s intervals)...")
    snapshots = extract_game_snapshots(
        data["pbp"], games_rolling, snapshot_interval=90
    )

    # ── Boxscore enrichment from PBP ──
    print("\n[5/12] Reconstructing live boxscore stats from PBP...")
    snapshots = enrich_snapshots_with_boxscore(
        snapshots, data["pbp"], data["games"]
    )

    # ── v3: Time-damp all LIVE features ──
    print("\n[6/12] Time-damping live features (v3 fix)...")
    snapshots = damp_live_features(snapshots)

    # ── v3: Remove mismatched features (skip boxscore overlay) ──
    print("\n[7/12] Removing train/inference mismatched features...")
    snapshots = drop_mismatched_features(snapshots)

    # ── Merge everything ──
    print("\n[8/12] Merging all features...")
    df = merge_all_features(
        snapshots, games_rolling, team_profiles, data["fatigue"]
    )

    # ── Recency + game-stage weights ──
    print("\n[9/12] Computing sample weights (v3: stage-weighted)...")
    df = compute_sample_weights(df, data["games"])

    # ── Define feature sets ──
    pregame_features = get_pregame_features(df)
    live_features = get_live_features(df)
    print(f"\n  Pre-game features: {len(pregame_features)}")
    print(f"  Live features: {len(live_features)}")

    # ── Out-of-fold predictions ──
    print("\n[10/12] Generating out-of-fold predictions...")
    df = generate_oof_predictions(df, pregame_features, live_features)

    # ── Merge live observations ──
    if os.path.exists("live_observations.sqlite"):
        df = merge_live_observations(df)

    # ── Edge computation ──
    print("\n[11/12] Computing edges and training edge model...")
    df = compute_edges(df)
    edge_model, edge_features = train_edge_model(df, live_features)
    print_feature_importance(
        edge_model, edge_features, title="Edge Quality"
    )

    # ── Backtest ──
    print("\n[12/12] Backtesting strategy...")
    results = backtest_strategy(df, edge_model, edge_features)

    # ── Train final production models on ALL data ──
    print("\n" + "=" * 60)
    print("TRAINING FINAL PRODUCTION MODELS (all data)")
    print("=" * 60)

    X_pre = df[pregame_features].values
    X_live = df[live_features].values
    y_cls = df["HOME_WON"].values
    y_margin = df["FINAL_MARGIN"].values
    w = df["SAMPLE_WEIGHT"].values

    print("\n  Training final market proxy...")
    final_proxy = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7,
        min_child_weight=20, reg_alpha=0.5, reg_lambda=2.0,
        random_state=42, verbosity=0,
    )
    final_proxy.fit(X_pre, y_cls)

    print("  Training final live win model (v3 regularization)...")
    final_win = xgb.XGBClassifier(
        n_estimators=600, max_depth=5, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.6,
        min_child_weight=15, reg_alpha=0.3, reg_lambda=2.0,
        gamma=0.2, random_state=42, verbosity=0,
    )
    final_win.fit(X_live, y_cls, sample_weight=w)

    print("  Training final margin model (v3 regularization)...")
    final_margin_mdl = xgb.XGBRegressor(
        n_estimators=600, max_depth=5, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.6,
        min_child_weight=15, reg_alpha=0.3, reg_lambda=2.0,
        gamma=0.2, random_state=42, verbosity=0,
    )
    final_margin_mdl.fit(X_live, y_margin, sample_weight=w)

    print_feature_importance(
        final_win, live_features, title="Win Probability (v3)"
    )
    print_feature_importance(
        final_margin_mdl, live_features, title="Margin (v3)"
    )

    # ── Save as v3 (separate from v2 so we can compare) ──
    print("\n" + "=" * 60)
    print("SAVING v3 MODELS")
    print("=" * 60)
    save_all_models(
        models={
            "v3_win_probability": final_win,
            "v3_margin": final_margin_mdl,
            "v3_market_proxy": final_proxy,
            "v3_edge_model": edge_model,
        },
        feature_sets={
            "v3_live_features": live_features,
            "v3_pregame_features": pregame_features,
            "v3_edge_features": edge_features,
        }
    )

    print("\n" + "=" * 60)
    print("V3 COMPLETE!")
    print("=" * 60)