"""
Courtside Alpha — Trade & Model Performance Analyzer
=====================================================
Loads all collected data (trades + live observation snapshots)
and prints a comprehensive analysis of model behavior, edge quality,
and trading performance.

Usage:
    python analyze.py
"""

import sqlite3
import json
from datetime import datetime
from collections import defaultdict

TRADES_DB = "trades.sqlite"
OBS_DB    = "live_observations.sqlite"

# ── Helpers ──────────────────────────────────────────────────────────────────

def pct(v):
    return f"{v * 100:.1f}%" if v is not None else "—"

def fmt_score(home, away):
    return f"{home}–{away}" if home is not None else "—"

def period_label(p):
    if p is None: return "?"
    if p <= 4: return f"Q{p}"
    return f"OT{p - 4}"

def secs_to_clock(s):
    if s is None: return ""
    m, sec = divmod(int(s), 60)
    return f"{m}:{sec:02d}"

# ── Load Data ────────────────────────────────────────────────────────────────

def load_trades():
    conn = sqlite3.connect(TRADES_DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM simulated_trades ORDER BY timestamp"
    ).fetchall()
    wallet = conn.execute(
        "SELECT usdc_balance FROM wallet_state WHERE id = 1"
    ).fetchone()
    conn.close()
    return [dict(r) for r in rows], wallet["usdc_balance"] if wallet else None

def load_snapshots():
    conn = sqlite3.connect(OBS_DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM snapshots ORDER BY recorded_at"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

# ── Trade Analysis ───────────────────────────────────────────────────────────

def analyze_trades(trades, wallet_bal):
    print("=" * 70)
    print("  TRADE ANALYSIS")
    print("=" * 70)

    buys  = [t for t in trades if t["action"].startswith("BUY")]
    sells = [t for t in trades if t["action"].startswith("SELL")]

    won   = [t for t in trades if t["status"] == "WON"]
    lost  = [t for t in trades if t["status"] == "LOST"]
    closed_with_pnl = [t for t in trades if t["status"] == "CLOSED" and t["pnl"] is not None and t["pnl"] != 0]
    open_trades = [t for t in trades if t["status"] == "OPEN"]

    # Combine won + closed-with-positive-pnl as "wins", lost + closed-with-negative-pnl as "losses"
    wins  = [t for t in trades if t["status"] == "WON" or (t["status"] == "CLOSED" and t["pnl"] is not None and t["pnl"] > 0)]
    losses = [t for t in trades if t["status"] == "LOST" or (t["status"] == "CLOSED" and t["pnl"] is not None and t["pnl"] < 0)]

    total_pnl = sum(t["pnl"] or 0 for t in trades)
    total_staked = sum(t["stake_amount"] for t in trades)
    win_pnl = sum(t["pnl"] for t in wins)
    loss_pnl = sum(t["pnl"] for t in losses)

    print(f"\n  Total trades:     {len(trades)}")
    print(f"    BUY orders:     {len(buys)}")
    print(f"    SELL orders:    {len(sells)}")
    print(f"    Open:           {len(open_trades)}")
    print(f"    Resolved:       {len(wins) + len(losses)}")
    print(f"      Wins:         {len(wins)}")
    print(f"      Losses:       {len(losses)}")

    if len(wins) + len(losses) > 0:
        print(f"\n  Win rate:         {len(wins) / (len(wins) + len(losses)) * 100:.1f}%")
    print(f"\n  Total PnL:        ${total_pnl:+.2f}")
    print(f"    Won PnL:        ${win_pnl:+.2f}")
    print(f"    Lost PnL:       ${loss_pnl:+.2f}")
    print(f"  Total staked:     ${total_staked:,.0f}")
    if total_staked > 0:
        print(f"  ROI:              {total_pnl / total_staked * 100:+.2f}%")
    if wallet_bal is not None:
        print(f"\n  Wallet balance:   ${wallet_bal:,.2f}")
        print(f"  Net from $10k:    ${wallet_bal - 10000:+,.2f}")

    # Avg win/loss size
    if wins:
        print(f"\n  Avg win:          ${win_pnl / len(wins):+.2f}")
    if losses:
        print(f"  Avg loss:         ${loss_pnl / len(losses):+.2f}")
    if wins and losses:
        print(f"  Win/loss ratio:   {(win_pnl / len(wins)) / abs(loss_pnl / len(losses)):.2f}x")

    return buys, sells, wins, losses, open_trades


def analyze_edge_quality(trades):
    """Check: when the model sees a large edge, does it actually win more?"""
    print("\n" + "=" * 70)
    print("  EDGE QUALITY ANALYSIS")
    print("=" * 70)

    resolved = [t for t in trades
                if t["pnl"] is not None and t["pnl"] != 0
                and t["action"].startswith("SELL")]

    if not resolved:
        print("\n  No resolved sell trades to analyze.")
        return

    # Bucket by edge size at entry (look at corresponding BUY)
    # For SELL rows, the edge = |model - market| at sell time
    # We want edge at BUY time — find the BUY for same game
    buy_by_game = {}
    for t in trades:
        if t["action"].startswith("BUY"):
            gid = t["game_id"]
            buy_by_game[gid] = t

    buckets = {"0-10%": [], "10-20%": [], "20-30%": [], "30-50%": [], "50%+": []}

    for sell in resolved:
        buy = buy_by_game.get(sell["game_id"])
        if not buy:
            continue
        edge = abs(buy["model_implied_prob"] - buy["market_implied_prob"]) * 100

        if edge < 10:     buckets["0-10%"].append(sell)
        elif edge < 20:   buckets["10-20%"].append(sell)
        elif edge < 30:   buckets["20-30%"].append(sell)
        elif edge < 50:   buckets["30-50%"].append(sell)
        else:             buckets["50%+"].append(sell)

    print(f"\n  {'Edge Bucket':<14} {'Trades':>7} {'Win%':>7} {'Avg PnL':>10} {'Total PnL':>12}")
    print("  " + "─" * 52)

    for bucket, sell_trades in buckets.items():
        if not sell_trades:
            print(f"  {bucket:<14} {'0':>7}")
            continue
        w = sum(1 for s in sell_trades if s["pnl"] > 0)
        avg = sum(s["pnl"] for s in sell_trades) / len(sell_trades)
        tot = sum(s["pnl"] for s in sell_trades)
        wr = w / len(sell_trades) * 100
        print(f"  {bucket:<14} {len(sell_trades):>7} {wr:>6.1f}% ${avg:>+8.2f} ${tot:>+10.2f}")


def analyze_by_game(trades):
    """Break down performance by game."""
    print("\n" + "=" * 70)
    print("  PERFORMANCE BY GAME")
    print("=" * 70)

    games = defaultdict(lambda: {"trades": 0, "pnl": 0.0, "wins": 0, "losses": 0, "label": ""})

    for t in trades:
        gid = t["game_id"]
        games[gid]["trades"] += 1
        games[gid]["pnl"] += t["pnl"] or 0
        if t["pnl"] is not None and t["pnl"] > 0:
            games[gid]["wins"] += 1
        elif t["pnl"] is not None and t["pnl"] < 0:
            games[gid]["losses"] += 1
        if not games[gid]["label"]:
            games[gid]["label"] = t["target_team"]

    print(f"\n  {'Game':<35} {'Trades':>7} {'W':>4} {'L':>4} {'PnL':>12}")
    print("  " + "─" * 64)

    for gid, g in sorted(games.items(), key=lambda x: -x[1]["pnl"]):
        label = g["label"][:33]
        print(f"  {label:<35} {g['trades']:>7} {g['wins']:>4} {g['losses']:>4} ${g['pnl']:>+10.2f}")


def analyze_sell_reasons(trades):
    """Break down by sell trigger type."""
    print("\n" + "=" * 70)
    print("  SELL TRIGGER BREAKDOWN")
    print("=" * 70)

    reasons = defaultdict(lambda: {"count": 0, "pnl": 0.0, "wins": 0})

    for t in trades:
        if not t["action"].startswith("SELL"):
            continue
        # Extract reason from action like "SELL(TRAILING_STOP(...))"
        action = t["action"]
        if "TRAILING_STOP" in action:
            reason = "TRAILING_STOP"
        elif "EDGE_FLIP" in action:
            reason = "EDGE_FLIP"
        elif "STOP_LOSS" in action:
            reason = "STOP_LOSS"
        elif "TIME_DECAY" in action:
            reason = "TIME_DECAY"
        elif "CONFIDENCE" in action:
            reason = "CONFIDENCE_DROP"
        else:
            reason = action[:30]

        reasons[reason]["count"] += 1
        reasons[reason]["pnl"] += t["pnl"] or 0
        if t["pnl"] is not None and t["pnl"] > 0:
            reasons[reason]["wins"] += 1

    print(f"\n  {'Reason':<22} {'Count':>7} {'Win%':>7} {'Avg PnL':>10} {'Total PnL':>12}")
    print("  " + "─" * 60)

    for reason, data in sorted(reasons.items(), key=lambda x: -x[1]["pnl"]):
        wr = data["wins"] / data["count"] * 100 if data["count"] else 0
        avg = data["pnl"] / data["count"] if data["count"] else 0
        print(f"  {reason:<22} {data['count']:>7} {wr:>6.1f}% ${avg:>+8.2f} ${data['pnl']:>+10.2f}")


def analyze_home_away(trades):
    """Win rate by bought_home."""
    print("\n" + "=" * 70)
    print("  HOME vs AWAY PERFORMANCE")
    print("=" * 70)

    for side, label in [(1, "HOME"), (0, "AWAY")]:
        side_trades = [t for t in trades if t.get("bought_home") == side]
        resolved = [t for t in side_trades if t["pnl"] is not None and t["pnl"] != 0 and t["action"].startswith("SELL")]
        if not resolved:
            print(f"\n  {label}: no resolved trades")
            continue
        wins = sum(1 for t in resolved if t["pnl"] > 0)
        total_pnl = sum(t["pnl"] for t in resolved)
        print(f"\n  {label}:")
        print(f"    Total trades:   {len(side_trades)}")
        print(f"    Resolved sells: {len(resolved)}")
        print(f"    Win rate:       {wins / len(resolved) * 100:.1f}%")
        print(f"    Total PnL:      ${total_pnl:+.2f}")
        print(f"    Avg PnL:        ${total_pnl / len(resolved):+.2f}")


# ── Snapshot / Live Observation Analysis ─────────────────────────────────────

def analyze_snapshots(snapshots):
    print("\n" + "=" * 70)
    print("  LIVE OBSERVATION SNAPSHOTS")
    print("=" * 70)

    print(f"\n  Total snapshots:  {len(snapshots)}")

    # Unique games
    games = defaultdict(list)
    for s in snapshots:
        key = f"{s['home_tricode']} vs {s['away_tricode']}"
        games[key].append(s)

    print(f"  Unique games:     {len(games)}")

    # Snapshots with market data
    with_market = [s for s in snapshots if s["polymarket_home_prob"] is not None]
    print(f"  With market odds: {len(with_market)} ({len(with_market)/len(snapshots)*100:.0f}%)")

    print(f"\n  {'Game':<22} {'Snaps':>6} {'Period':>8} {'Mkt':>5} {'Model':>7} {'Edge Range':>14}")
    print("  " + "─" * 65)

    for game_label, snaps in sorted(games.items(), key=lambda x: -len(x[1])):
        periods = set(s["period"] for s in snaps if s["period"])
        period_str = "-".join(f"Q{p}" for p in sorted(periods)) if periods else "?"

        market_probs = [s["polymarket_home_prob"] for s in snaps if s["polymarket_home_prob"] is not None]
        model_probs = [s["model_win_prob"] for s in snaps if s["model_win_prob"] is not None]
        edges = [s["model_edge"] for s in snaps if s["model_edge"] is not None]

        mkt_str = f"{min(market_probs)*100:.0f}-{max(market_probs)*100:.0f}%" if market_probs else "—"
        mdl_str = f"{min(model_probs)*100:.0f}-{max(model_probs)*100:.0f}%" if model_probs else "—"
        edge_str = f"{min(edges)*100:+.0f} to {max(edges)*100:+.0f}%" if edges else "—"

        print(f"  {game_label:<22} {len(snaps):>6} {period_str:>8} {mkt_str:>5} {mdl_str:>7} {edge_str:>14}")


def analyze_model_vs_market(snapshots):
    """How does the model compare to the market across all snapshots?"""
    print("\n" + "=" * 70)
    print("  MODEL vs MARKET ACCURACY")
    print("=" * 70)

    paired = [s for s in snapshots
              if s["polymarket_home_prob"] is not None
              and s["model_win_prob"] is not None]

    if not paired:
        print("\n  No snapshots with both model and market data.")
        return

    # Check which was closer to the actual outcome
    with_outcome = [s for s in paired if s["home_won"] is not None]

    print(f"\n  Snapshots with market + model: {len(paired)}")
    print(f"  Snapshots with known outcome:  {len(with_outcome)}")

    if not with_outcome:
        # Still show model vs market divergence stats
        edges = [s["model_win_prob"] - s["polymarket_home_prob"] for s in paired]
        abs_edges = [abs(e) for e in edges]
        print(f"\n  Avg |edge| (model - market): {sum(abs_edges)/len(abs_edges)*100:.1f}%")
        print(f"  Max |edge|:                  {max(abs_edges)*100:.1f}%")
        print(f"  Model > market:              {sum(1 for e in edges if e > 0)} / {len(edges)}")
        return

    model_correct = 0
    market_correct = 0
    model_brier = 0.0
    market_brier = 0.0

    for s in with_outcome:
        actual = float(s["home_won"])
        mp = s["model_win_prob"]
        mkp = s["polymarket_home_prob"]

        model_brier += (mp - actual) ** 2
        market_brier += (mkp - actual) ** 2

        if (mp > 0.5 and actual == 1) or (mp < 0.5 and actual == 0):
            model_correct += 1
        if (mkp > 0.5 and actual == 1) or (mkp < 0.5 and actual == 0):
            market_correct += 1

    n = len(with_outcome)
    print(f"\n  Model accuracy:    {model_correct / n * 100:.1f}% ({model_correct}/{n})")
    print(f"  Market accuracy:   {market_correct / n * 100:.1f}% ({market_correct}/{n})")
    print(f"  Model Brier:       {model_brier / n:.4f} (lower = better)")
    print(f"  Market Brier:      {market_brier / n:.4f}")

    if model_brier / n < market_brier / n:
        improvement = (1 - model_brier / market_brier) * 100
        print(f"\n  Model beats market by {improvement:.1f}% (Brier score)")
    else:
        print(f"\n  Market has better calibration than model")


def analyze_model_by_period(snapshots):
    """Model accuracy by game period — is Q1 really worse?"""
    print("\n" + "=" * 70)
    print("  MODEL ACCURACY BY QUARTER")
    print("=" * 70)

    with_outcome = [s for s in snapshots
                    if s["home_won"] is not None
                    and s["model_win_prob"] is not None
                    and s["period"] is not None]

    if not with_outcome:
        print("\n  No snapshots with outcomes to analyze by period.")
        return

    periods = defaultdict(lambda: {"correct": 0, "total": 0, "brier": 0.0, "abs_edge": []})

    for s in with_outcome:
        p = s["period"]
        actual = float(s["home_won"])
        mp = s["model_win_prob"]

        periods[p]["total"] += 1
        periods[p]["brier"] += (mp - actual) ** 2
        if (mp > 0.5 and actual == 1) or (mp < 0.5 and actual == 0):
            periods[p]["correct"] += 1
        if s["polymarket_home_prob"] is not None:
            periods[p]["abs_edge"].append(abs(mp - s["polymarket_home_prob"]))

    print(f"\n  {'Period':<10} {'Snaps':>7} {'Accuracy':>10} {'Brier':>8} {'Avg |Edge|':>12}")
    print("  " + "─" * 49)

    for p in sorted(periods.keys()):
        d = periods[p]
        acc = d["correct"] / d["total"] * 100
        brier = d["brier"] / d["total"]
        avg_edge = sum(d["abs_edge"]) / len(d["abs_edge"]) * 100 if d["abs_edge"] else 0
        print(f"  {period_label(p):<10} {d['total']:>7} {acc:>9.1f}% {brier:>8.4f} {avg_edge:>11.1f}%")


def analyze_score_trades(trades):
    """Show trades that have game state context."""
    print("\n" + "=" * 70)
    print("  TRADES WITH LIVE GAME CONTEXT")
    print("=" * 70)

    with_score = [t for t in trades if t.get("home_score") is not None]

    if not with_score:
        print("\n  No trades with game state recorded yet.")
        print("  (New trades from the updated engine will include this.)")
        return

    print(f"\n  Trades with score context: {len(with_score)} / {len(trades)}")
    print(f"\n  {'Time':<20} {'Action':<22} {'Score':>10} {'Qtr':>5} {'Mkt':>7} {'Mdl':>7} {'Edge':>7} {'PnL':>8}")
    print("  " + "─" * 90)

    for t in with_score[-30:]:  # Show last 30
        ts = t["timestamp"][:19].replace("T", " ")
        action = t["action"][:20]
        score = f"{t['home_score']}-{t['away_score']}"
        qtr = period_label(t.get("period"))
        mkt = pct(t["market_implied_prob"])
        mdl = pct(t["model_implied_prob"])
        edge_val = (t["model_implied_prob"] - t["market_implied_prob"]) * 100
        edge = f"{edge_val:+.1f}%"
        pnl = f"${t['pnl']:+.2f}" if t["pnl"] is not None else "—"
        print(f"  {ts:<20} {action:<22} {score:>10} {qtr:>5} {mkt:>7} {mdl:>7} {edge:>7} {pnl:>8}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + "  COURTSIDE ALPHA — PERFORMANCE ANALYSIS".center(68) + "║")
    print("║" + f"  Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    # Load data
    trades, wallet_bal = load_trades()
    snapshots = load_snapshots()

    # Trade analysis
    analyze_trades(trades, wallet_bal)
    analyze_edge_quality(trades)
    analyze_sell_reasons(trades)
    analyze_by_game(trades)
    analyze_home_away(trades)
    analyze_score_trades(trades)

    # Live observation analysis
    analyze_snapshots(snapshots)
    analyze_model_vs_market(snapshots)
    analyze_model_by_period(snapshots)

    print("\n" + "=" * 70)
    print("  END OF REPORT")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
