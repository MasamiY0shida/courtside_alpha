"use client";

import { useEffect, useRef, useState } from "react";
import {
  ComposedChart, Area, Line, XAxis, YAxis,
  Tooltip, Legend, ResponsiveContainer, ReferenceLine,
  BarChart, Bar, LineChart,
} from "recharts";

interface Trade {
  id:                  string;
  timestamp:           string;
  game_id:             string;
  target_team:         string;
  action:              string;
  market_implied_prob: number;
  model_implied_prob:  number;
  stake_amount:        number;
  status:              string;
  pnl:                 number | null;
  order_hash:          string | null;
  signed_tx:           string | null;
  bought_home:         boolean | null;
}

interface WalletInfo {
  address:      string;
  usdc_balance: number;
  chain_id:     number;
  chain:        string;
  initial_usdc: number;
}

interface LiveGame {
  game_id:      string;
  home_team:    string;
  away_team:    string;
  home_team_id: number;
  away_team_id: number;
  score:        { home: number; away: number };
  period:       number;
  game_clock:   string;
  predictions: {
    win_probability:   number;
    proxy_probability: number;
    predicted_margin:  number;
    edge:              number;
    abs_edge:          number;
    edge_confidence:   number;
    kelly_size:        number;
  };
  market_odds: {
    polymarket_prob:  number | null;
    market_edge:      number | null;
    market_abs_edge:  number | null;
    source:           string | null;
    volume:           number | null;
    spread:           number | null;
    total:            number | null;
  };
  signals:      unknown[];
  signal_count: number;
}

function fmt(dt: string) {
  return new Date(dt).toLocaleString(undefined, {
    month: "short", day: "numeric",
    hour: "2-digit", minute: "2-digit",
  });
}

function shortHash(h: string | null, len = 10): string {
  if (!h) return "—";
  return h.length > len + 4 ? `${h.slice(0, len)}…` : h;
}

type Tab = "overview" | "live" | "positions" | "history" | "analytics";

interface FeedEvent {
  id:        string;
  type:      "NBA_DATA" | "ML_PREDICTION" | "TRADE";
  message:   string;
  detail:    string;
  timestamp: Date;
}

interface ToastNotification {
  id:    string;
  trade: Trade;
}

export default function Dashboard() {
  const [trades, setTrades]           = useState<Trade[]>([]);
  const [wallet, setWallet]           = useState<WalletInfo | null>(null);
  const [liveGames, setLiveGames]     = useState<LiveGame[]>([]);
  const [loading, setLoading]         = useState(true);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [connected, setConnected]     = useState(false);
  const [activeTab, setActiveTab]     = useState<Tab>("overview");
  const [feedEvents, setFeedEvents]   = useState<FeedEvent[]>([]);
  const [toasts, setToasts]           = useState<ToastNotification[]>([]);
  const prevTradesRef  = useRef<Trade[]>([]);
  const prevGamesRef   = useRef<LiveGame[]>([]);
  const isFirstPollRef = useRef(true);

  useEffect(() => {
    const fetchAll = async () => {
      try {
        const [tradesRes, walletRes] = await Promise.all([
          fetch("/api/trades"),
          fetch("/api/wallet"),
        ]);
        const newTrades: Trade[] = await tradesRes.json();
        setTrades(newTrades);
        setWallet(await walletRes.json());

        // Fetch live games from server.py (non-blocking — may not be running)
        let newGames: LiveGame[] = [];
        try {
          const gamesRes = await fetch("/api/games");
          if (gamesRes.ok) {
            const data = await gamesRes.json();
            newGames = data.games ?? [];
            setLiveGames(newGames);
          }
        } catch { /* server.py may not be running */ }

        const now = new Date();

        if (!isFirstPollRef.current) {
          const newEvents: FeedEvent[] = [];

          // ── Detect new trades ──
          const prevIds = new Set(prevTradesRef.current.map(t => t.id));
          for (const trade of newTrades) {
            if (!prevIds.has(trade.id)) {
              const edge = (trade.model_implied_prob - trade.market_implied_prob) * 100;
              newEvents.push({
                id:        `trade-${trade.id}`,
                type:      "TRADE",
                message:   `${trade.action}  ${trade.target_team}`,
                detail:    `${edge >= 0 ? "+" : ""}${edge.toFixed(1)}% edge · $${trade.stake_amount} staked`,
                timestamp: now,
              });
              // Toast for each new trade (auto-dismiss after 6s)
              const toastId = `toast-${trade.id}`;
              setToasts(prev => [...prev, { id: toastId, trade }]);
              setTimeout(() => setToasts(prev => prev.filter(t => t.id !== toastId)), 6000);
            }
          }

          // ── Detect live game changes ──
          const prevGamesMap = new Map(prevGamesRef.current.map(g => [g.game_id, g]));
          for (const game of newGames) {
            const prev = prevGamesMap.get(game.game_id);
            if (!prev) {
              // New game appeared
              newEvents.push({
                id:        `nba-new-${game.game_id}-${now.getTime()}`,
                type:      "NBA_DATA",
                message:   `${game.home_team} vs ${game.away_team}`,
                detail:    `New game tracked · Q${game.period} · ${game.score.home}–${game.score.away}`,
                timestamp: now,
              });
              newEvents.push({
                id:        `pred-new-${game.game_id}-${now.getTime()}`,
                type:      "ML_PREDICTION",
                message:   `${game.home_team} win prob ${(game.predictions.win_probability * 100).toFixed(1)}%`,
                detail:    `edge ${game.predictions.edge >= 0 ? "+" : ""}${(game.predictions.edge * 100).toFixed(1)}% · conf ${(game.predictions.edge_confidence * 100).toFixed(0)}%`,
                timestamp: now,
              });
            } else {
              // Score changed
              if (prev.score.home !== game.score.home || prev.score.away !== game.score.away) {
                newEvents.push({
                  id:        `nba-score-${game.game_id}-${now.getTime()}`,
                  type:      "NBA_DATA",
                  message:   `${game.home_team} ${game.score.home}–${game.score.away} ${game.away_team}`,
                  detail:    `Q${game.period} ${game.game_clock} · score update`,
                  timestamp: now,
                });
              }
              // Prediction shifted >0.5%
              if (Math.abs(game.predictions.win_probability - prev.predictions.win_probability) > 0.005) {
                const edgePct = game.predictions.edge * 100;
                newEvents.push({
                  id:        `pred-${game.game_id}-${now.getTime()}`,
                  type:      "ML_PREDICTION",
                  message:   `${game.home_team} win prob → ${(game.predictions.win_probability * 100).toFixed(1)}%`,
                  detail:    `edge ${edgePct >= 0 ? "+" : ""}${edgePct.toFixed(1)}% · conf ${(game.predictions.edge_confidence * 100).toFixed(0)}%`,
                  timestamp: now,
                });
              }
            }
          }

          if (newEvents.length > 0) {
            setFeedEvents(prev => [...newEvents, ...prev].slice(0, 50));
          }
        } else {
          // First poll — seed the feed with a snapshot of current state
          isFirstPollRef.current = false;
          const initEvents: FeedEvent[] = [];
          if (newGames.length > 0) {
            initEvents.push({
              id:        `init-pred-${now.getTime()}`,
              type:      "ML_PREDICTION",
              message:   `XGBoost models active · ${newGames.length} game${newGames.length > 1 ? "s" : ""} tracked`,
              detail:    `265 features · 4 models per game`,
              timestamp: now,
            });
            initEvents.push({
              id:        `init-games-${now.getTime()}`,
              type:      "NBA_DATA",
              message:   newGames.map(g => `${g.home_team} vs ${g.away_team}`).join(" · "),
              detail:    `NBA live scoreboard · polling every 30s`,
              timestamp: now,
            });
          }
          if (newTrades.length > 0) {
            initEvents.push({
              id:        `init-trades-${now.getTime()}`,
              type:      "TRADE",
              message:   `${newTrades.length} trade${newTrades.length > 1 ? "s" : ""} on record`,
              detail:    `Latest: ${newTrades[0]?.target_team ?? "—"} · ${newTrades[0]?.action ?? ""}`,
              timestamp: now,
            });
          }
          if (initEvents.length > 0) setFeedEvents(initEvents);
        }

        prevTradesRef.current = newTrades;
        prevGamesRef.current  = newGames;
        setLastUpdated(now);
        setConnected(true);
      } catch {
        setConnected(false);
      } finally {
        setLoading(false);
      }
    };

    fetchAll();
    const interval = setInterval(fetchAll, 3000);
    return () => clearInterval(interval);
  }, []);

  // ── Derived stats ──
  const isWon  = (t: Trade) => t.status === "WON"  || (t.status === "CLOSED" && t.pnl != null && t.pnl > 0);
  const isLost = (t: Trade) => t.status === "LOST" || (t.status === "CLOSED" && t.pnl != null && t.pnl < 0);
  const won         = trades.filter(isWon).length;
  const lost        = trades.filter(isLost).length;
  const open        = trades.filter(t => t.status === "OPEN").length;
  const resolved    = won + lost;
  const winRate     = resolved > 0 ? (won / resolved) * 100 : 0;
  const totalPnl    = trades.reduce((s, t) => s + (t.pnl ?? 0), 0);
  const totalStaked = trades.reduce((s, t) => s + t.stake_amount, 0);
  const avgEdge     = trades.length > 0
    ? trades.reduce((s, t) => s + Math.abs(alignedProbs(t).edge), 0) / trades.length * 100
    : null;
  const signedCount = trades.filter(t => t.order_hash).length;

  // ROI and avg stake
  const roi      = totalStaked > 0 ? (totalPnl / totalStaked) * 100 : null;
  const avgStake = trades.length > 0 ? totalStaked / trades.length : null;

  // Current streak (trades assumed newest-first from API)
  const resolvedOrdered = trades.filter(t => t.status === "WON" || t.status === "LOST" || t.status === "CLOSED");
  let streakCount = 0;
  let streakType: "W" | "L" | null = null;
  for (const t of resolvedOrdered) {
    const s = isWon(t) ? "W" : "L";
    if (streakType === null) { streakType = s; streakCount = 1; }
    else if (s === streakType) { streakCount++; }
    else break;
  }

  // Best / worst resolved trade by PnL
  const resolvedWithPnl = trades.filter(t => t.pnl != null);
  const bestTrade  = resolvedWithPnl.length > 0 ? resolvedWithPnl.reduce((b, t) => t.pnl! > b.pnl! ? t : b) : null;
  const lostTrades = resolvedWithPnl.filter(t => t.pnl! < 0);
  const worstTrade = lostTrades.length > 0 ? lostTrades.reduce((w, t) => t.pnl! < w.pnl! ? t : w) : null;

  // PnL split by outcome
  const wonPnl  = trades.filter(isWon).reduce((s, t) => s + (t.pnl ?? 0), 0);
  const lostPnl = trades.filter(isLost).reduce((s, t) => s + (t.pnl ?? 0), 0);

  // Win rate by action type
  const buyHomeTrades  = trades.filter(t => t.action !== "BUY_AWAY" && (t.status === "WON" || t.status === "LOST" || t.status === "CLOSED"));
  const buyHomeWins    = buyHomeTrades.filter(isWon).length;
  const buyHomeWinRate = buyHomeTrades.length > 0 ? (buyHomeWins / buyHomeTrades.length) * 100 : null;
  const buyAwayTrades  = trades.filter(t => t.action === "BUY_AWAY" && (t.status === "WON" || t.status === "LOST" || t.status === "CLOSED"));
  const buyAwayWins    = buyAwayTrades.filter(isWon).length;
  const buyAwayWinRate = buyAwayTrades.length > 0 ? (buyAwayWins / buyAwayTrades.length) * 100 : null;

  // Most targeted team
  const teamCounts       = trades.reduce((acc, t) => { acc[t.target_team] = (acc[t.target_team] ?? 0) + 1; return acc; }, {} as Record<string, number>);
  const mostTargetedTeam = Object.entries(teamCounts).sort((a, b) => b[1] - a[1])[0] ?? null;

  // Last trade
  const lastTrade = trades.length > 0 ? trades[0] : null;

  // Cumulative PnL sparkline (chronological order)
  const sparkData = (() => {
    let running = 0;
    return [...trades].reverse().map(t => { running += t.pnl ?? 0; return { pnl: +running.toFixed(2) }; });
  })();

  // Market edge stats from live games
  const gamesWithMarket = liveGames.filter(g => g.market_odds?.polymarket_prob != null);
  const avgMarketEdge = gamesWithMarket.length > 0
    ? gamesWithMarket.reduce((s, g) => s + Math.abs(g.market_odds.market_edge ?? 0), 0) / gamesWithMarket.length * 100
    : null;

  // Live signal stats
  const totalSignals   = liveGames.reduce((s, g) => s + g.signal_count, 0);
  const gamesInPlay    = liveGames.filter(g => g.period >= 1 && g.period <= 4).length;
  const bestLiveGame   = liveGames.length > 0
    ? liveGames.reduce((best, g) => {
        const ea = Math.abs(g.market_odds?.market_edge ?? g.predictions.edge);
        const eb = Math.abs(best.market_odds?.market_edge ?? best.predictions.edge);
        return ea > eb ? g : best;
      })
    : null;
  const bestLiveEdgePct = bestLiveGame
    ? Math.abs(bestLiveGame.market_odds?.market_edge ?? bestLiveGame.predictions.edge) * 100
    : 0;

  const openTrades = trades.filter(t => t.status === "OPEN");

  return (
    <main className="min-h-screen bg-gray-950 text-gray-100 font-mono">

      {/* ── Sticky header ── */}
      <div className="sticky top-0 z-10 border-b border-gray-800 bg-gray-950/90 backdrop-blur">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div>
            <h1 className="text-lg font-bold text-green-400 tracking-widest">COURTSIDE ALPHA</h1>
            <p className="text-xs text-gray-600 mt-0.5">Quantitative prediction market engine · testnet order signing</p>
          </div>
          <div className="flex items-center gap-4 text-xs">
            {lastUpdated && (
              <span className="text-gray-600 hidden sm:block">
                Updated {lastUpdated.toLocaleTimeString()}
              </span>
            )}
            <span className={`flex items-center gap-1.5 font-bold ${connected ? "text-green-400" : "text-red-400"}`}>
              <span className={`w-2 h-2 rounded-full ${connected ? "bg-green-400 animate-pulse" : "bg-red-500"}`} />
              {connected ? "LIVE" : "OFFLINE"}
            </span>
          </div>
        </div>
      </div>

      {/* ── Tab bar ── */}
      <div className="border-b border-gray-800 bg-gray-950">
        <div className="max-w-7xl mx-auto px-6">
          <nav className="flex gap-1 -mb-px">
            {(
              [
                { id: "overview",   label: "Overview" },
                { id: "live",       label: "Live Games",       badge: liveGames.length > 0 ? liveGames.length : undefined },
                { id: "positions",  label: "Active Positions", badge: open > 0 ? open : undefined },
                { id: "history",    label: "Trade History",    badge: trades.length > 0 ? trades.length : undefined },
                { id: "analytics",  label: "Analytics" },
              ] as { id: Tab; label: string; badge?: number }[]
            ).map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-4 py-3 text-xs font-bold uppercase tracking-widest border-b-2 transition-colors flex items-center gap-2 ${
                  activeTab === tab.id
                    ? "border-green-400 text-green-400"
                    : "border-transparent text-gray-500 hover:text-gray-300 hover:border-gray-700"
                }`}
              >
                {tab.label}
                {tab.badge != null && (
                  <span className="bg-yellow-500/20 text-yellow-400 border border-yellow-700/40 rounded-full px-1.5 py-px text-xs leading-none">
                    {tab.badge}
                  </span>
                )}
              </button>
            ))}
          </nav>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-8 space-y-10">

        {/* ── Overview tab ── */}
        {activeTab === "overview" && (
          <>
            {/* Wallet panel */}
            {wallet && (
              <section>
                <SectionHeader label="Testnet Wallet" color="text-cyan-400" count={-1} />
                <div className="bg-gray-900 border border-cyan-900/40 rounded-lg p-5 space-y-3">
                  <div className="flex flex-wrap items-center gap-x-8 gap-y-2 text-xs">
                    <DataPoint label="Address">
                      <span className="text-cyan-300 font-semibold">{wallet.address}</span>
                    </DataPoint>
                    <DataPoint label="Chain">
                      <span className="text-gray-400">{wallet.chain} (ID {wallet.chain_id})</span>
                    </DataPoint>
                    <DataPoint label="Signed orders">
                      <span className="text-purple-400 font-semibold">{signedCount}</span>
                    </DataPoint>
                  </div>
                  <div className="flex items-end gap-6 pt-1">
                    <div>
                      <p className="text-gray-500 text-xs uppercase tracking-widest mb-1">Fake USDC Balance</p>
                      <p className={`text-3xl font-bold ${wallet.usdc_balance >= wallet.initial_usdc ? "text-green-400" : wallet.usdc_balance >= wallet.initial_usdc * 0.8 ? "text-yellow-400" : "text-red-400"}`}>
                        ${wallet.usdc_balance.toFixed(2)}
                      </p>
                      <p className="text-gray-600 text-xs mt-0.5">
                        started at ${wallet.initial_usdc.toLocaleString()} ·{" "}
                        <span className={wallet.usdc_balance >= wallet.initial_usdc ? "text-green-500" : "text-red-500"}>
                          {wallet.usdc_balance >= wallet.initial_usdc ? "+" : ""}
                          ${(wallet.usdc_balance - wallet.initial_usdc).toFixed(2)} net
                        </span>
                      </p>
                    </div>
                    <div className="h-12 w-px bg-gray-800" />
                    <div>
                      <p className="text-gray-500 text-xs uppercase tracking-widest mb-1">EIP-712 Signing</p>
                      <p className="text-xs text-gray-300">
                        Polymarket CLOB orders are signed with a secp256k1 test key.<br />
                        Each order hash is stored in the DB — ready for CLOB submission.
                      </p>
                    </div>
                  </div>
                </div>
              </section>
            )}

            {/* Stats grid — 8 cards */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <StatCard
                label="Simulated PnL"
                value={`${totalPnl >= 0 ? "+" : ""}$${totalPnl.toFixed(2)}`}
                color={totalPnl >= 0 ? "text-green-400" : "text-red-400"}
                sub={`$${totalStaked.toFixed(0)} total staked`}
              />
              <StatCard
                label="ROI"
                value={roi != null ? `${roi >= 0 ? "+" : ""}${roi.toFixed(2)}%` : "—"}
                color={roi != null ? (roi >= 0 ? "text-green-400" : "text-red-400") : "text-gray-500"}
                sub={`on $${totalStaked.toFixed(0)} staked`}
              />
              <StatCard
                label="Win Rate"
                value={`${winRate.toFixed(1)}%`}
                color="text-blue-400"
                sub={resolved > 0 ? `${won}W / ${lost}L` : "No resolved bets yet"}
              />
              <StatCard
                label="Avg Stake"
                value={avgStake != null ? `$${avgStake.toFixed(0)}` : "—"}
                color="text-gray-300"
                sub={`${trades.length} trades total`}
              />
              <StatCard
                label="Open Positions"
                value={String(open)}
                color="text-yellow-400"
                sub={open > 0 ? "awaiting resolution" : "none active"}
              />
              <StatCard
                label="Live Games"
                value={String(liveGames.length)}
                color="text-cyan-400"
                sub={gamesWithMarket.length > 0 ? `${gamesWithMarket.length} with market odds` : "no market data"}
              />
              <StatCard
                label="Avg Edge"
                value={avgEdge != null ? `${avgEdge.toFixed(1)}%` : "—"}
                color="text-purple-400"
                sub="model vs market (trades)"
              />
              <StatCard
                label="Market Edge"
                value={avgMarketEdge != null ? `${avgMarketEdge.toFixed(1)}%` : "—"}
                color="text-orange-400"
                sub="live model vs Polymarket"
              />
            </div>

            {/* Cumulative PnL sparkline + performance breakdown */}
            <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
              <div className="xl:col-span-2 bg-gray-900 border border-gray-800 rounded-lg p-5">
                <p className="text-xs text-gray-500 uppercase tracking-widest mb-1">Cumulative PnL</p>
                <p className="text-xs text-gray-600 mb-4">Running total profit/loss across all trades in chronological order.</p>
                <SparklineChart data={sparkData} />
              </div>
              <div className="space-y-4">
                {/* Resolved vs open ratio */}
                <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                  <p className="text-xs text-gray-500 uppercase tracking-widest mb-3">Trade Breakdown</p>
                  {trades.length > 0 ? (
                    <>
                      <div className="flex gap-0.5 h-2.5 rounded-full overflow-hidden mb-2">
                        {won  > 0 && <div className="bg-green-500"  style={{ flex: won  }} />}
                        {lost > 0 && <div className="bg-red-500"    style={{ flex: lost }} />}
                        {open > 0 && <div className="bg-yellow-500" style={{ flex: open }} />}
                      </div>
                      <div className="flex gap-4 text-xs">
                        <span className="text-green-400">{won} WON</span>
                        <span className="text-red-400">{lost} LOST</span>
                        <span className="text-yellow-400">{open} OPEN</span>
                      </div>
                      <div className="mt-3 pt-3 border-t border-gray-800 grid grid-cols-2 gap-2">
                        <DataPoint label="Won PnL">
                          <span className="text-green-400 font-bold">+${wonPnl.toFixed(2)}</span>
                        </DataPoint>
                        <DataPoint label="Lost PnL">
                          <span className="text-red-400 font-bold">${lostPnl.toFixed(2)}</span>
                        </DataPoint>
                      </div>
                    </>
                  ) : (
                    <p className="text-gray-600 text-xs">No trades yet</p>
                  )}
                </div>
                {/* Current streak */}
                <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                  <p className="text-xs text-gray-500 uppercase tracking-widest mb-2">Current Streak</p>
                  {streakType ? (
                    <p className={`text-3xl font-bold ${streakType === "W" ? "text-green-400" : "text-red-400"}`}>
                      {streakType}{streakCount}
                    </p>
                  ) : (
                    <p className="text-gray-600 text-sm">No resolved trades yet</p>
                  )}
                </div>
              </div>
            </div>

            {/* Trade insights */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className={`bg-gray-900 rounded-lg p-4 ${bestTrade ? "border border-green-900/30" : "border border-gray-800"}`}>
                <p className="text-xs text-gray-500 uppercase tracking-widest mb-2">Best Trade</p>
                {bestTrade ? (
                  <>
                    <p className="text-green-400 font-bold text-xl">+${bestTrade.pnl!.toFixed(2)}</p>
                    <p className="text-gray-400 text-xs mt-1 truncate" title={bestTrade.target_team}>{bestTrade.target_team}</p>
                  </>
                ) : <p className="text-gray-600 text-sm">—</p>}
              </div>
              <div className={`bg-gray-900 rounded-lg p-4 ${worstTrade ? "border border-red-900/30" : "border border-gray-800"}`}>
                <p className="text-xs text-gray-500 uppercase tracking-widest mb-2">Worst Trade</p>
                {worstTrade ? (
                  <>
                    <p className="text-red-400 font-bold text-xl">${worstTrade.pnl!.toFixed(2)}</p>
                    <p className="text-gray-400 text-xs mt-1 truncate" title={worstTrade.target_team}>{worstTrade.target_team}</p>
                  </>
                ) : <p className="text-gray-600 text-sm">—</p>}
              </div>
              <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <p className="text-xs text-gray-500 uppercase tracking-widest mb-2">Most Traded</p>
                {mostTargetedTeam ? (
                  <>
                    <p className="text-gray-200 font-bold text-sm truncate" title={mostTargetedTeam[0]}>{mostTargetedTeam[0]}</p>
                    <p className="text-gray-600 text-xs mt-1">{mostTargetedTeam[1]} trades</p>
                  </>
                ) : <p className="text-gray-600 text-sm">—</p>}
              </div>
              <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <p className="text-xs text-gray-500 uppercase tracking-widest mb-2">Last Trade</p>
                {lastTrade ? (
                  <>
                    <p className="text-gray-200 font-bold">{timeSince(lastTrade.timestamp)}</p>
                    <p className="text-gray-600 text-xs mt-1 truncate" title={lastTrade.target_team}>{lastTrade.target_team}</p>
                  </>
                ) : <p className="text-gray-600 text-sm">—</p>}
              </div>
            </div>

            {/* Win rate by action */}
            {(buyHomeWinRate != null || buyAwayWinRate != null) && (
              <section>
                <SectionHeader label="Win Rate by Action" count={-1} color="text-blue-400" />
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <ActionWinRateCard action="BUY_HOME" winRate={buyHomeWinRate} count={buyHomeTrades.length} wins={buyHomeWins} />
                  <ActionWinRateCard action="BUY_AWAY" winRate={buyAwayWinRate} count={buyAwayTrades.length} wins={buyAwayWins} />
                </div>
              </section>
            )}

            {/* Live Pipeline Feed */}
            <section>
              <SectionHeader label="System Pipeline Feed" count={-1} pulse color="text-green-400" />
              <LivePipelineFeed events={feedEvents} />
            </section>

            {/* Live signals summary */}
            {liveGames.length > 0 && (
              <section>
                <SectionHeader label="Live Signal Summary" count={-1} pulse color="text-cyan-400" />
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                  <StatCard
                    label="Active Signals"
                    value={String(totalSignals)}
                    color={totalSignals > 0 ? "text-yellow-400" : "text-gray-500"}
                    sub="across all live games"
                  />
                  <StatCard
                    label="Games In-Play"
                    value={String(gamesInPlay)}
                    color="text-cyan-400"
                    sub={`of ${liveGames.length} tracked`}
                  />
                  <StatCard
                    label="Best Live Edge"
                    value={bestLiveEdgePct > 0 ? `${bestLiveEdgePct.toFixed(1)}%` : "—"}
                    color={bestLiveEdgePct >= 5 ? "text-green-400" : "text-gray-400"}
                    sub={bestLiveGame ? `${bestLiveGame.home_team} vs ${bestLiveGame.away_team}` : "—"}
                  />
                  <StatCard
                    label="Market Data"
                    value={`${gamesWithMarket.length}/${liveGames.length}`}
                    color="text-orange-400"
                    sub="games with Polymarket odds"
                  />
                </div>
                {bestLiveGame && bestLiveEdgePct >= 5 && (
                  <div className="bg-green-950/30 border border-green-800/40 rounded-lg p-4 flex items-center justify-between">
                    <div>
                      <p className="text-xs text-green-600 uppercase tracking-widest mb-1">Signal Detected</p>
                      <p className="text-green-300 font-bold">{bestLiveGame.home_team} vs {bestLiveGame.away_team}</p>
                      <p className="text-gray-400 text-xs mt-0.5">Q{bestLiveGame.period} · {bestLiveGame.score.home}–{bestLiveGame.score.away}</p>
                    </div>
                    <div className="text-right">
                      <p className="text-green-400 font-bold text-2xl">{bestLiveEdgePct.toFixed(1)}%</p>
                      <p className="text-gray-500 text-xs">edge</p>
                    </div>
                  </div>
                )}
              </section>
            )}
          </>
        )}

        {/* ── Live Games tab ── */}
        {activeTab === "live" && (
          <section>
            <SectionHeader label="Live Games — Model vs Market" count={liveGames.length} pulse color="text-cyan-400" />
            {liveGames.length === 0 ? (
              <div className="text-gray-600 text-sm py-12 text-center border border-gray-800 rounded-lg">
                No live games detected — waiting for active NBA games…
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
                {liveGames.map(g => (
                  <LiveGameCard key={g.game_id} game={g} />
                ))}
              </div>
            )}
          </section>
        )}

        {/* ── Active Positions tab ── */}
        {activeTab === "positions" && (
          <section>
            <SectionHeader label="Active Positions" count={openTrades.length} pulse color="text-yellow-400" />
            {loading ? (
              <div className="text-gray-600 text-sm py-6">Loading…</div>
            ) : openTrades.length === 0 ? (
              <div className="text-gray-600 text-sm py-12 text-center border border-gray-800 rounded-lg">
                No open positions — waiting for edge signal…
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
                {openTrades.map(t => (
                  <ActivePositionCard key={t.id} trade={t} />
                ))}
              </div>
            )}
          </section>
        )}

        {/* ── Trade History tab ── */}
        {activeTab === "history" && (
          <section>
            <SectionHeader label="Trade History" count={trades.length} color="text-gray-400" />
            {loading ? (
              <div className="text-gray-600 text-sm py-12 text-center">Loading…</div>
            ) : trades.length === 0 ? (
              <div className="text-gray-600 text-sm py-12 text-center border border-gray-800 rounded-lg">
                No trades logged yet — waiting for edge signal…
              </div>
            ) : (
              <div className="overflow-x-auto rounded-lg border border-gray-800">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="bg-gray-900 text-gray-500 text-left uppercase tracking-wider">
                      <th className="px-4 py-3">Time</th>
                      <th className="px-4 py-3">Game / Team</th>
                      <th className="px-4 py-3">Action</th>
                      <th className="px-4 py-3">Stake</th>
                      <th className="px-4 py-3">
                        <div>Mkt Win %</div>
                        <div className="text-gray-600 normal-case font-normal text-xs tracking-normal">P(bet team wins)</div>
                      </th>
                      <th className="px-4 py-3">
                        <div>Mdl Win %</div>
                        <div className="text-gray-600 normal-case font-normal text-xs tracking-normal">P(bet team wins)</div>
                      </th>
                      <th className="px-4 py-3">Edge</th>
                      <th className="px-4 py-3">Order Hash</th>
                      <th className="px-4 py-3">Status</th>
                      <th className="px-4 py-3 text-right">PnL</th>
                    </tr>
                  </thead>
                  <tbody>
                    {trades.map(t => {
                      const { mkt, mdl, edge } = alignedProbs(t);
                      const edgePct = edge * 100;
                      return (
                        <tr
                          key={t.id}
                          className="border-t border-gray-800/60 hover:bg-gray-900/50 transition-colors"
                        >
                          <td className="px-4 py-3 text-gray-500 whitespace-nowrap">{fmt(t.timestamp)}</td>
                          <td className="px-4 py-3 max-w-xs text-gray-300" title={t.target_team}>
                            <div className="truncate">{t.target_team}</div>
                            {t.bought_home !== null && (
                              <div className="flex items-center gap-1 mt-0.5">
                                <span className={`text-xs font-medium ${t.bought_home ? "text-blue-400" : "text-violet-400"}`}>
                                  {t.bought_home ? "▲ HOME" : "▽ AWAY"}
                                </span>
                                <span className="text-xs text-gray-600">· {betTeam(t)}</span>
                              </div>
                            )}
                          </td>
                          <td className="px-4 py-3">
                            <span className={`font-semibold ${t.action.startsWith("BUY") ? "text-green-400" : "text-red-400"}`}>
                              {t.action}
                            </span>
                          </td>
                          <td className="px-4 py-3 text-gray-400">${t.stake_amount.toFixed(0)}</td>
                          <td className="px-4 py-3">
                            <div className="text-gray-300">{(mkt * 100).toFixed(1)}%</div>
                            <div className="text-xs text-gray-600 mt-0.5">{betTeam(t)}</div>
                          </td>
                          <td className="px-4 py-3">
                            <div className="text-gray-300">{(mdl * 100).toFixed(1)}%</div>
                            <div className="text-xs text-gray-600 mt-0.5">{betTeam(t)}</div>
                          </td>
                          <td className={`px-4 py-3 font-bold ${edgePct >= 0 ? "text-green-400" : "text-red-400"}`}>
                            {edgePct >= 0 ? "+" : ""}{edgePct.toFixed(1)}%
                          </td>
                          <td className="px-4 py-3">
                            {t.order_hash ? (
                              <span className="text-cyan-500 cursor-help font-mono" title={t.order_hash}>
                                {shortHash(t.order_hash, 12)}
                              </span>
                            ) : (
                              <span className="text-gray-700">—</span>
                            )}
                          </td>
                          <td className="px-4 py-3">
                            <StatusBadge status={t.status} />
                          </td>
                          <td className={`px-4 py-3 text-right font-semibold ${(t.pnl ?? 0) >= 0 ? "text-green-400" : "text-red-400"}`}>
                            {t.pnl != null ? `${t.pnl >= 0 ? "+" : ""}$${t.pnl.toFixed(2)}` : "—"}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        )}

        {/* ── Analytics tab ── */}
        {activeTab === "analytics" && (
          <section>
            <SectionHeader label="Model vs Market — Confidence Intervals" count={-1} color="text-purple-400" />
            {trades.length < 2 ? (
              <div className="text-gray-600 text-sm py-12 text-center border border-gray-800 rounded-lg">
                Need at least 2 trades to display charts…
              </div>
            ) : (
              <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
                <div className="xl:col-span-2 bg-gray-900 border border-gray-800 rounded-lg p-5">
                  <p className="text-xs text-gray-500 mb-1 uppercase tracking-widest">Probability Over Trades</p>
                  <p className="text-xs text-gray-600 mb-4">
                    Purple band = model ±5% CI (edge threshold). Trades trigger when market exits this band.
                  </p>
                  <ProbabilityChart trades={[...trades].slice(0, 30).reverse()} />
                </div>
                <div className="bg-gray-900 border border-gray-800 rounded-lg p-5">
                  <p className="text-xs text-gray-500 mb-1 uppercase tracking-widest">Edge Distribution</p>
                  <p className="text-xs text-gray-600 mb-4">
                    Model edge per trade. Green = positive (model &gt; market).
                  </p>
                  <EdgeChart trades={[...trades].slice(0, 20).reverse()} />
                </div>
              </div>
            )}
          </section>
        )}

      </div>

      {/* Trade toast notifications — fixed bottom-right */}
      <ToastStack toasts={toasts} onDismiss={(id) => setToasts(prev => prev.filter(t => t.id !== id))} />
    </main>
  );
}

// ── Sub-components ──

function StatCard({
  label, value, color, sub,
}: {
  label: string; value: string; color: string; sub?: string;
}) {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
      <p className="text-gray-500 text-xs uppercase tracking-widest mb-1">{label}</p>
      <p className={`text-2xl font-bold ${color}`}>{value}</p>
      {sub && <p className="text-gray-600 text-xs mt-1">{sub}</p>}
    </div>
  );
}

function SectionHeader({
  label, count, pulse, color,
}: {
  label: string; count: number; pulse?: boolean; color: string;
}) {
  return (
    <h2 className={`text-xs font-bold uppercase tracking-widest mb-4 flex items-center gap-2 ${color}`}>
      {pulse && <span className="w-2 h-2 rounded-full bg-yellow-400 animate-pulse" />}
      {label}
      {count >= 0 && <span className="text-gray-600 font-normal">({count})</span>}
    </h2>
  );
}

/** Return market/model probs aligned to the traded team's perspective.
 *  Raw DB values are always P(home wins). When bought_home=false we flip both. */
function alignedProbs(t: Trade): { mkt: number; mdl: number; edge: number } {
  const flip = t.bought_home === false;
  const mkt = flip ? 1 - t.market_implied_prob : t.market_implied_prob;
  const mdl = flip ? 1 - t.model_implied_prob  : t.model_implied_prob;
  return { mkt, mdl, edge: mdl - mkt };
}

/** The team whose win-probability the market%/model% columns are showing.
 *  For BUY rows the action encodes the team directly. For SELL rows use target_team. */
function betTeam(t: Trade): string {
  if (t.action.startsWith("BUY_")) {
    return t.action.slice(4).replace(/_/g, " ")
      .toLowerCase().replace(/\b\w/g, c => c.toUpperCase());
  }
  return t.target_team;
}

function ActivePositionCard({ trade }: { trade: Trade }) {
  const { edge } = alignedProbs(trade);
  const edgePct  = edge * 100;
  const isBuyHome = trade.bought_home !== false;
  return (
    <div className="bg-gray-900 border border-yellow-900/30 rounded-lg p-4 space-y-3 hover:border-yellow-700/40 transition-colors">
      <div className="flex items-start justify-between gap-3">
        <p className="text-gray-200 text-sm font-semibold leading-snug flex-1 min-w-0 truncate" title={trade.target_team}>
          {trade.target_team}
        </p>
        <StatusBadge status={trade.status} />
      </div>

      <div className="grid grid-cols-2 gap-x-6 gap-y-2 text-xs">
        <DataPoint label="Action">
          <span className={`font-bold ${isBuyHome ? "text-green-400" : "text-orange-400"}`}>
            {trade.action}
          </span>
        </DataPoint>
        <DataPoint label="Stake">
          <span className="text-gray-200 font-semibold">${trade.stake_amount.toFixed(0)} USDC</span>
        </DataPoint>
        <DataPoint label="Market odds">
          <span className="text-gray-300">{(alignedProbs(trade).mkt * 100).toFixed(1)}%</span>
        </DataPoint>
        <DataPoint label="Model confidence">
          <span className="text-gray-300">{(alignedProbs(trade).mdl * 100).toFixed(1)}%</span>
        </DataPoint>
      </div>

      {trade.order_hash && (
        <div className="pt-1 border-t border-gray-800 text-xs">
          <span className="text-gray-600">Order hash </span>
          <span className="text-cyan-500 font-mono" title={trade.order_hash}>
            {shortHash(trade.order_hash, 14)}
          </span>
        </div>
      )}

      <div className="flex items-center justify-between pt-2 border-t border-gray-800">
        <span className="text-gray-600 text-xs">{fmt(trade.timestamp)}</span>
        <span className={`text-sm font-bold ${edge >= 0 ? "text-green-400" : "text-red-400"}`}>
          EDGE {edge >= 0 ? "+" : ""}{edge.toFixed(1)}%
        </span>
      </div>
    </div>
  );
}

function DataPoint({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <span className="text-gray-600 block">{label}</span>
      <div className="mt-0.5">{children}</div>
    </div>
  );
}

function LiveGameCard({ game }: { game: LiveGame }) {
  const modelProb  = game.predictions.win_probability;
  const marketProb = game.market_odds?.polymarket_prob;
  const marketEdge = game.market_odds?.market_edge;
  const proxyProb  = game.predictions.proxy_probability;
  const margin     = game.predictions.predicted_margin;
  const confidence = game.predictions.edge_confidence;
  const kelly      = game.predictions.kelly_size;
  const hasMarket  = marketProb != null;

  const edgeVal    = hasMarket ? (marketEdge ?? 0) : game.predictions.edge;
  const edgePct    = edgeVal * 100;

  return (
    <div className="bg-gray-900 border border-cyan-900/30 rounded-lg p-4 space-y-3 hover:border-cyan-700/40 transition-colors">
      {/* Header: teams + score */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-gray-200 font-bold text-sm">{game.home_team}</span>
          <span className="text-gray-600 text-xs">vs</span>
          <span className="text-gray-200 font-bold text-sm">{game.away_team}</span>
        </div>
        <div className="text-right">
          <span className="text-gray-100 font-bold text-lg">{game.score.home}-{game.score.away}</span>
          <span className="text-gray-500 text-xs ml-2">Q{game.period}</span>
        </div>
      </div>

      {/* Probability comparison bars — both values are P(home_team wins) */}
      <div className="space-y-2">
        <ProbBar label={`Model · P(${game.home_team} wins)`} value={modelProb} color="bg-green-500" />
        {hasMarket ? (
          <ProbBar label={`Market · P(${game.home_team} wins)`} value={marketProb!} color="bg-blue-500" />
        ) : (
          <ProbBar label={`Proxy · P(${game.home_team} wins)`} value={proxyProb} color="bg-gray-500" />
        )}
      </div>

      {/* Edge + metrics */}
      <div className="grid grid-cols-3 gap-x-4 gap-y-1 text-xs">
        <DataPoint label="Edge">
          <span className={`font-bold ${edgePct >= 0 ? "text-green-400" : "text-red-400"}`}>
            {edgePct >= 0 ? "+" : ""}{edgePct.toFixed(1)}%
          </span>
          <span className="text-gray-600 ml-1 text-xs">
            {edgePct >= 0 ? `↑${game.home_team}` : `↑${game.away_team}`}
          </span>
        </DataPoint>
        <DataPoint label="Confidence">
          <span className={`font-bold ${confidence >= 0.6 ? "text-green-400" : "text-gray-400"}`}>
            {(confidence * 100).toFixed(0)}%
          </span>
        </DataPoint>
        <DataPoint label="Kelly">
          <span className="text-purple-400 font-bold">{(kelly * 100).toFixed(1)}%</span>
        </DataPoint>
        <DataPoint label="Margin">
          <span className="text-gray-300">{margin >= 0 ? "+" : ""}{margin.toFixed(1)}</span>
        </DataPoint>
        <DataPoint label="Signals">
          <span className={game.signal_count > 0 ? "text-yellow-400 font-bold" : "text-gray-600"}>
            {game.signal_count}
          </span>
        </DataPoint>
        {hasMarket && game.market_odds.volume != null && (
          <DataPoint label="Volume">
            <span className="text-gray-400">${(game.market_odds.volume! / 1000).toFixed(0)}k</span>
          </DataPoint>
        )}
      </div>

      {/* Source tag */}
      <div className="flex items-center justify-between pt-2 border-t border-gray-800 text-xs">
        <span className="text-gray-600">
          {hasMarket ? `via ${game.market_odds.source}` : "proxy model baseline"}
        </span>
        <span className={`font-bold text-sm ${Math.abs(edgePct) >= 5 ? (edgePct >= 0 ? "text-green-400" : "text-red-400") : "text-gray-500"}`}>
          {Math.abs(edgePct) >= 5 ? "SIGNAL" : "NO EDGE"}
        </span>
      </div>
    </div>
  );
}

function ProbBar({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="text-gray-500 w-12 text-right">{label}</span>
      <div className="flex-1 bg-gray-800 rounded-full h-3 overflow-hidden">
        <div className={`${color} h-full rounded-full transition-all duration-500`} style={{ width: `${Math.max(2, value * 100)}%` }} />
      </div>
      <span className="text-gray-300 w-12 font-bold">{(value * 100).toFixed(1)}%</span>
    </div>
  );
}

// ── Chart tooltip ──
function ProbTooltip({ active, payload, label }: {
  active?: boolean; payload?: Array<{ dataKey: string; value: number }>; label?: string;
}) {
  if (!active || !payload?.length) return null;
  const model  = payload.find(p => p.dataKey === "model");
  const market = payload.find(p => p.dataKey === "market");
  const edge   = model && market ? model.value - market.value : null;
  return (
    <div className="bg-gray-950 border border-gray-700 rounded-md px-3 py-2 text-xs space-y-1">
      <p className="text-gray-400 font-semibold">{label}</p>
      {model  && <p className="text-purple-400">Model:  {model.value.toFixed(1)}%</p>}
      {market && <p className="text-yellow-400">Market: {market.value.toFixed(1)}%</p>}
      {edge != null && (
        <p className={`font-bold ${edge >= 0 ? "text-green-400" : "text-red-400"}`}>
          Edge: {edge >= 0 ? "+" : ""}{edge.toFixed(1)}%
        </p>
      )}
    </div>
  );
}

function ProbabilityChart({ trades }: { trades: Trade[] }) {
  const CI = 5;
  const data = trades.map(t => {
    const modelPct = +(t.model_implied_prob * 100).toFixed(1);
    const low      = +Math.max(0,   modelPct - CI).toFixed(1);
    const high     = +Math.min(100, modelPct + CI).toFixed(1);
    return {
      label:   t.target_team.split(" ").slice(-1)[0],
      market:  +(t.market_implied_prob * 100).toFixed(1),
      model:   modelPct,
      ciBase:  low,
      ciWidth: +(high - low).toFixed(1),
    };
  });

  return (
    <ResponsiveContainer width="100%" height={260}>
      <ComposedChart data={data} margin={{ top: 8, right: 8, left: -8, bottom: 0 }}>
        <defs>
          <linearGradient id="ciGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%"   stopColor="#8b5cf6" stopOpacity={0.30} />
            <stop offset="100%" stopColor="#8b5cf6" stopOpacity={0.05} />
          </linearGradient>
        </defs>
        <XAxis dataKey="label" tick={{ fill: "#6b7280", fontSize: 10 }} tickLine={false} axisLine={{ stroke: "#374151" }} />
        <YAxis domain={[0, 100]} tick={{ fill: "#6b7280", fontSize: 10 }} tickLine={false} axisLine={false}
               tickFormatter={v => `${v}%`} width={38} />
        <Tooltip content={<ProbTooltip />} />
        <Legend wrapperStyle={{ fontSize: "11px", paddingTop: "10px" }}
                formatter={v => <span style={{ color: "#9ca3af" }}>{v}</span>} />
        <ReferenceLine y={50} stroke="#374151" strokeDasharray="3 3" />
        {/* CI band via stacked areas */}
        <Area type="monotone" dataKey="ciBase"  stackId="ci" stroke="none" fill="transparent" legendType="none" tooltipType="none" />
        <Area type="monotone" dataKey="ciWidth" stackId="ci" stroke="none" fill="url(#ciGrad)" name="Model CI (±5%)" legendType="square" />
        {/* Probability lines */}
        <Line type="monotone" dataKey="model"  name="Model Probability"  stroke="#8b5cf6" strokeWidth={2}
              dot={{ r: 3, fill: "#8b5cf6", strokeWidth: 0 }} activeDot={{ r: 5 }} />
        <Line type="monotone" dataKey="market" name="Market Probability" stroke="#f59e0b" strokeWidth={2}
              strokeDasharray="5 3" dot={{ r: 3, fill: "#f59e0b", strokeWidth: 0 }} activeDot={{ r: 5 }} />
      </ComposedChart>
    </ResponsiveContainer>
  );
}

function EdgeChart({ trades }: { trades: Trade[] }) {
  const data = trades.map(t => ({
    label: t.target_team.split(" ").slice(-1)[0],
    edge:  +((t.model_implied_prob - t.market_implied_prob) * 100).toFixed(1),
  }));

  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart data={data} margin={{ top: 8, right: 8, left: -8, bottom: 0 }}>
        <XAxis dataKey="label" tick={{ fill: "#6b7280", fontSize: 10 }} tickLine={false} axisLine={{ stroke: "#374151" }} />
        <YAxis tick={{ fill: "#6b7280", fontSize: 10 }} tickLine={false} axisLine={false}
               tickFormatter={v => `${v}%`} width={38} />
        <Tooltip
          contentStyle={{ backgroundColor: "#0f172a", border: "1px solid #334155", borderRadius: "6px", fontSize: "11px" }}
          labelStyle={{ color: "#94a3b8" }}
          formatter={(v: number) => [`${v >= 0 ? "+" : ""}${v}%`, "Edge"]}
        />
        <ReferenceLine y={0} stroke="#374151" />
        <Bar dataKey="edge" name="Edge" shape={(props: any) => {
          const { x, y, width, height } = props;
          const fill = (props.value ?? 0) >= 0 ? "#4ade80" : "#f87171";
          return <rect x={x} y={y} width={width} height={Math.abs(height)} fill={fill} fillOpacity={0.85} rx={2} />;
        }} />
      </BarChart>
    </ResponsiveContainer>
  );
}

function timeSince(dateStr: string): string {
  const diff = Date.now() - new Date(dateStr).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}

function SparklineChart({ data }: { data: Array<{ pnl: number }> }) {
  if (data.length < 2) {
    return (
      <div className="h-40 flex items-center justify-center text-gray-700 text-xs">
        Not enough data yet
      </div>
    );
  }
  const color = data[data.length - 1].pnl >= 0 ? "#4ade80" : "#f87171";
  return (
    <ResponsiveContainer width="100%" height={160}>
      <LineChart data={data} margin={{ top: 4, right: 8, left: -8, bottom: 0 }}>
        <YAxis tick={{ fill: "#6b7280", fontSize: 10 }} tickLine={false} axisLine={false}
               tickFormatter={v => `$${v}`} width={48} />
        <Tooltip
          contentStyle={{ backgroundColor: "#0f172a", border: "1px solid #334155", borderRadius: "6px", fontSize: "11px" }}
          formatter={(v: number) => [`${v >= 0 ? "+" : ""}$${v.toFixed(2)}`, "Cum. PnL"]}
          labelFormatter={() => ""}
        />
        <ReferenceLine y={0} stroke="#374151" strokeDasharray="3 3" />
        <Line type="monotone" dataKey="pnl" stroke={color} strokeWidth={2} dot={false} activeDot={{ r: 4 }} />
      </LineChart>
    </ResponsiveContainer>
  );
}

function ActionWinRateCard({ action, winRate, count, wins }: {
  action: string; winRate: number | null; count: number; wins: number;
}) {
  const isHome = action !== "BUY_AWAY";
  return (
    <div className={`bg-gray-900 border rounded-lg p-4 ${isHome ? "border-green-900/30" : "border-orange-900/30"}`}>
      <div className="flex items-center justify-between mb-3">
        <span className={`text-xs font-bold uppercase tracking-widest ${isHome ? "text-green-400" : "text-orange-400"}`}>
          {action}
        </span>
        <span className="text-gray-500 text-xs">{count} resolved</span>
      </div>
      {winRate != null ? (
        <>
          <p className={`text-3xl font-bold ${isHome ? "text-green-400" : "text-orange-400"}`}>
            {winRate.toFixed(1)}%
          </p>
          <p className="text-gray-600 text-xs mt-1">{wins}W / {count - wins}L</p>
          <div className="mt-3 bg-gray-800 rounded-full h-2 overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-500 ${isHome ? "bg-green-500" : "bg-orange-500"}`}
              style={{ width: `${winRate}%` }}
            />
          </div>
        </>
      ) : (
        <p className="text-gray-600 text-sm">No resolved trades</p>
      )}
    </div>
  );
}

function StatusBadge({ status }: { status: string }) {
  const styles: Record<string, string> = {
    OPEN:   "bg-yellow-900/50 text-yellow-300 border border-yellow-800/50",
    WON:    "bg-green-900/50  text-green-300  border border-green-800/50",
    LOST:   "bg-red-900/50    text-red-300    border border-red-800/50",
    CLOSED: "bg-gray-800/50   text-gray-300   border border-gray-700/50",
  };
  return (
    <span className={`px-2 py-0.5 rounded text-xs font-bold whitespace-nowrap ${styles[status] ?? "bg-gray-700 text-gray-300"}`}>
      {status}
    </span>
  );
}

// ── Live Pipeline Feed ──

const FEED_CONFIG = {
  NBA_DATA:      { label: "NBA DATA",   textColor: "text-cyan-400",  dotColor: "bg-cyan-400"  },
  ML_PREDICTION: { label: "ML MODEL",   textColor: "text-amber-400", dotColor: "bg-amber-400" },
  TRADE:         { label: "TRADE",      textColor: "text-green-400", dotColor: "bg-green-400" },
} as const;

function LivePipelineFeed({ events }: { events: FeedEvent[] }) {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = 0;
  }, [events.length]);

  if (events.length === 0) {
    return (
      <div className="bg-gray-900 border border-gray-800 rounded-lg h-48 flex items-center justify-center">
        <p className="text-gray-600 text-xs">Waiting for system events…</p>
      </div>
    );
  }

  return (
    <div ref={scrollRef} className="bg-gray-900 border border-gray-800 rounded-lg overflow-y-auto h-56 divide-y divide-gray-800/50">
      {events.map((ev, i) => {
        const cfg = FEED_CONFIG[ev.type];
        return (
          <div
            key={ev.id}
            className={`flex items-start gap-3 px-4 py-2.5 text-xs ${i === 0 ? "bg-gray-800/50" : "hover:bg-gray-800/20"} transition-colors`}
          >
            <span className="text-gray-600 whitespace-nowrap pt-0.5 w-24 shrink-0 tabular-nums">
              {ev.timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}
            </span>
            <span className={`w-2 h-2 rounded-full mt-1 shrink-0 ${cfg.dotColor} ${i === 0 ? "animate-pulse" : ""}`} />
            <span className={`font-bold uppercase tracking-wider shrink-0 w-20 ${cfg.textColor}`}>
              {cfg.label}
            </span>
            <div className="min-w-0 leading-relaxed">
              <span className="text-gray-200">{ev.message}</span>
              {ev.detail && <span className="text-gray-500 ml-2">{ev.detail}</span>}
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ── Trade Toast Notifications ──

function ToastStack({
  toasts,
  onDismiss,
}: {
  toasts: ToastNotification[];
  onDismiss: (id: string) => void;
}) {
  if (toasts.length === 0) return null;
  return (
    <div className="fixed bottom-6 right-6 z-50 flex flex-col gap-3 pointer-events-none">
      {toasts.map(toast => {
        const edge  = (toast.trade.model_implied_prob - toast.trade.market_implied_prob) * 100;
        const isBuy = toast.trade.action.startsWith("BUY");
        return (
          <div
            key={toast.id}
            className="pointer-events-auto bg-gray-900 border border-green-700/60 rounded-lg p-4 w-72 shadow-xl shadow-black/50 animate-slide-in"
          >
            <div className="flex items-start justify-between gap-2">
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse shrink-0" />
                <span className="text-green-400 font-bold text-xs uppercase tracking-widest">Trade Executed</span>
              </div>
              <button
                onClick={() => onDismiss(toast.id)}
                className="text-gray-600 hover:text-gray-400 text-xs leading-none"
              >
                ✕
              </button>
            </div>
            <p className="text-gray-100 font-semibold text-sm mt-2 leading-snug truncate" title={toast.trade.target_team}>
              {toast.trade.target_team}
            </p>
            <div className="flex items-center gap-4 mt-2 text-xs">
              <span className={`font-bold ${isBuy ? "text-green-400" : "text-red-400"}`}>{toast.trade.action}</span>
              <span className={`font-bold ${edge >= 0 ? "text-green-400" : "text-red-400"}`}>
                {edge >= 0 ? "+" : ""}{edge.toFixed(1)}% edge
              </span>
              <span className="text-gray-500">${toast.trade.stake_amount}</span>
            </div>
            {toast.trade.order_hash && (
              <p className="text-cyan-700 text-xs mt-2 font-mono truncate" title={toast.trade.order_hash}>
                {toast.trade.order_hash.slice(0, 22)}…
              </p>
            )}
          </div>
        );
      })}
    </div>
  );
}
