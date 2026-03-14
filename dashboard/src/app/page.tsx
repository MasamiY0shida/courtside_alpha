"use client";

import { useEffect, useState } from "react";

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
}

export default function Dashboard() {
  const [trades, setTrades]     = useState<Trade[]>([]);
  const [loading, setLoading]   = useState(true);

  // Poll the backend every 3 s
  useEffect(() => {
    const fetchTrades = async () => {
      try {
        const res  = await fetch("/api/trades");
        const data = await res.json();
        setTrades(data);
      } catch {
        // backend might not be up yet — silently retry
      } finally {
        setLoading(false);
      }
    };

    fetchTrades();
    const interval = setInterval(fetchTrades, 3000);
    return () => clearInterval(interval);
  }, []);

  // ── Derived stats ──
  const totalPnl     = trades.reduce((s, t) => s + (t.pnl ?? 0), 0);
  const won          = trades.filter(t => t.status === "WON").length;
  const lost         = trades.filter(t => t.status === "LOST").length;
  const open         = trades.filter(t => t.status === "OPEN").length;
  const resolved     = won + lost;
  const winRate      = resolved > 0 ? (won / resolved) * 100 : 0;

  return (
    <main className="min-h-screen bg-gray-950 text-gray-100 p-6 font-mono">
      {/* Header */}
      <h1 className="text-2xl font-bold text-green-400 mb-6">
        NBA Shadow Trader — Live Dashboard
      </h1>

      {/* Stats bar */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <StatCard label="Simulated PnL" value={`$${totalPnl.toFixed(2)}`}
          color={totalPnl >= 0 ? "text-green-400" : "text-red-400"} />
        <StatCard label="Win Rate"      value={`${winRate.toFixed(1)}%`}    color="text-blue-400" />
        <StatCard label="Open Positions" value={String(open)}               color="text-yellow-400" />
        <StatCard label="Total Trades"  value={String(trades.length)}       color="text-gray-300" />
      </div>

      {/* Trade feed */}
      <h2 className="text-lg font-semibold mb-3 text-gray-300">Live Trade Feed</h2>
      {loading ? (
        <p className="text-gray-500">Loading…</p>
      ) : trades.length === 0 ? (
        <p className="text-gray-500">No trades logged yet — waiting for edge…</p>
      ) : (
        <div className="overflow-x-auto rounded-lg border border-gray-800">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-gray-900 text-gray-400 text-left">
                <th className="p-3">Time</th>
                <th className="p-3">Market</th>
                <th className="p-3">Action</th>
                <th className="p-3">Market %</th>
                <th className="p-3">Model %</th>
                <th className="p-3">Edge</th>
                <th className="p-3">Status</th>
                <th className="p-3">PnL</th>
              </tr>
            </thead>
            <tbody>
              {trades.map(t => {
                const edge = (t.model_implied_prob - t.market_implied_prob) * 100;
                return (
                  <tr key={t.id} className="border-t border-gray-800 hover:bg-gray-900">
                    <td className="p-3 text-gray-400">{new Date(t.timestamp).toLocaleTimeString()}</td>
                    <td className="p-3 max-w-xs truncate">{t.target_team}</td>
                    <td className="p-3">
                      <span className={t.action !== "BUY_AWAY" ? "text-green-400" : "text-red-400"}>
                        {t.action}
                      </span>
                    </td>
                    <td className="p-3">{(t.market_implied_prob * 100).toFixed(1)}%</td>
                    <td className="p-3">{(t.model_implied_prob  * 100).toFixed(1)}%</td>
                    <td className={`p-3 font-bold ${edge >= 0 ? "text-green-400" : "text-red-400"}`}>
                      {edge >= 0 ? "+" : ""}{edge.toFixed(1)}%
                    </td>
                    <td className="p-3">
                      <StatusBadge status={t.status} />
                    </td>
                    <td className={`p-3 ${(t.pnl ?? 0) >= 0 ? "text-green-400" : "text-red-400"}`}>
                      {t.pnl != null ? `$${t.pnl.toFixed(2)}` : "—"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </main>
  );
}

function StatCard({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
      <p className="text-gray-400 text-xs uppercase tracking-wide mb-1">{label}</p>
      <p className={`text-2xl font-bold ${color}`}>{value}</p>
    </div>
  );
}

function StatusBadge({ status }: { status: string }) {
  const styles: Record<string, string> = {
    OPEN: "bg-yellow-900 text-yellow-300",
    WON:  "bg-green-900  text-green-300",
    LOST: "bg-red-900    text-red-300",
  };
  return (
    <span className={`px-2 py-0.5 rounded text-xs font-bold ${styles[status] ?? "bg-gray-700 text-gray-300"}`}>
      {status}
    </span>
  );
}
