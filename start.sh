#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  NBA Shadow Trader — one-command launcher
#  Usage:  ./start.sh          (starts all 4 services)
#          ./start.sh --stop   (kills all 4 services)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PIDFILE="$ROOT/.shadow-trader.pids"

# ── Stop mode ────────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--stop" ]]; then
    if [[ -f "$PIDFILE" ]]; then
        echo "Stopping services..."
        while read -r pid name; do
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid" 2>/dev/null && echo "  Stopped $name (PID $pid)"
            fi
        done < "$PIDFILE"
        rm -f "$PIDFILE"
    else
        echo "No running services found."
    fi
    exit 0
fi

# ── Preflight checks ────────────────────────────────────────────────────────
VENV="$ROOT/venv/bin/python"
if [[ ! -x "$VENV" ]]; then
    echo "ERROR: Python venv not found. Run:  python3 -m venv venv && ./venv/bin/pip install -r alpha-engine/requirements.txt"
    exit 1
fi

CARGO="${CARGO:-$(command -v cargo 2>/dev/null || echo "$HOME/.cargo/bin/cargo")}"
if [[ ! -x "$CARGO" ]]; then
    echo "ERROR: cargo not found. Install Rust: https://rustup.rs"
    exit 1
fi

if ! command -v npm &>/dev/null; then
    echo "ERROR: npm not found. Install Node.js ≥ 20."
    exit 1
fi

if [[ ! -d "$ROOT/dashboard/node_modules" ]]; then
    echo "Installing dashboard dependencies..."
    (cd "$ROOT/dashboard" && npm install)
fi

# ── Clean up any previous run ────────────────────────────────────────────────
if [[ -f "$PIDFILE" ]]; then
    "$0" --stop 2>/dev/null || true
fi

mkdir -p "$ROOT/logs"
> "$PIDFILE"

# ── Launch services ──────────────────────────────────────────────────────────
echo ""
echo "  NBA Shadow Trader"
echo "  ─────────────────"
echo ""

# 1. Live Game Server (port 8000)
$VENV -m uvicorn server:app --port 8000 \
    > "$ROOT/logs/server.log" 2>&1 &
echo "$! server.py:8000" >> "$PIDFILE"
echo "  [1/4] Live Game Server     → http://localhost:8000"

# 2. Alpha Engine (port 8001)
$VENV "$ROOT/alpha-engine/main.py" \
    > "$ROOT/logs/alpha-engine.log" 2>&1 &
echo "$! alpha-engine:8001" >> "$PIDFILE"
echo "  [2/4] Alpha Engine         → http://localhost:8001"

# 3. Execution Engine (port 4000)
(cd "$ROOT/execution-engine" && RUST_LOG=info "$CARGO" run --release) \
    > "$ROOT/logs/execution-engine.log" 2>&1 &
echo "$! execution-engine:4000" >> "$PIDFILE"
echo "  [3/4] Execution Engine     → http://localhost:4000"

# 4. Dashboard (port 3000)
(cd "$ROOT/dashboard" && npm run dev) \
    > "$ROOT/logs/dashboard.log" 2>&1 &
echo "$! dashboard:3000" >> "$PIDFILE"
echo "  [4/4] Dashboard            → http://localhost:3000"

echo ""
echo "  All services starting. Logs in ./logs/"
echo ""
echo "  Open the dashboard:  http://localhost:3000"
echo "  Stop everything:     ./start.sh --stop"
echo ""
