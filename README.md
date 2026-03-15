# Courtside Alpha

Quantitative prediction market engine — real-time NBA model vs. Polymarket odds with testnet order signing.

## Quick start

```bash
./start.sh              # launches all 4 services
open http://localhost:3000  # open the dashboard
./start.sh --stop       # shut everything down
```

Logs are written to `./logs/`. For manual per-service startup, see below.

---

## Prerequisites

| Tool | Version |
|------|---------|
| Python | ≥ 3.11 |
| Rust + Cargo | ≥ 1.78 |
| Node.js | ≥ 20 |

---

## One-time setup

Create a single venv at the project root — it covers all Python services:

```bash
# From the project root
python3 -m venv venv
source venv/bin/activate
pip install -r alpha-engine/requirements.txt
pip install pyarrow
```

> **Important:** Always use `./venv/bin/python` (or `./venv/bin/pip`) directly instead of bare `python`/`uvicorn`, to avoid Anaconda or system Python shadowing the venv.

---

## 1 — Live Game Server (Python · port 8000)

```bash
./venv/bin/python -m uvicorn server:app --reload --port 8000
```

Polls the NBA live API for scores and game state. The Execution Engine reads from this.

Health check: http://localhost:8000/health

---

## 2 — Alpha Engine (Python · port 8001)

```bash
./venv/bin/python alpha-engine/main.py
```

Loads the trained XGBoost model suite and serves win-probability predictions.

Health check: http://localhost:8001/health

---

## 3 — Execution Engine (Rust · port 4000)

```bash
cd execution-engine
cargo run --release
```

Connects to Polymarket WebSocket, queries the Alpha Engine on each price tick, and logs shadow trades to `trades.sqlite`.

Optional env vars:

| Variable | Default | Description |
|----------|---------|-------------|
| `RUST_LOG` | `info` | Log level (`debug` for verbose) |

```bash
RUST_LOG=debug cargo run --release
```

Health check: http://localhost:4000/health

---

## 4 — Dashboard (Next.js · port 3000)

```bash
cd dashboard
npm install
npm run dev
```

Open http://localhost:3000 — polls the Execution Engine every 3 s for live trades and wallet state.

---

## Data flow

```
Polymarket WS ──► Execution Engine (port 4000) ──► SQLite (trades.sqlite)
                        │         │                        ▲
                        │         └──► Alpha Engine        │
                        │              POST /predict       │
                        │              (port 8001)         │
                        │                                  │
                        └──► Live Game Server         Dashboard (port 3000)
                              GET /games               GET /trades, /wallet
                              (port 8000)
```

---

## Project structure

```
unihack/
├── server.py               # Live game state server (port 8000)
├── features.py             # Shared FeatureEngine
├── market_data.py          # Polymarket odds helpers
├── model.py / model_v2.py  # Model training scripts
├── trades.sqlite           # Shadow trade log (auto-created)
├── alpha-engine/           # FastAPI ML inference server (port 8001)
│   ├── main.py
│   └── requirements.txt
├── execution-engine/       # Rust WebSocket + shadow trader (port 4000)
│   ├── Cargo.toml
│   └── src/main.rs
├── dashboard/              # Next.js real-time UI (port 3000)
│   ├── package.json
│   └── src/
└── venv/                   # Shared Python venv (gitignored)
```
