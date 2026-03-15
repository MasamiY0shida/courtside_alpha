// ============================================================================
//  NBA Execution Engine — Module B  (v2 · Testnet Order Signing)
//  Polymarket live odds ingestion + shadow trading + EIP-712 order signing
//
//  Flow:
//    1. Query Polymarket REST API → discover live NBA markets
//    2. Open WebSocket to Polymarket CLOB → stream price ticks
//    3. On each tick → query server.py for full v2 model predictions
//       (265-feature pipeline with live boxscore, momentum, lag features)
//       Falls back to alpha-engine if server.py doesn't track the game.
//    4. If |model_prob - market_prob| > EDGE_THRESHOLD →
//         a. Build a Polymarket CLOB order and sign it (EIP-712, test key)
//         b. Log signed trade to SQLite, deduct fake USDC from wallet
//    5. Background task settles resolved trades & restores USDC balance
//    6. HTTP server exposes /trades, /wallet, /health to the dashboard
// ============================================================================

mod wallet;

use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::{Context, Result};
use axum::{extract::State, routing::get, Json, Router};
use chrono::Utc;
use futures_util::{SinkExt, StreamExt};
use reqwest::Client;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tower_http::cors::CorsLayer;
use tracing::{error, info, warn};
use uuid::Uuid;

// ── Configuration ────────────────────────────────────────────────────────────

const GAMMA_API:        &str = "https://gamma-api.polymarket.com";
const POLYMARKET_WS:    &str = "wss://ws-subscriptions-clob.polymarket.com/ws/market";
const ALPHA_ENGINE_URL: &str = "http://127.0.0.1:8001";
/// server.py live inference server (for live game state lookups).
const SERVER_URL:       &str = "http://127.0.0.1:8000";
const DB_PATH:          &str = "../trades.sqlite";

/// Minimum |model_prob − market_prob| to trigger a signed shadow trade.
const EDGE_THRESHOLD: f64 = 0.05;

/// Simulated USDC stake per trade.
const STAKE_USDC: f64 = 50.0;

/// Starting fake USDC balance for the test wallet.
const INITIAL_USDC: f64 = 10_000.0;

// ── Shared application state ──────────────────────────────────────────────────

#[derive(Clone)]
struct AppState {
    db:     Arc<Mutex<Connection>>,
    wallet: Arc<wallet::TestWallet>,
}

// ── Polymarket types ──────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Market {
    condition_id:    String,
    question:        String,
    game_start_time: String,
    tokens:          Vec<Token>,
}

#[derive(Debug, Clone)]
struct Token {
    token_id:  String,
    team_name: String,
    is_home:   bool,
    team_id:   Option<i64>,
    #[allow(dead_code)]
    price:     f64,
}

// ── Gamma API types ───────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct GammaEvent {
    #[allow(dead_code)]
    title:   String,
    active:  bool,
    closed:  bool,
    markets: Vec<GammaMarket>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GammaMarket {
    question:           String,
    condition_id:       String,
    clob_token_ids:     Option<String>,
    outcomes:           Option<String>,
    outcome_prices:     Option<String>,
    sports_market_type: Option<String>,
    game_start_time:    Option<String>,
    active:             bool,
    closed:             bool,
}

// ── WebSocket types ───────────────────────────────────────────────────────────

#[derive(Serialize)]
struct WsSubscribe {
    #[serde(rename = "type")]
    msg_type:   &'static str,
    channel:    &'static str,
    assets_ids: Vec<String>,
}

// ── Alpha Engine types ────────────────────────────────────────────────────────

#[derive(Serialize)]
struct GameStateRequest {
    home_team_id:      i64,
    away_team_id:      i64,
    period:            i32,
    game_seconds_left: f64,
    home_score:        f64,
    away_score:        f64,
}

#[derive(Deserialize)]
struct PredictResponse {
    win_probability:   f64,
    #[allow(dead_code)]
    proxy_probability: f64,
    #[allow(dead_code)]
    predicted_margin:  f64,
    #[allow(dead_code)]
    edge:              f64,
    #[allow(dead_code)]
    abs_edge:          f64,
    edge_confidence:   f64,
    #[allow(dead_code)]
    kelly_size:        f64,
    model_loaded:      bool,
}

// ── server.py combined prediction (full v2 pipeline with all 265 features) ───

/// Predictions extracted from server.py GET /games response.
/// These use the full feature pipeline (boxscore, momentum, lag) unlike the
/// alpha-engine fallback which only receives 6 fields.
struct ServerPrediction {
    game_seconds_left: f64,
    win_probability:   f64,
    edge_confidence:   f64,
}

// ── Sell policy configuration ─────────────────────────────────────────────────

/// If market price moves this far against our entry, close the position.
const STOP_LOSS_PP: f64 = 0.15;

/// Once unrealised gain exceeds this, activate trailing stop.
const TRAILING_ACTIVATE_PP: f64 = 0.10;

/// After trailing stop activates, sell if price drops this far from peak.
const TRAILING_DROP_PP: f64 = 0.05;

/// Close all positions when fewer than this many seconds remain in the game
/// to avoid binary resolution coin-flip risk.
const TIME_DECAY_SECS: f64 = 120.0;

/// Only scale into an existing position if current edge exceeds entry edge by
/// at least this many percentage points (prevents duplicate buys at same odds).
const SCALE_IN_EDGE_INCREASE: f64 = 0.10;

/// Maximum number of BUY trades allowed per game (condition_id).
const MAX_BUYS_PER_GAME: u32 = 3;

// ── Position tracking ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct OpenPosition {
    /// condition_id of the Polymarket market.
    condition_id: String,
    /// Token we bought (its token_id).
    token_id:     String,
    /// Team name we bet on.
    team_name:    String,
    /// "BUY_HOME" or "BUY_AWAY" — which side we hold.
    #[allow(dead_code)]
    side_label:   String,
    /// True if we bought the home outcome.
    bought_home:  bool,
    /// Market price at time of entry (home-win implied prob).
    entry_price:  f64,
    /// Best market price seen since entry (for trailing stop).
    peak_price:   f64,
    /// Model prob at time of entry.
    entry_model:  f64,
    /// Timestamp of entry.
    #[allow(dead_code)]
    entered_at:   chrono::DateTime<Utc>,
}

#[derive(Debug)]
enum SellReason {
    EdgeFlip { old_edge: f64, new_edge: f64 },
    StopLoss { entry: f64, current: f64 },
    TrailingStop { peak: f64, current: f64 },
    TimeDecay { seconds_left: f64 },
}

impl std::fmt::Display for SellReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SellReason::EdgeFlip { old_edge, new_edge } =>
                write!(f, "EDGE_FLIP(was {:+.3} now {:+.3})", old_edge, new_edge),
            SellReason::StopLoss { entry, current } =>
                write!(f, "STOP_LOSS(entry {:.3} now {:.3})", entry, current),
            SellReason::TrailingStop { peak, current } =>
                write!(f, "TRAILING_STOP(peak {:.3} now {:.3})", peak, current),
            SellReason::TimeDecay { seconds_left } =>
                write!(f, "TIME_DECAY({:.0}s left)", seconds_left),
        }
    }
}

/// Check whether an open position should be sold.
/// Returns Some(reason) if we should exit, None if we should hold.
fn check_sell_triggers(
    pos:             &OpenPosition,
    current_price:   f64,      // current home-win market prob
    model_prob:      f64,      // current model home-win prob
    game_secs_left:  f64,
    edge_confidence: f64,
) -> Option<SellReason> {
    // The "effective price" for our position:
    // If we bought home, favourable = price goes up.
    // If we bought away, favourable = price goes down (1 - price goes up).
    let (our_entry, our_current, our_peak) = if pos.bought_home {
        (pos.entry_price, current_price, pos.peak_price)
    } else {
        (1.0 - pos.entry_price, 1.0 - current_price, 1.0 - pos.peak_price)
    };

    // 1. Time decay — close before final buzzer
    if game_secs_left < TIME_DECAY_SECS && game_secs_left > 0.0 {
        return Some(SellReason::TimeDecay { seconds_left: game_secs_left });
    }

    // 2. Stop loss — market moved against us
    if our_current < our_entry - STOP_LOSS_PP {
        return Some(SellReason::StopLoss {
            entry:   our_entry,
            current: our_current,
        });
    }

    // 3. Trailing stop — we were up, now giving it back
    if our_peak >= our_entry + TRAILING_ACTIVATE_PP
        && our_current < our_peak - TRAILING_DROP_PP
    {
        return Some(SellReason::TrailingStop {
            peak:    our_peak,
            current: our_current,
        });
    }

    // 4. Edge flip — model now favours the other side with confidence
    let old_edge = pos.entry_model - pos.entry_price;
    let new_edge = model_prob - current_price;
    let flipped = (old_edge > 0.0 && new_edge < -EDGE_THRESHOLD)
               || (old_edge < 0.0 && new_edge > EDGE_THRESHOLD);
    if flipped && edge_confidence >= 0.60 {
        return Some(SellReason::EdgeFlip {
            old_edge,
            new_edge,
        });
    }

    None
}

// ── In-memory registry ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct MarketEntry {
    condition_id:    String,
    question:        String,
    token_id:        String,
    team_name:       String,
    is_home:         bool,
    home_team_id:    Option<i64>,
    away_team_id:    Option<i64>,
    game_start_time: String,
}

// ── HTTP API types ────────────────────────────────────────────────────────────

/// Row returned by GET /trades.
#[derive(Serialize)]
struct SimulatedTrade {
    id:                  String,
    timestamp:           String,
    game_id:             String,
    target_team:         String,
    action:              String,
    market_implied_prob: f64,
    model_implied_prob:  f64,
    stake_amount:        f64,
    status:              String,
    pnl:                 Option<f64>,
    /// EIP-712 order hash ("0x…") — null for legacy unsigned rows.
    order_hash:          Option<String>,
    /// 65-byte EIP-712 signature ("0x…") — null for legacy rows.
    signed_tx:           Option<String>,
    /// True if we hold the home team outcome. Dashboard flips display when false.
    bought_home:         Option<bool>,
}

/// Response for GET /wallet.
#[derive(Serialize)]
struct WalletInfo {
    address:      String,
    usdc_balance: f64,
    chain_id:     u64,
    chain:        &'static str,
    initial_usdc: f64,
}

/// Gamma single-market detail (used by settlement).
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GammaMarketDetail {
    closed:         bool,
    active:         bool,
    outcome_prices: Option<String>,
}

// ── SQLite helpers ────────────────────────────────────────────────────────────

fn init_db(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS simulated_trades (
            id                  TEXT PRIMARY KEY,
            timestamp           TEXT NOT NULL,
            game_id             TEXT NOT NULL,
            target_team         TEXT NOT NULL,
            action              TEXT NOT NULL,
            market_implied_prob REAL NOT NULL,
            model_implied_prob  REAL NOT NULL,
            stake_amount        REAL NOT NULL,
            status              TEXT NOT NULL DEFAULT 'OPEN',
            pnl                 REAL,
            order_hash          TEXT,
            signed_tx           TEXT
        );
        CREATE TABLE IF NOT EXISTS wallet_state (
            id           INTEGER PRIMARY KEY CHECK (id = 1),
            address      TEXT NOT NULL DEFAULT '',
            usdc_balance REAL NOT NULL DEFAULT 10000.0
        );",
    )?;
    // Migrate existing databases that predate the signing columns
    let _ = conn.execute("ALTER TABLE simulated_trades ADD COLUMN order_hash  TEXT",    []);
    let _ = conn.execute("ALTER TABLE simulated_trades ADD COLUMN signed_tx   TEXT",    []);
    let _ = conn.execute("ALTER TABLE simulated_trades ADD COLUMN bought_home INTEGER", []);
    info!("SQLite ready at {DB_PATH}");
    Ok(())
}

/// Initialise wallet_state row; preserves an existing balance across restarts.
fn init_wallet_state(conn: &Connection, address: &str) -> Result<()> {
    let count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM wallet_state",
        [],
        |r| r.get(0),
    )?;
    if count == 0 {
        conn.execute(
            "INSERT INTO wallet_state (id, address, usdc_balance) VALUES (1, ?1, ?2)",
            params![address, INITIAL_USDC],
        )?;
        info!("Wallet state initialised: address={address}  balance=${INITIAL_USDC}");
    } else {
        // Update the address (key is deterministic so this is idempotent)
        conn.execute(
            "UPDATE wallet_state SET address = ?1 WHERE id = 1",
            params![address],
        )?;
        let bal: f64 = conn.query_row(
            "SELECT usdc_balance FROM wallet_state WHERE id = 1",
            [],
            |r| r.get(0),
        )?;
        info!("Wallet state loaded: address={address}  balance=${bal:.2}");
    }
    Ok(())
}

/// Log a signed (or unsigned) shadow trade and deduct stake from wallet balance.
fn log_trade(
    conn:         &Connection,
    game_id:      &str,
    target_team:  &str,
    action:       &str,
    market_prob:  f64,
    model_prob:   f64,
    bought_home:  bool,
    signed_order: Option<&wallet::SignedOrder>,
) -> Result<()> {
    let id        = Uuid::new_v4().to_string();
    let timestamp = Utc::now().to_rfc3339();

    let order_hash: Option<String> = signed_order.map(|s| s.order_hash.clone());
    let signed_tx:  Option<String> = signed_order.map(|s| s.signed_tx.clone());

    // Ensure bought_home column exists (idempotent migration)
    let _ = conn.execute(
        "ALTER TABLE simulated_trades ADD COLUMN bought_home INTEGER",
        [],
    );

    // Probs stored as raw P(home wins) for reconstruction; display layer flips for away.
    conn.execute(
        "INSERT INTO simulated_trades
            (id, timestamp, game_id, target_team, action,
             market_implied_prob, model_implied_prob, stake_amount, status, pnl,
             order_hash, signed_tx, bought_home)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 'OPEN', NULL, ?9, ?10, ?11)",
        params![
            id, timestamp, game_id, target_team, action,
            market_prob, model_prob, STAKE_USDC,
            order_hash, signed_tx, bought_home as i64
        ],
    )?;

    // Deduct stake from the testnet wallet balance
    conn.execute(
        "UPDATE wallet_state SET usdc_balance = usdc_balance - ?1 WHERE id = 1",
        params![STAKE_USDC],
    )?;

    let hash_snippet = signed_order
        .map(|s| &s.order_hash[..12])
        .unwrap_or("(unsigned)");

    info!(
        "TRADE SIGNED  id={id}  game={game_id}  action={action}  \
         market={market_prob:.3}  model={model_prob:.3}  \
         edge={:+.3}  stake={STAKE_USDC} USDC  hash={hash_snippet}",
        model_prob - market_prob,
    );
    Ok(())
}

/// Log a SELL trade — credits the wallet with proceeds based on current market price.
fn log_sell_trade(
    conn:         &Connection,
    game_id:      &str,
    target_team:  &str,
    reason:       &SellReason,
    entry_price:  f64,
    exit_price:   f64,
    model_prob:   f64,
    bought_home:  bool,
    signed_order: Option<&wallet::SignedOrder>,
) -> Result<()> {
    let id        = Uuid::new_v4().to_string();
    let timestamp = Utc::now().to_rfc3339();
    let action    = format!("SELL({})", reason);

    let order_hash: Option<String> = signed_order.map(|s| s.order_hash.clone());
    let signed_tx:  Option<String> = signed_order.map(|s| s.signed_tx.clone());

    // PnL: proportional to price movement since entry
    // If bought home at 0.40, now selling at 0.55 → profit = stake * (0.55/0.40 - 1)
    let pnl = if entry_price > 0.001 {
        STAKE_USDC * (exit_price / entry_price - 1.0)
    } else {
        0.0
    };

    conn.execute(
        "INSERT INTO simulated_trades
            (id, timestamp, game_id, target_team, action,
             market_implied_prob, model_implied_prob, stake_amount, status, pnl,
             order_hash, signed_tx, bought_home)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 'CLOSED', ?9, ?10, ?11, ?12)",
        params![
            id, timestamp, game_id, target_team, action,
            exit_price, model_prob, STAKE_USDC, pnl,
            order_hash, signed_tx, bought_home as i64
        ],
    )?;

    // Credit wallet: original stake + pnl
    let balance_delta = STAKE_USDC + pnl;
    conn.execute(
        "UPDATE wallet_state SET usdc_balance = usdc_balance + ?1 WHERE id = 1",
        params![balance_delta],
    )?;

    info!(
        "TRADE SOLD  id={id}  game={game_id}  reason={reason}  \
         entry={entry_price:.3}  exit={exit_price:.3}  \
         pnl={pnl:+.2} USDC  wallet_delta={balance_delta:+.2}",
    );
    Ok(())
}

// ── HTTP handlers ─────────────────────────────────────────────────────────────

async fn handle_health() -> Json<serde_json::Value> {
    Json(serde_json::json!({"status": "ok"}))
}

async fn handle_get_trades(
    State(state): State<AppState>,
) -> Json<Vec<SimulatedTrade>> {
    let conn = state.db.lock().await;
    let mut stmt = conn
        .prepare(
            "SELECT id, timestamp, game_id, target_team, action, \
             market_implied_prob, model_implied_prob, stake_amount, status, pnl, \
             order_hash, signed_tx, bought_home \
             FROM simulated_trades ORDER BY timestamp DESC",
        )
        .unwrap();

    let trades: Vec<SimulatedTrade> = stmt
        .query_map([], |row| {
            Ok(SimulatedTrade {
                id:                  row.get(0)?,
                timestamp:           row.get(1)?,
                game_id:             row.get(2)?,
                target_team:         row.get(3)?,
                action:              row.get(4)?,
                market_implied_prob: row.get(5)?,
                model_implied_prob:  row.get(6)?,
                stake_amount:        row.get(7)?,
                status:              row.get(8)?,
                pnl:                 row.get(9)?,
                order_hash:          row.get(10)?,
                signed_tx:           row.get(11)?,
                bought_home:         row.get::<_, Option<i64>>(12).ok().flatten().map(|v| v != 0),
            })
        })
        .unwrap()
        .filter_map(|r| r.ok())
        .collect();

    Json(trades)
}

async fn handle_get_wallet(
    State(state): State<AppState>,
) -> Json<WalletInfo> {
    let conn = state.db.lock().await;
    let usdc_balance: f64 = conn
        .query_row(
            "SELECT usdc_balance FROM wallet_state WHERE id = 1",
            [],
            |r| r.get(0),
        )
        .unwrap_or(INITIAL_USDC);

    Json(WalletInfo {
        address:      state.wallet.address.clone(),
        usdc_balance,
        chain_id:     wallet::CHAIN_ID,
        chain:        "Polygon (test key — no real funds)",
        initial_usdc: INITIAL_USDC,
    })
}

async fn run_http_server(state: AppState) -> Result<()> {
    let app = Router::new()
        .route("/health", get(handle_health))
        .route("/trades", get(handle_get_trades))
        .route("/wallet", get(handle_get_wallet))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], 4000));
    info!("HTTP server listening on {addr}");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

// ── Trade settlement ──────────────────────────────────────────────────────────

async fn check_market_resolution(http: &Client, condition_id: &str) -> Option<bool> {
    let url  = format!("{GAMMA_API}/markets/{condition_id}");
    let resp = http.get(&url).send().await.ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let market: GammaMarketDetail = resp.json().await.ok()?;
    if !market.closed || market.active {
        return None;
    }
    let prices_str = market.outcome_prices?;
    let prices: Vec<String> = serde_json::from_str(&prices_str).ok()?;
    let home_price: f64 = prices.get(1)?.parse().ok()?;
    Some(home_price > 0.99)
}

/// Every 5 minutes: settle resolved markets and update wallet USDC balance.
async fn settle_trades(db: Arc<Mutex<Connection>>, http: Client) {
    loop {
        tokio::time::sleep(tokio::time::Duration::from_secs(300)).await;

        let open_trades: Vec<(String, String, f64, String, f64, f64, bool)> = {
            let conn = db.lock().await;
            let mut stmt = match conn.prepare(
                "SELECT id, game_id, market_implied_prob, action, stake_amount, model_implied_prob, \
                        COALESCE(bought_home, CASE WHEN model_implied_prob > market_implied_prob THEN 1 ELSE 0 END) \
                 FROM simulated_trades WHERE status = 'OPEN'",
            ) {
                Ok(s)  => s,
                Err(e) => { error!("settle prepare: {e}"); continue; }
            };
            stmt.query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, f64>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, f64>(4)?,
                    row.get::<_, f64>(5)?,
                    row.get::<_, i64>(6).map(|v| v != 0).unwrap_or(true),
                ))
            })
            .unwrap()
            .filter_map(|r| r.ok())
            .collect()
        };

        if open_trades.is_empty() {
            continue;
        }

        let mut resolved: std::collections::HashMap<String, Option<bool>> =
            std::collections::HashMap::new();

        for (trade_id, game_id, market_prob, action, stake_amount, _model_prob, bought_home) in &open_trades {
            let home_won = if let Some(&cached) = resolved.get(game_id) {
                cached
            } else {
                let r = check_market_resolution(&http, game_id).await;
                resolved.insert(game_id.clone(), r);
                r
            };

            let home_won = match home_won {
                Some(v) => v,
                None    => continue,
            };

            let won = if *bought_home { home_won } else { !home_won };

            let (status, pnl) = if won {
                ("WON", stake_amount * (1.0 / market_prob - 1.0))
            } else {
                ("LOST", -stake_amount)
            };

            let conn = db.lock().await;
            match conn.execute(
                "UPDATE simulated_trades SET status = ?1, pnl = ?2 WHERE id = ?3",
                params![status, pnl, trade_id],
            ) {
                Ok(_) => {
                    // Return stake + winnings to wallet.
                    // For WON: stake_amount + pnl = stake/market_prob (full return + profit).
                    // For LOST: stake_amount + pnl = 0 (stake was already deducted on entry).
                    let balance_delta = stake_amount + pnl;
                    if let Err(e) = conn.execute(
                        "UPDATE wallet_state SET usdc_balance = usdc_balance + ?1 WHERE id = 1",
                        params![balance_delta],
                    ) {
                        error!("wallet balance update failed: {e}");
                    }
                    info!(
                        "Trade settled  id={trade_id}  status={status}  \
                         pnl={pnl:+.2} USDC  wallet_delta={balance_delta:+.2}"
                    );
                }
                Err(e) => error!("settle update failed for {trade_id}: {e}"),
            }
        }
    }
}

// ── NBA team name → NBA API team ID mapping ──────────────────────────────────

fn team_name_to_id(name: &str) -> Option<i64> {
    let lower = name.to_lowercase();
    match lower.as_str() {
        "hawks"           | "atlanta"      => Some(1610612737),
        "celtics"         | "boston"        => Some(1610612738),
        "nets"            | "brooklyn"     => Some(1610612751),
        "hornets"         | "charlotte"    => Some(1610612766),
        "bulls"           | "chicago"      => Some(1610612741),
        "cavaliers" | "cavs" | "cleveland" => Some(1610612739),
        "mavericks" | "mavs" | "dallas"    => Some(1610612742),
        "nuggets"         | "denver"       => Some(1610612743),
        "pistons"         | "detroit"      => Some(1610612765),
        "warriors"        | "golden state" => Some(1610612744),
        "rockets"         | "houston"      => Some(1610612745),
        "pacers"          | "indiana"      => Some(1610612754),
        "clippers"        | "la clippers"  => Some(1610612746),
        "lakers"          | "la lakers" | "los angeles lakers" => Some(1610612747),
        "grizzlies"       | "memphis"      => Some(1610612763),
        "heat"            | "miami"        => Some(1610612748),
        "bucks"           | "milwaukee"    => Some(1610612749),
        "timberwolves" | "wolves" | "minnesota" => Some(1610612750),
        "pelicans"        | "new orleans"  => Some(1610612740),
        "knicks"          | "new york"     => Some(1610612752),
        "thunder"         | "oklahoma city" | "okc" => Some(1610612760),
        "magic"           | "orlando"      => Some(1610612753),
        "76ers" | "sixers" | "philadelphia" => Some(1610612755),
        "suns"            | "phoenix"      => Some(1610612756),
        "trail blazers" | "blazers" | "portland" => Some(1610612757),
        "kings"           | "sacramento"   => Some(1610612758),
        "spurs"           | "san antonio"  => Some(1610612759),
        "raptors"         | "toronto"      => Some(1610612761),
        "jazz"            | "utah"         => Some(1610612762),
        "wizards"         | "washington"   => Some(1610612764),
        _ => None,
    }
}

// ── Gamma API: discover live NBA moneyline markets ───────────────────────────

async fn fetch_nba_markets(http: &Client) -> Result<Vec<Market>> {
    use std::time::{SystemTime, UNIX_EPOCH};

    let now_secs  = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs() as i64;
    let cutoff    = now_secs + 48 * 3600;

    let mut markets = Vec::new();
    let mut offset  = 0usize;
    let limit       = 100usize;

    loop {
        let url = format!(
            "{GAMMA_API}/events?tag_slug=nba&active=true&closed=false\
             &limit={limit}&offset={offset}"
        );
        let events: Vec<GammaEvent> = http
            .get(&url).send().await.context("GET Gamma /events failed")?
            .json().await.context("deserialise Gamma /events failed")?;

        let page_len = events.len();

        for event in events {
            if event.closed || !event.active { continue; }
            if !event.title.contains("vs.")  { continue; }

            for gm in event.markets {
                if gm.closed || !gm.active { continue; }
                if gm.sports_market_type.as_deref() != Some("moneyline") { continue; }

                let game_start_time = gm.game_start_time.clone().unwrap_or_default();
                if !game_start_time.is_empty() {
                    if let Ok(ts) = parse_game_time(&game_start_time) {
                        if ts > cutoff { continue; }
                    }
                }

                let token_ids: Vec<String> = match &gm.clob_token_ids {
                    Some(s) => serde_json::from_str(s).unwrap_or_default(),
                    None    => continue,
                };
                if token_ids.len() < 2 { continue; }

                let team_names: Vec<String> = match &gm.outcomes {
                    Some(s) => serde_json::from_str(s).unwrap_or_else(|_| {
                        vec!["Away".into(), "Home".into()]
                    }),
                    None => vec!["Away".into(), "Home".into()],
                };

                let prices: Vec<f64> = match &gm.outcome_prices {
                    Some(s) => {
                        let raw: Vec<String> = serde_json::from_str(s).unwrap_or_default();
                        raw.iter().map(|p| p.parse().unwrap_or(0.5)).collect()
                    }
                    None => vec![0.5; token_ids.len()],
                };

                let tokens: Vec<Token> = token_ids.into_iter().enumerate().map(|(i, id)| {
                    let name = team_names.get(i).cloned().unwrap_or_else(|| "Unknown".into());
                    let tid = team_name_to_id(&name);
                    Token {
                        token_id:  id,
                        team_name: name,
                        is_home:   i == 1,
                        team_id:   tid,
                        price:     *prices.get(i).unwrap_or(&0.5),
                    }
                }).collect();

                info!(
                    "Game market: \"{}\"  start={}  away={} ({:.0}%)  home={} ({:.0}%)",
                    gm.question, game_start_time,
                    tokens[0].team_name, tokens[0].price * 100.0,
                    tokens[1].team_name, tokens[1].price * 100.0,
                );

                markets.push(Market {
                    condition_id:    gm.condition_id,
                    question:        gm.question,
                    game_start_time,
                    tokens,
                });
            }
        }

        if page_len < limit { break; }
        offset += limit;
    }

    info!("Found {} live NBA game moneyline markets", markets.len());
    Ok(markets)
}

fn parse_game_time(s: &str) -> Result<i64> {
    let normalised = s.replace(' ', "T").replace("+00", "+00:00");
    let dt = chrono::DateTime::parse_from_rfc3339(&normalised)
        .context("parse game_start_time")?;
    Ok(dt.timestamp())
}

// ── Alpha Engine ──────────────────────────────────────────────────────────────

#[allow(dead_code)]
async fn get_model_prob(http: &Client, gs: &GameStateRequest) -> Result<f64> {
    let resp: PredictResponse = http
        .post(format!("{ALPHA_ENGINE_URL}/predict"))
        .json(gs)
        .send().await.context("POST /predict failed")?
        .json().await.context("deserialise /predict failed")?;

    if !resp.model_loaded {
        warn!("Alpha Engine returned naive prior (model not loaded)");
    }
    Ok(resp.win_probability)
}

/// Search pre-fetched server.py /games JSON for a game matching these team IDs.
/// Returns full predictions from the v2 pipeline (265 features, boxscore, momentum).
fn find_server_prediction(
    games_data: &serde_json::Value,
    home_team_id: i64,
    away_team_id: i64,
) -> Option<ServerPrediction> {
    let games = games_data.get("games")?.as_array()?;

    for game in games {
        let g_home = game.get("home_team_id")?.as_i64()?;
        let g_away = game.get("away_team_id")?.as_i64()?;

        if g_home != home_team_id || g_away != away_team_id {
            continue;
        }

        // Extract game clock for seconds-left computation
        let period = game.get("period")?.as_i64()? as i32;
        let game_clock = game.get("game_clock").and_then(|v| v.as_str()).unwrap_or("");
        let game_seconds_left = {
            let clock = game_clock.replace("PT", "").replace("S", "");
            let parts: Vec<&str> = clock.split('M').collect();
            let mins: f64 = parts.first().and_then(|s| s.parse().ok()).unwrap_or(0.0);
            let secs: f64 = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0.0);
            let remaining_periods = (4 - period).max(0) as f64;
            remaining_periods * 720.0 + mins * 60.0 + secs
        };

        // Extract full v2 predictions
        let preds = game.get("predictions")?;
        let win_probability = preds.get("win_probability")?.as_f64()?;
        let edge_confidence = preds.get("edge_confidence")?.as_f64()?;

        return Some(ServerPrediction {
            game_seconds_left,
            win_probability,
            edge_confidence,
        });
    }

    None
}

fn parse_price(v: &serde_json::Value) -> Option<f64> {
    match v {
        serde_json::Value::Number(n) => n.as_f64(),
        serde_json::Value::String(s) => s.parse().ok(),
        _ => None,
    }
}

// ── Position persistence (prevents duplicate buys across WS reconnections) ──

/// Load existing OPEN BUY trades from SQLite and reconstruct position state.
/// Called at the start of each WS session so the bot knows what it already holds.
fn load_open_positions(
    conn: &Connection,
    registry: &std::collections::HashMap<String, MarketEntry>,
) -> (
    std::collections::HashMap<String, OpenPosition>,
    std::collections::HashMap<String, u32>,
) {
    let mut positions = std::collections::HashMap::new();
    let mut buy_counts: std::collections::HashMap<String, u32> =
        std::collections::HashMap::new();

    let mut stmt = match conn.prepare(
        "SELECT game_id, action, market_implied_prob, model_implied_prob, timestamp, \
                COALESCE(bought_home, CASE WHEN model_implied_prob > market_implied_prob THEN 1 ELSE 0 END) \
         FROM simulated_trades WHERE status = 'OPEN' AND action LIKE 'BUY_%' \
         ORDER BY timestamp ASC",
    ) {
        Ok(s) => s,
        Err(e) => {
            error!("Failed to load open positions: {e}");
            return (positions, buy_counts);
        }
    };

    let rows: Vec<(String, String, f64, f64, String, bool)> = stmt
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, f64>(2)?,
                row.get::<_, f64>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, i64>(5).map(|v| v != 0).unwrap_or(false),
            ))
        })
        .unwrap()
        .filter_map(|r| r.ok())
        .collect();

    for (cond_id, action, entry_price, entry_model, ts, bought_home) in rows {
        *buy_counts.entry(cond_id.clone()).or_insert(0) += 1;

        // Only keep the earliest trade as the reference position
        if positions.contains_key(&cond_id) {
            continue;
        }

        // Resolve token info from the current session's registry
        let (token_id, team_name) = registry
            .values()
            .find(|e| e.condition_id == cond_id && e.is_home == bought_home)
            .map(|e| (e.token_id.clone(), e.team_name.clone()))
            .unwrap_or_else(|| {
                let team = action
                    .strip_prefix("BUY_")
                    .unwrap_or(&action)
                    .to_string();
                (String::new(), team)
            });

        let entered_at = chrono::DateTime::parse_from_rfc3339(&ts)
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_else(|_| Utc::now());

        positions.insert(
            cond_id.clone(),
            OpenPosition {
                condition_id: cond_id,
                token_id,
                team_name,
                side_label: action,
                bought_home,
                entry_price,
                peak_price: entry_price, // conservative: reset peak to entry
                entry_model,
                entered_at,
            },
        );
    }

    if !positions.is_empty() {
        info!(
            "Restored {} open positions from DB ({} total buys across all games)",
            positions.len(),
            buy_counts.values().sum::<u32>(),
        );
    }

    (positions, buy_counts)
}

// ── WebSocket ingestion ───────────────────────────────────────────────────────

async fn run_ws_ingestion(
    markets: Vec<Market>,
    http:    Client,
    db:      Arc<Mutex<Connection>>,
    wallet:  Arc<wallet::TestWallet>,
) -> Result<()> {
    let mut registry: std::collections::HashMap<String, MarketEntry> =
        std::collections::HashMap::new();
    let mut all_token_ids: Vec<String> = Vec::new();

    for m in &markets {
        let home_tid = m.tokens.iter().find(|t| t.is_home).and_then(|t| t.team_id);
        let away_tid = m.tokens.iter().find(|t| !t.is_home).and_then(|t| t.team_id);

        for t in &m.tokens {
            registry.insert(
                t.token_id.clone(),
                MarketEntry {
                    condition_id:    m.condition_id.clone(),
                    question:        m.question.clone(),
                    token_id:        t.token_id.clone(),
                    team_name:       t.team_name.clone(),
                    is_home:         t.is_home,
                    home_team_id:    home_tid,
                    away_team_id:    away_tid,
                    game_start_time: m.game_start_time.clone(),
                },
            );
            all_token_ids.push(t.token_id.clone());
        }
    }

    // Load existing open positions from DB so we don't re-buy after WS reconnection
    let (mut positions, mut buy_counts) = {
        let conn = db.lock().await;
        load_open_positions(&conn, &registry)
    };

    if all_token_ids.is_empty() {
        warn!("No NBA markets found — WebSocket will subscribe to empty list.");
        return Ok(());
    }

    info!(
        "Connecting to Polymarket WebSocket — {} tokens / {} markets",
        all_token_ids.len(), markets.len()
    );

    let (mut ws_stream, _) = connect_async(POLYMARKET_WS)
        .await
        .context("WebSocket connect failed")?;
    info!("WebSocket connected");

    let sub      = WsSubscribe { msg_type: "subscribe", channel: "price_change", assets_ids: all_token_ids };
    let sub_json = serde_json::to_string(&sub)?;
    ws_stream.send(Message::Text(sub_json)).await?;
    info!("Subscription sent — tracking {} open positions", positions.len());

    while let Some(msg) = ws_stream.next().await {
        let msg = match msg {
            Ok(m)  => m,
            Err(e) => { error!("WS recv error: {e}"); break; }
        };

        let text = match msg {
            Message::Text(t)  => t,
            Message::Ping(d)  => {
                if let Err(e) = ws_stream.send(Message::Pong(d)).await {
                    error!("pong failed: {e}");
                }
                continue;
            }
            Message::Close(_) => { warn!("WebSocket closed by server"); break; }
            _ => continue,
        };

        let raw: serde_json::Value = match serde_json::from_str(&text) {
            Ok(v)  => v,
            Err(e) => { warn!("JSON parse error: {e}"); continue; }
        };

        let ticks: Vec<serde_json::Value> = if raw.is_array() {
            raw.as_array().cloned().unwrap_or_default()
        } else {
            vec![raw]
        };

        // Fetch server.py /games once per WS message (covers all ticks in batch)
        let server_games: Option<serde_json::Value> = match http
            .get(format!("{SERVER_URL}/games"))
            .send().await
        {
            Ok(r) if r.status().is_success() => r.json().await.ok(),
            _ => None,
        };

        for tick_val in ticks {
            let event_type = tick_val
                .get("event_type").or_else(|| tick_val.get("type"))
                .and_then(|v| v.as_str()).unwrap_or("");
            if !event_type.contains("price") && !event_type.is_empty() { continue; }

            let asset_id = match tick_val.get("asset_id").and_then(|v| v.as_str()) {
                Some(id) => id.to_string(),
                None     => continue,
            };
            let market_prob = match tick_val.get("price").and_then(parse_price) {
                Some(p) => p,
                None    => continue,
            };
            if !(0.01..=0.99).contains(&market_prob) { continue; }

            let entry = match registry.get(&asset_id) {
                Some(e) => e.clone(),
                None    => continue,
            };

            info!(
                "Tick  game=\"{}\"  team={}  is_home={}  market_prob={:.3}",
                entry.question, entry.team_name, entry.is_home, market_prob
            );

            // Only process home token — gives P(home wins) directly.
            if !entry.is_home { continue; }

            // Resolve team IDs
            let home_tid = entry.home_team_id.unwrap_or(0);
            let away_tid = entry.away_team_id.unwrap_or(0);

            // ── Tier 1: Use server.py full v2 predictions (265 features) ─────
            // ── Tier 2: Fall back to alpha-engine (6 fields, ~93 features zero)
            let (model_prob, secs_left, edge_confidence) =
                if let Some(sp) = server_games.as_ref()
                    .and_then(|g| find_server_prediction(g, home_tid, away_tid))
                {
                    (sp.win_probability, sp.game_seconds_left, sp.edge_confidence)
                } else {
                    // Alpha-engine fallback sends pregame state (scores=0).
                    // This is only valid before the game starts — for live games it
                    // produces nonsense predictions.  Skip rather than trade blind.
                    let game_started = if !entry.game_start_time.is_empty() {
                        parse_game_time(&entry.game_start_time)
                            .map(|ts| ts < Utc::now().timestamp())
                            .unwrap_or(false)
                    } else {
                        false
                    };
                    if game_started {
                        warn!(
                            "server.py unavailable for live game \"{}\" — \
                             skipping to avoid stale pregame prediction",
                            entry.question
                        );
                        continue;
                    }

                    // Alpha-engine fallback — pregame markets only
                    let game_state = GameStateRequest {
                        home_team_id: home_tid, away_team_id: away_tid,
                        period: 1, game_seconds_left: 2880.0,
                        home_score: 0.0, away_score: 0.0,
                    };
                    match http
                        .post(format!("{ALPHA_ENGINE_URL}/predict"))
                        .json(&game_state)
                        .send().await
                    {
                        Ok(r) => match r.json::<PredictResponse>().await {
                            Ok(p) => (p.win_probability, 2880.0, p.edge_confidence),
                            Err(e) => { warn!("Alpha Engine deserialise error: {e}"); continue; }
                        },
                        Err(e) => { warn!("Alpha Engine error: {e}"); continue; }
                    }
                };

            let edge = model_prob - market_prob;

            // ── SELL CHECK: do we have an open position on this game? ────
            let mut should_sell: Option<(SellReason, OpenPosition)> = None;

            if let Some(pos) = positions.get_mut(&entry.condition_id) {
                // Update peak price (stored as home-win prob) for trailing stop
                if pos.bought_home && market_prob > pos.peak_price {
                    pos.peak_price = market_prob;
                } else if !pos.bought_home && market_prob < pos.peak_price {
                    pos.peak_price = market_prob;
                }

                if let Some(reason) = check_sell_triggers(
                    pos, market_prob, model_prob, secs_left, edge_confidence,
                ) {
                    should_sell = Some((reason, pos.clone()));
                }
            }

            if let Some((reason, pos)) = should_sell {
                info!(
                    "SELL SIGNAL  game=\"{}\"  reason={}  positions_open={}",
                    entry.question, reason, positions.len()
                );

                let exit_price  = if pos.bought_home { market_prob } else { 1.0 - market_prob };
                let sell_entry  = if pos.bought_home { pos.entry_price } else { 1.0 - pos.entry_price };

                let signed = wallet.sign_order(
                    &pos.token_id, market_prob, STAKE_USDC, wallet::Side::Sell,
                ).ok();

                let db_guard = db.lock().await;
                if let Err(e) = log_sell_trade(
                    &db_guard,
                    &pos.condition_id,
                    &pos.team_name,
                    &reason,
                    sell_entry,
                    exit_price,
                    model_prob,
                    pos.bought_home,
                    signed.as_ref(),
                ) {
                    error!("DB sell write failed: {e}");
                }
                drop(db_guard);

                positions.remove(&entry.condition_id);
                buy_counts.remove(&entry.condition_id);
                info!("Position closed — {} open games remaining", positions.len());
                continue;
            }

            // If we have a position but no sell signal, only allow scale-in
            // when the edge has grown significantly on the same side
            if positions.contains_key(&entry.condition_id) {
                let pos = positions.get(&entry.condition_id).unwrap();
                let entry_edge = pos.entry_model - pos.entry_price;
                let count = buy_counts.get(&entry.condition_id).copied().unwrap_or(1);

                let same_side = (edge > 0.0 && entry_edge > 0.0)
                             || (edge < 0.0 && entry_edge < 0.0);
                let edge_grown = edge.abs() >= entry_edge.abs() + SCALE_IN_EDGE_INCREASE;

                if count >= MAX_BUYS_PER_GAME || !same_side || !edge_grown {
                    continue;
                }
                info!(
                    "SCALE-IN  game=\"{}\"  entry_edge={:+.3}  current_edge={:+.3}  \
                     buy #{}/{}",
                    entry.question, entry_edge, edge, count + 1, MAX_BUYS_PER_GAME
                );
            }

            // ── BUY CHECK: no open position, look for edge ──────────────
            // Only buy when model has a POSITIVE edge on the chosen side:
            //   edge > threshold  → model says home is underpriced → buy home token
            //   edge < -threshold → model says home is overpriced  → buy away token
            if edge.abs() < EDGE_THRESHOLD { continue; }
            if edge_confidence < 0.60 {
                info!(
                    "Edge {edge:+.3} found but confidence too low ({:.1}%) — skipping",
                    edge_confidence * 100.0
                );
                continue;
            }

            let bought_home = edge > 0.0;

            // Resolve the away team name from the registry
            let away_team_name = registry.values()
                .find(|e| e.condition_id == entry.condition_id && !e.is_home)
                .map(|e| e.team_name.clone())
                .unwrap_or_else(|| "AWAY".to_string());

            let action = if bought_home {
                format!("BUY_{}", entry.team_name.to_uppercase().replace(' ', "_"))
            } else {
                format!("BUY_{}", away_team_name.to_uppercase().replace(' ', "_"))
            };

            info!(
                "EDGE FOUND  game=\"{}\"  home={}  edge={edge:+.3}  conf={:.1}%  action={action}",
                entry.question, entry.team_name, edge_confidence * 100.0
            );

            // Sign the buy order — Buy for home token, Sell for away (short home)
            let side = if bought_home { wallet::Side::Buy } else { wallet::Side::Sell };
            let signed = wallet.sign_order(&entry.token_id, market_prob, STAKE_USDC, side).ok();

            if signed.is_none() {
                warn!("Order signing failed — logging trade without signature");
            }

            let db_guard = db.lock().await;
            if let Err(e) = log_trade(
                &db_guard,
                &entry.condition_id,
                &entry.question,
                &action,
                market_prob,
                model_prob,
                bought_home,
                signed.as_ref(),
            ) {
                error!("DB write failed: {e}");
            }
            drop(db_guard);

            // Track the open position (only store first entry as reference)
            if !positions.contains_key(&entry.condition_id) {
                positions.insert(entry.condition_id.clone(), OpenPosition {
                    condition_id: entry.condition_id.clone(),
                    token_id:     entry.token_id.clone(),
                    team_name:    entry.team_name.clone(),
                    side_label:   action,
                    bought_home,
                    entry_price:  market_prob,
                    peak_price:   market_prob,
                    entry_model:  model_prob,
                    entered_at:   Utc::now(),
                });
            }
            *buy_counts.entry(entry.condition_id.clone()).or_insert(0) += 1;
            let count = buy_counts[&entry.condition_id];
            info!(
                "Position buy #{count}/{MAX_BUYS_PER_GAME} — {} open games total",
                positions.len()
            );
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;

    #[test]
    fn test_db_initialization() {
        let conn = Connection::open_in_memory().unwrap();
        init_db(&conn).expect("init_db failed");

        // Verify tables exist
        let mut stmt = conn.prepare("SELECT name FROM sqlite_master WHERE type='table' AND name='simulated_trades'").unwrap();
        assert!(stmt.exists([]).unwrap());

        let mut stmt = conn.prepare("SELECT name FROM sqlite_master WHERE type='table' AND name='wallet_state'").unwrap();
        assert!(stmt.exists([]).unwrap());
    }

    #[test]
    fn test_init_wallet_state() {
        let conn = Connection::open_in_memory().unwrap();
        init_db(&conn).unwrap();
        
        let address = "0x1234567890123456789012345678901234567890";
        init_wallet_state(&conn, address).expect("init_wallet_state failed");

        let (addr, bal): (String, f64) = conn.query_row(
            "SELECT address, usdc_balance FROM wallet_state WHERE id = 1",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        ).unwrap();

        assert_eq!(addr, address);
        assert_eq!(bal, INITIAL_USDC);

        // Test idempotency and balance preservation
        let new_bal = 5000.0;
        conn.execute("UPDATE wallet_state SET usdc_balance = ?1 WHERE id = 1", params![new_bal]).unwrap();
        
        init_wallet_state(&conn, address).unwrap();
        let bal_after: f64 = conn.query_row("SELECT usdc_balance FROM wallet_state WHERE id = 1", [], |r| r.get(0)).unwrap();
        assert_eq!(bal_after, new_bal);
    }
}

// ── Entry point ───────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("execution_engine=info".parse()?)
                .add_directive("info".parse()?),
        )
        .init();

    info!("NBA Execution Engine v2 (testnet signing) starting…");

    // ── Wallet
    let wallet = Arc::new(
        wallet::TestWallet::new().context("failed to initialise test wallet")?
    );
    info!("Test wallet address : {}", wallet.address);
    info!("Chain               : Polygon (ID {}) — no real funds", wallet::CHAIN_ID);

    // ── Database
    let conn = Connection::open(DB_PATH).context("open SQLite")?;
    init_db(&conn)?;
    init_wallet_state(&conn, &wallet.address)?;
    let db = Arc::new(Mutex::new(conn));

    // ── HTTP client
    let http = Client::builder()
        .user_agent("nba-execution-engine/2.0")
        .timeout(std::time::Duration::from_secs(10))
        .build()?;

    // ── Shared app state
    let app_state = AppState { db: Arc::clone(&db), wallet: Arc::clone(&wallet) };

    // ── Spawn HTTP API server (port 4000)
    {
        let state = app_state.clone();
        tokio::spawn(async move {
            if let Err(e) = run_http_server(state).await {
                error!("HTTP server error: {e}");
            }
        });
    }

    // ── Spawn trade settlement (every 5 min)
    {
        let db_settle   = Arc::clone(&db);
        let http_settle = http.clone();
        tokio::spawn(async move { settle_trades(db_settle, http_settle).await; });
    }

    // ── Check server.py (primary prediction source — full v2 pipeline)
    match http.get(format!("{SERVER_URL}/health")).send().await {
        Ok(r) if r.status().is_success() => {
            let body: serde_json::Value = r.json().await.unwrap_or_default();
            info!("server.py healthy (primary predictions): {body}");
        }
        Ok(r)  => warn!("server.py HTTP {} — will fall back to alpha-engine", r.status()),
        Err(e) => warn!("server.py unreachable ({e}) — will fall back to alpha-engine"),
    }

    // ── Check Alpha Engine (fallback when server.py doesn't track a game)
    match http.get(format!("{ALPHA_ENGINE_URL}/health")).send().await {
        Ok(r) if r.status().is_success() => {
            let body: serde_json::Value = r.json().await.unwrap_or_default();
            info!("Alpha Engine healthy (fallback): {body}");
        }
        Ok(r)  => warn!("Alpha Engine HTTP {}", r.status()),
        Err(e) => warn!("Alpha Engine unreachable ({e}) — no fallback available"),
    }

    // ── Market discovery + WebSocket ingestion (retry loop)
    loop {
        let markets = fetch_nba_markets(&http).await.unwrap_or_else(|e| {
            error!("Market discovery failed: {e}");
            vec![]
        });

        if markets.is_empty() {
            warn!("No active NBA markets on Polymarket — retrying in 60 s…");
            tokio::time::sleep(tokio::time::Duration::from_secs(60)).await;
            continue;
        }

        info!("(Re)connecting WebSocket ingestion loop…");
        match run_ws_ingestion(markets, http.clone(), Arc::clone(&db), Arc::clone(&wallet)).await {
            Ok(_)  => warn!("WS loop exited cleanly — reconnecting in 5 s"),
            Err(e) => error!("WS loop error: {e} — reconnecting in 5 s"),
        }
        tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
    }
}
