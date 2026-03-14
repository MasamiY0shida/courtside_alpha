// ============================================================================
//  NBA Execution Engine — Module B
//  Polymarket live odds ingestion + shadow trading
//
//  Flow:
//    1. Query Polymarket REST API → discover live NBA markets
//    2. Open WebSocket to Polymarket CLOB → stream price ticks
//    3. On each tick → ask Alpha Engine for model probability
//    4. If |model_prob - market_prob| > EDGE_THRESHOLD → log shadow trade
//    5. All trades land in SQLite `simulated_trades` table
// ============================================================================

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

/// Gamma API — market/event discovery (supports tag_slug filtering).
const GAMMA_API:        &str = "https://gamma-api.polymarket.com";
const POLYMARKET_WS:    &str = "wss://ws-subscriptions-clob.polymarket.com/ws/market";
const ALPHA_ENGINE_URL: &str = "http://127.0.0.1:8000";
const DB_PATH:          &str = "../trades.sqlite";

/// Minimum edge (model_prob - market_prob) required to trigger a shadow trade.
const EDGE_THRESHOLD: f64 = 0.05;

/// Simulated USDC stake per trade.
const STAKE_USDC: f64 = 50.0;

// ── Polymarket types ──────────────────────────────────────────────────────────

/// Canonical market representation used throughout the engine.
#[derive(Debug, Clone)]
struct Market {
    condition_id:    String,
    /// e.g. "Grizzlies vs. Pistons"
    question:        String,
    /// UTC game tip-off time from Polymarket, e.g. "2026-03-14 01:00:00+00"
    game_start_time: String,
    tokens:          Vec<Token>,
}

/// One side of a moneyline market — maps to a specific team winning.
#[derive(Debug, Clone)]
struct Token {
    token_id:  String,
    /// Team name, e.g. "Grizzlies" or "Pistons"
    team_name: String,
    /// NBA convention: outcomes[0] = away team, outcomes[1] = home team
    is_home:   bool,
    /// Last known price from REST snapshot; WebSocket ticks override this.
    #[allow(dead_code)]
    price:     f64,
}

// ── Gamma API types (market discovery) ───────────────────────────────────────

/// Top-level event returned by the Gamma /events endpoint.
#[derive(Debug, Deserialize)]
struct GammaEvent {
    #[allow(dead_code)]
    title:   String,
    active:  bool,
    closed:  bool,
    markets: Vec<GammaMarket>,
}

/// Individual market nested inside a GammaEvent.
/// Field names are camelCase in JSON; `rename_all` handles the mapping.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GammaMarket {
    question:           String,
    condition_id:       String,
    /// JSON-encoded string: `["token_id_away", "token_id_home"]`
    clob_token_ids:     Option<String>,
    /// JSON-encoded string: `["AwayTeam", "HomeTeam"]`
    outcomes:           Option<String>,
    /// JSON-encoded string: `["0.45", "0.55"]` — may be null
    outcome_prices:     Option<String>,
    /// "moneyline" | "spreads" | "totals" | "points" | etc.
    sports_market_type: Option<String>,
    /// UTC tip-off timestamp, e.g. "2026-03-14 01:00:00+00"
    game_start_time:    Option<String>,
    active:             bool,
    closed:             bool,
}

// ── WebSocket message types ───────────────────────────────────────────────────

/// Outbound: subscribe to price-change events for a list of asset IDs.
#[derive(Serialize)]
struct WsSubscribe {
    #[serde(rename = "type")]
    msg_type:  &'static str,
    channel:   &'static str,
    assets_ids: Vec<String>,
}

// Polymarket WS messages are parsed directly as serde_json::Value
// because the schema varies between price_change, book, and last_trade_price events.

// ── Alpha Engine types ────────────────────────────────────────────────────────

#[derive(Serialize)]
struct GameStateRequest {
    period:           i32,
    game_seconds_left: f64,
    home_score:       f64,
    away_score:       f64,
    margin:           f64,
    abs_margin:       f64,
    total_points:     f64,
    scoring_pace:     f64,
    // remaining fields default to 0 — FastAPI fills them
}

#[derive(Deserialize)]
struct PredictResponse {
    win_probability: f64,
    model_loaded:    bool,
}

// ── In-memory market registry ─────────────────────────────────────────────────

/// Maps a Polymarket token_id → enough context to log a meaningful trade.
#[derive(Debug, Clone)]
struct MarketEntry {
    condition_id:    String,
    question:        String,
    #[allow(dead_code)]
    token_id:        String,
    team_name:       String,
    /// True when this token represents the home team winning.
    /// Market price of this token = P(home wins) directly.
    is_home:         bool,
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
}

/// Gamma API response for a single market (used by settlement).
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
            pnl                 REAL
        );",
    )?;
    info!("SQLite ready at {DB_PATH}");
    Ok(())
}

fn log_trade(
    conn:        &Connection,
    game_id:     &str,
    target_team: &str,
    action:      &str,      // "BUY_YES" | "BUY_NO"
    market_prob: f64,
    model_prob:  f64,
) -> Result<()> {
    let id        = Uuid::new_v4().to_string();
    let timestamp = Utc::now().to_rfc3339();

    conn.execute(
        "INSERT INTO simulated_trades
            (id, timestamp, game_id, target_team, action,
             market_implied_prob, model_implied_prob, stake_amount, status, pnl)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 'OPEN', NULL)",
        params![
            id, timestamp, game_id, target_team, action,
            market_prob, model_prob, STAKE_USDC
        ],
    )?;

    info!(
        "TRADE LOGGED  id={id}  game={game_id}  action={action}  \
         market={market_prob:.3}  model={model_prob:.3}  \
         edge={:.3}  stake={STAKE_USDC} USDC",
        model_prob - market_prob
    );
    Ok(())
}

// ── HTTP API server ───────────────────────────────────────────────────────────

async fn handle_health() -> Json<serde_json::Value> {
    Json(serde_json::json!({"status": "ok"}))
}

async fn handle_get_trades(
    State(db): State<Arc<Mutex<Connection>>>,
) -> Json<Vec<SimulatedTrade>> {
    let conn = db.lock().await;
    let mut stmt = conn
        .prepare(
            "SELECT id, timestamp, game_id, target_team, action, \
             market_implied_prob, model_implied_prob, stake_amount, status, pnl \
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
            })
        })
        .unwrap()
        .filter_map(|r| r.ok())
        .collect();

    Json(trades)
}

async fn run_http_server(db: Arc<Mutex<Connection>>) -> Result<()> {
    let app = Router::new()
        .route("/health", get(handle_health))
        .route("/trades", get(handle_get_trades))
        .layer(CorsLayer::permissive())
        .with_state(db);

    let addr = SocketAddr::from(([0, 0, 0, 0], 4000));
    info!("HTTP server listening on {addr}");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

// ── Trade settlement ──────────────────────────────────────────────────────────

/// Query the Gamma API for a single market's resolution status.
/// Returns Some(true) if home team won, Some(false) if away team won, None if unresolved.
async fn check_market_resolution(http: &Client, condition_id: &str) -> Option<bool> {
    let url = format!("{GAMMA_API}/markets/{condition_id}");
    let resp = http.get(&url).send().await.ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let market: GammaMarketDetail = resp.json().await.ok()?;
    if !market.closed || market.active {
        return None; // not yet resolved
    }
    // outcome_prices: ["away_price", "home_price"] — winner has price "1"
    let prices_str = market.outcome_prices?;
    let prices: Vec<String> = serde_json::from_str(&prices_str).ok()?;
    let home_price: f64 = prices.get(1)?.parse().ok()?;
    Some(home_price > 0.99)
}

/// Background task: every 5 minutes, check open trades and settle resolved markets.
async fn settle_trades(db: Arc<Mutex<Connection>>, http: Client) {
    loop {
        tokio::time::sleep(tokio::time::Duration::from_secs(300)).await;

        // Collect open trades (release lock immediately after)
        let open_trades: Vec<(String, String, f64, String)> = {
            let conn = db.lock().await;
            let mut stmt = match conn.prepare(
                "SELECT id, game_id, market_implied_prob, action \
                 FROM simulated_trades WHERE status = 'OPEN'",
            ) {
                Ok(s) => s,
                Err(e) => { error!("settle query prepare failed: {e}"); continue; }
            };
            stmt.query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, f64>(2)?,
                    row.get::<_, String>(3)?,
                ))
            })
            .unwrap()
            .filter_map(|r| r.ok())
            .collect()
        };

        if open_trades.is_empty() {
            continue;
        }

        // Cache resolution results per game_id to avoid duplicate API calls
        let mut resolved: std::collections::HashMap<String, Option<bool>> =
            std::collections::HashMap::new();

        for (trade_id, game_id, market_prob, action) in &open_trades {
            let home_won = if let Some(&cached) = resolved.get(game_id) {
                cached
            } else {
                let result = check_market_resolution(&http, game_id).await;
                resolved.insert(game_id.clone(), result);
                result
            };

            let home_won = match home_won {
                Some(v) => v,
                None => continue, // market not yet resolved
            };

            // Determine outcome: "BUY_AWAY" backs away team; everything else backs home
            let won = if action == "BUY_AWAY" { !home_won } else { home_won };

            let (status, pnl) = if won {
                ("WON", STAKE_USDC * (1.0 / market_prob - 1.0))
            } else {
                ("LOST", -STAKE_USDC)
            };

            let conn = db.lock().await;
            match conn.execute(
                "UPDATE simulated_trades SET status = ?1, pnl = ?2 WHERE id = ?3",
                params![status, pnl, trade_id],
            ) {
                Ok(_) => info!(
                    "Trade settled  id={trade_id}  status={status}  pnl={pnl:.2} USDC"
                ),
                Err(e) => error!("settle update failed for {trade_id}: {e}"),
            }
        }
    }
}

// ── Gamma API: fetch live NBA game moneyline markets ─────────────────────────

/// Only ingest moneyline markets for games starting within the next 48 hours.
/// This ensures we're pricing live game outcomes, not season-long futures.
async fn fetch_nba_markets(http: &Client) -> Result<Vec<Market>> {
    use std::time::{SystemTime, UNIX_EPOCH};

    let now_secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    // Accept games whose tip-off is up to 48 h from now (includes tonight + tomorrow)
    let cutoff_secs = now_secs + 48 * 3600;

    let mut markets: Vec<Market> = Vec::new();
    let mut offset = 0usize;
    let limit = 100usize;

    loop {
        let url = format!(
            "{GAMMA_API}/events?tag_slug=nba&active=true&closed=false\
             &limit={limit}&offset={offset}"
        );

        let events: Vec<GammaEvent> = http
            .get(&url)
            .send()
            .await
            .context("GET Gamma /events failed")?
            .json()
            .await
            .context("deserialise Gamma /events failed")?;

        let page_len = events.len();

        for event in events {
            if event.closed || !event.active {
                continue;
            }

            // Only process game events (title contains "vs.")
            if !event.title.contains("vs.") {
                continue;
            }

            for gm in event.markets {
                if gm.closed || !gm.active {
                    continue;
                }

                // Only moneyline — the straight "which team wins" market
                if gm.sports_market_type.as_deref() != Some("moneyline") {
                    continue;
                }

                // Gate by game start time — skip futures beyond 48 h
                let game_start_time = gm.game_start_time.clone().unwrap_or_default();
                if !game_start_time.is_empty() {
                    // Parse "2026-03-14 01:00:00+00" → unix timestamp
                    if let Ok(ts) = parse_game_time(&game_start_time) {
                        if ts > cutoff_secs {
                            continue; // too far in the future
                        }
                    }
                }

                // clobTokenIds is a JSON-encoded string
                let token_ids: Vec<String> = match &gm.clob_token_ids {
                    Some(s) => serde_json::from_str(s).unwrap_or_default(),
                    None    => continue,
                };
                if token_ids.len() < 2 {
                    continue;
                }

                // Outcomes: ["AwayTeam", "HomeTeam"] by NBA convention
                let team_names: Vec<String> = match &gm.outcomes {
                    Some(s) => serde_json::from_str(s).unwrap_or_else(|_| {
                        vec!["Away".into(), "Home".into()]
                    }),
                    None => vec!["Away".into(), "Home".into()],
                };

                // Prices: parallel array to token_ids / team_names
                let prices: Vec<f64> = match &gm.outcome_prices {
                    Some(s) => {
                        let raw: Vec<String> = serde_json::from_str(s).unwrap_or_default();
                        raw.iter().map(|p| p.parse().unwrap_or(0.5)).collect()
                    }
                    None => vec![0.5; token_ids.len()],
                };

                // index 0 = away team, index 1 = home team
                let tokens: Vec<Token> = token_ids
                    .into_iter()
                    .enumerate()
                    .map(|(i, id)| Token {
                        token_id:  id,
                        team_name: team_names.get(i).cloned().unwrap_or_else(|| "Unknown".into()),
                        is_home:   i == 1,
                        price:     *prices.get(i).unwrap_or(&0.5),
                    })
                    .collect();

                info!(
                    "Game market: \"{}\"  start={}  away={} ({:.0}%)  home={} ({:.0}%)",
                    gm.question,
                    game_start_time,
                    tokens[0].team_name, tokens[0].price * 100.0,
                    tokens[1].team_name, tokens[1].price * 100.0,
                );

                markets.push(Market {
                    condition_id:    gm.condition_id,
                    question:        gm.question,
                    game_start_time: game_start_time,
                    tokens,
                });
            }
        }

        if page_len < limit {
            break;
        }
        offset += limit;
    }

    info!("Found {} live NBA game moneyline markets", markets.len());
    Ok(markets)
}

/// Parse Polymarket's game start time string to a Unix timestamp.
/// Handles "2026-03-14 01:00:00+00" and "2026-03-14T01:00:00Z".
fn parse_game_time(s: &str) -> Result<i64> {
    // Normalise to RFC-3339 style and lean on chrono
    let normalised = s
        .replace(' ', "T")
        .replace("+00", "+00:00");
    let dt = chrono::DateTime::parse_from_rfc3339(&normalised)
        .context("parse game_start_time")?;
    Ok(dt.timestamp())
}

// ── Alpha Engine: get win probability ────────────────────────────────────────

async fn get_model_prob(http: &Client, game_state: &GameStateRequest) -> Result<f64> {
    let resp: PredictResponse = http
        .post(format!("{ALPHA_ENGINE_URL}/predict"))
        .json(game_state)
        .send()
        .await
        .context("POST /predict failed")?
        .json()
        .await
        .context("deserialise /predict failed")?;

    if !resp.model_loaded {
        warn!("Alpha Engine returned naive prior (model not trained yet)");
    }
    Ok(resp.win_probability)
}

// ── Parse price from a JSON value (string or number) ─────────────────────────

fn parse_price(v: &serde_json::Value) -> Option<f64> {
    match v {
        serde_json::Value::Number(n) => n.as_f64(),
        serde_json::Value::String(s) => s.parse().ok(),
        _ => None,
    }
}

// ── WebSocket ingestion loop ──────────────────────────────────────────────────

async fn run_ws_ingestion(
    markets:  Vec<Market>,
    http:     Client,
    db:       Arc<Mutex<Connection>>,
) -> Result<()> {
    // Build token_id → MarketEntry lookup
    let mut registry: std::collections::HashMap<String, MarketEntry> =
        std::collections::HashMap::new();

    let mut all_token_ids: Vec<String> = Vec::new();

    for m in &markets {
        for t in &m.tokens {
            registry.insert(
                t.token_id.clone(),
                MarketEntry {
                    condition_id:    m.condition_id.clone(),
                    question:        m.question.clone(),
                    token_id:        t.token_id.clone(),
                    team_name:       t.team_name.clone(),
                    is_home:         t.is_home,
                    game_start_time: m.game_start_time.clone(),
                },
            );
            all_token_ids.push(t.token_id.clone());
        }
    }

    if all_token_ids.is_empty() {
        warn!("No NBA markets found — WebSocket will subscribe to an empty list. \
               Check market filters or try again later.");
        return Ok(());
    }

    info!(
        "Connecting to Polymarket WebSocket — subscribing to {} tokens across {} markets",
        all_token_ids.len(),
        markets.len()
    );

    let (mut ws_stream, _) = connect_async(POLYMARKET_WS)
        .await
        .context("WebSocket connect failed")?;

    info!("WebSocket connected");

    // Send subscription
    let sub = WsSubscribe {
        msg_type:   "subscribe",
        channel:    "price_change",
        assets_ids: all_token_ids,
    };
    let sub_json = serde_json::to_string(&sub)?;
    ws_stream.send(Message::Text(sub_json)).await?;
    info!("Subscription sent");

    // ── Main event loop ──
    while let Some(msg) = ws_stream.next().await {
        let msg = match msg {
            Ok(m)  => m,
            Err(e) => { error!("WS recv error: {e}"); break; }
        };

        let text = match msg {
            Message::Text(t)  => t,
            Message::Ping(d)  => {
                // respond to keepalive pings
                if let Err(e) = ws_stream.send(Message::Pong(d)).await {
                    error!("pong failed: {e}");
                }
                continue;
            }
            Message::Close(_) => {
                warn!("WebSocket closed by server");
                break;
            }
            _ => continue,
        };

        // Parse the raw JSON
        let raw: serde_json::Value = match serde_json::from_str(&text) {
            Ok(v)  => v,
            Err(e) => { warn!("JSON parse error: {e} — raw: {text}"); continue; }
        };

        // Normalise into a flat list of ticks
        let ticks: Vec<serde_json::Value> = if raw.is_array() {
            raw.as_array().cloned().unwrap_or_default()
        } else {
            vec![raw]
        };

        for tick_val in ticks {
            // We only care about price_change events
            let event_type = tick_val
                .get("event_type")
                .or_else(|| tick_val.get("type"))
                .and_then(|v| v.as_str())
                .unwrap_or("");

            if !event_type.contains("price") && !event_type.is_empty() {
                continue; // skip order-book / book snapshots
            }

            let asset_id = match tick_val.get("asset_id").and_then(|v| v.as_str()) {
                Some(id) => id.to_string(),
                None     => continue,
            };

            let market_prob = match tick_val.get("price").and_then(parse_price) {
                Some(p) => p,
                None    => continue,
            };

            // Clamp to sane probability range
            if !(0.01..=0.99).contains(&market_prob) {
                continue;
            }

            let entry = match registry.get(&asset_id) {
                Some(e) => e.clone(),
                None    => continue,
            };

            info!(
                "Tick  game=\"{}\"  team={}  is_home={}  market_prob={:.3}  start={}",
                entry.question, entry.team_name, entry.is_home,
                market_prob, entry.game_start_time
            );

            // Only use the HOME team token to get P(home wins) directly.
            // The away token price = 1 - P(home wins), which is redundant.
            if !entry.is_home {
                continue;
            }

            // market_prob is now P(home team wins) from the market.
            // Build a pre-game snapshot for the Alpha Engine.
            // When live PBP is wired up this will carry real in-game state.
            let game_state = GameStateRequest {
                period:            1,
                game_seconds_left: 2880.0,
                home_score:        0.0,
                away_score:        0.0,
                margin:            0.0,
                abs_margin:        0.0,
                total_points:      0.0,
                scoring_pace:      0.0,
            };

            let model_prob = match get_model_prob(&http, &game_state).await {
                Ok(p)  => p,
                Err(e) => {
                    warn!("Alpha Engine error: {e}");
                    continue;
                }
            };

            let edge = model_prob - market_prob;

            if edge.abs() < EDGE_THRESHOLD {
                continue; // no meaningful mispricing
            }

            // Positive edge → market underprices home team → BUY home token
            // Negative edge → market overprices home team → BUY away token
            let action = if edge > 0.0 {
                format!("BUY_{}", entry.team_name.to_uppercase().replace(' ', "_"))
            } else {
                "BUY_AWAY".to_string()
            };

            info!(
                "EDGE FOUND  game=\"{}\"  home={}  edge={edge:+.3}  action={action}",
                entry.question, entry.team_name
            );

            // Log shadow trade
            let db_guard = db.lock().await;
            if let Err(e) = log_trade(
                &db_guard,
                &entry.condition_id,
                &entry.question,
                &action,
                market_prob,
                model_prob,
            ) {
                error!("DB write failed: {e}");
            }
        }
    }

    Ok(())
}

// ── Entry point ───────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    // Logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("execution_engine=info".parse()?)
                .add_directive("info".parse()?),
        )
        .init();

    info!("NBA Execution Engine starting...");

    // ── Database
    let conn = Connection::open(DB_PATH).context("open SQLite")?;
    init_db(&conn)?;
    let db = Arc::new(Mutex::new(conn));

    // ── HTTP client (shared for all calls)
    let http = Client::builder()
        .user_agent("nba-execution-engine/0.1")
        .timeout(std::time::Duration::from_secs(10))
        .build()?;

    // ── Spawn HTTP API server (port 4000) — serves /trades and /health
    {
        let db_http = Arc::clone(&db);
        tokio::spawn(async move {
            if let Err(e) = run_http_server(db_http).await {
                error!("HTTP server error: {e}");
            }
        });
    }

    // ── Spawn trade settlement background task (runs every 5 min)
    {
        let db_settle = Arc::clone(&db);
        let http_settle = http.clone();
        tokio::spawn(async move {
            settle_trades(db_settle, http_settle).await;
        });
    }

    // ── Check Alpha Engine health
    match http
        .get(format!("{ALPHA_ENGINE_URL}/health"))
        .send()
        .await
    {
        Ok(r) if r.status().is_success() => {
            let body: serde_json::Value = r.json().await.unwrap_or_default();
            info!("Alpha Engine healthy: {body}");
        }
        Ok(r) => warn!("Alpha Engine responded with HTTP {}", r.status()),
        Err(e) => warn!("Alpha Engine unreachable ({e}). Trades will use naive prior."),
    }

    // ── Discover NBA markets and run WebSocket ingestion — retry indefinitely
    loop {
        let markets = fetch_nba_markets(&http).await.unwrap_or_else(|e| {
            error!("Market discovery failed: {e}");
            vec![]
        });

        if markets.is_empty() {
            warn!("No active NBA markets on Polymarket right now — retrying in 60 s…");
            tokio::time::sleep(tokio::time::Duration::from_secs(60)).await;
            continue;
        }

        info!("(Re)connecting WebSocket ingestion loop…");
        match run_ws_ingestion(markets, http.clone(), Arc::clone(&db)).await {
            Ok(_)  => warn!("WS loop exited cleanly — reconnecting in 5 s"),
            Err(e) => error!("WS loop error: {e} — reconnecting in 5 s"),
        }
        tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
    }
}
