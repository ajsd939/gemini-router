//! Local OpenAI-compatible server.
//!
//! Binds to 127.0.0.1 only. Requests are authenticated with one of the access
//! tokens from the Connect-tools panel and routed through the same
//! [`crate::router`] used by the desktop UI, so modes and cooldowns are shared.

use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::extract::{DefaultBodyLimit, Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_util::StreamExt;
use serde_json::{Value, json};
use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::ReceiverStream;
use tower_http::cors::{Any, CorsLayer};

use crate::config::Mode;
use crate::logging::status as log_status;
use crate::router::{self, RouteError};
use crate::state::{AppState, lock};

/// Coding tools send large conversations; axum's 2 MiB default is too small.
const MAX_BODY_BYTES: usize = 32 * 1024 * 1024;

/// User-Agent substrings that identify known coding tools.
const KNOWN_TOOLS: [&str; 20] = [
    "opencode",
    "cline",
    "roo-code",
    "roocode",
    "continue",
    "cursor",
    "aider",
    "claude-code",
    "windsurf",
    "zed",
    "copilot",
    "codex",
    "kilo",
    "sweep",
    "open-webui",
    "openwebui",
    "librechat",
    "vscode",
    "jetbrains",
    "neovim",
];

pub struct ServerHandle {
    pub address: String,
    pub port: u16,
    shutdown: Option<oneshot::Sender<()>>,
    task: tokio::task::JoinHandle<()>,
}

impl ServerHandle {
    pub async fn shutdown(mut self) {
        if let Some(tx) = self.shutdown.take() {
            let _ = tx.send(());
        }
        let _ = tokio::time::timeout(Duration::from_secs(3), self.task).await;
    }
}

/// Bind 127.0.0.1:port and start serving.
pub async fn start(state: Arc<AppState>, port: u16) -> Result<ServerHandle, String> {
    let listener =
        tokio::net::TcpListener::bind(std::net::SocketAddr::from(([127, 0, 0, 1], port)))
            .await
            .map_err(|e| format!("could not bind 127.0.0.1:{port}: {e}"))?;
    let bound = listener.local_addr().map(|a| a.port()).unwrap_or(port);
    let address = format!("http://127.0.0.1:{bound}/v1");

    let app = build_router(state);
    let (tx, rx) = oneshot::channel::<()>();
    let task = tokio::spawn(async move {
        let _ = axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                let _ = rx.await;
            })
            .await;
    });

    Ok(ServerHandle {
        address,
        port: bound,
        shutdown: Some(tx),
        task,
    })
}

fn build_router(state: Arc<AppState>) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/models/{id}", get(retrieve_model))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/responses", post(responses))
        .route("/v1/completions", post(completions))
        .route("/v1/messages", post(anthropic_messages))
        .route("/v1/messages/count_tokens", post(anthropic_count_tokens))
        .route("/v1/settings", get(get_settings).post(post_settings))
        .fallback(not_found)
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .layer(cors)
        .with_state(state)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn now_secs() -> i64 {
    chrono::Utc::now().timestamp()
}

fn openai_error(status: StatusCode, message: &str, kind: &str) -> Response {
    (
        status,
        Json(json!({
            "error": {
                "message": message,
                "type": kind,
                "code": status.as_u16(),
                "param": Value::Null
            }
        })),
    )
        .into_response()
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

fn authorize(state: &AppState, headers: &HeaderMap) -> Result<(), Response> {
    let tokens: Vec<String> = lock(&state.settings)
        .access_tokens
        .iter()
        .map(|t| t.token.clone())
        .collect();
    if tokens.is_empty() {
        return Ok(());
    }

    let bearer = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .map(|v| v.trim())
        .and_then(|v| {
            let lower = v.to_ascii_lowercase();
            lower
                .strip_prefix("bearer ")
                .map(|_| v[7..].trim().to_string())
        });
    let api_key = headers
        .get("x-api-key")
        .and_then(|v| v.to_str().ok())
        .map(|v| v.trim().to_string());

    match bearer.or(api_key) {
        Some(provided) => {
            if tokens
                .iter()
                .any(|t| constant_time_eq(t.as_bytes(), provided.as_bytes()))
            {
                Ok(())
            } else {
                Err(openai_error(
                    StatusCode::UNAUTHORIZED,
                    "invalid access token",
                    "invalid_request_error",
                ))
            }
        }
        None => Err(openai_error(
            StatusCode::UNAUTHORIZED,
            "missing access token. Send it as `Authorization: Bearer <token>` or `x-api-key`.",
            "invalid_request_error",
        )),
    }
}

/// Identify the calling tool from headers, falling back to the User-Agent.
fn detect_source(headers: &HeaderMap) -> String {
    for name in ["x-source", "x-client-name", "x-tool", "x-title", "x-app"] {
        if let Some(value) = headers.get(name).and_then(|v| v.to_str().ok()) {
            let value = value.trim();
            if !value.is_empty() {
                return value.to_string();
            }
        }
    }
    if let Some(ua) = headers.get("user-agent").and_then(|v| v.to_str().ok()) {
        let lower = ua.to_ascii_lowercase();
        for tool in KNOWN_TOOLS {
            if lower.contains(tool) {
                return tool.to_string();
            }
        }
        if let Some(first) = ua.split_whitespace().next() {
            let token = first.split('/').next().unwrap_or(first).trim();
            if !token.is_empty() {
                return token.to_string();
            }
        }
    }
    "unknown-client".to_string()
}

/// The mode for this request: header override, then the `model` field if it names
/// a mode, then the globally selected mode.
fn resolve_mode(state: &AppState, body: &Value, headers: &HeaderMap) -> Mode {
    if let Some(mode) = headers
        .get("x-gemini-router-mode")
        .and_then(|v| v.to_str().ok())
        .and_then(Mode::parse)
    {
        return mode;
    }
    if let Some(mode) = body
        .get("model")
        .and_then(|v| v.as_str())
        .and_then(Mode::parse)
    {
        return mode;
    }
    state.mode()
}

fn anthropic_error(status: StatusCode, kind: &str, message: &str) -> Response {
    (
        status,
        Json(json!({
            "type": "error",
            "error": { "type": kind, "message": message }
        })),
    )
        .into_response()
}

fn anthropic_route_error(
    state: &AppState,
    source: &str,
    mode: Mode,
    err: RouteError,
    what: &str,
) -> Response {
    let (status, kind) = match &err {
        RouteError::NoKey | RouteError::KeyRejectedStored | RouteError::KeyRejected(_) => {
            (StatusCode::UNAUTHORIZED, "authentication_error")
        }
        RouteError::NoModelAvailable { .. } => (StatusCode::TOO_MANY_REQUESTS, "rate_limit_error"),
        RouteError::AllFailed { .. } => (StatusCode::BAD_GATEWAY, "api_error"),
        RouteError::Request(_) => (StatusCode::BAD_REQUEST, "invalid_request_error"),
    };
    let message = err.to_string();
    state.log_event(
        source,
        mode,
        None,
        log_status::ERROR,
        None,
        format!("{what} failed: {message}"),
    );
    anthropic_error(status, kind, &message)
}

fn route_error_response(
    state: &AppState,
    source: &str,
    mode: Mode,
    err: RouteError,
    what: &str,
) -> Response {
    let (status, kind) = match &err {
        RouteError::NoKey | RouteError::KeyRejectedStored | RouteError::KeyRejected(_) => {
            (StatusCode::UNAUTHORIZED, "authentication_error")
        }
        RouteError::NoModelAvailable { .. } => (StatusCode::TOO_MANY_REQUESTS, "rate_limit_error"),
        RouteError::AllFailed { .. } => (StatusCode::BAD_GATEWAY, "upstream_error"),
        RouteError::Request(_) => (StatusCode::BAD_REQUEST, "invalid_request_error"),
    };
    let message = err.to_string();
    state.log_event(
        source,
        mode,
        None,
        log_status::ERROR,
        None,
        format!("{what} failed: {message}"),
    );
    openai_error(status, &message, kind)
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

async fn health() -> Response {
    Json(json!({ "status": "ok" })).into_response()
}

async fn not_found() -> Response {
    openai_error(
        StatusCode::NOT_FOUND,
        "unknown endpoint. Supported: /v1/models, /v1/chat/completions, /v1/responses, /v1/completions, /v1/messages, /v1/messages/count_tokens, /v1/settings, /health",
        "invalid_request_error",
    )
}

async fn list_models(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    if let Err(response) = authorize(&state, &headers) {
        return response;
    }
    let mut data: Vec<Value> = Vec::new();
    // Mode aliases can be passed as the `model` field to pick a routing mode.
    for mode in [Mode::Flash, Mode::FlashLite, Mode::Both] {
        data.push(crate::openai::model_object(
            mode.as_str(),
            "gemini-router (routing mode)",
        ));
    }
    for (model, chain) in state.models_with_chain() {
        data.push(crate::openai::model_object(&model, chain));
    }
    Json(json!({ "object": "list", "data": data })).into_response()
}

/// `GET /v1/models/{id}` — some tools probe this before sending a request.
async fn retrieve_model(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Response {
    if let Err(response) = authorize(&state, &headers) {
        return response;
    }
    let is_mode = Mode::parse(&id).is_some();
    let is_model = state
        .models_with_chain()
        .iter()
        .any(|(model, _)| model == &id);
    if !is_mode && !is_model {
        return openai_error(
            StatusCode::NOT_FOUND,
            &format!("model {id} not found"),
            "invalid_request_error",
        );
    }
    let owner = if is_mode {
        "gemini-router (routing mode)"
    } else {
        "gemini-router"
    };
    Json(crate::openai::model_object(&id, owner)).into_response()
}

async fn get_settings(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    if let Err(response) = authorize(&state, &headers) {
        return response;
    }
    let (flash, lite) = state.chains();
    let settings = lock(&state.settings).clone();
    Json(json!({
        "mode": settings.mode.as_str(),
        "port": settings.port,
        "flash_chain": flash,
        "flash_lite_chain": lite,
        "wait_enabled": settings.api_wait_enabled,
        "max_wait_secs": settings.api_max_wait_secs
    }))
    .into_response()
}

/// Change the routing mode from a client. Applies to the next request.
async fn post_settings(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    if let Err(response) = authorize(&state, &headers) {
        return response;
    }
    let source = detect_source(&headers);
    let Some(mode) = body
        .get("mode")
        .and_then(|v| v.as_str())
        .and_then(Mode::parse)
    else {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "expected {\"mode\": \"flash\" | \"flash_lite\" | \"both\"}",
            "invalid_request_error",
        );
    };
    lock(&state.settings).mode = mode;
    if let Err(e) = state.persist_settings() {
        return openai_error(StatusCode::INTERNAL_SERVER_ERROR, &e, "server_error");
    }
    state.log_event(
        &source,
        mode,
        None,
        log_status::OK,
        None,
        format!("mode changed to {} via the local server", mode.as_str()),
    );
    state.emit_status();
    Json(json!({ "mode": mode.as_str() })).into_response()
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    if let Err(response) = authorize(&state, &headers) {
        return response;
    }
    let source = detect_source(&headers);
    let mode = resolve_mode(&state, &body, &headers);
    let stream = body
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let include_usage = body
        .pointer("/stream_options/include_usage")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let gemini_body = match crate::openai::chat_request_to_gemini(&body) {
        Ok(body) => body,
        Err(e) => return openai_error(StatusCode::BAD_REQUEST, &e, "invalid_request_error"),
    };

    if stream {
        return stream_response(
            &state,
            &source,
            mode,
            gemini_body,
            include_usage,
            Shape::Chat,
        )
        .await;
    }

    let allow_wait = lock(&state.settings).api_wait_enabled;
    match router::complete(&state, mode, &source, gemini_body, allow_wait).await {
        Ok(outcome) => Json(crate::openai::gemini_response_to_chat(
            &outcome.value,
            &outcome.model,
            now_secs(),
        ))
        .into_response(),
        Err(e) => route_error_response(&state, &source, mode, e, "chat completion"),
    }
}

async fn completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    if let Err(response) = authorize(&state, &headers) {
        return response;
    }
    let source = detect_source(&headers);
    let mode = resolve_mode(&state, &body, &headers);
    let stream = body
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let include_usage = body
        .pointer("/stream_options/include_usage")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let chat = match crate::openai::completions_request_to_chat(&body) {
        Ok(chat) => chat,
        Err(e) => return openai_error(StatusCode::BAD_REQUEST, &e, "invalid_request_error"),
    };
    let gemini_body = match crate::openai::chat_request_to_gemini(&chat) {
        Ok(body) => body,
        Err(e) => return openai_error(StatusCode::BAD_REQUEST, &e, "invalid_request_error"),
    };

    if stream {
        return stream_response(
            &state,
            &source,
            mode,
            gemini_body,
            include_usage,
            Shape::Completion,
        )
        .await;
    }

    let allow_wait = lock(&state.settings).api_wait_enabled;
    match router::complete(&state, mode, &source, gemini_body, allow_wait).await {
        Ok(outcome) => Json(crate::openai::gemini_response_to_completion(
            &outcome.value,
            &outcome.model,
            now_secs(),
        ))
        .into_response(),
        Err(e) => route_error_response(&state, &source, mode, e, "completion"),
    }
}

/// `POST /v1/responses` — the OpenAI Responses API used by Codex.
async fn responses(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    if let Err(response) = authorize(&state, &headers) {
        return response;
    }
    let source = detect_source(&headers);
    let mode = resolve_mode(&state, &body, &headers);
    let stream = body.get("stream").and_then(Value::as_bool).unwrap_or(false);

    let tools = crate::responses::ToolMap::from_request(&body);
    let gemini_body = match crate::responses::request_to_gemini(&body, &tools) {
        Ok(body) => body,
        Err(e) => return openai_error(StatusCode::BAD_REQUEST, &e, "invalid_request_error"),
    };

    if stream {
        return responses_stream_response(&state, &source, mode, gemini_body, tools).await;
    }

    let allow_wait = lock(&state.settings).api_wait_enabled;
    match router::complete(&state, mode, &source, gemini_body, allow_wait).await {
        Ok(outcome) => {
            let requested = body
                .get("model")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let model = if requested.is_empty() {
                outcome.model.clone()
            } else {
                requested
            };
            Json(crate::responses::response_from_gemini(
                &outcome.value,
                &model,
                now_secs(),
                &tools,
            ))
            .into_response()
        }
        Err(e) => route_error_response(&state, &source, mode, e, "response"),
    }
}

fn responses_sse(event: &str, data: &Value) -> String {
    format!("event: {event}\ndata: {data}\n\n")
}

/// Route a streaming Responses request and relay Gemini SSE as Responses SSE.
async fn responses_stream_response(
    state: &Arc<AppState>,
    source: &str,
    mode: Mode,
    gemini_body: Value,
    tools: crate::responses::ToolMap,
) -> Response {
    let allow_wait = lock(&state.settings).api_wait_enabled;
    let outcome = match router::complete_stream(state, mode, source, gemini_body, allow_wait).await
    {
        Ok(outcome) => outcome,
        Err(e) => return route_error_response(state, source, mode, e, "streaming response"),
    };

    let model = outcome.model.clone();
    let upstream = outcome.value;
    let (tx, rx) = mpsc::channel::<Result<String, std::io::Error>>(32);

    tokio::spawn(async move {
        let mut translator = crate::responses::StreamTranslator::new(&model, tools);
        for (event, data) in translator.start() {
            if tx.send(Ok(responses_sse(&event, &data))).await.is_err() {
                return;
            }
        }

        let mut stream = upstream.bytes_stream();
        let mut buffer = String::new();
        let mut finished = false;
        let mut errored = false;

        while !finished {
            match stream.next().await {
                Some(Ok(bytes)) => {
                    buffer.push_str(&String::from_utf8_lossy(&bytes).replace("\r\n", "\n"));
                }
                Some(Err(e)) => {
                    let data = json!({
                        "type": "response.failed",
                        "response": {
                            "id": format!("resp_{}", crate::openai::short_id()),
                            "object": "response",
                            "status": "failed",
                            "error": {
                                "code": "upstream_error",
                                "message": format!("upstream stream failed: {e}")
                            }
                        }
                    });
                    let _ = tx.send(Ok(responses_sse("response.failed", &data))).await;
                    errored = true;
                    finished = true;
                }
                None => {
                    buffer.push_str("\n\n");
                    finished = true;
                }
            }

            for chunk in take_sse_chunks(&mut buffer) {
                for (event, data) in translator.translate(&chunk) {
                    if tx.send(Ok(responses_sse(&event, &data))).await.is_err() {
                        return;
                    }
                }
            }
        }

        if !errored {
            for (event, data) in translator.finish() {
                if tx.send(Ok(responses_sse(&event, &data))).await.is_err() {
                    return;
                }
            }
        }
    });

    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .header("connection", "keep-alive")
        .header("x-accel-buffering", "no")
        .body(Body::from_stream(ReceiverStream::new(rx)))
        .unwrap_or_else(|_| {
            openai_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "could not build the stream response",
                "server_error",
            )
        })
}

/// `POST /v1/messages` — the Anthropic Messages API used by Claude Code.
async fn anthropic_messages(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    if let Err(response) = authorize(&state, &headers) {
        return response;
    }
    let source = detect_source(&headers);
    let mode = resolve_mode(&state, &body, &headers);
    let stream = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let requested_model = body
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();

    let gemini_body = match crate::anthropic::request_to_gemini(&body) {
        Ok(body) => body,
        Err(e) => return anthropic_error(StatusCode::BAD_REQUEST, "invalid_request_error", &e),
    };

    if stream {
        return anthropic_stream_response(&state, &source, mode, gemini_body, requested_model)
            .await;
    }

    let allow_wait = lock(&state.settings).api_wait_enabled;
    match router::complete(&state, mode, &source, gemini_body, allow_wait).await {
        Ok(outcome) => {
            let model = if requested_model.is_empty() {
                outcome.model.clone()
            } else {
                requested_model
            };
            Json(crate::anthropic::response_to_anthropic(
                &outcome.value,
                &model,
            ))
            .into_response()
        }
        Err(e) => anthropic_route_error(&state, &source, mode, e, "message"),
    }
}

/// `POST /v1/messages/count_tokens` — Claude Code uses this for context sizing.
async fn anthropic_count_tokens(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    if let Err(response) = authorize(&state, &headers) {
        return response;
    }
    Json(json!({ "input_tokens": crate::anthropic::estimate_input_tokens(&body) })).into_response()
}

fn anthropic_sse(event: &str, data: &Value) -> String {
    format!("event: {event}\ndata: {data}\n\n")
}

/// Route an Anthropic streaming request and relay Gemini SSE as Anthropic SSE.
async fn anthropic_stream_response(
    state: &Arc<AppState>,
    source: &str,
    mode: Mode,
    gemini_body: Value,
    requested_model: String,
) -> Response {
    let allow_wait = lock(&state.settings).api_wait_enabled;
    let outcome = match router::complete_stream(state, mode, source, gemini_body, allow_wait).await
    {
        Ok(outcome) => outcome,
        Err(e) => return anthropic_route_error(state, source, mode, e, "streaming message"),
    };

    let model = if requested_model.is_empty() {
        outcome.model.clone()
    } else {
        requested_model
    };
    let upstream = outcome.value;
    let (tx, rx) = mpsc::channel::<Result<String, std::io::Error>>(32);

    tokio::spawn(async move {
        let mut translator = crate::anthropic::StreamTranslator::new(&model);
        for (event, data) in translator.start() {
            if tx.send(Ok(anthropic_sse(&event, &data))).await.is_err() {
                return;
            }
        }

        let mut stream = upstream.bytes_stream();
        let mut buffer = String::new();
        let mut finished = false;
        let mut errored = false;

        while !finished {
            match stream.next().await {
                Some(Ok(bytes)) => {
                    buffer.push_str(&String::from_utf8_lossy(&bytes).replace("\r\n", "\n"));
                }
                Some(Err(e)) => {
                    let data = json!({
                        "type": "error",
                        "error": {
                            "type": "api_error",
                            "message": format!("upstream stream failed: {e}")
                        }
                    });
                    let _ = tx.send(Ok(anthropic_sse("error", &data))).await;
                    errored = true;
                    finished = true;
                }
                None => {
                    buffer.push_str("\n\n");
                    finished = true;
                }
            }

            for chunk in take_sse_chunks(&mut buffer) {
                for (event, data) in translator.translate(&chunk) {
                    if tx.send(Ok(anthropic_sse(&event, &data))).await.is_err() {
                        return;
                    }
                }
                if translator.is_finished() {
                    finished = true;
                }
            }
        }

        if !errored && !translator.is_finished() {
            for (event, data) in translator.finish() {
                if tx.send(Ok(anthropic_sse(&event, &data))).await.is_err() {
                    return;
                }
            }
        }
    });

    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .header("connection", "keep-alive")
        .header("x-accel-buffering", "no")
        .body(Body::from_stream(ReceiverStream::new(rx)))
        .unwrap_or_else(|_| {
            anthropic_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "api_error",
                "could not build the stream response",
            )
        })
}

#[derive(Clone, Copy)]
enum Shape {
    Chat,
    Completion,
}

/// Route a streaming request and relay the upstream SSE as OpenAI SSE.
async fn stream_response(
    state: &Arc<AppState>,
    source: &str,
    mode: Mode,
    gemini_body: Value,
    include_usage: bool,
    shape: Shape,
) -> Response {
    let allow_wait = lock(&state.settings).api_wait_enabled;
    let outcome = match router::complete_stream(state, mode, source, gemini_body, allow_wait).await
    {
        Ok(outcome) => outcome,
        Err(e) => return route_error_response(state, source, mode, e, "streaming request"),
    };

    let model = outcome.model.clone();
    let upstream = outcome.value;
    let id = crate::openai::chunk_id();
    let created = now_secs();
    let (tx, rx) = mpsc::channel::<Result<String, std::io::Error>>(32);

    tokio::spawn(async move {
        let mut stream = upstream.bytes_stream();
        let mut buffer = String::new();
        let mut last_usage: Option<Value> = None;
        let mut translator = crate::openai::StreamTranslator::default();
        let mut finished = false;

        while !finished {
            match stream.next().await {
                Some(Ok(bytes)) => {
                    // Normalise CRLF so a blank line is always "\n\n".
                    buffer.push_str(&String::from_utf8_lossy(&bytes).replace("\r\n", "\n"));
                }
                Some(Err(e)) => {
                    let _ = tx
                        .send(Ok(sse_data(&json!({
                            "error": { "message": format!("upstream stream failed: {e}"), "type": "upstream_error" }
                        }))))
                        .await;
                    finished = true;
                }
                None => {
                    // Flush a final event that arrived without its blank line.
                    buffer.push_str("\n\n");
                    finished = true;
                }
            }

            for chunk in take_sse_chunks(&mut buffer) {
                if chunk.get("usageMetadata").is_some() {
                    last_usage = Some(chunk.clone());
                }
                for out in translate_chunk(shape, &mut translator, &chunk, &id, &model, created) {
                    if tx.send(Ok(sse_data(&out))).await.is_err() {
                        return;
                    }
                }
            }
        }

        if include_usage {
            if let Some(usage) = last_usage {
                let chunk = crate::openai::stream_usage_chunk(&id, &model, created, &usage);
                if tx.send(Ok(sse_data(&chunk))).await.is_err() {
                    return;
                }
            }
        }
        let _ = tx.send(Ok("data: [DONE]\n\n".to_string())).await;
    });

    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .header("connection", "keep-alive")
        .header("x-accel-buffering", "no")
        .body(Body::from_stream(ReceiverStream::new(rx)))
        .unwrap_or_else(|_| {
            openai_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "could not build the stream response",
                "server_error",
            )
        })
}

fn sse_data(value: &Value) -> String {
    format!("data: {value}\n\n")
}

/// Drain every complete SSE event from `buffer` and return its JSON payloads.
fn take_sse_chunks(buffer: &mut String) -> Vec<Value> {
    let mut chunks = Vec::new();
    while let Some(cut) = buffer.find("\n\n") {
        let event: String = buffer.drain(..cut + 2).collect();
        for line in event.lines() {
            let Some(payload) = line.strip_prefix("data:") else {
                continue;
            };
            let payload = payload.trim();
            if payload.is_empty() || payload == "[DONE]" {
                continue;
            }
            if let Ok(chunk) = serde_json::from_str::<Value>(payload) {
                chunks.push(chunk);
            }
        }
    }
    chunks
}

fn translate_chunk(
    shape: Shape,
    translator: &mut crate::openai::StreamTranslator,
    chunk: &Value,
    id: &str,
    model: &str,
    created: i64,
) -> Vec<Value> {
    match shape {
        Shape::Chat => translator.translate(chunk, id, model, created),
        Shape::Completion => {
            // Legacy completions only carry text.
            let mut out = Vec::new();
            let text: String = chunk
                .pointer("/candidates/0/content/parts")
                .and_then(|p| p.as_array())
                .map(|parts| {
                    parts
                        .iter()
                        .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
                        .collect()
                })
                .unwrap_or_default();
            if !text.is_empty() {
                out.push(json!({
                    "id": id, "object": "text_completion", "created": created, "model": model,
                    "choices": [{ "text": text, "index": 0, "logprobs": Value::Null, "finish_reason": Value::Null }]
                }));
            }
            if let Some(reason) = chunk
                .pointer("/candidates/0/finishReason")
                .and_then(|v| v.as_str())
            {
                let mapped = if reason == "MAX_TOKENS" {
                    "length"
                } else {
                    "stop"
                };
                out.push(json!({
                    "id": id, "object": "text_completion", "created": created, "model": model,
                    "choices": [{ "text": "", "index": 0, "logprobs": Value::Null, "finish_reason": mapped }]
                }));
            }
            out
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;

    fn headers(pairs: &[(&str, &str)]) -> HeaderMap {
        let mut map = HeaderMap::new();
        for (k, v) in pairs {
            map.insert(
                axum::http::HeaderName::from_bytes(k.as_bytes()).unwrap(),
                HeaderValue::from_str(v).unwrap(),
            );
        }
        map
    }

    #[test]
    fn detects_known_tools_from_the_user_agent() {
        assert_eq!(
            detect_source(&headers(&[("user-agent", "opencode/1.2.3")])),
            "opencode"
        );
        assert_eq!(
            detect_source(&headers(&[("user-agent", "Cline/3.0.0 (VSCode)")])),
            "cline"
        );
    }

    #[test]
    fn explicit_source_header_wins_over_user_agent() {
        assert_eq!(
            detect_source(&headers(&[
                ("user-agent", "opencode/1.0"),
                ("x-source", "my-custom-tool")
            ])),
            "my-custom-tool"
        );
    }

    #[test]
    fn falls_back_to_the_product_token_then_unknown() {
        assert_eq!(
            detect_source(&headers(&[("user-agent", "python-requests/2.31.0")])),
            "python-requests"
        );
        assert_eq!(detect_source(&headers(&[])), "unknown-client");
    }

    #[test]
    fn constant_time_compare_behaves_like_equality() {
        assert!(constant_time_eq(b"abc", b"abc"));
        assert!(!constant_time_eq(b"abc", b"abd"));
        assert!(!constant_time_eq(b"abc", b"abcd"));
        assert!(constant_time_eq(b"", b""));
    }

    #[test]
    fn sse_parser_handles_partial_events() {
        let mut buffer = String::from("data: {\"a\":1}\n\ndata: {\"b\":2}\n");
        let first = take_sse_chunks(&mut buffer);
        assert_eq!(first.len(), 1);
        assert_eq!(first[0]["a"], 1);
        // The second event has no blank line yet, so it is still buffered.
        assert!(take_sse_chunks(&mut buffer).is_empty());

        buffer.push_str("\n");
        let rest = take_sse_chunks(&mut buffer);
        assert_eq!(rest.len(), 1);
        assert_eq!(rest[0]["b"], 2);
    }

    #[test]
    fn sse_data_is_a_well_formed_event() {
        let event = sse_data(&json!({"a":1}));
        assert!(event.starts_with("data: "));
        assert!(event.ends_with("\n\n"));
    }

    #[test]
    fn legacy_stream_shape_emits_text_completion_chunks() {
        let chunk =
            json!({"candidates":[{"content":{"parts":[{"text":"hi"}]},"finishReason":"STOP"}]});
        let mut translator = crate::openai::StreamTranslator::default();
        let out = translate_chunk(Shape::Completion, &mut translator, &chunk, "id", "m", 1);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0]["object"], "text_completion");
        assert_eq!(out[0]["choices"][0]["text"], "hi");
        assert_eq!(out[1]["choices"][0]["finish_reason"], "stop");
    }
}
