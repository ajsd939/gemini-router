//! Request log entries shared between the router, the local server and the UI.

use serde::{Deserialize, Serialize};

/// The log is a ring buffer; oldest entries are dropped past this length.
pub const MAX_LOG_ENTRIES: usize = 500;

/// Status values used in `LogEntry::status`. Kept as plain strings so the
/// frontend can style them without a shared enum.
pub mod status {
    /// Request completed successfully.
    pub const OK: &str = "ok";
    /// A model was skipped or failed and we moved on (429, unavailable, 5xx).
    pub const FALLBACK: &str = "fallback";
    /// "both" mode restarted from the top of flash_chain.
    pub const RESTART: &str = "restart";
    /// The router blocked waiting for a model to become available again.
    pub const WAIT: &str = "wait";
    /// The request failed and was returned to the caller.
    pub const ERROR: &str = "error";
    /// A transient error was retried against the same model.
    pub const RETRY: &str = "retry";
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    pub id: u64,
    /// Local wall-clock time, `%Y-%m-%d %H:%M:%S`.
    pub timestamp: String,
    /// Identified tool name, `app` for the desktop UI, or `unknown-client`.
    pub source: String,
    pub mode: String,
    pub model: Option<String>,
    pub status: String,
    pub latency_ms: Option<u64>,
    /// Human-readable explanation, including fallback and restart events.
    pub detail: String,
}

impl LogEntry {
    pub fn new(
        id: u64,
        source: impl Into<String>,
        mode: impl Into<String>,
        model: Option<String>,
        status: impl Into<String>,
        latency_ms: Option<u64>,
        detail: impl Into<String>,
    ) -> Self {
        LogEntry {
            id,
            timestamp: chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
            source: source.into(),
            mode: mode.into(),
            model,
            status: status.into(),
            latency_ms,
            detail: detail.into(),
        }
    }
}
