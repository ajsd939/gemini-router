//! Shared, thread-safe application state.
//!
//! Per-model availability is tracked per key slot, because two API keys can come
//! from two different Google Cloud projects and therefore have independent
//! quotas. The UI shows the state for one selected slot at a time.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard};
use std::time::Duration;

use chrono::{DateTime, Local, Utc};
use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager};

use crate::config::{self, Config, Mode, Settings};
use crate::keystore::{KeyStore, StorageKind};
use crate::logging::{LogEntry, MAX_LOG_ENTRIES, status as log_status};
use crate::server::ServerHandle;

/// Lock helper: a poisoned mutex should not take the whole app down.
pub fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

// ---------------------------------------------------------------------------
// Model availability registry
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
struct ModelState {
    cooling_until: Option<DateTime<Utc>>,
    exhausted_until: Option<DateTime<Utc>>,
    /// Why the model is blocked, shown next to the deadline in the UI.
    reason: Option<String>,
}

#[derive(Default)]
pub struct Registry {
    /// (key slot, model) -> state
    map: HashMap<(u32, String), ModelState>,
}

/// Snapshot row for the UI model-status list.
#[derive(Debug, Clone, Serialize)]
pub struct ModelStateView {
    pub model: String,
    /// `flash` or `flash_lite`.
    pub chain: &'static str,
    /// `available` | `cooling` | `exhausted`.
    pub status: &'static str,
    pub detail: String,
    pub cooling_until: Option<String>,
    pub exhausted_until: Option<String>,
}

impl Registry {
    fn entry(&mut self, slot: u32, model: &str) -> &mut ModelState {
        self.map.entry((slot, model.to_string())).or_default()
    }

    pub fn is_available(&self, slot: u32, model: &str, now: DateTime<Utc>) -> bool {
        self.blocked_until(slot, model, now).is_none()
    }

    /// The instant this model becomes available again, if it is currently blocked.
    pub fn blocked_until(
        &self,
        slot: u32,
        model: &str,
        now: DateTime<Utc>,
    ) -> Option<DateTime<Utc>> {
        let state = self.map.get(&(slot, model.to_string()))?;
        let cooling = state.cooling_until.filter(|t| *t > now);
        let exhausted = state.exhausted_until.filter(|t| *t > now);
        match (cooling, exhausted) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        }
    }

    /// Earliest instant any of `models` becomes available for `slot`.
    pub fn earliest_available(
        &self,
        slot: u32,
        models: &[String],
        now: DateTime<Utc>,
    ) -> Option<DateTime<Utc>> {
        models
            .iter()
            .filter_map(|m| self.blocked_until(slot, m, now))
            .min()
    }

    /// Earliest availability across every slot and model given.
    pub fn earliest_available_any(
        &self,
        slots: &[u32],
        models: &[String],
        now: DateTime<Utc>,
    ) -> Option<DateTime<Utc>> {
        slots
            .iter()
            .filter_map(|slot| self.earliest_available(*slot, models, now))
            .min()
    }

    pub fn mark_cooling(&mut self, slot: u32, model: &str, until: DateTime<Utc>, reason: String) {
        let entry = self.entry(slot, model);
        entry.cooling_until = Some(until);
        entry.reason = Some(reason);
    }

    pub fn mark_exhausted(&mut self, slot: u32, model: &str, until: DateTime<Utc>, reason: String) {
        let entry = self.entry(slot, model);
        entry.exhausted_until = Some(until);
        entry.reason = Some(reason);
    }

    /// A successful call clears any block on this model.
    pub fn clear(&mut self, slot: u32, model: &str) {
        let entry = self.entry(slot, model);
        entry.cooling_until = None;
        entry.exhausted_until = None;
        entry.reason = None;
    }

    /// Drop blocks whose deadline has passed so the UI shows them as available.
    pub fn sweep(&mut self, now: DateTime<Utc>) {
        for state in self.map.values_mut() {
            let cooling_expired = state.cooling_until.map(|t| t <= now).unwrap_or(false);
            let exhausted_expired = state.exhausted_until.map(|t| t <= now).unwrap_or(false);
            if cooling_expired {
                state.cooling_until = None;
            }
            if exhausted_expired {
                state.exhausted_until = None;
            }
            // Keep the explanation while the other block is still active.
            if cooling_expired && exhausted_expired {
                state.reason = None;
            }
        }
    }

    /// Build the UI rows for one key slot.
    pub fn snapshot(
        &mut self,
        slot: u32,
        models: &[(String, &'static str)],
        now: DateTime<Utc>,
    ) -> Vec<ModelStateView> {
        self.sweep(now);
        models
            .iter()
            .map(|(model, chain)| {
                let state = self.map.get(&(slot, model.clone()));
                let cooling = state.and_then(|s| s.cooling_until).filter(|t| *t > now);
                let exhausted = state.and_then(|s| s.exhausted_until).filter(|t| *t > now);
                let reason = state.and_then(|s| s.reason.clone());

                let (status, detail) = match (cooling, exhausted) {
                    (_, Some(until)) => (
                        "exhausted",
                        format!(
                            "Daily-exhausted until {}",
                            until.with_timezone(&Local).format("%Y-%m-%d %H:%M:%S")
                        ),
                    ),
                    (Some(until), None) => (
                        "cooling",
                        format!(
                            "Cooling down until {}",
                            until.with_timezone(&Local).format("%H:%M:%S")
                        ),
                    ),
                    (None, None) => ("available", "Available".to_string()),
                };

                let detail = match reason {
                    Some(r) if status != "available" => format!("{detail} · {r}"),
                    _ => detail,
                };

                ModelStateView {
                    model: model.clone(),
                    chain,
                    status,
                    detail,
                    cooling_until: cooling
                        .map(|t| t.with_timezone(&Local).format("%H:%M:%S").to_string()),
                    exhausted_until: exhausted.map(|t| {
                        t.with_timezone(&Local)
                            .format("%Y-%m-%d %H:%M:%S")
                            .to_string()
                    }),
                }
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Serialisable views for the frontend
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct SlotView {
    pub index: u32,
    pub label: String,
    pub present: bool,
    /// Masked preview such as `AIza…1234`, never the stored value.
    pub preview: String,
    pub storage: StorageKind,
    pub source: String,
    pub status: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct TokenView {
    pub id: String,
    pub label: String,
    pub token: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SettingsView {
    pub mode: Mode,
    pub port: u16,
    pub key_slots: u32,
    pub access_tokens: Vec<TokenView>,
    pub api_wait_enabled: bool,
    pub api_max_wait_secs: u64,
    pub ui_max_wait_secs: u64,
    pub request_timeout_secs: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ServerStatus {
    pub running: bool,
    pub address: String,
    pub port: u16,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Snapshot {
    pub config: Config,
    /// Absolute path of config.json, shown in the UI so hand edits are easy to find.
    pub config_path: String,
    pub settings: SettingsView,
    pub key_slots: Vec<SlotView>,
    pub keychain_ok: bool,
    pub keychain_error: Option<String>,
    pub models: Vec<ModelStateView>,
    /// Key slot the model list was generated for.
    pub status_slot: u32,
    pub server: ServerStatus,
    pub log: Vec<LogEntry>,
    pub env_key_present: bool,
}

// ---------------------------------------------------------------------------
// App state
// ---------------------------------------------------------------------------

pub struct AppState {
    pub app: AppHandle,
    pub data_dir: PathBuf,
    pub http: reqwest::Client,
    pub config: Mutex<Config>,
    pub settings: Mutex<Settings>,
    pub registry: Mutex<Registry>,
    pub keys: Mutex<KeyStore>,
    pub log: Mutex<Vec<LogEntry>>,
    log_seq: AtomicU64,
    pub server: Mutex<Option<ServerHandle>>,
    /// Last bind error, surfaced in the UI.
    pub server_error: Mutex<Option<String>>,
}

impl AppState {
    pub fn new(app: AppHandle) -> Result<Self, String> {
        let data_dir = app
            .path()
            .app_data_dir()
            .map_err(|e| format!("could not resolve the app data directory: {e}"))?;
        std::fs::create_dir_all(&data_dir)
            .map_err(|e| format!("could not create {}: {e}", data_dir.display()))?;

        let config = config::load_config(&data_dir);
        let mut settings = config::load_settings(&data_dir);
        // Keep the port coherent across the two files.
        if settings.port == 0 {
            settings.port = config.server_port;
        }

        let keys = KeyStore::new(&data_dir, settings.key_slots);
        let timeout = settings.request_timeout_secs.clamp(5, 3600);
        // Use a per-read timeout rather than a whole-request timeout: streaming
        // responses can legitimately stay open for many minutes, but a stalled
        // connection must still fail eventually.
        let http = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(15))
            .read_timeout(Duration::from_secs(timeout))
            .user_agent(concat!("gemini-router/", env!("CARGO_PKG_VERSION")))
            .build()
            .map_err(|e| format!("could not build the HTTP client: {e}"))?;

        let state = AppState {
            app,
            data_dir,
            http,
            config: Mutex::new(config),
            settings: Mutex::new(settings),
            registry: Mutex::new(Registry::default()),
            keys: Mutex::new(keys),
            log: Mutex::new(Vec::new()),
            log_seq: AtomicU64::new(1),
            server: Mutex::new(None),
            server_error: Mutex::new(None),
        };
        state.log_event(
            "app",
            state.mode(),
            None,
            log_status::OK,
            None,
            "router started",
        );
        Ok(state)
    }

    // -- accessors ---------------------------------------------------------

    pub fn mode(&self) -> Mode {
        lock(&self.settings).mode
    }

    pub fn chains(&self) -> (Vec<String>, Vec<String>) {
        let cfg = lock(&self.config);
        (cfg.flash_chain.clone(), cfg.flash_lite_chain.clone())
    }

    pub fn models_with_chain(&self) -> Vec<(String, &'static str)> {
        let cfg = lock(&self.config);
        let mut out: Vec<(String, &'static str)> = Vec::new();
        for m in &cfg.flash_chain {
            out.push((m.clone(), "flash"));
        }
        for m in &cfg.flash_lite_chain {
            if !out.iter().any(|(existing, _)| existing == m) {
                out.push((m.clone(), "flash_lite"));
            }
        }
        out
    }

    /// The chain sequence for a mode, in routing order.
    ///
    /// Duplicates are removed: a model that appears in both chains would
    /// otherwise be attempted twice in `both` mode.
    pub fn chain_for(&self, mode: Mode) -> Vec<String> {
        let (flash, lite) = self.chains();
        let mut chain: Vec<String> = match mode {
            Mode::Flash => flash,
            Mode::FlashLite => lite,
            Mode::Both => flash.into_iter().chain(lite).collect(),
        };
        let mut seen = std::collections::HashSet::new();
        chain.retain(|model| seen.insert(model.clone()));
        chain
    }

    pub fn slot_count(&self) -> u32 {
        lock(&self.settings).key_slots.max(1)
    }

    pub fn key_for_slot(&self, slot: u32) -> Option<String> {
        lock(&self.keys).get(slot).map(|s| s.to_string())
    }

    pub fn env_key_present(&self) -> bool {
        std::env::var(crate::keystore::ENV_FALLBACK_KEY)
            .map(|v| !v.trim().is_empty())
            .unwrap_or(false)
    }

    // -- logging / events --------------------------------------------------

    pub fn log_event(
        &self,
        source: &str,
        mode: Mode,
        model: Option<&str>,
        status: &str,
        latency_ms: Option<u64>,
        detail: impl Into<String>,
    ) {
        let id = self.log_seq.fetch_add(1, Ordering::Relaxed);
        let entry = LogEntry::new(
            id,
            source,
            mode.as_str(),
            model.map(|m| m.to_string()),
            status,
            latency_ms,
            detail,
        );
        {
            let mut log = lock(&self.log);
            log.push(entry.clone());
            if log.len() > MAX_LOG_ENTRIES {
                let excess = log.len() - MAX_LOG_ENTRIES;
                log.drain(0..excess);
            }
        }
        let _ = self.app.emit("router:log", &entry);
    }

    pub fn emit_status(&self) {
        let payload = self.snapshot(None);
        let _ = self.app.emit("router:status", &payload);
    }

    /// Build the full UI snapshot. `status_slot` defaults to the first slot holding a key.
    pub fn snapshot(&self, status_slot: Option<u32>) -> Snapshot {
        let now = Utc::now();
        let config = lock(&self.config).clone();
        let settings = lock(&self.settings).clone();

        let (key_slots, keychain, usable) = {
            let keys = lock(&self.keys);
            let mut views: Vec<SlotView> = keys
                .slots
                .iter()
                .map(|s| SlotView {
                    index: s.index,
                    label: format!("Key {}", s.index + 1),
                    present: s.present(),
                    preview: s
                        .value
                        .as_deref()
                        .map(crate::keystore::preview)
                        .unwrap_or_default(),
                    storage: s.storage,
                    source: s.source.clone(),
                    status: s.status.label().to_string(),
                })
                .collect();
            views.sort_by_key(|v| v.index);
            (views, keys.keychain.clone(), keys.usable_slots())
        };

        let slot = status_slot.or_else(|| usable.first().copied()).unwrap_or(0);

        let models = {
            let mut registry = lock(&self.registry);
            let list = {
                let cfg = lock(&self.config);
                let mut out: Vec<(String, &'static str)> = Vec::new();
                for m in &cfg.flash_chain {
                    out.push((m.clone(), "flash"));
                }
                for m in &cfg.flash_lite_chain {
                    if !out.iter().any(|(existing, _)| existing == m) {
                        out.push((m.clone(), "flash_lite"));
                    }
                }
                out
            };
            registry.snapshot(slot, &list, now)
        };

        let (running, address, error) = {
            let server = lock(&self.server);
            match server.as_ref() {
                Some(handle) => (true, handle.address.clone(), None),
                None => (
                    false,
                    format!("http://127.0.0.1:{}/v1", settings.port),
                    lock(&self.server_error).clone(),
                ),
            }
        };

        Snapshot {
            config_path: config::config_path(&self.data_dir).display().to_string(),
            config,
            key_slots,
            keychain_ok: keychain.is_ok(),
            keychain_error: keychain.err(),
            settings: SettingsView {
                mode: settings.mode,
                port: settings.port,
                key_slots: settings.key_slots,
                access_tokens: settings
                    .access_tokens
                    .iter()
                    .map(|t| TokenView {
                        id: t.id.clone(),
                        label: t.label.clone(),
                        token: t.token.clone(),
                    })
                    .collect(),
                api_wait_enabled: settings.api_wait_enabled,
                api_max_wait_secs: settings.api_max_wait_secs,
                ui_max_wait_secs: settings.ui_max_wait_secs,
                request_timeout_secs: settings.request_timeout_secs,
            },
            models,
            status_slot: slot,
            server: ServerStatus {
                running,
                address,
                port: settings.port,
                error,
            },
            log: lock(&self.log).clone(),
            env_key_present: self.env_key_present(),
        }
    }

    /// Persist settings to disk.
    pub fn persist_settings(&self) -> Result<(), String> {
        let settings = lock(&self.settings).clone();
        config::save_settings(&self.data_dir, &settings)
    }
}

/// Convenience: `snapshot` for tests and non-Tauri contexts cannot exist without
/// an `AppHandle`, so the pure-Rust tests live in the other modules.
#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration as ChronoDuration;

    fn models() -> Vec<String> {
        vec!["a".to_string(), "b".to_string()]
    }

    #[test]
    fn unknown_model_is_available() {
        let reg = Registry::default();
        assert!(reg.is_available(0, "never-seen", Utc::now()));
    }

    #[test]
    fn cooling_blocks_only_until_the_deadline() {
        let mut reg = Registry::default();
        let now = Utc::now();
        reg.mark_cooling(0, "a", now + ChronoDuration::seconds(30), "rpm".into());
        assert!(!reg.is_available(0, "a", now));
        assert!(reg.is_available(0, "a", now + ChronoDuration::seconds(31)));
        // A different slot is unaffected.
        assert!(reg.is_available(1, "a", now));
    }

    #[test]
    fn daily_exhaustion_wins_over_a_shorter_cooldown() {
        let mut reg = Registry::default();
        let now = Utc::now();
        reg.mark_cooling(0, "a", now + ChronoDuration::seconds(5), "rpm".into());
        reg.mark_exhausted(0, "a", now + ChronoDuration::hours(4), "rpd".into());
        assert_eq!(
            reg.blocked_until(0, "a", now),
            Some(now + ChronoDuration::hours(4))
        );
    }

    #[test]
    fn clear_and_sweep_release_models() {
        let mut reg = Registry::default();
        let now = Utc::now();
        reg.mark_cooling(0, "a", now + ChronoDuration::seconds(5), "rpm".into());
        reg.sweep(now + ChronoDuration::seconds(6));
        assert!(reg.is_available(0, "a", now + ChronoDuration::seconds(6)));

        reg.mark_exhausted(0, "b", now + ChronoDuration::hours(1), "rpd".into());
        reg.clear(0, "b");
        assert!(reg.is_available(0, "b", now));
    }

    #[test]
    fn earliest_available_picks_the_minimum() {
        let mut reg = Registry::default();
        let now = Utc::now();
        reg.mark_cooling(0, "a", now + ChronoDuration::seconds(90), "rpm".into());
        reg.mark_cooling(0, "b", now + ChronoDuration::seconds(10), "rpm".into());
        let earliest = reg.earliest_available(0, &models(), now).unwrap();
        assert_eq!(earliest, now + ChronoDuration::seconds(10));
    }

    #[test]
    fn sweep_keeps_the_reason_while_a_second_block_is_active() {
        let mut reg = Registry::default();
        let now = Utc::now();
        reg.mark_cooling(0, "a", now + ChronoDuration::seconds(5), "rpm".into());
        reg.mark_exhausted(0, "a", now + ChronoDuration::hours(4), "rpd".into());
        let after_cooling = now + ChronoDuration::seconds(6);
        reg.sweep(after_cooling);
        let state = reg.map.get(&(0, "a".to_string())).unwrap();
        assert_eq!(state.cooling_until, None);
        assert!(state.exhausted_until.is_some());
        assert_eq!(state.reason.as_deref(), Some("rpd"));
    }

    #[test]
    fn registry_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Registry>();
        assert_send_sync::<std::sync::Mutex<Registry>>();
    }
}
