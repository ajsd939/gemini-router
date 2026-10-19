//! Tauri commands — the entire API surface exposed to the web frontend.

use std::sync::Arc;

use serde::Serialize;
use tauri::State;

use crate::config::{self, Config, Mode};
use crate::gemini::{self, KeyValidation};
use crate::inject;
use crate::keystore::{self, KeyStatus};
use crate::logging::status as log_status;
use crate::router;
use crate::server;
use crate::state::{AppState, Snapshot, lock};

type CmdResult<T> = Result<T, String>;

/// Upper bound on key slots so the UI cannot grow without limit.
const MAX_KEY_SLOTS: u32 = 16;

#[derive(Serialize)]
pub struct SaveKeyResult {
    pub snapshot: Snapshot,
    pub message: String,
    /// Non-empty when the key had to go to a plaintext file.
    pub warning: String,
    /// True only when the key was actually stored, so the UI can keep the
    /// pasted value on a failed save instead of clearing it.
    pub saved: bool,
}

#[derive(Serialize)]
pub struct PromptResult {
    pub model: String,
    pub text: String,
    pub latency_ms: u64,
    pub key_slot: u32,
    pub restarts: u32,
    pub fallbacks: u32,
}

// ---------------------------------------------------------------------------
// Snapshot / mode / config
// ---------------------------------------------------------------------------

#[tauri::command]
pub fn get_snapshot(state: State<'_, Arc<AppState>>, status_slot: Option<u32>) -> Snapshot {
    state.snapshot(status_slot)
}

#[tauri::command]
pub fn set_mode(state: State<'_, Arc<AppState>>, mode: String) -> CmdResult<Snapshot> {
    let st = state.inner();
    let mode = Mode::parse(&mode).ok_or_else(|| format!("unknown mode: {mode}"))?;
    {
        let mut settings = lock(&st.settings);
        if settings.mode == mode {
            return Ok(st.snapshot(None));
        }
        settings.mode = mode;
    }
    st.persist_settings()?;
    st.log_event(
        "app",
        mode,
        None,
        log_status::OK,
        None,
        format!("mode changed to {}", mode.as_str()),
    );
    st.emit_status();
    Ok(st.snapshot(None))
}

#[tauri::command]
pub fn set_config_json(state: State<'_, Arc<AppState>>, json_text: String) -> CmdResult<Snapshot> {
    let st = state.inner();
    let parsed: Config =
        serde_json::from_str(&json_text).map_err(|e| format!("config is not valid JSON: {e}"))?;
    let config = parsed.sanitised()?;
    config::save_config(&st.data_dir, &config)?;
    *lock(&st.config) = config.clone();
    st.log_event(
        "app",
        st.mode(),
        None,
        log_status::OK,
        None,
        format!(
            "chains updated: flash=[{}], flash_lite=[{}]",
            config.flash_chain.join(", "),
            config.flash_lite_chain.join(", ")
        ),
    );
    st.emit_status();
    Ok(st.snapshot(None))
}

/// Re-read `config.json` from disk, picking up hand edits without a restart.
#[tauri::command]
pub fn reload_config(state: State<'_, Arc<AppState>>) -> CmdResult<Snapshot> {
    let st = state.inner();
    let config = config::load_config(&st.data_dir);
    *lock(&st.config) = config.clone();
    st.log_event(
        "app",
        st.mode(),
        None,
        log_status::OK,
        None,
        format!(
            "reloaded config.json: flash=[{}], flash_lite=[{}]",
            config.flash_chain.join(", "),
            config.flash_lite_chain.join(", ")
        ),
    );
    st.emit_status();
    Ok(st.snapshot(None))
}

// ---------------------------------------------------------------------------
// Key area
// ---------------------------------------------------------------------------

#[tauri::command]
pub fn add_key_slot(state: State<'_, Arc<AppState>>) -> CmdResult<Snapshot> {
    let st = state.inner();
    let count = {
        let mut settings = lock(&st.settings);
        if settings.key_slots >= MAX_KEY_SLOTS {
            return Err(format!("key slots are capped at {MAX_KEY_SLOTS}"));
        }
        settings.key_slots += 1;
        settings.key_slots
    };
    lock(&st.keys).ensure_slots(&st.data_dir, count);
    st.persist_settings()?;
    st.log_event(
        "app",
        st.mode(),
        None,
        log_status::OK,
        None,
        format!("added a key slot ({count} total)"),
    );
    st.emit_status();
    Ok(st.snapshot(None))
}

#[tauri::command]
pub fn save_key(
    state: State<'_, Arc<AppState>>,
    slot: u32,
    value: String,
) -> CmdResult<SaveKeyResult> {
    let st = state.inner();
    if slot >= MAX_KEY_SLOTS {
        return Err(format!("slot index must be below {MAX_KEY_SLOTS}"));
    }

    // Keep the slot count in step if the UI saved into a freshly added slot.
    // (Deliberately scoped: the settings lock must not be held while
    // persist_settings re-locks it.)
    let needs_grow = slot + 1 > lock(&st.settings).key_slots;
    if needs_grow {
        lock(&st.settings).key_slots = slot + 1;
        let _ = st.persist_settings();
    }

    let outcome = keystore::save_key(&st.data_dir, slot, &value);
    {
        let count = st.slot_count();
        let mut keys = lock(&st.keys);
        keys.ensure_slots(&st.data_dir, count);
        keys.reload(&st.data_dir, count);
        keys.set_status(slot, outcome.status);
    }

    // Never log the key itself, only where it went.
    let detail = match outcome.storage {
        keystore::StorageKind::Keychain => {
            format!("key slot {} saved to the OS keychain", slot + 1)
        }
        keystore::StorageKind::File => format!(
            "key slot {} saved to a restricted file (keychain unavailable)",
            slot + 1
        ),
        keystore::StorageKind::None => format!("key slot {} not saved", slot + 1),
    };
    st.log_event(
        "app",
        st.mode(),
        None,
        if outcome.status == KeyStatus::Saved {
            log_status::OK
        } else {
            log_status::ERROR
        },
        None,
        detail,
    );
    st.emit_status();

    Ok(SaveKeyResult {
        snapshot: st.snapshot(Some(slot)),
        message: outcome.message,
        warning: outcome.warning,
        saved: outcome.status == KeyStatus::Saved,
    })
}

#[tauri::command]
pub fn clear_key(state: State<'_, Arc<AppState>>, slot: u32) -> CmdResult<Snapshot> {
    let st = state.inner();
    keystore::delete_key(&st.data_dir, slot);
    let count = st.slot_count();
    lock(&st.keys).reload(&st.data_dir, count);
    st.log_event(
        "app",
        st.mode(),
        None,
        log_status::OK,
        None,
        format!("cleared key slot {}", slot + 1),
    );
    st.emit_status();
    Ok(st.snapshot(Some(slot)))
}

/// Validate a key against the API without consuming model quota.
#[tauri::command]
pub async fn test_key(state: State<'_, Arc<AppState>>, slot: u32) -> CmdResult<Snapshot> {
    let st = state.inner().clone();
    let key = st.key_for_slot(slot).ok_or_else(|| "No key".to_string())?;

    let (status, detail) = match gemini::validate_key(&st.http, &key).await {
        KeyValidation::Valid => (KeyStatus::Saved, "key accepted by the API".to_string()),
        KeyValidation::Rejected(why) => (
            KeyStatus::Rejected,
            format!("key rejected by the API: {why}"),
        ),
        // A network problem says nothing about the key, so leave the status alone.
        KeyValidation::Unreachable(why) => (
            st.keys
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .slot(slot)
                .map(|s| s.status)
                .unwrap_or(KeyStatus::NoKey),
            format!("could not verify the key: {why}"),
        ),
    };

    lock(&st.keys).set_status(slot, status);
    st.log_event(
        "app",
        st.mode(),
        None,
        if status == KeyStatus::Rejected {
            log_status::ERROR
        } else {
            log_status::OK
        },
        None,
        format!("key slot {}: {detail}", slot + 1),
    );
    st.emit_status();
    Ok(st.snapshot(Some(slot)))
}

// ---------------------------------------------------------------------------
// Prompt
// ---------------------------------------------------------------------------

#[tauri::command]
pub async fn send_prompt(
    state: State<'_, Arc<AppState>>,
    prompt: String,
    mode: Option<String>,
) -> CmdResult<PromptResult> {
    let st = state.inner().clone();
    if prompt.trim().is_empty() {
        return Err("the prompt is empty".to_string());
    }
    let mode = mode
        .as_deref()
        .and_then(Mode::parse)
        .unwrap_or_else(|| st.mode());

    let outcome = router::complete(&st, mode, "app", router::prompt_body(&prompt), true)
        .await
        .map_err(|e| e.to_string())?;

    Ok(PromptResult {
        text: router::extract_text(&outcome.value),
        model: outcome.model,
        latency_ms: outcome.latency_ms,
        key_slot: outcome.key_slot,
        restarts: outcome.restarts,
        fallbacks: outcome.fallbacks,
    })
}

// ---------------------------------------------------------------------------
// Connect tools
// ---------------------------------------------------------------------------

#[tauri::command]
pub async fn start_server(
    state: State<'_, Arc<AppState>>,
    port: Option<u16>,
) -> CmdResult<Snapshot> {
    let st = state.inner().clone();
    if lock(&st.server).is_some() {
        return Err("the server is already running".to_string());
    }
    let requested = port.unwrap_or_else(|| lock(&st.settings).port);

    match server::start(st.clone(), requested).await {
        Ok(handle) => {
            let bound = handle.port;
            let address = handle.address.clone();
            *lock(&st.server) = Some(handle);
            *lock(&st.server_error) = None;
            {
                let mut settings = lock(&st.settings);
                settings.port = bound;
            }
            let _ = st.persist_settings();
            st.log_event(
                "app",
                st.mode(),
                None,
                log_status::OK,
                None,
                format!("local OpenAI-compatible server started at {address}"),
            );
            st.emit_status();
            Ok(st.snapshot(None))
        }
        Err(e) => {
            *lock(&st.server_error) = Some(e.clone());
            st.log_event(
                "app",
                st.mode(),
                None,
                log_status::ERROR,
                None,
                format!("server failed to start: {e}"),
            );
            st.emit_status();
            Err(e)
        }
    }
}

#[tauri::command]
pub async fn stop_server(state: State<'_, Arc<AppState>>) -> CmdResult<Snapshot> {
    let st = state.inner().clone();
    let handle = lock(&st.server).take();
    match handle {
        Some(handle) => {
            let address = handle.address.clone();
            handle.shutdown().await;
            st.log_event(
                "app",
                st.mode(),
                None,
                log_status::OK,
                None,
                format!("local server at {address} stopped"),
            );
        }
        None => {}
    }
    st.emit_status();
    Ok(st.snapshot(None))
}

#[tauri::command]
pub fn update_server_settings(
    state: State<'_, Arc<AppState>>,
    port: Option<u16>,
    api_wait_enabled: Option<bool>,
    api_max_wait_secs: Option<u64>,
) -> CmdResult<Snapshot> {
    let st = state.inner();
    if let Some(port) = port {
        if port == 0 {
            return Err("port must be between 1 and 65535".to_string());
        }
    }
    {
        let mut settings = lock(&st.settings);
        if let Some(port) = port {
            settings.port = port;
        }
        if let Some(enabled) = api_wait_enabled {
            settings.api_wait_enabled = enabled;
        }
        if let Some(secs) = api_max_wait_secs {
            settings.api_max_wait_secs = secs.clamp(5, 3600);
        }
    }
    st.persist_settings()?;
    st.emit_status();
    Ok(st.snapshot(None))
}

/// Add one more local access token slot.
#[tauri::command]
pub fn add_access_token(
    state: State<'_, Arc<AppState>>,
    label: Option<String>,
) -> CmdResult<Snapshot> {
    let st = state.inner();
    let index = {
        let settings = lock(&st.settings);
        settings.access_tokens.len()
    };
    let token = config::AccessToken {
        id: uuid::Uuid::new_v4().simple().to_string(),
        label: label
            .filter(|l| !l.trim().is_empty())
            .unwrap_or_else(|| format!("Token {}", index + 1)),
        token: format!("sk-gr-{}", uuid::Uuid::new_v4().simple()),
    };
    let name = token.label.clone();
    lock(&st.settings).access_tokens.push(token);
    st.persist_settings()?;
    st.log_event(
        "app",
        st.mode(),
        None,
        log_status::OK,
        None,
        format!("added local access token \"{name}\""),
    );
    st.emit_status();
    Ok(st.snapshot(None))
}

#[tauri::command]
pub fn remove_access_token(state: State<'_, Arc<AppState>>, id: String) -> CmdResult<Snapshot> {
    let st = state.inner();
    lock(&st.settings).access_tokens.retain(|t| t.id != id);
    st.persist_settings()?;
    st.emit_status();
    Ok(st.snapshot(None))
}

#[tauri::command]
pub fn regenerate_access_token(state: State<'_, Arc<AppState>>, id: String) -> CmdResult<Snapshot> {
    let st = state.inner();
    {
        let mut settings = lock(&st.settings);
        let Some(token) = settings.access_tokens.iter_mut().find(|t| t.id == id) else {
            return Err("that access token no longer exists".to_string());
        };
        token.token = format!("sk-gr-{}", uuid::Uuid::new_v4().simple());
    }
    st.persist_settings()?;
    st.log_event(
        "app",
        st.mode(),
        None,
        log_status::OK,
        None,
        "regenerated a local access token",
    );
    st.emit_status();
    Ok(st.snapshot(None))
}

// ---------------------------------------------------------------------------
// Inject
// ---------------------------------------------------------------------------

/// The config file each coding tool owns, so the UI can show what will change.
#[tauri::command]
pub fn inject_targets() -> CmdResult<Vec<inject::InjectTarget>> {
    inject::targets()
}

/// Point one coding tool at the local server, pinned to a routing mode.
///
/// `mode` is `flash`, `flash_lite`, `both`, or `None` to keep the app's current
/// mode. Starts the server and creates an access token if needed.
#[tauri::command]
pub async fn inject_tool(
    state: State<'_, Arc<AppState>>,
    target: String,
    mode: Option<String>,
) -> CmdResult<inject::InjectResult> {
    let st = state.inner().clone();
    let parsed = match mode.as_deref().map(str::trim).filter(|m| !m.is_empty()) {
        Some("current") | None => None,
        Some(value) => Some(Mode::parse(value).ok_or_else(|| format!("unknown mode: {value}"))?),
    };
    inject::inject_tool(&st, &target, parsed).await
}

// ---------------------------------------------------------------------------
// Request log
// ---------------------------------------------------------------------------

#[tauri::command]
pub fn clear_log(state: State<'_, Arc<AppState>>) -> Snapshot {
    let st = state.inner();
    lock(&st.log).clear();
    st.emit_status();
    st.snapshot(None)
}

#[tauri::command]
pub fn refresh(state: State<'_, Arc<AppState>>, status_slot: Option<u32>) -> Snapshot {
    state.snapshot(status_slot)
}
