//! API key storage.
//!
//! Preferred backend is the OS keychain (Windows Credential Manager / macOS
//! Keychain / Secret Service) via the `keyring` crate. If no keychain is
//! available we fall back to a file in the app data directory with permissions
//! restricted to the current user, and the caller is expected to warn the user.
//!
//! `GEMINI_API_KEY` is honoured as a last-resort fallback for slot 0 only.
//! Key values are never logged.

use std::fs;
use std::path::{Path, PathBuf};

use serde::Serialize;

/// Keychain service name; slot index forms the account name.
const SERVICE: &str = "gemini-router";
pub const ENV_FALLBACK_KEY: &str = "GEMINI_API_KEY";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum StorageKind {
    Keychain,
    File,
    None,
}

/// Status shown in the UI key area.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum KeyStatus {
    NoKey,
    Saved,
    Rejected,
}

impl KeyStatus {
    pub fn label(self) -> &'static str {
        match self {
            KeyStatus::NoKey => "No key",
            KeyStatus::Saved => "Key saved",
            KeyStatus::Rejected => "Key rejected by API",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Slot {
    pub index: u32,
    pub value: Option<String>,
    pub storage: StorageKind,
    pub status: KeyStatus,
    /// Where the value came from: `keychain`, `file`, `env:GEMINI_API_KEY`, `none`.
    pub source: String,
}

impl Slot {
    pub fn present(&self) -> bool {
        self.value.is_some()
    }
}

pub struct KeyStore {
    pub slots: Vec<Slot>,
    /// `Ok(())` when the OS keychain initialised, otherwise the reason it did not.
    pub keychain: Result<(), String>,
}

impl KeyStore {
    pub fn new(data_dir: &Path, slot_count: u32) -> Self {
        let keychain = keychain_status();
        let count = slot_count.max(1);
        let mut slots = Vec::with_capacity(count as usize);
        for index in 0..count {
            slots.push(load_slot(data_dir, index));
        }
        KeyStore { slots, keychain }
    }

    pub fn get(&self, index: u32) -> Option<&str> {
        self.slots
            .iter()
            .find(|s| s.index == index)
            .and_then(|s| s.value.as_deref())
    }

    pub fn slot(&self, index: u32) -> Option<&Slot> {
        self.slots.iter().find(|s| s.index == index)
    }

    /// Grow the slot list to `count` entries, loading any that are new.
    pub fn ensure_slots(&mut self, data_dir: &Path, count: u32) {
        let count = count.max(1);
        for index in 0..count {
            if !self.slots.iter().any(|s| s.index == index) {
                self.slots.push(load_slot(data_dir, index));
            }
        }
        self.slots.retain(|s| s.index < count);
        self.slots.sort_by_key(|s| s.index);
    }

    /// Refresh every slot from its backing store.
    pub fn reload(&mut self, data_dir: &Path, count: u32) {
        self.keychain = keychain_status();
        let count = count.max(1);
        self.slots = (0..count).map(|i| load_slot(data_dir, i)).collect();
    }

    pub fn set_status(&mut self, index: u32, status: KeyStatus) {
        if let Some(slot) = self.slots.iter_mut().find(|s| s.index == index) {
            slot.status = status;
        }
    }

    /// All slots that currently hold a usable key, in slot order.
    pub fn usable_slots(&self) -> Vec<u32> {
        let mut out: Vec<u32> = self
            .slots
            .iter()
            .filter(|s| s.value.is_some() && s.status != KeyStatus::Rejected)
            .map(|s| s.index)
            .collect();
        out.sort_unstable();
        out
    }
}

pub fn slot_file(data_dir: &Path, index: u32) -> PathBuf {
    data_dir.join("keys").join(format!("slot-{index}.key"))
}

fn entry(index: u32) -> Result<keyring::Entry, keyring::Error> {
    keyring::Entry::new(SERVICE, &format!("slot-{index}"))
}

/// Probe the OS keychain without storing anything.
pub fn keychain_status() -> Result<(), String> {
    match keyring::Entry::store_status() {
        Ok(()) => Ok(()),
        Err(e) => Err(e.to_string()),
    }
}

fn load_slot(data_dir: &Path, index: u32) -> Slot {
    // 1. OS keychain.
    if let Ok(entry) = entry(index) {
        if let Ok(value) = entry.get_password() {
            let value = value.trim().to_string();
            if !value.is_empty() {
                return Slot {
                    index,
                    value: Some(value),
                    storage: StorageKind::Keychain,
                    status: KeyStatus::Saved,
                    source: "keychain".into(),
                };
            }
        }
    }

    // 2. Restricted file in the app data directory.
    if let Ok(text) = fs::read_to_string(slot_file(data_dir, index)) {
        let text = text.trim().to_string();
        if !text.is_empty() {
            return Slot {
                index,
                value: Some(text),
                storage: StorageKind::File,
                status: KeyStatus::Saved,
                source: "file".into(),
            };
        }
    }

    // 3. Environment variable, slot 0 only.
    if index == 0 {
        if let Ok(value) = std::env::var(ENV_FALLBACK_KEY) {
            let value = value.trim().to_string();
            if !value.is_empty() {
                return Slot {
                    index,
                    value: Some(value),
                    storage: StorageKind::None,
                    status: KeyStatus::Saved,
                    source: format!("env:{ENV_FALLBACK_KEY}"),
                };
            }
        }
    }

    Slot {
        index,
        value: None,
        storage: StorageKind::None,
        status: KeyStatus::NoKey,
        source: "none".into(),
    }
}

pub struct SaveOutcome {
    pub storage: StorageKind,
    pub status: KeyStatus,
    /// Primary message, e.g. "Key saved".
    pub message: String,
    /// Non-empty when the key could not go into the keychain.
    pub warning: String,
}

/// Clean up a pasted API key.
///
/// Keys are commonly pasted from a `.env` file, a terminal (`export …`), or a
/// web page with quotes or a trailing newline. Anything that cannot be part of
/// a key is a hard error rather than something to store and fail on later.
pub fn normalise_key(raw: &str) -> Result<String, String> {
    // Invisible characters that survive a copy from some editors.
    let mut value: String = raw
        .chars()
        .filter(|c| !matches!(c, '\u{feff}' | '\u{200b}' | '\u{200c}' | '\u{200d}'))
        .collect();
    value = value.trim().to_string();

    // Peel off wrappers such as `export GEMINI_API_KEY="AIza…"`.
    loop {
        let before = value.clone();
        let lower = value.to_ascii_lowercase();

        for prefix in ["export ", "set "] {
            if lower.starts_with(prefix) {
                value = value[prefix.len()..].trim().to_string();
                break;
            }
        }
        for prefix in ["GEMINI_API_KEY=", "GOOGLE_API_KEY=", "API_KEY="] {
            if let Some(rest) = value.strip_prefix(prefix) {
                value = rest.trim().to_string();
                break;
            }
        }
        let lower = value.to_ascii_lowercase();
        if lower.starts_with("bearer ") {
            value = value[7..].trim().to_string();
        }
        if value == before {
            break;
        }
    }

    // Matching surrounding quotes.
    if value.chars().count() >= 2 {
        let first = value.chars().next().unwrap_or(' ');
        let last = value.chars().last().unwrap_or(' ');
        if (first == '"' && last == '"') || (first == '\'' && last == '\'') {
            value = value[1..value.len() - 1].trim().to_string();
        }
    }

    if value.is_empty() {
        return Err("no key was provided".to_string());
    }
    if value.chars().any(char::is_whitespace) {
        return Err(
            "the key contains a space or line break. Copy it again as one continuous string."
                .to_string(),
        );
    }
    if !value.chars().all(|c| c.is_ascii_graphic()) {
        return Err("the key contains characters that cannot appear in an API key".to_string());
    }
    Ok(value)
}

/// A soft check for the usual Google API key shape. Never blocks a save.
fn key_shape_warning(value: &str) -> Option<String> {
    let looks_like_google = value.starts_with("AIza") && value.len() == 39;
    if looks_like_google {
        None
    } else {
        Some(format!(
            "Saved, but this does not look like a Google API key (usually 39 characters starting \
             with \"AIza\"). If requests fail with 401/403, re-copy the key from Google AI Studio."
        ))
    }
}

pub fn save_key(data_dir: &Path, index: u32, raw: &str) -> SaveOutcome {
    let value = match normalise_key(raw) {
        Ok(value) => value,
        Err(message) => {
            return SaveOutcome {
                storage: StorageKind::None,
                status: KeyStatus::NoKey,
                message: format!("Failed to save key: {message}"),
                warning: String::new(),
            };
        }
    };

    let mut outcome = store_key(data_dir, index, &value);
    if outcome.status == KeyStatus::Saved {
        if let Some(warning) = key_shape_warning(&value) {
            outcome.warning = if outcome.warning.is_empty() {
                warning
            } else {
                format!("{warning} {}", outcome.warning)
            };
        }
    }
    outcome
}

/// Store an already-normalised key, preferring the OS keychain.
fn store_key(data_dir: &Path, index: u32, value: &str) -> SaveOutcome {
    match entry(index).and_then(|e| e.set_password(value)) {
        Ok(()) => {
            // Drop any stale plaintext copy for this slot.
            let _ = fs::remove_file(slot_file(data_dir, index));
            SaveOutcome {
                storage: StorageKind::Keychain,
                status: KeyStatus::Saved,
                message: KeyStatus::Saved.label().to_string(),
                warning: String::new(),
            }
        }
        Err(err) => {
            let path = slot_file(data_dir, index);
            if let Some(parent) = path.parent() {
                if let Err(e) = fs::create_dir_all(parent) {
                    return SaveOutcome {
                        storage: StorageKind::None,
                        status: KeyStatus::NoKey,
                        message: format!("Failed to save key: {e}"),
                        warning: String::new(),
                    };
                }
            }
            match fs::write(&path, value) {
                Ok(()) => {
                    let warning = match restrict_permissions(&path) {
                        Ok(()) => format!(
                            "OS keychain unavailable ({err}). Key stored in {} with permissions \
                             restricted to your user account.",
                            path.display()
                        ),
                        Err(perm_err) => format!(
                            "OS keychain unavailable ({err}). Key stored in {} but permissions \
                             could NOT be restricted ({perm_err}). Treat this file as sensitive.",
                            path.display()
                        ),
                    };
                    SaveOutcome {
                        storage: StorageKind::File,
                        status: KeyStatus::Saved,
                        message: KeyStatus::Saved.label().to_string(),
                        warning,
                    }
                }
                Err(write_err) => SaveOutcome {
                    storage: StorageKind::None,
                    status: KeyStatus::NoKey,
                    message: format!("Failed to save key: {write_err}"),
                    warning: String::new(),
                },
            }
        }
    }
}

pub fn delete_key(data_dir: &Path, index: u32) {
    if let Ok(entry) = entry(index) {
        let _ = entry.delete_credential();
    }
    let _ = fs::remove_file(slot_file(data_dir, index));
}

/// Remove inherited ACEs and grant only the current user. Best effort: a failure
/// is surfaced to the user as a warning rather than blocking the save.
fn restrict_permissions(path: &Path) -> Result<(), String> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = fs::Permissions::from_mode(0o600);
        fs::set_permissions(path, mode).map_err(|e| e.to_string())
    }

    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        /// CREATE_NO_WINDOW so icacls does not flash a console.
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;

        let user = std::env::var("USERNAME").map_err(|_| "USERNAME is not set".to_string())?;
        let output = std::process::Command::new("icacls")
            .arg(path)
            .arg("/inheritance:r")
            .arg("/grant:r")
            .arg(format!("{user}:F"))
            .creation_flags(CREATE_NO_WINDOW)
            .output()
            .map_err(|e| e.to_string())?;
        if output.status.success() {
            Ok(())
        } else {
            Err(String::from_utf8_lossy(&output.stderr).trim().to_string())
        }
    }

    #[cfg(not(any(unix, windows)))]
    {
        let _ = path;
        Err("unsupported platform".into())
    }
}

/// Redact a key value from any string before it is logged or returned.
pub fn redact(text: &str, key: Option<&str>) -> String {
    match key {
        Some(k) if !k.is_empty() => text.replace(k, "***"),
        _ => text.to_string(),
    }
}

/// A short, non-reversible label for a key, safe to show or log.
pub fn preview(value: &str) -> String {
    let chars: Vec<char> = value.chars().collect();
    if chars.len() <= 8 {
        return "••••".to_string();
    }
    let head: String = chars[..4].iter().collect();
    let tail: String = chars[chars.len() - 4..].iter().collect();
    format!("{head}…{tail}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preview_never_reveals_the_middle() {
        let p = preview("AIzaSyD-abcdefghijklmnop1234");
        assert!(p.starts_with("AIza"));
        assert!(p.ends_with("1234"));
        assert!(p.contains('…'));
        assert!(!p.contains("efghijkl"));
    }

    #[test]
    fn preview_handles_short_values() {
        assert_eq!(preview("abc"), "••••");
    }

    #[test]
    fn redact_removes_every_occurrence() {
        let key = "SECRETKEY123";
        let text = format!("url?key={key} and again {key}");
        let out = redact(&text, Some(key));
        assert!(!out.contains(key));
        assert_eq!(out.matches("***").count(), 2);
    }

    #[test]
    fn redact_is_a_noop_without_a_key() {
        assert_eq!(redact("hello", None), "hello");
    }

    #[test]
    fn normalise_key_strips_common_paste_wrappers() {
        let key = format!("AIza{}", "x".repeat(35));
        for raw in [
            key.clone(),
            format!("  {key}\n"),
            format!("\"{key}\""),
            format!("'{key}'"),
            format!("GEMINI_API_KEY={key}"),
            format!("export GEMINI_API_KEY=\"{key}\""),
            format!("set GEMINI_API_KEY={key}"),
            format!("Bearer {key}"),
            format!("\u{feff}{key}\u{200b}"),
        ] {
            assert_eq!(normalise_key(&raw).unwrap(), key, "failed for {raw:?}");
        }
    }

    #[test]
    fn normalise_key_rejects_empty_and_whitespace() {
        assert!(normalise_key("   ").is_err());
        assert!(normalise_key("AIza abc").is_err());
        assert!(normalise_key("AIza\nabc").is_err());
        assert!(normalise_key("AIza\u{00a0}abc").is_err());
    }

    #[test]
    fn shape_warning_only_fires_for_odd_keys() {
        assert!(key_shape_warning(&format!("AIza{}", "x".repeat(35))).is_none());
        assert!(key_shape_warning("not-a-google-key").is_some());
    }
}
