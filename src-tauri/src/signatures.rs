//! Side table for Gemini `thoughtSignature` values.
//!
//! Gemini 3 requires the signature that came with a function call to be sent
//! back alongside that same call in later turns.  OpenAI and Anthropic clients
//! have no field for it, so the router remembers signatures under the tool-call
//! id that the client sees.  When the client replays history, the request
//! translators look the id up and attach the signature to the matching
//! `functionCall` part.
//!
//! The table is persisted because desktop sessions routinely outlive a router
//! restart.  A missing or corrupt cache is never fatal: it only means old tool
//! calls lose their signatures.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::Value;

const FILE_NAME: &str = "thought_signatures.json";
const MAX_AGE_SECS: u64 = 7 * 24 * 60 * 60;

/// Sentinel accepted by Gemini 3 when a function call did not originate from a
/// response that carried a signature.  It is a fallback for history imported
/// from other models (or created before this cache existed); real signatures
/// are always preferred.
pub const SKIP_VALIDATOR: &str = "skip_thought_signature_validator";

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Entry {
    signature: String,
    saved_at: u64,
}

#[derive(Default)]
struct Store {
    path: Option<PathBuf>,
    entries: HashMap<String, Entry>,
}

#[derive(Debug, Serialize, Deserialize)]
struct DiskStore {
    version: u32,
    signatures: HashMap<String, Entry>,
}

static STORE: OnceLock<Mutex<Store>> = OnceLock::new();

fn global() -> &'static Mutex<Store> {
    STORE.get_or_init(|| Mutex::new(Store::default()))
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

/// Load saved signatures from `data_dir`.  Called once during app startup.
pub fn init(data_dir: &Path) {
    let path = data_dir.join(FILE_NAME);
    let mut entries = HashMap::new();

    match std::fs::read_to_string(&path) {
        Ok(text) => match serde_json::from_str::<DiskStore>(&text) {
            Ok(disk) if disk.version == 1 => {
                let cutoff = now().saturating_sub(MAX_AGE_SECS);
                entries = disk
                    .signatures
                    .into_iter()
                    .filter(|(_, entry)| entry.saved_at >= cutoff)
                    .collect();
            }
            Ok(_) => {
                eprintln!(
                    "ignoring {}: unsupported thought-signature cache version",
                    path.display()
                );
            }
            Err(error) => {
                // A corrupt cache must never prevent the router from starting.
                eprintln!("could not read {}: {error}", path.display());
            }
        },
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            eprintln!("could not read {}: {error}", path.display());
        }
    }

    let mut store = global().lock().unwrap_or_else(|error| error.into_inner());
    store.path = Some(path);
    store.entries = entries;
}

/// Remember the signature that belongs to `call_id`.
pub fn put(call_id: &str, signature: &str) {
    if call_id.is_empty() || signature.is_empty() {
        return;
    }

    let mut store = global().lock().unwrap_or_else(|error| error.into_inner());
    let cutoff = now().saturating_sub(MAX_AGE_SECS);
    store.entries.retain(|_, entry| entry.saved_at >= cutoff);
    store.entries.insert(
        call_id.to_string(),
        Entry {
            signature: signature.to_string(),
            saved_at: now(),
        },
    );
    store.save();
}

/// Extract the signature attached to a Gemini response part.
///
/// The REST API puts `thoughtSignature` next to `functionCall`, but a few
/// proxies nest it, so accept both shapes.
pub fn from_part(part: &Value) -> Option<String> {
    part.get("thoughtSignature")
        .or_else(|| part.get("thought_signature"))
        .or_else(|| {
            part.get("functionCall")
                .and_then(|call| call.get("thoughtSignature"))
        })
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

/// Resolve the signature to replay for a tool call.
///
/// Clients that preserve a custom extension field win; otherwise fall back to
/// the side table keyed by the client-visible call id.
pub fn for_replay(call: &Value, call_id: &str) -> Option<String> {
    call.get("thoughtSignature")
        .or_else(|| call.get("thought_signature"))
        // Gemini's own OpenAI-compatible endpoint exposes the signature here,
        // and some clients preserve the field verbatim.
        .or_else(|| call.pointer("/extra_content/google/thought_signature"))
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .or_else(|| get(call_id))
}

/// Look up the signature for a tool-call id.
pub fn get(call_id: &str) -> Option<String> {
    if call_id.is_empty() {
        return None;
    }
    let store = global().lock().unwrap_or_else(|error| error.into_inner());
    store
        .entries
        .get(call_id)
        .map(|entry| entry.signature.clone())
}

impl Store {
    fn save(&self) {
        let Some(path) = &self.path else {
            return;
        };
        let disk = DiskStore {
            version: 1,
            signatures: self.entries.clone(),
        };
        let Ok(text) = serde_json::to_string(&disk) else {
            return;
        };
        // The cache is advisory; a failed write must not fail the request.
        let _ = std::fs::write(path, text);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stores_and_reads_a_signature() {
        let id = format!("call_{}", uuid::Uuid::new_v4().simple());
        put(&id, "sig");
        assert_eq!(get(&id).as_deref(), Some("sig"));
    }

    #[test]
    fn empty_values_are_ignored() {
        assert!(get("").is_none());
        put("", "sig");
        put("call_empty", "");
        assert!(get("call_empty").is_none());
    }
}
