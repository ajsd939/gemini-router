//! Persistent configuration.
//!
//! `config.json` holds the model chains exactly as documented (the user owns this
//! file and may edit it by hand). `settings.json` holds runtime state that the UI
//! mutates: current mode, port, key-slot count, access tokens, wait policy.

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Routing mode. The three variants map 1:1 onto the modes in the README.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    Flash,
    FlashLite,
    Both,
}

impl Default for Mode {
    fn default() -> Self {
        Mode::Both
    }
}

impl std::fmt::Display for Mode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl Mode {
    pub fn as_str(self) -> &'static str {
        match self {
            Mode::Flash => "flash",
            Mode::FlashLite => "flash_lite",
            Mode::Both => "both",
        }
    }

    pub fn parse(s: &str) -> Option<Mode> {
        match s.trim().to_ascii_lowercase().as_str() {
            "flash" => Some(Mode::Flash),
            "flash_lite" | "flash-lite" | "flashlite" | "lite" => Some(Mode::FlashLite),
            "both" | "all" => Some(Mode::Both),
            _ => None,
        }
    }
}

/// Defaults below were checked against https://ai.google.dev/gemini-api/docs/models
/// (updated 2026-10-13). Two IDs from the original spec do not exist as endpoints
/// and were dropped after confirmation:
///   * `gemini-3.7-flash`  -> no such model
///   * `gemini-3.5-flash`  -> no such model (only `gemini-3.5-flash-lite`)
pub const DEFAULT_FLASH_CHAIN: [&str; 2] = ["gemini-3.8-flash", "gemini-3.6-flash"];
pub const DEFAULT_FLASH_LITE_CHAIN: [&str; 2] = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"];
pub const DEFAULT_SERVER_PORT: u16 = 8787;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub flash_chain: Vec<String>,
    pub flash_lite_chain: Vec<String>,
    pub default_mode: Mode,
    pub server_port: u16,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            flash_chain: DEFAULT_FLASH_CHAIN.iter().map(|s| s.to_string()).collect(),
            flash_lite_chain: DEFAULT_FLASH_LITE_CHAIN
                .iter()
                .map(|s| s.to_string())
                .collect(),
            default_mode: Mode::Both,
            server_port: DEFAULT_SERVER_PORT,
        }
    }
}

fn normalise_chain(chain: &mut Vec<String>) {
    let mut seen = std::collections::HashSet::new();
    chain.retain_mut(|model| {
        *model = model.trim().to_string();
        !model.is_empty() && seen.insert(model.clone())
    });
}

impl Config {
    /// Validate and normalise a config loaded from disk or sent by the UI.
    pub fn sanitised(mut self) -> Result<Config, String> {
        normalise_chain(&mut self.flash_chain);
        normalise_chain(&mut self.flash_lite_chain);
        if self.flash_chain.is_empty() {
            return Err("flash_chain must contain at least one model".into());
        }
        if self.flash_lite_chain.is_empty() {
            return Err("flash_lite_chain must contain at least one model".into());
        }
        if self.server_port == 0 {
            return Err("server_port must be between 1 and 65535".into());
        }
        Ok(self)
    }
}

/// One access token accepted by the local OpenAI-compatible server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessToken {
    pub id: String,
    pub label: String,
    pub token: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    /// Active mode. Shared by the UI and the local server; changes apply to the
    /// next request.
    pub mode: Mode,
    pub port: u16,
    /// Number of Gemini API key slots. Starts at 1; the UI can add more.
    pub key_slots: u32,
    pub access_tokens: Vec<AccessToken>,
    /// Whether the local server may block waiting for a model to come back.
    pub api_wait_enabled: bool,
    pub api_max_wait_secs: u64,
    /// How long the desktop UI is willing to block waiting for a model.
    pub ui_max_wait_secs: u64,
    /// Total attempts (not retries) per model for transient failures.
    pub transient_attempts: u32,
    /// Cooldown applied to a per-minute 429 when the response has no retry delay.
    pub default_cooldown_secs: u64,
    pub request_timeout_secs: u64,
    /// Safety valve for the "both" mode restart rule.
    pub max_restarts: u32,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            mode: Mode::Both,
            port: DEFAULT_SERVER_PORT,
            key_slots: 1,
            access_tokens: Vec::new(),
            api_wait_enabled: false,
            api_max_wait_secs: 300,
            ui_max_wait_secs: 300,
            transient_attempts: 3,
            default_cooldown_secs: 60,
            request_timeout_secs: 120,
            max_restarts: 8,
        }
    }
}

pub fn config_path(data_dir: &Path) -> PathBuf {
    data_dir.join("config.json")
}

pub fn settings_path(data_dir: &Path) -> PathBuf {
    data_dir.join("settings.json")
}

pub fn load_config(data_dir: &Path) -> Config {
    let path = config_path(data_dir);
    match fs::read_to_string(&path) {
        Ok(text) => match serde_json::from_str::<Config>(&text) {
            Ok(cfg) => cfg.sanitised().unwrap_or_default(),
            Err(e) => {
                // Never overwrite a hand-edited file just because it has a typo:
                // the user needs it intact to fix it. Defaults apply for this run.
                eprintln!(
                    "[gemini-router] config.json is invalid ({e}); using defaults without overwriting it"
                );
                Config::default()
            }
        },
        Err(_) => write_default_config(data_dir).unwrap_or_default(),
    }
}

fn write_default_config(data_dir: &Path) -> Result<Config, ()> {
    let cfg = Config::default();
    let _ = save_config(data_dir, &cfg);
    Ok(cfg)
}

pub fn save_config(data_dir: &Path, cfg: &Config) -> Result<(), String> {
    let path = config_path(data_dir);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("create {}: {e}", parent.display()))?;
    }
    let text = serde_json::to_string_pretty(cfg).map_err(|e| e.to_string())?;
    fs::write(&path, text).map_err(|e| format!("write {}: {e}", path.display()))
}

pub fn load_settings(data_dir: &Path) -> Settings {
    let path = settings_path(data_dir);
    match fs::read_to_string(&path) {
        Ok(text) => serde_json::from_str::<Settings>(&text).unwrap_or_default(),
        Err(_) => {
            let s = Settings::default();
            let _ = save_settings(data_dir, &s);
            s
        }
    }
}

pub fn save_settings(data_dir: &Path, settings: &Settings) -> Result<(), String> {
    let path = settings_path(data_dir);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("create {}: {e}", parent.display()))?;
    }
    let text = serde_json::to_string_pretty(settings).map_err(|e| e.to_string())?;
    fs::write(&path, text).map_err(|e| format!("write {}: {e}", path.display()))
}

/// Next midnight in Pacific Time, returned as a UTC instant.
///
/// Daily (RPD/TPD) Gemini quotas reset at midnight Pacific, so an exhausted model
/// is blocked until this instant regardless of the machine's local timezone.
pub fn next_midnight_pacific() -> chrono::DateTime<chrono::Utc> {
    use chrono::{NaiveTime, TimeZone, Utc};
    let tz = chrono_tz::America::Los_Angeles;
    let now_local = Utc::now().with_timezone(&tz);
    let tomorrow = now_local
        .date_naive()
        .succ_opt()
        .unwrap_or_else(|| now_local.date_naive());
    let naive = tomorrow.and_time(NaiveTime::MIN);
    tz.from_local_datetime(&naive)
        .earliest()
        .map(|dt| dt.with_timezone(&Utc))
        // Pacific DST transitions can make midnight ambiguous/nonexistent;
        // interpreting it as UTC is only ever off by an hour.
        .unwrap_or_else(|| Utc.from_utc_datetime(&naive))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_spec() {
        let cfg = Config::default();
        assert_eq!(
            cfg.flash_chain,
            vec!["gemini-3.8-flash", "gemini-3.6-flash"]
        );
        assert_eq!(
            cfg.flash_lite_chain,
            vec!["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
        );
        assert_eq!(cfg.default_mode, Mode::Both);
        assert_eq!(cfg.server_port, 8787);
    }

    #[test]
    fn mode_round_trips_through_json() {
        assert_eq!(
            serde_json::to_string(&Mode::FlashLite).unwrap(),
            "\"flash_lite\""
        );
        let m: Mode = serde_json::from_str("\"flash_lite\"").unwrap();
        assert_eq!(m, Mode::FlashLite);
        assert_eq!(Mode::parse("Flash-Lite"), Some(Mode::FlashLite));
        assert_eq!(Mode::parse("nope"), None);
    }

    #[test]
    fn partial_config_falls_back_to_defaults() {
        let cfg: Config = serde_json::from_str(r#"{"server_port": 9000}"#).unwrap();
        assert_eq!(cfg.server_port, 9000);
        assert_eq!(cfg.flash_chain, Config::default().flash_chain);
    }

    #[test]
    fn pacific_midnight_is_in_the_future_and_early_utc() {
        let mid = next_midnight_pacific();
        assert!(mid > chrono::Utc::now());
        // Midnight Pacific is 07:00/08:00 UTC.
        use chrono::Timelike;
        let hour = mid.hour();
        assert!(hour == 7 || hour == 8, "unexpected UTC hour {hour}");
        assert_eq!(mid.minute(), 0);
    }

    #[test]
    fn invalid_config_file_is_left_untouched() {
        let dir = std::env::temp_dir().join(format!(
            "gemini-router-config-test-{}",
            uuid::Uuid::new_v4()
        ));
        fs::create_dir_all(&dir).unwrap();
        let path = config_path(&dir);
        fs::write(&path, "{ not valid json").unwrap();

        let cfg = load_config(&dir);
        assert_eq!(cfg.flash_chain, Config::default().flash_chain);
        // A typo must not destroy the user's hand-edited file.
        assert_eq!(fs::read_to_string(&path).unwrap(), "{ not valid json");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn sanitise_rejects_empty_chain() {
        let mut cfg = Config::default();
        cfg.flash_chain.clear();
        assert!(cfg.sanitised().is_err());
    }

    #[test]
    fn sanitise_trims_and_deduplicates_model_ids() {
        let mut cfg = Config::default();
        cfg.flash_chain = vec![
            " gemini-3.8-flash ".into(),
            "gemini-3.8-flash".into(),
            "".into(),
            "gemini-3.6-flash".into(),
        ];
        let cfg = cfg.sanitised().unwrap();
        assert_eq!(
            cfg.flash_chain,
            vec!["gemini-3.8-flash", "gemini-3.6-flash"]
        );
    }
}
