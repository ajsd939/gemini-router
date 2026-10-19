//! One-click "inject" support for the three coding tools.
//!
//! Each injection ensures the local server and an access token exist, then
//! writes the tool's user-level config so it points at 127.0.0.1.
//!
//! The routing mode is injected per request rather than by changing the app's
//! global mode: every tool sends the mode name (`flash`, `flash_lite`, `both`)
//! as its model, which the local server already accepts as a per-request
//! routing override. Passing `None` leaves the tool on the app's current mode by
//! naming a concrete chain model instead. Tokens are written only to user config
//! files and are never logged.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::Serialize;
use serde_json::{Map, Value, json};

use crate::config::{AccessToken, Mode};
use crate::logging::status as log_status;
use crate::server;
use crate::state::{AppState, lock};

pub const CLAUDE: &str = "claude";
pub const CODEX: &str = "codex";
pub const OPENCODE: &str = "opencode";

/// Routing aliases the local server accepts in the `model` field.
const MODE_ALIASES: [&str; 3] = ["flash", "flash_lite", "both"];

/// One tool that can be injected, with the config file it owns.
#[derive(Debug, Clone, Serialize)]
pub struct InjectTarget {
    pub id: String,
    pub label: String,
    pub path: String,
    pub exists: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct InjectResult {
    pub target: String,
    pub label: String,
    /// `flash`, `flash_lite`, `both`, or `current`.
    pub mode: String,
    pub path: String,
    pub message: String,
    /// Non-empty when the tool needs an extra manual step.
    pub warning: String,
}

fn home() -> Result<PathBuf, String> {
    std::env::var_os("USERPROFILE")
        .or_else(|| std::env::var_os("HOME"))
        .map(PathBuf::from)
        .ok_or_else(|| "could not locate the user home directory".to_string())
}

fn claude_path() -> Result<PathBuf, String> {
    Ok(home()?.join(".claude").join("settings.json"))
}

fn codex_path() -> Result<PathBuf, String> {
    Ok(home()?.join(".codex").join("config.toml"))
}

fn opencode_dir() -> Result<PathBuf, String> {
    Ok(home()?.join(".config").join("opencode"))
}

/// Prefer `opencode.json` when it exists, but write to `opencode.jsonc` when
/// that is the only config file the user has. Creating a second `opencode.json`
/// next to an existing `.jsonc` would silently be ignored by OpenCode.
fn choose_opencode_path(dir: &Path) -> PathBuf {
    let json = dir.join("opencode.json");
    let jsonc = dir.join("opencode.jsonc");
    if !json.exists() && jsonc.exists() {
        jsonc
    } else {
        json
    }
}

fn opencode_path() -> Result<PathBuf, String> {
    Ok(choose_opencode_path(&opencode_dir()?))
}

/// The config files each tool reads, for the inject panel.
pub fn targets() -> Result<Vec<InjectTarget>, String> {
    let claude = claude_path()?;
    let codex = codex_path()?;
    let opencode = opencode_path()?;
    Ok(vec![
        target(CLAUDE, "Claude Code", claude),
        target(CODEX, "Codex", codex),
        target(OPENCODE, "OpenCode", opencode),
    ])
}

fn target(id: &str, label: &str, path: PathBuf) -> InjectTarget {
    InjectTarget {
        id: id.to_string(),
        label: label.to_string(),
        path: path.display().to_string(),
        exists: path.exists(),
    }
}

fn ensure_parent(path: &Path) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("create {}: {e}", parent.display()))?;
    }
    Ok(())
}

fn write_text(path: &Path, text: &str) -> Result<(), String> {
    ensure_parent(path)?;
    fs::write(path, text).map_err(|e| format!("write {}: {e}", path.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(path, fs::Permissions::from_mode(0o600));
    }
    Ok(())
}

fn read_json(path: &Path) -> Result<Value, String> {
    if !path.exists() {
        return Ok(json!({}));
    }
    let text = fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    if text.trim().is_empty() {
        return Ok(json!({}));
    }
    let is_jsonc = path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| extension.eq_ignore_ascii_case("jsonc"))
        .unwrap_or(false);
    let text = if is_jsonc { strip_jsonc(&text) } else { text };
    serde_json::from_str(&text).map_err(|e| format!("{} is not valid JSON: {e}", path.display()))
}

/// Remove `//` and `/* … */` comments and trailing commas from JSONC.
///
/// OpenCode writes its config as JSONC. `serde_json` cannot parse comments, so
/// the file is normalised before parsing. The rewritten file is plain JSON
/// (which is also valid JSONC); existing comments are not preserved.
fn strip_jsonc(input: &str) -> String {
    let chars: Vec<char> = input.chars().collect();
    let mut without_comments = String::with_capacity(input.len());
    let mut index = 0;
    let mut in_string = false;
    let mut escaped = false;

    while index < chars.len() {
        let character = chars[index];
        if in_string {
            without_comments.push(character);
            if escaped {
                escaped = false;
            } else if character == '\\' {
                escaped = true;
            } else if character == '"' {
                in_string = false;
            }
            index += 1;
            continue;
        }
        if character == '"' {
            in_string = true;
            without_comments.push(character);
            index += 1;
            continue;
        }
        if character == '/' && chars.get(index + 1) == Some(&'/') {
            while index < chars.len() && chars[index] != '\n' {
                index += 1;
            }
            continue;
        }
        if character == '/' && chars.get(index + 1) == Some(&'*') {
            index += 2;
            while index + 1 < chars.len() && !(chars[index] == '*' && chars[index + 1] == '/') {
                index += 1;
            }
            index = (index + 2).min(chars.len());
            continue;
        }
        without_comments.push(character);
        index += 1;
    }

    let chars: Vec<char> = without_comments.chars().collect();
    let mut output = String::with_capacity(without_comments.len());
    let mut in_string = false;
    let mut escaped = false;
    for (index, character) in chars.iter().enumerate() {
        if in_string {
            output.push(*character);
            if escaped {
                escaped = false;
            } else if *character == '\\' {
                escaped = true;
            } else if *character == '"' {
                in_string = false;
            }
            continue;
        }
        if *character == '"' {
            in_string = true;
            output.push(*character);
            continue;
        }
        if *character == ',' {
            let mut next = index + 1;
            while next < chars.len() && chars[next].is_whitespace() {
                next += 1;
            }
            if next < chars.len() && (chars[next] == '}' || chars[next] == ']') {
                continue;
            }
        }
        output.push(*character);
    }
    output
}

fn write_json(path: &Path, value: &Value) -> Result<(), String> {
    let text = serde_json::to_string_pretty(value).map_err(|e| e.to_string())?;
    write_text(path, &text)
}

fn ensure_token(state: &Arc<AppState>) -> Result<String, String> {
    if let Some(token) = lock(&state.settings)
        .access_tokens
        .first()
        .map(|token| token.token.clone())
        .filter(|token| !token.is_empty())
    {
        return Ok(token);
    }

    let token = AccessToken {
        id: uuid::Uuid::new_v4().simple().to_string(),
        label: "Injected tool".to_string(),
        token: format!("sk-gr-{}", uuid::Uuid::new_v4().simple()),
    };
    let value = token.token.clone();
    lock(&state.settings).access_tokens.push(token);
    state.persist_settings()?;
    state.log_event(
        "app",
        state.mode(),
        None,
        log_status::OK,
        None,
        "created a local access token for tool injection",
    );
    Ok(value)
}

async fn ensure_server(state: &Arc<AppState>) -> Result<u16, String> {
    if let Some(handle) = lock(&state.server).as_ref() {
        return Ok(handle.port);
    }

    let requested = lock(&state.settings).port;
    match server::start(state.clone(), requested).await {
        Ok(handle) => {
            let bound = handle.port;
            *lock(&state.server) = Some(handle);
            *lock(&state.server_error) = None;
            lock(&state.settings).port = bound;
            state.persist_settings()?;
            state.log_event(
                "app",
                state.mode(),
                None,
                log_status::OK,
                None,
                format!("local server started at http://127.0.0.1:{bound}/v1 for injection"),
            );
            Ok(bound)
        }
        Err(error) => {
            *lock(&state.server_error) = Some(error.clone());
            Err(error)
        }
    }
}

/// The model string a tool should send: a routing alias when a mode is pinned,
/// otherwise a concrete chain model so the app's global mode applies.
fn alias_for(mode: Option<Mode>, fallback_model: &str) -> String {
    match mode {
        Some(mode) => mode.as_str().to_string(),
        None => fallback_model.to_string(),
    }
}

/// Apply the Claude Code settings to an already-parsed object.
///
/// The routing mode travels in `X-Gemini-Router-Mode`, which the local server
/// checks before the `model` field. That keeps `ANTHROPIC_MODEL` set to a model
/// Claude Code recognizes, so it neither warns about an unknown model nor makes
/// up a context size. `ANTHROPIC_MODEL` outranks a `model` saved in
/// `settings.json`, so that saved default is left untouched.
fn apply_claude_settings(
    root: &mut Value,
    port: u16,
    token: &str,
    mode: Option<Mode>,
) -> Result<(), String> {
    let object = root
        .as_object_mut()
        .ok_or_else(|| "settings.json must contain a JSON object".to_string())?;

    let env = object
        .entry("env")
        .or_insert_with(|| Value::Object(Map::new()))
        .as_object_mut()
        .ok_or_else(|| "`env` must be an object".to_string())?;

    // Claude Code appends /v1/messages itself, so the base URL has no /v1.
    env.insert(
        "ANTHROPIC_BASE_URL".into(),
        json!(format!("http://127.0.0.1:{port}")),
    );
    // Claude Code warns when both auth variables are set ("auth may not work as
    // expected"), so use the bearer token only. The router accepts both styles.
    env.remove("ANTHROPIC_API_KEY");
    env.insert("ANTHROPIC_AUTH_TOKEN".into(), json!(token));

    // Recognized aliases keep Claude Code's model picker and context sizing
    // happy; the header below decides the actual Gemini chain.
    env.insert("ANTHROPIC_MODEL".into(), json!("sonnet"));
    env.insert("ANTHROPIC_SMALL_FAST_MODEL".into(), json!("haiku"));

    // Remove mode aliases written by an earlier version of this injection, so
    // they cannot make Claude Code report an unrecognized model.
    for key in [
        "ANTHROPIC_DEFAULT_OPUS_MODEL",
        "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    ] {
        let should_remove = env
            .get(key)
            .and_then(Value::as_str)
            .map(|v| {
                Mode::parse(v).is_some()
                    || v.to_lowercase().contains("gemini")
                    || v.to_lowercase().contains("flash")
            })
            .unwrap_or(false);
        if should_remove {
            env.remove(key);
        }
    }

    match mode {
        Some(mode) => {
            env.insert(
                "ANTHROPIC_CUSTOM_HEADERS".into(),
                json!(format!("X-Gemini-Router-Mode: {}", mode.as_str())),
            );
        }
        // Follow the app's global mode: no header override.
        None => {
            env.remove("ANTHROPIC_CUSTOM_HEADERS");
        }
    }

    // Gemini model IDs are not in Claude Code's model table, so it would
    // assume a 200K window and warn on every start. This tells it the real
    // window. For an unrecognized ID that does not start with `claude-` (all
    // of ours do not), the variable takes effect on its own and silences the
    // warning. Gemini's current models all take 1M tokens.
    env.insert("CLAUDE_CODE_MAX_CONTEXT_TOKENS".into(), json!("1000000"));

    // Keep Claude Code from calling Anthropic endpoints that a local router
    // cannot serve (telemetry, feature flags).
    env.insert(
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC".into(),
        json!("1"),
    );

    Ok(())
}

fn inject_claude(
    port: u16,
    token: &str,
    mode: Option<Mode>,
) -> Result<(PathBuf, String, String), String> {
    let path = claude_path()?;
    let mut root = read_json(&path)?;
    apply_claude_settings(&mut root, port, token, mode)
        .map_err(|e| format!("{}: {e}", path.display()))?;

    write_json(&path, &root)?;
    Ok((
        path,
        "Injected into Claude Code. Restart Claude Code if it is already running.".to_string(),
        String::new(),
    ))
}

fn is_root_key(line: &str, key: &str) -> bool {
    line.strip_prefix(key)
        .map(|rest| rest.trim_start().starts_with('='))
        .unwrap_or(false)
}

/// Remove a previous injection from `config.toml` without touching other tables.
fn strip_codex_block(text: &str) -> String {
    let mut out = String::new();
    let mut skipping = false;
    let mut in_table = false;
    let mut table = String::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') {
            in_table = true;
            skipping = trimmed == "[model_providers.gemini-router]";
            if skipping {
                continue;
            }
            table = trimmed.trim_matches(['[', ']']).to_string();
        } else if skipping {
            continue;
        }
        // Root keys can only appear before the first table header.
        if !in_table
            && (is_root_key(trimmed, "model")
                || is_root_key(trimmed, "model_provider")
                || is_root_key(trimmed, "model_context_window"))
        {
            continue;
        }
        // Older versions of this injection appended the root keys after the
        // user's last table, which put them inside `[windows]`. Remove those
        // leftovers so they cannot confuse Codex's config parser.
        if table == "windows"
            && (is_root_key(trimmed, "model")
                || is_root_key(trimmed, "model_provider")
                || is_root_key(trimmed, "model_context_window"))
        {
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }
    collapse_blank_lines(out.trim_end())
}

/// Keep at most one blank line between blocks.
fn collapse_blank_lines(text: &str) -> String {
    let mut out: Vec<&str> = Vec::new();
    for line in text.lines() {
        let blank = line.trim().is_empty();
        if blank
            && out
                .last()
                .map(|previous| previous.trim().is_empty())
                .unwrap_or(true)
        {
            continue;
        }
        out.push(line);
    }
    out.join("\n").trim_end().to_string()
}

fn toml_string(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

/// Build the Codex config with the router's root keys and provider table.
fn codex_config_text(existing: &str, port: u16, model: &str) -> String {
    let existing = strip_codex_block(existing);
    // Root keys must come *before* the first table header. Writing them after
    // the user's tables would nest them inside the last table instead.
    let mut text = format!(
        "model = \"{model}\"\n\
         model_provider = \"gemini-router\"\n\
         model_context_window = 1000000\n",
        model = toml_string(model),
    );
    if !existing.is_empty() {
        text.push('\n');
        text.push_str(&existing);
        text.push('\n');
    }
    text.push_str(&format!(
        "\n[model_providers.gemini-router]\n\
         name = \"Gemini Router\"\n\
         base_url = \"http://127.0.0.1:{port}/v1\"\n\
         env_key = \"GEMINI_ROUTER_API_KEY\"\n\
         wire_api = \"responses\"\n",
    ));
    text
}

fn inject_codex(port: u16, token: &str, model: &str) -> Result<(PathBuf, String, String), String> {
    let path = codex_path()?;
    let existing = if path.exists() {
        fs::read_to_string(&path).map_err(|e| format!("read {}: {e}", path.display()))?
    } else {
        String::new()
    };

    let text = codex_config_text(&existing, port, model);
    write_text(&path, &text)?;

    let mut warnings: Vec<String> = Vec::new();
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        match std::process::Command::new("setx")
            .arg("GEMINI_ROUTER_API_KEY")
            .arg(token)
            .creation_flags(CREATE_NO_WINDOW)
            .output()
        {
            Ok(output) if output.status.success() => {}
            Ok(output) => warnings.push(format!(
                "config.toml was written, but setx failed: {}. Set GEMINI_ROUTER_API_KEY to the \
                 token shown under Access tokens.",
                String::from_utf8_lossy(&output.stderr).trim()
            )),
            Err(e) => warnings.push(format!(
                "config.toml was written, but the GEMINI_ROUTER_API_KEY environment variable could \
                 not be set: {e}. Set it to the token shown under Access tokens."
            )),
        }
    }
    #[cfg(not(windows))]
    {
        warnings.push(
            "Set GEMINI_ROUTER_API_KEY in your environment to the token shown under Access tokens."
                .to_string(),
        );
    }
    Ok((
        path,
        "Injected into Codex. Open a new terminal so it sees GEMINI_ROUTER_API_KEY.".to_string(),
        warnings.join(" "),
    ))
}

fn inject_opencode(
    port: u16,
    token: &str,
    model: &str,
) -> Result<(PathBuf, String, String), String> {
    let path = opencode_path()?;
    let mut root = read_json(&path)?;
    let object = root
        .as_object_mut()
        .ok_or_else(|| format!("{} must contain a JSON object", path.display()))?;

    object
        .entry("$schema")
        .or_insert_with(|| json!("https://opencode.ai/config.json"));

    // All three routing aliases are exposed so the mode can be switched from
    // OpenCode's own model picker.
    let mut model_map = Map::new();
    for alias in MODE_ALIASES {
        model_map.insert(
            alias.to_string(),
            json!({ "name": format!("Gemini Router - {alias}") }),
        );
    }
    let provider = json!({
        "npm": "@ai-sdk/openai-compatible",
        "name": "Gemini Router",
        "options": {
            "baseURL": format!("http://127.0.0.1:{port}/v1"),
            "apiKey": token
        },
        "models": model_map
    });
    object
        .entry("provider")
        .or_insert_with(|| json!({}))
        .as_object_mut()
        .ok_or_else(|| format!("{}: `provider` must be an object", path.display()))?
        .insert("gemini-router".into(), provider);
    object.insert("model".into(), json!(format!("gemini-router/{model}")));

    write_json(&path, &root)?;

    let mut warning = String::new();
    if path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| extension.eq_ignore_ascii_case("json"))
        .unwrap_or(false)
    {
        let jsonc = path.with_file_name("opencode.jsonc");
        if jsonc.exists() {
            warning = format!(
                "{} also exists. OpenCode may prefer it over opencode.json; if the provider does \
                 not appear, delete one of the two files or copy the `gemini-router` provider \
                 block into the .jsonc file.",
                jsonc.display()
            );
        }
    }

    Ok((
        path,
        "Injected into OpenCode. Restart OpenCode if it is already running.".to_string(),
        warning,
    ))
}

pub async fn inject_tool(
    state: &Arc<AppState>,
    target: &str,
    mode: Option<Mode>,
) -> Result<InjectResult, String> {
    let token = ensure_token(state)?;
    let port = ensure_server(state).await?;

    let (flash, _lite) = state.chains();
    let fallback_model = flash
        .first()
        .cloned()
        .unwrap_or_else(|| "gemini-3.8-flash".to_string());
    let model = alias_for(mode, &fallback_model);

    let (path, label, message, warning) = match target {
        CLAUDE => {
            let (path, message, warning) = inject_claude(port, &token, mode)?;
            (path, "Claude Code", message, warning)
        }
        CODEX => {
            let (path, message, warning) = inject_codex(port, &token, &model)?;
            (path, "Codex", message, warning)
        }
        OPENCODE => {
            let (path, message, warning) = inject_opencode(port, &token, &model)?;
            (path, "OpenCode", message, warning)
        }
        other => return Err(format!("unknown inject target: {other}")),
    };

    let mode_label = mode.map(Mode::as_str).unwrap_or("current").to_string();
    state.log_event(
        "app",
        state.mode(),
        None,
        log_status::OK,
        None,
        format!(
            "injected {label} with mode {mode_label} -> {}",
            path.display()
        ),
    );
    state.emit_status();

    Ok(InjectResult {
        target: target.to_string(),
        label: label.to_string(),
        mode: mode_label,
        path: path.display().to_string(),
        message,
        warning,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codex_block_replacement_is_idempotent() {
        let first = "model = \"old\"\nmodel_provider = \"old\"\n\n[model_providers.gemini-router]\nbase_url = \"http://old\"\n";
        let stripped = strip_codex_block(first);
        assert!(!stripped.contains("gemini-router"));
        assert!(!stripped.contains("old\""));
    }

    #[test]
    fn codex_keeps_model_keys_inside_other_tables() {
        let mixed = "model = \"old\"\n\n[profiles.fast]\nmodel = \"keep-me\"\n";
        let stripped = strip_codex_block(mixed);
        assert!(!stripped.contains("model = \"old\""));
        assert!(stripped.contains("model = \"keep-me\""));
    }

    #[test]
    fn codex_other_tables_survive() {
        let text = "[other]\nkey = \"value\"\n\n[model_providers.gemini-router]\nbase_url = \"http://old\"\n\n[after]\nx = 1\n";
        let stripped = strip_codex_block(text);
        assert!(stripped.contains("[other]"));
        assert!(stripped.contains("key = \"value\""));
        assert!(stripped.contains("[after]"));
        assert!(stripped.contains("x = 1"));
        assert!(!stripped.contains("gemini-router"));
    }

    #[test]
    fn codex_root_keys_are_written_before_tables() {
        let existing = "model_reasoning_effort = \"medium\"\n\n[windows]\nsandbox = \"elevated\"\n\n[model_providers.gemini-router]\nbase_url = \"http://old\"\nwire_api = \"chat\"\n";
        let text = codex_config_text(existing, 8787, "flash");
        let model = text.find("model = \"flash\"").unwrap();
        let windows = text.find("[windows]").unwrap();
        let provider = text.find("[model_providers.gemini-router]").unwrap();
        assert!(model < windows);
        assert!(windows < provider);
        assert!(text.contains("wire_api = \"responses\""));
        assert!(!text.contains("wire_api = \"chat\""));
        assert!(text.contains("model_context_window = 1000000"));
        // The old provider table is replaced, not duplicated.
        assert_eq!(text.matches("[model_providers.gemini-router]").count(), 1);
    }

    #[test]
    fn blank_lines_collapse() {
        assert_eq!(collapse_blank_lines("a\n\n\n\nb\n\n"), "a\n\nb");
    }

    #[test]
    fn a_pinned_mode_becomes_the_model_alias() {
        assert_eq!(alias_for(Some(Mode::FlashLite), "x"), "flash_lite");
        assert_eq!(alias_for(None, "x"), "x");
    }

    #[test]
    fn toml_strings_are_escaped() {
        assert_eq!(toml_string("a\"b\\c"), "a\\\"b\\\\c");
    }

    #[test]
    fn claude_settings_pin_the_mode_and_keep_unrelated_keys() {
        let mut root = json!({ "theme": "dark", "model": "haiku" });
        apply_claude_settings(&mut root, 8787, "sk-gr-x", Some(Mode::FlashLite)).unwrap();
        // ANTHROPIC_MODEL outranks the settings `model`, so the saved default
        // must survive untouched.
        assert_eq!(root["model"], "haiku");
        assert_eq!(root["theme"], "dark");
        assert_eq!(root["env"]["ANTHROPIC_BASE_URL"], "http://127.0.0.1:8787");
        // A recognized alias avoids the "isn't recognized" startup warning.
        assert_eq!(root["env"]["ANTHROPIC_MODEL"], "sonnet");
        // Gemini models are not in Claude Code's table; tell it the real
        // window so it does not fall back to 200K.
        assert_eq!(root["env"]["CLAUDE_CODE_MAX_CONTEXT_TOKENS"], "1000000");
        assert_eq!(
            root["env"]["ANTHROPIC_CUSTOM_HEADERS"],
            "X-Gemini-Router-Mode: flash_lite"
        );
    }

    #[test]
    fn claude_settings_use_one_auth_variable() {
        let mut root = json!({ "env": { "ANTHROPIC_API_KEY": "old" } });
        apply_claude_settings(&mut root, 1, "sk-gr-x", Some(Mode::Both)).unwrap();
        // Claude Code warns when both are set.
        assert!(root["env"].get("ANTHROPIC_API_KEY").is_none());
        assert_eq!(root["env"]["ANTHROPIC_AUTH_TOKEN"], "sk-gr-x");
    }

    #[test]
    fn claude_settings_follow_the_app_mode_without_a_header() {
        let mut root = json!({
            "env": {
                "ANTHROPIC_CUSTOM_HEADERS": "X-Gemini-Router-Mode: both",
                "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "200000"
            }
        });
        apply_claude_settings(&mut root, 1, "t", None).unwrap();
        assert!(root["env"].get("ANTHROPIC_CUSTOM_HEADERS").is_none());
        // An older injected value is corrected, not preserved.
        assert_eq!(root["env"]["CLAUDE_CODE_MAX_CONTEXT_TOKENS"], "1000000");
    }

    #[test]
    fn claude_settings_drop_old_mode_aliases() {
        let mut root = json!({ "env": {
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": "flash_lite",
            "ANTHROPIC_DEFAULT_OPUS_MODEL": "my-custom-model"
        }});
        apply_claude_settings(&mut root, 1, "t", None).unwrap();
        assert!(root["env"].get("ANTHROPIC_DEFAULT_HAIKU_MODEL").is_none());
        // A user's own custom model must not be touched.
        assert_eq!(
            root["env"]["ANTHROPIC_DEFAULT_OPUS_MODEL"],
            "my-custom-model"
        );
    }

    #[test]
    fn claude_env_is_created_when_missing() {
        let mut root = json!({});
        apply_claude_settings(&mut root, 9, "t", Some(Mode::Flash)).unwrap();
        assert_eq!(root["env"]["ANTHROPIC_AUTH_TOKEN"], "t");
    }

    #[test]
    fn jsonc_comments_and_trailing_commas_are_removed() {
        let input = r#"{
            // line comment
            "url": "http://example.com/path",
            /* block comment */
            "items": [1, 2,],
        }"#;
        let parsed: Value = serde_json::from_str(&strip_jsonc(input)).unwrap();
        assert_eq!(parsed["url"], "http://example.com/path");
        assert_eq!(parsed["items"][1], 2);
    }

    #[test]
    fn opencode_jsonc_is_used_when_it_is_the_only_config() {
        let dir = std::env::temp_dir().join(format!(
            "gemini-router-test-{}",
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        assert_eq!(choose_opencode_path(&dir), dir.join("opencode.json"));

        std::fs::write(dir.join("opencode.jsonc"), "{}").unwrap();
        assert_eq!(choose_opencode_path(&dir), dir.join("opencode.jsonc"));

        // If both exist, the documented `opencode.json` wins.
        std::fs::write(dir.join("opencode.json"), "{}").unwrap();
        assert_eq!(choose_opencode_path(&dir), dir.join("opencode.json"));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
