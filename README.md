# Gemini Router

A Tauri 2 desktop app (Rust backend, plain HTML/CSS/JS frontend) that sends prompts to the
Gemini API, routes them across model chains using a single API key at a time, and doubles as a
local OpenAI-compatible server so coding tools get the same routing.

```
┌────────────────────────────┐        ┌───────────────────────┐
│ Desktop UI (prompt, keys,  │        │ Coding tools          │
│ status, log)               │        │ (opencode, Cline, …)  │
└─────────────┬──────────────┘        └───────────┬───────────┘
              │ Tauri commands                    │ OpenAI HTTP
              ▼                                   ▼
        ┌───────────────────────────────────────────────┐
        │ router.rs — chains, cooldowns, fallback, wait │
        └───────────────────┬───────────────────────────┘
                            ▼
                  generativelanguage.googleapis.com
```

---

## Prerequisites

* **Rust** stable (tested on 1.99, MSVC toolchain) — `rustup` with the `x86_64-pc-windows-msvc` target
* **Node.js** — *not required*; the frontend is plain static files with no build step
* **WebView2** — preinstalled on Windows 11; the Tauri installer also ships it
* Windows users need the MSVC linker (Visual Studio Build Tools) for any Rust build

No C compiler, CMake, or NASM is required — see [TLS backend](#tls-backend) below.

## Run it

```bash
cd src-tauri
cargo run            # dev: opens the app window
cargo test           # unit tests for routing, 429 classification, OpenAI translation
cargo build --release
```

Installers (`msi`/`nsis`) are produced by `cargo tauri build` if the Tauri CLI is installed
(`cargo install tauri-cli --version "^2"`). That is optional; `cargo run` is enough to use the app.

Icons are generated, not committed as binaries:

```bash
python tools/make_icons.py
```

---

## Configuration

Everything lives in the Tauri app data directory:

| Platform | Path |
| --- | --- |
| Windows | `%APPDATA%\dev.gemini-router.desktop\` |
| macOS | `~/Library/Application Support/dev.gemini-router.desktop/` |
| Linux | `~/.local/share/dev.gemini-router.desktop/` |

* `config.json` — the model chains. Exactly the shape from the brief:

  ```json
  {
    "flash_chain": ["gemini-3.8-flash", "gemini-3.6-flash"],
    "flash_lite_chain": ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"],
    "default_mode": "both",
    "server_port": 8787
  }
  ```

  Created with these defaults on first launch, then never overwritten, so hand edits survive.
  `server_port` is only the *initial* port; later changes come from the UI. The Mode area can
  edit both chains and write this file; **Reload config.json** re-reads hand edits without a
  restart. A file with a JSON error is left untouched (defaults apply for that run) so a typo
  cannot destroy your chains.

* `settings.json` — mode, port, key-slot count, access tokens, wait policy, timeouts.
* `keys/slot-N.key` — only created when no OS keychain is available (see below).

## API keys

The key is stored through the `keyring` crate, which on Windows is the Credential Manager
(service `gemini-router`, account `slot-0`, `slot-1`, …). The key area shows
**Key saved**, **No key**, or **Key rejected by API** — the last one comes from a real call to
`models.list`, which does not consume model quota.

* **Never logged.** The key travels in the `x-goog-api-key` header, never a query string, and
  every error string is passed through a redaction step before it is logged or returned.
* **Paste-safe.** Values are normalised before storage: surrounding whitespace, newlines,
  zero-width characters, matching quotes, `export `, `SET `, `GEMINI_API_KEY=`, and `Bearer ` are
  removed. A key with internal whitespace or non-ASCII characters is rejected with an explanation
  instead of being stored, and the field keeps the text so it can be fixed. Saving also verifies
  the key against `models.list` immediately.
* **Shape warning.** A saved key that is not 39 characters starting with `AIza` still works but
  gets a warning, since that is the usual Google API key format.
* **Fallback:** `GEMINI_API_KEY` is read as a last resort for slot 1.
* **Rejected-key failover.** A 401/403 marks only that slot as *rejected* and the request
  continues with the next slot; it does not abort the whole request. A key is still never
  switched in the middle of a single model attempt.
* **No keychain?** The key is written to `keys/slot-N.key` and the permissions are locked down
  (`icacls /inheritance:r /grant:r <user>:F` on Windows, `chmod 600` on Unix). The UI warns you
  explicitly, and warns again if permissions could not be restricted.

## Modes

Switching is instant from the UI or over HTTP (`POST /v1/settings`, or by sending the mode name
as the `model` field). It applies from the next request.

| Mode | Behaviour |
| --- | --- |
| `flash` | Walks `flash_chain` in order. If every model is unavailable, waits for the earliest one and retries. |
| `flash_lite` | Same, on `flash_lite_chain`. |
| `both` | `flash_chain` then `flash_lite_chain`. On reaching the end of `flash_lite_chain` it re-checks `gemini-3.8-flash`: available → restart from the top of `flash_chain`; not available → wait for the earliest model and retry. |

## Routing

* Each request goes to the **first available** model in the active chain.
* **HTTP 429** is classified from the response body:
  * *Per-minute* (RPM/TPM, or a spend-based limit) → cooling down until the `RetryInfo.retryDelay`
    (or `Retry-After`) delay, defaulting to 60 s.
  * *Per-day* (RPD/TPD, a `RESOURCE_EXHAUSTED` body with a daily `quotaId`) → daily-exhausted
    until the next **midnight Pacific Time**, computed with `chrono-tz` so it stays correct
    regardless of your machine's timezone.
* **Transient errors** (5xx, timeouts, connection failures) retry the *same* model with
  exponential backoff plus jitter — 500 ms, 1 s, 2 s, capped at 8 s, 3 attempts total — then fall
  through to the next model.
* **No model available** and waiting is not allowed → the caller gets an error naming the earliest
  time a model becomes available, in local time and RFC 3339 UTC.
* Availability lives in `Arc<Mutex<Registry>>`, keyed by `(key slot, model)`, with expired entries
  swept on every read.

## Connect tools

Start/stop the server, set the port, and manage access tokens in section 5.

* Binds **127.0.0.1 only**.
* Accepts request bodies up to **32 MiB** (coding tools routinely send large conversations).
* The Gemini HTTP client uses a 15 s connect timeout and a per-read timeout
  (`request_timeout_secs`, default 120 s), so long SSE responses are not cut off while tokens
  are still arriving.
* Base URL: `http://127.0.0.1:8787/v1`
* Auth: `Authorization: Bearer <token>` or `x-api-key: <token>`. While the token list is empty,
  requests are accepted without auth.
* **Add key slot** adds one more access token per press. Tokens can be copied, regenerated or removed.

| Endpoint | Notes |
| --- | --- |
| `GET /v1/models` | Chain models, plus `flash` / `flash_lite` / `both` as routing aliases |
| `GET /v1/models/{id}` | One model or routing alias, for tools that probe it first |
| `POST /v1/chat/completions` | Streaming and non-streaming |
| `POST /v1/responses` | OpenAI Responses API used by Codex, streaming and non-streaming |
| `POST /v1/completions` | Legacy, streaming and non-streaming |
| `GET /v1/settings` | Current mode, chains, wait policy |
| `POST /v1/settings` | `{"mode":"flash_lite"}` — changes routing for the next request |
| `GET /health` | Liveness |

Example:

```bash
OPENAI_BASE_URL=http://127.0.0.1:8787/v1
OPENAI_API_KEY=sk-gr-...        # a token from the Connect-tools panel
```

Supported translation: system/developer/user/assistant/tool messages, function tools and
`tool_choice`, `temperature`, `top_p`, `max_tokens`/`max_completion_tokens`, `stop`, SSE
streaming, and `stream_options.include_usage`. Tool calls map both ways, including correlating
`tool_call_id` back to the function name for `functionResponse`.

`/v1/responses` accepts Codex's `input` items (messages, function calls, custom/freeform tool
calls and their outputs), flat and namespaced function tools, and custom tools. Custom tools are
exposed to Gemini as a single-string `input` function and translated back to `custom_tool_call`
items. Hosted tools and reasoning items are ignored. Responses SSE follows the OpenAI event
sequence (`response.created`, `response.output_item.*`, `response.output_text.delta`,
`response.completed`).

Tool parameter schemas are reduced to the JSON Schema subset Gemini accepts, using an
**allowlist** rather than a list of known-bad keys. `propertyNames`, `patternProperties`,
`exclusiveMinimum`/`exclusiveMaximum`, `const`, `oneOf`, `$ref`, `additionalProperties` and other
draft-07 keywords are converted where possible (`oneOf` → `anyOf`, `const` → one-value `enum`,
exclusive bounds → inclusive bounds) and dropped otherwise, so an arbitrary MCP schema cannot
produce Gemini's `400 Invalid JSON payload`.

Ignored: `n` > 1, logprobs, logit_bias, response_format, seed.

By default the server does **not** block waiting for a model — it returns `429` with the earliest
availability in the message. Tick *Let the server wait* to make it block instead.

## Inject into a coding tool

Section 5 has an **Inject into a coding tool** panel with one button per tool. Each press starts
the local server and creates an access token if needed, then writes the tool's user-level config:

| Tool | File written | How the mode is pinned |
| --- | --- | --- |
| Claude Code | `~/.claude/settings.json` (`env`) | `ANTHROPIC_CUSTOM_HEADERS: X-Gemini-Router-Mode` |
| Codex | `~/.codex/config.toml` | `model = "…"` under `[model_providers.gemini-router]` |
| OpenCode | `~/.config/opencode/opencode.json` or `opencode.jsonc` | `model = "gemini-router/<alias>"` |

The **Mode** dropdown chooses what gets written:

* `flash` / `flash_lite` / `both` — the tool sends the mode name as its model, which the server
  already accepts as a **per-request mode override**. The app's own global mode is untouched, so
  two tools can run on two different chains at once.
* `current app mode` — a concrete chain model is written instead, so the tool follows whatever
  mode the app is set to. Changing the app's mode then changes the tool's routing too.

Notes:

* Claude Code's base URL has no `/v1` suffix because it appends `/v1/messages` itself; the other
  two use `http://127.0.0.1:<port>/v1`.
* Claude Code gets `ANTHROPIC_MODEL=sonnet` / `ANTHROPIC_SMALL_FAST_MODEL=haiku`, and the mode
  rides in `ANTHROPIC_CUSTOM_HEADERS`, which the server checks before the `model` field. Only
  `ANTHROPIC_AUTH_TOKEN` is set — setting both auth variables makes Claude Code warn that auth may
  not work. `CLAUDE_CODE_MAX_CONTEXT_TOKENS=1000000` tells Claude Code the real window for the
  Gemini model IDs (which are not in its model table), so it neither assumes 200K nor prints the
  "isn't recognized" warning. The saved `model` in `settings.json` is left untouched, and mode
  aliases written by older injections are removed.
* Codex gets `GEMINI_ROUTER_API_KEY` through `setx` on Windows. The provider uses
  `wire_api = "responses"` (Codex no longer accepts `"chat"`), and `model_context_window =
  1000000` tells Codex the Gemini models' real window when its model catalog has no entry.
* OpenCode exposes all three aliases in its model picker, so the mode can be switched there too.
  If only `opencode.jsonc` exists, that file is used instead of creating an `opencode.json`.
  JSONC comments and trailing commas are accepted when reading; the rewritten file is plain JSON
  (still valid JSONC), so comments are not preserved. If both files exist, `opencode.json` wins and
  the panel warns you.
* Re-injecting is idempotent: only the `gemini-router` provider block and the root `model` /
  `model_provider` / `model_context_window` keys are replaced, and other tables in `config.toml`
  survive. If a file is not valid JSON it is left untouched rather than overwritten.
* The access token is written into these files, so they are user-only (`chmod 600` on Unix). The
  token is never written to the request log.
* Restart the tool after injecting. Codex also needs a new terminal to see the environment variable.

## Request log

One row per event with timestamp, source, mode, model, status, latency and detail. Fallbacks and
restarts each get their own row, for example:

```
14:02:11  opencode  both        gemini-3.8-flash  fallback   —  429 (per-minute quota exhausted, retry after 37s) …
14:02:11  opencode  both        gemini-3.8-flash  restart    —  reached the end of flash_lite_chain and gemini-3.8-flash is available again …
14:02:12  opencode  both        gemini-3.8-flash  ok        812 ms  handled by gemini-3.8-flash (key slot 1)
```

**Source** is the tool name when the server can identify it — from `X-Source` / `X-Client-Name` /
`X-Tool` / `X-Title`, otherwise a known User-Agent substring, otherwise the UA product token.
The desktop UI logs as `app`. The brief said unidentified requests should log as `app`; that would
conflate the two origins, so unidentified server clients log as `unknown-client` or their product
token instead. The log keeps the newest 500 rows.

---

## Deviations and notes

1. **TLS backend.** `reqwest` uses `native-tls` (schannel on Windows) rather than `rustls`.
   The `rustls` feature pulls in `aws-lc-sys`, which requires CMake **and** NASM to assemble on
   Windows; this machine has neither. OS-native TLS needs no extra toolchain. If you prefer
   bundled roots, install NASM + CMake and switch the feature in `src-tauri/Cargo.toml`.
2. **keyring 4.x features.** Pinned to `v1,windows-native-keyring-store` with default features off,
   which keeps the D-Bus stack out of Windows builds. For macOS or Linux add
   `apple-native-keyring-store` or `zbus-secret-service-keyring-store`; without them the app falls
   back to the restricted file, as designed.
3. **Restart loop is bounded.** The `both` restart rule could in principle spin, so it is capped by
   `max_restarts` (default 8, in `settings.json`) and the cap being hit is logged.
4. **Unknown models are parked, not retried.** A `404`, or a `400` that says the model is not
   found/supported, cools that model for an hour and falls through, instead of failing the request.
   This is beyond the brief but stops a bad chain entry from breaking every request.
5. **Per-request mode override.** Sending `flash`, `flash_lite` or `both` as the `model` field
   (or the `X-Gemini-Router-Mode` header) routes for that request only. Extra convenience; the
   global mode is unchanged.
6. **CSP is disabled** (`"csp": null`, the Tauri default) to avoid blank-window failures. Since
   the frontend is local static files with no remote content, tightening this is safe if you want.
7. **`n > 1`, logprobs, etc. are ignored** rather than rejected — most coding tools never send them.

## Layout

```
gemini-router/
├─ src/                        frontend (index.html, styles.css, main.js) — no build step
├─ src-tauri/
│  ├─ Cargo.toml
│  ├─ tauri.conf.json
│  ├─ capabilities/default.json
│  ├─ icons/                   generated by tools/make_icons.py
│  └─ src/
│     ├─ main.rs               Tauri setup + command registration
│     ├─ commands.rs           the command surface the UI calls
│     ├─ config.rs             config.json / settings.json, Pacific midnight
│     ├─ inject.rs             one-click tool config injection ← per-mode
│     ├─ keystore.rs           keyring + restricted-file fallback, redaction
│     ├─ gemini.rs             API client, 429 classification
│     ├─ router.rs             chains, cooldowns, fallback, wait  ← the core
│     ├─ server.rs             axum OpenAI-compatible server
│     ├─ openai.rs             OpenAI ⇄ Gemini translation
│     ├─ state.rs              AppState + per-(slot, model) registry
│     └─ logging.rs            log entries
└─ tools/make_icons.py
```

## Tests

`cargo test` covers the parts where a mistake is silent and expensive: 429 per-minute vs per-day
classification (including mixed bodies and `RetryInfo` parsing), `retryDelay` formats, status →
error-variant mapping, key redaction, registry cooldown/exhaustion/sweep semantics, Pacific-midnight
arithmetic, OpenAI ⇄ Gemini message and tool-call translation (including parallel tool results and
consecutive same-role turns), SSE chunk translation with finish state carried across chunks, and
that a malformed `config.json` is never overwritten.
