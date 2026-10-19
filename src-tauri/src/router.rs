//! Request routing across the model chains.
//!
//! One implementation serves both the desktop UI and the local
//! OpenAI-compatible server: [`route`] is generic over the actual API call, so
//! streaming and non-streaming requests follow exactly the same mode rules,
//! cooldown bookkeeping and fallback logging.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::{DateTime, Local, Utc};
use serde_json::Value;
use thiserror::Error;

use crate::config::{Mode, next_midnight_pacific};
use crate::gemini::{self, GeminiError, LimitClass};
use crate::keystore::KeyStatus;
use crate::logging::status as log_status;
use crate::state::{AppState, lock};

/// Boxed future returned by the per-attempt call closure.
///
/// The closure owns everything it needs (the shared state is cloned into it), so
/// the future it produces is always `'static`.
pub type CallFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T, GeminiError>> + Send + 'a>>;

#[derive(Debug, Error)]
pub enum RouteError {
    #[error("no Gemini API key is configured. Add one in the Key area.")]
    NoKey,
    #[error("the stored Gemini API key was rejected by the API")]
    KeyRejectedStored,
    #[error("the Gemini API rejected this key: {0}")]
    KeyRejected(String),
    #[error(
        "no model in the {mode} chain is available, and this request may not wait. \
         The earliest one becomes available at {local} (local time) / {utc} (UTC)."
    )]
    NoModelAvailable {
        mode: String,
        local: String,
        utc: String,
    },
    #[error("every model in the {mode} chain failed: {detail}")]
    AllFailed { mode: String, detail: String },
    #[error("{0}")]
    Request(String),
}

pub struct RouteOutcome<T> {
    pub model: String,
    pub key_slot: u32,
    pub value: T,
    pub latency_ms: u64,
    pub restarts: u32,
    pub fallbacks: u32,
}

/// Exponential backoff with jitter for transient (5xx / network) failures.
fn backoff_delay(attempt: u32) -> Duration {
    let shift = attempt.saturating_sub(1).min(4);
    let base = 500u64.saturating_mul(1u64 << shift).min(8_000);
    // Cheap, dependency-free jitter: sub-second clock noise.
    let jitter = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| (d.subsec_nanos() as u64) % 250)
        .unwrap_or(0);
    Duration::from_millis(base + jitter)
}

fn fmt_local(t: DateTime<Utc>) -> String {
    t.with_timezone(&Local)
        .format("%Y-%m-%d %H:%M:%S")
        .to_string()
}

/// Route one request through the active chain.
///
/// `call` performs a single attempt against a concrete (key, model) pair. It is
/// retried in place for transient failures, then the router falls through to the
/// next model.
pub async fn route<T: Send + 'static>(
    state: &Arc<AppState>,
    mode: Mode,
    source: &str,
    allow_wait: bool,
    mut call: impl FnMut(String, String) -> CallFuture<'static, T>,
) -> Result<RouteOutcome<T>, RouteError> {
    let settings = lock(&state.settings).clone();
    let (flash, _lite) = state.chains();
    let chain = state.chain_for(mode);

    if chain.is_empty() {
        return Err(RouteError::Request("no models configured".into()));
    }

    let wait_budget = if allow_wait {
        if source == "app" {
            settings.ui_max_wait_secs
        } else {
            settings.api_max_wait_secs
        }
    } else {
        0
    };
    let deadline = Instant::now() + Duration::from_secs(wait_budget);

    let started = Instant::now();
    let attempts = settings.transient_attempts.max(1);
    let mut restarts = 0u32;
    let mut fallbacks = 0u32;
    let mut last_failure: Option<String> = None;
    let mut last_auth_failure: Option<String> = None;

    'outer: loop {
        // Re-read the usable slots on every pass: a rejected key must not be
        // retried, and a wait may have lasted long enough for a slot to be
        // cleared by the user. Multiple keys are failover only after a key has
        // nothing left to try; within a single model attempt we never hop keys.
        let active_slots = lock(&state.keys).usable_slots();
        if active_slots.is_empty() {
            let any_present = lock(&state.keys).slots.iter().any(|s| s.present());
            return Err(if any_present {
                RouteError::KeyRejectedStored
            } else {
                RouteError::NoKey
            });
        }

        for slot in active_slots.iter().copied() {
            let key = match state.key_for_slot(slot) {
                Some(k) => k,
                None => continue,
            };

            let mut index = 0usize;
            loop {
                // ---- end of the active chain for this key ------------------
                if index >= chain.len() {
                    if mode == Mode::Both && !flash.is_empty() {
                        let top = flash[0].clone();
                        let available = {
                            let mut registry = lock(&state.registry);
                            let now = Utc::now();
                            registry.sweep(now);
                            registry.is_available(slot, &top, now)
                        };
                        if available {
                            restarts += 1;
                            if restarts <= settings.max_restarts {
                                state.log_event(
                                    source,
                                    mode,
                                    Some(&top),
                                    log_status::RESTART,
                                    None,
                                    format!(
                                        "reached the end of flash_lite_chain and {top} is available \
                                         again — restarting from the top of flash_chain (restart #{restarts})"
                                    ),
                                );
                                index = 0;
                                continue;
                            }
                            state.log_event(
                                source,
                                mode,
                                Some(&top),
                                log_status::RESTART,
                                None,
                                format!(
                                    "restart limit ({}) reached; not restarting again",
                                    settings.max_restarts
                                ),
                            );
                        }
                    }
                    break;
                }

                // ---- pick the first available model -----------------------
                let model = chain[index].clone();
                let available = {
                    let mut registry = lock(&state.registry);
                    let now = Utc::now();
                    registry.sweep(now);
                    registry.is_available(slot, &model, now)
                };
                if !available {
                    index += 1;
                    continue;
                }

                // ---- attempt, with transient retries ----------------------
                let mut outcome: Result<T, GeminiError> = Err(GeminiError::Transient {
                    detail: "no attempt was made".into(),
                });
                let mut attempts_used = 0u32;
                for attempt in 1..=attempts {
                    attempts_used = attempt;
                    let attempt_started = Instant::now();
                    match call(key.clone(), model.clone()).await {
                        Ok(value) => {
                            outcome = Ok(value);
                            break;
                        }
                        Err(GeminiError::Transient { detail }) => {
                            state.log_event(
                                source,
                                mode,
                                Some(&model),
                                log_status::RETRY,
                                Some(attempt_started.elapsed().as_millis() as u64),
                                format!(
                                    "transient failure (attempt {attempt}/{attempts}): {detail}"
                                ),
                            );
                            outcome = Err(GeminiError::Transient { detail });
                            if attempt < attempts {
                                tokio::time::sleep(backoff_delay(attempt)).await;
                            }
                        }
                        Err(other) => {
                            outcome = Err(other);
                            break;
                        }
                    }
                }

                match outcome {
                    Ok(value) => {
                        lock(&state.registry).clear(slot, &model);
                        let latency_ms = started.elapsed().as_millis() as u64;
                        let detail = if attempts_used > 1 {
                            format!(
                                "handled by {model} (key slot {}) after {attempts_used} attempts",
                                slot + 1
                            )
                        } else {
                            format!("handled by {model} (key slot {})", slot + 1)
                        };
                        state.log_event(
                            source,
                            mode,
                            Some(&model),
                            log_status::OK,
                            Some(latency_ms),
                            detail,
                        );
                        return Ok(RouteOutcome {
                            model,
                            key_slot: slot,
                            value,
                            latency_ms,
                            restarts,
                            fallbacks,
                        });
                    }

                    Err(GeminiError::RateLimited {
                        class,
                        retry_after,
                        detail,
                    }) => {
                        let (until, kind, reason) = match class {
                            LimitClass::PerDay => (
                                next_midnight_pacific(),
                                "daily quota exhausted",
                                "daily quota (RPD/TPD) exhausted",
                            ),
                            LimitClass::PerMinute => (
                                Utc::now()
                                    + chrono::Duration::from_std(retry_after)
                                        .unwrap_or_else(|_| chrono::Duration::seconds(60)),
                                "per-minute quota exhausted",
                                "per-minute quota (RPM/TPM) exhausted",
                            ),
                        };
                        {
                            let mut registry = lock(&state.registry);
                            match class {
                                LimitClass::PerDay => {
                                    registry.mark_exhausted(slot, &model, until, reason.to_string())
                                }
                                LimitClass::PerMinute => {
                                    registry.mark_cooling(slot, &model, until, reason.to_string())
                                }
                            }
                        }
                        fallbacks += 1;
                        let blocked_until = match class {
                            LimitClass::PerDay => {
                                format!("daily-exhausted until {}", fmt_local(until))
                            }
                            LimitClass::PerMinute => format!(
                                "cooling down until {}",
                                until.with_timezone(&Local).format("%H:%M:%S")
                            ),
                        };
                        state.log_event(
                            source,
                            mode,
                            Some(&model),
                            log_status::FALLBACK,
                            None,
                            format!(
                                "429 ({kind}, retry after {}s) — {detail}; {blocked_until}; falling through \
                                 to the next model",
                                retry_after.as_secs()
                            ),
                        );
                        last_failure = Some(detail);
                        index += 1;
                        continue;
                    }

                    Err(GeminiError::Unavailable { detail }) => {
                        // The model is in the chain but cannot serve requests (unknown ID,
                        // region, unsupported). Park it for an hour rather than hammering it.
                        let until = Utc::now() + chrono::Duration::hours(1);
                        lock(&state.registry).mark_cooling(
                            slot,
                            &model,
                            until,
                            "model unavailable".to_string(),
                        );
                        fallbacks += 1;
                        last_failure = Some(detail.clone());
                        state.log_event(
                            source,
                            mode,
                            Some(&model),
                            log_status::FALLBACK,
                            None,
                            format!("{detail}; parking it for an hour and falling through"),
                        );
                        index += 1;
                        continue;
                    }

                    Err(GeminiError::Transient { detail }) => {
                        fallbacks += 1;
                        let text = format!(
                            "transient failures on all {attempts} attempts: {detail}; falling through"
                        );
                        state.log_event(
                            source,
                            mode,
                            Some(&model),
                            log_status::FALLBACK,
                            None,
                            text.clone(),
                        );
                        last_failure = Some(text);
                        index += 1;
                        continue;
                    }

                    Err(GeminiError::Auth { detail }) => {
                        // Do not abort the whole request: another slot may hold a
                        // key from a different project. Mark this one unusable,
                        // then move on to the next slot.
                        lock(&state.keys).set_status(slot, KeyStatus::Rejected);
                        state.emit_status();
                        state.log_event(
                            source,
                            mode,
                            Some(&model),
                            log_status::ERROR,
                            None,
                            format!(
                                "key slot {} was rejected by the API: {detail}; trying the next key slot",
                                slot + 1
                            ),
                        );
                        last_auth_failure = Some(detail);
                        break;
                    }

                    Err(GeminiError::Client { status, detail }) => {
                        state.log_event(
                            source,
                            mode,
                            Some(&model),
                            log_status::ERROR,
                            None,
                            format!("request rejected with HTTP {status}: {detail}"),
                        );
                        return Err(RouteError::Request(format!("HTTP {status}: {detail}")));
                    }
                }
            }
        }

        // ---- nothing available anywhere -----------------------------------
        let now = Utc::now();
        let earliest = lock(&state.registry).earliest_available_any(&active_slots, &chain, now);

        if allow_wait {
            if let Some(when) = earliest {
                let remaining = deadline.saturating_duration_since(Instant::now());
                if !remaining.is_zero() {
                    let sleep_for = (when - now)
                        .to_std()
                        .unwrap_or_else(|_| Duration::from_secs(1))
                        .min(Duration::from_secs(30))
                        .min(remaining);
                    if !sleep_for.is_zero() {
                        state.log_event(
                            source,
                            mode,
                            None,
                            log_status::WAIT,
                            None,
                            format!(
                                "every model in the {} chain is unavailable; waiting {}s until {}",
                                mode.as_str(),
                                sleep_for.as_secs(),
                                fmt_local(when)
                            ),
                        );
                        tokio::time::sleep(sleep_for).await;
                        continue 'outer;
                    }
                }
            }
        }

        // Nothing is blocked but everything failed: report the underlying cause.
        if earliest.is_none() {
            if let Some(detail) = last_failure {
                return Err(RouteError::AllFailed {
                    mode: mode.as_str().to_string(),
                    detail,
                });
            }
            if let Some(detail) = last_auth_failure {
                return Err(RouteError::KeyRejected(detail));
            }
        }

        let (local, utc) = match earliest {
            Some(t) => (fmt_local(t), t.to_rfc3339()),
            None => ("unknown".to_string(), "unknown".to_string()),
        };
        state.log_event(
            source,
            mode,
            None,
            log_status::ERROR,
            None,
            format!(
                "no model available in the {} chain; earliest availability {local}",
                mode.as_str()
            ),
        );
        return Err(RouteError::NoModelAvailable {
            mode: mode.as_str().to_string(),
            local,
            utc,
        });
    }
}

/// Non-streaming completion using the shared router.
pub async fn complete(
    state: &Arc<AppState>,
    mode: Mode,
    source: &str,
    body: Value,
    allow_wait: bool,
) -> Result<RouteOutcome<Value>, RouteError> {
    let settings = lock(&state.settings).clone();
    route(state, mode, source, allow_wait, move |key, model| {
        let state = state.clone();
        let body = body.clone();
        let settings = settings.clone();
        Box::pin(async move { gemini::generate(&state.http, &key, &model, &body, &settings).await })
    })
    .await
}

/// Streaming completion. The returned response is already status 200, so the
/// caller owns the SSE body.
pub async fn complete_stream(
    state: &Arc<AppState>,
    mode: Mode,
    source: &str,
    body: Value,
    allow_wait: bool,
) -> Result<RouteOutcome<reqwest::Response>, RouteError> {
    let settings = lock(&state.settings).clone();
    route(state, mode, source, allow_wait, move |key, model| {
        let state = state.clone();
        let body = body.clone();
        let settings = settings.clone();
        Box::pin(async move {
            gemini::generate_stream(&state.http, &key, &model, &body, &settings).await
        })
    })
    .await
}

/// Concatenate the text parts of the first candidate.
pub fn extract_text(value: &Value) -> String {
    value
        .pointer("/candidates/0/content/parts")
        .and_then(|p| p.as_array())
        .map(|parts| {
            parts
                .iter()
                .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
                .collect::<Vec<_>>()
                .join("")
        })
        .unwrap_or_default()
}

/// Build a minimal Gemini request body for the desktop prompt box.
pub fn prompt_body(prompt: &str) -> Value {
    serde_json::json!({
        "contents": [{
            "role": "user",
            "parts": [{ "text": prompt }]
        }]
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backoff_grows_then_caps() {
        for attempt in 1..=6u32 {
            let d = backoff_delay(attempt);
            assert!(
                d >= Duration::from_millis(400),
                "attempt {attempt} gave {d:?}"
            );
            assert!(
                d <= Duration::from_millis(8_250),
                "attempt {attempt} gave {d:?}"
            );
        }
        // The last step must be capped rather than growing without bound.
        assert!(backoff_delay(50) <= Duration::from_millis(8_250));
    }

    #[test]
    fn extracts_text_parts_and_ignores_tool_calls() {
        let value = serde_json::json!({
            "candidates": [{
                "content": { "parts": [
                    { "text": "hello " },
                    { "functionCall": { "name": "f", "args": {} } },
                    { "text": "world" }
                ]}
            }]
        });
        assert_eq!(extract_text(&value), "hello world");
    }

    #[test]
    fn extract_text_is_empty_when_there_is_no_candidate() {
        assert_eq!(extract_text(&serde_json::json!({})), "");
    }

    #[test]
    fn prompt_body_has_a_single_user_turn() {
        let body = prompt_body("hi");
        assert_eq!(body["contents"][0]["role"], "user");
        assert_eq!(body["contents"][0]["parts"][0]["text"], "hi");
    }
}
