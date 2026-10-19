//! Gemini API client: request sending plus 429 classification.
//!
//! The API key is always sent in the `x-goog-api-key` header (never a query
//! string) so it cannot leak into URLs, and it is stripped from every error
//! message before the message leaves this module.

use std::time::Duration;

use serde_json::Value;
use thiserror::Error;

pub const API_BASE: &str = "https://generativelanguage.googleapis.com/v1beta";

/// Which quota a 429 hit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LimitClass {
    /// RPM / TPM (or a spend-based limit): short cooldown, then retry.
    PerMinute,
    /// RPD / TPD: blocked until the next midnight Pacific.
    PerDay,
}

#[derive(Debug, Error)]
pub enum GeminiError {
    #[error("rate limited: {detail}")]
    RateLimited {
        class: LimitClass,
        retry_after: Duration,
        detail: String,
    },
    /// 5xx or a transport failure: retry the same model with backoff.
    #[error("transient failure: {detail}")]
    Transient { detail: String },
    /// 401/403: the key itself is the problem.
    #[error("key rejected by API: {detail}")]
    Auth { detail: String },
    /// The model exists in config but the API says it cannot serve requests
    /// (unknown model, unsupported, not available in this region).
    #[error("model unavailable: {detail}")]
    Unavailable { detail: String },
    /// Anything else in the 4xx range: a bad request that retrying will not fix.
    #[error("request rejected ({status}): {detail}")]
    Client { status: u16, detail: String },
}

/// Shorten an API error body so logs stay readable.
fn summarise(body: &str, key: &str) -> String {
    let no_key = crate::keystore::redact(body, Some(key));
    let flattened: String = no_key.split_whitespace().collect::<Vec<_>>().join(" ");
    let text = if let Ok(json) = serde_json::from_str::<Value>(&no_key) {
        json.pointer("/error/message")
            .and_then(|m| m.as_str())
            .map(|m| m.to_string())
            .unwrap_or(flattened.clone())
    } else {
        flattened
    };
    if text.chars().count() > 400 {
        format!("{}…", text.chars().take(400).collect::<String>())
    } else {
        text
    }
}

fn transport_detail(err: reqwest::Error, key: &str) -> String {
    let mut parts: Vec<String> = Vec::new();
    if err.is_timeout() {
        parts.push("timed out".into());
    }
    if err.is_connect() {
        parts.push("connection failed".into());
    }
    if err.is_decode() {
        parts.push("invalid response body".into());
    }
    let base = if parts.is_empty() {
        err.to_string()
    } else {
        format!("{} ({err})", parts.join(", "))
    };
    crate::keystore::redact(&base, Some(key))
}

/// Parse a Google `retryDelay` / `Retry-After` style duration such as `37s`,
/// `1.5s`, or a bare number of seconds.
fn parse_retry_delay(raw: &str) -> Option<f64> {
    let s = raw.trim();
    if s.is_empty() {
        return None;
    }
    // Proto3 JSON durations look like "37s" / "1.5s"; also accept "37" and "PT37S".
    let numeric = s
        .trim_end_matches('s')
        .trim_end_matches('S')
        .trim_start_matches("PT")
        .trim_start_matches("pt");
    numeric
        .parse::<f64>()
        .ok()
        .filter(|v| v.is_finite() && *v >= 0.0)
}

/// Walk the error JSON collecting `(key, string-value)` pairs.
fn collect_strings<'a>(value: &'a Value, out: &mut Vec<(String, String)>) {
    match value {
        Value::Object(map) => {
            for (k, v) in map {
                match v {
                    Value::String(s) => out.push((k.clone(), s.clone())),
                    Value::Number(n) => out.push((k.clone(), n.to_string())),
                    _ => collect_strings(v, out),
                }
            }
        }
        Value::Array(items) => {
            for item in items {
                collect_strings(item, out);
            }
        }
        _ => {}
    }
}

/// Classify a 429 body into per-minute vs per-day and extract the retry delay.
pub fn classify_429(
    body: &str,
    retry_after_header: Option<u64>,
    default_cooldown: Duration,
) -> (LimitClass, Duration, String) {
    let mut pairs: Vec<(String, String)> = Vec::new();
    if let Ok(json) = serde_json::from_str::<Value>(body) {
        collect_strings(&json, &mut pairs);
    } else {
        pairs.push(("body".into(), body.to_string()));
    }

    // Retry delay: RetryInfo.details[].retryDelay, else the Retry-After header.
    let retry_delay = pairs
        .iter()
        .filter(|(k, _)| {
            k.eq_ignore_ascii_case("retryDelay") || k.eq_ignore_ascii_case("retry_after")
        })
        .find_map(|(_, v)| parse_retry_delay(v))
        .or_else(|| retry_after_header.map(|s| s as f64))
        .map(Duration::from_secs_f64)
        .unwrap_or(default_cooldown);

    // Daily quota markers. Google uses quota IDs such as
    // "GenerateRequestsPerDayPerProjectPerModel-FreeTier" and metrics like
    // "generativelanguage.googleapis.com/generate_content_free_tier_requests".
    const DAILY_MARKERS: [&str; 9] = [
        "perday",
        "per day",
        "per_day",
        "daily",
        "requests per day",
        "tokens per day",
        "rpd",
        "tpd",
        "dailyquota",
    ];
    const MINUTE_MARKERS: [&str; 8] = [
        "perminute",
        "per minute",
        "per_minute",
        "perminutepers", // e.g. GenerateRequestsPerMinutePerProjectPerModel
        "rpm",
        "tpm",
        "requestsperminute",
        "tokensperminute",
    ];

    let quota_text: String = pairs
        .iter()
        .filter(|(k, _)| {
            let k = k.to_ascii_lowercase();
            k.contains("quota") || k.contains("violation") || k.contains("metric") || k == "message"
        })
        .map(|(_, v)| v.to_ascii_lowercase())
        .collect::<Vec<_>>()
        .join(" | ");

    let is_daily = DAILY_MARKERS.iter().any(|m| quota_text.contains(m));
    let is_minutely = MINUTE_MARKERS.iter().any(|m| quota_text.contains(m));

    // A daily marker wins: being exhausted for the day dominates a short wait.
    let class = if is_daily {
        LimitClass::PerDay
    } else {
        let _ = is_minutely;
        LimitClass::PerMinute
    };

    let detail = if quota_text.is_empty() {
        "429 RESOURCE_EXHAUSTED".to_string()
    } else {
        let short = quota_text.chars().take(240).collect::<String>();
        format!("429 RESOURCE_EXHAUSTED ({short})")
    };

    (class, retry_delay, detail)
}

/// Map a non-200 status to a typed error.
fn classify_status(
    status: u16,
    body: &str,
    retry_after_header: Option<u64>,
    key: &str,
    cfg: &crate::config::Settings,
) -> GeminiError {
    let detail = summarise(body, key);
    match status {
        429 => {
            let (class, retry_after, why) = classify_429(
                body,
                retry_after_header,
                Duration::from_secs(cfg.default_cooldown_secs.max(1)),
            );
            let why = crate::keystore::redact(&why, Some(key));
            GeminiError::RateLimited {
                class,
                retry_after,
                detail: format!("{detail} — {why}"),
            }
        }
        401 | 403 => GeminiError::Auth { detail },
        404 => GeminiError::Unavailable {
            detail: format!("HTTP 404 — {detail}"),
        },
        400 => {
            let lower = detail.to_ascii_lowercase();
            let looks_like_model_problem = lower.contains("not found")
                || lower.contains("not supported")
                || lower.contains("does not exist")
                || lower.contains("unsupported")
                || lower.contains("is not available");
            if looks_like_model_problem {
                GeminiError::Unavailable { detail }
            } else {
                GeminiError::Client { status, detail }
            }
        }
        500..=599 => GeminiError::Transient {
            detail: format!("HTTP {status} — {detail}"),
        },
        _ => GeminiError::Client { status, detail },
    }
}

fn endpoint(model: &str, method: &str, stream: bool) -> String {
    if stream {
        format!("{API_BASE}/models/{model}:{method}?alt=sse")
    } else {
        format!("{API_BASE}/models/{model}:{method}")
    }
}

/// Send a non-streaming `generateContent` request.
pub async fn generate(
    http: &reqwest::Client,
    key: &str,
    model: &str,
    body: &Value,
    settings: &crate::config::Settings,
) -> Result<Value, GeminiError> {
    let response = http
        .post(endpoint(model, "generateContent", false))
        .header("x-goog-api-key", key)
        .json(body)
        .send()
        .await
        .map_err(|e| GeminiError::Transient {
            detail: transport_detail(e, key),
        })?;

    let status = response.status().as_u16();
    let retry_after = response
        .headers()
        .get("retry-after")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.trim().parse::<u64>().ok());

    let text = response.text().await.map_err(|e| GeminiError::Transient {
        detail: transport_detail(e, key),
    })?;

    if status == 200 {
        serde_json::from_str::<Value>(&text).map_err(|e| GeminiError::Client {
            status,
            detail: crate::keystore::redact(&format!("invalid JSON from API: {e}"), Some(key)),
        })
    } else {
        Err(classify_status(status, &text, retry_after, key, settings))
    }
}

/// Open a streaming `streamGenerateContent` request (SSE). The caller owns the
/// response and is responsible for reading the body.
pub async fn generate_stream(
    http: &reqwest::Client,
    key: &str,
    model: &str,
    body: &Value,
    settings: &crate::config::Settings,
) -> Result<reqwest::Response, GeminiError> {
    let response = http
        .post(endpoint(model, "streamGenerateContent", true))
        .header("x-goog-api-key", key)
        .json(body)
        .send()
        .await
        .map_err(|e| GeminiError::Transient {
            detail: transport_detail(e, key),
        })?;

    let status = response.status().as_u16();
    if status == 200 {
        return Ok(response);
    }

    let retry_after = response
        .headers()
        .get("retry-after")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.trim().parse::<u64>().ok());
    let text = response.text().await.unwrap_or_default();
    Err(classify_status(status, &text, retry_after, key, settings))
}

/// Outcome of a key validation call.
#[derive(Debug)]
pub enum KeyValidation {
    Valid,
    /// The API rejected the key itself (401/403, or a 400 about the API key).
    Rejected(String),
    /// The API could not be reached, so validity is unknown.
    Unreachable(String),
}

/// Validate a key without consuming model quota (`models.list`).
pub async fn validate_key(http: &reqwest::Client, key: &str) -> KeyValidation {
    let response = match http
        .get(format!("{API_BASE}/models?pageSize=1"))
        .header("x-goog-api-key", key)
        .send()
        .await
    {
        Ok(response) => response,
        Err(e) => return KeyValidation::Unreachable(transport_detail(e, key)),
    };

    let status = response.status().as_u16();
    let text = response.text().await.unwrap_or_default();
    let detail = summarise(&text, key);
    match status {
        200 => KeyValidation::Valid,
        401 | 403 => KeyValidation::Rejected(detail),
        400 if detail.to_ascii_lowercase().contains("api key") => KeyValidation::Rejected(detail),
        _ => KeyValidation::Unreachable(format!("HTTP {status}: {detail}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Settings;

    fn settings() -> Settings {
        Settings::default()
    }

    #[test]
    fn parses_proto_duration_suffixes() {
        assert_eq!(parse_retry_delay("37s"), Some(37.0));
        assert_eq!(parse_retry_delay("1.5s"), Some(1.5));
        assert_eq!(parse_retry_delay("42"), Some(42.0));
        assert_eq!(parse_retry_delay("PT37S"), Some(37.0));
        assert_eq!(parse_retry_delay("nonsense"), None);
        assert_eq!(parse_retry_delay(""), None);
    }

    #[test]
    fn rpm_429_uses_retry_info_delay() {
        let body = r#"{
          "error": {
            "code": 429,
            "status": "RESOURCE_EXHAUSTED",
            "message": "Quota exceeded for quota metric 'Generate requests per minute'",
            "details": [
              {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "37s"},
              {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [
                 {"quotaMetric": "generativelanguage.googleapis.com/generate_content_requests",
                  "quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"}
              ]}
            ]
          }
        }"#;
        let (class, delay, detail) = classify_429(body, None, Duration::from_secs(60));
        assert_eq!(class, LimitClass::PerMinute);
        assert_eq!(delay, Duration::from_secs(37));
        assert!(detail.contains("RESOURCE_EXHAUSTED"));
    }

    #[test]
    fn rpm_429_defaults_to_60s_without_retry_info() {
        let body =
            r#"{"error":{"code":429,"status":"RESOURCE_EXHAUSTED","message":"Per minute quota"}}"#;
        let (class, delay, _) = classify_429(body, None, Duration::from_secs(60));
        assert_eq!(class, LimitClass::PerMinute);
        assert_eq!(delay, Duration::from_secs(60));
    }

    #[test]
    fn rpd_429_is_per_day() {
        let body = r#"{
          "error": {
            "code": 429,
            "status": "RESOURCE_EXHAUSTED",
            "message": "Quota exceeded for quota metric 'Generate requests per day per project per model'",
            "details": [
              {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [
                 {"quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
                  "quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
                  "quotaValue": "50"}
              ]}
            ]
          }
        }"#;
        let (class, _, _) = classify_429(body, None, Duration::from_secs(60));
        assert_eq!(class, LimitClass::PerDay);
    }

    #[test]
    fn retry_after_header_is_a_fallback() {
        let body = r#"{"error":{"code":429,"message":"slow down"}}"#;
        let (_, delay, _) = classify_429(body, Some(12), Duration::from_secs(60));
        assert_eq!(delay, Duration::from_secs(12));
    }

    #[test]
    fn daily_marker_beats_minute_marker() {
        // Mixed bodies must not be misread as a 60s cooldown.
        let body = r#"{"error":{"message":"requests per minute and requests per day exhausted",
            "details":[{"quotaId":"GenerateRequestsPerDayPerProjectPerModel"}]}}"#;
        let (class, _, _) = classify_429(body, None, Duration::from_secs(60));
        assert_eq!(class, LimitClass::PerDay);
    }

    #[test]
    fn statuses_map_to_the_right_variants() {
        let s = settings();
        let key = "SECRET";
        assert!(matches!(
            classify_status(401, "{}", None, key, &s),
            GeminiError::Auth { .. }
        ));
        assert!(matches!(
            classify_status(403, "{}", None, key, &s),
            GeminiError::Auth { .. }
        ));
        assert!(matches!(
            classify_status(404, "{}", None, key, &s),
            GeminiError::Unavailable { .. }
        ));
        assert!(matches!(
            classify_status(500, "{}", None, key, &s),
            GeminiError::Transient { .. }
        ));
        assert!(matches!(
            classify_status(503, "{}", None, key, &s),
            GeminiError::Transient { .. }
        ));
        assert!(matches!(
            classify_status(400, "bad arg", None, key, &s),
            GeminiError::Client { .. }
        ));
        // A 400 that says the model is missing should fall through instead of failing hard.
        assert!(matches!(
            classify_status(
                400,
                r#"{"error":{"message":"models/foo is not found"}}"#,
                None,
                key,
                &s
            ),
            GeminiError::Unavailable { .. }
        ));
    }

    #[test]
    fn detail_never_contains_the_key() {
        let key = "AIzaSyTOPSECRET";
        let s = settings();
        let err = classify_status(400, &format!("bad arg for key {key}"), None, key, &s);
        // Check the rendered message too: it is what reaches the log and the UI.
        let rendered = err.to_string();
        assert!(!rendered.contains(key), "the key leaked: {rendered}");
        assert!(rendered.contains("***"));
    }

    #[test]
    fn endpoint_uses_header_auth_and_sse_when_streaming() {
        assert_eq!(
            endpoint("gemini-3.8-flash", "generateContent", false),
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent"
        );
        assert!(
            endpoint("m", "streamGenerateContent", true)
                .ends_with(":streamGenerateContent?alt=sse")
        );
        // The key must never appear in a URL.
        assert!(!endpoint("m", "generateContent", false).contains("key="));
    }
}
