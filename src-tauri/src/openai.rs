//! Translation between the OpenAI wire format and the Gemini wire format.
//!
//! Supported: system/developer/user/assistant/tool messages, function tools and
//! `tool_choice`, temperature/top_p/max tokens/stop sequences, and streaming.
//!
//! Not supported (ignored): `n` > 1, logprobs, logit_bias, response_format,
//! seed, and OpenAI's own structured-output extensions.

use std::collections::HashMap;

use serde_json::{Map, Value, json};

/// JSON Schema keywords Gemini's `Schema` message understands.
///
/// This is an allowlist, not a denylist: MCP tool schemas routinely carry
/// draft-07 keywords (`propertyNames`, `exclusiveMinimum`, `patternProperties`,
/// `const`, `if`/`then`, …) that Gemini's strict protobuf parser rejects with
/// `400 Invalid JSON payload`. Anything not listed here is dropped.
const SUPPORTED_SCHEMA_KEYS: [&str; 19] = [
    "type",
    "format",
    "description",
    "nullable",
    "enum",
    "properties",
    "required",
    "items",
    "anyOf",
    "minItems",
    "maxItems",
    "minLength",
    "maxLength",
    "minProperties",
    "maxProperties",
    "minimum",
    "maximum",
    "pattern",
    "propertyOrdering",
];

pub(crate) fn short_id() -> String {
    uuid::Uuid::new_v4()
        .simple()
        .to_string()
        .chars()
        .take(12)
        .collect()
}

/// Flatten an OpenAI message `content` (string or array of parts) into text.
pub(crate) fn content_to_text(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(|item| match item {
                Value::String(s) => Some(s.clone()),
                Value::Object(o) => o
                    .get("text")
                    .and_then(|t| t.as_str())
                    .map(|s| s.to_string()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(""),
        Some(Value::Null) | None => String::new(),
        Some(other) => other.to_string(),
    }
}

/// Append `parts` to the previous turn when it has the same role.
///
/// Gemini expects alternating turns. OpenAI clients routinely send several
/// consecutive `tool` messages (one per parallel tool call) and sometimes
/// several consecutive user messages; merging them into one content block keeps
/// the translated conversation valid without changing its meaning.
pub(crate) fn push_content(contents: &mut Vec<Value>, role: &str, mut parts: Vec<Value>) {
    if parts.is_empty() {
        return;
    }
    if let Some(last) = contents.last_mut() {
        let same_role = last.get("role").and_then(|r| r.as_str()) == Some(role);
        if same_role {
            if let Some(existing) = last.get_mut("parts").and_then(|p| p.as_array_mut()) {
                existing.append(&mut parts);
                return;
            }
        }
    }
    contents.push(json!({ "role": role, "parts": parts }));
}

/// OpenAI sends tool arguments as a JSON *string*; Gemini wants an object.
fn parse_args(value: &Value) -> Value {
    match value {
        Value::String(s) => serde_json::from_str::<Value>(s).unwrap_or_else(|_| json!({})),
        Value::Object(_) => value.clone(),
        _ => json!({}),
    }
}

/// Keep integer bounds as integers; only fall back to a float when needed.
fn schema_number(value: &Value) -> Option<Value> {
    if value.is_i64() || value.is_u64() {
        Some(value.clone())
    } else {
        value.as_f64().map(|number| json!(number))
    }
}

/// Reduce an arbitrary JSON Schema to the subset Gemini accepts.
pub(crate) fn clean_schema(value: &Value) -> Value {
    match value {
        // Draft-06+ boolean schemas (`true` / `false`) are not valid here.
        Value::Bool(_) => json!({}),
        Value::Object(map) => {
            let mut out = Map::new();
            for (key, child) in map {
                match key.as_str() {
                    // `type` may be ["string", "null"]; Gemini wants one type.
                    "type" => match child {
                        Value::String(_) => {
                            out.insert("type".into(), child.clone());
                        }
                        Value::Array(types) => {
                            let nullable = types.iter().any(|t| t.as_str() == Some("null"));
                            if let Some(name) = types
                                .iter()
                                .filter_map(Value::as_str)
                                .find(|name| *name != "null")
                            {
                                out.insert("type".into(), json!(name));
                            }
                            if nullable {
                                out.insert("nullable".into(), json!(true));
                            }
                        }
                        _ => {}
                    },
                    // Gemini has `anyOf` but no `oneOf`; treating oneOf as anyOf
                    // keeps the variants instead of erasing the schema.
                    "oneOf" => {
                        out.insert("anyOf".into(), clean_schema(child));
                    }
                    // Draft-07 exclusive bounds are numbers; map them to the
                    // inclusive fields Gemini understands.
                    "exclusiveMinimum" => {
                        if let Some(number) = schema_number(child) {
                            out.insert("minimum".into(), number);
                        }
                    }
                    "exclusiveMaximum" => {
                        if let Some(number) = schema_number(child) {
                            out.insert("maximum".into(), number);
                        }
                    }
                    // `const` has no Gemini equivalent; a one-value enum does.
                    "const" => {
                        out.insert("enum".into(), json!([child]));
                    }
                    // Tuple validation is unsupported; keep the first shape.
                    "items" => {
                        let item = match child {
                            Value::Array(items) => {
                                items.first().cloned().unwrap_or_else(|| json!({}))
                            }
                            other => other.clone(),
                        };
                        out.insert("items".into(), clean_schema(&item));
                    }
                    // Property names are user data, so only the schemas inside
                    // are cleaned. An empty map is still a valid object schema.
                    "properties" => {
                        let mut properties = Map::new();
                        if let Value::Object(entries) = child {
                            for (name, schema) in entries {
                                properties.insert(name.clone(), clean_schema(schema));
                            }
                        }
                        out.insert("properties".into(), Value::Object(properties));
                    }
                    // Gemini's enum is a string array; a numeric enum would be
                    // rejected, so keep the string members and drop the rest.
                    "enum" => {
                        let values: Vec<Value> = child
                            .as_array()
                            .map(|items| {
                                items
                                    .iter()
                                    .filter(|item| item.is_string())
                                    .cloned()
                                    .collect()
                            })
                            .unwrap_or_default();
                        if !values.is_empty() {
                            out.insert("enum".into(), Value::Array(values));
                        }
                    }
                    "required" => {
                        let names: Vec<Value> = child
                            .as_array()
                            .map(|items| {
                                items
                                    .iter()
                                    .filter(|item| item.is_string())
                                    .cloned()
                                    .collect()
                            })
                            .unwrap_or_default();
                        if !names.is_empty() {
                            out.insert("required".into(), Value::Array(names));
                        }
                    }
                    _ if SUPPORTED_SCHEMA_KEYS.contains(&key.as_str()) => {
                        out.insert(key.clone(), clean_schema(child));
                    }
                    // Everything else (propertyNames, patternProperties, $ref,
                    // additionalProperties, $schema, strict, default, …) would
                    // make Gemini return 400, so drop it.
                    _ => {}
                }
            }
            Value::Object(out)
        }
        Value::Array(items) => Value::Array(items.iter().map(clean_schema).collect()),
        other => other.clone(),
    }
}

fn tools_to_gemini(tools: &Value) -> Option<Value> {
    let array = tools.as_array()?;
    let declarations: Vec<Value> = array
        .iter()
        .filter_map(|tool| {
            // Accept both {"type":"function","function":{...}} and a bare function object.
            let f = tool.get("function").unwrap_or(tool);
            let name = f.get("name")?.as_str()?.to_string();
            let mut decl = Map::new();
            decl.insert("name".into(), json!(name));
            if let Some(description) = f.get("description").and_then(|v| v.as_str()) {
                decl.insert("description".into(), json!(description));
            }
            if let Some(parameters) = f.get("parameters") {
                decl.insert("parameters".into(), clean_schema(parameters));
            }
            Some(Value::Object(decl))
        })
        .collect();
    if declarations.is_empty() {
        None
    } else {
        Some(json!([{ "functionDeclarations": declarations }]))
    }
}

fn tool_config_to_gemini(tool_choice: &Value) -> Option<Value> {
    let (mode, allowed) = match tool_choice {
        Value::String(s) => match s.as_str() {
            "none" => ("NONE", None),
            "required" | "any" => ("ANY", None),
            _ => ("AUTO", None),
        },
        Value::Object(o) => {
            let name = o
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(|n| n.as_str())
                .map(|n| n.to_string());
            match name {
                Some(n) => ("ANY", Some(vec![n])),
                None => ("AUTO", None),
            }
        }
        _ => ("AUTO", None),
    };
    let mut config = Map::new();
    let mut calling = Map::new();
    calling.insert("mode".into(), json!(mode));
    if let Some(names) = allowed {
        calling.insert("allowedFunctionNames".into(), json!(names));
    }
    config.insert("functionCallingConfig".into(), Value::Object(calling));
    Some(json!({ "toolConfig": config }))
}

fn generation_config(req: &Value) -> Option<Value> {
    let mut generation = Map::new();
    if let Some(t) = req.get("temperature").and_then(|v| v.as_f64()) {
        generation.insert("temperature".into(), json!(t));
    }
    if let Some(p) = req.get("top_p").and_then(|v| v.as_f64()) {
        generation.insert("topP".into(), json!(p));
    } else if let Some(p) = req.get("topP").and_then(|v| v.as_f64()) {
        generation.insert("topP".into(), json!(p));
    }
    let max = req
        .get("max_completion_tokens")
        .or_else(|| req.get("max_tokens"))
        .and_then(|v| v.as_u64());
    if let Some(m) = max {
        if m > 0 {
            generation.insert("maxOutputTokens".into(), json!(m));
        }
    }
    match req.get("stop") {
        Some(Value::String(s)) if !s.is_empty() => {
            generation.insert("stopSequences".into(), json!([s]));
        }
        Some(Value::Array(items)) => {
            let stops: Vec<&str> = items.iter().filter_map(|v| v.as_str()).collect();
            if !stops.is_empty() {
                generation.insert("stopSequences".into(), json!(stops));
            }
        }
        _ => {}
    }
    if generation.is_empty() {
        None
    } else {
        Some(Value::Object(generation))
    }
}

/// Convert an OpenAI `/v1/chat/completions` body into a Gemini request body.
pub fn chat_request_to_gemini(req: &Value) -> Result<Value, String> {
    let messages = req
        .get("messages")
        .and_then(|m| m.as_array())
        .ok_or_else(|| "`messages` must be a non-empty array".to_string())?;
    if messages.is_empty() {
        return Err("`messages` must be a non-empty array".to_string());
    }

    let mut contents: Vec<Value> = Vec::new();
    let mut system_parts: Vec<Value> = Vec::new();
    // OpenAI correlates tool results by id; Gemini needs the function name.
    let mut id_to_name: HashMap<String, String> = HashMap::new();

    for message in messages {
        let role = message
            .get("role")
            .and_then(|r| r.as_str())
            .unwrap_or("user");
        match role {
            "system" | "developer" => {
                let text = content_to_text(message.get("content"));
                if !text.is_empty() {
                    system_parts.push(json!({ "text": text }));
                }
            }
            "user" => {
                let text = content_to_text(message.get("content"));
                let mut parts: Vec<Value> = Vec::new();
                if !text.is_empty() {
                    parts.push(json!({ "text": text }));
                }
                push_content(&mut contents, "user", parts);
            }
            "assistant" => {
                let mut parts: Vec<Value> = Vec::new();
                let text = content_to_text(message.get("content"));
                if !text.is_empty() {
                    parts.push(json!({ "text": text }));
                }
                if let Some(calls) = message.get("tool_calls").and_then(|t| t.as_array()) {
                    // Gemini only needs the signature on the first function
                    // call of a parallel group.  If the original signature is
                    // gone (imported history, older conversation), the
                    // documented sentinel keeps the request valid.
                    let mut first_tool = true;
                    for call in calls {
                        let f = call.get("function").cloned().unwrap_or(Value::Null);
                        let name = f
                            .get("name")
                            .and_then(|n| n.as_str())
                            .unwrap_or_default()
                            .to_string();
                        if name.is_empty() {
                            continue;
                        }
                        let id = call
                            .get("id")
                            .and_then(|i| i.as_str())
                            .unwrap_or_default()
                            .to_string();
                        if !id.is_empty() {
                            id_to_name.insert(id.clone(), name.clone());
                        }
                        let args = parse_args(&f.get("arguments").cloned().unwrap_or(Value::Null));
                        let mut part = json!({ "functionCall": { "name": name, "args": args } });
                        if first_tool {
                            let signature = crate::signatures::for_replay(call, &id)
                                .unwrap_or_else(|| crate::signatures::SKIP_VALIDATOR.to_string());
                            part["thoughtSignature"] = json!(signature);
                            first_tool = false;
                        }
                        parts.push(part);
                    }
                }
                push_content(&mut contents, "model", parts);
            }
            // Legacy "function" role and modern "tool" role both become functionResponse.
            "tool" | "function" => {
                let name = message
                    .get("tool_call_id")
                    .and_then(|v| v.as_str())
                    .and_then(|id| id_to_name.get(id).cloned())
                    .or_else(|| {
                        message
                            .get("name")
                            .and_then(|v| v.as_str())
                            .map(str::to_string)
                    })
                    .unwrap_or_else(|| "tool".to_string());
                let text = content_to_text(message.get("content"));
                let response = serde_json::from_str::<Value>(&text)
                    .ok()
                    .filter(|v| v.is_object())
                    .unwrap_or_else(|| json!({ "content": text }));
                push_content(
                    &mut contents,
                    "user",
                    vec![json!({ "functionResponse": { "name": name, "response": response } })],
                );
            }
            _ => {}
        }
    }

    if contents.is_empty() {
        return Err("no usable messages were found in the request".to_string());
    }

    let mut body = Map::new();
    body.insert("contents".into(), Value::Array(contents));
    if !system_parts.is_empty() {
        body.insert("systemInstruction".into(), json!({ "parts": system_parts }));
    }
    if let Some(generation) = generation_config(req) {
        body.insert("generationConfig".into(), generation);
    }
    if let Some(tools) = req.get("tools") {
        if let Some(converted) = tools_to_gemini(tools) {
            body.insert("tools".into(), converted);
        }
    }
    // `toolConfig` without `tools` is rejected by Gemini, and some clients send
    // a stale tool_choice after dropping the tools array.
    if body.contains_key("tools") {
        if let Some(choice) = req.get("tool_choice") {
            if let Some(config) = tool_config_to_gemini(choice) {
                for (k, v) in config.as_object().into_iter().flatten() {
                    body.insert(k.clone(), v.clone());
                }
            }
        }
    }

    Ok(Value::Object(body))
}

/// Convert a legacy `/v1/completions` body into a chat-style body.
pub fn completions_request_to_chat(req: &Value) -> Result<Value, String> {
    let prompt = match req.get("prompt") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(|v| v.as_str())
            .collect::<Vec<_>>()
            .join("\n"),
        _ => return Err("`prompt` must be a string or an array of strings".to_string()),
    };
    let mut chat = req.clone();
    if let Some(obj) = chat.as_object_mut() {
        obj.insert(
            "messages".into(),
            json!([{ "role": "user", "content": prompt }]),
        );
        obj.remove("prompt");
        obj.remove("echo");
        obj.remove("best_of");
    }
    Ok(chat)
}

fn finish_reason(resp: &Value, has_tool_calls: bool) -> &'static str {
    if has_tool_calls {
        return "tool_calls";
    }
    // A blocked prompt yields no candidates at all.
    if resp.pointer("/candidates/0").is_none() {
        if resp.pointer("/promptFeedback/blockReason").is_some() {
            return "content_filter";
        }
    }
    match resp
        .pointer("/candidates/0/finishReason")
        .and_then(|v| v.as_str())
    {
        Some("MAX_TOKENS") => "length",
        Some("SAFETY")
        | Some("RECITATION")
        | Some("BLOCKLIST")
        | Some("PROHIBITED_CONTENT")
        | Some("SPII")
        | Some("IMAGE_SAFETY") => "content_filter",
        _ => "stop",
    }
}

fn usage_object(resp: &Value) -> Value {
    let meta = resp.get("usageMetadata");
    let prompt = meta
        .and_then(|m| m.get("promptTokenCount"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let completion = meta
        .and_then(|m| m.get("candidatesTokenCount"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let total = meta
        .and_then(|m| m.get("totalTokenCount"))
        .and_then(|v| v.as_u64())
        .unwrap_or(prompt + completion);
    json!({ "prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total })
}

fn extract_tool_calls(resp: &Value) -> Vec<Value> {
    resp.pointer("/candidates/0/content/parts")
        .and_then(|p| p.as_array())
        .map(|parts| {
            parts
                .iter()
                .enumerate()
                .filter_map(|(index, part)| {
                    let call = part.get("functionCall")?;
                    let name = call.get("name")?.as_str()?.to_string();
                    let args = call.get("args").cloned().unwrap_or_else(|| json!({}));
                    let id = format!("call_{index}_{}", short_id());
                    let signature = crate::signatures::from_part(part);
                    if let Some(signature) = &signature {
                        crate::signatures::put(&id, signature);
                    }
                    let mut call = json!({
                        "id": id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": serde_json::to_string(&args).unwrap_or_else(|_| "{}".into())
                        }
                    });
                    if let Some(signature) = signature {
                        call["extra_content"] =
                            json!({ "google": { "thought_signature": signature } });
                    }
                    Some(call)
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Convert a Gemini response into an OpenAI `chat.completion`.
pub fn gemini_response_to_chat(resp: &Value, model: &str, created: i64) -> Value {
    let text = crate::router::extract_text(resp);
    let tool_calls = extract_tool_calls(resp);
    let has_tools = !tool_calls.is_empty();

    let mut message = Map::new();
    message.insert("role".into(), json!("assistant"));
    if has_tools {
        message.insert(
            "content".into(),
            if text.is_empty() {
                Value::Null
            } else {
                json!(text)
            },
        );
        message.insert("tool_calls".into(), Value::Array(tool_calls));
    } else {
        message.insert("content".into(), json!(text));
    }

    json!({
        "id": format!("chatcmpl-{}", short_id()),
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": Value::Object(message),
            "finish_reason": finish_reason(resp, has_tools)
        }],
        "usage": usage_object(resp)
    })
}

/// Convert a Gemini response into a legacy `text_completion`.
pub fn gemini_response_to_completion(resp: &Value, model: &str, created: i64) -> Value {
    json!({
        "id": format!("cmpl-{}", short_id()),
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [{
            "text": crate::router::extract_text(resp),
            "index": 0,
            "logprobs": Value::Null,
            "finish_reason": finish_reason(resp, false)
        }],
        "usage": usage_object(resp)
    })
}

fn delta_chunk(id: &str, model: &str, created: i64, delta: Value, finish: Option<&str>) -> Value {
    json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{ "index": 0, "delta": delta, "finish_reason": finish }]
    })
}

/// State carried across a streaming response.
///
/// Gemini can send the `functionCall` part and the `finishReason` in different
/// SSE chunks, so the translator must remember that a tool call was emitted or
/// the final chunk would incorrectly report `stop` instead of `tool_calls`.
#[derive(Debug, Default)]
pub struct StreamTranslator {
    emitted_tool_calls: bool,
    /// Gemini puts the signature on the first function-call part of a turn.
    first_tool_id: Option<String>,
}

impl StreamTranslator {
    /// Convert one Gemini SSE chunk into zero or more OpenAI delta chunks.
    pub fn translate(&mut self, chunk: &Value, id: &str, model: &str, created: i64) -> Vec<Value> {
        let mut out: Vec<Value> = Vec::new();

        if let Some(parts) = chunk
            .pointer("/candidates/0/content/parts")
            .and_then(|p| p.as_array())
        {
            let text: String = parts
                .iter()
                .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
                .collect();
            if !text.is_empty() {
                out.push(delta_chunk(
                    id,
                    model,
                    created,
                    json!({ "content": text }),
                    None,
                ));
            }
            for (index, part) in parts.iter().enumerate() {
                if let Some(call) = part.get("functionCall") {
                    self.emitted_tool_calls = true;
                    let name = call
                        .get("name")
                        .and_then(|n| n.as_str())
                        .unwrap_or_default()
                        .to_string();
                    let args = call.get("args").cloned().unwrap_or_else(|| json!({}));
                    let tool_id = format!("call_{index}_{}", short_id());
                    let signature = crate::signatures::from_part(part);
                    if let Some(signature) = &signature {
                        crate::signatures::put(&tool_id, signature);
                    }
                    if self.first_tool_id.is_none() {
                        self.first_tool_id = Some(tool_id.clone());
                    }
                    let mut tool_call = json!({
                        "index": index,
                        "id": tool_id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": serde_json::to_string(&args).unwrap_or_else(|_| "{}".into())
                        }
                    });
                    if let Some(signature) = signature {
                        tool_call["extra_content"] =
                            json!({ "google": { "thought_signature": signature } });
                    }
                    out.push(delta_chunk(
                        id,
                        model,
                        created,
                        json!({ "tool_calls": [tool_call] }),
                        None,
                    ));
                } else if let Some(signature) = crate::signatures::from_part(part) {
                    // A signature may arrive in a follow-up part.  Only attach
                    // it when no visible text is present, so text signatures
                    // are not misattributed to the tool call.
                    if part.get("text").is_none() {
                        if let Some(tool_id) = &self.first_tool_id {
                            crate::signatures::put(tool_id, &signature);
                        }
                    }
                }
            }
        }

        if let Some(reason) = chunk
            .pointer("/candidates/0/finishReason")
            .and_then(|v| v.as_str())
        {
            let mapped = match reason {
                "MAX_TOKENS" => "length",
                "SAFETY" | "RECITATION" | "BLOCKLIST" | "PROHIBITED_CONTENT" | "SPII"
                | "IMAGE_SAFETY" => "content_filter",
                _ if self.emitted_tool_calls => "tool_calls",
                _ => "stop",
            };
            out.push(delta_chunk(id, model, created, json!({}), Some(mapped)));
        }

        out
    }
}

/// Convert one Gemini SSE chunk into zero or more OpenAI delta chunks.
///
/// Convenience wrapper for a single chunk; streaming callers should keep a
/// [`StreamTranslator`] so state survives across chunks. Only tests use the
/// single-chunk form; the server always holds a translator.
#[cfg(test)]
pub fn gemini_chunk_to_openai(chunk: &Value, id: &str, model: &str, created: i64) -> Vec<Value> {
    StreamTranslator::default().translate(chunk, id, model, created)
}

pub fn stream_usage_chunk(id: &str, model: &str, created: i64, resp: &Value) -> Value {
    json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [],
        "usage": usage_object(resp)
    })
}

pub fn chunk_id() -> String {
    format!("chatcmpl-{}", short_id())
}

/// One entry of the `/v1/models` listing.
pub fn model_object(id: &str, owned_by: &str) -> Value {
    json!({
        "id": id,
        "object": "model",
        "created": 0,
        "owned_by": owned_by
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_user_message_round_trips() {
        let req = json!({"messages":[{"role":"user","content":"hello"}]});
        let body = chat_request_to_gemini(&req).unwrap();
        assert_eq!(body["contents"][0]["role"], "user");
        assert_eq!(body["contents"][0]["parts"][0]["text"], "hello");
    }

    #[test]
    fn system_messages_become_system_instruction() {
        let req = json!({"messages":[
            {"role":"system","content":"be terse"},
            {"role":"developer","content":"and correct"},
            {"role":"user","content":"hi"}
        ]});
        let body = chat_request_to_gemini(&req).unwrap();
        let parts = body["systemInstruction"]["parts"].as_array().unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0]["text"], "be terse");
        assert_eq!(parts[1]["text"], "and correct");
        // The system text must not leak into the conversation contents.
        assert_eq!(body["contents"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn assistant_becomes_model_role() {
        let req = json!({"messages":[
            {"role":"user","content":"hi"},
            {"role":"assistant","content":"hello"},
            {"role":"user","content":"bye"}
        ]});
        let body = chat_request_to_gemini(&req).unwrap();
        let roles: Vec<&str> = body["contents"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c["role"].as_str().unwrap())
            .collect();
        assert_eq!(roles, vec!["user", "model", "user"]);
    }

    #[test]
    fn tool_calls_and_results_map_symmetrically() {
        let req = json!({"messages":[
            {"role":"user","content":"weather?"},
            {"role":"assistant","tool_calls":[
                {"id":"call_abc","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}}
            ]},
            {"role":"tool","tool_call_id":"call_abc","content":"{\"temp\":21}"}
        ]});
        let body = chat_request_to_gemini(&req).unwrap();
        let call = &body["contents"][1]["parts"][0]["functionCall"];
        assert_eq!(call["name"], "get_weather");
        // Arguments arrive as a JSON string and must become a real object.
        assert_eq!(call["args"]["city"], "Paris");

        let result = &body["contents"][2]["parts"][0]["functionResponse"];
        // The name is recovered from the earlier tool_call id.
        assert_eq!(result["name"], "get_weather");
        assert_eq!(result["response"]["temp"], 21);
    }

    #[test]
    fn parallel_tool_results_share_one_user_turn() {
        let req = json!({"messages":[
            {"role":"user","content":"weather in Paris and Rome?"},
            {"role":"assistant","tool_calls":[
                {"id":"a","type":"function","function":{"name":"weather","arguments":"{\"city\":\"Paris\"}"}},
                {"id":"b","type":"function","function":{"name":"weather","arguments":"{\"city\":\"Rome\"}"}}
            ]},
            {"role":"tool","tool_call_id":"a","content":"{\"temp\":21}"},
            {"role":"tool","tool_call_id":"b","content":"{\"temp\":25}"}
        ]});
        let body = chat_request_to_gemini(&req).unwrap();
        let contents = body["contents"].as_array().unwrap();
        // user, model, user — the two tool results must not become two user turns.
        assert_eq!(contents.len(), 3);
        assert_eq!(contents[2]["role"], "user");
        let parts = contents[2]["parts"].as_array().unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0]["functionResponse"]["name"], "weather");
        assert_eq!(parts[1]["functionResponse"]["name"], "weather");
    }

    #[test]
    fn consecutive_user_messages_merge_into_one_turn() {
        let req = json!({"messages":[
            {"role":"user","content":"first"},
            {"role":"user","content":"second"}
        ]});
        let body = chat_request_to_gemini(&req).unwrap();
        let contents = body["contents"].as_array().unwrap();
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0]["parts"].as_array().unwrap().len(), 2);
        assert_eq!(contents[0]["parts"][1]["text"], "second");
    }

    #[test]
    fn malformed_tool_arguments_do_not_panic() {
        let req = json!({"messages":[
            {"role":"assistant","tool_calls":[
                {"id":"x","type":"function","function":{"name":"f","arguments":"{not json"}}
            ]}
        ]});
        let body = chat_request_to_gemini(&req).unwrap();
        assert_eq!(
            body["contents"][0]["parts"][0]["functionCall"]["args"],
            json!({})
        );
    }

    #[test]
    fn tools_become_function_declarations_without_unsupported_keys() {
        let req = json!({
            "messages":[{"role":"user","content":"hi"}],
            "tools":[{"type":"function","function":{
                "name":"get_weather",
                "description":"Get weather",
                "strict":true,
                "parameters":{
                    "$schema":"http://json-schema.org/draft-07/schema#",
                    "type":"object",
                    "additionalProperties":false,
                    "properties":{"city":{"type":"string","description":"City"}},
                    "required":["city"]
                }
            }}]
        });
        let body = chat_request_to_gemini(&req).unwrap();
        let decl = &body["tools"][0]["functionDeclarations"][0];
        assert_eq!(decl["name"], "get_weather");
        assert_eq!(decl["description"], "Get weather");
        assert!(decl["parameters"].get("$schema").is_none());
        assert!(decl["parameters"].get("additionalProperties").is_none());
        assert_eq!(decl["parameters"]["properties"]["city"]["type"], "string");
    }

    #[test]
    fn schema_keys_gemini_rejects_are_removed() {
        // The exact shape that made Gemini return
        // "Unknown name propertyNames / exclusiveMinimum".
        let schema = json!({
            "type": "object",
            "propertyNames": { "pattern": "^[a-z]+$" },
            "patternProperties": { "^x-": { "type": "string" } },
            "$defs": { "unused": { "type": "string" } },
            "additionalProperties": false,
            "properties": {
                "count": {
                    "type": ["integer", "null"],
                    "exclusiveMinimum": 0,
                    "exclusiveMaximum": 100
                },
                "mode": { "const": "fast" },
                "choice": { "oneOf": [{ "type": "string" }, { "type": "number" }] }
            },
            "required": ["count", "mode"]
        });
        let cleaned = clean_schema(&schema);

        for rejected in [
            "propertyNames",
            "patternProperties",
            "$defs",
            "additionalProperties",
            "exclusiveMinimum",
            "exclusiveMaximum",
            "const",
            "oneOf",
        ] {
            assert!(
                !serde_json::to_string(&cleaned).unwrap().contains(rejected),
                "{rejected} survived cleaning"
            );
        }
        // Structurally valid replacements are kept.
        assert_eq!(cleaned["properties"]["count"]["type"], "integer");
        assert_eq!(cleaned["properties"]["count"]["nullable"], true);
        assert_eq!(cleaned["properties"]["count"]["minimum"], 0);
        assert_eq!(cleaned["properties"]["count"]["maximum"], 100);
        assert_eq!(cleaned["properties"]["mode"]["enum"][0], "fast");
        assert!(cleaned["properties"]["choice"]["anyOf"].is_array());
        assert_eq!(cleaned["required"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn boolean_and_tuple_schemas_are_normalised() {
        let cleaned = clean_schema(&json!({
            "type": "object",
            "properties": {
                "anything": true,
                "pair": { "type": "array", "items": [{ "type": "string" }] }
            }
        }));
        assert_eq!(cleaned["properties"]["anything"], json!({}));
        assert_eq!(cleaned["properties"]["pair"]["items"]["type"], "string");
    }

    #[test]
    fn tool_choice_none_and_specific_function() {
        let base = json!({"messages":[{"role":"user","content":"hi"}],
            "tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}]});

        let mut none = base.clone();
        none["tool_choice"] = json!("none");
        assert_eq!(
            chat_request_to_gemini(&none).unwrap()["toolConfig"]["functionCallingConfig"]["mode"],
            "NONE"
        );

        let mut one = base.clone();
        one["tool_choice"] = json!({"type":"function","function":{"name":"f"}});
        assert_eq!(
            chat_request_to_gemini(&one).unwrap()["toolConfig"]["functionCallingConfig"]["mode"],
            "ANY"
        );

        let mut required = base;
        required["tool_choice"] = json!("required");
        assert_eq!(
            chat_request_to_gemini(&required).unwrap()["toolConfig"]["functionCallingConfig"]["mode"],
            "ANY"
        );
    }

    #[test]
    fn tool_choice_without_tools_is_ignored() {
        let req = json!({
            "messages":[{"role":"user","content":"hi"}],
            "tool_choice":"none"
        });
        assert!(
            chat_request_to_gemini(&req)
                .unwrap()
                .get("toolConfig")
                .is_none()
        );
    }

    #[test]
    fn generation_config_maps_openai_names() {
        let req = json!({
            "messages":[{"role":"user","content":"hi"}],
            "temperature":0.3, "top_p":0.9, "max_tokens":256, "stop":["END"]
        });
        let generation = &chat_request_to_gemini(&req).unwrap()["generationConfig"];
        assert_eq!(generation["temperature"], 0.3);
        assert_eq!(generation["topP"], 0.9);
        assert_eq!(generation["maxOutputTokens"], 256);
        assert_eq!(generation["stopSequences"][0], "END");
    }

    #[test]
    fn max_completion_tokens_wins_over_max_tokens() {
        let req = json!({"messages":[{"role":"user","content":"hi"}],
            "max_tokens":10, "max_completion_tokens":99});
        assert_eq!(
            chat_request_to_gemini(&req).unwrap()["generationConfig"]["maxOutputTokens"],
            99
        );
    }

    #[test]
    fn empty_messages_are_rejected() {
        assert!(chat_request_to_gemini(&json!({"messages":[]})).is_err());
        assert!(chat_request_to_gemini(&json!({})).is_err());
        // A request with only a system message has nothing to send.
        assert!(
            chat_request_to_gemini(&json!({"messages":[{"role":"system","content":"x"}]})).is_err()
        );
    }

    #[test]
    fn response_with_tool_call_sets_tool_calls_finish_reason() {
        let resp = json!({
            "candidates":[{"content":{"parts":[{"functionCall":{"name":"get_weather","args":{"city":"Paris"}}}]},
                           "finishReason":"STOP"}],
            "usageMetadata":{"promptTokenCount":9,"candidatesTokenCount":4,"totalTokenCount":13}
        });
        let out = gemini_response_to_chat(&resp, "gemini-3.8-flash", 1);
        assert_eq!(out["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(out["choices"][0]["message"]["content"], Value::Null);
        let call = &out["choices"][0]["message"]["tool_calls"][0];
        assert_eq!(call["type"], "function");
        assert_eq!(call["function"]["name"], "get_weather");
        // Arguments must be a JSON *string* on the way out.
        let args: Value =
            serde_json::from_str(call["function"]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(args["city"], "Paris");
        assert_eq!(out["usage"]["total_tokens"], 13);
    }

    #[test]
    fn response_with_text_has_stop_finish_reason() {
        let resp =
            json!({"candidates":[{"content":{"parts":[{"text":"hi"}]},"finishReason":"STOP"}]});
        let out = gemini_response_to_chat(&resp, "m", 1);
        assert_eq!(out["choices"][0]["message"]["content"], "hi");
        assert_eq!(out["choices"][0]["finish_reason"], "stop");
    }

    #[test]
    fn max_tokens_finish_reason_becomes_length() {
        let resp = json!({"candidates":[{"content":{"parts":[{"text":"x"}]},"finishReason":"MAX_TOKENS"}]});
        assert_eq!(
            gemini_response_to_chat(&resp, "m", 1)["choices"][0]["finish_reason"],
            "length"
        );
    }

    #[test]
    fn blocked_prompt_becomes_content_filter() {
        let resp = json!({"promptFeedback":{"blockReason":"SAFETY"}});
        assert_eq!(
            gemini_response_to_chat(&resp, "m", 1)["choices"][0]["finish_reason"],
            "content_filter"
        );
    }

    #[test]
    fn streaming_chunk_emits_text_delta() {
        let chunk = json!({"candidates":[{"content":{"parts":[{"text":"Hel"}]}}]});
        let out = gemini_chunk_to_openai(&chunk, "id", "m", 1);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0]["object"], "chat.completion.chunk");
        assert_eq!(out[0]["choices"][0]["delta"]["content"], "Hel");
        assert_eq!(out[0]["choices"][0]["finish_reason"], Value::Null);
    }

    #[test]
    fn streaming_chunk_with_finish_reason_closes_the_stream() {
        let chunk =
            json!({"candidates":[{"content":{"parts":[{"text":"o"}]},"finishReason":"STOP"}]});
        let out = gemini_chunk_to_openai(&chunk, "id", "m", 1);
        assert_eq!(out.len(), 2);
        assert_eq!(out[1]["choices"][0]["finish_reason"], "stop");
    }

    #[test]
    fn streaming_tool_call_is_reported_as_tool_calls() {
        let chunk = json!({"candidates":[{"content":{"parts":[
            {"functionCall":{"name":"f","args":{"a":1}}}]},"finishReason":"STOP"}]});
        let out = gemini_chunk_to_openai(&chunk, "id", "m", 1);
        let tc = &out[0]["choices"][0]["delta"]["tool_calls"][0];
        assert_eq!(tc["function"]["name"], "f");
        assert_eq!(out[1]["choices"][0]["finish_reason"], "tool_calls");
    }

    #[test]
    fn streamed_tool_call_finish_reason_survives_a_later_chunk() {
        let mut translator = StreamTranslator::default();
        let call = json!({"candidates":[{"content":{"parts":[
            {"functionCall":{"name":"f","args":{"a":1}}}]}}]});
        let first = translator.translate(&call, "id", "m", 1);
        assert_eq!(first.len(), 1);
        assert!(first[0]["choices"][0]["delta"]["tool_calls"].is_array());

        // Gemini often sends the finish reason on its own in the next chunk.
        let end = json!({"candidates":[{"finishReason":"STOP"}]});
        let last = translator.translate(&end, "id", "m", 1);
        assert_eq!(last.len(), 1);
        assert_eq!(last[0]["choices"][0]["finish_reason"], "tool_calls");
    }

    #[test]
    fn streaming_chunk_without_candidates_is_ignored() {
        assert!(gemini_chunk_to_openai(&json!({"usageMetadata":{}}), "id", "m", 1).is_empty());
    }

    #[test]
    fn legacy_completions_prompt_becomes_a_user_message() {
        let req = json!({"prompt":"complete this","max_tokens":5});
        let chat = completions_request_to_chat(&req).unwrap();
        assert_eq!(chat["messages"][0]["role"], "user");
        assert_eq!(chat["messages"][0]["content"], "complete this");
        assert!(chat.get("prompt").is_none());
        assert!(completions_request_to_chat(&json!({})).is_err());
    }

    #[test]
    fn completion_response_shape_is_legacy_compatible() {
        let resp =
            json!({"candidates":[{"content":{"parts":[{"text":"done"}]},"finishReason":"STOP"}]});
        let out = gemini_response_to_completion(&resp, "m", 7);
        assert_eq!(out["object"], "text_completion");
        assert_eq!(out["choices"][0]["text"], "done");
        assert_eq!(out["choices"][0]["finish_reason"], "stop");
    }

    #[test]
    fn thought_signature_survives_a_tool_call_round_trip() {
        let resp = json!({
            "candidates": [{ "content": { "parts": [{
                "functionCall": { "name": "shell", "args": { "cmd": "ls" } },
                "thoughtSignature": "sig-1"
            }] } }]
        });
        let out = gemini_response_to_chat(&resp, "m", 1);
        let call_id = out["choices"][0]["message"]["tool_calls"][0]["id"]
            .as_str()
            .unwrap()
            .to_string();

        let req = json!({
            "messages": [
                {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": call_id,
                        "type": "function",
                        "function": { "name": "shell", "arguments": "{\"cmd\":\"ls\"}" }
                    }]
                },
                { "role": "tool", "tool_call_id": call_id, "content": "ok" }
            ]
        });
        let body = chat_request_to_gemini(&req).unwrap();
        assert_eq!(body["contents"][0]["parts"][0]["thoughtSignature"], "sig-1");
    }

    #[test]
    fn missing_thought_signature_uses_the_skip_sentinel() {
        let id = format!("call_missing_{}", short_id());
        let req = json!({
            "messages": [{
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": id,
                    "type": "function",
                    "function": { "name": "f", "arguments": "{}" }
                }]
            }]
        });
        let body = chat_request_to_gemini(&req).unwrap();
        assert_eq!(
            body["contents"][0]["parts"][0]["thoughtSignature"],
            crate::signatures::SKIP_VALIDATOR
        );
    }

    #[test]
    fn only_the_first_parallel_tool_call_gets_a_signature() {
        let req = json!({
            "messages": [{
                "role": "assistant",
                "content": null,
                "tool_calls": [
                    {
                        "id": format!("call_par_a_{}", short_id()),
                        "type": "function",
                        "function": { "name": "a", "arguments": "{}" }
                    },
                    {
                        "id": format!("call_par_b_{}", short_id()),
                        "type": "function",
                        "function": { "name": "b", "arguments": "{}" }
                    }
                ]
            }]
        });
        let body = chat_request_to_gemini(&req).unwrap();
        assert_eq!(
            body["contents"][0]["parts"][0]["thoughtSignature"],
            crate::signatures::SKIP_VALIDATOR
        );
        assert!(
            body["contents"][0]["parts"][1]
                .get("thoughtSignature")
                .is_none()
        );
    }

    #[test]
    fn extra_content_signature_is_replayed() {
        let req = json!({
            "messages": [{
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": format!("call_extra_{}", short_id()),
                    "type": "function",
                    "extra_content": {
                        "google": { "thought_signature": "sig-extra" }
                    },
                    "function": { "name": "f", "arguments": "{}" }
                }]
            }]
        });
        let body = chat_request_to_gemini(&req).unwrap();
        assert_eq!(
            body["contents"][0]["parts"][0]["thoughtSignature"],
            "sig-extra"
        );
    }

    #[test]
    fn streaming_thought_signature_is_cached_for_replay() {
        let mut translator = StreamTranslator::default();
        let chunk = json!({
            "candidates": [{ "content": { "parts": [{
                "functionCall": { "name": "f", "args": { "a": 1 } },
                "thoughtSignature": "sig-stream"
            }] } }]
        });
        let out = translator.translate(&chunk, "id", "m", 1);
        let call_id = out[0]["choices"][0]["delta"]["tool_calls"][0]["id"]
            .as_str()
            .unwrap()
            .to_string();
        assert_eq!(
            crate::signatures::get(&call_id).as_deref(),
            Some("sig-stream")
        );
    }

    #[test]
    fn streaming_signature_in_a_followup_part_attaches_to_the_first_call() {
        let mut translator = StreamTranslator::default();
        let call = json!({
            "candidates": [{ "content": { "parts": [{
                "functionCall": { "name": "f", "args": { "a": 1 } }
            }] } }]
        });
        let first = translator.translate(&call, "id", "m", 1);
        let call_id = first[0]["choices"][0]["delta"]["tool_calls"][0]["id"]
            .as_str()
            .unwrap()
            .to_string();

        let signature = json!({
            "candidates": [{ "content": { "parts": [
                { "thoughtSignature": "sig-late" }
            ] } }]
        });
        translator.translate(&signature, "id", "m", 1);
        assert_eq!(
            crate::signatures::get(&call_id).as_deref(),
            Some("sig-late")
        );
    }

    #[test]
    fn generated_ids_are_unique() {
        assert_ne!(short_id(), short_id());
        assert_eq!(chunk_id().chars().count(), "chatcmpl-".len() + 12);
    }
}
