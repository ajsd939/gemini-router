//! Translation between the OpenAI Responses API and the Gemini wire format.
//!
//! Codex no longer accepts `wire_api = "chat"`, so the router exposes
//! `POST /v1/responses`.  Only the parts Codex actually sends are implemented:
//! instructions/messages, function and freeform (custom) tools, tool calls and
//! tool results, usage, and SSE streaming.  Hosted tools (`web_search`,
//! `local_shell`, …) and reasoning items are ignored.
//!
//! The Responses API is translated to a Chat Completions request first and then
//! through [`crate::openai::chat_request_to_gemini`], so the existing schema
//! sanitizer and Gemini message handling stay in one place.

use std::collections::{HashMap, HashSet};

use serde_json::{Map, Value, json};

/// The namespace Codex uses for ordinary (non-prefixed) tools.
const DEFAULT_NAMESPACE: &str = "functions";

/// Gemini rejects function names longer than 64 characters.
const MAX_TOOL_NAME: usize = 64;

/// Per-request tool metadata needed to translate tool calls back to Codex.
#[derive(Debug, Clone, Default)]
pub struct ToolMap {
    /// Gemini-side names of freeform (`type: "custom"`) tools.
    custom: HashSet<String>,
    /// Gemini-side name → original namespace and name.
    namespaces: HashMap<String, NamespacedTool>,
}

#[derive(Debug, Clone)]
struct NamespacedTool {
    namespace: String,
    name: String,
}

impl ToolMap {
    /// Collect custom tools and namespaces from a Responses request body.
    pub fn from_request(req: &Value) -> Self {
        let mut map = Self::default();
        let Some(tools) = req.get("tools").and_then(Value::as_array) else {
            return map;
        };
        for tool in tools {
            match tool.get("type").and_then(Value::as_str).unwrap_or("") {
                "custom" => {
                    if let Some(name) = tool.get("name").and_then(Value::as_str) {
                        map.custom.insert(name.to_string());
                    }
                }
                "namespace" => {
                    let namespace = tool
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or(DEFAULT_NAMESPACE);
                    let Some(inner) = tool.get("tools").and_then(Value::as_array) else {
                        continue;
                    };
                    for tool in inner {
                        let kind = tool.get("type").and_then(Value::as_str).unwrap_or("");
                        if kind != "function" && kind != "custom" {
                            continue;
                        }
                        let Some(name) = tool.get("name").and_then(Value::as_str) else {
                            continue;
                        };
                        let gemini = gemini_name(namespace, name);
                        if kind == "custom" {
                            map.custom.insert(gemini.clone());
                        }
                        map.namespaces.insert(
                            gemini,
                            NamespacedTool {
                                namespace: namespace.to_string(),
                                name: name.to_string(),
                            },
                        );
                    }
                }
                _ => {}
            }
        }
        map
    }

    /// The name Gemini should see for a tool.
    pub fn gemini_name(&self, namespace: Option<&str>, name: &str) -> String {
        match namespace {
            Some(namespace) if !namespace.is_empty() && namespace != DEFAULT_NAMESPACE => {
                gemini_name(namespace, name)
            }
            _ => name.to_string(),
        }
    }

    /// The namespace and original name for a Gemini function call.
    fn output_name(&self, gemini: &str) -> (Option<String>, String) {
        match self.namespaces.get(gemini) {
            Some(tool) => (Some(tool.namespace.clone()), tool.name.clone()),
            None => (None, gemini.to_string()),
        }
    }

    fn is_custom(&self, gemini: &str) -> bool {
        self.custom.contains(gemini)
    }
}

fn truncate_tool_name(name: &str) -> String {
    if name.len() <= MAX_TOOL_NAME {
        name.to_string()
    } else {
        name.chars().take(MAX_TOOL_NAME).collect()
    }
}

fn gemini_name(namespace: &str, name: &str) -> String {
    if namespace.is_empty() || namespace == DEFAULT_NAMESPACE {
        truncate_tool_name(name)
    } else {
        truncate_tool_name(&format!("{namespace}__{name}"))
    }
}

/// Translate a Responses API request into a Gemini request body.
pub fn request_to_gemini(req: &Value, tools: &ToolMap) -> Result<Value, String> {
    let mut messages: Vec<Value> = Vec::new();

    if let Some(instructions) = req.get("instructions").and_then(Value::as_str) {
        if !instructions.trim().is_empty() {
            messages.push(json!({ "role": "system", "content": instructions }));
        }
    }

    match req.get("input") {
        Some(Value::String(text)) => {
            if !text.is_empty() {
                messages.push(json!({ "role": "user", "content": text }));
            }
        }
        Some(Value::Array(items)) => {
            for item in items {
                push_input_item(item, &mut messages, tools);
            }
        }
        Some(Value::Null) | None => {}
        Some(_) => {
            return Err("`input` must be a string or an array of input items".to_string());
        }
    }

    if messages.is_empty() {
        return Err("no usable input items were found in the request".to_string());
    }

    let mut chat = Map::new();
    chat.insert("messages".into(), Value::Array(messages));

    if let Some(converted) = req.get("tools").map(|t| tools_to_chat(t, tools)) {
        if !converted.is_empty() {
            chat.insert("tools".into(), Value::Array(converted));
            if let Some(choice) = req
                .get("tool_choice")
                .and_then(|c| tool_choice_to_chat(c, tools))
            {
                chat.insert("tool_choice".into(), choice);
            }
        }
    }

    // Responses calls it `max_output_tokens`; Chat Completions calls it
    // `max_tokens`, which `generation_config` already understands.
    if let Some(max) = req.get("max_output_tokens") {
        chat.insert("max_tokens".into(), max.clone());
    }
    for key in ["temperature", "top_p"] {
        if let Some(value) = req.get(key) {
            chat.insert(key.into(), value.clone());
        }
    }

    crate::openai::chat_request_to_gemini(&Value::Object(chat))
}

fn push_input_item(item: &Value, messages: &mut Vec<Value>, tools: &ToolMap) {
    let kind = item.get("type").and_then(Value::as_str).unwrap_or("");
    match kind {
        "message" => push_message(item, messages),
        "function_call" => push_tool_call(item, messages, tools, false),
        "custom_tool_call" => push_tool_call(item, messages, tools, true),
        "function_call_output" | "custom_tool_call_output" => push_tool_output(item, messages),
        _ => {
            // Be forgiving: some clients omit `type` on ordinary messages.
            if item.get("role").is_some() && item.get("content").is_some() {
                push_message(item, messages);
            }
        }
    }
}

fn push_message(item: &Value, messages: &mut Vec<Value>) {
    let role = match item.get("role").and_then(Value::as_str).unwrap_or("user") {
        "developer" | "system" => "system",
        "assistant" => "assistant",
        "tool" | "function" => "tool",
        _ => "user",
    };
    let text = content_text(item.get("content"));
    if text.is_empty() {
        return;
    }
    messages.push(json!({ "role": role, "content": text }));
}

fn push_tool_call(item: &Value, messages: &mut Vec<Value>, tools: &ToolMap, custom: bool) {
    let original = item.get("name").and_then(Value::as_str).unwrap_or_default();
    if original.is_empty() {
        return;
    }
    let namespace = item.get("namespace").and_then(Value::as_str);
    let name = tools.gemini_name(namespace, original);
    let call_id = item
        .get("call_id")
        .and_then(Value::as_str)
        .or_else(|| item.get("id").and_then(Value::as_str))
        .unwrap_or("call_unknown")
        .to_string();

    let arguments = if custom {
        let input = item
            .get("input")
            .and_then(Value::as_str)
            .unwrap_or_default();
        json!({ "input": input }).to_string()
    } else {
        match item.get("arguments") {
            Some(Value::String(text)) => text.clone(),
            Some(value) => value.to_string(),
            None => "{}".to_string(),
        }
    };

    let call = json!({
        "id": call_id,
        "type": "function",
        "function": { "name": name, "arguments": arguments }
    });

    if let Some(last) = messages.last_mut() {
        if last.get("role").and_then(Value::as_str) == Some("assistant") {
            if let Some(object) = last.as_object_mut() {
                let calls = object.entry("tool_calls").or_insert_with(|| json!([]));
                if let Some(array) = calls.as_array_mut() {
                    array.push(call);
                }
                object.entry("content").or_insert(Value::Null);
                return;
            }
        }
    }

    messages.push(json!({
        "role": "assistant",
        "content": Value::Null,
        "tool_calls": [call]
    }));
}

fn push_tool_output(item: &Value, messages: &mut Vec<Value>) {
    let call_id = item
        .get("call_id")
        .and_then(Value::as_str)
        .or_else(|| item.get("id").and_then(Value::as_str))
        .unwrap_or_default();
    if call_id.is_empty() {
        return;
    }
    let output = output_text(item.get("output"));
    messages.push(json!({
        "role": "tool",
        "tool_call_id": call_id,
        "content": output
    }));
}

/// Flatten Responses content items (input_text/output_text/images) into text.
fn content_text(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(|item| match item {
                Value::String(text) => Some(text.clone()),
                Value::Object(object) => object
                    .get("text")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(""),
        Some(Value::Null) | None => String::new(),
        Some(other) => other.to_string(),
    }
}

fn output_text(output: Option<&Value>) -> String {
    match output {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(|item| match item {
                Value::String(text) => Some(text.clone()),
                Value::Object(object) => object
                    .get("text")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n"),
        Some(Value::Null) | None => String::new(),
        Some(other) => other.to_string(),
    }
}

fn tools_to_chat(value: &Value, tools: &ToolMap) -> Vec<Value> {
    let mut out = Vec::new();
    let Some(items) = value.as_array() else {
        return out;
    };
    for tool in items {
        match tool.get("type").and_then(Value::as_str).unwrap_or("") {
            "function" => push_function_tool(tool, None, tools, &mut out),
            "custom" => push_custom_tool(tool, None, tools, &mut out),
            "namespace" => {
                let namespace = tool
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or(DEFAULT_NAMESPACE);
                let Some(inner) = tool.get("tools").and_then(Value::as_array) else {
                    continue;
                };
                for tool in inner {
                    match tool.get("type").and_then(Value::as_str).unwrap_or("") {
                        "function" => push_function_tool(tool, Some(namespace), tools, &mut out),
                        "custom" => push_custom_tool(tool, Some(namespace), tools, &mut out),
                        _ => {}
                    }
                }
            }
            // Hosted tools (web_search, local_shell, image_generation, …) are
            // not functions Gemini can call, so skip them instead of sending an
            // invalid tool declaration.
            _ => {}
        }
    }
    out
}

fn push_function_tool(
    tool: &Value,
    namespace: Option<&str>,
    tools: &ToolMap,
    out: &mut Vec<Value>,
) {
    let Some(name) = tool.get("name").and_then(Value::as_str) else {
        return;
    };
    let name = tools.gemini_name(namespace, name);
    let description = tool
        .get("description")
        .and_then(Value::as_str)
        .unwrap_or("");
    let parameters = tool
        .get("parameters")
        .cloned()
        .unwrap_or_else(|| json!({ "type": "object", "properties": {} }));
    out.push(json!({
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters
        }
    }));
}

fn push_custom_tool(tool: &Value, namespace: Option<&str>, tools: &ToolMap, out: &mut Vec<Value>) {
    let Some(name) = tool.get("name").and_then(Value::as_str) else {
        return;
    };
    let name = tools.gemini_name(namespace, name);
    let description = tool
        .get("description")
        .and_then(Value::as_str)
        .unwrap_or("");
    out.push(json!({
        "type": "function",
        "function": {
            "name": name,
            "description": format!(
                "{description}\n\nThis is a freeform tool. Put the complete raw tool input in the `input` string; do not wrap it in JSON."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "The complete raw input for this freeform tool."
                    }
                },
                "required": ["input"]
            }
        }
    }));
}

fn tool_choice_to_chat(choice: &Value, tools: &ToolMap) -> Option<Value> {
    match choice {
        Value::String(kind) => Some(json!(kind)),
        Value::Object(object) => {
            // Accept both the Responses shape ({type, name}) and a nested
            // Chat Completions shape ({type, function: {name}}).
            let name = object.get("name").and_then(Value::as_str).or_else(|| {
                object
                    .get("function")
                    .and_then(|function| function.get("name"))
                    .and_then(Value::as_str)
            })?;
            let namespace = object.get("namespace").and_then(Value::as_str);
            let name = tools.gemini_name(namespace, name);
            Some(json!({
                "type": "function",
                "function": { "name": name }
            }))
        }
        _ => None,
    }
}

/// Convert a non-streaming Gemini response into a Responses API response.
pub fn response_from_gemini(resp: &Value, model: &str, created_at: i64, tools: &ToolMap) -> Value {
    let text = extract_text(resp);
    let calls = extract_calls(resp);
    let mut output = Vec::new();

    // A text-only turn always has a message item, even when the text is empty;
    // a tool-only turn does not need an empty assistant message.
    if !text.is_empty() || calls.is_empty() {
        output.push(message_item(&text, "completed"));
    }
    for call in &calls {
        output.push(call_item(call, tools));
    }

    response_object(
        &format!("resp_{}", crate::openai::short_id()),
        model,
        created_at,
        "completed",
        output,
        Some(&usage_object(resp)),
    )
}

/// Extract the visible answer text, skipping Gemini "thought" parts.
pub fn extract_text(resp: &Value) -> String {
    resp.pointer("/candidates/0/content/parts")
        .and_then(Value::as_array)
        .map(|parts| {
            parts
                .iter()
                .filter(|part| part.get("thought").and_then(Value::as_bool) != Some(true))
                .filter_map(|part| part.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("")
        })
        .unwrap_or_default()
}

#[derive(Debug, Clone)]
struct GeminiCall {
    name: String,
    args: Value,
    signature: Option<String>,
}

fn extract_calls(resp: &Value) -> Vec<GeminiCall> {
    resp.pointer("/candidates/0/content/parts")
        .and_then(Value::as_array)
        .map(|parts| {
            let mut calls: Vec<GeminiCall> = Vec::new();
            for part in parts {
                if let Some(call) = part.get("functionCall") {
                    let Some(name) = call.get("name").and_then(Value::as_str) else {
                        continue;
                    };
                    let args = call.get("args").cloned().unwrap_or_else(|| json!({}));
                    calls.push(GeminiCall {
                        name: name.to_string(),
                        args,
                        signature: crate::signatures::from_part(part),
                    });
                } else if let Some(signature) = crate::signatures::from_part(part) {
                    // In the non-streaming API the signature normally sits on
                    // the first function-call part, but tolerate a separate
                    // signature-only part as well.
                    if part.get("text").is_none() {
                        if let Some(first) = calls.first_mut() {
                            if first.signature.is_none() {
                                first.signature = Some(signature);
                            }
                        }
                    }
                }
            }
            calls
        })
        .unwrap_or_default()
}

fn message_item(text: &str, status: &str) -> Value {
    json!({
        "type": "message",
        "id": format!("msg_{}", crate::openai::short_id()),
        "role": "assistant",
        "status": status,
        "content": [{
            "type": "output_text",
            "text": text,
            "annotations": []
        }]
    })
}

fn call_item(call: &GeminiCall, tools: &ToolMap) -> Value {
    let (namespace, name) = tools.output_name(&call.name);
    let call_id = format!("call_{}", crate::openai::short_id());
    let item_id = if tools.is_custom(&call.name) {
        format!("ctc_{}", crate::openai::short_id())
    } else {
        format!("fc_{}", crate::openai::short_id())
    };
    if let Some(signature) = &call.signature {
        // Cache under both ids: clients may reply with `call_id` or with the
        // output item `id`.
        crate::signatures::put(&call_id, signature);
        crate::signatures::put(&item_id, signature);
    }

    let mut item = if tools.is_custom(&call.name) {
        let input = call
            .args
            .get("input")
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| serde_json::to_string(&call.args).unwrap_or_default());
        json!({
            "type": "custom_tool_call",
            "id": item_id,
            "call_id": call_id,
            "name": name,
            "input": input,
            "status": "completed"
        })
    } else {
        let arguments = serde_json::to_string(&call.args).unwrap_or_else(|_| "{}".to_string());
        json!({
            "type": "function_call",
            "id": item_id,
            "call_id": call_id,
            "name": name,
            "arguments": arguments,
            "status": "completed"
        })
    };
    if let Some(namespace) = namespace {
        item["namespace"] = json!(namespace);
    }
    item
}

fn response_object(
    id: &str,
    model: &str,
    created_at: i64,
    status: &str,
    output: Vec<Value>,
    usage: Option<&Value>,
) -> Value {
    json!({
        "id": id,
        "object": "response",
        "created_at": created_at,
        "status": status,
        "model": model,
        "output": output,
        "usage": usage.cloned().unwrap_or(Value::Null),
        "error": Value::Null,
        "incomplete_details": Value::Null,
        "instructions": Value::Null,
        "metadata": {},
        "parallel_tool_calls": true,
        "tool_choice": "auto",
        "tools": []
    })
}

/// Build the Responses `usage` object from Gemini metadata.
pub fn usage_object(resp: &Value) -> Value {
    let meta = resp.get("usageMetadata");
    let input = meta
        .and_then(|m| m.get("promptTokenCount"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let candidates = meta
        .and_then(|m| m.get("candidatesTokenCount"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let thoughts = meta
        .and_then(|m| m.get("thoughtsTokenCount"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let cached = meta
        .and_then(|m| m.get("cachedContentTokenCount"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let total = meta
        .and_then(|m| m.get("totalTokenCount"))
        .and_then(Value::as_u64)
        .unwrap_or(input + candidates + thoughts);

    json!({
        "input_tokens": input,
        "input_tokens_details": { "cached_tokens": cached },
        "output_tokens": candidates + thoughts,
        "output_tokens_details": { "reasoning_tokens": thoughts },
        "total_tokens": total
    })
}

struct TextItem {
    id: String,
    output_index: usize,
    text: String,
}

/// State carried across a streaming Responses request.
pub struct StreamTranslator {
    response_id: String,
    created_at: i64,
    model: String,
    tools: ToolMap,
    next_output_index: usize,
    text: Option<TextItem>,
    output: Vec<Value>,
    usage: Value,
    /// Gemini puts the signature on the first function-call part of a turn.
    first_call_id: Option<String>,
}

impl StreamTranslator {
    pub fn new(model: &str, tools: ToolMap) -> Self {
        Self {
            response_id: format!("resp_{}", crate::openai::short_id()),
            created_at: chrono::Utc::now().timestamp(),
            model: model.to_string(),
            tools,
            next_output_index: 0,
            text: None,
            output: Vec::new(),
            usage: usage_object(&json!({})),
            first_call_id: None,
        }
    }

    /// Events emitted before any upstream bytes arrive.
    pub fn start(&self) -> Vec<(String, Value)> {
        let response = response_object(
            &self.response_id,
            &self.model,
            self.created_at,
            "in_progress",
            Vec::new(),
            None,
        );
        vec![
            (
                "response.created".to_string(),
                json!({ "type": "response.created", "response": response.clone() }),
            ),
            (
                "response.in_progress".to_string(),
                json!({ "type": "response.in_progress", "response": response }),
            ),
        ]
    }

    /// Translate one Gemini SSE chunk into Responses events.
    pub fn translate(&mut self, chunk: &Value) -> Vec<(String, Value)> {
        let mut events = Vec::new();

        if chunk.get("usageMetadata").is_some() {
            self.usage = usage_object(chunk);
        }

        if let Some(parts) = chunk
            .pointer("/candidates/0/content/parts")
            .and_then(Value::as_array)
        {
            for part in parts {
                if part.get("thought").and_then(Value::as_bool) == Some(true) {
                    continue;
                }
                if let Some(text) = part.get("text").and_then(Value::as_str) {
                    if !text.is_empty() {
                        self.push_text(text, &mut events);
                    }
                }
                if let Some(call) = part.get("functionCall") {
                    let name = call.get("name").and_then(Value::as_str).unwrap_or_default();
                    if name.is_empty() {
                        continue;
                    }
                    // Close the text item first so its `done` event is ordered
                    // before the tool call that follows it.
                    self.finalize_text(&mut events);
                    let args = call.get("args").cloned().unwrap_or_else(|| json!({}));
                    let signature = crate::signatures::from_part(part);
                    self.push_call(name, &args, signature.as_deref(), &mut events);
                } else if let Some(signature) = crate::signatures::from_part(part) {
                    if part.get("text").is_none() {
                        if let Some(call_id) = &self.first_call_id {
                            crate::signatures::put(call_id, &signature);
                        }
                    }
                }
            }
        }

        events
    }

    /// Emit the closing events, including the final `response.completed`.
    pub fn finish(&mut self) -> Vec<(String, Value)> {
        let mut events = Vec::new();
        self.finalize_text(&mut events);

        // Codex expects a completed response even when the model produced
        // nothing.  An empty message keeps the turn well formed.
        if self.output.is_empty() {
            let item = message_item("", "completed");
            let index = self.next_output_index;
            self.next_output_index += 1;
            events.push((
                "response.output_item.done".to_string(),
                json!({ "type": "response.output_item.done", "output_index": index, "item": item.clone() }),
            ));
            self.output.push(item);
        }

        let response = response_object(
            &self.response_id,
            &self.model,
            self.created_at,
            "completed",
            self.output.clone(),
            Some(&self.usage),
        );
        events.push((
            "response.completed".to_string(),
            json!({ "type": "response.completed", "response": response }),
        ));
        events
    }

    fn push_text(&mut self, delta: &str, events: &mut Vec<(String, Value)>) {
        if self.text.is_none() {
            let id = format!("msg_{}", crate::openai::short_id());
            let index = self.next_output_index;
            self.next_output_index += 1;
            events.push((
                "response.output_item.added".to_string(),
                json!({
                    "type": "response.output_item.added",
                    "output_index": index,
                    "item": {
                        "type": "message",
                        "id": id,
                        "role": "assistant",
                        "status": "in_progress",
                        "content": []
                    }
                }),
            ));
            events.push((
                "response.content_part.added".to_string(),
                json!({
                    "type": "response.content_part.added",
                    "item_id": id,
                    "output_index": index,
                    "content_index": 0,
                    "part": { "type": "output_text", "text": "", "annotations": [] }
                }),
            ));
            self.text = Some(TextItem {
                id,
                output_index: index,
                text: String::new(),
            });
        }
        let Some(item) = self.text.as_mut() else {
            return;
        };
        item.text.push_str(delta);
        events.push((
            "response.output_text.delta".to_string(),
            json!({
                "type": "response.output_text.delta",
                "item_id": item.id,
                "output_index": item.output_index,
                "content_index": 0,
                "delta": delta
            }),
        ));
    }

    fn finalize_text(&mut self, events: &mut Vec<(String, Value)>) {
        let Some(item) = self.text.take() else {
            return;
        };
        let content = json!([{ "type": "output_text", "text": item.text, "annotations": [] }]);
        events.push((
            "response.output_text.done".to_string(),
            json!({
                "type": "response.output_text.done",
                "item_id": item.id,
                "output_index": item.output_index,
                "content_index": 0,
                "text": item.text
            }),
        ));
        events.push((
            "response.content_part.done".to_string(),
            json!({
                "type": "response.content_part.done",
                "item_id": item.id,
                "output_index": item.output_index,
                "content_index": 0,
                "part": { "type": "output_text", "text": item.text, "annotations": [] }
            }),
        ));
        let done = json!({
            "type": "message",
            "id": item.id,
            "role": "assistant",
            "status": "completed",
            "content": content
        });
        events.push((
            "response.output_item.done".to_string(),
            json!({
                "type": "response.output_item.done",
                "output_index": item.output_index,
                "item": done.clone()
            }),
        ));
        self.output.push(done);
    }

    fn push_call(
        &mut self,
        gemini_name: &str,
        args: &Value,
        signature: Option<&str>,
        events: &mut Vec<(String, Value)>,
    ) {
        let (namespace, name) = self.tools.output_name(gemini_name);
        let call_id = format!("call_{}", crate::openai::short_id());
        let index = self.next_output_index;
        self.next_output_index += 1;
        let custom = self.tools.is_custom(gemini_name);
        let item_id = if custom {
            format!("ctc_{}", crate::openai::short_id())
        } else {
            format!("fc_{}", crate::openai::short_id())
        };
        if let Some(signature) = signature {
            // Cache under both ids: clients may reply with `call_id` or with
            // the output item `id`.
            crate::signatures::put(&call_id, signature);
            crate::signatures::put(&item_id, signature);
        }
        if self.first_call_id.is_none() {
            self.first_call_id = Some(call_id.clone());
        }

        if custom {
            let input = args
                .get("input")
                .and_then(Value::as_str)
                .map(str::to_string)
                .unwrap_or_else(|| serde_json::to_string(args).unwrap_or_default());
            let mut added = json!({
                "type": "custom_tool_call",
                "id": item_id,
                "call_id": call_id,
                "name": name,
                "input": "",
                "status": "in_progress"
            });
            if let Some(namespace) = &namespace {
                added["namespace"] = json!(namespace);
            }
            events.push((
                "response.output_item.added".to_string(),
                json!({ "type": "response.output_item.added", "output_index": index, "item": added }),
            ));
            events.push((
                "response.custom_tool_call_input.delta".to_string(),
                json!({
                    "type": "response.custom_tool_call_input.delta",
                    "item_id": added["id"],
                    "call_id": added["call_id"],
                    "output_index": index,
                    "delta": input
                }),
            ));
            let mut done = added.clone();
            done["input"] = json!(input);
            done["status"] = json!("completed");
            events.push((
                "response.output_item.done".to_string(),
                json!({ "type": "response.output_item.done", "output_index": index, "item": done.clone() }),
            ));
            self.output.push(done);
        } else {
            let arguments = serde_json::to_string(args).unwrap_or_else(|_| "{}".to_string());
            let mut added = json!({
                "type": "function_call",
                "id": item_id,
                "call_id": call_id,
                "name": name,
                "arguments": "",
                "status": "in_progress"
            });
            if let Some(namespace) = &namespace {
                added["namespace"] = json!(namespace);
            }
            events.push((
                "response.output_item.added".to_string(),
                json!({ "type": "response.output_item.added", "output_index": index, "item": added }),
            ));
            events.push((
                "response.function_call_arguments.delta".to_string(),
                json!({
                    "type": "response.function_call_arguments.delta",
                    "item_id": added["id"],
                    "output_index": index,
                    "delta": arguments
                }),
            ));
            let mut done = added.clone();
            done["arguments"] = json!(arguments);
            done["status"] = json!("completed");
            events.push((
                "response.output_item.done".to_string(),
                json!({ "type": "response.output_item.done", "output_index": index, "item": done.clone() }),
            ));
            self.output.push(done);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_input_becomes_a_user_message() {
        let req = json!({ "input": "hello" });
        let body = request_to_gemini(&req, &ToolMap::default()).unwrap();
        assert_eq!(body["contents"][0]["role"], "user");
        assert_eq!(body["contents"][0]["parts"][0]["text"], "hello");
    }

    #[test]
    fn instructions_become_a_system_instruction() {
        let req = json!({
            "instructions": "be terse",
            "input": [{ "type": "message", "role": "user", "content": [{ "type": "input_text", "text": "hi" }] }]
        });
        let body = request_to_gemini(&req, &ToolMap::default()).unwrap();
        assert_eq!(body["systemInstruction"]["parts"][0]["text"], "be terse");
        assert_eq!(body["contents"][0]["parts"][0]["text"], "hi");
    }

    #[test]
    fn function_tools_are_nested_for_chat_completions() {
        let req = json!({
            "input": "hi",
            "tools": [{
                "type": "function",
                "name": "shell",
                "description": "run a command",
                "parameters": { "type": "object", "properties": { "cmd": { "type": "string" } } }
            }]
        });
        let map = ToolMap::from_request(&req);
        let body = request_to_gemini(&req, &map).unwrap();
        let decl = &body["tools"][0]["functionDeclarations"][0];
        assert_eq!(decl["name"], "shell");
        assert_eq!(decl["parameters"]["properties"]["cmd"]["type"], "string");
    }

    #[test]
    fn custom_tools_become_single_string_functions() {
        let req = json!({
            "input": "hi",
            "tools": [{
                "type": "custom",
                "name": "apply_patch",
                "description": "edit files",
                "format": { "type": "grammar", "syntax": "lark", "definition": "..." }
            }]
        });
        let map = ToolMap::from_request(&req);
        assert!(map.is_custom("apply_patch"));
        let body = request_to_gemini(&req, &map).unwrap();
        let decl = &body["tools"][0]["functionDeclarations"][0];
        assert_eq!(decl["name"], "apply_patch");
        assert_eq!(decl["parameters"]["required"][0], "input");
    }

    #[test]
    fn function_call_history_becomes_an_assistant_tool_call() {
        let req = json!({
            "input": [
                { "type": "message", "role": "user", "content": "run it" },
                { "type": "function_call", "name": "shell", "arguments": "{\"cmd\":\"ls\"}", "call_id": "call_1" },
                { "type": "function_call_output", "call_id": "call_1", "output": "a.txt" }
            ]
        });
        let body = request_to_gemini(&req, &ToolMap::default()).unwrap();
        let contents = body["contents"].as_array().unwrap();
        assert_eq!(contents[1]["role"], "model");
        assert_eq!(contents[1]["parts"][0]["functionCall"]["name"], "shell");
        assert_eq!(contents[2]["role"], "user");
        assert_eq!(
            contents[2]["parts"][0]["functionResponse"]["response"]["content"],
            "a.txt"
        );
    }

    #[test]
    fn response_text_is_a_message_item() {
        let resp = json!({
            "candidates": [{ "content": { "parts": [{ "text": "hello" }] } }],
            "usageMetadata": { "promptTokenCount": 3, "candidatesTokenCount": 2, "totalTokenCount": 5 }
        });
        let out = response_from_gemini(&resp, "flash", 1, &ToolMap::default());
        assert_eq!(out["object"], "response");
        assert_eq!(out["output"][0]["type"], "message");
        assert_eq!(out["output"][0]["content"][0]["text"], "hello");
        assert_eq!(out["usage"]["input_tokens"], 3);
        assert_eq!(out["usage"]["output_tokens"], 2);
        assert_eq!(out["usage"]["total_tokens"], 5);
    }

    #[test]
    fn response_custom_call_uses_input_string() {
        let req = json!({
            "input": "hi",
            "tools": [{ "type": "custom", "name": "apply_patch", "description": "edit" }]
        });
        let map = ToolMap::from_request(&req);
        let resp = json!({
            "candidates": [{ "content": { "parts": [{
                "functionCall": { "name": "apply_patch", "args": { "input": "*** Begin Patch" } }
            }] } }]
        });
        let out = response_from_gemini(&resp, "flash", 1, &map);
        assert_eq!(out["output"][0]["type"], "custom_tool_call");
        assert_eq!(out["output"][0]["input"], "*** Begin Patch");
    }

    #[test]
    fn streaming_closes_text_before_tool_calls() {
        let mut translator = StreamTranslator::new("flash", ToolMap::default());
        let mut events = translator.start();
        events.extend(translator.translate(&json!({
            "candidates": [{ "content": { "parts": [{ "text": "thinking" }] } }]
        })));
        events.extend(translator.translate(&json!({
            "candidates": [{ "content": { "parts": [{
                "functionCall": { "name": "shell", "args": { "cmd": "ls" } }
            }] } }]
        })));
        events.extend(translator.finish());

        let text_done = events
            .iter()
            .position(|(kind, _)| kind == "response.output_text.done")
            .unwrap();
        let call_added = events
            .iter()
            .position(|(kind, data)| {
                kind == "response.output_item.added" && data["item"]["type"] == "function_call"
            })
            .unwrap();
        assert!(text_done < call_added);
        assert!(events.iter().any(|(kind, _)| kind == "response.completed"));
    }

    #[test]
    fn streaming_completed_has_required_usage_fields() {
        let mut translator = StreamTranslator::new("flash", ToolMap::default());
        translator.translate(&json!({
            "usageMetadata": { "promptTokenCount": 7, "candidatesTokenCount": 4, "totalTokenCount": 11 }
        }));
        let events = translator.finish();
        let (_, completed) = events
            .iter()
            .find(|(kind, _)| kind == "response.completed")
            .unwrap();
        let usage = &completed["response"]["usage"];
        assert_eq!(usage["input_tokens"], 7);
        assert_eq!(usage["output_tokens"], 4);
        assert_eq!(usage["total_tokens"], 11);
        assert!(usage["output_tokens_details"]["reasoning_tokens"].is_number());
    }

    #[test]
    fn response_function_call_signature_is_cached_and_replayed() {
        let resp = json!({
            "candidates": [{ "content": { "parts": [{
                "functionCall": { "name": "shell", "args": { "cmd": "ls" } },
                "thoughtSignature": "sig-r"
            }] } }]
        });
        let out = response_from_gemini(&resp, "flash", 1, &ToolMap::default());
        let item = &out["output"][0];
        let call_id = item["call_id"].as_str().unwrap().to_string();
        let item_id = item["id"].as_str().unwrap().to_string();
        assert_eq!(crate::signatures::get(&call_id).as_deref(), Some("sig-r"));
        assert_eq!(crate::signatures::get(&item_id).as_deref(), Some("sig-r"));

        let req = json!({
            "input": [
                { "type": "message", "role": "user", "content": "run" },
                { "type": "function_call", "name": "shell", "arguments": "{\"cmd\":\"ls\"}", "call_id": call_id },
                { "type": "function_call_output", "call_id": call_id, "output": "ok" }
            ]
        });
        let body = request_to_gemini(&req, &ToolMap::default()).unwrap();
        assert_eq!(body["contents"][1]["parts"][0]["thoughtSignature"], "sig-r");
    }

    #[test]
    fn streaming_response_caches_signature_for_replay() {
        let mut translator = StreamTranslator::new("flash", ToolMap::default());
        let events = translator.translate(&json!({
            "candidates": [{ "content": { "parts": [{
                "functionCall": { "name": "shell", "args": { "cmd": "ls" } },
                "thoughtSignature": "sig-rs"
            }] } }]
        }));
        let (_, done) = events
            .iter()
            .find(|(kind, data)| {
                kind == "response.output_item.done" && data["item"]["type"] == "function_call"
            })
            .unwrap();
        let call_id = done["item"]["call_id"].as_str().unwrap();
        assert_eq!(crate::signatures::get(call_id).as_deref(), Some("sig-rs"));
    }
}
