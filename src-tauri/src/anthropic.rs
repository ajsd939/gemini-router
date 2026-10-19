//! Translation between the Anthropic Messages API and the Gemini wire format.
//!
//! Claude Code speaks the Anthropic Messages API. This module is the bridge used
//! by the local server's `/v1/messages` endpoint: it translates text, images,
//! tool_use/tool_result blocks, system prompts, generation options and streaming
//! SSE events. Anything Claude Code sends that Gemini has no concept of (for
//! example thinking signatures or cache_control) is ignored.

use std::collections::HashMap;

use serde_json::{Map, Value, json};

use crate::openai::{clean_schema, content_to_text, push_content, short_id};

/// One Anthropic SSE event: `(event name, JSON data)`.
pub type Event = (String, Value);

fn content_blocks(content: Option<&Value>) -> Vec<Value> {
    match content {
        Some(Value::String(text)) => vec![json!({ "type": "text", "text": text })],
        Some(Value::Array(items)) => items.clone(),
        _ => Vec::new(),
    }
}

fn system_parts(system: Option<&Value>) -> Vec<Value> {
    match system {
        Some(Value::String(text)) if !text.is_empty() => vec![json!({ "text": text })],
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(|block| {
                if block.get("type").and_then(Value::as_str).unwrap_or("text") != "text" {
                    return None;
                }
                block
                    .get("text")
                    .and_then(Value::as_str)
                    .filter(|text| !text.is_empty())
                    .map(|text| json!({ "text": text }))
            })
            .collect(),
        _ => Vec::new(),
    }
}

fn image_part(block: &Value) -> Option<Value> {
    let source = block.get("source")?;
    if source.get("type").and_then(Value::as_str) != Some("base64") {
        // Gemini has no remote-URL image part; only inline base64 is supported.
        return None;
    }
    let mime_type = source.get("media_type").and_then(Value::as_str)?;
    let data = source.get("data").and_then(Value::as_str)?;
    Some(json!({ "inlineData": { "mimeType": mime_type, "data": data } }))
}

fn tool_result_part(block: &Value, id_to_name: &HashMap<String, String>) -> Value {
    let id = block
        .get("tool_use_id")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let name = id_to_name
        .get(id)
        .cloned()
        .or_else(|| {
            block
                .get("name")
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .unwrap_or_else(|| "tool".to_string());

    let text = content_to_text(block.get("content"));
    let is_error = block
        .get("is_error")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let response = if is_error {
        json!({ "error": text })
    } else {
        serde_json::from_str::<Value>(&text)
            .ok()
            .filter(|value| value.is_object())
            .unwrap_or_else(|| json!({ "content": text }))
    };

    json!({ "functionResponse": { "name": name, "response": response } })
}

fn tool_config(choice: &Value) -> Option<Value> {
    let (mode, allowed) = match choice {
        Value::String(value) => match value.as_str() {
            "none" => ("NONE", None),
            "any" | "required" => ("ANY", None),
            _ => ("AUTO", None),
        },
        Value::Object(object) => match object.get("type").and_then(Value::as_str).unwrap_or("auto")
        {
            "none" => ("NONE", None),
            "any" => ("ANY", None),
            "tool" => {
                let names = object
                    .get("name")
                    .and_then(Value::as_str)
                    .map(|name| vec![name.to_string()]);
                ("ANY", names)
            }
            _ => ("AUTO", None),
        },
        _ => return None,
    };

    let mut calling = Map::new();
    calling.insert("mode".into(), json!(mode));
    if let Some(names) = allowed {
        calling.insert("allowedFunctionNames".into(), json!(names));
    }
    Some(json!({ "toolConfig": { "functionCallingConfig": calling } }))
}

fn tools_to_gemini(tools: Option<&Value>) -> Option<Value> {
    let declarations: Vec<Value> = tools
        .and_then(Value::as_array)
        .map(|tools| {
            tools
                .iter()
                .filter_map(|tool| {
                    let name = tool.get("name").and_then(Value::as_str)?;
                    let mut declaration = Map::new();
                    declaration.insert("name".into(), json!(name));
                    if let Some(description) = tool.get("description").and_then(Value::as_str) {
                        declaration.insert("description".into(), json!(description));
                    }
                    if let Some(schema) = tool.get("input_schema") {
                        declaration.insert("parameters".into(), clean_schema(schema));
                    }
                    Some(Value::Object(declaration))
                })
                .collect()
        })
        .unwrap_or_default();

    if declarations.is_empty() {
        None
    } else {
        Some(json!([{ "functionDeclarations": declarations }]))
    }
}

fn generation_config(req: &Value) -> Option<Value> {
    let mut generation = Map::new();
    if let Some(max) = req.get("max_tokens").and_then(Value::as_u64) {
        generation.insert("maxOutputTokens".into(), json!(max));
    }
    if let Some(temperature) = req.get("temperature").and_then(Value::as_f64) {
        generation.insert("temperature".into(), json!(temperature));
    }
    if let Some(top_p) = req.get("top_p").and_then(Value::as_f64) {
        generation.insert("topP".into(), json!(top_p));
    }
    if let Some(top_k) = req.get("top_k").and_then(Value::as_u64) {
        generation.insert("topK".into(), json!(top_k));
    }
    if let Some(stops) = req.get("stop_sequences").and_then(Value::as_array) {
        let stops: Vec<&str> = stops.iter().filter_map(Value::as_str).collect();
        if !stops.is_empty() {
            generation.insert("stopSequences".into(), json!(stops));
        }
    }
    if generation.is_empty() {
        None
    } else {
        Some(Value::Object(generation))
    }
}

/// Translate an Anthropic `/v1/messages` request into a Gemini request body.
pub fn request_to_gemini(req: &Value) -> Result<Value, String> {
    let messages = req
        .get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| "`messages` must be a non-empty array".to_string())?;
    if messages.is_empty() {
        return Err("`messages` must be a non-empty array".to_string());
    }

    let mut contents: Vec<Value> = Vec::new();
    // Anthropic correlates tool_result blocks with tool_use ids; Gemini needs
    // the function name instead, so remember the mapping while walking history.
    let mut id_to_name: HashMap<String, String> = HashMap::new();

    for message in messages {
        let role = message
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or("user");
        let blocks = content_blocks(message.get("content"));

        match role {
            "assistant" => {
                let mut parts = Vec::new();
                for block in &blocks {
                    match block.get("type").and_then(Value::as_str).unwrap_or("") {
                        "text" => {
                            if let Some(text) = block.get("text").and_then(Value::as_str) {
                                if !text.is_empty() {
                                    parts.push(json!({ "text": text }));
                                }
                            }
                        }
                        "tool_use" => {
                            let Some(name) = block.get("name").and_then(Value::as_str) else {
                                continue;
                            };
                            let id = block
                                .get("id")
                                .and_then(Value::as_str)
                                .unwrap_or_default()
                                .to_string();
                            if !id.is_empty() {
                                id_to_name.insert(id, name.to_string());
                            }
                            let args = block.get("input").cloned().unwrap_or_else(|| json!({}));
                            parts.push(json!({
                                "functionCall": { "name": name, "args": args }
                            }));
                        }
                        // Thinking blocks are Anthropic-specific; Gemini does not
                        // accept them back as history.
                        "thinking" | "redacted_thinking" => {}
                        _ => {}
                    }
                }
                push_content(&mut contents, "model", parts);
            }
            "user" => {
                let mut parts = Vec::new();
                for block in &blocks {
                    match block.get("type").and_then(Value::as_str).unwrap_or("") {
                        "text" => {
                            if let Some(text) = block.get("text").and_then(Value::as_str) {
                                if !text.is_empty() {
                                    parts.push(json!({ "text": text }));
                                }
                            }
                        }
                        "image" => {
                            if let Some(part) = image_part(block) {
                                parts.push(part);
                            }
                        }
                        "tool_result" => {
                            parts.push(tool_result_part(block, &id_to_name));
                        }
                        _ => {}
                    }
                }
                push_content(&mut contents, "user", parts);
            }
            _ => {}
        }
    }

    if contents.is_empty() {
        return Err("no usable messages were found in the request".to_string());
    }

    let mut body = Map::new();
    body.insert("contents".into(), Value::Array(contents));

    let system = system_parts(req.get("system"));
    if !system.is_empty() {
        body.insert("systemInstruction".into(), json!({ "parts": system }));
    }
    if let Some(generation) = generation_config(req) {
        body.insert("generationConfig".into(), generation);
    }
    if let Some(tools) = tools_to_gemini(req.get("tools")) {
        body.insert("tools".into(), tools);
        if let Some(choice) = req.get("tool_choice") {
            if let Some(config) = tool_config(choice) {
                for (key, value) in config.as_object().into_iter().flatten() {
                    body.insert(key.clone(), value.clone());
                }
            }
        }
    }

    Ok(Value::Object(body))
}

/// Rough local token estimate for `/v1/messages/count_tokens`.
pub fn estimate_input_tokens(req: &Value) -> u64 {
    let text = serde_json::to_string(req).unwrap_or_default();
    let chars = text.chars().count() as u64;
    (chars / 4).max(1)
}

fn stop_reason(resp: &Value, has_tool_use: bool) -> &'static str {
    if has_tool_use {
        return "tool_use";
    }
    if resp.pointer("/candidates/0").is_none()
        && resp.pointer("/promptFeedback/blockReason").is_some()
    {
        return "refusal";
    }
    match resp
        .pointer("/candidates/0/finishReason")
        .and_then(Value::as_str)
    {
        Some("MAX_TOKENS") => "max_tokens",
        Some("SAFETY")
        | Some("RECITATION")
        | Some("BLOCKLIST")
        | Some("PROHIBITED_CONTENT")
        | Some("SPII")
        | Some("IMAGE_SAFETY") => "refusal",
        _ => "end_turn",
    }
}

fn usage(resp: &Value) -> Value {
    let meta = resp.get("usageMetadata");
    let input = meta
        .and_then(|m| m.get("promptTokenCount"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let output = meta
        .and_then(|m| m.get("candidatesTokenCount"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    json!({ "input_tokens": input, "output_tokens": output })
}

/// Translate one Gemini response into an Anthropic `message` object.
pub fn response_to_anthropic(resp: &Value, model: &str) -> Value {
    let mut content = Vec::new();
    let mut has_tool_use = false;

    if let Some(parts) = resp
        .pointer("/candidates/0/content/parts")
        .and_then(Value::as_array)
    {
        for part in parts {
            if let Some(text) = part.get("text").and_then(Value::as_str) {
                if !text.is_empty() {
                    content.push(json!({ "type": "text", "text": text }));
                }
            }
            if let Some(call) = part.get("functionCall") {
                let name = call.get("name").and_then(Value::as_str).unwrap_or("tool");
                let input = call.get("args").cloned().unwrap_or_else(|| json!({}));
                content.push(json!({
                    "type": "tool_use",
                    "id": format!("toolu_{}", short_id()),
                    "name": name,
                    "input": input
                }));
                has_tool_use = true;
            }
        }
    }

    json!({
        "id": format!("msg_{}", short_id()),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason(resp, has_tool_use),
        "stop_sequence": Value::Null,
        "usage": usage(resp)
    })
}

/// Streaming translator producing Anthropic SSE events.
pub struct StreamTranslator {
    message_id: String,
    model: String,
    started: bool,
    finished: bool,
    next_index: usize,
    open_text: Option<usize>,
    emitted_tool: bool,
    input_tokens: u64,
    output_tokens: u64,
}

impl StreamTranslator {
    pub fn new(model: &str) -> Self {
        StreamTranslator {
            message_id: format!("msg_{}", short_id()),
            model: model.to_string(),
            started: false,
            finished: false,
            next_index: 0,
            open_text: None,
            emitted_tool: false,
            input_tokens: 0,
            output_tokens: 0,
        }
    }

    pub fn is_finished(&self) -> bool {
        self.finished
    }

    pub fn start(&mut self) -> Vec<Event> {
        if self.started {
            return Vec::new();
        }
        self.started = true;
        let message = json!({
            "id": self.message_id,
            "type": "message",
            "role": "assistant",
            "model": self.model,
            "content": [],
            "stop_reason": Value::Null,
            "stop_sequence": Value::Null,
            "usage": { "input_tokens": self.input_tokens, "output_tokens": 0 }
        });
        vec![(
            "message_start".to_string(),
            json!({ "type": "message_start", "message": message }),
        )]
    }

    fn update_usage(&mut self, chunk: &Value) {
        if let Some(meta) = chunk.get("usageMetadata") {
            if let Some(input) = meta.get("promptTokenCount").and_then(Value::as_u64) {
                self.input_tokens = input;
            }
            if let Some(output) = meta.get("candidatesTokenCount").and_then(Value::as_u64) {
                self.output_tokens = output;
            }
        }
    }

    fn ensure_text_block(&mut self, out: &mut Vec<Event>) -> usize {
        if let Some(index) = self.open_text {
            return index;
        }
        let index = self.next_index;
        self.next_index += 1;
        self.open_text = Some(index);
        out.push((
            "content_block_start".to_string(),
            json!({
                "type": "content_block_start",
                "index": index,
                "content_block": { "type": "text", "text": "" }
            }),
        ));
        index
    }

    fn close_text_block(&mut self, out: &mut Vec<Event>) {
        if let Some(index) = self.open_text.take() {
            out.push((
                "content_block_stop".to_string(),
                json!({ "type": "content_block_stop", "index": index }),
            ));
        }
    }

    fn finalize(&mut self, reason: Option<&str>, out: &mut Vec<Event>) {
        if self.finished {
            return;
        }
        self.finished = true;
        let stop = if self.emitted_tool {
            "tool_use"
        } else {
            match reason {
                Some("MAX_TOKENS") => "max_tokens",
                Some("SAFETY")
                | Some("RECITATION")
                | Some("BLOCKLIST")
                | Some("PROHIBITED_CONTENT")
                | Some("SPII")
                | Some("IMAGE_SAFETY") => "refusal",
                _ => "end_turn",
            }
        };
        out.push((
            "message_delta".to_string(),
            json!({
                "type": "message_delta",
                "delta": { "stop_reason": stop, "stop_sequence": Value::Null },
                "usage": { "output_tokens": self.output_tokens }
            }),
        ));
        out.push((
            "message_stop".to_string(),
            json!({ "type": "message_stop" }),
        ));
    }

    /// Translate one Gemini SSE chunk into Anthropic events.
    pub fn translate(&mut self, chunk: &Value) -> Vec<Event> {
        let mut out = self.start();
        if self.finished {
            return out;
        }
        self.update_usage(chunk);

        if let Some(parts) = chunk
            .pointer("/candidates/0/content/parts")
            .and_then(Value::as_array)
        {
            for part in parts {
                if let Some(text) = part.get("text").and_then(Value::as_str) {
                    if !text.is_empty() {
                        let index = self.ensure_text_block(&mut out);
                        out.push((
                            "content_block_delta".to_string(),
                            json!({
                                "type": "content_block_delta",
                                "index": index,
                                "delta": { "type": "text_delta", "text": text }
                            }),
                        ));
                    }
                }
                if let Some(call) = part.get("functionCall") {
                    self.close_text_block(&mut out);
                    let index = self.next_index;
                    self.next_index += 1;
                    self.emitted_tool = true;
                    let name = call.get("name").and_then(Value::as_str).unwrap_or("tool");
                    let id = format!("toolu_{}", short_id());
                    out.push((
                        "content_block_start".to_string(),
                        json!({
                            "type": "content_block_start",
                            "index": index,
                            "content_block": {
                                "type": "tool_use",
                                "id": id,
                                "name": name,
                                "input": {}
                            }
                        }),
                    ));
                    let args = call.get("args").cloned().unwrap_or_else(|| json!({}));
                    let partial = serde_json::to_string(&args).unwrap_or_else(|_| "{}".into());
                    out.push((
                        "content_block_delta".to_string(),
                        json!({
                            "type": "content_block_delta",
                            "index": index,
                            "delta": { "type": "input_json_delta", "partial_json": partial }
                        }),
                    ));
                    out.push((
                        "content_block_stop".to_string(),
                        json!({ "type": "content_block_stop", "index": index }),
                    ));
                }
            }
        }

        if let Some(reason) = chunk
            .pointer("/candidates/0/finishReason")
            .and_then(Value::as_str)
        {
            self.close_text_block(&mut out);
            self.finalize(Some(reason), &mut out);
        }

        out
    }

    /// Close the stream if Gemini ended without a finishReason.
    pub fn finish(&mut self) -> Vec<Event> {
        let mut out = self.start();
        if self.finished {
            return out;
        }
        self.close_text_block(&mut out);
        self.finalize(None, &mut out);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_and_system_are_translated() {
        let req = json!({
            "model": "claude-sonnet-4-5",
            "max_tokens": 100,
            "system": "be terse",
            "messages": [{ "role": "user", "content": "hello" }]
        });
        let body = request_to_gemini(&req).unwrap();
        assert_eq!(body["contents"][0]["role"], "user");
        assert_eq!(body["contents"][0]["parts"][0]["text"], "hello");
        assert_eq!(body["systemInstruction"]["parts"][0]["text"], "be terse");
        assert_eq!(body["generationConfig"]["maxOutputTokens"], 100);
    }

    #[test]
    fn tool_use_history_maps_ids_back_to_names() {
        let req = json!({
            "max_tokens": 100,
            "messages": [
                { "role": "user", "content": "weather?" },
                { "role": "assistant", "content": [
                    { "type": "text", "text": "checking" },
                    { "type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": { "city": "Paris" } }
                ]},
                { "role": "user", "content": [
                    { "type": "tool_result", "tool_use_id": "toolu_1", "content": "{\"temp\":21}" }
                ]}
            ]
        });
        let body = request_to_gemini(&req).unwrap();
        let contents = body["contents"].as_array().unwrap();
        assert_eq!(contents.len(), 3);
        assert_eq!(contents[1]["role"], "model");
        assert_eq!(
            contents[1]["parts"][1]["functionCall"]["name"],
            "get_weather"
        );
        assert_eq!(
            contents[1]["parts"][1]["functionCall"]["args"]["city"],
            "Paris"
        );
        assert_eq!(
            contents[2]["parts"][0]["functionResponse"]["name"],
            "get_weather"
        );
        assert_eq!(
            contents[2]["parts"][0]["functionResponse"]["response"]["temp"],
            21
        );
    }

    #[test]
    fn parallel_tool_results_share_one_user_turn() {
        let req = json!({
            "max_tokens": 100,
            "messages": [
                { "role": "user", "content": "both?" },
                { "role": "assistant", "content": [
                    { "type": "tool_use", "id": "a", "name": "f", "input": {} },
                    { "type": "tool_use", "id": "b", "name": "g", "input": {} }
                ]},
                { "role": "user", "content": [
                    { "type": "tool_result", "tool_use_id": "a", "content": "1" },
                    { "type": "tool_result", "tool_use_id": "b", "content": "2" }
                ]}
            ]
        });
        let body = request_to_gemini(&req).unwrap();
        let contents = body["contents"].as_array().unwrap();
        assert_eq!(contents.len(), 3);
        assert_eq!(contents[2]["role"], "user");
        assert_eq!(contents[2]["parts"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn tools_and_tool_choice_are_translated() {
        let req = json!({
            "max_tokens": 100,
            "messages": [{ "role": "user", "content": "hi" }],
            "tools": [{
                "name": "get_weather",
                "description": "Weather",
                "input_schema": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "additionalProperties": false,
                    "properties": { "city": { "type": "string" } },
                    "required": ["city"]
                }
            }],
            "tool_choice": { "type": "tool", "name": "get_weather" }
        });
        let body = request_to_gemini(&req).unwrap();
        let declaration = &body["tools"][0]["functionDeclarations"][0];
        assert_eq!(declaration["name"], "get_weather");
        assert!(declaration["parameters"].get("$schema").is_none());
        assert!(
            declaration["parameters"]
                .get("additionalProperties")
                .is_none()
        );
        assert_eq!(body["toolConfig"]["functionCallingConfig"]["mode"], "ANY");
        assert_eq!(
            body["toolConfig"]["functionCallingConfig"]["allowedFunctionNames"][0],
            "get_weather"
        );
    }

    #[test]
    fn image_base64_becomes_inline_data() {
        let req = json!({
            "max_tokens": 10,
            "messages": [{ "role": "user", "content": [
                { "type": "text", "text": "what is this?" },
                { "type": "image", "source": { "type": "base64", "media_type": "image/png", "data": "AAAA" } }
            ]}]
        });
        let body = request_to_gemini(&req).unwrap();
        assert_eq!(
            body["contents"][0]["parts"][1]["inlineData"]["mimeType"],
            "image/png"
        );
        assert_eq!(
            body["contents"][0]["parts"][1]["inlineData"]["data"],
            "AAAA"
        );
    }

    #[test]
    fn response_text_maps_to_anthropic_message() {
        let resp = json!({
            "candidates": [{ "content": { "parts": [{ "text": "hello" }] }, "finishReason": "STOP" }],
            "usageMetadata": { "promptTokenCount": 7, "candidatesTokenCount": 2 }
        });
        let out = response_to_anthropic(&resp, "claude-sonnet-4-5");
        assert_eq!(out["type"], "message");
        assert_eq!(out["content"][0]["type"], "text");
        assert_eq!(out["content"][0]["text"], "hello");
        assert_eq!(out["stop_reason"], "end_turn");
        assert_eq!(out["usage"]["input_tokens"], 7);
        assert_eq!(out["usage"]["output_tokens"], 2);
    }

    #[test]
    fn response_tool_call_maps_to_tool_use() {
        let resp = json!({
            "candidates": [{ "content": { "parts": [
                { "functionCall": { "name": "get_weather", "args": { "city": "Paris" } } }
            ]}, "finishReason": "STOP" }]
        });
        let out = response_to_anthropic(&resp, "m");
        assert_eq!(out["stop_reason"], "tool_use");
        assert_eq!(out["content"][0]["type"], "tool_use");
        assert_eq!(out["content"][0]["name"], "get_weather");
        assert_eq!(out["content"][0]["input"]["city"], "Paris");
        assert!(
            out["content"][0]["id"]
                .as_str()
                .unwrap()
                .starts_with("toolu_")
        );
    }

    #[test]
    fn streaming_text_emits_anthropic_events() {
        let mut translator = StreamTranslator::new("m");
        let start = translator.start();
        assert_eq!(start[0].0, "message_start");

        let chunk = json!({ "candidates": [{ "content": { "parts": [{ "text": "Hel" }] } }] });
        let events = translator.translate(&chunk);
        let names: Vec<&str> = events.iter().map(|(name, _)| name.as_str()).collect();
        assert_eq!(names, vec!["content_block_start", "content_block_delta"]);

        let end = json!({ "candidates": [{ "finishReason": "STOP" }] });
        let events = translator.translate(&end);
        let names: Vec<&str> = events.iter().map(|(name, _)| name.as_str()).collect();
        assert_eq!(
            names,
            vec!["content_block_stop", "message_delta", "message_stop"]
        );
    }

    #[test]
    fn streaming_tool_call_emits_input_json_delta() {
        let mut translator = StreamTranslator::new("m");
        let chunk = json!({ "candidates": [{ "content": { "parts": [
            { "functionCall": { "name": "f", "args": { "a": 1 } } }
        ]}, "finishReason": "STOP" }] });
        let events = translator.translate(&chunk);
        let names: Vec<&str> = events.iter().map(|(name, _)| name.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop"
            ]
        );
        let delta = events
            .iter()
            .find(|(name, _)| name == "content_block_delta")
            .unwrap();
        assert_eq!(delta.1["delta"]["type"], "input_json_delta");
        let stop = events
            .iter()
            .find(|(name, _)| name == "message_delta")
            .unwrap();
        assert_eq!(stop.1["delta"]["stop_reason"], "tool_use");
    }

    #[test]
    fn stream_finish_without_finish_reason_closes_cleanly() {
        let mut translator = StreamTranslator::new("m");
        let events = translator.finish();
        let names: Vec<&str> = events.iter().map(|(name, _)| name.as_str()).collect();
        assert_eq!(
            names,
            vec!["message_start", "message_delta", "message_stop"]
        );
    }

    #[test]
    fn empty_messages_are_rejected() {
        assert!(request_to_gemini(&json!({ "messages": [] })).is_err());
        assert!(request_to_gemini(&json!({})).is_err());
    }

    #[test]
    fn token_estimate_is_positive() {
        assert!(
            estimate_input_tokens(&json!({ "messages": [{ "role": "user", "content": "hi" }] }))
                > 0
        );
    }
}
