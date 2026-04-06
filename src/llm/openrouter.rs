use anodized::spec;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Semaphore;
use tracing::{debug, info, instrument};

use crate::config::LlmBackendConfig;
use crate::error::InboxError;
use crate::message::LlmResponse;

use super::{LlmClient, LlmCompletion, LlmRequest, ToolCall};

pub struct OpenRouterClient {
    pub model: String,
    pub api_key: String,
    pub base_url: String,
    pub retries: u32,
    pub timeout: Duration,
    semaphore: Option<Arc<Semaphore>>,
    client: reqwest::Client,
}

impl OpenRouterClient {
    /// Create an `OpenRouterClient` from backend config.
    ///
    /// # Panics
    /// Panics if the TLS backend cannot be initialised (extremely unlikely in practice).
    #[must_use]
    #[spec(requires:
        !cfg.model.trim().is_empty()
        && !cfg.base_url.trim().is_empty()
        && cfg.timeout_secs > 0
        && cfg.retries > 0
    )]
    pub fn from_config(cfg: &LlmBackendConfig) -> Self {
        let client = crate::tls::client_builder()
            .connect_timeout(Duration::from_secs(cfg.connect_timeout_secs))
            .timeout(Duration::from_secs(cfg.timeout_secs))
            .build()
            .expect("Failed to build OpenRouter HTTP client");

        Self {
            model: cfg.model.clone(),
            api_key: cfg.api_key.clone().unwrap_or_default(),
            base_url: cfg.base_url.clone(),
            retries: cfg.retries,
            timeout: Duration::from_secs(cfg.timeout_secs),
            semaphore: cfg.max_concurrent.map(|n| Arc::new(Semaphore::new(n))),
            client,
        }
    }
}

// ── Wire types ────────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<&'a str>,
}

#[derive(Serialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<RawToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct RawToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: RawFunction,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct RawFunction {
    pub name: String,
    pub arguments: String,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: ResponseMessage,
}

#[derive(Deserialize)]
struct ResponseMessage {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<RawToolCall>>,
}

// ── LlmClient impl ────────────────────────────────────────────────────────────

#[async_trait]
impl LlmClient for OpenRouterClient {
    fn name(&self) -> &'static str {
        "openrouter"
    }
    fn model(&self) -> &str {
        &self.model
    }
    fn retries(&self) -> u32 {
        self.retries
    }

    #[instrument(skip(self, req), fields(model = %self.model))]
    async fn complete(&self, req: LlmRequest) -> Result<LlmCompletion, InboxError> {
        let LlmRequest {
            system_prompt,
            user_content,
            tool_definitions,
            require_initial_tool_call,
            images,
            ..
        } = req;

        let user_message_content: serde_json::Value = if images.is_empty() {
            serde_json::Value::String(user_content)
        } else {
            let mut parts = vec![serde_json::json!({"type": "text", "text": user_content})];
            for (mime, b64) in &images {
                parts.push(serde_json::json!({
                    "type": "image_url",
                    "image_url": { "url": format!("data:{mime};base64,{b64}") }
                }));
            }
            serde_json::Value::Array(parts)
        };

        let messages = vec![
            ChatMessage {
                role: "system".into(),
                content: serde_json::Value::String(system_prompt),
                tool_calls: None,
                tool_call_id: None,
            },
            ChatMessage {
                role: "user".into(),
                content: user_message_content,
                tool_calls: None,
                tool_call_id: None,
            },
        ];

        let body = ChatRequest {
            model: &self.model,
            messages,
            tools: tool_definitions,
            tool_choice: if require_initial_tool_call {
                Some("required")
            } else {
                None
            },
        };

        let _permit = if let Some(sem) = &self.semaphore {
            Some(sem.acquire().await.expect("semaphore closed"))
        } else {
            None
        };

        let url = format!("{}/chat/completions", self.base_url);
        debug!(url = %url, model = %self.model, "Sending OpenRouter request");
        let resp = self
            .client
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| InboxError::Llm(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(InboxError::Llm(format!(
                "OpenRouter API error {status}: {text}"
            )));
        }

        let chat: ChatResponse = resp
            .json()
            .await
            .map_err(|e| InboxError::Llm(format!("OpenRouter parse error: {e}")))?;

        let choice = chat
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| InboxError::Llm("OpenRouter returned no choices".into()))?;

        // Tool calls?
        if let Some(tool_calls) = choice.message.tool_calls {
            info!(
                tool_count = tool_calls.len(),
                tool_names = ?tool_calls
                    .iter()
                    .map(|tc| tc.function.name.clone())
                    .collect::<Vec<_>>(),
                "OpenRouter returned tool calls"
            );
            let calls = tool_calls
                .into_iter()
                .map(|tc| ToolCall {
                    id: tc.id,
                    name: tc.function.name,
                    arguments: serde_json::from_str(&tc.function.arguments)
                        .unwrap_or(serde_json::Value::Null),
                })
                .collect();
            return Ok(LlmCompletion::ToolCalls(calls));
        }

        // Text response — parse JSON
        let text = choice.message.content.unwrap_or_default();
        debug!(
            response_len = text.len(),
            response_preview = %truncate_for_log(&text, 1200),
            "OpenRouter returned assistant text"
        );

        parse_llm_json_response(&text, "openrouter").map(LlmCompletion::Message)
    }
}

#[spec(requires: max_chars > 0)]
fn truncate_for_log(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        s.to_owned()
    } else {
        let mut out: String = s.chars().take(max_chars).collect();
        out.push_str("…<truncated>");
        out
    }
}

/// Parse the structured JSON response from the LLM into an `LlmResponse`.
///
/// # Errors
/// Returns an error if the text is not valid JSON or missing required fields.
#[spec(requires: !backend.trim().is_empty())]
pub fn parse_llm_json_response(text: &str, backend: &str) -> Result<LlmResponse, InboxError> {
    let cleaned = strip_markdown_fences(text);

    let json: serde_json::Value = match serde_json::from_str(cleaned) {
        Ok(v) => v,
        Err(first_err) => {
            // Fallback: extract the first complete {…} object from the text.
            // Handles think-tag artifacts (e.g. `</think>` after the JSON) and
            // duplicate objects that some models emit.
            match extract_first_json_object(cleaned)
                .and_then(|s| serde_json::from_str::<serde_json::Value>(s).ok())
            {
                Some(v) => {
                    debug!(
                        backend,
                        "LLM response had extra content around JSON — extracted first object"
                    );
                    v
                }
                None => {
                    return Err(InboxError::Llm(format!(
                        "LLM JSON parse error: {first_err}. Raw: {text}"
                    )));
                }
            }
        }
    };

    let title = json["title"].as_str().unwrap_or("(no title)").to_owned();
    let tags = json["tags"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(str::to_owned))
                .collect()
        })
        .unwrap_or_default();
    let summary = json["summary"].as_str().unwrap_or("").to_owned();
    let excerpt = json["excerpt"].as_str().map(str::to_owned);

    Ok(LlmResponse {
        title,
        tags,
        summary,
        excerpt,
        produced_by: backend.to_owned(),
    })
}

fn strip_markdown_fences(s: &str) -> &str {
    let s = s.trim();
    let s = s
        .strip_prefix("```json")
        .or_else(|| s.strip_prefix("```"))
        .unwrap_or(s);
    s.strip_suffix("```").unwrap_or(s).trim()
}

/// Scan `s` for the first balanced `{…}` JSON object and return the slice.
/// Handles string literals (including escaped quotes) so brace characters
/// inside string values are not counted.
fn extract_first_json_object(s: &str) -> Option<&str> {
    let start = s.find('{')?;
    let tail = &s[start..];
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escape = false;

    for (i, c) in tail.char_indices() {
        if escape {
            escape = false;
            continue;
        }
        if c == '\\' && in_string {
            escape = true;
            continue;
        }
        if c == '"' {
            in_string = !in_string;
            continue;
        }
        if in_string {
            continue;
        }
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&s[start..start + i + c.len_utf8()]);
                }
            }
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests;
