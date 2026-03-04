use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::Duration;
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
    client: reqwest::Client,
}

impl OpenRouterClient {
    /// Create an `OpenRouterClient` from backend config.
    ///
    /// # Panics
    /// Panics if the TLS backend cannot be initialised (extremely unlikely in practice).
    #[must_use]
    pub fn from_config(cfg: &LlmBackendConfig) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(cfg.timeout_secs))
            .build()
            .expect("Failed to build OpenRouter HTTP client");

        Self {
            model: cfg.model.clone(),
            api_key: cfg.api_key.clone().unwrap_or_default(),
            base_url: cfg.base_url.clone(),
            retries: cfg.retries,
            timeout: Duration::from_secs(cfg.timeout_secs),
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
            ..
        } = req;

        let messages = vec![
            ChatMessage {
                role: "system".into(),
                content: serde_json::Value::String(system_prompt),
                tool_calls: None,
                tool_call_id: None,
            },
            ChatMessage {
                role: "user".into(),
                content: serde_json::Value::String(user_content),
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
pub fn parse_llm_json_response(text: &str, backend: &str) -> Result<LlmResponse, InboxError> {
    // Strip optional markdown fences
    let cleaned = strip_markdown_fences(text);
    let json: serde_json::Value = serde_json::from_str(cleaned)
        .map_err(|e| InboxError::Llm(format!("LLM JSON parse error: {e}. Raw: {text}")))?;

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

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn make_client(base_url: &str) -> OpenRouterClient {
        OpenRouterClient {
            model: "test-model".into(),
            api_key: "test-key".into(),
            base_url: base_url.to_owned(),
            retries: 1,
            timeout: std::time::Duration::from_secs(5),
            client: reqwest::Client::new(),
        }
    }

    #[test]
    fn parse_json_response_full() {
        let json = r#"{"title":"T","tags":["a","b"],"summary":"S","excerpt":"E"}"#;
        let r = parse_llm_json_response(json, "test").unwrap();
        assert_eq!(r.title, "T");
        assert_eq!(r.tags, vec!["a", "b"]);
        assert_eq!(r.summary, "S");
        assert_eq!(r.excerpt.as_deref(), Some("E"));
        assert_eq!(r.produced_by, "test");
    }

    #[test]
    fn parse_json_strips_markdown_fences() {
        let json = "```json\n{\"title\":\"T\",\"summary\":\"S\",\"tags\":[]}\n```";
        let r = parse_llm_json_response(json, "x").unwrap();
        assert_eq!(r.title, "T");
    }

    #[test]
    fn parse_json_strips_bare_fences() {
        let json = "```\n{\"title\":\"T\",\"summary\":\"S\",\"tags\":[]}\n```";
        let r = parse_llm_json_response(json, "x").unwrap();
        assert_eq!(r.title, "T");
    }

    #[test]
    fn parse_json_missing_fields_defaults() {
        let json = r"{}";
        let r = parse_llm_json_response(json, "x").unwrap();
        assert_eq!(r.title, "(no title)");
        assert!(r.tags.is_empty());
        assert_eq!(r.summary, "");
        assert!(r.excerpt.is_none());
    }

    #[test]
    fn parse_json_invalid_returns_error() {
        let result = parse_llm_json_response("not json", "x");
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn complete_success() {
        let server = MockServer::start().await;
        let body = serde_json::json!({
            "choices": [{
                "message": {
                    "content": r#"{"title":"T","tags":[],"summary":"S"}"#
                }
            }]
        });
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "application/json")
                    .set_body_json(body),
            )
            .mount(&server)
            .await;

        let client = make_client(&server.uri());
        let req = LlmRequest::simple("sys", "user");
        let result = client.complete(req).await.unwrap();
        assert!(matches!(result, LlmCompletion::Message(_)));
    }

    #[tokio::test]
    async fn complete_error_status() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(500).set_body_string("oops"))
            .mount(&server)
            .await;

        let client = make_client(&server.uri());
        let req = LlmRequest::simple("sys", "user");
        let result = client.complete(req).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn complete_empty_choices_error() {
        let server = MockServer::start().await;
        let body = serde_json::json!({ "choices": [] });
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "application/json")
                    .set_body_json(body),
            )
            .mount(&server)
            .await;

        let client = make_client(&server.uri());
        let req = LlmRequest::simple("sys", "user");
        let result = client.complete(req).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn complete_tool_calls() {
        let server = MockServer::start().await;
        let body = serde_json::json!({
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": { "name": "scrape_page", "arguments": "{\"url\":\"http://example.com\"}" }
                    }]
                }
            }]
        });
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "application/json")
                    .set_body_json(body),
            )
            .mount(&server)
            .await;

        let client = make_client(&server.uri());
        let req = LlmRequest::simple("sys", "user");
        let result = client.complete(req).await.unwrap();
        assert!(matches!(result, LlmCompletion::ToolCalls(_)));
    }
}
