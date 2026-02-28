use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::{debug, instrument};

use crate::config::LlmBackendConfig;
use crate::error::InboxError;

use super::openrouter::parse_llm_json_response;
use super::{LlmClient, LlmCompletion, LlmRequest, ToolCall};

pub struct OllamaClient {
    pub model: String,
    pub base_url: String,
    pub retries: u32,
    pub timeout: Duration,
    client: reqwest::Client,
}

impl OllamaClient {
    /// Create an `OllamaClient` from backend config.
    ///
    /// # Panics
    /// Panics if the TLS backend cannot be initialised (extremely unlikely in practice).
    #[must_use]
    pub fn from_config(cfg: &LlmBackendConfig) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(cfg.timeout_secs))
            .build()
            .expect("Failed to build Ollama HTTP client");

        Self {
            model: cfg.model.clone(),
            base_url: cfg.base_url.clone(),
            retries: cfg.retries,
            timeout: Duration::from_secs(cfg.timeout_secs),
            client,
        }
    }
}

// ── Wire types ────────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct OllamaChatRequest<'a> {
    model: &'a str,
    messages: Vec<OllamaMessage>,
    stream: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<serde_json::Value>,
}

#[derive(Serialize, Deserialize, Clone)]
struct OllamaMessage {
    role: String,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OllamaToolCall>>,
}

#[derive(Serialize, Deserialize, Clone)]
struct OllamaToolCall {
    function: OllamaFunction,
}

#[derive(Serialize, Deserialize, Clone)]
struct OllamaFunction {
    name: String,
    arguments: serde_json::Value,
}

#[derive(Deserialize)]
struct OllamaChatResponse {
    message: OllamaResponseMessage,
}

#[derive(Deserialize)]
struct OllamaResponseMessage {
    content: String,
    #[serde(default)]
    tool_calls: Option<Vec<OllamaToolCall>>,
}

// ── LlmClient impl ────────────────────────────────────────────────────────────

#[async_trait]
impl LlmClient for OllamaClient {
    fn name(&self) -> &'static str {
        "ollama"
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
            ..
        } = req;

        let messages = vec![
            OllamaMessage {
                role: "system".into(),
                content: system_prompt,
                tool_calls: None,
            },
            OllamaMessage {
                role: "user".into(),
                content: user_content,
                tool_calls: None,
            },
        ];

        let body = OllamaChatRequest {
            model: &self.model,
            messages,
            stream: false,
            tools: tool_definitions,
        };

        let resp = self
            .client
            .post(format!("{}/api/chat", self.base_url))
            .json(&body)
            .send()
            .await
            .map_err(|e| InboxError::Llm(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(InboxError::Llm(format!(
                "Ollama API error {status}: {text}"
            )));
        }

        let chat: OllamaChatResponse = resp
            .json()
            .await
            .map_err(|e| InboxError::Llm(format!("Ollama parse error: {e}")))?;

        // Tool calls?
        if let Some(tool_calls) = chat.message.tool_calls {
            debug!(
                tool_count = tool_calls.len(),
                tool_names = ?tool_calls
                    .iter()
                    .map(|tc| tc.function.name.clone())
                    .collect::<Vec<_>>(),
                "Ollama returned tool calls"
            );
            let calls = tool_calls
                .into_iter()
                .map(|tc| ToolCall {
                    id: uuid::Uuid::new_v4().to_string(),
                    name: tc.function.name,
                    arguments: tc.function.arguments,
                })
                .collect();
            return Ok(LlmCompletion::ToolCalls(calls));
        }

        debug!(
            response_len = chat.message.content.len(),
            response_preview = %truncate_for_log(&chat.message.content, 1200),
            "Ollama returned assistant text"
        );

        parse_llm_json_response(&chat.message.content, "ollama").map(LlmCompletion::Message)
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

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn make_client(base_url: &str) -> OllamaClient {
        OllamaClient {
            model: "llama3".into(),
            base_url: base_url.to_owned(),
            retries: 1,
            timeout: std::time::Duration::from_secs(5),
            client: reqwest::Client::new(),
        }
    }

    #[tokio::test]
    async fn complete_success() {
        let server = MockServer::start().await;
        let body = serde_json::json!({
            "message": {
                "role": "assistant",
                "content": r#"{"title":"T","tags":[],"summary":"S"}"#
            }
        });
        Mock::given(method("POST"))
            .and(path("/api/chat"))
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
            .and(path("/api/chat"))
            .respond_with(ResponseTemplate::new(503).set_body_string("unavailable"))
            .mount(&server)
            .await;

        let client = make_client(&server.uri());
        let req = LlmRequest::simple("sys", "user");
        assert!(client.complete(req).await.is_err());
    }

    #[tokio::test]
    async fn complete_tool_calls() {
        let server = MockServer::start().await;
        let body = serde_json::json!({
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "function": { "name": "scrape_page", "arguments": {"url": "http://x.com"} }
                }]
            }
        });
        Mock::given(method("POST"))
            .and(path("/api/chat"))
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
