use anodized::spec;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Semaphore;
use tracing::{debug, info, instrument};

use crate::config::LlmBackendConfig;
use crate::error::InboxError;

use super::openrouter::parse_llm_json_response;
use super::{LlmClient, LlmCompletion, LlmRequest, ToolCall};

pub struct OllamaClient {
    pub model: String,
    pub base_url: String,
    pub retries: u32,
    pub timeout: Duration,
    /// `None` = omit from request (model decides); `Some(true/false)` = explicit.
    pub think: Option<bool>,
    /// Extended timeout applied when thinking is active. `None` = use `self.timeout`.
    pub think_timeout: Option<Duration>,
    pub thinking_supported: bool,
    semaphore: Option<Arc<Semaphore>>,
    client: reqwest::Client,
}

impl OllamaClient {
    /// Create an `OllamaClient` from backend config.
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
            .timeout(Duration::from_secs(cfg.timeout_secs))
            .build()
            .expect("Failed to build Ollama HTTP client");

        Self {
            model: cfg.model.clone(),
            base_url: cfg.base_url.clone(),
            retries: cfg.retries,
            timeout: Duration::from_secs(cfg.timeout_secs),
            think: cfg.think,
            think_timeout: cfg.think_timeout_secs.map(Duration::from_secs),
            thinking_supported: cfg.thinking_supported,
            semaphore: cfg.max_concurrent.map(|n| Arc::new(Semaphore::new(n))),
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
    #[serde(skip_serializing_if = "Option::is_none")]
    think: Option<bool>,
}

#[derive(Serialize, Deserialize, Clone)]
struct OllamaMessage {
    role: String,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OllamaToolCall>>,
    /// Base64-encoded images for vision-capable models.
    #[serde(skip_serializing_if = "Option::is_none")]
    images: Option<Vec<String>>,
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
    thinking: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<OllamaToolCall>>,
}

// ── LlmClient impl ────────────────────────────────────────────────────────────

#[async_trait]
impl LlmClient for OllamaClient {
    fn name(&self) -> &'static str {
        "ollama"
    }
    fn model(&self) -> &str {
        &self.model
    }
    fn retries(&self) -> u32 {
        self.retries
    }
    fn thinking_supported(&self) -> bool {
        self.thinking_supported
    }

    #[instrument(skip(self, req), fields(model = %self.model))]
    async fn complete(&self, req: LlmRequest) -> Result<LlmCompletion, InboxError> {
        let LlmRequest {
            system_prompt,
            user_content,
            tool_definitions,
            images,
            think: req_think,
            ..
        } = req;

        // Per-request think flag takes precedence over backend default.
        let effective_think = req_think.or(self.think);

        let image_b64s: Option<Vec<String>> = if images.is_empty() {
            None
        } else {
            Some(images.into_iter().map(|(_, b64)| b64).collect())
        };

        let messages = vec![
            OllamaMessage {
                role: "system".into(),
                content: system_prompt,
                tool_calls: None,
                images: None,
            },
            OllamaMessage {
                role: "user".into(),
                content: user_content,
                tool_calls: None,
                images: image_b64s,
            },
        ];

        let body = OllamaChatRequest {
            model: &self.model,
            messages,
            stream: false,
            tools: tool_definitions,
            think: effective_think,
        };

        let _permit = if let Some(sem) = &self.semaphore {
            Some(sem.acquire().await.expect("semaphore closed"))
        } else {
            None
        };

        let url = format!("{}/api/chat", self.base_url);
        debug!(url = %url, model = %self.model, think = ?effective_think, "Sending Ollama request");
        let request = self.client.post(&url).json(&body);
        let request = match (effective_think, self.think_timeout) {
            (Some(true), Some(timeout)) => request.timeout(timeout),
            _ => request,
        };
        let resp = request
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

        if let Some(thinking) = &chat.message.thinking
            && !thinking.is_empty()
        {
            debug!(thinking = %truncate_for_log(thinking, 2000), "Ollama model thinking trace");
        }

        // Tool calls?
        if let Some(tool_calls) = chat.message.tool_calls {
            info!(
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
            think: None,
            think_timeout: None,
            thinking_supported: false,
            semaphore: None,
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

    #[tokio::test]
    async fn complete_with_images_sends_images_field() {
        use wiremock::matchers::body_partial_json;

        let server = MockServer::start().await;
        let body = serde_json::json!({
            "message": { "role": "assistant", "content": r#"{"title":"T","tags":[],"summary":"S"}"# }
        });
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .and(body_partial_json(serde_json::json!({
                "messages": [
                    { "role": "system" },
                    { "role": "user", "images": ["aGVsbG8="] }
                ]
            })))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "application/json")
                    .set_body_json(body),
            )
            .mount(&server)
            .await;

        let client = make_client(&server.uri());
        let mut req = LlmRequest::simple("sys", "user");
        req.images = vec![("image/png".into(), "aGVsbG8=".into())];
        let result = client.complete(req).await.unwrap();
        assert!(matches!(result, LlmCompletion::Message(_)));
    }
}
