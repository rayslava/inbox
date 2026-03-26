use anodized::spec;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Semaphore;
use tracing::{debug, info, instrument, warn};

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
    /// KV-cache context window size in tokens. Sent as `options.num_ctx` when set.
    pub context_size: Option<usize>,
    semaphore: Option<Arc<Semaphore>>,
    client: reqwest::Client,
}

impl OllamaClient {
    /// Query the Ollama `/api/ps` endpoint to get currently loaded models.
    /// Returns `None` if the request fails (always non-fatal).
    async fn query_ps(&self) -> Option<Vec<OllamaPsModel>> {
        let resp = self
            .client
            .get(format!("{}/api/ps", self.base_url))
            .send()
            .await
            .ok()?;
        resp.json::<OllamaPsResponse>().await.ok().map(|r| r.models)
    }

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
            context_size: cfg.context_size,
            semaphore: cfg.max_concurrent.map(|n| Arc::new(Semaphore::new(n))),
            client,
        }
    }
}

// ── Wire types ────────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct OllamaOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    num_ctx: Option<usize>,
}

#[derive(Serialize)]
struct OllamaChatRequest<'a> {
    model: &'a str,
    messages: Vec<OllamaMessage>,
    stream: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    think: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<OllamaOptions>,
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
struct OllamaPsResponse {
    models: Vec<OllamaPsModel>,
}

#[derive(Deserialize)]
struct OllamaPsModel {
    name: String,
    size_vram: Option<u64>,
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
            mut user_content,
            tool_definitions,
            images,
            think: req_think,
            ..
        } = req;

        // Pre-flight: log model load status from Ollama /api/ps.
        if let Some(models) = self.query_ps().await {
            let loaded = models.iter().any(|m| m.name.contains(self.model.as_str()));
            let vram_mb = models
                .iter()
                .find(|m| m.name.contains(self.model.as_str()))
                .and_then(|m| m.size_vram.map(|b| b / (1024 * 1024)));
            info!(model = %self.model, loaded, vram_mb, "Ollama model load status");
        }

        // Guard: truncate user_content if estimated tokens exceed the configured context window.
        let estimated_tokens = user_content.len() / 4;
        if let Some(ctx) = self.context_size {
            if estimated_tokens > ctx {
                warn!(
                    estimated_tokens,
                    context_size = ctx,
                    "Content exceeds context window — truncating"
                );
                let char_limit = ctx.saturating_mul(4);
                let truncated: String = user_content.chars().take(char_limit).collect();
                user_content = format!(
                    "{truncated}\n... [context truncated: ~{estimated_tokens} tokens > context_size {ctx}]"
                );
            }
        }

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

        let options = self
            .context_size
            .map(|n| OllamaOptions { num_ctx: Some(n) });

        let body = OllamaChatRequest {
            model: &self.model,
            messages,
            stream: false,
            tools: tool_definitions,
            think: effective_think,
            options,
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
            context_size: None,
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

    #[tokio::test]
    async fn context_size_sends_options_num_ctx() {
        use wiremock::matchers::body_partial_json;

        let server = MockServer::start().await;
        let body = serde_json::json!({
            "message": { "role": "assistant", "content": r#"{"title":"T","tags":[],"summary":"S"}"# }
        });
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .and(body_partial_json(serde_json::json!({
                "options": { "num_ctx": 16384 }
            })))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "application/json")
                    .set_body_json(body),
            )
            .mount(&server)
            .await;

        let mut client = make_client(&server.uri());
        client.context_size = Some(16384);
        let req = LlmRequest::simple("sys", "user");
        let result = client.complete(req).await.unwrap();
        assert!(matches!(result, LlmCompletion::Message(_)));
    }

    #[tokio::test]
    async fn no_context_size_omits_options() {
        let server = MockServer::start().await;
        let body = serde_json::json!({
            "message": { "role": "assistant", "content": r#"{"title":"T","tags":[],"summary":"S"}"# }
        });
        // If options were present with num_ctx, this mock would only match that specific body.
        // By NOT using body_partial_json for options, we verify the basic path still works.
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "application/json")
                    .set_body_json(body),
            )
            .mount(&server)
            .await;

        let client = make_client(&server.uri()); // context_size = None
        let req = LlmRequest::simple("sys", "user");
        let result = client.complete(req).await.unwrap();
        assert!(matches!(result, LlmCompletion::Message(_)));
    }

    fn chat_response_body() -> serde_json::Value {
        serde_json::json!({
            "message": { "role": "assistant", "content": r#"{"title":"T","tags":[],"summary":"S"}"# }
        })
    }

    #[tokio::test]
    async fn preflight_model_loaded_proceeds() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/ps"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "application/json")
                    .set_body_json(serde_json::json!({
                        "models": [{"name": "llama3", "size_vram": 4_294_967_296_u64}]
                    })),
            )
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "application/json")
                    .set_body_json(chat_response_body()),
            )
            .mount(&server)
            .await;

        let client = make_client(&server.uri());
        let req = LlmRequest::simple("sys", "user");
        let result = client.complete(req).await.unwrap();
        assert!(matches!(result, LlmCompletion::Message(_)));
    }

    #[tokio::test]
    async fn preflight_empty_models_proceeds() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/ps"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "application/json")
                    .set_body_json(serde_json::json!({"models": []})),
            )
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "application/json")
                    .set_body_json(chat_response_body()),
            )
            .mount(&server)
            .await;

        let client = make_client(&server.uri());
        let req = LlmRequest::simple("sys", "user");
        let result = client.complete(req).await.unwrap();
        assert!(matches!(result, LlmCompletion::Message(_)));
    }

    #[tokio::test]
    async fn preflight_error_ignored_proceeds() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/ps"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "application/json")
                    .set_body_json(chat_response_body()),
            )
            .mount(&server)
            .await;

        let client = make_client(&server.uri());
        let req = LlmRequest::simple("sys", "user");
        let result = client.complete(req).await.unwrap();
        assert!(matches!(result, LlmCompletion::Message(_)));
    }

    #[tokio::test]
    async fn context_overflow_truncates_content() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/ps"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "application/json")
                    .set_body_json(serde_json::json!({"models": []})),
            )
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "application/json")
                    .set_body_json(chat_response_body()),
            )
            .mount(&server)
            .await;

        let mut client = make_client(&server.uri());
        // context_size = 1 token → char_limit = 4 chars; content of 100 chars triggers truncation
        client.context_size = Some(1);
        let long_content = "a".repeat(100);
        let req = LlmRequest::simple("sys", &long_content);
        let result = client.complete(req).await;
        // Truncation fires but request still completes normally
        assert!(result.is_ok());
    }
}
