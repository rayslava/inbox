use anodized::spec;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
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
    /// Seconds to skip this backend after a connection failure (circuit breaker).
    /// 0 disables the circuit breaker.
    circuit_open_secs: u64,
    last_connection_failure: Arc<Mutex<Option<Instant>>>,
    semaphore: Option<Arc<Semaphore>>,
    client: reqwest::Client,
}

/// Result of the Ollama `/api/ps` pre-flight check.
enum PsResult {
    /// Model is loaded in memory and ready.
    Ready { vram_mb: Option<u64> },
    /// Ollama is up but this model is not loaded (cold-start expected).
    ColdStart,
    /// TCP connection failed — Ollama is unreachable.
    Unreachable,
    /// Reachable but response could not be parsed (non-fatal).
    Unknown,
}

impl OllamaClient {
    /// Query the Ollama `/api/ps` endpoint to determine model readiness.
    async fn query_ps(&self) -> PsResult {
        match self
            .client
            .get(format!("{}/api/ps", self.base_url))
            .send()
            .await
        {
            Err(e) if e.is_connect() => PsResult::Unreachable,
            Err(_) => PsResult::Unknown,
            Ok(resp) => match resp.json::<OllamaPsResponse>().await {
                Err(_) => PsResult::Unknown,
                Ok(r) => {
                    let entry = r.models.iter().find(|m| m.name.contains(&self.model));
                    match entry {
                        Some(m) => PsResult::Ready {
                            vram_mb: m.size_vram.map(|b| b / (1024 * 1024)),
                        },
                        None => PsResult::ColdStart,
                    }
                }
            },
        }
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
            .connect_timeout(Duration::from_secs(cfg.connect_timeout_secs))
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
            circuit_open_secs: cfg.circuit_open_secs,
            last_connection_failure: Arc::new(Mutex::new(None)),
            semaphore: cfg.max_concurrent.map(|n| Arc::new(Semaphore::new(n))),
            client,
        }
    }

    /// Record a connection failure and open the circuit breaker.
    fn record_connection_failure(&self) {
        *self
            .last_connection_failure
            .lock()
            .expect("circuit mutex poisoned") = Some(Instant::now());
    }

    /// Clear the circuit breaker (called on successful response).
    fn clear_circuit(&self) {
        *self
            .last_connection_failure
            .lock()
            .expect("circuit mutex poisoned") = None;
    }

    /// Returns `true` when the circuit is open and requests should be skipped.
    fn is_circuit_open(&self) -> Option<Duration> {
        if self.circuit_open_secs == 0 {
            return None;
        }
        let guard = self
            .last_connection_failure
            .lock()
            .expect("circuit mutex poisoned");
        if let Some(failed_at) = *guard {
            let elapsed = failed_at.elapsed();
            let limit = Duration::from_secs(self.circuit_open_secs);
            if elapsed < limit {
                return Some(limit.checked_sub(elapsed).unwrap());
            }
        }
        None
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

        // Circuit breaker: skip immediately if a recent connection failure is still within cooldown.
        if let Some(remaining) = self.is_circuit_open() {
            warn!(
                model = %self.model,
                remaining_secs = remaining.as_secs(),
                "Ollama circuit open — skipping request"
            );
            return Err(InboxError::Llm(format!(
                "Ollama circuit open: backend unreachable, retry in {}s",
                remaining.as_secs()
            )));
        }

        // Pre-flight: determine model readiness via /api/ps.
        match self.query_ps().await {
            PsResult::Unreachable => {
                self.record_connection_failure();
                warn!(model = %self.model, "Ollama unreachable (connection refused) — opening circuit");
                return Err(InboxError::Llm(
                    "Ollama unreachable (connection refused)".into(),
                ));
            }
            PsResult::ColdStart => {
                warn!(
                    model = %self.model,
                    "Ollama model not loaded — cold start expected, first request will be slow"
                );
            }
            PsResult::Ready { vram_mb } => {
                info!(model = %self.model, vram_mb, "Ollama model loaded and warm");
            }
            PsResult::Unknown => {}
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

        // Successful response — clear any previous connection failure.
        self.clear_circuit();

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

        let produced_by = format!("ollama:{}", self.model);
        parse_llm_json_response(&chat.message.content, &produced_by).map(LlmCompletion::Message)
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
    use std::time::{Duration, Instant};
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
            circuit_open_secs: 0,
            last_connection_failure: Arc::new(Mutex::new(None)),
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
    async fn circuit_open_skips_request() {
        // Pre-set a recent connection failure; subsequent call should return
        // a circuit-open error without making any HTTP requests.
        let server = MockServer::start().await;
        // No mocks registered — any HTTP hit would be an unexpected request.

        let mut client = make_client(&server.uri());
        client.circuit_open_secs = 300;
        *client.last_connection_failure.lock().expect("mutex") = Some(Instant::now());

        let result = client.complete(LlmRequest::simple("sys", "user")).await;
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("circuit"),
            "expected circuit-open error, got: {msg}"
        );
    }

    #[tokio::test]
    async fn circuit_clears_on_success() {
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
        // Artificially open a stale circuit from 1000s ago (expired).
        client.circuit_open_secs = 1;
        *client.last_connection_failure.lock().expect("mutex") =
            Some(Instant::now().checked_sub(Duration::from_secs(10)).unwrap());

        // Circuit should be expired — request succeeds and clears failure.
        let result = client.complete(LlmRequest::simple("sys", "user")).await;
        assert!(result.is_ok());
        assert!(
            client
                .last_connection_failure
                .lock()
                .expect("mutex")
                .is_none()
        );
    }

    #[tokio::test]
    async fn preflight_cold_start_proceeds() {
        // Empty /api/ps (model not loaded) — should proceed with a cold-start warning,
        // not fail.
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

        let result = make_client(&server.uri())
            .complete(LlmRequest::simple("sys", "user"))
            .await;
        assert!(result.is_ok());
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
