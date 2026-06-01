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
    pub vision_supported: bool,
    /// KV-cache context window size in tokens. Sent as `options.num_ctx` when set.
    pub context_size: Option<usize>,
    /// Ollama `format` constraint (e.g. `"json"`). Applied only on turns with no
    /// tool definitions, so it never suppresses tool calls mid-loop. `None` =
    /// unconstrained free-text output.
    pub format: Option<String>,
    /// Skip this backend for a cooldown after a connection failure.
    circuit: super::CircuitBreaker,
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
    /// # Errors
    /// Returns an error if the HTTP client cannot be built.
    #[spec(requires:
        !cfg.model.trim().is_empty()
        && !cfg.base_url.trim().is_empty()
        && cfg.timeout_secs > 0
        && cfg.retries > 0
    )]
    pub fn from_config(cfg: &LlmBackendConfig) -> Result<Self, InboxError> {
        let client = crate::tls::client_builder()
            .connect_timeout(Duration::from_secs(cfg.connect_timeout_secs))
            .timeout(Duration::from_secs(cfg.timeout_secs))
            .build()
            .map_err(|e| InboxError::Llm(format!("Failed to build Ollama HTTP client: {e}")))?;

        Ok(Self {
            model: cfg.model.clone(),
            base_url: cfg.base_url.clone(),
            retries: cfg.retries,
            timeout: Duration::from_secs(cfg.timeout_secs),
            think: cfg.think,
            think_timeout: cfg.think_timeout_secs.map(Duration::from_secs),
            thinking_supported: cfg.thinking_supported,
            vision_supported: cfg.vision_supported,
            context_size: cfg.context_size,
            format: cfg.format.clone(),
            circuit: super::CircuitBreaker::new(cfg.circuit_open_secs),
            semaphore: cfg.max_concurrent.map(|n| Arc::new(Semaphore::new(n))),
            client,
        })
    }

    /// Short-circuit when a recent connection failure is still within the
    /// circuit-breaker cooldown.
    fn enforce_circuit(&self) -> Result<(), InboxError> {
        if let Some(remaining) = self.circuit.remaining() {
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
        Ok(())
    }

    /// Pre-flight via `/api/ps` to log model readiness and trip the circuit
    /// breaker if the daemon is unreachable.
    async fn preflight_ps(&self) -> Result<(), InboxError> {
        match self.query_ps().await {
            PsResult::Unreachable => {
                self.circuit.record_failure();
                warn!(model = %self.model, "Ollama unreachable (connection refused) — opening circuit");
                Err(InboxError::Llm(
                    "Ollama unreachable (connection refused)".into(),
                ))
            }
            PsResult::ColdStart => {
                warn!(
                    model = %self.model,
                    "Ollama model not loaded — cold start expected, first request will be slow"
                );
                Ok(())
            }
            PsResult::Ready { vram_mb } => {
                info!(model = %self.model, vram_mb, "Ollama model loaded and warm");
                Ok(())
            }
            PsResult::Unknown => Ok(()),
        }
    }

    /// Trim user content if the rough token budget (system + tools + user)
    /// would exceed `context_size`. Ollama silently left-truncates past
    /// `num_ctx`, often dropping the system prompt — we'd rather drop tail
    /// context than the JSON instruction.
    fn truncate_user_to_context(
        &self,
        mut user_content: String,
        system_prompt: &str,
        tool_definitions: &[serde_json::Value],
    ) -> String {
        let Some(ctx) = self.context_size else {
            return user_content;
        };
        let tool_chars: usize = tool_definitions.iter().map(|t| t.to_string().len()).sum();
        let overhead_tokens = (system_prompt.len() + tool_chars) / 4;
        let user_tokens = user_content.len() / 4;
        let total_tokens = overhead_tokens + user_tokens;
        if total_tokens <= ctx {
            return user_content;
        }
        let user_budget_tokens = ctx.saturating_sub(overhead_tokens);
        warn!(
            total_tokens,
            overhead_tokens,
            user_tokens,
            context_size = ctx,
            user_budget_tokens,
            "Request exceeds context window — truncating user content (raise context_size if the model supports it)"
        );
        let char_limit = user_budget_tokens.saturating_mul(4);
        let truncated: String = user_content.chars().take(char_limit).collect();
        user_content = format!(
            "{truncated}\n... [context truncated: ~{total_tokens} tokens > context_size {ctx}]"
        );
        user_content
    }

    /// Assemble the `/api/chat` body. `format` is suppressed when tools are
    /// offered so the model is free to emit `tool_calls` instead of JSON.
    fn build_chat_request<'a>(
        &'a self,
        system_prompt: &'a str,
        user_content: String,
        images: Vec<(String, String)>,
        tool_definitions: Vec<serde_json::Value>,
        effective_think: Option<bool>,
    ) -> OllamaChatRequest<'a> {
        let image_b64s: Option<Vec<String>> = if images.is_empty() {
            None
        } else {
            Some(images.into_iter().map(|(_, b64)| b64).collect())
        };
        let messages = vec![
            OllamaMessage {
                role: "system".into(),
                content: system_prompt.to_owned(),
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
        let format = if tool_definitions.is_empty() {
            self.format.as_deref()
        } else {
            None
        };
        OllamaChatRequest {
            model: &self.model,
            messages,
            stream: false,
            tools: tool_definitions,
            think: effective_think,
            format,
            options,
        }
    }

    /// Send the prepared body and return the parsed chat response on a 2xx.
    async fn send_chat(
        &self,
        body: &OllamaChatRequest<'_>,
        effective_think: Option<bool>,
    ) -> Result<OllamaChatResponse, InboxError> {
        // `acquire` errors only if the semaphore is closed, which never happens
        // here (we hold the `Arc` and never call `close`). Treat the impossible
        // error as "no permit" and proceed rather than panicking.
        let _permit = match &self.semaphore {
            Some(sem) => sem.acquire().await.ok(),
            None => None,
        };

        let url = format!("{}/api/chat", self.base_url);
        debug!(url = %url, model = %self.model, think = ?effective_think, "Sending Ollama request");
        let request = self.client.post(&url).json(body);
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

        resp.json::<OllamaChatResponse>()
            .await
            .map_err(|e| InboxError::Llm(format!("Ollama parse error: {e}")))
    }

    /// Convert a chat response into either a `ToolCalls` or a parsed JSON
    /// `Message`. Resets the circuit on success.
    fn parse_chat_response(&self, chat: OllamaChatResponse) -> Result<LlmCompletion, InboxError> {
        if let Some(thinking) = &chat.message.thinking
            && !thinking.is_empty()
        {
            debug!(thinking = %truncate_for_log(thinking, 2000), "Ollama model thinking trace");
        }

        self.circuit.clear();

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
    format: Option<&'a str>,
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
    fn vision_supported(&self) -> bool {
        self.vision_supported
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

        self.enforce_circuit()?;
        self.preflight_ps().await?;

        let user_content =
            self.truncate_user_to_context(user_content, &system_prompt, &tool_definitions);
        let effective_think = req_think.or(self.think);
        let body = self.build_chat_request(
            &system_prompt,
            user_content,
            images,
            tool_definitions,
            effective_think,
        );

        let chat = self.send_chat(&body, effective_think).await?;
        self.parse_chat_response(chat)
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
mod tests;
