use std::path::{Path, PathBuf};

use anodized::spec;
use async_trait::async_trait;
use tracing::{debug, info, warn};

use crate::config::{Config, FallbackMode, LlmConfig};
use crate::error::InboxError;
use crate::message::{EnrichedMessage, LlmResponse};
use crate::pipeline::url_fetcher::UrlFetcher;

pub mod ollama;
pub mod openrouter;
pub mod tools;

// ── Public request / response types ──────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct LlmRequest {
    pub system_prompt: String,
    pub user_content: String,
    pub msg_id: uuid::Uuid,
    pub attachments_dir: PathBuf,
    pub tool_definitions: Vec<serde_json::Value>,
    pub require_initial_tool_call: bool,
    /// Base64-encoded image attachments to include in the vision prompt.
    /// Each entry is `(mime_type, base64_data)`.
    pub images: Vec<(String, String)>,
}

impl LlmRequest {
    #[must_use]
    #[spec(requires: cfg.url_content_max_chars > 0)]
    pub fn from_enriched(
        enriched: &EnrichedMessage,
        cfg: &LlmConfig,
        attachments_dir: &Path,
        guidance_block: &str,
        require_initial_tool_call: bool,
    ) -> Self {
        use crate::message::{MediaKind, SourceMetadata};

        // Prepend forwarded attribution so the LLM has attribution context.
        let mut user_content = if let SourceMetadata::Telegram {
            forwarded_from: Some(ff),
            ..
        } = &enriched.original.metadata
        {
            format!("Forwarded from {ff}\n\n{}", enriched.original.text)
        } else {
            enriched.original.text.clone()
        };

        for uc in &enriched.url_contents {
            use std::fmt::Write as _;
            let _ = write!(user_content, "\n\n--- Page: {} ---", uc.url);
            if let Some(title) = &uc.page_title {
                let _ = write!(user_content, "\nTitle: {title}");
            }
            if !uc.headings.is_empty() {
                let _ = write!(user_content, "\nHeadings: {}", uc.headings.join(" | "));
            }
            if !uc.text.is_empty() {
                let _ = write!(user_content, "\n{}", uc.text);
            }
        }

        // Collect images for vision analysis.
        let images: Vec<(String, String)> = enriched
            .original
            .attachments
            .iter()
            .filter(|a| a.media_kind == MediaKind::Image)
            .filter_map(|a| {
                let bytes = std::fs::read(&a.saved_path).ok()?;
                if bytes.len() > cfg.vision_max_bytes {
                    warn!(
                        path = %a.saved_path.display(),
                        size = bytes.len(),
                        limit = cfg.vision_max_bytes,
                        "Image too large for vision analysis, skipping"
                    );
                    return None;
                }
                let mime = a
                    .mime_type
                    .clone()
                    .unwrap_or_else(|| "image/jpeg".to_owned());
                let b64 =
                    base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &bytes);
                Some((mime, b64))
            })
            .collect();

        let mut system_prompt = cfg.prompts.base_system.clone();
        if !guidance_block.trim().is_empty() {
            system_prompt.push_str("\n\n");
            system_prompt.push_str(cfg.prompts.tool_guidance_header.trim());
            system_prompt.push('\n');
            system_prompt.push_str(guidance_block.trim());
        }
        if !images.is_empty() {
            system_prompt.push('\n');
            system_prompt.push_str(cfg.prompts.vision_prompt_note.trim());
        }

        Self {
            system_prompt,
            user_content,
            msg_id: enriched.original.id,
            attachments_dir: attachments_dir.to_path_buf(),
            tool_definitions: Vec::new(),
            require_initial_tool_call,
            images,
        }
    }

    #[must_use]
    pub fn simple(system_prompt: impl Into<String>, user_content: impl Into<String>) -> Self {
        Self {
            system_prompt: system_prompt.into(),
            user_content: user_content.into(),
            msg_id: uuid::Uuid::nil(),
            attachments_dir: PathBuf::new(),
            tool_definitions: Vec::new(),
            require_initial_tool_call: false,
            images: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum LlmCompletion {
    Message(LlmResponse),
    ToolCalls(Vec<ToolCall>),
}

#[derive(Debug, Clone)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

pub enum LlmOutcome {
    Success(LlmResponse),
    RawFallback,
    Discard,
}

// ── LlmClient trait ───────────────────────────────────────────────────────────

#[async_trait]
pub trait LlmClient: Send + Sync {
    fn name(&self) -> &str;
    fn model(&self) -> &str;
    fn retries(&self) -> u32;
    async fn complete(&self, req: LlmRequest) -> Result<LlmCompletion, InboxError>;
}

// ── LlmChain ─────────────────────────────────────────────────────────────────

pub struct LlmChain {
    backends: Vec<Box<dyn LlmClient>>,
    fallback: FallbackMode,
    max_tool_turns: usize,
    tool_executor: Option<tools::ToolExecutor>,
}

impl LlmChain {
    #[must_use]
    pub fn new(
        backends: Vec<Box<dyn LlmClient>>,
        fallback: FallbackMode,
        max_tool_turns: usize,
        tool_executor: Option<tools::ToolExecutor>,
    ) -> Self {
        Self {
            backends,
            fallback,
            max_tool_turns,
            tool_executor,
        }
    }

    /// Try each backend in order with retries. On exhaustion, apply fallback policy.
    #[spec(requires: self.max_tool_turns > 0)]
    pub async fn complete(&self, req: LlmRequest) -> LlmOutcome {
        let tool_defs = self
            .tool_executor
            .as_ref()
            .map_or_else(Vec::new, tools::ToolExecutor::active_tool_definitions);

        for backend in &self.backends {
            for attempt in 0..backend.retries() {
                let start = std::time::Instant::now();
                let mut req_attempt = req.clone();
                req_attempt.tool_definitions = tool_defs.clone();
                let mut turns = 0usize;
                let mut required_tool_prompts = 0usize;

                loop {
                    let tool_names: Vec<&str> = req_attempt
                        .tool_definitions
                        .iter()
                        .filter_map(|d| d["function"]["name"].as_str())
                        .collect();
                    let system_preview: String =
                        req_attempt.system_prompt.chars().take(300).collect();
                    let content_preview: String =
                        req_attempt.user_content.chars().take(600).collect();
                    debug!(
                        backend = backend.name(),
                        model = backend.model(),
                        turn = turns + 1,
                        tools = ?tool_names,
                        system_len = req_attempt.system_prompt.len(),
                        content_len = req_attempt.user_content.len(),
                        system_preview = %system_preview,
                        content_preview = %content_preview,
                        "LLM request"
                    );

                    match backend.complete(req_attempt.clone()).await {
                        Ok(LlmCompletion::Message(resp)) => {
                            if req_attempt.require_initial_tool_call
                                && turns == 0
                                && !tool_defs.is_empty()
                            {
                                if required_tool_prompts < 3 {
                                    debug!(
                                        backend = backend.name(),
                                        prompt_attempt = required_tool_prompts + 1,
                                        "Re-prompting model to make required initial tool call"
                                    );
                                    req_attempt.user_content.push_str(
                                        "\n\nA tool call is required before final JSON because URLs are present. First analyze and call exactly one best retrieval tool, then continue.",
                                    );
                                    required_tool_prompts += 1;
                                    continue;
                                }
                                warn!(
                                    backend = backend.name(),
                                    "Required initial tool call was not produced"
                                );
                                break;
                            }
                            metrics::counter!(
                                crate::telemetry::LLM_REQUESTS,
                                "backend" => backend.name().to_owned(),
                                "status" => "success"
                            )
                            .increment(1);
                            metrics::histogram!(
                                crate::telemetry::LLM_DURATION,
                                "backend" => backend.name().to_owned()
                            )
                            .record(start.elapsed().as_secs_f64());
                            return LlmOutcome::Success(resp);
                        }
                        Ok(LlmCompletion::ToolCalls(calls)) => {
                            if calls.is_empty() {
                                warn!(
                                    backend = backend.name(),
                                    "LLM returned empty tool call list"
                                );
                                break;
                            }
                            if turns >= self.max_tool_turns {
                                warn!(
                                    backend = backend.name(),
                                    max_turns = self.max_tool_turns,
                                    "Max tool turns reached"
                                );
                                break;
                            }
                            let Some(executor) = &self.tool_executor else {
                                warn!(
                                    backend = backend.name(),
                                    "Tool call requested but no executor configured"
                                );
                                break;
                            };

                            let output = execute_tool_calls(executor, &calls, &req_attempt).await;
                            req_attempt
                                .user_content
                                .push_str("\n\n--- Tool execution results ---\n");
                            req_attempt.user_content.push_str(&output);
                            req_attempt.require_initial_tool_call = false;
                            turns += 1;
                        }
                        Err(e) => {
                            let elapsed_ms = start.elapsed().as_millis();
                            warn!(
                                ?e,
                                backend = backend.name(),
                                model = backend.model(),
                                attempt = attempt + 1,
                                total_attempts = backend.retries(),
                                elapsed_ms,
                                "LLM attempt failed"
                            );
                            break;
                        }
                    }
                }
                metrics::counter!(
                    crate::telemetry::LLM_REQUESTS,
                    "backend" => backend.name().to_owned(),
                    "status" => "failure"
                )
                .increment(1);
            }
            warn!(
                backend = backend.name(),
                model = backend.model(),
                retries = backend.retries(),
                "LLM backend exhausted all retries"
            );
        }

        warn!(
            backend_count = self.backends.len(),
            "All LLM backends failed, applying fallback"
        );
        match self.fallback {
            FallbackMode::Raw => LlmOutcome::RawFallback,
            FallbackMode::Discard => LlmOutcome::Discard,
        }
    }

    #[must_use]
    pub fn max_tool_turns(&self) -> usize {
        self.max_tool_turns
    }
}

#[spec(requires: !calls.is_empty())]
async fn execute_tool_calls(
    executor: &tools::ToolExecutor,
    calls: &[ToolCall],
    req: &LlmRequest,
) -> String {
    let mut outputs = Vec::with_capacity(calls.len());
    for call in calls {
        info!(tool = %call.name, "Executing LLM tool call");
        match executor
            .execute(
                &call.name,
                &call.arguments,
                req.msg_id,
                req.attachments_dir.as_path(),
            )
            .await
        {
            Ok(tools::ToolResult::Text(text) | tools::ToolResult::Attachment { text, .. }) => {
                let result_preview: String = text.chars().take(120).collect();
                info!(
                    tool = %call.name,
                    result_len = text.len(),
                    result_preview = %result_preview,
                    "Tool call result"
                );
                outputs.push(format!("tool `{}`: {text}", call.name));
            }
            Err(e) => {
                warn!(tool = %call.name, ?e, "Tool call failed");
                outputs.push(format!("tool `{}` error: {e}", call.name));
            }
        }
    }

    outputs.join("\n")
}

// ── Builder ───────────────────────────────────────────────────────────────────

use crate::config::{LlmBackendConfig, LlmBackendType};

#[must_use]
pub fn build_chain(cfg: &Config) -> LlmChain {
    let backends: Vec<Box<dyn LlmClient>> = cfg.llm.backends.iter().map(build_backend).collect();

    let tool_executor = Some(tools::from_tooling(
        &cfg.tooling,
        UrlFetcher::new(&cfg.url_fetch),
    ));

    LlmChain::new(
        backends,
        cfg.llm.fallback,
        cfg.llm.max_tool_turns,
        tool_executor,
    )
}

fn build_backend(cfg: &LlmBackendConfig) -> Box<dyn LlmClient> {
    match cfg.backend_type {
        LlmBackendType::Openrouter => Box::new(openrouter::OpenRouterClient::from_config(cfg)),
        LlmBackendType::Ollama => Box::new(ollama::OllamaClient::from_config(cfg)),
    }
}

#[cfg(test)]
mod tests;

// ── Test helpers (also used by integration tests) ─────────────────────────────

#[cfg(any(test, feature = "test-helpers"))]
pub mod mock;
