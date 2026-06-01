use std::path::{Path, PathBuf};

use anodized::spec;
use async_trait::async_trait;
use tracing::warn;

use crate::config::{FallbackMode, LlmConfig};
use crate::error::InboxError;
use crate::message::{EnrichedMessage, LlmResponse};

pub mod free_router;
pub mod ollama;
pub mod openrouter;
pub mod tools;

mod chain;
mod chain_tools;
#[cfg(test)]
mod chain_tools_tests;
mod circuit;
pub use chain::LlmChain;
pub(crate) use chain::VisionUnavailable;
#[cfg(test)]
use chain_tools::append_missing_source_links;
pub(crate) use circuit::CircuitBreaker;

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
    /// Whether recognized image text (from the image-analysis stage) has been
    /// folded into `user_content`. When `true`, non-vision backends can still
    /// produce a useful result from the text alone (images may be stripped).
    pub has_image_text: bool,
    /// Per-request thinking mode override.
    /// Set to `Some(true)` by the chain when `activate_thinking` tool is called.
    /// `None` = use backend default.
    pub think: Option<bool>,
    /// Recursive depth of `llm_call` tool invocations. `0` = top-level request.
    pub llm_depth: u32,
    /// Optional channel to report per-turn progress events back to the caller.
    pub progress_tx: Option<tokio::sync::mpsc::UnboundedSender<LlmTurnProgress>>,
    /// Source adapter name (e.g. "telegram", "email", "http") for memory linking.
    pub source_name: String,
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
        use std::fmt::Write as _;

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

        // Append text recognized from images by the analysis stage, and collect
        // the names of attachments that were transcribed. `has_image_text` lets a
        // text-only backend summarize from the recognized text.
        let mut has_image_text = false;
        let mut transcribed: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for a in &enriched.original.image_analyses {
            if !a.recognized_text.trim().is_empty() {
                has_image_text = true;
                transcribed.insert(a.attachment_name.as_str());
                let _ = write!(
                    user_content,
                    "\n\n--- Image text ({}) ---\n{}",
                    a.attachment_name, a.recognized_text
                );
            }
        }

        // Encode images for vision per attachment: skip any image already
        // transcribed (its text is the distilled content — re-sending the raw
        // image is wasteful), but keep images with no extracted text so a vision
        // backend in enrichment can still try them.
        let images: Vec<(String, String)> = enriched
            .original
            .attachments
            .iter()
            .filter(|a| a.media_kind == MediaKind::Image)
            .filter(|a| !transcribed.contains(a.original_name.as_str()))
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
            has_image_text,
            think: None,
            llm_depth: 0,
            progress_tx: None,
            source_name: enriched.original.source_name().to_owned(),
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
            has_image_text: false,
            think: None,
            llm_depth: 0,
            progress_tx: None,
            source_name: String::new(),
        }
    }

    /// Whether this request carries image attachments that require a
    /// vision-capable backend. Derived from `images` so it can never desync
    /// when a non-vision backend invocation strips the images.
    #[must_use]
    pub fn needs_vision(&self) -> bool {
        !self.images.is_empty()
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
    Success {
        response: LlmResponse,
        /// `backend:model` identifiers of additional models consulted during
        /// this run (e.g. via `llm_call` sub-calls). Primary model is in
        /// `response.produced_by`.
        helpers: Vec<String>,
        /// Total tool invocations across the turn loop.
        tool_calls_made: usize,
    },
    RawFallback {
        source_urls: Vec<String>,
        tool_results: Vec<(String, String)>,
        helpers: Vec<String>,
        tool_calls_made: usize,
    },
    Discard,
}

/// Progress event emitted after each tool-call turn by the LLM chain.
#[derive(Debug)]
pub struct LlmTurnProgress {
    pub turn: usize,
    pub max_turns: usize,
    pub tools_called: Vec<String>,
}

// ── LlmClient trait ───────────────────────────────────────────────────────────

#[async_trait]
pub trait LlmClient: Send + Sync {
    fn name(&self) -> &str;
    fn model(&self) -> &str;
    fn retries(&self) -> u32;
    /// Whether this backend supports the `think` field. Controls whether
    /// `activate_thinking` is offered in the tool list.
    fn thinking_supported(&self) -> bool {
        false
    }
    /// Whether this backend can consume image attachments (vision input).
    /// Image-bearing requests are routed only to backends that return `true`,
    /// unless the request already carries recognized image text.
    fn vision_supported(&self) -> bool {
        false
    }
    async fn complete(&self, req: LlmRequest) -> Result<LlmCompletion, InboxError>;
    /// Whether the backend is usable right now. A backend that recently hit a
    /// transient outage (429/5xx/timeout) reports `false` for a cooldown window
    /// so the chain skips it without an attempt and falls through to the next
    /// (local) backend. Default: always available.
    fn is_available(&self) -> bool {
        true
    }
    /// Trip the backend's cooldown after a transient "service unavailable"
    /// failure. Default: no-op (backends without a circuit ignore it).
    fn mark_unavailable(&self) {}
    /// Call the backend and return the plain-text result plus the
    /// `backend:model` identifier of the model that produced it.
    ///
    /// Default implementation wraps the system prompt to elicit a JSON
    /// `{"summary": "..."}` response and extracts the `summary` field.
    async fn complete_raw(&self, mut req: LlmRequest) -> Result<(String, String), InboxError> {
        req.system_prompt.push_str(
            "\n\nRespond ONLY with a JSON object: {\"summary\": \"<your complete answer here>\"}",
        );
        match self.complete(req).await? {
            LlmCompletion::Message(resp) => Ok((resp.summary, resp.produced_by)),
            LlmCompletion::ToolCalls(_) => Err(InboxError::Llm(
                "llm_call: unexpected tool calls in sub-request".into(),
            )),
        }
    }
}

pub(crate) fn activate_thinking_tool_def() -> serde_json::Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": "activate_thinking",
            "description": "Activate extended thinking/reasoning mode for this request. \
                            Call this when the task requires deep analysis, complex \
                            multi-step reasoning, or careful deliberation.",
            "parameters": { "type": "object", "properties": {} }
        }
    })
}

pub(crate) fn llm_call_tool_def() -> serde_json::Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": "llm_call",
            "description": "Invoke the LLM with a custom system prompt and content. \
                            Returns the model's plain-text response. Use for sub-tasks: \
                            summarization, extraction, translation, analysis, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "system_prompt": {
                        "type": "string",
                        "description": "System prompt for the sub-call"
                    },
                    "content": {
                        "type": "string",
                        "description": "User content to process"
                    }
                },
                "required": ["system_prompt", "content"]
            }
        }
    })
}

/// Exponential backoff before a retry attempt. `attempt` is the loop counter
/// (the first retry is `attempt == 1`); returns `500ms · 2^(attempt-1)`,
/// saturating so a large attempt count can never overflow. Shared by every LLM
/// retry loop so the schedule stays identical across backends.
#[must_use]
pub(crate) fn retry_backoff(attempt: u32) -> std::time::Duration {
    let exp = attempt.saturating_sub(1);
    let delay_ms = 500u64.saturating_mul(2u64.saturating_pow(exp));
    std::time::Duration::from_millis(delay_ms)
}

// ── Builder ───────────────────────────────────────────────────────────────────

mod builder;
pub use builder::{BuildResult, build_chain};

#[cfg(test)]
mod circuit_tests;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_builder;
#[cfg(test)]
mod tests_chain_tool_exec;
#[cfg(test)]
mod tests_resilience;
#[cfg(test)]
mod tests_thinking;
#[cfg(test)]
mod tests_vision;

// ── Test helpers (also used by integration tests) ─────────────────────────────

#[cfg(any(test, feature = "test-helpers"))]
pub mod mock;
