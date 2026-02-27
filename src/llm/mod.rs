use std::path::{Path, PathBuf};

use anodized::contract;
use async_trait::async_trait;
use tracing::warn;

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
}

impl LlmRequest {
    #[must_use]
    pub fn from_enriched(
        enriched: &EnrichedMessage,
        cfg: &LlmConfig,
        attachments_dir: &Path,
        tool_prompt_block: &str,
    ) -> Self {
        let mut user_content = enriched.original.text.clone();

        for uc in &enriched.url_contents {
            use std::fmt::Write as _;
            let _ = write!(
                user_content,
                "\n\n--- Content from {} ---\n{}",
                uc.url, uc.text
            );
        }

        let mut system_prompt = if cfg.system_prompt.is_empty() {
            default_system_prompt()
        } else {
            cfg.system_prompt.clone()
        };
        if !tool_prompt_block.trim().is_empty() {
            system_prompt.push_str("\n\nTool-specific guidance:\n");
            system_prompt.push_str(tool_prompt_block.trim());
        }

        Self {
            system_prompt,
            user_content,
            msg_id: enriched.original.id,
            attachments_dir: attachments_dir.to_path_buf(),
            tool_definitions: Vec::new(),
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
        }
    }
}

fn default_system_prompt() -> String {
    r#"You are a personal inbox assistant. Given a captured note or web content, respond with a JSON object containing:
- "title": a short descriptive title (max 80 chars)
- "tags": array of relevant tag strings (max 5, lowercase, no spaces — use underscores)
- "summary": a 1-3 sentence summary of the content
- "excerpt": (optional) a single key quote or sentence worth preserving verbatim, or null

Respond ONLY with the JSON object, no markdown fences."#
        .into()
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

                loop {
                    match backend.complete(req_attempt.clone()).await {
                        Ok(LlmCompletion::Message(resp)) => {
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
                            turns += 1;
                        }
                        Err(e) => {
                            warn!(?e, backend = backend.name(), attempt, "LLM attempt failed");
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
        }

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

#[contract(requires: !calls.is_empty())]
async fn execute_tool_calls(
    executor: &tools::ToolExecutor,
    calls: &[ToolCall],
    req: &LlmRequest,
) -> String {
    let mut outputs = Vec::with_capacity(calls.len());
    for call in calls {
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
                outputs.push(format!("tool `{}`: {text}", call.name));
            }
            Err(e) => {
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
mod tests {
    use super::*;
    use crate::config::FallbackMode;
    use crate::message::{EnrichedMessage, IncomingMessage, MessageSource, SourceMetadata};
    use crate::url_content::UrlContent;

    fn make_enriched(text: &str) -> EnrichedMessage {
        let msg = IncomingMessage::new(
            MessageSource::Http,
            text.into(),
            SourceMetadata::Http {
                remote_addr: None,
                user_agent: None,
            },
        );
        EnrichedMessage {
            original: msg,
            urls: vec![],
            url_contents: vec![],
        }
    }

    #[test]
    fn llm_request_default_system_prompt() {
        let cfg = crate::test_helpers::no_llm_config();
        let enriched = make_enriched("hello");
        let req = LlmRequest::from_enriched(&enriched, &cfg, std::path::Path::new("/tmp"), "");
        assert!(!req.system_prompt.is_empty());
        assert_eq!(req.user_content, "hello");
    }

    #[test]
    fn llm_request_custom_system_prompt() {
        let mut cfg = crate::test_helpers::no_llm_config();
        cfg.system_prompt = "custom prompt".into();
        let enriched = make_enriched("text");
        let req = LlmRequest::from_enriched(&enriched, &cfg, std::path::Path::new("/tmp"), "");
        assert_eq!(req.system_prompt, "custom prompt");
    }

    #[test]
    fn llm_request_appends_url_contents() {
        let cfg = crate::test_helpers::no_llm_config();
        let mut enriched = make_enriched("base text");
        enriched.url_contents.push(UrlContent {
            url: "http://example.com".into(),
            text: "page content".into(),
            page_title: None,
        });
        let req = LlmRequest::from_enriched(&enriched, &cfg, std::path::Path::new("/tmp"), "");
        assert!(req.user_content.contains("page content"));
        assert!(req.user_content.contains("http://example.com"));
    }

    #[test]
    fn llm_request_appends_tool_prompt_block() {
        let cfg = crate::test_helpers::no_llm_config();
        let enriched = make_enriched("base text");
        let req = LlmRequest::from_enriched(
            &enriched,
            &cfg,
            std::path::Path::new("/tmp"),
            "Tool crawl_url: prefer markdown first",
        );
        assert!(req.system_prompt.contains("Tool-specific guidance:"));
        assert!(req.system_prompt.contains("prefer markdown first"));
    }

    #[tokio::test]
    async fn chain_returns_success() {
        let resp = crate::test_helpers::default_llm_response();
        let chain = crate::test_helpers::mock_llm_chain(resp.clone());
        let enriched = make_enriched("test");
        let cfg = crate::test_helpers::no_llm_config();
        let req = LlmRequest::from_enriched(&enriched, &cfg, std::path::Path::new("/tmp"), "");
        let outcome = chain.complete(req).await;
        assert!(matches!(outcome, LlmOutcome::Success(_)));
    }

    #[tokio::test]
    async fn chain_raw_fallback_when_no_backends() {
        let chain = LlmChain::new(vec![], FallbackMode::Raw, 5, None);
        let req = LlmRequest::simple("s", "u");
        let outcome = chain.complete(req).await;
        assert!(matches!(outcome, LlmOutcome::RawFallback));
    }

    #[tokio::test]
    async fn chain_discard_fallback_when_no_backends() {
        let chain = LlmChain::new(vec![], FallbackMode::Discard, 5, None);
        let req = LlmRequest::simple("s", "u");
        let outcome = chain.complete(req).await;
        assert!(matches!(outcome, LlmOutcome::Discard));
    }

    #[test]
    fn max_tool_turns_accessor() {
        let chain = LlmChain::new(vec![], FallbackMode::Raw, 7, None);
        assert_eq!(chain.max_tool_turns(), 7);
    }
}

// ── Test helpers (also used by integration tests) ─────────────────────────────

#[cfg(any(test, feature = "test-helpers"))]
pub mod mock {
    use super::{InboxError, LlmClient, LlmCompletion, LlmRequest, LlmResponse, async_trait};

    pub struct MockLlm {
        pub response: LlmResponse,
        pub name: String,
    }

    impl MockLlm {
        #[must_use]
        pub fn new(response: LlmResponse) -> Self {
            Self {
                response,
                name: "mock".into(),
            }
        }
    }

    #[async_trait]
    impl LlmClient for MockLlm {
        fn name(&self) -> &str {
            &self.name
        }
        fn retries(&self) -> u32 {
            1
        }
        async fn complete(&self, _req: LlmRequest) -> Result<LlmCompletion, InboxError> {
            Ok(LlmCompletion::Message(self.response.clone()))
        }
    }
}
