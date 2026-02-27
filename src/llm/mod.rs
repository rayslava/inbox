use async_trait::async_trait;
use tracing::warn;

use crate::config::{FallbackMode, LlmConfig};
use crate::error::InboxError;
use crate::message::{EnrichedMessage, LlmResponse};

pub mod ollama;
pub mod openrouter;
pub mod tools;

// ── Public request / response types ──────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct LlmRequest {
    pub system_prompt: String,
    pub user_content: String,
}

impl LlmRequest {
    #[must_use]
    pub fn from_enriched(enriched: &EnrichedMessage, cfg: &LlmConfig) -> Self {
        let mut user_content = enriched.original.text.clone();

        for uc in &enriched.url_contents {
            use std::fmt::Write as _;
            let _ = write!(
                user_content,
                "\n\n--- Content from {} ---\n{}",
                uc.url, uc.text
            );
        }

        Self {
            system_prompt: if cfg.system_prompt.is_empty() {
                default_system_prompt()
            } else {
                cfg.system_prompt.clone()
            },
            user_content,
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
}

impl LlmChain {
    #[must_use]
    pub fn new(
        backends: Vec<Box<dyn LlmClient>>,
        fallback: FallbackMode,
        max_tool_turns: usize,
    ) -> Self {
        Self {
            backends,
            fallback,
            max_tool_turns,
        }
    }

    /// Try each backend in order with retries. On exhaustion, apply fallback policy.
    pub async fn complete(&self, req: LlmRequest) -> LlmOutcome {
        for backend in &self.backends {
            for attempt in 0..backend.retries() {
                let start = std::time::Instant::now();
                match backend.complete(req.clone()).await {
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
                    Ok(LlmCompletion::ToolCalls(_)) => {
                        // Simple chain doesn't process tool calls; backends handle them internally.
                        warn!(
                            backend = backend.name(),
                            attempt, "Unexpected tool call in chain"
                        );
                    }
                    Err(e) => {
                        warn!(?e, backend = backend.name(), attempt, "LLM attempt failed");
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

// ── Builder ───────────────────────────────────────────────────────────────────

use crate::config::{LlmBackendConfig, LlmBackendType};

#[must_use]
pub fn build_chain(cfg: &LlmConfig) -> LlmChain {
    let backends: Vec<Box<dyn LlmClient>> = cfg.backends.iter().map(|b| build_backend(b)).collect();

    LlmChain::new(backends, cfg.fallback, cfg.max_tool_turns)
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
        let req = LlmRequest::from_enriched(&enriched, &cfg);
        assert!(!req.system_prompt.is_empty());
        assert_eq!(req.user_content, "hello");
    }

    #[test]
    fn llm_request_custom_system_prompt() {
        let mut cfg = crate::test_helpers::no_llm_config();
        cfg.system_prompt = "custom prompt".into();
        let enriched = make_enriched("text");
        let req = LlmRequest::from_enriched(&enriched, &cfg);
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
        let req = LlmRequest::from_enriched(&enriched, &cfg);
        assert!(req.user_content.contains("page content"));
        assert!(req.user_content.contains("http://example.com"));
    }

    #[tokio::test]
    async fn chain_returns_success() {
        let resp = crate::test_helpers::default_llm_response();
        let chain = crate::test_helpers::mock_llm_chain(resp.clone());
        let enriched = make_enriched("test");
        let cfg = crate::test_helpers::no_llm_config();
        let req = LlmRequest::from_enriched(&enriched, &cfg);
        let outcome = chain.complete(req).await;
        assert!(matches!(outcome, LlmOutcome::Success(_)));
    }

    #[tokio::test]
    async fn chain_raw_fallback_when_no_backends() {
        let chain = LlmChain::new(vec![], FallbackMode::Raw, 5);
        let req = LlmRequest {
            system_prompt: "s".into(),
            user_content: "u".into(),
        };
        let outcome = chain.complete(req).await;
        assert!(matches!(outcome, LlmOutcome::RawFallback));
    }

    #[tokio::test]
    async fn chain_discard_fallback_when_no_backends() {
        let chain = LlmChain::new(vec![], FallbackMode::Discard, 5);
        let req = LlmRequest {
            system_prompt: "s".into(),
            user_content: "u".into(),
        };
        let outcome = chain.complete(req).await;
        assert!(matches!(outcome, LlmOutcome::Discard));
    }

    #[test]
    fn max_tool_turns_accessor() {
        let chain = LlmChain::new(vec![], FallbackMode::Raw, 7);
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
