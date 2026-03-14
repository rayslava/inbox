/// Test utilities: `MockLlm`-based chains, sample data, temp dirs.
///
/// Compiled only when the `test-helpers` feature is enabled.
use std::path::PathBuf;
use std::sync::Arc;

use crate::{
    config::{FallbackMode, LlmConfig},
    llm::{LlmChain, LlmClient, mock::MockLlm},
    message::LlmResponse,
};

/// Build a `LlmChain` backed by a `MockLlm` that always returns `response`.
#[must_use]
pub fn mock_llm_chain(response: LlmResponse) -> Arc<LlmChain> {
    let client = Box::new(MockLlm::new(response)) as Box<dyn LlmClient>;
    Arc::new(LlmChain::new(vec![client], FallbackMode::Raw, 3, None, 1))
}

/// Build a `LlmChain` backed by a `MockLlm` that always returns `InboxError::Llm`,
/// with `FallbackMode::Discard` so the pipeline propagates the failure as `Err`.
#[must_use]
pub fn failing_llm_chain(message: impl Into<String>) -> Arc<LlmChain> {
    let client = Box::new(MockLlm::failing(message)) as Box<dyn LlmClient>;
    Arc::new(LlmChain::new(
        vec![client],
        FallbackMode::Discard,
        3,
        None,
        1,
    ))
}

/// Build a default `LlmResponse` for use in tests.
#[must_use]
pub fn default_llm_response() -> LlmResponse {
    LlmResponse {
        title: "Test title".into(),
        tags: vec!["test".into()],
        summary: "A test summary.".into(),
        excerpt: None,
        produced_by: "mock".into(),
    }
}

/// Create a temporary directory and return its path.
///
/// # Panics
/// Panics if the OS cannot create a temp directory.
#[must_use]
pub fn temp_dir() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().to_owned();
    (dir, path)
}

/// Build a minimal `LlmConfig` with no real backends.
#[must_use]
pub fn no_llm_config() -> LlmConfig {
    LlmConfig {
        fallback: FallbackMode::Raw,
        url_content_max_chars: 4000,
        max_tool_turns: 3,
        max_llm_tool_depth: 1,
        vision_max_bytes: 5 * 1024 * 1024,
        prompts: crate::config::LlmPromptsConfig::default(),
        backends: vec![],
    }
}
