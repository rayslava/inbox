//! Bridges the daemon's `LlmChain` to the `inbox_core::LlmBackend` boundary,
//! exposing only a prompt→answer surface to downstream crates. The rich
//! `LlmRequest`/tool machinery stays internal to the `inbox` binary.

use async_trait::async_trait;
use inbox_core::{CoreError, LlmBackend};

use super::LlmChain;

#[async_trait]
impl LlmBackend for LlmChain {
    async fn complete_text(&self, system: &str, user: &str) -> Result<(String, String), CoreError> {
        // Delegate to the inherent plain-text path (uses `complete_raw`, no tool
        // loop / structured-JSON parsing). UFCS resolves to the inherent method.
        LlmChain::complete_text(self, system, user)
            .await
            .ok_or_else(|| CoreError::Llm("LLM produced no answer".into()))
    }
}
