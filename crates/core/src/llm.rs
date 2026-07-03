use crate::CoreError;
use async_trait::async_trait;

/// Minimal LLM completion boundary for downstream crates (brain RAG, `kb-web`).
///
/// The rich internal request/tool machinery (`LlmRequest`, tool calls, progress
/// events, multi-backend fallback) stays in the `inbox` binary; `core` exposes
/// only a prompt→answer surface. Implemented by the daemon's `LlmChain`.
#[async_trait]
pub trait LlmBackend: Send + Sync {
    /// Complete a `system`+`user` prompt, returning `(answer, "backend:model")`.
    ///
    /// # Errors
    /// Returns [`CoreError`] if the chain produced no structured answer (every
    /// backend failed or the request was discarded).
    async fn complete_text(&self, system: &str, user: &str) -> Result<(String, String), CoreError>;
}
