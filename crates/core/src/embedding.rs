use crate::CoreError;
use async_trait::async_trait;

/// Produces an embedding vector for a piece of text.
///
/// Implemented by an adapter in the `inbox` binary (currently an Ollama-native
/// `/api/embed` client); `core` itself stays transport-free. Downstream crates
/// (`kb-web` RAG, `omi-bridge`) depend on this trait, never the concrete client.
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Embed `text`, returning the model's vector.
    ///
    /// `text` is expected to be **non-empty**; the concrete inbox adapter
    /// enforces this via an `anodized` precondition on its inherent `embed`.
    ///
    /// # Errors
    /// Returns [`CoreError`] if the embedding backend request fails or its
    /// response cannot be parsed (mapped from the adapter's own error type).
    async fn embed(&self, text: &str) -> Result<Vec<f32>, CoreError>;
}
