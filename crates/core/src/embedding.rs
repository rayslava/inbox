use crate::CoreError;
use async_trait::async_trait;

/// Produces an embedding vector for a piece of text.
///
/// Implemented by an adapter in the `inbox` binary (currently an Ollama-native
/// `/api/embed` client); `core` itself stays transport-free. Downstream crates
/// (`kb-web` RAG, `omi-bridge`) depend on this trait, never the concrete client.
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Embed `text` verbatim, returning the model's vector.
    ///
    /// `text` is expected to be **non-empty**; the concrete inbox adapter
    /// enforces this via an `anodized` precondition on its inherent `embed`.
    ///
    /// # Errors
    /// Returns [`CoreError`] if the embedding backend request fails or its
    /// response cannot be parsed (mapped from the adapter's own error type).
    async fn embed(&self, text: &str) -> Result<Vec<f32>, CoreError>;

    /// Embed a **document/passage** for storage. Asymmetric embedders (e.g.
    /// nomic) prepend a task prefix here; the default is verbatim [`Self::embed`].
    ///
    /// # Errors
    /// Returns [`CoreError`] on backend or parse failure.
    async fn embed_document(&self, text: &str) -> Result<Vec<f32>, CoreError> {
        self.embed(text).await
    }

    /// Embed a **search query**. Asymmetric embedders prepend a different task
    /// prefix here; the default is verbatim [`Self::embed`].
    ///
    /// # Errors
    /// Returns [`CoreError`] on backend or parse failure.
    async fn embed_query(&self, text: &str) -> Result<Vec<f32>, CoreError> {
        self.embed(text).await
    }
}
