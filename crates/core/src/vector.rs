use crate::CoreError;
use async_trait::async_trait;

/// A recalled memory/KB entry with its relevance score.
#[derive(Debug, Clone)]
pub struct MemoryEntry {
    pub key: String,
    pub value: String,
    pub score: f64,
}

/// A source node linked to a memory (e.g. the note/message it came from).
#[derive(Debug, Clone)]
pub struct SourceEntry {
    pub kind: String,
    pub source_id: String,
    pub title: String,
}

/// Persistent associative memory + KB store: save entries, link provenance, and
/// recall by hybrid vector/text search.
///
/// Implemented in the `inbox` binary over the embedded Grafeo engine; `core`
/// exposes only the value types and async surface so downstream crates
/// (`kb-web` RAG, `brain`) depend on the trait, never on grafeo.
#[async_trait]
pub trait VectorStore: Send + Sync {
    /// Upsert a memory under `key` with `value` (embedded if a provider is set).
    ///
    /// # Errors
    /// Returns [`CoreError`] if the underlying store write fails.
    async fn save(&self, key: &str, value: &str) -> Result<(), CoreError>;

    /// Link a memory to a provenance source node.
    ///
    /// # Errors
    /// Returns [`CoreError`] if the link cannot be created.
    async fn link_source(
        &self,
        memory_key: &str,
        source_kind: &str,
        source_id: &str,
        title: &str,
    ) -> Result<(), CoreError>;

    /// Link two memories with a named relation.
    ///
    /// # Errors
    /// Returns [`CoreError`] if the relation cannot be created.
    async fn link_memories(
        &self,
        from_key: &str,
        to_key: &str,
        relation: &str,
    ) -> Result<(), CoreError>;

    /// Recall up to `limit` entries most relevant to `query`.
    ///
    /// # Errors
    /// Returns [`CoreError`] if the query fails.
    async fn recall(&self, query: &str, limit: usize) -> Result<Vec<MemoryEntry>, CoreError>;

    /// Recall entries reachable within `hops` graph steps of `query` matches.
    ///
    /// # Errors
    /// Returns [`CoreError`] if the traversal fails.
    async fn context(&self, query: &str, hops: u32) -> Result<Vec<MemoryEntry>, CoreError>;

    /// List the source nodes linked to `memory_key`.
    ///
    /// # Errors
    /// Returns [`CoreError`] if the lookup fails.
    async fn sources(&self, memory_key: &str) -> Result<Vec<SourceEntry>, CoreError>;
}
