use serde::Deserialize;

/// Configuration for the persistent LLM memory store (Grafeo graph database).
#[derive(Debug, Clone, Deserialize)]
pub struct MemoryConfig {
    /// Enable the memory tools (`memory_save` / `memory_recall` / `memory_link` / `memory_context`).
    #[serde(default)]
    pub enabled: bool,
    /// Grafeo database path. Defaults to `{attachments_dir}/memory.grafeo`.
    pub db_path: Option<String>,
    /// OpenAI-compatible embeddings endpoint base URL
    /// (e.g. `http://localhost:11434/v1` for Ollama).
    pub embedding_endpoint: Option<String>,
    /// Embedding model name (e.g. `nomic-embed-text`).
    pub embedding_model: Option<String>,
    /// Embedding vector dimensions. Auto-detected via probe call if not set.
    pub embedding_dims: Option<usize>,
    /// Optional API key for the embedding endpoint.
    pub embedding_api_key: Option<String>,

    // ── Pre-load settings (active whenever `enabled = true`) ─────────────
    /// Maximum number of recalled memories to inject into the LLM context.
    #[serde(default = "default_preload_max_memories")]
    pub preload_max_memories: usize,
    /// Graph traversal depth when fetching related memories.
    #[serde(default = "default_preload_graph_hops")]
    pub preload_graph_hops: u32,
    /// Pre-load recent user feedback (especially low-rated) as behavioural guidance.
    #[serde(default = "super::infra::bool_true")]
    pub preload_feedback: bool,
    /// Maximum number of recent feedback entries to inject.
    #[serde(default = "default_preload_max_feedback")]
    pub preload_max_feedback: usize,
    /// Only include feedback with rating at or below this value (1-3).
    #[serde(default = "default_preload_feedback_max_rating")]
    pub preload_feedback_max_rating: u8,
}

fn default_preload_max_memories() -> usize {
    5
}
fn default_preload_graph_hops() -> u32 {
    2
}
fn default_preload_max_feedback() -> usize {
    10
}
fn default_preload_feedback_max_rating() -> u8 {
    2
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            db_path: None,
            embedding_endpoint: None,
            embedding_model: None,
            embedding_dims: None,
            embedding_api_key: None,
            preload_max_memories: default_preload_max_memories(),
            preload_graph_hops: default_preload_graph_hops(),
            preload_feedback: true,
            preload_max_feedback: default_preload_max_feedback(),
            preload_feedback_max_rating: default_preload_feedback_max_rating(),
        }
    }
}
