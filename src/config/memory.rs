use serde::Deserialize;

/// Configuration for the persistent LLM memory store (Grafeo graph database).
#[derive(Debug, Clone, Deserialize, Default)]
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
}
