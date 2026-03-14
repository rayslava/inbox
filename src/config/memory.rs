use serde::Deserialize;

/// Configuration for the persistent LLM memory store.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct MemoryConfig {
    /// Enable the memory tool (`memory_save` / `memory_recall`).
    #[serde(default)]
    pub enabled: bool,
    /// `SQLite` database path. Defaults to `{attachments_dir}/memory.db`.
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
