use serde::Deserialize;

/// Wire dialect of the embeddings endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingApi {
    /// Ollama-native `POST /api/embed`, response `{"embeddings": [[...]]}`.
    #[default]
    Ollama,
    /// OpenAI-compatible `POST /embeddings` (e.g. llama.cpp `server` at
    /// `.../v1`), response `{"data": [{"embedding": [...]}]}`.
    Openai,
}

/// Configuration for the persistent LLM memory store (Grafeo graph database).
#[derive(Debug, Clone, Deserialize)]
pub struct MemoryConfig {
    /// Enable the memory tools (`memory_save` / `memory_recall` / `memory_link` / `memory_context`).
    #[serde(default)]
    pub enabled: bool,
    /// Grafeo database path. Defaults to `{attachments_dir}/memory.grafeo`.
    pub db_path: Option<String>,
    /// Root directory of the org corpus to index into the KB (used by the
    /// `index-kb` command / reindex). No default — required to run indexing.
    pub kb_root: Option<String>,
    /// Embeddings endpoint base URL. For `ollama` this is the Ollama base
    /// (e.g. `http://localhost:11434`); for `openai` it includes the version
    /// prefix (e.g. `http://localhost:8080/v1` for a llama.cpp `server`).
    pub embedding_endpoint: Option<String>,
    /// Wire dialect of `embedding_endpoint`. Default: `ollama`.
    #[serde(default)]
    pub embedding_api: EmbeddingApi,
    /// Embedding model name (e.g. `nomic-embed-text`).
    pub embedding_model: Option<String>,
    /// Embedding vector dimensions. Auto-detected via probe call if not set.
    pub embedding_dims: Option<usize>,
    /// Optional API key for the embedding endpoint.
    pub embedding_api_key: Option<String>,
    /// Task prefix prepended to stored documents/passages before embedding.
    /// Asymmetric embedders need this — nomic: `"search_document: "`. `None` = none.
    pub embedding_document_prefix: Option<String>,
    /// Task prefix prepended to search queries before embedding.
    /// nomic: `"search_query: "`. `None` = none.
    pub embedding_query_prefix: Option<String>,

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
            kb_root: None,
            embedding_endpoint: None,
            embedding_api: EmbeddingApi::default(),
            embedding_model: None,
            embedding_dims: None,
            embedding_api_key: None,
            embedding_document_prefix: None,
            embedding_query_prefix: None,
            preload_max_memories: default_preload_max_memories(),
            preload_graph_hops: default_preload_graph_hops(),
            preload_feedback: true,
            preload_max_feedback: default_preload_max_feedback(),
            preload_feedback_max_rating: default_preload_feedback_max_rating(),
        }
    }
}
