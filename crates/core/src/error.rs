use thiserror::Error;

/// Error type for `inbox-core` trait boundaries.
///
/// Intentionally free of heavy transport deps (no `reqwest`/`askama`): adapters
/// in the `inbox` binary map their concrete errors into these variants, keeping
/// `core` dependency-light. Most variants carry a message string; the `#[from]`
/// variants cover errors whose source types are themselves dependency-light.
///
/// The string categories **mirror the daemon's `InboxError`** one-to-one so the
/// boundary conversion (`From<InboxError>`, in the `inbox` crate) is
/// category-preserving. The daemon's two heavy variants (`Http(reqwest::Error)`,
/// `Template(askama::Error)`) have no `core` equivalent by design and degrade to
/// the `Fetch` / `Output` string categories at the boundary.
#[derive(Debug, Error)]
pub enum CoreError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("URL parse error: {0}")]
    UrlParse(#[from] url::ParseError),

    #[error("Config error: {0}")]
    Config(String),

    #[error("LLM error: {0}")]
    Llm(String),

    #[error("LLM tool error: {0}")]
    LlmTool(String),

    #[error("Embedding error: {0}")]
    Embedding(String),

    #[error("Vector store error: {0}")]
    VectorStore(String),

    #[error("Fetch error: {0}")]
    Fetch(String),

    #[error("Attachment error: {0}")]
    Attachment(String),

    #[error("Auth error: {0}")]
    Auth(String),

    #[error("Pipeline error: {0}")]
    Pipeline(String),

    #[error("Adapter error: {0}")]
    Adapter(String),

    #[error("Output error: {0}")]
    Output(String),

    #[error("Memory error: {0}")]
    Memory(String),
}
