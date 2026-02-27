use thiserror::Error;

#[derive(Debug, Error)]
pub enum InboxError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Template render error: {0}")]
    Template(#[from] askama::Error),

    #[error("URL parse error: {0}")]
    UrlParse(#[from] url::ParseError),

    #[error("Config error: {0}")]
    Config(String),

    #[error("LLM error: {0}")]
    Llm(String),

    #[error("LLM tool error: {0}")]
    LlmTool(String),

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
}
