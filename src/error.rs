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

    #[error("Memory error: {0}")]
    Memory(String),
}

/// Boundary conversion into the dependency-light `inbox_core::CoreError`.
///
/// Category-preserving for every string variant; the two heavy variants
/// (`Http`/`Template`, whose source types `core` deliberately excludes) degrade
/// to the `Fetch`/`Output` string categories carrying their `Display` text.
impl From<InboxError> for inbox_core::CoreError {
    fn from(err: InboxError) -> Self {
        use inbox_core::CoreError as C;
        match err {
            InboxError::Io(e) => C::Io(e),
            InboxError::Json(e) => C::Json(e),
            InboxError::UrlParse(e) => C::UrlParse(e),
            InboxError::Http(e) => C::Fetch(e.to_string()),
            InboxError::Template(e) => C::Output(e.to_string()),
            InboxError::Config(s) => C::Config(s),
            InboxError::Llm(s) => C::Llm(s),
            InboxError::LlmTool(s) => C::LlmTool(s),
            InboxError::Attachment(s) => C::Attachment(s),
            InboxError::Auth(s) => C::Auth(s),
            InboxError::Pipeline(s) => C::Pipeline(s),
            InboxError::Adapter(s) => C::Adapter(s),
            InboxError::Output(s) => C::Output(s),
            InboxError::Memory(s) => C::Memory(s),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::InboxError;
    use inbox_core::CoreError;

    #[test]
    fn string_variants_map_category_preserving() {
        assert!(matches!(
            CoreError::from(InboxError::Config("x".into())),
            CoreError::Config(_)
        ));
        assert!(matches!(
            CoreError::from(InboxError::Llm("x".into())),
            CoreError::Llm(_)
        ));
        assert!(matches!(
            CoreError::from(InboxError::LlmTool("x".into())),
            CoreError::LlmTool(_)
        ));
        assert!(matches!(
            CoreError::from(InboxError::Attachment("x".into())),
            CoreError::Attachment(_)
        ));
        assert!(matches!(
            CoreError::from(InboxError::Auth("x".into())),
            CoreError::Auth(_)
        ));
        assert!(matches!(
            CoreError::from(InboxError::Pipeline("x".into())),
            CoreError::Pipeline(_)
        ));
        assert!(matches!(
            CoreError::from(InboxError::Adapter("x".into())),
            CoreError::Adapter(_)
        ));
        assert!(matches!(
            CoreError::from(InboxError::Output("x".into())),
            CoreError::Output(_)
        ));
        assert!(matches!(
            CoreError::from(InboxError::Memory("x".into())),
            CoreError::Memory(_)
        ));
    }

    #[test]
    fn source_variants_map_and_preserve_message() {
        let io = CoreError::from(InboxError::Io(std::io::Error::other("disk")));
        assert!(matches!(io, CoreError::Io(_)));
        assert!(io.to_string().contains("disk"));

        let json = InboxError::Json(serde_json::from_str::<i32>("nope").unwrap_err());
        assert!(matches!(CoreError::from(json), CoreError::Json(_)));

        let urlp = InboxError::UrlParse(url::Url::parse("http://[bad").unwrap_err());
        assert!(matches!(CoreError::from(urlp), CoreError::UrlParse(_)));
    }
}
