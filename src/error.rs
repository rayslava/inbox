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

/// Reverse boundary conversion, so daemon code that works in `InboxError` can
/// `?` a `CoreError` returned by a core-trait impl. Category-preserving; the two
/// core-only categories (`Embedding`/`VectorStore`) fold into `Memory` and
/// `Fetch` into `Adapter`, carrying their message.
impl From<inbox_core::CoreError> for InboxError {
    fn from(err: inbox_core::CoreError) -> Self {
        use inbox_core::CoreError as C;
        match err {
            C::Io(e) => InboxError::Io(e),
            C::Json(e) => InboxError::Json(e),
            C::UrlParse(e) => InboxError::UrlParse(e),
            C::Config(s) => InboxError::Config(s),
            C::Llm(s) => InboxError::Llm(s),
            C::LlmTool(s) => InboxError::LlmTool(s),
            // Core-only categories fold into the nearest daemon category.
            C::Embedding(s) | C::VectorStore(s) | C::Memory(s) => InboxError::Memory(s),
            C::Fetch(s) | C::Adapter(s) => InboxError::Adapter(s),
            C::Attachment(s) => InboxError::Attachment(s),
            C::Auth(s) => InboxError::Auth(s),
            C::Pipeline(s) => InboxError::Pipeline(s),
            C::Output(s) => InboxError::Output(s),
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

    #[test]
    fn reverse_core_to_inbox_is_category_preserving() {
        use inbox_core::CoreError as C;
        assert!(matches!(
            InboxError::from(C::Config("x".into())),
            InboxError::Config(_)
        ));
        assert!(matches!(
            InboxError::from(C::Llm("x".into())),
            InboxError::Llm(_)
        ));
        assert!(matches!(
            InboxError::from(C::LlmTool("x".into())),
            InboxError::LlmTool(_)
        ));
        // Core-only categories fold into the nearest daemon category.
        assert!(matches!(
            InboxError::from(C::Embedding("x".into())),
            InboxError::Memory(_)
        ));
        assert!(matches!(
            InboxError::from(C::VectorStore("x".into())),
            InboxError::Memory(_)
        ));
        assert!(matches!(
            InboxError::from(C::Fetch("x".into())),
            InboxError::Adapter(_)
        ));
        assert!(matches!(
            InboxError::from(C::Attachment("x".into())),
            InboxError::Attachment(_)
        ));
        assert!(matches!(
            InboxError::from(C::Auth("x".into())),
            InboxError::Auth(_)
        ));
        assert!(matches!(
            InboxError::from(C::Pipeline("x".into())),
            InboxError::Pipeline(_)
        ));
        assert!(matches!(
            InboxError::from(C::Adapter("x".into())),
            InboxError::Adapter(_)
        ));
        assert!(matches!(
            InboxError::from(C::Output("x".into())),
            InboxError::Output(_)
        ));
        assert!(matches!(
            InboxError::from(C::Memory("x".into())),
            InboxError::Memory(_)
        ));
        assert!(matches!(
            InboxError::from(C::Io(std::io::Error::other("d"))),
            InboxError::Io(_)
        ));
        let json = C::Json(serde_json::from_str::<i32>("nope").unwrap_err());
        assert!(matches!(InboxError::from(json), InboxError::Json(_)));
        let urlp = C::UrlParse(url::Url::parse("http://[bad").unwrap_err());
        assert!(matches!(InboxError::from(urlp), InboxError::UrlParse(_)));
    }
}
