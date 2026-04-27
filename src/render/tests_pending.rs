use crate::message::{
    EnrichedMessage, IncomingMessage, LlmResponse, MessageSource, ProcessedMessage,
    ProcessingHints, SourceMetadata,
};
use crate::render::render_org_node;

fn make_processed(text: &str, llm_response: Option<LlmResponse>) -> ProcessedMessage {
    let msg = IncomingMessage::new(
        MessageSource::Http,
        text.into(),
        SourceMetadata::Http {
            remote_addr: None,
            user_agent: None,
        },
    );
    ProcessedMessage {
        enriched: EnrichedMessage {
            original: msg,
            urls: vec![],
            url_contents: vec![],
        },
        llm_response,
        fallback_source_urls: vec![],
        fallback_tool_results: vec![],
        fallback_title: None,
        enrichment: crate::message::EnrichmentMetadata::default(),
    }
}

#[test]
fn render_fallback_adds_inbox_pending_tag() {
    let msg = make_processed("Some text", None);
    let result = render_org_node(&msg, std::path::Path::new("/tmp")).unwrap();
    assert!(
        result.contains(":inbox_pending:"),
        "fallback render must include :inbox_pending: tag, got:\n{result}"
    );
}

#[test]
fn render_with_llm_response_has_no_inbox_pending_tag() {
    let resp = LlmResponse {
        title: "Title".into(),
        tags: vec![],
        summary: "summary".into(),
        excerpt: None,
        produced_by: "mock".into(),
    };
    let msg = make_processed("text", Some(resp));
    let result = render_org_node(&msg, std::path::Path::new("/tmp")).unwrap();
    assert!(
        !result.contains(":inbox_pending:"),
        "successful LLM render must NOT include :inbox_pending: tag, got:\n{result}"
    );
}

#[test]
fn merge_tags_deduplicates_suggested_tags() {
    let mut msg = IncomingMessage::new(
        MessageSource::Http,
        "text".into(),
        SourceMetadata::Http {
            remote_addr: None,
            user_agent: None,
        },
    );
    msg.user_tags = vec!["Rust".into()];
    msg.preprocessing_hints = ProcessingHints {
        suggested_tags: vec!["rust".into(), "async".into()],
        ..Default::default()
    };
    let processed = ProcessedMessage {
        enriched: EnrichedMessage {
            original: msg,
            urls: vec![],
            url_contents: vec![],
        },
        llm_response: None,
        fallback_source_urls: vec![],
        fallback_tool_results: vec![],
        fallback_title: None,
        enrichment: crate::message::EnrichmentMetadata::default(),
    };
    let result = render_org_node(&processed, std::path::Path::new("/tmp")).unwrap();
    // "rust" (case-insensitive dup of "Rust") should not appear twice; "async" added once.
    let rust_count = result.matches("rust").count() + result.matches("Rust").count();
    assert_eq!(
        rust_count, 1,
        "deduplicated tag should appear only once: {result}"
    );
    assert!(
        result.contains("async"),
        "unique suggested tag should appear: {result}"
    );
    assert!(
        result.contains("inbox_pending"),
        "pending tag still present: {result}"
    );
}
