//! Render tests for raw-fallback paths and enrichment metadata.

use super::tests::make_processed;
use super::*;
use crate::message::{
    EnrichedMessage, IncomingMessage, LlmResponse, MessageSource, ProcessedMessage, SourceMetadata,
};

#[test]
fn render_fallback_uses_tool_content_as_summary() {
    let msg = IncomingMessage::new(
        MessageSource::Http,
        "Original raw text".into(),
        SourceMetadata::Http {
            remote_addr: None,
            user_agent: None,
        },
    );
    let processed = ProcessedMessage {
        enriched: EnrichedMessage {
            original: msg,
            urls: vec![],
            url_contents: vec![],
        },
        llm_response: None,
        incomplete: crate::message::ProcessingCompleteness::Complete,
        fallback_source_urls: vec![],
        fallback_tool_results: vec![(
            "scrape_page".to_owned(),
            "Tool gathered summary content".to_owned(),
        )],
        fallback_title: None,
        enrichment: crate::message::EnrichmentMetadata::default(),
    };
    let result = render_org_node(&processed, std::path::Path::new("/tmp")).unwrap();
    assert!(
        result.contains("Tool gathered summary content"),
        "fallback_tool_results should be used as summary: {result}"
    );
    assert!(
        !result.contains("Original raw text") || result.contains("Tool gathered summary content"),
        "tool content should take precedence over raw text: {result}"
    );
}

#[test]
fn render_fallback_source_urls_in_roam_refs() {
    let msg = IncomingMessage::new(
        MessageSource::Http,
        "Some note".into(),
        SourceMetadata::Http {
            remote_addr: None,
            user_agent: None,
        },
    );
    let processed = ProcessedMessage {
        enriched: EnrichedMessage {
            original: msg,
            urls: vec![],
            url_contents: vec![],
        },
        llm_response: None,
        incomplete: crate::message::ProcessingCompleteness::Complete,
        fallback_source_urls: vec![
            "https://tool-found.example.com/page1".into(),
            "https://tool-found.example.com/page2".into(),
        ],
        fallback_tool_results: vec![],
        fallback_title: None,
        enrichment: crate::message::EnrichmentMetadata::default(),
    };
    let result = render_org_node(&processed, std::path::Path::new("/tmp")).unwrap();
    assert!(
        result.contains("https://tool-found.example.com/page1"),
        "fallback_source_urls[0] should appear in ROAM_REFS: {result}"
    );
    assert!(
        result.contains("https://tool-found.example.com/page2"),
        "fallback_source_urls[1] should appear in ROAM_REFS: {result}"
    );
}

#[test]
fn render_fallback_tool_results_joined_cleanly() {
    let msg = IncomingMessage::new(
        MessageSource::Http,
        String::new(),
        SourceMetadata::Http {
            remote_addr: None,
            user_agent: None,
        },
    );
    let processed = ProcessedMessage {
        enriched: EnrichedMessage {
            original: msg,
            urls: vec![],
            url_contents: vec![],
        },
        llm_response: None,
        incomplete: crate::message::ProcessingCompleteness::Complete,
        fallback_source_urls: vec![],
        fallback_tool_results: vec![
            ("web_search".to_owned(), "First result content".to_owned()),
            ("scrape_page".to_owned(), "Second result content".to_owned()),
        ],
        fallback_title: None,
        enrichment: crate::message::EnrichmentMetadata::default(),
    };
    let result = render_org_node(&processed, std::path::Path::new("/tmp")).unwrap();
    assert!(
        result.contains("First result content"),
        "first tool result should appear: {result}"
    );
    assert!(
        result.contains("Second result content"),
        "second tool result should appear: {result}"
    );
    assert!(
        !result.contains("--- Tool execution results ---"),
        "LLM separator markers should not appear in output: {result}"
    );
    assert!(
        !result.contains("tool `web_search`"),
        "tool name prefixes should not appear in output: {result}"
    );
}

#[test]
fn render_fallback_title_used_when_present() {
    let msg = IncomingMessage::new(
        MessageSource::Http,
        String::new(),
        SourceMetadata::Http {
            remote_addr: None,
            user_agent: None,
        },
    );
    let processed = ProcessedMessage {
        enriched: EnrichedMessage {
            original: msg,
            urls: vec![],
            url_contents: vec![],
        },
        llm_response: None,
        incomplete: crate::message::ProcessingCompleteness::Complete,
        fallback_source_urls: vec![],
        fallback_tool_results: vec![],
        fallback_title: Some("Five Word Generated Title".to_owned()),
        enrichment: crate::message::EnrichmentMetadata::default(),
    };
    let result = render_org_node(&processed, std::path::Path::new("/tmp")).unwrap();
    assert!(
        result.contains("* Five Word Generated Title"),
        "fallback_title should be used as heading: {result}"
    );
}

#[test]
fn render_empty_text_image_uses_media_kind() {
    use crate::message::Attachment;

    let mut msg = IncomingMessage::new(
        MessageSource::Telegram,
        String::new(),
        SourceMetadata::Telegram {
            chat_id: 1,
            message_id: 3,
            username: None,
            forwarded_from: None,
        },
    );
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("photo.jpg");
    std::fs::write(&path, b"jpg").unwrap();
    msg.attachments.push(Attachment {
        original_name: "photo.jpg".into(),
        saved_path: path,
        mime_type: Some("image/jpeg".into()),
        media_kind: crate::message::MediaKind::Image,
    });
    let processed = ProcessedMessage {
        enriched: EnrichedMessage {
            original: msg,
            urls: vec![],
            url_contents: vec![],
        },
        llm_response: None,
        incomplete: crate::message::ProcessingCompleteness::Complete,
        fallback_source_urls: vec![],
        fallback_tool_results: vec![],
        fallback_title: None,
        enrichment: crate::message::EnrichmentMetadata::default(),
    };
    let result = render_org_node(&processed, tmp.path()).unwrap();
    assert!(
        result.contains("* Image"),
        "empty-text image should use 'Image' as title: {result}"
    );
}

#[test]
fn render_untitled_when_nothing_available() {
    let msg = IncomingMessage::new(
        MessageSource::Http,
        String::new(),
        SourceMetadata::Http {
            remote_addr: None,
            user_agent: None,
        },
    );
    let processed = ProcessedMessage {
        enriched: EnrichedMessage {
            original: msg,
            urls: vec![],
            url_contents: vec![],
        },
        llm_response: None,
        incomplete: crate::message::ProcessingCompleteness::Complete,
        fallback_source_urls: vec![],
        fallback_tool_results: vec![],
        fallback_title: None,
        enrichment: crate::message::EnrichmentMetadata::default(),
    };
    let result = render_org_node(&processed, std::path::Path::new("/tmp")).unwrap();
    assert!(
        result.contains("* (untitled)"),
        "should fall back to (untitled) when nothing available: {result}"
    );
}

// ── EnrichmentMetadata rendering ──────────────────────────────────────────────

fn with_enrichment(text: &str, enrichment: crate::message::EnrichmentMetadata) -> ProcessedMessage {
    let mut msg = make_processed(
        text,
        Some(LlmResponse {
            title: "T".into(),
            tags: vec![],
            summary: "S".into(),
            excerpt: None,
            produced_by: "free_router:primary/model".into(),
        }),
    );
    msg.enrichment = enrichment;
    msg
}

#[test]
fn render_enriched_by_contains_backend_and_model() {
    let msg = with_enrichment("x", crate::message::EnrichmentMetadata::default());
    let out = render_org_node(&msg, std::path::Path::new("/tmp")).unwrap();
    assert!(
        out.contains(":ENRICHED_BY: free_router:primary/model"),
        "got:\n{out}"
    );
}

#[test]
fn render_enriched_with_lists_helpers_when_non_empty() {
    let msg = with_enrichment(
        "x",
        crate::message::EnrichmentMetadata {
            helpers: vec!["free_router:helper/one".into(), "ollama:llama3".into()],
            ..Default::default()
        },
    );
    let out = render_org_node(&msg, std::path::Path::new("/tmp")).unwrap();
    assert!(
        out.contains(":ENRICHED_WITH: free_router:helper/one, ollama:llama3"),
        "expected :ENRICHED_WITH: line with both helpers, got:\n{out}"
    );
}

#[test]
fn render_omits_enriched_with_when_no_helpers() {
    let msg = with_enrichment("x", crate::message::EnrichmentMetadata::default());
    let out = render_org_node(&msg, std::path::Path::new("/tmp")).unwrap();
    assert!(
        !out.contains("ENRICHED_WITH"),
        "no helpers should mean no property line; got:\n{out}"
    );
}

#[test]
fn render_stats_properties_appear_when_nonzero() {
    let msg = with_enrichment(
        "x",
        crate::message::EnrichmentMetadata {
            helpers: vec![],
            memories_recalled: 3,
            urls_fetched: 2,
            tool_calls_made: 5,
        },
    );
    let out = render_org_node(&msg, std::path::Path::new("/tmp")).unwrap();
    assert!(out.contains(":MEMORIES_RECALLED: 3"), "got:\n{out}");
    assert!(out.contains(":URLS_FETCHED: 2"), "got:\n{out}");
    assert!(out.contains(":TOOL_CALLS: 5"), "got:\n{out}");
}

#[test]
fn render_stats_properties_omitted_when_zero() {
    let msg = with_enrichment("x", crate::message::EnrichmentMetadata::default());
    let out = render_org_node(&msg, std::path::Path::new("/tmp")).unwrap();
    assert!(!out.contains("MEMORIES_RECALLED"), "got:\n{out}");
    assert!(!out.contains("URLS_FETCHED"), "got:\n{out}");
    assert!(!out.contains("TOOL_CALLS"), "got:\n{out}");
}
