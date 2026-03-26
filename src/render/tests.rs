use super::*;
use crate::message::{
    EnrichedMessage, IncomingMessage, LlmResponse, MessageSource, ProcessedMessage, SourceMetadata,
};

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
    }
}

#[test]
fn render_with_llm_response() {
    let resp = LlmResponse {
        title: "My Title".into(),
        tags: vec!["rust".into(), "test".into()],
        summary: "A summary.".into(),
        excerpt: Some("Key quote".into()),
        produced_by: "mock".into(),
    };
    let msg = make_processed("raw text", Some(resp));
    let result = render_org_node(&msg, std::path::Path::new("/tmp")).unwrap();
    assert!(result.contains("* My Title"));
    assert!(result.contains(":rust:test:"));
    assert!(result.contains("A summary."));
    assert!(result.contains("Key quote"));
    assert!(result.contains(":ENRICHED_BY: mock"));
}

#[test]
fn render_without_llm_response_raw_fallback() {
    let msg = make_processed("First line\nSecond line", None);
    let result = render_org_node(&msg, std::path::Path::new("/tmp")).unwrap();
    assert!(result.contains("* First line"));
    assert!(result.contains(":ENRICHED_BY: none"));
    assert!(result.contains("First line"));
}

#[test]
fn render_empty_text_untitled() {
    let msg = make_processed("", None);
    let result = render_org_node(&msg, std::path::Path::new("/tmp")).unwrap();
    assert!(result.contains("(untitled)"));
}

#[test]
fn attachment_names_joined() {
    let tmpl = OrgNodeTemplate {
        title: "t",
        tags: &[],
        id: "id",
        created: "now",
        source: "http",
        urls: &[],
        roam_refs: &[],
        attachments: &[
            AttachmentRef {
                name: "a.pdf",
                path_rel: "a.pdf".to_owned(),
            },
            AttachmentRef {
                name: "b.jpg",
                path_rel: "b.jpg".to_owned(),
            },
        ],
        llm_backend: "mock",
        summary: "s",
        excerpt: None,
        raw_text: "",
        forwarded_from: None,
        media_kinds: &[],
    };
    assert_eq!(tmpl.attachment_names(), "a.pdf b.jpg");
}

#[test]
fn render_with_url_in_enriched() {
    let msg_inner = IncomingMessage::new(
        MessageSource::Http,
        "text".into(),
        SourceMetadata::Http {
            remote_addr: None,
            user_agent: None,
        },
    );
    let url: url::Url = "https://example.com/page".parse().unwrap();
    let msg = ProcessedMessage {
        enriched: EnrichedMessage {
            original: msg_inner,
            urls: vec![url],
            url_contents: vec![],
        },
        llm_response: None,
        fallback_source_urls: vec![],
        fallback_tool_results: vec![],
        fallback_title: None,
    };
    let result = render_org_node(&msg, std::path::Path::new("/tmp")).unwrap();
    assert!(result.contains("https://example.com/page"));
}

#[test]
fn render_roam_refs_collects_links_from_summary_and_excerpt() {
    let resp = LlmResponse {
        title: "My Title".into(),
        tags: vec![],
        summary: "See https://a.example/path and https://b.example/.".into(),
        excerpt: Some("Quote from https://c.example/info".into()),
        produced_by: "mock".into(),
    };
    let msg = make_processed("raw text", Some(resp));
    let result = render_org_node(&msg, std::path::Path::new("/tmp")).unwrap();
    assert!(result.contains(":ROAM_REFS:"));
    assert!(result.contains("https://a.example/path"));
    assert!(result.contains("https://b.example/"));
    assert!(result.contains("https://c.example/info"));
}

#[test]
fn render_heading_is_immediately_followed_by_properties_drawer() {
    let resp = LlmResponse {
        title: "My Title".into(),
        tags: vec![],
        summary: "A summary.".into(),
        excerpt: None,
        produced_by: "mock".into(),
    };
    let msg = make_processed("raw text", Some(resp));
    let result = render_org_node(&msg, std::path::Path::new("/tmp")).unwrap();
    assert!(
        result.starts_with("* My Title\n:PROPERTIES:\n"),
        "expected heading directly followed by drawer, got:\n{result}"
    );
}

#[test]
fn render_forwarded_from_appears_in_drawer() {
    let msg = IncomingMessage::new(
        MessageSource::Telegram,
        "forwarded content".into(),
        SourceMetadata::Telegram {
            chat_id: 1,
            message_id: 1,
            username: None,
            forwarded_from: Some("@bob".into()),
        },
    );
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
    };
    let result = render_org_node(&processed, std::path::Path::new("/tmp")).unwrap();
    assert!(
        result.contains(":FORWARDED_FROM: @bob"),
        "drawer should contain FORWARDED_FROM: {result}"
    );
}

#[test]
fn render_no_forwarded_property_when_absent() {
    let msg = make_processed("plain", None);
    let result = render_org_node(&msg, std::path::Path::new("/tmp")).unwrap();
    assert!(
        !result.contains("FORWARDED_FROM"),
        "FORWARDED_FROM should not appear when absent: {result}"
    );
}

#[test]
fn render_voice_message_media_kind_in_drawer() {
    use crate::message::Attachment;

    let mut msg = IncomingMessage::new(
        MessageSource::Telegram,
        "voice note".into(),
        SourceMetadata::Telegram {
            chat_id: 1,
            message_id: 2,
            username: None,
            forwarded_from: None,
        },
    );
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("voice.ogg");
    std::fs::write(&path, b"ogg").unwrap();
    msg.attachments.push(Attachment {
        original_name: "voice.ogg".into(),
        saved_path: path,
        mime_type: Some("audio/ogg".into()),
        media_kind: crate::message::MediaKind::VoiceMessage,
    });
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
    };
    let result = render_org_node(&processed, tmp.path()).unwrap();
    assert!(
        result.contains(":MEDIA_KIND: voice_message"),
        "drawer should contain MEDIA_KIND: {result}"
    );
}

#[test]
fn render_no_media_kind_for_documents() {
    use crate::message::Attachment;

    let mut msg = IncomingMessage::new(
        MessageSource::Http,
        "doc".into(),
        SourceMetadata::Http {
            remote_addr: None,
            user_agent: None,
        },
    );
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("file.pdf");
    std::fs::write(&path, b"pdf").unwrap();
    msg.attachments.push(Attachment {
        original_name: "file.pdf".into(),
        saved_path: path,
        mime_type: Some("application/pdf".into()),
        media_kind: crate::message::MediaKind::Document,
    });
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
    };
    let result = render_org_node(&processed, tmp.path()).unwrap();
    assert!(
        !result.contains("MEDIA_KIND"),
        "MEDIA_KIND should not appear for document attachments: {result}"
    );
}

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
        fallback_source_urls: vec![],
        fallback_tool_results: vec![(
            "scrape_page".to_owned(),
            "Tool gathered summary content".to_owned(),
        )],
        fallback_title: None,
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
        fallback_source_urls: vec![
            "https://tool-found.example.com/page1".into(),
            "https://tool-found.example.com/page2".into(),
        ],
        fallback_tool_results: vec![],
        fallback_title: None,
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
        fallback_source_urls: vec![],
        fallback_tool_results: vec![
            ("web_search".to_owned(), "First result content".to_owned()),
            ("scrape_page".to_owned(), "Second result content".to_owned()),
        ],
        fallback_title: None,
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
        fallback_source_urls: vec![],
        fallback_tool_results: vec![],
        fallback_title: Some("Five Word Generated Title".to_owned()),
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
        fallback_source_urls: vec![],
        fallback_tool_results: vec![],
        fallback_title: None,
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
        fallback_source_urls: vec![],
        fallback_tool_results: vec![],
        fallback_title: None,
    };
    let result = render_org_node(&processed, std::path::Path::new("/tmp")).unwrap();
    assert!(
        result.contains("* (untitled)"),
        "should fall back to (untitled) when nothing available: {result}"
    );
}
