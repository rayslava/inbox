use super::*;
use crate::message::{
    EnrichedMessage, IncomingMessage, LlmResponse, MessageSource, ProcessedMessage, SourceMetadata,
};

pub(super) fn make_processed(text: &str, llm_response: Option<LlmResponse>) -> ProcessedMessage {
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
        incomplete: crate::message::ProcessingCompleteness::Complete,
        fallback_source_urls: vec![],
        fallback_tool_results: vec![],
        fallback_title: None,
        enrichment: crate::message::EnrichmentMetadata::default(),
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

fn make_with_attachment(media_kind: crate::message::MediaKind) -> ProcessedMessage {
    use crate::message::Attachment;
    let mut msg = make_processed("", None);
    msg.enriched.original.attachments.push(Attachment {
        original_name: "file".into(),
        saved_path: std::path::PathBuf::from("/tmp/file"),
        mime_type: None,
        media_kind,
    });
    msg
}

#[test]
fn fallback_title_uses_attachment_kind_when_text_empty() {
    use crate::message::MediaKind;
    let cases = [
        (MediaKind::Image, "Image"),
        (MediaKind::Audio, "Audio"),
        (MediaKind::Video, "Video"),
        (MediaKind::VoiceMessage, "Voice Message"),
        (MediaKind::Sticker, "Sticker"),
        (MediaKind::Animation, "Animation"),
    ];
    for (kind, expected_title) in cases {
        let msg = make_with_attachment(kind);
        let out = render_org_node(&msg, std::path::Path::new("/tmp")).unwrap();
        assert!(
            out.contains(&format!("* {expected_title}")),
            "expected `* {expected_title}` for kind {kind:?}, got:\n{out}"
        );
    }
}

#[test]
fn fallback_title_document_attachment_is_untitled() {
    use crate::message::MediaKind;
    // Document/Other map to None in the fallback chain → "(untitled)".
    let msg = make_with_attachment(MediaKind::Document);
    let out = render_org_node(&msg, std::path::Path::new("/tmp")).unwrap();
    assert!(out.contains("(untitled)"), "got:\n{out}");
}

#[test]
fn fallback_title_uses_explicit_override() {
    let mut msg = make_processed("", None);
    msg.fallback_title = Some("Explicit override".into());
    let out = render_org_node(&msg, std::path::Path::new("/tmp")).unwrap();
    assert!(out.contains("* Explicit override"), "got:\n{out}");
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
        enrichment_helpers: &[],
        memories_recalled: 0,
        urls_fetched: 0,
        tool_calls_made: 0,
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
        incomplete: crate::message::ProcessingCompleteness::Complete,
        fallback_source_urls: vec![],
        fallback_tool_results: vec![],
        fallback_title: None,
        enrichment: crate::message::EnrichmentMetadata::default(),
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
        incomplete: crate::message::ProcessingCompleteness::Complete,
        fallback_source_urls: vec![],
        fallback_tool_results: vec![],
        fallback_title: None,
        enrichment: crate::message::EnrichmentMetadata::default(),
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
        incomplete: crate::message::ProcessingCompleteness::Complete,
        fallback_source_urls: vec![],
        fallback_tool_results: vec![],
        fallback_title: None,
        enrichment: crate::message::EnrichmentMetadata::default(),
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
        incomplete: crate::message::ProcessingCompleteness::Complete,
        fallback_source_urls: vec![],
        fallback_tool_results: vec![],
        fallback_title: None,
        enrichment: crate::message::EnrichmentMetadata::default(),
    };
    let result = render_org_node(&processed, tmp.path()).unwrap();
    assert!(
        !result.contains("MEDIA_KIND"),
        "MEDIA_KIND should not appear for document attachments: {result}"
    );
}
