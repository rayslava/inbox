use crate::config::FallbackMode;
use crate::message::{
    Attachment, EnrichedMessage, IncomingMessage, MediaKind, MessageSource, SourceMetadata,
};
use crate::url_content::UrlContent;
use async_trait::async_trait;

use super::{
    LlmChain, LlmClient, LlmCompletion, LlmOutcome, LlmRequest, ToolCall,
    append_missing_source_links,
};
use crate::error::InboxError;

fn make_enriched(text: &str) -> EnrichedMessage {
    let msg = IncomingMessage::new(
        MessageSource::Http,
        text.into(),
        SourceMetadata::Http {
            remote_addr: None,
            user_agent: None,
        },
    );
    EnrichedMessage {
        original: msg,
        urls: vec![],
        url_contents: vec![],
    }
}

fn make_enriched_telegram(text: &str, forwarded_from: Option<String>) -> EnrichedMessage {
    let mut msg = IncomingMessage::new(
        MessageSource::Telegram,
        text.into(),
        SourceMetadata::Telegram {
            chat_id: 1,
            message_id: 1,
            username: None,
            forwarded_from,
        },
    );
    msg.attachments = vec![];
    EnrichedMessage {
        original: msg,
        urls: vec![],
        url_contents: vec![],
    }
}

#[test]
fn llm_request_default_system_prompt() {
    let cfg = crate::test_helpers::no_llm_config();
    let enriched = make_enriched("hello");
    let req = LlmRequest::from_enriched(&enriched, &cfg, std::path::Path::new("/tmp"), "", false);
    assert!(!req.system_prompt.is_empty());
    assert_eq!(req.user_content, "hello");
}

#[test]
fn llm_request_custom_system_prompt() {
    let mut cfg = crate::test_helpers::no_llm_config();
    cfg.prompts.base_system = "custom prompt".into();
    let enriched = make_enriched("text");
    let req = LlmRequest::from_enriched(&enriched, &cfg, std::path::Path::new("/tmp"), "", false);
    assert_eq!(req.system_prompt, "custom prompt");
}

#[test]
fn llm_request_appends_url_contents() {
    let cfg = crate::test_helpers::no_llm_config();
    let mut enriched = make_enriched("base text");
    enriched.url_contents.push(UrlContent {
        url: "http://example.com".into(),
        text: "page content".into(),
        page_title: None,
        headings: vec![],
    });
    let req = LlmRequest::from_enriched(&enriched, &cfg, std::path::Path::new("/tmp"), "", false);
    assert!(req.user_content.contains("page content"));
    assert!(req.user_content.contains("http://example.com"));
}

#[test]
fn llm_request_formats_headings_and_title() {
    let cfg = crate::test_helpers::no_llm_config();
    let mut enriched = make_enriched("note");
    enriched.url_contents.push(UrlContent {
        url: "http://example.com".into(),
        text: "body text".into(),
        page_title: Some("My Page".into()),
        headings: vec!["Intro".into(), "Details".into()],
    });
    let req = LlmRequest::from_enriched(&enriched, &cfg, std::path::Path::new("/tmp"), "", false);
    assert!(req.user_content.contains("Title: My Page"));
    assert!(req.user_content.contains("Headings: Intro | Details"));
    assert!(req.user_content.contains("body text"));
}

#[test]
fn llm_request_appends_tool_prompt_block() {
    let cfg = crate::test_helpers::no_llm_config();
    let enriched = make_enriched("base text");
    let req = LlmRequest::from_enriched(
        &enriched,
        &cfg,
        std::path::Path::new("/tmp"),
        "Tool crawl_url: prefer markdown first",
        false,
    );
    assert!(
        req.system_prompt
            .contains(&cfg.prompts.tool_guidance_header)
    );
    assert!(req.system_prompt.contains("prefer markdown first"));
}

#[tokio::test]
async fn chain_returns_success() {
    let resp = crate::test_helpers::default_llm_response();
    let chain = crate::test_helpers::mock_llm_chain(resp.clone());
    let enriched = make_enriched("test");
    let cfg = crate::test_helpers::no_llm_config();
    let req = LlmRequest::from_enriched(&enriched, &cfg, std::path::Path::new("/tmp"), "", false);
    let outcome = chain.complete(req).await;
    assert!(matches!(outcome, LlmOutcome::Success(_)));
}

#[tokio::test]
async fn chain_raw_fallback_when_no_backends() {
    let chain = LlmChain::new(vec![], FallbackMode::Raw, 5, None, 1, 0, 0);
    let req = LlmRequest::simple("s", "u");
    let outcome = chain.complete(req).await;
    assert!(matches!(outcome, LlmOutcome::RawFallback { .. }));
}

#[tokio::test]
async fn chain_discard_fallback_when_no_backends() {
    let chain = LlmChain::new(vec![], FallbackMode::Discard, 5, None, 1, 0, 0);
    let req = LlmRequest::simple("s", "u");
    let outcome = chain.complete(req).await;
    assert!(matches!(outcome, LlmOutcome::Discard));
}

#[test]
fn max_tool_turns_accessor() {
    let chain = LlmChain::new(vec![], FallbackMode::Raw, 7, None, 1, 0, 0);
    assert_eq!(chain.max_tool_turns(), 7);
}

#[test]
fn append_missing_source_links_adds_sources_block() {
    let resp = crate::message::LlmResponse {
        title: "Title".into(),
        tags: vec![],
        summary: "Summary body".into(),
        excerpt: None,
        produced_by: "mock".into(),
    };
    let out = append_missing_source_links(
        resp,
        &[
            "https://example.com/a".into(),
            "https://example.com/b".into(),
        ],
    );
    assert!(out.summary.contains("Sources:"));
    assert!(out.summary.contains("https://example.com/a"));
    assert!(out.summary.contains("https://example.com/b"));
}

#[test]
fn append_missing_source_links_skips_already_present_links() {
    let resp = crate::message::LlmResponse {
        title: "Title".into(),
        tags: vec![],
        summary: "Summary uses https://example.com/a".into(),
        excerpt: Some("Quote with https://example.com/b".into()),
        produced_by: "mock".into(),
    };
    let out = append_missing_source_links(
        resp,
        &[
            "https://example.com/a".into(),
            "https://example.com/b".into(),
        ],
    );
    assert!(!out.summary.contains("Sources:"));
}

struct ToolCallsLlm;
#[async_trait]
impl LlmClient for ToolCallsLlm {
    fn name(&self) -> &'static str {
        "toolcalls"
    }
    fn model(&self) -> &'static str {
        "test-model"
    }
    fn retries(&self) -> u32 {
        1
    }
    async fn complete(&self, _req: LlmRequest) -> Result<LlmCompletion, InboxError> {
        Ok(LlmCompletion::ToolCalls(vec![ToolCall {
            id: "t1".into(),
            name: "scrape_page".into(),
            arguments: serde_json::json!({"url":"https://example.com"}),
        }]))
    }
}

struct EmptyToolCallsLlm;
#[async_trait]
impl LlmClient for EmptyToolCallsLlm {
    fn name(&self) -> &'static str {
        "empty_toolcalls"
    }
    fn model(&self) -> &'static str {
        "test-model"
    }
    fn retries(&self) -> u32 {
        1
    }
    async fn complete(&self, _req: LlmRequest) -> Result<LlmCompletion, InboxError> {
        Ok(LlmCompletion::ToolCalls(vec![]))
    }
}

#[tokio::test]
async fn chain_tool_calls_without_executor_falls_back() {
    let chain = LlmChain::new(
        vec![Box::new(ToolCallsLlm) as Box<dyn LlmClient>],
        FallbackMode::Raw,
        2,
        None,
        1,
        0,
        0,
    );
    let req = LlmRequest::simple("s", "u");
    let outcome = chain.complete(req).await;
    assert!(matches!(outcome, LlmOutcome::RawFallback { .. }));
}

#[tokio::test]
async fn chain_empty_tool_calls_falls_back() {
    let chain = LlmChain::new(
        vec![Box::new(EmptyToolCallsLlm) as Box<dyn LlmClient>],
        FallbackMode::Raw,
        2,
        None,
        1,
        0,
        0,
    );
    let req = LlmRequest::simple("s", "u");
    let outcome = chain.complete(req).await;
    assert!(matches!(outcome, LlmOutcome::RawFallback { .. }));
}

#[test]
fn llm_request_prepends_forwarded_from_to_content() {
    let cfg = crate::test_helpers::no_llm_config();
    let enriched = make_enriched_telegram("original text", Some("@alice".into()));
    let req = LlmRequest::from_enriched(&enriched, &cfg, std::path::Path::new("/tmp"), "", false);
    assert!(
        req.user_content.starts_with("Forwarded from @alice"),
        "user_content should start with forwarded attribution: {:?}",
        req.user_content
    );
    assert!(
        req.user_content.contains("original text"),
        "user_content should still contain original text"
    );
}

#[test]
fn llm_request_no_forwarded_prefix_for_non_forwarded() {
    let cfg = crate::test_helpers::no_llm_config();
    let enriched = make_enriched_telegram("plain text", None);
    let req = LlmRequest::from_enriched(&enriched, &cfg, std::path::Path::new("/tmp"), "", false);
    assert_eq!(req.user_content, "plain text");
}

#[test]
fn llm_request_collects_images_within_size_limit() {
    use std::io::Write as _;

    let mut cfg = crate::test_helpers::no_llm_config();
    cfg.vision_max_bytes = 1024 * 1024;

    let tmp = tempfile::tempdir().unwrap();
    let img_path = tmp.path().join("photo.jpg");
    // Write a minimal non-empty file to simulate an image.
    let mut f = std::fs::File::create(&img_path).unwrap();
    f.write_all(b"fake-jpeg-bytes").unwrap();

    let mut enriched = make_enriched("test");
    enriched.original.attachments.push(Attachment {
        original_name: "photo.jpg".into(),
        saved_path: img_path,
        mime_type: Some("image/jpeg".into()),
        media_kind: MediaKind::Image,
    });

    let req = LlmRequest::from_enriched(&enriched, &cfg, tmp.path(), "", false);
    assert_eq!(req.images.len(), 1, "one image should be collected");
    assert_eq!(req.images[0].0, "image/jpeg");
    assert!(req.system_prompt.contains("Images are attached"));
}

#[test]
fn llm_request_skips_images_exceeding_size_limit() {
    use std::io::Write as _;

    let mut cfg = crate::test_helpers::no_llm_config();
    cfg.vision_max_bytes = 5; // tiny limit

    let tmp = tempfile::tempdir().unwrap();
    let img_path = tmp.path().join("big.jpg");
    let mut f = std::fs::File::create(&img_path).unwrap();
    f.write_all(b"this-is-more-than-5-bytes").unwrap();

    let mut enriched = make_enriched("test");
    enriched.original.attachments.push(Attachment {
        original_name: "big.jpg".into(),
        saved_path: img_path,
        mime_type: Some("image/jpeg".into()),
        media_kind: MediaKind::Image,
    });

    let req = LlmRequest::from_enriched(&enriched, &cfg, tmp.path(), "", false);
    assert!(req.images.is_empty(), "oversized image should be skipped");
    assert!(
        !req.system_prompt.contains("Images are attached"),
        "vision note should not be added when no images collected"
    );
}

#[test]
fn llm_request_ignores_non_image_attachments_for_vision() {
    use std::io::Write as _;

    let cfg = crate::test_helpers::no_llm_config();
    let tmp = tempfile::tempdir().unwrap();
    let audio_path = tmp.path().join("voice.ogg");
    std::fs::File::create(&audio_path)
        .unwrap()
        .write_all(b"ogg-data")
        .unwrap();

    let mut enriched = make_enriched("test");
    enriched.original.attachments.push(Attachment {
        original_name: "voice.ogg".into(),
        saved_path: audio_path,
        mime_type: Some("audio/ogg".into()),
        media_kind: MediaKind::VoiceMessage,
    });

    let req = LlmRequest::from_enriched(&enriched, &cfg, tmp.path(), "", false);
    assert!(
        req.images.is_empty(),
        "audio attachment should not produce vision images"
    );
}
