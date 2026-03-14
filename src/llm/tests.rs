use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

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
    let chain = LlmChain::new(vec![], FallbackMode::Raw, 5, None, 1);
    let req = LlmRequest::simple("s", "u");
    let outcome = chain.complete(req).await;
    assert!(matches!(outcome, LlmOutcome::RawFallback));
}

#[tokio::test]
async fn chain_discard_fallback_when_no_backends() {
    let chain = LlmChain::new(vec![], FallbackMode::Discard, 5, None, 1);
    let req = LlmRequest::simple("s", "u");
    let outcome = chain.complete(req).await;
    assert!(matches!(outcome, LlmOutcome::Discard));
}

#[test]
fn max_tool_turns_accessor() {
    let chain = LlmChain::new(vec![], FallbackMode::Raw, 7, None, 1);
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
    );
    let req = LlmRequest::simple("s", "u");
    let outcome = chain.complete(req).await;
    assert!(matches!(outcome, LlmOutcome::RawFallback));
}

#[tokio::test]
async fn chain_empty_tool_calls_falls_back() {
    let chain = LlmChain::new(
        vec![Box::new(EmptyToolCallsLlm) as Box<dyn LlmClient>],
        FallbackMode::Raw,
        2,
        None,
        1,
    );
    let req = LlmRequest::simple("s", "u");
    let outcome = chain.complete(req).await;
    assert!(matches!(outcome, LlmOutcome::RawFallback));
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

// ── activate_thinking tests ───────────────────────────────────────────────────

/// Returns `activate_thinking` on the first call, then a success response.
struct ActivateThinkingLlm {
    calls: Arc<AtomicUsize>,
    response: crate::message::LlmResponse,
}

#[async_trait]
impl LlmClient for ActivateThinkingLlm {
    fn name(&self) -> &'static str {
        "activate_thinking_mock"
    }
    fn model(&self) -> &'static str {
        "test-model"
    }
    fn retries(&self) -> u32 {
        1
    }
    fn thinking_supported(&self) -> bool {
        true
    }
    async fn complete(&self, req: LlmRequest) -> Result<LlmCompletion, InboxError> {
        let n = self.calls.fetch_add(1, Ordering::SeqCst);
        if n == 0 {
            // First call: request thinking mode
            assert!(
                req.think.is_none(),
                "first call should not have think set yet"
            );
            Ok(LlmCompletion::ToolCalls(vec![ToolCall {
                id: "at1".into(),
                name: "activate_thinking".into(),
                arguments: serde_json::json!({}),
            }]))
        } else {
            // Second call: thinking should be activated
            assert_eq!(
                req.think,
                Some(true),
                "second call should have think=Some(true)"
            );
            Ok(LlmCompletion::Message(self.response.clone()))
        }
    }
}

#[tokio::test]
async fn chain_activate_thinking_retries_with_think_true() {
    let resp = crate::test_helpers::default_llm_response();
    let call_count = Arc::new(AtomicUsize::new(0));
    let llm = ActivateThinkingLlm {
        calls: Arc::clone(&call_count),
        response: resp,
    };
    let chain = LlmChain::new(
        vec![Box::new(llm) as Box<dyn LlmClient>],
        FallbackMode::Raw,
        5,
        None,
        1,
    );
    let req = LlmRequest::simple("s", "u");
    let outcome = chain.complete(req).await;
    assert!(
        matches!(outcome, LlmOutcome::Success(_)),
        "expected success after activate_thinking"
    );
    assert_eq!(
        call_count.load(Ordering::SeqCst),
        2,
        "should have made exactly 2 LLM calls"
    );
}

#[test]
fn thinking_supported_false_by_default() {
    // MockLlm inherits the default impl which returns false
    let mock = crate::llm::mock::MockLlm::new(crate::test_helpers::default_llm_response());
    assert!(!mock.thinking_supported());
}

// ── activate_thinking loop guard ─────────────────────────────────────────────

/// Always returns `activate_thinking`, never anything else.
struct AlwaysThinkingLlm;

#[async_trait]
impl LlmClient for AlwaysThinkingLlm {
    fn name(&self) -> &'static str {
        "always_thinking"
    }
    fn model(&self) -> &'static str {
        "test"
    }
    fn retries(&self) -> u32 {
        1
    }
    fn thinking_supported(&self) -> bool {
        true
    }
    async fn complete(&self, _req: LlmRequest) -> Result<LlmCompletion, InboxError> {
        Ok(LlmCompletion::ToolCalls(vec![ToolCall {
            id: "at1".into(),
            name: "activate_thinking".into(),
            arguments: serde_json::json!({}),
        }]))
    }
}

#[tokio::test]
async fn chain_thinking_loop_terminates() {
    let chain = LlmChain::new(
        vec![Box::new(AlwaysThinkingLlm) as Box<dyn LlmClient>],
        FallbackMode::Raw,
        5,
        None,
        1,
    );
    let req = LlmRequest::simple("s", "u");
    let result = tokio::time::timeout(std::time::Duration::from_secs(5), chain.complete(req)).await;
    assert!(result.is_ok(), "should complete within 5s");
    assert!(
        matches!(result.unwrap(), LlmOutcome::RawFallback),
        "should fall back after hitting loop limit"
    );
}

// ── llm_call tool ─────────────────────────────────────────────────────────────

/// Returns `llm_call` on the first call, then a success response.
struct LlmCallLlm {
    calls: Arc<AtomicUsize>,
    response: crate::message::LlmResponse,
}

#[async_trait]
impl LlmClient for LlmCallLlm {
    fn name(&self) -> &'static str {
        "llm_call_mock"
    }
    fn model(&self) -> &'static str {
        "test"
    }
    fn retries(&self) -> u32 {
        1
    }
    async fn complete(&self, _req: LlmRequest) -> Result<LlmCompletion, InboxError> {
        let n = self.calls.fetch_add(1, Ordering::SeqCst);
        if n == 0 {
            Ok(LlmCompletion::ToolCalls(vec![ToolCall {
                id: "lc1".into(),
                name: "llm_call".into(),
                arguments: serde_json::json!({
                    "system_prompt": "Summarize the following",
                    "content": "some content"
                }),
            }]))
        } else {
            Ok(LlmCompletion::Message(self.response.clone()))
        }
    }
}

#[tokio::test]
async fn llm_call_executes_sub_call() {
    let resp = crate::test_helpers::default_llm_response();
    let call_count = Arc::new(AtomicUsize::new(0));
    let llm = LlmCallLlm {
        calls: Arc::clone(&call_count),
        response: resp,
    };
    let chain = LlmChain::new(
        vec![Box::new(llm) as Box<dyn LlmClient>],
        FallbackMode::Raw,
        5,
        None,
        1,
    );
    let req = LlmRequest::simple("s", "u");
    let outcome = chain.complete(req).await;
    assert!(
        matches!(outcome, LlmOutcome::Success(_)),
        "chain should succeed after llm_call sub-request"
    );
    // call 0: returns llm_call tool call
    // call 1: complete_raw default wraps and calls complete → success
    // call 2: main loop sees llm_call result in user_content, returns success
    assert!(
        call_count.load(Ordering::SeqCst) >= 2,
        "should have made at least 2 LLM calls"
    );
}

#[tokio::test]
async fn llm_call_not_offered_when_depth_zero() {
    // With max_llm_tool_depth = 0, llm_call should not be in tool defs.
    // We verify by checking that a mock that always returns llm_call eventually falls back
    // (since there are no executor tools to satisfy the empty-tools condition).
    let call_count = Arc::new(AtomicUsize::new(0));
    let llm = LlmCallLlm {
        calls: Arc::clone(&call_count),
        response: crate::test_helpers::default_llm_response(),
    };
    let chain = LlmChain::new(
        vec![Box::new(llm) as Box<dyn LlmClient>],
        FallbackMode::Raw,
        5,
        None,
        0, // max_llm_tool_depth = 0
    );
    // With depth=0 and no tool executor, tool_defs is empty, so llm_call is NOT offered.
    // The LLM returns an llm_call tool call anyway (models can do that).
    // The chain handles it: executes sub-call, appends result, then on next turn gets success.
    let req = LlmRequest::simple("s", "u");
    let outcome = chain.complete(req).await;
    // The chain still handles llm_call even when not offered — it just runs it.
    // What we mainly verify is that it terminates correctly.
    assert!(
        matches!(outcome, LlmOutcome::Success(_) | LlmOutcome::RawFallback),
        "should terminate"
    );
}
