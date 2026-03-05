use crate::config::FallbackMode;
use crate::message::{EnrichedMessage, IncomingMessage, MessageSource, SourceMetadata};
use crate::url_content::UrlContent;
use async_trait::async_trait;

use super::{LlmChain, LlmClient, LlmCompletion, LlmOutcome, LlmRequest, ToolCall};
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
    let chain = LlmChain::new(vec![], FallbackMode::Raw, 5, None);
    let req = LlmRequest::simple("s", "u");
    let outcome = chain.complete(req).await;
    assert!(matches!(outcome, LlmOutcome::RawFallback));
}

#[tokio::test]
async fn chain_discard_fallback_when_no_backends() {
    let chain = LlmChain::new(vec![], FallbackMode::Discard, 5, None);
    let req = LlmRequest::simple("s", "u");
    let outcome = chain.complete(req).await;
    assert!(matches!(outcome, LlmOutcome::Discard));
}

#[test]
fn max_tool_turns_accessor() {
    let chain = LlmChain::new(vec![], FallbackMode::Raw, 7, None);
    assert_eq!(chain.max_tool_turns(), 7);
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
    );
    let req = LlmRequest::simple("s", "u");
    let outcome = chain.complete(req).await;
    assert!(matches!(outcome, LlmOutcome::RawFallback));
}
