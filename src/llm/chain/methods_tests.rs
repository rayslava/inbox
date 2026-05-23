//! Tests for the auxiliary `LlmChain` methods (`complete_text`,
//! `execute_llm_tool_call`).

use crate::config::FallbackMode;
use crate::llm::mock::MockLlm;
use crate::llm::{LlmClient, LlmRequest, ToolCall};
use crate::message::LlmResponse;
use crate::test_helpers::default_llm_response;

use super::LlmChain;

fn chain(backends: Vec<Box<dyn LlmClient>>) -> LlmChain {
    LlmChain::new(backends, FallbackMode::Raw, 2, None, 1, 0, 0)
}

fn response_with_summary(summary: &str, produced_by: &str) -> LlmResponse {
    LlmResponse {
        summary: summary.into(),
        produced_by: produced_by.into(),
        ..default_llm_response()
    }
}

fn llm_call(system: &str, content: &str) -> ToolCall {
    ToolCall {
        id: "t1".into(),
        name: "llm_call".into(),
        arguments: serde_json::json!({ "system_prompt": system, "content": content }),
    }
}

#[tokio::test]
async fn complete_text_returns_first_non_empty_summary() {
    let backend = MockLlm::new(response_with_summary("the answer", "mock-a"));
    let chain = chain(vec![Box::new(backend)]);

    let out = chain.complete_text("sys", "user").await;
    assert_eq!(out, Some(("the answer".to_owned(), "mock-a".to_owned())));
}

#[tokio::test]
async fn complete_text_skips_empty_then_uses_next() {
    // First backend yields an empty summary (skipped); second provides text.
    let empty = MockLlm::new(response_with_summary("   ", "mock-empty"));
    let good = MockLlm::new(response_with_summary("real text", "mock-good"));
    let chain = chain(vec![Box::new(empty), Box::new(good)]);

    let out = chain.complete_text("sys", "user").await;
    assert_eq!(out, Some(("real text".to_owned(), "mock-good".to_owned())));
}

#[tokio::test]
async fn complete_text_returns_none_when_all_fail() {
    let chain = chain(vec![
        Box::new(MockLlm::failing("boom")),
        Box::new(MockLlm::failing("boom2")),
    ]);

    assert!(chain.complete_text("sys", "user").await.is_none());
}

#[tokio::test]
async fn execute_llm_tool_call_returns_sub_result() {
    let backend = MockLlm::new(response_with_summary("sub answer", "mock-sub"));
    let chain = chain(vec![Box::new(backend)]);

    let parent = LlmRequest::simple("parent-sys", "parent-user");
    let (text, produced_by) = chain
        .execute_llm_tool_call(&llm_call("be helpful", "summarize X"), &parent)
        .await;

    assert_eq!(text, "sub answer");
    assert_eq!(produced_by, "mock-sub");
}

#[tokio::test]
async fn execute_llm_tool_call_reports_exhaustion() {
    let chain = chain(vec![Box::new(MockLlm::failing("down"))]);

    let parent = LlmRequest::simple("parent-sys", "parent-user");
    let (text, produced_by) = chain
        .execute_llm_tool_call(&llm_call("sys", "content"), &parent)
        .await;

    assert_eq!(text, "llm_call failed: all backends exhausted");
    assert!(produced_by.is_empty());
}
