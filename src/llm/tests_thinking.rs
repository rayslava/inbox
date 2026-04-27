use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;

use crate::config::FallbackMode;
use crate::error::InboxError;

use super::{LlmChain, LlmClient, LlmCompletion, LlmOutcome, LlmRequest, ToolCall};

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
        0,
        0,
    );
    let req = LlmRequest::simple("s", "u");
    let outcome = chain.complete(req).await;
    assert!(
        matches!(outcome, LlmOutcome::Success { .. }),
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
    let mock = crate::llm::mock::MockLlm::new(crate::test_helpers::default_llm_response());
    assert!(!mock.thinking_supported());
}

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
        0,
        0,
    );
    let req = LlmRequest::simple("s", "u");
    let result = tokio::time::timeout(std::time::Duration::from_secs(5), chain.complete(req)).await;
    assert!(result.is_ok(), "should complete within 5s");
    assert!(
        matches!(result.unwrap(), LlmOutcome::RawFallback { .. }),
        "should fall back after hitting loop limit"
    );
}

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
        0,
        0,
    );
    let req = LlmRequest::simple("s", "u");
    let outcome = chain.complete(req).await;
    assert!(
        matches!(outcome, LlmOutcome::Success { .. }),
        "chain should succeed after llm_call sub-request"
    );
    assert!(
        call_count.load(Ordering::SeqCst) >= 2,
        "should have made at least 2 LLM calls"
    );
}

#[tokio::test]
async fn llm_call_not_offered_when_depth_zero() {
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
        0,
        0,
        0,
    );
    let req = LlmRequest::simple("s", "u");
    let outcome = chain.complete(req).await;
    assert!(
        matches!(
            outcome,
            LlmOutcome::Success { .. } | LlmOutcome::RawFallback { .. }
        ),
        "should terminate"
    );
}
