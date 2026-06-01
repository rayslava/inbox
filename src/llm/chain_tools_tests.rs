//! Tests for `chain_tools::retry_inner` — specifically the early-abort on a
//! transient "service unavailable" error so the inner retry budget is not burnt
//! on a backend that will fail the same way immediately.

use crate::test_helpers::default_llm_response;

use super::chain_tools::retry_inner;
use super::mock::MockLlm;
use super::{LlmCompletion, LlmRequest};

/// A 429 on the first call aborts the inner loop — the scripted success that
/// follows is never reached, so the 429 error bubbles up to the chain.
#[tokio::test]
async fn aborts_inner_retries_on_service_unavailable() {
    let backend = MockLlm::scripted(vec![
        Err("free_router API error 429 Too Many Requests: quota exhausted".into()),
        Ok(LlmCompletion::Message(default_llm_response())),
    ]);
    let req = LlmRequest::simple("s", "u");
    let err = retry_inner(&backend, &req, 5)
        .await
        .expect_err("transient error must abort, not fall through to the scripted Ok");
    match err {
        crate::error::InboxError::Llm(m) => assert!(m.contains("API error 429"), "{m}"),
        other => panic!("unexpected error variant: {other:?}"),
    }
}

/// A non-transient soft error is retried, so the scripted success is reached.
#[tokio::test]
async fn retries_soft_errors_until_success() {
    let backend = MockLlm::scripted(vec![
        Err("model returned malformed chunk".into()),
        Ok(LlmCompletion::Message(default_llm_response())),
    ]);
    let req = LlmRequest::simple("s", "u");
    let out = retry_inner(&backend, &req, 3)
        .await
        .expect("soft error should be retried and then succeed");
    assert!(matches!(out, LlmCompletion::Message(_)));
}
