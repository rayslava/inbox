//! Task 3: chain-level service-unavailable cooldown — a transient backend is
//! abandoned mid-run and skipped (without an attempt) on subsequent runs while
//! its cooldown is open.

use crate::llm::mock::MockLlm;
use crate::llm::{FallbackMode, LlmClient, LlmCompletion, LlmOutcome, LlmRequest};
use crate::message::LlmResponse;

use super::LlmChain;

fn resp(summary: &str, produced_by: &str) -> LlmResponse {
    LlmResponse {
        title: "T".into(),
        tags: vec![],
        summary: summary.into(),
        excerpt: None,
        produced_by: produced_by.into(),
    }
}

fn chain(backends: Vec<Box<dyn crate::llm::LlmClient>>) -> LlmChain {
    LlmChain::new(backends, FallbackMode::Raw, 5, None, 1, 0, 0)
}

fn summary_of(outcome: LlmOutcome) -> String {
    match outcome {
        LlmOutcome::Success { response, .. } => response.summary,
        LlmOutcome::RawFallback { .. } => panic!("expected Success, got RawFallback"),
        LlmOutcome::Discard => panic!("expected Success, got Discard"),
    }
}

/// A backend already in cooldown is skipped without an attempt; the chain falls
/// through to the next backend. Backend-1 is scripted to succeed, so if it were
/// attempted its summary would win — proving the skip by the summary that wins.
#[tokio::test]
async fn cooling_backend_is_skipped_without_attempt() {
    let b1 = MockLlm::new(resp("from-b1", "b1")).with_cooldown(300);
    b1.mark_unavailable();
    let b2 = MockLlm::new(resp("from-b2", "b2"));

    let chain = chain(vec![Box::new(b1), Box::new(b2)]);
    let out = chain.complete(LlmRequest::simple("s", "u")).await;
    assert_eq!(summary_of(out), "from-b2");
}

/// A 429 on the first run trips backend-1's cooldown; it is then skipped on the
/// second run even though its script's next entry is a success.
#[tokio::test]
async fn transient_error_trips_cooldown_and_skips_next_run() {
    let b1 = MockLlm::scripted(vec![
        Err("free_router API error 429 Too Many Requests: quota".into()),
        Ok(LlmCompletion::Message(resp("from-b1", "b1"))),
    ])
    .with_cooldown(300);
    let b2 = MockLlm::new(resp("from-b2", "b2"));

    let chain = chain(vec![Box::new(b1), Box::new(b2)]);

    // Run 1: b1 hits 429 → tripped → falls through to b2.
    let first = chain.complete(LlmRequest::simple("s", "u")).await;
    assert_eq!(summary_of(first), "from-b2");

    // Run 2: b1 is in cooldown → skipped (its scripted success is never reached).
    let second = chain.complete(LlmRequest::simple("s", "u")).await;
    assert_eq!(summary_of(second), "from-b2");
}
