//! Task 3: per-backend vision routing — `decide()` matrix and chain-level
//! skip / strip / run behavior for image-bearing requests.

use crate::llm::mock::MockLlm;
use crate::llm::{FallbackMode, LlmOutcome, LlmRequest};
use crate::message::LlmResponse;

use std::borrow::Cow;

use super::LlmChain;
use super::vision::{VisionDecision, decide, prepare};

fn resp() -> LlmResponse {
    LlmResponse {
        title: "T".into(),
        tags: vec![],
        summary: "S".into(),
        excerpt: None,
        produced_by: "mock".into(),
    }
}

fn image_request(has_text: bool) -> LlmRequest {
    let mut req = LlmRequest::simple("sys", "user");
    req.images.push(("image/png".into(), "Zm9v".into()));
    req.has_image_text = has_text;
    req
}

fn chain(backends: Vec<Box<dyn crate::llm::LlmClient>>) -> LlmChain {
    LlmChain::new(backends, FallbackMode::Raw, 5, None, 1, 0, 0)
}

#[test]
fn prepare_run_borrows_for_vision_backend() {
    let backend = MockLlm::new(resp()).with_vision();
    match prepare(&image_request(false), &backend) {
        Some(Cow::Borrowed(_)) => {}
        other => panic!("expected borrowed Run, got {:?}", other.is_some()),
    }
}

#[test]
fn prepare_strips_images_for_text_backend_with_text() {
    let backend = MockLlm::new(resp()); // non-vision
    match prepare(&image_request(true), &backend) {
        Some(Cow::Owned(r)) => assert!(r.images.is_empty(), "images stripped"),
        _ => panic!("expected owned, image-stripped request"),
    }
}

#[test]
fn prepare_skips_text_backend_without_image_text() {
    let backend = MockLlm::new(resp()); // non-vision
    assert!(prepare(&image_request(false), &backend).is_none());
}

#[tokio::test]
async fn complete_vision_text_no_backend_for_empty_images() {
    use crate::llm::VisionUnavailable;
    let chain = chain(vec![Box::new(MockLlm::new(resp()).with_vision())]);
    assert_eq!(
        chain.complete_vision_text("s", "u", vec![]).await,
        Err(VisionUnavailable::NoVisionBackend)
    );
}

#[tokio::test]
async fn complete_vision_text_all_unavailable_when_backend_errors() {
    use crate::llm::VisionUnavailable;
    let chain = chain(vec![Box::new(MockLlm::failing("boom").with_vision())]);
    let out = chain
        .complete_vision_text("s", "u", vec![("image/png".into(), "Zm9v".into())])
        .await;
    assert_eq!(
        out,
        Err(VisionUnavailable::AllUnavailable),
        "an eligible vision backend that errors yields AllUnavailable"
    );
}

#[test]
fn decide_non_image_request_always_runs() {
    let backend = MockLlm::new(resp()); // non-vision
    let req = LlmRequest::simple("s", "u");
    assert_eq!(decide(&req, &backend), VisionDecision::Run);
}

#[test]
fn decide_vision_backend_runs_image_request() {
    let backend = MockLlm::new(resp()).with_vision();
    assert_eq!(decide(&image_request(false), &backend), VisionDecision::Run);
}

#[test]
fn decide_text_backend_strips_when_image_text_present() {
    let backend = MockLlm::new(resp());
    assert_eq!(
        decide(&image_request(true), &backend),
        VisionDecision::StripImages
    );
}

#[test]
fn decide_text_backend_skips_without_image_text() {
    let backend = MockLlm::new(resp());
    assert_eq!(
        decide(&image_request(false), &backend),
        VisionDecision::Skip
    );
}

#[tokio::test]
async fn chain_skips_text_only_backend_for_image_without_text() {
    let chain = chain(vec![Box::new(MockLlm::new(resp()))]); // non-vision
    let outcome = chain.complete(image_request(false)).await;
    // Backend skipped ⇒ no success, falls through to raw fallback.
    assert!(matches!(outcome, LlmOutcome::RawFallback { .. }));
}

#[tokio::test]
async fn chain_runs_text_backend_when_image_text_present() {
    let chain = chain(vec![Box::new(MockLlm::new(resp()))]); // non-vision
    let outcome = chain.complete(image_request(true)).await;
    assert!(matches!(outcome, LlmOutcome::Success { .. }));
}

#[tokio::test]
async fn chain_runs_vision_backend_for_image_request() {
    let chain = chain(vec![Box::new(MockLlm::new(resp()).with_vision())]);
    let outcome = chain.complete(image_request(false)).await;
    assert!(matches!(outcome, LlmOutcome::Success { .. }));
}

#[tokio::test]
async fn chain_prefers_vision_backend_over_skipped_text_backend() {
    // Text-only backend first (would be skipped), vision backend second.
    let chain = chain(vec![
        Box::new(MockLlm::new(resp())),
        Box::new(MockLlm::new(resp()).with_vision()),
    ]);
    let outcome = chain.complete(image_request(false)).await;
    assert!(matches!(outcome, LlmOutcome::Success { .. }));
}

#[tokio::test]
async fn chain_image_all_skipped_defers_even_under_discard() {
    // No backend can serve the image; Discard policy must NOT drop it — the LLM
    // never ran, so the request is deferred to a raw fallback instead.
    let chain = LlmChain::new(
        vec![Box::new(MockLlm::new(resp()))], // non-vision
        FallbackMode::Discard,
        5,
        None,
        1,
        0,
        0,
    );
    let outcome = chain.complete(image_request(false)).await;
    assert!(matches!(outcome, LlmOutcome::RawFallback { .. }));
}

#[tokio::test]
async fn chain_non_image_still_discards_under_discard_policy() {
    // The image guard must not change behavior for non-image requests: a failing
    // backend under Discard still yields Discard.
    let chain = LlmChain::new(
        vec![Box::new(MockLlm::failing("boom"))],
        FallbackMode::Discard,
        5,
        None,
        1,
        0,
        0,
    );
    let outcome = chain.complete(LlmRequest::simple("s", "u")).await;
    assert!(matches!(outcome, LlmOutcome::Discard));
}
