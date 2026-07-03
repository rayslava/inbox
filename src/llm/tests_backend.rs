use inbox_core::LlmBackend;

use crate::message::LlmResponse;

#[tokio::test]
async fn llm_backend_complete_text_returns_summary_and_producer() {
    let resp = LlmResponse {
        title: "T".into(),
        tags: vec![],
        summary: "the answer".into(),
        excerpt: None,
        produced_by: "mock".into(),
    };
    let chain = crate::test_helpers::mock_llm_chain(resp);
    let backend: &dyn LlmBackend = chain.as_ref();
    let (text, by) = backend
        .complete_text("system", "user")
        .await
        .expect("complete_text succeeds with a mock backend");
    assert_eq!(text, "the answer");
    assert_eq!(by, "mock");
}

#[tokio::test]
async fn llm_backend_complete_text_maps_no_answer_to_error() {
    use crate::config::FallbackMode;
    use crate::llm::LlmChain;

    // No backends → inherent complete_text returns None → CoreError::Llm.
    let chain = LlmChain::new(vec![], FallbackMode::Raw, 5, None, 1, 0, 0);
    let backend: &dyn LlmBackend = &chain;
    assert!(backend.complete_text("s", "u").await.is_err());
}
