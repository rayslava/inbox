//! End-to-end tests for `inbox_core::brain::answer` wired to the real
//! grafeo-backed `MemoryStore` (as `&dyn VectorStore`) and a mock `LlmChain`
//! (as `&dyn LlmBackend`).

use crate::kb_index;
use crate::memory::MemoryStore;
use crate::message::LlmResponse;
use inbox_core::VectorStore;
use inbox_core::brain::RetrievalMode;

fn mock_response(summary: &str) -> LlmResponse {
    LlmResponse {
        title: String::new(),
        tags: vec![],
        summary: summary.to_owned(),
        excerpt: None,
        produced_by: "mock".to_owned(),
    }
}

#[tokio::test]
async fn brain_answers_from_kb_with_citations() {
    let store = MemoryStore::new_in_memory().expect("store");
    kb_index::index_content(
        &store,
        "org",
        "capital-note",
        "/cap.org",
        "* Geography\nThe capital of France is Paris.\n",
    )
    .await
    .expect("index");

    let chain = crate::test_helpers::mock_llm_chain(mock_response("Paris."));

    let answer = inbox_core::brain::answer(
        &store,
        chain.as_ref(),
        "what is the capital of France",
        5,
        RetrievalMode::KbOnly,
    )
    .await
    .expect("answer");

    assert!(answer.text.contains("Paris"));
    assert!(answer.note_ids.contains(&"capital-note".to_owned()));
    assert!(answer.to_org().contains("[[id:capital-note]]"));
}

#[tokio::test]
async fn brain_handles_empty_kb() {
    let store = MemoryStore::new_in_memory().expect("store");
    let chain = crate::test_helpers::mock_llm_chain(mock_response("unused"));

    let answer = inbox_core::brain::answer(
        &store,
        chain.as_ref(),
        "anything at all",
        5,
        RetrievalMode::KbOnly,
    )
    .await
    .expect("answer");

    assert!(answer.note_ids.is_empty());
    assert!(answer.text.contains("couldn't find"));
}

#[tokio::test]
async fn memory_only_answers_from_behavioral_memory_without_citations() {
    let store = MemoryStore::new_in_memory().expect("store");
    // KB chunk that must NOT surface under MemoryOnly.
    kb_index::index_content(&store, "org", "kb-note", "/k.org", "* T\nthe sky is blue\n")
        .await
        .expect("index");
    // Behavioral memory the query should draw from.
    VectorStore::save(&store, "fact:sky", "the sky is blue")
        .await
        .expect("save");

    let chain = crate::test_helpers::mock_llm_chain(mock_response("Blue."));
    let answer = inbox_core::brain::answer(
        &store,
        chain.as_ref(),
        "what color is the sky",
        5,
        RetrievalMode::MemoryOnly,
    )
    .await
    .expect("answer");

    assert!(answer.text.contains("Blue"));
    // Behavioral memory keys are not org notes → no citations.
    assert!(answer.note_ids.is_empty());
}

#[tokio::test]
async fn hybrid_caps_kb_and_still_cites_notes() {
    let store = MemoryStore::new_in_memory().expect("store");
    kb_index::index_content(
        &store,
        "org",
        "capital-note",
        "/cap.org",
        "* Geography\nThe capital of France is Paris.\n",
    )
    .await
    .expect("index");
    VectorStore::save(&store, "fact:france", "France is in Europe")
        .await
        .expect("save");

    let chain = crate::test_helpers::mock_llm_chain(mock_response("Paris."));
    let answer = inbox_core::brain::answer(
        &store,
        chain.as_ref(),
        "capital of France",
        6,
        RetrievalMode::Hybrid { kb_quota: 4 },
    )
    .await
    .expect("answer");

    assert!(answer.text.contains("Paris"));
    // KB chunk still cited; behavioral memory contributes context but no citation.
    assert!(answer.note_ids.contains(&"capital-note".to_owned()));
}
