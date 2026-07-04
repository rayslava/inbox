//! Live end-to-end check of the Phase 1 `/ask` surface over a **real TCP
//! socket**: a real grafeo-backed store, real heading-split indexing, real
//! per-kind retrieval, and deterministic citations — served by the same
//! `ask_router` `main.rs` mounts. Only the LLM is stubbed (no external API,
//! per the repo's no-real-calls rule); everything else is the production path.

use std::sync::Arc;

use inbox::kb_index;
use inbox::memory::MemoryStore;
use inbox::message::LlmResponse;
use inbox::test_helpers::mock_llm_chain;
use inbox::web::ask::{AskState, ask_router};

/// Bind `ask_router` on an ephemeral loopback port with `store` and an LLM
/// stubbed to echo `answer`; return the full `/ask` URL.
async fn spawn_ask(store: Arc<MemoryStore>, answer: &str) -> String {
    let llm = mock_llm_chain(LlmResponse {
        title: String::new(),
        tags: vec![],
        summary: answer.to_owned(),
        excerpt: None,
        produced_by: "stub".to_owned(),
    });
    let router = ask_router(AskState {
        memory_store: Some(store),
        llm,
    });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let addr = listener.local_addr().expect("addr");
    tokio::spawn(async move {
        let _ = axum::serve(listener, router).await;
    });
    format!("http://{addr}/ask")
}

/// Index one org note and save one behavioral memory into a fresh store.
async fn seed_store() -> Arc<MemoryStore> {
    let store = Arc::new(MemoryStore::new_in_memory().expect("store"));
    kb_index::index_content(
        &store,
        "org",
        "capital-note",
        "/cap.org",
        "* Geography\nThe capital of France is Paris.\n",
    )
    .await
    .expect("index");
    store
        .save("fact:france", "France is a country in Europe")
        .await
        .expect("save");
    store
}

#[tokio::test]
async fn ask_kb_mode_answers_over_the_wire_with_citation() {
    let url = spawn_ask(seed_store().await, "Paris.").await;
    let resp = reqwest::Client::new()
        .post(&url)
        .json(&serde_json::json!({"question": "capital of France", "mode": "kb"}))
        .send()
        .await
        .expect("request");
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.expect("json");
    assert!(body["answer"].as_str().unwrap().contains("Paris"));
    // Deterministic citation parsed from the chunk id, not invented by the LLM.
    assert_eq!(body["note_ids"][0], "capital-note");
    assert!(
        body["org"]
            .as_str()
            .unwrap()
            .contains("[[id:capital-note]]")
    );
}

#[tokio::test]
async fn ask_hybrid_mode_answers_over_the_wire() {
    let url = spawn_ask(seed_store().await, "Paris.").await;
    let resp = reqwest::Client::new()
        .post(&url)
        .json(&serde_json::json!({"question": "France", "mode": "hybrid", "top_k": 6}))
        .send()
        .await
        .expect("request");
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.expect("json");
    // KB chunk is still cited; behavioral memory blends in as context.
    assert_eq!(body["note_ids"][0], "capital-note");
}

#[tokio::test]
async fn ask_unknown_mode_is_bad_request_over_the_wire() {
    let url = spawn_ask(seed_store().await, "unused").await;
    let resp = reqwest::Client::new()
        .post(&url)
        .json(&serde_json::json!({"question": "hi", "mode": "bogus"}))
        .send()
        .await
        .expect("request");
    assert_eq!(resp.status(), 400);
}
