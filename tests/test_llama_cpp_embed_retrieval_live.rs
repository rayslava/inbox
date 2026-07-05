//! Opt-in end-to-end retrieval check against a real llama.cpp `--embeddings`
//! server, exercising the whole embeddings path: `MemoryStore` open + probe,
//! KB indexing via `embed_document` (with the nomic `search_document:` prefix),
//! grafeo vector storage, and `kb_recall` via `embed_query`. Also proves
//! cross-lingual retrieval (RU query → EN note) that motivated v2-moe.
//!
//! Skipped unless `LLAMACPP_EMBED_URL` is set to the `OpenAI` base (e.g.
//! `http://127.0.0.1:32002/v1`). Run:
//! `LLAMACPP_EMBED_URL=http://127.0.0.1:32002/v1 cargo test --test \
//! test_llama_cpp_embed_retrieval_live -- --nocapture`

use inbox::config::{EmbeddingApi, MemoryConfig};
use inbox::kb_index;
use inbox::memory::MemoryStore;

async fn open_store(endpoint: String) -> (MemoryStore, tempfile::TempDir) {
    let cfg = MemoryConfig {
        enabled: true,
        embedding_endpoint: Some(endpoint),
        embedding_api: EmbeddingApi::Openai,
        embedding_model: Some("nomic-embed-text-v2-moe".into()),
        embedding_document_prefix: Some("search_document: ".into()),
        embedding_query_prefix: Some("search_query: ".into()),
        ..MemoryConfig::default()
    };
    let dir = tempfile::tempdir().expect("tempdir");
    let store = MemoryStore::open(&cfg, &dir.path().join("m.grafeo"))
        .await
        .expect("open store (is the embed server up?)");
    (store, dir)
}

#[tokio::test]
async fn kb_retrieval_end_to_end_over_real_embeddings() {
    let Ok(endpoint) = std::env::var("LLAMACPP_EMBED_URL") else {
        eprintln!("skipping: set LLAMACPP_EMBED_URL (e.g. http://127.0.0.1:32002/v1)");
        return;
    };
    let (store, _dir) = open_store(endpoint).await;

    kb_index::index_content(
        &store,
        "org",
        "note-paris",
        "/paris.org",
        "* Geography\nThe capital of France is Paris.\n",
    )
    .await
    .expect("index paris");
    kb_index::index_content(
        &store,
        "org",
        "note-borscht",
        "/food.org",
        "* Еда\nРецепт борща с говядиной и свёклой.\n",
    )
    .await
    .expect("index borscht");

    // English query → English note.
    let en = store
        .kb_recall("what is the capital of France", 3)
        .await
        .expect("recall en");
    eprintln!(
        "EN hits: {:?}",
        en.iter().map(|e| &e.key).collect::<Vec<_>>()
    );
    assert!(
        en.first().is_some_and(|e| e.key.contains("note-paris")),
        "EN query should rank the Paris note first"
    );

    // Cross-lingual: Russian query → English note (the v2-moe payoff).
    let ru = store
        .kb_recall("столица Франции", 3)
        .await
        .expect("recall ru");
    eprintln!(
        "RU hits: {:?}",
        ru.iter().map(|e| &e.key).collect::<Vec<_>>()
    );
    assert!(
        ru.iter().any(|e| e.key.contains("note-paris")),
        "RU query should retrieve the EN Paris note (cross-lingual)"
    );
}
