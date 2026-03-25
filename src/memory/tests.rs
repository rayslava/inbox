use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use super::{MemoryStore, embed::EmbedClient};

#[tokio::test]
async fn save_and_recall() {
    let store = MemoryStore::new_in_memory().unwrap();
    store.save("greeting", "Hello, world!").await.unwrap();

    let results = store.recall("Hello", 5).await.unwrap();
    assert!(!results.is_empty(), "should find saved memory");
    assert_eq!(results[0].key, "greeting");
    assert_eq!(results[0].value, "Hello, world!");
}

#[tokio::test]
async fn save_overwrites_existing_key() {
    let store = MemoryStore::new_in_memory().unwrap();
    store.save("key", "value one").await.unwrap();
    store.save("key", "value two").await.unwrap();

    let results = store.recall("value", 5).await.unwrap();
    assert_eq!(results.len(), 1, "should have exactly one entry");
    assert_eq!(results[0].value, "value two", "should have updated value");
}

#[tokio::test]
async fn recall_returns_empty_for_unknown_query() {
    let store = MemoryStore::new_in_memory().unwrap();
    let results = store.recall("xyzzy_nonexistent_42", 5).await.unwrap();
    assert!(results.is_empty());
}

#[tokio::test]
async fn recall_fallback_returns_recent_entries() {
    let store = MemoryStore::new_in_memory().unwrap();
    store.save("a", "first entry").await.unwrap();
    store.save("b", "second entry").await.unwrap();

    // Empty query string triggers fallback to recent
    let results = store.recall("", 10).await.unwrap();
    assert_eq!(results.len(), 2);
}

#[tokio::test]
async fn recall_multiple_entries_fts() {
    let store = MemoryStore::new_in_memory().unwrap();
    store
        .save("rust_info", "Rust is a systems programming language")
        .await
        .unwrap();
    store
        .save("python_info", "Python is a scripting language")
        .await
        .unwrap();
    store.save("weather", "It is sunny today").await.unwrap();

    let results = store.recall("programming language", 5).await.unwrap();
    let keys: Vec<&str> = results.iter().map(|e| e.key.as_str()).collect();
    assert!(
        keys.contains(&"rust_info") || keys.contains(&"python_info"),
        "BM25 should find programming language entries, got: {keys:?}"
    );
}

#[tokio::test]
async fn memory_store_open_creates_db() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("test.grafeo");
    let cfg = crate::config::MemoryConfig::default();
    let store = MemoryStore::open(&cfg, &db_path).await;
    assert!(store.is_ok(), "open should succeed: {:?}", store.err());
    assert!(db_path.exists(), "DB file should be created");
}

// ── Graph relationship tests ──────────────────────────────────────────────────

#[tokio::test]
async fn link_memories_creates_relationship() {
    let store = MemoryStore::new_in_memory().unwrap();
    store.save("alice", "Alice is a developer").await.unwrap();
    store
        .save("project", "inbox is Alice's project")
        .await
        .unwrap();

    store
        .link_memories("alice", "project", "works_on")
        .await
        .unwrap();

    let ctx = store.context("alice", 1).await.unwrap();
    let keys: Vec<&str> = ctx.iter().map(|e| e.key.as_str()).collect();
    assert!(keys.contains(&"project"), "should find linked memory");
}

#[tokio::test]
async fn link_source_creates_source_node() {
    let store = MemoryStore::new_in_memory().unwrap();
    store.save("fact", "The sky is blue").await.unwrap();

    store
        .link_source("fact", "telegram", "msg_456", "Chat message")
        .await
        .unwrap();

    let sources = store.sources("fact").await.unwrap();
    assert_eq!(sources.len(), 1);
    assert_eq!(sources[0].kind, "telegram");
    assert_eq!(sources[0].source_id, "msg_456");
    assert_eq!(sources[0].title, "Chat message");
}

#[tokio::test]
async fn sources_returns_multiple_linked_sources() {
    let store = MemoryStore::new_in_memory().unwrap();
    store.save("topic", "Rust programming").await.unwrap();

    store
        .link_source("topic", "telegram", "msg_1", "Chat about Rust")
        .await
        .unwrap();
    store
        .link_source("topic", "email", "email_42", "Rust newsletter")
        .await
        .unwrap();

    let sources = store.sources("topic").await.unwrap();
    assert_eq!(sources.len(), 2, "should find both sources");

    let kinds: Vec<&str> = sources.iter().map(|s| s.kind.as_str()).collect();
    assert!(kinds.contains(&"telegram"), "should have telegram source");
    assert!(kinds.contains(&"email"), "should have email source");
}

#[tokio::test]
async fn sources_returns_empty_for_unlinked_memory() {
    let store = MemoryStore::new_in_memory().unwrap();
    store.save("lonely", "no sources").await.unwrap();

    let sources = store.sources("lonely").await.unwrap();
    assert!(sources.is_empty());
}

#[tokio::test]
async fn sources_returns_empty_for_nonexistent_key() {
    let store = MemoryStore::new_in_memory().unwrap();
    let sources = store.sources("does_not_exist").await.unwrap();
    assert!(sources.is_empty());
}

#[tokio::test]
async fn context_traverses_multiple_hops() {
    let store = MemoryStore::new_in_memory().unwrap();
    store.save("a", "node a").await.unwrap();
    store.save("b", "node b").await.unwrap();
    store.save("c", "node c").await.unwrap();

    store.link_memories("a", "b", "related_to").await.unwrap();
    store.link_memories("b", "c", "related_to").await.unwrap();

    // 1 hop from a should find b
    let ctx1 = store.context("a", 1).await.unwrap();
    let keys1: Vec<&str> = ctx1.iter().map(|e| e.key.as_str()).collect();
    assert!(keys1.contains(&"b"), "1 hop should find b");

    // 2 hops from a should find both b and c
    let ctx2 = store.context("a", 2).await.unwrap();
    let keys2: Vec<&str> = ctx2.iter().map(|e| e.key.as_str()).collect();
    assert!(keys2.contains(&"c"), "2 hops should find c, got: {keys2:?}");
}

// ── Embed client tests ────────────────────────────────────────────────────────

#[tokio::test]
async fn embed_client_returns_vector_on_success() {
    let server = MockServer::start().await;
    let body = serde_json::json!({
        "embeddings": [[0.1f32, 0.2f32, 0.3f32]]
    });
    Mock::given(method("POST"))
        .and(path("/api/embed"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(body),
        )
        .mount(&server)
        .await;

    let client = EmbedClient::new(server.uri(), "test-model".into(), None);
    let vec = client.embed("hello world").await.unwrap();
    assert_eq!(vec.len(), 3);
    assert!((vec[0] - 0.1).abs() < 1e-6);
}

#[tokio::test]
async fn embed_client_returns_error_on_api_failure() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/embed"))
        .respond_with(ResponseTemplate::new(500).set_body_string("Internal Server Error"))
        .mount(&server)
        .await;

    let client = EmbedClient::new(server.uri(), "test-model".into(), None);
    let result = client.embed("hello").await;
    assert!(result.is_err(), "should fail on 500 response");
}

#[tokio::test]
async fn embed_client_returns_error_on_missing_embedding_field() {
    let server = MockServer::start().await;
    let body = serde_json::json!({"not_embeddings": [[]]});
    Mock::given(method("POST"))
        .and(path("/api/embed"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(body),
        )
        .mount(&server)
        .await;

    let client = EmbedClient::new(server.uri(), "test-model".into(), None);
    let result = client.embed("hello").await;
    assert!(
        result.is_err(),
        "should fail when embedding field is missing"
    );
}

#[tokio::test]
async fn embed_client_uses_api_key_when_set() {
    use wiremock::matchers::header;

    let server = MockServer::start().await;
    let body = serde_json::json!({
        "embeddings": [[0.5f32]]
    });
    Mock::given(method("POST"))
        .and(path("/api/embed"))
        .and(header("Authorization", "Bearer test-key"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(body),
        )
        .mount(&server)
        .await;

    let client = EmbedClient::new(server.uri(), "test-model".into(), Some("test-key".into()));
    let result = client.embed("hello").await;
    assert!(result.is_ok(), "should succeed with valid auth header");
}
