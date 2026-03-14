use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use super::{MemoryStore, embed::EmbedClient, search::cosine};

#[test]
fn cosine_identical_vectors() {
    let v = vec![1.0f32, 0.0, 0.0];
    assert!((cosine(&v, &v) - 1.0).abs() < 1e-6);
}

#[test]
fn cosine_orthogonal_vectors() {
    let a = vec![1.0f32, 0.0];
    let b = vec![0.0f32, 1.0];
    assert!(cosine(&a, &b).abs() < 1e-6);
}

#[test]
fn cosine_zero_vector_returns_zero() {
    let a = vec![0.0f32, 0.0];
    let b = vec![1.0f32, 2.0];
    assert_eq!(cosine(&a, &b), 0.0);
}

#[tokio::test]
async fn save_and_recall() {
    let store = MemoryStore::new_in_memory().unwrap();
    store.save("greeting", "Hello, world!").await.unwrap();

    let results = store.recall("greeting", 5).await.unwrap();
    assert!(!results.is_empty(), "should find saved memory");
    assert_eq!(results[0].key, "greeting");
    assert_eq!(results[0].value, "Hello, world!");
}

#[tokio::test]
async fn save_overwrites_existing_key() {
    let store = MemoryStore::new_in_memory().unwrap();
    store.save("key", "value one").await.unwrap();
    store.save("key", "value two").await.unwrap();

    let results = store.recall("key", 5).await.unwrap();
    assert_eq!(results.len(), 1, "should have exactly one entry");
    assert_eq!(results[0].value, "value two", "should have updated value");
}

#[tokio::test]
async fn recall_returns_empty_for_unknown_query() {
    let store = MemoryStore::new_in_memory().unwrap();
    // FTS5 returns empty for queries that don't match anything
    let results = store.recall("xyzzy_nonexistent_42", 5).await.unwrap();
    // Either empty (no match) or falls back to latest (no entries at all)
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
        "FTS should find programming language entries, got: {keys:?}"
    );
}

#[tokio::test]
async fn memory_store_open_creates_db_file() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let cfg = crate::config::MemoryConfig::default();
    let store = MemoryStore::open(&cfg, &db_path).await;
    assert!(store.is_ok(), "open should succeed: {:?}", store.err());
    assert!(db_path.exists(), "DB file should be created");
}

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
    // Response missing the `embeddings[0]` field
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
    assert!(result.is_err(), "should fail when embedding field is missing");
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
