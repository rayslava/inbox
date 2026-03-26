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

// ── Feedback tests ───────────────────────────────────────────────────────────

#[tokio::test]
async fn save_and_query_feedback() {
    use chrono::Utc;

    let store = MemoryStore::new_in_memory().unwrap();
    let entry = crate::feedback::FeedbackEntry {
        message_id: "00000000-0000-0000-0000-000000000001".into(),
        rating: 3,
        comment: "great summary".into(),
        created_at: Utc::now(),
        source: "web_ui".into(),
        title: "Test Article".into(),
    };

    store.save_feedback(&entry).await.unwrap();

    let loaded = store
        .query_feedback("00000000-0000-0000-0000-000000000001")
        .await
        .unwrap();
    let loaded = loaded.expect("should find feedback");
    assert_eq!(loaded.rating, 3);
    assert_eq!(loaded.comment, "great summary");
    assert_eq!(loaded.source, "web_ui");
    assert_eq!(loaded.title, "Test Article");
}

#[tokio::test]
async fn feedback_upsert_updates_existing() {
    use chrono::Utc;

    let store = MemoryStore::new_in_memory().unwrap();
    let mid = "00000000-0000-0000-0000-000000000002";

    let entry1 = crate::feedback::FeedbackEntry {
        message_id: mid.into(),
        rating: 1,
        comment: String::new(),
        created_at: Utc::now(),
        source: "telegram".into(),
        title: "Bad".into(),
    };
    store.save_feedback(&entry1).await.unwrap();

    let entry2 = crate::feedback::FeedbackEntry {
        message_id: mid.into(),
        rating: 3,
        comment: "actually good".into(),
        created_at: Utc::now(),
        source: "telegram".into(),
        title: "Bad".into(),
    };
    store.save_feedback(&entry2).await.unwrap();

    let loaded = store.query_feedback(mid).await.unwrap().unwrap();
    assert_eq!(loaded.rating, 3);
    assert_eq!(loaded.comment, "actually good");
}

#[tokio::test]
async fn query_feedback_returns_none_for_unknown() {
    let store = MemoryStore::new_in_memory().unwrap();
    let result = store.query_feedback("nonexistent").await.unwrap();
    assert!(result.is_none());
}

#[tokio::test]
async fn feedback_stats_empty() {
    let store = MemoryStore::new_in_memory().unwrap();
    let stats = store.feedback_stats().await.unwrap();
    assert_eq!(stats.total, 0);
    assert_eq!(stats.by_rating, [0, 0, 0]);
    assert!((stats.average - 0.0).abs() < f64::EPSILON);
}

#[tokio::test]
async fn feedback_stats_with_entries() {
    use chrono::Utc;

    let store = MemoryStore::new_in_memory().unwrap();
    for (i, rating) in [1u8, 2, 3, 3].iter().enumerate() {
        let entry = crate::feedback::FeedbackEntry {
            message_id: format!("msg-{i}"),
            rating: *rating,
            comment: String::new(),
            created_at: Utc::now(),
            source: "test".into(),
            title: format!("title {i}"),
        };
        store.save_feedback(&entry).await.unwrap();
    }

    let stats = store.feedback_stats().await.unwrap();
    assert_eq!(stats.total, 4);
    assert_eq!(stats.by_rating, [1, 1, 2]);
    let expected_avg = (1.0 + 2.0 + 3.0 + 3.0) / 4.0;
    assert!((stats.average - expected_avg).abs() < 1e-6);
}

#[tokio::test]
async fn update_feedback_comment() {
    use chrono::Utc;

    let store = MemoryStore::new_in_memory().unwrap();
    let mid = "msg-comment";

    let entry = crate::feedback::FeedbackEntry {
        message_id: mid.into(),
        rating: 2,
        comment: String::new(),
        created_at: Utc::now(),
        source: "telegram".into(),
        title: "Test".into(),
    };
    store.save_feedback(&entry).await.unwrap();

    let updated = store
        .update_feedback_comment(mid, "needs better tags")
        .await
        .unwrap();
    assert!(updated);

    let loaded = store.query_feedback(mid).await.unwrap().unwrap();
    assert_eq!(loaded.comment, "needs better tags");
}

#[tokio::test]
async fn update_feedback_comment_returns_false_for_missing() {
    let store = MemoryStore::new_in_memory().unwrap();
    let updated = store
        .update_feedback_comment("nonexistent", "hello")
        .await
        .unwrap();
    assert!(!updated);
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
