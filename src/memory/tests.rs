use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use super::{MemoryStore, embed::EmbedClient, resolve_embed_client};

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

    let client =
        EmbedClient::new(server.uri(), "test-model".into(), None).expect("build embed client");
    let vec = client.embed("hello world").await.unwrap();
    assert_eq!(vec.len(), 3);
    assert!((vec[0] - 0.1).abs() < 1e-6);
}

#[tokio::test]
async fn embedding_provider_trait_path_maps_success_and_error() {
    use inbox_core::{CoreError, EmbeddingProvider};

    // Success path through the core trait object (not the inherent method).
    let ok_server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/embed"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(serde_json::json!({ "embeddings": [[0.5f32, 0.6f32]] })),
        )
        .mount(&ok_server)
        .await;
    let ok_client =
        EmbedClient::new(ok_server.uri(), "test-model".into(), None).expect("build embed client");
    let ok_provider: &dyn EmbeddingProvider = &ok_client;
    assert_eq!(ok_provider.embed("hi").await.expect("ok path").len(), 2);

    // Error path: the adapter's InboxError::Memory maps into CoreError::Memory.
    let err_server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/embed"))
        .respond_with(ResponseTemplate::new(500))
        .mount(&err_server)
        .await;
    let err_client =
        EmbedClient::new(err_server.uri(), "test-model".into(), None).expect("build embed client");
    let err_provider: &dyn EmbeddingProvider = &err_client;
    assert!(matches!(
        err_provider.embed("hi").await,
        Err(CoreError::Memory(_))
    ));
}

#[tokio::test]
async fn embed_client_returns_error_on_api_failure() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/embed"))
        .respond_with(ResponseTemplate::new(500).set_body_string("Internal Server Error"))
        .mount(&server)
        .await;

    let client =
        EmbedClient::new(server.uri(), "test-model".into(), None).expect("build embed client");
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

    let client =
        EmbedClient::new(server.uri(), "test-model".into(), None).expect("build embed client");
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

    let client = EmbedClient::new(server.uri(), "test-model".into(), Some("test-key".into()))
        .expect("build embed client");
    let result = client.embed("hello").await;
    assert!(result.is_ok(), "should succeed with valid auth header");
}

// ── resolve_embed_client tests ───────────────────────────────────────────────

#[tokio::test]
async fn resolve_embed_client_returns_none_when_endpoint_missing() {
    let cfg = crate::config::MemoryConfig::default();
    assert!(resolve_embed_client(&cfg).await.is_none());
}

#[tokio::test]
async fn resolve_embed_client_skips_probe_when_dims_set() {
    // No mock server is mounted: success here proves the probe was not issued.
    let cfg = crate::config::MemoryConfig {
        embedding_endpoint: Some("http://127.0.0.1:1/unreachable".into()),
        embedding_dims: Some(384),
        ..Default::default()
    };

    assert!(resolve_embed_client(&cfg).await.is_some());
}

#[tokio::test]
async fn resolve_embed_client_returns_some_on_probe_success() {
    let server = MockServer::start().await;
    let body = serde_json::json!({"embeddings": [[0.1f32, 0.2f32]]});
    Mock::given(method("POST"))
        .and(path("/api/embed"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(body),
        )
        .mount(&server)
        .await;

    let cfg = crate::config::MemoryConfig {
        embedding_endpoint: Some(server.uri()),
        ..Default::default()
    };

    assert!(resolve_embed_client(&cfg).await.is_some());
}

// ── recall_entries vector-path tests ─────────────────────────────────────────

fn build_db_with_embedded_memory(key: &str, value: &str, embedding: &[f32]) -> grafeo::GrafeoDB {
    let db = grafeo::GrafeoDB::new_in_memory();
    db.create_text_index("Memory", "value").ok();
    super::queries::create_indexes(&db, embedding.len());
    super::queries::upsert_memory(&db, key, value, Some(embedding)).expect("upsert with embedding");
    db
}

#[test]
fn recall_entries_hybrid_search_returns_match_when_query_text_matches() {
    let db = build_db_with_embedded_memory("rust", "Rust is a systems language", &[0.1, 0.2, 0.3]);

    let results =
        super::queries::recall_entries(&db, "Rust", Some(&[0.1, 0.2, 0.3]), 5).expect("recall ok");

    assert!(!results.is_empty(), "hybrid search should return entry");
    assert_eq!(results[0].key, "rust");
}

#[test]
fn recall_entries_vector_only_path_when_query_text_empty() {
    // Empty query text skips both BM25 and the hybrid path's text component;
    // the raw cosine_similarity branch on lines 172-198 fires.
    let db = build_db_with_embedded_memory("topic", "irrelevant text", &[1.0, 0.0, 0.0]);

    let results =
        super::queries::recall_entries(&db, "", Some(&[1.0, 0.0, 0.0]), 5).expect("recall ok");

    assert!(
        !results.is_empty(),
        "vector-only path should match identical embedding"
    );
    assert_eq!(results[0].key, "topic");
}

#[test]
fn recall_entries_vector_path_filters_by_similarity_threshold() {
    // Orthogonal vectors → cosine 0 < 0.5 threshold → vector path returns no hits,
    // and with empty query string we hit fallback_recent which still returns the entry.
    let db = build_db_with_embedded_memory("topic", "hello", &[1.0, 0.0, 0.0]);

    let results =
        super::queries::recall_entries(&db, "", Some(&[0.0, 1.0, 0.0]), 5).expect("recall ok");

    // fallback_recent returns all Memory nodes regardless of similarity.
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].key, "topic");
}

#[test]
fn recall_entries_falls_back_to_recent_when_nothing_matches() {
    let db = grafeo::GrafeoDB::new_in_memory();
    db.create_text_index("Memory", "value").ok();
    super::queries::upsert_memory(&db, "k1", "first", None).unwrap();
    super::queries::upsert_memory(&db, "k2", "second", None).unwrap();

    let results = super::queries::recall_entries(&db, "", None, 10).expect("recall ok");
    assert_eq!(results.len(), 2);
}

#[tokio::test]
async fn resolve_embed_client_returns_none_on_probe_failure() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/embed"))
        .respond_with(ResponseTemplate::new(500))
        .mount(&server)
        .await;

    let cfg = crate::config::MemoryConfig {
        embedding_endpoint: Some(server.uri()),
        ..Default::default()
    };

    assert!(resolve_embed_client(&cfg).await.is_none());
}

#[tokio::test]
async fn vector_store_trait_path_exercises_all_methods() {
    use inbox_core::VectorStore;

    let store = MemoryStore::new_in_memory().expect("in-memory store");
    let vs: &dyn VectorStore = &store;

    vs.save("k1", "the quick brown fox").await.expect("save k1");
    vs.save("k2", "a memory about dogs").await.expect("save k2");

    let hits = vs.recall("quick fox", 5).await.expect("recall");
    assert!(hits.iter().any(|e| e.key == "k1"));

    vs.link_source("k1", "note", "id1", "Title")
        .await
        .expect("link_source");
    let srcs = vs.sources("k1").await.expect("sources");
    assert!(srcs.iter().any(|s| s.source_id == "id1"));

    vs.link_memories("k1", "k2", "related")
        .await
        .expect("link_memories");
    // Graph traversal result may vary; exercising the path is enough here.
    let _ = vs.context("fox", 1).await.expect("context");
}
