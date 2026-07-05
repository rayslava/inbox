use wiremock::matchers::{body_string_contains, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use super::{MemoryStore, embed::EmbedClient, resolve_embed_client};
use crate::config::EmbeddingApi;

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

    let client = EmbedClient::new(
        server.uri(),
        EmbeddingApi::Ollama,
        "test-model".into(),
        None,
    )
    .expect("build embed client");
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
    let ok_client = EmbedClient::new(
        ok_server.uri(),
        EmbeddingApi::Ollama,
        "test-model".into(),
        None,
    )
    .expect("build embed client");
    let ok_provider: &dyn EmbeddingProvider = &ok_client;
    assert_eq!(ok_provider.embed("hi").await.expect("ok path").len(), 2);

    // Error path: the adapter's InboxError::Memory maps into CoreError::Memory.
    let err_server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/embed"))
        .respond_with(ResponseTemplate::new(500))
        .mount(&err_server)
        .await;
    let err_client = EmbedClient::new(
        err_server.uri(),
        EmbeddingApi::Ollama,
        "test-model".into(),
        None,
    )
    .expect("build embed client");
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

    let client = EmbedClient::new(
        server.uri(),
        EmbeddingApi::Ollama,
        "test-model".into(),
        None,
    )
    .expect("build embed client");
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

    let client = EmbedClient::new(
        server.uri(),
        EmbeddingApi::Ollama,
        "test-model".into(),
        None,
    )
    .expect("build embed client");
    let result = client.embed("hello").await;
    assert!(
        result.is_err(),
        "should fail when embedding field is missing"
    );
}

#[tokio::test]
async fn embed_client_openai_api_parses_data_embedding() {
    let server = MockServer::start().await;
    // OpenAI/llama.cpp shape: {"data": [{"embedding": [...]}]} at POST /embeddings.
    let body = serde_json::json!({
        "data": [{"embedding": [0.7f32, 0.8f32, 0.9f32]}]
    });
    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(body),
        )
        .mount(&server)
        .await;

    let client = EmbedClient::new(
        server.uri(),
        EmbeddingApi::Openai,
        "test-model".into(),
        None,
    )
    .expect("build embed client");
    let vec = client.embed("hello world").await.unwrap();
    assert_eq!(vec.len(), 3);
    assert!((vec[2] - 0.9).abs() < 1e-6);
}

#[tokio::test]
async fn embed_client_rejects_malformed_vector_element() {
    // A `null` element must fail the whole parse, not be silently dropped
    // (a truncated vector would corrupt dimension detection / the index).
    for (api, route, body) in [
        (
            EmbeddingApi::Ollama,
            "/api/embed",
            serde_json::json!({ "embeddings": [[0.1f32, serde_json::Value::Null, 0.3f32]] }),
        ),
        (
            EmbeddingApi::Openai,
            "/embeddings",
            serde_json::json!({ "data": [{ "embedding": [0.1f32, serde_json::Value::Null, 0.3f32] }] }),
        ),
    ] {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path(route))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "application/json")
                    .set_body_json(body),
            )
            .mount(&server)
            .await;

        let client = EmbedClient::new(server.uri(), api, "test-model".into(), None)
            .expect("build embed client");
        assert!(
            client.embed("hello").await.is_err(),
            "{api:?}: malformed element must error, not truncate"
        );
    }
}

#[tokio::test]
async fn embed_applies_task_prefixes() {
    let server = MockServer::start().await;
    let ok = |v: f64| {
        ResponseTemplate::new(200)
            .insert_header("content-type", "application/json")
            .set_body_json(serde_json::json!({ "embeddings": [[v]] }))
    };
    // Distinct mocks keyed on the (prefixed) input actually sent.
    Mock::given(method("POST"))
        .and(path("/api/embed"))
        .and(body_string_contains("search_document: hello"))
        .respond_with(ok(0.1))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/embed"))
        .and(body_string_contains("search_query: hello"))
        .respond_with(ok(0.2))
        .mount(&server)
        .await;

    let client = EmbedClient::new(server.uri(), EmbeddingApi::Ollama, "m".into(), None)
        .expect("build")
        .with_prefixes(
            Some("search_document: ".into()),
            Some("search_query: ".into()),
        );

    // Each routes to the mock matching its prefix; a missing prefix → no match → err.
    assert!(
        client.embed_document("hello").await.is_ok(),
        "document prefix not applied"
    );
    assert!(
        client.embed_query("hello").await.is_ok(),
        "query prefix not applied"
    );
}

#[tokio::test]
async fn embed_prefixes_apply_through_trait_object() {
    use inbox_core::EmbeddingProvider;
    let server = MockServer::start().await;
    for frag in ["search_document: hi", "search_query: hi"] {
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .and(body_string_contains(frag))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "application/json")
                    .set_body_json(serde_json::json!({ "embeddings": [[0.4f32]] })),
            )
            .mount(&server)
            .await;
    }
    let client = EmbedClient::new(server.uri(), EmbeddingApi::Ollama, "m".into(), None)
        .expect("build")
        .with_prefixes(
            Some("search_document: ".into()),
            Some("search_query: ".into()),
        );
    let provider: &dyn EmbeddingProvider = &client;
    assert!(provider.embed_document("hi").await.is_ok());
    assert!(provider.embed_query("hi").await.is_ok());
}

#[tokio::test]
async fn embed_without_prefixes_sends_verbatim() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/embed"))
        .and(body_string_contains("\"input\":\"hello\""))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(serde_json::json!({ "embeddings": [[0.3f32]] })),
        )
        .mount(&server)
        .await;

    let client =
        EmbedClient::new(server.uri(), EmbeddingApi::Ollama, "m".into(), None).expect("build");
    // No prefixes configured → document/query embed the raw text.
    assert!(client.embed_document("hello").await.is_ok());
    assert!(client.embed_query("hello").await.is_ok());
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

    let client = EmbedClient::new(
        server.uri(),
        EmbeddingApi::Ollama,
        "test-model".into(),
        Some("test-key".into()),
    )
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

const TEST_FP: &str = "m|3|cosine|none|v1|";

fn build_db_with_fp(key: &str, value: &str, embedding: &[f32], fp: &str) -> grafeo::GrafeoDB {
    let db = grafeo::GrafeoDB::new_in_memory();
    db.create_text_index("Memory", "value").ok();
    super::queries::create_indexes(&db, embedding.len());
    super::queries::upsert_memory(&db, key, value, Some(embedding), fp)
        .expect("upsert with embedding");
    db
}

fn build_db_with_embedded_memory(key: &str, value: &str, embedding: &[f32]) -> grafeo::GrafeoDB {
    build_db_with_fp(key, value, embedding, TEST_FP)
}

#[test]
fn recall_entries_returns_match_for_query_with_text_and_vector() {
    let db = build_db_with_embedded_memory("rust", "Rust is a systems language", &[0.1, 0.2, 0.3]);

    let results = super::queries::recall_entries(&db, "Rust", Some(&[0.1, 0.2, 0.3]), 5, TEST_FP)
        .expect("recall ok");

    assert!(!results.is_empty(), "should return the matching entry");
    assert_eq!(results[0].key, "rust");
}

#[test]
fn recall_entries_text_survives_active_vector_hit() {
    // Mixed store: one active-fingerprint memory the query vector matches, plus a
    // stale-fingerprint memory that only matches by text. The text match must not
    // be hidden by the vector hit (codex regression guard).
    let db = build_db_with_fp("active", "quantum entanglement", &[1.0, 0.0, 0.0], TEST_FP);
    super::queries::upsert_memory(
        &db,
        "stale",
        "vintage typewriter",
        Some(&[0.0, 1.0, 0.0]),
        "OLD",
    )
    .expect("insert stale");

    let hits =
        super::queries::recall_entries(&db, "typewriter", Some(&[1.0, 0.0, 0.0]), 5, TEST_FP)
            .expect("recall ok");
    let keys: Vec<&str> = hits.iter().map(|e| e.key.as_str()).collect();
    assert!(keys.contains(&"active"), "active vector match present");
    assert!(
        keys.contains(&"stale"),
        "stale-fingerprint text match must survive alongside the vector hit"
    );
}

#[test]
fn recall_entries_vector_only_path_when_query_text_empty() {
    // Empty query text skips both BM25 and the hybrid path's text component;
    // the raw cosine_similarity fallback fires.
    let db = build_db_with_embedded_memory("topic", "irrelevant text", &[1.0, 0.0, 0.0]);

    let results = super::queries::recall_entries(&db, "", Some(&[1.0, 0.0, 0.0]), 5, TEST_FP)
        .expect("recall ok");

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

    let results = super::queries::recall_entries(&db, "", Some(&[0.0, 1.0, 0.0]), 5, TEST_FP)
        .expect("recall ok");

    // fallback_recent returns all Memory nodes regardless of similarity.
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].key, "topic");
}

#[test]
fn recall_entries_vector_path_isolates_stale_fingerprint() {
    // A memory embedded under a different fingerprint must not match on the
    // vector (cosine) path even with an identical vector — only the active space.
    let db = build_db_with_fp("stale", "irrelevant text", &[1.0, 0.0, 0.0], "OLD-FP");

    // Vector path under a different active fingerprint: no vector hit → empty query
    // falls back to recent (score 0), not a vector match.
    let vec_hits =
        super::queries::recall_entries(&db, "", Some(&[1.0, 0.0, 0.0]), 5, "NEW-FP").expect("ok");
    assert!(
        vec_hits.iter().all(|e| e.score == 0.0),
        "stale-fingerprint vector must not produce a similarity match"
    );

    // But text recall is space-agnostic — the memory is still findable by content.
    let text_hits =
        super::queries::recall_entries(&db, "irrelevant", None, 5, "NEW-FP").expect("ok");
    assert!(
        text_hits.iter().any(|e| e.key == "stale"),
        "text recall must still find a stale-fingerprint memory"
    );
}

#[test]
fn upsert_memory_update_path_refreshes_embedding_and_fingerprint() {
    let db = grafeo::GrafeoDB::new_in_memory();
    db.create_text_index("Memory", "value").ok();
    super::queries::create_indexes(&db, 3);
    // Insert, then update the same key with a new embedding + fingerprint
    // (exercises the SET-with-embedding branch).
    super::queries::upsert_memory(&db, "k", "first", Some(&[1.0, 0.0, 0.0]), "FP1").unwrap();
    super::queries::upsert_memory(&db, "k", "second", Some(&[0.0, 1.0, 0.0]), "FP2").unwrap();

    // The vector path under the new fingerprint matches the refreshed vector.
    let now =
        super::queries::recall_entries(&db, "", Some(&[0.0, 1.0, 0.0]), 5, "FP2").expect("ok");
    assert!(
        now.iter()
            .any(|e| e.key == "k" && e.value == "second" && e.score > 0.5),
        "update must refresh both value and vector under the new fingerprint"
    );
    // The old fingerprint no longer matches on the vector path.
    let old =
        super::queries::recall_entries(&db, "", Some(&[0.0, 1.0, 0.0]), 5, "FP1").expect("ok");
    assert!(
        old.iter().all(|e| e.score == 0.0),
        "old fingerprint must not vector-match the refreshed embedding"
    );
}

#[test]
fn recall_entries_reserves_slot_for_text_when_vectors_saturate() {
    // Vector hits saturate `limit`, but a text-only match on a stale-fingerprint
    // memory must still win a reserved slot (codex saturation guard).
    let db = grafeo::GrafeoDB::new_in_memory();
    db.create_text_index("Memory", "value").ok();
    super::queries::create_indexes(&db, 3);
    for (k, v) in [
        ("a1", "alpha one"),
        ("a2", "alpha two"),
        ("a3", "alpha three"),
    ] {
        super::queries::upsert_memory(&db, k, v, Some(&[1.0, 0.0, 0.0]), TEST_FP).unwrap();
    }
    super::queries::upsert_memory(
        &db,
        "stale",
        "unique borscht recipe",
        Some(&[0.0, 1.0, 0.0]),
        "OLD",
    )
    .unwrap();

    let hits = super::queries::recall_entries(&db, "borscht", Some(&[1.0, 0.0, 0.0]), 3, TEST_FP)
        .expect("recall ok");
    let keys: Vec<&str> = hits.iter().map(|e| e.key.as_str()).collect();
    assert_eq!(hits.len(), 3);
    assert!(
        keys.contains(&"stale"),
        "a text match must reserve a slot despite saturated vector hits, got {keys:?}"
    );
}

#[test]
fn recall_entries_falls_back_to_recent_when_nothing_matches() {
    let db = grafeo::GrafeoDB::new_in_memory();
    db.create_text_index("Memory", "value").ok();
    super::queries::upsert_memory(&db, "k1", "first", None, TEST_FP).unwrap();
    super::queries::upsert_memory(&db, "k2", "second", None, TEST_FP).unwrap();

    let results = super::queries::recall_entries(&db, "", None, 10, TEST_FP).expect("recall ok");
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
async fn kb_save_and_recall_with_embeddings() {
    use crate::memory::kb;

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/embed"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(serde_json::json!({ "embeddings": [[0.1f32, 0.2, 0.3]] })),
        )
        .mount(&server)
        .await;

    let dir = tempfile::tempdir().expect("tempdir");
    let cfg = crate::config::MemoryConfig {
        embedding_endpoint: Some(server.uri()),
        embedding_dims: Some(3),
        ..Default::default()
    };
    let store = MemoryStore::open(&cfg, &dir.path().join("m.grafeo"))
        .await
        .expect("open store");

    let id = kb::kb_id("org", "n1", "v1", "h1");
    store
        .kb_save(&id, "quantum computing basics", "org", "n1", "/n1.org")
        .await
        .expect("kb_save (insert with embedding)");

    let hits = store
        .kb_recall("quantum", 5)
        .await
        .expect("kb_recall (hybrid)");
    assert!(hits.iter().any(|e| e.key.starts_with("kb:")));

    // Same id again → update path (SET value + embedding).
    store
        .kb_save(&id, "quantum computing revised", "org", "n1", "/n1.org")
        .await
        .expect("kb_save (update)");
}

#[tokio::test]
async fn kb_chunks_do_not_pollute_memory_recall() {
    use crate::memory::kb;

    let store = MemoryStore::new_in_memory().expect("in-memory store");

    store
        .save("fox-memory", "the quick brown fox")
        .await
        .expect("save m1");
    store
        .save("dog-memory", "a dog barks loudly")
        .await
        .expect("save m2");

    let before: Vec<String> = store
        .recall("fox", 10)
        .await
        .expect("recall")
        .into_iter()
        .map(|e| e.value)
        .collect();
    assert!(before.iter().any(|v| v.contains("quick brown fox")));

    // Bulk-insert KB chunks whose text overlaps the query.
    for i in 0..5 {
        store
            .kb_save(
                &kb::kb_id("org", &format!("note{i}"), "v1", &format!("h{i}")),
                "the quick brown fox appears in this document too",
                "org",
                &format!("note{i}"),
                &format!("/note{i}.org"),
            )
            .await
            .expect("kb_save");
    }

    // Behavioral recall is byte-for-byte unchanged — KB volume did not leak in.
    let after: Vec<String> = store
        .recall("fox", 10)
        .await
        .expect("recall2")
        .into_iter()
        .map(|e| e.value)
        .collect();
    assert_eq!(before, after);
    assert!(after.iter().all(|v| !v.contains("document too")));

    // KB-only recall returns chunks (namespaced ids), never memories.
    let kb_hits = store.kb_recall("fox", 10).await.expect("kb_recall");
    assert!(!kb_hits.is_empty());
    assert!(kb_hits.iter().all(|e| e.key.starts_with("kb:")));
    assert!(kb_hits.iter().any(|e| e.value.contains("document too")));

    // Behavioral recall never returns a KB chunk id.
    let mem_hits = store.recall("fox", 10).await.expect("recall3");
    assert!(mem_hits.iter().all(|e| !e.key.starts_with("kb:")));
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
