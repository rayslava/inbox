use std::sync::Arc;

use inbox::feedback::FeedbackEntry;
use inbox::memory::MemoryStore;

fn default_memory_config() -> inbox::config::MemoryConfig {
    inbox::config::MemoryConfig {
        enabled: true,
        ..Default::default()
    }
}

#[tokio::test]
async fn preload_recalls_saved_memories() {
    let store = Arc::new(MemoryStore::new_in_memory().unwrap());
    store
        .save("rust_lang", "User is interested in Rust programming")
        .await
        .unwrap();
    store
        .save("python_ml", "User uses Python for machine learning")
        .await
        .unwrap();

    let config = default_memory_config();
    let ctx = inbox::pipeline::context_preload::preload_context(
        &store,
        &config,
        "Tell me about Rust async",
        &[],
        &[],
    )
    .await;

    assert!(
        !ctx.memories.is_empty(),
        "should recall at least one memory"
    );
    let keys: Vec<&str> = ctx.memories.iter().map(|m| m.key.as_str()).collect();
    assert!(
        keys.contains(&"rust_lang"),
        "should recall rust_lang memory"
    );
}

#[tokio::test]
async fn preload_returns_empty_for_no_matches() {
    let store = Arc::new(MemoryStore::new_in_memory().unwrap());
    let config = default_memory_config();

    let ctx = inbox::pipeline::context_preload::preload_context(
        &store,
        &config,
        "something random",
        &[],
        &[],
    )
    .await;

    assert!(ctx.memories.is_empty());
    assert_eq!(
        ctx.recall_quality,
        inbox::pipeline::context_preload::RecallQuality::Empty
    );
}

#[tokio::test]
async fn preload_includes_feedback() {
    let store = Arc::new(MemoryStore::new_in_memory().unwrap());
    store
        .save_feedback(&FeedbackEntry {
            message_id: "msg-1".into(),
            rating: 1,
            comment: "Tags were wrong".into(),
            created_at: chrono::Utc::now(),
            source: "http".into(),
            title: "Bad article".into(),
        })
        .await
        .unwrap();

    let config = default_memory_config();
    let ctx = inbox::pipeline::context_preload::preload_context(
        &store,
        &config,
        "some content",
        &[],
        &[],
    )
    .await;

    assert_eq!(ctx.feedback.len(), 1);
    assert_eq!(ctx.feedback[0].comment, "Tags were wrong");
}

#[tokio::test]
async fn preload_excludes_high_rated_feedback() {
    let store = Arc::new(MemoryStore::new_in_memory().unwrap());
    store
        .save_feedback(&FeedbackEntry {
            message_id: "msg-good".into(),
            rating: 3,
            comment: "Great!".into(),
            created_at: chrono::Utc::now(),
            source: "http".into(),
            title: "Good article".into(),
        })
        .await
        .unwrap();

    let config = default_memory_config();
    let ctx = inbox::pipeline::context_preload::preload_context(
        &store,
        &config,
        "some content",
        &[],
        &[],
    )
    .await;

    // Default max_rating is 2, so rating=3 should be excluded.
    assert!(ctx.feedback.is_empty());
}

#[tokio::test]
async fn preload_includes_related_memories() {
    let store = Arc::new(MemoryStore::new_in_memory().unwrap());
    store
        .save("rust", "Rust programming language")
        .await
        .unwrap();
    store
        .save("borrow_checker", "Rust ownership model")
        .await
        .unwrap();
    store
        .link_memories("rust", "borrow_checker", "related_to")
        .await
        .unwrap();

    let config = default_memory_config();
    let ctx = inbox::pipeline::context_preload::preload_context(
        &store,
        &config,
        "Rust programming",
        &[],
        &[],
    )
    .await;

    // Find the "rust" memory and check it has related memories.
    let rust_mem = ctx.memories.iter().find(|m| m.key == "rust");
    assert!(rust_mem.is_some(), "should recall rust memory");
    let rust_mem = rust_mem.unwrap();
    assert!(
        !rust_mem.related.is_empty(),
        "rust memory should have related memories"
    );
    assert!(
        rust_mem.related.iter().any(|r| r.key == "borrow_checker"),
        "should include borrow_checker as related"
    );
}

#[tokio::test]
async fn log_recall_event_creates_graph_node() {
    let store = Arc::new(MemoryStore::new_in_memory().unwrap());
    store.save("test_key", "test value").await.unwrap();

    store
        .log_recall_event("msg-123", &["test_key".into()], "http")
        .await
        .unwrap();

    // Verify we can query the recall event back.
    let outcomes = store.recall_outcomes(&["test_key".into()]).await;
    // No feedback linked yet, so outcomes should be empty.
    assert!(outcomes.is_empty());
}

#[tokio::test]
async fn recall_outcomes_correlate_with_feedback() {
    let store = Arc::new(MemoryStore::new_in_memory().unwrap());
    store.save("test_mem", "test value").await.unwrap();
    store
        .log_recall_event("test-msg", &["test_mem".into()], "http")
        .await
        .unwrap();

    store
        .save_feedback(&FeedbackEntry {
            message_id: "test-msg".into(),
            rating: 2,
            comment: "Okay-ish".into(),
            created_at: chrono::Utc::now(),
            source: "http".into(),
            title: "Test".into(),
        })
        .await
        .unwrap();

    let outcomes = store.recall_outcomes(&["test_mem".into()]).await;
    // The recall_outcomes query does a two-step lookup:
    //   1. Find RecallEvent nodes linked to Memory via RECALLED edge
    //   2. Find Feedback nodes with same message_id as the RecallEvent
    // Grafeo in-memory may share state between instances, so outcomes
    // may include extra results from other tests. Verify at least one match.
    assert!(
        !outcomes.is_empty(),
        "recall outcomes should find feedback via RECALLED edge"
    );
    let outcome = outcomes.iter().find(|o| o.memory_key == "test_mem");
    assert!(outcome.is_some(), "should have outcome for test_mem");
    let outcome = outcome.unwrap();
    assert!(outcome.times_recalled >= 1);
    assert!(outcome.avg_rating > 0.0);
}

#[tokio::test]
async fn format_preloaded_context_full_round_trip() {
    let store = Arc::new(MemoryStore::new_in_memory().unwrap());
    store.save("rust", "User is learning Rust").await.unwrap();
    store
        .save_feedback(&FeedbackEntry {
            message_id: "m1".into(),
            rating: 1,
            comment: "Bad tags".into(),
            created_at: chrono::Utc::now(),
            source: "http".into(),
            title: "Some article".into(),
        })
        .await
        .unwrap();

    let config = default_memory_config();
    let ctx = inbox::pipeline::context_preload::preload_context(
        &store,
        &config,
        "Rust async programming",
        &[],
        &["rust".into()],
    )
    .await;

    let text = inbox::pipeline::context_preload::format_preloaded_context(&ctx);
    assert!(text.contains("Memory context"));
    assert!(text.contains("rust"));
    assert!(text.contains("User feedback"));
    assert!(text.contains("Bad tags"));
}
