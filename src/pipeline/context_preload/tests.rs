use crate::feedback::FeedbackEntry;
use crate::memory::{RecallOutcome, RelatedMemory};

use super::{PreloadedContext, RecallQuality, RecalledMemory, format_preloaded_context};

#[test]
fn format_empty_context_shows_no_memories_guidance() {
    let ctx = PreloadedContext::default();
    let text = format_preloaded_context(&ctx);
    assert!(text.contains("no relevant memories found"));
    assert!(text.contains("memory_save"));
}

#[test]
fn format_memories_only() {
    let ctx = PreloadedContext {
        memories: vec![RecalledMemory {
            key: "rust".into(),
            value: "User likes Rust".into(),
            score: 0.85,
            related: vec![RelatedMemory {
                key: "systems".into(),
                value: "Systems programming".into(),
                relation: "RELATED_TO".into(),
                direction: "outgoing".into(),
            }],
            outcome: None,
        }],
        feedback: Vec::new(),
        recall_quality: RecallQuality::Strong,
    };
    let text = format_preloaded_context(&ctx);
    assert!(text.contains("[0.85] rust: User likes Rust"));
    assert!(text.contains("RELATED_TO"));
    assert!(text.contains("systems: Systems programming"));
    assert!(text.contains("memory_save"));
    assert!(!text.contains("feedback"));
}

#[test]
fn format_with_outcomes() {
    let ctx = PreloadedContext {
        memories: vec![RecalledMemory {
            key: "borrow".into(),
            value: "Borrow checker details".into(),
            score: 0.62,
            related: Vec::new(),
            outcome: Some(RecallOutcome {
                memory_key: "borrow".into(),
                times_recalled: 2,
                avg_rating: 1.5,
                sample_comments: vec!["Too surface-level".into()],
            }),
        }],
        feedback: Vec::new(),
        recall_quality: RecallQuality::Weak,
    };
    let text = format_preloaded_context(&ctx);
    assert!(text.contains("used 2 times"));
    assert!(text.contains("avg rating 1.5"));
    assert!(text.contains("⚠"));
    assert!(text.contains("Too surface-level"));
}

#[test]
fn format_feedback_only() {
    let ctx = PreloadedContext {
        memories: Vec::new(),
        feedback: vec![FeedbackEntry {
            message_id: "abc".into(),
            rating: 1,
            comment: "Tags were wrong".into(),
            created_at: chrono::Utc::now(),
            source: "telegram".into(),
            title: "Bad article".into(),
        }],
        recall_quality: RecallQuality::Empty,
    };
    let text = format_preloaded_context(&ctx);
    assert!(text.contains("Bad article"));
    assert!(text.contains("rated 1/3"));
    assert!(text.contains("Tags were wrong"));
    assert!(text.contains("Avoid patterns"));
}

#[test]
fn format_combined() {
    let ctx = PreloadedContext {
        memories: vec![RecalledMemory {
            key: "k".into(),
            value: "v".into(),
            score: 0.9,
            related: Vec::new(),
            outcome: None,
        }],
        feedback: vec![FeedbackEntry {
            message_id: "x".into(),
            rating: 2,
            comment: String::new(),
            created_at: chrono::Utc::now(),
            source: "http".into(),
            title: "Some page".into(),
        }],
        recall_quality: RecallQuality::Strong,
    };
    let text = format_preloaded_context(&ctx);
    assert!(text.contains("Memory context"));
    assert!(text.contains("User feedback"));
    // Empty comment should not produce a colon.
    assert!(text.contains("\"Some page\" rated 2/3\n"));
}
