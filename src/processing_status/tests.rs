use super::*;

#[test]
fn insert_then_snapshot_returns_one_entry() {
    let tracker = ProcessingTracker::new();
    let id = Uuid::new_v4();
    tracker.insert(id, "telegram".into(), "hello world".into());
    let snap = tracker.snapshot();
    assert_eq!(snap.len(), 1);
    assert_eq!(snap[0].id, id);
    assert_eq!(snap[0].source, "telegram");
    assert!(matches!(snap[0].stage, ProcessingStage::Received));
}

#[test]
fn advance_updates_stage() {
    let tracker = ProcessingTracker::new();
    let id = Uuid::new_v4();
    tracker.insert(id, "http".into(), "test".into());
    tracker.advance(id, ProcessingStage::Enriching);
    let snap = tracker.snapshot();
    assert!(matches!(snap[0].stage, ProcessingStage::Enriching));
}

#[test]
fn snapshot_prunes_old_done_entries() {
    let tracker = ProcessingTracker::new();
    let id = Uuid::new_v4();
    tracker.insert(id, "http".into(), "test".into());
    tracker.advance(
        id,
        ProcessingStage::Done {
            title: "my title".into(),
        },
    );

    // Manually backdate the updated_at to simulate expiry
    if let Some(mut entry) = tracker.entries.get_mut(&id) {
        entry.updated_at = Utc::now() - chrono::Duration::seconds(DONE_RETAIN_SECS + 1);
    }

    let snap = tracker.snapshot();
    assert!(snap.is_empty(), "expired Done entry should be pruned");
}

#[tokio::test]
async fn noop_notifier_does_nothing() {
    let mut n = NoopNotifier;
    n.advance(ProcessingStage::Enriching).await;
    n.advance(ProcessingStage::RunningLlm {
        turn: 0,
        max_turns: 5,
        last_tools: vec![],
    })
    .await;
    n.advance(ProcessingStage::Done { title: "x".into() }).await;
}
