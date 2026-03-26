use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use dashmap::DashMap;
use uuid::Uuid;

use crate::message::{Attachment, MediaKind, RetryableMessage};

use super::*;

fn dummy_attachment(name: &str) -> Attachment {
    Attachment {
        original_name: name.to_owned(),
        saved_path: PathBuf::from(format!("/tmp/{name}")),
        mime_type: Some("image/jpeg".to_owned()),
        media_kind: MediaKind::Image,
    }
}

// ── get_or_create ────────────────────────────────────────────────────────────

#[test]
fn get_or_create_returns_new_on_first_call() {
    let groups = new_map();
    let (state, is_new) = get_or_create(&groups, "group-1");
    assert!(is_new);
    assert_eq!(groups.len(), 1);

    let inner = state.inner.lock().unwrap();
    assert!(inner.text.is_empty());
    assert!(inner.attachments.is_empty());
}

#[test]
fn get_or_create_returns_existing_on_second_call() {
    let groups = new_map();
    let (first, is_new_1) = get_or_create(&groups, "group-1");
    let (second, is_new_2) = get_or_create(&groups, "group-1");

    assert!(is_new_1);
    assert!(!is_new_2);

    let id_1 = first.inner.lock().unwrap().msg_id;
    let id_2 = second.inner.lock().unwrap().msg_id;
    assert_eq!(id_1, id_2, "should return same state with same msg_id");
}

#[test]
fn get_or_create_different_groups_are_independent() {
    let groups = new_map();
    let (_, new_a) = get_or_create(&groups, "group-a");
    let (_, new_b) = get_or_create(&groups, "group-b");
    assert!(new_a);
    assert!(new_b);
    assert_eq!(groups.len(), 2);
}

// ── set_metadata ─────────────────────────────────────────────────────────────

#[test]
fn set_metadata_populates_fields() {
    let groups = new_map();
    let (state, _) = get_or_create(&groups, "g");

    let chat_id = teloxide::types::ChatId(42);
    let msg_id_tg = teloxide::types::MessageId(7);
    set_metadata(
        &state,
        chat_id,
        7,
        Some("alice".to_owned()),
        Some("bob".to_owned()),
        Some(msg_id_tg),
    );

    let inner = state.inner.lock().unwrap();
    assert_eq!(inner.chat_id, chat_id);
    assert_eq!(inner.first_message_id, 7);
    assert_eq!(inner.username.as_deref(), Some("alice"));
    assert_eq!(inner.forwarded_from.as_deref(), Some("bob"));
    assert_eq!(inner.sent_status_id, Some(msg_id_tg));
}

// ── add_content ──────────────────────────────────────────────────────────────

#[test]
fn add_content_sets_text_on_first_call() {
    let groups = new_map();
    let (state, _) = get_or_create(&groups, "g");

    add_content(&state, "caption one".to_owned(), vec![]);
    let inner = state.inner.lock().unwrap();
    assert_eq!(inner.text, "caption one");
}

#[test]
fn add_content_appends_text_with_newline() {
    let groups = new_map();
    let (state, _) = get_or_create(&groups, "g");

    add_content(&state, "first".to_owned(), vec![]);
    add_content(&state, "second".to_owned(), vec![]);

    let inner = state.inner.lock().unwrap();
    assert_eq!(inner.text, "first\nsecond");
}

#[test]
fn add_content_skips_empty_text() {
    let groups = new_map();
    let (state, _) = get_or_create(&groups, "g");

    add_content(&state, "caption".to_owned(), vec![]);
    add_content(&state, String::new(), vec![]);

    let inner = state.inner.lock().unwrap();
    assert_eq!(inner.text, "caption");
}

#[test]
fn add_content_accumulates_attachments() {
    let groups = new_map();
    let (state, _) = get_or_create(&groups, "g");

    add_content(&state, String::new(), vec![dummy_attachment("a.jpg")]);
    add_content(&state, String::new(), vec![dummy_attachment("b.jpg")]);
    add_content(
        &state,
        String::new(),
        vec![dummy_attachment("c.jpg"), dummy_attachment("d.jpg")],
    );

    let inner = state.inner.lock().unwrap();
    assert_eq!(inner.attachments.len(), 4);
    let names: Vec<_> = inner
        .attachments
        .iter()
        .map(|a| a.original_name.as_str())
        .collect();
    assert_eq!(names, vec!["a.jpg", "b.jpg", "c.jpg", "d.jpg"]);
}

// ── spawn_flush ──────────────────────────────────────────────────────────────

#[tokio::test]
async fn spawn_flush_sends_combined_message_after_timeout() {
    let groups = new_map();
    let (state, _) = get_or_create(&groups, "grp");

    set_metadata(
        &state,
        teloxide::types::ChatId(100),
        1,
        Some("user".to_owned()),
        None,
        None,
    );
    add_content(
        &state,
        "album caption".to_owned(),
        vec![dummy_attachment("photo1.jpg")],
    );
    add_content(&state, String::new(), vec![dummy_attachment("photo2.jpg")]);

    let (tx, mut rx) = tokio::sync::mpsc::channel::<IncomingMessage>(4);

    let client = reqwest::Client::new();
    let bot = teloxide::Bot::with_client("fake:token", client);
    let retry_store: Arc<DashMap<Uuid, RetryableMessage>> = Arc::new(DashMap::new());

    spawn_flush(
        groups.clone(),
        "grp".to_owned(),
        state,
        Duration::from_millis(50),
        tx,
        FlushContext {
            bot,
            retry_store,
            notify_cfg: crate::adapters::telegram_notifier::NotifyConfig {
                retries: 3,
                retry_base_ms: 100,
            },
            feedback_msg_map: Arc::new(DashMap::new()),
        },
    );

    // Wait for flush to fire.
    tokio::time::sleep(Duration::from_millis(200)).await;

    let msg = rx.try_recv().expect("should receive combined message");
    assert_eq!(msg.text, "album caption");
    assert_eq!(msg.attachments.len(), 2);
    assert_eq!(msg.attachments[0].original_name, "photo1.jpg");
    assert_eq!(msg.attachments[1].original_name, "photo2.jpg");
    assert_eq!(msg.source, MessageSource::Telegram);

    // Group should be removed from the map after flush.
    assert!(groups.is_empty());
}

#[tokio::test]
async fn spawn_flush_waits_for_pending_downloads() {
    let groups = new_map();
    let (state, _) = get_or_create(&groups, "grp");

    // Simulate one in-flight download.
    state.pending_downloads.fetch_add(1, Ordering::Release);

    set_metadata(&state, teloxide::types::ChatId(100), 1, None, None, None);
    add_content(
        &state,
        "partial".to_owned(),
        vec![dummy_attachment("a.jpg")],
    );

    let (tx, mut rx) = tokio::sync::mpsc::channel::<IncomingMessage>(4);

    let client = reqwest::Client::new();
    let bot = teloxide::Bot::with_client("fake:token", client);
    let retry_store: Arc<DashMap<Uuid, RetryableMessage>> = Arc::new(DashMap::new());

    let state_clone = Arc::clone(groups.get("grp").unwrap().value());
    spawn_flush(
        groups.clone(),
        "grp".to_owned(),
        state_clone.clone(),
        Duration::from_millis(50),
        tx,
        FlushContext {
            bot,
            retry_store,
            notify_cfg: crate::adapters::telegram_notifier::NotifyConfig {
                retries: 3,
                retry_base_ms: 100,
            },
            feedback_msg_map: Arc::new(DashMap::new()),
        },
    );

    // After timeout elapses, message should NOT be sent yet (download pending).
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(
        rx.try_recv().is_err(),
        "should not flush while downloads pending"
    );

    // Complete the download and add the attachment.
    add_content(&state_clone, String::new(), vec![dummy_attachment("b.jpg")]);
    state_clone
        .pending_downloads
        .fetch_sub(1, Ordering::Release);

    // Now the flush should proceed.
    tokio::time::sleep(Duration::from_millis(400)).await;
    let msg = rx
        .try_recv()
        .expect("should receive after download completes");
    assert_eq!(msg.attachments.len(), 2);
}
