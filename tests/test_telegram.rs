/// Integration tests for the Telegram adapter using `teloxide_tests`.
///
/// These tests dispatch mock Telegram updates through the real handler logic
/// without touching the Telegram API.
use std::path::PathBuf;
use std::sync::Arc;

use dashmap::DashMap;
use inbox::adapters::telegram::build_handler;
use inbox::adapters::telegram_notifier::NotifyConfig;
use inbox::message::{IncomingMessage, MessageSource, RetryableMessage, SourceMetadata};
use teloxide::dptree;
use teloxide_tests::{
    MockBot, MockCallbackQuery, MockMessageAnimation, MockMessageAudio, MockMessageContact,
    MockMessageDocument, MockMessageLocation, MockMessagePhoto, MockMessagePoll,
    MockMessageSticker, MockMessageText, MockMessageVideo, MockMessageVoice,
};
use tokio::sync::mpsc;
use uuid::Uuid;

/// Default `MockUser` ID used by `teloxide_tests`.
const DEFAULT_USER_ID: i64 = 12_345_678;

// ── Helper ────────────────────────────────────────────────────────────────────

fn make_channel() -> (
    mpsc::Sender<IncomingMessage>,
    mpsc::Receiver<IncomingMessage>,
) {
    mpsc::channel(10)
}

fn temp_attachments() -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_path_buf();
    (dir, path)
}

fn default_handler(
    allowed: Vec<i64>,
    dir: PathBuf,
) -> teloxide::dispatching::UpdateHandler<teloxide::RequestError> {
    let retry_store: Arc<DashMap<Uuid, RetryableMessage>> = Arc::new(DashMap::new());
    build_handler(
        allowed,
        dir,
        60,
        3,
        1500,
        NotifyConfig {
            retries: 3,
            retry_base_ms: 100,
        },
        retry_store,
    )
}

fn handler_with_short_media_group_timeout(
    dir: PathBuf,
) -> teloxide::dispatching::UpdateHandler<teloxide::RequestError> {
    let retry_store: Arc<DashMap<Uuid, RetryableMessage>> = Arc::new(DashMap::new());
    build_handler(
        vec![],
        dir,
        60,
        3,
        100,
        NotifyConfig {
            retries: 3,
            retry_base_ms: 100,
        },
        retry_store,
    )
}

fn handler_with_store(
    dir: PathBuf,
    retry_store: Arc<DashMap<Uuid, RetryableMessage>>,
) -> teloxide::dispatching::UpdateHandler<teloxide::RequestError> {
    build_handler(
        vec![],
        dir,
        60,
        3,
        1500,
        NotifyConfig {
            retries: 3,
            retry_base_ms: 100,
        },
        retry_store,
    )
}

fn make_retryable(text: &str) -> RetryableMessage {
    let msg = IncomingMessage::new(
        MessageSource::Telegram,
        text.into(),
        SourceMetadata::Telegram {
            chat_id: 123,
            message_id: 1,
            username: None,
            forwarded_from: None,
        },
    );
    RetryableMessage::from(&msg)
}

// ── Text messages ─────────────────────────────────────────────────────────────

#[tokio::test]
async fn text_message_is_enqueued() {
    let (tx, mut rx) = make_channel();
    let (_tmp, dir) = temp_attachments();

    let mut bot = MockBot::new(
        MockMessageText::new().text("Hello inbox!"),
        default_handler(vec![], dir),
    );
    bot.dependencies(dptree::deps![tx]);
    bot.dispatch().await;

    let msg = rx.try_recv().expect("message should be enqueued");
    assert_eq!(msg.text, "Hello inbox!");
    assert_eq!(msg.source_name(), "telegram");
}

#[tokio::test]
async fn text_message_default_text_is_enqueued() {
    let (tx, mut rx) = make_channel();
    let (_tmp, dir) = temp_attachments();

    let mut bot = MockBot::new(MockMessageText::new(), default_handler(vec![], dir));
    bot.dependencies(dptree::deps![tx]);
    bot.dispatch().await;

    let msg = rx.try_recv().expect("message should be enqueued");
    assert_eq!(msg.text, MockMessageText::TEXT);
}

// ── Allow-list filtering ───────────────────────────────────────────────────────

#[tokio::test]
async fn allowed_user_is_enqueued() {
    let (tx, mut rx) = make_channel();
    let (_tmp, dir) = temp_attachments();

    // Allow the default test user (12345678)
    let mut bot = MockBot::new(
        MockMessageText::new().text("allowed"),
        default_handler(vec![DEFAULT_USER_ID], dir),
    );
    bot.dependencies(dptree::deps![tx]);
    bot.dispatch().await;

    let msg = rx.try_recv();
    assert!(msg.is_ok(), "allowed user should be enqueued");
}

#[tokio::test]
async fn blocked_user_is_not_enqueued() {
    let (tx, mut rx) = make_channel();
    let (_tmp, dir) = temp_attachments();

    // Allow only user 999, but the test message comes from user 12345678
    let mut bot = MockBot::new(
        MockMessageText::new().text("blocked"),
        default_handler(vec![999], dir),
    );
    bot.dependencies(dptree::deps![tx]);
    bot.dispatch().await;

    let msg = rx.try_recv();
    assert!(msg.is_err(), "blocked user should not produce a message");
}

// ── Special message types ─────────────────────────────────────────────────────

#[tokio::test]
async fn location_message_is_formatted() {
    let (tx, mut rx) = make_channel();
    let (_tmp, dir) = temp_attachments();

    let mock = MockMessageLocation::new().latitude(48.85).longitude(2.35);

    let mut bot = MockBot::new(mock, default_handler(vec![], dir));
    bot.dependencies(dptree::deps![tx]);
    bot.dispatch().await;

    let msg = rx.try_recv().expect("location message should be enqueued");
    assert!(
        msg.text.contains("48.85"),
        "text should contain latitude: {:?}",
        msg.text
    );
    assert!(
        msg.text.contains("2.35"),
        "text should contain longitude: {:?}",
        msg.text
    );
    assert!(msg.text.starts_with('📍'));
}

#[tokio::test]
async fn contact_message_is_formatted() {
    let (tx, mut rx) = make_channel();
    let (_tmp, dir) = temp_attachments();

    let mock = MockMessageContact::new()
        .first_name("Alice".to_string())
        .last_name("Smith".to_string())
        .phone_number("+12025550100".to_string());

    let mut bot = MockBot::new(mock, default_handler(vec![], dir));
    bot.dependencies(dptree::deps![tx]);
    bot.dispatch().await;

    let msg = rx.try_recv().expect("contact message should be enqueued");
    assert!(
        msg.text.contains("Alice"),
        "should contain first name: {:?}",
        msg.text
    );
    assert!(
        msg.text.contains("Smith"),
        "should contain last name: {:?}",
        msg.text
    );
    assert!(
        msg.text.contains("+12025550100"),
        "should contain phone: {:?}",
        msg.text
    );
    assert!(msg.text.starts_with('👤'));
}

#[tokio::test]
async fn poll_message_is_formatted() {
    use teloxide::types::{PollOption, PollType};

    let (tx, mut rx) = make_channel();
    let (_tmp, dir) = temp_attachments();

    let options = vec![
        PollOption {
            text: "Rust".to_string(),
            voter_count: 5,
            text_entities: None,
        },
        PollOption {
            text: "Python".to_string(),
            voter_count: 3,
            text_entities: None,
        },
    ];

    let mock = MockMessagePoll::new()
        .question("Best language?".to_string())
        .options(options)
        .poll_type(PollType::Regular);

    let mut bot = MockBot::new(mock, default_handler(vec![], dir));
    bot.dependencies(dptree::deps![tx]);
    bot.dispatch().await;

    let msg = rx.try_recv().expect("poll message should be enqueued");
    assert!(
        msg.text.contains("Best language?"),
        "should contain poll question: {:?}",
        msg.text
    );
    assert!(
        msg.text.contains("Rust"),
        "should contain option: {:?}",
        msg.text
    );
    assert!(msg.text.starts_with('📊'));
}

// ── Media messages with file download ────────────────────────────────────────

#[tokio::test]
async fn document_message_downloads_attachment() {
    let (tx, mut rx) = make_channel();
    let (_tmp, dir) = temp_attachments();

    let mock = MockMessageDocument::new()
        .file_name("report.pdf".to_string())
        .caption("My report".to_string());

    let mut bot = MockBot::new(mock, default_handler(vec![], dir));
    bot.dependencies(dptree::deps![tx]);
    bot.dispatch().await;

    let msg = rx.try_recv().expect("document message should be enqueued");
    assert_eq!(msg.text, "My report");
    assert_eq!(msg.attachments.len(), 1, "should have one attachment");
    assert_eq!(msg.attachments[0].original_name, "report.pdf");
}

#[tokio::test]
async fn photo_message_is_enqueued_with_caption() {
    // Note: teloxide_tests' mock server cannot register photo arrays in its file
    // store (find_file doesn't recurse into arrays), so the file download will
    // fail and be reported inline. We only verify the caption text is present
    // and the message is enqueued — not the attachment.
    let (tx, mut rx) = make_channel();
    let (_tmp, dir) = temp_attachments();

    let mock = MockMessagePhoto::new().caption("A nice photo".to_string());

    let mut bot = MockBot::new(mock, default_handler(vec![], dir));
    bot.dependencies(dptree::deps![tx]);
    bot.dispatch().await;

    let msg = rx.try_recv().expect("photo message should be enqueued");
    assert!(
        msg.text.contains("A nice photo"),
        "caption should appear in text: {:?}",
        msg.text
    );
}

#[tokio::test]
async fn voice_message_downloads_attachment() {
    let (tx, mut rx) = make_channel();
    let (_tmp, dir) = temp_attachments();

    let mut bot = MockBot::new(MockMessageVoice::new(), default_handler(vec![], dir));
    bot.dependencies(dptree::deps![tx]);
    bot.dispatch().await;

    let msg = rx.try_recv().expect("voice message should be enqueued");
    assert_eq!(msg.attachments.len(), 1, "should have voice attachment");
    assert_eq!(msg.attachments[0].original_name, "voice.ogg");
}

// ── Additional media types ─────────────────────────────────────────────────────

#[tokio::test]
async fn audio_message_is_enqueued() {
    let (tx, mut rx) = make_channel();
    let (_tmp, dir) = temp_attachments();

    let mut bot = MockBot::new(MockMessageAudio::new(), default_handler(vec![], dir));
    bot.dependencies(dptree::deps![tx]);
    bot.dispatch().await;

    let msg = rx.try_recv().expect("audio message should be enqueued");
    // Either downloaded successfully or skipped with a note.
    assert!(
        msg.attachments.len() == 1 || msg.text.contains("[attachment skipped"),
        "audio should produce attachment or skip note: {msg:?}"
    );
}

#[tokio::test]
async fn video_message_is_enqueued() {
    let (tx, mut rx) = make_channel();
    let (_tmp, dir) = temp_attachments();

    let mut bot = MockBot::new(MockMessageVideo::new(), default_handler(vec![], dir));
    bot.dependencies(dptree::deps![tx]);
    bot.dispatch().await;

    let msg = rx.try_recv().expect("video message should be enqueued");
    assert!(
        msg.attachments.len() == 1 || msg.text.contains("[attachment skipped"),
        "video should produce attachment or skip note: {msg:?}"
    );
}

#[tokio::test]
async fn sticker_message_is_enqueued() {
    let (tx, mut rx) = make_channel();
    let (_tmp, dir) = temp_attachments();

    let mut bot = MockBot::new(MockMessageSticker::new(), default_handler(vec![], dir));
    bot.dependencies(dptree::deps![tx]);
    bot.dispatch().await;

    let msg = rx.try_recv().expect("sticker message should be enqueued");
    assert!(
        msg.attachments.len() == 1 || msg.text.contains("[attachment skipped"),
        "sticker should produce attachment or skip note: {msg:?}"
    );
}

#[tokio::test]
async fn animation_message_is_enqueued() {
    let (tx, mut rx) = make_channel();
    let (_tmp, dir) = temp_attachments();

    let mut bot = MockBot::new(MockMessageAnimation::new(), default_handler(vec![], dir));
    bot.dependencies(dptree::deps![tx]);
    bot.dispatch().await;

    let msg = rx.try_recv().expect("animation message should be enqueued");
    assert!(
        msg.attachments.len() == 1 || msg.text.contains("[attachment skipped"),
        "animation should produce attachment or skip note: {msg:?}"
    );
}

// ── Callback query handling ────────────────────────────────────────────────────

#[tokio::test]
async fn callback_query_non_retry_data_is_ignored() {
    let (tx, mut rx) = make_channel();
    let (_tmp, dir) = temp_attachments();

    let mut bot = MockBot::new(
        MockCallbackQuery::new().data("some_other_action"),
        default_handler(vec![], dir),
    );
    bot.dependencies(dptree::deps![tx]);
    bot.dispatch().await;

    // Callback was answered but no message enqueued.
    let r = bot.get_responses();
    assert!(
        !r.answered_callback_queries.is_empty(),
        "callback must be acked"
    );
    assert!(
        rx.try_recv().is_err(),
        "non-retry data should not enqueue a message"
    );
}

#[tokio::test]
async fn callback_query_invalid_uuid_is_ignored() {
    let (tx, mut rx) = make_channel();
    let (_tmp, dir) = temp_attachments();

    let mut bot = MockBot::new(
        MockCallbackQuery::new().data("retry:not-a-uuid"),
        default_handler(vec![], dir),
    );
    bot.dependencies(dptree::deps![tx]);
    bot.dispatch().await;

    assert!(
        rx.try_recv().is_err(),
        "invalid uuid should not enqueue a message"
    );
}

#[tokio::test]
async fn callback_query_missing_store_key_is_ignored() {
    let (tx, mut rx) = make_channel();
    let (_tmp, dir) = temp_attachments();
    let key = Uuid::new_v4();

    // Store is empty — the key is absent.
    let mut bot = MockBot::new(
        MockCallbackQuery::new().data(format!("retry:{key}")),
        default_handler(vec![], dir),
    );
    bot.dependencies(dptree::deps![tx]);
    bot.dispatch().await;

    assert!(
        rx.try_recv().is_err(),
        "missing store key should not enqueue a message"
    );
}

#[tokio::test]
async fn callback_query_retry_reenqueues_message() {
    let (tx, mut rx) = make_channel();
    let (_tmp, dir) = temp_attachments();

    let retry_store: Arc<DashMap<Uuid, RetryableMessage>> = Arc::new(DashMap::new());
    let key = Uuid::new_v4();
    retry_store.insert(key, make_retryable("retry me"));

    let mut bot = MockBot::new(
        MockCallbackQuery::new().data(format!("retry:{key}")),
        handler_with_store(dir, retry_store.clone()),
    );
    bot.dependencies(dptree::deps![tx]);
    bot.dispatch().await;

    let msg = rx.try_recv().expect("retry should re-enqueue the message");
    assert_eq!(msg.text, "retry me");
    assert!(
        !retry_store.contains_key(&key),
        "key must be removed from store after retry"
    );
}

// ── Media group aggregation ──────────────────────────────────────────────────

#[tokio::test]
async fn media_group_message_is_buffered_not_sent_immediately() {
    // teloxide_tests runs dispatch in a nested runtime that is destroyed after
    // dispatch(), so we cannot test the delayed flush end-to-end here.
    // Instead we verify that a document with media_group_id does NOT produce an
    // immediate message (proving the buffering path is entered), while a
    // document without media_group_id does (see single_document_without_media_group_is_not_delayed).
    let (tx, mut rx) = make_channel();
    let (_tmp, dir) = temp_attachments();

    let mut bot = MockBot::new(
        MockMessageDocument::new()
            .file_name("grouped.pdf".to_string())
            .caption("Album part".to_string())
            .media_group_id("mg-456"),
        handler_with_short_media_group_timeout(dir),
    );
    bot.dependencies(dptree::deps![tx]);
    bot.dispatch().await;

    // The message must NOT arrive immediately — it's buffered for the media group.
    assert!(
        rx.try_recv().is_err(),
        "media group message should be buffered, not sent immediately"
    );
}

#[tokio::test]
async fn single_document_without_media_group_is_not_delayed() {
    let (tx, mut rx) = make_channel();
    let (_tmp, dir) = temp_attachments();

    let mut bot = MockBot::new(
        MockMessageDocument::new()
            .file_name("solo.pdf".to_string())
            .caption("Solo doc".to_string()),
        handler_with_short_media_group_timeout(dir),
    );
    bot.dependencies(dptree::deps![tx]);
    bot.dispatch().await;

    // Should arrive immediately (no media group delay).
    let msg = rx
        .try_recv()
        .expect("single document should be enqueued immediately");
    assert_eq!(msg.text, "Solo doc");
    assert_eq!(msg.attachments.len(), 1);
    assert_eq!(msg.attachments[0].original_name, "solo.pdf");
}
