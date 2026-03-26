/// Integration tests for the Telegram adapter using `teloxide_tests`.
///
/// These tests dispatch mock Telegram updates through the real handler logic
/// without touching the Telegram API.
use std::path::PathBuf;
use std::sync::Arc;

use dashmap::DashMap;
use inbox::adapters::telegram::{HandlerConfig, build_handler};
use inbox::adapters::telegram_notifier::NotifyConfig;
use inbox::message::IncomingMessage;
use teloxide::dptree;
use teloxide_tests::{
    MockBot, MockMessageContact, MockMessageLocation, MockMessagePoll, MockMessageText,
};
use tokio::sync::mpsc;

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
    build_handler(HandlerConfig {
        allowed_user_ids: allowed,
        attachments_dir: dir,
        file_download_timeout_secs: 60,
        file_download_retries: 3,
        media_group_timeout_ms: 1500,
        notify_cfg: NotifyConfig {
            retries: 3,
            retry_base_ms: 100,
        },
        retry_store: Arc::new(DashMap::new()),
        memory_store: None,
        feedback_msg_map: Arc::new(DashMap::new()),
    })
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
