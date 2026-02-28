/// Integration tests for the Telegram adapter using `teloxide_tests`.
///
/// These tests dispatch mock Telegram updates through the real handler logic
/// without touching the Telegram API.
use inbox::adapters::telegram::build_handler;
use inbox::message::IncomingMessage;
use teloxide::dptree;
use teloxide_tests::{
    MockBot, MockMessageContact, MockMessageDocument, MockMessageLocation, MockMessagePhoto,
    MockMessagePoll, MockMessageText, MockMessageVoice,
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

// ── Text messages ─────────────────────────────────────────────────────────────

#[tokio::test]
async fn text_message_is_enqueued() {
    let (tx, mut rx) = make_channel();
    let (_tmp, dir) = temp_attachments();

    let mut bot = MockBot::new(
        MockMessageText::new().text("Hello inbox!"),
        build_handler(vec![], dir),
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

    let mut bot = MockBot::new(MockMessageText::new(), build_handler(vec![], dir));
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
        build_handler(vec![DEFAULT_USER_ID], dir),
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
        build_handler(vec![999], dir),
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

    let mut bot = MockBot::new(mock, build_handler(vec![], dir));
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

    let mut bot = MockBot::new(mock, build_handler(vec![], dir));
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

    let mut bot = MockBot::new(mock, build_handler(vec![], dir));
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

    let mut bot = MockBot::new(mock, build_handler(vec![], dir));
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

    let mut bot = MockBot::new(mock, build_handler(vec![], dir));
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

    let mut bot = MockBot::new(MockMessageVoice::new(), build_handler(vec![], dir));
    bot.dependencies(dptree::deps![tx]);
    bot.dispatch().await;

    let msg = rx.try_recv().expect("voice message should be enqueued");
    assert_eq!(msg.attachments.len(), 1, "should have voice attachment");
    assert_eq!(msg.attachments[0].original_name, "voice.ogg");
}
