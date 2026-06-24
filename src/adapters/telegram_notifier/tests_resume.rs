//! Tests for the `TelegramResumeNotifier` (pending-item resume notifications).

use std::sync::Arc;

use chrono::Utc;
use dashmap::DashMap;
use teloxide::prelude::*;
use teloxide_tests::{MockBot, MockMessageText};
use uuid::Uuid;

use super::tests::dummy_retryable;
use crate::adapters::telegram::{TelegramShared, build_bot};
use crate::adapters::telegram_notifier::resume::TelegramResumeNotifier;
use crate::message::{ProcessingHints, RetryableMessage, SourceMetadata};
use crate::pending::PendingItem;
use crate::url_content::UrlContent;

// ── TelegramResumeNotifier tests ──────────────────────────────────────────────

fn dummy_http_pending_item() -> PendingItem {
    PendingItem {
        id: Uuid::new_v4(),
        created_at: Utc::now(),
        retry_count: 0,
        last_retry_at: None,
        incoming: RetryableMessage {
            text: "msg".into(),
            metadata: SourceMetadata::Http {
                remote_addr: None,
                user_agent: None,
            },
            attachments: vec![],
            user_tags: vec![],
            preprocessing_hints: ProcessingHints::default(),
            received_at: Utc::now(),
            image_analyses: Vec::new(),
        },
        url_contents: vec![UrlContent {
            url: "https://example.com".into(),
            text: "text".into(),
            page_title: None,
            headings: vec![],
        }],
        tool_results: vec![],
        source_urls: vec![],
        fallback_title: None,
        telegram_status_msg_id: None,
        source: "http".into(),
        url_count: 1,
        tool_count: 0,
    }
}

#[test]
fn telegram_shared_new_starts_empty() {
    let shared = TelegramShared::new();
    assert!(shared.retry_store.is_empty());
    assert!(shared.feedback_msg_map.is_empty());
}

#[test]
fn build_bot_succeeds_for_nonempty_token() {
    assert!(build_bot("fake:token").is_ok());
}

#[test]
fn from_shared_reuses_the_same_maps() {
    let shared = TelegramShared::new();
    shared.retry_store.insert(Uuid::new_v4(), dummy_retryable());
    shared.feedback_msg_map.insert(7, Uuid::new_v4());

    let bot = build_bot("fake:token").expect("bot builds for non-empty token");
    let notifier = TelegramResumeNotifier::from_shared(bot, &shared);

    // Same underlying allocations — resume notifications and live callbacks
    // must resolve against one shared map, not separate copies.
    assert!(Arc::ptr_eq(&notifier.retry_store, &shared.retry_store));
    assert!(Arc::ptr_eq(
        &notifier.feedback_msg_map,
        &shared.feedback_msg_map
    ));
}

#[tokio::test]
async fn notify_done_noop_for_non_telegram_item() {
    // An HTTP item has no chat_id, so notify_done should return Ok immediately
    // without making any bot calls.
    let client = reqwest::Client::new();
    let bot = teloxide::Bot::with_client("fake:token", client);
    let notifier = TelegramResumeNotifier {
        bot,
        feedback_msg_map: Arc::new(DashMap::new()),
        retry_store: Arc::new(DashMap::new()),
    };

    let item = dummy_http_pending_item();
    let result = notifier.notify_done(&item, "Some Title", item.id).await;
    assert!(result.is_ok());
}

fn dummy_telegram_pending_item(status_msg_id: Option<i32>) -> PendingItem {
    PendingItem {
        id: Uuid::new_v4(),
        created_at: Utc::now(),
        retry_count: 0,
        last_retry_at: None,
        incoming: RetryableMessage {
            text: "msg".into(),
            metadata: SourceMetadata::Telegram {
                chat_id: 1,
                message_id: 1,
                username: None,
                forwarded_from: None,
            },
            attachments: vec![],
            user_tags: vec![],
            preprocessing_hints: ProcessingHints::default(),
            received_at: Utc::now(),
            image_analyses: Vec::new(),
        },
        url_contents: vec![],
        tool_results: vec![],
        source_urls: vec![],
        fallback_title: None,
        telegram_status_msg_id: status_msg_id,
        source: "telegram".into(),
        url_count: 0,
        tool_count: 0,
    }
}

#[tokio::test]
async fn notify_done_edits_existing_status_message() {
    let feedback_map: Arc<DashMap<i32, Uuid>> = Arc::new(DashMap::new());
    let retry_store: Arc<DashMap<Uuid, RetryableMessage>> = Arc::new(DashMap::new());
    let key = Uuid::new_v4();
    retry_store.insert(key, dummy_retryable());

    let fb = feedback_map.clone();
    let rs = retry_store.clone();

    let handler = Update::filter_message().endpoint(move |bot: Bot, msg: Message| {
        let fb = fb.clone();
        let rs = rs.clone();
        async move {
            // Pre-send the status message so the notifier has something to edit.
            let sent = bot.send_message(msg.chat.id, "⏳ Processing…").await?;
            let notifier = TelegramResumeNotifier {
                bot,
                feedback_msg_map: fb,
                retry_store: rs,
            };
            let mut item = dummy_telegram_pending_item(Some(sent.id.0));
            item.id = key;
            // Override chat_id to match the mock chat
            if let SourceMetadata::Telegram { chat_id, .. } = &mut item.incoming.metadata {
                *chat_id = msg.chat.id.0;
            }
            let _ = notifier.notify_done(&item, "Resumed Title", key).await;
            Ok::<(), teloxide::RequestError>(())
        }
    });

    let mut mock = MockBot::new(MockMessageText::new(), handler);
    mock.dispatch().await;

    let r = mock.get_responses();
    // Expect exactly one edit with the resumed title and feedback buttons.
    assert_eq!(r.edited_messages_text.len(), 1);
    assert_eq!(
        r.edited_messages_text[0].bot_request.text,
        "✅ Resumed Title"
    );
    assert!(r.edited_messages_text[0].bot_request.reply_markup.is_some());

    // Retry store entry for this id must be cleared; feedback map must record
    // the edited message id.
    assert!(!retry_store.contains_key(&key));
    assert!(!feedback_map.is_empty());
}

#[tokio::test]
async fn notify_failed_marks_failed_without_feedback_buttons() {
    let feedback_map: Arc<DashMap<i32, Uuid>> = Arc::new(DashMap::new());
    let retry_store: Arc<DashMap<Uuid, RetryableMessage>> = Arc::new(DashMap::new());
    let key = Uuid::new_v4();
    retry_store.insert(key, dummy_retryable());

    let fb = feedback_map.clone();
    let rs = retry_store.clone();

    let handler = Update::filter_message().endpoint(move |bot: Bot, msg: Message| {
        let fb = fb.clone();
        let rs = rs.clone();
        async move {
            let sent = bot.send_message(msg.chat.id, "⏳ Retrying…").await?;
            let notifier = TelegramResumeNotifier {
                bot,
                feedback_msg_map: fb,
                retry_store: rs,
            };
            let mut item = dummy_telegram_pending_item(Some(sent.id.0));
            item.id = key;
            if let SourceMetadata::Telegram { chat_id, .. } = &mut item.incoming.metadata {
                *chat_id = msg.chat.id.0;
            }
            let _ = notifier.notify_failed(&item, "Stuck Title", key).await;
            Ok::<(), teloxide::RequestError>(())
        }
    });

    let mut mock = MockBot::new(MockMessageText::new(), handler);
    mock.dispatch().await;

    let r = mock.get_responses();
    assert_eq!(r.edited_messages_text.len(), 1);
    assert_eq!(
        r.edited_messages_text[0].bot_request.text,
        "❌ Failed after retries: Stuck Title"
    );
    // No feedback buttons on a failed item, and no success recorded.
    assert!(r.edited_messages_text[0].bot_request.reply_markup.is_none());
    assert!(feedback_map.is_empty());
    assert!(!retry_store.contains_key(&key));
}

#[tokio::test]
async fn notify_done_sends_new_message_when_status_id_missing() {
    let feedback_map: Arc<DashMap<i32, Uuid>> = Arc::new(DashMap::new());
    let retry_store: Arc<DashMap<Uuid, RetryableMessage>> = Arc::new(DashMap::new());
    let key = Uuid::new_v4();

    let fb = feedback_map.clone();
    let rs = retry_store.clone();

    let handler = Update::filter_message().endpoint(move |bot: Bot, msg: Message| {
        let fb = fb.clone();
        let rs = rs.clone();
        async move {
            let notifier = TelegramResumeNotifier {
                bot,
                feedback_msg_map: fb,
                retry_store: rs,
            };
            let mut item = dummy_telegram_pending_item(None);
            item.id = key;
            if let SourceMetadata::Telegram { chat_id, .. } = &mut item.incoming.metadata {
                *chat_id = msg.chat.id.0;
            }
            let _ = notifier.notify_done(&item, "Fresh Title", key).await;
            Ok::<(), teloxide::RequestError>(())
        }
    });

    let mut mock = MockBot::new(MockMessageText::new(), handler);
    mock.dispatch().await;

    let r = mock.get_responses();
    assert!(
        r.edited_messages_text.is_empty(),
        "no edit should happen without a status_msg_id"
    );
    assert_eq!(r.sent_messages_text.len(), 1);
    assert_eq!(r.sent_messages_text[0].bot_request.text, "✅ Fresh Title");
    assert!(r.sent_messages_text[0].bot_request.reply_markup.is_some());
    assert!(!feedback_map.is_empty());
}
