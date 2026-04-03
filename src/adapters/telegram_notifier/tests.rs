use std::sync::{Arc, Mutex};

use chrono::Utc;
use dashmap::DashMap;
use teloxide::prelude::*;
use teloxide::types::MessageId;
use teloxide_tests::{MockBot, MockMessageText};
use uuid::Uuid;

use super::*;
use crate::adapters::telegram_notifier::resume::TelegramResumeNotifier;
use crate::message::{
    IncomingMessage, MessageSource, ProcessingHints, RetryableMessage, SourceMetadata,
};
use crate::pending::PendingItem;
use crate::url_content::UrlContent;

// ── Pure-function unit tests ───────────────────────────────────────────────────

#[tokio::test]
async fn stage_text_formats_correctly() {
    assert_eq!(stage_text(&ProcessingStage::Received), "⏳ Processing…");
    assert_eq!(
        stage_text(&ProcessingStage::Enriching),
        "🔍 Fetching content…"
    );
    assert_eq!(
        stage_text(&ProcessingStage::RunningLlm {
            turn: 0,
            max_turns: 5,
            last_tools: vec![],
        }),
        "🤖 Analysing…"
    );
    assert_eq!(
        stage_text(&ProcessingStage::RunningLlm {
            turn: 2,
            max_turns: 5,
            last_tools: vec!["web_search".into()],
        }),
        "🤖 Analysing… (turn 2/5 · web_search)"
    );
    assert_eq!(stage_text(&ProcessingStage::Writing), "✍️ Saving…");
    assert_eq!(
        stage_text(&ProcessingStage::Done {
            title: "My Title".into()
        }),
        "✅ My Title"
    );
    assert_eq!(
        stage_text(&ProcessingStage::Failed {
            reason: "oops".into()
        }),
        "❌ Failed: oops"
    );
}

#[test]
fn is_terminal_done_and_failed() {
    assert!(is_terminal(&ProcessingStage::Done { title: "x".into() }));
    assert!(is_terminal(&ProcessingStage::Failed { reason: "y".into() }));
    assert!(!is_terminal(&ProcessingStage::Received));
    assert!(!is_terminal(&ProcessingStage::Enriching));
    assert!(!is_terminal(&ProcessingStage::RunningLlm {
        turn: 0,
        max_turns: 5,
        last_tools: vec![]
    }));
    assert!(!is_terminal(&ProcessingStage::Writing));
}

#[test]
fn terminal_stages_get_more_retries() {
    // Terminal stages should receive more retries than non-terminal ones.
    assert!(!is_terminal(&ProcessingStage::Enriching));
    assert!(is_terminal(&ProcessingStage::Done { title: "x".into() }));
}

// ── Dispatch-backed notifier tests ────────────────────────────────────────────
//
// `teloxide_tests` only starts its mock HTTP server inside `dispatch()`, so
// direct `bot.*` calls must be issued from within a handler endpoint.

fn dummy_retryable() -> RetryableMessage {
    let msg = IncomingMessage::new(
        MessageSource::Telegram,
        "test".into(),
        SourceMetadata::Telegram {
            chat_id: 1,
            message_id: 1,
            username: None,
            forwarded_from: None,
        },
    );
    RetryableMessage::from(&msg)
}

/// Tests that `advance(Enriching)` edits the previously-sent status message.
#[tokio::test]
async fn advance_enriching_edits_status_message() {
    let store: Arc<DashMap<Uuid, RetryableMessage>> = Arc::new(DashMap::new());
    let s = store.clone();
    let key = Uuid::new_v4();

    let handler = Update::filter_message().endpoint(move |bot: Bot, msg: Message| {
        let s = s.clone();
        async move {
            let sent = bot.send_message(msg.chat.id, "⏳ Processing…").await?;
            let mut notifier = TelegramNotifier::new(
                bot,
                msg.chat.id,
                sent.id,
                s,
                key,
                dummy_retryable(),
                NotifyConfig {
                    retries: 3,
                    retry_base_ms: 100,
                },
            );
            notifier.advance(ProcessingStage::Enriching).await;
            Ok::<(), teloxide::RequestError>(())
        }
    });

    let mut mock = MockBot::new(MockMessageText::new(), handler);
    mock.dispatch().await;

    let r = mock.get_responses();
    assert_eq!(r.edited_messages_text.len(), 1);
    assert_eq!(
        r.edited_messages_text[0].bot_request.text,
        "🔍 Fetching content…"
    );
    assert!(r.edited_messages_text[0].bot_request.reply_markup.is_none());
}

/// Tests that `advance(Failed)` stores the retryable and adds a retry button.
#[tokio::test]
async fn advance_failed_inserts_retry_store_and_adds_button() {
    let store: Arc<DashMap<Uuid, RetryableMessage>> = Arc::new(DashMap::new());
    let s = store.clone();
    let key = Uuid::new_v4();

    let handler = Update::filter_message().endpoint(move |bot: Bot, msg: Message| {
        let s = s.clone();
        async move {
            let sent = bot.send_message(msg.chat.id, "⏳ Processing…").await?;
            let mut notifier = TelegramNotifier::new(
                bot,
                msg.chat.id,
                sent.id,
                s,
                key,
                dummy_retryable(),
                NotifyConfig {
                    retries: 3,
                    retry_base_ms: 100,
                },
            );
            notifier
                .advance(ProcessingStage::Failed {
                    reason: "pipeline error".into(),
                })
                .await;
            Ok::<(), teloxide::RequestError>(())
        }
    });

    let mut mock = MockBot::new(MockMessageText::new(), handler);
    mock.dispatch().await;

    assert!(
        store.contains_key(&key),
        "retryable must be inserted into store on failure"
    );
    let r = mock.get_responses();
    assert_eq!(r.edited_messages_text.len(), 1);
    assert!(
        r.edited_messages_text[0]
            .bot_request
            .text
            .contains("pipeline error")
    );
    assert!(
        r.edited_messages_text[0].bot_request.reply_markup.is_some(),
        "retry inline button must be present"
    );
}

/// Tests that `advance(Done)` removes a pre-existing retryable from the store.
#[tokio::test]
async fn advance_done_removes_from_retry_store() {
    let store: Arc<DashMap<Uuid, RetryableMessage>> = Arc::new(DashMap::new());
    let s = store.clone();
    let key = Uuid::new_v4();
    store.insert(key, dummy_retryable()); // pre-populate

    let handler = Update::filter_message().endpoint(move |bot: Bot, msg: Message| {
        let s = s.clone();
        async move {
            let sent = bot.send_message(msg.chat.id, "⏳ Processing…").await?;
            let mut notifier = TelegramNotifier::new(
                bot,
                msg.chat.id,
                sent.id,
                s,
                key,
                dummy_retryable(),
                NotifyConfig {
                    retries: 3,
                    retry_base_ms: 100,
                },
            );
            notifier
                .advance(ProcessingStage::Done {
                    title: "Saved".into(),
                })
                .await;
            Ok::<(), teloxide::RequestError>(())
        }
    });

    let mut mock = MockBot::new(MockMessageText::new(), handler);
    mock.dispatch().await;

    assert!(
        !store.contains_key(&key),
        "retryable must be removed from store on done"
    );
    let r = mock.get_responses();
    assert!(
        r.edited_messages_text[0].bot_request.reply_markup.is_some(),
        "feedback inline buttons must be present on Done"
    );
}

/// Tests that `send_status_reply` sends "⏳ Processing…" and returns a `MessageId`.
#[tokio::test]
async fn send_status_reply_sends_processing_message() {
    let result: Arc<Mutex<bool>> = Arc::new(Mutex::new(false));
    let r = result.clone();

    let handler = Update::filter_message().endpoint(move |bot: Bot, msg: Message| {
        let r = r.clone();
        async move {
            let id = send_status_reply(
                &bot,
                msg.chat.id,
                None,
                NotifyConfig {
                    retries: 3,
                    retry_base_ms: 100,
                },
            )
            .await;
            *r.lock().unwrap() = id.is_some();
            Ok::<(), teloxide::RequestError>(())
        }
    });

    let mut mock = MockBot::new(MockMessageText::new(), handler);
    mock.dispatch().await;

    assert!(
        *result.lock().unwrap(),
        "send_status_reply must return Some"
    );
    let r = mock.get_responses();
    assert_eq!(r.sent_messages_text[0].bot_request.text, "⏳ Processing…");
}

/// Tests that `send_status_reply` with a `reply_to` sets the `reply_parameters`.
#[tokio::test]
async fn send_status_reply_with_reply_to_sets_parameters() {
    let handler = Update::filter_message().endpoint(move |bot: Bot, msg: Message| async move {
        // reply to the incoming message itself (id = 1 in mock)
        send_status_reply(
            &bot,
            msg.chat.id,
            Some(msg.id),
            NotifyConfig {
                retries: 3,
                retry_base_ms: 100,
            },
        )
        .await;
        Ok::<(), teloxide::RequestError>(())
    });

    let mut mock = MockBot::new(MockMessageText::new(), handler);
    mock.dispatch().await;

    let r = mock.get_responses();
    let params = r.sent_messages_text[0]
        .bot_request
        .reply_parameters
        .as_ref()
        .expect("reply_parameters must be set when reply_to is Some");
    assert_eq!(params.message_id, MessageId(1));
}

/// Tests that `build_telegram_notifier` sends the initial message and returns a notifier.
#[tokio::test]
async fn build_telegram_notifier_sends_initial_and_returns_notifier() {
    let result: Arc<Mutex<bool>> = Arc::new(Mutex::new(false));
    let r = result.clone();
    let store: Arc<DashMap<Uuid, RetryableMessage>> = Arc::new(DashMap::new());
    let s = store.clone();

    let handler = Update::filter_message().endpoint(move |bot: Bot, msg: Message| {
        let r = r.clone();
        let s = s.clone();
        async move {
            let key = Uuid::new_v4();
            let notifier = build_telegram_notifier(
                &bot,
                BuildNotifierArgs {
                    chat_id: msg.chat.id,
                    reply_to: None,
                    retry_store: s,
                    retry_key: key,
                    retryable: dummy_retryable(),
                    cfg: NotifyConfig {
                        retries: 3,
                        retry_base_ms: 100,
                    },
                    feedback_msg_map: None,
                },
            )
            .await;
            *r.lock().unwrap() = notifier.is_some();
            Ok::<(), teloxide::RequestError>(())
        }
    });

    let mut mock = MockBot::new(MockMessageText::new(), handler);
    mock.dispatch().await;

    assert!(
        *result.lock().unwrap(),
        "build_telegram_notifier must return Some"
    );
    let r = mock.get_responses();
    assert!(!r.sent_messages_text.is_empty());
    assert_eq!(r.sent_messages_text[0].bot_request.text, "⏳ Processing…");
}

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
