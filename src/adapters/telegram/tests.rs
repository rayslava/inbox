use std::sync::Arc;

use dashmap::DashMap;
use teloxide::prelude::*;
use teloxide_tests::{MockBot, MockCallbackQuery, MockMessageText};
use uuid::Uuid;

use super::FeedbackMessageMap;
use super::handlers;
use crate::memory::MemoryStore;

/// Dispatches a single `fb:` callback through `handle_feedback_callback`.
async fn dispatch_feedback(
    data: String,
    store: Option<Arc<MemoryStore>>,
    map: FeedbackMessageMap,
) -> teloxide_tests::Responses {
    let mut query = MockCallbackQuery::new();
    query.data = Some(data);
    query.message = Some(MockMessageText::new().text("✅ Saved").build());

    let handler = Update::filter_callback_query().endpoint(
        move |bot: Bot, q: teloxide::types::CallbackQuery| {
            let store = store.clone();
            let map = map.clone();
            async move {
                handlers::handle_feedback_callback(
                    &bot,
                    q.id,
                    q.data.as_deref().unwrap_or(""),
                    q.message.as_ref(),
                    store.as_ref(),
                    &map,
                )
                .await;
                Ok::<(), teloxide::RequestError>(())
            }
        },
    );

    let mut mock = MockBot::new(query, handler);
    mock.dispatch().await;
    mock.get_responses()
}

/// A button press confirms with a toast and clears the inline keyboard even
/// when no memory store is configured.
#[tokio::test]
async fn feedback_without_store_confirms_and_clears_buttons() {
    let inbox_id = Uuid::new_v4();
    let map: FeedbackMessageMap = Arc::new(DashMap::new());

    let r = dispatch_feedback(format!("fb:2:{inbox_id}"), None, map.clone()).await;

    assert_eq!(r.answered_callback_queries.len(), 1);
    assert_eq!(
        r.answered_callback_queries[0].text.as_deref(),
        Some("Thanks! Rated \u{2b50}\u{2b50}")
    );

    assert_eq!(r.edited_messages_text.len(), 1);
    let edited = &r.edited_messages_text[0];
    assert!(edited.bot_request.text.contains("Rated: \u{2b50}\u{2b50}"));
    assert!(
        button_count(edited.bot_request.reply_markup.as_ref()) == 0,
        "inline keyboard must be cleared after feedback"
    );
}

/// A button press persists the rating, registers the message for reply-as-comment
/// and clears the inline keyboard when a memory store is present.
#[tokio::test]
async fn feedback_with_store_saves_and_clears_buttons() {
    let store = Arc::new(MemoryStore::new_in_memory().expect("in-memory store"));
    let inbox_id = Uuid::new_v4();
    let map: FeedbackMessageMap = Arc::new(DashMap::new());

    let r = dispatch_feedback(format!("fb:3:{inbox_id}"), Some(store.clone()), map.clone()).await;

    let saved = store
        .query_feedback(&inbox_id.to_string())
        .await
        .expect("query feedback");
    assert_eq!(saved.map(|f| f.rating), Some(3));

    assert_eq!(map.iter().count(), 1, "message id must be registered");

    assert_eq!(r.answered_callback_queries.len(), 1);
    assert_eq!(r.edited_messages_text.len(), 1);
    assert_eq!(
        button_count(r.edited_messages_text[0].bot_request.reply_markup.as_ref()),
        0
    );
}

/// Malformed callback data is still acknowledged, but edits nothing.
#[tokio::test]
async fn feedback_malformed_data_only_acknowledges() {
    let map: FeedbackMessageMap = Arc::new(DashMap::new());

    let r = dispatch_feedback("fb:bogus".to_string(), None, map).await;

    assert_eq!(r.answered_callback_queries.len(), 1);
    assert!(r.answered_callback_queries[0].text.is_none());
    assert!(r.edited_messages_text.is_empty());
}

/// Counts the inline buttons present in an edit request's reply markup.
fn button_count(markup: Option<&teloxide::types::ReplyMarkup>) -> usize {
    match markup {
        Some(teloxide::types::ReplyMarkup::InlineKeyboard(kb)) => {
            kb.inline_keyboard.iter().map(Vec::len).sum()
        }
        _ => 0,
    }
}
