//! Resume-specific Telegram notifier for completed retries.
//!
//! When the background resume task successfully processes a pending item that
//! originated from Telegram, this notifier edits the original status message to
//! show the final title and attaches star-rating feedback buttons.

use std::sync::Arc;

use dashmap::DashMap;
use teloxide::payloads::{EditMessageTextSetters, SendMessageSetters};
use teloxide::prelude::Requester;
use teloxide::types::{ChatId, InlineKeyboardButton, InlineKeyboardMarkup, MessageId};
use tracing::{debug, warn};
use uuid::Uuid;

use crate::adapters::telegram::FeedbackMessageMap;
use crate::error::InboxError;
use crate::message::RetryableMessage;
use crate::pending::PendingItem;
use crate::pending::PendingStore;

/// Sends a completion notification (with feedback buttons) for a resumed item.
///
/// Uses the `chat_id` from the stored [`PendingItem`] metadata and the
/// `telegram_status_msg_id` to edit the original status message. Falls back
/// to `send_message` if editing fails.
pub struct TelegramResumeNotifier {
    pub bot: teloxide::Bot,
    pub feedback_msg_map: FeedbackMessageMap,
    pub retry_store: Arc<DashMap<Uuid, RetryableMessage>>,
}

impl TelegramResumeNotifier {
    /// Build a resume notifier that shares the live adapter's maps.
    #[must_use]
    pub fn from_shared(
        bot: teloxide::Bot,
        shared: &crate::adapters::telegram::TelegramShared,
    ) -> Self {
        Self {
            bot,
            feedback_msg_map: shared.feedback_msg_map.clone(),
            retry_store: shared.retry_store.clone(),
        }
    }

    /// Notify the user that retry succeeded: edit the original status message
    /// to show `✅ {title}` and attach star-rating buttons.
    #[anodized::spec(requires: !title.is_empty())]
    pub async fn notify_done(
        &self,
        item: &PendingItem,
        title: &str,
        inbox_id: Uuid,
    ) -> Result<(), InboxError> {
        let Some(chat_id_raw) = PendingStore::telegram_chat_id(item) else {
            return Ok(());
        };

        let chat_id = ChatId(chat_id_raw);
        let text = format!("✅ {title}");
        let markup = InlineKeyboardMarkup::new(vec![vec![
            InlineKeyboardButton::callback("⭐", format!("fb:1:{inbox_id}")),
            InlineKeyboardButton::callback("⭐⭐", format!("fb:2:{inbox_id}")),
            InlineKeyboardButton::callback("⭐⭐⭐", format!("fb:3:{inbox_id}")),
        ]]);

        let final_msg_id = if let Some(msg_id) = item.telegram_status_msg_id {
            let sent_msg_id = MessageId(msg_id);
            let edit_result = self
                .bot
                .edit_message_text(chat_id, sent_msg_id, &text)
                .reply_markup(markup.clone())
                .await;
            match edit_result {
                Ok(_) => sent_msg_id,
                Err(e) => {
                    warn!(?e, %inbox_id, "Failed to edit status message on resume; sending new message");
                    self.send_new_message(chat_id, &text, markup, inbox_id)
                        .await?
                }
            }
        } else {
            self.send_new_message(chat_id, &text, markup, inbox_id)
                .await?
        };

        self.feedback_msg_map.insert(final_msg_id.0, inbox_id);
        self.retry_store.remove(&inbox_id);
        debug!(
            %inbox_id,
            msg_id = final_msg_id.0,
            "Attached feedback buttons on resume"
        );
        Ok(())
    }

    /// Notify the user that all retries were exhausted: edit the status message
    /// to a failed marker. No feedback buttons — the node was never completed,
    /// so it must not be presented as a successful result.
    #[anodized::spec(requires: !title.is_empty())]
    pub async fn notify_failed(
        &self,
        item: &PendingItem,
        title: &str,
        inbox_id: Uuid,
    ) -> Result<(), InboxError> {
        let Some(chat_id_raw) = PendingStore::telegram_chat_id(item) else {
            return Ok(());
        };
        let chat_id = ChatId(chat_id_raw);
        let text = format!("❌ Failed after retries: {title}");

        if let Some(msg_id) = item.telegram_status_msg_id {
            let sent_msg_id = MessageId(msg_id);
            if let Err(e) = self
                .bot
                .edit_message_text(chat_id, sent_msg_id, &text)
                .await
            {
                warn!(?e, %inbox_id, "Failed to edit status on exhaustion; sending new message");
                self.bot.send_message(chat_id, &text).await.map_err(|e| {
                    InboxError::Adapter(format!("resume notify_failed: {e} (id={inbox_id})"))
                })?;
            }
        } else {
            self.bot.send_message(chat_id, &text).await.map_err(|e| {
                InboxError::Adapter(format!("resume notify_failed: {e} (id={inbox_id})"))
            })?;
        }

        self.retry_store.remove(&inbox_id);
        debug!(%inbox_id, "Marked resumed item as failed after exhausting retries");
        Ok(())
    }

    async fn send_new_message(
        &self,
        chat_id: ChatId,
        text: &str,
        markup: InlineKeyboardMarkup,
        inbox_id: Uuid,
    ) -> Result<MessageId, InboxError> {
        self.bot
            .send_message(chat_id, text)
            .reply_markup(markup)
            .await
            .map(|m| m.id)
            .map_err(|e| InboxError::Adapter(format!("resume send_message: {e} (id={inbox_id})")))
    }
}
