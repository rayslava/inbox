use std::sync::Arc;
use std::time::Duration;

use dashmap::DashMap;
use teloxide::payloads::{EditMessageTextSetters, SendMessageSetters};
use teloxide::prelude::Requester;
use teloxide::types::{InlineKeyboardButton, InlineKeyboardMarkup};
use tracing::warn;
use uuid::Uuid;

use crate::message::RetryableMessage;
use crate::processing_status::{ProcessingStage, StatusNotifier};

const MAX_NOTIFY_RETRIES: u32 = 3;
const TERMINAL_NOTIFY_RETRIES: u32 = 5;
const NOTIFY_RETRY_BASE_MS: u64 = 500;

pub struct TelegramNotifier {
    bot: teloxide::Bot,
    chat_id: teloxide::types::ChatId,
    sent_msg_id: teloxide::types::MessageId,
    retry_store: Arc<DashMap<Uuid, RetryableMessage>>,
    retry_key: Uuid,
    retryable: RetryableMessage,
}

impl TelegramNotifier {
    #[must_use]
    pub fn new(
        bot: teloxide::Bot,
        chat_id: teloxide::types::ChatId,
        sent_msg_id: teloxide::types::MessageId,
        retry_store: Arc<DashMap<Uuid, RetryableMessage>>,
        retry_key: Uuid,
        retryable: RetryableMessage,
    ) -> Self {
        Self {
            bot,
            chat_id,
            sent_msg_id,
            retry_store,
            retry_key,
            retryable,
        }
    }
}

pub(super) fn stage_text(stage: &ProcessingStage) -> String {
    match stage {
        ProcessingStage::Received => "⏳ Processing…".to_owned(),
        ProcessingStage::Enriching => "🔍 Fetching content…".to_owned(),
        ProcessingStage::RunningLlm => "🤖 Analysing…".to_owned(),
        ProcessingStage::Writing => "✍️ Saving…".to_owned(),
        ProcessingStage::Done { title } => format!("✅ {title}"),
        ProcessingStage::Failed { reason } => format!("❌ Failed: {reason}"),
    }
}

fn is_terminal(stage: &ProcessingStage) -> bool {
    matches!(
        stage,
        ProcessingStage::Done { .. } | ProcessingStage::Failed { .. }
    )
}

#[async_trait::async_trait]
impl StatusNotifier for TelegramNotifier {
    async fn advance(&mut self, stage: ProcessingStage) {
        let text = stage_text(&stage);
        let retries = if is_terminal(&stage) {
            TERMINAL_NOTIFY_RETRIES
        } else {
            MAX_NOTIFY_RETRIES
        };

        let is_failed = matches!(stage, ProcessingStage::Failed { .. });
        let is_done = matches!(stage, ProcessingStage::Done { .. });

        if is_failed {
            self.retry_store
                .insert(self.retry_key, self.retryable.clone());
        } else if is_done {
            self.retry_store.remove(&self.retry_key);
        }

        let markup = is_failed.then(|| {
            InlineKeyboardMarkup::new(vec![vec![InlineKeyboardButton::callback(
                "🔄 Retry",
                format!("retry:{}", self.retry_key),
            )]])
        });

        let mut edited = false;
        for attempt in 0..retries {
            if attempt > 0 {
                let backoff = Duration::from_millis(NOTIFY_RETRY_BASE_MS * 2u64.pow(attempt - 1));
                tokio::time::sleep(backoff).await;
            }
            let req = self
                .bot
                .edit_message_text(self.chat_id, self.sent_msg_id, &text);
            let result = if let Some(ref m) = markup {
                req.reply_markup(m.clone()).await
            } else {
                req.await
            };
            match result {
                Ok(_) => {
                    edited = true;
                    break;
                }
                Err(e) => {
                    warn!(
                        ?e,
                        attempt = attempt + 1,
                        retries,
                        "Failed to edit Telegram status message"
                    );
                }
            }
        }

        if !edited {
            warn!("All edit retries exhausted, falling back to send_message");
            match self.bot.send_message(self.chat_id, &text).await {
                Ok(sent) => {
                    self.sent_msg_id = sent.id;
                }
                Err(e) => {
                    warn!(?e, "Fallback send_message also failed");
                }
            }
        }
    }
}

/// Send an initial "⏳ Processing…" reply and return the sent message ID.
///
/// Retries up to `MAX_NOTIFY_RETRIES` times with exponential backoff on transient failures.
/// Returns `None` if all attempts fail (e.g. bot lacks permission).
pub async fn send_status_reply(
    bot: &teloxide::Bot,
    chat_id: teloxide::types::ChatId,
    reply_to: Option<teloxide::types::MessageId>,
) -> Option<teloxide::types::MessageId> {
    for attempt in 0..MAX_NOTIFY_RETRIES {
        if attempt > 0 {
            let backoff = Duration::from_millis(NOTIFY_RETRY_BASE_MS * 2u64.pow(attempt - 1));
            tokio::time::sleep(backoff).await;
        }
        let req = bot.send_message(chat_id, "⏳ Processing…");
        let req = if let Some(id) = reply_to {
            req.reply_parameters(teloxide::types::ReplyParameters::new(id))
        } else {
            req
        };
        match req.await {
            Ok(sent) => return Some(sent.id),
            Err(e) => {
                warn!(
                    ?e,
                    attempt = attempt + 1,
                    MAX_NOTIFY_RETRIES,
                    "Failed to send initial Telegram status message"
                );
            }
        }
    }
    None
}

/// Send an initial "⏳ Processing…" reply and return a fully-initialised notifier.
///
/// Retries sending up to `MAX_NOTIFY_RETRIES` times. Returns `None` if all attempts fail.
pub async fn build_telegram_notifier(
    bot: &teloxide::Bot,
    chat_id: teloxide::types::ChatId,
    reply_to: Option<teloxide::types::MessageId>,
    retry_store: Arc<DashMap<Uuid, RetryableMessage>>,
    retry_key: Uuid,
    retryable: RetryableMessage,
) -> Option<TelegramNotifier> {
    let sent_msg_id = send_status_reply(bot, chat_id, reply_to).await?;
    Some(TelegramNotifier::new(
        bot.clone(),
        chat_id,
        sent_msg_id,
        retry_store,
        retry_key,
        retryable,
    ))
}

#[cfg(test)]
mod tests;
