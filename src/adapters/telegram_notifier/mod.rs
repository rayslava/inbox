use std::time::Duration;

use teloxide::payloads::SendMessageSetters;
use teloxide::prelude::Requester;
use tracing::warn;

use crate::processing_status::{ProcessingStage, StatusNotifier};

const MAX_NOTIFY_RETRIES: u32 = 3;
const TERMINAL_NOTIFY_RETRIES: u32 = 5;
const NOTIFY_RETRY_BASE_MS: u64 = 500;

pub struct TelegramNotifier {
    bot: teloxide::Bot,
    chat_id: teloxide::types::ChatId,
    sent_msg_id: teloxide::types::MessageId,
}

impl TelegramNotifier {
    #[must_use]
    pub fn new(
        bot: teloxide::Bot,
        chat_id: teloxide::types::ChatId,
        sent_msg_id: teloxide::types::MessageId,
    ) -> Self {
        Self {
            bot,
            chat_id,
            sent_msg_id,
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

        let mut edited = false;
        for attempt in 0..retries {
            if attempt > 0 {
                let backoff = Duration::from_millis(NOTIFY_RETRY_BASE_MS * 2u64.pow(attempt - 1));
                tokio::time::sleep(backoff).await;
            }
            match self
                .bot
                .edit_message_text(self.chat_id, self.sent_msg_id, &text)
                .await
            {
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

/// Send an initial "⏳ Processing…" reply and return a notifier for the sent message.
///
/// Returns `None` if the initial send fails (e.g. bot lacks reply permission).
pub async fn build_telegram_notifier(
    bot: &teloxide::Bot,
    chat_id: teloxide::types::ChatId,
    reply_to: teloxide::types::MessageId,
) -> Option<TelegramNotifier> {
    match bot
        .send_message(chat_id, "⏳ Processing…")
        .reply_parameters(teloxide::types::ReplyParameters::new(reply_to))
        .await
    {
        Ok(sent) => Some(TelegramNotifier::new(bot.clone(), chat_id, sent.id)),
        Err(e) => {
            warn!(?e, "Failed to send initial Telegram status message");
            None
        }
    }
}

#[cfg(test)]
mod tests;
