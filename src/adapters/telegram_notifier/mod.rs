use teloxide::payloads::SendMessageSetters;
use teloxide::prelude::Requester;
use tracing::warn;

use crate::processing_status::{ProcessingStage, StatusNotifier};

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

#[async_trait::async_trait]
impl StatusNotifier for TelegramNotifier {
    async fn advance(&mut self, stage: ProcessingStage) {
        let text = stage_text(&stage);
        if let Err(e) = self
            .bot
            .edit_message_text(self.chat_id, self.sent_msg_id, text)
            .await
        {
            warn!(?e, "Failed to update Telegram status message");
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
