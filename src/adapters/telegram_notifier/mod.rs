use std::sync::Arc;
use std::time::Duration;

use dashmap::DashMap;
use teloxide::payloads::{EditMessageTextSetters, SendMessageSetters};
use teloxide::prelude::Requester;
use teloxide::types::{InlineKeyboardButton, InlineKeyboardMarkup};
use tracing::warn;
use uuid::Uuid;

use crate::adapters::telegram::FeedbackMessageMap;
use crate::message::RetryableMessage;
use crate::processing_status::{ProcessingStage, StatusNotifier};

/// Hard upper bound on retry delay regardless of base and attempt count.
const MAX_BACKOFF_MS: u64 = 30_000;

/// Retry policy for status notification calls.
#[derive(Clone, Copy)]
pub struct NotifyConfig {
    pub retries: u32,
    pub retry_base_ms: u64,
}

pub struct TelegramNotifier {
    bot: teloxide::Bot,
    chat_id: teloxide::types::ChatId,
    sent_msg_id: teloxide::types::MessageId,
    retry_store: Arc<DashMap<Uuid, RetryableMessage>>,
    retry_key: Uuid,
    retryable: RetryableMessage,
    /// Retry policy. Terminal stages (Done/Failed) use `cfg.retries * 2`.
    cfg: NotifyConfig,
    feedback_msg_map: Option<FeedbackMessageMap>,
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
        cfg: NotifyConfig,
    ) -> Self {
        Self {
            bot,
            chat_id,
            sent_msg_id,
            retry_store,
            retry_key,
            retryable,
            cfg,
            feedback_msg_map: None,
        }
    }

    #[must_use]
    pub fn with_feedback_map(mut self, map: FeedbackMessageMap) -> Self {
        self.feedback_msg_map = Some(map);
        self
    }
}

pub(super) fn stage_text(stage: &ProcessingStage) -> String {
    match stage {
        ProcessingStage::Received => "⏳ Processing…".to_owned(),
        ProcessingStage::Enriching => "🔍 Fetching content…".to_owned(),
        ProcessingStage::RunningLlm {
            turn,
            max_turns,
            last_tools,
        } => {
            if *turn == 0 || last_tools.is_empty() {
                "🤖 Analysing…".to_owned()
            } else {
                let tools_str = last_tools.join(", ");
                format!("🤖 Analysing… (turn {turn}/{max_turns} · {tools_str})")
            }
        }
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
            self.cfg.retries * 2
        } else {
            self.cfg.retries
        };

        let is_failed = matches!(stage, ProcessingStage::Failed { .. });
        let is_done = matches!(stage, ProcessingStage::Done { .. });

        if is_failed {
            self.retry_store
                .insert(self.retry_key, self.retryable.clone());
        } else if is_done {
            self.retry_store.remove(&self.retry_key);
        }

        let markup = if is_failed {
            Some(InlineKeyboardMarkup::new(vec![vec![
                InlineKeyboardButton::callback(
                    "\u{1f504} Retry",
                    format!("retry:{}", self.retry_key),
                ),
            ]]))
        } else if is_done {
            let key = self.retry_key;
            Some(InlineKeyboardMarkup::new(vec![vec![
                InlineKeyboardButton::callback("\u{2b50}", format!("fb:1:{key}")),
                InlineKeyboardButton::callback("\u{2b50}\u{2b50}", format!("fb:2:{key}")),
                InlineKeyboardButton::callback("\u{2b50}\u{2b50}\u{2b50}", format!("fb:3:{key}")),
            ]]))
        } else {
            None
        };

        let mut edited = false;
        for attempt in 0..retries {
            if attempt > 0 {
                let delay_ms = (self.cfg.retry_base_ms * 2u64.pow(attempt - 1)).min(MAX_BACKOFF_MS);
                tokio::time::sleep(Duration::from_millis(delay_ms)).await;
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

        // On Done, register the bot message ID for reply-as-comment.
        if is_done {
            if let Some(ref map) = self.feedback_msg_map {
                map.insert(self.sent_msg_id.0, self.retry_key);
            }
        }
    }

    fn telegram_status_msg_id(&self) -> Option<i32> {
        Some(self.sent_msg_id.0)
    }
}

/// Send an initial "⏳ Processing…" reply and return the sent message ID.
///
/// Retries up to `retries` times with exponential backoff (capped at 30 s) on transient failures.
/// Returns `None` if all attempts fail (e.g. bot lacks permission).
pub async fn send_status_reply(
    bot: &teloxide::Bot,
    chat_id: teloxide::types::ChatId,
    reply_to: Option<teloxide::types::MessageId>,
    cfg: NotifyConfig,
) -> Option<teloxide::types::MessageId> {
    for attempt in 0..cfg.retries {
        if attempt > 0 {
            let delay_ms = (cfg.retry_base_ms * 2u64.pow(attempt - 1)).min(MAX_BACKOFF_MS);
            tokio::time::sleep(Duration::from_millis(delay_ms)).await;
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
                    retries = cfg.retries,
                    "Failed to send initial Telegram status message"
                );
            }
        }
    }
    None
}

/// Arguments for building a Telegram notifier.
pub struct BuildNotifierArgs {
    pub chat_id: teloxide::types::ChatId,
    pub reply_to: Option<teloxide::types::MessageId>,
    pub retry_store: Arc<DashMap<Uuid, RetryableMessage>>,
    pub retry_key: Uuid,
    pub retryable: RetryableMessage,
    pub cfg: NotifyConfig,
    pub feedback_msg_map: Option<FeedbackMessageMap>,
}

/// Send an initial "⏳ Processing…" reply and return a fully-initialised notifier.
///
/// Uses `cfg` for both the initial send and subsequent status edits.
/// Returns `None` if all send attempts fail.
pub async fn build_telegram_notifier(
    bot: &teloxide::Bot,
    args: BuildNotifierArgs,
) -> Option<TelegramNotifier> {
    let sent_msg_id = send_status_reply(bot, args.chat_id, args.reply_to, args.cfg).await?;
    let mut notifier = TelegramNotifier::new(
        bot.clone(),
        args.chat_id,
        sent_msg_id,
        args.retry_store,
        args.retry_key,
        args.retryable,
        args.cfg,
    );
    if let Some(map) = args.feedback_msg_map {
        notifier = notifier.with_feedback_map(map);
    }
    Some(notifier)
}

pub mod resume;

#[cfg(test)]
mod tests;
