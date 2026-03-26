use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use dashmap::DashMap;
use tokio::sync::mpsc;
use tracing::{info, warn};
use uuid::Uuid;

use crate::adapters::telegram_media_group::{self, MediaGroupMap};
use crate::memory::MemoryStore;
use crate::message::{IncomingMessage, MessageSource, RetryableMessage, SourceMetadata};
use crate::processing_status::StatusNotifier;

use super::FeedbackMessageMap;
use super::files::{DownloadConfig, extract_forward_origin, extract_message_content};

pub(super) struct MessageContext {
    pub(super) attachments_dir: PathBuf,
    pub(super) dl_cfg: DownloadConfig,
    pub(super) media_group_timeout_ms: u64,
    pub(super) notify_cfg: crate::adapters::telegram_notifier::NotifyConfig,
    pub(super) retry_store: Arc<DashMap<Uuid, RetryableMessage>>,
    pub(super) media_groups: MediaGroupMap,
    pub(super) memory_store: Option<Arc<MemoryStore>>,
    pub(super) feedback_msg_map: FeedbackMessageMap,
}

async fn try_reply_as_comment(msg: &teloxide::types::Message, ctx: &MessageContext) -> bool {
    let Some(reply) = msg.reply_to_message() else {
        return false;
    };
    let Some(store) = &ctx.memory_store else {
        return false;
    };
    let reply_msg_id = reply.id.0;
    let Some(entry) = ctx.feedback_msg_map.get(&reply_msg_id) else {
        return false;
    };
    let inbox_id = entry.value().to_string();
    let comment = msg.text().unwrap_or("").to_owned();
    if comment.is_empty() {
        return false;
    }
    let store = Arc::clone(store);
    match store.update_feedback_comment(&inbox_id, &comment).await {
        Ok(true) => info!(inbox_id, "Feedback comment added via reply"),
        Ok(false) => warn!(inbox_id, "No feedback found for reply-as-comment"),
        Err(e) => warn!(?e, inbox_id, "Failed to save feedback comment"),
    }
    true
}

pub(super) async fn handle_message(
    bot: teloxide::Bot,
    msg: teloxide::types::Message,
    tx: mpsc::Sender<IncomingMessage>,
    ctx: MessageContext,
) {
    if try_reply_as_comment(&msg, &ctx).await {
        return;
    }

    let username = msg.from.as_ref().and_then(|u| u.username.clone());
    let message_id = msg.id.0;
    let forwarded_from = extract_forward_origin(&msg);

    if let Some(group_id) = msg.media_group_id() {
        let group_id_str = group_id.0.clone();
        let (state, is_first) =
            telegram_media_group::get_or_create(&ctx.media_groups, &group_id_str);

        let msg_id = state
            .inner
            .lock()
            .expect("media group mutex poisoned")
            .msg_id;

        if is_first {
            let sent_id = crate::adapters::telegram_notifier::send_status_reply(
                &bot,
                msg.chat.id,
                Some(msg.id),
                ctx.notify_cfg,
            )
            .await;
            telegram_media_group::set_metadata(
                &state,
                msg.chat.id,
                message_id,
                username,
                forwarded_from,
                sent_id,
            );
            telegram_media_group::spawn_flush(
                ctx.media_groups,
                group_id_str.clone(),
                state.clone(),
                Duration::from_millis(ctx.media_group_timeout_ms),
                tx,
                telegram_media_group::FlushContext {
                    bot: bot.clone(),
                    retry_store: ctx.retry_store,
                    notify_cfg: ctx.notify_cfg,
                    feedback_msg_map: ctx.feedback_msg_map,
                },
            );
        }

        state.pending_downloads.fetch_add(1, Ordering::Release);
        let (text, attachments) =
            extract_message_content(&bot, &msg, &ctx.attachments_dir, &ctx.dl_cfg, msg_id).await;
        telegram_media_group::add_content(&state, text, attachments);
        state.pending_downloads.fetch_sub(1, Ordering::Release);

        return;
    }

    let sent_id = crate::adapters::telegram_notifier::send_status_reply(
        &bot,
        msg.chat.id,
        Some(msg.id),
        ctx.notify_cfg,
    )
    .await;
    let msg_id = Uuid::new_v4();

    let (text, attachments) =
        extract_message_content(&bot, &msg, &ctx.attachments_dir, &ctx.dl_cfg, msg_id).await;

    let mut incoming = IncomingMessage::with_id(
        msg_id,
        MessageSource::Telegram,
        text,
        SourceMetadata::Telegram {
            chat_id: msg.chat.id.0,
            message_id,
            username,
            forwarded_from,
        },
    );
    incoming.attachments = attachments;

    if let Some(sent_msg_id) = sent_id {
        let retry_key = incoming.id;
        let retryable = RetryableMessage::from(&incoming);
        incoming.status_notifier = Some(Box::new(
            crate::adapters::telegram_notifier::TelegramNotifier::new(
                bot.clone(),
                msg.chat.id,
                sent_msg_id,
                ctx.retry_store,
                retry_key,
                retryable,
                ctx.notify_cfg,
            )
            .with_feedback_map(ctx.feedback_msg_map),
        ));
    }

    metrics::counter!(
        crate::telemetry::MESSAGES_RECEIVED,
        "source" => "telegram"
    )
    .increment(1);

    if let Err(e) = tx.send(incoming).await {
        warn!(?e, "Failed to enqueue Telegram message");
    }
}

pub(super) async fn handle_callback_query(
    bot: teloxide::Bot,
    query: teloxide::types::CallbackQuery,
    tx: mpsc::Sender<IncomingMessage>,
    retry_store: Arc<DashMap<Uuid, RetryableMessage>>,
    notify_cfg: crate::adapters::telegram_notifier::NotifyConfig,
    memory_store: Option<Arc<MemoryStore>>,
    feedback_msg_map: FeedbackMessageMap,
) -> Result<(), teloxide::RequestError> {
    use teloxide::prelude::Requester;

    let teloxide::types::CallbackQuery {
        id: callback_id,
        data,
        message,
        ..
    } = query;

    let data_str = data.as_deref().unwrap_or("");

    if data_str.starts_with("fb:") {
        bot.answer_callback_query(callback_id).await?;
        handle_feedback_callback(
            &bot,
            data_str,
            message.as_ref(),
            memory_store.as_ref(),
            &feedback_msg_map,
        )
        .await;
        return Ok(());
    }

    bot.answer_callback_query(callback_id).await?;

    let Some(uuid_str) = data_str.strip_prefix("retry:") else {
        return Ok(());
    };
    let Ok(key) = Uuid::parse_str(uuid_str) else {
        return Ok(());
    };
    let Some((_, retryable)) = retry_store.remove(&key) else {
        return Ok(());
    };

    let (chat_id, reply_to) = match &message {
        Some(teloxide::types::MaybeInaccessibleMessage::Regular(m)) => {
            (Some(m.chat.id), Some(m.id))
        }
        Some(teloxide::types::MaybeInaccessibleMessage::Inaccessible(m)) => (Some(m.chat.id), None),
        None => (None, None),
    };

    let Some(chat_id) = chat_id else {
        return Ok(());
    };

    let mut msg = IncomingMessage::new(
        MessageSource::Telegram,
        retryable.text.clone(),
        retryable.metadata.clone(),
    );
    msg.attachments = retryable.attachments.clone();
    msg.user_tags = retryable.user_tags.clone();
    msg.preprocessing_hints = retryable.preprocessing_hints.clone();

    let retry_key = msg.id;
    let fresh_retryable = RetryableMessage::from(&msg);
    let notifier = crate::adapters::telegram_notifier::build_telegram_notifier(
        &bot,
        crate::adapters::telegram_notifier::BuildNotifierArgs {
            chat_id,
            reply_to,
            retry_store,
            retry_key,
            retryable: fresh_retryable,
            cfg: notify_cfg,
            feedback_msg_map: Some(feedback_msg_map.clone()),
        },
    )
    .await;
    msg.status_notifier = notifier.map(|n| Box::new(n) as Box<dyn StatusNotifier>);

    if let Err(e) = tx.send(msg).await {
        warn!(?e, "Failed to enqueue retried Telegram message");
    }

    Ok(())
}

pub(super) async fn handle_feedback_callback(
    bot: &teloxide::Bot,
    data_str: &str,
    message: Option<&teloxide::types::MaybeInaccessibleMessage>,
    memory_store: Option<&Arc<MemoryStore>>,
    feedback_msg_map: &FeedbackMessageMap,
) {
    use crate::feedback::{FeedbackEntry, FeedbackRating};
    use chrono::Utc;
    use teloxide::payloads::EditMessageTextSetters;
    use teloxide::prelude::Requester;

    let parts: Vec<&str> = data_str.splitn(3, ':').collect();
    if parts.len() < 3 {
        return;
    }
    let Ok(rating_val) = parts[1].parse::<u8>() else {
        return;
    };
    let Some(rating) = FeedbackRating::new(rating_val) else {
        return;
    };
    let Ok(inbox_id) = Uuid::parse_str(parts[2]) else {
        return;
    };

    let Some(store) = memory_store else {
        warn!("Feedback callback received but memory store is not enabled");
        return;
    };

    let entry = FeedbackEntry {
        message_id: inbox_id.to_string(),
        rating: rating.value(),
        comment: String::new(),
        created_at: Utc::now(),
        source: "telegram".into(),
        title: String::new(),
    };

    if let Err(e) = store.save_feedback(&entry).await {
        warn!(?e, "Failed to save Telegram feedback");
        return;
    }

    if let Some(teloxide::types::MaybeInaccessibleMessage::Regular(m)) = message {
        feedback_msg_map.insert(m.id.0, inbox_id);
    }

    let stars = rating.to_string();
    if let Some(teloxide::types::MaybeInaccessibleMessage::Regular(m)) = message {
        let text = format!("{}\n\nRated: {stars}", m.text().unwrap_or(""));
        if let Err(e) = bot
            .edit_message_text(m.chat.id, m.id, text)
            .reply_markup(teloxide::types::InlineKeyboardMarkup::default())
            .await
        {
            warn!(?e, "Failed to edit message after feedback");
        }
    }
}
