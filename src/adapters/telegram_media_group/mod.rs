use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use dashmap::DashMap;
use tokio::sync::mpsc;
use tracing::{info, warn};
use uuid::Uuid;

use crate::adapters::telegram_notifier;
use crate::message::{
    Attachment, IncomingMessage, MessageSource, RetryableMessage, SourceMetadata,
};
use crate::processing_status::StatusNotifier;

pub(crate) struct MediaGroupInner {
    pub msg_id: Uuid,
    pub text: String,
    pub attachments: Vec<Attachment>,
    pub chat_id: teloxide::types::ChatId,
    pub first_message_id: i32,
    pub username: Option<String>,
    pub forwarded_from: Option<String>,
    pub sent_status_id: Option<teloxide::types::MessageId>,
}

pub(crate) struct MediaGroupState {
    pub inner: Mutex<MediaGroupInner>,
    pub pending_downloads: AtomicU32,
}

pub(crate) type MediaGroupMap = Arc<DashMap<String, Arc<MediaGroupState>>>;

pub(crate) fn new_map() -> MediaGroupMap {
    Arc::new(DashMap::new())
}

/// Get or create a media group state. Returns `(state, is_new)`.
pub(crate) fn get_or_create(
    groups: &MediaGroupMap,
    group_id: &str,
) -> (Arc<MediaGroupState>, bool) {
    use dashmap::mapref::entry::Entry;
    match groups.entry(group_id.to_owned()) {
        Entry::Occupied(e) => (Arc::clone(e.get()), false),
        Entry::Vacant(e) => {
            let state = Arc::new(MediaGroupState {
                inner: Mutex::new(MediaGroupInner {
                    msg_id: Uuid::new_v4(),
                    text: String::new(),
                    attachments: Vec::new(),
                    chat_id: teloxide::types::ChatId(0),
                    first_message_id: 0,
                    username: None,
                    forwarded_from: None,
                    sent_status_id: None,
                }),
                pending_downloads: AtomicU32::new(0),
            });
            e.insert(Arc::clone(&state));
            (state, true)
        }
    }
}

/// Set first-part metadata on a newly created group.
pub(crate) fn set_metadata(
    state: &MediaGroupState,
    chat_id: teloxide::types::ChatId,
    message_id: i32,
    username: Option<String>,
    forwarded_from: Option<String>,
    sent_status_id: Option<teloxide::types::MessageId>,
) {
    let mut inner = state.inner.lock().expect("media group mutex poisoned");
    inner.chat_id = chat_id;
    inner.first_message_id = message_id;
    inner.username = username;
    inner.forwarded_from = forwarded_from;
    inner.sent_status_id = sent_status_id;
}

/// Append extracted content from one part of the media group.
pub(crate) fn add_content(state: &MediaGroupState, text: String, attachments: Vec<Attachment>) {
    let mut inner = state.inner.lock().expect("media group mutex poisoned");
    if !text.is_empty() {
        if inner.text.is_empty() {
            inner.text = text;
        } else {
            inner.text.push('\n');
            inner.text.push_str(&text);
        }
    }
    inner.attachments.extend(attachments);
}

/// Groups the bot handle, retry store and notify config needed when flushing a media group.
pub(crate) struct FlushContext {
    pub bot: teloxide::Bot,
    pub retry_store: Arc<DashMap<Uuid, RetryableMessage>>,
    pub notify_cfg: crate::adapters::telegram_notifier::NotifyConfig,
}

/// Spawn the delayed flush task for a media group.
pub(crate) fn spawn_flush(
    groups: MediaGroupMap,
    group_id: String,
    state: Arc<MediaGroupState>,
    timeout: Duration,
    tx: mpsc::Sender<IncomingMessage>,
    ctx: FlushContext,
) {
    tokio::spawn(async move {
        tokio::time::sleep(timeout).await;

        // Wait for in-flight downloads to finish (poll with short intervals).
        let deadline = tokio::time::Instant::now() + Duration::from_secs(300);
        while state.pending_downloads.load(Ordering::Acquire) > 0 {
            if tokio::time::Instant::now() >= deadline {
                warn!(
                    media_group_id = group_id,
                    "Timed out waiting for media group downloads"
                );
                break;
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }

        // Remove from map so late arrivals become standalone messages.
        groups.remove(&group_id);

        flush(state, &group_id, tx, ctx).await;
    });
}

#[cfg(test)]
mod tests;

async fn flush(
    state: Arc<MediaGroupState>,
    group_id: &str,
    tx: mpsc::Sender<IncomingMessage>,
    ctx: FlushContext,
) {
    let FlushContext {
        bot,
        retry_store,
        notify_cfg,
    } = ctx;
    let (incoming_base, sent_status_id, chat_id) = {
        let inner = state.inner.lock().expect("media group mutex poisoned");
        let n_attachments = inner.attachments.len();
        info!(
            media_group_id = group_id,
            attachments = n_attachments,
            "Flushing media group"
        );

        let mut incoming = IncomingMessage::with_id(
            inner.msg_id,
            MessageSource::Telegram,
            inner.text.clone(),
            SourceMetadata::Telegram {
                chat_id: inner.chat_id.0,
                message_id: inner.first_message_id,
                username: inner.username.clone(),
                forwarded_from: inner.forwarded_from.clone(),
            },
        );
        incoming.attachments.clone_from(&inner.attachments);

        let sent = inner.sent_status_id;
        let cid = inner.chat_id;
        (incoming, sent, cid)
    };

    let mut incoming = incoming_base;

    if let Some(sent_msg_id) = sent_status_id {
        let retry_key = incoming.id;
        let retryable = RetryableMessage::from(&incoming);
        incoming.status_notifier = Some(Box::new(telegram_notifier::TelegramNotifier::new(
            bot,
            chat_id,
            sent_msg_id,
            retry_store,
            retry_key,
            retryable,
            notify_cfg,
        )) as Box<dyn StatusNotifier>);
    }

    metrics::counter!(
        crate::telemetry::MESSAGES_RECEIVED,
        "source" => "telegram"
    )
    .increment(1);

    if let Err(e) = tx.send(incoming).await {
        warn!(?e, "Failed to enqueue Telegram media group message");
    }
}
