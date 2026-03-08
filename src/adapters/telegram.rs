use std::fmt::Write;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anodized::spec;
use dashmap::DashMap;
use teloxide::net::Download;
use teloxide::prelude::Requester;
use tokio::io::{AsyncSeekExt, SeekFrom};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};
use uuid::Uuid;

const RECONNECT_BACKOFF_INIT: Duration = Duration::from_secs(1);
const RECONNECT_BACKOFF_MAX: Duration = Duration::from_secs(60);
const STABLE_THRESHOLD: Duration = Duration::from_secs(30);

const FILE_DOWNLOAD_RETRY_BASE_MS: u64 = 1_000;

use crate::adapters::telegram_notifier::{build_telegram_notifier, send_status_reply};
use crate::config::TelegramConfig;
use crate::error::InboxError;
use crate::message::{IncomingMessage, MessageSource, RetryableMessage, SourceMetadata};
use crate::processing_status::StatusNotifier;

use super::InputAdapter;

pub struct TelegramAdapter {
    pub cfg: TelegramConfig,
    pub attachments_dir: std::path::PathBuf,
}

#[async_trait::async_trait]
impl InputAdapter for TelegramAdapter {
    fn name(&self) -> &'static str {
        "telegram"
    }

    async fn run(
        self: Box<Self>,
        tx: mpsc::Sender<IncomingMessage>,
        shutdown: CancellationToken,
    ) -> Result<(), InboxError> {
        use teloxide::prelude::*;

        if self.cfg.bot_token.is_empty() {
            return Err(InboxError::Adapter("Telegram bot_token is empty".into()));
        }

        info!("Telegram adapter starting");

        let retry_store: Arc<DashMap<Uuid, RetryableMessage>> = Arc::new(DashMap::new());
        let mut backoff = RECONNECT_BACKOFF_INIT;

        loop {
            let bot = Bot::new(&self.cfg.bot_token);
            let handler = build_handler(
                self.cfg.allowed_user_ids.clone(),
                self.attachments_dir.clone(),
                self.cfg.file_download_timeout_secs,
                self.cfg.file_download_retries,
                retry_store.clone(),
            );

            // Note: no enable_ctrlc_handler() — shutdown is handled via CancellationToken.
            let mut dispatcher = Dispatcher::builder(bot, handler)
                .dependencies(dptree::deps![tx.clone()])
                .build();

            let started = Instant::now();

            let mut dispatch_task = tokio::task::spawn(async move {
                dispatcher.dispatch().await;
            });

            tokio::select! {
                () = shutdown.cancelled() => {
                    dispatch_task.abort();
                    info!("Telegram adapter shutdown");
                    return Ok(());
                }
                result = &mut dispatch_task => {
                    if let Err(ref e) = result {
                        warn!(panic = ?e, "Telegram dispatcher panicked, will reconnect");
                    }
                    // fall through to reconnect logic in either case
                }
            }

            if shutdown.is_cancelled() {
                return Ok(());
            }

            // A long-lived session indicates a clean environment; reset backoff.
            if started.elapsed() >= STABLE_THRESHOLD {
                backoff = RECONNECT_BACKOFF_INIT;
            }

            metrics::counter!(
                crate::telemetry::ADAPTER_RECONNECTS,
                "adapter" => "telegram"
            )
            .increment(1);

            warn!(
                delay_secs = backoff.as_secs(),
                "Telegram dispatcher exited unexpectedly, reconnecting"
            );

            tokio::select! {
                () = shutdown.cancelled() => return Ok(()),
                () = tokio::time::sleep(backoff) => {}
            }

            backoff = (backoff * 2).min(RECONNECT_BACKOFF_MAX);
        }
    }
}

/// Build the teloxide update handler tree. Exposed for integration tests.
#[must_use]
pub fn build_handler(
    allowed_user_ids: Vec<i64>,
    attachments_dir: PathBuf,
    file_download_timeout_secs: u64,
    file_download_retries: u32,
    retry_store: Arc<DashMap<Uuid, RetryableMessage>>,
) -> teloxide::dispatching::UpdateHandler<teloxide::RequestError> {
    use teloxide::prelude::*;

    let retry_store_msg = retry_store.clone();
    let retry_store_cb = retry_store;

    let message_handler = Update::filter_message().endpoint(
        move |bot: Bot, msg: Message, tx: mpsc::Sender<IncomingMessage>| {
            let allowed = allowed_user_ids.clone();
            let attachments_dir = attachments_dir.clone();
            let retry_store = retry_store_msg.clone();
            async move {
                let user_id = msg
                    .from
                    .as_ref()
                    .map_or(0_i64, |u| i64::try_from(u.id.0).unwrap_or(i64::MAX));

                if !allowed.is_empty() && !allowed.contains(&user_id) {
                    return Ok(());
                }

                let username = msg.from.as_ref().and_then(|u| u.username.clone());
                let chat_id_i64 = msg.chat.id.0;
                let message_id = msg.id.0;
                let forwarded_from = extract_forward_origin(&msg);

                // Send "⏳ Processing…" immediately so the user sees feedback before download.
                let sent_id = send_status_reply(&bot, msg.chat.id, Some(msg.id)).await;

                // Download attached files (may block; user sees "⏳" during this).
                let (text, attachments) = extract_message_content(
                    &bot,
                    &msg,
                    &attachments_dir,
                    file_download_timeout_secs,
                    file_download_retries,
                )
                .await;

                let mut incoming = IncomingMessage::new(
                    MessageSource::Telegram,
                    text,
                    SourceMetadata::Telegram {
                        chat_id: chat_id_i64,
                        message_id,
                        username,
                        forwarded_from,
                    },
                );
                incoming.attachments = attachments;

                // Build notifier with retry context (references the already-sent status message).
                if let Some(sent_msg_id) = sent_id {
                    let retry_key = incoming.id;
                    let retryable = RetryableMessage::from(&incoming);
                    incoming.status_notifier = Some(Box::new(
                        crate::adapters::telegram_notifier::TelegramNotifier::new(
                            bot.clone(),
                            msg.chat.id,
                            sent_msg_id,
                            retry_store,
                            retry_key,
                            retryable,
                        ),
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

                Ok::<_, teloxide::RequestError>(())
            }
        },
    );

    let callback_handler = Update::filter_callback_query().endpoint(
        move |bot: Bot,
              query: teloxide::types::CallbackQuery,
              tx: mpsc::Sender<IncomingMessage>| {
            let retry_store = retry_store_cb.clone();
            async move { handle_callback_query(bot, query, tx, retry_store).await }
        },
    );

    dptree::entry()
        .branch(message_handler)
        .branch(callback_handler)
}

async fn handle_callback_query(
    bot: teloxide::Bot,
    query: teloxide::types::CallbackQuery,
    tx: mpsc::Sender<IncomingMessage>,
    retry_store: Arc<DashMap<Uuid, RetryableMessage>>,
) -> Result<(), teloxide::RequestError> {
    use teloxide::prelude::Requester;

    // Destructure query before any moves so all fields remain accessible.
    let teloxide::types::CallbackQuery {
        id: callback_id,
        data,
        message,
        ..
    } = query;

    // Acknowledge the button press immediately (removes the spinner in Telegram).
    bot.answer_callback_query(callback_id).await?;

    let data_str = data.as_deref().unwrap_or("");
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
    let notifier = build_telegram_notifier(
        &bot,
        chat_id,
        reply_to,
        retry_store,
        retry_key,
        fresh_retryable,
    )
    .await;
    msg.status_notifier = notifier.map(|n| Box::new(n) as Box<dyn StatusNotifier>);

    if let Err(e) = tx.send(msg).await {
        warn!(?e, "Failed to enqueue retried Telegram message");
    }

    Ok(())
}

/// Return `(file_id, filename, media_kind)` for the first downloadable file in the message.
fn classify_message_file(
    msg: &teloxide::types::Message,
) -> Option<(String, String, crate::message::MediaKind)> {
    use crate::message::MediaKind;

    if let Some(doc) = msg.document() {
        let name = sanitize_filename(&doc.file_name.clone().unwrap_or_else(|| "document".into()));
        return Some((doc.file.id.to_string(), name, MediaKind::Document));
    }
    if let Some(photos) = msg.photo() {
        return photos
            .last()
            .map(|p| (p.file.id.to_string(), "photo.jpg".into(), MediaKind::Image));
    }
    if let Some(audio) = msg.audio() {
        let name = sanitize_filename(
            &audio
                .file_name
                .clone()
                .unwrap_or_else(|| "audio.mp3".into()),
        );
        return Some((audio.file.id.to_string(), name, MediaKind::Audio));
    }
    if let Some(voice) = msg.voice() {
        return Some((
            voice.file.id.to_string(),
            "voice.ogg".into(),
            MediaKind::VoiceMessage,
        ));
    }
    if let Some(video) = msg.video() {
        let name = sanitize_filename(
            &video
                .file_name
                .clone()
                .unwrap_or_else(|| "video.mp4".into()),
        );
        return Some((video.file.id.to_string(), name, MediaKind::Video));
    }
    if let Some(sticker) = msg.sticker() {
        return Some((
            sticker.file.id.to_string(),
            "sticker.webp".into(),
            MediaKind::Sticker,
        ));
    }
    if let Some(anim) = msg.animation() {
        let name = sanitize_filename(
            &anim
                .file_name
                .clone()
                .unwrap_or_else(|| "animation.mp4".into()),
        );
        return Some((anim.file.id.to_string(), name, MediaKind::Animation));
    }
    None
}

async fn extract_message_content(
    bot: &teloxide::Bot,
    msg: &teloxide::types::Message,
    attachments_dir: &std::path::Path,
    file_download_timeout_secs: u64,
    file_download_retries: u32,
) -> (String, Vec<crate::message::Attachment>) {
    let mut text = msg
        .text()
        .or_else(|| msg.caption())
        .unwrap_or("")
        .to_owned();
    let mut attachments = Vec::new();

    if let Some(loc) = msg.location() {
        text = format!("📍 {},{}", loc.latitude, loc.longitude);
    }

    if let Some(contact) = msg.contact() {
        text = format!(
            "👤 {} {} {}",
            contact.first_name,
            contact.last_name.as_deref().unwrap_or(""),
            contact.phone_number
        );
    }

    if let Some(poll) = msg.poll() {
        let options = poll
            .options
            .iter()
            .map(|o| o.text.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        text = format!("📊 {} [{}]", poll.question, options);
    }

    let file_info = classify_message_file(msg);

    if let Some((file_id, filename, media_kind)) = file_info {
        match download_telegram_file(
            bot,
            &file_id,
            &filename,
            attachments_dir,
            media_kind,
            file_download_timeout_secs,
            file_download_retries,
        )
        .await
        {
            Ok(att) => attachments.push(att),
            Err(e) => {
                warn!(?e, "Failed to download Telegram file");
                let _ = write!(text, "\n[attachment skipped: {e}]");
            }
        }
    }

    (text, attachments)
}

async fn download_telegram_file(
    bot: &teloxide::Bot,
    file_id: &str,
    filename: &str,
    attachments_dir: &std::path::Path,
    media_kind: crate::message::MediaKind,
    timeout_secs: u64,
    retries: u32,
) -> Result<crate::message::Attachment, InboxError> {
    #[spec(requires: !file_id.is_empty() && !filename.is_empty())]
    fn validate_input(file_id: &str, filename: &str) {
        let _ = (file_id, filename);
    }
    validate_input(file_id, filename);

    let file = bot
        .get_file(teloxide::types::FileId(file_id.to_owned()))
        .await
        .map_err(|e| InboxError::Adapter(format!("Telegram getFile error: {e}")))?;

    let id = uuid::Uuid::new_v4();
    let save_path =
        crate::pipeline::url_fetcher::attachment_save_path(attachments_dir, id, filename);

    if let Some(parent) = save_path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(InboxError::Io)?;
    }

    let mut dst = tokio::fs::File::create(&save_path)
        .await
        .map_err(InboxError::Io)?;

    let timeout = Duration::from_secs(timeout_secs);

    for attempt in 0..retries {
        if attempt > 0 {
            let backoff =
                Duration::from_millis(FILE_DOWNLOAD_RETRY_BASE_MS * 2u64.pow(attempt - 1));
            tokio::time::sleep(backoff).await;
            // Truncate file so partial data from previous attempt is discarded.
            dst.set_len(0).await.map_err(InboxError::Io)?;
            dst.seek(SeekFrom::Start(0)).await.map_err(InboxError::Io)?;
        }

        match tokio::time::timeout(timeout, bot.download_file(&file.path, &mut dst)).await {
            Ok(Ok(())) => {
                let mime = mime_guess::from_path(&save_path)
                    .first_raw()
                    .map(str::to_owned);
                return Ok(crate::message::Attachment {
                    original_name: filename.to_owned(),
                    saved_path: save_path,
                    mime_type: mime,
                    media_kind,
                });
            }
            Ok(Err(e)) => {
                warn!(
                    ?e,
                    attempt = attempt + 1,
                    retries,
                    "Telegram file download attempt failed"
                );
            }
            Err(_) => {
                warn!(
                    attempt = attempt + 1,
                    retries, timeout_secs, "Telegram file download attempt timed out"
                );
            }
        }
    }

    Err(InboxError::Adapter(
        "all Telegram file download attempts failed".into(),
    ))
}

/// Extract a human-readable display name from a forwarded message's origin, if any.
fn extract_forward_origin(msg: &teloxide::types::Message) -> Option<String> {
    use teloxide::types::MessageOrigin;

    match msg.forward_origin()? {
        MessageOrigin::User { sender_user, .. } => Some(
            sender_user
                .username
                .as_ref()
                .map_or_else(|| sender_user.full_name(), |u| format!("@{u}")),
        ),
        MessageOrigin::HiddenUser {
            sender_user_name, ..
        } => Some(sender_user_name.clone()),
        MessageOrigin::Chat { sender_chat, .. } => Some(sender_chat.username().map_or_else(
            || sender_chat.title().unwrap_or("unknown").to_owned(),
            |u| format!("@{u}"),
        )),
        MessageOrigin::Channel { chat, .. } => Some(chat.username().map_or_else(
            || chat.title().unwrap_or("unknown").to_owned(),
            |u| format!("@{u}"),
        )),
    }
}

#[must_use]
#[spec(requires: !name.is_empty())]
fn sanitize_filename(name: &str) -> String {
    let basename = std::path::Path::new(name)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("attachment");

    let cleaned: String = basename
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || matches!(c, '.' | '-' | '_') {
                c
            } else {
                '_'
            }
        })
        .collect();

    if cleaned.is_empty() {
        "attachment".to_owned()
    } else {
        cleaned
    }
}
