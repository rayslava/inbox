use std::fmt::Write;
use std::path::PathBuf;

use anodized::spec;
use teloxide::net::Download;
use teloxide::prelude::Requester;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use crate::config::TelegramConfig;
use crate::error::InboxError;
use crate::message::{IncomingMessage, MessageSource, SourceMetadata};

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

        let bot = Bot::new(&self.cfg.bot_token);
        info!("Telegram adapter starting");

        let handler = build_handler(
            self.cfg.allowed_user_ids.clone(),
            self.attachments_dir.clone(),
        );

        let mut dispatcher = Dispatcher::builder(bot, handler)
            .dependencies(dptree::deps![tx])
            .enable_ctrlc_handler()
            .build();

        tokio::select! {
            () = shutdown.cancelled() => { info!("Telegram adapter shutdown"); }
            () = dispatcher.dispatch() => {}
        }

        Ok(())
    }
}

/// Build the teloxide update handler tree. Exposed for integration tests.
#[must_use]
pub fn build_handler(
    allowed_user_ids: Vec<i64>,
    attachments_dir: PathBuf,
) -> teloxide::dispatching::UpdateHandler<teloxide::RequestError> {
    use teloxide::prelude::*;

    Update::filter_message().endpoint(
        move |bot: Bot, msg: Message, tx: mpsc::Sender<IncomingMessage>| {
            let allowed = allowed_user_ids.clone();
            let attachments_dir = attachments_dir.clone();
            async move {
                let user_id = msg
                    .from
                    .as_ref()
                    .map_or(0_i64, |u| i64::try_from(u.id.0).unwrap_or(i64::MAX));

                if !allowed.is_empty() && !allowed.contains(&user_id) {
                    return Ok(());
                }

                let username = msg.from.as_ref().and_then(|u| u.username.clone());
                let chat_id = msg.chat.id.0;
                let message_id = msg.id.0;

                let (text, attachments) =
                    extract_message_content(&bot, &msg, &attachments_dir).await;

                let mut incoming = IncomingMessage::new(
                    MessageSource::Telegram,
                    text,
                    SourceMetadata::Telegram {
                        chat_id,
                        message_id,
                        username,
                    },
                );
                incoming.attachments = attachments;

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
    )
}

async fn extract_message_content(
    bot: &teloxide::Bot,
    msg: &teloxide::types::Message,
    attachments_dir: &std::path::Path,
) -> (String, Vec<crate::message::Attachment>) {
    use crate::message::MediaKind;

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

    // File to download: (file_id_str, filename, media_kind)
    let file_info: Option<(String, String, MediaKind)> = if let Some(doc) = msg.document() {
        let name = sanitize_filename(&doc.file_name.clone().unwrap_or_else(|| "document".into()));
        Some((doc.file.id.to_string(), name, MediaKind::Document))
    } else if let Some(photos) = msg.photo() {
        photos
            .last()
            .map(|p| (p.file.id.to_string(), "photo.jpg".into(), MediaKind::Image))
    } else if let Some(audio) = msg.audio() {
        let name = audio
            .file_name
            .clone()
            .unwrap_or_else(|| "audio.mp3".into());
        Some((
            audio.file.id.to_string(),
            sanitize_filename(&name),
            MediaKind::Audio,
        ))
    } else if let Some(voice) = msg.voice() {
        Some((
            voice.file.id.to_string(),
            "voice.ogg".into(),
            MediaKind::VoiceMessage,
        ))
    } else if let Some(video) = msg.video() {
        let name = video
            .file_name
            .clone()
            .unwrap_or_else(|| "video.mp4".into());
        Some((
            video.file.id.to_string(),
            sanitize_filename(&name),
            MediaKind::Video,
        ))
    } else if let Some(sticker) = msg.sticker() {
        Some((
            sticker.file.id.to_string(),
            "sticker.webp".into(),
            MediaKind::Sticker,
        ))
    } else if let Some(anim) = msg.animation() {
        let name = anim
            .file_name
            .clone()
            .unwrap_or_else(|| "animation.mp4".into());
        Some((
            anim.file.id.to_string(),
            sanitize_filename(&name),
            MediaKind::Animation,
        ))
    } else {
        None
    };

    if let Some((file_id, filename, media_kind)) = file_info {
        match download_telegram_file(bot, &file_id, &filename, attachments_dir, media_kind).await {
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

    bot.download_file(&file.path, &mut dst)
        .await
        .map_err(|e| InboxError::Adapter(format!("Telegram download error: {e}")))?;

    let mime = mime_guess::from_path(&save_path)
        .first_raw()
        .map(str::to_owned);

    Ok(crate::message::Attachment {
        original_name: filename.to_owned(),
        saved_path: save_path,
        mime_type: mime,
        media_kind,
    })
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
