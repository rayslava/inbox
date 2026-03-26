use std::fmt::Write;
use std::time::Duration;

use anodized::spec;
use teloxide::net::Download;
use teloxide::prelude::Requester;
use tokio::io::{AsyncSeekExt, SeekFrom};
use tracing::warn;
use uuid::Uuid;

use crate::error::InboxError;

pub(super) const FILE_DOWNLOAD_RETRY_BASE_MS: u64 = 1_000;

pub(super) struct DownloadConfig {
    pub(super) timeout_secs: u64,
    pub(super) retries: u32,
}

pub(super) fn classify_message_file(
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

pub(super) async fn extract_message_content(
    bot: &teloxide::Bot,
    msg: &teloxide::types::Message,
    attachments_dir: &std::path::Path,
    dl_cfg: &DownloadConfig,
    msg_id: Uuid,
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
            dl_cfg,
            msg_id,
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
    dl_cfg: &DownloadConfig,
    msg_id: Uuid,
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

    let id = msg_id;
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

    let timeout = Duration::from_secs(dl_cfg.timeout_secs);

    for attempt in 0..dl_cfg.retries {
        if attempt > 0 {
            let backoff =
                Duration::from_millis(FILE_DOWNLOAD_RETRY_BASE_MS * 2u64.pow(attempt - 1));
            tokio::time::sleep(backoff).await;
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
                    retries = dl_cfg.retries,
                    "Telegram file download attempt failed"
                );
            }
            Err(_) => {
                warn!(
                    attempt = attempt + 1,
                    retries = dl_cfg.retries,
                    timeout_secs = dl_cfg.timeout_secs,
                    "Telegram file download attempt timed out"
                );
            }
        }
    }

    Err(InboxError::Adapter(
        "all Telegram file download attempts failed".into(),
    ))
}

pub(super) fn extract_forward_origin(msg: &teloxide::types::Message) -> Option<String> {
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
