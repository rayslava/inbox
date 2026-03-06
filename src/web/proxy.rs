use axum::{
    body::Bytes,
    extract::{Multipart, State},
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
};
use tracing::warn;

use crate::message::{Attachment, IncomingMessage, MediaKind, MessageSource, SourceMetadata};
use crate::pipeline::url_fetcher::attachment_save_path;

use super::AdminState;
use super::auth;

pub(crate) async fn inbox_handler(
    State(state): State<AdminState>,
    headers: HeaderMap,
    body: Bytes,
) -> impl IntoResponse {
    let admin = &state.cfg.admin;
    if !auth::is_authenticated(&headers, &state.sessions, admin.session_ttl_days) {
        return StatusCode::UNAUTHORIZED;
    }

    let Some(ref tx) = state.inbox_tx else {
        return StatusCode::SERVICE_UNAVAILABLE;
    };

    let text = String::from_utf8_lossy(&body).into_owned();
    if text.is_empty() {
        return StatusCode::BAD_REQUEST;
    }

    let msg = IncomingMessage::new(
        MessageSource::Http,
        text,
        SourceMetadata::Http {
            remote_addr: None,
            user_agent: headers
                .get("user-agent")
                .and_then(|v| v.to_str().ok())
                .map(str::to_owned),
        },
    );

    metrics::counter!(crate::telemetry::MESSAGES_RECEIVED, "source" => "http").increment(1);

    match tx.send(msg).await {
        Ok(()) => StatusCode::ACCEPTED,
        Err(_) => StatusCode::SERVICE_UNAVAILABLE,
    }
}

pub(crate) async fn upload_handler(
    State(state): State<AdminState>,
    headers: HeaderMap,
    mut multipart: Multipart,
) -> impl IntoResponse {
    let admin = &state.cfg.admin;
    if !auth::is_authenticated(&headers, &state.sessions, admin.session_ttl_days) {
        return StatusCode::UNAUTHORIZED;
    }

    let Some(ref tx) = state.inbox_tx else {
        return StatusCode::SERVICE_UNAVAILABLE;
    };

    let mut text = String::new();
    let mut attachments: Vec<Attachment> = Vec::new();

    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.name().unwrap_or("").to_owned();
        let filename = field.file_name().map(str::to_owned);
        let content_type = field.content_type().map(str::to_owned).unwrap_or_default();

        let data = match field.bytes().await {
            Ok(b) => b,
            Err(e) => {
                warn!(?e, field = name, "Multipart read error in proxy upload");
                continue;
            }
        };

        if name == "text" {
            text = String::from_utf8_lossy(&data).into_owned();
        } else if let Some(fname) = filename
            && let Ok(att) = save_bytes(&state.attachments_dir, &fname, &content_type, &data).await
        {
            attachments.push(att);
        }
    }

    let mut msg = IncomingMessage::new(
        MessageSource::Http,
        text,
        SourceMetadata::Http {
            remote_addr: None,
            user_agent: headers
                .get("user-agent")
                .and_then(|v| v.to_str().ok())
                .map(str::to_owned),
        },
    );
    msg.attachments = attachments;

    metrics::counter!(crate::telemetry::MESSAGES_RECEIVED, "source" => "http").increment(1);

    match tx.send(msg).await {
        Ok(()) => StatusCode::ACCEPTED,
        Err(_) => StatusCode::SERVICE_UNAVAILABLE,
    }
}

async fn save_bytes(
    attachments_dir: &std::path::Path,
    filename: &str,
    mime: &str,
    data: &[u8],
) -> Result<Attachment, crate::error::InboxError> {
    let id = uuid::Uuid::new_v4();
    let safe_name = sanitize_filename(filename);
    let path = attachment_save_path(attachments_dir, id, &safe_name);

    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(crate::error::InboxError::Io)?;
    }

    tokio::fs::write(&path, data)
        .await
        .map_err(crate::error::InboxError::Io)?;

    let media_kind = if mime.is_empty() {
        MediaKind::from_mime(mime_guess::from_path(&path).first_raw().unwrap_or(""))
    } else {
        MediaKind::from_mime(mime)
    };

    Ok(Attachment {
        original_name: safe_name,
        saved_path: path,
        mime_type: if mime.is_empty() {
            None
        } else {
            Some(mime.to_owned())
        },
        media_kind,
    })
}

fn sanitize_filename(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_alphanumeric() || matches!(c, '.' | '-' | '_') {
                c
            } else {
                '_'
            }
        })
        .collect()
}
