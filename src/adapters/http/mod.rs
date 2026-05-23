use std::sync::Arc;

use anodized::spec;
use axum::{
    Router,
    body::Bytes,
    extract::{DefaultBodyLimit, Multipart, State},
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
    routing::post,
};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tower_http::cors::{Any, CorsLayer};
use tracing::{info, warn};

use uuid::Uuid;

use crate::config::HttpAdapterConfig;
use crate::error::InboxError;
use crate::message::{Attachment, IncomingMessage, MediaKind, MessageSource, SourceMetadata};
use crate::pipeline::url_fetcher::attachment_save_path;

use super::InputAdapter;

pub struct HttpAdapter {
    pub cfg: HttpAdapterConfig,
    pub attachments_dir: std::path::PathBuf,
}

#[async_trait::async_trait]
impl InputAdapter for HttpAdapter {
    fn name(&self) -> &'static str {
        "http"
    }

    async fn run(
        self: Box<Self>,
        tx: mpsc::Sender<IncomingMessage>,
        shutdown: CancellationToken,
    ) -> Result<(), InboxError> {
        let bind_addr = self.cfg.bind_addr;
        let router = build_router(Arc::new(*self), tx);

        info!(%bind_addr, "HTTP adapter listening");

        let listener = tokio::net::TcpListener::bind(bind_addr)
            .await
            .map_err(InboxError::Io)?;

        axum::serve(listener, router)
            .with_graceful_shutdown(async move { shutdown.cancelled().await })
            .await
            .map_err(|e| InboxError::Adapter(e.to_string()))?;

        Ok(())
    }
}

// ── State ─────────────────────────────────────────────────────────────────────

#[derive(Clone)]
struct AppState {
    tx: mpsc::Sender<IncomingMessage>,
    adapter: Arc<HttpAdapter>,
}

// ── Router ────────────────────────────────────────────────────────────────────

fn build_router(adapter: Arc<HttpAdapter>, tx: mpsc::Sender<IncomingMessage>) -> Router {
    let state = AppState { tx, adapter };

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([axum::http::Method::POST, axum::http::Method::OPTIONS])
        .allow_headers([
            axum::http::header::CONTENT_TYPE,
            axum::http::header::AUTHORIZATION,
        ])
        .max_age(std::time::Duration::from_secs(3600));

    Router::new()
        .route("/inbox", post(inbox_handler))
        .route("/inbox/upload", post(upload_handler))
        .layer(cors)
        .layer(DefaultBodyLimit::max(50 * 1024 * 1024)) // 50 MB
        .with_state(state)
}

// ── Handlers ──────────────────────────────────────────────────────────────────

async fn inbox_handler(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> impl IntoResponse {
    if !check_auth(&headers, &state.adapter.cfg.auth_token) {
        return StatusCode::UNAUTHORIZED;
    }

    let content_type = headers
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");

    let msg_id = Uuid::new_v4();

    let (text, attachment) = if content_type.starts_with("application/json") {
        let json: serde_json::Value = match serde_json::from_slice(&body) {
            Ok(v) => v,
            Err(_) => return StatusCode::BAD_REQUEST,
        };
        (json["text"].as_str().unwrap_or("").to_owned(), None)
    } else if content_type.starts_with("text/") || content_type.is_empty() {
        (String::from_utf8_lossy(&body).into_owned(), None)
    } else {
        // Binary body — treat as single file upload
        let filename = derive_filename_from_headers(&headers).unwrap_or_else(|| "upload".into());
        let mime = content_type.to_owned();
        let att = save_bytes(
            &state.adapter.attachments_dir,
            &filename,
            &mime,
            &body,
            msg_id,
        )
        .await;
        (filename.clone(), att.ok())
    };

    enqueue_message(&state, text, attachment, &headers, msg_id).await
}

async fn upload_handler(
    State(state): State<AppState>,
    headers: HeaderMap,
    mut multipart: Multipart,
) -> impl IntoResponse {
    if !check_auth(&headers, &state.adapter.cfg.auth_token) {
        return StatusCode::UNAUTHORIZED;
    }

    let mut text = String::new();
    let mut attachments: Vec<Attachment> = Vec::new();

    let msg_id = Uuid::new_v4();

    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.name().unwrap_or("").to_owned();
        let filename = field.file_name().map(str::to_owned);
        let content_type = field.content_type().map(str::to_owned).unwrap_or_default();

        let data = match field.bytes().await {
            Ok(b) => b,
            Err(e) => {
                warn!(?e, field = name, "Multipart read error");
                continue;
            }
        };

        if name == "text" {
            text = String::from_utf8_lossy(&data).into_owned();
        } else if let Some(fname) = filename
            && let Ok(att) = save_bytes(
                &state.adapter.attachments_dir,
                &fname,
                &content_type,
                &data,
                msg_id,
            )
            .await
        {
            attachments.push(att);
        }
    }

    let mut msg = IncomingMessage::with_id(
        msg_id,
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

    match state.tx.send(msg).await {
        Ok(()) => StatusCode::ACCEPTED,
        Err(_) => StatusCode::SERVICE_UNAVAILABLE,
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn check_auth(headers: &HeaderMap, expected_token: &str) -> bool {
    if expected_token.is_empty() {
        return true; // auth disabled
    }
    headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .is_some_and(|t| t == expected_token)
}

async fn enqueue_message(
    state: &AppState,
    text: String,
    attachment: Option<Attachment>,
    headers: &HeaderMap,
    msg_id: Uuid,
) -> StatusCode {
    let remote_addr = None; // axum ConnectInfo requires extra extractor setup
    let user_agent = headers
        .get("user-agent")
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);

    let mut msg = IncomingMessage::with_id(
        msg_id,
        MessageSource::Http,
        text,
        SourceMetadata::Http {
            remote_addr,
            user_agent,
        },
    );

    if let Some(att) = attachment {
        msg.attachments.push(att);
    }

    metrics::counter!(crate::telemetry::MESSAGES_RECEIVED, "source" => "http").increment(1);

    match state.tx.send(msg).await {
        Ok(()) => StatusCode::ACCEPTED,
        Err(_) => StatusCode::SERVICE_UNAVAILABLE,
    }
}

async fn save_bytes(
    attachments_dir: &std::path::Path,
    filename: &str,
    mime: &str,
    data: &[u8],
    msg_id: Uuid,
) -> Result<Attachment, InboxError> {
    #[spec(requires: !filename.is_empty() && !data.is_empty())]
    fn validate_input(filename: &str, data: &[u8]) {
        let _ = (filename, data);
    }
    validate_input(filename, data);

    let id = msg_id;
    let safe_name = sanitize_filename(filename);
    let path = attachment_save_path(attachments_dir, id, &safe_name);

    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(InboxError::Io)?;
    }

    tokio::fs::write(&path, data)
        .await
        .map_err(InboxError::Io)?;

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

fn derive_filename_from_headers(headers: &HeaderMap) -> Option<String> {
    headers
        .get("content-disposition")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| {
            v.split(';')
                .find_map(|p| p.trim().strip_prefix("filename="))
                .map(|f| f.trim_matches('"').to_owned())
        })
}

#[cfg(test)]
mod tests;
