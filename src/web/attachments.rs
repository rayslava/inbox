use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use tracing::warn;

use super::AdminState;

/// Serve a file from the attachments directory.
///
/// Path components that would escape the base dir are rejected with 403.
pub(crate) async fn serve_attachment(
    State(state): State<AdminState>,
    Path(path): Path<String>,
) -> Response {
    // Reject any attempt at path traversal before touching the filesystem
    if path.contains("..") {
        return StatusCode::FORBIDDEN.into_response();
    }

    let base = &state.cfg.general.attachments_dir;
    let file_path = base.join(&path);

    let data = match tokio::fs::read(&file_path).await {
        Ok(d) => d,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return StatusCode::NOT_FOUND.into_response();
        }
        Err(e) => {
            warn!(?e, path, "Attachment read error");
            return StatusCode::INTERNAL_SERVER_ERROR.into_response();
        }
    };

    let mime = mime_guess::from_path(&file_path)
        .first_or_octet_stream()
        .to_string();

    ([(axum::http::header::CONTENT_TYPE, mime)], data).into_response()
}
