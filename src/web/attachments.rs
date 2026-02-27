use anodized::contract;
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use std::path::{Component, PathBuf};
use tracing::warn;

use super::AdminState;

/// Serve a file from the attachments directory.
///
/// Path components that would escape the base dir are rejected with 403.
#[contract(requires: !path.is_empty())]
pub(crate) async fn serve_attachment(
    State(state): State<AdminState>,
    Path(path): Path<String>,
) -> Response {
    let Some(rel_path) = normalize_relative_path(&path) else {
        return StatusCode::FORBIDDEN.into_response();
    };

    let base = &state.cfg.general.attachments_dir;
    let base_canon = tokio::fs::canonicalize(base)
        .await
        .unwrap_or_else(|_| base.clone());
    let file_path = base.join(&rel_path);
    let file_canon = match tokio::fs::canonicalize(&file_path).await {
        Ok(path) => path,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return StatusCode::NOT_FOUND.into_response();
        }
        Err(e) => {
            warn!(?e, path, "Attachment canonicalize error");
            return StatusCode::INTERNAL_SERVER_ERROR.into_response();
        }
    };

    if !file_canon.starts_with(&base_canon) {
        return StatusCode::FORBIDDEN.into_response();
    }

    let data = match tokio::fs::read(&file_canon).await {
        Ok(d) => d,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return StatusCode::NOT_FOUND.into_response();
        }
        Err(e) => {
            warn!(?e, path, "Attachment read error");
            return StatusCode::INTERNAL_SERVER_ERROR.into_response();
        }
    };

    let mime = mime_guess::from_path(&file_canon)
        .first_or_octet_stream()
        .to_string();

    ([(axum::http::header::CONTENT_TYPE, mime)], data).into_response()
}

#[must_use]
#[contract(requires: !path.is_empty())]
fn normalize_relative_path(path: &str) -> Option<PathBuf> {
    let mut normalized = PathBuf::new();
    for component in std::path::Path::new(path).components() {
        match component {
            Component::CurDir => {}
            Component::Normal(part) => normalized.push(part),
            Component::RootDir | Component::Prefix(_) | Component::ParentDir => return None,
        }
    }

    if normalized.as_os_str().is_empty() {
        return None;
    }

    Some(normalized)
}

#[cfg(test)]
mod tests {
    use super::normalize_relative_path;
    use std::path::Path;

    #[test]
    fn normalize_relative_path_accepts_safe_relative() {
        let p = normalize_relative_path("aa/bb/file.png").expect("normalized");
        assert_eq!(p, Path::new("aa/bb/file.png"));
    }

    #[test]
    fn normalize_relative_path_rejects_parent_dir() {
        assert!(normalize_relative_path("../etc/passwd").is_none());
    }

    #[test]
    fn normalize_relative_path_rejects_absolute() {
        assert!(normalize_relative_path("/tmp/file").is_none());
    }
}
