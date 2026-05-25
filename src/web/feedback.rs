use axum::{
    Json,
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use chrono::Utc;

use crate::feedback::{FeedbackEntry, FeedbackRequest};
use crate::web::auth;

use super::AdminState;

// ── POST /feedback ───────────────────────────────────────────────────────────

pub(crate) async fn submit_handler(
    State(state): State<AdminState>,
    headers: HeaderMap,
    Json(req): Json<FeedbackRequest>,
) -> Response {
    let admin = &state.cfg.admin;
    if !auth::is_authenticated(&headers, &state.sessions, admin.session_ttl_days) {
        tracing::debug!("Web feedback rejected: unauthenticated");
        return StatusCode::UNAUTHORIZED.into_response();
    }

    let rating_val = req.rating.value();
    let has_comment = req.comment.as_ref().is_some_and(|c| !c.is_empty());
    tracing::debug!(
        message_id = %req.message_id,
        rating = rating_val,
        has_comment,
        "Web feedback received"
    );

    let Some(store) = &state.memory_store else {
        tracing::warn!(
            message_id = %req.message_id,
            "Web feedback dropped: memory store is not enabled"
        );
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "Memory store is not enabled",
        )
            .into_response();
    };

    let entry = FeedbackEntry {
        message_id: req.message_id.to_string(),
        rating: rating_val,
        comment: req.comment.unwrap_or_default(),
        created_at: Utc::now(),
        source: "web_ui".into(),
        title: String::new(),
    };

    match store.save_feedback(&entry).await {
        Ok(()) => {
            let is_htmx = headers.get("HX-Request").is_some();
            tracing::debug!(
                message_id = %req.message_id,
                rating = rating_val,
                htmx = is_htmx,
                "Web feedback saved"
            );
            if is_htmx {
                let stars = req.rating.to_string();
                (
                    StatusCode::OK,
                    [(axum::http::header::CONTENT_TYPE, "text/html; charset=utf-8")],
                    format!(r#"<span class="feedback-done">Rated: {stars}</span>"#),
                )
                    .into_response()
            } else {
                (StatusCode::OK, Json(serde_json::json!({"ok": true}))).into_response()
            }
        }
        Err(e) => {
            tracing::warn!(?e, message_id = %req.message_id, "Failed to save feedback");
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
    }
}

// ── GET /feedback/{message_id} ───────────────────────────────────────────────

pub(crate) async fn get_handler(
    State(state): State<AdminState>,
    headers: HeaderMap,
    Path(message_id): Path<String>,
) -> Response {
    let admin = &state.cfg.admin;
    if !auth::is_authenticated(&headers, &state.sessions, admin.session_ttl_days) {
        tracing::debug!(message_id, "Web feedback lookup rejected: unauthenticated");
        return StatusCode::UNAUTHORIZED.into_response();
    }

    let Some(store) = &state.memory_store else {
        tracing::warn!(
            message_id,
            "Web feedback lookup: memory store is not enabled"
        );
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "Memory store is not enabled",
        )
            .into_response();
    };

    match store.query_feedback(&message_id).await {
        Ok(Some(entry)) => {
            tracing::debug!(message_id, rating = entry.rating, "Web feedback lookup hit");
            Json(entry).into_response()
        }
        Ok(None) => {
            tracing::debug!(message_id, "Web feedback lookup miss");
            StatusCode::NOT_FOUND.into_response()
        }
        Err(e) => {
            tracing::warn!(?e, message_id, "Failed to query feedback");
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
    }
}
