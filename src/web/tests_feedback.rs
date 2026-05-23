//! Tests for the admin feedback endpoints.

use super::tests::{make_router, test_state};
use super::*;
use axum::body::Body;
use axum::http::Request;
use std::sync::Arc;
use tower::ServiceExt;

// ── Feedback endpoint tests ──────────────────────────────────────────────────

#[tokio::test]
async fn feedback_post_without_session_returns_401() {
    let router = make_router(true);
    let req = Request::builder()
        .method("POST")
        .uri("/feedback")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"message_id":"00000000-0000-0000-0000-000000000001","rating":3}"#,
        ))
        .unwrap();
    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn feedback_post_without_memory_store_returns_503() {
    use axum::http::header;
    use chrono::Utc;

    let state = test_state(true);
    let token = auth::generate_session_token();
    state.sessions.insert(token.clone(), Utc::now());

    let router = admin_router(AdminRouterArgs {
        cfg: state.cfg,
        readiness: state.readiness,
        session_store: state.sessions,
        metrics_handle: state.metrics_handle,
        log_store: state.log_store,
        tracker: state.tracker,
        inbox_tx: state.inbox_tx,
        attachments_dir: state.attachments_dir,
        memory_store: None,
    });
    let req = Request::builder()
        .method("POST")
        .uri("/feedback")
        .header("content-type", "application/json")
        .header(header::COOKIE, format!("session={token}"))
        .body(Body::from(
            r#"{"message_id":"00000000-0000-0000-0000-000000000001","rating":3}"#,
        ))
        .unwrap();
    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn feedback_post_with_memory_store_returns_ok() {
    use crate::memory::MemoryStore;
    use axum::http::header;
    use chrono::Utc;

    let state = test_state(true);
    let token = auth::generate_session_token();
    state.sessions.insert(token.clone(), Utc::now());

    let store = Arc::new(MemoryStore::new_in_memory().unwrap());
    let router = admin_router(AdminRouterArgs {
        cfg: state.cfg,
        readiness: state.readiness,
        session_store: state.sessions,
        metrics_handle: state.metrics_handle,
        log_store: state.log_store,
        tracker: state.tracker,
        inbox_tx: state.inbox_tx,
        attachments_dir: state.attachments_dir,
        memory_store: Some(store),
    });
    let req = Request::builder()
        .method("POST")
        .uri("/feedback")
        .header("content-type", "application/json")
        .header(header::COOKIE, format!("session={token}"))
        .body(Body::from(
            r#"{"message_id":"00000000-0000-0000-0000-000000000001","rating":3}"#,
        ))
        .unwrap();
    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn feedback_post_htmx_returns_html_fragment() {
    use crate::memory::MemoryStore;
    use axum::http::header;
    use chrono::Utc;

    let state = test_state(true);
    let token = auth::generate_session_token();
    state.sessions.insert(token.clone(), Utc::now());

    let store = Arc::new(MemoryStore::new_in_memory().unwrap());
    let router = admin_router(AdminRouterArgs {
        cfg: state.cfg,
        readiness: state.readiness,
        session_store: state.sessions,
        metrics_handle: state.metrics_handle,
        log_store: state.log_store,
        tracker: state.tracker,
        inbox_tx: state.inbox_tx,
        attachments_dir: state.attachments_dir,
        memory_store: Some(store),
    });
    let req = Request::builder()
        .method("POST")
        .uri("/feedback")
        .header("content-type", "application/json")
        .header(header::COOKIE, format!("session={token}"))
        .header("HX-Request", "true")
        .body(Body::from(
            r#"{"message_id":"00000000-0000-0000-0000-000000000001","rating":2}"#,
        ))
        .unwrap();
    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let ct = resp
        .headers()
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap();
    assert!(ct.contains("text/html"));
    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let text = String::from_utf8_lossy(&body);
    assert!(text.contains("feedback-done"));
}

#[tokio::test]
async fn feedback_get_without_session_returns_401() {
    let router = make_router(true);
    let req = Request::builder()
        .uri("/feedback/00000000-0000-0000-0000-000000000001")
        .body(Body::empty())
        .unwrap();
    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn feedback_get_returns_entry_after_post() {
    use crate::memory::MemoryStore;
    use axum::http::header;
    use chrono::Utc;

    let state = test_state(true);
    let token = auth::generate_session_token();
    state.sessions.insert(token.clone(), Utc::now());

    let store = Arc::new(MemoryStore::new_in_memory().unwrap());
    let router = admin_router(AdminRouterArgs {
        cfg: state.cfg,
        readiness: state.readiness,
        session_store: state.sessions.clone(),
        metrics_handle: state.metrics_handle,
        log_store: state.log_store,
        tracker: state.tracker,
        inbox_tx: state.inbox_tx,
        attachments_dir: state.attachments_dir,
        memory_store: Some(store),
    });

    // First, POST feedback
    let req = Request::builder()
        .method("POST")
        .uri("/feedback")
        .header("content-type", "application/json")
        .header(header::COOKIE, format!("session={token}"))
        .body(Body::from(
            r#"{"message_id":"00000000-0000-0000-0000-000000000099","rating":1,"comment":"poor"}"#,
        ))
        .unwrap();
    let resp = router.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // Then, GET it back
    let req = Request::builder()
        .uri("/feedback/00000000-0000-0000-0000-000000000099")
        .header(header::COOKIE, format!("session={token}"))
        .body(Body::empty())
        .unwrap();
    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let text = String::from_utf8_lossy(&body);
    assert!(text.contains("00000000-0000-0000-0000-000000000099"));
    assert!(text.contains("poor"));
}

#[tokio::test]
async fn feedback_get_not_found() {
    use crate::memory::MemoryStore;
    use axum::http::header;
    use chrono::Utc;

    let state = test_state(true);
    let token = auth::generate_session_token();
    state.sessions.insert(token.clone(), Utc::now());

    let store = Arc::new(MemoryStore::new_in_memory().unwrap());
    let router = admin_router(AdminRouterArgs {
        cfg: state.cfg,
        readiness: state.readiness,
        session_store: state.sessions,
        metrics_handle: state.metrics_handle,
        log_store: state.log_store,
        tracker: state.tracker,
        inbox_tx: state.inbox_tx,
        attachments_dir: state.attachments_dir,
        memory_store: Some(store),
    });
    let req = Request::builder()
        .uri("/feedback/nonexistent")
        .header(header::COOKIE, format!("session={token}"))
        .body(Body::empty())
        .unwrap();
    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn proxy_inbox_with_session_and_tx_returns_accepted() {
    use chrono::Utc;
    use tokio::sync::mpsc;

    let state = test_state(true);
    let (tx, _rx) = mpsc::channel(8);
    let token = auth::generate_session_token();
    state.sessions.insert(token.clone(), Utc::now());

    let router = admin_router(AdminRouterArgs {
        cfg: state.cfg,
        readiness: state.readiness,
        session_store: state.sessions,
        metrics_handle: state.metrics_handle,
        log_store: state.log_store,
        tracker: state.tracker,
        inbox_tx: Some(tx),
        attachments_dir: state.attachments_dir,
        memory_store: state.memory_store,
    });
    let req = Request::builder()
        .method("POST")
        .uri("/capture")
        .header("content-type", "text/plain")
        .header(axum::http::header::COOKIE, format!("session={token}"))
        .body(Body::from("hello world"))
        .unwrap();
    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::ACCEPTED);
}

#[tokio::test]
async fn proxy_inbox_without_session_returns_401_not_404() {
    let router = make_router(true);
    let req = Request::builder()
        .method("POST")
        .uri("/capture")
        .header("content-type", "text/plain")
        .body(Body::from("hello"))
        .unwrap();
    let resp = router.oneshot(req).await.unwrap();
    // 401 = route exists but unauthenticated; 404 = route missing
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn proxy_upload_without_session_returns_401_not_404() {
    let router = make_router(true);
    let req = Request::builder()
        .method("POST")
        .uri("/capture/upload")
        .header("content-type", "multipart/form-data; boundary=abc")
        .body(Body::from(
            "--abc\r\nContent-Disposition: form-data; name=\"text\"\r\n\r\nhello\r\n--abc--",
        ))
        .unwrap();
    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn status_with_valid_session_returns_json() {
    use axum::http::header;
    use chrono::Utc;

    let state = test_state(true);
    let token = auth::generate_session_token();
    state.sessions.insert(token.clone(), Utc::now());

    let router = admin_router(AdminRouterArgs {
        cfg: state.cfg,
        readiness: state.readiness,
        session_store: state.sessions,
        metrics_handle: state.metrics_handle,
        log_store: state.log_store,
        tracker: state.tracker,
        inbox_tx: state.inbox_tx,
        attachments_dir: state.attachments_dir,
        memory_store: state.memory_store,
    });
    let req = Request::builder()
        .uri("/status")
        .header(header::COOKIE, format!("session={token}"))
        .body(Body::empty())
        .unwrap();
    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let ct = resp
        .headers()
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap();
    assert!(ct.contains("application/json"));
}

// ── Proxy edge-case tests ────────────────────────────────────────────────────

#[tokio::test]
async fn proxy_inbox_empty_body_returns_400() {
    use axum::http::header;
    use chrono::Utc;
    use tokio::sync::mpsc;

    let state = test_state(true);
    let (tx, _rx) = mpsc::channel(8);
    let token = auth::generate_session_token();
    state.sessions.insert(token.clone(), Utc::now());

    let router = admin_router(AdminRouterArgs {
        cfg: state.cfg,
        readiness: state.readiness,
        session_store: state.sessions,
        metrics_handle: state.metrics_handle,
        log_store: state.log_store,
        tracker: state.tracker,
        inbox_tx: Some(tx),
        attachments_dir: state.attachments_dir,
        memory_store: state.memory_store,
    });
    let req = Request::builder()
        .method("POST")
        .uri("/capture")
        .header("content-type", "text/plain")
        .header(header::COOKIE, format!("session={token}"))
        .body(Body::empty())
        .unwrap();
    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn proxy_inbox_no_tx_returns_503() {
    use axum::http::header;
    use chrono::Utc;

    let state = test_state(true);
    let token = auth::generate_session_token();
    state.sessions.insert(token.clone(), Utc::now());

    // inbox_tx is None by default in test_state
    let router = admin_router(AdminRouterArgs {
        cfg: state.cfg,
        readiness: state.readiness,
        session_store: state.sessions,
        metrics_handle: state.metrics_handle,
        log_store: state.log_store,
        tracker: state.tracker,
        inbox_tx: None,
        attachments_dir: state.attachments_dir,
        memory_store: state.memory_store,
    });
    let req = Request::builder()
        .method("POST")
        .uri("/capture")
        .header("content-type", "text/plain")
        .header(header::COOKIE, format!("session={token}"))
        .body(Body::from("some content"))
        .unwrap();
    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
}
