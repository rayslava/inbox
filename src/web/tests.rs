use super::*;
use crate::config::{
    AdaptersConfig, FallbackMode, GeneralConfig, LlmConfig, LlmPromptsConfig, PipelineConfig,
    SyncthingConfig, ToolingConfig, UrlFetchConfig, WebUiConfig,
};
use crate::health::ReadinessState;
use crate::processing_status::ProcessingTracker;
use axum::body::Body;
use axum::http::Request;
use std::sync::Arc;
use tower::ServiceExt;

fn test_state(ready: bool) -> AdminState {
    let dir = tempfile::tempdir().unwrap();
    let cfg = Arc::new(Config {
        general: GeneralConfig {
            output_file: dir.path().join("inbox.org"),
            attachments_dir: dir.path().to_path_buf(),
            log_level: "info".into(),
            log_format: "pretty".into(),
        },
        admin: crate::config::AdminConfig::default(),
        web_ui: WebUiConfig::default(),
        pipeline: PipelineConfig::default(),
        llm: LlmConfig {
            fallback: FallbackMode::default(),
            url_content_max_chars: 4000,
            max_tool_turns: 5,
            max_llm_tool_depth: 1,
            inner_retries: 0,
            vision_max_bytes: 5 * 1024 * 1024,
            tool_result_max_chars: 0,
            prompts: LlmPromptsConfig::default(),
            backends: vec![],
        },
        adapters: AdaptersConfig::default(),
        url_fetch: UrlFetchConfig::default(),
        syncthing: SyncthingConfig::default(),
        tooling: ToolingConfig::default(),
        memory: crate::config::MemoryConfig::default(),
    });
    let readiness = ReadinessState::new(ready);
    let sessions = auth::new_session_store();
    let handle = metrics_exporter_prometheus::PrometheusBuilder::new()
        .build_recorder()
        .handle();
    let attachments_dir = dir.path().to_path_buf();
    AdminState {
        cfg,
        readiness,
        sessions,
        metrics_handle: handle,
        log_store: LogStore::new(100),
        tracker: Arc::new(ProcessingTracker::new()),
        inbox_tx: None,
        attachments_dir,
    }
}

fn make_router(ready: bool) -> Router {
    let state = test_state(ready);
    admin_router(AdminRouterArgs {
        cfg: state.cfg,
        readiness: state.readiness,
        session_store: state.sessions,
        metrics_handle: state.metrics_handle,
        log_store: state.log_store,
        tracker: state.tracker,
        inbox_tx: state.inbox_tx,
        attachments_dir: state.attachments_dir,
    })
}

#[tokio::test]
async fn health_live_returns_200() {
    let router = make_router(true);
    let req = Request::builder()
        .uri("/health/live")
        .body(Body::empty())
        .unwrap();
    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn health_ready_when_ready_returns_200() {
    let router = make_router(true);
    let req = Request::builder()
        .uri("/health/ready")
        .body(Body::empty())
        .unwrap();
    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn health_ready_when_not_ready_returns_503() {
    let router = make_router(false);
    let req = Request::builder()
        .uri("/health/ready")
        .body(Body::empty())
        .unwrap();
    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn metrics_returns_200() {
    let router = make_router(true);
    let req = Request::builder()
        .uri("/metrics")
        .body(Body::empty())
        .unwrap();
    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn login_get_returns_html() {
    let router = make_router(true);
    let req = Request::builder()
        .uri("/login")
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
    assert!(ct.contains("text/html"));
}

#[tokio::test]
async fn ui_without_session_redirects_to_login() {
    let router = make_router(true);
    let req = Request::builder().uri("/ui").body(Body::empty()).unwrap();
    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::SEE_OTHER);
    let loc = resp.headers().get("location").unwrap().to_str().unwrap();
    assert_eq!(loc, "login");
}

#[tokio::test]
async fn login_post_invalid_credentials_returns_html() {
    let router = make_router(true);
    let req = Request::builder()
        .method("POST")
        .uri("/login")
        .header("content-type", "application/x-www-form-urlencoded")
        .body(Body::from("username=admin&password=wrongpass"))
        .unwrap();
    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK); // returns login form with error
}

#[tokio::test]
async fn logout_redirects_to_login() {
    let router = make_router(true);
    let req = Request::builder()
        .uri("/logout")
        .body(Body::empty())
        .unwrap();
    let resp = router.oneshot(req).await.unwrap();
    // Redirect to login
    let status = resp.status();
    assert!(status.is_redirection(), "expected redirect, got {status}");
}

#[tokio::test]
async fn attachment_traversal_returns_403() {
    let router = make_router(true);
    let req = Request::builder()
        .uri("/attachments/../../etc/passwd")
        .body(Body::empty())
        .unwrap();
    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
}

#[tokio::test]
async fn attachment_not_found_returns_404() {
    let router = make_router(true);
    let req = Request::builder()
        .uri("/attachments/nonexistent.txt")
        .body(Body::empty())
        .unwrap();
    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn ui_nodes_without_session_redirects_to_login() {
    let router = make_router(true);
    let req = Request::builder()
        .uri("/ui/nodes")
        .body(Body::empty())
        .unwrap();
    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::SEE_OTHER);
}

#[tokio::test]
async fn logs_entries_without_session_redirects_to_login() {
    let router = make_router(true);
    let req = Request::builder()
        .uri("/logs/entries")
        .body(Body::empty())
        .unwrap();
    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::SEE_OTHER);
}

#[tokio::test]
async fn ui_nodes_with_session_returns_html_fragment() {
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
    });
    let req = Request::builder()
        .uri("/ui/nodes")
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
    assert!(ct.contains("text/html"));
    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let text = String::from_utf8_lossy(&body);
    // Fragment should NOT contain <html> or <head>
    assert!(!text.contains("<html"));
    assert!(!text.contains("<head"));
}

#[tokio::test]
async fn logs_entries_with_session_returns_html_fragment() {
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
    });
    let req = Request::builder()
        .uri("/logs/entries")
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
    assert!(ct.contains("text/html"));
    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let text = String::from_utf8_lossy(&body);
    assert!(!text.contains("<html"));
}

#[tokio::test]
async fn status_without_session_redirects_to_login() {
    let router = make_router(true);
    let req = Request::builder()
        .uri("/status")
        .body(Body::empty())
        .unwrap();
    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::SEE_OTHER);
    let loc = resp.headers().get("location").unwrap().to_str().unwrap();
    assert_eq!(loc, "login");
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
