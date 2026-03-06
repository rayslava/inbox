use std::path::PathBuf;
use std::sync::Arc;

use askama::Template;
use axum::{
    Form, Router,
    extract::State,
    http::{HeaderMap, StatusCode, header},
    response::{IntoResponse, Redirect, Response},
    routing::{get, post},
};
use chrono::Utc;
use tokio::sync::mpsc;
use tracing::warn;

use crate::config::Config;
use crate::health::ReadinessState;
use crate::log_capture::LogStore;
use crate::message::IncomingMessage;
use crate::processing_status::ProcessingTracker;

pub mod attachments;
pub mod auth;
pub mod proxy;
pub mod ui;

// ── Shared state ──────────────────────────────────────────────────────────────

#[derive(Clone)]
pub(crate) struct AdminState {
    pub cfg: Arc<Config>,
    pub readiness: ReadinessState,
    pub sessions: Arc<auth::SessionStore>,
    pub metrics_handle: metrics_exporter_prometheus::PrometheusHandle,
    pub log_store: Arc<LogStore>,
    pub tracker: Arc<ProcessingTracker>,
    pub inbox_tx: Option<mpsc::Sender<IncomingMessage>>,
    pub attachments_dir: PathBuf,
}

// ── Router ────────────────────────────────────────────────────────────────────

pub struct AdminRouterArgs {
    pub cfg: Arc<Config>,
    pub readiness: ReadinessState,
    pub session_store: Arc<auth::SessionStore>,
    pub metrics_handle: metrics_exporter_prometheus::PrometheusHandle,
    pub log_store: Arc<LogStore>,
    pub tracker: Arc<ProcessingTracker>,
    pub inbox_tx: Option<mpsc::Sender<IncomingMessage>>,
    pub attachments_dir: PathBuf,
}

pub fn admin_router(args: AdminRouterArgs) -> Router {
    let web_ui_enabled = args.cfg.web_ui.enabled;
    let state = AdminState {
        cfg: args.cfg,
        readiness: args.readiness,
        sessions: args.session_store,
        metrics_handle: args.metrics_handle,
        log_store: args.log_store,
        tracker: args.tracker,
        inbox_tx: args.inbox_tx,
        attachments_dir: args.attachments_dir,
    };

    let mut router = Router::new()
        .route("/health/live", get(live_handler))
        .route("/health/ready", get(ready_handler))
        .route("/metrics", get(metrics_handler));

    if web_ui_enabled {
        router = router
            .route("/login", get(login_get).post(login_post))
            .route("/logout", get(logout_handler))
            .route("/ui", get(ui_handler))
            .route("/logs", get(logs_handler))
            .route("/status", get(status_handler))
            .route("/attachments/{*path}", get(attachments::serve_attachment))
            .route("/api/inbox", post(proxy::inbox_handler))
            .route("/api/inbox/upload", post(proxy::upload_handler));
    }

    router.with_state(state)
}

// ── Health handlers ───────────────────────────────────────────────────────────

async fn live_handler() -> impl IntoResponse {
    (StatusCode::OK, "alive")
}

async fn ready_handler(State(state): State<AdminState>) -> impl IntoResponse {
    if state.readiness.is_ready() {
        (StatusCode::OK, r#"{"status":"ready"}"#)
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, r#"{"status":"not_ready"}"#)
    }
}

async fn metrics_handler(State(state): State<AdminState>) -> impl IntoResponse {
    state.metrics_handle.render()
}

// ── Auth handlers ─────────────────────────────────────────────────────────────

async fn login_get() -> impl IntoResponse {
    html_response(
        ui::LoginTemplate { error: None }
            .render()
            .unwrap_or_default(),
    )
}

async fn login_post(
    State(state): State<AdminState>,
    Form(form): Form<auth::LoginForm>,
) -> Response {
    let admin = &state.cfg.admin;

    let valid = form.username == admin.username
        && !admin.password_hash.is_empty()
        && auth::verify_password(&admin.password_hash, &form.password);

    if !valid {
        warn!(user = form.username, "Failed login attempt");
        return html_response(
            ui::LoginTemplate {
                error: Some("Invalid username or password.".into()),
            }
            .render()
            .unwrap_or_default(),
        );
    }

    let token = auth::generate_session_token();
    state.sessions.insert(token.clone(), Utc::now());

    let ttl_secs = admin.session_ttl_days * 86_400;
    let cookie = format!("session={token}; HttpOnly; SameSite=Lax; Path=/; Max-Age={ttl_secs}");

    Response::builder()
        .status(StatusCode::SEE_OTHER)
        .header(header::LOCATION, "/ui")
        .header(header::SET_COOKIE, cookie)
        .body(axum::body::Body::empty())
        .unwrap()
}

async fn logout_handler(State(state): State<AdminState>, headers: HeaderMap) -> Response {
    if let Some(token) = auth::extract_session_token(&headers) {
        state.sessions.remove(&token);
    }
    let clear = "session=; Max-Age=0; HttpOnly; SameSite=Lax; Path=/";
    let mut resp = Redirect::to("/login").into_response();
    resp.headers_mut().insert(
        header::SET_COOKIE,
        header::HeaderValue::from_str(clear).expect("static cookie value"),
    );
    resp
}

// ── UI handler ────────────────────────────────────────────────────────────────

async fn ui_handler(State(state): State<AdminState>, headers: HeaderMap) -> Response {
    let admin = &state.cfg.admin;
    if !auth::is_authenticated(&headers, &state.sessions, admin.session_ttl_days) {
        return Redirect::to("/login").into_response();
    }

    let org_path = &state.cfg.general.output_file;
    let content = tokio::fs::read_to_string(org_path)
        .await
        .unwrap_or_default();
    let mut nodes = ui::parse_org_nodes(&content, &state.cfg.general.attachments_dir);
    nodes.reverse(); // most-recent first

    let inbox_url = if state.inbox_tx.is_some() {
        "/api/inbox".to_owned()
    } else {
        String::new()
    };
    let auth_token = String::new();

    html_response(
        ui::InboxUiTemplate {
            nodes,
            inbox_url,
            auth_token,
        }
        .render()
        .unwrap_or_default(),
    )
}

// ── Status handler ───────────────────────────────────────────────────────────

async fn status_handler(State(state): State<AdminState>, headers: HeaderMap) -> Response {
    let admin = &state.cfg.admin;
    if !auth::is_authenticated(&headers, &state.sessions, admin.session_ttl_days) {
        return Redirect::to("/login").into_response();
    }
    axum::Json(state.tracker.snapshot()).into_response()
}

// ── Logs handler ─────────────────────────────────────────────────────────────

async fn logs_handler(State(state): State<AdminState>, headers: HeaderMap) -> Response {
    let admin = &state.cfg.admin;
    if !auth::is_authenticated(&headers, &state.sessions, admin.session_ttl_days) {
        return Redirect::to("/login").into_response();
    }
    let entries = state.log_store.recent();
    html_response(ui::LogsTemplate { entries }.render().unwrap_or_default())
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn html_response(body: String) -> Response {
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
        body,
    )
        .into_response()
}

#[cfg(test)]
mod tests {
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
                vision_max_bytes: 5 * 1024 * 1024,
                prompts: LlmPromptsConfig::default(),
                backends: vec![],
            },
            adapters: AdaptersConfig::default(),
            url_fetch: UrlFetchConfig::default(),
            syncthing: SyncthingConfig::default(),
            tooling: ToolingConfig::default(),
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
        assert_eq!(loc, "/login");
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
        // Redirect to /login
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
    async fn status_without_session_redirects_to_login() {
        let router = make_router(true);
        let req = Request::builder()
            .uri("/status")
            .body(Body::empty())
            .unwrap();
        let resp = router.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::SEE_OTHER);
        let loc = resp.headers().get("location").unwrap().to_str().unwrap();
        assert_eq!(loc, "/login");
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
}
