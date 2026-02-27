use std::sync::Arc;

use askama::Template;
use axum::{
    Form, Router,
    extract::State,
    http::{HeaderMap, StatusCode, header},
    response::{IntoResponse, Redirect, Response},
    routing::get,
};
use chrono::Utc;
use tracing::warn;

use crate::config::Config;
use crate::health::ReadinessState;

pub mod attachments;
pub mod auth;
pub mod ui;

// ── Shared state ──────────────────────────────────────────────────────────────

#[derive(Clone)]
pub(crate) struct AdminState {
    pub cfg: Arc<Config>,
    pub readiness: ReadinessState,
    pub sessions: Arc<auth::SessionStore>,
    pub metrics_handle: metrics_exporter_prometheus::PrometheusHandle,
}

// ── Router ────────────────────────────────────────────────────────────────────

pub fn admin_router(
    cfg: Arc<Config>,
    readiness: ReadinessState,
    session_store: Arc<auth::SessionStore>,
    metrics_handle: metrics_exporter_prometheus::PrometheusHandle,
) -> Router {
    let web_ui_enabled = cfg.web_ui.enabled;
    let state = AdminState {
        cfg,
        readiness,
        sessions: session_store,
        metrics_handle,
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
            .route("/attachments/{*path}", get(attachments::serve_attachment));
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

    let http_cfg = &state.cfg.adapters.http;
    let inbox_url = if http_cfg.enabled {
        let addr = http_cfg.bind_addr;
        let host = if addr.ip().is_unspecified() {
            "localhost".to_string()
        } else {
            addr.ip().to_string()
        };
        format!("http://{}:{}/inbox", host, addr.port())
    } else {
        String::new()
    };
    let auth_token = http_cfg.auth_token.clone();

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
    use crate::config::{GeneralConfig, LlmConfig};
    use crate::health::ReadinessState;
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
            web_ui: Default::default(),
            llm: LlmConfig {
                fallback: Default::default(),
                url_content_max_chars: 4000,
                max_tool_turns: 5,
                system_prompt: String::new(),
                backends: vec![],
            },
            adapters: Default::default(),
            url_fetch: Default::default(),
            syncthing: Default::default(),
            tools: vec![],
        });
        let readiness = ReadinessState::new(ready);
        let sessions = auth::new_session_store();
        let handle = metrics_exporter_prometheus::PrometheusBuilder::new()
            .build_recorder()
            .handle();
        AdminState {
            cfg,
            readiness,
            sessions,
            metrics_handle: handle,
        }
    }

    fn make_router(ready: bool) -> Router {
        let state = test_state(ready);
        admin_router(
            state.cfg,
            state.readiness,
            state.sessions,
            state.metrics_handle,
        )
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
}
