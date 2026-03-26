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
use crate::memory::MemoryStore;
use crate::message::IncomingMessage;
use crate::processing_status::ProcessingTracker;

pub mod attachments;
pub mod auth;
pub mod feedback;
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
    pub memory_store: Option<Arc<MemoryStore>>,
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
    pub memory_store: Option<Arc<MemoryStore>>,
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
        memory_store: args.memory_store,
    };

    let mut router = Router::new()
        .route("/health/live", get(live_handler))
        .route("/health/ready", get(ready_handler))
        .route("/metrics", get(metrics_handler));

    if web_ui_enabled {
        router = router
            .route("/", get(root_redirect))
            .route("/favicon.svg", get(favicon_handler))
            .route("/login", get(login_get).post(login_post))
            .route("/logout", get(logout_handler))
            .route("/ui", get(ui_handler))
            .route("/ui/nodes", get(ui_nodes_handler))
            .route("/logs", get(logs_handler))
            .route("/logs/entries", get(logs_entries_handler))
            .route("/status", get(status_handler))
            .route("/attachments/{*path}", get(attachments::serve_attachment))
            .route("/capture", post(proxy::inbox_handler))
            .route("/capture/upload", post(proxy::upload_handler))
            .route("/feedback", post(feedback::submit_handler))
            .route("/feedback/{message_id}", get(feedback::get_handler));
    }

    router.with_state(state)
}

// ── Root / favicon ───────────────────────────────────────────────────────────

async fn root_redirect() -> Redirect {
    Redirect::to("ui")
}

const FAVICON_SVG: &str = include_str!("../../static/favicon.svg");

async fn favicon_handler() -> impl IntoResponse {
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "image/svg+xml")],
        FAVICON_SVG,
    )
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
        .header(header::LOCATION, "ui")
        .header(header::SET_COOKIE, cookie)
        .body(axum::body::Body::empty())
        .unwrap()
}

async fn logout_handler(State(state): State<AdminState>, headers: HeaderMap) -> Response {
    if let Some(token) = auth::extract_session_token(&headers) {
        state.sessions.remove(&token);
    }
    let clear = "session=; Max-Age=0; HttpOnly; SameSite=Lax; Path=/";
    let mut resp = Redirect::to("login").into_response();
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
        return Redirect::to("login").into_response();
    }

    let org_path = &state.cfg.general.output_file;
    let content = tokio::fs::read_to_string(org_path)
        .await
        .unwrap_or_default();
    let mut nodes = ui::parse_org_nodes(&content, &state.cfg.general.attachments_dir);
    nodes.reverse(); // most-recent first

    let inbox_url = if state.inbox_tx.is_some() {
        "capture".to_owned()
    } else {
        String::new()
    };

    html_response(
        ui::InboxUiTemplate { nodes, inbox_url }
            .render()
            .unwrap_or_default(),
    )
}

// ── Fragment handlers ─────────────────────────────────────────────────────────

async fn ui_nodes_handler(State(state): State<AdminState>, headers: HeaderMap) -> Response {
    let admin = &state.cfg.admin;
    if !auth::is_authenticated(&headers, &state.sessions, admin.session_ttl_days) {
        return Redirect::to("login").into_response();
    }

    let org_path = &state.cfg.general.output_file;
    let content = tokio::fs::read_to_string(org_path)
        .await
        .unwrap_or_default();
    let mut nodes = ui::parse_org_nodes(&content, &state.cfg.general.attachments_dir);
    nodes.reverse();

    html_response(
        ui::InboxNodesTemplate { nodes }
            .render()
            .unwrap_or_default(),
    )
}

async fn logs_entries_handler(State(state): State<AdminState>, headers: HeaderMap) -> Response {
    let admin = &state.cfg.admin;
    if !auth::is_authenticated(&headers, &state.sessions, admin.session_ttl_days) {
        return Redirect::to("login").into_response();
    }
    let entries = state.log_store.recent();
    html_response(
        ui::LogsEntriesTemplate { entries }
            .render()
            .unwrap_or_default(),
    )
}

// ── Status handler ───────────────────────────────────────────────────────────

async fn status_handler(State(state): State<AdminState>, headers: HeaderMap) -> Response {
    let admin = &state.cfg.admin;
    if !auth::is_authenticated(&headers, &state.sessions, admin.session_ttl_days) {
        return Redirect::to("login").into_response();
    }
    axum::Json(state.tracker.snapshot()).into_response()
}

// ── Logs handler ─────────────────────────────────────────────────────────────

async fn logs_handler(State(state): State<AdminState>, headers: HeaderMap) -> Response {
    let admin = &state.cfg.admin;
    if !auth::is_authenticated(&headers, &state.sessions, admin.session_ttl_days) {
        return Redirect::to("login").into_response();
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
mod tests;
#[cfg(test)]
mod tests_proxy;
