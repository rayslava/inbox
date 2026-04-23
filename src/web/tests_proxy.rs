use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use chrono::Utc;
use tokio::sync::mpsc;
use tower::ServiceExt;

use super::proxy::sanitize_filename;
use super::{AdminRouterArgs, admin_router, auth};
use crate::config::{
    AdaptersConfig, Config, FallbackMode, GeneralConfig, LlmConfig, LlmPromptsConfig,
    PipelineConfig, SyncthingConfig, ToolingConfig, UrlFetchConfig, WebUiConfig,
};
use crate::health::ReadinessState;
use crate::log_capture::LogStore;
use crate::processing_status::ProcessingTracker;

#[test]
fn sanitize_filename_passes_safe_chars() {
    assert_eq!(sanitize_filename("report.pdf"), "report.pdf");
    assert_eq!(sanitize_filename("my-file_v2.txt"), "my-file_v2.txt");
}

#[test]
fn sanitize_filename_replaces_spaces_and_special() {
    assert_eq!(sanitize_filename("my file (1).pdf"), "my_file__1_.pdf");
}

#[test]
fn sanitize_filename_replaces_path_separators() {
    let result = sanitize_filename("../../etc/passwd");
    // Dots and dashes are preserved, slashes become underscores.
    assert_eq!(result, ".._.._etc_passwd");
}

#[test]
fn sanitize_filename_unicode_replaced() {
    // Non-ASCII alphanumeric chars are kept, but control chars and symbols are replaced.
    let result = sanitize_filename("café☕.txt");
    assert!(
        std::path::Path::new(&result)
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("txt"))
    );
    assert!(result.contains("caf"));
}

// ── Proxy upload handler tests ────────────────────────────────────────────────

struct ProxyHarness {
    router: axum::Router,
    token: String,
    _tmp: tempfile::TempDir,
}

fn proxy_harness(
    with_tx: bool,
) -> (
    ProxyHarness,
    Option<mpsc::Receiver<crate::message::IncomingMessage>>,
) {
    let tmp = tempfile::tempdir().unwrap();
    let cfg = Arc::new(Config {
        general: GeneralConfig {
            output_file: tmp.path().join("inbox.org"),
            attachments_dir: tmp.path().to_path_buf(),
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

    let sessions = auth::new_session_store();
    let token = auth::generate_session_token();
    sessions.insert(token.clone(), Utc::now());

    let (tx, rx) = if with_tx {
        let (tx, rx) = mpsc::channel(8);
        (Some(tx), Some(rx))
    } else {
        (None, None)
    };

    let handle = metrics_exporter_prometheus::PrometheusBuilder::new()
        .build_recorder()
        .handle();

    let router = admin_router(AdminRouterArgs {
        cfg,
        readiness: ReadinessState::new(true),
        session_store: sessions,
        metrics_handle: handle,
        log_store: LogStore::new(100),
        tracker: Arc::new(ProcessingTracker::new()),
        inbox_tx: tx,
        attachments_dir: tmp.path().to_path_buf(),
        memory_store: None,
    });

    (
        ProxyHarness {
            router,
            token,
            _tmp: tmp,
        },
        rx,
    )
}

#[tokio::test]
async fn proxy_upload_multipart_text_and_file_returns_accepted() {
    let (harness, mut rx) = proxy_harness(true);
    let rx = rx.as_mut().unwrap();

    let boundary = "BOUNDARY";
    let body = format!(
        "--{boundary}\r\n\
Content-Disposition: form-data; name=\"text\"\r\n\r\n\
captured via web\r\n\
--{boundary}\r\n\
Content-Disposition: form-data; name=\"file\"; filename=\"note.txt\"\r\n\
Content-Type: text/plain\r\n\r\n\
body bytes\r\n\
--{boundary}--\r\n"
    );

    let req = Request::builder()
        .method("POST")
        .uri("/capture/upload")
        .header(
            "content-type",
            format!("multipart/form-data; boundary={boundary}"),
        )
        .header(header::COOKIE, format!("session={}", harness.token))
        .body(Body::from(body))
        .unwrap();

    let resp = harness.router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::ACCEPTED);

    let msg = rx.recv().await.expect("message enqueued");
    assert_eq!(msg.text, "captured via web");
    assert_eq!(msg.attachments.len(), 1);
    assert_eq!(msg.attachments[0].original_name, "note.txt");
}

#[tokio::test]
async fn proxy_upload_with_session_but_no_tx_returns_503() {
    let (harness, _) = proxy_harness(false);

    let boundary = "BOUNDARY";
    let body = format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"text\"\r\n\r\nhello\r\n--{boundary}--\r\n"
    );

    let req = Request::builder()
        .method("POST")
        .uri("/capture/upload")
        .header(
            "content-type",
            format!("multipart/form-data; boundary={boundary}"),
        )
        .header(header::COOKIE, format!("session={}", harness.token))
        .body(Body::from(body))
        .unwrap();

    let resp = harness.router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn proxy_upload_file_without_filename_is_skipped() {
    let (harness, mut rx) = proxy_harness(true);
    let rx = rx.as_mut().unwrap();

    // Unnamed file field (no filename) must not become an attachment.
    let boundary = "BOUNDARY";
    let body = format!(
        "--{boundary}\r\n\
Content-Disposition: form-data; name=\"text\"\r\n\r\n\
just text\r\n\
--{boundary}\r\n\
Content-Disposition: form-data; name=\"attachment\"\r\n\
Content-Type: application/octet-stream\r\n\r\n\
raw bytes\r\n\
--{boundary}--\r\n"
    );

    let req = Request::builder()
        .method("POST")
        .uri("/capture/upload")
        .header(
            "content-type",
            format!("multipart/form-data; boundary={boundary}"),
        )
        .header(header::COOKIE, format!("session={}", harness.token))
        .body(Body::from(body))
        .unwrap();

    let resp = harness.router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::ACCEPTED);

    let msg = rx.recv().await.expect("message enqueued");
    assert_eq!(msg.text, "just text");
    assert!(
        msg.attachments.is_empty(),
        "filename-less field must not produce an attachment"
    );
}
