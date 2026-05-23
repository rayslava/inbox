//! HTTP adapter tests.

use super::*;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use tokio::sync::mpsc;
use tower::ServiceExt;

fn make_adapter(auth_token: &str) -> Arc<HttpAdapter> {
    let dir = tempfile::tempdir().unwrap();
    Arc::new(HttpAdapter {
        cfg: HttpAdapterConfig {
            enabled: true,
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            auth_token: auth_token.to_owned(),
        },
        attachments_dir: dir.path().to_path_buf(),
    })
}

#[test]
fn check_auth_empty_token_allows_all() {
    let headers = HeaderMap::new();
    assert!(check_auth(&headers, ""));
}

#[test]
fn check_auth_valid_bearer() {
    let mut headers = HeaderMap::new();
    headers.insert("authorization", "Bearer secret123".parse().unwrap());
    assert!(check_auth(&headers, "secret123"));
}

#[test]
fn check_auth_wrong_bearer() {
    let mut headers = HeaderMap::new();
    headers.insert("authorization", "Bearer wrong".parse().unwrap());
    assert!(!check_auth(&headers, "secret123"));
}

#[test]
fn check_auth_missing_header_with_token() {
    let headers = HeaderMap::new();
    assert!(!check_auth(&headers, "secret123"));
}

#[test]
fn sanitize_filename_replaces_specials() {
    assert_eq!(sanitize_filename("hello world.txt"), "hello_world.txt");
    assert_eq!(sanitize_filename("file/../../etc"), "file_.._.._etc");
}

#[test]
fn sanitize_filename_keeps_safe_chars() {
    assert_eq!(
        sanitize_filename("report-2024_v1.pdf"),
        "report-2024_v1.pdf"
    );
}

#[test]
fn derive_filename_no_header() {
    let headers = HeaderMap::new();
    assert!(derive_filename_from_headers(&headers).is_none());
}

#[test]
fn derive_filename_with_quoted() {
    let mut headers = HeaderMap::new();
    headers.insert(
        "content-disposition",
        "attachment; filename=\"report.pdf\"".parse().unwrap(),
    );
    assert_eq!(
        derive_filename_from_headers(&headers),
        Some("report.pdf".into())
    );
}

#[tokio::test]
async fn post_inbox_text_plain_returns_accepted() {
    let (tx, _rx) = mpsc::channel(10);
    let adapter = make_adapter("");
    let router = build_router(adapter, tx);

    let req = Request::builder()
        .method("POST")
        .uri("/inbox")
        .header("content-type", "text/plain")
        .body(Body::from("hello inbox"))
        .unwrap();

    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::ACCEPTED);
}

#[tokio::test]
async fn post_inbox_json_returns_accepted() {
    let (tx, _rx) = mpsc::channel(10);
    let adapter = make_adapter("");
    let router = build_router(adapter, tx);

    let req = Request::builder()
        .method("POST")
        .uri("/inbox")
        .header("content-type", "application/json")
        .body(Body::from(r#"{"text":"hello"}"#))
        .unwrap();

    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::ACCEPTED);
}

#[tokio::test]
async fn post_inbox_with_auth_valid_returns_accepted() {
    let (tx, _rx) = mpsc::channel(10);
    let adapter = make_adapter("mytoken");
    let router = build_router(adapter, tx);

    let req = Request::builder()
        .method("POST")
        .uri("/inbox")
        .header("authorization", "Bearer mytoken")
        .header("content-type", "text/plain")
        .body(Body::from("secured"))
        .unwrap();

    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::ACCEPTED);
}

#[tokio::test]
async fn post_inbox_with_auth_invalid_returns_unauthorized() {
    let (tx, _rx) = mpsc::channel(10);
    let adapter = make_adapter("mytoken");
    let router = build_router(adapter, tx);

    let req = Request::builder()
        .method("POST")
        .uri("/inbox")
        .header("authorization", "Bearer wrong")
        .header("content-type", "text/plain")
        .body(Body::from("secured"))
        .unwrap();

    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn post_inbox_invalid_json_returns_bad_request() {
    let (tx, _rx) = mpsc::channel(10);
    let adapter = make_adapter("");
    let router = build_router(adapter, tx);

    let req = Request::builder()
        .method("POST")
        .uri("/inbox")
        .header("content-type", "application/json")
        .body(Body::from("not json{"))
        .unwrap();

    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn post_inbox_channel_closed_returns_service_unavailable() {
    let (tx, rx) = mpsc::channel(1);
    drop(rx); // close receiver so send() fails
    let adapter = make_adapter("");
    let router = build_router(adapter, tx);

    let req = Request::builder()
        .method("POST")
        .uri("/inbox")
        .header("content-type", "text/plain")
        .body(Body::from("msg"))
        .unwrap();

    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn post_inbox_binary_body_treated_as_upload() {
    let (tx, mut rx) = mpsc::channel(10);
    let adapter = make_adapter("");
    let router = build_router(Arc::clone(&adapter), tx);

    let req = Request::builder()
        .method("POST")
        .uri("/inbox")
        .header("content-type", "image/png")
        .header("content-disposition", "attachment; filename=\"pic.png\"")
        .body(Body::from(&b"PNGDATA"[..]))
        .unwrap();

    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::ACCEPTED);

    let msg = rx.recv().await.expect("message enqueued");
    assert_eq!(msg.attachments.len(), 1);
    assert_eq!(msg.attachments[0].original_name, "pic.png");
    assert_eq!(msg.attachments[0].media_kind, MediaKind::Image);
}

#[tokio::test]
async fn post_upload_multipart_accepts_text_and_file() {
    let (tx, mut rx) = mpsc::channel(10);
    let adapter = make_adapter("");
    let router = build_router(Arc::clone(&adapter), tx);

    let boundary = "BOUNDARY";
    let body = format!(
        "--{boundary}\r\n\
Content-Disposition: form-data; name=\"text\"\r\n\r\n\
hello multipart\r\n\
--{boundary}\r\n\
Content-Disposition: form-data; name=\"file\"; filename=\"note.txt\"\r\n\
Content-Type: text/plain\r\n\r\n\
body bytes\r\n\
--{boundary}--\r\n"
    );

    let req = Request::builder()
        .method("POST")
        .uri("/inbox/upload")
        .header(
            "content-type",
            format!("multipart/form-data; boundary={boundary}"),
        )
        .body(Body::from(body))
        .unwrap();

    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::ACCEPTED);

    let msg = rx.recv().await.expect("message enqueued");
    assert_eq!(msg.text, "hello multipart");
    assert_eq!(msg.attachments.len(), 1);
    assert_eq!(msg.attachments[0].original_name, "note.txt");
}

#[tokio::test]
async fn post_upload_unauthorized() {
    let (tx, _rx) = mpsc::channel(10);
    let adapter = make_adapter("mytoken");
    let router = build_router(adapter, tx);

    let req = Request::builder()
        .method("POST")
        .uri("/inbox/upload")
        .header("content-type", "multipart/form-data; boundary=x")
        .body(Body::from("--x--\r\n"))
        .unwrap();

    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn save_bytes_writes_and_classifies_by_mime() {
    let dir = tempfile::tempdir().unwrap();
    let msg_id = Uuid::new_v4();
    let att = save_bytes(
        dir.path(),
        "photo.jpg",
        "image/jpeg",
        b"\xff\xd8\xff\xe0fake",
        msg_id,
    )
    .await
    .expect("save");
    assert_eq!(att.media_kind, MediaKind::Image);
    assert!(att.saved_path.exists());
    assert_eq!(att.mime_type.as_deref(), Some("image/jpeg"));
}

#[tokio::test]
async fn save_bytes_empty_mime_falls_back_to_extension_guess() {
    let dir = tempfile::tempdir().unwrap();
    let msg_id = Uuid::new_v4();
    let att = save_bytes(dir.path(), "song.mp3", "", b"ID3data", msg_id)
        .await
        .expect("save");
    assert_eq!(att.media_kind, MediaKind::Audio);
    assert!(att.mime_type.is_none());
}

#[test]
fn derive_filename_unquoted() {
    let mut headers = HeaderMap::new();
    headers.insert(
        "content-disposition",
        "attachment; filename=plain.txt".parse().unwrap(),
    );
    assert_eq!(
        derive_filename_from_headers(&headers),
        Some("plain.txt".into())
    );
}

#[test]
fn check_auth_bearer_without_prefix_is_rejected() {
    let mut headers = HeaderMap::new();
    headers.insert("authorization", "secret123".parse().unwrap());
    assert!(!check_auth(&headers, "secret123"));
}
