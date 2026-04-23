/// Integration tests for URL fetching using wiremock.
use std::io::Write as _;

use flate2::Compression;
use flate2::write::GzEncoder;
use inbox::{
    config::UrlFetchConfig,
    pipeline::{url_classifier::UrlKind, url_fetcher::UrlFetcher},
};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{method, path},
};

fn test_fetch_cfg() -> UrlFetchConfig {
    UrlFetchConfig {
        enabled: true,
        user_agent: "inbox-test/1.0".into(),
        timeout_secs: 5,
        max_redirects: 3,
        max_body_bytes: 1024 * 1024,
        skip_domains: vec![],
        nitter_base_url: None,
    }
}

#[tokio::test]
async fn fetcher_scrapes_html_page() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/article"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/html; charset=utf-8")
                .set_body_string(
                    "<html><head><title>Test Page</title></head>\
                    <body><p>Hello from the test page</p></body></html>",
                ),
        )
        .mount(&server)
        .await;

    let fetcher = UrlFetcher::new(&test_fetch_cfg());
    let url = format!("{}/article", server.uri()).parse().unwrap();

    let result = fetcher.fetch_page(&url).await;

    assert!(result.is_some(), "fetch_page should return content");
    let page = result.unwrap();
    assert!(
        page.text.contains("Hello from the test page"),
        "should extract body text: {:?}",
        page.text
    );
    assert_eq!(
        page.page_title.as_deref(),
        Some("Test Page"),
        "should extract page title"
    );
}

#[tokio::test]
async fn fetcher_head_returns_page_kind_for_html() {
    let server = MockServer::start().await;

    Mock::given(method("HEAD"))
        .and(path("/page"))
        .respond_with(ResponseTemplate::new(200).insert_header("content-type", "text/html"))
        .mount(&server)
        .await;

    let fetcher = UrlFetcher::new(&test_fetch_cfg());
    let url = format!("{}/page", server.uri()).parse().unwrap();

    let kind = inbox::pipeline::url_classifier::classify_url(&url, &fetcher).await;
    assert!(
        matches!(kind, UrlKind::Page),
        "HTML should be classified as Page"
    );
}

#[tokio::test]
async fn fetcher_head_returns_file_kind_for_pdf() {
    let server = MockServer::start().await;

    Mock::given(method("HEAD"))
        .and(path("/doc.pdf"))
        .respond_with(ResponseTemplate::new(200).insert_header("content-type", "application/pdf"))
        .mount(&server)
        .await;

    let fetcher = UrlFetcher::new(&test_fetch_cfg());
    let url = format!("{}/doc.pdf", server.uri()).parse().unwrap();

    let kind = inbox::pipeline::url_classifier::classify_url(&url, &fetcher).await;
    assert!(
        matches!(kind, UrlKind::File { .. }),
        "PDF should be classified as File"
    );
}

#[tokio::test]
async fn fetcher_returns_none_for_server_error() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/broken"))
        .respond_with(ResponseTemplate::new(500))
        .mount(&server)
        .await;

    let fetcher = UrlFetcher::new(&test_fetch_cfg());
    let url = format!("{}/broken", server.uri()).parse().unwrap();

    let result = fetcher.fetch_page(&url).await;
    assert!(result.is_none(), "500 response should produce None");
}

#[tokio::test]
async fn fetcher_follows_redirect() {
    let server = MockServer::start().await;

    // /old redirects → /new
    Mock::given(method("GET"))
        .and(path("/old"))
        .respond_with(
            ResponseTemplate::new(301).insert_header("location", format!("{}/new", server.uri())),
        )
        .mount(&server)
        .await;

    Mock::given(method("GET"))
        .and(path("/new"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/html")
                .set_body_string(
                    "<html><head><title>Redirected</title></head>\
                    <body><p>Landed after redirect</p></body></html>",
                ),
        )
        .mount(&server)
        .await;

    let fetcher = UrlFetcher::new(&test_fetch_cfg());
    let url = format!("{}/old", server.uri()).parse().unwrap();

    let result = fetcher.fetch_page(&url).await;
    assert!(
        result.is_some(),
        "should follow redirect and return content"
    );
    let page = result.unwrap();
    assert!(
        page.text.contains("Landed after redirect"),
        "text after redirect: {:?}",
        page.text
    );
    assert_eq!(page.page_title.as_deref(), Some("Redirected"));
}

#[tokio::test]
async fn fetcher_decodes_gzip_body() {
    let html = "<html><head><title>Compressed</title></head>\
                <body><p>Gzip content decoded</p></body></html>";

    // Gzip-compress the HTML body
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(html.as_bytes()).unwrap();
    let compressed = encoder.finish().unwrap();

    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/gz"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/html; charset=utf-8")
                .insert_header("content-encoding", "gzip")
                .set_body_bytes(compressed),
        )
        .mount(&server)
        .await;

    let fetcher = UrlFetcher::new(&test_fetch_cfg());
    let url = format!("{}/gz", server.uri()).parse().unwrap();

    let result = fetcher.fetch_page(&url).await;
    assert!(result.is_some(), "should decode gzip body");
    let page = result.unwrap();
    assert!(
        page.text.contains("Gzip content decoded"),
        "decompressed text: {:?}",
        page.text
    );
    assert_eq!(page.page_title.as_deref(), Some("Compressed"));
}

#[tokio::test]
async fn fetcher_download_file_writes_attachment() {
    use inbox::pipeline::url_fetcher::attachment_save_path;
    use uuid::Uuid;

    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/data/report.pdf"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/pdf")
                .set_body_bytes(&b"%PDF-1.4 fake content"[..]),
        )
        .mount(&server)
        .await;

    let fetcher = UrlFetcher::new(&test_fetch_cfg());
    let tmp = tempfile::tempdir().unwrap();
    let msg_id = Uuid::new_v4();
    let url = format!("{}/data/report.pdf", server.uri()).parse().unwrap();

    let att = fetcher
        .download_file(&url, msg_id, tmp.path())
        .await
        .expect("download_file should succeed");

    assert_eq!(att.original_name, "report.pdf");
    assert_eq!(att.mime_type.as_deref(), Some("application/pdf"));
    assert_eq!(att.media_kind, inbox::message::MediaKind::Document);

    let expected_path = attachment_save_path(tmp.path(), msg_id, "report.pdf");
    assert_eq!(att.saved_path, expected_path);
    let written = tokio::fs::read(&expected_path).await.unwrap();
    assert_eq!(written, b"%PDF-1.4 fake content");
}

#[tokio::test]
async fn fetcher_download_file_returns_none_on_500() {
    use uuid::Uuid;

    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/fail"))
        .respond_with(ResponseTemplate::new(500))
        .mount(&server)
        .await;

    let fetcher = UrlFetcher::new(&test_fetch_cfg());
    let tmp = tempfile::tempdir().unwrap();
    let url = format!("{}/fail", server.uri()).parse().unwrap();

    let att = fetcher
        .download_file(&url, Uuid::new_v4(), tmp.path())
        .await;
    assert!(att.is_none(), "500 should produce None, not panic");
}

#[tokio::test]
async fn fetcher_download_file_strips_mime_parameters() {
    use uuid::Uuid;

    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/song.mp3"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "audio/mpeg; charset=binary")
                .set_body_bytes(&b"ID3\x03\x00\x00\x00fake"[..]),
        )
        .mount(&server)
        .await;

    let fetcher = UrlFetcher::new(&test_fetch_cfg());
    let tmp = tempfile::tempdir().unwrap();
    let url = format!("{}/song.mp3", server.uri()).parse().unwrap();

    let att = fetcher
        .download_file(&url, Uuid::new_v4(), tmp.path())
        .await
        .expect("download should succeed");
    // The `; charset=binary` parameter should be stripped before storing.
    assert_eq!(att.mime_type.as_deref(), Some("audio/mpeg"));
    assert_eq!(att.media_kind, inbox::message::MediaKind::Audio);
}

#[tokio::test]
async fn fetcher_respects_max_body_bytes() {
    let server = MockServer::start().await;
    let big_body = "<html><head><title>T</title></head><body><p>".to_string()
        + &"x".repeat(2048)
        + "</p></body></html>";

    Mock::given(method("GET"))
        .and(path("/big"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/html")
                .set_body_string(big_body),
        )
        .mount(&server)
        .await;

    let mut cfg = test_fetch_cfg();
    cfg.max_body_bytes = 64; // truncate aggressively so the body is clipped
    let fetcher = UrlFetcher::new(&cfg);
    let url = format!("{}/big", server.uri()).parse().unwrap();

    // With only 64 bytes of the response available, HTML parsing extracts
    // whatever it can — most importantly: no panic and Some() is returned.
    let page = fetcher.fetch_page(&url).await;
    assert!(page.is_some(), "truncated body must still return Some");
}

#[test]
fn rewrite_twitter_url_for_twitter_com() {
    use inbox::pipeline::url_fetcher::rewrite_twitter_url;
    use url::Url;
    let u = Url::parse("https://twitter.com/user/status/12345").unwrap();
    let out = rewrite_twitter_url(&u, Some("https://nitter.example.net")).unwrap();
    assert_eq!(out.host_str(), Some("nitter.example.net"));
    assert_eq!(out.path(), "/user/status/12345");
}

#[test]
fn rewrite_twitter_url_for_x_com() {
    use inbox::pipeline::url_fetcher::rewrite_twitter_url;
    use url::Url;
    let u = Url::parse("https://x.com/user/status/12345?s=20").unwrap();
    let out = rewrite_twitter_url(&u, Some("https://nitter.example.net")).unwrap();
    assert_eq!(out.host_str(), Some("nitter.example.net"));
    assert_eq!(out.query(), Some("s=20"));
}

#[test]
fn rewrite_twitter_url_none_without_nitter() {
    use inbox::pipeline::url_fetcher::rewrite_twitter_url;
    use url::Url;
    let u = Url::parse("https://twitter.com/user").unwrap();
    assert!(rewrite_twitter_url(&u, None).is_none());
}

#[test]
fn rewrite_twitter_url_none_for_other_host() {
    use inbox::pipeline::url_fetcher::rewrite_twitter_url;
    use url::Url;
    let u = Url::parse("https://example.com/user").unwrap();
    assert!(rewrite_twitter_url(&u, Some("https://nitter.example.net")).is_none());
}

#[test]
fn rewrite_twitter_url_handles_subdomain() {
    use inbox::pipeline::url_fetcher::rewrite_twitter_url;
    use url::Url;
    let u = Url::parse("https://mobile.twitter.com/foo").unwrap();
    let out = rewrite_twitter_url(&u, Some("https://nitter.example.net")).unwrap();
    assert_eq!(out.path(), "/foo");
}

/// Live network test — only runs when `TEST_WITH_NETWORK=1` is set.
#[tokio::test]
async fn fetcher_live_https_example_com() {
    if std::env::var("TEST_WITH_NETWORK").is_err() {
        return;
    }

    let fetcher = UrlFetcher::new(&UrlFetchConfig {
        enabled: true,
        user_agent: "inbox-test/1.0".into(),
        timeout_secs: 10,
        max_redirects: 5,
        max_body_bytes: 1024 * 1024,
        skip_domains: vec![],
        nitter_base_url: None,
    });

    let url = "https://example.com".parse().unwrap();
    let result = fetcher.fetch_page(&url).await;

    assert!(result.is_some(), "https://example.com should be reachable");
    let page = result.unwrap();
    assert!(page.page_title.is_some(), "example.com should have a title");
    assert!(
        page.text.contains("Example Domain"),
        "expected 'Example Domain' in text, got: {:?}",
        page.text
    );
}
