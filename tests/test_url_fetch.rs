/// Integration tests for URL fetching using wiremock.
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
