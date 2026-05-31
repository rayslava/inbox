//! `process_url` branch coverage with wiremock — split out of `pipeline/tests.rs`
//! to keep that file under the line limit.

use std::sync::Arc;

use super::Pipeline;
use super::tests::{make_msg, test_config};
use crate::processing_status::ProcessingTracker;

fn pipeline_with_fetch_cfg(fetch_cfg: crate::config::UrlFetchConfig) -> Arc<Pipeline> {
    let dir = tempfile::tempdir().unwrap();
    let mut cfg = test_config(crate::config::JsShellPolicy::ToolOnly);
    cfg.general.attachments_dir = dir.path().to_path_buf();
    cfg.url_fetch = fetch_cfg;
    cfg.llm.url_content_max_chars = 200;
    // Leak the tempdir so attachments survive for the duration of the test.
    Box::leak(Box::new(dir));
    let cfg = Arc::new(cfg);

    let llm = crate::test_helpers::mock_llm_chain(crate::test_helpers::default_llm_response());
    let writer = Arc::new(crate::output::NullWriter);
    let tracker = Arc::new(ProcessingTracker::new());
    Arc::new(Pipeline::new(cfg, llm, writer, tracker, None, None).expect("build pipeline"))
}

fn fetch_cfg_enabled() -> crate::config::UrlFetchConfig {
    crate::config::UrlFetchConfig {
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
async fn process_url_fetches_html_page_into_url_contents() {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let server = MockServer::start().await;
    Mock::given(method("HEAD"))
        .and(path("/article"))
        .respond_with(ResponseTemplate::new(200).insert_header("content-type", "text/html"))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/article"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/html")
                .set_body_string(
                    "<html><head><title>T</title></head><body><p>hello body</p></body></html>",
                ),
        )
        .mount(&server)
        .await;

    let pipeline = pipeline_with_fetch_cfg(fetch_cfg_enabled());
    let msg = make_msg(&format!("look at {}/article please", server.uri()));
    let enriched = pipeline.enrich(msg).await.expect("enrich ok");

    assert_eq!(enriched.url_contents.len(), 1);
    assert!(enriched.url_contents[0].text.contains("hello body"));
    assert!(enriched.original.attachments.is_empty());
}

#[tokio::test]
async fn process_url_downloads_file_attachments() {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let server = MockServer::start().await;
    Mock::given(method("HEAD"))
        .and(path("/file.bin"))
        .respond_with(
            ResponseTemplate::new(200).insert_header("content-type", "application/octet-stream"),
        )
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/file.bin"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/octet-stream")
                .set_body_bytes(vec![1u8, 2, 3, 4]),
        )
        .mount(&server)
        .await;

    let pipeline = pipeline_with_fetch_cfg(fetch_cfg_enabled());
    let msg = make_msg(&format!("download {}/file.bin", server.uri()));
    let enriched = pipeline.enrich(msg).await.expect("enrich ok");

    assert!(enriched.url_contents.is_empty());
    assert_eq!(enriched.original.attachments.len(), 1);
}

#[tokio::test]
async fn process_url_skips_domain_in_skip_list() {
    use wiremock::matchers::method;
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let server = MockServer::start().await;
    // If the test ever reaches the server, the mock would respond — the
    // assertion that url_contents is empty proves it didn't.
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(200).set_body_string("should not be reached"))
        .mount(&server)
        .await;

    let host = server.uri().trim_start_matches("http://").to_owned();
    // Strip the port so skip_domain matches by host only.
    let bare_host = host.split(':').next().unwrap().to_owned();
    let mut fc = fetch_cfg_enabled();
    fc.skip_domains = vec![bare_host];

    let pipeline = pipeline_with_fetch_cfg(fc);
    let msg = make_msg(&format!("see {}/anything", server.uri()));
    let enriched = pipeline.enrich(msg).await.expect("enrich ok");

    assert!(enriched.url_contents.is_empty());
    assert!(enriched.original.attachments.is_empty());
}

#[tokio::test]
async fn process_url_drops_page_matching_js_shell_policy() {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let server = MockServer::start().await;
    Mock::given(method("HEAD"))
        .and(path("/spa"))
        .respond_with(ResponseTemplate::new(200).insert_header("content-type", "text/html"))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/spa"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/html")
                .set_body_string(
                    "<html><body>This page doesn't work properly without JavaScript enabled \
                     please enable it to continue</body></html>",
                ),
        )
        .mount(&server)
        .await;

    let pipeline = pipeline_with_fetch_cfg(fetch_cfg_enabled());
    let msg = make_msg(&format!("see {}/spa", server.uri()));
    let enriched = pipeline.enrich(msg).await.expect("enrich ok");

    assert!(
        enriched.url_contents.is_empty(),
        "JS-shell policy should suppress page content"
    );
}

#[tokio::test]
async fn process_url_unknown_kind_falls_back_to_page_fetch() {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let server = MockServer::start().await;
    // HEAD with a content-type that fails MIME parse → UrlKind::Unknown.
    Mock::given(method("HEAD"))
        .and(path("/weird"))
        .respond_with(ResponseTemplate::new(200).insert_header("content-type", "garbage"))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/weird"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/html")
                .set_body_string("<html><body><p>unknown fallback body</p></body></html>"),
        )
        .mount(&server)
        .await;

    let pipeline = pipeline_with_fetch_cfg(fetch_cfg_enabled());
    let msg = make_msg(&format!("see {}/weird", server.uri()));
    let enriched = pipeline.enrich(msg).await.expect("enrich ok");

    assert_eq!(enriched.url_contents.len(), 1);
    assert!(
        enriched.url_contents[0]
            .text
            .contains("unknown fallback body")
    );
}
