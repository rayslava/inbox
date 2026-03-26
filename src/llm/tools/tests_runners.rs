use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use super::runners::{CrawlToolCfg, HttpToolCfg, run_crawler_tool, run_http_tool, run_shell_tool};

#[tokio::test]
async fn run_shell_tool_empty_argv_errors() {
    let result = run_shell_tool(&[], "http://x.com", "", 5).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn run_shell_tool_echo() {
    let argv = vec!["echo".to_owned(), "hello {url}".to_owned()];
    let result = run_shell_tool(&argv, "world", "", 5).await.unwrap();
    assert!(result.text().contains("hello world"));
}

#[tokio::test]
async fn run_http_tool_plain_text_response() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/content"))
        .respond_with(ResponseTemplate::new(200).set_body_string("page text"))
        .mount(&server)
        .await;

    let client = reqwest::Client::new();
    let cfg = HttpToolCfg {
        endpoint: &format!("{}/content", server.uri()),
        method: "GET",
        auth_header: None,
        body_template: None,
        response_path: "",
        timeout_secs: 5,
    };
    let result = run_http_tool(&client, cfg, "http://x.com", "")
        .await
        .unwrap();
    assert_eq!(result.text(), "page text");
}

#[tokio::test]
async fn run_http_tool_json_path_extraction() {
    let server = MockServer::start().await;
    let body = serde_json::json!({"data": {"content": "extracted text"}});
    Mock::given(method("GET"))
        .and(path("/json"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(body),
        )
        .mount(&server)
        .await;

    let client = reqwest::Client::new();
    let cfg = HttpToolCfg {
        endpoint: &format!("{}/json", server.uri()),
        method: "GET",
        auth_header: None,
        body_template: None,
        response_path: "data.content",
        timeout_secs: 5,
    };
    let result = run_http_tool(&client, cfg, "", "").await.unwrap();
    assert_eq!(result.text(), "extracted text");
}

#[tokio::test]
async fn run_http_tool_post_with_body() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/post"))
        .respond_with(ResponseTemplate::new(200).set_body_string("ok"))
        .mount(&server)
        .await;

    let client = reqwest::Client::new();
    let endpoint = format!("{}/post", server.uri());
    let cfg = HttpToolCfg {
        endpoint: &endpoint,
        method: "POST",
        auth_header: Some("X-Key: myvalue"),
        body_template: Some(r#"{"url":"{url}"}"#),
        response_path: "",
        timeout_secs: 5,
    };
    let result = run_http_tool(&client, cfg, "http://x.com", "")
        .await
        .unwrap();
    assert_eq!(result.text(), "ok");
}

#[tokio::test]
async fn run_crawler_tool_markdown_then_html_fallback() {
    let server = MockServer::start().await;
    let body = serde_json::json!({
        "success": true,
        "results": [{
            "metadata": {"title": "T"},
            "markdown": {"raw_markdown": "# md"},
            "cleaned_html": "<p>html</p>"
        }]
    });
    Mock::given(method("POST"))
        .and(path("/crawl"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(body),
        )
        .mount(&server)
        .await;

    let client = reqwest::Client::new();
    let cfg = CrawlToolCfg {
        endpoint: &format!("{}/crawl", server.uri()),
        auth_header: None,
        timeout_secs: 5,
        priority: 10,
    };
    let result = run_crawler_tool(&client, cfg, "http://x.com")
        .await
        .unwrap();
    assert!(result.text().contains("Title: T"));
    assert!(result.text().contains("# md"));
}

#[tokio::test]
async fn run_crawler_tool_uses_html_when_markdown_missing() {
    let server = MockServer::start().await;
    let body = serde_json::json!({
        "success": true,
        "results": [{
            "metadata": {"title": "T"},
            "markdown": {"raw_markdown": ""},
            "cleaned_html": "<p>html only</p>"
        }]
    });
    Mock::given(method("POST"))
        .and(path("/crawl"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(body),
        )
        .mount(&server)
        .await;

    let client = reqwest::Client::new();
    let cfg = CrawlToolCfg {
        endpoint: &format!("{}/crawl", server.uri()),
        auth_header: None,
        timeout_secs: 5,
        priority: 10,
    };
    let result = run_crawler_tool(&client, cfg, "http://x.com")
        .await
        .unwrap();
    assert!(result.text().contains("HTML fallback:"));
    assert!(result.text().contains("html only"));
}

#[tokio::test]
async fn run_crawler_tool_errors_when_results_missing() {
    let server = MockServer::start().await;
    let body = serde_json::json!({
        "success": true,
        "results": []
    });
    Mock::given(method("POST"))
        .and(path("/crawl"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(body),
        )
        .mount(&server)
        .await;

    let client = reqwest::Client::new();
    let cfg = CrawlToolCfg {
        endpoint: &format!("{}/crawl", server.uri()),
        auth_header: None,
        timeout_secs: 5,
        priority: 10,
    };
    let result = run_crawler_tool(&client, cfg, "http://x.com").await;
    assert!(result.is_err());
}

#[tokio::test]
async fn run_crawler_tool_handles_empty_content() {
    let server = MockServer::start().await;
    let body = serde_json::json!({
        "success": true,
        "results": [{
            "metadata": {"title": ""},
            "markdown": {"raw_markdown": ""},
            "cleaned_html": ""
        }]
    });
    Mock::given(method("POST"))
        .and(path("/crawl"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(body),
        )
        .mount(&server)
        .await;

    let client = reqwest::Client::new();
    let cfg = CrawlToolCfg {
        endpoint: &format!("{}/crawl", server.uri()),
        auth_header: None,
        timeout_secs: 5,
        priority: 10,
    };
    let result = run_crawler_tool(&client, cfg, "http://x.com")
        .await
        .expect("crawler result");
    assert!(result.text().contains("no markdown/html"));
}
