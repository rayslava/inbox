use crate::config::{ToolBackendConfig, UrlFetchConfig};
use crate::pipeline::url_fetcher::UrlFetcher;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use super::runners::{
    CrawlToolCfg, HttpToolCfg, resolve_env_vars, run_crawler_tool, run_http_tool, run_shell_tool,
};
use super::{Tool, ToolExecutor, ToolResult, default_tools, from_tooling};

fn test_fetcher() -> UrlFetcher {
    UrlFetcher::new(&UrlFetchConfig {
        enabled: true,
        user_agent: "test/1.0".into(),
        timeout_secs: 5,
        max_redirects: 3,
        max_body_bytes: 1024 * 1024,
        skip_domains: vec![],
    })
}

#[test]
fn tool_openai_definition_has_name() {
    let tool = Tool {
        name: "scrape_page".into(),
        description: "desc".into(),
        enabled: true,
        retries: 0,
        backend: ToolBackendConfig::Internal,
    };
    let def = tool.openai_definition();
    assert_eq!(def["function"]["name"], "scrape_page");
}

#[test]
fn active_tool_definitions_filters_disabled() {
    let executor = default_tools(test_fetcher());
    let defs = executor.active_tool_definitions();
    assert_eq!(defs.len(), 2);
}

#[test]
fn active_tool_definitions_empty_when_all_disabled() {
    let tools = vec![Tool {
        name: "scrape_page".into(),
        description: "d".into(),
        enabled: false,
        retries: 0,
        backend: ToolBackendConfig::Internal,
    }];
    let executor = ToolExecutor::new(tools, test_fetcher());
    assert!(executor.active_tool_definitions().is_empty());
}

#[test]
fn tool_result_text() {
    let r = ToolResult::Text("hello".into());
    assert_eq!(r.text(), "hello");
}

#[test]
fn tool_result_attachment_text() {
    use crate::message::{Attachment, MediaKind};
    let r = ToolResult::Attachment {
        text: "downloaded".into(),
        attachment: Attachment {
            original_name: "f.pdf".into(),
            saved_path: std::path::PathBuf::from("/tmp/f.pdf"),
            mime_type: None,
            media_kind: MediaKind::Document,
        },
    };
    assert_eq!(r.text(), "downloaded");
}

#[test]
fn resolve_env_vars_expands_known() {
    unsafe { std::env::set_var("TEST_TOOL_VAR_XYZ", "secret") };
    let result = resolve_env_vars("Bearer ${TEST_TOOL_VAR_XYZ}");
    assert_eq!(result, "Bearer secret");
}

#[test]
fn resolve_env_vars_unknown_becomes_empty() {
    let result = resolve_env_vars("${NONEXISTENT_VAR_12345}");
    assert_eq!(result, "");
}

#[tokio::test]
async fn execute_unknown_tool_errors() {
    let executor = default_tools(test_fetcher());
    let id = uuid::Uuid::new_v4();
    let result = executor
        .execute(
            "nonexistent",
            &serde_json::json!({}),
            id,
            std::path::Path::new("/tmp"),
        )
        .await;
    assert!(result.is_err());
}

#[tokio::test]
async fn execute_scrape_page_missing_url_arg_errors() {
    let executor = default_tools(test_fetcher());
    let id = uuid::Uuid::new_v4();
    let result = executor
        .execute(
            "scrape_page",
            &serde_json::json!({}),
            id,
            std::path::Path::new("/tmp"),
        )
        .await;
    assert!(result.is_err());
}

#[tokio::test]
async fn execute_download_file_missing_url_arg_errors() {
    let executor = default_tools(test_fetcher());
    let id = uuid::Uuid::new_v4();
    let result = executor
        .execute(
            "download_file",
            &serde_json::json!({}),
            id,
            std::path::Path::new("/tmp"),
        )
        .await;
    assert!(result.is_err());
}

#[tokio::test]
async fn execute_crawl_url_missing_url_arg_errors() {
    let tools = vec![Tool {
        name: "crawl_url".into(),
        description: "crawl".into(),
        enabled: true,
        retries: 0,
        backend: ToolBackendConfig::Crawler {
            endpoint: "http://localhost:11235/crawl".into(),
            auth_header: None,
            timeout_secs: 5,
            priority: 10,
        },
    }];
    let executor = ToolExecutor::new(tools, test_fetcher());
    let id = uuid::Uuid::new_v4();
    let result = executor
        .execute(
            "crawl_url",
            &serde_json::json!({}),
            id,
            std::path::Path::new("/tmp"),
        )
        .await;
    assert!(result.is_err());
}

#[tokio::test]
async fn execute_crawl_url_with_wrong_backend_errors() {
    let tools = vec![Tool {
        name: "crawl_url".into(),
        description: "crawl".into(),
        enabled: true,
        retries: 0,
        backend: ToolBackendConfig::Internal,
    }];
    let executor = ToolExecutor::new(tools, test_fetcher());
    let id = uuid::Uuid::new_v4();
    let result = executor
        .execute(
            "crawl_url",
            &serde_json::json!({"url":"https://example.com"}),
            id,
            std::path::Path::new("/tmp"),
        )
        .await;
    assert!(result.is_err());
}

#[tokio::test]
async fn execute_scrape_page_with_crawler_backend_errors() {
    let tools = vec![Tool {
        name: "scrape_page".into(),
        description: "scrape".into(),
        enabled: true,
        retries: 0,
        backend: ToolBackendConfig::Crawler {
            endpoint: "http://localhost:11235/crawl".into(),
            auth_header: None,
            timeout_secs: 5,
            priority: 10,
        },
    }];
    let executor = ToolExecutor::new(tools, test_fetcher());
    let id = uuid::Uuid::new_v4();
    let result = executor
        .execute(
            "scrape_page",
            &serde_json::json!({"url":"https://example.com"}),
            id,
            std::path::Path::new("/tmp"),
        )
        .await;
    assert!(result.is_err());
}

#[tokio::test]
async fn execute_download_file_with_crawler_backend_errors() {
    let tools = vec![Tool {
        name: "download_file".into(),
        description: "download".into(),
        enabled: true,
        retries: 0,
        backend: ToolBackendConfig::Crawler {
            endpoint: "http://localhost:11235/crawl".into(),
            auth_header: None,
            timeout_secs: 5,
            priority: 10,
        },
    }];
    let executor = ToolExecutor::new(tools, test_fetcher());
    let id = uuid::Uuid::new_v4();
    let result = executor
        .execute(
            "download_file",
            &serde_json::json!({"url":"https://example.com"}),
            id,
            std::path::Path::new("/tmp"),
        )
        .await;
    assert!(result.is_err());
}

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

#[test]
fn from_tooling_builds_executor() {
    let cfg = crate::config::ToolingConfig::default();
    let executor = from_tooling(&cfg, test_fetcher());
    assert!(!executor.active_tool_definitions().is_empty());
}
