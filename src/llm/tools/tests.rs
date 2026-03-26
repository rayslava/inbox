use crate::config::{ToolBackendConfig, UrlFetchConfig};
use crate::pipeline::url_fetcher::UrlFetcher;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use super::runners::resolve_env_vars;
use super::{Tool, ToolExecutor, ToolResult, default_tools, from_tooling};

fn test_fetcher() -> UrlFetcher {
    UrlFetcher::new(&UrlFetchConfig {
        enabled: true,
        user_agent: "test/1.0".into(),
        timeout_secs: 5,
        max_redirects: 3,
        max_body_bytes: 1024 * 1024,
        skip_domains: vec![],
        nitter_base_url: None,
    })
}

#[test]
fn tool_openai_definition_has_name() {
    let tool = Tool {
        name: "scrape_page".into(),
        description: "desc".into(),
        enabled: true,
        retries: 0,
        backend: ToolBackendConfig::Internal { timeout_secs: 15 },
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
        backend: ToolBackendConfig::Internal { timeout_secs: 15 },
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
            "",
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
            "",
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
            "",
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
            "",
        )
        .await;
    assert!(result.is_err());
}

#[tokio::test]
async fn execute_web_search_missing_query_arg_errors() {
    let tools = vec![Tool {
        name: "web_search".into(),
        description: "search".into(),
        enabled: true,
        retries: 0,
        backend: ToolBackendConfig::KagiSearch {
            endpoint: "https://kagi.com/api/v0/search".into(),
            api_token: Some("token".into()),
            timeout_secs: 5,
            default_limit: 3,
            max_snippet_chars: 120,
        },
    }];
    let executor = ToolExecutor::new(tools, test_fetcher());
    let id = uuid::Uuid::new_v4();
    let result = executor
        .execute(
            "web_search",
            &serde_json::json!({}),
            id,
            std::path::Path::new("/tmp"),
            "",
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
        backend: ToolBackendConfig::Internal { timeout_secs: 15 },
    }];
    let executor = ToolExecutor::new(tools, test_fetcher());
    let id = uuid::Uuid::new_v4();
    let result = executor
        .execute(
            "crawl_url",
            &serde_json::json!({"url":"https://example.com"}),
            id,
            std::path::Path::new("/tmp"),
            "",
        )
        .await;
    assert!(result.is_err());
}

#[tokio::test]
async fn execute_web_search_with_wrong_backend_errors() {
    let tools = vec![Tool {
        name: "web_search".into(),
        description: "search".into(),
        enabled: true,
        retries: 0,
        backend: ToolBackendConfig::Internal { timeout_secs: 15 },
    }];
    let executor = ToolExecutor::new(tools, test_fetcher());
    let id = uuid::Uuid::new_v4();
    let result = executor
        .execute(
            "web_search",
            &serde_json::json!({"query":"rust async"}),
            id,
            std::path::Path::new("/tmp"),
            "",
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
            "",
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
            "",
        )
        .await;
    assert!(result.is_err());
}

#[test]
fn from_tooling_builds_executor() {
    let cfg = crate::config::ToolingConfig::default();
    let executor = from_tooling(&cfg, test_fetcher());
    assert!(!executor.active_tool_definitions().is_empty());
}

#[tokio::test]
async fn internal_scrape_respects_timeout() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/slow"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string("ok")
                .set_delay(std::time::Duration::from_secs(5)),
        )
        .mount(&server)
        .await;

    let tools = vec![Tool {
        name: "scrape_page".into(),
        description: "scrape".into(),
        enabled: true,
        retries: 0,
        backend: ToolBackendConfig::Internal { timeout_secs: 1 },
    }];
    let executor = ToolExecutor::new(tools, test_fetcher());
    let id = uuid::Uuid::new_v4();
    let result = executor
        .execute(
            "scrape_page",
            &serde_json::json!({"url": format!("{}/slow", server.uri())}),
            id,
            std::path::Path::new("/tmp"),
            "",
        )
        .await;
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("timed out"),
        "expected timeout error, got: {err_msg}"
    );
}

#[tokio::test]
async fn exponential_backoff_increases_delay() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/slow"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string("ok")
                .set_delay(std::time::Duration::from_secs(10)),
        )
        .mount(&server)
        .await;

    let tools = vec![Tool {
        name: "scrape_page".into(),
        description: "scrape".into(),
        enabled: true,
        retries: 1,
        backend: ToolBackendConfig::Internal { timeout_secs: 1 },
    }];
    let executor = ToolExecutor::new(tools, test_fetcher());
    let id = uuid::Uuid::new_v4();
    let start = std::time::Instant::now();
    let result = executor
        .execute(
            "scrape_page",
            &serde_json::json!({"url": format!("{}/slow", server.uri())}),
            id,
            std::path::Path::new("/tmp"),
            "",
        )
        .await;
    let elapsed = start.elapsed();
    assert!(result.is_err());
    assert!(
        elapsed.as_secs() >= 3,
        "expected backoff delay, elapsed: {elapsed:?}"
    );
}
