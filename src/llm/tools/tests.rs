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
    .expect("build test fetcher")
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
    let executor = default_tools(test_fetcher()).expect("build tools");
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
    let executor = ToolExecutor::new(tools, test_fetcher()).expect("build executor");
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
    let executor = default_tools(test_fetcher()).expect("build tools");
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
    let executor = default_tools(test_fetcher()).expect("build tools");
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
    let executor = default_tools(test_fetcher()).expect("build tools");
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
    let executor = ToolExecutor::new(tools, test_fetcher()).expect("build executor");
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
    let executor = ToolExecutor::new(tools, test_fetcher()).expect("build executor");
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
async fn execute_keenable_search_missing_query_arg_errors() {
    let tools = vec![Tool {
        name: "keenable_search".into(),
        description: "search".into(),
        enabled: true,
        retries: 0,
        backend: ToolBackendConfig::KeenableSearch {
            endpoint: "https://api.keenable.ai/v1/search".into(),
            api_key: Some("key".into()),
            timeout_secs: 5,
            default_limit: 3,
            max_snippet_chars: 120,
        },
    }];
    let executor = ToolExecutor::new(tools, test_fetcher()).expect("build executor");
    let id = uuid::Uuid::new_v4();
    let result = executor
        .execute(
            "keenable_search",
            &serde_json::json!({}),
            id,
            std::path::Path::new("/tmp"),
            "",
        )
        .await;
    assert!(result.is_err());
}

#[tokio::test]
async fn execute_keenable_search_with_wrong_backend_errors() {
    let tools = vec![Tool {
        name: "keenable_search".into(),
        description: "search".into(),
        enabled: true,
        retries: 0,
        backend: ToolBackendConfig::Internal { timeout_secs: 15 },
    }];
    let executor = ToolExecutor::new(tools, test_fetcher()).expect("build executor");
    let id = uuid::Uuid::new_v4();
    let result = executor
        .execute(
            "keenable_search",
            &serde_json::json!({"query":"rust async"}),
            id,
            std::path::Path::new("/tmp"),
            "",
        )
        .await;
    assert!(result.is_err());
}

#[tokio::test]
async fn execute_crawl_url_blank_url_errors() {
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
    let executor = ToolExecutor::new(tools, test_fetcher()).expect("build executor");
    let id = uuid::Uuid::new_v4();
    let result = executor
        .execute(
            "crawl_url",
            &serde_json::json!({"url":"   "}),
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
    let executor = ToolExecutor::new(tools, test_fetcher()).expect("build executor");
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
    let executor = ToolExecutor::new(tools, test_fetcher()).expect("build executor");
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
    let executor = ToolExecutor::new(tools, test_fetcher()).expect("build executor");
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
    let executor = ToolExecutor::new(tools, test_fetcher()).expect("build executor");
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
    let executor = from_tooling(&cfg, test_fetcher()).expect("build tools");
    assert!(!executor.active_tool_definitions().is_empty());
}

#[tokio::test]
async fn execute_scrape_page_internal_returns_text() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/page"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/html")
                .set_body_string("<html><body><p>Hello scrape world</p></body></html>"),
        )
        .mount(&server)
        .await;

    let tools = vec![Tool {
        name: "scrape_page".into(),
        description: "scrape".into(),
        enabled: true,
        retries: 0,
        backend: ToolBackendConfig::Internal { timeout_secs: 5 },
    }];
    let executor = ToolExecutor::new(tools, test_fetcher()).expect("build executor");
    let url = format!("{}/page", server.uri());
    let result = executor
        .execute(
            "scrape_page",
            &serde_json::json!({ "url": url }),
            uuid::Uuid::new_v4(),
            std::path::Path::new("/tmp"),
            "",
        )
        .await
        .unwrap();
    assert!(result.text().contains("Hello scrape world"));
}

#[tokio::test]
async fn execute_download_file_internal_saves_attachment() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/file.bin"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/octet-stream")
                .set_body_bytes(vec![1u8, 2, 3, 4]),
        )
        .mount(&server)
        .await;

    let tools = vec![Tool {
        name: "download_file".into(),
        description: "download".into(),
        enabled: true,
        retries: 0,
        backend: ToolBackendConfig::Internal { timeout_secs: 5 },
    }];
    let executor = ToolExecutor::new(tools, test_fetcher()).expect("build executor");
    let dir = tempfile::tempdir().unwrap();
    let url = format!("{}/file.bin", server.uri());
    let result = executor
        .execute(
            "download_file",
            &serde_json::json!({ "url": url }),
            uuid::Uuid::new_v4(),
            dir.path(),
            "",
        )
        .await
        .unwrap();
    assert!(result.text().contains("Downloaded"));
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
    let executor = ToolExecutor::new(tools, test_fetcher()).expect("build executor");
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

// ── dispatch_once additional branch coverage ─────────────────────────────────

fn nitter_fetcher(nitter_base: &str) -> UrlFetcher {
    UrlFetcher::new(&UrlFetchConfig {
        enabled: true,
        user_agent: "test/1.0".into(),
        timeout_secs: 5,
        max_redirects: 3,
        max_body_bytes: 1024 * 1024,
        skip_domains: vec![],
        nitter_base_url: Some(nitter_base.into()),
    })
    .expect("build nitter fetcher")
}

#[tokio::test]
async fn execute_memory_save_with_source_links_to_source() {
    use std::sync::Arc;

    let store = Arc::new(crate::memory::MemoryStore::new_in_memory().unwrap());
    let mut executor = default_tools(test_fetcher()).expect("build tools");
    super::add_memory_tools(&mut executor, Arc::clone(&store));

    let id = uuid::Uuid::new_v4();
    executor
        .execute(
            "memory_save",
            &serde_json::json!({"key": "k1", "value": "v1"}),
            id,
            std::path::Path::new("/tmp"),
            "telegram",
        )
        .await
        .expect("memory_save with source ok");

    let sources = store.sources("k1").await.unwrap();
    assert_eq!(sources.len(), 1);
    assert_eq!(sources[0].kind, "telegram");
    assert_eq!(sources[0].source_id, id.to_string());
}

#[tokio::test]
async fn execute_memory_context_empty_returns_no_results_message() {
    use std::sync::Arc;

    let store = Arc::new(crate::memory::MemoryStore::new_in_memory().unwrap());
    store.save("lonely", "no links").await.unwrap();

    let mut executor = default_tools(test_fetcher()).expect("build tools");
    super::add_memory_tools(&mut executor, Arc::clone(&store));

    let result = executor
        .execute(
            "memory_context",
            &serde_json::json!({"query": "lonely"}),
            uuid::Uuid::new_v4(),
            std::path::Path::new("/tmp"),
            "",
        )
        .await
        .unwrap();
    assert!(
        result.text().contains("No connected memories"),
        "expected empty-context message, got: {}",
        result.text()
    );
}

#[tokio::test]
async fn execute_registered_but_unhandled_tool_name_errors() {
    // A tool registered with a name `dispatch_once` does not match in its
    // `match name` arm exercises the `_ =>` branch (line 234).
    let tools = vec![Tool {
        name: "made_up_tool".into(),
        description: "no handler".into(),
        enabled: true,
        retries: 0,
        backend: ToolBackendConfig::Internal { timeout_secs: 5 },
    }];
    let executor = ToolExecutor::new(tools, test_fetcher()).expect("build executor");
    let result = executor
        .execute(
            "made_up_tool",
            &serde_json::json!({}),
            uuid::Uuid::new_v4(),
            std::path::Path::new("/tmp"),
            "",
        )
        .await;
    assert!(result.is_err(), "should error on unhandled tool name");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("No handler for tool"),
        "expected handler error, got: {msg}"
    );
}

#[tokio::test]
async fn execute_scrape_page_rewrites_twitter_via_nitter() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/some_user/status/123"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/html")
                .set_body_string("<html><body><p>nitter mirror</p></body></html>"),
        )
        .mount(&server)
        .await;

    let tools = vec![Tool {
        name: "scrape_page".into(),
        description: "scrape".into(),
        enabled: true,
        retries: 0,
        backend: ToolBackendConfig::Internal { timeout_secs: 5 },
    }];
    let executor = ToolExecutor::new(tools, nitter_fetcher(&server.uri()))
        .expect("build executor with nitter");
    let result = executor
        .execute(
            "scrape_page",
            &serde_json::json!({"url": "https://twitter.com/some_user/status/123"}),
            uuid::Uuid::new_v4(),
            std::path::Path::new("/tmp"),
            "",
        )
        .await
        .unwrap();
    assert!(result.text().contains("nitter mirror"));
}

#[tokio::test]
async fn execute_scrape_page_shell_backend_runs_command() {
    let tools = vec![Tool {
        name: "scrape_page".into(),
        description: "scrape".into(),
        enabled: true,
        retries: 0,
        backend: ToolBackendConfig::Shell {
            argv: vec!["printf".into(), "shell scrape ok for {url}".into()],
            timeout_secs: 5,
        },
    }];
    let executor = ToolExecutor::new(tools, test_fetcher()).expect("build executor");
    let result = executor
        .execute(
            "scrape_page",
            &serde_json::json!({"url": "https://example.com/page"}),
            uuid::Uuid::new_v4(),
            std::path::Path::new("/tmp"),
            "",
        )
        .await
        .unwrap();
    assert!(result.text().contains("shell scrape ok"));
    assert!(result.text().contains("https://example.com/page"));
}

#[tokio::test]
async fn execute_scrape_page_http_backend_returns_body() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/forward"))
        .respond_with(ResponseTemplate::new(200).set_body_string("scraped via http backend"))
        .mount(&server)
        .await;

    let tools = vec![Tool {
        name: "scrape_page".into(),
        description: "scrape".into(),
        enabled: true,
        retries: 0,
        backend: ToolBackendConfig::Http {
            endpoint: format!("{}/forward", server.uri()),
            method: "GET".into(),
            auth_header: None,
            body_template: None,
            response_path: String::new(),
            timeout_secs: 5,
        },
    }];
    let executor = ToolExecutor::new(tools, test_fetcher()).expect("build executor");
    let result = executor
        .execute(
            "scrape_page",
            &serde_json::json!({"url": "https://example.com/page"}),
            uuid::Uuid::new_v4(),
            std::path::Path::new("/tmp"),
            "",
        )
        .await
        .unwrap();
    assert!(result.text().contains("scraped via http backend"));
}

#[tokio::test]
async fn execute_download_file_internal_returns_failure_text_on_404() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/missing.bin"))
        .respond_with(ResponseTemplate::new(404))
        .mount(&server)
        .await;

    let tools = vec![Tool {
        name: "download_file".into(),
        description: "download".into(),
        enabled: true,
        retries: 0,
        backend: ToolBackendConfig::Internal { timeout_secs: 5 },
    }];
    let executor = ToolExecutor::new(tools, test_fetcher()).expect("build executor");
    let dir = tempfile::tempdir().unwrap();
    let result = executor
        .execute(
            "download_file",
            &serde_json::json!({"url": format!("{}/missing.bin", server.uri())}),
            uuid::Uuid::new_v4(),
            dir.path(),
            "",
        )
        .await
        .unwrap();
    assert!(
        result.text().contains("Failed to download"),
        "expected failure text, got: {}",
        result.text()
    );
}

#[tokio::test]
async fn execute_download_file_shell_backend_runs_command() {
    let dir = tempfile::tempdir().unwrap();
    let tools = vec![Tool {
        name: "download_file".into(),
        description: "download".into(),
        enabled: true,
        retries: 0,
        backend: ToolBackendConfig::Shell {
            argv: vec!["printf".into(), "saved {url} to {filename}".into()],
            timeout_secs: 5,
        },
    }];
    let executor = ToolExecutor::new(tools, test_fetcher()).expect("build executor");
    let result = executor
        .execute(
            "download_file",
            &serde_json::json!({"url": "https://example.com/file.bin"}),
            uuid::Uuid::new_v4(),
            dir.path(),
            "",
        )
        .await
        .unwrap();
    assert!(result.text().contains("saved https://example.com/file.bin"));
}

#[tokio::test]
async fn execute_download_file_http_backend_returns_body() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/dl"))
        .respond_with(ResponseTemplate::new(200).set_body_string("downloaded via http backend"))
        .mount(&server)
        .await;

    let tools = vec![Tool {
        name: "download_file".into(),
        description: "download".into(),
        enabled: true,
        retries: 0,
        backend: ToolBackendConfig::Http {
            endpoint: format!("{}/dl", server.uri()),
            method: "POST".into(),
            auth_header: None,
            body_template: Some(r#"{"url":"{url}"}"#.into()),
            response_path: String::new(),
            timeout_secs: 5,
        },
    }];
    let executor = ToolExecutor::new(tools, test_fetcher()).expect("build executor");
    let dir = tempfile::tempdir().unwrap();
    let result = executor
        .execute(
            "download_file",
            &serde_json::json!({"url": "https://example.com/big.bin"}),
            uuid::Uuid::new_v4(),
            dir.path(),
            "",
        )
        .await
        .unwrap();
    assert!(result.text().contains("downloaded via http backend"));
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
    let executor = ToolExecutor::new(tools, test_fetcher()).expect("build executor");
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
