use crate::config::{ToolBackendConfig, UrlFetchConfig};
use crate::pipeline::url_fetcher::UrlFetcher;
use wiremock::matchers::{header, method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

use super::runners::{
    CrawlToolCfg, DuckDuckGoSearchToolCfg, HttpToolCfg, KagiSearchToolCfg, resolve_env_vars,
    run_crawler_tool, run_duckduckgo_search_tool, run_http_tool, run_kagi_search_tool,
    run_shell_tool,
};
use super::{Tool, ToolExecutor, ToolResult, add_memory_tools, default_tools, from_tooling};

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

#[tokio::test]
async fn run_kagi_search_tool_formats_results() {
    let server = MockServer::start().await;
    let body = serde_json::json!({
        "meta": {"id":"x","node":"test","ms":1},
        "data": [{
            "title": "Rust",
            "url": "https://www.rust-lang.org/",
            "snippet": "A language empowering everyone.",
            "published": null
        }]
    });

    Mock::given(method("GET"))
        .and(path("/api/v0/search"))
        .and(query_param("q", "rust language"))
        .and(query_param("limit", "3"))
        .and(header("Authorization", "Bot kagi-token"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(body),
        )
        .mount(&server)
        .await;

    let client = reqwest::Client::new();
    let cfg = KagiSearchToolCfg {
        endpoint: &format!("{}/api/v0/search", server.uri()),
        api_token: Some("kagi-token"),
        timeout_secs: 5,
        default_limit: 5,
        max_snippet_chars: 120,
    };

    let result = run_kagi_search_tool(&client, cfg, "rust language", Some(3))
        .await
        .unwrap();
    assert!(result.text().contains("Kagi web_search results"));
    assert!(result.text().contains("Rust"));
    assert!(result.text().contains("https://www.rust-lang.org/"));
}

#[tokio::test]
async fn run_kagi_search_tool_requires_non_empty_token_when_configured() {
    let client = reqwest::Client::new();
    let cfg = KagiSearchToolCfg {
        endpoint: "https://kagi.com/api/v0/search",
        api_token: Some(""),
        timeout_secs: 5,
        default_limit: 5,
        max_snippet_chars: 120,
    };
    let result = run_kagi_search_tool(&client, cfg, "rust", None).await;
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
        )
        .await;
    let elapsed = start.elapsed();
    assert!(result.is_err());
    // First attempt: 1s timeout, then 2s backoff, then 1s timeout = 4s total minimum
    assert!(
        elapsed.as_secs() >= 3,
        "expected backoff delay, elapsed: {elapsed:?}"
    );
}

static DDG_HTML_FIXTURE: &str = r#"<!DOCTYPE html><html><body>
<div class="results_links_deep">
  <a class="result__a" href="/l/?uddg=https%3A%2F%2Fwww.rust-lang.org%2F&amp;rut=x">Rust Programming Language</a>
  <span class="result__snippet">A language empowering everyone to build reliable software.</span>
</div>
<div class="results_links_deep">
  <a class="result__a" href="/l/?uddg=https%3A%2F%2Fdoc.rust-lang.org%2F&amp;rut=y">Rust Documentation</a>
  <span class="result__snippet">The Rust reference and book.</span>
</div>
</body></html>"#;

#[tokio::test]
async fn run_duckduckgo_search_formats_results() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/html/"))
        .and(query_param("q", "rust language"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/html; charset=utf-8")
                .set_body_string(DDG_HTML_FIXTURE),
        )
        .mount(&server)
        .await;

    let client = reqwest::Client::new();
    let cfg = DuckDuckGoSearchToolCfg {
        endpoint: &format!("{}/html/", server.uri()),
        timeout_secs: 5,
        default_limit: 5,
        max_snippet_chars: 320,
    };

    let result = run_duckduckgo_search_tool(&client, cfg, "rust language", None)
        .await
        .unwrap();
    assert!(result.text().contains("DuckDuckGo search results"));
    assert!(result.text().contains("Rust Programming Language"));
    assert!(result.text().contains("https://www.rust-lang.org/"));
    assert!(result.text().contains("Rust Documentation"));
}

#[tokio::test]
async fn run_duckduckgo_search_tool_empty_query_errors() {
    let client = reqwest::Client::new();
    let cfg = DuckDuckGoSearchToolCfg {
        endpoint: "https://duckduckgo.com/html/",
        timeout_secs: 5,
        default_limit: 5,
        max_snippet_chars: 120,
    };
    let result = run_duckduckgo_search_tool(&client, cfg, "  ", None).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn execute_duckduckgo_search_missing_query_arg_errors() {
    let tools = vec![crate::llm::tools::Tool {
        name: "duckduckgo_search".into(),
        description: "search".into(),
        enabled: true,
        retries: 0,
        backend: crate::config::ToolBackendConfig::DuckDuckGoSearch {
            endpoint: "https://duckduckgo.com/html/".into(),
            timeout_secs: 5,
            default_limit: 3,
            max_snippet_chars: 120,
        },
    }];
    let executor = super::ToolExecutor::new(tools, test_fetcher());
    let id = uuid::Uuid::new_v4();
    let result = executor
        .execute(
            "duckduckgo_search",
            &serde_json::json!({}),
            id,
            std::path::Path::new("/tmp"),
        )
        .await;
    assert!(result.is_err());
}

#[tokio::test]
async fn execute_duckduckgo_search_with_wrong_backend_errors() {
    let tools = vec![crate::llm::tools::Tool {
        name: "duckduckgo_search".into(),
        description: "search".into(),
        enabled: true,
        retries: 0,
        backend: crate::config::ToolBackendConfig::Internal { timeout_secs: 15 },
    }];
    let executor = super::ToolExecutor::new(tools, test_fetcher());
    let id = uuid::Uuid::new_v4();
    let result = executor
        .execute(
            "duckduckgo_search",
            &serde_json::json!({"query":"rust async"}),
            id,
            std::path::Path::new("/tmp"),
        )
        .await;
    assert!(result.is_err());
}

// ── Memory tool tests ─────────────────────────────────────────────────────────

#[tokio::test]
async fn execute_memory_save_and_recall() {
    use std::sync::Arc;
    let store = Arc::new(crate::memory::MemoryStore::new_in_memory().unwrap());
    let mut executor = default_tools(test_fetcher());
    add_memory_tools(&mut executor, store);

    let id = uuid::Uuid::new_v4();
    let dir = std::path::Path::new("/tmp");

    // Save a memory
    let save_result = executor
        .execute(
            "memory_save",
            &serde_json::json!({"key": "user_name", "value": "Alice"}),
            id,
            dir,
        )
        .await
        .unwrap();
    assert!(save_result.text().contains("user_name"));

    // Recall it back
    let recall_result = executor
        .execute(
            "memory_recall",
            &serde_json::json!({"query": "user_name"}),
            id,
            dir,
        )
        .await
        .unwrap();
    assert!(recall_result.text().contains("Alice"));
}

#[tokio::test]
async fn execute_memory_save_missing_key_errors() {
    use std::sync::Arc;
    let store = Arc::new(crate::memory::MemoryStore::new_in_memory().unwrap());
    let mut executor = default_tools(test_fetcher());
    add_memory_tools(&mut executor, store);

    let id = uuid::Uuid::new_v4();
    let result = executor
        .execute(
            "memory_save",
            &serde_json::json!({"value": "no key provided"}),
            id,
            std::path::Path::new("/tmp"),
        )
        .await;
    assert!(result.is_err());
}

#[tokio::test]
async fn execute_memory_recall_missing_query_errors() {
    use std::sync::Arc;
    let store = Arc::new(crate::memory::MemoryStore::new_in_memory().unwrap());
    let mut executor = default_tools(test_fetcher());
    add_memory_tools(&mut executor, store);

    let id = uuid::Uuid::new_v4();
    let result = executor
        .execute(
            "memory_recall",
            &serde_json::json!({}),
            id,
            std::path::Path::new("/tmp"),
        )
        .await;
    assert!(result.is_err());
}

#[tokio::test]
async fn execute_memory_recall_no_results_returns_text() {
    use std::sync::Arc;
    let store = Arc::new(crate::memory::MemoryStore::new_in_memory().unwrap());
    let mut executor = default_tools(test_fetcher());
    add_memory_tools(&mut executor, store);

    let id = uuid::Uuid::new_v4();
    let result = executor
        .execute(
            "memory_recall",
            &serde_json::json!({"query": "xyzzy_nonexistent_42"}),
            id,
            std::path::Path::new("/tmp"),
        )
        .await
        .unwrap();
    assert!(result.text().contains("No memories found"));
}

#[test]
fn add_memory_tools_registers_two_tools() {
    use std::sync::Arc;
    let store = Arc::new(crate::memory::MemoryStore::new_in_memory().unwrap());
    let mut executor = default_tools(test_fetcher());
    let before = executor.active_tool_definitions().len();
    add_memory_tools(&mut executor, store);
    let after = executor.active_tool_definitions().len();
    assert_eq!(after, before + 2);
}

#[tokio::test]
async fn execute_memory_save_without_store_errors() {
    let tools = vec![Tool {
        name: "memory_save".into(),
        description: "save".into(),
        enabled: true,
        retries: 0,
        backend: crate::config::ToolBackendConfig::Memory,
    }];
    let executor = ToolExecutor::new(tools, test_fetcher());
    let result = executor
        .execute(
            "memory_save",
            &serde_json::json!({"key": "k", "value": "v"}),
            uuid::Uuid::new_v4(),
            std::path::Path::new("/tmp"),
        )
        .await;
    assert!(result.is_err());
}

#[tokio::test]
async fn execute_scrape_page_with_memory_backend_errors() {
    let tools = vec![Tool {
        name: "scrape_page".into(),
        description: "scrape".into(),
        enabled: true,
        retries: 0,
        backend: crate::config::ToolBackendConfig::Memory,
    }];
    let executor = ToolExecutor::new(tools, test_fetcher());
    let result = executor
        .execute(
            "scrape_page",
            &serde_json::json!({"url": "https://example.com"}),
            uuid::Uuid::new_v4(),
            std::path::Path::new("/tmp"),
        )
        .await;
    assert!(result.is_err());
}

/// Manual integration test: `TEST_WITH_DDG=1 cargo test duckduckgo_live_search -- --nocapture`.
///
/// Verifies that the real `DuckDuckGo` HTML endpoint returns parseable results.
/// Skipped automatically unless `TEST_WITH_DDG=1` is set.
#[tokio::test]
async fn duckduckgo_live_search() {
    if std::env::var("TEST_WITH_DDG").as_deref() != Ok("1") {
        return;
    }
    let client = reqwest::Client::new();
    let cfg = DuckDuckGoSearchToolCfg {
        endpoint: "https://duckduckgo.com/html/",
        timeout_secs: 15,
        default_limit: 3,
        max_snippet_chars: 320,
    };
    let result = run_duckduckgo_search_tool(&client, cfg, "Rust programming language", Some(3))
        .await
        .expect("DDG live search should succeed");
    let text = result.text();
    println!("DDG result:\n{text}");
    assert!(
        text.contains("DuckDuckGo search results"),
        "Expected formatted results header, got: {text}"
    );
    assert!(!text.is_empty(), "Expected non-empty results");
}

/// Manual integration test: `TEST_WITH_KAGI=1 KAGI_API_TOKEN=<token> cargo test kagi_live`.
///
/// Verifies the real Kagi Search API call succeeds and returns formatted results.
/// Skipped automatically unless `TEST_WITH_KAGI=1` is set.
#[tokio::test]
async fn kagi_live_search() {
    if std::env::var("TEST_WITH_KAGI").as_deref() != Ok("1") {
        return;
    }
    let token =
        std::env::var("KAGI_API_TOKEN").expect("KAGI_API_TOKEN must be set when TEST_WITH_KAGI=1");

    let client = reqwest::Client::new();
    let cfg = KagiSearchToolCfg {
        endpoint: "https://kagi.com/api/v0/search",
        api_token: Some(&token),
        timeout_secs: 15,
        default_limit: 3,
        max_snippet_chars: 320,
    };

    let result = run_kagi_search_tool(&client, cfg, "Rust programming language", Some(3))
        .await
        .expect("Kagi live search should succeed");

    let text = result.text();
    println!("Kagi result:\n{text}");
    assert!(
        text.contains("Kagi web_search results"),
        "Expected formatted results header, got: {text}"
    );
    assert!(!text.is_empty(), "Expected non-empty results");
}
