use crate::config::UrlFetchConfig;
use crate::pipeline::url_fetcher::UrlFetcher;
use wiremock::matchers::{header, method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

use super::runners::{
    DuckDuckGoSearchToolCfg, KagiSearchToolCfg, run_duckduckgo_search_tool, run_kagi_search_tool,
};
use super::{Tool, ToolExecutor, add_memory_tools, default_tools};

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
            "",
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
            "",
        )
        .await;
    assert!(result.is_err());
}

#[tokio::test]
async fn execute_memory_save_and_recall() {
    use std::sync::Arc;
    let store = Arc::new(crate::memory::MemoryStore::new_in_memory().unwrap());
    let mut executor = default_tools(test_fetcher());
    add_memory_tools(&mut executor, store);

    let id = uuid::Uuid::new_v4();
    let dir = std::path::Path::new("/tmp");

    let save_result = executor
        .execute(
            "memory_save",
            &serde_json::json!({"key": "user_name", "value": "Alice"}),
            id,
            dir,
            "",
        )
        .await
        .unwrap();
    assert!(save_result.text().contains("user_name"));

    let recall_result = executor
        .execute(
            "memory_recall",
            &serde_json::json!({"query": "user_name"}),
            id,
            dir,
            "",
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
            "",
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
            "",
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
            "",
        )
        .await
        .unwrap();
    assert!(result.text().contains("No memories found"));
}

#[test]
fn add_memory_tools_registers_four_tools() {
    use std::sync::Arc;
    let store = Arc::new(crate::memory::MemoryStore::new_in_memory().unwrap());
    let mut executor = default_tools(test_fetcher());
    let before = executor.active_tool_definitions().len();
    add_memory_tools(&mut executor, store);
    let after = executor.active_tool_definitions().len();
    assert_eq!(after, before + 4);
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
            "",
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
            "",
        )
        .await;
    assert!(result.is_err());
}

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
