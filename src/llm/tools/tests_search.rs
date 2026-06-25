use wiremock::matchers::{body_partial_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use crate::config::{ToolBackendConfig, UrlFetchConfig};
use crate::pipeline::url_fetcher::UrlFetcher;

use super::runners::{KeenableSearchToolCfg, run_keenable_search_tool};
use super::{Tool, ToolExecutor};

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

fn keenable_results_body() -> serde_json::Value {
    serde_json::json!({
        "results": [
            {
                "title": "Rust",
                "url": "https://www.rust-lang.org/",
                "description": "Long description.",
                "snippet": "A language empowering everyone."
            },
            {
                "title": "Rust Docs",
                "url": "https://doc.rust-lang.org/",
                "description": "The Rust reference and book.",
                "snippet": ""
            },
            {
                "title": "Crates",
                "url": "https://crates.io/",
                "description": "Package registry.",
                "snippet": "Find packages."
            }
        ]
    })
}

#[tokio::test]
async fn run_keenable_search_tool_formats_results() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/search"))
        .and(header("X-API-Key", "keen-key"))
        .and(body_partial_json(
            serde_json::json!({ "query": "rust language" }),
        ))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(keenable_results_body()),
        )
        .mount(&server)
        .await;

    let client = reqwest::Client::new();
    let cfg = KeenableSearchToolCfg {
        endpoint: &format!("{}/v1/search", server.uri()),
        api_key: Some("keen-key"),
        timeout_secs: 5,
        default_limit: 5,
        max_snippet_chars: 120,
    };

    let result = run_keenable_search_tool(&client, cfg, "rust language", None)
        .await
        .expect("keenable search ok");
    let text = result.text();
    assert!(text.contains("Keenable keenable_search results"));
    assert!(text.contains("Rust"));
    assert!(text.contains("https://www.rust-lang.org/"));
    assert!(text.contains("A language empowering everyone."));
}

#[tokio::test]
async fn execute_keenable_search_dispatches_to_backend() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/search"))
        .and(header("X-API-Key", "keen-key"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(keenable_results_body()),
        )
        .mount(&server)
        .await;

    let tools = vec![Tool {
        name: "keenable_search".into(),
        description: "search".into(),
        enabled: true,
        retries: 0,
        backend: ToolBackendConfig::KeenableSearch {
            endpoint: format!("{}/v1/search", server.uri()),
            api_key: Some("keen-key".into()),
            timeout_secs: 5,
            default_limit: 5,
            max_snippet_chars: 120,
        },
    }];
    let executor = ToolExecutor::new(tools, test_fetcher()).expect("build executor");
    let id = uuid::Uuid::new_v4();
    let result = executor
        .execute(
            "keenable_search",
            &serde_json::json!({"query":"rust language"}),
            id,
            std::path::Path::new("/tmp"),
            "",
        )
        .await
        .expect("keenable dispatch ok");
    assert!(result.text().contains("Keenable keenable_search results"));
    assert!(result.text().contains("https://www.rust-lang.org/"));
}

#[tokio::test]
async fn run_keenable_search_tool_falls_back_to_description() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/search"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(keenable_results_body()),
        )
        .mount(&server)
        .await;

    let client = reqwest::Client::new();
    let cfg = KeenableSearchToolCfg {
        endpoint: &format!("{}/v1/search", server.uri()),
        api_key: Some("keen-key"),
        timeout_secs: 5,
        default_limit: 5,
        max_snippet_chars: 120,
    };

    let result = run_keenable_search_tool(&client, cfg, "rust", None)
        .await
        .expect("keenable search ok");
    // Second result has an empty snippet -> the description is used instead.
    assert!(result.text().contains("The Rust reference and book."));
}

#[tokio::test]
async fn run_keenable_search_tool_truncates_to_limit() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/search"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(keenable_results_body()),
        )
        .mount(&server)
        .await;

    let client = reqwest::Client::new();
    let cfg = KeenableSearchToolCfg {
        endpoint: &format!("{}/v1/search", server.uri()),
        api_key: Some("keen-key"),
        timeout_secs: 5,
        default_limit: 5,
        max_snippet_chars: 120,
    };

    let result = run_keenable_search_tool(&client, cfg, "rust", Some(2))
        .await
        .expect("keenable search ok");
    let text = result.text();
    assert!(text.contains("1. Rust"));
    assert!(text.contains("2. Rust Docs"));
    assert!(!text.contains("Crates"));
}

#[tokio::test]
async fn run_keenable_search_tool_empty_results() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/search"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(serde_json::json!({ "results": [] })),
        )
        .mount(&server)
        .await;

    let client = reqwest::Client::new();
    let cfg = KeenableSearchToolCfg {
        endpoint: &format!("{}/v1/search", server.uri()),
        api_key: Some("keen-key"),
        timeout_secs: 5,
        default_limit: 5,
        max_snippet_chars: 120,
    };

    let result = run_keenable_search_tool(&client, cfg, "rust", None)
        .await
        .expect("keenable search ok");
    assert!(result.text().contains("no results"));
}

#[tokio::test]
async fn run_keenable_search_tool_non_2xx_errors() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/search"))
        .respond_with(
            ResponseTemplate::new(401)
                .insert_header("content-type", "application/json")
                .set_body_json(serde_json::json!({ "error": "Missing API key" })),
        )
        .mount(&server)
        .await;

    let client = reqwest::Client::new();
    let cfg = KeenableSearchToolCfg {
        endpoint: &format!("{}/v1/search", server.uri()),
        api_key: Some("bad-key"),
        timeout_secs: 5,
        default_limit: 5,
        max_snippet_chars: 120,
    };

    let result = run_keenable_search_tool(&client, cfg, "rust", None).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn run_keenable_search_tool_requires_non_empty_key_when_configured() {
    let client = reqwest::Client::new();
    let cfg = KeenableSearchToolCfg {
        endpoint: "https://api.keenable.ai/v1/search",
        api_key: Some(""),
        timeout_secs: 5,
        default_limit: 5,
        max_snippet_chars: 120,
    };
    let result = run_keenable_search_tool(&client, cfg, "rust", None).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn run_keenable_search_tool_requires_key() {
    let client = reqwest::Client::new();
    let cfg = KeenableSearchToolCfg {
        endpoint: "https://api.keenable.ai/v1/search",
        api_key: None,
        timeout_secs: 5,
        default_limit: 5,
        max_snippet_chars: 120,
    };
    let result = run_keenable_search_tool(&client, cfg, "rust", None).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn run_keenable_search_tool_empty_query_errors() {
    let client = reqwest::Client::new();
    let cfg = KeenableSearchToolCfg {
        endpoint: "https://api.keenable.ai/v1/search",
        api_key: Some("keen-key"),
        timeout_secs: 5,
        default_limit: 5,
        max_snippet_chars: 120,
    };
    let result = run_keenable_search_tool(&client, cfg, "  ", None).await;
    assert!(result.is_err());
}
