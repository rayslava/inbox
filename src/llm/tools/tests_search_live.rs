//! Opt-in live web-search tests (require `TEST_WITH_DDG` / `TEST_WITH_KAGI` /
//! `TEST_WITH_KEENABLE`).

use super::runners::{
    DuckDuckGoSearchToolCfg, KagiSearchToolCfg, KeenableSearchToolCfg, run_duckduckgo_search_tool,
    run_kagi_search_tool, run_keenable_search_tool,
};

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

#[tokio::test]
async fn keenable_live_search() {
    if std::env::var("TEST_WITH_KEENABLE").as_deref() != Ok("1") {
        return;
    }
    let key = std::env::var("KEENABLE_API_KEY")
        .expect("KEENABLE_API_KEY must be set when TEST_WITH_KEENABLE=1");

    let client = reqwest::Client::new();
    let cfg = KeenableSearchToolCfg {
        endpoint: "https://api.keenable.ai/v1/search",
        api_key: Some(&key),
        timeout_secs: 15,
        default_limit: 3,
        max_snippet_chars: 320,
    };

    let result = run_keenable_search_tool(&client, cfg, "Rust programming language", Some(3))
        .await
        .expect("Keenable live search should succeed");

    let text = result.text();
    println!("Keenable result:\n{text}");
    assert!(
        text.contains("Keenable keenable_search results"),
        "Expected formatted results header, got: {text}"
    );
    assert!(!text.is_empty(), "Expected non-empty results");
}
