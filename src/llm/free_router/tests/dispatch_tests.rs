//! HTTP-backed tests for dispatch, hedging, refresh, and degraded boot.

use std::time::Duration;

use serde_json::json;
use wiremock::matchers::{body_partial_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use crate::llm::free_router::FreeRouterClient;
use crate::llm::free_router::pool::{FALLBACK_MODEL_ID, PoolState};
use crate::llm::{LlmClient, LlmCompletion, LlmRequest};

use super::fixtures::{backend_cfg, fixed_pool, json_choice, sample_model};

#[tokio::test]
async fn tool_call_uses_tool_pool() {
    let chat = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_partial_json(json!({ "model": "a/tool-high" })))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(json_choice(r#"{"title":"T","tags":[],"summary":"S"}"#)),
        )
        .expect(1)
        .mount(&chat)
        .await;

    let cfg = backend_cfg("http://unused.invalid/list", &chat.uri(), 1);
    let client = FreeRouterClient::with_pool(&cfg, fixed_pool());

    let mut req = LlmRequest::simple("sys", "user");
    req.tool_definitions = vec![json!({"type":"function","function":{"name":"x","parameters":{}}})];

    let result = client.complete(req).await.unwrap();
    assert!(matches!(result, LlmCompletion::Message(_)));
}

#[tokio::test]
async fn non_tool_call_can_use_any_model() {
    let chat = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(json_choice(r#"{"title":"T","tags":[],"summary":"S"}"#)),
        )
        .mount(&chat)
        .await;

    let cfg = backend_cfg("http://unused.invalid/list", &chat.uri(), 1);
    let client = FreeRouterClient::with_pool(&cfg, fixed_pool());

    let req = LlmRequest::simple("sys", "user");
    let result = client.complete(req).await.unwrap();
    assert!(matches!(result, LlmCompletion::Message(_)));
}

#[tokio::test]
async fn hedged_fanout_picks_fastest_success() {
    let chat = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_partial_json(json!({ "model": "a/tool-high" })))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(json_choice(r#"{"title":"SLOW","tags":[],"summary":"S"}"#))
                .set_delay(Duration::from_millis(800)),
        )
        .mount(&chat)
        .await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_partial_json(json!({ "model": "b/tool-low" })))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(json_choice(r#"{"title":"FAST","tags":[],"summary":"S"}"#)),
        )
        .mount(&chat)
        .await;

    let cfg = backend_cfg("http://unused.invalid/list", &chat.uri(), 2);
    let client = FreeRouterClient::with_pool(&cfg, fixed_pool());

    let mut req = LlmRequest::simple("sys", "user");
    req.tool_definitions = vec![json!({"type":"function","function":{"name":"x","parameters":{}}})];

    let started = std::time::Instant::now();
    let result = client.complete(req).await.unwrap();
    let elapsed = started.elapsed();

    assert!(
        elapsed < Duration::from_millis(700),
        "expected fast candidate to win; total elapsed = {elapsed:?}"
    );
    match result {
        LlmCompletion::Message(r) => assert_eq!(r.title, "FAST"),
        LlmCompletion::ToolCalls(_) => panic!("unexpected tool call result"),
    }
}

#[tokio::test]
async fn all_fail_triggers_refresh_and_retry() {
    let list = MockServer::start().await;
    let chat = MockServer::start().await;

    // Replacement pool advertised by the refreshed list endpoint.
    let refreshed_list = json!({
        "models": [
            {
                "id": "fresh/model",
                "score": 1500,
                "contextLength": 32000,
                "supportsTools": true,
                "supportsToolChoice": true,
                "supportsStructuredOutputs": false,
                "supportsReasoning": false,
                "healthStatus": "passed"
            }
        ]
    });
    Mock::given(method("GET"))
        .and(path("/"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(refreshed_list),
        )
        .mount(&list)
        .await;

    // Original pool members always fail.
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_partial_json(json!({ "model": "a/tool-high" })))
        .respond_with(ResponseTemplate::new(500).set_body_string("boom"))
        .mount(&chat)
        .await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_partial_json(json!({ "model": "b/tool-low" })))
        .respond_with(ResponseTemplate::new(500).set_body_string("boom"))
        .mount(&chat)
        .await;
    // The refreshed model succeeds.
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_partial_json(json!({ "model": "fresh/model" })))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(json_choice(
                    r#"{"title":"REFRESHED","tags":[],"summary":"S"}"#,
                )),
        )
        .expect(1)
        .mount(&chat)
        .await;

    let cfg = backend_cfg(&format!("{}/", list.uri()), &chat.uri(), 1);
    let client = FreeRouterClient::with_pool(&cfg, fixed_pool());

    let mut req = LlmRequest::simple("sys", "user");
    req.tool_definitions = vec![json!({"type":"function","function":{"name":"x","parameters":{}}})];

    let result = client.complete(req).await.unwrap();
    match result {
        LlmCompletion::Message(r) => assert_eq!(r.title, "REFRESHED"),
        LlmCompletion::ToolCalls(_) => panic!("unexpected tool call result"),
    }
}

#[tokio::test]
async fn exhaustion_without_refresh_change_propagates_error() {
    let list = MockServer::start().await;
    let chat = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(500).set_body_string("still broken"))
        .mount(&list)
        .await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(500).set_body_string("boom"))
        .mount(&chat)
        .await;

    let cfg = backend_cfg(&format!("{}/", list.uri()), &chat.uri(), 1);
    let client = FreeRouterClient::with_pool(&cfg, fixed_pool());
    let req = LlmRequest::simple("sys", "user");
    let result = client.complete(req).await;
    assert!(result.is_err());
}

#[tokio::test(flavor = "multi_thread")]
async fn degraded_bootstrap_when_list_unreachable() {
    let list = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(500).set_body_string("unavailable"))
        .mount(&list)
        .await;

    let cfg = backend_cfg(&format!("{}/", list.uri()), "http://unused.invalid", 1);
    let client = FreeRouterClient::from_config(&cfg);

    let tool_candidates = client.candidate_models(true).await;
    let general_candidates = client.candidate_models(false).await;
    assert_eq!(tool_candidates.len(), 1);
    assert_eq!(tool_candidates[0].id, FALLBACK_MODEL_ID);
    assert_eq!(general_candidates.len(), 1);
}

#[tokio::test]
async fn hard_error_short_circuits_retries() {
    let chat = MockServer::start().await;
    // Every attempt returns 401 — classified as a hard error. With
    // per_model_retries=2, we'd normally see up to 3 hits per model; the
    // short-circuit path must stop at 1 and propagate.
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_partial_json(json!({ "model": "a/tool-high" })))
        .respond_with(ResponseTemplate::new(401).set_body_string("unauthorized"))
        .expect(1)
        .mount(&chat)
        .await;

    let mut cfg = backend_cfg("http://unused.invalid/list", &chat.uri(), 1);
    cfg.per_model_retries = 2;

    let pool = PoolState {
        tool_models: vec![sample_model(
            "a/tool-high",
            1000.0,
            64_000,
            true,
            true,
            false,
            "passed",
        )],
        general_models: vec![sample_model(
            "a/tool-high",
            1000.0,
            64_000,
            true,
            true,
            false,
            "passed",
        )],
    };
    let client = FreeRouterClient::with_pool(&cfg, pool);

    let mut req = LlmRequest::simple("sys", "user");
    req.tool_definitions = vec![json!({"type":"function","function":{"name":"x","parameters":{}}})];

    let result = client.complete(req).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn tool_call_falls_back_to_general_when_no_tool_models() {
    let chat = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_partial_json(json!({ "model": "c/no-tool" })))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(json_choice(r#"{"title":"T","tags":[],"summary":"S"}"#)),
        )
        .expect(1)
        .mount(&chat)
        .await;

    let general_only = PoolState {
        tool_models: vec![],
        general_models: vec![sample_model(
            "c/no-tool",
            900.0,
            128_000,
            false,
            false,
            false,
            "passed",
        )],
    };
    let cfg = backend_cfg("http://unused.invalid/list", &chat.uri(), 1);
    let client = FreeRouterClient::with_pool(&cfg, general_only);

    let mut req = LlmRequest::simple("sys", "user");
    req.tool_definitions = vec![json!({"type":"function","function":{"name":"x","parameters":{}}})];

    let result = client.complete(req).await.unwrap();
    assert!(matches!(result, LlmCompletion::Message(_)));
}

#[tokio::test]
async fn thinking_supported_reflects_pool_and_preference() {
    let reasoning_pool = PoolState {
        tool_models: vec![sample_model(
            "r/reasoning",
            1200.0,
            64_000,
            true,
            false,
            true,
            "passed",
        )],
        general_models: vec![sample_model(
            "r/reasoning",
            1200.0,
            64_000,
            true,
            false,
            true,
            "passed",
        )],
    };

    // prefer_reasoning=false ⇒ thinking_supported is always false.
    let cfg_off = backend_cfg("http://unused.invalid/list", "http://unused.invalid", 1);
    let client_off = FreeRouterClient::with_pool(&cfg_off, reasoning_pool.clone());
    assert!(!client_off.thinking_supported());

    // prefer_reasoning=true with a reasoning-capable model in the pool ⇒ true.
    let mut cfg_on = backend_cfg("http://unused.invalid/list", "http://unused.invalid", 1);
    cfg_on.prefer_reasoning = true;
    let client_on = FreeRouterClient::with_pool(&cfg_on, reasoning_pool);
    assert!(client_on.thinking_supported());

    // prefer_reasoning=true but no reasoning models ⇒ false.
    let no_reasoning_pool = PoolState {
        tool_models: vec![sample_model(
            "x/plain", 900.0, 16_000, true, false, false, "passed",
        )],
        general_models: vec![sample_model(
            "x/plain", 900.0, 16_000, true, false, false, "passed",
        )],
    };
    let client_no = FreeRouterClient::with_pool(&cfg_on, no_reasoning_pool);
    assert!(!client_no.thinking_supported());
}
