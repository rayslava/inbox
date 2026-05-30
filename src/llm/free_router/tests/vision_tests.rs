//! Task 2: vision-aware pool partition, `OpenRouter` modality join, and
//! vision-only candidate selection.

use std::time::Duration;

use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use crate::llm::free_router::FreeRouterClient;
use crate::llm::free_router::pool::{
    PoolPreferences, PoolState, build_pool, fetch_pool, fetch_vision_model_ids,
};
use crate::llm::{LlmClient, LlmRequest};

use super::fixtures::{backend_cfg, sample_model, sample_vision_model};

fn prefs() -> PoolPreferences {
    PoolPreferences {
        min_context_length: 0,
        prefer_structured_outputs: false,
        prefer_reasoning: false,
    }
}

#[test]
fn build_pool_partitions_vision_models() {
    let mut vision = sample_model("a/vision", 1000.0, 64_000, true, false, false, "passed");
    vision.supports_vision = true;
    let models = vec![
        vision,
        sample_model("b/text", 900.0, 64_000, true, false, false, "passed"),
    ];
    let pool = build_pool(models, prefs());
    assert_eq!(pool.vision_models.len(), 1);
    assert_eq!(pool.vision_models[0].id, "a/vision");
    // Vision models remain part of the general superset.
    assert_eq!(pool.general_models.len(), 2);
}

async fn mock_openrouter_models(server: &MockServer) {
    Mock::given(method("GET"))
        .and(path("/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [
                {"id": "a/vision", "architecture": {"input_modalities": ["text", "image"]}},
                {"id": "b/text", "architecture": {"input_modalities": ["text"]}}
            ]
        })))
        .mount(server)
        .await;
}

#[tokio::test]
async fn fetch_vision_model_ids_selects_image_inputs() {
    let server = MockServer::start().await;
    mock_openrouter_models(&server).await;

    let ids = fetch_vision_model_ids(
        &reqwest::Client::new(),
        &server.uri(),
        Duration::from_secs(5),
    )
    .await;
    assert!(ids.contains("a/vision"));
    assert!(!ids.contains("b/text"));
}

#[tokio::test]
async fn fetch_vision_model_ids_graceful_on_http_error() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/models"))
        .respond_with(ResponseTemplate::new(503))
        .mount(&server)
        .await;

    let ids = fetch_vision_model_ids(
        &reqwest::Client::new(),
        &server.uri(),
        Duration::from_secs(5),
    )
    .await;
    assert!(ids.is_empty());
}

#[tokio::test]
async fn fetch_vision_model_ids_graceful_on_unreachable() {
    let ids = fetch_vision_model_ids(
        &reqwest::Client::new(),
        "http://127.0.0.1:0",
        Duration::from_millis(200),
    )
    .await;
    assert!(ids.is_empty());
}

#[tokio::test]
async fn fetch_pool_joins_vision_metadata() {
    let server = MockServer::start().await;
    // shir-man index at "/"; OpenRouter models at "/models".
    Mock::given(method("GET"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "models": [
                {"id": "a/vision", "supportsTools": true, "supportsToolChoice": true, "healthStatus": "passed"},
                {"id": "b/text", "supportsTools": true, "supportsToolChoice": true, "healthStatus": "passed"}
            ]
        })))
        .mount(&server)
        .await;
    mock_openrouter_models(&server).await;

    let api_url = format!("{}/", server.uri());
    let pool = fetch_pool(
        &reqwest::Client::new(),
        &api_url,
        &server.uri(),
        Duration::from_secs(5),
        prefs(),
    )
    .await
    .expect("pool");

    assert_eq!(pool.vision_models.len(), 1);
    assert_eq!(pool.vision_models[0].id, "a/vision");
    assert!(pool.vision_models[0].supports_vision);
}

#[tokio::test]
async fn candidate_models_vision_intersects_tools() {
    let pool = PoolState {
        tool_models: vec![],
        general_models: vec![],
        vision_models: vec![
            sample_vision_model("v/tool", 100.0, true),
            sample_vision_model("v/notool", 90.0, false),
        ],
    };
    let cfg = backend_cfg("http://unused.invalid/list", "http://unused.invalid", 1);
    let client = FreeRouterClient::with_pool(&cfg, pool);

    // needs_tools + needs_vision ⇒ only the vision model that also has tools.
    let both = client.candidate_models(true, true).await;
    assert_eq!(both.len(), 1);
    assert_eq!(both[0].id, "v/tool");

    // needs_vision only ⇒ all vision models.
    let vision_only = client.candidate_models(false, true).await;
    assert_eq!(vision_only.len(), 2);
}

#[tokio::test]
async fn candidate_models_empty_when_no_vision() {
    let pool = PoolState {
        tool_models: vec![sample_model(
            "t/only", 100.0, 16_000, true, false, false, "passed",
        )],
        general_models: vec![sample_model(
            "t/only", 100.0, 16_000, true, false, false, "passed",
        )],
        vision_models: vec![],
    };
    let cfg = backend_cfg("http://unused.invalid/list", "http://unused.invalid", 1);
    let client = FreeRouterClient::with_pool(&cfg, pool);

    let vision = client.candidate_models(false, true).await;
    assert!(vision.is_empty());
    assert!(!client.vision_supported());
}

#[tokio::test]
async fn vision_supported_reflects_pool() {
    let cfg = backend_cfg("http://unused.invalid/list", "http://unused.invalid", 1);
    let pool = PoolState {
        tool_models: vec![],
        general_models: vec![],
        vision_models: vec![sample_vision_model("v/x", 100.0, true)],
    };
    let client = FreeRouterClient::with_pool(&cfg, pool);
    assert!(client.vision_supported());
}

#[test]
fn degraded_fallback_has_no_vision_models() {
    let pool = PoolState::degraded_fallback();
    assert!(pool.vision_models.is_empty());
    assert!(!pool.is_empty());
}

/// A vision request against a healthy pool that simply lacks vision models must
/// fail fast without forcing a recovery refresh (the list endpoint is asserted
/// to receive zero calls). Prevents per-retry refresh storms before the chain
/// (Task 3) starts skipping free-router for unsupported vision.
#[tokio::test]
async fn vision_request_with_no_vision_models_fails_fast_without_refresh() {
    let list = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(200))
        .expect(0)
        .mount(&list)
        .await;

    let pool = PoolState {
        tool_models: vec![sample_model(
            "t/only", 100.0, 16_000, true, false, false, "passed",
        )],
        general_models: vec![sample_model(
            "t/only", 100.0, 16_000, true, false, false, "passed",
        )],
        vision_models: vec![],
    };
    let cfg = backend_cfg(&format!("{}/", list.uri()), "http://unused.invalid", 1);
    let client = FreeRouterClient::with_pool(&cfg, pool);

    let mut req = LlmRequest::simple("sys", "user");
    req.images.push(("image/png".into(), "Zm9v".into()));

    let result = client.complete(req).await;
    assert!(result.is_err());
    // `list` is dropped at end of scope; its expect(0) verifies no refresh fired.
}
