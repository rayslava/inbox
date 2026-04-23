//! Shared fixtures for `free_router` tests.

use crate::config::{LlmBackendConfig, LlmBackendType};

use crate::llm::free_router::pool::{FreeModel, PoolState};

pub(super) fn sample_model(
    id: &str,
    score: f64,
    context_length: usize,
    supports_tools: bool,
    supports_structured: bool,
    supports_reasoning: bool,
    health: &str,
) -> FreeModel {
    FreeModel {
        id: id.into(),
        score,
        context_length,
        supports_tools,
        supports_tool_choice: supports_tools,
        supports_structured_outputs: supports_structured,
        supports_reasoning,
        health_status: health.into(),
    }
}

pub(super) fn fixed_pool() -> PoolState {
    PoolState {
        tool_models: vec![
            sample_model("a/tool-high", 1000.0, 64_000, true, true, false, "passed"),
            sample_model("b/tool-low", 800.0, 32_000, true, true, false, "passed"),
        ],
        general_models: vec![
            sample_model("a/tool-high", 1000.0, 64_000, true, true, false, "passed"),
            sample_model("c/no-tool", 900.0, 128_000, false, false, false, "passed"),
            sample_model("b/tool-low", 800.0, 32_000, true, true, false, "passed"),
        ],
    }
}

pub(super) fn backend_cfg(api_url: &str, base_url: &str, fanout: usize) -> LlmBackendConfig {
    LlmBackendConfig {
        backend_type: LlmBackendType::FreeRouter,
        model: "free_router:dynamic".into(),
        api_key: Some("test-key".into()),
        base_url: base_url.into(),
        retries: 1,
        timeout_secs: 5,
        think: None,
        think_timeout_secs: None,
        thinking_supported: false,
        max_concurrent: None,
        context_size: None,
        connect_timeout_secs: 5,
        circuit_open_secs: 0,
        api_url: api_url.into(),
        parallel_fanout: fanout,
        per_model_retries: 0,
        min_refresh_interval_secs: 0,
        min_context_length: 0,
        prefer_structured_outputs: false,
        prefer_reasoning: false,
    }
}

pub(super) fn json_choice(text: &str) -> serde_json::Value {
    serde_json::json!({
        "choices": [{
            "message": { "content": text }
        }]
    })
}
