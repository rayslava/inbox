//! Opt-in live check of the `llama_cpp` backend against a real llama.cpp
//! `server`. Skipped unless `LLAMACPP_BASE_URL` is set (e.g.
//! `http://127.0.0.1:32000/v1`), mirroring the repo's `TEST_WITH_OLLAMA`
//! convention — the default suite makes no real network calls.
//!
//! Run: `LLAMACPP_BASE_URL=http://127.0.0.1:32000/v1 cargo test --test \
//! test_llama_cpp_live -- --nocapture`

use inbox::config::{FallbackMode, LlmBackendConfig, LlmBackendType};
use inbox::llm::LlmChain;
use inbox::llm::openrouter::OpenRouterClient;
use inbox_core::LlmBackend;

fn llama_cpp_config(base_url: String) -> LlmBackendConfig {
    LlmBackendConfig {
        backend_type: LlmBackendType::LlamaCpp,
        model: "local".into(),
        api_key: None,
        base_url,
        retries: 1,
        timeout_secs: 120,
        think: None,
        think_timeout_secs: None,
        thinking_supported: false,
        vision_supported: false,
        max_concurrent: Some(1),
        context_size: None,
        format: None,
        connect_timeout_secs: 10,
        circuit_open_secs: 0,
        api_url: String::new(),
        parallel_fanout: 3,
        per_model_retries: 2,
        min_refresh_interval_secs: 300,
        min_context_length: 0,
        prefer_structured_outputs: false,
        prefer_reasoning: false,
    }
}

#[tokio::test]
async fn llama_cpp_completes_text_against_real_server() {
    let Ok(base_url) = std::env::var("LLAMACPP_BASE_URL") else {
        eprintln!("skipping: set LLAMACPP_BASE_URL to run (e.g. http://127.0.0.1:32000/v1)");
        return;
    };

    let client = OpenRouterClient::from_config_labeled(&llama_cpp_config(base_url), "llama_cpp")
        .expect("build llama_cpp client");
    let chain = LlmChain::new(vec![Box::new(client)], FallbackMode::Raw, 5, None, 1, 2, 0);

    // The `/ask` brain path: plain-text completion, no JSON contract.
    // The `/ask` brain path uses the core `LlmBackend` trait (Result, not the
    // inherent Option), so failures surface their cause here.
    let (answer, produced_by) = LlmBackend::complete_text(
        &chain,
        "You are a terse assistant. Answer in one short sentence.",
        "What is the capital of France?",
    )
    .await
    .expect("llama.cpp completion");

    eprintln!("llama.cpp produced_by={produced_by}\nanswer={answer}");
    assert!(!answer.trim().is_empty(), "answer must not be empty");
    assert!(
        produced_by.starts_with("llama_cpp:"),
        "provenance must carry the llama_cpp label, got {produced_by}"
    );
}
