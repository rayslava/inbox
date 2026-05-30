//! Task 1 vision-capability surface: trait default, per-backend config flag,
//! and `LlmRequest` vision derivation.

use crate::config::LlmBackendConfig;
use crate::message::LlmResponse;

use super::mock::MockLlm;
use super::ollama::OllamaClient;
use super::openrouter::OpenRouterClient;
use super::{LlmClient, LlmRequest};

fn backend_config(toml: &str) -> LlmBackendConfig {
    toml::from_str(toml).expect("valid backend config")
}

#[test]
fn trait_default_vision_unsupported() {
    let client = MockLlm::new(LlmResponse {
        title: "t".into(),
        tags: vec![],
        summary: "s".into(),
        excerpt: None,
        produced_by: "mock".into(),
    });
    assert!(!client.vision_supported());
}

#[test]
fn openrouter_reflects_vision_flag() {
    let enabled = backend_config(
        r#"type = "openrouter"
model = "x/y"
vision_supported = true
"#,
    );
    let disabled = backend_config(
        r#"type = "openrouter"
model = "x/y"
"#,
    );
    assert!(
        OpenRouterClient::from_config(&enabled)
            .expect("client")
            .vision_supported()
    );
    assert!(
        !OpenRouterClient::from_config(&disabled)
            .expect("client")
            .vision_supported()
    );
}

#[test]
fn ollama_reflects_vision_flag() {
    let enabled = backend_config(
        r#"type = "ollama"
model = "llava"
base_url = "http://localhost:11434"
vision_supported = true
"#,
    );
    let disabled = backend_config(
        r#"type = "ollama"
model = "llama3"
base_url = "http://localhost:11434"
"#,
    );
    assert!(
        OllamaClient::from_config(&enabled)
            .expect("client")
            .vision_supported()
    );
    assert!(
        !OllamaClient::from_config(&disabled)
            .expect("client")
            .vision_supported()
    );
}

#[test]
fn request_needs_vision_tracks_images() {
    let mut req = LlmRequest::simple("system", "user");
    assert!(!req.needs_vision());
    assert!(!req.has_image_text);

    req.images.push(("image/png".into(), "Zm9v".into()));
    assert!(req.needs_vision());

    // Stripping images flips the derived flag back — no desync possible.
    req.images.clear();
    assert!(!req.needs_vision());
}
