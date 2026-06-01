//! Parse tests for `[[llm.backends]]` — the recommended vision chain order
//! (`free_router` → `ollama` text → `ollama` vision) and the `vision_supported`
//! flags.

use serde::Deserialize;

use super::{LlmBackendConfig, LlmBackendType};

#[derive(Deserialize)]
struct Backends {
    backends: Vec<LlmBackendConfig>,
}

#[test]
fn three_backend_vision_chain_parses_with_expected_flags() {
    let toml = r#"
[[backends]]
type = "free_router"
api_key = "k"

[[backends]]
type = "ollama"
model = "llama3.2"
base_url = "http://localhost:11434"

[[backends]]
type = "ollama"
model = "llama3.2-vision"
base_url = "http://localhost:11434"
vision_supported = true
circuit_open_secs = 120
"#;

    let parsed: Backends = toml::from_str(toml).expect("config parses");
    let b = &parsed.backends;
    assert_eq!(b.len(), 3);

    assert_eq!(b[0].backend_type, LlmBackendType::FreeRouter);
    // free_router detects vision per-model; the static flag stays false.
    assert!(!b[0].vision_supported);

    assert_eq!(b[1].backend_type, LlmBackendType::Ollama);
    assert!(!b[1].vision_supported, "text ollama is not vision-capable");

    assert_eq!(b[2].backend_type, LlmBackendType::Ollama);
    assert!(
        b[2].vision_supported,
        "vision ollama must route image requests"
    );
    assert_eq!(b[2].circuit_open_secs, 120);
}

#[test]
fn circuit_open_secs_defaults_to_300() {
    let b: LlmBackendConfig = toml::from_str(
        r#"
type = "openrouter"
model = "openai/gpt-4o-mini"
"#,
    )
    .expect("backend parses");
    assert_eq!(b.circuit_open_secs, 300);
}
