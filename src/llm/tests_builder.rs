use crate::config::{
    AdaptersConfig, AdminConfig, Config, FallbackMode, GeneralConfig, LlmBackendConfig,
    LlmBackendType, LlmConfig, LlmPromptsConfig, PipelineConfig, SyncthingConfig, ToolingConfig,
    UrlFetchConfig, WebUiConfig,
};

use super::builder::{BuildResult, build_chain};

fn test_config(backends: Vec<LlmBackendConfig>, memory_enabled: bool) -> Config {
    Config {
        general: GeneralConfig {
            output_file: std::path::PathBuf::from("/tmp/inbox-builder-test.org"),
            attachments_dir: std::path::PathBuf::from("/tmp/inbox-builder-att"),
            log_level: "info".into(),
            log_format: "pretty".into(),
        },
        admin: AdminConfig::default(),
        web_ui: WebUiConfig::default(),
        pipeline: PipelineConfig::default(),
        llm: LlmConfig {
            fallback: FallbackMode::Raw,
            url_content_max_chars: 4000,
            max_tool_turns: 5,
            max_llm_tool_depth: 1,
            inner_retries: 0,
            vision_max_bytes: 5 * 1024 * 1024,
            tool_result_max_chars: 0,
            prompts: LlmPromptsConfig::default(),
            backends,
        },
        adapters: AdaptersConfig::default(),
        url_fetch: UrlFetchConfig::default(),
        syncthing: SyncthingConfig::default(),
        tooling: ToolingConfig::default(),
        memory: crate::config::MemoryConfig {
            enabled: memory_enabled,
            ..crate::config::MemoryConfig::default()
        },
    }
}

fn openrouter_backend() -> LlmBackendConfig {
    LlmBackendConfig {
        backend_type: LlmBackendType::Openrouter,
        model: "test/model".into(),
        api_key: Some("test-key".into()),
        base_url: "https://openrouter.ai/api/v1".into(),
        retries: 1,
        timeout_secs: 10,
        think: None,
        think_timeout_secs: None,
        thinking_supported: false,
        max_concurrent: None,
        context_size: None,
        connect_timeout_secs: 10,
        circuit_open_secs: 0,
    }
}

fn ollama_backend() -> LlmBackendConfig {
    LlmBackendConfig {
        backend_type: LlmBackendType::Ollama,
        model: "llama3".into(),
        api_key: None,
        base_url: "http://localhost:11434".into(),
        retries: 1,
        timeout_secs: 30,
        think: None,
        think_timeout_secs: None,
        thinking_supported: false,
        max_concurrent: Some(1),
        context_size: Some(4096),
        connect_timeout_secs: 10,
        circuit_open_secs: 0,
    }
}

#[tokio::test]
async fn build_chain_no_backends_no_memory() {
    let cfg = test_config(vec![], false);
    let BuildResult {
        chain,
        memory_store,
    } = build_chain(&cfg);
    assert!(memory_store.is_none());
    assert_eq!(chain.max_tool_turns(), 5);
}

#[tokio::test]
async fn build_chain_with_openrouter_backend() {
    let cfg = test_config(vec![openrouter_backend()], false);
    let BuildResult {
        chain,
        memory_store,
    } = build_chain(&cfg);
    assert!(memory_store.is_none());
    // Chain should have 1 backend.
    assert_eq!(chain.max_tool_turns(), 5);
}

#[tokio::test]
async fn build_chain_with_ollama_backend() {
    let cfg = test_config(vec![ollama_backend()], false);
    let result = build_chain(&cfg);
    assert!(result.memory_store.is_none());
}

#[tokio::test]
async fn build_chain_multiple_backends() {
    let cfg = test_config(vec![openrouter_backend(), ollama_backend()], false);
    let result = build_chain(&cfg);
    assert!(result.memory_store.is_none());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn build_chain_with_memory_enabled() {
    let dir = tempfile::tempdir().unwrap();
    let mut cfg = test_config(vec![], true);
    cfg.general.attachments_dir = dir.path().to_path_buf();
    let result = build_chain(&cfg);
    // Memory store should open successfully with local Grafeo.
    assert!(result.memory_store.is_some());
    assert_eq!(result.chain.max_tool_turns(), 5);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn build_chain_with_memory_custom_db_path() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("custom.grafeo");
    let mut cfg = test_config(vec![], true);
    cfg.memory.db_path = Some(db_path.to_string_lossy().into_owned());
    let result = build_chain(&cfg);
    assert!(result.memory_store.is_some());
    assert_eq!(result.chain.max_tool_turns(), 5);
}

#[tokio::test]
async fn build_result_fields_accessible() {
    let cfg = test_config(vec![openrouter_backend()], false);
    let result = build_chain(&cfg);
    // Verify the struct destructures properly.
    let _chain = result.chain;
    let _store = result.memory_store;
}
