use std::sync::Arc;

use crate::config::{Config, LlmBackendConfig, LlmBackendType};
use crate::pipeline::url_fetcher::UrlFetcher;

use super::tools;
use super::{LlmChain, LlmClient};

#[must_use]
#[anodized::spec(requires: cfg.llm.max_tool_turns > 0)]
pub fn build_chain(cfg: &Config) -> LlmChain {
    let backends: Vec<Box<dyn LlmClient>> = cfg.llm.backends.iter().map(build_backend).collect();

    let mut tool_executor = tools::from_tooling(&cfg.tooling, UrlFetcher::new(&cfg.url_fetch));

    if cfg.memory.enabled {
        wire_memory(cfg, &mut tool_executor);
    }

    LlmChain::new(
        backends,
        cfg.llm.fallback,
        cfg.llm.max_tool_turns,
        Some(tool_executor),
        cfg.llm.max_llm_tool_depth,
        cfg.llm.inner_retries,
        cfg.llm.tool_result_max_chars,
    )
}

fn wire_memory(cfg: &Config, executor: &mut tools::ToolExecutor) {
    let db_path = cfg.memory.db_path.as_deref().map_or_else(
        || cfg.general.attachments_dir.join("memory.grafeo"),
        std::path::PathBuf::from,
    );

    let mem_cfg = cfg.memory.clone();
    // Build the store synchronously by spinning up a local runtime for the async open.
    // In production this is called at startup (not in a hot path).
    let rt = tokio::runtime::Handle::try_current();
    match rt {
        Ok(handle) => {
            let store_result = tokio::task::block_in_place(|| {
                handle.block_on(crate::memory::MemoryStore::open(&mem_cfg, &db_path))
            });
            match store_result {
                Ok(store) => {
                    tools::add_memory_tools(executor, Arc::new(store));
                }
                Err(e) => {
                    tracing::warn!("Memory store failed to open, skipping memory tools: {e}");
                }
            }
        }
        Err(_) => {
            tracing::warn!("No tokio runtime available; skipping memory tools");
        }
    }
}

fn build_backend(cfg: &LlmBackendConfig) -> Box<dyn LlmClient> {
    match cfg.backend_type {
        LlmBackendType::Openrouter => {
            Box::new(super::openrouter::OpenRouterClient::from_config(cfg))
        }
        LlmBackendType::Ollama => Box::new(super::ollama::OllamaClient::from_config(cfg)),
    }
}
