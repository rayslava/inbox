use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::config::{Config, LlmBackendConfig, LlmBackendType, MemoryConfig};
use crate::memory::MemoryStore;
use crate::pipeline::url_fetcher::UrlFetcher;

use super::tools;
use super::{LlmChain, LlmClient};

const MEMORY_OPEN_MAX_WAIT: Duration = Duration::from_secs(120);
const MEMORY_OPEN_BASE_DELAY: Duration = Duration::from_secs(1);
const MEMORY_OPEN_MAX_DELAY: Duration = Duration::from_secs(15);
const MEMORY_LOCK_MARKER: &str = "locked by another process";

/// Build result containing the LLM chain and, if memory is enabled, a shared handle
/// to the `MemoryStore` for use by the feedback system and admin routes.
pub struct BuildResult {
    pub chain: LlmChain,
    pub memory_store: Option<Arc<MemoryStore>>,
}

#[must_use]
#[anodized::spec(requires: cfg.llm.max_tool_turns > 0)]
pub fn build_chain(cfg: &Config) -> BuildResult {
    let backends: Vec<Box<dyn LlmClient>> = cfg.llm.backends.iter().map(build_backend).collect();

    let mut tool_executor = tools::from_tooling(&cfg.tooling, UrlFetcher::new(&cfg.url_fetch));

    let memory_store = if cfg.memory.enabled {
        wire_memory(cfg, &mut tool_executor)
    } else {
        None
    };

    let chain = LlmChain::new(
        backends,
        cfg.llm.fallback,
        cfg.llm.max_tool_turns,
        Some(tool_executor),
        cfg.llm.max_llm_tool_depth,
        cfg.llm.inner_retries,
        cfg.llm.tool_result_max_chars,
    );

    BuildResult {
        chain,
        memory_store,
    }
}

fn wire_memory(cfg: &Config, executor: &mut tools::ToolExecutor) -> Option<Arc<MemoryStore>> {
    let db_path = cfg.memory.db_path.as_deref().map_or_else(
        || cfg.general.attachments_dir.join("memory.grafeo"),
        std::path::PathBuf::from,
    );

    let mem_cfg = cfg.memory.clone();
    let Ok(handle) = tokio::runtime::Handle::try_current() else {
        tracing::warn!("No tokio runtime available; skipping memory tools");
        return None;
    };

    let store = tokio::task::block_in_place(|| {
        handle.block_on(open_memory_with_retry(
            &mem_cfg,
            &db_path,
            MEMORY_OPEN_MAX_WAIT,
        ))
    })?;

    let store = Arc::new(store);
    tools::add_memory_tools(executor, Arc::clone(&store));
    Some(store)
}

/// Open `MemoryStore`, retrying with exponential backoff while the DB file is
/// locked by a concurrent process (e.g. Syncthing during a sync window).
/// Returns `None` if the deadline expires or the failure is not a lock error.
#[anodized::spec(requires: !max_wait.is_zero())]
pub(super) async fn open_memory_with_retry(
    cfg: &MemoryConfig,
    path: &Path,
    max_wait: Duration,
) -> Option<MemoryStore> {
    let deadline = Instant::now() + max_wait;
    let mut delay = MEMORY_OPEN_BASE_DELAY;
    let mut attempt: u32 = 0;

    loop {
        attempt += 1;
        match MemoryStore::open(cfg, path).await {
            Ok(store) => {
                if attempt > 1 {
                    tracing::info!(attempt, "Memory store opened after retries");
                }
                return Some(store);
            }
            Err(e) => {
                let msg = e.to_string();
                let is_lock = msg.contains(MEMORY_LOCK_MARKER);
                let now = Instant::now();
                if !is_lock || now >= deadline {
                    tracing::warn!(
                        attempt,
                        "Memory store failed to open, skipping memory tools: {e}"
                    );
                    return None;
                }
                let remaining = deadline.saturating_duration_since(now);
                let sleep_for = delay.min(remaining);
                tracing::info!(
                    attempt,
                    delay_ms = u64::try_from(sleep_for.as_millis()).unwrap_or(u64::MAX),
                    "Memory DB locked, retrying: {e}"
                );
                tokio::time::sleep(sleep_for).await;
                delay = (delay * 2).min(MEMORY_OPEN_MAX_DELAY);
            }
        }
    }
}

fn build_backend(cfg: &LlmBackendConfig) -> Box<dyn LlmClient> {
    match cfg.backend_type {
        LlmBackendType::Openrouter => {
            Box::new(super::openrouter::OpenRouterClient::from_config(cfg))
        }
        LlmBackendType::Ollama => Box::new(super::ollama::OllamaClient::from_config(cfg)),
        LlmBackendType::FreeRouter => {
            Box::new(super::free_router::FreeRouterClient::from_config(cfg))
        }
    }
}
