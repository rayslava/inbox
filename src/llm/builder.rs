use crate::config::{Config, LlmBackendConfig, LlmBackendType};
use crate::pipeline::url_fetcher::UrlFetcher;

use super::tools;
use super::{LlmChain, LlmClient};

#[must_use]
#[anodized::spec(requires: cfg.llm.max_tool_turns > 0)]
pub fn build_chain(cfg: &Config) -> LlmChain {
    let backends: Vec<Box<dyn LlmClient>> = cfg.llm.backends.iter().map(build_backend).collect();

    let tool_executor = Some(tools::from_tooling(
        &cfg.tooling,
        UrlFetcher::new(&cfg.url_fetch),
    ));

    LlmChain::new(
        backends,
        cfg.llm.fallback,
        cfg.llm.max_tool_turns,
        tool_executor,
    )
}

fn build_backend(cfg: &LlmBackendConfig) -> Box<dyn LlmClient> {
    match cfg.backend_type {
        LlmBackendType::Openrouter => {
            Box::new(super::openrouter::OpenRouterClient::from_config(cfg))
        }
        LlmBackendType::Ollama => Box::new(super::ollama::OllamaClient::from_config(cfg)),
    }
}
