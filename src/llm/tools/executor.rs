//! `ToolExecutor` construction and tool-call dispatch.

use std::path::Path;
use std::time::Duration;

use anodized::spec;
use tracing::instrument;
use url::Url;
use uuid::Uuid;

use crate::config::ToolBackendConfig;
use crate::error::InboxError;
use crate::pipeline::url_fetcher::UrlFetcher;

use super::runners::{
    CrawlToolCfg, DuckDuckGoSearchToolCfg, HttpToolCfg, KagiSearchToolCfg, run_crawler_tool,
    run_duckduckgo_search_tool, run_http_tool, run_kagi_search_tool, run_shell_tool,
};
use super::{Tool, ToolExecutor, ToolResult};

impl ToolExecutor {
    /// Create a `ToolExecutor`.
    ///
    /// # Errors
    /// Returns an error if the tool HTTP client cannot be built.
    #[spec(requires: tools.iter().all(|t| !t.name.trim().is_empty() && !t.description.trim().is_empty()))]
    pub fn new(tools: Vec<Tool>, fetcher: UrlFetcher) -> Result<Self, crate::error::InboxError> {
        let http_client = crate::tls::client_builder()
            .timeout(Duration::from_secs(30))
            .build()
            .map_err(|e| {
                crate::error::InboxError::LlmTool(format!("Failed to build tool HTTP client: {e}"))
            })?;
        Ok(Self {
            tools,
            fetcher,
            http_client,
            memory_store: None,
        })
    }

    #[must_use]
    pub fn active_tool_definitions(&self) -> Vec<serde_json::Value> {
        self.tools
            .iter()
            .filter(|t| t.enabled)
            .map(Tool::openai_definition)
            .collect()
    }

    /// Execute a named tool call, retrying up to `tool.retries` additional times on failure.
    ///
    /// # Errors
    /// Returns an error if the tool is unknown, arguments are invalid, or all attempts fail.
    #[spec(requires: !name.is_empty())]
    pub async fn execute(
        &self,
        name: &str,
        args: &serde_json::Value,
        msg_id: Uuid,
        attachments_dir: &Path,
        source_name: &str,
    ) -> Result<ToolResult, InboxError> {
        let tool = self
            .tools
            .iter()
            .find(|t| t.name == name && t.enabled)
            .ok_or_else(|| InboxError::LlmTool(format!("Unknown tool: {name}")))?;

        let start = std::time::Instant::now();
        let attempts = tool.retries + 1;
        let mut last_err = InboxError::LlmTool(format!("tool {name} never attempted"));
        for attempt in 0..attempts {
            if attempt > 0 {
                tracing::warn!(tool = %name, attempt, "Retrying tool call after failure");
                let backoff = Duration::from_secs(2u64.pow(attempt).min(16));
                tokio::time::sleep(backoff).await;
            }
            match self
                .dispatch_once(tool, name, args, msg_id, attachments_dir, source_name)
                .await
            {
                Ok(result) => {
                    metrics::counter!(crate::telemetry::TOOL_CALLS, "tool" => name.to_owned(), "status" => "success")
                        .increment(1);
                    metrics::histogram!(crate::telemetry::TOOL_DURATION, "tool" => name.to_owned())
                        .record(start.elapsed().as_secs_f64());
                    return Ok(result);
                }
                Err(e) => {
                    tracing::warn!(tool = %name, attempt = attempt + 1, ?e, "Tool attempt failed");
                    last_err = e;
                }
            }
        }
        metrics::counter!(crate::telemetry::TOOL_CALLS, "tool" => name.to_owned(), "status" => "failure")
            .increment(1);
        metrics::histogram!(crate::telemetry::TOOL_DURATION, "tool" => name.to_owned())
            .record(start.elapsed().as_secs_f64());
        Err(last_err)
    }

    #[spec(requires: !name.trim().is_empty())]
    async fn dispatch_once(
        &self,
        tool: &Tool,
        name: &str,
        args: &serde_json::Value,
        msg_id: Uuid,
        attachments_dir: &Path,
        source_name: &str,
    ) -> Result<ToolResult, InboxError> {
        match name {
            "memory_save" => self.run_memory_save(args, msg_id, source_name).await,
            "memory_recall" => self.run_memory_recall(args).await,
            "memory_link" => self.run_memory_link(args).await,
            "memory_context" => self.run_memory_context(args).await,
            "scrape_page" => self.run_scrape_call(&tool.backend, args).await,
            "download_file" => {
                self.run_download_call(&tool.backend, args, msg_id, attachments_dir)
                    .await
            }
            "crawl_url" => self.run_crawl_call(&tool.backend, args).await,
            "web_search" | "duckduckgo_search" => {
                self.run_web_search_call(&tool.backend, args).await
            }
            _ => Err(InboxError::LlmTool(format!("No handler for tool: {name}"))),
        }
    }

    async fn run_memory_save(
        &self,
        args: &serde_json::Value,
        msg_id: Uuid,
        source_name: &str,
    ) -> Result<ToolResult, InboxError> {
        let key = required_str(args, "key", "memory_save")?;
        let value = required_str(args, "value", "memory_save")?;
        let store = self.require_memory_store("memory_save")?;
        store
            .save(key, value)
            .await
            .map_err(|e| InboxError::LlmTool(e.to_string()))?;
        if !source_name.is_empty() {
            let _ = store
                .link_source(key, source_name, &msg_id.to_string(), "")
                .await;
        }
        Ok(ToolResult::Text(format!("Saved memory: {key}")))
    }

    async fn run_memory_recall(&self, args: &serde_json::Value) -> Result<ToolResult, InboxError> {
        let query = required_str(args, "query", "memory_recall")?;
        let store = self.require_memory_store("memory_recall")?;
        let entries = store
            .recall(query, 10)
            .await
            .map_err(|e| InboxError::LlmTool(e.to_string()))?;
        Ok(ToolResult::Text(format_memory_entries(
            &entries,
            "No memories found.",
        )))
    }

    async fn run_memory_link(&self, args: &serde_json::Value) -> Result<ToolResult, InboxError> {
        let from_key = required_str(args, "from_key", "memory_link")?;
        let to_key = required_str(args, "to_key", "memory_link")?;
        let relation = required_str(args, "relation", "memory_link")?;
        let store = self.require_memory_store("memory_link")?;
        store
            .link_memories(from_key, to_key, relation)
            .await
            .map_err(|e| InboxError::LlmTool(e.to_string()))?;
        Ok(ToolResult::Text(format!(
            "Linked {from_key} -> {to_key} ({relation})"
        )))
    }

    async fn run_memory_context(&self, args: &serde_json::Value) -> Result<ToolResult, InboxError> {
        let query = required_str(args, "query", "memory_context")?;
        let hops = u32::try_from(args["hops"].as_u64().unwrap_or(1).min(3)).unwrap_or(1);
        let store = self.require_memory_store("memory_context")?;
        let entries = store
            .context(query, hops)
            .await
            .map_err(|e| InboxError::LlmTool(e.to_string()))?;
        Ok(ToolResult::Text(format_memory_entries(
            &entries,
            "No connected memories found.",
        )))
    }

    async fn run_scrape_call(
        &self,
        backend: &ToolBackendConfig,
        args: &serde_json::Value,
    ) -> Result<ToolResult, InboxError> {
        let url_str = required_str(args, "url", "scrape_page")?;
        let url = Url::parse(url_str).map_err(InboxError::UrlParse)?;
        self.run_scrape(backend, &url).await
    }

    async fn run_download_call(
        &self,
        backend: &ToolBackendConfig,
        args: &serde_json::Value,
        msg_id: Uuid,
        attachments_dir: &Path,
    ) -> Result<ToolResult, InboxError> {
        let url_str = required_str(args, "url", "download_file")?;
        let url = Url::parse(url_str).map_err(InboxError::UrlParse)?;
        self.run_download(backend, &url, msg_id, attachments_dir)
            .await
    }

    async fn run_crawl_call(
        &self,
        backend: &ToolBackendConfig,
        args: &serde_json::Value,
    ) -> Result<ToolResult, InboxError> {
        let url_str = required_str(args, "url", "crawl_url")?;
        self.run_crawl(backend, url_str).await
    }

    async fn run_web_search_call(
        &self,
        backend: &ToolBackendConfig,
        args: &serde_json::Value,
    ) -> Result<ToolResult, InboxError> {
        let query = required_str(args, "query", "web_search")?;
        let limit = args["limit"].as_u64().and_then(|v| u32::try_from(v).ok());
        self.run_web_search(backend, query, limit).await
    }

    fn require_memory_store(
        &self,
        tool_name: &str,
    ) -> Result<&std::sync::Arc<crate::memory::MemoryStore>, InboxError> {
        self.memory_store
            .as_ref()
            .ok_or_else(|| InboxError::LlmTool(format!("{tool_name}: no memory store")))
    }

    #[instrument(skip(self, backend), fields(url = %url))]
    async fn run_scrape(
        &self,
        backend: &ToolBackendConfig,
        url: &Url,
    ) -> Result<ToolResult, InboxError> {
        use crate::pipeline::url_fetcher::rewrite_twitter_url;
        let rewritten;
        let effective_url = match rewrite_twitter_url(url, self.fetcher.nitter_base_url()) {
            Some(rw) => {
                rewritten = rw;
                &rewritten
            }
            None => url,
        };
        match backend {
            ToolBackendConfig::Internal { timeout_secs } => {
                let timeout = Duration::from_secs(u64::from(*timeout_secs));
                let content = tokio::time::timeout(timeout, self.fetcher.fetch_page(effective_url))
                    .await
                    .map_err(|_| {
                        InboxError::LlmTool(format!("scrape_page timed out after {timeout_secs}s"))
                    })?;
                Ok(ToolResult::Text(
                    content.map_or_else(|| "Failed to fetch page".into(), |c| c.text),
                ))
            }
            ToolBackendConfig::Shell { argv, timeout_secs } => {
                run_shell_tool(argv, effective_url.as_str(), "", *timeout_secs).await
            }
            ToolBackendConfig::Http {
                endpoint,
                method,
                auth_header,
                body_template,
                response_path,
                timeout_secs,
            } => {
                let cfg = HttpToolCfg {
                    endpoint,
                    method,
                    auth_header: auth_header.as_deref(),
                    body_template: body_template.as_deref(),
                    response_path,
                    timeout_secs: *timeout_secs,
                };
                run_http_tool(&self.http_client, cfg, effective_url.as_str(), "").await
            }
            ToolBackendConfig::Crawler { .. } => Err(InboxError::LlmTool(
                "scrape_page does not support crawler backend".into(),
            )),
            ToolBackendConfig::KagiSearch { .. }
            | ToolBackendConfig::DuckDuckGoSearch { .. }
            | ToolBackendConfig::Memory => Err(InboxError::LlmTool(
                "scrape_page does not support this backend".into(),
            )),
        }
    }

    #[instrument(skip(self, backend, attachments_dir), fields(url = %url))]
    async fn run_download(
        &self,
        backend: &ToolBackendConfig,
        url: &Url,
        msg_id: Uuid,
        attachments_dir: &Path,
    ) -> Result<ToolResult, InboxError> {
        match backend {
            ToolBackendConfig::Internal { timeout_secs } => {
                let timeout = Duration::from_secs(u64::from(*timeout_secs));
                let att = tokio::time::timeout(
                    timeout,
                    self.fetcher.download_file(url, msg_id, attachments_dir),
                )
                .await
                .map_err(|_| {
                    InboxError::LlmTool(format!("download_file timed out after {timeout_secs}s"))
                })?;
                match att {
                    Some(a) => {
                        let name = a.original_name.clone();
                        Ok(ToolResult::Attachment {
                            text: format!("Downloaded: {name}"),
                            attachment: a,
                        })
                    }
                    None => Ok(ToolResult::Text("Failed to download file".into())),
                }
            }
            ToolBackendConfig::Shell { argv, timeout_secs } => {
                let filename = crate::pipeline::url_fetcher::attachment_save_path(
                    attachments_dir,
                    msg_id,
                    "download",
                )
                .to_string_lossy()
                .into_owned();
                run_shell_tool(argv, url.as_str(), &filename, *timeout_secs).await
            }
            ToolBackendConfig::Http {
                endpoint,
                method,
                auth_header,
                body_template,
                response_path,
                timeout_secs,
            } => {
                let cfg = HttpToolCfg {
                    endpoint,
                    method,
                    auth_header: auth_header.as_deref(),
                    body_template: body_template.as_deref(),
                    response_path,
                    timeout_secs: *timeout_secs,
                };
                run_http_tool(&self.http_client, cfg, url.as_str(), "").await
            }
            ToolBackendConfig::Crawler { .. } => Err(InboxError::LlmTool(
                "download_file does not support crawler backend".into(),
            )),
            ToolBackendConfig::KagiSearch { .. }
            | ToolBackendConfig::DuckDuckGoSearch { .. }
            | ToolBackendConfig::Memory => Err(InboxError::LlmTool(
                "download_file does not support this backend".into(),
            )),
        }
    }

    #[instrument(skip(self, backend), fields(url = %url))]
    #[spec(requires: !url.trim().is_empty())]
    async fn run_crawl(
        &self,
        backend: &ToolBackendConfig,
        url: &str,
    ) -> Result<ToolResult, InboxError> {
        match backend {
            ToolBackendConfig::Crawler {
                endpoint,
                auth_header,
                timeout_secs,
                priority,
            } => {
                let cfg = CrawlToolCfg {
                    endpoint,
                    auth_header: auth_header.as_deref(),
                    timeout_secs: *timeout_secs,
                    priority: *priority,
                };
                run_crawler_tool(&self.http_client, cfg, url).await
            }
            _ => Err(InboxError::LlmTool(
                "crawl_url requires crawler backend".into(),
            )),
        }
    }

    #[spec(requires: !query.trim().is_empty())]
    async fn run_web_search(
        &self,
        backend: &ToolBackendConfig,
        query: &str,
        limit: Option<u32>,
    ) -> Result<ToolResult, InboxError> {
        match backend {
            ToolBackendConfig::KagiSearch {
                endpoint,
                api_token,
                timeout_secs,
                default_limit,
                max_snippet_chars,
            } => {
                let cfg = KagiSearchToolCfg {
                    endpoint,
                    api_token: api_token.as_deref(),
                    timeout_secs: *timeout_secs,
                    default_limit: *default_limit,
                    max_snippet_chars: *max_snippet_chars,
                };
                run_kagi_search_tool(&self.http_client, cfg, query, limit).await
            }
            ToolBackendConfig::DuckDuckGoSearch {
                endpoint,
                timeout_secs,
                default_limit,
                max_snippet_chars,
            } => {
                let cfg = DuckDuckGoSearchToolCfg {
                    endpoint,
                    timeout_secs: *timeout_secs,
                    default_limit: *default_limit,
                    max_snippet_chars: *max_snippet_chars,
                };
                run_duckduckgo_search_tool(&self.http_client, cfg, query, limit).await
            }
            _ => Err(InboxError::LlmTool(
                "web_search requires a search backend (kagi_search or duckduckgo_search)".into(),
            )),
        }
    }
}

fn required_str<'a>(
    args: &'a serde_json::Value,
    key: &str,
    tool_name: &str,
) -> Result<&'a str, InboxError> {
    args[key]
        .as_str()
        .ok_or_else(|| InboxError::LlmTool(format!("{tool_name} missing '{key}'")))
}

fn format_memory_entries(entries: &[crate::memory::MemoryEntry], empty_msg: &str) -> String {
    if entries.is_empty() {
        empty_msg.to_owned()
    } else {
        entries
            .iter()
            .map(|e| format!("{}: {}", e.key, e.value))
            .collect::<Vec<_>>()
            .join("\n")
    }
}
