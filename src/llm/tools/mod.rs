use std::path::Path;
use std::time::Duration;

use anodized::spec;
use tracing::instrument;
use url::Url;
use uuid::Uuid;

use crate::config::ToolBackendConfig;
use crate::error::InboxError;
use crate::message::Attachment;
use crate::pipeline::url_fetcher::UrlFetcher;

use runners::{
    CrawlToolCfg, DuckDuckGoSearchToolCfg, HttpToolCfg, KagiSearchToolCfg, run_crawler_tool,
    run_duckduckgo_search_tool, run_http_tool, run_kagi_search_tool, run_shell_tool,
};

mod builders;
mod runners;

pub use builders::{add_memory_tools, default_tools, from_tooling};

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_runners;
#[cfg(test)]
mod tests_search_memory;

// ── Tool definition ───────────────────────────────────────────────────────────

/// A configured tool the LLM can call.
pub struct Tool {
    pub name: String,
    pub description: String,
    pub enabled: bool,
    pub retries: u32,
    pub backend: ToolBackendConfig,
}

impl Tool {
    #[must_use]
    #[spec(requires: !self.name.trim().is_empty() && !self.description.trim().is_empty())]
    pub fn openai_definition(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": tool_parameters(&self.name),
            }
        })
    }
}

#[spec(requires: !name.trim().is_empty())]
fn tool_parameters(name: &str) -> serde_json::Value {
    match name {
        "scrape_page" => serde_json::json!({
            "type": "object",
            "properties": {
                "url": { "type": "string", "description": "The URL to scrape" }
            },
            "required": ["url"]
        }),
        "download_file" => serde_json::json!({
            "type": "object",
            "properties": {
                "url": { "type": "string", "description": "The URL of the file to download" }
            },
            "required": ["url"]
        }),
        "crawl_url" => serde_json::json!({
            "type": "object",
            "properties": {
                "url": { "type": "string", "description": "The URL to crawl" }
            },
            "required": ["url"]
        }),
        "web_search" | "duckduckgo_search" => serde_json::json!({
            "type": "object",
            "properties": {
                "query": { "type": "string", "description": "The web search query" },
                "limit": { "type": "integer", "description": "Optional max number of results (1-20)" }
            },
            "required": ["query"]
        }),
        "memory_save" => serde_json::json!({
            "type": "object",
            "properties": {
                "key": { "type": "string", "description": "Short identifier for the memory" },
                "value": { "type": "string", "description": "Content to remember" }
            },
            "required": ["key", "value"]
        }),
        "memory_recall" => serde_json::json!({
            "type": "object",
            "properties": {
                "query": { "type": "string", "description": "Search query to find relevant memories" }
            },
            "required": ["query"]
        }),
        "memory_link" => serde_json::json!({
            "type": "object",
            "properties": {
                "from_key": { "type": "string", "description": "Key of the source memory" },
                "to_key": { "type": "string", "description": "Key of the target memory" },
                "relation": { "type": "string", "description": "Relationship type (e.g. 'related_to', 'depends_on')" }
            },
            "required": ["from_key", "to_key", "relation"]
        }),
        "memory_context" => serde_json::json!({
            "type": "object",
            "properties": {
                "query": { "type": "string", "description": "Memory key to find connected memories for" },
                "hops": { "type": "integer", "description": "Number of graph traversal hops (default 1, max 3)" }
            },
            "required": ["query"]
        }),
        _ => serde_json::json!({ "type": "object", "properties": {} }),
    }
}

// ── ToolExecutor ──────────────────────────────────────────────────────────────

/// Dispatches LLM tool calls to their configured backend.
pub struct ToolExecutor {
    tools: Vec<Tool>,
    fetcher: UrlFetcher,
    http_client: reqwest::Client,
    pub(super) memory_store: Option<std::sync::Arc<crate::memory::MemoryStore>>,
}

impl ToolExecutor {
    /// Create a `ToolExecutor`.
    ///
    /// # Panics
    /// Panics if the TLS backend cannot be initialised (extremely unlikely in practice).
    #[must_use]
    #[spec(requires: tools.iter().all(|t| !t.name.trim().is_empty() && !t.description.trim().is_empty()))]
    pub fn new(tools: Vec<Tool>, fetcher: UrlFetcher) -> Self {
        let http_client = crate::tls::client_builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to build tool HTTP client");
        Self {
            tools,
            fetcher,
            http_client,
            memory_store: None,
        }
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
            "memory_save" => {
                let key = args["key"]
                    .as_str()
                    .ok_or_else(|| InboxError::LlmTool("memory_save missing 'key'".into()))?;
                let value = args["value"]
                    .as_str()
                    .ok_or_else(|| InboxError::LlmTool("memory_save missing 'value'".into()))?;
                let store = self
                    .memory_store
                    .as_ref()
                    .ok_or_else(|| InboxError::LlmTool("memory_save: no memory store".into()))?;
                store
                    .save(key, value)
                    .await
                    .map_err(|e| InboxError::LlmTool(e.to_string()))?;
                // Auto-link memory to the originating source.
                if !source_name.is_empty() {
                    let _ = store
                        .link_source(key, source_name, &msg_id.to_string(), "")
                        .await;
                }
                Ok(ToolResult::Text(format!("Saved memory: {key}")))
            }
            "memory_recall" => {
                let query = args["query"]
                    .as_str()
                    .ok_or_else(|| InboxError::LlmTool("memory_recall missing 'query'".into()))?;
                let store = self
                    .memory_store
                    .as_ref()
                    .ok_or_else(|| InboxError::LlmTool("memory_recall: no memory store".into()))?;
                let entries = store
                    .recall(query, 10)
                    .await
                    .map_err(|e| InboxError::LlmTool(e.to_string()))?;
                let text = if entries.is_empty() {
                    "No memories found.".to_owned()
                } else {
                    entries
                        .iter()
                        .map(|e| format!("{}: {}", e.key, e.value))
                        .collect::<Vec<_>>()
                        .join("\n")
                };
                Ok(ToolResult::Text(text))
            }
            "memory_link" => {
                let from_key = args["from_key"]
                    .as_str()
                    .ok_or_else(|| InboxError::LlmTool("memory_link missing 'from_key'".into()))?;
                let to_key = args["to_key"]
                    .as_str()
                    .ok_or_else(|| InboxError::LlmTool("memory_link missing 'to_key'".into()))?;
                let relation = args["relation"]
                    .as_str()
                    .ok_or_else(|| InboxError::LlmTool("memory_link missing 'relation'".into()))?;
                let store = self
                    .memory_store
                    .as_ref()
                    .ok_or_else(|| InboxError::LlmTool("memory_link: no memory store".into()))?;
                store
                    .link_memories(from_key, to_key, relation)
                    .await
                    .map_err(|e| InboxError::LlmTool(e.to_string()))?;
                Ok(ToolResult::Text(format!(
                    "Linked {from_key} -> {to_key} ({relation})"
                )))
            }
            "memory_context" => {
                let query = args["query"]
                    .as_str()
                    .ok_or_else(|| InboxError::LlmTool("memory_context missing 'query'".into()))?;
                let hops = args["hops"].as_u64().unwrap_or(1).min(3) as u32;
                let store = self
                    .memory_store
                    .as_ref()
                    .ok_or_else(|| InboxError::LlmTool("memory_context: no memory store".into()))?;
                let entries = store
                    .context(query, hops)
                    .await
                    .map_err(|e| InboxError::LlmTool(e.to_string()))?;
                let text = if entries.is_empty() {
                    "No connected memories found.".to_owned()
                } else {
                    entries
                        .iter()
                        .map(|e| format!("{}: {}", e.key, e.value))
                        .collect::<Vec<_>>()
                        .join("\n")
                };
                Ok(ToolResult::Text(text))
            }
            "scrape_page" => {
                let url_str = args["url"]
                    .as_str()
                    .ok_or_else(|| InboxError::LlmTool("scrape_page missing 'url'".into()))?;
                let url = Url::parse(url_str).map_err(InboxError::UrlParse)?;
                self.run_scrape(&tool.backend, &url).await
            }
            "download_file" => {
                let url_str = args["url"]
                    .as_str()
                    .ok_or_else(|| InboxError::LlmTool("download_file missing 'url'".into()))?;
                let url = Url::parse(url_str).map_err(InboxError::UrlParse)?;
                self.run_download(&tool.backend, &url, msg_id, attachments_dir)
                    .await
            }
            "crawl_url" => {
                let url_str = args["url"]
                    .as_str()
                    .ok_or_else(|| InboxError::LlmTool("crawl_url missing 'url'".into()))?;
                self.run_crawl(&tool.backend, url_str).await
            }
            "web_search" | "duckduckgo_search" => {
                let query = args["query"]
                    .as_str()
                    .ok_or_else(|| InboxError::LlmTool("web_search missing 'query'".into()))?;
                let limit = args["limit"].as_u64().and_then(|v| u32::try_from(v).ok());
                self.run_web_search(&tool.backend, query, limit).await
            }
            _ => Err(InboxError::LlmTool(format!("No handler for tool: {name}"))),
        }
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

// ── ToolResult ────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum ToolResult {
    Text(String),
    Attachment {
        text: String,
        attachment: Attachment,
    },
}

impl ToolResult {
    #[must_use]
    pub fn text(&self) -> &str {
        match self {
            Self::Text(t) => t,
            Self::Attachment { text, .. } => text,
        }
    }
}
