use std::path::Path;
use std::time::Duration;

use anodized::spec;
use tracing::instrument;
use url::Url;
use uuid::Uuid;

use crate::config::{ToolBackendConfig, ToolingConfig};
use crate::error::InboxError;
use crate::message::Attachment;
use crate::pipeline::url_fetcher::UrlFetcher;

use runners::{CrawlToolCfg, HttpToolCfg, run_crawler_tool, run_http_tool, run_shell_tool};

mod runners;

#[cfg(test)]
mod tests;

// ── Tool definition ───────────────────────────────────────────────────────────

/// A configured tool the LLM can call.
pub struct Tool {
    pub name: String,
    pub description: String,
    pub enabled: bool,
    pub backend: ToolBackendConfig,
}

impl Tool {
    #[must_use]
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
        _ => serde_json::json!({ "type": "object", "properties": {} }),
    }
}

// ── ToolExecutor ──────────────────────────────────────────────────────────────

/// Dispatches LLM tool calls to their configured backend.
pub struct ToolExecutor {
    tools: Vec<Tool>,
    fetcher: UrlFetcher,
    http_client: reqwest::Client,
}

impl ToolExecutor {
    /// Create a `ToolExecutor`.
    ///
    /// # Panics
    /// Panics if the TLS backend cannot be initialised (extremely unlikely in practice).
    #[must_use]
    pub fn new(tools: Vec<Tool>, fetcher: UrlFetcher) -> Self {
        let http_client = crate::tls::client_builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to build tool HTTP client");
        Self {
            tools,
            fetcher,
            http_client,
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

    /// Execute a named tool call.
    ///
    /// # Errors
    /// Returns an error if the tool is unknown, arguments are invalid, or the backend fails.
    #[spec(requires: !name.is_empty())]
    pub async fn execute(
        &self,
        name: &str,
        args: &serde_json::Value,
        msg_id: Uuid,
        attachments_dir: &Path,
    ) -> Result<ToolResult, InboxError> {
        let tool = self
            .tools
            .iter()
            .find(|t| t.name == name && t.enabled)
            .ok_or_else(|| InboxError::LlmTool(format!("Unknown tool: {name}")))?;

        match name {
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
            _ => Err(InboxError::LlmTool(format!("No handler for tool: {name}"))),
        }
    }

    #[instrument(skip(self, backend), fields(url = %url))]
    async fn run_scrape(
        &self,
        backend: &ToolBackendConfig,
        url: &Url,
    ) -> Result<ToolResult, InboxError> {
        match backend {
            ToolBackendConfig::Internal => {
                let content = self.fetcher.fetch_page(url).await;
                Ok(ToolResult::Text(
                    content.map_or_else(|| "Failed to fetch page".into(), |c| c.text),
                ))
            }
            ToolBackendConfig::Shell { argv, timeout_secs } => {
                run_shell_tool(argv, url.as_str(), "", *timeout_secs).await
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
                "scrape_page does not support crawler backend".into(),
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
            ToolBackendConfig::Internal => {
                let att = self
                    .fetcher
                    .download_file(url, msg_id, attachments_dir)
                    .await;
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
        }
    }

    #[instrument(skip(self, backend), fields(url = %url))]
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
}

// ── ToolResult ────────────────────────────────────────────────────────────────

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

// ── Builders ──────────────────────────────────────────────────────────────────

/// Build the default tool list used when tooling config is not customized.
#[must_use]
pub fn default_tools(fetcher: UrlFetcher) -> ToolExecutor {
    let tools = vec![
        Tool {
            name: "scrape_page".into(),
            description: "Fetch and extract readable text from a web page URL".into(),
            enabled: true,
            backend: ToolBackendConfig::Internal,
        },
        Tool {
            name: "download_file".into(),
            description: "Download a file from a URL and save it as an attachment".into(),
            enabled: true,
            backend: ToolBackendConfig::Internal,
        },
        Tool {
            name: "crawl_url".into(),
            description: "Crawl a URL and return markdown/html from crawler service".into(),
            enabled: false,
            backend: ToolBackendConfig::Crawler {
                endpoint: "http://localhost:11235/crawl".into(),
                auth_header: None,
                timeout_secs: 30,
                priority: 10,
            },
        },
    ];
    ToolExecutor::new(tools, fetcher)
}

#[must_use]
pub fn from_tooling(tooling: &ToolingConfig, fetcher: UrlFetcher) -> ToolExecutor {
    let scrape_desc = if tooling.scrape_page.description.trim().is_empty() {
        "Fetch and extract readable text from a web page URL".to_owned()
    } else {
        tooling.scrape_page.description.clone()
    };
    let download_desc = if tooling.download_file.description.trim().is_empty() {
        "Download a file from a URL and save it as an attachment".to_owned()
    } else {
        tooling.download_file.description.clone()
    };
    let crawl_desc = if tooling.crawl_url.description.trim().is_empty() {
        "Crawl a URL and return markdown/html from crawler service".to_owned()
    } else {
        tooling.crawl_url.description.clone()
    };

    let tools = vec![
        Tool {
            name: "scrape_page".into(),
            description: scrape_desc,
            enabled: tooling.scrape_page.enabled,
            backend: tooling.scrape_page.backend.clone(),
        },
        Tool {
            name: "download_file".into(),
            description: download_desc,
            enabled: tooling.download_file.enabled,
            backend: tooling.download_file.backend.clone(),
        },
        Tool {
            name: "crawl_url".into(),
            description: crawl_desc,
            enabled: tooling.crawl_url.enabled,
            backend: ToolBackendConfig::Crawler {
                endpoint: tooling.crawl_url.endpoint.clone(),
                auth_header: tooling.crawl_url.auth_header.clone(),
                timeout_secs: tooling.crawl_url.timeout_secs,
                priority: tooling.crawl_url.priority,
            },
        },
    ];
    ToolExecutor::new(tools, fetcher)
}
