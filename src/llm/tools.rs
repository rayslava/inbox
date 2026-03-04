use std::path::Path;
use std::time::Duration;

use anodized::spec;
use tokio::process::Command;
use tracing::{instrument, warn};
use url::Url;
use uuid::Uuid;

use crate::config::{ToolBackendConfig, ToolingConfig};
use crate::error::InboxError;
use crate::message::Attachment;
use crate::pipeline::url_fetcher::UrlFetcher;

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
        let http_client = reqwest::Client::builder()
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

/// Execute a shell tool. argv may contain `{url}` and `{filename}` placeholders.
/// Arguments are passed as separate argv entries — no shell interpolation.
async fn run_shell_tool(
    argv: &[String],
    url: &str,
    filename: &str,
    timeout_secs: u32,
) -> Result<ToolResult, InboxError> {
    if argv.is_empty() {
        return Err(InboxError::LlmTool("Shell tool has empty argv".into()));
    }

    let program = &argv[0];
    let processed_args: Vec<String> = argv[1..]
        .iter()
        .map(|a| a.replace("{url}", url).replace("{filename}", filename))
        .collect();

    let output = tokio::time::timeout(
        Duration::from_secs(u64::from(timeout_secs)),
        Command::new(program).args(&processed_args).output(),
    )
    .await
    .map_err(|_| InboxError::LlmTool(format!("Shell tool timed out after {timeout_secs}s")))?
    .map_err(|e| InboxError::LlmTool(format!("Shell tool exec error: {e}")))?;

    if !output.status.success() {
        warn!(
            program,
            status = ?output.status,
            stderr = %String::from_utf8_lossy(&output.stderr),
            "Shell tool exited with error"
        );
    }

    Ok(ToolResult::Text(
        String::from_utf8_lossy(&output.stdout).into_owned(),
    ))
}

/// Configuration bundle for an HTTP tool backend call.
struct HttpToolCfg<'a> {
    endpoint: &'a str,
    method: &'a str,
    auth_header: Option<&'a str>,
    body_template: Option<&'a str>,
    response_path: &'a str,
    timeout_secs: u32,
}

struct CrawlToolCfg<'a> {
    endpoint: &'a str,
    auth_header: Option<&'a str>,
    timeout_secs: u32,
    priority: i32,
}

/// Execute an HTTP tool backend.
async fn run_http_tool(
    client: &reqwest::Client,
    cfg: HttpToolCfg<'_>,
    url: &str,
    filename: &str,
) -> Result<ToolResult, InboxError> {
    #[spec(requires: !cfg.endpoint.is_empty() && cfg.timeout_secs > 0)]
    fn validate_http_cfg(cfg: &HttpToolCfg<'_>) {
        let _ = cfg;
    }
    validate_http_cfg(&cfg);

    let endpoint_resolved = cfg
        .endpoint
        .replace("{url}", url)
        .replace("{filename}", filename);

    let mut req = match cfg.method.to_uppercase().as_str() {
        "POST" => client.post(&endpoint_resolved),
        _ => client.get(&endpoint_resolved),
    };

    if let Some(auth) = cfg.auth_header {
        let resolved = resolve_env_vars(auth);
        if let Some((name, value)) = resolved.split_once(':') {
            req = req.header(name.trim(), value.trim());
        }
    }

    if let Some(tmpl) = cfg.body_template {
        let body = tmpl.replace("{url}", url).replace("{filename}", filename);
        req = req.header("content-type", "application/json").body(body);
    }

    let timeout = Duration::from_secs(u64::from(cfg.timeout_secs));
    let resp = tokio::time::timeout(timeout, req.send())
        .await
        .map_err(|_| {
            InboxError::LlmTool(format!("HTTP tool timed out after {}s", cfg.timeout_secs))
        })?
        .map_err(|e| InboxError::LlmTool(format!("HTTP tool request error: {e}")))?;

    let body = resp
        .text()
        .await
        .map_err(|e| InboxError::LlmTool(e.to_string()))?;

    if cfg.response_path.is_empty() {
        return Ok(ToolResult::Text(body));
    }

    // Extract nested field via dot-notation path.
    let json: serde_json::Value = serde_json::from_str(&body)?;
    let value = cfg
        .response_path
        .split('.')
        .try_fold(&json, |acc, key| acc.get(key));

    let text = value
        .and_then(serde_json::Value::as_str)
        .unwrap_or("")
        .to_owned();

    Ok(ToolResult::Text(text))
}

#[spec(requires: !url.is_empty())]
async fn run_crawler_tool(
    client: &reqwest::Client,
    cfg: CrawlToolCfg<'_>,
    url: &str,
) -> Result<ToolResult, InboxError> {
    let mut req = client.post(cfg.endpoint);
    if let Some(auth) = cfg.auth_header {
        let resolved = resolve_env_vars(auth);
        if let Some((name, value)) = resolved.split_once(':') {
            req = req.header(name.trim(), value.trim());
        }
    }

    let payload = serde_json::json!({
        "urls": [url],
        "priority": cfg.priority,
    });
    let timeout = Duration::from_secs(u64::from(cfg.timeout_secs));
    let resp = tokio::time::timeout(timeout, req.json(&payload).send())
        .await
        .map_err(|_| InboxError::LlmTool(format!("Crawler timed out after {}s", cfg.timeout_secs)))?
        .map_err(|e| InboxError::LlmTool(format!("Crawler request error: {e}")))?;

    let json: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| InboxError::LlmTool(format!("Crawler parse error: {e}")))?;

    let Some(result) = json
        .get("results")
        .and_then(serde_json::Value::as_array)
        .and_then(|arr| arr.first())
    else {
        return Err(InboxError::LlmTool("Crawler returned no results".into()));
    };

    let title = result
        .pointer("/metadata/title")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    let markdown = result
        .pointer("/markdown/raw_markdown")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    let html = result
        .get("cleaned_html")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");

    let mut out = String::new();
    if !title.is_empty() {
        out.push_str("Title: ");
        out.push_str(title);
        out.push_str("\n\n");
    }

    if !markdown.is_empty() {
        out.push_str("Markdown:\n");
        out.push_str(markdown);
    } else if !html.is_empty() {
        out.push_str("HTML fallback:\n");
        out.push_str(html);
    } else {
        out.push_str("Crawler returned no markdown/html content");
    }

    Ok(ToolResult::Text(out))
}

/// Expand `${VAR}` patterns in a string using environment variables.
fn resolve_env_vars(s: &str) -> String {
    let re = regex::Regex::new(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}").unwrap();
    re.replace_all(s, |caps: &regex::Captures<'_>| {
        std::env::var(&caps[1]).unwrap_or_default()
    })
    .into_owned()
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::UrlFetchConfig;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn test_fetcher() -> UrlFetcher {
        UrlFetcher::new(&UrlFetchConfig {
            enabled: true,
            user_agent: "test/1.0".into(),
            timeout_secs: 5,
            max_redirects: 3,
            max_body_bytes: 1024 * 1024,
            skip_domains: vec![],
        })
    }

    #[test]
    fn tool_openai_definition_has_name() {
        let tool = Tool {
            name: "scrape_page".into(),
            description: "desc".into(),
            enabled: true,
            backend: ToolBackendConfig::Internal,
        };
        let def = tool.openai_definition();
        assert_eq!(def["function"]["name"], "scrape_page");
    }

    #[test]
    fn active_tool_definitions_filters_disabled() {
        let executor = default_tools(test_fetcher());
        let defs = executor.active_tool_definitions();
        assert_eq!(defs.len(), 2);
    }

    #[test]
    fn active_tool_definitions_empty_when_all_disabled() {
        let tools = vec![Tool {
            name: "scrape_page".into(),
            description: "d".into(),
            enabled: false,
            backend: ToolBackendConfig::Internal,
        }];
        let executor = ToolExecutor::new(tools, test_fetcher());
        assert!(executor.active_tool_definitions().is_empty());
    }

    #[test]
    fn tool_result_text() {
        let r = ToolResult::Text("hello".into());
        assert_eq!(r.text(), "hello");
    }

    #[test]
    fn tool_result_attachment_text() {
        use crate::message::{Attachment, MediaKind};
        let r = ToolResult::Attachment {
            text: "downloaded".into(),
            attachment: Attachment {
                original_name: "f.pdf".into(),
                saved_path: std::path::PathBuf::from("/tmp/f.pdf"),
                mime_type: None,
                media_kind: MediaKind::Document,
            },
        };
        assert_eq!(r.text(), "downloaded");
    }

    #[test]
    fn resolve_env_vars_expands_known() {
        unsafe { std::env::set_var("TEST_TOOL_VAR_XYZ", "secret") };
        let result = resolve_env_vars("Bearer ${TEST_TOOL_VAR_XYZ}");
        assert_eq!(result, "Bearer secret");
    }

    #[test]
    fn resolve_env_vars_unknown_becomes_empty() {
        let result = resolve_env_vars("${NONEXISTENT_VAR_12345}");
        assert_eq!(result, "");
    }

    #[tokio::test]
    async fn execute_unknown_tool_errors() {
        let executor = default_tools(test_fetcher());
        let id = uuid::Uuid::new_v4();
        let result = executor
            .execute(
                "nonexistent",
                &serde_json::json!({}),
                id,
                std::path::Path::new("/tmp"),
            )
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn execute_scrape_page_missing_url_arg_errors() {
        let executor = default_tools(test_fetcher());
        let id = uuid::Uuid::new_v4();
        let result = executor
            .execute(
                "scrape_page",
                &serde_json::json!({}),
                id,
                std::path::Path::new("/tmp"),
            )
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn execute_download_file_missing_url_arg_errors() {
        let executor = default_tools(test_fetcher());
        let id = uuid::Uuid::new_v4();
        let result = executor
            .execute(
                "download_file",
                &serde_json::json!({}),
                id,
                std::path::Path::new("/tmp"),
            )
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn execute_crawl_url_missing_url_arg_errors() {
        let tools = vec![Tool {
            name: "crawl_url".into(),
            description: "crawl".into(),
            enabled: true,
            backend: ToolBackendConfig::Crawler {
                endpoint: "http://localhost:11235/crawl".into(),
                auth_header: None,
                timeout_secs: 5,
                priority: 10,
            },
        }];
        let executor = ToolExecutor::new(tools, test_fetcher());
        let id = uuid::Uuid::new_v4();
        let result = executor
            .execute(
                "crawl_url",
                &serde_json::json!({}),
                id,
                std::path::Path::new("/tmp"),
            )
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn execute_crawl_url_with_wrong_backend_errors() {
        let tools = vec![Tool {
            name: "crawl_url".into(),
            description: "crawl".into(),
            enabled: true,
            backend: ToolBackendConfig::Internal,
        }];
        let executor = ToolExecutor::new(tools, test_fetcher());
        let id = uuid::Uuid::new_v4();
        let result = executor
            .execute(
                "crawl_url",
                &serde_json::json!({"url":"https://example.com"}),
                id,
                std::path::Path::new("/tmp"),
            )
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn execute_scrape_page_with_crawler_backend_errors() {
        let tools = vec![Tool {
            name: "scrape_page".into(),
            description: "scrape".into(),
            enabled: true,
            backend: ToolBackendConfig::Crawler {
                endpoint: "http://localhost:11235/crawl".into(),
                auth_header: None,
                timeout_secs: 5,
                priority: 10,
            },
        }];
        let executor = ToolExecutor::new(tools, test_fetcher());
        let id = uuid::Uuid::new_v4();
        let result = executor
            .execute(
                "scrape_page",
                &serde_json::json!({"url":"https://example.com"}),
                id,
                std::path::Path::new("/tmp"),
            )
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn execute_download_file_with_crawler_backend_errors() {
        let tools = vec![Tool {
            name: "download_file".into(),
            description: "download".into(),
            enabled: true,
            backend: ToolBackendConfig::Crawler {
                endpoint: "http://localhost:11235/crawl".into(),
                auth_header: None,
                timeout_secs: 5,
                priority: 10,
            },
        }];
        let executor = ToolExecutor::new(tools, test_fetcher());
        let id = uuid::Uuid::new_v4();
        let result = executor
            .execute(
                "download_file",
                &serde_json::json!({"url":"https://example.com"}),
                id,
                std::path::Path::new("/tmp"),
            )
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn run_shell_tool_empty_argv_errors() {
        let result = run_shell_tool(&[], "http://x.com", "", 5).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn run_shell_tool_echo() {
        let argv = vec!["echo".to_owned(), "hello {url}".to_owned()];
        let result = run_shell_tool(&argv, "world", "", 5).await.unwrap();
        assert!(result.text().contains("hello world"));
    }

    #[tokio::test]
    async fn run_http_tool_plain_text_response() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/content"))
            .respond_with(ResponseTemplate::new(200).set_body_string("page text"))
            .mount(&server)
            .await;

        let client = reqwest::Client::new();
        let cfg = HttpToolCfg {
            endpoint: &format!("{}/content", server.uri()),
            method: "GET",
            auth_header: None,
            body_template: None,
            response_path: "",
            timeout_secs: 5,
        };
        let result = run_http_tool(&client, cfg, "http://x.com", "")
            .await
            .unwrap();
        assert_eq!(result.text(), "page text");
    }

    #[tokio::test]
    async fn run_http_tool_json_path_extraction() {
        let server = MockServer::start().await;
        let body = serde_json::json!({"data": {"content": "extracted text"}});
        Mock::given(method("GET"))
            .and(path("/json"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "application/json")
                    .set_body_json(body),
            )
            .mount(&server)
            .await;

        let client = reqwest::Client::new();
        let cfg = HttpToolCfg {
            endpoint: &format!("{}/json", server.uri()),
            method: "GET",
            auth_header: None,
            body_template: None,
            response_path: "data.content",
            timeout_secs: 5,
        };
        let result = run_http_tool(&client, cfg, "", "").await.unwrap();
        assert_eq!(result.text(), "extracted text");
    }

    #[tokio::test]
    async fn run_http_tool_post_with_body() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/post"))
            .respond_with(ResponseTemplate::new(200).set_body_string("ok"))
            .mount(&server)
            .await;

        let client = reqwest::Client::new();
        let endpoint = format!("{}/post", server.uri());
        let cfg = HttpToolCfg {
            endpoint: &endpoint,
            method: "POST",
            auth_header: Some("X-Key: myvalue"),
            body_template: Some(r#"{"url":"{url}"}"#),
            response_path: "",
            timeout_secs: 5,
        };
        let result = run_http_tool(&client, cfg, "http://x.com", "")
            .await
            .unwrap();
        assert_eq!(result.text(), "ok");
    }

    #[tokio::test]
    async fn run_crawler_tool_markdown_then_html_fallback() {
        let server = MockServer::start().await;
        let body = serde_json::json!({
            "success": true,
            "results": [{
                "metadata": {"title": "T"},
                "markdown": {"raw_markdown": "# md"},
                "cleaned_html": "<p>html</p>"
            }]
        });
        Mock::given(method("POST"))
            .and(path("/crawl"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "application/json")
                    .set_body_json(body),
            )
            .mount(&server)
            .await;

        let client = reqwest::Client::new();
        let cfg = CrawlToolCfg {
            endpoint: &format!("{}/crawl", server.uri()),
            auth_header: None,
            timeout_secs: 5,
            priority: 10,
        };
        let result = run_crawler_tool(&client, cfg, "http://x.com")
            .await
            .unwrap();
        assert!(result.text().contains("Title: T"));
        assert!(result.text().contains("# md"));
    }

    #[tokio::test]
    async fn run_crawler_tool_uses_html_when_markdown_missing() {
        let server = MockServer::start().await;
        let body = serde_json::json!({
            "success": true,
            "results": [{
                "metadata": {"title": "T"},
                "markdown": {"raw_markdown": ""},
                "cleaned_html": "<p>html only</p>"
            }]
        });
        Mock::given(method("POST"))
            .and(path("/crawl"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "application/json")
                    .set_body_json(body),
            )
            .mount(&server)
            .await;

        let client = reqwest::Client::new();
        let cfg = CrawlToolCfg {
            endpoint: &format!("{}/crawl", server.uri()),
            auth_header: None,
            timeout_secs: 5,
            priority: 10,
        };
        let result = run_crawler_tool(&client, cfg, "http://x.com")
            .await
            .unwrap();
        assert!(result.text().contains("HTML fallback:"));
        assert!(result.text().contains("html only"));
    }

    #[tokio::test]
    async fn run_crawler_tool_errors_when_results_missing() {
        let server = MockServer::start().await;
        let body = serde_json::json!({
            "success": true,
            "results": []
        });
        Mock::given(method("POST"))
            .and(path("/crawl"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "application/json")
                    .set_body_json(body),
            )
            .mount(&server)
            .await;

        let client = reqwest::Client::new();
        let cfg = CrawlToolCfg {
            endpoint: &format!("{}/crawl", server.uri()),
            auth_header: None,
            timeout_secs: 5,
            priority: 10,
        };
        let result = run_crawler_tool(&client, cfg, "http://x.com").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn run_crawler_tool_handles_empty_content() {
        let server = MockServer::start().await;
        let body = serde_json::json!({
            "success": true,
            "results": [{
                "metadata": {"title": ""},
                "markdown": {"raw_markdown": ""},
                "cleaned_html": ""
            }]
        });
        Mock::given(method("POST"))
            .and(path("/crawl"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "application/json")
                    .set_body_json(body),
            )
            .mount(&server)
            .await;

        let client = reqwest::Client::new();
        let cfg = CrawlToolCfg {
            endpoint: &format!("{}/crawl", server.uri()),
            auth_header: None,
            timeout_secs: 5,
            priority: 10,
        };
        let result = run_crawler_tool(&client, cfg, "http://x.com")
            .await
            .expect("crawler result");
        assert!(result.text().contains("no markdown/html"));
    }

    #[test]
    fn from_tooling_builds_executor() {
        let cfg = crate::config::ToolingConfig::default();
        let executor = from_tooling(&cfg, test_fetcher());
        assert!(!executor.active_tool_definitions().is_empty());
    }
}
