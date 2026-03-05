use serde::Deserialize;

use super::infra::bool_true;

// ── Tool config ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct NamedToolConfig {
    #[serde(default)]
    pub description: String,
    #[serde(default = "bool_true")]
    pub enabled: bool,
    #[serde(default)]
    pub prompt: String,
    #[serde(flatten)]
    pub backend: ToolBackendConfig,
}

impl Default for NamedToolConfig {
    fn default() -> Self {
        Self {
            description: String::new(),
            enabled: true,
            prompt: String::new(),
            backend: ToolBackendConfig::Internal,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct CrawlToolConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_crawl_description")]
    pub description: String,
    #[serde(default)]
    pub prompt: String,
    #[serde(default = "default_crawl_endpoint")]
    pub endpoint: String,
    #[serde(default)]
    pub auth_header: Option<String>,
    #[serde(default = "default_tool_timeout")]
    pub timeout_secs: u32,
    #[serde(default = "default_crawl_priority")]
    pub priority: i32,
}

impl Default for CrawlToolConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            description: default_crawl_description(),
            prompt: String::new(),
            endpoint: default_crawl_endpoint(),
            auth_header: None,
            timeout_secs: default_tool_timeout(),
            priority: default_crawl_priority(),
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct ToolingConfig {
    #[serde(default)]
    pub scrape_page: NamedToolConfig,
    #[serde(default)]
    pub download_file: NamedToolConfig,
    #[serde(default)]
    pub crawl_url: CrawlToolConfig,
}

impl ToolingConfig {
    #[must_use]
    pub fn prompt_block(&self) -> String {
        let mut lines = Vec::new();
        if self.scrape_page.enabled && !self.scrape_page.prompt.trim().is_empty() {
            lines.push(format!(
                "Tool scrape_page: {}",
                self.scrape_page.prompt.trim()
            ));
        }
        if self.download_file.enabled && !self.download_file.prompt.trim().is_empty() {
            lines.push(format!(
                "Tool download_file: {}",
                self.download_file.prompt.trim()
            ));
        }
        if self.crawl_url.enabled && !self.crawl_url.prompt.trim().is_empty() {
            lines.push(format!("Tool crawl_url: {}", self.crawl_url.prompt.trim()));
        }
        lines.join("\n")
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(tag = "backend", rename_all = "snake_case")]
pub enum ToolBackendConfig {
    #[default]
    Internal,
    Shell {
        argv: Vec<String>,
        #[serde(default = "default_tool_timeout")]
        timeout_secs: u32,
    },
    Http {
        endpoint: String,
        #[serde(default = "default_http_method")]
        method: String,
        #[serde(default)]
        auth_header: Option<String>,
        #[serde(default)]
        body_template: Option<String>,
        #[serde(default)]
        response_path: String,
        #[serde(default = "default_tool_timeout")]
        timeout_secs: u32,
    },
    Crawler {
        #[serde(default = "default_crawl_endpoint")]
        endpoint: String,
        #[serde(default)]
        auth_header: Option<String>,
        #[serde(default = "default_tool_timeout")]
        timeout_secs: u32,
        #[serde(default = "default_crawl_priority")]
        priority: i32,
    },
}

fn default_tool_timeout() -> u32 {
    15
}
fn default_http_method() -> String {
    "GET".into()
}
fn default_crawl_endpoint() -> String {
    "http://localhost:11235/crawl".into()
}
fn default_crawl_priority() -> i32 {
    10
}
fn default_crawl_description() -> String {
    "Crawl a URL and return markdown/html extracted by the crawler service".into()
}
