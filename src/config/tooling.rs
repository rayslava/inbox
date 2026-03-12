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
    /// Number of additional attempts after the first failure (0 = no retry).
    #[serde(default = "default_tool_retries")]
    pub retries: u32,
    #[serde(flatten)]
    pub backend: ToolBackendConfig,
}

impl Default for NamedToolConfig {
    fn default() -> Self {
        Self {
            description: String::new(),
            enabled: true,
            prompt: String::new(),
            retries: default_tool_retries(),
            backend: ToolBackendConfig::Internal {
                timeout_secs: default_tool_timeout(),
            },
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
    /// Number of additional attempts after the first failure (0 = no retry).
    #[serde(default = "default_tool_retries")]
    pub retries: u32,
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
            retries: default_tool_retries(),
        }
    }
}

/// Configuration for `[tooling.web_search]` — Kagi Search API backend.
#[derive(Debug, Clone, Deserialize)]
pub struct KagiSearchToolConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_kagi_description")]
    pub description: String,
    #[serde(default)]
    pub prompt: String,
    #[serde(default = "default_kagi_endpoint")]
    pub endpoint: String,
    #[serde(default)]
    pub api_token: Option<String>,
    #[serde(default = "default_tool_timeout")]
    pub timeout_secs: u32,
    #[serde(default = "default_web_search_limit")]
    pub default_limit: u32,
    #[serde(default = "default_web_search_max_snippet_chars")]
    pub max_snippet_chars: usize,
    /// Number of additional attempts after the first failure (0 = no retry).
    #[serde(default = "default_tool_retries")]
    pub retries: u32,
}

impl Default for KagiSearchToolConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            description: default_kagi_description(),
            prompt: String::new(),
            endpoint: default_kagi_endpoint(),
            api_token: None,
            timeout_secs: default_tool_timeout(),
            default_limit: default_web_search_limit(),
            max_snippet_chars: default_web_search_max_snippet_chars(),
            retries: default_tool_retries(),
        }
    }
}

/// Configuration for `[tooling.duckduckgo_search]` — `DuckDuckGo` HTML scraping backend.
#[derive(Debug, Clone, Deserialize)]
pub struct DuckDuckGoSearchToolConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_ddg_description")]
    pub description: String,
    #[serde(default)]
    pub prompt: String,
    #[serde(default = "default_ddg_endpoint")]
    pub endpoint: String,
    #[serde(default = "default_tool_timeout")]
    pub timeout_secs: u32,
    #[serde(default = "default_web_search_limit")]
    pub default_limit: u32,
    #[serde(default = "default_web_search_max_snippet_chars")]
    pub max_snippet_chars: usize,
    /// Number of additional attempts after the first failure (0 = no retry).
    #[serde(default = "default_tool_retries")]
    pub retries: u32,
}

impl Default for DuckDuckGoSearchToolConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            description: default_ddg_description(),
            prompt: String::new(),
            endpoint: default_ddg_endpoint(),
            timeout_secs: default_tool_timeout(),
            default_limit: default_web_search_limit(),
            max_snippet_chars: default_web_search_max_snippet_chars(),
            retries: default_tool_retries(),
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
    #[serde(default)]
    pub web_search: KagiSearchToolConfig,
    #[serde(default)]
    pub duckduckgo_search: DuckDuckGoSearchToolConfig,
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
        if self.web_search.enabled && !self.web_search.prompt.trim().is_empty() {
            lines.push(format!(
                "Tool web_search: {}",
                self.web_search.prompt.trim()
            ));
        }
        if self.duckduckgo_search.enabled && !self.duckduckgo_search.prompt.trim().is_empty() {
            lines.push(format!(
                "Tool duckduckgo_search: {}",
                self.duckduckgo_search.prompt.trim()
            ));
        }
        lines.join("\n")
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "backend", rename_all = "snake_case")]
pub enum ToolBackendConfig {
    Internal {
        #[serde(default = "default_tool_timeout")]
        timeout_secs: u32,
    },
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
    KagiSearch {
        #[serde(default = "default_kagi_endpoint")]
        endpoint: String,
        #[serde(default)]
        api_token: Option<String>,
        #[serde(default = "default_tool_timeout")]
        timeout_secs: u32,
        #[serde(default = "default_web_search_limit")]
        default_limit: u32,
        #[serde(default = "default_web_search_max_snippet_chars")]
        max_snippet_chars: usize,
    },
    DuckDuckGoSearch {
        #[serde(default = "default_ddg_endpoint")]
        endpoint: String,
        #[serde(default = "default_tool_timeout")]
        timeout_secs: u32,
        #[serde(default = "default_web_search_limit")]
        default_limit: u32,
        #[serde(default = "default_web_search_max_snippet_chars")]
        max_snippet_chars: usize,
    },
}

impl Default for ToolBackendConfig {
    fn default() -> Self {
        Self::Internal {
            timeout_secs: default_tool_timeout(),
        }
    }
}

fn default_tool_timeout() -> u32 {
    15
}
fn default_tool_retries() -> u32 {
    3
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
fn default_kagi_description() -> String {
    "Search the web via Kagi Search API and return top results".into()
}
fn default_kagi_endpoint() -> String {
    "https://kagi.com/api/v0/search".into()
}
fn default_ddg_description() -> String {
    "Search the web via DuckDuckGo and return top results".into()
}
fn default_ddg_endpoint() -> String {
    "https://duckduckgo.com/html/".into()
}
fn default_web_search_limit() -> u32 {
    5
}
fn default_web_search_max_snippet_chars() -> usize {
    320
}
