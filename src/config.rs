use std::net::SocketAddr;
use std::path::PathBuf;

use anodized::spec;
use serde::Deserialize;

use crate::error::InboxError;

// ── Top-level ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub general: GeneralConfig,
    #[serde(default)]
    pub admin: AdminConfig,
    #[serde(default)]
    pub web_ui: WebUiConfig,
    #[serde(default)]
    pub pipeline: PipelineConfig,
    pub llm: LlmConfig,
    #[serde(default)]
    pub adapters: AdaptersConfig,
    #[serde(default)]
    pub url_fetch: UrlFetchConfig,
    #[serde(default)]
    pub syncthing: SyncthingConfig,
    #[serde(default)]
    pub tooling: ToolingConfig,
}

// ── General ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct GeneralConfig {
    pub output_file: PathBuf,
    pub attachments_dir: PathBuf,
    #[serde(default = "default_log_level")]
    pub log_level: String,
    #[serde(default = "default_log_format")]
    pub log_format: String,
}

fn default_log_level() -> String {
    "info".into()
}
fn default_log_format() -> String {
    "pretty".into()
}

// ── Admin ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct AdminConfig {
    #[serde(default = "default_admin_bind")]
    pub bind_addr: SocketAddr,
    #[serde(default = "default_admin_user")]
    pub username: String,
    /// Argon2id hash generated via `inbox hash-password`.
    #[serde(default)]
    pub password_hash: String,
    #[serde(default = "default_session_ttl")]
    pub session_ttl_days: u64,
    #[serde(default = "default_drain_secs")]
    pub shutdown_drain_secs: u64,
}

impl Default for AdminConfig {
    fn default() -> Self {
        Self {
            bind_addr: default_admin_bind(),
            username: default_admin_user(),
            password_hash: String::new(),
            session_ttl_days: default_session_ttl(),
            shutdown_drain_secs: default_drain_secs(),
        }
    }
}

fn default_admin_bind() -> SocketAddr {
    "0.0.0.0:9090".parse().unwrap()
}
fn default_admin_user() -> String {
    "admin".into()
}
fn default_session_ttl() -> u64 {
    7
}
fn default_drain_secs() -> u64 {
    5
}

// ── Web UI ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct WebUiConfig {
    #[serde(default = "bool_true")]
    pub enabled: bool,
}

impl Default for WebUiConfig {
    fn default() -> Self {
        Self { enabled: true }
    }
}

fn bool_true() -> bool {
    true
}

// ── LLM ──────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct LlmConfig {
    #[serde(default)]
    pub fallback: FallbackMode,
    #[serde(default = "default_url_content_max_chars")]
    pub url_content_max_chars: usize,
    #[serde(default = "default_max_tool_turns")]
    pub max_tool_turns: usize,
    #[serde(default)]
    pub prompts: LlmPromptsConfig,
    #[serde(default)]
    pub backends: Vec<LlmBackendConfig>,
}

fn default_url_content_max_chars() -> usize {
    4000
}
fn default_max_tool_turns() -> usize {
    5
}

#[derive(Debug, Clone, Deserialize)]
pub struct LlmPromptsConfig {
    #[serde(default = "default_base_system_prompt")]
    pub base_system: String,
    #[serde(default = "default_tool_guidance_header")]
    pub tool_guidance_header: String,
    #[serde(default = "default_js_shell_tool_hint")]
    pub js_shell_tool_hint: String,
    #[serde(default = "bool_true")]
    pub require_tool_for_urls: bool,
    #[serde(default = "default_url_tool_decision")]
    pub url_tool_decision: String,
}

impl Default for LlmPromptsConfig {
    fn default() -> Self {
        Self {
            base_system: default_base_system_prompt(),
            tool_guidance_header: default_tool_guidance_header(),
            js_shell_tool_hint: default_js_shell_tool_hint(),
            require_tool_for_urls: true,
            url_tool_decision: default_url_tool_decision(),
        }
    }
}

fn default_base_system_prompt() -> String {
    r#"You are a personal inbox assistant. Given a captured note or web content, respond with a JSON object containing:
- "title": a short descriptive title (max 80 chars)
- "tags": array of relevant tag strings (max 5, lowercase, no spaces — use underscores)
- "summary": a 1-3 sentence summary of the content
- "excerpt": (optional) a single key quote or sentence worth preserving verbatim, or null

Respond ONLY with the JSON object, no markdown fences."#
        .into()
}

fn default_tool_guidance_header() -> String {
    "Tool-specific guidance:".into()
}

fn default_js_shell_tool_hint() -> String {
    "If URL content appears to be a JavaScript shell, call crawl_url for that URL and prefer markdown output.".into()
}

fn default_url_tool_decision() -> String {
    "When URLs are present, decide the best retrieval tool first and call it before producing final JSON. Use crawl_url for JS-heavy/app-shell pages, scrape_page for normal readable pages, and download_file for direct file links. URLs: {urls}".into()
}

#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FallbackMode {
    #[default]
    Raw,
    Discard,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LlmBackendConfig {
    #[serde(rename = "type")]
    pub backend_type: LlmBackendType,
    pub model: String,
    pub api_key: Option<String>,
    #[serde(default = "default_openrouter_base_url")]
    pub base_url: String,
    #[serde(default = "default_retries")]
    pub retries: u32,
    #[serde(default = "default_timeout_secs")]
    pub timeout_secs: u64,
}

fn default_openrouter_base_url() -> String {
    "https://openrouter.ai/api/v1".into()
}
fn default_retries() -> u32 {
    3
}
fn default_timeout_secs() -> u64 {
    30
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LlmBackendType {
    Openrouter,
    Ollama,
}

// ── Pipeline ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Default, Deserialize)]
pub struct PipelineConfig {
    #[serde(default)]
    pub web_content: WebContentConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct WebContentConfig {
    #[serde(default)]
    pub js_shell_policy: JsShellPolicy,
    #[serde(default = "default_js_shell_patterns")]
    pub js_shell_patterns: Vec<String>,
}

impl Default for WebContentConfig {
    fn default() -> Self {
        Self {
            js_shell_policy: JsShellPolicy::default(),
            js_shell_patterns: default_js_shell_patterns(),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum JsShellPolicy {
    #[default]
    Allow,
    ToolOnly,
    Drop,
}

fn default_js_shell_patterns() -> Vec<String> {
    vec![
        "doesn't work properly without javascript enabled".into(),
        "please enable it to continue".into(),
        "requires javascript".into(),
    ]
}

// ── Adapters ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Default, Deserialize)]
pub struct AdaptersConfig {
    #[serde(default)]
    pub telegram: TelegramConfig,
    #[serde(default)]
    pub http: HttpAdapterConfig,
    #[serde(default)]
    pub email: EmailConfig,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct TelegramConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub bot_token: String,
    #[serde(default)]
    pub allowed_user_ids: Vec<i64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct HttpAdapterConfig {
    #[serde(default = "bool_true")]
    pub enabled: bool,
    #[serde(default = "default_http_bind")]
    pub bind_addr: SocketAddr,
    #[serde(default)]
    pub auth_token: String,
}

impl Default for HttpAdapterConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            bind_addr: default_http_bind(),
            auth_token: String::new(),
        }
    }
}

fn default_http_bind() -> SocketAddr {
    "0.0.0.0:8080".parse().unwrap()
}

#[derive(Debug, Clone, Deserialize)]
pub struct EmailConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub host: String,
    #[serde(default = "default_imap_port")]
    pub port: u16,
    #[serde(default)]
    pub username: String,
    #[serde(default)]
    pub password: String,
    #[serde(default = "default_mailbox")]
    pub mailbox: String,
    #[serde(default = "bool_true")]
    pub mark_as_seen: bool,
    #[serde(default)]
    pub processed_mailbox: String,
}

impl Default for EmailConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            host: String::new(),
            port: default_imap_port(),
            username: String::new(),
            password: String::new(),
            mailbox: default_mailbox(),
            mark_as_seen: true,
            processed_mailbox: String::new(),
        }
    }
}

fn default_imap_port() -> u16 {
    993
}
fn default_mailbox() -> String {
    "INBOX".into()
}

// ── URL Fetch ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct UrlFetchConfig {
    #[serde(default = "bool_true")]
    pub enabled: bool,
    #[serde(default = "default_fetch_timeout")]
    pub timeout_secs: u64,
    #[serde(default = "default_max_redirects")]
    pub max_redirects: u32,
    #[serde(default = "default_max_body_bytes")]
    pub max_body_bytes: usize,
    #[serde(default = "default_user_agent")]
    pub user_agent: String,
    #[serde(default)]
    pub skip_domains: Vec<String>,
}

impl Default for UrlFetchConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            timeout_secs: default_fetch_timeout(),
            max_redirects: default_max_redirects(),
            max_body_bytes: default_max_body_bytes(),
            user_agent: default_user_agent(),
            skip_domains: Vec::new(),
        }
    }
}

fn default_fetch_timeout() -> u64 {
    10
}
fn default_max_redirects() -> u32 {
    5
}
fn default_max_body_bytes() -> usize {
    5 * 1024 * 1024
}
fn default_user_agent() -> String {
    "inbox-bot/1.0".into()
}

// ── Syncthing ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Default, Deserialize)]
pub struct SyncthingConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_syncthing_url")]
    pub api_url: String,
    #[serde(default)]
    pub api_key: String,
    /// Syncthing folder ID that contains the org output file.
    #[serde(default)]
    pub org_folder_id: String,
    /// Syncthing folder ID for attachments (may differ from org folder).
    #[serde(default)]
    pub attachments_folder_id: Option<String>,
    #[serde(default = "bool_true")]
    pub rescan_on_write: bool,
}

fn default_syncthing_url() -> String {
    "http://localhost:8384".into()
}

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

// ── Loading ───────────────────────────────────────────────────────────────────

/// Load config from a TOML file, interpolating `${VAR}` from the environment.
///
/// # Errors
/// Returns an error if the file cannot be read or the TOML is invalid.
#[spec(requires: path.exists())]
pub fn load(path: &std::path::Path) -> Result<Config, InboxError> {
    let raw = std::fs::read_to_string(path).map_err(InboxError::Io)?;
    let interpolated = interpolate_env(&raw);
    toml::from_str(&interpolated).map_err(|e| InboxError::Config(e.to_string()))
}

/// Replace `${VAR_NAME}` occurrences with the value of the named env var.
/// Unknown variables are left as-is (to avoid masking typos as empty strings).
fn interpolate_env(s: &str) -> String {
    let re = regex::Regex::new(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}").unwrap();
    re.replace_all(s, |caps: &regex::Captures<'_>| {
        let var = &caps[1];
        std::env::var(var).unwrap_or_else(|_| caps[0].to_owned())
    })
    .into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interpolate_known_var() {
        // SAFETY: single-threaded test, no other threads reading this env var
        unsafe { std::env::set_var("TEST_TOKEN_XYZ", "secret123") };
        let result = interpolate_env("token = \"${TEST_TOKEN_XYZ}\"");
        assert_eq!(result, "token = \"secret123\"");
    }

    #[test]
    fn interpolate_unknown_var_unchanged() {
        let result = interpolate_env("x = \"${DEFINITELY_NOT_SET_VAR_INBOX}\"");
        assert!(result.contains("${DEFINITELY_NOT_SET_VAR_INBOX}"));
    }

    #[test]
    fn tooling_prompt_block_collects_enabled_nonempty_prompts() {
        let mut tooling = ToolingConfig::default();
        tooling.scrape_page.prompt = "prompt one".into();
        tooling.download_file.prompt = "prompt two".into();
        tooling.crawl_url.enabled = true;
        tooling.crawl_url.prompt = "prompt three".into();
        let block = tooling.prompt_block();
        assert!(block.contains("Tool scrape_page: prompt one"));
        assert!(block.contains("Tool download_file: prompt two"));
        assert!(block.contains("Tool crawl_url: prompt three"));
    }

    #[test]
    fn tooling_prompt_block_ignores_disabled_or_empty_prompts() {
        let mut tooling = ToolingConfig::default();
        tooling.scrape_page.prompt = String::new();
        tooling.download_file.enabled = false;
        tooling.download_file.prompt = "ignored".into();
        tooling.crawl_url.enabled = false;
        tooling.crawl_url.prompt = "ignored".into();
        let block = tooling.prompt_block();
        assert!(block.is_empty());
    }
}
