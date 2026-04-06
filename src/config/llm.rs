use serde::Deserialize;

use super::infra::bool_true;

// ── LLM ───────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct LlmConfig {
    #[serde(default)]
    pub fallback: FallbackMode,
    #[serde(default = "default_url_content_max_chars")]
    pub url_content_max_chars: usize,
    #[serde(default = "default_max_tool_turns")]
    pub max_tool_turns: usize,
    /// Max depth for recursive `llm_call` tool invocations. Default: 1.
    #[serde(default = "default_max_llm_tool_depth")]
    pub max_llm_tool_depth: u32,
    /// Retries for individual LLM API calls within a tool loop (e.g. on network blip).
    /// Default: 2.
    #[serde(default = "default_inner_retries")]
    pub inner_retries: u32,
    /// Maximum image file size (bytes) to send to the LLM for vision analysis.
    /// Images larger than this are silently skipped.
    #[serde(default = "default_vision_max_bytes")]
    pub vision_max_bytes: usize,
    /// Maximum characters of a single tool result appended to the LLM context.
    /// Prevents context overflow from large scraped pages. `0` disables truncation.
    #[serde(default = "default_tool_result_max_chars")]
    pub tool_result_max_chars: usize,
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
fn default_max_llm_tool_depth() -> u32 {
    1
}
fn default_inner_retries() -> u32 {
    2
}
fn default_vision_max_bytes() -> usize {
    5 * 1024 * 1024
}
fn default_tool_result_max_chars() -> usize {
    20_000
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
    /// Appended to the system prompt when image attachments are present.
    #[serde(default = "default_vision_prompt_note")]
    pub vision_prompt_note: String,
}

impl Default for LlmPromptsConfig {
    fn default() -> Self {
        Self {
            base_system: default_base_system_prompt(),
            tool_guidance_header: default_tool_guidance_header(),
            js_shell_tool_hint: default_js_shell_tool_hint(),
            require_tool_for_urls: true,
            url_tool_decision: default_url_tool_decision(),
            vision_prompt_note: default_vision_prompt_note(),
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

fn default_vision_prompt_note() -> String {
    "Images are attached. Include a description of each image's content in your summary.".into()
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
    /// Ollama only: explicitly enable (`true`) or disable (`false`) the model's
    /// extended thinking/reasoning mode. `null` / omitted = model default.
    #[serde(default)]
    pub think: Option<bool>,
    /// Extended timeout used when thinking mode is active (Ollama only).
    /// Should be significantly longer than `timeout_secs`. `None` = use `timeout_secs`.
    pub think_timeout_secs: Option<u64>,
    /// Whether this backend supports the `think` field and the `activate_thinking` tool.
    /// Set to `true` for Ollama models that have a built-in reasoning mode (e.g. qwq, deepseek-r1).
    /// Defaults to `false`.
    #[serde(default)]
    pub thinking_supported: bool,
    /// Maximum number of concurrent in-flight requests to this backend.
    /// `None` means unlimited. Set to `1` for local Ollama instances.
    pub max_concurrent: Option<usize>,
    /// Context window size in tokens. For Ollama, sent as `options.num_ctx`.
    /// Not used for `OpenRouter` (cloud manages context).
    #[serde(default)]
    pub context_size: Option<usize>,
    /// TCP connection timeout in seconds, separate from the response read timeout.
    /// Allows fast failure when the server is not reachable, while still allowing
    /// `timeout_secs` to be long for slow CPU-based inference.
    /// Default: 10 seconds.
    #[serde(default = "default_connect_timeout_secs")]
    pub connect_timeout_secs: u64,
    /// Ollama only. After a connection failure (not a timeout), skip this backend
    /// for this many seconds before retrying. Prevents repeated failed attempts when
    /// the Ollama server is unreachable. Set to 0 to disable. Default: 300 seconds.
    #[serde(default = "default_circuit_open_secs")]
    pub circuit_open_secs: u64,
}

fn default_connect_timeout_secs() -> u64 {
    10
}

fn default_circuit_open_secs() -> u64 {
    300
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
