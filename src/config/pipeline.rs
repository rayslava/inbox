use std::path::PathBuf;

use serde::Deserialize;

// ── Pipeline ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct PipelineConfig {
    #[serde(default)]
    pub web_content: WebContentConfig,
    #[serde(default)]
    pub preprocessing: PreprocessingConfig,
    #[serde(default)]
    pub resume: ResumeConfig,
    #[serde(default)]
    pub image_analysis: ImageAnalysisConfig,
    /// Hashtags that mark a message as a plain memo: URL fetch and the LLM
    /// enrichment call are skipped and the content is written straight to org.
    /// Image OCR still runs, so an image memo sent while vision is down is held
    /// pending and re-OCR'd on resume rather than finalized empty.
    #[serde(default = "default_memo_tags")]
    pub memo_tags: Vec<String>,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            web_content: WebContentConfig::default(),
            preprocessing: PreprocessingConfig::default(),
            resume: ResumeConfig::default(),
            image_analysis: ImageAnalysisConfig::default(),
            memo_tags: default_memo_tags(),
        }
    }
}

fn default_memo_tags() -> Vec<String> {
    vec!["memo".into()]
}

// ── Image analysis ─────────────────────────────────────────────────────────────

/// Configuration for the vision-LLM image-analysis stage that classifies and
/// transcribes image attachments before pre-processing.
#[derive(Debug, Clone, Deserialize)]
pub struct ImageAnalysisConfig {
    /// Enable image analysis (classify + transcribe via a vision model).
    #[serde(default = "super::infra::bool_true")]
    pub enabled: bool,
    /// System prompt instructing the vision model to transcribe visible text.
    #[serde(default = "default_image_analysis_prompt")]
    pub prompt: String,
    /// Maximum images to analyze per message. Default: 4.
    #[serde(default = "default_image_max_attachments")]
    pub max_attachments: usize,
    /// Minimum recognized-text length (chars) to classify an image as an
    /// interface/screenshot rather than a plain photo. Default: 24.
    #[serde(default = "default_interface_min_chars")]
    pub interface_min_chars: usize,
}

impl Default for ImageAnalysisConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            prompt: default_image_analysis_prompt(),
            max_attachments: default_image_max_attachments(),
            interface_min_chars: default_interface_min_chars(),
        }
    }
}

fn default_image_analysis_prompt() -> String {
    "Transcribe all text visible in this image, preserving line breaks. \
     If the image contains no readable text, reply with an empty response."
        .into()
}

fn default_image_max_attachments() -> usize {
    4
}

fn default_interface_min_chars() -> usize {
    24
}

// ── Incomplete-processing resume ───────────────────────────────────────────────

/// Configuration for background retry of messages that fell back to raw mode.
#[derive(Debug, Clone, Deserialize)]
pub struct ResumeConfig {
    /// Enable the background resume task.
    #[serde(default)]
    pub enabled: bool,
    /// How often (seconds) to scan for pending items when idle. Default: 300 (5 min).
    #[serde(default = "default_resume_interval_secs")]
    pub interval_secs: u64,
    /// Maximum retry attempts before giving up. Default: 5.
    #[serde(default = "default_resume_max_retries")]
    pub max_retries: u32,
    /// Path to the pending `SQLite` database.
    /// Defaults to `{attachments_dir}/pending.db`.
    #[serde(default)]
    pub db_path: Option<PathBuf>,
}

impl Default for ResumeConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            interval_secs: default_resume_interval_secs(),
            max_retries: default_resume_max_retries(),
            db_path: None,
        }
    }
}

fn default_resume_interval_secs() -> u64 {
    300
}

fn default_resume_max_retries() -> u32 {
    5
}

// ── Pre-processing rules ───────────────────────────────────────────────────────

/// Configuration for the pre-processing stage that runs before URL enrichment.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct PreprocessingConfig {
    /// Rules evaluated in order; all matching rules are applied.
    #[serde(default)]
    pub rules: Vec<PreprocessingRule>,
}

/// A single pre-processing rule.
#[derive(Debug, Clone, Deserialize)]
pub struct PreprocessingRule {
    /// Human-readable name for logging.
    pub name: String,
    /// Condition that must be true for the rule to fire.
    pub condition: RuleCondition,
    /// Numeric threshold used by conditions that need one (e.g. `text_word_count_lt`).
    pub threshold: Option<usize>,
    /// Action to take when the condition matches.
    pub action: RuleAction,
    /// Tag to add (used by the `add_tag` action).
    pub tag: Option<String>,
    /// Extra guidance appended to the LLM system prompt when the rule fires.
    pub llm_hint: Option<String>,
}

/// Condition variants for pre-processing rules.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuleCondition {
    /// True when the message text contains fewer than `threshold` whitespace-separated words.
    TextWordCountLt,
    /// True when at least one image attachment is present.
    HasImageAttachment,
    /// True when at least one attachment of any kind is present.
    HasAttachment,
}

/// Action variants for pre-processing rules.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuleAction {
    /// Set the `force_web_search` hint so the LLM is guided to call `web_search`.
    ForceWebSearch,
    /// Add `tag` to the `suggested_tags` hint (merged into the org output).
    AddTag,
    /// Append `llm_hint` to the extra hints block without any other side effects.
    AddLlmHint,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_analysis_defaults() {
        let cfg: PipelineConfig = toml::from_str("").expect("empty config parses");
        assert!(cfg.image_analysis.enabled);
        assert_eq!(cfg.image_analysis.max_attachments, 4);
        assert_eq!(cfg.image_analysis.interface_min_chars, 24);
        assert!(!cfg.image_analysis.prompt.is_empty());
    }

    #[test]
    fn image_analysis_parses_overrides() {
        let cfg: PipelineConfig = toml::from_str(
            r#"
[image_analysis]
enabled = false
max_attachments = 2
interface_min_chars = 50
prompt = "custom"
"#,
        )
        .expect("config parses");
        assert!(!cfg.image_analysis.enabled);
        assert_eq!(cfg.image_analysis.max_attachments, 2);
        assert_eq!(cfg.image_analysis.interface_min_chars, 50);
        assert_eq!(cfg.image_analysis.prompt, "custom");
    }
}
