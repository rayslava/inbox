use serde::Deserialize;

// ── Pipeline ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Default, Deserialize)]
pub struct PipelineConfig {
    #[serde(default)]
    pub web_content: WebContentConfig,
    #[serde(default)]
    pub preprocessing: PreprocessingConfig,
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
