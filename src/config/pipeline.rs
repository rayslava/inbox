use serde::Deserialize;

// ── Pipeline ──────────────────────────────────────────────────────────────────

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
