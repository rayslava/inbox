use serde::Deserialize;

use crate::error::InboxError;

pub mod adapters;
pub mod infra;
pub mod llm;
pub mod memory;
pub mod pipeline;
pub mod tooling;

// Re-export everything so callers can use `crate::config::Foo` as before.
pub use adapters::{AdaptersConfig, EmailConfig, HttpAdapterConfig, TelegramConfig};
pub use infra::{AdminConfig, GeneralConfig, SyncthingConfig, UrlFetchConfig, WebUiConfig};
pub use llm::{FallbackMode, LlmBackendConfig, LlmBackendType, LlmConfig, LlmPromptsConfig};
pub use memory::MemoryConfig;
pub use pipeline::{
    JsShellPolicy, PipelineConfig, PreprocessingConfig, PreprocessingRule, RuleAction,
    RuleCondition, WebContentConfig,
};
pub use tooling::{
    CrawlToolConfig, DuckDuckGoSearchToolConfig, KagiSearchToolConfig, NamedToolConfig,
    ToolBackendConfig, ToolingConfig,
};

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
    #[serde(default)]
    pub memory: MemoryConfig,
}

// ── Loading ───────────────────────────────────────────────────────────────────

/// Load config from a TOML file, interpolating `${VAR}` from the environment.
///
/// # Errors
/// Returns an error if the file cannot be read or the TOML is invalid.
pub fn load(path: &std::path::Path) -> Result<Config, InboxError> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| InboxError::Config(format!("{}: {e}", path.display())))?;
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
    use std::io::Write;

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

    #[test]
    fn load_interpolates_env_and_deserializes() {
        // SAFETY: single-threaded test, no other threads reading this env var
        unsafe { std::env::set_var("INBOX_TEST_OUTPUT", "out.org") };

        let mut file = tempfile::NamedTempFile::new().expect("tempfile");
        writeln!(
            file,
            r#"
[general]
output_file = "${{INBOX_TEST_OUTPUT}}"
attachments_dir = "attachments"

[llm]
"#
        )
        .expect("write config");

        let cfg = load(file.path()).expect("load config");
        assert_eq!(cfg.general.output_file, std::path::PathBuf::from("out.org"));
        assert_eq!(
            cfg.general.attachments_dir,
            std::path::PathBuf::from("attachments")
        );
    }

    #[test]
    fn load_invalid_toml_returns_config_error() {
        let mut file = tempfile::NamedTempFile::new().expect("tempfile");
        writeln!(file, "[general").expect("write config");

        let err = load(file.path()).expect_err("must fail");
        assert!(matches!(err, InboxError::Config(_)));
    }
}
