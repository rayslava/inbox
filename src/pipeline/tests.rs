use super::*;
use crate::config::{
    AdaptersConfig, AdminConfig, Config, GeneralConfig, PipelineConfig, SyncthingConfig,
    ToolingConfig, UrlFetchConfig, WebUiConfig,
};

fn test_config(policy: crate::config::JsShellPolicy) -> Config {
    Config {
        general: GeneralConfig {
            output_file: std::path::PathBuf::from("/tmp/inbox-test.org"),
            attachments_dir: std::path::PathBuf::from("/tmp/inbox-test-att"),
            log_level: "info".into(),
            log_format: "pretty".into(),
        },
        admin: AdminConfig::default(),
        web_ui: WebUiConfig::default(),
        pipeline: PipelineConfig {
            web_content: crate::config::WebContentConfig {
                js_shell_policy: policy,
                js_shell_patterns: vec![
                    "doesn't work properly without javascript enabled".into(),
                    "please enable it to continue".into(),
                ],
            },
            preprocessing: crate::config::PreprocessingConfig::default(),
        },
        llm: crate::test_helpers::no_llm_config(),
        adapters: AdaptersConfig::default(),
        url_fetch: UrlFetchConfig::default(),
        syncthing: SyncthingConfig::default(),
        tooling: ToolingConfig::default(),
    }
}

#[test]
fn truncate_chars_within_limit() {
    assert_eq!(truncate_chars("hello", 10), "hello");
}

#[test]
fn truncate_chars_at_limit() {
    assert_eq!(truncate_chars("hello", 5), "hello");
}

#[test]
fn truncate_chars_exceeds_limit() {
    assert_eq!(truncate_chars("hello world", 5), "hello");
}

#[test]
fn truncate_chars_unicode() {
    // "héllo" — 5 chars, each may be multi-byte
    let s = "héllo";
    assert_eq!(truncate_chars(s, 3), "hél");
}

#[test]
fn js_shell_match_respects_policy() {
    let cfg = test_config(crate::config::JsShellPolicy::ToolOnly);
    assert!(matches_js_shell_policy(
        &cfg,
        "This page doesn't work properly without JavaScript enabled"
    ));
}

#[test]
fn js_shell_match_disabled_when_policy_not_tool_only() {
    let cfg = test_config(crate::config::JsShellPolicy::Allow);
    assert!(!matches_js_shell_policy(
        &cfg,
        "This page doesn't work properly without JavaScript enabled"
    ));
}
