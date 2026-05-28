use super::build_enabled;

fn minimal_config() -> crate::config::Config {
    use crate::config::{
        AdaptersConfig, AdminConfig, Config, GeneralConfig, PipelineConfig, SyncthingConfig,
        ToolingConfig, UrlFetchConfig, WebUiConfig,
    };
    Config {
        general: GeneralConfig {
            output_file: std::path::PathBuf::from("/tmp/inbox-test.org"),
            attachments_dir: std::path::PathBuf::from("/tmp/inbox-test-att"),
            log_level: "info".into(),
            log_format: "pretty".into(),
        },
        admin: AdminConfig::default(),
        web_ui: WebUiConfig::default(),
        pipeline: PipelineConfig::default(),
        llm: crate::test_helpers::no_llm_config(),
        // HttpAdapterConfig::default() has enabled = true, the others false.
        adapters: AdaptersConfig::default(),
        url_fetch: UrlFetchConfig::default(),
        syncthing: SyncthingConfig::default(),
        tooling: ToolingConfig::default(),
        memory: crate::config::MemoryConfig::default(),
    }
}

#[test]
fn build_enabled_returns_only_http_by_default() {
    let cfg = minimal_config();
    let adapters = build_enabled(&cfg, None);
    let names: Vec<&'static str> = adapters.iter().map(|a| a.name()).collect();
    assert_eq!(names, vec!["http"]);
}

#[test]
fn build_enabled_returns_empty_when_all_disabled() {
    let mut cfg = minimal_config();
    cfg.adapters.http.enabled = false;
    let adapters = build_enabled(&cfg, None);
    assert!(adapters.is_empty());
}

#[test]
fn build_enabled_includes_telegram_when_enabled() {
    let mut cfg = minimal_config();
    cfg.adapters.http.enabled = false;
    cfg.adapters.telegram.enabled = true;
    let adapters = build_enabled(&cfg, None);
    let names: Vec<&'static str> = adapters.iter().map(|a| a.name()).collect();
    assert_eq!(names, vec!["telegram"]);
}

#[test]
fn build_enabled_includes_email_when_enabled() {
    let mut cfg = minimal_config();
    cfg.adapters.http.enabled = false;
    cfg.adapters.email.enabled = true;
    let adapters = build_enabled(&cfg, None);
    let names: Vec<&'static str> = adapters.iter().map(|a| a.name()).collect();
    assert_eq!(names, vec!["email"]);
}

#[test]
fn build_enabled_returns_all_three_in_order() {
    let mut cfg = minimal_config();
    cfg.adapters.http.enabled = true;
    cfg.adapters.telegram.enabled = true;
    cfg.adapters.email.enabled = true;
    let adapters = build_enabled(&cfg, None);
    let names: Vec<&'static str> = adapters.iter().map(|a| a.name()).collect();
    assert_eq!(names, vec!["http", "telegram", "email"]);
}
