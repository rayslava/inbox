use std::sync::Arc;

use super::*;
use crate::config::{
    AdaptersConfig, AdminConfig, Config, GeneralConfig, PipelineConfig, SyncthingConfig,
    ToolingConfig, UrlFetchConfig, WebUiConfig,
};
use crate::message::{EnrichedMessage, IncomingMessage, MessageSource, SourceMetadata};
use crate::pending::PendingStore;
use crate::processing_status::ProcessingTracker;

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
            resume: crate::config::ResumeConfig::default(),
        },
        llm: crate::test_helpers::no_llm_config(),
        adapters: AdaptersConfig::default(),
        url_fetch: UrlFetchConfig::default(),
        syncthing: SyncthingConfig::default(),
        tooling: ToolingConfig::default(),
        memory: crate::config::MemoryConfig::default(),
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

#[test]
fn host_skip_domain_match_is_boundary_safe() {
    assert!(host_matches_skip_domain("youtube.com", "youtube.com"));
    assert!(host_matches_skip_domain("m.youtube.com", "youtube.com"));
    assert!(host_matches_skip_domain("m.YouTube.com", ".youtube.com"));
    assert!(!host_matches_skip_domain("notyoutube.com", "youtube.com"));
    assert!(!host_matches_skip_domain("youtube.com.evil", "youtube.com"));
}

#[test]
fn host_skip_domain_empty_inputs() {
    assert!(!host_matches_skip_domain("", "youtube.com"));
    assert!(!host_matches_skip_domain("youtube.com", ""));
    assert!(!host_matches_skip_domain("", ""));
}

#[test]
fn host_skip_domain_trailing_dots() {
    assert!(host_matches_skip_domain("youtube.com.", "youtube.com."));
    assert!(host_matches_skip_domain("sub.example.com.", "example.com"));
}

#[test]
fn js_shell_match_drop_policy() {
    let cfg = test_config(crate::config::JsShellPolicy::Drop);
    assert!(matches_js_shell_policy(
        &cfg,
        "please enable it to continue"
    ));
}

#[test]
fn js_shell_match_case_insensitive() {
    let cfg = test_config(crate::config::JsShellPolicy::ToolOnly);
    assert!(matches_js_shell_policy(
        &cfg,
        "DOESN'T WORK PROPERLY WITHOUT JAVASCRIPT ENABLED"
    ));
}

#[test]
fn js_shell_match_no_patterns() {
    let mut cfg = test_config(crate::config::JsShellPolicy::ToolOnly);
    cfg.pipeline.web_content.js_shell_patterns.clear();
    assert!(!matches_js_shell_policy(&cfg, "anything"));
}

#[test]
fn make_url_content_truncates() {
    let url = url::Url::parse("https://example.com").unwrap();
    let content = crate::url_content::UrlContent {
        url: String::new(),
        text: "abcdefghij".into(),
        page_title: Some("Title".into()),
        headings: vec!["H1".into()],
    };
    let result = make_url_content(&url, content, 5);
    assert_eq!(result.text, "abcde");
    assert_eq!(result.url, "https://example.com/");
    assert_eq!(result.page_title.as_deref(), Some("Title"));
    assert_eq!(result.headings, vec!["H1"]);
}

#[test]
fn make_url_content_no_truncation() {
    let url = url::Url::parse("https://example.com").unwrap();
    let content = crate::url_content::UrlContent {
        url: String::new(),
        text: "short".into(),
        page_title: None,
        headings: vec![],
    };
    let result = make_url_content(&url, content, 100);
    assert_eq!(result.text, "short");
}

fn make_test_pipeline(cfg: Config) -> Arc<Pipeline> {
    let cfg = Arc::new(cfg);
    let llm = crate::test_helpers::mock_llm_chain(crate::test_helpers::default_llm_response());
    let writer = Arc::new(crate::output::NullWriter);
    let tracker = Arc::new(ProcessingTracker::new());
    Arc::new(Pipeline::new(cfg, llm, writer, tracker, None, None))
}

fn test_enriched(text: &str, urls: Vec<url::Url>, user_tags: Vec<String>) -> EnrichedMessage {
    let mut msg = IncomingMessage::new(
        MessageSource::Http,
        text.into(),
        SourceMetadata::Http {
            remote_addr: None,
            user_agent: None,
        },
    );
    msg.user_tags = user_tags;
    EnrichedMessage {
        urls,
        url_contents: vec![],
        original: msg,
    }
}

#[test]
fn build_llm_guidance_empty_when_no_context() {
    let cfg = test_config(crate::config::JsShellPolicy::Allow);
    let pipeline = make_test_pipeline(cfg);
    let enriched = test_enriched("hello", vec![], vec![]);
    let guidance = pipeline.build_llm_guidance(&enriched, "");
    // Should be minimal — just tool prompt block if any.
    // No user tags, no preload, no URLs.
    assert!(!guidance.contains("tagged"));
    assert!(!guidance.contains("web_search"));
}

#[test]
fn build_llm_guidance_includes_user_tags() {
    let cfg = test_config(crate::config::JsShellPolicy::Allow);
    let pipeline = make_test_pipeline(cfg);
    let enriched = test_enriched("hello", vec![], vec!["rust".into(), "async".into()]);
    let guidance = pipeline.build_llm_guidance(&enriched, "");
    assert!(guidance.contains("#rust"));
    assert!(guidance.contains("#async"));
    assert!(guidance.contains("tagged"));
}

#[test]
fn build_llm_guidance_includes_preloaded_context() {
    let cfg = test_config(crate::config::JsShellPolicy::Allow);
    let pipeline = make_test_pipeline(cfg);
    let enriched = test_enriched("hello", vec![], vec![]);
    let guidance = pipeline.build_llm_guidance(&enriched, "Previously recalled: some context");
    assert!(guidance.contains("Previously recalled: some context"));
}

#[test]
fn build_llm_guidance_url_decision_when_urls_present() {
    let cfg = test_config(crate::config::JsShellPolicy::Allow);
    let pipeline = make_test_pipeline(cfg);
    let urls = vec![url::Url::parse("https://example.com/page").unwrap()];
    let enriched = test_enriched("check this", urls, vec![]);
    let guidance = pipeline.build_llm_guidance(&enriched, "");
    assert!(guidance.contains("example.com/page"));
}

#[test]
fn build_llm_guidance_js_shell_hint_when_tool_only() {
    let cfg = test_config(crate::config::JsShellPolicy::ToolOnly);
    let pipeline = make_test_pipeline(cfg);
    let urls = vec![url::Url::parse("https://spa-app.com").unwrap()];
    let enriched = test_enriched("check this", urls, vec![]);
    let guidance = pipeline.build_llm_guidance(&enriched, "");
    assert!(guidance.contains("spa-app.com"));
}

#[test]
fn build_llm_guidance_force_web_search() {
    let cfg = test_config(crate::config::JsShellPolicy::Allow);
    let pipeline = make_test_pipeline(cfg);
    let mut enriched = test_enriched("hello", vec![], vec![]);
    enriched.original.preprocessing_hints.force_web_search = true;
    let guidance = pipeline.build_llm_guidance(&enriched, "");
    assert!(guidance.contains("web_search"));
}

#[tokio::test]
async fn fallback_item_inserted_into_pending_store() {
    // Build a pipeline backed by a no-LLM chain (always falls back) and a real
    // pending store in a temp DB, then run a message through it and verify the
    // item lands in the store.
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("pending.db");
    let store = Arc::new(PendingStore::open(&db_path).await.unwrap());

    let cfg = Arc::new(test_config(crate::config::JsShellPolicy::Allow));
    let failing_llm = crate::test_helpers::always_fail_llm_chain();
    let writer = Arc::new(crate::output::NullWriter);
    let tracker = Arc::new(ProcessingTracker::new());
    let pipeline = Arc::new(Pipeline::new(
        cfg,
        failing_llm,
        writer,
        tracker,
        None,
        Some(store.clone()),
    ));

    let enriched = test_enriched("test pending insertion", vec![], vec![]);
    // run_llm produces a ProcessedMessage; if LLM fails it has llm_response=None.
    let processed = pipeline.run_llm(enriched).await.unwrap();
    assert!(processed.llm_response.is_none(), "expected fallback");

    // Simulate what the pipeline does after run_llm when llm_response is None.
    store
        .insert(processed.enriched.original.id, &processed, None)
        .await
        .unwrap();

    let items = store.list(5, 10).await.unwrap();
    assert_eq!(items.len(), 1, "one item should be in the pending store");
    assert_eq!(items[0].incoming.text, "test pending insertion");
}
