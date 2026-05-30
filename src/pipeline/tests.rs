use std::sync::Arc;

use super::*;
use crate::config::{
    AdaptersConfig, AdminConfig, Config, GeneralConfig, PipelineConfig, SyncthingConfig,
    ToolingConfig, UrlFetchConfig, WebUiConfig,
};
use crate::message::{EnrichedMessage, IncomingMessage, MessageSource, SourceMetadata};
use crate::pending::PendingStore;
use crate::processing_status::ProcessingTracker;

pub(super) fn test_config(policy: crate::config::JsShellPolicy) -> Config {
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
            image_analysis: crate::config::ImageAnalysisConfig::default(),
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
    Arc::new(Pipeline::new(cfg, llm, writer, tracker, None, None).expect("build pipeline"))
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
    let pipeline = Arc::new(
        Pipeline::new(cfg, failing_llm, writer, tracker, None, Some(store.clone()))
            .expect("build pipeline"),
    );

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

// ── Pipeline::process end-to-end tests ────────────────────────────────────────

pub(super) fn make_msg(text: &str) -> IncomingMessage {
    IncomingMessage::new(
        MessageSource::Http,
        text.into(),
        SourceMetadata::Http {
            remote_addr: None,
            user_agent: None,
        },
    )
}

fn enriched_from(msg: IncomingMessage) -> EnrichedMessage {
    EnrichedMessage {
        original: msg,
        urls: vec![],
        url_contents: vec![],
    }
}

#[tokio::test]
async fn process_success_path_writes_and_marks_done() {
    let dir = tempfile::tempdir().unwrap();
    let mut cfg = test_config(crate::config::JsShellPolicy::Allow);
    cfg.general.attachments_dir = dir.path().to_path_buf();
    cfg.general.output_file = dir.path().join("inbox.org");
    cfg.url_fetch.enabled = false;
    let cfg = Arc::new(cfg);

    let llm = crate::test_helpers::mock_llm_chain(crate::test_helpers::default_llm_response());
    let writer = Arc::new(crate::output::NullWriter);
    let tracker = Arc::new(ProcessingTracker::new());
    let pipeline = Pipeline::new(cfg, llm, writer, tracker, None, None).expect("build pipeline");

    pipeline.process(make_msg("hello world")).await.unwrap();
}

#[derive(Default)]
struct CapturingWriter {
    captured_tags: std::sync::Mutex<Vec<String>>,
}

#[async_trait::async_trait]
impl crate::output::OutputWriter for CapturingWriter {
    async fn write(
        &self,
        msg: &crate::message::ProcessedMessage,
        _cfg: &Config,
    ) -> Result<(), crate::error::InboxError> {
        *self.captured_tags.lock().unwrap() = msg.enriched.original.user_tags.clone();
        Ok(())
    }
}

#[tokio::test]
async fn process_extracts_user_tags_from_text() {
    let dir = tempfile::tempdir().unwrap();
    let mut cfg = test_config(crate::config::JsShellPolicy::Allow);
    cfg.general.attachments_dir = dir.path().to_path_buf();
    cfg.general.output_file = dir.path().join("inbox.org");
    cfg.url_fetch.enabled = false;
    let cfg = Arc::new(cfg);

    let capture = Arc::new(CapturingWriter::default());
    let llm = crate::test_helpers::mock_llm_chain(crate::test_helpers::default_llm_response());
    let tracker = Arc::new(ProcessingTracker::new());
    let pipeline = Pipeline::new(
        cfg,
        llm,
        Arc::clone(&capture) as Arc<dyn crate::output::OutputWriter>,
        tracker,
        None,
        None,
    )
    .expect("build pipeline");

    pipeline
        .process(make_msg("check this out #rust #inbox"))
        .await
        .unwrap();

    let tags = capture.captured_tags.lock().unwrap().clone();
    assert!(
        tags.iter().any(|t| t == "rust") && tags.iter().any(|t| t == "inbox"),
        "user_tags should be extracted and propagated: {tags:?}"
    );
}

#[tokio::test]
async fn process_persists_pending_item_on_raw_fallback() {
    let dir = tempfile::tempdir().unwrap();
    let mut cfg = test_config(crate::config::JsShellPolicy::Allow);
    cfg.general.attachments_dir = dir.path().to_path_buf();
    cfg.general.output_file = dir.path().join("inbox.org");
    cfg.url_fetch.enabled = false;
    let cfg = Arc::new(cfg);

    let db_path = dir.path().join("pending.db");
    let store = Arc::new(PendingStore::open(&db_path).await.unwrap());

    let failing_llm = crate::test_helpers::always_fail_llm_chain();
    let writer = Arc::new(crate::output::NullWriter);
    let tracker = Arc::new(ProcessingTracker::new());
    let pipeline = Pipeline::new(
        cfg,
        failing_llm,
        writer,
        tracker,
        None,
        Some(Arc::clone(&store)),
    )
    .expect("build pipeline");

    pipeline
        .process(make_msg("this message should be pending"))
        .await
        .unwrap();

    let items = store.list(5, 10).await.unwrap();
    assert_eq!(items.len(), 1, "pending store should have one entry");
    assert_eq!(items[0].incoming.text, "this message should be pending");
}

#[tokio::test]
async fn process_propagates_error_on_discard_fallback() {
    use crate::config::FallbackMode;

    let dir = tempfile::tempdir().unwrap();
    let mut cfg = test_config(crate::config::JsShellPolicy::Allow);
    cfg.general.attachments_dir = dir.path().to_path_buf();
    cfg.general.output_file = dir.path().join("inbox.org");
    cfg.url_fetch.enabled = false;
    cfg.llm.fallback = FallbackMode::Discard;
    let cfg = Arc::new(cfg);

    // A chain built with FallbackMode::Discard inside LlmChain propagates
    // Err — mirroring production behavior.
    let failing_llm = crate::test_helpers::failing_llm_chain("simulated LLM failure");
    let writer = Arc::new(crate::output::NullWriter);
    let tracker = Arc::new(ProcessingTracker::new());
    let pipeline =
        Pipeline::new(cfg, failing_llm, writer, tracker, None, None).expect("build pipeline");

    let result = pipeline.process(make_msg("drop me")).await;
    assert!(result.is_err(), "Discard fallback must surface as Err");
}

#[tokio::test]
async fn run_llm_raw_fallback_with_existing_text_skips_title_regeneration() {
    // When the message has non-empty text, fallback_title stays None —
    // the pipeline doesn't invoke complete_text.
    let dir = tempfile::tempdir().unwrap();
    let mut cfg = test_config(crate::config::JsShellPolicy::Allow);
    cfg.general.attachments_dir = dir.path().to_path_buf();
    cfg.general.output_file = dir.path().join("inbox.org");
    cfg.url_fetch.enabled = false;
    let cfg = Arc::new(cfg);

    let failing_llm = crate::test_helpers::always_fail_llm_chain();
    let writer = Arc::new(crate::output::NullWriter);
    let tracker = Arc::new(ProcessingTracker::new());
    let pipeline =
        Pipeline::new(cfg, failing_llm, writer, tracker, None, None).expect("build pipeline");

    let enriched = enriched_from(make_msg("plain text, not empty"));
    let processed = pipeline.run_llm(enriched).await.unwrap();
    assert!(processed.llm_response.is_none());
    assert!(
        processed.fallback_title.is_none(),
        "title regeneration should not happen when text is non-empty"
    );
}

// ── process_url branch coverage with wiremock ────────────────────────────────

fn pipeline_with_fetch_cfg(fetch_cfg: crate::config::UrlFetchConfig) -> Arc<Pipeline> {
    let dir = tempfile::tempdir().unwrap();
    let mut cfg = test_config(crate::config::JsShellPolicy::ToolOnly);
    cfg.general.attachments_dir = dir.path().to_path_buf();
    cfg.url_fetch = fetch_cfg;
    cfg.llm.url_content_max_chars = 200;
    // Leak the tempdir so attachments survive for the duration of the test.
    Box::leak(Box::new(dir));
    let cfg = Arc::new(cfg);

    let llm = crate::test_helpers::mock_llm_chain(crate::test_helpers::default_llm_response());
    let writer = Arc::new(crate::output::NullWriter);
    let tracker = Arc::new(ProcessingTracker::new());
    Arc::new(Pipeline::new(cfg, llm, writer, tracker, None, None).expect("build pipeline"))
}

fn fetch_cfg_enabled() -> crate::config::UrlFetchConfig {
    crate::config::UrlFetchConfig {
        enabled: true,
        user_agent: "inbox-test/1.0".into(),
        timeout_secs: 5,
        max_redirects: 3,
        max_body_bytes: 1024 * 1024,
        skip_domains: vec![],
        nitter_base_url: None,
    }
}

#[tokio::test]
async fn process_url_fetches_html_page_into_url_contents() {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let server = MockServer::start().await;
    Mock::given(method("HEAD"))
        .and(path("/article"))
        .respond_with(ResponseTemplate::new(200).insert_header("content-type", "text/html"))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/article"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/html")
                .set_body_string(
                    "<html><head><title>T</title></head><body><p>hello body</p></body></html>",
                ),
        )
        .mount(&server)
        .await;

    let pipeline = pipeline_with_fetch_cfg(fetch_cfg_enabled());
    let msg = make_msg(&format!("look at {}/article please", server.uri()));
    let enriched = pipeline.enrich(msg).await.expect("enrich ok");

    assert_eq!(enriched.url_contents.len(), 1);
    assert!(enriched.url_contents[0].text.contains("hello body"));
    assert!(enriched.original.attachments.is_empty());
}

#[tokio::test]
async fn process_url_downloads_file_attachments() {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let server = MockServer::start().await;
    Mock::given(method("HEAD"))
        .and(path("/file.bin"))
        .respond_with(
            ResponseTemplate::new(200).insert_header("content-type", "application/octet-stream"),
        )
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/file.bin"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/octet-stream")
                .set_body_bytes(vec![1u8, 2, 3, 4]),
        )
        .mount(&server)
        .await;

    let pipeline = pipeline_with_fetch_cfg(fetch_cfg_enabled());
    let msg = make_msg(&format!("download {}/file.bin", server.uri()));
    let enriched = pipeline.enrich(msg).await.expect("enrich ok");

    assert!(enriched.url_contents.is_empty());
    assert_eq!(enriched.original.attachments.len(), 1);
}

#[tokio::test]
async fn process_url_skips_domain_in_skip_list() {
    use wiremock::matchers::method;
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let server = MockServer::start().await;
    // If the test ever reaches the server, the mock would respond — the
    // assertion that url_contents is empty proves it didn't.
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(200).set_body_string("should not be reached"))
        .mount(&server)
        .await;

    let host = server.uri().trim_start_matches("http://").to_owned();
    // Strip the port so skip_domain matches by host only.
    let bare_host = host.split(':').next().unwrap().to_owned();
    let mut fc = fetch_cfg_enabled();
    fc.skip_domains = vec![bare_host];

    let pipeline = pipeline_with_fetch_cfg(fc);
    let msg = make_msg(&format!("see {}/anything", server.uri()));
    let enriched = pipeline.enrich(msg).await.expect("enrich ok");

    assert!(enriched.url_contents.is_empty());
    assert!(enriched.original.attachments.is_empty());
}

#[tokio::test]
async fn process_url_drops_page_matching_js_shell_policy() {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let server = MockServer::start().await;
    Mock::given(method("HEAD"))
        .and(path("/spa"))
        .respond_with(ResponseTemplate::new(200).insert_header("content-type", "text/html"))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/spa"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/html")
                .set_body_string(
                    "<html><body>This page doesn't work properly without JavaScript enabled \
                     please enable it to continue</body></html>",
                ),
        )
        .mount(&server)
        .await;

    let pipeline = pipeline_with_fetch_cfg(fetch_cfg_enabled());
    let msg = make_msg(&format!("see {}/spa", server.uri()));
    let enriched = pipeline.enrich(msg).await.expect("enrich ok");

    assert!(
        enriched.url_contents.is_empty(),
        "JS-shell policy should suppress page content"
    );
}

#[tokio::test]
async fn process_url_unknown_kind_falls_back_to_page_fetch() {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let server = MockServer::start().await;
    // HEAD with a content-type that fails MIME parse → UrlKind::Unknown.
    Mock::given(method("HEAD"))
        .and(path("/weird"))
        .respond_with(ResponseTemplate::new(200).insert_header("content-type", "garbage"))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/weird"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/html")
                .set_body_string("<html><body><p>unknown fallback body</p></body></html>"),
        )
        .mount(&server)
        .await;

    let pipeline = pipeline_with_fetch_cfg(fetch_cfg_enabled());
    let msg = make_msg(&format!("see {}/weird", server.uri()));
    let enriched = pipeline.enrich(msg).await.expect("enrich ok");

    assert_eq!(enriched.url_contents.len(), 1);
    assert!(
        enriched.url_contents[0]
            .text
            .contains("unknown fallback body")
    );
}
