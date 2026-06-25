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
            memo_tags: vec!["memo".into()],
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
    assert!(guidance.contains("web search tool"));
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
    let processed = pipeline.run_llm(enriched, true).await.unwrap();
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

#[tokio::test]
async fn image_only_vision_unavailable_yields_incomplete_node() {
    // An image-only message whose vision backends were all unavailable, with no
    // recognized image text, must be marked incomplete (held pending) and never
    // carry an LLM response — i.e. never reported as successfully processed.
    let cfg = Arc::new(test_config(crate::config::JsShellPolicy::Allow));
    let llm = crate::test_helpers::always_fail_llm_chain();
    let writer = Arc::new(crate::output::NullWriter);
    let tracker = Arc::new(ProcessingTracker::new());
    let pipeline =
        Arc::new(Pipeline::new(cfg, llm, writer, tracker, None, None).expect("build pipeline"));

    let mut msg = make_msg(""); // image-only, no caption
    msg.attachments.push(crate::message::Attachment {
        original_name: "shot.jpg".into(),
        saved_path: std::path::PathBuf::from("/tmp/inbox-test-none.jpg"),
        mime_type: Some("image/jpeg".into()),
        media_kind: crate::message::MediaKind::Image,
    });
    let enriched = enriched_from(msg);

    let processed = pipeline.run_llm(enriched, false).await.unwrap();
    assert!(
        processed.is_incomplete(),
        "image-only + vision-down must be incomplete"
    );
    assert!(
        processed.llm_response.is_none(),
        "incomplete node must not be reported as a successful LLM result"
    );
}

struct PanicOnComplete;
#[async_trait::async_trait]
impl crate::llm::LlmClient for PanicOnComplete {
    fn name(&self) -> &'static str {
        "panic"
    }
    fn model(&self) -> &'static str {
        "panic"
    }
    fn retries(&self) -> u32 {
        1
    }
    fn vision_supported(&self) -> bool {
        true
    }
    async fn complete(
        &self,
        _req: crate::llm::LlmRequest,
    ) -> Result<crate::llm::LlmCompletion, crate::error::InboxError> {
        panic!("LLM must not be called for an image-only vision-unavailable message");
    }
}

#[tokio::test]
async fn image_only_vision_unavailable_short_circuits_without_calling_llm() {
    // A real on-disk image makes req.images non-empty, so the empty-input branch
    // in run_llm fires: the node is built incomplete WITHOUT an LLM call (the
    // panic chain would fail the test if the LLM were invoked). The tempdir guard
    // cleans up even if the test panics.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("shot.jpg");
    std::fs::write(&path, b"fake-jpeg-bytes").expect("write temp image");

    let cfg = Arc::new(test_config(crate::config::JsShellPolicy::Allow));
    let llm = Arc::new(crate::llm::LlmChain::new(
        vec![Box::new(PanicOnComplete)],
        crate::config::FallbackMode::Raw,
        5,
        None,
        1,
        0,
        0,
    ));
    let writer = Arc::new(crate::output::NullWriter);
    let tracker = Arc::new(ProcessingTracker::new());
    let pipeline =
        Arc::new(Pipeline::new(cfg, llm, writer, tracker, None, None).expect("build pipeline"));

    let mut msg = make_msg(""); // image-only, no caption
    msg.attachments.push(crate::message::Attachment {
        original_name: "shot.jpg".into(),
        saved_path: path.clone(),
        mime_type: Some("image/jpeg".into()),
        media_kind: crate::message::MediaKind::Image,
    });
    let enriched = enriched_from(msg);

    let processed = pipeline.run_llm(enriched, false).await.unwrap();
    assert!(processed.is_incomplete());
    assert!(processed.llm_response.is_none());
}

#[tokio::test]
async fn resume_image_analysis_preserves_stored_ocr_when_reanalysis_yields_nothing() {
    // Re-OCR with no vision backend (and a missing file) produces no results;
    // the previously recognized text must be kept, not blanked.
    let pipeline = make_test_pipeline(test_config(crate::config::JsShellPolicy::Allow));
    let mut msg = make_msg("");
    msg.attachments.push(crate::message::Attachment {
        original_name: "x.jpg".into(),
        saved_path: std::path::PathBuf::from("/tmp/inbox-resume-missing.jpg"),
        mime_type: Some("image/jpeg".into()),
        media_kind: crate::message::MediaKind::Image,
    });
    msg.image_analyses
        .push(crate::message::ImageAnalysisResult {
            attachment_name: "x.jpg".into(),
            kind: crate::message::ImageAnalysisKind::Photo,
            recognized_text: "stored text".into(),
            produced_by: "old".into(),
        });

    let resolved = pipeline.resume_image_analysis(&mut msg).await;

    assert_eq!(msg.image_analyses.len(), 1);
    assert_eq!(msg.image_analyses[0].recognized_text, "stored text");
    assert_eq!(
        resolved,
        Some(false),
        "a skipped (unreadable) image is not resolved → stays pending"
    );
}

#[tokio::test]
async fn resume_image_analysis_none_when_no_image() {
    let pipeline = make_test_pipeline(test_config(crate::config::JsShellPolicy::Allow));
    let mut msg = make_msg("just text");
    assert_eq!(pipeline.resume_image_analysis(&mut msg).await, None);
}

#[tokio::test]
async fn resume_image_analysis_resolved_when_backend_runs_with_no_text() {
    // A recovered vision backend that runs and finds no readable text resolves
    // the image (Some(true)) so the node finalizes instead of looping pending.
    let resp = crate::message::LlmResponse {
        title: String::new(),
        tags: vec![],
        summary: String::new(), // empty OCR
        excerpt: None,
        produced_by: "mock-vision".into(),
    };
    let llm = Arc::new(crate::llm::LlmChain::new(
        vec![Box::new(crate::llm::mock::MockLlm::new(resp).with_vision())],
        crate::config::FallbackMode::Raw,
        5,
        None,
        1,
        0,
        0,
    ));
    let cfg = Arc::new(test_config(crate::config::JsShellPolicy::Allow));
    let writer = Arc::new(crate::output::NullWriter);
    let tracker = Arc::new(ProcessingTracker::new());
    let pipeline = Pipeline::new(cfg, llm, writer, tracker, None, None).expect("build pipeline");

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("shot.jpg");
    std::fs::write(&path, b"fake-jpeg-bytes").unwrap();
    let mut msg = make_msg("");
    msg.attachments.push(crate::message::Attachment {
        original_name: "shot.jpg".into(),
        saved_path: path,
        mime_type: Some("image/jpeg".into()),
        media_kind: crate::message::MediaKind::Image,
    });

    assert_eq!(
        pipeline.resume_image_analysis(&mut msg).await,
        Some(true),
        "backend analyzed the image (empty text still counts) → resolved"
    );
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

#[derive(Default)]
struct NodeCapturingWriter {
    node: std::sync::Mutex<String>,
    backend: std::sync::Mutex<String>,
}

#[async_trait::async_trait]
impl crate::output::OutputWriter for NodeCapturingWriter {
    async fn write(
        &self,
        msg: &crate::message::ProcessedMessage,
        cfg: &Config,
    ) -> Result<(), crate::error::InboxError> {
        let node = crate::render::render_org_node(msg, &cfg.general.attachments_dir)?;
        *self.node.lock().unwrap() = node;
        *self.backend.lock().unwrap() = msg
            .llm_response
            .as_ref()
            .map(|r| r.produced_by.clone())
            .unwrap_or_default();
        Ok(())
    }
}

#[tokio::test]
async fn process_memo_skips_llm_and_writes_final_node() {
    // A #memo message must skip the LLM entirely (PanicOnComplete fails the test
    // if invoked), produce a final node tagged :memo: but never :inbox_pending:,
    // and never land in the pending store.
    let dir = tempfile::tempdir().unwrap();
    let mut cfg = test_config(crate::config::JsShellPolicy::Allow);
    cfg.general.attachments_dir = dir.path().to_path_buf();
    cfg.general.output_file = dir.path().join("inbox.org");
    cfg.url_fetch.enabled = false;
    let cfg = Arc::new(cfg);

    let store = Arc::new(
        PendingStore::open(&dir.path().join("pending.db"))
            .await
            .unwrap(),
    );
    let llm = Arc::new(crate::llm::LlmChain::new(
        vec![Box::new(PanicOnComplete)],
        crate::config::FallbackMode::Raw,
        5,
        None,
        1,
        0,
        0,
    ));
    let capture = Arc::new(NodeCapturingWriter::default());
    let tracker = Arc::new(ProcessingTracker::new());
    let pipeline = Pipeline::new(
        cfg,
        llm,
        Arc::clone(&capture) as Arc<dyn crate::output::OutputWriter>,
        tracker,
        None,
        Some(store.clone()),
    )
    .expect("build pipeline");

    // A user-typed reserved tag must not forge a pending headline.
    pipeline
        .process(make_msg("Oil change 4851 km #memo #inbox_pending"))
        .await
        .unwrap();

    let node = capture.node.lock().unwrap().clone();
    assert!(node.contains(":memo:"), "memo tag must render: {node}");
    assert!(
        !node.contains(":inbox_pending:"),
        "memo node must not be pending even if user typed the tag: {node}"
    );
    assert!(
        node.contains("Oil change 4851 km"),
        "memo text preserved: {node}"
    );
    assert_eq!(*capture.backend.lock().unwrap(), "memo");

    let items = store.list(5, 10).await.unwrap();
    assert!(
        items.is_empty(),
        "memo must not be persisted, got {}",
        items.len()
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
    let processed = pipeline.run_llm(enriched, true).await.unwrap();
    assert!(processed.llm_response.is_none());
    assert!(
        processed.fallback_title.is_none(),
        "title regeneration should not happen when text is non-empty"
    );
}

// `process_url` branch coverage with wiremock lives in `tests_url.rs`.
