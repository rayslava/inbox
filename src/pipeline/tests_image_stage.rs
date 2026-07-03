//! Task 6: pipeline wiring of the image-analysis stage — analysis populates
//! results + interface tag, the stage is observable via the notifier, and the
//! stage is skipped when disabled.

use std::sync::{Arc, Mutex};

use super::Pipeline;
use super::tests::{make_msg, test_config};
use crate::config::{FallbackMode, JsShellPolicy};
use crate::llm::LlmChain;
use crate::llm::mock::MockLlm;
use crate::message::{
    Attachment, ImageAnalysisKind, ImageAnalysisResult, LlmResponse, MediaKind, ProcessedMessage,
};
use crate::output::OutputWriter;
use crate::processing_status::{ProcessingStage, ProcessingTracker, StatusNotifier};

#[derive(Default)]
struct AnalysisCapturingWriter {
    analyses: Mutex<Vec<ImageAnalysisResult>>,
    suggested_tags: Mutex<Vec<String>>,
}

#[async_trait::async_trait]
impl OutputWriter for AnalysisCapturingWriter {
    async fn write(
        &self,
        msg: &ProcessedMessage,
        _target: &inbox_core::OutputTarget<'_>,
    ) -> Result<(), inbox_core::CoreError> {
        *self.analyses.lock().unwrap() = msg.enriched.original.image_analyses.clone();
        *self.suggested_tags.lock().unwrap() = msg
            .enriched
            .original
            .preprocessing_hints
            .suggested_tags
            .clone();
        Ok(())
    }
}

/// A notifier that records every stage it is advanced through (by serde tag).
struct RecordingNotifier {
    stages: Arc<Mutex<Vec<String>>>,
}

#[async_trait::async_trait]
impl StatusNotifier for RecordingNotifier {
    async fn advance(&mut self, stage: ProcessingStage) {
        if let Ok(v) = serde_json::to_value(&stage)
            && let Some(s) = v.get("stage").and_then(serde_json::Value::as_str)
        {
            self.stages.lock().unwrap().push(s.to_owned());
        }
    }
}

fn vision_chain(summary: &str) -> Arc<LlmChain> {
    let resp = LlmResponse {
        title: String::new(),
        tags: vec![],
        summary: summary.into(),
        excerpt: None,
        produced_by: "mock-vision".into(),
    };
    Arc::new(LlmChain::new(
        vec![Box::new(MockLlm::new(resp).with_vision())],
        FallbackMode::Raw,
        5,
        None,
        1,
        0,
        0,
    ))
}

fn image_msg(dir: &std::path::Path) -> crate::message::IncomingMessage {
    let img = dir.join("photo.jpg");
    std::fs::write(&img, b"fake-bytes").unwrap();
    let mut msg = make_msg("");
    msg.attachments.push(Attachment {
        original_name: "photo.jpg".into(),
        saved_path: img,
        mime_type: Some("image/jpeg".into()),
        media_kind: MediaKind::Image,
    });
    msg
}

#[tokio::test]
async fn process_runs_image_analysis_and_tags_interface() {
    let dir = tempfile::tempdir().unwrap();
    let mut cfg = test_config(JsShellPolicy::Allow);
    cfg.general.attachments_dir = dir.path().to_path_buf();
    cfg.general.output_file = dir.path().join("inbox.org");
    cfg.url_fetch.enabled = false;

    let capture = Arc::new(AnalysisCapturingWriter::default());
    let pipeline = Pipeline::new(
        Arc::new(cfg),
        vision_chain("Username field\nPassword field\nSign in button"),
        Arc::clone(&capture) as Arc<dyn OutputWriter>,
        Arc::new(ProcessingTracker::new()),
        None,
        None,
    )
    .expect("build pipeline");

    pipeline.process(image_msg(dir.path())).await.unwrap();

    let analyses = capture.analyses.lock().unwrap().clone();
    assert_eq!(analyses.len(), 1, "one analysis result");
    assert_eq!(analyses[0].kind, ImageAnalysisKind::Interface);
    assert!(analyses[0].recognized_text.contains("Username field"));
    assert!(
        capture
            .suggested_tags
            .lock()
            .unwrap()
            .iter()
            .any(|t| t == "interface"),
        "interface image should add the `interface` suggested tag"
    );
}

#[tokio::test]
async fn process_advances_notifier_through_analyzing_images() {
    let dir = tempfile::tempdir().unwrap();
    let mut cfg = test_config(JsShellPolicy::Allow);
    cfg.general.attachments_dir = dir.path().to_path_buf();
    cfg.general.output_file = dir.path().join("inbox.org");
    cfg.url_fetch.enabled = false;

    let pipeline = Pipeline::new(
        Arc::new(cfg),
        vision_chain("Some screen text here that is long enough"),
        Arc::new(crate::output::NullWriter),
        Arc::new(ProcessingTracker::new()),
        None,
        None,
    )
    .expect("build pipeline");

    let stages = Arc::new(Mutex::new(Vec::new()));
    let mut msg = image_msg(dir.path());
    msg.status_notifier = Some(Box::new(RecordingNotifier {
        stages: Arc::clone(&stages),
    }));

    pipeline.process(msg).await.unwrap();

    let seen = stages.lock().unwrap();
    assert!(
        seen.iter().any(|s| s == "analyzing_images"),
        "notifier should advance through analyzing_images: {seen:?}"
    );
}

#[tokio::test]
async fn process_skips_analysis_when_disabled() {
    let dir = tempfile::tempdir().unwrap();
    let mut cfg = test_config(JsShellPolicy::Allow);
    cfg.general.attachments_dir = dir.path().to_path_buf();
    cfg.general.output_file = dir.path().join("inbox.org");
    cfg.url_fetch.enabled = false;
    cfg.pipeline.image_analysis.enabled = false;

    let capture = Arc::new(AnalysisCapturingWriter::default());
    let pipeline = Pipeline::new(
        Arc::new(cfg),
        vision_chain("text"),
        Arc::clone(&capture) as Arc<dyn OutputWriter>,
        Arc::new(ProcessingTracker::new()),
        None,
        None,
    )
    .expect("build pipeline");

    pipeline.process(image_msg(dir.path())).await.unwrap();

    assert!(
        capture.analyses.lock().unwrap().is_empty(),
        "disabled image analysis must not populate results"
    );
}
