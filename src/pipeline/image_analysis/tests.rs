//! Task 5: image-analysis classification heuristic and the `analyze_images`
//! orchestration (vision-LLM transcription, size cap, skip rules).

use std::path::PathBuf;

use crate::config::{FallbackMode, ImageAnalysisConfig};
use crate::llm::LlmChain;
use crate::llm::mock::MockLlm;
use crate::message::{
    Attachment, ImageAnalysisKind, IncomingMessage, LlmResponse, MediaKind, MessageSource,
    SourceMetadata,
};

use super::analyze_images;
use super::classify::classify;

// ── classify heuristic ────────────────────────────────────────────────────────

#[test]
fn classify_empty_is_photo() {
    assert_eq!(classify("", 24), ImageAnalysisKind::Photo);
    assert_eq!(classify("   \n  ", 24), ImageAnalysisKind::Photo);
}

#[test]
fn classify_short_text_is_photo() {
    assert_eq!(classify("hi there", 24), ImageAnalysisKind::Photo);
}

#[test]
fn classify_long_text_is_interface() {
    assert_eq!(
        classify("Username: admin   Password: ******   Sign in", 24),
        ImageAnalysisKind::Interface
    );
}

#[test]
fn classify_multiline_is_interface() {
    assert_eq!(
        classify("a\nb\nc", 1000),
        ImageAnalysisKind::Interface,
        "three non-blank lines look like an interface even below the char threshold"
    );
}

// ── analyze_images ────────────────────────────────────────────────────────────

fn vision_chain(summary: &str) -> LlmChain {
    let resp = LlmResponse {
        title: String::new(),
        tags: vec![],
        summary: summary.into(),
        excerpt: None,
        produced_by: "mock-vision".into(),
    };
    LlmChain::new(
        vec![Box::new(MockLlm::new(resp).with_vision())],
        FallbackMode::Raw,
        5,
        None,
        1,
        0,
        0,
    )
}

fn text_only_chain() -> LlmChain {
    let resp = LlmResponse {
        title: String::new(),
        tags: vec![],
        summary: "ignored".into(),
        excerpt: None,
        produced_by: "mock".into(),
    };
    LlmChain::new(
        vec![Box::new(MockLlm::new(resp))], // non-vision
        FallbackMode::Raw,
        5,
        None,
        1,
        0,
        0,
    )
}

/// A vision backend that panics if the chain ever calls it — used to prove the
/// size cap is a pre-call guard (the image is never sent to the LLM).
struct PanicLlm;

#[async_trait::async_trait]
impl crate::llm::LlmClient for PanicLlm {
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
        panic!("vision backend must not be called for an oversized image");
    }
}

fn panic_chain() -> LlmChain {
    LlmChain::new(
        vec![Box::new(PanicLlm)],
        FallbackMode::Raw,
        5,
        None,
        1,
        0,
        0,
    )
}

struct TempImage(PathBuf);
impl Drop for TempImage {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn msg_with_image(bytes: &[u8]) -> (IncomingMessage, TempImage) {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let path =
        std::env::temp_dir().join(format!("inbox-imgtest-{}-{nanos}.jpg", std::process::id()));
    std::fs::write(&path, bytes).expect("write temp image");
    let mut msg = IncomingMessage::new(
        MessageSource::Telegram,
        String::new(),
        SourceMetadata::Telegram {
            chat_id: 1,
            message_id: 1,
            username: None,
            forwarded_from: Some("@Evgeniya_Koroleva".into()),
        },
    );
    msg.attachments.push(Attachment {
        original_name: "photo.jpg".into(),
        saved_path: path.clone(),
        mime_type: Some("image/jpeg".into()),
        media_kind: MediaKind::Image,
    });
    (msg, TempImage(path))
}

#[tokio::test]
async fn analyze_transcribes_and_classifies() {
    let chain = vision_chain("Login failed\nRetry now\nContact support");
    let (msg, _tmp) = msg_with_image(b"fake-jpeg-bytes");
    let cfg = ImageAnalysisConfig::default();

    let results = analyze_images(&chain, &cfg, 5 * 1024 * 1024, &msg).await;
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].attachment_name, "photo.jpg");
    assert_eq!(results[0].kind, ImageAnalysisKind::Interface);
    assert!(results[0].recognized_text.contains("Login failed"));
    assert_eq!(results[0].produced_by, "mock-vision");
}

#[tokio::test]
async fn analyze_disabled_returns_empty() {
    let chain = vision_chain("text");
    let (msg, _tmp) = msg_with_image(b"bytes");
    let cfg = ImageAnalysisConfig {
        enabled: false,
        ..ImageAnalysisConfig::default()
    };
    assert!(
        analyze_images(&chain, &cfg, 5 * 1024 * 1024, &msg)
            .await
            .is_empty()
    );
}

#[tokio::test]
async fn analyze_skips_oversize_image_before_calling_llm() {
    // PanicLlm fails the test if reached: the size cap must skip the image by
    // stat, before any read/encode/LLM call.
    let chain = panic_chain();
    let (msg, _tmp) = msg_with_image(b"this-is-too-big");
    let cfg = ImageAnalysisConfig::default();
    // 4-byte cap < payload ⇒ skipped pre-call.
    assert!(analyze_images(&chain, &cfg, 4, &msg).await.is_empty());
}

#[tokio::test]
async fn analyze_without_vision_backend_returns_empty() {
    let chain = text_only_chain();
    let (msg, _tmp) = msg_with_image(b"bytes");
    let cfg = ImageAnalysisConfig::default();
    assert!(
        analyze_images(&chain, &cfg, 5 * 1024 * 1024, &msg)
            .await
            .is_empty()
    );
}

#[tokio::test]
async fn analyze_skips_unreadable_image() {
    // An image attachment whose file does not exist ⇒ stat fails ⇒ skipped,
    // non-fatally, with no result.
    let chain = panic_chain();
    let mut msg = IncomingMessage::new(
        MessageSource::Telegram,
        String::new(),
        SourceMetadata::Telegram {
            chat_id: 1,
            message_id: 1,
            username: None,
            forwarded_from: None,
        },
    );
    msg.attachments.push(Attachment {
        original_name: "missing.jpg".into(),
        saved_path: PathBuf::from("/nonexistent/inbox-missing-image.jpg"),
        mime_type: Some("image/jpeg".into()),
        media_kind: MediaKind::Image,
    });
    let cfg = ImageAnalysisConfig::default();
    assert!(
        analyze_images(&chain, &cfg, 5 * 1024 * 1024, &msg)
            .await
            .is_empty()
    );
}

#[tokio::test]
async fn analyze_skips_when_read_fails_after_stat() {
    // A directory passes the size stat but fails to read as a file ⇒ the read
    // branch returns None non-fatally (the LLM is never reached).
    let dir = tempfile::tempdir().unwrap();
    let chain = panic_chain();
    let mut msg = IncomingMessage::new(
        MessageSource::Telegram,
        String::new(),
        SourceMetadata::Telegram {
            chat_id: 1,
            message_id: 1,
            username: None,
            forwarded_from: None,
        },
    );
    msg.attachments.push(Attachment {
        original_name: "adir.jpg".into(),
        saved_path: dir.path().to_path_buf(),
        mime_type: Some("image/jpeg".into()),
        media_kind: MediaKind::Image,
    });
    let cfg = ImageAnalysisConfig::default();
    assert!(
        analyze_images(&chain, &cfg, 5 * 1024 * 1024, &msg)
            .await
            .is_empty()
    );
}

#[tokio::test]
async fn analyze_ignores_non_image_attachments() {
    let chain = vision_chain("text");
    let mut msg = IncomingMessage::new(
        MessageSource::Http,
        String::new(),
        SourceMetadata::Http {
            remote_addr: None,
            user_agent: None,
        },
    );
    msg.attachments.push(Attachment {
        original_name: "doc.pdf".into(),
        saved_path: PathBuf::from("/tmp/none.pdf"),
        mime_type: Some("application/pdf".into()),
        media_kind: MediaKind::Document,
    });
    let cfg = ImageAnalysisConfig::default();
    assert!(
        analyze_images(&chain, &cfg, 5 * 1024 * 1024, &msg)
            .await
            .is_empty()
    );
}
