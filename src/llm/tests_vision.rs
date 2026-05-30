//! Task 1 vision-capability surface: trait default, per-backend config flag,
//! and `LlmRequest` vision derivation.

use crate::config::LlmBackendConfig;
use crate::message::LlmResponse;

use super::mock::MockLlm;
use super::ollama::OllamaClient;
use super::openrouter::OpenRouterClient;
use super::{LlmClient, LlmRequest};

fn backend_config(toml: &str) -> LlmBackendConfig {
    toml::from_str(toml).expect("valid backend config")
}

#[test]
fn trait_default_vision_unsupported() {
    let client = MockLlm::new(LlmResponse {
        title: "t".into(),
        tags: vec![],
        summary: "s".into(),
        excerpt: None,
        produced_by: "mock".into(),
    });
    assert!(!client.vision_supported());
}

#[test]
fn openrouter_reflects_vision_flag() {
    let enabled = backend_config(
        r#"type = "openrouter"
model = "x/y"
vision_supported = true
"#,
    );
    let disabled = backend_config(
        r#"type = "openrouter"
model = "x/y"
"#,
    );
    assert!(
        OpenRouterClient::from_config(&enabled)
            .expect("client")
            .vision_supported()
    );
    assert!(
        !OpenRouterClient::from_config(&disabled)
            .expect("client")
            .vision_supported()
    );
}

#[test]
fn ollama_reflects_vision_flag() {
    let enabled = backend_config(
        r#"type = "ollama"
model = "llava"
base_url = "http://localhost:11434"
vision_supported = true
"#,
    );
    let disabled = backend_config(
        r#"type = "ollama"
model = "llama3"
base_url = "http://localhost:11434"
"#,
    );
    assert!(
        OllamaClient::from_config(&enabled)
            .expect("client")
            .vision_supported()
    );
    assert!(
        !OllamaClient::from_config(&disabled)
            .expect("client")
            .vision_supported()
    );
}

#[test]
fn request_needs_vision_tracks_images() {
    let mut req = LlmRequest::simple("system", "user");
    assert!(!req.needs_vision());
    assert!(!req.has_image_text);

    req.images.push(("image/png".into(), "Zm9v".into()));
    assert!(req.needs_vision());

    // Stripping images flips the derived flag back — no desync possible.
    req.images.clear();
    assert!(!req.needs_vision());
}

// ── Task 7: from_enriched folds recognized image text into the request ─────────

use crate::message::{
    Attachment, EnrichedMessage, ImageAnalysisKind, ImageAnalysisResult, IncomingMessage,
    MediaKind, MessageSource, SourceMetadata,
};

fn enriched_with(
    forwarded: Option<&str>,
    attachments: Vec<Attachment>,
    analyses: Vec<ImageAnalysisResult>,
) -> EnrichedMessage {
    let mut original = IncomingMessage::new(
        MessageSource::Telegram,
        String::new(),
        SourceMetadata::Telegram {
            chat_id: 1,
            message_id: 1,
            username: None,
            forwarded_from: forwarded.map(str::to_owned),
        },
    );
    original.attachments = attachments;
    original.image_analyses = analyses;
    EnrichedMessage {
        original,
        urls: vec![],
        url_contents: vec![],
    }
}

fn image_attachment(path: &str) -> Attachment {
    Attachment {
        original_name: "photo.jpg".into(),
        saved_path: std::path::PathBuf::from(path),
        mime_type: Some("image/jpeg".into()),
        media_kind: MediaKind::Image,
    }
}

fn analysis(text: &str) -> ImageAnalysisResult {
    ImageAnalysisResult {
        attachment_name: "photo.jpg".into(),
        kind: ImageAnalysisKind::Interface,
        recognized_text: text.into(),
        produced_by: "mock-vision".into(),
    }
}

#[test]
fn from_enriched_appends_image_text_and_omits_images() {
    // Recognized text present ⇒ folded into user_content, has_image_text set, and
    // images omitted (no need to re-send the raw image to enrichment).
    let enriched = enriched_with(
        Some("@Evgeniya_Koroleva"),
        vec![image_attachment("/tmp/does-not-exist.jpg")],
        vec![analysis("Login failed")],
    );
    let cfg = crate::test_helpers::no_llm_config();
    let req = LlmRequest::from_enriched(&enriched, &cfg, std::path::Path::new("/tmp"), "", false);

    assert!(req.has_image_text);
    assert!(req.user_content.contains("Image text"));
    assert!(req.user_content.contains("Login failed"));
    assert!(
        req.user_content
            .contains("Forwarded from @Evgeniya_Koroleva")
    );
    assert!(
        req.images.is_empty(),
        "images omitted once text is extracted"
    );
}

#[test]
fn from_enriched_no_image_text_when_no_analysis() {
    // No attachments, no analysis ⇒ has_image_text false, no images.
    let enriched = enriched_with(None, vec![], vec![]);
    let cfg = crate::test_helpers::no_llm_config();
    let req = LlmRequest::from_enriched(&enriched, &cfg, std::path::Path::new("/tmp"), "", false);
    assert!(!req.has_image_text);
    assert!(req.images.is_empty());
}

#[test]
fn from_enriched_keeps_images_when_no_recognized_text() {
    // Image present but analysis produced no text ⇒ images are still encoded so
    // a vision backend in enrichment can try.
    let dir = std::env::temp_dir();
    let path = dir.join(format!("inbox-fe-{}.jpg", std::process::id()));
    std::fs::write(&path, b"jpeg-bytes").expect("write temp image");

    let enriched = enriched_with(
        None,
        vec![image_attachment(path.to_str().unwrap())],
        vec![], // no analysis text
    );
    let cfg = crate::test_helpers::no_llm_config();
    let req = LlmRequest::from_enriched(&enriched, &cfg, std::path::Path::new("/tmp"), "", false);

    assert!(!req.has_image_text);
    assert_eq!(req.images.len(), 1, "image encoded when no extracted text");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn from_enriched_mixed_images_keeps_only_untranscribed() {
    // Two images: one transcribed (omitted), one with no text (kept). Per-
    // attachment omission must not drop the no-text image.
    let dir = std::env::temp_dir();
    let photo = dir.join(format!("inbox-fe-mix-{}.jpg", std::process::id()));
    std::fs::write(&photo, b"plain-photo-bytes").expect("write temp image");

    let screenshot = Attachment {
        original_name: "screen.jpg".into(),
        saved_path: std::path::PathBuf::from("/tmp/screen-not-read.jpg"),
        mime_type: Some("image/jpeg".into()),
        media_kind: MediaKind::Image,
    };
    let photo_att = Attachment {
        original_name: "plain.jpg".into(),
        saved_path: photo.clone(),
        mime_type: Some("image/jpeg".into()),
        media_kind: MediaKind::Image,
    };
    let screen_analysis = ImageAnalysisResult {
        attachment_name: "screen.jpg".into(),
        kind: ImageAnalysisKind::Interface,
        recognized_text: "Settings menu".into(),
        produced_by: "mock-vision".into(),
    };

    let enriched = enriched_with(None, vec![screenshot, photo_att], vec![screen_analysis]);
    let cfg = crate::test_helpers::no_llm_config();
    let req = LlmRequest::from_enriched(&enriched, &cfg, std::path::Path::new("/tmp"), "", false);

    assert!(req.has_image_text);
    assert!(req.user_content.contains("Settings menu"));
    // Only the un-transcribed photo is encoded; the screenshot (already OCR'd)
    // is omitted — so exactly one image and its saved_path was read.
    assert_eq!(req.images.len(), 1, "only the no-text image is kept");
    let _ = std::fs::remove_file(&photo);
}
