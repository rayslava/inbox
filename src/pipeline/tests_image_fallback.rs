//! Task 4: image-bearing messages always yield a non-empty raw fallback, and
//! `RetryableMessage` round-trips the analysis field (backward-compatible).

use crate::message::{
    Attachment, ImageAnalysisKind, ImageAnalysisResult, IncomingMessage, MediaKind, MessageSource,
    RetryableMessage, SourceMetadata,
};

use super::llm_stage::{
    FallbackPlan, ImageFallback, first_nonempty_line, image_fallback, plan_image_fallback,
};

fn title_of(fb: &ImageFallback) -> &str {
    match fb {
        ImageFallback::Ocr { title, .. } | ImageFallback::Metadata { title, .. } => title,
    }
}

fn telegram_image_msg(
    forwarded: Option<&str>,
    analyses: Vec<ImageAnalysisResult>,
) -> IncomingMessage {
    let mut msg = IncomingMessage::new(
        MessageSource::Telegram,
        String::new(),
        SourceMetadata::Telegram {
            chat_id: 1,
            message_id: 1,
            username: None,
            forwarded_from: forwarded.map(str::to_owned),
        },
    );
    msg.attachments.push(Attachment {
        original_name: "photo.jpg".into(),
        saved_path: std::path::PathBuf::from("/tmp/photo.jpg"),
        mime_type: Some("image/jpeg".into()),
        media_kind: MediaKind::Image,
    });
    msg.image_analyses = analyses;
    msg
}

fn analysis(kind: ImageAnalysisKind, text: &str) -> ImageAnalysisResult {
    ImageAnalysisResult {
        attachment_name: "photo.jpg".into(),
        kind,
        recognized_text: text.into(),
        produced_by: "free_router:model".into(),
    }
}

#[test]
fn recognized_text_drives_title_and_results() {
    let msg = telegram_image_msg(
        Some("@Evgeniya_Koroleva"),
        vec![analysis(
            ImageAnalysisKind::Interface,
            "Login failed\nRetry?",
        )],
    );
    match image_fallback(&msg).expect("image fallback") {
        ImageFallback::Ocr {
            extra_results,
            title,
        } => {
            assert_eq!(title, "Login failed");
            assert!(
                extra_results
                    .iter()
                    .any(|(label, text)| label == "image_ocr" && text.contains("Login failed"))
            );
        }
        ImageFallback::Metadata { .. } => panic!("expected OCR fallback"),
    }
}

#[test]
fn forwarded_image_without_text_uses_metadata() {
    let msg = telegram_image_msg(Some("@Evgeniya_Koroleva"), vec![]);
    match image_fallback(&msg).expect("image fallback") {
        ImageFallback::Metadata { summary, title } => {
            assert_eq!(title, "Image from @Evgeniya_Koroleva");
            assert!(summary.contains("Forwarded from @Evgeniya_Koroleva"));
            assert!(summary.contains("photo.jpg"));
            assert!(summary.contains("No text recognized"));
        }
        ImageFallback::Ocr { .. } => panic!("expected metadata fallback"),
    }
}

#[test]
fn non_forwarded_image_titles_by_filename() {
    let msg = telegram_image_msg(None, vec![]);
    assert_eq!(
        title_of(&image_fallback(&msg).expect("image fallback")),
        "Image: photo.jpg"
    );
}

#[test]
fn no_image_returns_none() {
    let msg = IncomingMessage::new(
        MessageSource::Http,
        "hello".into(),
        SourceMetadata::Http {
            remote_addr: None,
            user_agent: None,
        },
    );
    assert!(image_fallback(&msg).is_none());
}

#[test]
fn image_bearing_always_has_nonempty_title() {
    for analyses in [
        vec![],
        vec![analysis(ImageAnalysisKind::Photo, "")],
        vec![analysis(ImageAnalysisKind::Interface, "Some text")],
    ] {
        let fb = image_fallback(&telegram_image_msg(Some("@X"), analyses))
            .expect("image-bearing message must have a fallback");
        assert!(!title_of(&fb).is_empty());
    }
}

#[test]
fn plan_blank_tool_result_still_merges_metadata() {
    // Codex finding 1: a present-but-blank tool result must NOT suppress the
    // image metadata, or the node would render an empty summary.
    let msg = telegram_image_msg(Some("@X"), vec![]);
    let mut tool_results = vec![("shell".to_owned(), String::new())];
    let plan = plan_image_fallback(&mut tool_results, image_fallback(&msg));
    assert!(matches!(plan, FallbackPlan::Title(_)));
    // Blank entry dropped; metadata summary present and non-empty.
    assert_eq!(tool_results.len(), 1);
    assert_eq!(tool_results[0].0, "image");
    assert!(!tool_results[0].1.trim().is_empty());
}

#[test]
fn plan_nonblank_tool_result_defers_to_text_path() {
    // Codex finding 2: real tool content keeps the summary AND title (via the
    // caller's text path), so the metadata title is not forced on top of it.
    let msg = telegram_image_msg(Some("@X"), vec![]);
    let mut tool_results = vec![("scrape".to_owned(), "real page content".to_owned())];
    let plan = plan_image_fallback(&mut tool_results, image_fallback(&msg));
    assert_eq!(plan, FallbackPlan::DeferToTextPath);
    // Tool results unchanged — no metadata injected.
    assert_eq!(
        tool_results,
        vec![("scrape".to_owned(), "real page content".to_owned())]
    );
}

#[test]
fn plan_ocr_always_wins_over_tool_results() {
    let msg = telegram_image_msg(
        Some("@X"),
        vec![analysis(ImageAnalysisKind::Interface, "Hello")],
    );
    let mut tool_results = vec![("scrape".to_owned(), "real content".to_owned())];
    let plan = plan_image_fallback(&mut tool_results, image_fallback(&msg));
    assert_eq!(plan, FallbackPlan::Title("Hello".to_owned()));
    assert!(tool_results.iter().any(|(l, _)| l == "image_ocr"));
}

#[test]
fn first_nonempty_line_skips_blanks_and_caps() {
    assert_eq!(
        first_nonempty_line("\n  \n  Hello world  \nx"),
        Some("Hello world".to_owned())
    );
    assert_eq!(first_nonempty_line("   \n  "), None);
    let long = "a".repeat(200);
    assert_eq!(first_nonempty_line(&long).unwrap().chars().count(), 80);
}

#[test]
fn retryable_message_roundtrips_image_analyses() {
    let msg = telegram_image_msg(
        Some("@X"),
        vec![analysis(ImageAnalysisKind::Interface, "text")],
    );
    let r = RetryableMessage::from(&msg);
    let json = serde_json::to_string(&r).expect("serialize");
    let back: RetryableMessage = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back.image_analyses.len(), 1);
    assert_eq!(back.image_analyses[0].recognized_text, "text");
}

#[test]
fn retryable_message_loads_without_image_analyses_field() {
    // Backward-compat: pending rows written before the field must still load.
    let msg = telegram_image_msg(Some("@X"), vec![]);
    let r = RetryableMessage::from(&msg);
    let mut value = serde_json::to_value(&r).expect("to_value");
    value
        .as_object_mut()
        .expect("object")
        .remove("image_analyses");
    let back: RetryableMessage = serde_json::from_value(value).expect("legacy row loads");
    assert!(back.image_analyses.is_empty());
}
