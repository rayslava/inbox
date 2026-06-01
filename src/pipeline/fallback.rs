//! Image-derived fallback helpers: build a non-empty node (recognized OCR text
//! first, attachment metadata otherwise) for an image-bearing message when the
//! LLM is unavailable or skipped.

use anodized::spec;

/// A non-empty fallback derived from an image-bearing message.
pub(super) enum ImageFallback {
    /// Recognized image text (interface OCR) — always wins.
    Ocr {
        /// `(label, text)` pairs to merge into the fallback tool results.
        extra_results: Vec<(String, String)>,
        title: String,
    },
    /// Attachment metadata — used only when no real tool content exists.
    Metadata { summary: String, title: String },
}

/// Plan for how the raw-fallback title/summary should be produced.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum FallbackPlan {
    /// An image title was produced (and `tool_results` mutated to match).
    Title(String),
    /// No image-derived title; the caller's text/LLM path should run.
    DeferToTextPath,
}

/// Derive an image fallback (recognized text first, attachment metadata
/// otherwise). `None` when the message has no image.
pub(super) fn image_fallback(msg: &crate::message::IncomingMessage) -> Option<ImageFallback> {
    use crate::message::MediaKind;

    let recognized: Vec<&crate::message::ImageAnalysisResult> = msg
        .image_analyses
        .iter()
        .filter(|a| !a.recognized_text.trim().is_empty())
        .collect();

    if let Some(first) = recognized.first() {
        let title = first_nonempty_line(&first.recognized_text)
            .unwrap_or_else(|| "Recognized image text".to_owned());
        let extra_results = recognized
            .iter()
            .map(|a| ("image_ocr".to_owned(), a.recognized_text.clone()))
            .collect();
        return Some(ImageFallback::Ocr {
            extra_results,
            title,
        });
    }

    if msg
        .attachments
        .iter()
        .any(|a| a.media_kind == MediaKind::Image)
    {
        return Some(ImageFallback::Metadata {
            summary: image_metadata_summary(msg),
            title: image_metadata_title(msg),
        });
    }

    None
}

/// Apply an image fallback to `tool_results`, returning whether an image title
/// was produced. OCR always wins; metadata wins only when no existing tool
/// result has non-whitespace text — otherwise real tool content keeps both the
/// summary and (via the caller's text path) the title, avoiding both an
/// empty-summary node and a title/body mismatch.
pub(super) fn plan_image_fallback(
    tool_results: &mut Vec<(String, String)>,
    image: Option<ImageFallback>,
) -> FallbackPlan {
    let has_real_tool_text = tool_results.iter().any(|(_, t)| !t.trim().is_empty());
    match image {
        Some(ImageFallback::Ocr {
            extra_results,
            title,
        }) => {
            tool_results.extend(extra_results);
            FallbackPlan::Title(title)
        }
        Some(ImageFallback::Metadata { summary, title }) if !has_real_tool_text => {
            // Drop blank tool entries so the metadata summary is what renders.
            tool_results.retain(|(_, t)| !t.trim().is_empty());
            tool_results.push(("image".to_owned(), summary));
            FallbackPlan::Title(title)
        }
        _ => FallbackPlan::DeferToTextPath,
    }
}

/// First non-blank line, trimmed and capped at 80 chars.
#[spec(ensures: output.as_ref().is_none_or(|s| s.chars().count() <= 80))]
pub(super) fn first_nonempty_line(s: &str) -> Option<String> {
    s.lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .map(|l| l.chars().take(80).collect())
}

fn image_metadata_title(msg: &crate::message::IncomingMessage) -> String {
    use crate::message::{MediaKind, SourceMetadata};
    if let SourceMetadata::Telegram {
        forwarded_from: Some(ff),
        ..
    } = &msg.metadata
    {
        return format!("Image from {ff}");
    }
    match msg
        .attachments
        .iter()
        .find(|a| a.media_kind == MediaKind::Image)
    {
        Some(a) => format!("Image: {}", a.original_name),
        None => "Image".to_owned(),
    }
}

fn image_metadata_summary(msg: &crate::message::IncomingMessage) -> String {
    use std::fmt::Write as _;

    use crate::message::{MediaKind, SourceMetadata};

    let mut s = String::new();
    if let SourceMetadata::Telegram {
        forwarded_from: Some(ff),
        ..
    } = &msg.metadata
    {
        let _ = write!(s, "Forwarded from {ff}. ");
    }
    let names: Vec<String> = msg
        .attachments
        .iter()
        .filter(|a| a.media_kind == MediaKind::Image)
        .map(|a| match &a.mime_type {
            Some(m) => format!("{} ({m})", a.original_name),
            None => a.original_name.clone(),
        })
        .collect();
    let _ = write!(
        s,
        "Image attachment(s): {}. No text recognized.",
        names.join(", ")
    );
    s
}
