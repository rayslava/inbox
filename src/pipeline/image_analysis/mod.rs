//! Vision-LLM image analysis: classify an attachment as interface vs photo and
//! transcribe any visible text. Runs before pre-processing so the recognized
//! text can flow into enrichment. Non-fatal per image.

use tracing::warn;

use crate::config::ImageAnalysisConfig;
use crate::llm::LlmChain;
use crate::message::{Attachment, ImageAnalysisResult, IncomingMessage, MediaKind};

mod classify;
#[cfg(test)]
mod tests;

/// Result of the image-analysis stage: the per-image results plus whether any
/// image's text was left unread because every vision backend was unavailable
/// (a transient outage that should hold the node pending for retry).
#[derive(Debug, Default)]
pub struct ImageAnalysisOutcome {
    pub results: Vec<ImageAnalysisResult>,
    pub vision_unavailable: bool,
}

/// Outcome of analyzing one image.
enum AnalyzeOne {
    /// Vision produced a result.
    Done(ImageAnalysisResult),
    /// Non-fatal skip (unreadable/too large/no vision backend) — ignore.
    Skip,
    /// Every vision backend was unavailable — a transient outage.
    Unavailable,
}

/// Analyze the image attachments on `msg`. Returns empty/`false` when analysis
/// is disabled; caps at `cfg.max_attachments`; skips any image that fails to
/// read or exceeds `vision_max_bytes`. Sets `vision_unavailable` when an image's
/// text went unread because all vision backends were down.
pub async fn analyze_images(
    chain: &LlmChain,
    cfg: &ImageAnalysisConfig,
    vision_max_bytes: usize,
    msg: &IncomingMessage,
) -> ImageAnalysisOutcome {
    if !cfg.enabled {
        return ImageAnalysisOutcome::default();
    }
    let mut outcome = ImageAnalysisOutcome::default();
    for att in msg
        .attachments
        .iter()
        .filter(|a| a.media_kind == MediaKind::Image)
        .take(cfg.max_attachments)
    {
        match analyze_image(chain, cfg, vision_max_bytes, att).await {
            AnalyzeOne::Done(r) => outcome.results.push(r),
            AnalyzeOne::Unavailable => outcome.vision_unavailable = true,
            AnalyzeOne::Skip => {}
        }
    }
    outcome
}

/// Analyze a single image attachment. `Skip` when the file cannot be read, is
/// too large, or no vision backend is eligible; `Unavailable` when all vision
/// backends were down.
async fn analyze_image(
    chain: &LlmChain,
    cfg: &ImageAnalysisConfig,
    vision_max_bytes: usize,
    att: &Attachment,
) -> AnalyzeOne {
    // Reject oversized files by stat before reading them into memory.
    match tokio::fs::metadata(&att.saved_path).await {
        Ok(meta) if meta.len() > vision_max_bytes as u64 => {
            warn!(
                path = %att.saved_path.display(),
                size = meta.len(),
                limit = vision_max_bytes,
                "image analysis: too large, skipping"
            );
            return AnalyzeOne::Skip;
        }
        Ok(_) => {}
        Err(e) => {
            warn!(path = %att.saved_path.display(), ?e, "image analysis: stat failed");
            return AnalyzeOne::Skip;
        }
    }
    let bytes = match tokio::fs::read(&att.saved_path).await {
        Ok(b) => b,
        Err(e) => {
            warn!(path = %att.saved_path.display(), ?e, "image analysis: read failed");
            return AnalyzeOne::Skip;
        }
    };
    // Defensive: the file may have grown between stat and read.
    if bytes.len() > vision_max_bytes {
        warn!(
            path = %att.saved_path.display(),
            size = bytes.len(),
            limit = vision_max_bytes,
            "image analysis: too large after read, skipping"
        );
        return AnalyzeOne::Skip;
    }
    let mime = att
        .mime_type
        .clone()
        .unwrap_or_else(|| "image/jpeg".to_owned());
    let b64 = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &bytes);

    let (text, produced_by) = match chain
        .complete_vision_text(&cfg.prompt, "", vec![(mime, b64)])
        .await
    {
        Ok(pair) => pair,
        // Genuine outage: the text is unread and should be retried.
        Err(crate::llm::VisionUnavailable::AllUnavailable) => return AnalyzeOne::Unavailable,
        // No vision backend at all — nothing to retry; handled downstream.
        Err(crate::llm::VisionUnavailable::NoVisionBackend) => return AnalyzeOne::Skip,
    };
    let recognized = text.trim().to_owned();
    let kind = classify::classify(&recognized, cfg.interface_min_chars);

    AnalyzeOne::Done(ImageAnalysisResult {
        attachment_name: att.original_name.clone(),
        kind,
        recognized_text: recognized,
        produced_by,
    })
}
