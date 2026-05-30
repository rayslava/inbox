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

/// Analyze the image attachments on `msg`, returning per-image results. Returns
/// empty when analysis is disabled; caps at `cfg.max_attachments`; skips any
/// image that fails to read, exceeds `vision_max_bytes`, or yields no text.
pub async fn analyze_images(
    chain: &LlmChain,
    cfg: &ImageAnalysisConfig,
    vision_max_bytes: usize,
    msg: &IncomingMessage,
) -> Vec<ImageAnalysisResult> {
    if !cfg.enabled {
        return Vec::new();
    }
    let mut results = Vec::new();
    for att in msg
        .attachments
        .iter()
        .filter(|a| a.media_kind == MediaKind::Image)
        .take(cfg.max_attachments)
    {
        if let Some(r) = analyze_image(chain, cfg, vision_max_bytes, att).await {
            results.push(r);
        }
    }
    results
}

/// Analyze a single image attachment. `None` when the file cannot be read, is
/// too large, or no vision backend produced text.
async fn analyze_image(
    chain: &LlmChain,
    cfg: &ImageAnalysisConfig,
    vision_max_bytes: usize,
    att: &Attachment,
) -> Option<ImageAnalysisResult> {
    // Reject oversized files by stat before reading them into memory.
    match tokio::fs::metadata(&att.saved_path).await {
        Ok(meta) if meta.len() > vision_max_bytes as u64 => {
            warn!(
                path = %att.saved_path.display(),
                size = meta.len(),
                limit = vision_max_bytes,
                "image analysis: too large, skipping"
            );
            return None;
        }
        Ok(_) => {}
        Err(e) => {
            warn!(path = %att.saved_path.display(), ?e, "image analysis: stat failed");
            return None;
        }
    }
    let bytes = match tokio::fs::read(&att.saved_path).await {
        Ok(b) => b,
        Err(e) => {
            warn!(path = %att.saved_path.display(), ?e, "image analysis: read failed");
            return None;
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
        return None;
    }
    let mime = att
        .mime_type
        .clone()
        .unwrap_or_else(|| "image/jpeg".to_owned());
    let b64 = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &bytes);

    let (text, produced_by) = chain
        .complete_vision_text(&cfg.prompt, "", vec![(mime, b64)])
        .await?;
    let recognized = text.trim().to_owned();
    let kind = classify::classify(&recognized, cfg.interface_min_chars);

    Some(ImageAnalysisResult {
        attachment_name: att.original_name.clone(),
        kind,
        recognized_text: recognized,
        produced_by,
    })
}
