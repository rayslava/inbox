//! Memo short-circuit: a `#memo`-tagged message skips URL fetch and the LLM
//! enrichment call and is written straight to org. Image OCR (which runs before
//! this stage) is preserved, so an image memo whose vision backend was down is
//! still held incomplete and re-OCR'd on resume rather than finalized empty.

use anodized::spec;

use crate::message::{
    EnrichedMessage, EnrichmentMetadata, LlmResponse, ProcessedMessage, ProcessingCompleteness,
};

use super::fallback::first_nonempty_line;
use super::llm_stage::completeness_of;

/// Normalize a configured memo tag to match the stored form of user tags
/// (`extract_user_tags` lowercases and drops the leading `#`).
fn normalize_tag(tag: &str) -> String {
    tag.trim().trim_start_matches('#').to_lowercase()
}

/// Whether `user_tags` contains any of the configured memo tags.
///
/// `user_tags` are already lowercased and `#`-free; configured tags are
/// normalized the same way so a config value like `"#Memo"` still matches.
pub(crate) fn is_memo(user_tags: &[String], memo_tags: &[String]) -> bool {
    memo_tags
        .iter()
        .map(|m| normalize_tag(m))
        .any(|m| !m.is_empty() && user_tags.iter().any(|t| t.eq_ignore_ascii_case(&m)))
}

/// Build an org node for a memo without any LLM / network / memory call.
///
/// Recognized image text (OCR ran before this stage) is folded into the summary.
/// A complete memo is carried as a populated `llm_response` (`produced_by =
/// "memo"`) so it is final — never `:inbox_pending:`, never persisted. An
/// image memo whose vision was unavailable and yielded no text is left
/// incomplete (`llm_response = None`) so it is held pending and re-OCR'd on
/// resume, exactly like the non-memo image-only path.
#[spec(ensures: output.llm_response.is_some() == (output.incomplete == ProcessingCompleteness::Complete))]
pub(crate) fn processed_memo(
    enriched: EnrichedMessage,
    vision_available: bool,
) -> ProcessedMessage {
    let completeness = completeness_of(&enriched, vision_available);
    let original = &enriched.original;

    let ocr: Vec<&str> = original
        .image_analyses
        .iter()
        .map(|a| a.recognized_text.trim())
        .filter(|t| !t.is_empty())
        .collect();

    let text = original.text.trim();
    let summary = match (text.is_empty(), ocr.is_empty()) {
        (false, true) => text.to_owned(),
        (true, false) => ocr.join("\n\n"),
        (false, false) => format!("{text}\n\n{}", ocr.join("\n\n")),
        (true, true) => String::new(),
    };

    let title = first_nonempty_line(&original.text)
        .or_else(|| ocr.first().copied().and_then(first_nonempty_line))
        .unwrap_or_else(|| "(untitled)".to_owned());

    let urls_fetched = enriched.url_contents.len();

    // Incomplete (image vision-down, no text) mirrors the non-memo pending node:
    // no LLM response, held for re-OCR on resume.
    let llm_response = (completeness == ProcessingCompleteness::Complete).then(|| LlmResponse {
        title,
        tags: vec![],
        summary,
        excerpt: None,
        produced_by: "memo".to_owned(),
    });

    ProcessedMessage {
        enriched,
        llm_response,
        incomplete: completeness,
        fallback_source_urls: vec![],
        fallback_tool_results: vec![],
        fallback_title: None,
        enrichment: EnrichmentMetadata {
            urls_fetched,
            ..EnrichmentMetadata::default()
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{
        Attachment, ImageAnalysisKind, ImageAnalysisResult, IncomingMessage, MediaKind,
        MessageSource, SourceMetadata,
    };

    fn msg(text: &str) -> IncomingMessage {
        IncomingMessage::new(
            MessageSource::Http,
            text.to_owned(),
            SourceMetadata::Http {
                remote_addr: None,
                user_agent: None,
            },
        )
    }

    fn enriched(original: IncomingMessage) -> EnrichedMessage {
        EnrichedMessage {
            original,
            urls: vec![],
            url_contents: vec![],
        }
    }

    fn ocr_result(text: &str) -> ImageAnalysisResult {
        ImageAnalysisResult {
            attachment_name: "shot.png".to_owned(),
            kind: ImageAnalysisKind::Interface,
            recognized_text: text.to_owned(),
            produced_by: "vision:test".to_owned(),
        }
    }

    fn image_attachment() -> Attachment {
        Attachment {
            original_name: "shot.png".to_owned(),
            saved_path: std::path::PathBuf::from("/tmp/shot.png"),
            mime_type: Some("image/png".to_owned()),
            media_kind: MediaKind::Image,
        }
    }

    #[test]
    fn is_memo_matches_lowercase() {
        assert!(is_memo(&["memo".into()], &["memo".into()]));
    }

    #[test]
    fn is_memo_case_insensitive() {
        assert!(is_memo(&["memo".into()], &["Memo".into()]));
    }

    #[test]
    fn is_memo_config_tag_with_leading_hash_matches() {
        assert!(is_memo(&["memo".into()], &["#memo".into()]));
        assert!(is_memo(&["memo".into()], &[" #Memo ".into()]));
    }

    #[test]
    fn is_memo_empty_config_tag_never_matches() {
        assert!(!is_memo(&["memo".into()], &["#".into()]));
        assert!(!is_memo(&[String::new()], &[String::new()]));
    }

    #[test]
    fn is_memo_multiple_configured_tags() {
        assert!(is_memo(&["note".into()], &["memo".into(), "note".into()]));
    }

    #[test]
    fn is_memo_no_match() {
        assert!(!is_memo(&["rust".into()], &["memo".into()]));
        assert!(!is_memo(&[], &["memo".into()]));
    }

    #[test]
    fn processed_memo_text_only() {
        let p = processed_memo(enriched(msg("Oil change 4851 km")), true);
        let r = p.llm_response.expect("memo node has a response");
        assert_eq!(r.title, "Oil change 4851 km");
        assert_eq!(r.summary, "Oil change 4851 km");
        assert!(r.tags.is_empty());
        assert_eq!(r.produced_by, "memo");
        assert!(r.excerpt.is_none());
        assert_eq!(p.incomplete, ProcessingCompleteness::Complete);
    }

    #[test]
    fn processed_memo_title_first_line() {
        let p = processed_memo(enriched(msg("Buy milk\nand eggs")), true);
        let r = p.llm_response.expect("response");
        assert_eq!(r.title, "Buy milk");
        assert_eq!(r.summary, "Buy milk\nand eggs");
    }

    #[test]
    fn processed_memo_folds_ocr_when_text_empty() {
        let mut m = msg("");
        m.attachments.push(image_attachment());
        m.image_analyses = vec![ocr_result("Receipt total $42")];
        let p = processed_memo(enriched(m), true);
        let r = p.llm_response.expect("response");
        assert_eq!(r.summary, "Receipt total $42");
        assert_eq!(r.title, "Receipt total $42");
    }

    #[test]
    fn processed_memo_folds_ocr_after_text() {
        let mut m = msg("Note");
        m.attachments.push(image_attachment());
        m.image_analyses = vec![ocr_result("Screen text")];
        let p = processed_memo(enriched(m), true);
        let r = p.llm_response.expect("response");
        assert_eq!(r.summary, "Note\n\nScreen text");
        assert_eq!(r.title, "Note");
    }

    #[test]
    fn processed_memo_untitled_when_empty() {
        let p = processed_memo(enriched(msg("")), true);
        let r = p.llm_response.expect("response");
        assert_eq!(r.title, "(untitled)");
        assert!(r.summary.is_empty());
    }

    #[test]
    fn processed_memo_image_vision_down_is_incomplete() {
        // Image memo, no recognized text, vision unavailable: must be held
        // incomplete (no LLM response) for re-OCR on resume, not finalized empty.
        let mut m = msg("");
        m.attachments.push(image_attachment());
        let p = processed_memo(enriched(m), false);
        assert!(
            p.llm_response.is_none(),
            "incomplete memo must not be final"
        );
        assert_eq!(
            p.incomplete,
            ProcessingCompleteness::IncompleteVisionUnavailable
        );
    }

    #[test]
    fn processed_memo_image_with_ocr_is_complete_even_if_vision_flag_false() {
        // Recognized text present means vision worked: complete regardless.
        let mut m = msg("");
        m.attachments.push(image_attachment());
        m.image_analyses = vec![ocr_result("recognized")];
        let p = processed_memo(enriched(m), false);
        assert!(p.llm_response.is_some());
        assert_eq!(p.incomplete, ProcessingCompleteness::Complete);
    }
}
