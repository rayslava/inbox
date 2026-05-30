//! LLM enrichment stage: `run_llm` orchestration, success/raw-fallback
//! assembly, memory-context preload, and guidance construction.

use std::sync::Arc;

use tracing::{info, instrument};

use crate::error::InboxError;
use crate::message::{EnrichedMessage, ProcessedMessage};
use crate::processing_status::ProcessingStage;

use super::Pipeline;
use super::context_preload;

impl Pipeline {
    #[instrument(skip(self, enriched), fields(
        id = %enriched.original.id,
        url_count = enriched.urls.len(),
        content_count = enriched.url_contents.len(),
    ))]
    pub(crate) async fn run_llm(
        &self,
        enriched: EnrichedMessage,
    ) -> Result<ProcessedMessage, InboxError> {
        use crate::llm::{LlmOutcome, LlmRequest, LlmTurnProgress};

        let text_preview: String = enriched.original.text.chars().take(120).collect();
        info!(
            id = %enriched.original.id,
            attachment_count = enriched.original.attachments.len(),
            text_preview = %text_preview,
            "Starting LLM processing"
        );

        let (preloaded_text, memories_recalled) = self.preload_memory_context(&enriched).await;

        let (progress_tx, mut progress_rx) =
            tokio::sync::mpsc::unbounded_channel::<LlmTurnProgress>();

        let mut req = LlmRequest::from_enriched(
            &enriched,
            &self.config.llm,
            &self.config.general.attachments_dir,
            &self.build_llm_guidance(&enriched, &preloaded_text),
            // Only force a tool call if URLs are present but none were pre-fetched by
            // the pipeline. If url_contents is already populated the LLM prompt already
            // contains the page text — there is nothing for a tool call to add.
            self.config.llm.prompts.require_tool_for_urls
                && !enriched.urls.is_empty()
                && enriched.url_contents.is_empty(),
        );
        req.progress_tx = Some(progress_tx);

        let tracker = Arc::clone(&self.tracker);
        let id = enriched.original.id;

        let progress_future = async move {
            while let Some(evt) = progress_rx.recv().await {
                tracker.advance(
                    id,
                    ProcessingStage::RunningLlm {
                        turn: evt.turn,
                        max_turns: evt.max_turns,
                        last_tools: evt.tools_called,
                    },
                );
            }
        };

        let urls_fetched = enriched.url_contents.len();

        let (outcome, ()) = tokio::join!(self.llm.complete(req), progress_future);

        match outcome {
            LlmOutcome::Success {
                response,
                helpers,
                tool_calls_made,
            } => Ok(Self::processed_from_success(
                enriched,
                response,
                helpers,
                tool_calls_made,
                memories_recalled,
                urls_fetched,
            )),
            LlmOutcome::RawFallback {
                source_urls,
                tool_results,
                helpers,
                tool_calls_made,
            } => Ok(self
                .processed_from_raw_fallback(
                    enriched,
                    RawFallbackParts {
                        source_urls,
                        tool_results,
                        helpers,
                        tool_calls_made,
                    },
                    memories_recalled,
                    urls_fetched,
                )
                .await),
            LlmOutcome::Discard => {
                info!(id = %enriched.original.id, "Message discarded by LLM fallback policy");
                Err(InboxError::Pipeline(
                    "Message discarded by LLM fallback policy".into(),
                ))
            }
        }
    }

    fn processed_from_success(
        enriched: EnrichedMessage,
        response: crate::message::LlmResponse,
        helpers: Vec<String>,
        tool_calls_made: usize,
        memories_recalled: usize,
        urls_fetched: usize,
    ) -> ProcessedMessage {
        info!(
            id = %enriched.original.id,
            title = %response.title,
            tags = ?response.tags,
            backend = %response.produced_by,
            helper_count = helpers.len(),
            tool_calls = tool_calls_made,
            memories_recalled,
            "LLM processing succeeded"
        );
        let enrichment = crate::message::EnrichmentMetadata {
            helpers,
            memories_recalled,
            urls_fetched,
            tool_calls_made,
        };
        ProcessedMessage {
            enriched,
            llm_response: Some(response),
            fallback_source_urls: vec![],
            fallback_tool_results: vec![],
            fallback_title: None,
            enrichment,
        }
    }

    async fn processed_from_raw_fallback(
        &self,
        enriched: EnrichedMessage,
        parts: RawFallbackParts,
        memories_recalled: usize,
        urls_fetched: usize,
    ) -> ProcessedMessage {
        let RawFallbackParts {
            source_urls,
            mut tool_results,
            mut helpers,
            tool_calls_made,
        } = parts;
        let text_preview: String = enriched.original.text.chars().take(120).collect();
        info!(
            id = %enriched.original.id,
            text_preview = %text_preview,
            tool_calls = tool_calls_made,
            memories_recalled,
            "LLM unavailable, using raw fallback"
        );
        // An image-bearing message must never render empty. Recognized image text
        // (interface OCR) becomes the summary/title; lacking that — and lacking
        // real tool content — attachment metadata does. Otherwise we fall back to
        // the LLM-generated title over tool results.
        let image = image_fallback(&enriched.original);
        let fallback_title =
            if let FallbackPlan::Title(title) = plan_image_fallback(&mut tool_results, image) {
                Some(title)
            } else if enriched.original.text.is_empty() && !tool_results.is_empty() {
                let context = tool_results
                    .iter()
                    .map(|(_, t)| t.as_str())
                    .collect::<Vec<_>>()
                    .join("\n\n");
                match self
                    .llm
                    .complete_text(
                        "Generate a concise 5-word title for this content. \
                     Reply with only the title, no punctuation.",
                        &context,
                    )
                    .await
                {
                    Some((title, produced_by)) => {
                        if !produced_by.is_empty() && !helpers.contains(&produced_by) {
                            helpers.push(produced_by);
                        }
                        Some(title)
                    }
                    None => None,
                }
            } else {
                None
            };
        let enrichment = crate::message::EnrichmentMetadata {
            helpers,
            memories_recalled,
            urls_fetched,
            tool_calls_made,
        };
        ProcessedMessage {
            enriched,
            llm_response: None,
            fallback_source_urls: source_urls,
            fallback_tool_results: tool_results,
            fallback_title,
            enrichment,
        }
    }
}

/// Grouped args for `processed_from_raw_fallback` — keeps the clippy
/// `too_many_arguments` lint happy without sacrificing clarity.
struct RawFallbackParts {
    source_urls: Vec<String>,
    tool_results: Vec<(String, String)>,
    helpers: Vec<String>,
    tool_calls_made: usize,
}

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

impl Pipeline {
    /// Returns the formatted context string and the number of memories that
    /// were recalled (for observability in the org entry drawer).
    async fn preload_memory_context(&self, enriched: &EnrichedMessage) -> (String, usize) {
        let Some(ref store) = self.memory_store else {
            return (String::new(), 0);
        };

        let ctx = context_preload::preload_context(
            store,
            &self.config.memory,
            &enriched.original.text,
            &enriched.urls,
            &enriched.original.user_tags,
        )
        .await;

        let recalled = ctx.memories.len();

        if !ctx.memories.is_empty() {
            let keys: Vec<String> = ctx.memories.iter().map(|m| m.key.clone()).collect();
            let source = enriched.original.source_name();
            let _ = store
                .log_recall_event(&enriched.original.id.to_string(), &keys, source)
                .await;
        }

        (context_preload::format_preloaded_context(&ctx), recalled)
    }

    pub(super) fn build_llm_guidance(
        &self,
        enriched: &EnrichedMessage,
        preloaded_context: &str,
    ) -> String {
        let mut lines = Vec::new();

        if !preloaded_context.is_empty() {
            lines.push(preloaded_context.to_owned());
        }

        if !enriched.original.user_tags.is_empty() {
            let tag_list = enriched
                .original
                .user_tags
                .iter()
                .map(|t| format!("#{t}"))
                .collect::<Vec<_>>()
                .join(", ");
            lines.push(format!(
                "The user has explicitly tagged this message with: {tag_list}. \
                 Make sure these tags appear in your tags output."
            ));
        }

        let hints = &enriched.original.preprocessing_hints;
        if hints.force_web_search {
            lines.push(
                "Use the web_search tool to find more context before producing the final JSON."
                    .to_owned(),
            );
        }
        for hint in &hints.extra_llm_hints {
            lines.push(hint.clone());
        }

        let tool_lines = self.config.tooling.prompt_block();
        if !tool_lines.trim().is_empty() {
            lines.push(tool_lines);
        }

        if self.config.llm.prompts.require_tool_for_urls
            && !self.config.llm.prompts.url_tool_decision.trim().is_empty()
            && !enriched.urls.is_empty()
        {
            let urls = enriched
                .urls
                .iter()
                .map(url::Url::as_str)
                .collect::<Vec<_>>()
                .join(", ");
            let decision = self
                .config
                .llm
                .prompts
                .url_tool_decision
                .replace("{urls}", &urls);
            lines.push(decision);
        }

        if !self.config.llm.prompts.js_shell_tool_hint.trim().is_empty()
            && self.config.pipeline.web_content.js_shell_policy
                == crate::config::JsShellPolicy::ToolOnly
            && !enriched.urls.is_empty()
            && enriched.url_contents.is_empty()
        {
            let urls = enriched
                .urls
                .iter()
                .map(url::Url::as_str)
                .collect::<Vec<_>>()
                .join(", ");
            let hint = self
                .config
                .llm
                .prompts
                .js_shell_tool_hint
                .replace("{urls}", &urls);
            lines.push(hint);
        }

        lines.join("\n")
    }
}
