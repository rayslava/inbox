use std::sync::Arc;

use tokio::sync::mpsc;
use tracing::{debug, error, info, instrument, warn};

use crate::config::Config;
use crate::error::InboxError;
use crate::message::{EnrichedMessage, ImageAnalysisKind, IncomingMessage, MediaKind};
use crate::output::OutputWriter;
use crate::pending::PendingStore;
use crate::processing_status::{ProcessingStage, ProcessingTracker};

pub mod content_extractor;
pub mod context_preload;
pub mod image_analysis;
pub mod preprocess;
pub mod tags;
pub mod url_classifier;
pub mod url_extractor;
pub mod url_fetcher;

use url_classifier::{UrlKind, classify_url};
use url_extractor::extract_urls;
use url_fetcher::UrlFetcher;

mod fallback;
mod llm_stage;

pub struct Pipeline {
    pub config: Arc<Config>,
    pub llm: Arc<crate::llm::LlmChain>,
    pub writer: Arc<dyn OutputWriter>,
    pub fetcher: UrlFetcher,
    pub tracker: Arc<ProcessingTracker>,
    pub memory_store: Option<Arc<crate::memory::MemoryStore>>,
    pub pending: Option<Arc<PendingStore>>,
    pub in_flight: Arc<tokio::sync::Semaphore>,
}

impl Pipeline {
    /// Build a `Pipeline`.
    ///
    /// # Errors
    /// Returns an error if the URL fetcher's HTTP client cannot be built.
    pub fn new(
        config: Arc<Config>,
        llm: Arc<crate::llm::LlmChain>,
        writer: Arc<dyn OutputWriter>,
        tracker: Arc<ProcessingTracker>,
        memory_store: Option<Arc<crate::memory::MemoryStore>>,
        pending: Option<Arc<PendingStore>>,
    ) -> Result<Self, InboxError> {
        let fetcher = UrlFetcher::new(&config.url_fetch)?;
        let in_flight_limit =
            std::thread::available_parallelism().map_or(8, std::num::NonZeroUsize::get) * 4;
        Ok(Self {
            config,
            llm,
            writer,
            fetcher,
            tracker,
            memory_store,
            pending,
            in_flight: Arc::new(tokio::sync::Semaphore::new(in_flight_limit)),
        })
    }

    pub async fn run(self: Arc<Self>, mut rx: mpsc::Receiver<IncomingMessage>) {
        info!("Pipeline started, waiting for messages");
        while let Some(msg) = rx.recv().await {
            let Ok(permit) = Arc::clone(&self.in_flight).acquire_owned().await else {
                break;
            };
            let pipeline = Arc::clone(&self);
            tokio::spawn(async move {
                let _permit = permit;
                let source = msg.source_name();
                let timer_start = std::time::Instant::now();
                match pipeline.process(msg).await {
                    Ok(()) => {
                        metrics::counter!(
                            crate::telemetry::MESSAGES_PROCESSED,
                            "source" => source,
                            "status" => "success"
                        )
                        .increment(1);
                    }
                    Err(e) => {
                        error!(?e, source, "Pipeline error");
                        metrics::counter!(
                            crate::telemetry::MESSAGES_PROCESSED,
                            "source" => source,
                            "status" => "failure"
                        )
                        .increment(1);
                    }
                }
                let elapsed = timer_start.elapsed().as_secs_f64();
                metrics::histogram!(
                    crate::telemetry::PROCESSING_DURATION,
                    "source" => source
                )
                .record(elapsed);
            });
        }
        info!("Pipeline channel closed, exiting");
    }

    /// Run the vision-LLM image-analysis stage, populating `msg.image_analyses`.
    /// No-op when disabled or when the message carries no image attachments.
    /// Returns `true` when an image's text was left unread because every vision
    /// backend was unavailable (a transient outage to retry).
    async fn run_image_analysis(
        &self,
        id: uuid::Uuid,
        notifier: &mut Option<Box<dyn crate::processing_status::StatusNotifier>>,
        msg: &mut IncomingMessage,
    ) -> bool {
        if !self.config.pipeline.image_analysis.enabled
            || !msg
                .attachments
                .iter()
                .any(|a| a.media_kind == MediaKind::Image)
        {
            return false;
        }
        self.tracker.advance(id, ProcessingStage::AnalyzingImages);
        if let Some(n) = notifier {
            n.advance(ProcessingStage::AnalyzingImages).await;
        }
        let outcome = image_analysis::analyze_images(
            &self.llm,
            &self.config.pipeline.image_analysis,
            self.config.llm.vision_max_bytes,
            msg,
        )
        .await;
        if !outcome.results.is_empty() {
            info!(id = %id, count = outcome.results.len(), "Image analysis complete");
        }
        if outcome.vision_unavailable {
            warn!(id = %id, "Image text unread — all vision backends unavailable");
        }
        msg.image_analyses = outcome.results;
        outcome.vision_unavailable
    }

    /// Re-run image analysis for a resumed item (no tracker/notifier). Updates
    /// `msg.image_analyses` only when re-OCR yields results, preserving any
    /// stored analyses otherwise — a still-down backend or a missing attachment
    /// file must not discard previously recognized text.
    pub(crate) async fn resume_image_analysis(&self, msg: &mut IncomingMessage) {
        if !self.config.pipeline.image_analysis.enabled
            || !msg
                .attachments
                .iter()
                .any(|a| a.media_kind == MediaKind::Image)
        {
            return;
        }
        let outcome = image_analysis::analyze_images(
            &self.llm,
            &self.config.pipeline.image_analysis,
            self.config.llm.vision_max_bytes,
            msg,
        )
        .await;
        if !outcome.results.is_empty() {
            msg.image_analyses = outcome.results;
        }
    }

    /// Process a single incoming message through the full pipeline.
    ///
    /// # Errors
    /// Returns an error if enrichment, LLM completion, or output writing fails.
    #[instrument(skip(self, msg), fields(id = %msg.id, source = %msg.source))]
    pub async fn process(&self, mut msg: IncomingMessage) -> Result<(), InboxError> {
        let id = msg.id;
        let mut notifier = msg.status_notifier.take();

        let (cleaned_text, user_tags) = tags::extract_user_tags(&msg.text);
        if !user_tags.is_empty() {
            info!(id = %id, tags = ?user_tags, "Extracted user tags from message");
            msg.text = cleaned_text;
            msg.user_tags = user_tags;
        }

        self.tracker.insert(
            id,
            msg.source.as_str().to_owned(),
            msg.text.chars().take(80).collect(),
        );

        // Image analysis (interface classification + OCR) runs before
        // pre-processing so the recognized text can inform rules and enrichment.
        let vision_unavailable = self.run_image_analysis(id, &mut notifier, &mut msg).await;

        let mut hints = preprocess::run_preprocessing(&msg, &self.config.pipeline.preprocessing);
        // Tag interface/screenshot images so the org node is easy to find.
        if msg
            .image_analyses
            .iter()
            .any(|a| a.kind == ImageAnalysisKind::Interface)
            && !hints.suggested_tags.iter().any(|t| t == "interface")
        {
            hints.suggested_tags.push("interface".to_owned());
        }
        if hints.force_web_search || !hints.suggested_tags.is_empty() {
            info!(id = %id, force_web_search = hints.force_web_search,
                suggested_tags = ?hints.suggested_tags, "Pre-processing hints computed");
        }
        msg.preprocessing_hints = hints;

        let enriched = self
            .run_stage(
                id,
                &mut notifier,
                ProcessingStage::Enriching,
                self.enrich(msg),
            )
            .await?;

        let llm_initial = ProcessingStage::RunningLlm {
            turn: 0,
            max_turns: self.llm.max_tool_turns(),
            last_tools: vec![],
        };
        let processed = self
            .run_stage(
                id,
                &mut notifier,
                llm_initial,
                self.run_llm(enriched, vision_unavailable),
            )
            .await?;

        self.run_stage(
            id,
            &mut notifier,
            ProcessingStage::Writing,
            self.writer.write(&processed, &self.config),
        )
        .await?;

        // Persist items for background retry: raw fallbacks (no LLM response) and
        // incomplete nodes (vision unavailable — re-OCR on resume).
        if (processed.llm_response.is_none() || processed.is_incomplete())
            && let Some(ref store) = self.pending
        {
            let tg_msg_id = notifier.as_ref().and_then(|n| n.telegram_status_msg_id());
            if let Err(e) = store.insert(id, &processed, tg_msg_id).await {
                warn!(?e, %id, "Failed to persist pending item — resume unavailable for this message");
            }
        }

        let title = processed.llm_response.as_ref().map_or_else(
            || {
                processed
                    .enriched
                    .original
                    .text
                    .lines()
                    .next()
                    .unwrap_or("")
                    .to_owned()
            },
            |r| r.title.clone(),
        );
        // An incomplete node is not yet done — surface it as Pending so it is not
        // reported as successfully processed (it is retried on resume).
        let final_stage = if processed.is_incomplete() {
            ProcessingStage::Pending { title }
        } else {
            ProcessingStage::Done { title }
        };
        self.tracker.advance(id, final_stage.clone());
        if let Some(n) = &mut notifier {
            n.advance(final_stage).await;
        }
        Ok(())
    }

    async fn run_stage<T>(
        &self,
        id: uuid::Uuid,
        notifier: &mut Option<Box<dyn crate::processing_status::StatusNotifier>>,
        stage: ProcessingStage,
        fut: impl std::future::Future<Output = Result<T, InboxError>>,
    ) -> Result<T, InboxError> {
        self.tracker.advance(id, stage.clone());
        if let Some(n) = notifier {
            n.advance(stage).await;
        }
        match fut.await {
            Ok(v) => Ok(v),
            Err(e) => {
                let failed = ProcessingStage::Failed {
                    reason: e.to_string(),
                };
                self.tracker.advance(id, failed.clone());
                if let Some(n) = notifier {
                    n.advance(failed).await;
                }
                Err(e)
            }
        }
    }

    #[instrument(skip(self, msg), fields(id = %msg.id))]
    async fn enrich(&self, msg: IncomingMessage) -> Result<EnrichedMessage, InboxError> {
        if !self.config.url_fetch.enabled {
            debug!(id = %msg.id, "URL fetch disabled, skipping enrichment");
            return Ok(EnrichedMessage {
                urls: Vec::new(),
                url_contents: Vec::new(),
                original: msg,
            });
        }

        let urls = extract_urls(&msg.text);
        info!(id = %msg.id, url_count = urls.len(), "Extracted URLs from message");

        let mut url_contents = Vec::new();
        let mut attachments = msg.attachments.clone();

        for url in &urls {
            self.process_url(url, msg.id, &mut url_contents, &mut attachments)
                .await;
        }

        info!(
            id = %msg.id,
            url_count = urls.len(),
            content_count = url_contents.len(),
            attachment_count = attachments.len(),
            "Message enrichment complete"
        );

        Ok(EnrichedMessage {
            original: IncomingMessage { attachments, ..msg },
            urls,
            url_contents,
        })
    }

    async fn process_url(
        &self,
        url: &url::Url,
        msg_id: uuid::Uuid,
        url_contents: &mut Vec<crate::url_content::UrlContent>,
        attachments: &mut Vec<crate::message::Attachment>,
    ) {
        let host = url.host_str().unwrap_or("");
        if self
            .config
            .url_fetch
            .skip_domains
            .iter()
            .any(|d| host_matches_skip_domain(host, d))
        {
            debug!(%url, "Skipping URL — domain is in skip list");
            return;
        }

        match classify_url(url, &self.fetcher).await {
            UrlKind::Page => {
                if let Some(content) = self.fetcher.fetch_page(url).await {
                    if matches_js_shell_policy(&self.config, &content.text) {
                        debug!(
                            %url,
                            policy = ?self.config.pipeline.web_content.js_shell_policy,
                            "Page content matched JavaScript-shell policy; skipping direct content"
                        );
                        return;
                    }
                    debug!(
                        %url,
                        text_len = content.text.len(),
                        title = ?content.page_title,
                        "Page content fetched"
                    );
                    url_contents.push(make_url_content(
                        url,
                        content,
                        self.config.llm.url_content_max_chars,
                    ));
                } else {
                    warn!(%url, "Failed to fetch page content");
                }
            }
            UrlKind::File { ref mime } => {
                if let Some(att) = self
                    .fetcher
                    .download_file(url, msg_id, &self.config.general.attachments_dir)
                    .await
                {
                    debug!(%url, %mime, filename = %att.original_name, "File attachment added");
                    attachments.push(att);
                } else {
                    warn!(%url, %mime, "Failed to download file attachment");
                }
            }
            UrlKind::Unknown => {
                debug!(%url, "Unknown URL kind, attempting page fetch as fallback");
                if let Some(content) = self.fetcher.fetch_page(url).await {
                    if matches_js_shell_policy(&self.config, &content.text) {
                        debug!(
                            %url,
                            policy = ?self.config.pipeline.web_content.js_shell_policy,
                            "Page content matched JavaScript-shell policy; skipping direct content"
                        );
                        return;
                    }
                    url_contents.push(make_url_content(
                        url,
                        content,
                        self.config.llm.url_content_max_chars,
                    ));
                }
            }
        }
    }
}

fn host_matches_skip_domain(host: &str, skip_domain: &str) -> bool {
    let host = host.trim().trim_end_matches('.').to_ascii_lowercase();
    let domain = skip_domain
        .trim()
        .trim_start_matches('.')
        .trim_end_matches('.')
        .to_ascii_lowercase();

    if host.is_empty() || domain.is_empty() {
        return false;
    }

    if host == domain {
        return true;
    }

    host.strip_suffix(&domain)
        .is_some_and(|prefix| prefix.ends_with('.'))
}

fn make_url_content(
    url: &url::Url,
    content: crate::url_content::UrlContent,
    max_chars: usize,
) -> crate::url_content::UrlContent {
    crate::url_content::UrlContent {
        url: url.to_string(),
        text: truncate_chars(&content.text, max_chars),
        page_title: content.page_title,
        headings: content.headings,
    }
}

fn matches_js_shell_policy(config: &Config, text: &str) -> bool {
    use crate::config::JsShellPolicy;

    if !matches!(
        config.pipeline.web_content.js_shell_policy,
        JsShellPolicy::ToolOnly | JsShellPolicy::Drop
    ) {
        return false;
    }

    let haystack = text.to_ascii_lowercase();
    config
        .pipeline
        .web_content
        .js_shell_patterns
        .iter()
        .map(|p| p.trim().to_ascii_lowercase())
        .filter(|p| !p.is_empty())
        .any(|p| haystack.contains(&p))
}

fn truncate_chars(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_owned()
    } else {
        s.chars().take(max).collect()
    }
}

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_image_fallback;
#[cfg(test)]
mod tests_image_stage;
#[cfg(test)]
mod tests_url;
