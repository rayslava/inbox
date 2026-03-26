use std::sync::Arc;

use tokio::sync::mpsc;
use tracing::{debug, error, info, instrument, warn};

use crate::config::Config;
use crate::error::InboxError;
use crate::message::{EnrichedMessage, IncomingMessage, ProcessedMessage};
use crate::output::OutputWriter;
use crate::processing_status::{ProcessingStage, ProcessingTracker};

pub mod content_extractor;
pub mod context_preload;
pub mod preprocess;
pub mod tags;
pub mod url_classifier;
pub mod url_extractor;
pub mod url_fetcher;

use url_classifier::{UrlKind, classify_url};
use url_extractor::extract_urls;
use url_fetcher::UrlFetcher;

pub struct Pipeline {
    pub config: Arc<Config>,
    pub llm: Arc<crate::llm::LlmChain>,
    pub writer: Arc<dyn OutputWriter>,
    pub fetcher: UrlFetcher,
    pub tracker: Arc<ProcessingTracker>,
    pub memory_store: Option<Arc<crate::memory::MemoryStore>>,
    in_flight: Arc<tokio::sync::Semaphore>,
}

impl Pipeline {
    pub fn new(
        config: Arc<Config>,
        llm: Arc<crate::llm::LlmChain>,
        writer: Arc<dyn OutputWriter>,
        tracker: Arc<ProcessingTracker>,
        memory_store: Option<Arc<crate::memory::MemoryStore>>,
    ) -> Self {
        let fetcher = UrlFetcher::new(&config.url_fetch);
        let in_flight_limit =
            std::thread::available_parallelism().map_or(8, std::num::NonZeroUsize::get) * 4;
        Self {
            config,
            llm,
            writer,
            fetcher,
            tracker,
            memory_store,
            in_flight: Arc::new(tokio::sync::Semaphore::new(in_flight_limit)),
        }
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

        let hints = preprocess::run_preprocessing(&msg, &self.config.pipeline.preprocessing);
        if hints.force_web_search || !hints.suggested_tags.is_empty() {
            info!(id = %id, force_web_search = hints.force_web_search,
                suggested_tags = ?hints.suggested_tags, "Pre-processing hints computed");
        }
        msg.preprocessing_hints = hints;

        self.tracker.insert(
            id,
            msg.source.as_str().to_owned(),
            msg.text.chars().take(80).collect(),
        );

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
            .run_stage(id, &mut notifier, llm_initial, self.run_llm(enriched))
            .await?;

        self.run_stage(
            id,
            &mut notifier,
            ProcessingStage::Writing,
            self.writer.write(&processed, &self.config),
        )
        .await?;

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
        let done = ProcessingStage::Done { title };
        self.tracker.advance(id, done.clone());
        if let Some(n) = &mut notifier {
            n.advance(done).await;
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

    #[instrument(skip(self, enriched), fields(
        id = %enriched.original.id,
        url_count = enriched.urls.len(),
        content_count = enriched.url_contents.len(),
    ))]
    async fn run_llm(&self, enriched: EnrichedMessage) -> Result<ProcessedMessage, InboxError> {
        use crate::llm::{LlmOutcome, LlmRequest, LlmTurnProgress};

        let text_preview: String = enriched.original.text.chars().take(120).collect();
        info!(
            id = %enriched.original.id,
            attachment_count = enriched.original.attachments.len(),
            text_preview = %text_preview,
            "Starting LLM processing"
        );

        let preloaded_text = self.preload_memory_context(&enriched).await;

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

        let (outcome, ()) = tokio::join!(self.llm.complete(req), progress_future);

        match outcome {
            LlmOutcome::Success(resp) => {
                info!(
                    id = %enriched.original.id,
                    title = %resp.title,
                    tags = ?resp.tags,
                    backend = %resp.produced_by,
                    "LLM processing succeeded"
                );
                Ok(ProcessedMessage {
                    enriched,
                    llm_response: Some(resp),
                    fallback_source_urls: vec![],
                    fallback_tool_results: vec![],
                    fallback_title: None,
                })
            }
            LlmOutcome::RawFallback {
                source_urls,
                tool_results,
            } => {
                let text_preview: String = enriched.original.text.chars().take(120).collect();
                info!(
                    id = %enriched.original.id,
                    text_preview = %text_preview,
                    "LLM unavailable, using raw fallback"
                );
                let fallback_title =
                    if enriched.original.text.is_empty() && !tool_results.is_empty() {
                        let context = tool_results
                            .iter()
                            .map(|(_, t)| t.as_str())
                            .collect::<Vec<_>>()
                            .join("\n\n");
                        self.llm
                            .complete_text(
                                "Generate a concise 5-word title for this content. \
                                 Reply with only the title, no punctuation.",
                                &context,
                            )
                            .await
                    } else {
                        None
                    };
                Ok(ProcessedMessage {
                    enriched,
                    llm_response: None,
                    fallback_source_urls: source_urls,
                    fallback_tool_results: tool_results,
                    fallback_title,
                })
            }
            LlmOutcome::Discard => {
                info!(id = %enriched.original.id, "Message discarded by LLM fallback policy");
                Err(InboxError::Pipeline(
                    "Message discarded by LLM fallback policy".into(),
                ))
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

impl Pipeline {
    async fn preload_memory_context(&self, enriched: &EnrichedMessage) -> String {
        let Some(ref store) = self.memory_store else {
            return String::new();
        };

        let ctx = context_preload::preload_context(
            store,
            &self.config.memory,
            &enriched.original.text,
            &enriched.urls,
            &enriched.original.user_tags,
        )
        .await;

        if !ctx.memories.is_empty() {
            let keys: Vec<String> = ctx.memories.iter().map(|m| m.key.clone()).collect();
            let source = enriched.original.source_name();
            let _ = store
                .log_recall_event(&enriched.original.id.to_string(), &keys, source)
                .await;
        }

        context_preload::format_preloaded_context(&ctx)
    }

    fn build_llm_guidance(&self, enriched: &EnrichedMessage, preloaded_context: &str) -> String {
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
