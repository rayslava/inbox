use std::sync::Arc;

use anodized::contract;
use tokio::sync::mpsc;
use tracing::{debug, error, info, instrument, warn};

use crate::config::Config;
use crate::error::InboxError;
use crate::message::{EnrichedMessage, IncomingMessage, ProcessedMessage};
use crate::output::OutputWriter;

pub mod content_extractor;
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
    in_flight: Arc<tokio::sync::Semaphore>,
}

impl Pipeline {
    pub fn new(
        config: Arc<Config>,
        llm: Arc<crate::llm::LlmChain>,
        writer: Arc<dyn OutputWriter>,
    ) -> Self {
        let fetcher = UrlFetcher::new(&config.url_fetch);
        let in_flight_limit =
            std::thread::available_parallelism().map_or(8, std::num::NonZeroUsize::get) * 4;
        Self {
            config,
            llm,
            writer,
            fetcher,
            in_flight: Arc::new(tokio::sync::Semaphore::new(in_flight_limit)),
        }
    }

    #[contract(requires: true)]
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
    #[contract(requires: !msg.id.is_nil())]
    #[instrument(skip(self, msg), fields(id = %msg.id, source = %msg.source))]
    pub async fn process(&self, msg: IncomingMessage) -> Result<(), InboxError> {
        let enriched = self.enrich(msg).await?;
        let processed = self.run_llm(enriched).await?;
        self.writer.write(&processed, &self.config).await?;
        Ok(())
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
            let host = url.host_str().unwrap_or("");
            if self
                .config
                .url_fetch
                .skip_domains
                .iter()
                .any(|d| host.ends_with(d.as_str()))
            {
                debug!(%url, "Skipping URL — domain is in skip list");
                continue;
            }

            match classify_url(url, &self.fetcher).await {
                UrlKind::Page => {
                    if let Some(content) = self.fetcher.fetch_page(url).await {
                        debug!(
                            %url,
                            text_len = content.text.len(),
                            title = ?content.page_title,
                            "Page content fetched"
                        );
                        let truncated =
                            truncate_chars(&content.text, self.config.llm.url_content_max_chars);
                        url_contents.push(crate::url_content::UrlContent {
                            url: url.to_string(),
                            text: truncated,
                            page_title: content.page_title,
                        });
                    } else {
                        warn!(%url, "Failed to fetch page content");
                    }
                }
                UrlKind::File { ref mime } => {
                    if let Some(att) = self
                        .fetcher
                        .download_file(url, msg.id, &self.config.general.attachments_dir)
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
                        let truncated =
                            truncate_chars(&content.text, self.config.llm.url_content_max_chars);
                        url_contents.push(crate::url_content::UrlContent {
                            url: url.to_string(),
                            text: truncated,
                            page_title: content.page_title,
                        });
                    }
                }
            }
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

    #[instrument(skip(self, enriched), fields(
        id = %enriched.original.id,
        url_count = enriched.urls.len(),
        content_count = enriched.url_contents.len(),
    ))]
    async fn run_llm(&self, enriched: EnrichedMessage) -> Result<ProcessedMessage, InboxError> {
        use crate::llm::{LlmOutcome, LlmRequest};

        debug!(
            id = %enriched.original.id,
            attachment_count = enriched.original.attachments.len(),
            "Starting LLM processing"
        );

        let req = LlmRequest::from_enriched(
            &enriched,
            &self.config.llm,
            &self.config.general.attachments_dir,
            &self.config.tooling.prompt_block(),
        );
        match self.llm.complete(req).await {
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
                })
            }
            LlmOutcome::RawFallback => {
                info!(id = %enriched.original.id, "LLM unavailable, using raw fallback");
                Ok(ProcessedMessage {
                    enriched,
                    llm_response: None,
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

fn truncate_chars(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_owned()
    } else {
        s.chars().take(max).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_chars_within_limit() {
        assert_eq!(truncate_chars("hello", 10), "hello");
    }

    #[test]
    fn truncate_chars_at_limit() {
        assert_eq!(truncate_chars("hello", 5), "hello");
    }

    #[test]
    fn truncate_chars_exceeds_limit() {
        assert_eq!(truncate_chars("hello world", 5), "hello");
    }

    #[test]
    fn truncate_chars_unicode() {
        // "héllo" — 5 chars, each may be multi-byte
        let s = "héllo";
        assert_eq!(truncate_chars(s, 3), "hél");
    }
}
