//! Background task that retries messages that fell back to raw mode.
//!
//! Runs on a configurable interval and only processes items when the pipeline
//! is idle (no in-flight messages). On success, patches the org-mode output
//! file in-place and optionally notifies the original Telegram chat.

use std::sync::Arc;
use std::time::Duration;

use metrics::{counter, gauge};
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};

use crate::adapters::telegram_notifier::resume::TelegramResumeNotifier;
use crate::config::Config;
use crate::message::{EnrichedMessage, IncomingMessage, MessageSource};
use crate::output::org_patcher;
use crate::pending::{PendingItem, PendingStore};
use crate::pipeline::Pipeline;
use crate::render::{FAILED_TAG, PENDING_TAG, render_org_node};
use crate::telemetry;
use anodized::spec;

/// Maximum number of items to retry per scan cycle.
const BATCH_SIZE: u32 = 3;

/// Arguments for the background resume task.
pub struct ResumeTaskArgs {
    pub store: Arc<PendingStore>,
    pub pipeline: Arc<Pipeline>,
    pub config: Arc<Config>,
    pub telegram_notifier: Option<Arc<TelegramResumeNotifier>>,
    pub shutdown: CancellationToken,
}

/// Run the background resume loop.
///
/// Wakes every `interval_secs`, checks that the pipeline is idle, then
/// processes up to [`BATCH_SIZE`] pending items. Exits cleanly when
/// `shutdown` is cancelled.
#[spec(requires: args.config.pipeline.resume.interval_secs > 0)]
pub async fn run(args: ResumeTaskArgs) {
    let interval = Duration::from_secs(args.config.pipeline.resume.interval_secs);
    let max_retries = args.config.pipeline.resume.max_retries;

    loop {
        tokio::select! {
            () = tokio::time::sleep(interval) => {}
            () = args.shutdown.cancelled() => {
                info!("Resume task shutting down");
                break;
            }
        }

        // Only scan when the pipeline is idle (no permits consumed).
        // Try to acquire one permit non-blockingly; if it fails the pipeline is busy.
        let idle_permit = args.pipeline.in_flight.try_acquire();
        if idle_permit.is_err() {
            debug!("Pipeline busy, skipping resume scan");
            continue;
        }
        // Release the permit immediately — we just used it to probe idleness.
        drop(idle_permit);

        process_pending_batch(&args, max_retries).await;
    }
}

async fn process_pending_batch(args: &ResumeTaskArgs, max_retries: u32) {
    let items = match args.store.list(max_retries, BATCH_SIZE).await {
        Ok(v) => v,
        Err(e) => {
            warn!(?e, "Failed to list pending items");
            return;
        }
    };

    if items.is_empty() {
        debug!("No pending items to resume");
    } else {
        info!(count = items.len(), "Processing pending items batch");
    }

    for item in &items {
        retry_item(args, item, max_retries).await;
    }

    // Update metrics after the batch.
    update_pending_metrics(&args.store, max_retries).await;
}

async fn retry_item(args: &ResumeTaskArgs, item: &PendingItem, max_retries: u32) {
    let id = item.id;
    info!(%id, retry_count = item.retry_count, source = %item.source, "Retrying pending item");

    let enriched = build_enriched(item);

    match args.pipeline.run_llm(enriched).await {
        Ok(processed) if processed.llm_response.is_some() => {
            on_success(args, item, processed).await;
        }
        Ok(_) => {
            // Still falling back — increment retry.
            warn!(%id, "LLM still falling back on retry");
            if let Err(e) = args.store.increment_retry(id).await {
                warn!(?e, %id, "Failed to increment retry count");
            }
            let remaining = max_retries.saturating_sub(item.retry_count + 1);
            if remaining == 0 {
                warn!(%id, "All retries exhausted for pending item — marking as failed");
                on_exhausted(args, item).await;
            } else {
                info!(%id, remaining_retries = remaining, "Retry did not improve result");
            }
            counter!(telemetry::RESUME_ATTEMPTS, "status" => "failure").increment(1);
        }
        Err(e) => {
            warn!(?e, %id, "LLM error during retry");
            if let Err(e2) = args.store.increment_retry(id).await {
                warn!(?e2, %id, "Failed to increment retry count");
            }
            let remaining = max_retries.saturating_sub(item.retry_count + 1);
            if remaining == 0 {
                warn!(%id, "All retries exhausted (LLM error) — marking as failed");
                on_exhausted(args, item).await;
            }
            counter!(telemetry::RESUME_ATTEMPTS, "status" => "failure").increment(1);
        }
    }
}

async fn on_success(
    args: &ResumeTaskArgs,
    item: &PendingItem,
    processed: crate::message::ProcessedMessage,
) {
    let id = item.id;
    let title = processed
        .llm_response
        .as_ref()
        .map_or("", |r| r.title.as_str())
        .to_owned();

    let new_text = match render_org_node(&processed, &args.config.general.attachments_dir) {
        Ok(t) => t,
        Err(e) => {
            warn!(?e, %id, "Failed to render org node for resume patch");
            counter!(telemetry::RESUME_ATTEMPTS, "status" => "failure").increment(1);
            return;
        }
    };

    let output_path = &args.config.general.output_file;
    match org_patcher::patch_entry(output_path, id, &new_text).await {
        Ok(true) => {
            info!(%id, %title, "Successfully patched org entry");
        }
        Ok(false) => {
            warn!(%id, "Org entry not found in output file — removing from pending anyway");
            counter!(telemetry::RESUME_ATTEMPTS, "status" => "not_found").increment(1);
        }
        Err(e) => {
            warn!(?e, %id, "Failed to patch org file");
            counter!(telemetry::RESUME_ATTEMPTS, "status" => "failure").increment(1);
            return;
        }
    }

    // Remove from pending store.
    if let Err(e) = args.store.remove(id).await {
        warn!(?e, %id, "Failed to remove pending item after successful retry");
    }

    counter!(telemetry::RESUME_ATTEMPTS, "status" => "success").increment(1);

    // Notify Telegram if applicable.
    if item.source == "telegram" {
        if let Some(ref notifier) = args.telegram_notifier {
            if let Err(e) = notifier.notify_done(item, &title, id).await {
                warn!(?e, %id, "Failed to send Telegram resume notification");
            }
        }
    }

    // Trigger Syncthing rescan if configured.
    if args.config.syncthing.enabled && args.config.syncthing.rescan_on_write {
        crate::output::org_file::trigger_syncthing_rescans(&args.config.syncthing).await;
    }
}

/// Called when all retries for an item are exhausted.
///
/// Replaces `:inbox_pending:` with `:inbox_failed:` in the org file (in-place
/// text substitution, not a full re-render) and notifies Telegram if applicable.
async fn on_exhausted(args: &ResumeTaskArgs, item: &PendingItem) {
    let id = item.id;
    let output_path = &args.config.general.output_file;

    // Read, patch the tag, write back atomically.
    match tokio::fs::read_to_string(output_path).await {
        Ok(text) => {
            // Replace only the first occurrence associated with this entry's
            // heading — a simple string replace across the whole file is safe
            // because each UUID is unique, but the tag appears on the headline,
            // not the ID line, so swap globally (there is only one per ID).
            let needle = format!(":{PENDING_TAG}:");
            let replacement = format!(":{FAILED_TAG}:");
            if text.contains(&needle) {
                let patched = text.replacen(&needle, &replacement, 1);
                let tmp = output_path.with_extension("org.tmp");
                if let Err(e) = tokio::fs::write(&tmp, &patched).await {
                    warn!(?e, %id, "Failed to write tmp org file for exhausted patch");
                } else if let Err(e) = tokio::fs::rename(&tmp, output_path).await {
                    warn!(?e, %id, "Failed to rename tmp org file for exhausted patch");
                } else {
                    info!(%id, "Patched org entry tag to :inbox_failed:");
                    if args.config.syncthing.enabled && args.config.syncthing.rescan_on_write {
                        crate::output::org_file::trigger_syncthing_rescans(&args.config.syncthing)
                            .await;
                    }
                }
            }
        }
        Err(e) => warn!(?e, %id, "Could not read org file to patch exhausted tag"),
    }

    // Telegram notification.
    if item.source == "telegram" {
        if let Some(ref notifier) = args.telegram_notifier {
            let title = item
                .fallback_title
                .as_deref()
                .unwrap_or("(unknown)")
                .to_owned();
            if let Err(e) = notifier.notify_done(item, &title, id).await {
                warn!(?e, %id, "Failed to send Telegram exhausted notification");
            }
        }
    }

    counter!(telemetry::RESUME_ATTEMPTS, "status" => "exhausted").increment(1);
}

/// Reconstruct an [`EnrichedMessage`] from a stored [`PendingItem`].
///
/// The enriched URL contents and fallback tool results are pre-loaded so
/// the LLM stage can use them without re-fetching.
pub(crate) fn build_enriched(item: &PendingItem) -> EnrichedMessage {
    let source = match item.source.as_str() {
        "telegram" => MessageSource::Telegram,
        "email" => MessageSource::Email,
        _ => MessageSource::Http,
    };

    let mut incoming = IncomingMessage::with_id(
        item.id,
        source,
        item.incoming.text.clone(),
        item.incoming.metadata.clone(),
    );
    incoming.attachments.clone_from(&item.incoming.attachments);
    incoming.user_tags.clone_from(&item.incoming.user_tags);
    incoming
        .preprocessing_hints
        .clone_from(&item.incoming.preprocessing_hints);
    incoming.received_at = item.incoming.received_at;

    // Re-parse URLs from stored url_contents so we don't re-fetch.
    let urls: Vec<url::Url> = item
        .url_contents
        .iter()
        .filter_map(|uc| uc.url.parse().ok())
        .collect();

    EnrichedMessage {
        original: incoming,
        urls,
        url_contents: item.url_contents.clone(),
    }
}

async fn update_pending_metrics(store: &PendingStore, max_retries: u32) {
    match store.stats(max_retries).await {
        Ok(s) => {
            gauge!(telemetry::PENDING_ITEMS).set(f64::from(s.total_items));
            gauge!(telemetry::PENDING_EXHAUSTED).set(f64::from(s.exhausted_items));
            gauge!(telemetry::PENDING_DB_BYTES).set(u64_to_gauge_f64(s.db_bytes()));
            gauge!(telemetry::PENDING_DB_FREELIST_PAGES).set(u64_to_gauge_f64(s.db_freelist_count));
        }
        Err(e) => {
            warn!(?e, "Failed to collect pending store stats");
        }
    }
}

/// Convert a `u64` metric into an `f64` gauge value without triggering
/// `clippy::cast_precision_loss`.
///
/// Splits the value into two `u32` halves (each losslessly representable
/// as `f64`) and recombines them. Values above `2^53` round to the nearest
/// representable `f64`, which is acceptable for a monitoring gauge.
fn u64_to_gauge_f64(v: u64) -> f64 {
    let hi = u32::try_from(v >> 32).unwrap_or(u32::MAX);
    let lo = u32::try_from(v & 0xFFFF_FFFF).unwrap_or(u32::MAX);
    f64::from(hi).mul_add(4_294_967_296.0, f64::from(lo))
}
