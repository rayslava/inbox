use std::fmt::Write as _;
use std::sync::Arc;

use tracing::warn;

use crate::config::MemoryConfig;
use crate::feedback::FeedbackEntry;
use crate::memory::{MemoryEntry, MemoryStore, RecallOutcome, RelatedMemory};

#[cfg(test)]
mod tests;

// ── Types ────────────────────────────────────────────────────────────────────

/// A recalled memory with its similarity score, graph neighbors, and historical outcomes.
#[derive(Debug)]
pub struct RecalledMemory {
    pub key: String,
    pub value: String,
    pub score: f64,
    pub related: Vec<RelatedMemory>,
    pub outcome: Option<RecallOutcome>,
}

/// Pre-loaded context ready for injection into the LLM guidance block.
#[derive(Debug, Default)]
pub struct PreloadedContext {
    pub memories: Vec<RecalledMemory>,
    pub feedback: Vec<FeedbackEntry>,
    pub recall_quality: RecallQuality,
}

/// Qualitative assessment of how well the recall matched the message.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum RecallQuality {
    Strong,
    Weak,
    #[default]
    Empty,
}

// ── Pre-load logic ───────────────────────────────────────────────────────────

/// Build a recall query from message text, URLs, and user tags.
fn build_recall_query(text: &str, urls: &[url::Url], tags: &[String]) -> String {
    let mut query = text.chars().take(200).collect::<String>();

    for url in urls {
        if let Some(host) = url.host_str() {
            let _ = write!(query, " {host}");
        }
    }

    for tag in tags {
        let _ = write!(query, " {tag}");
    }

    query
}

/// Classify recall quality based on similarity scores.
fn classify_quality(entries: &[MemoryEntry]) -> RecallQuality {
    if entries.is_empty() {
        return RecallQuality::Empty;
    }
    if entries.iter().any(|e| e.score > 0.7) {
        RecallQuality::Strong
    } else if entries.iter().all(|e| e.score < 0.4) {
        RecallQuality::Weak
    } else {
        RecallQuality::Strong
    }
}

const MAX_RELATED_PER_MEMORY: usize = 3;

/// Fetch memories and feedback relevant to the incoming message.
pub async fn preload_context(
    store: &Arc<MemoryStore>,
    config: &MemoryConfig,
    text: &str,
    urls: &[url::Url],
    tags: &[String],
) -> PreloadedContext {
    let query = build_recall_query(text, urls, tags);
    if query.trim().is_empty() {
        return PreloadedContext::default();
    }

    // Recall memories.
    let entries = match store.recall(&query, config.preload_max_memories).await {
        Ok(e) => e,
        Err(e) => {
            warn!("Memory preload recall failed: {e}");
            Vec::new()
        }
    };

    let recall_quality = classify_quality(&entries);

    // Fetch graph relations and recall outcomes for each memory.
    let keys: Vec<String> = entries.iter().map(|e| e.key.clone()).collect();
    let outcomes = store.recall_outcomes(&keys).await;

    let mut memories = Vec::with_capacity(entries.len());
    for entry in entries {
        let related = match store
            .related_memories(&entry.key, config.preload_graph_hops)
            .await
        {
            Ok(mut r) => {
                r.truncate(MAX_RELATED_PER_MEMORY);
                r
            }
            Err(_) => Vec::new(),
        };

        let outcome = outcomes.iter().find(|o| o.memory_key == entry.key).cloned();

        memories.push(RecalledMemory {
            key: entry.key,
            value: entry.value,
            score: entry.score,
            related,
            outcome,
        });
    }

    // Fetch recent low-rated feedback.
    let feedback = if config.preload_feedback {
        store
            .recent_feedback(
                config.preload_feedback_max_rating,
                config.preload_max_feedback,
            )
            .await
            .unwrap_or_default()
    } else {
        Vec::new()
    };

    PreloadedContext {
        memories,
        feedback,
        recall_quality,
    }
}

// ── Formatting ───────────────────────────────────────────────────────────────

/// Format pre-loaded context into a guidance string for the LLM system prompt.
#[must_use]
pub fn format_preloaded_context(ctx: &PreloadedContext) -> String {
    if ctx.memories.is_empty() && ctx.feedback.is_empty() {
        if ctx.recall_quality == RecallQuality::Empty {
            return "--- Memory context: no relevant memories found ---\n\
                    No existing memories match this message. If this content contains notable facts, \
                    preferences, or patterns, use memory_save to persist them for future reference."
                .to_owned();
        }
        return String::new();
    }

    let mut out = String::new();

    if !ctx.memories.is_empty() {
        let quality_label = match ctx.recall_quality {
            RecallQuality::Strong => "strong",
            RecallQuality::Weak => "weak",
            RecallQuality::Empty => "none",
        };
        let _ = writeln!(
            out,
            "--- Memory context (recall quality: {quality_label}, {} matches) ---",
            ctx.memories.len()
        );

        for mem in &ctx.memories {
            let _ = write!(out, "[{:.2}] {}: {}", mem.score, mem.key, mem.value);
            if let Some(ref outcome) = mem.outcome {
                let _ = write!(
                    out,
                    " (used {} times, avg rating {:.1}{})",
                    outcome.times_recalled,
                    outcome.avg_rating,
                    if outcome.avg_rating < 2.0 { " ⚠" } else { "" }
                );
            }
            out.push('\n');

            for rel in &mem.related {
                let arrow = if rel.direction == "outgoing" {
                    "→"
                } else {
                    "←"
                };
                let label = if rel.relation.is_empty() {
                    "CONNECTED".to_owned()
                } else {
                    rel.relation.clone()
                };
                let _ = writeln!(out, "  {arrow} {label} {arrow} {}: {}", rel.key, rel.value);
            }

            if let Some(ref outcome) = mem.outcome {
                for comment in &outcome.sample_comments {
                    let _ = writeln!(out, "  Feedback: \"{comment}\"");
                }
            }
        }

        out.push_str(
            "Use memory_save to update outdated memories. \
             Use memory_link to connect new insights to existing knowledge.\n",
        );
    }

    if !ctx.feedback.is_empty() {
        let _ = writeln!(
            out,
            "\n--- User feedback on previous outputs (last {} low-rated) ---",
            ctx.feedback.len()
        );
        for fb in &ctx.feedback {
            let comment_part = if fb.comment.is_empty() {
                String::new()
            } else {
                format!(": \"{}\"", fb.comment)
            };
            let _ = writeln!(
                out,
                "- \"{}\" rated {}/3{}",
                fb.title, fb.rating, comment_part
            );
        }
        out.push_str(
            "Avoid patterns that received low ratings. Apply lessons from this feedback.\n",
        );
    }

    out
}
