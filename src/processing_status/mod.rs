use std::sync::Arc;

use anodized::spec;
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use uuid::Uuid;

/// How long to retain Done/Failed entries in the tracker before pruning.
pub(super) const DONE_RETAIN_SECS: i64 = 300;

// ── Stage ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "snake_case", tag = "stage")]
pub enum ProcessingStage {
    Received,
    Enriching,
    RunningLlm {
        turn: usize,
        max_turns: usize,
        last_tools: Vec<String>,
    },
    Writing,
    Done {
        title: String,
    },
    Failed {
        reason: String,
    },
}

// ── In-flight entry ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, serde::Serialize)]
pub struct InFlightEntry {
    pub id: Uuid,
    pub source: String,
    pub text_preview: String,
    pub started_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    #[serde(flatten)]
    pub stage: ProcessingStage,
}

// ── Tracker ───────────────────────────────────────────────────────────────────

pub struct ProcessingTracker {
    pub(super) entries: Arc<DashMap<Uuid, InFlightEntry>>,
}

impl ProcessingTracker {
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Arc::new(DashMap::new()),
        }
    }

    #[spec(requires: !source.is_empty())]
    pub fn insert(&self, id: Uuid, source: String, text_preview: String) {
        let entry = InFlightEntry {
            id,
            source,
            text_preview,
            started_at: Utc::now(),
            updated_at: Utc::now(),
            stage: ProcessingStage::Received,
        };
        self.entries.insert(id, entry);
        self.update_gauge();
    }

    pub fn advance(&self, id: Uuid, stage: ProcessingStage) {
        if let Some(mut entry) = self.entries.get_mut(&id) {
            entry.stage = stage;
            entry.updated_at = Utc::now();
        }
        self.update_gauge();
    }

    /// Return all in-flight entries, pruning Done/Failed older than `DONE_RETAIN_SECS`.
    #[must_use]
    pub fn snapshot(&self) -> Vec<InFlightEntry> {
        let cutoff = Utc::now() - chrono::Duration::seconds(DONE_RETAIN_SECS);
        self.entries.retain(|_, entry| match &entry.stage {
            ProcessingStage::Done { .. } | ProcessingStage::Failed { .. } => {
                entry.updated_at > cutoff
            }
            _ => true,
        });
        self.update_gauge();
        let mut result: Vec<InFlightEntry> =
            self.entries.iter().map(|e| e.value().clone()).collect();
        result.sort_by_key(|e| std::cmp::Reverse(e.started_at));
        result
    }

    fn update_gauge(&self) {
        let count = u32::try_from(self.entries.len()).unwrap_or(u32::MAX);
        metrics::gauge!(crate::telemetry::QUEUE_DEPTH).set(f64::from(count));
    }
}

impl Default for ProcessingTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ── StatusNotifier trait ──────────────────────────────────────────────────────

#[async_trait::async_trait]
pub trait StatusNotifier: Send + Sync {
    async fn advance(&mut self, stage: ProcessingStage);

    /// Returns the Telegram message ID of the status message, if this is a
    /// Telegram notifier. Used by the pending store to enable resume notifications.
    fn telegram_status_msg_id(&self) -> Option<i32> {
        None
    }
}

// ── NoopNotifier ──────────────────────────────────────────────────────────────

pub struct NoopNotifier;

#[async_trait::async_trait]
impl StatusNotifier for NoopNotifier {
    async fn advance(&mut self, _stage: ProcessingStage) {}
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests;
