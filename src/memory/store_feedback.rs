use std::sync::Arc;

use crate::error::InboxError;

use super::feedback;
use super::{MemoryStore, RecallOutcome, RelatedMemory};

impl MemoryStore {
    // ── Feedback methods ─────────────────────────────────────────────────────

    /// Save (upsert) a feedback entry and link it to the source message.
    ///
    /// # Errors
    /// Returns an error if the database write fails.
    pub async fn save_feedback(
        &self,
        entry: &crate::feedback::FeedbackEntry,
    ) -> Result<(), InboxError> {
        let start = std::time::Instant::now();
        let message_id = entry.message_id.clone();
        let rating = entry.rating;
        let comment = entry.comment.clone();
        let created_at = entry.created_at.timestamp();
        let source = entry.source.clone();
        let title = entry.title.clone();
        let db = Arc::clone(&self.db);

        let rating_str = rating.to_string();
        let source_label = source.clone();

        let result = tokio::task::spawn_blocking(move || {
            feedback::insert_feedback(
                &db,
                &message_id,
                rating,
                &comment,
                created_at,
                &source,
                &title,
            )
        })
        .await
        .map_err(|e| InboxError::Memory(e.to_string()))?;

        let status = if result.is_ok() { "success" } else { "failure" };
        metrics::counter!(crate::telemetry::FEEDBACK_TOTAL, "rating" => rating_str.clone(), "source" => source_label, "status" => status)
            .increment(1);
        metrics::histogram!(crate::telemetry::FEEDBACK_DURATION, "op" => "save")
            .record(start.elapsed().as_secs_f64());
        if result.is_ok() {
            metrics::gauge!(crate::telemetry::FEEDBACK_RATING_DISTRIBUTION, "rating" => rating_str)
                .increment(1.0);
        }
        result
    }

    /// Query feedback for a specific message.
    ///
    /// # Errors
    /// Returns an error if the database query fails.
    pub async fn query_feedback(
        &self,
        message_id: &str,
    ) -> Result<Option<crate::feedback::FeedbackEntry>, InboxError> {
        let start = std::time::Instant::now();
        let mid = message_id.to_owned();
        let db = Arc::clone(&self.db);

        let result = tokio::task::spawn_blocking(move || feedback::get_feedback(&db, &mid))
            .await
            .map_err(|e| InboxError::Memory(e.to_string()))?;

        let status = if result.is_ok() { "success" } else { "failure" };
        metrics::counter!(crate::telemetry::MEMORY_OPS, "op" => "feedback_query", "status" => status)
            .increment(1);
        metrics::histogram!(crate::telemetry::FEEDBACK_DURATION, "op" => "query")
            .record(start.elapsed().as_secs_f64());
        result
    }

    /// Compute aggregate feedback statistics.
    ///
    /// # Errors
    /// Returns an error if the database query fails.
    pub async fn feedback_stats(&self) -> Result<crate::feedback::FeedbackStats, InboxError> {
        let start = std::time::Instant::now();
        let db = Arc::clone(&self.db);

        let result = tokio::task::spawn_blocking(move || feedback::get_feedback_stats(&db))
            .await
            .map_err(|e| InboxError::Memory(e.to_string()))?;

        let status = if result.is_ok() { "success" } else { "failure" };
        metrics::counter!(crate::telemetry::MEMORY_OPS, "op" => "feedback_stats", "status" => status)
            .increment(1);
        metrics::histogram!(crate::telemetry::FEEDBACK_DURATION, "op" => "stats")
            .record(start.elapsed().as_secs_f64());
        result
    }

    /// Update the comment on an existing feedback entry.
    /// Returns `true` if the feedback existed and was updated.
    ///
    /// # Errors
    /// Returns an error if the database write fails.
    pub async fn update_feedback_comment(
        &self,
        message_id: &str,
        comment: &str,
    ) -> Result<bool, InboxError> {
        let start = std::time::Instant::now();
        let mid = message_id.to_owned();
        let cmt = comment.to_owned();
        let db = Arc::clone(&self.db);

        let result =
            tokio::task::spawn_blocking(move || feedback::update_feedback_comment(&db, &mid, &cmt))
                .await
                .map_err(|e| InboxError::Memory(e.to_string()))?;

        let status = if result.is_ok() { "success" } else { "failure" };
        metrics::counter!(crate::telemetry::FEEDBACK_COMMENTS_TOTAL, "source" => "direct", "status" => status)
            .increment(1);
        metrics::histogram!(crate::telemetry::FEEDBACK_DURATION, "op" => "update_comment")
            .record(start.elapsed().as_secs_f64());
        result
    }

    // ── Pre-load methods ─────────────────────────────────────────────────

    /// Find memories related to a given key via graph edges, returning relation types.
    ///
    /// # Errors
    /// Returns an error if the graph query fails.
    pub async fn related_memories(
        &self,
        memory_key: &str,
        hops: u32,
    ) -> Result<Vec<RelatedMemory>, InboxError> {
        let start = std::time::Instant::now();
        let key = memory_key.to_owned();
        let db = Arc::clone(&self.db);

        let result = tokio::task::spawn_blocking(move || {
            super::queries::graph_related_memories(&db, &key, hops)
        })
        .await
        .map_err(|e| InboxError::Memory(e.to_string()))?;

        let status = if result.is_ok() { "success" } else { "failure" };
        metrics::counter!(crate::telemetry::MEMORY_OPS, "op" => "related_memories", "status" => status)
            .increment(1);
        metrics::histogram!(crate::telemetry::MEMORY_DURATION, "op" => "related_memories")
            .record(start.elapsed().as_secs_f64());
        result
    }

    /// Fetch recent feedback entries with rating at or below `max_rating`.
    ///
    /// # Errors
    /// Returns an error if the database query fails.
    pub async fn recent_feedback(
        &self,
        max_rating: u8,
        limit: usize,
    ) -> Result<Vec<crate::feedback::FeedbackEntry>, InboxError> {
        let start = std::time::Instant::now();
        let db = Arc::clone(&self.db);

        let result = tokio::task::spawn_blocking(move || {
            feedback::get_recent_feedback(&db, max_rating, limit)
        })
        .await
        .map_err(|e| InboxError::Memory(e.to_string()))?;

        let status = if result.is_ok() { "success" } else { "failure" };
        metrics::counter!(crate::telemetry::MEMORY_OPS, "op" => "recent_feedback", "status" => status)
            .increment(1);
        metrics::histogram!(crate::telemetry::FEEDBACK_DURATION, "op" => "recent")
            .record(start.elapsed().as_secs_f64());
        result
    }

    /// Log which memories were recalled for a given message, creating a `:RecallEvent` node.
    ///
    /// # Errors
    /// Returns an error if the database write fails.
    pub async fn log_recall_event(
        &self,
        message_id: &str,
        recalled_keys: &[String],
        source_name: &str,
    ) -> Result<(), InboxError> {
        let start = std::time::Instant::now();
        let mid = message_id.to_owned();
        let keys = recalled_keys.to_vec();
        let src = source_name.to_owned();
        let db = Arc::clone(&self.db);

        let result = tokio::task::spawn_blocking(move || {
            super::queries::insert_recall_event(&db, &mid, &keys, &src)
        })
        .await
        .map_err(|e| InboxError::Memory(e.to_string()))?;

        let status = if result.is_ok() { "success" } else { "failure" };
        metrics::counter!(crate::telemetry::MEMORY_OPS, "op" => "log_recall", "status" => status)
            .increment(1);
        metrics::histogram!(crate::telemetry::MEMORY_DURATION, "op" => "log_recall")
            .record(start.elapsed().as_secs_f64());
        result
    }

    /// Find historical recall outcomes for a set of memory keys by correlating
    /// recall events with feedback.
    pub async fn recall_outcomes(&self, memory_keys: &[String]) -> Vec<RecallOutcome> {
        let start = std::time::Instant::now();
        let keys = memory_keys.to_vec();
        let db = Arc::clone(&self.db);

        let result =
            tokio::task::spawn_blocking(move || super::queries::query_recall_outcomes(&db, &keys))
                .await
                .unwrap_or_default();

        metrics::counter!(crate::telemetry::MEMORY_OPS, "op" => "recall_outcomes", "status" => "success")
            .increment(1);
        metrics::histogram!(crate::telemetry::MEMORY_DURATION, "op" => "recall_outcomes")
            .record(start.elapsed().as_secs_f64());
        result
    }
}
