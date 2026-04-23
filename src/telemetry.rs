// Metric name constants — used with the `metrics` facade macros throughout the codebase.
// Register descriptions once at startup via `describe_metrics()`.

use metrics::{Unit, describe_counter, describe_gauge, describe_histogram};

// ── Counters ─────────────────────────────────────────────────────────────────

/// Labels: source = "telegram" | "http" | "email"
pub const MESSAGES_RECEIVED: &str = "inbox_messages_received_total";

/// Labels: source, status = "success" | "failure"
pub const MESSAGES_PROCESSED: &str = "inbox_messages_processed_total";

/// Labels: backend = "openrouter" | "ollama" | "mock", status = "success" | "failure"
pub const LLM_REQUESTS: &str = "inbox_llm_requests_total";

/// Labels: tool, status = "success" | "failure"
pub const TOOL_CALLS: &str = "inbox_tool_calls_total";

/// Labels: status = "success" | "failure"
pub const URL_FETCHES: &str = "inbox_url_fetches_total";

/// Labels: status = "success" | "failure"
pub const WRITES_TOTAL: &str = "inbox_writes_total";

pub const WRITE_ERRORS: &str = "inbox_write_errors_total";

/// Labels: adapter = "telegram" | "email"
pub const ADAPTER_RECONNECTS: &str = "inbox_adapter_reconnects_total";

/// Labels: op = "save" | "recall" | "`link_source`" | "`link_memories`" | "context" | "sources",
///         status = "success" | "failure"
pub const MEMORY_OPS: &str = "inbox_memory_ops_total";

// ── Histograms ────────────────────────────────────────────────────────────────

/// Labels: source
pub const PROCESSING_DURATION: &str = "inbox_processing_duration_seconds";

/// Labels: backend
pub const LLM_DURATION: &str = "inbox_llm_duration_seconds";

/// Labels: tool
pub const TOOL_DURATION: &str = "inbox_tool_duration_seconds";

/// Labels: kind = "page" | "file"
pub const URL_FETCH_DURATION: &str = "inbox_url_fetch_duration_seconds";

/// Labels: (none)
pub const WRITE_DURATION: &str = "inbox_write_duration_seconds";

/// Labels: op
pub const MEMORY_DURATION: &str = "inbox_memory_duration_seconds";

/// Labels: rating = "1" | "2" | "3", source = "telegram" | "`web_ui`" | "http",
///         status = "success" | "failure"
pub const FEEDBACK_TOTAL: &str = "inbox_feedback_total";

/// Labels: source, status = "success" | "failure"
pub const FEEDBACK_COMMENTS_TOTAL: &str = "inbox_feedback_comments_total";

/// Labels: op = "save" | "query" | "stats" | "`update_comment`"
pub const FEEDBACK_DURATION: &str = "inbox_feedback_duration_seconds";

// ── Gauges ────────────────────────────────────────────────────────────────────

pub const QUEUE_DEPTH: &str = "inbox_queue_depth";

/// Labels: rating = "1" | "2" | "3"
pub const FEEDBACK_RATING_DISTRIBUTION: &str = "inbox_feedback_rating";

// ── Pending / resume metrics ──────────────────────────────────────────────────

/// Gauge: current number of items awaiting LLM retry.
pub const PENDING_ITEMS: &str = "inbox_pending_items";

/// Gauge: items that have exhausted `max_retries` and will not be retried.
pub const PENDING_EXHAUSTED: &str = "inbox_pending_exhausted";

/// Gauge: estimated `SQLite` database file size in bytes.
pub const PENDING_DB_BYTES: &str = "inbox_pending_db_bytes";

/// Gauge: number of free (unused) pages in the `SQLite` store (fragmentation indicator).
pub const PENDING_DB_FREELIST_PAGES: &str = "inbox_pending_db_freelist_pages";

/// Counter. Labels: status = "success" | "failure" | "`not_found`"
pub const RESUME_ATTEMPTS: &str = "inbox_resume_attempts_total";

// ─────────────────────────────────────────────────────────────────────────────

/// Call once at startup to register metric descriptions with the recorder.
pub fn describe_metrics() {
    describe_core_metrics();
    describe_memory_metrics();
    describe_feedback_metrics();
    describe_pending_metrics();
}

fn describe_core_metrics() {
    describe_counter!(
        MESSAGES_RECEIVED,
        Unit::Count,
        "Total messages received by all adapters"
    );
    describe_counter!(
        MESSAGES_PROCESSED,
        Unit::Count,
        "Total messages through the pipeline (success or failure)"
    );
    describe_counter!(LLM_REQUESTS, Unit::Count, "Total LLM API requests");
    describe_counter!(TOOL_CALLS, Unit::Count, "Total LLM tool call executions");
    describe_counter!(URL_FETCHES, Unit::Count, "Total URL fetch attempts");
    describe_counter!(
        WRITES_TOTAL,
        Unit::Count,
        "Total output writes (success or failure)"
    );
    describe_counter!(
        WRITE_ERRORS,
        Unit::Count,
        "Total errors writing to the org output file"
    );
    describe_counter!(
        ADAPTER_RECONNECTS,
        Unit::Count,
        "Total adapter reconnection attempts after unexpected disconnects"
    );
    describe_histogram!(
        PROCESSING_DURATION,
        Unit::Seconds,
        "End-to-end pipeline processing time per message"
    );
    describe_histogram!(LLM_DURATION, Unit::Seconds, "LLM request duration");
    describe_histogram!(
        TOOL_DURATION,
        Unit::Seconds,
        "LLM tool call execution duration"
    );
    describe_histogram!(
        URL_FETCH_DURATION,
        Unit::Seconds,
        "URL fetch duration by kind"
    );
    describe_histogram!(WRITE_DURATION, Unit::Seconds, "Output write duration");
    describe_gauge!(
        QUEUE_DEPTH,
        Unit::Count,
        "Current number of messages waiting in the pipeline queue"
    );
}

fn describe_memory_metrics() {
    describe_counter!(MEMORY_OPS, Unit::Count, "Total memory store operations");
    describe_histogram!(
        MEMORY_DURATION,
        Unit::Seconds,
        "Memory store operation duration"
    );
}

fn describe_feedback_metrics() {
    describe_counter!(
        FEEDBACK_TOTAL,
        Unit::Count,
        "Total user feedback submissions by rating and source"
    );
    describe_counter!(
        FEEDBACK_COMMENTS_TOTAL,
        Unit::Count,
        "Total feedback comment additions"
    );
    describe_histogram!(
        FEEDBACK_DURATION,
        Unit::Seconds,
        "Feedback storage operation duration"
    );
    describe_gauge!(
        FEEDBACK_RATING_DISTRIBUTION,
        Unit::Count,
        "Current feedback count per rating level"
    );
}

fn describe_pending_metrics() {
    describe_gauge!(
        PENDING_ITEMS,
        Unit::Count,
        "Pending items awaiting LLM retry"
    );
    describe_gauge!(
        PENDING_EXHAUSTED,
        Unit::Count,
        "Pending items that have exhausted max retries"
    );
    describe_gauge!(
        PENDING_DB_BYTES,
        Unit::Bytes,
        "Estimated pending SQLite database file size in bytes"
    );
    describe_gauge!(
        PENDING_DB_FREELIST_PAGES,
        Unit::Count,
        "Free (unused) pages in the pending SQLite store"
    );
    describe_counter!(
        RESUME_ATTEMPTS,
        Unit::Count,
        "Total resume attempts by status (success, failure, not_found)"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metric_constants_are_nonempty() {
        assert!(!MESSAGES_RECEIVED.is_empty());
        assert!(!MESSAGES_PROCESSED.is_empty());
        assert!(!LLM_REQUESTS.is_empty());
        assert!(!TOOL_CALLS.is_empty());
        assert!(!TOOL_DURATION.is_empty());
        assert!(!URL_FETCHES.is_empty());
        assert!(!URL_FETCH_DURATION.is_empty());
        assert!(!WRITES_TOTAL.is_empty());
        assert!(!WRITE_ERRORS.is_empty());
        assert!(!WRITE_DURATION.is_empty());
        assert!(!PROCESSING_DURATION.is_empty());
        assert!(!LLM_DURATION.is_empty());
        assert!(!QUEUE_DEPTH.is_empty());
        assert!(!ADAPTER_RECONNECTS.is_empty());
        assert!(!MEMORY_OPS.is_empty());
        assert!(!MEMORY_DURATION.is_empty());
        assert!(!FEEDBACK_TOTAL.is_empty());
        assert!(!FEEDBACK_COMMENTS_TOTAL.is_empty());
        assert!(!FEEDBACK_DURATION.is_empty());
        assert!(!FEEDBACK_RATING_DISTRIBUTION.is_empty());
    }

    #[test]
    fn describe_metrics_does_not_panic() {
        // describe_metrics() can be called multiple times safely
        describe_metrics();
    }
}
