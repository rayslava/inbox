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

/// Labels: status = "success" | "failure"
pub const URL_FETCHES: &str = "inbox_url_fetches_total";

pub const WRITE_ERRORS: &str = "inbox_write_errors_total";

/// Labels: adapter = "telegram" | "email"
pub const ADAPTER_RECONNECTS: &str = "inbox_adapter_reconnects_total";

/// Labels: op = "save" | "recall" | "link" | "context" | "sources", status = "success" | "failure"
pub const MEMORY_OPS: &str = "inbox_memory_ops_total";

// ── Histograms ────────────────────────────────────────────────────────────────

/// Labels: source
pub const PROCESSING_DURATION: &str = "inbox_processing_duration_seconds";

/// Labels: backend
pub const LLM_DURATION: &str = "inbox_llm_duration_seconds";

/// Labels: op
pub const MEMORY_DURATION: &str = "inbox_memory_duration_seconds";

// ── Gauges ────────────────────────────────────────────────────────────────────

pub const QUEUE_DEPTH: &str = "inbox_queue_depth";

// ─────────────────────────────────────────────────────────────────────────────

/// Call once at startup to register metric descriptions with the recorder.
pub fn describe_metrics() {
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
    describe_counter!(URL_FETCHES, Unit::Count, "Total URL fetch attempts");
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
    describe_counter!(
        MEMORY_OPS,
        Unit::Count,
        "Total memory store operations (save, recall, link, context, sources)"
    );
    describe_histogram!(
        MEMORY_DURATION,
        Unit::Seconds,
        "Memory store operation duration"
    );
    describe_gauge!(
        QUEUE_DEPTH,
        Unit::Count,
        "Current number of messages waiting in the pipeline queue"
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
        assert!(!URL_FETCHES.is_empty());
        assert!(!WRITE_ERRORS.is_empty());
        assert!(!PROCESSING_DURATION.is_empty());
        assert!(!LLM_DURATION.is_empty());
        assert!(!QUEUE_DEPTH.is_empty());
        assert!(!ADAPTER_RECONNECTS.is_empty());
        assert!(!MEMORY_OPS.is_empty());
        assert!(!MEMORY_DURATION.is_empty());
    }

    #[test]
    fn describe_metrics_does_not_panic() {
        // describe_metrics() can be called multiple times safely
        describe_metrics();
    }
}
