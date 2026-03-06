use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use tracing::field::{Field, Visit};
use tracing_subscriber::Layer;

pub const CAPACITY: usize = 500;

#[derive(Clone)]
pub struct LogEntry {
    pub timestamp: String,
    pub level: String,
    pub target: String,
    pub message: String,
    /// Structured fields from the tracing event (excludes the "message" field).
    pub fields: Vec<(String, String)>,
}

pub struct LogStore {
    entries: Mutex<VecDeque<LogEntry>>,
    capacity: usize,
}

impl LogStore {
    #[must_use]
    pub fn new(capacity: usize) -> Arc<Self> {
        Arc::new(Self {
            entries: Mutex::new(VecDeque::with_capacity(capacity)),
            capacity,
        })
    }

    /// # Panics
    /// Panics if the internal mutex is poisoned (only possible after a previous panic).
    pub fn push(&self, entry: LogEntry) {
        let mut guard = self.entries.lock().expect("log store mutex poisoned");
        if guard.len() >= self.capacity {
            guard.pop_front();
        }
        guard.push_back(entry);
    }

    /// Returns entries newest-first.
    ///
    /// # Panics
    /// Panics if the internal mutex is poisoned (only possible after a previous panic).
    #[must_use]
    pub fn recent(&self) -> Vec<LogEntry> {
        self.entries
            .lock()
            .expect("log store mutex poisoned")
            .iter()
            .rev()
            .cloned()
            .collect()
    }
}

pub struct LogCaptureLayer {
    store: Arc<LogStore>,
}

impl LogCaptureLayer {
    #[must_use]
    pub fn new(store: Arc<LogStore>) -> Self {
        Self { store }
    }
}

impl<S: tracing::Subscriber> Layer<S> for LogCaptureLayer {
    fn on_event(
        &self,
        event: &tracing::Event<'_>,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        let meta = event.metadata();
        let mut visitor = FieldCollectorVisitor::default();
        event.record(&mut visitor);

        self.store.push(LogEntry {
            timestamp: chrono::Utc::now().format("%H:%M:%S%.3f").to_string(),
            level: meta.level().to_string(),
            target: meta.target().to_string(),
            message: visitor.message,
            fields: visitor.fields,
        });
    }
}

#[derive(Default)]
struct FieldCollectorVisitor {
    message: String,
    fields: Vec<(String, String)>,
}

impl Visit for FieldCollectorVisitor {
    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "message" {
            value.clone_into(&mut self.message);
        } else {
            self.fields
                .push((field.name().to_string(), value.to_string()));
        }
    }

    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            // fmt::Arguments delegates Debug to Display, so no extra quotes.
            self.message = format!("{value:?}");
        } else {
            self.fields
                .push((field.name().to_string(), format!("{value:?}")));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tracing_subscriber::prelude::*;

    #[test]
    fn log_store_enforces_capacity_and_newest_first() {
        let store = LogStore::new(2);
        store.push(LogEntry {
            timestamp: "00:00:01.000".into(),
            level: "INFO".into(),
            target: "t1".into(),
            message: "first".into(),
            fields: vec![],
        });
        store.push(LogEntry {
            timestamp: "00:00:02.000".into(),
            level: "INFO".into(),
            target: "t2".into(),
            message: "second".into(),
            fields: vec![],
        });
        store.push(LogEntry {
            timestamp: "00:00:03.000".into(),
            level: "INFO".into(),
            target: "t3".into(),
            message: "third".into(),
            fields: vec![],
        });

        let recent = store.recent();
        assert_eq!(recent.len(), 2);
        assert_eq!(recent[0].message, "third");
        assert_eq!(recent[1].message, "second");
    }

    #[test]
    fn capture_layer_records_message_and_metadata() {
        let store = LogStore::new(10);
        let layer = LogCaptureLayer::new(Arc::clone(&store));
        let subscriber = tracing_subscriber::registry().with(layer);
        let _guard = tracing::subscriber::set_default(subscriber);

        tracing::info!(target: "inbox::capture_test", "hello from capture layer");

        let recent = store.recent();
        assert_eq!(recent.len(), 1);
        assert_eq!(recent[0].target, "inbox::capture_test");
        assert_eq!(recent[0].level, "INFO");
        assert!(recent[0].message.contains("hello from capture layer"));
        assert!(!recent[0].timestamp.is_empty());
        assert!(recent[0].fields.is_empty());
    }

    #[test]
    fn capture_layer_records_structured_fields() {
        let store = LogStore::new(10);
        let layer = LogCaptureLayer::new(Arc::clone(&store));
        let subscriber = tracing_subscriber::registry().with(layer);
        let _guard = tracing::subscriber::set_default(subscriber);

        tracing::info!(
            tool = "scrape_page",
            url = "https://example.com",
            "Tool executed"
        );

        let recent = store.recent();
        assert_eq!(recent.len(), 1);
        assert!(recent[0].message.contains("Tool executed"));
        let field_names: Vec<&str> = recent[0].fields.iter().map(|(k, _)| k.as_str()).collect();
        assert!(field_names.contains(&"tool"));
        assert!(field_names.contains(&"url"));
        let tool_val = recent[0]
            .fields
            .iter()
            .find(|(k, _)| k == "tool")
            .map(|(_, v)| v.as_str());
        assert_eq!(tool_val, Some("scrape_page"));
    }
}
