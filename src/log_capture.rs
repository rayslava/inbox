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
        let mut visitor = MessageVisitor::default();
        event.record(&mut visitor);

        self.store.push(LogEntry {
            timestamp: chrono::Utc::now().format("%H:%M:%S%.3f").to_string(),
            level: meta.level().to_string(),
            target: meta.target().to_string(),
            message: visitor.message,
        });
    }
}

#[derive(Default)]
struct MessageVisitor {
    message: String,
}

impl Visit for MessageVisitor {
    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "message" {
            value.clone_into(&mut self.message);
        }
    }

    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            // fmt::Arguments delegates Debug to Display, so no extra quotes.
            self.message = format!("{value:?}");
        }
    }
}
