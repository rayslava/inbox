//! A small time-based circuit breaker shared by the LLM backend clients. After
//! `record_failure`, the breaker reports "open" for `open_secs`, during which a
//! backend is skipped; `clear` (on a success) closes it early. Extracted from the
//! original inline Ollama implementation so cloud clients can reuse it for
//! service-unavailable (429/5xx/timeout) cooldown.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anodized::spec;

/// Shared, cheaply-cloneable cooldown gate. `open_secs == 0` disables it.
#[derive(Clone)]
pub(crate) struct CircuitBreaker {
    open_secs: u64,
    last_failure: Arc<Mutex<Option<Instant>>>,
}

impl CircuitBreaker {
    #[must_use]
    pub(crate) fn new(open_secs: u64) -> Self {
        Self {
            open_secs,
            last_failure: Arc::new(Mutex::new(None)),
        }
    }

    /// Open the circuit, starting the cooldown window now.
    pub(crate) fn record_failure(&self) {
        *self
            .last_failure
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(Instant::now());
    }

    /// Close the circuit (called on a successful response).
    pub(crate) fn clear(&self) {
        *self
            .last_failure
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
    }

    /// Remaining cooldown while the circuit is open; `None` when closed (disabled,
    /// no recorded failure, or the window has elapsed).
    #[must_use]
    #[spec(ensures: output.as_ref().is_none_or(|d| d.as_secs() <= self.open_secs))]
    pub(crate) fn remaining(&self) -> Option<Duration> {
        if self.open_secs == 0 {
            return None;
        }
        let guard = self
            .last_failure
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(failed_at) = *guard {
            let elapsed = failed_at.elapsed();
            let limit = Duration::from_secs(self.open_secs);
            if elapsed < limit {
                // `elapsed < limit` guarantees a positive remainder; checked_sub
                // returns None only on the boundary, which maps to "closed".
                return limit.checked_sub(elapsed);
            }
        }
        None
    }

    /// Test-only: open the circuit as if the failure happened at `at` (allows
    /// simulating an elapsed cooldown without sleeping).
    #[cfg(test)]
    pub(crate) fn open_since(&self, at: Instant) {
        *self
            .last_failure
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(at);
    }

    /// Test-only: whether a failure timestamp is recorded, regardless of whether
    /// the cooldown window has elapsed. Lets a test observe that `clear` ran even
    /// when the window was already expired.
    #[cfg(test)]
    pub(crate) fn has_recorded_failure(&self) -> bool {
        self.last_failure
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .is_some()
    }
}
