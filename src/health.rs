use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Shared readiness state. Adapters and the admin server both hold a clone.
#[derive(Clone, Debug)]
pub struct ReadinessState {
    inner: Arc<AtomicBool>,
}

impl ReadinessState {
    #[must_use]
    pub fn new(ready: bool) -> Self {
        Self {
            inner: Arc::new(AtomicBool::new(ready)),
        }
    }

    pub fn set_ready(&self) {
        self.inner.store(true, Ordering::SeqCst);
    }

    pub fn set_not_ready(&self) {
        self.inner.store(false, Ordering::SeqCst);
    }

    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.inner.load(Ordering::SeqCst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starts_ready() {
        let s = ReadinessState::new(true);
        assert!(s.is_ready());
    }

    #[test]
    fn starts_not_ready() {
        let s = ReadinessState::new(false);
        assert!(!s.is_ready());
    }

    #[test]
    fn set_ready_and_not_ready() {
        let s = ReadinessState::new(false);
        s.set_ready();
        assert!(s.is_ready());
        s.set_not_ready();
        assert!(!s.is_ready());
    }

    #[test]
    fn clone_shares_state() {
        let s = ReadinessState::new(false);
        let s2 = s.clone();
        s.set_ready();
        assert!(s2.is_ready());
    }
}
