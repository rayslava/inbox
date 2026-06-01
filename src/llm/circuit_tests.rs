//! Unit tests for the shared `CircuitBreaker`.

use std::time::{Duration, Instant};

use super::CircuitBreaker;

#[test]
fn disabled_when_open_secs_zero() {
    let cb = CircuitBreaker::new(0);
    cb.record_failure();
    assert!(cb.remaining().is_none(), "open_secs=0 disables the breaker");
}

#[test]
fn opens_on_failure_and_reports_remaining() {
    let cb = CircuitBreaker::new(300);
    assert!(cb.remaining().is_none(), "closed before any failure");
    cb.record_failure();
    assert!(cb.remaining().is_some());
    let remaining = cb.remaining().expect("open");
    assert!(remaining.as_secs() <= 300);
    assert!(
        remaining.as_secs() >= 299,
        "freshly opened ⇒ near full window"
    );
}

#[test]
fn clear_closes_the_circuit() {
    let cb = CircuitBreaker::new(300);
    cb.record_failure();
    assert!(cb.remaining().is_some());
    cb.clear();
    assert!(cb.remaining().is_none());
}

#[test]
fn expired_window_is_closed() {
    let cb = CircuitBreaker::new(1);
    cb.open_since(Instant::now().checked_sub(Duration::from_secs(10)).unwrap());
    assert!(cb.remaining().is_none(), "elapsed > window ⇒ closed");
}

#[test]
fn within_window_is_open() {
    let cb = CircuitBreaker::new(300);
    cb.open_since(Instant::now().checked_sub(Duration::from_secs(10)).unwrap());
    assert!(cb.remaining().is_some());
    assert!(cb.remaining().expect("open").as_secs() <= 290);
}
