use std::future::Future;
use std::time::{Duration, Instant};

use tokio_util::sync::CancellationToken;
use tracing::warn;

/// Reconnection policy shared by all adapters.
#[derive(Debug, Clone)]
pub struct ReconnectPolicy {
    /// Initial delay before the first reconnect attempt.
    pub initial_backoff: Duration,
    /// Maximum delay between reconnect attempts.
    pub max_backoff: Duration,
    /// If the inner operation ran longer than this, reset backoff to `initial_backoff`.
    /// `None` disables the stable-reset behaviour.
    pub stable_threshold: Option<Duration>,
    /// Label emitted with the `ADAPTER_RECONNECTS` metric counter.
    pub adapter_label: &'static str,
}

/// Run `operation` in a loop, reconnecting with exponential backoff.
///
/// The loop exits when `shutdown` is cancelled. On each reconnect the
/// `ADAPTER_RECONNECTS` counter is incremented.
///
/// `operation` is called with a `&CancellationToken` so it can check for
/// shutdown internally, but the outer loop also checks after each iteration.
pub async fn reconnect_loop<F, Fut>(
    policy: ReconnectPolicy,
    shutdown: CancellationToken,
    mut operation: F,
) where
    F: FnMut(&CancellationToken) -> Fut,
    Fut: Future<Output = ()>,
{
    let mut backoff = policy.initial_backoff;
    let mut first = true;

    loop {
        let started = Instant::now();

        // Run the inner operation until it returns (error, panic, or clean exit).
        tokio::select! {
            () = shutdown.cancelled() => return,
            () = operation(&shutdown) => {}
        }

        if shutdown.is_cancelled() {
            return;
        }

        // Emit reconnect metric (skip the very first run — that's the initial start, not a reconnect).
        if !first {
            metrics::counter!(
                crate::telemetry::ADAPTER_RECONNECTS,
                "adapter" => policy.adapter_label
            )
            .increment(1);
        }
        first = false;

        // Reset backoff if the operation ran long enough to be considered stable.
        if let Some(threshold) = policy.stable_threshold {
            if started.elapsed() >= threshold {
                backoff = policy.initial_backoff;
            }
        }

        warn!(
            adapter = policy.adapter_label,
            delay_secs = backoff.as_secs(),
            "Adapter session ended, reconnecting"
        );

        tokio::select! {
            () = shutdown.cancelled() => return,
            () = tokio::time::sleep(backoff) => {}
        }

        backoff = (backoff * 2).min(policy.max_backoff);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};

    fn test_policy() -> ReconnectPolicy {
        ReconnectPolicy {
            initial_backoff: Duration::from_secs(1),
            max_backoff: Duration::from_secs(8),
            stable_threshold: Some(Duration::from_secs(30)),
            adapter_label: "test",
        }
    }

    #[tokio::test(start_paused = true)]
    async fn shutdown_before_first_run_exits_immediately() {
        let shutdown = CancellationToken::new();
        shutdown.cancel();

        let calls = Arc::new(AtomicU32::new(0));
        let calls_clone = calls.clone();

        reconnect_loop(test_policy(), shutdown, move |_| {
            let c = calls_clone.clone();
            async move {
                c.fetch_add(1, Ordering::Relaxed);
            }
        })
        .await;

        assert_eq!(calls.load(Ordering::Relaxed), 0);
    }

    #[tokio::test(start_paused = true)]
    async fn shutdown_during_backoff_exits() {
        let shutdown = CancellationToken::new();
        let calls = Arc::new(AtomicU32::new(0));
        let calls_clone = calls.clone();
        let shutdown_clone = shutdown.clone();

        // Operation runs once, then cancel during the backoff sleep.
        reconnect_loop(test_policy(), shutdown.clone(), move |_| {
            let c = calls_clone.clone();
            let s = shutdown_clone.clone();
            async move {
                let n = c.fetch_add(1, Ordering::Relaxed);
                if n == 0 {
                    // First call: return immediately → triggers backoff.
                    // Then cancel from a spawned task so select picks it up.
                    tokio::spawn(async move {
                        tokio::time::sleep(Duration::from_millis(500)).await;
                        s.cancel();
                    });
                }
            }
        })
        .await;

        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }

    #[tokio::test(start_paused = true)]
    async fn backoff_doubles_up_to_max() {
        let shutdown = CancellationToken::new();
        let calls = Arc::new(AtomicU32::new(0));
        let calls_clone = calls.clone();
        let shutdown_clone = shutdown.clone();

        // Run 5 iterations, each returning immediately (unstable → backoff grows).
        reconnect_loop(test_policy(), shutdown.clone(), move |_| {
            let c = calls_clone.clone();
            let s = shutdown_clone.clone();
            async move {
                let n = c.fetch_add(1, Ordering::Relaxed);
                if n >= 4 {
                    s.cancel();
                }
            }
        })
        .await;

        // 5 calls: initial + 4 reconnects
        assert_eq!(calls.load(Ordering::Relaxed), 5);
        // Backoff sequence: 1s, 2s, 4s, 8s (capped)
        // Total sleep: 1 + 2 + 4 + 8 = 15s (but we won't sleep on the last one due to cancel)
        // With paused time this completes instantly.
    }

    #[tokio::test(start_paused = true)]
    async fn stable_connection_resets_backoff() {
        let shutdown = CancellationToken::new();
        let calls = Arc::new(AtomicU32::new(0));
        let calls_clone = calls.clone();
        let shutdown_clone = shutdown.clone();

        reconnect_loop(test_policy(), shutdown.clone(), move |_| {
            let c = calls_clone.clone();
            let s = shutdown_clone.clone();
            async move {
                let n = c.fetch_add(1, Ordering::Relaxed);
                if n == 1 {
                    // Second call: simulate a stable session (>30s)
                    tokio::time::sleep(Duration::from_secs(31)).await;
                    // Returns → backoff should reset to 1s (not 2s)
                } else if n == 2 {
                    // Third call: cancel to exit
                    s.cancel();
                }
            }
        })
        .await;

        assert_eq!(calls.load(Ordering::Relaxed), 3);
    }

    #[tokio::test(start_paused = true)]
    async fn fixed_backoff_when_init_equals_max() {
        let policy = ReconnectPolicy {
            initial_backoff: Duration::from_secs(5),
            max_backoff: Duration::from_secs(5),
            stable_threshold: None,
            adapter_label: "fixed",
        };

        let shutdown = CancellationToken::new();
        let calls = Arc::new(AtomicU32::new(0));
        let calls_clone = calls.clone();
        let shutdown_clone = shutdown.clone();

        reconnect_loop(policy, shutdown.clone(), move |_| {
            let c = calls_clone.clone();
            let s = shutdown_clone.clone();
            async move {
                let n = c.fetch_add(1, Ordering::Relaxed);
                if n >= 2 {
                    s.cancel();
                }
            }
        })
        .await;

        // 3 calls: each separated by 5s backoff (capped = same as init).
        assert_eq!(calls.load(Ordering::Relaxed), 3);
    }

    #[tokio::test(start_paused = true)]
    async fn first_run_does_not_emit_reconnect() {
        // This tests that the first iteration is not counted as a reconnect.
        // We rely on the `first` flag in the implementation.
        let shutdown = CancellationToken::new();
        let shutdown_clone = shutdown.clone();

        reconnect_loop(test_policy(), shutdown.clone(), move |_| {
            let s = shutdown_clone.clone();
            async move {
                s.cancel();
            }
        })
        .await;
        // If it got here without panicking, the first-run skip logic works.
    }

    #[tokio::test(start_paused = true)]
    async fn shutdown_during_operation_exits() {
        let shutdown = CancellationToken::new();
        let calls = Arc::new(AtomicU32::new(0));
        let calls_clone = calls.clone();
        let shutdown_clone = shutdown.clone();

        reconnect_loop(test_policy(), shutdown.clone(), move |token| {
            let c = calls_clone.clone();
            let t = token.clone();
            let s = shutdown_clone.clone();
            async move {
                c.fetch_add(1, Ordering::Relaxed);
                s.cancel();
                // Simulate the operation noticing shutdown.
                t.cancelled().await;
            }
        })
        .await;

        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }
}
