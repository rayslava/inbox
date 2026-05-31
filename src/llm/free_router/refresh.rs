//! Pool refresh + startup-seed lifecycle for the free-router backend. Split out
//! of `free_router.rs` to keep that file focused on the client and dispatch.

use std::time::{Duration, Instant};

use tokio::sync::RwLock;
use tracing::{info, warn};

use super::PoolStateWithStamp;
use super::pool::{PoolPreferences, PoolState, fetch_pool};

/// Fetch a fresh pool and install it into `state`, ignoring the refresh
/// interval. Shared by `refresh_now` and the deferred startup refresh.
///
/// An empty fetch result never replaces a non-empty pool — a transient
/// top-models hiccup (zero healthy models for a moment) must not blank the
/// backend. If the current pool is *also* empty we seed the degraded
/// `openrouter/free` fallback so the backend keeps serving and can later
/// self-heal on the next successful refresh.
pub(super) async fn refresh_into(
    state: &RwLock<PoolStateWithStamp>,
    client: &reqwest::Client,
    api_url: &str,
    base_url: &str,
    list_timeout: Duration,
    prefs: PoolPreferences,
) {
    match fetch_pool(client, api_url, base_url, list_timeout, prefs).await {
        Ok(new_pool) if !new_pool.is_empty() => {
            info!(
                tool_models = new_pool.tool_models.len(),
                general_models = new_pool.general_models.len(),
                vision_models = new_pool.vision_models.len(),
                "Free-router pool refreshed"
            );
            let mut guard = state.write().await;
            guard.pool = new_pool;
            guard.last_refreshed = Instant::now();
        }
        Ok(_) => {
            warn!("Free-router refresh returned no healthy models; keeping current pool");
            let mut guard = state.write().await;
            if guard.pool.is_empty() {
                warn!("Free-router pool empty after refresh; seeding degraded fallback");
                guard.pool = PoolState::degraded_fallback();
            }
            guard.last_refreshed = Instant::now();
        }
        Err(e) => {
            warn!(?e, "Free-router pool refresh failed; keeping current pool");
            let mut guard = state.write().await;
            if guard.pool.is_empty() {
                warn!("Free-router pool empty and refresh failed; seeding degraded fallback");
                guard.pool = PoolState::degraded_fallback();
            }
            guard.last_refreshed = Instant::now();
        }
    }
}

/// Decide the startup pool. Returns the seed plus whether the caller should
/// spawn a background refresh to fill it.
///
/// The synchronous list fetch relies on `block_in_place`, which panics on a
/// current-thread runtime; it is therefore done inline only on a multi-thread
/// runtime. Off that (current-thread runtimes, e.g. some tests), we seed the
/// degraded `openrouter/free` fallback and signal the caller to refresh it in
/// the background. With no runtime at all we cannot fetch or spawn, so the
/// degraded seed stands until the first `complete()` triggers a refresh.
pub(super) fn initial_pool(
    client: &reqwest::Client,
    api_url: &str,
    base_url: &str,
    timeout: Duration,
    prefs: PoolPreferences,
) -> (PoolState, bool) {
    let Ok(handle) = tokio::runtime::Handle::try_current() else {
        warn!("No tokio runtime available; seeding free-router with degraded fallback");
        return (PoolState::degraded_fallback(), false);
    };

    if handle.runtime_flavor() != tokio::runtime::RuntimeFlavor::MultiThread {
        warn!(
            "Free-router init off the multi-thread runtime; deferring pool fetch to background refresh"
        );
        return (PoolState::degraded_fallback(), true);
    }

    let fut = fetch_pool(client, api_url, base_url, timeout, prefs);
    match tokio::task::block_in_place(|| handle.block_on(fut)) {
        Ok(pool) if !pool.is_empty() => {
            info!(
                tool_models = pool.tool_models.len(),
                general_models = pool.general_models.len(),
                vision_models = pool.vision_models.len(),
                "Free-router pool initialised"
            );
            (pool, false)
        }
        Ok(_) => {
            warn!(
                "Free-router top-models list returned no healthy models; using degraded fallback"
            );
            metrics::counter!(
                crate::telemetry::LLM_REQUESTS,
                "backend" => "free_router",
                "status" => "degraded",
            )
            .increment(1);
            (PoolState::degraded_fallback(), false)
        }
        Err(e) => {
            warn!(
                ?e,
                "Free-router initial list fetch failed; using degraded fallback"
            );
            metrics::counter!(
                crate::telemetry::LLM_REQUESTS,
                "backend" => "free_router",
                "status" => "degraded",
            )
            .increment(1);
            (PoolState::degraded_fallback(), false)
        }
    }
}
