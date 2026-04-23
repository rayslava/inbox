//! Free-router backend: dynamic `OpenRouter` model pool with hedged dispatch.
//!
//! Fetches the shir-man `top-models` index to discover free `OpenRouter` models,
//! partitions them by tool-call capability, and serves each `complete()` call
//! by racing `parallel_fanout` candidates in parallel (first valid wins).
//! Refresh is reactive: triggered only when every candidate in a call errors,
//! paced by `min_refresh_interval_secs`.

use std::sync::Arc;
use std::time::{Duration, Instant};

use anodized::spec;
use async_trait::async_trait;
use futures::FutureExt;
use tokio::sync::{RwLock, Semaphore};
use tracing::{debug, info, instrument, warn};

use crate::config::LlmBackendConfig;
use crate::error::InboxError;

use super::openrouter::call_chat_completion;
use super::{LlmClient, LlmCompletion, LlmRequest};

mod pool;
#[cfg(test)]
mod tests;

use pool::{FreeModel, PoolPreferences, PoolState, fetch_pool};

pub struct FreeRouterClient {
    pub api_url: String,
    pub base_url: String,
    pub api_key: String,
    pub retries: u32,
    pub parallel_fanout: usize,
    pub per_model_retries: u32,
    pub min_refresh_interval: Duration,
    pub timeout: Duration,
    pub list_timeout: Duration,
    prefs: PoolPreferences,
    state: Arc<RwLock<PoolStateWithStamp>>,
    semaphore: Option<Arc<Semaphore>>,
    client: reqwest::Client,
}

struct PoolStateWithStamp {
    pool: PoolState,
    last_refreshed: Instant,
}

impl FreeRouterClient {
    /// Build a `FreeRouterClient` from backend config. Performs a synchronous
    /// (blocking) initial pool fetch using the current tokio runtime; on failure
    /// falls back to `openrouter/free`.
    ///
    /// # Panics
    /// Panics if the TLS backend cannot be initialised.
    #[must_use]
    #[spec(requires:
        !cfg.api_url.trim().is_empty()
        && !cfg.base_url.trim().is_empty()
        && cfg.timeout_secs > 0
        && cfg.parallel_fanout > 0
    )]
    pub fn from_config(cfg: &LlmBackendConfig) -> Self {
        let client = crate::tls::client_builder()
            .connect_timeout(Duration::from_secs(cfg.connect_timeout_secs))
            .timeout(Duration::from_secs(cfg.timeout_secs))
            .build()
            .expect("Failed to build FreeRouter HTTP client");

        let prefs = PoolPreferences::from(cfg);
        let list_timeout = Duration::from_secs(cfg.timeout_secs);
        let initial = initial_pool(&client, &cfg.api_url, list_timeout, prefs);

        Self {
            api_url: cfg.api_url.clone(),
            base_url: cfg.base_url.clone(),
            api_key: cfg.api_key.clone().unwrap_or_default(),
            retries: cfg.retries,
            parallel_fanout: cfg.parallel_fanout,
            per_model_retries: cfg.per_model_retries,
            min_refresh_interval: Duration::from_secs(cfg.min_refresh_interval_secs),
            timeout: Duration::from_secs(cfg.timeout_secs),
            list_timeout,
            prefs,
            state: Arc::new(RwLock::new(PoolStateWithStamp {
                pool: initial,
                last_refreshed: Instant::now(),
            })),
            semaphore: cfg.max_concurrent.map(|n| Arc::new(Semaphore::new(n))),
            client,
        }
    }

    /// Construct a client with a pre-built pool. Test-only — skips the
    /// startup list fetch so unit tests can drive pool contents directly.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn with_pool(cfg: &LlmBackendConfig, pool: PoolState) -> Self {
        let client = crate::tls::client_builder()
            .connect_timeout(Duration::from_secs(cfg.connect_timeout_secs))
            .timeout(Duration::from_secs(cfg.timeout_secs))
            .build()
            .expect("Failed to build FreeRouter HTTP client");

        Self {
            api_url: cfg.api_url.clone(),
            base_url: cfg.base_url.clone(),
            api_key: cfg.api_key.clone().unwrap_or_default(),
            retries: cfg.retries,
            parallel_fanout: cfg.parallel_fanout,
            per_model_retries: cfg.per_model_retries,
            min_refresh_interval: Duration::from_secs(cfg.min_refresh_interval_secs),
            timeout: Duration::from_secs(cfg.timeout_secs),
            list_timeout: Duration::from_secs(cfg.timeout_secs),
            prefs: PoolPreferences::from(cfg),
            state: Arc::new(RwLock::new(PoolStateWithStamp {
                pool,
                last_refreshed: Instant::now(),
            })),
            semaphore: cfg.max_concurrent.map(|n| Arc::new(Semaphore::new(n))),
            client,
        }
    }

    async fn candidate_models(&self, needs_tools: bool) -> Vec<FreeModel> {
        let guard = self.state.read().await;
        if needs_tools {
            let pool = &guard.pool.tool_models;
            if pool.is_empty() {
                warn!(
                    "Free-router: no tool-capable models; falling back to general pool with tools still requested"
                );
                guard.pool.general_models.clone()
            } else {
                pool.clone()
            }
        } else {
            guard.pool.general_models.clone()
        }
    }

    /// Trigger a reactive pool refresh if enough time has elapsed since the
    /// last one. Best-effort; failures are logged and do not propagate.
    async fn maybe_refresh(&self) {
        {
            let guard = self.state.read().await;
            if guard.last_refreshed.elapsed() < self.min_refresh_interval {
                return;
            }
        }
        match fetch_pool(&self.client, &self.api_url, self.list_timeout, self.prefs).await {
            Ok(new_pool) => {
                info!(
                    tool_models = new_pool.tool_models.len(),
                    general_models = new_pool.general_models.len(),
                    "Free-router pool refreshed"
                );
                let mut guard = self.state.write().await;
                guard.pool = new_pool;
                guard.last_refreshed = Instant::now();
            }
            Err(e) => {
                warn!(?e, "Free-router pool refresh failed; keeping current pool");
                let mut guard = self.state.write().await;
                guard.last_refreshed = Instant::now();
            }
        }
    }
}

/// Fetch the pool synchronously at startup by blocking on the current runtime.
/// Degraded fallback seeds `openrouter/free` if the fetch fails.
fn initial_pool(
    client: &reqwest::Client,
    api_url: &str,
    timeout: Duration,
    prefs: PoolPreferences,
) -> PoolState {
    let fut = fetch_pool(client, api_url, timeout, prefs);
    let outcome = tokio::runtime::Handle::try_current()
        .ok()
        .map(|h| tokio::task::block_in_place(|| h.block_on(fut)));

    match outcome {
        Some(Ok(pool)) if !pool.is_empty() => {
            info!(
                tool_models = pool.tool_models.len(),
                general_models = pool.general_models.len(),
                "Free-router pool initialised"
            );
            pool
        }
        Some(Ok(_)) => {
            warn!(
                "Free-router top-models list returned no healthy models; using degraded fallback"
            );
            metrics::counter!(
                crate::telemetry::LLM_REQUESTS,
                "backend" => "free_router",
                "status" => "degraded",
            )
            .increment(1);
            PoolState::degraded_fallback()
        }
        Some(Err(e)) => {
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
            PoolState::degraded_fallback()
        }
        None => {
            warn!("No tokio runtime available; seeding free-router with degraded fallback");
            PoolState::degraded_fallback()
        }
    }
}

#[async_trait]
impl LlmClient for FreeRouterClient {
    fn name(&self) -> &'static str {
        "free_router"
    }

    fn model(&self) -> &'static str {
        // Dynamic pool — no single model ID. Report the backend label so chain
        // logs remain meaningful.
        "free_router:dynamic"
    }

    fn retries(&self) -> u32 {
        self.retries.max(1)
    }

    fn thinking_supported(&self) -> bool {
        if !self.prefs.prefer_reasoning {
            return false;
        }
        // Best-effort read: we can't await inside a sync trait method, so use
        // try_read. If the lock is momentarily held, default to false — chain
        // logic only consults this to decide whether to offer activate_thinking.
        self.state
            .try_read()
            .is_ok_and(|g| g.pool.general_models.iter().any(|m| m.supports_reasoning))
    }

    #[instrument(skip(self, req), fields(backend = "free_router"))]
    async fn complete(&self, req: LlmRequest) -> Result<LlmCompletion, InboxError> {
        let _permit = if let Some(sem) = &self.semaphore {
            Some(sem.acquire().await.expect("semaphore closed"))
        } else {
            None
        };

        let needs_tools = !req.tool_definitions.is_empty();
        let candidates = self.candidate_models(needs_tools).await;
        if candidates.is_empty() {
            return Err(InboxError::Llm(
                "free-router pool is empty (both tool and general)".into(),
            ));
        }

        debug!(
            needs_tools,
            pool_size = candidates.len(),
            fanout = self.parallel_fanout,
            "free-router dispatching"
        );

        let fanout = self.parallel_fanout.max(1);
        let mut last_err: Option<InboxError> = None;
        for batch in candidates.chunks(fanout) {
            match self.race_batch(batch, &req).await {
                Ok(completion) => return Ok(completion),
                Err(e) => {
                    warn!(
                        ?e,
                        models = ?batch.iter().map(|m| m.id.as_str()).collect::<Vec<_>>(),
                        "free-router batch exhausted"
                    );
                    last_err = Some(e);
                }
            }
        }

        // All batches failed — attempt a reactive refresh and, if the pool
        // changed, retry once with the freshly fetched candidates.
        self.maybe_refresh().await;
        let refreshed = self.candidate_models(needs_tools).await;
        if !refreshed.is_empty() && !refreshed_is_same(&refreshed, &candidates) {
            return self.complete_with(refreshed, req.clone()).await;
        }

        Err(last_err
            .unwrap_or_else(|| InboxError::Llm("free-router exhausted all candidates".into())))
    }
}

fn refreshed_is_same(a: &[FreeModel], b: &[FreeModel]) -> bool {
    a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| x.id == y.id)
}

impl FreeRouterClient {
    /// Variant of `complete` used after a reactive refresh replaces the pool.
    /// Kept separate so the top-level `complete` method does not recurse
    /// unboundedly.
    async fn complete_with(
        &self,
        candidates: Vec<FreeModel>,
        req: LlmRequest,
    ) -> Result<LlmCompletion, InboxError> {
        let fanout = self.parallel_fanout.max(1);
        let mut last_err: Option<InboxError> = None;
        for batch in candidates.chunks(fanout) {
            match self.race_batch(batch, &req).await {
                Ok(completion) => return Ok(completion),
                Err(e) => last_err = Some(e),
            }
        }
        Err(last_err
            .unwrap_or_else(|| InboxError::Llm("free-router refreshed pool exhausted".into())))
    }

    /// Race a batch of models in parallel. First `Ok` wins; on first success,
    /// pending futures are dropped and their in-flight requests cancelled.
    async fn race_batch(
        &self,
        models: &[FreeModel],
        req: &LlmRequest,
    ) -> Result<LlmCompletion, InboxError> {
        let futures = models.iter().map(|m| {
            let model_id = m.id.clone();
            let req = req.clone();
            self.call_one_model_with_retries(model_id, req).boxed()
        });

        match futures::future::select_ok(futures).await {
            Ok((completion, _rest)) => Ok(completion),
            Err(e) => Err(e),
        }
    }

    /// Invoke a single model with its per-model retry budget.
    #[instrument(skip(self, req), fields(model = %model_id))]
    async fn call_one_model_with_retries(
        &self,
        model_id: String,
        req: LlmRequest,
    ) -> Result<LlmCompletion, InboxError> {
        let total_attempts = self.per_model_retries.saturating_add(1);
        let mut last_err: Option<InboxError> = None;
        for attempt in 0..total_attempts {
            if attempt > 0 {
                let delay_ms = 500u64.saturating_mul(2u64.saturating_pow(attempt - 1));
                tokio::time::sleep(Duration::from_millis(delay_ms)).await;
            }
            let start = std::time::Instant::now();
            let result = call_chat_completion(
                &self.client,
                &self.base_url,
                &self.api_key,
                &model_id,
                &req,
                "free_router",
            )
            .await;
            match result {
                Ok(c) => {
                    metrics::counter!(
                        crate::telemetry::LLM_REQUESTS,
                        "backend" => "free_router",
                        "status" => "success",
                    )
                    .increment(1);
                    metrics::histogram!(
                        crate::telemetry::LLM_DURATION,
                        "backend" => "free_router",
                    )
                    .record(start.elapsed().as_secs_f64());
                    return Ok(c);
                }
                Err(e) if is_hard_error(&e) => {
                    metrics::counter!(
                        crate::telemetry::LLM_REQUESTS,
                        "backend" => "free_router",
                        "status" => "hard_failure",
                    )
                    .increment(1);
                    return Err(e);
                }
                Err(e) => {
                    debug!(?e, attempt, model = %model_id, "free-router per-model retry");
                    last_err = Some(e);
                }
            }
        }
        metrics::counter!(
            crate::telemetry::LLM_REQUESTS,
            "backend" => "free_router",
            "status" => "failure",
        )
        .increment(1);
        Err(last_err.unwrap_or_else(|| {
            InboxError::Llm(format!("free-router: model {model_id} exhausted retries"))
        }))
    }
}

/// Errors that should abort retries on the current model rather than wasting
/// the retry budget on something that will never succeed. Auth and malformed-
/// request failures are deterministic across attempts.
fn is_hard_error(err: &InboxError) -> bool {
    let InboxError::Llm(msg) = err else {
        return false;
    };
    msg.contains(" 401") || msg.contains(" 403") || msg.contains(" 400")
}
