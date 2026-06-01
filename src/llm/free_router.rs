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
use tracing::{debug, instrument, warn};

use crate::config::LlmBackendConfig;
use crate::error::InboxError;

use super::openrouter::call_chat_completion;
use super::{LlmClient, LlmCompletion, LlmRequest};

mod pool;
mod refresh;
#[cfg(test)]
mod tests;

use pool::{FreeModel, PoolPreferences, PoolState};
use refresh::{initial_pool, refresh_into};

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
    circuit: super::CircuitBreaker,
}

pub(super) struct PoolStateWithStamp {
    pool: PoolState,
    last_refreshed: Instant,
}

impl FreeRouterClient {
    /// Build a `FreeRouterClient` from backend config. Performs a synchronous
    /// (blocking) initial pool fetch using the current tokio runtime; on failure
    /// falls back to `openrouter/free`.
    ///
    /// # Errors
    /// Returns an error if the HTTP client cannot be built.
    #[spec(requires:
        !cfg.api_url.trim().is_empty()
        && !cfg.base_url.trim().is_empty()
        && cfg.timeout_secs > 0
        && cfg.parallel_fanout > 0
    )]
    pub fn from_config(cfg: &LlmBackendConfig) -> Result<Self, InboxError> {
        let client = crate::tls::client_builder()
            .connect_timeout(Duration::from_secs(cfg.connect_timeout_secs))
            .timeout(Duration::from_secs(cfg.timeout_secs))
            .build()
            .map_err(|e| InboxError::Llm(format!("Failed to build FreeRouter HTTP client: {e}")))?;

        let prefs = PoolPreferences::from(cfg);
        let list_timeout = Duration::from_secs(cfg.timeout_secs);
        let (initial, defer_refresh) =
            initial_pool(&client, &cfg.api_url, &cfg.base_url, list_timeout, prefs);

        let state = Arc::new(RwLock::new(PoolStateWithStamp {
            pool: initial,
            last_refreshed: Instant::now(),
        }));

        // Off the multi-thread runtime the synchronous fetch is skipped (it would
        // panic via `block_in_place`), so fill the degraded seed in the
        // background instead of blocking construction.
        if defer_refresh && let Ok(handle) = tokio::runtime::Handle::try_current() {
            let state = Arc::clone(&state);
            let client = client.clone();
            let api_url = cfg.api_url.clone();
            let base_url = cfg.base_url.clone();
            handle.spawn(async move {
                refresh_into(&state, &client, &api_url, &base_url, list_timeout, prefs).await;
            });
        }

        Ok(Self {
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
            state,
            semaphore: cfg.max_concurrent.map(|n| Arc::new(Semaphore::new(n))),
            client,
            circuit: super::CircuitBreaker::new(cfg.circuit_open_secs),
        })
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
            circuit: super::CircuitBreaker::new(cfg.circuit_open_secs),
        }
    }

    /// Whether both the tool and general pools are empty (a genuinely drained
    /// pool, as opposed to merely lacking vision-capable models).
    async fn pool_is_empty(&self) -> bool {
        self.state.read().await.pool.is_empty()
    }

    async fn candidate_models(&self, needs_tools: bool, needs_vision: bool) -> Vec<FreeModel> {
        let guard = self.state.read().await;

        // Vision requests must never reach a non-vision model. Draw from the
        // vision pool only; when tools are also needed, prefer the intersection
        // of vision-and-tool models, relaxing the tool constraint (not the
        // vision one) if that intersection is empty — mirroring the tool→general
        // leniency below.
        if needs_vision {
            let vision = &guard.pool.vision_models;
            if needs_tools {
                let both: Vec<FreeModel> = vision
                    .iter()
                    .filter(|m| m.supports_tools && m.supports_tool_choice)
                    .cloned()
                    .collect();
                if both.is_empty() {
                    if !vision.is_empty() {
                        warn!(
                            "Free-router: no vision+tool models; using vision pool with tools still requested"
                        );
                    }
                    return vision.clone();
                }
                return both;
            }
            return vision.clone();
        }

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
        self.refresh_now().await;
    }

    /// Fetch a fresh pool and install it, ignoring the refresh interval.
    ///
    /// An empty fetch result never replaces a non-empty pool — a transient
    /// top-models hiccup (zero healthy models for a moment) must not blank the
    /// backend. If the current pool is *also* empty we seed the degraded
    /// `openrouter/free` fallback so the backend keeps serving and can later
    /// self-heal on the next successful refresh.
    async fn refresh_now(&self) {
        refresh_into(
            &self.state,
            &self.client,
            &self.api_url,
            &self.base_url,
            self.list_timeout,
            self.prefs,
        )
        .await;
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

    fn vision_supported(&self) -> bool {
        // Best-effort read, same rationale as `thinking_supported`. The chain
        // consults this to decide whether to route image requests here.
        self.state
            .try_read()
            .is_ok_and(|g| !g.pool.vision_models.is_empty())
    }

    fn is_available(&self) -> bool {
        self.circuit.remaining().is_none()
    }

    fn mark_unavailable(&self) {
        self.circuit.record_failure();
    }

    #[instrument(skip(self, req), fields(backend = "free_router"))]
    async fn complete(&self, req: LlmRequest) -> Result<LlmCompletion, InboxError> {
        if let Some(d) = self.circuit.remaining() {
            return Err(InboxError::Llm(format!(
                "free_router circuit open: cooldown {}s remaining",
                d.as_secs()
            )));
        }
        let result = self.complete_inner(req).await;
        // Every candidate was exhausted before `complete_inner` returns an error,
        // so a transient failure means the whole backend could not serve this
        // request — trip the cooldown; clear it on any success.
        match &result {
            Ok(_) => self.circuit.clear(),
            Err(e) if !super::chain::is_service_available(e) => self.circuit.record_failure(),
            Err(_) => {}
        }
        result
    }
}

impl FreeRouterClient {
    async fn complete_inner(&self, req: LlmRequest) -> Result<LlmCompletion, InboxError> {
        // `acquire` errors only if the semaphore is closed, which never happens
        // here (we hold the `Arc` and never call `close`). Treat the impossible
        // error as "no permit" and proceed rather than panicking.
        let _permit = match &self.semaphore {
            Some(sem) => sem.acquire().await.ok(),
            None => None,
        };

        let needs_tools = !req.tool_definitions.is_empty();
        let needs_vision = req.needs_vision();
        let mut candidates = self.candidate_models(needs_tools, needs_vision).await;
        if candidates.is_empty() {
            // A vision request against an otherwise-healthy pool simply has no
            // vision-capable models — refreshing cannot conjure them. Fail fast so
            // the chain moves to the next backend instead of forcing a network
            // refresh on every retry.
            if needs_vision && !self.pool_is_empty().await {
                return Err(InboxError::Llm(
                    "free-router has no vision-capable models".into(),
                ));
            }
            // Genuinely drained pool (e.g. a prior refresh returned zero healthy
            // models). complete() never reaches `maybe_refresh` on an empty pool,
            // so force a recovery refresh — bypassing the interval — and retry.
            warn!(
                needs_vision,
                "Free-router pool empty — forcing recovery refresh"
            );
            self.refresh_now().await;
            candidates = self.candidate_models(needs_tools, needs_vision).await;
            if candidates.is_empty() {
                return Err(InboxError::Llm(
                    "free-router pool is empty for this request (no matching models)".into(),
                ));
            }
        }

        debug!(
            needs_tools,
            needs_vision,
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
        let refreshed = self.candidate_models(needs_tools, needs_vision).await;
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
                tokio::time::sleep(super::retry_backoff(attempt)).await;
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
                // A transient outage (429/5xx/timeout) recurs immediately; abort
                // this model's retries so the batch moves on without burning the
                // per-model budget against a rate-limited model.
                Err(e) if !super::chain::is_service_available(&e) => {
                    debug!(?e, model = %model_id, "free-router model unavailable; aborting retries");
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
