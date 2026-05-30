//! Free-router model pool — fetching, scoring, partitioning.
//!
//! The pool is fed by the shir-man "top-models" API, which returns capability-
//! flagged `OpenRouter` models. We keep healthy ones only, assign each model an
//! *effective score* (API rank plus soft preferences from backend config), and
//! expose two ordered vectors: `tool_models` (tool-calling capable) and
//! `general_models` (superset; used when the call does not need tools).

use std::time::Duration;

use anodized::spec;
use serde::Deserialize;
use tracing::{debug, warn};

use crate::config::LlmBackendConfig;
use crate::error::InboxError;

/// Model ID used when the top-models endpoint is unreachable at startup.
pub const FALLBACK_MODEL_ID: &str = "openrouter/free";

/// Preference scoring weights. Each knob nudges ordering but never drops a
/// model — chosen so a single preference is worth less than the gap between
/// top and mid-ranked models from the API.
const PREF_BONUS: f64 = 50.0;

/// Upper bound on the optional `OpenRouter` modality fetch. The list fetch may use
/// a long timeout for slow CPU inference; the non-fatal join must not inherit it.
const VISION_FETCH_TIMEOUT_CAP: Duration = Duration::from_secs(5);

// The `supports_*` booleans mostly mirror the top-models API schema 1:1
// (`supportsTools`, `supportsToolChoice`, `supportsStructuredOutputs`,
// `supportsReasoning`). `supports_vision` is the exception: the index does not
// expose modality, so it is populated from OpenRouter `/api/v1/models`
// metadata after deserialization. Grouping these flags into a bitfield would
// make the deserializer diverge from the upstream JSON, so we opt out of the
// `struct_excessive_bools` pedantic lint for this type.
#[expect(
    clippy::struct_excessive_bools,
    reason = "Flags mirror the top-models API response; vision is joined from OpenRouter metadata."
)]
#[derive(Debug, Clone, Deserialize)]
pub struct FreeModel {
    pub id: String,
    #[serde(default)]
    pub score: f64,
    #[serde(default, rename = "contextLength")]
    pub context_length: usize,
    #[serde(default, rename = "supportsTools")]
    pub supports_tools: bool,
    #[serde(default, rename = "supportsToolChoice")]
    pub supports_tool_choice: bool,
    #[serde(default, rename = "supportsStructuredOutputs")]
    pub supports_structured_outputs: bool,
    #[serde(default, rename = "supportsReasoning")]
    pub supports_reasoning: bool,
    /// Accepts image input. Not from the top-models index — joined from
    /// `OpenRouter` `/api/v1/models` (`architecture.input_modalities`).
    #[serde(default, skip)]
    pub supports_vision: bool,
    #[serde(default, rename = "healthStatus")]
    pub health_status: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TopModelsResponse {
    #[serde(default)]
    pub models: Vec<FreeModel>,
}

/// Scoring preferences derived from `LlmBackendConfig`. Stored separately so
/// refresh paths do not have to re-read the whole backend config.
#[derive(Debug, Clone, Copy)]
pub struct PoolPreferences {
    pub min_context_length: usize,
    pub prefer_structured_outputs: bool,
    pub prefer_reasoning: bool,
}

impl From<&LlmBackendConfig> for PoolPreferences {
    fn from(cfg: &LlmBackendConfig) -> Self {
        Self {
            min_context_length: cfg.min_context_length,
            prefer_structured_outputs: cfg.prefer_structured_outputs,
            prefer_reasoning: cfg.prefer_reasoning,
        }
    }
}

/// Snapshot of the ordered model pool. Replaced atomically on refresh.
#[expect(
    clippy::struct_field_names,
    reason = "Each field is a distinct pool of models; the `_models` suffix is the clearest name."
)]
#[derive(Debug, Clone, Default)]
pub struct PoolState {
    pub tool_models: Vec<FreeModel>,
    pub general_models: Vec<FreeModel>,
    /// Subset of `general_models` that accept image input. Empty unless the
    /// `OpenRouter` modality join succeeded — never seeded with the degraded
    /// fallback alias, whose modality is unknown.
    pub vision_models: Vec<FreeModel>,
}

impl PoolState {
    /// Seed pool used when the top-models endpoint is unreachable at boot.
    /// Uses the router-managed `openrouter/free` alias so the daemon still runs.
    #[must_use]
    pub fn degraded_fallback() -> Self {
        let fallback = FreeModel {
            id: FALLBACK_MODEL_ID.into(),
            score: 0.0,
            context_length: 0,
            supports_tools: true,
            supports_tool_choice: true,
            supports_structured_outputs: false,
            supports_reasoning: false,
            supports_vision: false,
            health_status: "passed".into(),
        };
        Self {
            tool_models: vec![fallback.clone()],
            general_models: vec![fallback],
            // The fallback alias has no known modality — leaving the vision pool
            // empty makes vision requests skip free-router rather than risk a
            // blind image dispatch.
            vision_models: vec![],
        }
    }

    /// True if both pools are empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tool_models.is_empty() && self.general_models.is_empty()
    }
}

/// Fetch the top-models list, join `OpenRouter` modality metadata, and build an
/// ordered `PoolState`. `base_url` is the `OpenRouter` API root (e.g.
/// `https://openrouter.ai/api/v1`); its `/models` endpoint supplies vision
/// capability. A failed modality join is non-fatal — models stay non-vision.
///
/// # Errors
/// Returns an error if the top-models request fails or cannot be parsed.
#[spec(requires: !api_url.trim().is_empty())]
pub async fn fetch_pool(
    client: &reqwest::Client,
    api_url: &str,
    base_url: &str,
    timeout: Duration,
    prefs: PoolPreferences,
) -> Result<PoolState, InboxError> {
    let resp = client
        .get(api_url)
        .timeout(timeout)
        .send()
        .await
        .map_err(|e| InboxError::Llm(format!("free-router list fetch failed: {e}")))?;

    if !resp.status().is_success() {
        return Err(InboxError::Llm(format!(
            "free-router list HTTP {}: {}",
            resp.status(),
            resp.text().await.unwrap_or_default()
        )));
    }

    let body: TopModelsResponse = resp
        .json()
        .await
        .map_err(|e| InboxError::Llm(format!("free-router list parse error: {e}")))?;

    let mut models = body.models;
    // The modality join is optional and non-fatal; cap its timeout so a slow
    // /models endpoint never gates pool availability for text-only routing.
    let vision_timeout = timeout.min(VISION_FETCH_TIMEOUT_CAP);
    let vision_ids = fetch_vision_model_ids(client, base_url, vision_timeout).await;
    for m in &mut models {
        m.supports_vision = vision_ids.contains(&m.id);
    }

    Ok(build_pool(models, prefs))
}

// ── OpenRouter modality metadata ──────────────────────────────────────────────

#[derive(Deserialize)]
struct OpenRouterModels {
    #[serde(default)]
    data: Vec<OpenRouterModel>,
}

#[derive(Deserialize)]
struct OpenRouterModel {
    id: String,
    #[serde(default)]
    architecture: OpenRouterArch,
}

#[derive(Deserialize, Default)]
struct OpenRouterArch {
    #[serde(default)]
    input_modalities: Vec<String>,
}

/// Return the set of `OpenRouter` model ids that accept image input, read from
/// `{base_url}/models` (`architecture.input_modalities` contains `"image"`).
///
/// Non-fatal by contract: any transport, status, or parse failure yields an
/// empty set (logged), so a missing join simply leaves the vision pool empty.
pub async fn fetch_vision_model_ids(
    client: &reqwest::Client,
    base_url: &str,
    timeout: Duration,
) -> std::collections::HashSet<String> {
    let url = format!("{}/models", base_url.trim_end_matches('/'));
    let result = client.get(&url).timeout(timeout).send().await;
    let resp = match result {
        Ok(r) if r.status().is_success() => r,
        Ok(r) => {
            warn!(status = %r.status(), "free-router vision metadata HTTP error");
            return std::collections::HashSet::new();
        }
        Err(e) => {
            warn!(?e, "free-router vision metadata fetch failed");
            return std::collections::HashSet::new();
        }
    };
    match resp.json::<OpenRouterModels>().await {
        Ok(body) => body
            .data
            .into_iter()
            .filter(|m| m.architecture.input_modalities.iter().any(|s| s == "image"))
            .map(|m| m.id)
            .collect(),
        Err(e) => {
            warn!(?e, "free-router vision metadata parse failed");
            std::collections::HashSet::new()
        }
    }
}

/// Partition + sort healthy models into the two ordered pools.
#[must_use]
pub fn build_pool(models: Vec<FreeModel>, prefs: PoolPreferences) -> PoolState {
    let mut healthy: Vec<FreeModel> = models
        .into_iter()
        .filter(|m| m.health_status.eq_ignore_ascii_case("passed"))
        .collect();

    healthy.sort_by(|a, b| {
        score_model(b, prefs)
            .partial_cmp(&score_model(a, prefs))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let tool_models: Vec<FreeModel> = healthy
        .iter()
        .filter(|m| m.supports_tools && m.supports_tool_choice)
        .cloned()
        .collect();

    let vision_models: Vec<FreeModel> = healthy
        .iter()
        .filter(|m| m.supports_vision)
        .cloned()
        .collect();

    let general_models = healthy;

    debug!(
        tool_count = tool_models.len(),
        general_count = general_models.len(),
        vision_count = vision_models.len(),
        "Free-router pool built"
    );

    if tool_models.is_empty() && !general_models.is_empty() {
        warn!(
            "Free-router pool has no tool-capable models; tool calls will fall back to general pool"
        );
    }

    PoolState {
        tool_models,
        general_models,
        vision_models,
    }
}

/// Compute the effective score for a model: API score plus soft-preference bonuses.
/// Preferences never drop a model, only reorder.
#[must_use]
pub fn score_model(m: &FreeModel, prefs: PoolPreferences) -> f64 {
    let mut s = m.score;
    if prefs.min_context_length > 0 && m.context_length >= prefs.min_context_length {
        s += PREF_BONUS;
    }
    if prefs.prefer_structured_outputs && m.supports_structured_outputs {
        s += PREF_BONUS;
    }
    if prefs.prefer_reasoning && m.supports_reasoning {
        s += PREF_BONUS;
    }
    s
}
