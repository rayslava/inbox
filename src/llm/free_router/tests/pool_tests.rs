//! Pure-logic tests: list parsing, scoring, pool partitioning.

use serde_json::json;

use crate::llm::free_router::pool::{PoolPreferences, TopModelsResponse, build_pool, score_model};

use super::fixtures::sample_model;

#[test]
fn build_pool_drops_unhealthy_and_partitions_tool_models() {
    let models = vec![
        sample_model("ok/tool", 1000.0, 16_000, true, false, false, "passed"),
        sample_model(
            "skip/unhealthy",
            2000.0,
            16_000,
            true,
            false,
            false,
            "imperfect",
        ),
        sample_model("ok/no-tool", 900.0, 16_000, false, false, false, "passed"),
    ];
    let prefs = PoolPreferences {
        min_context_length: 0,
        prefer_structured_outputs: false,
        prefer_reasoning: false,
    };
    let pool = build_pool(models, prefs);
    assert_eq!(pool.tool_models.len(), 1);
    assert_eq!(pool.tool_models[0].id, "ok/tool");
    assert_eq!(pool.general_models.len(), 2);
    assert!(!pool.general_models.iter().any(|m| m.id == "skip/unhealthy"));
}

#[test]
fn score_model_preferences_reorder_never_drop() {
    let prefs = PoolPreferences {
        min_context_length: 32_000,
        prefer_structured_outputs: true,
        prefer_reasoning: false,
    };
    let m_plain = sample_model("plain", 1000.0, 8_000, true, false, false, "passed");
    let m_preferred = sample_model("preferred", 1000.0, 64_000, true, true, false, "passed");
    assert!(score_model(&m_preferred, prefs) > score_model(&m_plain, prefs));
    // Neither is dropped — both remain in the pool after build.
    let pool = build_pool(vec![m_plain.clone(), m_preferred.clone()], prefs);
    assert_eq!(pool.general_models.len(), 2);
    // Preferred is ranked first due to bonus.
    assert_eq!(pool.general_models[0].id, "preferred");
}

#[test]
fn score_model_reasoning_preference_bonus() {
    let prefs = PoolPreferences {
        min_context_length: 0,
        prefer_structured_outputs: false,
        prefer_reasoning: true,
    };
    let plain = sample_model("plain", 1000.0, 16_000, true, false, false, "passed");
    let reasoning = sample_model("reasoning", 1000.0, 16_000, true, false, true, "passed");
    assert!(score_model(&reasoning, prefs) > score_model(&plain, prefs));
}

#[test]
fn parse_top_models_payload() {
    let payload = json!({
        "models": [
            {
                "id": "inclusionai/ling-2.6-flash:free",
                "score": 1060,
                "contextLength": 262_144,
                "supportsTools": true,
                "supportsToolChoice": true,
                "supportsStructuredOutputs": true,
                "supportsReasoning": false,
                "healthStatus": "passed"
            },
            {
                "id": "dying/model",
                "score": 10,
                "contextLength": 2048,
                "supportsTools": false,
                "supportsToolChoice": false,
                "supportsStructuredOutputs": false,
                "supportsReasoning": false,
                "healthStatus": "imperfect"
            }
        ]
    });
    let parsed: TopModelsResponse = serde_json::from_value(payload).unwrap();
    assert_eq!(parsed.models.len(), 2);
    let prefs = PoolPreferences {
        min_context_length: 0,
        prefer_structured_outputs: false,
        prefer_reasoning: false,
    };
    let pool = build_pool(parsed.models, prefs);
    assert_eq!(pool.tool_models.len(), 1);
    assert_eq!(pool.tool_models[0].id, "inclusionai/ling-2.6-flash:free");
}
