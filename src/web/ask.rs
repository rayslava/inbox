//! Local private second-brain query endpoint (`POST /ask`). Served on a
//! **dedicated loopback listener** (default 127.0.0.1), NOT on the admin server
//! — the admin server defaults to all interfaces, and `/ask` reads the full
//! private KB with no auth, so it must never ride a public bind. Used by Emacs
//! and local HTTP callers.

use std::sync::Arc;

use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::post,
};
use serde::{Deserialize, Serialize};

use crate::llm::LlmChain;
use crate::memory::MemoryStore;

const DEFAULT_TOP_K: usize = 6;
const MAX_TOP_K: usize = 50;

/// State for the local `/ask` server: the shared store + LLM chain.
#[derive(Clone)]
pub struct AskState {
    pub memory_store: Option<Arc<MemoryStore>>,
    pub llm: Arc<LlmChain>,
}

/// Router for the local ask server — serves only `POST /ask`.
pub fn ask_router(state: AskState) -> Router {
    Router::new()
        .route("/ask", post(ask_handler))
        .with_state(state)
}

#[derive(Deserialize)]
pub struct AskRequest {
    pub question: String,
    #[serde(default)]
    pub top_k: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct AskResponse {
    /// The answer text alone.
    pub answer: String,
    /// Cited note ids (org-roam `:ID:`s), deterministic from retrieval.
    pub note_ids: Vec<String>,
    /// Answer plus a `Sources:` list of `[[id:...]]` links, ready for org.
    pub org: String,
}

/// Core `/ask` logic: validate, run the brain, shape the response. Returns an
/// `(status, message)` on any client/serving error.
pub(crate) async fn run_ask(
    memory_store: Option<&Arc<MemoryStore>>,
    llm: &LlmChain,
    question: &str,
    top_k: Option<usize>,
) -> Result<AskResponse, (StatusCode, String)> {
    if question.trim().is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "question must not be empty".to_owned(),
        ));
    }
    let Some(store) = memory_store else {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "memory/brain is disabled".to_owned(),
        ));
    };
    let k = top_k.unwrap_or(DEFAULT_TOP_K).clamp(1, MAX_TOP_K);

    let answer = inbox_core::brain::answer(store.as_ref(), llm, question, k)
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("brain error: {e}"),
            )
        })?;

    Ok(AskResponse {
        org: answer.to_org(),
        answer: answer.text,
        note_ids: answer.note_ids,
    })
}

pub(crate) async fn ask_handler(
    State(state): State<AskState>,
    Json(req): Json<AskRequest>,
) -> Response {
    match run_ask(
        state.memory_store.as_ref(),
        &state.llm,
        &req.question,
        req.top_k,
    )
    .await
    {
        Ok(resp) => Json(resp).into_response(),
        Err((code, msg)) => (code, msg).into_response(),
    }
}

#[cfg(all(test, feature = "test-helpers"))]
mod tests {
    use super::{AskState, MemoryStore, ask_router, run_ask};
    use crate::kb_index;
    use crate::message::LlmResponse;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use std::sync::Arc;
    use tower::ServiceExt;

    fn mock_chain(summary: &str) -> Arc<crate::llm::LlmChain> {
        crate::test_helpers::mock_llm_chain(LlmResponse {
            title: String::new(),
            tags: vec![],
            summary: summary.to_owned(),
            excerpt: None,
            produced_by: "mock".to_owned(),
        })
    }

    #[tokio::test]
    async fn empty_question_is_bad_request() {
        let chain = mock_chain("x");
        let err = run_ask(None, &chain, "   ", None).await.unwrap_err();
        assert_eq!(err.0, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn missing_memory_is_unavailable() {
        let chain = mock_chain("x");
        let err = run_ask(None, &chain, "hi", None).await.unwrap_err();
        assert_eq!(err.0, StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn ask_router_returns_answer_json_with_citation() {
        let store = Arc::new(MemoryStore::new_in_memory().expect("store"));
        kb_index::index_content(&store, "org", "n1", "/n1.org", "* T\nthe sky is blue\n")
            .await
            .expect("index");
        let router = ask_router(AskState {
            memory_store: Some(store),
            llm: mock_chain("Blue."),
        });

        let req = Request::builder()
            .method("POST")
            .uri("/ask")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"question":"what color is the sky"}"#))
            .unwrap();
        let resp = router.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
            .await
            .unwrap();
        let body = String::from_utf8_lossy(&bytes);
        assert!(body.contains("Blue"), "answer body: {body}");
        assert!(body.contains("[[id:n1]]"), "citation missing: {body}");
    }

    #[tokio::test]
    async fn ask_router_503_when_memory_disabled() {
        let router = ask_router(AskState {
            memory_store: None,
            llm: mock_chain("x"),
        });
        let req = Request::builder()
            .method("POST")
            .uri("/ask")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"question":"hi"}"#))
            .unwrap();
        let resp = router.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    }
}
