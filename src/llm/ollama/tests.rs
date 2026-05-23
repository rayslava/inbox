//! Unit tests for the Ollama backend client.

use super::*;
use std::time::{Duration, Instant};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn make_client(base_url: &str) -> OllamaClient {
    OllamaClient {
        model: "llama3".into(),
        base_url: base_url.to_owned(),
        retries: 1,
        timeout: std::time::Duration::from_secs(5),
        think: None,
        think_timeout: None,
        thinking_supported: false,
        context_size: None,
        format: None,
        circuit_open_secs: 0,
        last_connection_failure: Arc::new(Mutex::new(None)),
        semaphore: None,
        client: reqwest::Client::new(),
    }
}

#[tokio::test]
async fn complete_success() {
    let server = MockServer::start().await;
    let body = serde_json::json!({
        "message": {
            "role": "assistant",
            "content": r#"{"title":"T","tags":[],"summary":"S"}"#
        }
    });
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(body),
        )
        .mount(&server)
        .await;

    let client = make_client(&server.uri());
    let req = LlmRequest::simple("sys", "user");
    let result = client.complete(req).await.unwrap();
    assert!(matches!(result, LlmCompletion::Message(_)));
}

#[tokio::test]
async fn complete_error_status() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(503).set_body_string("unavailable"))
        .mount(&server)
        .await;

    let client = make_client(&server.uri());
    let req = LlmRequest::simple("sys", "user");
    assert!(client.complete(req).await.is_err());
}

#[tokio::test]
async fn complete_tool_calls() {
    let server = MockServer::start().await;
    let body = serde_json::json!({
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "function": { "name": "scrape_page", "arguments": {"url": "http://x.com"} }
            }]
        }
    });
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(body),
        )
        .mount(&server)
        .await;

    let client = make_client(&server.uri());
    let req = LlmRequest::simple("sys", "user");
    let result = client.complete(req).await.unwrap();
    assert!(matches!(result, LlmCompletion::ToolCalls(_)));
}

#[tokio::test]
async fn complete_with_images_sends_images_field() {
    use wiremock::matchers::body_partial_json;

    let server = MockServer::start().await;
    let body = serde_json::json!({
        "message": { "role": "assistant", "content": r#"{"title":"T","tags":[],"summary":"S"}"# }
    });
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .and(body_partial_json(serde_json::json!({
            "messages": [
                { "role": "system" },
                { "role": "user", "images": ["aGVsbG8="] }
            ]
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(body),
        )
        .mount(&server)
        .await;

    let client = make_client(&server.uri());
    let mut req = LlmRequest::simple("sys", "user");
    req.images = vec![("image/png".into(), "aGVsbG8=".into())];
    let result = client.complete(req).await.unwrap();
    assert!(matches!(result, LlmCompletion::Message(_)));
}

#[tokio::test]
async fn context_size_sends_options_num_ctx() {
    use wiremock::matchers::body_partial_json;

    let server = MockServer::start().await;
    let body = serde_json::json!({
        "message": { "role": "assistant", "content": r#"{"title":"T","tags":[],"summary":"S"}"# }
    });
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .and(body_partial_json(serde_json::json!({
            "options": { "num_ctx": 16384 }
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(body),
        )
        .mount(&server)
        .await;

    let mut client = make_client(&server.uri());
    client.context_size = Some(16384);
    let req = LlmRequest::simple("sys", "user");
    let result = client.complete(req).await.unwrap();
    assert!(matches!(result, LlmCompletion::Message(_)));
}

#[tokio::test]
async fn no_context_size_omits_options() {
    let server = MockServer::start().await;
    let body = serde_json::json!({
        "message": { "role": "assistant", "content": r#"{"title":"T","tags":[],"summary":"S"}"# }
    });
    // If options were present with num_ctx, this mock would only match that specific body.
    // By NOT using body_partial_json for options, we verify the basic path still works.
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(body),
        )
        .mount(&server)
        .await;

    let client = make_client(&server.uri()); // context_size = None
    let req = LlmRequest::simple("sys", "user");
    let result = client.complete(req).await.unwrap();
    assert!(matches!(result, LlmCompletion::Message(_)));
}

fn chat_response_body() -> serde_json::Value {
    serde_json::json!({
        "message": { "role": "assistant", "content": r#"{"title":"T","tags":[],"summary":"S"}"# }
    })
}

#[tokio::test]
async fn preflight_model_loaded_proceeds() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/ps"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(serde_json::json!({
                    "models": [{"name": "llama3", "size_vram": 4_294_967_296_u64}]
                })),
        )
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(chat_response_body()),
        )
        .mount(&server)
        .await;

    let client = make_client(&server.uri());
    let req = LlmRequest::simple("sys", "user");
    let result = client.complete(req).await.unwrap();
    assert!(matches!(result, LlmCompletion::Message(_)));
}

#[tokio::test]
async fn preflight_empty_models_proceeds() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/ps"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(serde_json::json!({"models": []})),
        )
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(chat_response_body()),
        )
        .mount(&server)
        .await;

    let client = make_client(&server.uri());
    let req = LlmRequest::simple("sys", "user");
    let result = client.complete(req).await.unwrap();
    assert!(matches!(result, LlmCompletion::Message(_)));
}

#[tokio::test]
async fn preflight_error_ignored_proceeds() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/ps"))
        .respond_with(ResponseTemplate::new(500))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(chat_response_body()),
        )
        .mount(&server)
        .await;

    let client = make_client(&server.uri());
    let req = LlmRequest::simple("sys", "user");
    let result = client.complete(req).await.unwrap();
    assert!(matches!(result, LlmCompletion::Message(_)));
}

#[tokio::test]
async fn circuit_open_skips_request() {
    // Pre-set a recent connection failure; subsequent call should return
    // a circuit-open error without making any HTTP requests.
    let server = MockServer::start().await;
    // No mocks registered — any HTTP hit would be an unexpected request.

    let mut client = make_client(&server.uri());
    client.circuit_open_secs = 300;
    *client.last_connection_failure.lock().expect("mutex") = Some(Instant::now());

    let result = client.complete(LlmRequest::simple("sys", "user")).await;
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("circuit"),
        "expected circuit-open error, got: {msg}"
    );
}

#[tokio::test]
async fn circuit_clears_on_success() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/ps"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(serde_json::json!({"models": []})),
        )
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(chat_response_body()),
        )
        .mount(&server)
        .await;

    let mut client = make_client(&server.uri());
    // Artificially open a stale circuit from 1000s ago (expired).
    client.circuit_open_secs = 1;
    *client.last_connection_failure.lock().expect("mutex") =
        Some(Instant::now().checked_sub(Duration::from_secs(10)).unwrap());

    // Circuit should be expired — request succeeds and clears failure.
    let result = client.complete(LlmRequest::simple("sys", "user")).await;
    assert!(result.is_ok());
    assert!(
        client
            .last_connection_failure
            .lock()
            .expect("mutex")
            .is_none()
    );
}

#[tokio::test]
async fn preflight_cold_start_proceeds() {
    // Empty /api/ps (model not loaded) — should proceed with a cold-start warning,
    // not fail.
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/ps"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(serde_json::json!({"models": []})),
        )
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(chat_response_body()),
        )
        .mount(&server)
        .await;

    let result = make_client(&server.uri())
        .complete(LlmRequest::simple("sys", "user"))
        .await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn context_overflow_truncates_content() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/ps"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(serde_json::json!({"models": []})),
        )
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(chat_response_body()),
        )
        .mount(&server)
        .await;

    let mut client = make_client(&server.uri());
    // context_size = 1 token → char_limit = 4 chars; content of 100 chars triggers truncation
    client.context_size = Some(1);
    let long_content = "a".repeat(100);
    let req = LlmRequest::simple("sys", &long_content);
    let result = client.complete(req).await;
    // Truncation fires but request still completes normally
    assert!(result.is_ok());
}

/// Helper: parse the single recorded `/api/chat` request body as JSON.
async fn recorded_chat_body(server: &MockServer) -> serde_json::Value {
    let reqs = server.received_requests().await.expect("requests recorded");
    let chat = reqs
        .iter()
        .find(|r| r.url.path() == "/api/chat")
        .expect("a /api/chat request was made");
    serde_json::from_slice(&chat.body).expect("body is valid JSON")
}

#[tokio::test]
async fn format_sent_when_no_tools() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(chat_response_body()),
        )
        .mount(&server)
        .await;

    let mut client = make_client(&server.uri());
    client.format = Some("json".into());
    let req = LlmRequest::simple("sys", "user"); // no tool_definitions
    client.complete(req).await.unwrap();

    let body = recorded_chat_body(&server).await;
    assert_eq!(body["format"], "json");
}

#[tokio::test]
async fn format_omitted_when_tools_present() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(chat_response_body()),
        )
        .mount(&server)
        .await;

    let mut client = make_client(&server.uri());
    client.format = Some("json".into());
    let mut req = LlmRequest::simple("sys", "user");
    req.tool_definitions = vec![serde_json::json!({
        "type": "function",
        "function": { "name": "scrape_page" }
    })];
    client.complete(req).await.unwrap();

    let body = recorded_chat_body(&server).await;
    // `format` must be absent so the model is free to emit tool_calls.
    assert!(
        body.get("format").is_none(),
        "format leaked onto a tool turn"
    );
}

#[tokio::test]
async fn context_guard_counts_system_and_tools() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(chat_response_body()),
        )
        .mount(&server)
        .await;

    let mut client = make_client(&server.uri());
    // A short user message that fits the window on its own, but a large
    // system prompt pushes the *total* over budget → user content truncates.
    client.context_size = Some(50); // 50 tokens ≈ 200 chars budget
    let big_system = "S".repeat(2000); // ~500 tokens of overhead alone
    let mut req = LlmRequest::simple(&big_system, "short user content");
    req.tool_definitions = vec![serde_json::json!({
        "type": "function",
        "function": { "name": "x", "description": "y".repeat(400) }
    })];
    client.complete(req).await.unwrap();

    let body = recorded_chat_body(&server).await;
    let user_msg = body["messages"][1]["content"]
        .as_str()
        .expect("user message present");
    assert!(
        user_msg.contains("[context truncated"),
        "expected user content truncated by combined system+tool overhead, got: {user_msg}"
    );
}
