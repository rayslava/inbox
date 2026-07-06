use super::*;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

fn make_client(base_url: &str) -> OpenRouterClient {
    make_labeled_client(base_url, "openrouter")
}

fn make_labeled_client(base_url: &str, label: &'static str) -> OpenRouterClient {
    OpenRouterClient {
        model: "test-model".into(),
        api_key: "test-key".into(),
        base_url: base_url.to_owned(),
        retries: 1,
        timeout: std::time::Duration::from_secs(5),
        label,
        vision_supported: false,
        semaphore: None,
        client: reqwest::Client::new(),
        circuit: crate::llm::CircuitBreaker::new(0),
    }
}

#[tokio::test]
async fn llama_cpp_label_flows_to_name_and_provenance() {
    let server = MockServer::start().await;
    let body = serde_json::json!({
        "choices": [{ "message": { "content": r#"{"title":"T","tags":[],"summary":"S"}"# } }]
    });
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(&server)
        .await;

    let client = make_labeled_client(&server.uri(), "llama_cpp");
    assert_eq!(client.name(), "llama_cpp");
    let result = client
        .complete(LlmRequest::simple("sys", "user"))
        .await
        .unwrap();
    match result {
        LlmCompletion::Message(m) => {
            // `:ENRICHED_BY:` provenance carries the label, not "openrouter".
            assert_eq!(m.produced_by, "llama_cpp:test-model");
        }
        LlmCompletion::ToolCalls(_) => panic!("expected message, got tool calls"),
    }
}

#[test]
fn from_config_labeled_sets_llama_cpp_name() {
    let cfg = crate::config::LlmBackendConfig {
        backend_type: crate::config::LlmBackendType::LlamaCpp,
        model: "qwen2.5".into(),
        api_key: None,
        base_url: "http://localhost:8080/v1".into(),
        retries: 1,
        timeout_secs: 30,
        think: None,
        think_timeout_secs: None,
        thinking_supported: false,
        vision_supported: false,
        max_concurrent: None,
        context_size: None,
        format: None,
        connect_timeout_secs: 10,
        circuit_open_secs: 0,
        api_url: String::new(),
        parallel_fanout: 3,
        per_model_retries: 2,
        min_refresh_interval_secs: 300,
        min_context_length: 0,
        prefer_structured_outputs: false,
        prefer_reasoning: false,
    };
    let client = OpenRouterClient::from_config_labeled(&cfg, "llama_cpp").expect("build");
    assert_eq!(client.name(), "llama_cpp");
    assert_eq!(client.model(), "qwen2.5");
}

#[test]
fn parse_json_response_full() {
    let json = r#"{"title":"T","tags":["a","b"],"summary":"S","excerpt":"E"}"#;
    let r = parse_llm_json_response(json, "test").unwrap();
    assert_eq!(r.title, "T");
    assert_eq!(r.tags, vec!["a", "b"]);
    assert_eq!(r.summary, "S");
    assert_eq!(r.excerpt.as_deref(), Some("E"));
    assert_eq!(r.produced_by, "test");
}

#[test]
fn parse_json_tolerates_raw_control_chars_in_strings() {
    // saiga et al. emit a fenced object with LITERAL newlines/tabs inside the
    // string value, which strict serde_json rejects; the lenient path recovers.
    let json = "```json\n{\n  \"summary\": \"line one\n\tline two\"\n}\n```";
    let r = parse_llm_json_response(json, "llama_cpp").unwrap();
    assert!(r.summary.contains("line one"));
    assert!(r.summary.contains("line two"));
}

#[test]
fn parse_json_control_chars_recovered_after_surrounding_text() {
    // Control chars inside the string AND extra prose around the object.
    let json = "Here you go:\n{\"title\":\"T\",\"summary\":\"a\nb\"}\nThanks!";
    let r = parse_llm_json_response(json, "x").unwrap();
    assert_eq!(r.title, "T");
    assert!(r.summary.contains('a') && r.summary.contains('b'));
}

#[test]
fn parse_json_still_errors_on_non_json() {
    assert!(parse_llm_json_response("not json at all, sorry", "x").is_err());
}

#[test]
fn parse_json_backslash_before_raw_control_is_fail_safe() {
    // A backslash immediately before a raw control char inside a string is not
    // recoverable (the char is consumed by the escape branch verbatim) — but it
    // fails safely (no corruption), same as before the lenient path existed.
    let json = "{\"summary\":\"ab\\\nx\"}";
    assert!(parse_llm_json_response(json, "x").is_err());
}

#[test]
fn parse_json_escapes_all_control_char_kinds_with_surrounding_text() {
    // \r, \t, and an arbitrary control char () inside the string, plus
    // prose around the object → forces the escape-then-extract fallback path.
    let json = "prose here\n{\"summary\":\"a\rb\tc\u{0001}d\"}\ntrailing";
    let r = parse_llm_json_response(json, "x").unwrap();
    assert!(r.summary.contains('a') && r.summary.contains('d'));
}

#[test]
fn parse_json_unwraps_single_element_array() {
    // Some models (e.g. gemma) wrap the object in a JSON array.
    let json = r#"[{"title":"T","tags":["a"],"summary":"S","excerpt":"E"}]"#;
    let r = parse_llm_json_response(json, "x").unwrap();
    assert_eq!(r.title, "T");
    assert_eq!(r.tags, vec!["a"]);
    assert_eq!(r.summary, "S");
    assert_eq!(r.excerpt.as_deref(), Some("E"));
}

#[test]
fn parse_json_array_skips_non_objects() {
    let json = r#"["junk", {"title":"T","tags":[],"summary":"S"}]"#;
    let r = parse_llm_json_response(json, "x").unwrap();
    assert_eq!(r.title, "T");
}

#[test]
fn parse_json_strips_markdown_fences() {
    let json = "```json\n{\"title\":\"T\",\"summary\":\"S\",\"tags\":[]}\n```";
    let r = parse_llm_json_response(json, "x").unwrap();
    assert_eq!(r.title, "T");
}

#[test]
fn parse_json_strips_bare_fences() {
    let json = "```\n{\"title\":\"T\",\"summary\":\"S\",\"tags\":[]}\n```";
    let r = parse_llm_json_response(json, "x").unwrap();
    assert_eq!(r.title, "T");
}

#[test]
fn parse_json_missing_fields_defaults() {
    let json = r"{}";
    let r = parse_llm_json_response(json, "x").unwrap();
    assert_eq!(r.title, "(no title)");
    assert!(r.tags.is_empty());
    assert_eq!(r.summary, "");
    assert!(r.excerpt.is_none());
}

#[test]
fn parse_json_invalid_returns_error() {
    let result = parse_llm_json_response("not json", "x");
    assert!(result.is_err());
}

#[test]
fn parse_json_strips_think_tag_trailing() {
    // qwen3.5 sometimes emits </think> after the JSON object
    let text = r#"{"title":"T","tags":[],"summary":"S"}
</think>"#;
    let r = parse_llm_json_response(text, "x").unwrap();
    assert_eq!(r.title, "T");
}

#[test]
fn parse_json_strips_duplicate_json_after_think_tag() {
    // Model emits </think> then a duplicate JSON object
    let text = "```json\n{\"title\":\"T\",\"tags\":[],\"summary\":\"S\"}\n</think>\n{\"title\":\"T2\",\"tags\":[],\"summary\":\"S2\"}\n```";
    let r = parse_llm_json_response(text, "x").unwrap();
    assert_eq!(r.title, "T");
    assert_eq!(r.summary, "S");
}

#[test]
fn parse_json_extracts_first_object_from_preamble() {
    // Preamble text before the JSON object
    let text = "Here is the result:\n{\"title\":\"T\",\"tags\":[],\"summary\":\"S\"}";
    let r = parse_llm_json_response(text, "x").unwrap();
    assert_eq!(r.title, "T");
}

#[test]
fn parse_json_truly_malformed_returns_error() {
    let result = parse_llm_json_response("no braces here at all", "x");
    assert!(result.is_err());
}

#[tokio::test]
async fn empty_api_key_omits_authorization_header() {
    let server = MockServer::start().await;
    let body = serde_json::json!({
        "choices": [{ "message": { "content": r#"{"title":"T","tags":[],"summary":"S"}"# } }]
    });
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(|req: &Request| !req.headers.contains_key("authorization"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(&server)
        .await;

    // A llama.cpp backend configured with no key must not send `Bearer `.
    let mut client = make_labeled_client(&server.uri(), "llama_cpp");
    client.api_key = String::new();
    let result = client.complete(LlmRequest::simple("sys", "user")).await;
    assert!(result.is_ok(), "no-key request should succeed: {result:?}");
}

#[tokio::test]
async fn non_empty_api_key_sends_bearer_header() {
    let server = MockServer::start().await;
    let body = serde_json::json!({
        "choices": [{ "message": { "content": r#"{"title":"T","tags":[],"summary":"S"}"# } }]
    });
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("authorization", "Bearer test-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(&server)
        .await;

    let client = make_client(&server.uri()); // api_key = "test-key"
    let result = client.complete(LlmRequest::simple("sys", "user")).await;
    assert!(result.is_ok(), "keyed request should succeed: {result:?}");
}

#[tokio::test]
async fn complete_success() {
    let server = MockServer::start().await;
    let body = serde_json::json!({
        "choices": [{
            "message": {
                "content": r#"{"title":"T","tags":[],"summary":"S"}"#
            }
        }]
    });
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
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
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(500).set_body_string("oops"))
        .mount(&server)
        .await;

    let client = make_client(&server.uri());
    let req = LlmRequest::simple("sys", "user");
    let result = client.complete(req).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn complete_empty_choices_error() {
    let server = MockServer::start().await;
    let body = serde_json::json!({ "choices": [] });
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(body),
        )
        .mount(&server)
        .await;

    let client = make_client(&server.uri());
    let req = LlmRequest::simple("sys", "user");
    let result = client.complete(req).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn complete_tool_calls() {
    let server = MockServer::start().await;
    let body = serde_json::json!({
        "choices": [{
            "message": {
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": { "name": "scrape_page", "arguments": "{\"url\":\"http://example.com\"}" }
                }]
            }
        }]
    });
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
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

/// Build a client with an active cooldown window so a transient failure makes
/// `is_available` flip to `false`.
fn make_client_with_cooldown(base_url: &str) -> OpenRouterClient {
    OpenRouterClient {
        model: "test-model".into(),
        api_key: "test-key".into(),
        base_url: base_url.to_owned(),
        retries: 1,
        timeout: std::time::Duration::from_secs(5),
        label: "openrouter",
        vision_supported: false,
        semaphore: None,
        client: reqwest::Client::new(),
        circuit: crate::llm::CircuitBreaker::new(300),
    }
}

#[tokio::test]
async fn rate_limit_trips_cooldown_and_short_circuits_next_call() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(429).set_body_string("rate limited"))
        .mount(&server)
        .await;

    let client = make_client_with_cooldown(&server.uri());
    assert!(client.is_available());

    // First call hits the 429 and trips the cooldown.
    let first = client.complete(LlmRequest::simple("s", "u")).await;
    assert!(first.is_err());
    assert!(!client.is_available(), "429 must open the circuit");

    // Second call is short-circuited by the open circuit — no HTTP request.
    let second = client.complete(LlmRequest::simple("s", "u")).await;
    match second {
        Err(crate::error::InboxError::Llm(m)) => {
            assert!(m.contains("circuit open"), "{m}");
        }
        other => panic!("expected circuit-open error, got {other:?}"),
    }
}

#[tokio::test]
async fn complete_with_images_sends_array_content() {
    use wiremock::matchers::body_partial_json;

    let server = MockServer::start().await;
    let body = serde_json::json!({
        "choices": [{ "message": { "content": r#"{"title":"T","tags":[],"summary":"S"}"# } }]
    });
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        // The user message content must be an array when images are present.
        .and(body_partial_json(serde_json::json!({
            "messages": [
                { "role": "system" },
                { "role": "user", "content": [{ "type": "text" }] }
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
    let mut req = LlmRequest::simple("sys", "user text");
    req.images = vec![("image/png".into(), "aGVsbG8=".into())];
    let result = client.complete(req).await.unwrap();
    assert!(matches!(result, LlmCompletion::Message(_)));
}

#[tokio::test]
async fn complete_rate_limited_returns_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(429)
                .set_body_string(r#"{"error":{"message":"Rate limit exceeded"}}"#),
        )
        .mount(&server)
        .await;

    let result = make_client(&server.uri())
        .complete(LlmRequest::simple("s", "u"))
        .await;
    assert!(result.is_err());
    assert!(
        result.unwrap_err().to_string().contains("429"),
        "error message should contain status code"
    );
}

#[tokio::test]
async fn complete_malformed_json_response_returns_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_string("this is not json"),
        )
        .mount(&server)
        .await;

    let result = make_client(&server.uri())
        .complete(LlmRequest::simple("s", "u"))
        .await;
    assert!(result.is_err());
}
