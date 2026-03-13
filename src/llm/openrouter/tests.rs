use super::*;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn make_client(base_url: &str) -> OpenRouterClient {
    OpenRouterClient {
        model: "test-model".into(),
        api_key: "test-key".into(),
        base_url: base_url.to_owned(),
        retries: 1,
        timeout: std::time::Duration::from_secs(5),
        semaphore: None,
        client: reqwest::Client::new(),
    }
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
