//! Tests for the error classifiers — `is_service_unavailable` (transient) and
//! its non-overlap with `is_deterministic_error`.

use std::time::Duration;

use crate::error::InboxError;

use super::classify::{is_deterministic_error, is_service_unavailable};

fn llm(msg: &str) -> InboxError {
    InboxError::Llm(msg.to_owned())
}

#[test]
fn http_5xx_and_429_are_service_unavailable() {
    for code in [
        "429 Too Many Requests",
        "500 Internal Server Error",
        "502 Bad Gateway",
        "503 Service Unavailable",
        "504 Gateway Timeout",
    ] {
        let e = llm(&format!("free_router API error {code}: upstream said no"));
        assert!(is_service_unavailable(&e), "should be transient: {code}");
    }
    assert!(is_service_unavailable(&llm(
        "Ollama API error 503 Service Unavailable: down"
    )));
}

#[test]
fn circuit_open_is_service_unavailable() {
    assert!(is_service_unavailable(&llm(
        "Ollama circuit open: backend unreachable, retry in 42s"
    )));
}

#[test]
fn non_transient_statuses_are_not() {
    for code in [
        "400 Bad Request",
        "401 Unauthorized",
        "403 Forbidden",
        "404 Not Found",
    ] {
        let e = llm(&format!("free_router API error {code}: nope"));
        assert!(
            !is_service_unavailable(&e),
            "should NOT be transient: {code}"
        );
    }
}

#[test]
fn status_digits_in_body_do_not_false_positive() {
    // A 200 response whose body merely mentions 500/503 must not be classified.
    let e = llm("free_router API error 200 OK: the model wrote 'error 500 occurred' in text");
    assert!(!is_service_unavailable(&e));
}

#[test]
fn transient_markers_in_response_body_do_not_false_positive() {
    // Header is a non-transient 4xx; the body merely echoes transient phrases.
    for body in [
        "upstream said API error 503",
        "the gateway reported error sending request",
        "the model timed out internally",
        "connection refused by a downstream service",
    ] {
        let e = llm(&format!("free_router API error 400 Bad Request: {body}"));
        assert!(
            !is_service_unavailable(&e),
            "body must not classify: {body}"
        );
    }
}

#[test]
fn json_parse_is_deterministic_not_transient() {
    let e = llm("free_router JSON parse error: expected value at line 1");
    assert!(is_deterministic_error(&e));
    assert!(!is_service_unavailable(&e), "parse error is not transient");
}

#[test]
fn non_llm_variants_are_not_transient() {
    assert!(!is_service_unavailable(&InboxError::Pipeline("x".into())));
    assert!(!is_service_unavailable(&InboxError::Config("y".into())));
}

#[tokio::test]
async fn real_reqwest_timeout_is_service_unavailable() {
    // Pins the actual timeout wording this crate's reqwest produces.
    let server = wiremock::MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::method("GET"))
        .respond_with(wiremock::ResponseTemplate::new(200).set_delay(Duration::from_secs(5)))
        .mount(&server)
        .await;
    let client = reqwest::Client::new();
    let err = client
        .get(server.uri())
        .timeout(Duration::from_millis(150))
        .send()
        .await
        .expect_err("request must time out");
    assert!(
        is_service_unavailable(&llm(&err.to_string())),
        "reqwest timeout should classify transient: {err}"
    );
}

#[tokio::test]
async fn real_connection_refused_is_service_unavailable() {
    // Port 1 on loopback is closed → connection refused / connect error.
    let client = reqwest::Client::new();
    let err = client
        .get("http://127.0.0.1:1/")
        .timeout(Duration::from_millis(300))
        .send()
        .await
        .expect_err("connection must fail");
    assert!(
        is_service_unavailable(&llm(&err.to_string())),
        "connection failure should classify transient: {err}"
    );
}
