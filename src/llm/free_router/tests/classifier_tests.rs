//! Tests for the `is_hard_error` classifier.

use crate::error::InboxError;
use crate::llm::free_router::is_hard_error;

#[test]
fn is_hard_error_detects_auth_codes() {
    assert!(is_hard_error(&InboxError::Llm(
        "free_router API error 401 Unauthorized: nope".into()
    )));
    assert!(is_hard_error(&InboxError::Llm(
        "free_router API error 403 Forbidden: nope".into()
    )));
    assert!(is_hard_error(&InboxError::Llm(
        "free_router API error 400 Bad Request: nope".into()
    )));
}

#[test]
fn is_hard_error_leaves_transient_codes() {
    assert!(!is_hard_error(&InboxError::Llm(
        "free_router API error 500 Internal Server Error".into()
    )));
    assert!(!is_hard_error(&InboxError::Llm(
        "free_router API error 429 Too Many Requests".into()
    )));
    assert!(!is_hard_error(&InboxError::Llm("network blip".into())));
}
