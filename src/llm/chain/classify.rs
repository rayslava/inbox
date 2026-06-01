//! Error classifiers for the backend chain: which failures recur on retry
//! (deterministic) and which are transient "service unavailable" outages that
//! should trip a cooldown and fall through to the next backend.

use crate::error::InboxError;

/// Errors that will recur identically on a retry of the *same* backend, so
/// burning the remaining retry budget on them is wasted time. A JSON parse
/// failure means the model produced unparseable output (e.g. Markdown prose);
/// re-running the same model with the same prompt yields the same result, often
/// after minutes of slow local inference each time.
pub(crate) fn is_deterministic_error(err: &InboxError) -> bool {
    let InboxError::Llm(msg) = err else {
        return false;
    };
    msg.contains("JSON parse error")
}

/// Whether `err` leaves the backend usable — i.e. it is **not** a transient
/// "service unavailable" failure (rate limit, upstream 5xx, transport
/// timeout/connection error, or an open circuit). Returns `false` for those
/// transient outages so the caller trips the cooldown and falls through to the
/// next (local) backend; `true` for any other error (auth, JSON parse, …),
/// which does not indicate the service is down.
///
/// Matched against the flattened `InboxError::Llm` string; HTTP statuses are
/// anchored on the `"API error {status}"` prefix the clients emit, so a status
/// digit in a response body cannot false-positive. Callers must classify
/// deterministic errors (e.g. JSON parse) *first*.
pub(crate) fn is_service_available(err: &InboxError) -> bool {
    const HTTP: [&str; 5] = [
        "API error 429",
        "API error 500",
        "API error 502",
        "API error 503",
        "API error 504",
    ];
    let InboxError::Llm(msg) = err else {
        return true;
    };
    // Errors are either "{header}: {body}" (HTTP responses — `body` is the raw
    // model output and must NOT drive classification) or a bare reqwest transport
    // string. Scan only the header (before the first `:`) so a status code or
    // transport phrase echoed inside a response body cannot false-positive.
    let header = msg.split_once(':').map_or(msg.as_str(), |(h, _)| h);
    if HTTP.iter().any(|m| header.contains(m)) || header.contains("circuit open") {
        return false;
    }
    // reqwest flattens transport failures via `to_string()` to
    // "error sending request for url (...)" — the timeout/connect cause lives in
    // the source chain, not the Display (pinned by the real-timeout test). The
    // marker sits before the URL's "http:" colon, so it stays in `header`.
    let lower = header.to_ascii_lowercase();
    ![
        "error sending request",
        "error trying to connect",
        "timed out",
        "connection refused",
        "dns error",
    ]
    .iter()
    .any(|m| lower.contains(m))
}
