/// Integration tests for the web module: auth helpers and org parser.
use std::path::Path;

use inbox::web::{auth, ui};

// ── Auth ──────────────────────────────────────────────────────────────────────

#[test]
fn session_token_is_64_hex_chars() {
    let token = auth::generate_session_token();
    assert_eq!(token.len(), 64, "token should be 32 bytes as hex");
    assert!(
        token.chars().all(|c| c.is_ascii_hexdigit()),
        "token should be hex"
    );
}

#[test]
fn session_tokens_are_unique() {
    let t1 = auth::generate_session_token();
    let t2 = auth::generate_session_token();
    assert_ne!(t1, t2, "two consecutive tokens should differ");
}

#[test]
fn is_authenticated_returns_false_for_missing_cookie() {
    use axum::http::HeaderMap;
    let store = auth::new_session_store();
    let headers = HeaderMap::new();
    assert!(!auth::is_authenticated(&headers, &store, 30));
}

#[test]
fn is_authenticated_returns_false_for_unknown_token() {
    use axum::http::{HeaderMap, header};
    let store = auth::new_session_store();
    let mut headers = HeaderMap::new();
    headers.insert(header::COOKIE, "session=nonexistent_token".parse().unwrap());
    assert!(!auth::is_authenticated(&headers, &store, 30));
}

#[test]
fn is_authenticated_returns_true_for_fresh_session() {
    use axum::http::{HeaderMap, header};
    use chrono::Utc;
    let store = auth::new_session_store();
    let token = auth::generate_session_token();
    store.insert(token.clone(), Utc::now());

    let mut headers = HeaderMap::new();
    headers.insert(header::COOKIE, format!("session={token}").parse().unwrap());
    assert!(auth::is_authenticated(&headers, &store, 30));
}

// ── Org parser ────────────────────────────────────────────────────────────────

const SAMPLE_ORG: &str = r"* My capture title :tag1:tag2:

:PROPERTIES:
:ID:       a1b2c3d4-0000-0000-0000-000000000001
:CREATED:  [2024-01-15 Mon 10:30]
:SOURCE:   http
:ENRICHED_BY: mock
:END:

This is the summary text.

#+begin_quote
Key excerpt here.
#+end_quote

* Another entry

:PROPERTIES:
:ID:       a1b2c3d4-0000-0000-0000-000000000002
:CREATED:  [2024-01-16 Tue 11:00]
:SOURCE:   telegram
:ENRICHED_BY: mock
:END:

Second summary.
";

#[test]
fn parse_org_nodes_extracts_two_nodes() {
    let nodes = ui::parse_org_nodes(SAMPLE_ORG, Path::new("/attachments"));
    assert_eq!(nodes.len(), 2, "should find two headlines");
}

#[test]
fn parse_org_nodes_extracts_title_and_tags() {
    let nodes = ui::parse_org_nodes(SAMPLE_ORG, Path::new("/attachments"));
    let first = &nodes[0];
    assert_eq!(first.title, "My capture title");
    assert_eq!(first.tags, vec!["tag1", "tag2"]);
}

#[test]
fn parse_org_nodes_extracts_properties() {
    let nodes = ui::parse_org_nodes(SAMPLE_ORG, Path::new("/attachments"));
    let first = &nodes[0];
    assert_eq!(first.source, "http");
    assert!(first.created.contains("2024-01-15"));
}

#[test]
fn parse_org_nodes_extracts_summary() {
    let nodes = ui::parse_org_nodes(SAMPLE_ORG, Path::new("/attachments"));
    assert!(nodes[0].summary.contains("summary text"));
}

#[test]
fn parse_org_nodes_extracts_excerpt() {
    let nodes = ui::parse_org_nodes(SAMPLE_ORG, Path::new("/attachments"));
    let excerpt = nodes[0].excerpt.as_deref().unwrap_or("");
    assert!(excerpt.contains("Key excerpt"), "missing excerpt content");
}

#[test]
fn parse_org_nodes_second_node_has_no_excerpt() {
    let nodes = ui::parse_org_nodes(SAMPLE_ORG, Path::new("/attachments"));
    assert!(nodes[1].excerpt.is_none());
}
