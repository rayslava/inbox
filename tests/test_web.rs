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

// ── Attachment link parsing ───────────────────────────────────────────────────

const NODE_WITH_IMAGE_ATTACHMENT: &str = r"* Entry with image :photo:

:PROPERTIES:
:ID:       a1b2c3d4-0000-0000-0000-000000000001
:CREATED:  [2024-01-15 Mon 10:30]
:SOURCE:   telegram
:END:

Summary above.

[[attachment:photo.jpg][photo.jpg]]
";

#[test]
fn parse_org_nodes_extracts_image_attachment() {
    let nodes = ui::parse_org_nodes(NODE_WITH_IMAGE_ATTACHMENT, Path::new("/attachments"));
    let node = &nodes[0];
    assert_eq!(node.attachments.len(), 1, "expected one attachment");
    let html = &node.attachments[0].html;
    assert!(
        html.contains("<img "),
        "image attachment should render <img>"
    );
    assert!(
        html.contains("attachments/"),
        "URL should be namespaced under attachments/"
    );
}

const NODE_WITH_AUDIO_ATTACHMENT: &str = r"* Voice note :voice:

:PROPERTIES:
:ID:       a1b2c3d4-0000-0000-0000-000000000002
:CREATED:  [2024-01-15 Mon 10:30]
:SOURCE:   telegram
:END:

[[attachment:voice.ogg][voice.ogg]]
";

#[test]
fn parse_org_nodes_extracts_audio_attachment() {
    let nodes = ui::parse_org_nodes(NODE_WITH_AUDIO_ATTACHMENT, Path::new("/attachments"));
    let html = &nodes[0].attachments[0].html;
    assert!(
        html.contains("<audio "),
        "audio attachment should render <audio>"
    );
}

const NODE_WITH_VIDEO_ATTACHMENT: &str = r"* Clip :video:

:PROPERTIES:
:ID:       a1b2c3d4-0000-0000-0000-000000000003
:CREATED:  [2024-01-15 Mon 10:30]
:SOURCE:   telegram
:END:

[[attachment:clip.mp4][clip.mp4]]
";

#[test]
fn parse_org_nodes_extracts_video_attachment() {
    let nodes = ui::parse_org_nodes(NODE_WITH_VIDEO_ATTACHMENT, Path::new("/attachments"));
    let html = &nodes[0].attachments[0].html;
    assert!(
        html.contains("<video "),
        "video attachment should render <video>"
    );
}

const NODE_WITH_DOCUMENT_ATTACHMENT: &str = r"* Doc :doc:

:PROPERTIES:
:ID:       a1b2c3d4-0000-0000-0000-000000000004
:CREATED:  [2024-01-15 Mon 10:30]
:SOURCE:   http
:END:

[[attachment:report.pdf][report.pdf]]
";

#[test]
fn parse_org_nodes_document_attachment_renders_as_link() {
    let nodes = ui::parse_org_nodes(NODE_WITH_DOCUMENT_ATTACHMENT, Path::new("/attachments"));
    let html = &nodes[0].attachments[0].html;
    assert!(
        html.contains("class=\"doc-link\""),
        "non-media should render as a link"
    );
    assert!(html.contains("report.pdf"));
}

const NODE_WITH_FILE_LINK: &str = r"* Legacy file link :legacy:

:PROPERTIES:
:ID:       a1b2c3d4-0000-0000-0000-000000000005
:CREATED:  [2024-01-15 Mon 10:30]
:SOURCE:   http
:END:

[[file:/var/inbox/attachments/a1b2c3d4-0000-0000-0000-000000000005/thing.txt][thing.txt]]
";

#[test]
fn parse_org_nodes_supports_legacy_file_links() {
    let nodes = ui::parse_org_nodes(NODE_WITH_FILE_LINK, Path::new("/var/inbox/attachments"));
    assert_eq!(
        nodes[0].attachments.len(),
        1,
        "legacy file link should be detected"
    );
}

const NODE_WITH_TRAVERSAL_ATTEMPT: &str = r"* Malicious :bad:

:PROPERTIES:
:ID:       a1b2c3d4-0000-0000-0000-000000000006
:CREATED:  [2024-01-15 Mon 10:30]
:SOURCE:   http
:END:

[[attachment:../../../etc/passwd][evil]]
";

#[test]
fn parse_org_nodes_rejects_path_traversal_in_attachment() {
    let nodes = ui::parse_org_nodes(NODE_WITH_TRAVERSAL_ATTEMPT, Path::new("/attachments"));
    assert!(
        nodes[0].attachments.is_empty(),
        "attachment:../ should be rejected"
    );
}
