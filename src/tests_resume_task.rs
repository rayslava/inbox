use chrono::Utc;
use uuid::Uuid;

use crate::message::{MessageSource, RetryableMessage, SourceMetadata};
use crate::pending::PendingItem;
use crate::resume_task::build_enriched;
use crate::url_content::UrlContent;

fn dummy_pending(source: &str) -> PendingItem {
    PendingItem {
        id: Uuid::new_v4(),
        created_at: Utc::now(),
        retry_count: 0,
        last_retry_at: None,
        incoming: RetryableMessage {
            text: "test message".into(),
            metadata: SourceMetadata::Http {
                remote_addr: None,
                user_agent: None,
            },
            attachments: vec![],
            user_tags: vec!["tag1".into()],
            preprocessing_hints: Default::default(),
            received_at: Utc::now(),
        },
        url_contents: vec![UrlContent {
            url: "https://example.com".into(),
            text: "page text".into(),
            page_title: Some("Example".into()),
            headings: vec![],
        }],
        tool_results: vec![("scrape_page".into(), "content".into())],
        source_urls: vec!["https://example.com".into()],
        fallback_title: Some("Title".into()),
        telegram_status_msg_id: None,
        source: source.into(),
        url_count: 1,
        tool_count: 1,
    }
}

#[test]
fn build_enriched_http_source() {
    let item = dummy_pending("http");
    let enriched = build_enriched(&item).unwrap();
    assert_eq!(enriched.original.source, MessageSource::Http);
    assert_eq!(enriched.original.text, "test message");
    assert_eq!(enriched.url_contents.len(), 1);
    assert_eq!(enriched.urls.len(), 1);
    assert_eq!(enriched.original.user_tags, vec!["tag1"]);
}

#[test]
fn build_enriched_telegram_source() {
    let mut item = dummy_pending("telegram");
    item.incoming.metadata = SourceMetadata::Telegram {
        chat_id: 42,
        message_id: 7,
        username: None,
        forwarded_from: None,
    };
    let enriched = build_enriched(&item).unwrap();
    assert_eq!(enriched.original.source, MessageSource::Telegram);
}

#[test]
fn build_enriched_email_source() {
    let item = dummy_pending("email");
    let enriched = build_enriched(&item).unwrap();
    assert_eq!(enriched.original.source, MessageSource::Email);
}

#[test]
fn build_enriched_preserves_id() {
    let item = dummy_pending("http");
    let id = item.id;
    let enriched = build_enriched(&item).unwrap();
    assert_eq!(enriched.original.id, id);
}

#[test]
fn build_enriched_skips_invalid_urls() {
    let mut item = dummy_pending("http");
    item.url_contents = vec![
        UrlContent {
            url: "not-a-url".into(),
            text: "bad".into(),
            page_title: None,
            headings: vec![],
        },
        UrlContent {
            url: "https://valid.example.com".into(),
            text: "good".into(),
            page_title: None,
            headings: vec![],
        },
    ];
    let enriched = build_enriched(&item).unwrap();
    // Only the valid URL is parsed into enriched.urls; url_contents keeps both.
    assert_eq!(enriched.urls.len(), 1);
    assert_eq!(enriched.url_contents.len(), 2);
}
