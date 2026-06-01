use chrono::Utc;
use uuid::Uuid;

use super::store::PendingStore;
use crate::message::{EnrichedMessage, IncomingMessage, ProcessedMessage};
use crate::message::{MessageSource, ProcessingHints, RetryableMessage, SourceMetadata};
use crate::url_content::UrlContent;

fn dummy_retryable() -> RetryableMessage {
    RetryableMessage {
        text: "hello world".into(),
        metadata: SourceMetadata::Http {
            remote_addr: None,
            user_agent: None,
        },
        attachments: vec![],
        user_tags: vec![],
        preprocessing_hints: ProcessingHints::default(),
        received_at: Utc::now(),
        image_analyses: Vec::new(),
    }
}

fn dummy_processed(id: Uuid) -> ProcessedMessage {
    let retryable = dummy_retryable();
    let incoming = IncomingMessage::with_id(
        id,
        MessageSource::Http,
        retryable.text.clone(),
        retryable.metadata.clone(),
    );
    ProcessedMessage {
        enriched: EnrichedMessage {
            original: incoming,
            urls: vec![],
            url_contents: vec![UrlContent {
                url: "https://example.com".into(),
                text: "page text".into(),
                page_title: Some("Example".into()),
                headings: vec![],
            }],
        },
        llm_response: None,
        incomplete: crate::message::ProcessingCompleteness::Complete,
        fallback_source_urls: vec!["https://found.example".into()],
        fallback_tool_results: vec![("scrape_page".into(), "scraped content".into())],
        fallback_title: Some("Fallback Title".into()),
        enrichment: crate::message::EnrichmentMetadata::default(),
    }
}

async fn open_in_memory() -> PendingStore {
    // sqlx doesn't support migrate! with in-memory for path-based migrations;
    // use a temp file instead.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.db");
    // Keep dir alive for the duration of the test by leaking it — acceptable in tests.
    std::mem::forget(dir);
    PendingStore::open(&path).await.unwrap()
}

#[tokio::test]
async fn insert_and_list() {
    let store = open_in_memory().await;
    let id = Uuid::new_v4();
    let msg = dummy_processed(id);

    store.insert(id, &msg, None).await.unwrap();

    let items = store.list(5, 10).await.unwrap();
    assert_eq!(items.len(), 1);
    let item = &items[0];
    assert_eq!(item.id, id);
    assert_eq!(item.retry_count, 0);
    assert!(item.last_retry_at.is_none());
    assert_eq!(item.url_contents.len(), 1);
    assert_eq!(item.tool_results.len(), 1);
    assert_eq!(item.source_urls.len(), 1);
    assert_eq!(item.fallback_title.as_deref(), Some("Fallback Title"));
    assert_eq!(item.url_count, 1);
    assert_eq!(item.tool_count, 1);
    assert_eq!(item.source, "http");
}

#[tokio::test]
async fn insert_is_idempotent() {
    let store = open_in_memory().await;
    let id = Uuid::new_v4();
    let msg = dummy_processed(id);

    store.insert(id, &msg, None).await.unwrap();
    store.insert(id, &msg, None).await.unwrap(); // INSERT OR IGNORE

    assert_eq!(store.list(5, 10).await.unwrap().len(), 1);
}

#[tokio::test]
async fn increment_retry_and_filter() {
    let store = open_in_memory().await;
    let id = Uuid::new_v4();
    let msg = dummy_processed(id);

    store.insert(id, &msg, None).await.unwrap();

    // With max_retries=1, item (retry_count=0) is listed.
    assert_eq!(store.list(1, 10).await.unwrap().len(), 1);

    store.increment_retry(id).await.unwrap();

    // Now retry_count=1 >= max_retries=1 → excluded.
    assert_eq!(store.list(1, 10).await.unwrap().len(), 0);

    // But with max_retries=2 it should still appear.
    assert_eq!(store.list(2, 10).await.unwrap().len(), 1);

    let item = &store.list(2, 10).await.unwrap()[0];
    assert_eq!(item.retry_count, 1);
    assert!(item.last_retry_at.is_some());
}

#[tokio::test]
async fn remove() {
    let store = open_in_memory().await;
    let id = Uuid::new_v4();
    let msg = dummy_processed(id);

    store.insert(id, &msg, None).await.unwrap();
    store.remove(id).await.unwrap();

    assert!(store.list(5, 10).await.unwrap().is_empty());
}

#[tokio::test]
async fn stats_counts() {
    let store = open_in_memory().await;

    for _ in 0..3 {
        let id = Uuid::new_v4();
        store.insert(id, &dummy_processed(id), None).await.unwrap();
    }

    // Exhaust one item (retry_count >= 2).
    let items = store.list(5, 10).await.unwrap();
    store.increment_retry(items[0].id).await.unwrap();
    store.increment_retry(items[0].id).await.unwrap();

    let s = store.stats(2).await.unwrap();
    assert_eq!(s.total_items, 3);
    assert_eq!(s.exhausted_items, 1);
    assert!(s.db_page_count > 0);
    assert!(s.db_page_size > 0);
    assert!(s.db_bytes() > 0);
}

#[tokio::test]
async fn telegram_chat_id_extraction() {
    let store = open_in_memory().await;
    let id = Uuid::new_v4();
    let mut msg = dummy_processed(id);
    msg.enriched.original.metadata = SourceMetadata::Telegram {
        chat_id: 42,
        message_id: 1,
        username: None,
        forwarded_from: None,
    };
    // Re-build retryable with Telegram metadata.
    store.insert(id, &msg, Some(99)).await.unwrap();

    let items = store.list(5, 10).await.unwrap();
    assert_eq!(items[0].telegram_status_msg_id, Some(99));
    assert_eq!(items[0].source, "telegram");
    assert_eq!(PendingStore::telegram_chat_id(&items[0]), Some(42));
}

// ── open_optional tests ──────────────────────────────────────────────────────

fn minimal_config(attachments_dir: std::path::PathBuf) -> crate::config::Config {
    use crate::config::{
        AdaptersConfig, AdminConfig, Config, GeneralConfig, PipelineConfig, SyncthingConfig,
        ToolingConfig, UrlFetchConfig, WebUiConfig,
    };
    Config {
        general: GeneralConfig {
            output_file: attachments_dir.join("inbox.org"),
            attachments_dir,
            log_level: "info".into(),
            log_format: "pretty".into(),
        },
        admin: AdminConfig::default(),
        web_ui: WebUiConfig::default(),
        pipeline: PipelineConfig::default(),
        llm: crate::test_helpers::no_llm_config(),
        adapters: AdaptersConfig::default(),
        url_fetch: UrlFetchConfig::default(),
        syncthing: SyncthingConfig::default(),
        tooling: ToolingConfig::default(),
        memory: crate::config::MemoryConfig::default(),
    }
}

#[tokio::test]
async fn open_optional_returns_none_when_resume_disabled() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = minimal_config(dir.path().to_path_buf());
    assert!(!cfg.pipeline.resume.enabled, "default should be disabled");
    assert!(super::open_optional(&cfg).await.is_none());
}

#[tokio::test]
async fn open_optional_opens_store_at_default_path_when_enabled() {
    let dir = tempfile::tempdir().unwrap();
    let mut cfg = minimal_config(dir.path().to_path_buf());
    cfg.pipeline.resume.enabled = true;

    let store = super::open_optional(&cfg).await;
    assert!(store.is_some(), "should open store under attachments_dir");
    assert!(
        dir.path().join("pending.db").exists(),
        "default path should be {{attachments_dir}}/pending.db"
    );
}

#[tokio::test]
async fn open_optional_uses_explicit_db_path() {
    let dir = tempfile::tempdir().unwrap();
    let custom = dir.path().join("subdir").join("custom.db");
    std::fs::create_dir_all(custom.parent().unwrap()).unwrap();

    let mut cfg = minimal_config(dir.path().to_path_buf());
    cfg.pipeline.resume.enabled = true;
    cfg.pipeline.resume.db_path = Some(custom.clone());

    let store = super::open_optional(&cfg).await;
    assert!(store.is_some());
    assert!(custom.exists(), "explicit db_path should be created");
}

#[tokio::test]
async fn open_optional_returns_none_on_open_failure() {
    // Point at a path whose parent doesn't exist and can't be created
    // (a non-directory file in the parent slot).
    let dir = tempfile::tempdir().unwrap();
    let blocker = dir.path().join("not-a-dir");
    std::fs::write(&blocker, b"i am a regular file").unwrap();
    let bad_path = blocker.join("pending.db");

    let mut cfg = minimal_config(dir.path().to_path_buf());
    cfg.pipeline.resume.enabled = true;
    cfg.pipeline.resume.db_path = Some(bad_path);

    assert!(super::open_optional(&cfg).await.is_none());
}
