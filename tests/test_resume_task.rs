//! Integration test for the resume task's happy path.
//!
//! Drives one iteration of `retry_item` against a real `SQLite` pending store,
//! a real `Pipeline` with a mock LLM chain, and a real org output file.
//! Verifies:
//! - `:inbox_pending:` is replaced by `:inbox_failed:` when retries exhaust,
//! - on success, the pending row is removed from the store,
//! - on success, the org entry is patched in place.

use std::sync::Arc;

use chrono::Utc;
use uuid::Uuid;

use inbox::{
    config::{
        AdaptersConfig, AdminConfig, Config, GeneralConfig, PipelineConfig, SyncthingConfig,
        ToolingConfig, UrlFetchConfig, WebUiConfig,
    },
    message::{ProcessingHints, RetryableMessage, SourceMetadata},
    output::{OutputWriter, org_file::OrgFileWriter},
    pending::{PendingItem, PendingStore},
    pipeline::Pipeline,
    processing_status::ProcessingTracker,
    resume_task::{ResumeTaskArgs, retry_item_for_test},
    test_helpers as helpers,
};

fn minimal_config(attachments_dir: std::path::PathBuf, output_file: std::path::PathBuf) -> Config {
    Config {
        general: GeneralConfig {
            output_file,
            attachments_dir,
            log_level: "debug".into(),
            log_format: "pretty".into(),
        },
        admin: AdminConfig::default(),
        web_ui: WebUiConfig::default(),
        pipeline: PipelineConfig::default(),
        llm: helpers::no_llm_config(),
        adapters: AdaptersConfig::default(),
        url_fetch: UrlFetchConfig {
            enabled: false,
            ..UrlFetchConfig::default()
        },
        syncthing: SyncthingConfig::default(),
        tooling: ToolingConfig::default(),
        memory: inbox::config::MemoryConfig::default(),
    }
}

fn pending_item(id: Uuid) -> PendingItem {
    PendingItem {
        id,
        created_at: Utc::now(),
        retry_count: 0,
        last_retry_at: None,
        incoming: RetryableMessage {
            text: "Original message".into(),
            metadata: SourceMetadata::Http {
                remote_addr: None,
                user_agent: None,
            },
            attachments: vec![],
            user_tags: vec![],
            preprocessing_hints: ProcessingHints::default(),
            received_at: Utc::now(),
        },
        url_contents: vec![],
        tool_results: vec![],
        source_urls: vec![],
        fallback_title: Some("Original Fallback".into()),
        telegram_status_msg_id: None,
        source: "http".into(),
        url_count: 0,
        tool_count: 0,
    }
}

/// Pre-existing org entry shape the pipeline produces on raw-fallback — the
/// bit `resume_task` has to patch.
fn pending_org_entry(id: Uuid) -> String {
    format!(
        "* Original Fallback :inbox_pending:\n\
:PROPERTIES:\n\
:ID:       {id}\n\
:END:\n\
Some placeholder body.\n\
"
    )
}

#[tokio::test]
async fn retry_item_success_removes_pending_and_patches_org() {
    let (_tmp, dir) = helpers::temp_dir();
    let db_path = dir.join("pending.db");
    let output_file = dir.join("inbox.org");

    // Seed the org file with a pending entry matching the item we're about to retry.
    let id = Uuid::new_v4();
    tokio::fs::write(&output_file, pending_org_entry(id))
        .await
        .unwrap();

    // Seed the pending store directly — bypass Pipeline::process's insert path.
    let store = Arc::new(PendingStore::open(&db_path).await.unwrap());
    let item = pending_item(id);
    // Manually insert via a raw SQL path isn't exposed; instead craft the item
    // in-memory and drive retry_item against it directly. retry_item reads the
    // store only to update retry counts / remove on success.
    // For remove() to find the row, we still need it in the DB — use
    // PendingStore::insert via a ProcessedMessage shim.
    seed_pending(&store, &item).await;

    let cfg = Arc::new(minimal_config(dir.clone(), output_file.clone()));
    let llm = helpers::mock_llm_chain(helpers::default_llm_response());
    let writer = Arc::new(OrgFileWriter) as Arc<dyn OutputWriter>;
    let tracker = Arc::new(ProcessingTracker::new());
    let pipeline = Arc::new(Pipeline::new(
        Arc::clone(&cfg),
        llm,
        writer,
        Arc::clone(&tracker),
        None,
        Some(Arc::clone(&store)),
    ));

    let args = ResumeTaskArgs {
        store: Arc::clone(&store),
        pipeline,
        config: Arc::clone(&cfg),
        telegram_notifier: None,
        shutdown: tokio_util::sync::CancellationToken::new(),
    };

    retry_item_for_test(&args, &item, 3).await;

    // Pending row should be removed.
    let remaining = store.list(3, 10).await.unwrap();
    assert!(
        remaining.iter().all(|i| i.id != id),
        "pending row should be removed after successful retry"
    );

    // Org file should now contain the LLM-enriched title from the mock response
    // ("Test title") instead of the original fallback.
    let patched = tokio::fs::read_to_string(&output_file).await.unwrap();
    assert!(
        patched.contains("* Test title"),
        "org entry should be patched with the enriched title; got:\n{patched}"
    );
    assert!(
        !patched.contains(":inbox_pending:"),
        "pending tag should be gone; got:\n{patched}"
    );
}

#[tokio::test]
async fn retry_item_exhausts_and_flips_tag_to_failed() {
    let (_tmp, dir) = helpers::temp_dir();
    let db_path = dir.join("pending.db");
    let output_file = dir.join("inbox.org");

    let id = Uuid::new_v4();
    tokio::fs::write(&output_file, pending_org_entry(id))
        .await
        .unwrap();

    let store = Arc::new(PendingStore::open(&db_path).await.unwrap());
    let mut item = pending_item(id);
    // One more retry away from exhaustion.
    item.retry_count = 2;
    seed_pending(&store, &item).await;

    let cfg = Arc::new(minimal_config(dir.clone(), output_file.clone()));
    // Chain that always falls back (Raw mode → llm_response is None).
    let llm = helpers::always_fail_llm_chain();
    let writer = Arc::new(OrgFileWriter) as Arc<dyn OutputWriter>;
    let tracker = Arc::new(ProcessingTracker::new());
    let pipeline = Arc::new(Pipeline::new(
        Arc::clone(&cfg),
        llm,
        writer,
        Arc::clone(&tracker),
        None,
        Some(Arc::clone(&store)),
    ));

    let args = ResumeTaskArgs {
        store: Arc::clone(&store),
        pipeline,
        config: Arc::clone(&cfg),
        telegram_notifier: None,
        shutdown: tokio_util::sync::CancellationToken::new(),
    };

    retry_item_for_test(&args, &item, 3).await;

    let patched = tokio::fs::read_to_string(&output_file).await.unwrap();
    assert!(
        patched.contains(":inbox_failed:"),
        "tag should flip to inbox_failed when retries exhaust; got:\n{patched}"
    );
    assert!(
        !patched.contains(":inbox_pending:"),
        "pending tag should be gone; got:\n{patched}"
    );
}

/// Helper that seeds the pending store by calling `insert` with a minimally
/// constructed `ProcessedMessage`. The store's `insert` signature requires a
/// real `ProcessedMessage`, so assemble one from the pending-item fields.
async fn seed_pending(store: &PendingStore, item: &PendingItem) {
    use inbox::message::{
        EnrichedMessage, EnrichmentMetadata, IncomingMessage, MessageSource, ProcessedMessage,
    };

    let mut incoming = IncomingMessage::with_id(
        item.id,
        MessageSource::Http,
        item.incoming.text.clone(),
        item.incoming.metadata.clone(),
    );
    incoming.received_at = item.incoming.received_at;

    let processed = ProcessedMessage {
        enriched: EnrichedMessage {
            original: incoming,
            urls: vec![],
            url_contents: vec![],
        },
        llm_response: None,
        fallback_source_urls: vec![],
        fallback_tool_results: vec![],
        fallback_title: item.fallback_title.clone(),
        enrichment: EnrichmentMetadata::default(),
    };
    store.insert(item.id, &processed, None).await.unwrap();

    // Bump the retry counter to match `item.retry_count` so the exhausted-
    // tag-flip test observes the correct "already at retry_count=2" state.
    for _ in 0..item.retry_count {
        store.increment_retry(item.id).await.unwrap();
    }
}
