use std::path::Path;

use anodized::spec;
use chrono::{DateTime, Utc};
use sqlx::sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePoolOptions};
use sqlx::{Row, SqlitePool};
use uuid::Uuid;

use crate::error::InboxError;
use crate::message::{ProcessedMessage, RetryableMessage, SourceMetadata};
use crate::url_content::UrlContent;

/// A pending item stored in the `SQLite` database awaiting LLM retry.
#[derive(Debug)]
pub struct PendingItem {
    pub id: Uuid,
    pub created_at: DateTime<Utc>,
    pub retry_count: u32,
    pub last_retry_at: Option<DateTime<Utc>>,
    pub incoming: RetryableMessage,
    pub url_contents: Vec<UrlContent>,
    pub tool_results: Vec<(String, String)>,
    pub source_urls: Vec<String>,
    pub fallback_title: Option<String>,
    pub telegram_status_msg_id: Option<i32>,
    /// From generated column — available without extra deserialization.
    pub source: String,
    pub url_count: u32,
    pub tool_count: u32,
}

/// Aggregate statistics about the pending queue and its `SQLite` store.
#[derive(Debug, Clone)]
pub struct PendingStats {
    pub total_items: u32,
    pub exhausted_items: u32,
    pub db_page_count: u64,
    pub db_page_size: u64,
    pub db_freelist_count: u64,
}

impl PendingStats {
    /// Estimated database file size in bytes.
    #[must_use]
    pub fn db_bytes(&self) -> u64 {
        self.db_page_count * self.db_page_size
    }
}

/// SQLite-backed store for messages awaiting LLM retry.
pub struct PendingStore {
    pool: SqlitePool,
}

impl PendingStore {
    /// Open (or create) the pending-items database at `path`.
    ///
    /// Runs embedded migrations on first open. Enables WAL mode.
    ///
    /// # Errors
    /// Returns an error if the database cannot be opened or migrations fail.
    pub async fn open(path: &Path) -> Result<Self, InboxError> {
        let opts = SqliteConnectOptions::new()
            .filename(path)
            .create_if_missing(true)
            .journal_mode(SqliteJournalMode::Wal);

        let pool = SqlitePoolOptions::new()
            .max_connections(4)
            .connect_with(opts)
            .await
            .map_err(|e| InboxError::Pipeline(format!("pending db open: {e}")))?;

        sqlx::migrate!("src/pending/migrations")
            .run(&pool)
            .await
            .map_err(|e| InboxError::Pipeline(format!("pending db migrate: {e}")))?;

        Ok(Self { pool })
    }

    /// Persist a fallback-processed message for later LLM retry.
    ///
    /// Extracts all relevant context from `msg` in one place. No-op if the
    /// id already exists (`INSERT OR IGNORE`).
    #[spec(requires: msg.llm_response.is_none())]
    pub async fn insert(
        &self,
        id: Uuid,
        msg: &ProcessedMessage,
        telegram_status_msg_id: Option<i32>,
    ) -> Result<(), InboxError> {
        let retryable = RetryableMessage::from(&msg.enriched.original);
        let incoming_json = serde_json::to_string(&retryable)
            .map_err(|e| InboxError::Pipeline(format!("serialize incoming: {e}")))?;
        let url_contents_json = serde_json::to_string(&msg.enriched.url_contents)
            .map_err(|e| InboxError::Pipeline(format!("serialize url_contents: {e}")))?;
        let tool_results_json = serde_json::to_string(&msg.fallback_tool_results)
            .map_err(|e| InboxError::Pipeline(format!("serialize tool_results: {e}")))?;
        let source_urls_json = serde_json::to_string(&msg.fallback_source_urls)
            .map_err(|e| InboxError::Pipeline(format!("serialize source_urls: {e}")))?;
        let id_str = id.to_string();

        sqlx::query(
            r"
            INSERT OR IGNORE INTO pending_items
                (id, incoming, url_contents, tool_results, source_urls,
                 fallback_title, telegram_status_msg_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ",
        )
        .bind(&id_str)
        .bind(&incoming_json)
        .bind(&url_contents_json)
        .bind(&tool_results_json)
        .bind(&source_urls_json)
        .bind(&msg.fallback_title)
        .bind(telegram_status_msg_id)
        .execute(&self.pool)
        .await
        .map_err(|e| InboxError::Pipeline(format!("pending insert: {e}")))?;

        Ok(())
    }

    /// Return up to `limit` pending items with `retry_count < max_retries`,
    /// ordered oldest-first by `received_at`.
    #[spec(requires: limit > 0)]
    pub async fn list(&self, max_retries: u32, limit: u32) -> Result<Vec<PendingItem>, InboxError> {
        let rows = sqlx::query(
            r"
            SELECT
                id,
                created_at,
                retry_count,
                last_retry_at,
                incoming,
                url_contents,
                tool_results,
                source_urls,
                fallback_title,
                telegram_status_msg_id,
                source,
                url_count,
                tool_count
            FROM pending_items
            WHERE retry_count < ?
            ORDER BY received_at ASC
            LIMIT ?
            ",
        )
        .bind(i64::from(max_retries))
        .bind(i64::from(limit))
        .fetch_all(&self.pool)
        .await
        .map_err(|e| InboxError::Pipeline(format!("pending list: {e}")))?;

        let mut items = Vec::with_capacity(rows.len());
        for row in rows {
            let id_str: String = row.get("id");
            let id = Uuid::parse_str(&id_str)
                .map_err(|e| InboxError::Pipeline(format!("pending id parse: {e}")))?;

            let created_at_str: String = row.get("created_at");
            let created_at: DateTime<Utc> = created_at_str
                .parse()
                .map_err(|e| InboxError::Pipeline(format!("pending created_at parse: {e}")))?;

            let last_retry_at: Option<DateTime<Utc>> = row
                .get::<Option<String>, _>("last_retry_at")
                .map(|s| {
                    s.parse()
                        .map_err(|e| InboxError::Pipeline(format!("pending last_retry_at: {e}")))
                })
                .transpose()?;

            let incoming_json: String = row.get("incoming");
            let incoming: RetryableMessage = serde_json::from_str(&incoming_json)
                .map_err(|e| InboxError::Pipeline(format!("pending incoming deser: {e}")))?;

            let url_contents_json: String = row.get("url_contents");
            let url_contents: Vec<UrlContent> = serde_json::from_str(&url_contents_json)
                .map_err(|e| InboxError::Pipeline(format!("pending url_contents deser: {e}")))?;

            let tool_results_json: String = row.get("tool_results");
            let tool_results: Vec<(String, String)> = serde_json::from_str(&tool_results_json)
                .map_err(|e| InboxError::Pipeline(format!("pending tool_results deser: {e}")))?;

            let source_urls_json: String = row.get("source_urls");
            let source_urls: Vec<String> = serde_json::from_str(&source_urls_json)
                .map_err(|e| InboxError::Pipeline(format!("pending source_urls deser: {e}")))?;

            items.push(PendingItem {
                id,
                created_at,
                retry_count: u32::try_from(row.get::<i64, _>("retry_count")).unwrap_or(u32::MAX),
                last_retry_at,
                incoming,
                url_contents,
                tool_results,
                source_urls,
                fallback_title: row.get("fallback_title"),
                telegram_status_msg_id: row
                    .get::<Option<i64>, _>("telegram_status_msg_id")
                    .map(|v| i32::try_from(v).unwrap_or(i32::MAX)),
                source: row.get::<Option<String>, _>("source").unwrap_or_default(),
                url_count: u32::try_from(row.get::<Option<i64>, _>("url_count").unwrap_or(0))
                    .unwrap_or(u32::MAX),
                tool_count: u32::try_from(row.get::<Option<i64>, _>("tool_count").unwrap_or(0))
                    .unwrap_or(u32::MAX),
            });
        }
        Ok(items)
    }

    /// Increment the retry counter and record the timestamp of this attempt.
    ///
    /// # Errors
    /// Returns an error if the database update fails.
    pub async fn increment_retry(&self, id: Uuid) -> Result<(), InboxError> {
        let id_str = id.to_string();
        sqlx::query(
            r"
            UPDATE pending_items
            SET retry_count   = retry_count + 1,
                last_retry_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE id = ?
            ",
        )
        .bind(&id_str)
        .execute(&self.pool)
        .await
        .map_err(|e| InboxError::Pipeline(format!("pending increment_retry: {e}")))?;
        Ok(())
    }

    /// Remove a successfully processed pending item.
    ///
    /// # Errors
    /// Returns an error if the database delete fails.
    pub async fn remove(&self, id: Uuid) -> Result<(), InboxError> {
        let id_str = id.to_string();
        sqlx::query("DELETE FROM pending_items WHERE id = ?")
            .bind(&id_str)
            .execute(&self.pool)
            .await
            .map_err(|e| InboxError::Pipeline(format!("pending remove: {e}")))?;
        Ok(())
    }

    /// Collect queue statistics including `SQLite` PRAGMA values.
    ///
    /// # Errors
    /// Returns an error if any of the backing queries or PRAGMAs fail.
    pub async fn stats(&self, max_retries: u32) -> Result<PendingStats, InboxError> {
        let row = sqlx::query(
            r"
            SELECT
                COUNT(*) AS total_items,
                COUNT(CASE WHEN retry_count >= ? THEN 1 END) AS exhausted_items
            FROM pending_items
            ",
        )
        .bind(i64::from(max_retries))
        .fetch_one(&self.pool)
        .await
        .map_err(|e| InboxError::Pipeline(format!("pending stats: {e}")))?;

        let page_count: i64 = sqlx::query_scalar("SELECT * FROM pragma_page_count()")
            .fetch_one(&self.pool)
            .await
            .map_err(|e| InboxError::Pipeline(format!("pragma page_count: {e}")))?;

        let page_size: i64 = sqlx::query_scalar("SELECT * FROM pragma_page_size()")
            .fetch_one(&self.pool)
            .await
            .map_err(|e| InboxError::Pipeline(format!("pragma page_size: {e}")))?;

        let freelist: i64 = sqlx::query_scalar("SELECT * FROM pragma_freelist_count()")
            .fetch_one(&self.pool)
            .await
            .map_err(|e| InboxError::Pipeline(format!("pragma freelist_count: {e}")))?;

        Ok(PendingStats {
            total_items: u32::try_from(row.get::<i64, _>("total_items")).unwrap_or(u32::MAX),
            exhausted_items: u32::try_from(row.get::<i64, _>("exhausted_items"))
                .unwrap_or(u32::MAX),
            db_page_count: u64::try_from(page_count).unwrap_or(0),
            db_page_size: u64::try_from(page_size).unwrap_or(0),
            db_freelist_count: u64::try_from(freelist).unwrap_or(0),
        })
    }

    /// Returns the `chat_id` from a pending item's incoming metadata, if it was from Telegram.
    #[must_use]
    pub fn telegram_chat_id(item: &PendingItem) -> Option<i64> {
        match &item.incoming.metadata {
            SourceMetadata::Telegram { chat_id, .. } => Some(*chat_id),
            _ => None,
        }
    }
}
