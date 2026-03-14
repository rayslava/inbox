use std::path::Path;
use std::sync::Arc;

use anodized::spec;
use tracing::warn;

use crate::config::MemoryConfig;
use crate::error::InboxError;

pub mod embed;
pub mod search;

#[cfg(test)]
mod tests;

// ── Public types ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct MemoryEntry {
    pub key: String,
    pub value: String,
    pub score: f64,
}

// ── MemoryStore ───────────────────────────────────────────────────────────────

pub struct MemoryStore {
    conn: Arc<std::sync::Mutex<rusqlite::Connection>>,
    embed_client: Option<embed::EmbedClient>,
}

impl MemoryStore {
    /// Open (or create) a `MemoryStore` at `db_path`, applying the schema and
    /// optionally probing the embedding endpoint to detect vector dimensions.
    ///
    /// # Errors
    /// Returns an error if the database cannot be opened or schema migration fails.
    pub async fn open(cfg: &MemoryConfig, db_path: &Path) -> Result<Self, InboxError> {
        let path = db_path.to_owned();
        let conn = tokio::task::spawn_blocking(move || {
            let c =
                rusqlite::Connection::open(&path).map_err(|e| InboxError::Memory(e.to_string()))?;
            apply_schema(&c)?;
            Ok::<_, InboxError>(c)
        })
        .await
        .map_err(|e| InboxError::Memory(e.to_string()))??;

        let conn = Arc::new(std::sync::Mutex::new(conn));

        let embed_client = resolve_embed_client(cfg).await;

        Ok(Self { conn, embed_client })
    }

    /// Save (upsert) a key-value pair. Embeds the value if an embedding client is configured.
    ///
    /// # Errors
    /// Returns an error if the database write fails.
    ///
    /// # Panics
    /// Panics if the internal mutex is poisoned.
    #[spec(requires: !key.trim().is_empty())]
    pub async fn save(&self, key: &str, value: &str) -> Result<(), InboxError> {
        let embedding: Option<Vec<u8>> = if let Some(embed) = &self.embed_client {
            embed
                .embed(value)
                .await
                .ok()
                .map(|v| embed::vec_to_blob(&v))
        } else {
            None
        };

        let key = key.to_owned();
        let value = value.to_owned();
        let conn = Arc::clone(&self.conn);

        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            conn.execute(
                "INSERT INTO memories (key, value, embedding) VALUES (?1, ?2, ?3) \
                 ON CONFLICT(key) DO UPDATE SET \
                   value = excluded.value, \
                   embedding = excluded.embedding, \
                   updated_at = datetime('now')",
                rusqlite::params![key, value, embedding],
            )
            .map_err(|e| InboxError::Memory(e.to_string()))?;
            Ok::<_, InboxError>(())
        })
        .await
        .map_err(|e| InboxError::Memory(e.to_string()))??;

        Ok(())
    }

    /// Recall at most `limit` entries matching `query`.
    ///
    /// Uses hybrid vector + FTS5 recall with Reciprocal Rank Fusion when embeddings are
    /// enabled; falls back to FTS5-only when they are not.
    ///
    /// # Errors
    /// Returns an error if the database query fails.
    ///
    /// # Panics
    /// Panics if the internal mutex is poisoned.
    #[spec(requires: limit > 0)]
    pub async fn recall(&self, query: &str, limit: usize) -> Result<Vec<MemoryEntry>, InboxError> {
        let query_vec: Option<Vec<f32>> = if let Some(embed) = &self.embed_client {
            embed.embed(query).await.ok()
        } else {
            None
        };

        let query = query.to_owned();
        let conn = Arc::clone(&self.conn);

        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().unwrap();
            search::hybrid_recall(&conn, &query, query_vec.as_deref(), limit)
        })
        .await
        .map_err(|e| InboxError::Memory(e.to_string()))?
    }

    /// Test helper: create a `MemoryStore` backed by an in-memory `SQLite` database.
    ///
    /// # Errors
    /// Returns an error if the schema cannot be applied.
    #[cfg(any(test, feature = "test-helpers"))]
    pub fn new_in_memory() -> Result<Self, InboxError> {
        let conn = rusqlite::Connection::open_in_memory()
            .map_err(|e| InboxError::Memory(e.to_string()))?;
        apply_schema(&conn)?;
        Ok(Self {
            conn: Arc::new(std::sync::Mutex::new(conn)),
            embed_client: None,
        })
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn apply_schema(conn: &rusqlite::Connection) -> Result<(), InboxError> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS memories (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            key        TEXT NOT NULL UNIQUE,
            value      TEXT NOT NULL,
            embedding  BLOB,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
            USING fts5(key, value, content='memories', content_rowid='id');

        CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
            INSERT INTO memories_fts(rowid, key, value)
                VALUES (new.id, new.key, new.value);
        END;

        CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, key, value)
                VALUES ('delete', old.id, old.key, old.value);
            INSERT INTO memories_fts(rowid, key, value)
                VALUES (new.id, new.key, new.value);
        END;",
    )
    .map_err(|e| InboxError::Memory(e.to_string()))
}

async fn resolve_embed_client(cfg: &MemoryConfig) -> Option<embed::EmbedClient> {
    let endpoint = cfg.embedding_endpoint.as_ref()?;

    let client = embed::EmbedClient::new(
        endpoint.clone(),
        cfg.embedding_model
            .clone()
            .unwrap_or_else(|| "nomic-embed-text".into()),
        cfg.embedding_api_key.clone(),
    );

    // Probe to verify the endpoint works (or use configured dims to skip probe).
    let probe_ok = match cfg.embedding_dims {
        Some(_) => true,
        None => match client.embed("probe").await {
            Ok(_) => true,
            Err(e) => {
                warn!("Embedding probe failed, disabling embeddings: {e}");
                false
            }
        },
    };

    if probe_ok { Some(client) } else { None }
}
