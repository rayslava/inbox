use std::path::Path;
use std::sync::Arc;

use anodized::spec;
use grafeo::GrafeoDB;
use tracing::warn;

use crate::config::MemoryConfig;
use crate::error::InboxError;

pub mod embed;
pub(crate) mod feedback;
mod queries;
mod store_feedback;
mod util;

#[cfg(test)]
mod tests;

// ── Public types ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct MemoryEntry {
    pub key: String,
    pub value: String,
    pub score: f64,
}

#[derive(Debug, Clone)]
pub struct SourceEntry {
    pub kind: String,
    pub source_id: String,
    pub title: String,
}

#[derive(Debug, Clone)]
pub struct RelatedMemory {
    pub key: String,
    pub value: String,
    pub relation: String,
    pub direction: String,
}

#[derive(Debug, Clone)]
pub struct RecallOutcome {
    pub memory_key: String,
    pub times_recalled: u32,
    pub avg_rating: f64,
    pub sample_comments: Vec<String>,
}

// ── MemoryStore ───────────────────────────────────────────────────────────────

pub struct MemoryStore {
    db: Arc<GrafeoDB>,
    embed_client: Option<embed::EmbedClient>,
}

impl MemoryStore {
    /// Open (or create) a `MemoryStore` backed by Grafeo at `db_path`,
    /// optionally probing the embedding endpoint to detect vector dimensions.
    ///
    /// # Errors
    /// Returns an error if the database cannot be opened or indexes cannot be created.
    pub async fn open(cfg: &MemoryConfig, db_path: &Path) -> Result<Self, InboxError> {
        let path = db_path.to_owned();
        let db = tokio::task::spawn_blocking(move || {
            let grafeo_cfg = grafeo::Config::persistent(&path);
            GrafeoDB::with_config(grafeo_cfg)
                .map_err(|e| InboxError::Memory(format!("Grafeo open: {e}")))
        })
        .await
        .map_err(|e| InboxError::Memory(e.to_string()))??;

        let db = Arc::new(db);

        let embed_client = resolve_embed_client(cfg).await;

        // Detect embedding dimensions via probe or config.
        let embedding_dims = if let Some(dims) = cfg.embedding_dims {
            Some(dims)
        } else if let Some(ref client) = embed_client {
            client.embed("probe").await.ok().map(|v| v.len())
        } else {
            None
        };

        // Create indexes if we know the dimensions.
        if let Some(dims) = embedding_dims {
            let db_ref = Arc::clone(&db);
            tokio::task::spawn_blocking(move || {
                queries::create_indexes(&db_ref, dims);
            })
            .await
            .map_err(|e| InboxError::Memory(e.to_string()))?;
        }

        // Always create the text index for BM25 recall.
        let db_ref = Arc::clone(&db);
        tokio::task::spawn_blocking(move || {
            if let Err(e) = db_ref.create_text_index("Memory", "value") {
                // Ignore "already exists" style errors on reopening.
                warn!("Text index creation (may already exist): {e}");
            }
        })
        .await
        .map_err(|e| InboxError::Memory(e.to_string()))?;

        Ok(Self { db, embed_client })
    }

    /// Save (upsert) a key-value pair. Embeds the value if an embedding client is configured.
    ///
    /// # Errors
    /// Returns an error if the database write fails.
    #[spec(requires: !key.trim().is_empty())]
    pub async fn save(&self, key: &str, value: &str) -> Result<(), InboxError> {
        let start = std::time::Instant::now();
        let embedding: Option<Vec<f32>> = if let Some(embed) = &self.embed_client {
            embed.embed(value).await.ok()
        } else {
            None
        };

        let key = key.to_owned();
        let value = value.to_owned();
        let db = Arc::clone(&self.db);

        let result = tokio::task::spawn_blocking(move || {
            queries::upsert_memory(&db, &key, &value, embedding.as_deref())
        })
        .await
        .map_err(|e| InboxError::Memory(e.to_string()))?;

        let status = if result.is_ok() { "success" } else { "failure" };
        metrics::counter!(crate::telemetry::MEMORY_OPS, "op" => "save", "status" => status)
            .increment(1);
        metrics::histogram!(crate::telemetry::MEMORY_DURATION, "op" => "save")
            .record(start.elapsed().as_secs_f64());
        result
    }

    /// Link a memory to a source (e.g. email, telegram message).
    ///
    /// # Errors
    /// Returns an error if the relationship cannot be created.
    pub async fn link_source(
        &self,
        memory_key: &str,
        source_kind: &str,
        source_id: &str,
        title: &str,
    ) -> Result<(), InboxError> {
        let start = std::time::Instant::now();
        let memory_key = memory_key.to_owned();
        let source_kind = source_kind.to_owned();
        let source_id = source_id.to_owned();
        let title = title.to_owned();
        let db = Arc::clone(&self.db);

        let result = tokio::task::spawn_blocking(move || {
            queries::link_memory_source(&db, &memory_key, &source_kind, &source_id, &title)
        })
        .await
        .map_err(|e| InboxError::Memory(e.to_string()))?;

        let status = if result.is_ok() { "success" } else { "failure" };
        metrics::counter!(crate::telemetry::MEMORY_OPS, "op" => "link_source", "status" => status)
            .increment(1);
        metrics::histogram!(crate::telemetry::MEMORY_DURATION, "op" => "link_source")
            .record(start.elapsed().as_secs_f64());
        result
    }

    /// Link two memories together with a named relationship.
    ///
    /// # Errors
    /// Returns an error if the relationship cannot be created.
    pub async fn link_memories(
        &self,
        from_key: &str,
        to_key: &str,
        relation: &str,
    ) -> Result<(), InboxError> {
        let start = std::time::Instant::now();
        let from_key = from_key.to_owned();
        let to_key = to_key.to_owned();
        let relation = relation.to_owned();
        let db = Arc::clone(&self.db);

        let result = tokio::task::spawn_blocking(move || {
            queries::link_memory_to_memory(&db, &from_key, &to_key, &relation)
        })
        .await
        .map_err(|e| InboxError::Memory(e.to_string()))?;

        let status = if result.is_ok() { "success" } else { "failure" };
        metrics::counter!(crate::telemetry::MEMORY_OPS, "op" => "link_memories", "status" => status)
            .increment(1);
        metrics::histogram!(crate::telemetry::MEMORY_DURATION, "op" => "link_memories")
            .record(start.elapsed().as_secs_f64());
        result
    }

    /// Recall at most `limit` entries matching `query`.
    ///
    /// Uses hybrid vector + BM25 recall with Reciprocal Rank Fusion when embeddings are
    /// enabled; falls back to BM25-only or recent entries when they are not.
    ///
    /// # Errors
    /// Returns an error if the database query fails.
    #[spec(requires: limit > 0)]
    pub async fn recall(&self, query: &str, limit: usize) -> Result<Vec<MemoryEntry>, InboxError> {
        let start = std::time::Instant::now();
        let query_vec: Option<Vec<f32>> = if let Some(embed) = &self.embed_client {
            embed.embed(query).await.ok()
        } else {
            None
        };

        let query = query.to_owned();
        let db = Arc::clone(&self.db);

        let result = tokio::task::spawn_blocking(move || {
            queries::recall_entries(&db, &query, query_vec.as_deref(), limit)
        })
        .await
        .map_err(|e| InboxError::Memory(e.to_string()))?;

        let status = if result.is_ok() { "success" } else { "failure" };
        metrics::counter!(crate::telemetry::MEMORY_OPS, "op" => "recall", "status" => status)
            .increment(1);
        metrics::histogram!(crate::telemetry::MEMORY_DURATION, "op" => "recall")
            .record(start.elapsed().as_secs_f64());
        result
    }

    /// Find memories connected to a given memory key via graph traversal.
    ///
    /// # Errors
    /// Returns an error if the graph query fails.
    pub async fn context(&self, query: &str, hops: u32) -> Result<Vec<MemoryEntry>, InboxError> {
        let start = std::time::Instant::now();
        let query = query.to_owned();
        let db = Arc::clone(&self.db);

        let result = tokio::task::spawn_blocking(move || queries::graph_context(&db, &query, hops))
            .await
            .map_err(|e| InboxError::Memory(e.to_string()))?;

        let status = if result.is_ok() { "success" } else { "failure" };
        metrics::counter!(crate::telemetry::MEMORY_OPS, "op" => "context", "status" => status)
            .increment(1);
        metrics::histogram!(crate::telemetry::MEMORY_DURATION, "op" => "context")
            .record(start.elapsed().as_secs_f64());
        result
    }

    /// Find sources linked to a memory by key.
    ///
    /// # Errors
    /// Returns an error if the graph query fails.
    pub async fn sources(&self, memory_key: &str) -> Result<Vec<SourceEntry>, InboxError> {
        let start = std::time::Instant::now();
        let key = memory_key.to_owned();
        let db = Arc::clone(&self.db);

        let result = tokio::task::spawn_blocking(move || queries::find_sources(&db, &key))
            .await
            .map_err(|e| InboxError::Memory(e.to_string()))?;

        let status = if result.is_ok() { "success" } else { "failure" };
        metrics::counter!(crate::telemetry::MEMORY_OPS, "op" => "sources", "status" => status)
            .increment(1);
        metrics::histogram!(crate::telemetry::MEMORY_DURATION, "op" => "sources")
            .record(start.elapsed().as_secs_f64());
        result
    }

    /// Test helper: create a `MemoryStore` backed by an in-memory Grafeo database.
    ///
    /// # Errors
    /// Returns an error if the database cannot be created.
    #[cfg(any(test, feature = "test-helpers"))]
    pub fn new_in_memory() -> Result<Self, InboxError> {
        let db = GrafeoDB::new_in_memory();
        let db = Arc::new(db);

        // Create text index for BM25.
        if let Err(e) = db.create_text_index("Memory", "value") {
            warn!("Text index creation: {e}");
        }

        Ok(Self {
            db,
            embed_client: None,
        })
    }
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
