use std::path::Path;
use std::sync::Arc;

use anodized::spec;
use grafeo::GrafeoDB;
use tracing::warn;

use crate::config::MemoryConfig;
use crate::error::InboxError;

pub mod embed;
pub(crate) mod feedback;

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
                create_indexes(&db_ref, dims);
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
            upsert_memory(&db, &key, &value, embedding.as_deref())
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
            link_memory_source(&db, &memory_key, &source_kind, &source_id, &title)
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
            link_memory_to_memory(&db, &from_key, &to_key, &relation)
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
            recall_entries(&db, &query, query_vec.as_deref(), limit)
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

        let result = tokio::task::spawn_blocking(move || graph_context(&db, &query, hops))
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

        let result = tokio::task::spawn_blocking(move || find_sources(&db, &key))
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

    // ── Feedback methods ─────────────────────────────────────────────────────

    /// Save (upsert) a feedback entry and link it to the source message.
    ///
    /// # Errors
    /// Returns an error if the database write fails.
    pub async fn save_feedback(
        &self,
        entry: &crate::feedback::FeedbackEntry,
    ) -> Result<(), InboxError> {
        let start = std::time::Instant::now();
        let message_id = entry.message_id.clone();
        let rating = entry.rating;
        let comment = entry.comment.clone();
        let created_at = entry.created_at.timestamp();
        let source = entry.source.clone();
        let title = entry.title.clone();
        let db = Arc::clone(&self.db);

        let rating_str = rating.to_string();
        let source_label = source.clone();

        let result = tokio::task::spawn_blocking(move || {
            feedback::insert_feedback(
                &db,
                &message_id,
                rating,
                &comment,
                created_at,
                &source,
                &title,
            )
        })
        .await
        .map_err(|e| InboxError::Memory(e.to_string()))?;

        let status = if result.is_ok() { "success" } else { "failure" };
        metrics::counter!(crate::telemetry::FEEDBACK_TOTAL, "rating" => rating_str.clone(), "source" => source_label, "status" => status)
            .increment(1);
        metrics::histogram!(crate::telemetry::FEEDBACK_DURATION, "op" => "save")
            .record(start.elapsed().as_secs_f64());
        if result.is_ok() {
            metrics::gauge!(crate::telemetry::FEEDBACK_RATING_DISTRIBUTION, "rating" => rating_str)
                .increment(1.0);
        }
        result
    }

    /// Query feedback for a specific message.
    ///
    /// # Errors
    /// Returns an error if the database query fails.
    pub async fn query_feedback(
        &self,
        message_id: &str,
    ) -> Result<Option<crate::feedback::FeedbackEntry>, InboxError> {
        let start = std::time::Instant::now();
        let mid = message_id.to_owned();
        let db = Arc::clone(&self.db);

        let result = tokio::task::spawn_blocking(move || feedback::get_feedback(&db, &mid))
            .await
            .map_err(|e| InboxError::Memory(e.to_string()))?;

        let status = if result.is_ok() { "success" } else { "failure" };
        metrics::counter!(crate::telemetry::MEMORY_OPS, "op" => "feedback_query", "status" => status)
            .increment(1);
        metrics::histogram!(crate::telemetry::FEEDBACK_DURATION, "op" => "query")
            .record(start.elapsed().as_secs_f64());
        result
    }

    /// Compute aggregate feedback statistics.
    ///
    /// # Errors
    /// Returns an error if the database query fails.
    pub async fn feedback_stats(&self) -> Result<crate::feedback::FeedbackStats, InboxError> {
        let start = std::time::Instant::now();
        let db = Arc::clone(&self.db);

        let result = tokio::task::spawn_blocking(move || feedback::get_feedback_stats(&db))
            .await
            .map_err(|e| InboxError::Memory(e.to_string()))?;

        let status = if result.is_ok() { "success" } else { "failure" };
        metrics::counter!(crate::telemetry::MEMORY_OPS, "op" => "feedback_stats", "status" => status)
            .increment(1);
        metrics::histogram!(crate::telemetry::FEEDBACK_DURATION, "op" => "stats")
            .record(start.elapsed().as_secs_f64());
        result
    }

    /// Update the comment on an existing feedback entry.
    /// Returns `true` if the feedback existed and was updated.
    ///
    /// # Errors
    /// Returns an error if the database write fails.
    pub async fn update_feedback_comment(
        &self,
        message_id: &str,
        comment: &str,
    ) -> Result<bool, InboxError> {
        let start = std::time::Instant::now();
        let mid = message_id.to_owned();
        let cmt = comment.to_owned();
        let db = Arc::clone(&self.db);

        let result =
            tokio::task::spawn_blocking(move || feedback::update_feedback_comment(&db, &mid, &cmt))
                .await
                .map_err(|e| InboxError::Memory(e.to_string()))?;

        let status = if result.is_ok() { "success" } else { "failure" };
        metrics::counter!(crate::telemetry::FEEDBACK_COMMENTS_TOTAL, "source" => "direct", "status" => status)
            .increment(1);
        metrics::histogram!(crate::telemetry::FEEDBACK_DURATION, "op" => "update_comment")
            .record(start.elapsed().as_secs_f64());
        result
    }

    // ── Pre-load methods ─────────────────────────────────────────────────

    /// Find memories related to a given key via graph edges, returning relation types.
    ///
    /// # Errors
    /// Returns an error if the graph query fails.
    pub async fn related_memories(
        &self,
        memory_key: &str,
        hops: u32,
    ) -> Result<Vec<RelatedMemory>, InboxError> {
        let start = std::time::Instant::now();
        let key = memory_key.to_owned();
        let db = Arc::clone(&self.db);

        let result = tokio::task::spawn_blocking(move || graph_related_memories(&db, &key, hops))
            .await
            .map_err(|e| InboxError::Memory(e.to_string()))?;

        let status = if result.is_ok() { "success" } else { "failure" };
        metrics::counter!(crate::telemetry::MEMORY_OPS, "op" => "related_memories", "status" => status)
            .increment(1);
        metrics::histogram!(crate::telemetry::MEMORY_DURATION, "op" => "related_memories")
            .record(start.elapsed().as_secs_f64());
        result
    }

    /// Fetch recent feedback entries with rating at or below `max_rating`.
    ///
    /// # Errors
    /// Returns an error if the database query fails.
    pub async fn recent_feedback(
        &self,
        max_rating: u8,
        limit: usize,
    ) -> Result<Vec<crate::feedback::FeedbackEntry>, InboxError> {
        let start = std::time::Instant::now();
        let db = Arc::clone(&self.db);

        let result = tokio::task::spawn_blocking(move || {
            feedback::get_recent_feedback(&db, max_rating, limit)
        })
        .await
        .map_err(|e| InboxError::Memory(e.to_string()))?;

        let status = if result.is_ok() { "success" } else { "failure" };
        metrics::counter!(crate::telemetry::MEMORY_OPS, "op" => "recent_feedback", "status" => status)
            .increment(1);
        metrics::histogram!(crate::telemetry::FEEDBACK_DURATION, "op" => "recent")
            .record(start.elapsed().as_secs_f64());
        result
    }

    /// Log which memories were recalled for a given message, creating a `:RecallEvent` node.
    ///
    /// # Errors
    /// Returns an error if the database write fails.
    pub async fn log_recall_event(
        &self,
        message_id: &str,
        recalled_keys: &[String],
        source_name: &str,
    ) -> Result<(), InboxError> {
        let start = std::time::Instant::now();
        let mid = message_id.to_owned();
        let keys = recalled_keys.to_vec();
        let src = source_name.to_owned();
        let db = Arc::clone(&self.db);

        let result =
            tokio::task::spawn_blocking(move || insert_recall_event(&db, &mid, &keys, &src))
                .await
                .map_err(|e| InboxError::Memory(e.to_string()))?;

        let status = if result.is_ok() { "success" } else { "failure" };
        metrics::counter!(crate::telemetry::MEMORY_OPS, "op" => "log_recall", "status" => status)
            .increment(1);
        metrics::histogram!(crate::telemetry::MEMORY_DURATION, "op" => "log_recall")
            .record(start.elapsed().as_secs_f64());
        result
    }

    /// Find historical recall outcomes for a set of memory keys by correlating
    /// recall events with feedback.
    pub async fn recall_outcomes(&self, memory_keys: &[String]) -> Vec<RecallOutcome> {
        let start = std::time::Instant::now();
        let keys = memory_keys.to_vec();
        let db = Arc::clone(&self.db);

        let result = tokio::task::spawn_blocking(move || query_recall_outcomes(&db, &keys))
            .await
            .unwrap_or_default();

        metrics::counter!(crate::telemetry::MEMORY_OPS, "op" => "recall_outcomes", "status" => "success")
            .increment(1);
        metrics::histogram!(crate::telemetry::MEMORY_DURATION, "op" => "recall_outcomes")
            .record(start.elapsed().as_secs_f64());
        result
    }
}

// ── Internal helpers ─────────────────────────────────────────────────────────

fn create_indexes(db: &GrafeoDB, dims: usize) {
    let query = format!(
        "CREATE VECTOR INDEX IF NOT EXISTS mem_vec_idx \
         ON :Memory(embedding) DIMENSION {dims} METRIC 'cosine'"
    );
    if let Err(e) = db.session().execute(&query) {
        warn!("Vector index creation (may already exist): {e}");
    }
}

fn upsert_memory(
    db: &GrafeoDB,
    key: &str,
    value: &str,
    embedding: Option<&[f32]>,
) -> Result<(), InboxError> {
    let session = db.session();

    // Check if memory with this key already exists.
    let existing = session
        .execute(&format!(
            "MATCH (m:Memory {{key: '{key_esc}'}}) RETURN m.key",
            key_esc = gql_escape(key)
        ))
        .map_err(|e| InboxError::Memory(format!("upsert check: {e}")))?;

    if existing.is_empty() {
        // Insert new memory node.
        if let Some(emb) = embedding {
            let vec_str = format_vector(emb);
            session
                .execute(&format!(
                    "INSERT (:Memory {{key: '{key_esc}', value: '{val_esc}', \
                     embedding: vector({vec_str})}})",
                    key_esc = gql_escape(key),
                    val_esc = gql_escape(value),
                ))
                .map_err(|e| InboxError::Memory(format!("insert with embedding: {e}")))?;
        } else {
            session
                .execute(&format!(
                    "INSERT (:Memory {{key: '{key_esc}', value: '{val_esc}'}})",
                    key_esc = gql_escape(key),
                    val_esc = gql_escape(value),
                ))
                .map_err(|e| InboxError::Memory(format!("insert: {e}")))?;
        }
    } else {
        // Update existing memory.
        if let Some(emb) = embedding {
            let vec_str = format_vector(emb);
            session
                .execute(&format!(
                    "MATCH (m:Memory {{key: '{key_esc}'}}) \
                     SET m.value = '{val_esc}', m.embedding = vector({vec_str})",
                    key_esc = gql_escape(key),
                    val_esc = gql_escape(value),
                ))
                .map_err(|e| InboxError::Memory(format!("update with embedding: {e}")))?;
        } else {
            session
                .execute(&format!(
                    "MATCH (m:Memory {{key: '{key_esc}'}}) SET m.value = '{val_esc}'",
                    key_esc = gql_escape(key),
                    val_esc = gql_escape(value),
                ))
                .map_err(|e| InboxError::Memory(format!("update: {e}")))?;
        }
    }

    Ok(())
}

fn link_memory_source(
    db: &GrafeoDB,
    memory_key: &str,
    source_kind: &str,
    source_id: &str,
    title: &str,
) -> Result<(), InboxError> {
    let session = db.session();

    // Ensure source node exists (upsert by source_id).
    let existing = session
        .execute(&format!(
            "MATCH (s:Source {{source_id: '{sid}'}}) RETURN s.source_id",
            sid = gql_escape(source_id),
        ))
        .map_err(|e| InboxError::Memory(format!("source check: {e}")))?;

    if existing.is_empty() {
        session
            .execute(&format!(
                "INSERT (:Source {{kind: '{kind}', source_id: '{sid}', title: '{title}'}})",
                kind = gql_escape(source_kind),
                sid = gql_escape(source_id),
                title = gql_escape(title),
            ))
            .map_err(|e| InboxError::Memory(format!("source insert: {e}")))?;
    }

    // Create edge.
    session
        .execute(&format!(
            "MATCH (m:Memory {{key: '{key}'}}), (s:Source {{source_id: '{sid}'}}) \
             INSERT (m)-[:FROM_SOURCE]->(s)",
            key = gql_escape(memory_key),
            sid = gql_escape(source_id),
        ))
        .map_err(|e| InboxError::Memory(format!("link source: {e}")))?;

    Ok(())
}

fn link_memory_to_memory(
    db: &GrafeoDB,
    from_key: &str,
    to_key: &str,
    relation: &str,
) -> Result<(), InboxError> {
    let session = db.session();
    // Use uppercase edge type from relation.
    let edge_type = relation.to_uppercase().replace(' ', "_");
    session
        .execute(&format!(
            "MATCH (a:Memory {{key: '{from}'}}), (b:Memory {{key: '{to}'}}) \
             INSERT (a)-[:{edge_type}]->(b)",
            from = gql_escape(from_key),
            to = gql_escape(to_key),
        ))
        .map_err(|e| InboxError::Memory(format!("link memories: {e}")))?;
    Ok(())
}

fn recall_entries(
    db: &GrafeoDB,
    query: &str,
    query_vec: Option<&[f32]>,
    limit: usize,
) -> Result<Vec<MemoryEntry>, InboxError> {
    // Try hybrid search first (if both text and vector indexes exist).
    if query_vec.is_some() {
        if let Ok(results) = db.hybrid_search(
            "Memory",
            "value",
            "embedding",
            query,
            query_vec,
            limit,
            None,
        ) {
            if !results.is_empty() {
                return Ok(node_ids_to_entries(db, &results));
            }
        }
    }

    // Fall back to text-only search.
    if !query.trim().is_empty() {
        if let Ok(results) = db.text_search("Memory", "value", query, limit) {
            if !results.is_empty() {
                return Ok(node_ids_to_entries(db, &results));
            }
        }
    }

    // Fall back to vector-only search.
    if let Some(qvec) = query_vec {
        let session = db.session();
        let vec_str = format_vector(qvec);
        let result = session
            .execute(&format!(
                "MATCH (m:Memory) \
                 WHERE m.embedding IS NOT NULL \
                 WITH m, cosine_similarity(m.embedding, vector({vec_str})) AS score \
                 WHERE score > 0.5 \
                 RETURN m.key, m.value, score \
                 ORDER BY score DESC LIMIT {limit}"
            ))
            .map_err(|e| InboxError::Memory(format!("vector recall: {e}")))?;

        if !result.is_empty() {
            let mut entries = Vec::new();
            for row in result.iter() {
                if row.len() >= 3 {
                    entries.push(MemoryEntry {
                        key: value_to_string(&row[0]),
                        value: value_to_string(&row[1]),
                        score: value_to_f64(&row[2]),
                    });
                }
            }
            return Ok(entries);
        }
    }

    // Final fallback: most recent entries.
    fallback_recent(db, limit)
}

fn graph_context(db: &GrafeoDB, query: &str, hops: u32) -> Result<Vec<MemoryEntry>, InboxError> {
    let session = db.session();
    // Find memories matching the query, then traverse outward.
    let hop_pattern = if hops <= 1 {
        "-[*1..1]-".to_owned()
    } else {
        format!("-[*1..{hops}]-")
    };

    let result = session
        .execute(&format!(
            "MATCH (m:Memory {{key: '{key}'}}){hop_pattern}(n) \
             WHERE n:Memory \
             RETURN n.key, n.value",
            key = gql_escape(query),
        ))
        .map_err(|e| InboxError::Memory(format!("graph context: {e}")))?;

    let mut entries = Vec::new();
    for row in result.iter() {
        if let (Some(key), Some(value)) = (row.first(), row.get(1)) {
            entries.push(MemoryEntry {
                key: value_to_string(key),
                value: value_to_string(value),
                score: 0.0,
            });
        }
    }
    Ok(entries)
}

fn find_sources(db: &GrafeoDB, memory_key: &str) -> Result<Vec<SourceEntry>, InboxError> {
    let session = db.session();
    let result = session
        .execute(&format!(
            "MATCH (m:Memory {{key: '{key}'}})-[:FROM_SOURCE]->(s:Source) \
             RETURN s.kind, s.source_id, s.title",
            key = gql_escape(memory_key),
        ))
        .map_err(|e| InboxError::Memory(format!("find sources: {e}")))?;

    let mut entries = Vec::new();
    for row in result.iter() {
        if row.len() >= 3 {
            entries.push(SourceEntry {
                kind: value_to_string(&row[0]),
                source_id: value_to_string(&row[1]),
                title: value_to_string(&row[2]),
            });
        }
    }
    Ok(entries)
}

fn graph_related_memories(
    db: &GrafeoDB,
    memory_key: &str,
    hops: u32,
) -> Result<Vec<RelatedMemory>, InboxError> {
    let session = db.session();
    let key = gql_escape(memory_key);
    let hops = hops.clamp(1, 3);
    let mut entries = Vec::new();

    // Outgoing edges.
    let hop_pat = format!("-[*1..{hops}]->");
    let out = session
        .execute(&format!(
            "MATCH (m:Memory {{key: '{key}'}}){hop_pat}(n:Memory) \
             RETURN n.key, n.value"
        ))
        .map_err(|e| InboxError::Memory(format!("related out: {e}")))?;

    for row in out.iter() {
        if row.len() >= 2 {
            entries.push(RelatedMemory {
                key: value_to_string(&row[0]),
                value: value_to_string(&row[1]),
                relation: String::new(),
                direction: "outgoing".into(),
            });
        }
    }

    // Incoming edges.
    let hop_pat = format!("<-[*1..{hops}]-");
    let inc = session
        .execute(&format!(
            "MATCH (m:Memory {{key: '{key}'}}){hop_pat}(n:Memory) \
             RETURN n.key, n.value"
        ))
        .map_err(|e| InboxError::Memory(format!("related in: {e}")))?;

    for row in inc.iter() {
        if row.len() >= 2 {
            let related_key = value_to_string(&row[0]);
            // Avoid duplicates from bidirectional traversal.
            if !entries.iter().any(|e| e.key == related_key) {
                entries.push(RelatedMemory {
                    key: related_key,
                    value: value_to_string(&row[1]),
                    relation: String::new(),
                    direction: "incoming".into(),
                });
            }
        }
    }

    // Try to resolve relation types for direct (1-hop) connections.
    resolve_direct_relations(db, memory_key, &mut entries);

    Ok(entries)
}

fn resolve_direct_relations(db: &GrafeoDB, memory_key: &str, entries: &mut [RelatedMemory]) {
    let session = db.session();
    let key = gql_escape(memory_key);

    // Query direct outgoing edges with relation label.
    // Grafeo variable-length paths don't expose edge labels, so we query 1-hop
    // edges explicitly and attempt to infer the label from the result set.
    for direction in &["out", "in"] {
        let query = if *direction == "out" {
            format!(
                "MATCH (m:Memory {{key: '{key}'}})-[r]->(n:Memory) \
                 RETURN n.key, labels(r)"
            )
        } else {
            format!(
                "MATCH (n:Memory)-[r]->(m:Memory {{key: '{key}'}}) \
                 RETURN n.key, labels(r)"
            )
        };

        if let Ok(rows) = session.execute(&query) {
            for row in rows.iter() {
                if row.len() >= 2 {
                    let nk = value_to_string(&row[0]);
                    let label = value_to_string(&row[1]);
                    if let Some(entry) = entries.iter_mut().find(|e| e.key == nk) {
                        if entry.relation.is_empty() {
                            entry.relation = label;
                        }
                    }
                }
            }
        }
    }
}

fn insert_recall_event(
    db: &GrafeoDB,
    message_id: &str,
    recalled_keys: &[String],
    source_name: &str,
) -> Result<(), InboxError> {
    let session = db.session();
    let mid = gql_escape(message_id);
    let src = gql_escape(source_name);
    let ts = chrono::Utc::now().timestamp();

    session
        .execute(&format!(
            "INSERT (:RecallEvent {{message_id: '{mid}', recalled_at: {ts}, source: '{src}'}})"
        ))
        .map_err(|e| InboxError::Memory(format!("recall event insert: {e}")))?;

    for key in recalled_keys {
        let k = gql_escape(key);
        // Link to memory.
        let _ = session.execute(&format!(
            "MATCH (e:RecallEvent {{message_id: '{mid}'}}), (m:Memory {{key: '{k}'}}) \
             INSERT (e)-[:RECALLED]->(m)"
        ));
    }

    // Link to source if exists.
    let _ = session.execute(&format!(
        "MATCH (e:RecallEvent {{message_id: '{mid}'}}), (s:Source {{source_id: '{mid}'}}) \
         INSERT (e)-[:FOR_MESSAGE]->(s)"
    ));

    Ok(())
}

fn query_recall_outcomes(db: &GrafeoDB, memory_keys: &[String]) -> Vec<RecallOutcome> {
    let session = db.session();
    let mut outcomes = Vec::new();

    for key in memory_keys {
        let k = gql_escape(key);

        // Step 1: Find recall events that recalled this memory.
        let Ok(event_rows) = session.execute(&format!(
            "MATCH (m:Memory {{key: '{k}'}})<-[:RECALLED]-(e:RecallEvent) \
             RETURN e.message_id"
        )) else {
            continue;
        };

        if event_rows.is_empty() {
            continue;
        }

        // Step 2: For each recall event, find feedback via the shared message_id.
        let mut total_rating = 0.0_f64;
        let mut count = 0u32;
        let mut comments = Vec::new();

        for row in event_rows.iter() {
            let Some(mid_val) = row.first() else {
                continue;
            };
            let mid = gql_escape(&value_to_string(mid_val));

            let Ok(fb_rows) = session.execute(&format!(
                "MATCH (f:Feedback {{message_id: '{mid}'}}) \
                 RETURN f.rating, f.comment"
            )) else {
                continue;
            };

            for fb_row in fb_rows.iter() {
                if fb_row.len() >= 2 {
                    total_rating += value_to_f64(&fb_row[0]);
                    count += 1;
                    let comment = value_to_string(&fb_row[1]);
                    if !comment.is_empty() && comments.len() < 3 {
                        comments.push(comment);
                    }
                }
            }
        }

        if count > 0 {
            outcomes.push(RecallOutcome {
                memory_key: key.clone(),
                times_recalled: count,
                avg_rating: total_rating / f64::from(count),
                sample_comments: comments,
            });
        }
    }

    outcomes
}

fn fallback_recent(db: &GrafeoDB, limit: usize) -> Result<Vec<MemoryEntry>, InboxError> {
    let session = db.session();
    let result = session
        .execute(&format!(
            "MATCH (m:Memory) RETURN m.key, m.value LIMIT {limit}"
        ))
        .map_err(|e| InboxError::Memory(format!("fallback recent: {e}")))?;

    let mut entries = Vec::new();
    for row in result.iter() {
        if let (Some(key), Some(value)) = (row.first(), row.get(1)) {
            entries.push(MemoryEntry {
                key: value_to_string(key),
                value: value_to_string(value),
                score: 0.0,
            });
        }
    }
    Ok(entries)
}

fn node_ids_to_entries(db: &GrafeoDB, results: &[(grafeo::NodeId, f64)]) -> Vec<MemoryEntry> {
    let mut entries = Vec::with_capacity(results.len());
    for &(node_id, score) in results {
        if let Some(node) = db.get_node(node_id) {
            let key = node
                .get_property("key")
                .map(value_to_string)
                .unwrap_or_default();
            let value = node
                .get_property("value")
                .map(value_to_string)
                .unwrap_or_default();
            entries.push(MemoryEntry { key, value, score });
        }
    }
    entries
}

fn value_to_string(v: &grafeo::Value) -> String {
    match v {
        grafeo::Value::String(s) => s.to_string(),
        other => strip_quotes(&other.to_string()),
    }
}

fn value_to_f64(v: &grafeo::Value) -> f64 {
    if let Some(f) = v.as_float64() {
        return f;
    }
    // Grafeo may store small numbers as integers; parse via string to avoid truncation.
    if let Some(i) = v.as_int64() {
        return f64::from(i32::try_from(i).unwrap_or(0));
    }
    0.0
}

fn strip_quotes(s: &str) -> String {
    s.trim_matches('"').trim_matches('\'').to_owned()
}

fn format_vector(v: &[f32]) -> String {
    let parts: Vec<String> = v.iter().map(|f| format!("{f}")).collect();
    format!("[{}]", parts.join(", "))
}

fn gql_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('\'', "\\'")
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
