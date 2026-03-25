use std::path::Path;
use std::sync::Arc;

use anodized::spec;
use grafeo::GrafeoDB;
use tracing::warn;

use crate::config::MemoryConfig;
use crate::error::InboxError;

pub mod embed;

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
        metrics::counter!(crate::telemetry::MEMORY_OPS, "op" => "link", "status" => status)
            .increment(1);
        metrics::histogram!(crate::telemetry::MEMORY_DURATION, "op" => "link")
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
        metrics::counter!(crate::telemetry::MEMORY_OPS, "op" => "link", "status" => status)
            .increment(1);
        metrics::histogram!(crate::telemetry::MEMORY_DURATION, "op" => "link")
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
    v.as_float64().unwrap_or(0.0)
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
