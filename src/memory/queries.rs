use grafeo::{GrafeoDB, NodeId};
use tracing::warn;

use crate::error::InboxError;

use super::{MemoryEntry, RecallOutcome, RelatedMemory, SourceEntry};

pub(super) fn create_indexes(db: &GrafeoDB, dims: usize) {
    // Grafeo's GQL dialect does not support IF NOT EXISTS on vector indexes.
    // Attempt creation and silently ignore errors — the index already exists
    // on subsequent restarts.
    let query = format!(
        "CREATE VECTOR INDEX mem_vec_idx \
         ON :Memory(embedding) DIMENSION {dims} METRIC 'cosine'"
    );
    if let Err(e) = db.session().execute(&query) {
        let msg = e.to_string();
        if !msg.contains("already exists") && !msg.contains("duplicate") {
            warn!("Vector index creation failed: {e}");
        }
    }
}

pub(super) fn upsert_memory(
    db: &GrafeoDB,
    key: &str,
    value: &str,
    embedding: Option<&[f32]>,
) -> Result<(), InboxError> {
    let session = db.session();

    let existing = session
        .execute(&format!(
            "MATCH (m:Memory {{key: '{key_esc}'}}) RETURN m.key",
            key_esc = gql_escape(key)
        ))
        .map_err(|e| InboxError::Memory(format!("upsert check: {e}")))?;

    if existing.is_empty() {
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

pub(super) fn link_memory_source(
    db: &GrafeoDB,
    memory_key: &str,
    source_kind: &str,
    source_id: &str,
    title: &str,
) -> Result<(), InboxError> {
    let session = db.session();

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

pub(super) fn link_memory_to_memory(
    db: &GrafeoDB,
    from_key: &str,
    to_key: &str,
    relation: &str,
) -> Result<(), InboxError> {
    let session = db.session();
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

pub(super) fn recall_entries(
    db: &GrafeoDB,
    query: &str,
    query_vec: Option<&[f32]>,
    limit: usize,
) -> Result<Vec<MemoryEntry>, InboxError> {
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

    if !query.trim().is_empty() {
        if let Ok(results) = db.text_search("Memory", "value", query, limit) {
            if !results.is_empty() {
                return Ok(node_ids_to_entries(db, &results));
            }
        }
    }

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

    fallback_recent(db, limit)
}

pub(super) fn graph_context(
    db: &GrafeoDB,
    query: &str,
    hops: u32,
) -> Result<Vec<MemoryEntry>, InboxError> {
    let session = db.session();
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

pub(super) fn find_sources(
    db: &GrafeoDB,
    memory_key: &str,
) -> Result<Vec<SourceEntry>, InboxError> {
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

pub(super) fn graph_related_memories(
    db: &GrafeoDB,
    memory_key: &str,
    hops: u32,
) -> Result<Vec<RelatedMemory>, InboxError> {
    let session = db.session();
    let key = gql_escape(memory_key);
    let hops = hops.clamp(1, 3);
    let mut entries = Vec::new();

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

    resolve_direct_relations(db, memory_key, &mut entries);

    Ok(entries)
}

fn resolve_direct_relations(db: &GrafeoDB, memory_key: &str, entries: &mut [RelatedMemory]) {
    let session = db.session();
    let key = gql_escape(memory_key);

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

pub(super) fn insert_recall_event(
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
        let _ = session.execute(&format!(
            "MATCH (e:RecallEvent {{message_id: '{mid}'}}), (m:Memory {{key: '{k}'}}) \
             INSERT (e)-[:RECALLED]->(m)"
        ));
    }

    let _ = session.execute(&format!(
        "MATCH (e:RecallEvent {{message_id: '{mid}'}}), (s:Source {{source_id: '{mid}'}}) \
         INSERT (e)-[:FOR_MESSAGE]->(s)"
    ));

    Ok(())
}

pub(super) fn query_recall_outcomes(db: &GrafeoDB, memory_keys: &[String]) -> Vec<RecallOutcome> {
    let session = db.session();
    let mut outcomes = Vec::new();

    for key in memory_keys {
        let k = gql_escape(key);

        let Ok(event_rows) = session.execute(&format!(
            "MATCH (m:Memory {{key: '{k}'}})<-[:RECALLED]-(e:RecallEvent) \
             RETURN e.message_id"
        )) else {
            continue;
        };

        if event_rows.is_empty() {
            continue;
        }

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

fn node_ids_to_entries(db: &GrafeoDB, results: &[(NodeId, f64)]) -> Vec<MemoryEntry> {
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

use super::util::{format_vector, gql_escape, value_to_f64, value_to_string};
