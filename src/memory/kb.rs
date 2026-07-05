//! KB-chunk storage: a `:KbChunk` node label kept **separate** from behavioral
//! `:Memory` nodes so Grafeo's label-scoped `hybrid_search`/`text_search` can
//! never mix the two — behavioral recall stays isolated by construction. Chunks
//! are content-agnostic (org notes today; tax docs / PDFs / articles later),
//! distinguished by a `source` property and a namespaced `id`.

use grafeo::{GrafeoDB, NodeId};
use tracing::warn;

use crate::error::InboxError;

use super::MemoryEntry;
use super::util::{format_vector, gql_escape, value_to_string};

/// Identifies the vector space a chunk was embedded in. Chunks from a different
/// fingerprint are never co-queried; changing any field forces a re-embed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbeddingFingerprint {
    pub model: String,
    pub dims: usize,
    pub metric: String,
    pub normalization: String,
    pub chunker_version: String,
    /// The `embed_document` task prefix baked into every stored chunk vector
    /// (empty for symmetric embedders). Part of the vector-space identity:
    /// changing it must invalidate old chunks, since query vectors would then
    /// be compared against documents embedded under a different task prompt.
    pub doc_prefix: String,
}

impl EmbeddingFingerprint {
    /// Stable single-line tag stored on each chunk and compared on upsert.
    #[must_use]
    pub fn tag(&self) -> String {
        format!(
            "{}|{}|{}|{}|{}|{}",
            self.model,
            self.dims,
            self.metric,
            self.normalization,
            self.chunker_version,
            self.doc_prefix
        )
    }
}

/// Namespaced id for a behavioral memory: `memory:<source>:<key>`.
#[must_use]
pub fn memory_id(source: &str, key: &str) -> String {
    format!("memory:{source}:{key}")
}

/// Namespaced id for a KB chunk: `kb:<source>:<doc-id>:<chunker>:<hash>`.
#[must_use]
pub fn kb_id(source: &str, doc_id: &str, chunker: &str, hash: &str) -> String {
    format!("kb:{source}:{doc_id}:{chunker}:{hash}")
}

pub(super) fn create_kb_indexes(db: &GrafeoDB, dims: usize) {
    let query = format!(
        "CREATE VECTOR INDEX kb_vec_idx \
         ON :KbChunk(embedding) DIMENSION {dims} METRIC 'cosine'"
    );
    if let Err(e) = db.session().execute(&query) {
        let msg = e.to_string();
        if !msg.contains("already exists") && !msg.contains("duplicate") {
            warn!("KB vector index creation failed: {e}");
        }
    }
}

/// A KB chunk to write, bundled so the upsert stays under the argument-count lint.
pub(super) struct KbChunkWrite<'a> {
    pub id: &'a str,
    pub value: &'a str,
    pub embedding: Option<&'a [f32]>,
    pub source: &'a str,
    pub note_id: &'a str,
    pub path: &'a str,
    pub fingerprint: &'a str,
}

/// Upsert a KB chunk under its namespaced `id`. Rejects a chunk whose
/// `fingerprint` differs from `active_fp` so the store never mixes vector spaces.
pub(super) fn upsert_kb_chunk(
    db: &GrafeoDB,
    chunk: &KbChunkWrite<'_>,
    active_fp: &str,
) -> Result<(), InboxError> {
    if chunk.fingerprint != active_fp {
        return Err(InboxError::Memory(format!(
            "kb upsert rejected: chunk fingerprint '{}' != active '{active_fp}' \
             (re-embed required before mixing vector spaces)",
            chunk.fingerprint
        )));
    }

    let session = db.session();
    let id_esc = gql_escape(chunk.id);
    let props = format!(
        "id: '{id_esc}', value: '{val}', kind: 'kb-chunk', source: '{src}', \
         note_id: '{nid}', path: '{path}', fingerprint: '{fp}'",
        val = gql_escape(chunk.value),
        src = gql_escape(chunk.source),
        nid = gql_escape(chunk.note_id),
        path = gql_escape(chunk.path),
        fp = gql_escape(chunk.fingerprint),
    );

    let existing = session
        .execute(&format!("MATCH (c:KbChunk {{id: '{id_esc}'}}) RETURN c.id"))
        .map_err(|e| InboxError::Memory(format!("kb upsert check: {e}")))?;

    let stmt = if existing.is_empty() {
        chunk.embedding.map_or_else(
            || format!("INSERT (:KbChunk {{{props}}})"),
            |emb| {
                format!(
                    "INSERT (:KbChunk {{{props}, embedding: vector({})}})",
                    format_vector(emb)
                )
            },
        )
    } else {
        let set_embedding = chunk
            .embedding
            .map(|emb| format!(", c.embedding = vector({})", format_vector(emb)))
            .unwrap_or_default();
        format!(
            "MATCH (c:KbChunk {{id: '{id_esc}'}}) \
             SET c.value = '{val}', c.fingerprint = '{fp}'{set_embedding}",
            val = gql_escape(chunk.value),
            fp = gql_escape(chunk.fingerprint),
        )
    };

    session
        .execute(&stmt)
        .map_err(|e| InboxError::Memory(format!("kb upsert: {e}")))?;
    Ok(())
}

/// KB-only recall: hybrid vector+BM25 over `:KbChunk` (never touches `:Memory`).
/// Results are filtered to `active_fp` so a stale-fingerprint chunk (left over
/// from a model/chunker change not yet re-embedded) is never co-queried.
/// Query/index errors are swallowed to an empty result, so this is infallible.
pub(super) fn kb_recall_entries(
    db: &GrafeoDB,
    query: &str,
    query_vec: Option<&[f32]>,
    limit: usize,
    active_fp: &str,
) -> Vec<MemoryEntry> {
    // Over-fetch so stale-fingerprint chunks filtered out below don't crowd out
    // valid current ones in the top-k.
    let fetch = limit.saturating_mul(4).max(limit);

    if query_vec.is_some()
        && let Ok(results) = db.hybrid_search(
            "KbChunk",
            "value",
            "embedding",
            query,
            query_vec,
            fetch,
            None,
        )
        && !results.is_empty()
    {
        let entries = kb_node_ids_to_entries(db, &results, active_fp, limit);
        if !entries.is_empty() {
            return entries;
        }
    }

    if !query.trim().is_empty()
        && let Ok(results) = db.text_search("KbChunk", "value", query, fetch)
        && !results.is_empty()
    {
        let entries = kb_node_ids_to_entries(db, &results, active_fp, limit);
        if !entries.is_empty() {
            return entries;
        }
    }

    Vec::new()
}

fn kb_node_ids_to_entries(
    db: &GrafeoDB,
    results: &[(NodeId, f64)],
    active_fp: &str,
    limit: usize,
) -> Vec<MemoryEntry> {
    let mut entries = Vec::new();
    for &(node_id, score) in results {
        if entries.len() >= limit {
            break;
        }
        let Some(node) = db.get_node(node_id) else {
            continue;
        };
        let fp = node
            .get_property("fingerprint")
            .map(value_to_string)
            .unwrap_or_default();
        if fp != active_fp {
            continue;
        }
        let key = node
            .get_property("id")
            .map(value_to_string)
            .unwrap_or_default();
        let value = node
            .get_property("value")
            .map(value_to_string)
            .unwrap_or_default();
        entries.push(MemoryEntry { key, value, score });
    }
    entries
}

#[cfg(test)]
mod tests {
    use super::{
        EmbeddingFingerprint, KbChunkWrite, kb_id, kb_recall_entries, memory_id, upsert_kb_chunk,
    };
    use crate::error::InboxError;
    use grafeo::GrafeoDB;

    #[test]
    fn fingerprint_tag_is_stable() {
        let fp = EmbeddingFingerprint {
            model: "nomic".into(),
            dims: 768,
            metric: "cosine".into(),
            normalization: "none".into(),
            chunker_version: "v1".into(),
            doc_prefix: String::new(),
        };
        assert_eq!(fp.tag(), "nomic|768|cosine|none|v1|");
        // Enabling a document prefix changes the vector-space identity, so old
        // unprefixed chunks stop matching and are filtered from recall.
        let prefixed = EmbeddingFingerprint {
            doc_prefix: "search_document: ".into(),
            ..fp.clone()
        };
        assert_ne!(fp.tag(), prefixed.tag());
    }

    #[test]
    fn namespaced_ids_have_expected_shape() {
        assert_eq!(memory_id("wallabag", "k42"), "memory:wallabag:k42");
        assert_eq!(
            kb_id("taxes", "2024-return", "v1", "abc"),
            "kb:taxes:2024-return:v1:abc"
        );
    }

    #[test]
    fn upsert_rejects_fingerprint_mismatch() {
        let db = GrafeoDB::new_in_memory();
        let chunk = KbChunkWrite {
            id: "kb:org:n:v1:h",
            value: "x",
            embedding: None,
            source: "org",
            note_id: "n",
            path: "/n.org",
            fingerprint: "OTHER",
        };
        let err = upsert_kb_chunk(&db, &chunk, "ACTIVE").expect_err("must reject");
        assert!(matches!(err, InboxError::Memory(_)));
    }

    #[test]
    fn recall_filters_stale_fingerprint() {
        let db = GrafeoDB::new_in_memory();
        let _ = db.create_text_index("KbChunk", "value");
        let chunk = KbChunkWrite {
            id: "kb:org:n:v1:h",
            value: "quantum note",
            embedding: None,
            source: "org",
            note_id: "n",
            path: "/n.org",
            fingerprint: "OLD",
        };
        upsert_kb_chunk(&db, &chunk, "OLD").expect("insert with matching fp");

        // Same active fingerprint → chunk is returned.
        let hit = kb_recall_entries(&db, "quantum", None, 5, "OLD");
        assert!(hit.iter().any(|e| e.value.contains("quantum")));

        // Active fingerprint changed → stale chunk is filtered out at recall.
        let none = kb_recall_entries(&db, "quantum", None, 5, "NEW");
        assert!(none.is_empty());
    }
}
