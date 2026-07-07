//! KB-chunk storage: a `:KbChunk` node label kept **separate** from behavioral
//! `:Memory` nodes so Grafeo's label-scoped `hybrid_search`/`text_search` can
//! never mix the two — behavioral recall stays isolated by construction. Chunks
//! are content-agnostic (org notes today; tax docs / PDFs / articles later),
//! distinguished by a `source` property and a namespaced `id`.

use std::collections::HashSet;

use grafeo::{GrafeoDB, NodeId};
use tracing::warn;

use crate::error::InboxError;

use super::MemoryEntry;
use super::util::{format_vector, gql_escape, value_to_f64, value_to_string};

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

    let existing = session
        .execute(&format!("MATCH (c:KbChunk {{id: '{id_esc}'}}) RETURN c.id"))
        .map_err(|e| InboxError::Memory(format!("kb upsert check: {e}")))?;

    let stmt = if existing.is_empty() {
        kb_insert_stmt(chunk)
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

/// Build the `INSERT (:KbChunk {…})` statement for a chunk (with optional
/// embedding). Used by the upsert insert-branch and the transactional reindex.
fn kb_insert_stmt(chunk: &KbChunkWrite<'_>) -> String {
    let props = format!(
        "id: '{id}', value: '{val}', kind: 'kb-chunk', source: '{src}', \
         note_id: '{nid}', path: '{path}', fingerprint: '{fp}'",
        id = gql_escape(chunk.id),
        val = gql_escape(chunk.value),
        src = gql_escape(chunk.source),
        nid = gql_escape(chunk.note_id),
        path = gql_escape(chunk.path),
        fp = gql_escape(chunk.fingerprint),
    );
    chunk.embedding.map_or_else(
        || format!("INSERT (:KbChunk {{{props}}})"),
        |emb| {
            format!(
                "INSERT (:KbChunk {{{props}, embedding: vector({})}})",
                format_vector(emb)
            )
        },
    )
}

/// A pre-embedded chunk ready for a transactional reindex write.
pub(super) struct PreparedChunk {
    pub id: String,
    pub note_id: String,
    pub text: String,
    pub embedding: Option<Vec<f32>>,
}

/// Atomically replace the chunk set for a `(source, path[, note_scope])` unit:
/// in a single Grafeo transaction, delete the prior chunks and insert `prepared`
/// (already embedded outside the transaction). Rolls back on any failure so a
/// source is never left empty and concurrent recall never sees a gap.
///
/// `note_scope = None` deletes every chunk at `(source, path)` — the org-file
/// case (subtrees may have been added/removed). `Some(note_id)` scopes deletion
/// to one note — the shared-attachment case (a file linked by several notes).
pub(super) fn reindex_chunks(
    db: &GrafeoDB,
    source: &str,
    path: &str,
    note_scope: Option<&str>,
    prepared: &[PreparedChunk],
    active_fp: &str,
) -> Result<(), InboxError> {
    let session = db.session();
    session
        .execute("START TRANSACTION")
        .map_err(|e| InboxError::Memory(format!("kb reindex begin: {e}")))?;

    let run = || -> Result<(), InboxError> {
        let note_pred = note_scope
            .map(|n| format!(" AND c.note_id = '{}'", gql_escape(n)))
            .unwrap_or_default();
        let del = format!(
            "MATCH (c:KbChunk) WHERE c.source = '{}' AND c.path = '{}'{note_pred} DETACH DELETE c",
            gql_escape(source),
            gql_escape(path),
        );
        session
            .execute(&del)
            .map_err(|e| InboxError::Memory(format!("kb reindex delete: {e}")))?;

        // Chunk ids are content-hashed, so two identical chunks under the same
        // note collapse to one id — insert each id once (the old upsert path
        // deduped this; a blind insert would create duplicate nodes).
        let mut inserted: HashSet<&str> = HashSet::new();
        for c in prepared {
            if !inserted.insert(c.id.as_str()) {
                continue;
            }
            let write = KbChunkWrite {
                id: &c.id,
                value: &c.text,
                embedding: c.embedding.as_deref(),
                source,
                note_id: &c.note_id,
                path,
                fingerprint: active_fp,
            };
            session
                .execute(&kb_insert_stmt(&write))
                .map_err(|e| InboxError::Memory(format!("kb reindex insert: {e}")))?;
        }
        Ok(())
    };

    match run() {
        Ok(()) => session
            .execute("COMMIT")
            .map(|_| ())
            .map_err(|e| InboxError::Memory(format!("kb reindex commit: {e}"))),
        Err(e) => {
            let _ = session.execute("ROLLBACK");
            Err(e)
        }
    }
}

/// Look up cached extraction **output** for an attachment file. A hit requires
/// the same canonical `path`, `file_hash`, **and** `extraction_fp`, so a changed
/// file or a changed extractor config (languages/backend/version) misses and
/// forces re-extraction. The returned text is re-chunkable for any owning note
/// without re-running OCR. Query errors degrade to a miss (`None`).
pub(super) fn kb_source_lookup(
    db: &GrafeoDB,
    canonical_path: &str,
    file_hash: &str,
    extraction_fp: &str,
) -> Option<String> {
    // `ORDER BY indexed_at DESC LIMIT 1`: the write path keeps one row per path,
    // but should a duplicate ever slip in (e.g. an interleaved concurrent put),
    // lookup stays deterministic and prefers the newest entry.
    let q = format!(
        "MATCH (s:KbSource {{path: '{p}'}}) \
         WHERE s.file_hash = '{h}' AND s.extraction_fp = '{fp}' \
         RETURN s.text ORDER BY s.indexed_at DESC LIMIT 1",
        p = gql_escape(canonical_path),
        h = gql_escape(file_hash),
        fp = gql_escape(extraction_fp),
    );
    let res = db.session().execute(&q).ok()?;
    res.iter()
        .next()
        .and_then(|row| row.first().map(value_to_string))
}

/// Upsert the extraction-output cache entry for `canonical_path`. The path is the
/// identity (one row per file): in a single transaction the prior entry is
/// cleared and the new one inserted, so a changed `file_hash`/`extraction_fp`
/// supersedes stale text atomically — no torn clear-without-insert state, and no
/// duplicate row from an interleaved writer. This is a cache — a rolled-back
/// write only causes a later re-extraction, never a correctness loss.
pub(super) fn kb_source_store(
    db: &GrafeoDB,
    canonical_path: &str,
    file_hash: &str,
    extraction_fp: &str,
    text: &str,
    indexed_at: &str,
) -> Result<(), InboxError> {
    let session = db.session();
    let p = gql_escape(canonical_path);
    session
        .execute("START TRANSACTION")
        .map_err(|e| InboxError::Memory(format!("kb source begin: {e}")))?;

    let run = || -> Result<(), InboxError> {
        session
            .execute(&format!(
                "MATCH (s:KbSource {{path: '{p}'}}) DETACH DELETE s"
            ))
            .map_err(|e| InboxError::Memory(format!("kb source clear: {e}")))?;
        session
            .execute(&format!(
                "INSERT (:KbSource {{path: '{p}', file_hash: '{h}', \
                 extraction_fp: '{fp}', text: '{t}', indexed_at: '{ts}'}})",
                h = gql_escape(file_hash),
                fp = gql_escape(extraction_fp),
                t = gql_escape(text),
                ts = gql_escape(indexed_at),
            ))
            .map_err(|e| InboxError::Memory(format!("kb source insert: {e}")))?;
        Ok(())
    };

    match run() {
        Ok(()) => session
            .execute("COMMIT")
            .map(|_| ())
            .map_err(|e| InboxError::Memory(format!("kb source commit: {e}"))),
        Err(e) => {
            let _ = session.execute("ROLLBACK");
            Err(e)
        }
    }
}

/// KB-only recall over `:KbChunk` (never touches `:Memory`): hybrid vector+BM25,
/// then BM25 text, then a **pure-vector cosine fallback** so semantic-only
/// queries (cross-lingual, paraphrase, vague) still match when there is no
/// lexical overlap. Results are filtered to `active_fp` so a stale-fingerprint
/// chunk (left over from a model/chunker/prefix change not yet re-embedded) is
/// never co-queried. Query/index errors are swallowed to an empty result, so
/// this is infallible.
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

    // Pure-vector fallback: `hybrid_search` needs lexical overlap to seed
    // candidates, so a semantic-only query (cross-lingual, paraphrase, vague)
    // otherwise finds nothing. Scan embeddings by cosine, fingerprint-filtered
    // in the query so stale vector spaces never leak in. Mirrors memory recall.
    if let Some(qvec) = query_vec {
        let session = db.session();
        let vec_str = format_vector(qvec);
        if let Ok(result) = session.execute(&format!(
            "MATCH (c:KbChunk) \
             WHERE c.embedding IS NOT NULL AND c.fingerprint = '{fp}' \
             WITH c, cosine_similarity(c.embedding, vector({vec_str})) AS score \
             WHERE score > 0.5 \
             RETURN c.id, c.value, score \
             ORDER BY score DESC LIMIT {limit}",
            fp = gql_escape(active_fp),
        )) {
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
            if !entries.is_empty() {
                return entries;
            }
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
        EmbeddingFingerprint, KbChunkWrite, PreparedChunk, kb_id, kb_recall_entries,
        kb_source_lookup, kb_source_store, memory_id, reindex_chunks, upsert_kb_chunk,
    };
    use crate::error::InboxError;
    use grafeo::GrafeoDB;

    fn note_chunk_count(db: &GrafeoDB, note_id: &str) -> usize {
        db.session()
            .execute(&format!(
                "MATCH (c:KbChunk {{note_id: '{note_id}'}}) RETURN c.id"
            ))
            .map_or(0, |r| r.iter().count())
    }

    #[test]
    fn reindex_note_scope_only_replaces_that_note() {
        let db = GrafeoDB::new_in_memory();
        let prep = |note: &str, h: &str, t: &str| PreparedChunk {
            id: kb_id("attachment", note, "v1", h),
            note_id: note.to_owned(),
            text: t.to_owned(),
            embedding: None,
        };
        // Two notes share one attachment path.
        reindex_chunks(
            &db,
            "attachment",
            "/f.pdf",
            Some("A"),
            &[prep("A", "h1", "a")],
            "FP",
        )
        .expect("A");
        reindex_chunks(
            &db,
            "attachment",
            "/f.pdf",
            Some("B"),
            &[prep("B", "h2", "b")],
            "FP",
        )
        .expect("B");
        assert_eq!(note_chunk_count(&db, "A"), 1);
        assert_eq!(note_chunk_count(&db, "B"), 1);

        // Re-index only note A → B's chunk for the same path must survive.
        reindex_chunks(
            &db,
            "attachment",
            "/f.pdf",
            Some("A"),
            &[prep("A", "h3", "a2")],
            "FP",
        )
        .expect("A2");
        assert_eq!(note_chunk_count(&db, "A"), 1, "A replaced, still one chunk");
        assert_eq!(
            note_chunk_count(&db, "B"),
            1,
            "B's chunk untouched by A's reindex"
        );
    }

    #[test]
    fn reindex_deduplicates_identical_chunk_ids() {
        let db = GrafeoDB::new_in_memory();
        let dup = || PreparedChunk {
            id: kb_id("org", "n", "v1", "hh"),
            note_id: "n".to_owned(),
            text: "same".to_owned(),
            embedding: None,
        };
        // Two identical chunks share one content-hashed id.
        reindex_chunks(&db, "org", "/n.org", None, &[dup(), dup()], "FP").expect("reindex");
        let n = db
            .session()
            .execute("MATCH (c:KbChunk {id: 'kb:org:n:v1:hh'}) RETURN c.id")
            .map_or(0, |r| r.iter().count());
        assert_eq!(n, 1, "duplicate chunk ids must collapse to one node");
    }

    #[test]
    fn kb_source_cache_hits_only_on_matching_hash_and_fp() {
        let db = GrafeoDB::new_in_memory();
        kb_source_store(
            &db,
            "/a.pdf",
            "hash1",
            "shell|eng|v1|false",
            "cached text",
            "t0",
        )
        .expect("store");

        // Exact match → hit.
        assert_eq!(
            kb_source_lookup(&db, "/a.pdf", "hash1", "shell|eng|v1|false").as_deref(),
            Some("cached text")
        );
        // Changed file bytes → miss (must re-extract).
        assert!(kb_source_lookup(&db, "/a.pdf", "hash2", "shell|eng|v1|false").is_none());
        // Changed extractor config → miss (must re-extract).
        assert!(kb_source_lookup(&db, "/a.pdf", "hash1", "shell|rus|v1|false").is_none());
        // Different file → miss.
        assert!(kb_source_lookup(&db, "/b.pdf", "hash1", "shell|eng|v1|false").is_none());
    }

    #[test]
    fn kb_source_store_supersedes_stale_entry() {
        let db = GrafeoDB::new_in_memory();
        kb_source_store(&db, "/a.pdf", "h1", "fp1", "old", "t0").expect("first");
        // Re-extract with new bytes+config → single row, new text; old gone.
        kb_source_store(&db, "/a.pdf", "h2", "fp2", "new", "t1").expect("second");

        assert_eq!(
            kb_source_lookup(&db, "/a.pdf", "h2", "fp2").as_deref(),
            Some("new")
        );
        assert!(
            kb_source_lookup(&db, "/a.pdf", "h1", "fp1").is_none(),
            "stale entry must not linger"
        );
        let rows = db
            .session()
            .execute("MATCH (s:KbSource {path: '/a.pdf'}) RETURN s.path")
            .map_or(0, |r| r.iter().count());
        assert_eq!(rows, 1, "one cache row per file");
    }

    #[test]
    fn kb_source_lookup_is_deterministic_across_duplicate_rows() {
        // Even if two rows for one path/hash/fp ever coexist (e.g. an interleaved
        // concurrent put), lookup must deterministically return the newest.
        let db = GrafeoDB::new_in_memory();
        for (ts, text) in [("t0", "older"), ("t1", "newer")] {
            db.session()
                .execute(&format!(
                    "INSERT (:KbSource {{path: '/a.pdf', file_hash: 'h', \
                     extraction_fp: 'fp', text: '{text}', indexed_at: '{ts}'}})"
                ))
                .expect("insert dup");
        }
        assert_eq!(
            kb_source_lookup(&db, "/a.pdf", "h", "fp").as_deref(),
            Some("newer"),
            "newest indexed_at must win"
        );
    }

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

    #[test]
    fn recall_finds_by_vector_without_lexical_overlap() {
        let db = GrafeoDB::new_in_memory();
        let chunk = KbChunkWrite {
            id: "kb:org:n:v1:h",
            value: "alpha bravo charlie",
            embedding: Some(&[1.0, 0.0, 0.0]),
            source: "org",
            note_id: "n",
            path: "/n.org",
            fingerprint: "FP",
        };
        upsert_kb_chunk(&db, &chunk, "FP").expect("insert with embedding");

        // Query text shares no tokens with the chunk; only the vector matches.
        // Without the pure-vector fallback this returns empty (the cross-lingual bug).
        let hits = kb_recall_entries(
            &db,
            "zzz totally unrelated words",
            Some(&[1.0, 0.0, 0.0]),
            5,
            "FP",
        );
        assert!(
            hits.iter().any(|e| e.value.contains("alpha")),
            "pure-vector recall must find a semantically-matching chunk with no lexical overlap"
        );

        // A stale fingerprint must still exclude it even on the vector path.
        let stale = kb_recall_entries(
            &db,
            "zzz totally unrelated words",
            Some(&[1.0, 0.0, 0.0]),
            5,
            "OTHER",
        );
        assert!(
            stale.is_empty(),
            "vector fallback must honour the fingerprint filter"
        );
    }
}
