use std::collections::HashMap;

use anodized::spec;

use crate::error::InboxError;

use super::MemoryEntry;
use super::embed::blob_to_vec;

/// Cosine similarity between two equal-length f32 slices. Returns 0 if either norm is zero.
#[must_use]
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

/// Hybrid recall: vector search + FTS5 keyword search merged with Reciprocal Rank Fusion.
///
/// If neither search yields results, falls back to most-recently-updated entries.
///
/// # Errors
/// Returns an `InboxError` on DB failure.
#[spec(requires: limit > 0)]
pub fn hybrid_recall(
    conn: &rusqlite::Connection,
    fts_query: &str,
    query_vec: Option<&[f32]>,
    limit: usize,
) -> Result<Vec<MemoryEntry>, InboxError> {
    // ── Vector search ────────────────────────────────────────────────────────
    let vec_ranked: Vec<i64> = if let Some(qvec) = query_vec {
        let mut stmt = conn
            .prepare("SELECT id, embedding FROM memories WHERE embedding IS NOT NULL")
            .map_err(|e| InboxError::Memory(e.to_string()))?;

        let mut scored: Vec<(i64, f32)> = stmt
            .query_map([], |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, Vec<u8>>(1)?))
            })
            .map_err(|e| InboxError::Memory(e.to_string()))?
            .filter_map(std::result::Result::ok)
            .filter_map(|(id, blob)| {
                let v = blob_to_vec(&blob);
                if v.len() == qvec.len() {
                    Some((id, cosine(qvec, &v)))
                } else {
                    None
                }
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit * 2);
        scored.into_iter().map(|(id, _)| id).collect()
    } else {
        Vec::new()
    };

    // ── FTS5 keyword search ──────────────────────────────────────────────────
    let fts_ranked: Vec<i64> = if fts_query.trim().is_empty() {
        Vec::new()
    } else {
        let n = i64::try_from(limit * 2).unwrap_or(20);
        conn.prepare(
            "SELECT rowid FROM memories_fts WHERE memories_fts MATCH ? ORDER BY rank LIMIT ?",
        )
        .and_then(|mut stmt| {
            stmt.query_map(rusqlite::params![fts_query, n], |row| row.get::<_, i64>(0))
                .map(|rows| rows.filter_map(std::result::Result::ok).collect())
        })
        .unwrap_or_default()
    };

    // ── Reciprocal Rank Fusion ───────────────────────────────────────────────
    let mut scores: HashMap<i64, f64> = HashMap::new();
    let mut rank: u32 = 0;
    for id in &vec_ranked {
        *scores.entry(*id).or_insert(0.0) += 0.7 / (61.0 + f64::from(rank));
        rank = rank.saturating_add(1);
    }
    rank = 0;
    for id in &fts_ranked {
        *scores.entry(*id).or_insert(0.0) += 0.3 / (61.0 + f64::from(rank));
        rank = rank.saturating_add(1);
    }

    let mut ranked: Vec<(i64, f64)> = scores.into_iter().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    ranked.truncate(limit);

    if ranked.is_empty() {
        return fallback_recent(conn, limit);
    }

    // Fetch full entries for ranked IDs
    let mut entries = Vec::with_capacity(ranked.len());
    for (id, score) in ranked {
        if let Ok(entry) = conn.query_row(
            "SELECT key, value FROM memories WHERE id = ?",
            rusqlite::params![id],
            |row| {
                Ok(MemoryEntry {
                    key: row.get(0)?,
                    value: row.get(1)?,
                    score,
                })
            },
        ) {
            entries.push(entry);
        }
    }

    Ok(entries)
}

fn fallback_recent(
    conn: &rusqlite::Connection,
    limit: usize,
) -> Result<Vec<MemoryEntry>, InboxError> {
    let n = i64::try_from(limit).unwrap_or(10);
    let mut stmt = conn
        .prepare("SELECT key, value FROM memories ORDER BY updated_at DESC LIMIT ?")
        .map_err(|e| InboxError::Memory(e.to_string()))?;
    let entries = stmt
        .query_map(rusqlite::params![n], |row| {
            Ok(MemoryEntry {
                key: row.get(0)?,
                value: row.get(1)?,
                score: 0.0,
            })
        })
        .map_err(|e| InboxError::Memory(e.to_string()))?
        .filter_map(std::result::Result::ok)
        .collect();
    Ok(entries)
}
