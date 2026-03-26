use chrono::DateTime;
use grafeo::GrafeoDB;

use crate::error::InboxError;
use crate::feedback::{FeedbackEntry, FeedbackStats};

use super::{gql_escape, value_to_string};

/// Extract a numeric value from a Grafeo `Value`, handling both integers and floats.
fn value_to_i64(v: &grafeo::Value) -> i64 {
    if let Some(i) = v.as_int64() {
        return i;
    }
    // Fallback for Grafeo returning small integers as floats (ratings 1-3).
    if let Some(f) = v.as_float64() {
        if f.is_finite() {
            // Parse the rounded float's string representation to avoid truncation casts.
            let s = format!("{:.0}", f.round());
            if let Ok(i) = s.parse::<i64>() {
                return i;
            }
        }
    }
    0
}

/// Extract a value as a `u8` rating (clamped to 0..=255).
fn value_to_u8(v: &grafeo::Value) -> u8 {
    u8::try_from(value_to_i64(v).clamp(0, i64::from(u8::MAX))).unwrap_or(0)
}

// ── Insert / upsert ──────────────────────────────────────────────────────────

pub(super) fn insert_feedback(
    db: &GrafeoDB,
    message_id: &str,
    rating: u8,
    comment: &str,
    created_at: i64,
    source: &str,
    title: &str,
) -> Result<(), InboxError> {
    let session = db.session();
    let mid = gql_escape(message_id);
    let cmt = gql_escape(comment);
    let src = gql_escape(source);
    let ttl = gql_escape(title);

    let existing = session
        .execute(&format!(
            "MATCH (f:Feedback {{message_id: '{mid}'}}) RETURN f.message_id"
        ))
        .map_err(|e| InboxError::Memory(format!("feedback check: {e}")))?;

    if existing.is_empty() {
        session
            .execute(&format!(
                "INSERT (:Feedback {{message_id: '{mid}', rating: {rating}, \
                 comment: '{cmt}', created_at: {created_at}, source: '{src}', title: '{ttl}'}})"
            ))
            .map_err(|e| InboxError::Memory(format!("feedback insert: {e}")))?;
    } else {
        session
            .execute(&format!(
                "MATCH (f:Feedback {{message_id: '{mid}'}}) \
                 SET f.rating = {rating}, f.comment = '{cmt}', \
                 f.created_at = {created_at}, f.source = '{src}', f.title = '{ttl}'"
            ))
            .map_err(|e| InboxError::Memory(format!("feedback update: {e}")))?;
    }

    // Link to existing Source node if present.
    let source_exists = session
        .execute(&format!(
            "MATCH (s:Source {{source_id: '{mid}'}}) RETURN s.source_id"
        ))
        .map_err(|e| InboxError::Memory(format!("feedback source check: {e}")))?;

    if !source_exists.is_empty() {
        // Only create edge if not already linked.
        let edge_exists = session
            .execute(&format!(
                "MATCH (f:Feedback {{message_id: '{mid}'}})-[:FEEDBACK_FOR]->(s:Source {{source_id: '{mid}'}}) \
                 RETURN f.message_id"
            ))
            .map_err(|e| InboxError::Memory(format!("feedback edge check: {e}")))?;

        if edge_exists.is_empty() {
            session
                .execute(&format!(
                    "MATCH (f:Feedback {{message_id: '{mid}'}}), (s:Source {{source_id: '{mid}'}}) \
                     INSERT (f)-[:FEEDBACK_FOR]->(s)"
                ))
                .map_err(|e| InboxError::Memory(format!("feedback link: {e}")))?;
        }
    }

    Ok(())
}

// ── Query single ─────────────────────────────────────────────────────────────

pub(super) fn get_feedback(
    db: &GrafeoDB,
    message_id: &str,
) -> Result<Option<FeedbackEntry>, InboxError> {
    let session = db.session();
    let mid = gql_escape(message_id);

    let result = session
        .execute(&format!(
            "MATCH (f:Feedback {{message_id: '{mid}'}}) \
             RETURN f.message_id, f.rating, f.comment, f.created_at, f.source, f.title"
        ))
        .map_err(|e| InboxError::Memory(format!("feedback query: {e}")))?;

    let mut iter = result.iter();
    let row = match iter.next() {
        Some(r) if r.len() >= 6 => r,
        _ => return Ok(None),
    };

    let epoch = value_to_i64(&row[3]);
    let created_at = DateTime::from_timestamp(epoch, 0).unwrap_or_default();

    Ok(Some(FeedbackEntry {
        message_id: value_to_string(&row[0]),
        rating: value_to_u8(&row[1]),
        comment: value_to_string(&row[2]),
        created_at,
        source: value_to_string(&row[4]),
        title: value_to_string(&row[5]),
    }))
}

// ── Stats ────────────────────────────────────────────────────────────────────

pub(super) fn get_feedback_stats(db: &GrafeoDB) -> Result<FeedbackStats, InboxError> {
    let session = db.session();

    let result = session
        .execute("MATCH (f:Feedback) RETURN f.rating")
        .map_err(|e| InboxError::Memory(format!("feedback stats: {e}")))?;

    let mut by_rating = [0u64; 3];
    let mut total = 0u64;
    let mut sum = 0u64;

    for row in result.iter() {
        if let Some(val) = row.first() {
            let rating = value_to_u8(val);
            if (1..=3).contains(&rating) {
                by_rating[(rating - 1) as usize] += 1;
                total += 1;
                sum += u64::from(rating);
            }
        }
    }

    let average = if total > 0 {
        // Rating sums and counts stay small (max 3 * count), so f64 is exact here.
        let sum_f: f64 = f64::from(u32::try_from(sum).unwrap_or(u32::MAX));
        let total_f: f64 = f64::from(u32::try_from(total).unwrap_or(u32::MAX));
        sum_f / total_f
    } else {
        0.0
    };

    Ok(FeedbackStats {
        total,
        by_rating,
        average,
    })
}

// ── Recent feedback ──────────────────────────────────────────────────

pub(super) fn get_recent_feedback(
    db: &GrafeoDB,
    max_rating: u8,
    limit: usize,
) -> Result<Vec<FeedbackEntry>, InboxError> {
    let session = db.session();

    let result = session
        .execute("MATCH (f:Feedback) RETURN f.message_id, f.rating, f.comment, f.created_at, f.source, f.title")
        .map_err(|e| InboxError::Memory(format!("recent feedback: {e}")))?;

    let mut entries = Vec::new();
    for row in result.iter() {
        if row.len() < 6 {
            continue;
        }
        let rating = value_to_u8(&row[1]);
        if rating > max_rating || rating == 0 {
            continue;
        }
        let epoch = value_to_i64(&row[3]);
        let created_at = DateTime::from_timestamp(epoch, 0).unwrap_or_default();
        entries.push(FeedbackEntry {
            message_id: value_to_string(&row[0]),
            rating,
            comment: value_to_string(&row[2]),
            created_at,
            source: value_to_string(&row[4]),
            title: value_to_string(&row[5]),
        });
    }

    // Sort by created_at descending (Grafeo may not support ORDER BY).
    entries.sort_by(|a, b| b.created_at.cmp(&a.created_at));
    entries.truncate(limit);

    Ok(entries)
}

// ── Update comment ───────────────────────────────────────────────────────────

pub(super) fn update_feedback_comment(
    db: &GrafeoDB,
    message_id: &str,
    comment: &str,
) -> Result<bool, InboxError> {
    let session = db.session();
    let mid = gql_escape(message_id);
    let cmt = gql_escape(comment);

    let existing = session
        .execute(&format!(
            "MATCH (f:Feedback {{message_id: '{mid}'}}) RETURN f.message_id"
        ))
        .map_err(|e| InboxError::Memory(format!("feedback comment check: {e}")))?;

    if existing.is_empty() {
        return Ok(false);
    }

    session
        .execute(&format!(
            "MATCH (f:Feedback {{message_id: '{mid}'}}) SET f.comment = '{cmt}'"
        ))
        .map_err(|e| InboxError::Memory(format!("feedback comment update: {e}")))?;

    Ok(true)
}
