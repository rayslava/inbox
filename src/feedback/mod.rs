use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[cfg(test)]
mod tests;

// ── Rating ───────────────────────────────────────────────────────────────────

/// A 1–3 star rating: 1 = bad, 2 = ok, 3 = good.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "u8", into = "u8")]
pub struct FeedbackRating(u8);

impl FeedbackRating {
    /// Create a rating from a raw value (1–3). Returns `None` if out of range.
    #[must_use]
    pub fn new(value: u8) -> Option<Self> {
        (1..=3).contains(&value).then_some(Self(value))
    }

    #[must_use]
    pub fn value(self) -> u8 {
        self.0
    }
}

impl From<FeedbackRating> for u8 {
    fn from(r: FeedbackRating) -> Self {
        r.0
    }
}

impl TryFrom<u8> for FeedbackRating {
    type Error = String;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        Self::new(value).ok_or_else(|| format!("rating must be 1–3, got {value}"))
    }
}

impl std::fmt::Display for FeedbackRating {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let stars = match self.0 {
            1 => "\u{2b50}",
            2 => "\u{2b50}\u{2b50}",
            3 => "\u{2b50}\u{2b50}\u{2b50}",
            _ => unreachable!(),
        };
        f.write_str(stars)
    }
}

// ── Request / Entry / Stats ──────────────────────────────────────────────────

/// Incoming feedback from any adapter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackRequest {
    pub message_id: Uuid,
    pub rating: FeedbackRating,
    #[serde(default)]
    pub comment: Option<String>,
}

/// A stored feedback record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackEntry {
    pub message_id: String,
    pub rating: u8,
    pub comment: String,
    pub created_at: DateTime<Utc>,
    pub source: String,
    pub title: String,
}

/// Aggregate feedback statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FeedbackStats {
    pub total: u64,
    /// Counts per rating: index 0 = rating 1, index 1 = rating 2, index 2 = rating 3.
    pub by_rating: [u64; 3],
    pub average: f64,
}
