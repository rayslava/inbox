//! Tri-state result type for vision completions, so callers can distinguish a
//! genuine outage (retry later) from "no vision backend at all" (skip).

/// Why a vision completion produced no text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum VisionUnavailable {
    /// Every vision-capable backend errored or was in cooldown — a transient
    /// outage. The image text is unread; the node should be held pending and
    /// retried once a backend recovers.
    AllUnavailable,
    /// No vision-capable backend is configured/eligible for this request, so
    /// there is nothing to retry.
    NoVisionBackend,
}
