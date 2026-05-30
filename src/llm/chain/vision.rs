//! Per-backend routing decision for image-bearing requests.
//!
//! Kept out of `chain.rs` so the main tool-loop driver stays focused (and under
//! the file-size limit). The rule: a request carrying images must never reach a
//! backend that cannot see them, *unless* recognized image text has already been
//! folded into the prompt — in which case a text-only backend can still help.

use std::borrow::Cow;

use tracing::debug;

use crate::llm::{LlmClient, LlmRequest};

/// What the chain should do with one backend for a given request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum VisionDecision {
    /// Send the request unchanged.
    Run,
    /// Send the request with image attachments removed (text-only backend, but
    /// recognized image text is available in the prompt).
    StripImages,
    /// Skip this backend entirely — it cannot serve this image request.
    Skip,
}

/// Decide how `backend` should handle `req`.
#[must_use]
pub(super) fn decide(req: &LlmRequest, backend: &dyn LlmClient) -> VisionDecision {
    if !req.needs_vision() || backend.vision_supported() {
        VisionDecision::Run
    } else if req.has_image_text {
        VisionDecision::StripImages
    } else {
        VisionDecision::Skip
    }
}

/// Prepare the per-backend request: the original for `Run`, an image-stripped
/// clone for `StripImages`, or `None` (logged) when the backend is skipped.
#[must_use]
pub(super) fn prepare<'a>(
    req: &'a LlmRequest,
    backend: &dyn LlmClient,
) -> Option<Cow<'a, LlmRequest>> {
    match decide(req, backend) {
        VisionDecision::Run => Some(Cow::Borrowed(req)),
        VisionDecision::StripImages => {
            let mut stripped = req.clone();
            stripped.images.clear();
            Some(Cow::Owned(stripped))
        }
        VisionDecision::Skip => {
            debug!(
                backend = backend.name(),
                model = backend.model(),
                "Skipping non-vision backend for an image request"
            );
            None
        }
    }
}
