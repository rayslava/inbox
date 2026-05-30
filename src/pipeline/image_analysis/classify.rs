//! Cheap, local heuristic to classify recognized image text as an
//! interface/screenshot vs an ordinary photo. Kept minimal on purpose — richer
//! signals (UI vocabulary, filename, dimensions) are deferred until a real
//! misclassification is observed.

use crate::message::ImageAnalysisKind;

/// Classify by recognized-text shape: enough characters, or several non-blank
/// lines, looks like an interface; little or no text is a plain photo.
#[must_use]
pub(super) fn classify(recognized: &str, interface_min_chars: usize) -> ImageAnalysisKind {
    let text = recognized.trim();
    if text.is_empty() {
        return ImageAnalysisKind::Photo;
    }
    let nonblank_lines = text.lines().filter(|l| !l.trim().is_empty()).count();
    if text.chars().count() >= interface_min_chars || nonblank_lines >= 3 {
        ImageAnalysisKind::Interface
    } else {
        ImageAnalysisKind::Photo
    }
}
