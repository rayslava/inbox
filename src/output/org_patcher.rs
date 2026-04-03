//! In-place patching of org-mode files by entry `:ID:` property.
//!
//! Finds the org headline whose `:PROPERTIES:` drawer contains
//! `:ID:       <uuid>` and replaces the entire entry (up to the next
//! top-level headline or end-of-file) with `new_content`. The write
//! is atomic: the replacement is written to a `.tmp` file and then
//! renamed over the original.

use std::path::Path;

use anodized::spec;
use tokio::fs;
use uuid::Uuid;

use crate::error::InboxError;

/// Replace the org entry whose `:ID:` property equals `id` with `new_content`.
///
/// Returns `Ok(true)` if the entry was found and replaced, `Ok(false)` if not
/// found (e.g. it was already removed from the file). The replacement is
/// written atomically via a `.tmp` rename.
#[spec(requires: !new_content.is_empty())]
pub async fn patch_entry(path: &Path, id: Uuid, new_content: &str) -> Result<bool, InboxError> {
    let text = match fs::read_to_string(path).await {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(e) => {
            return Err(InboxError::Io(e));
        }
    };

    let id_str = id.to_string();
    let Some((entry_start, entry_end)) = locate_entry(&text, &id_str) else {
        return Ok(false);
    };

    let mut result = String::with_capacity(text.len());
    result.push_str(&text[..entry_start]);
    result.push_str(new_content);
    if !new_content.ends_with('\n') {
        result.push('\n');
    }
    result.push_str(&text[entry_end..]);

    let tmp_path = path.with_extension("org.tmp");
    fs::write(&tmp_path, &result)
        .await
        .map_err(InboxError::Io)?;
    fs::rename(&tmp_path, path).await.map_err(InboxError::Io)?;

    Ok(true)
}

/// Locate the byte range `[start, end)` of the org entry with the given `:ID:`.
///
/// `start` is the position of the `*` that begins the headline.
/// `end`   is the position of the `*` that begins the next top-level headline,
///         or the end of the file.
fn locate_entry(text: &str, id: &str) -> Option<(usize, usize)> {
    let id_needle = format!(":ID:       {id}");

    // Collect positions of all top-level headlines ("* " at column 0).
    let mut headline_positions: Vec<usize> = Vec::new();

    if text.starts_with("* ") {
        headline_positions.push(0);
    }
    let mut search = 0;
    while let Some(pos) = text[search..].find("\n* ") {
        let abs = search + pos + 1; // skip the '\n'
        headline_positions.push(abs);
        search = abs + 1;
    }

    // For each headline, determine the entry's extent and check for the ID.
    for (i, &start) in headline_positions.iter().enumerate() {
        let end = headline_positions.get(i + 1).copied().unwrap_or(text.len());

        let entry_slice = &text[start..end];
        if entry_slice.contains(&id_needle) {
            return Some((start, end));
        }
    }

    None
}
