//! `kb_index`: walk the org corpus, split notes into heading chunks, and upsert
//! them into the shared store as `kind=kb-chunk` (see `memory::kb`). The chunker
//! is pure (`chunk`); this module handles note-id extraction, file/dir walking,
//! and the embed+store round-trip via [`MemoryStore::kb_save`].

pub mod chunk;

use std::path::{Path, PathBuf};

use tracing::warn;

use crate::error::InboxError;
use crate::memory::{MemoryStore, kb};

/// Source tag for chunks indexed from the local org corpus.
pub const ORG_SOURCE: &str = "org";

/// Extract a note id: the org-roam file-level `:ID:` if present, else the file
/// stem. Scans only the leading lines where a top-level PROPERTIES drawer lives.
#[must_use]
pub fn extract_note_id(content: &str, path: &Path) -> String {
    for line in content.lines().take(15) {
        if let Some(rest) = line.trim().strip_prefix(":ID:") {
            let id = rest.trim();
            if !id.is_empty() {
                return id.to_owned();
            }
        }
    }
    path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_owned()
}

/// Chunk `content` and upsert each chunk. Returns the number of chunks stored.
///
/// # Errors
/// Returns an error if a chunk fails to embed or write.
pub async fn index_content(
    store: &MemoryStore,
    source: &str,
    note_id: &str,
    path: &str,
    content: &str,
) -> Result<usize, InboxError> {
    let chunks = chunk::chunk_org(content);
    for c in &chunks {
        let id = kb::kb_id(source, note_id, chunk::CHUNKER_VERSION, &c.hash);
        store.kb_save(&id, &c.text, source, note_id, path).await?;
    }
    Ok(chunks.len())
}

/// Read and index a single org file. Returns the number of chunks stored.
///
/// # Errors
/// Returns an error if the file cannot be read or a chunk fails to store.
pub async fn index_file(store: &MemoryStore, path: &Path) -> Result<usize, InboxError> {
    let content = tokio::fs::read_to_string(path)
        .await
        .map_err(|e| InboxError::Memory(format!("read {}: {e}", path.display())))?;
    let note_id = extract_note_id(&content, path);
    index_content(
        store,
        ORG_SOURCE,
        &note_id,
        &path.to_string_lossy(),
        &content,
    )
    .await
}

/// Index every `*.org` file under `dir` (recursively), skipping hidden entries,
/// `*.gpg`, and `*.sync-conflict*`. A file that fails to index is logged and
/// skipped rather than aborting the sweep. Returns the total chunks stored.
///
/// # Errors
/// Returns an error only if the directory cannot be scanned.
pub async fn index_directory(store: &MemoryStore, dir: &Path) -> Result<usize, InboxError> {
    let dir = dir.to_path_buf();
    let files = tokio::task::spawn_blocking(move || collect_org_files(&dir))
        .await
        .map_err(|e| InboxError::Memory(e.to_string()))?;

    let mut total = 0;
    for path in files {
        match index_file(store, &path).await {
            Ok(n) => total += n,
            Err(e) => warn!("kb_index: skipping {}: {e}", path.display()),
        }
    }
    Ok(total)
}

/// Recursively collect indexable `*.org` files, skipping hidden entries,
/// `*.gpg`, and Syncthing conflict copies.
fn collect_org_files(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with('.') || name.contains(".sync-conflict") {
                continue;
            }
            // `file_type()` does not follow symlinks, so a symlinked directory is
            // neither recursed into nor indexed — avoids symlink-cycle loops.
            let Ok(ft) = entry.file_type() else {
                continue;
            };
            if ft.is_dir() {
                stack.push(entry.path());
            } else if ft.is_file() && name.ends_with(".org") {
                out.push(entry.path());
            }
        }
    }
    out
}

#[cfg(test)]
mod tests;
