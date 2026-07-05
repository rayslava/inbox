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

/// Extract the file-level note id: the `:ID:` **inside the top-of-file
/// `:PROPERTIES:` drawer** (before the first heading), else the file stem. Only a
/// drawer-scoped `:ID:` counts — a stray `:ID:` line in body text does not.
#[must_use]
pub fn extract_note_id(content: &str, path: &Path) -> String {
    let mut in_props = false;
    for line in content.lines().take(30) {
        let t = line.trim();
        // File-level drawer only: stop at the first heading.
        let stars = t.chars().take_while(|&c| c == '*').count();
        if stars > 0 && t[stars..].starts_with(' ') {
            break;
        }
        if t.eq_ignore_ascii_case(":PROPERTIES:") {
            in_props = true;
        } else if in_props {
            if t.eq_ignore_ascii_case(":END:") {
                break;
            }
            let b = t.as_bytes();
            if b.len() >= 4 && b[..4].eq_ignore_ascii_case(b":id:") {
                let id = t[4..].trim();
                if !id.is_empty() {
                    return id.to_owned();
                }
            }
        }
    }
    path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_owned()
}

/// Chunk `content` and upsert each chunk under its **owning entry id** (the
/// chunk's inherited `:ID:`, falling back to `fallback_note_id` when a chunk has
/// none). Returns the number of chunks stored.
///
/// # Errors
/// Returns an error if a chunk fails to embed or write.
pub async fn index_content(
    store: &MemoryStore,
    source: &str,
    fallback_note_id: &str,
    path: &str,
    content: &str,
) -> Result<usize, InboxError> {
    let inputs: Vec<(String, String, String)> = chunk::chunk_org(content)
        .into_iter()
        .map(|c| {
            let note_id = c.note_id.unwrap_or_else(|| fallback_note_id.to_owned());
            let id = kb::kb_id(source, &note_id, chunk::CHUNKER_VERSION, &c.hash);
            (id, note_id, c.text)
        })
        .collect();
    // Whole-file replace (note_scope = None): a file's subtrees may have been
    // added or removed, so clear all its chunks and write the current set.
    store.kb_reindex(source, path, None, inputs).await
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

/// Index the whole org corpus configured at `[memory].kb_root` into the KB:
/// open the memory store, then `index_directory` over the root. The `main`
/// `index-kb` command is a thin wrapper over this (kept testable here).
///
/// Opens only the `MemoryStore` (async, no `block_on`, runtime-flavor safe) —
/// it does **not** build the LLM chain, so indexing never triggers backend
/// startup side effects (e.g. a `free_router` pool fetch).
///
/// # Errors
/// Returns an error if `kb_root` is unset or not a directory, memory is
/// disabled, the store cannot be opened, or the sweep fails.
pub async fn index_corpus(cfg: &crate::config::Config) -> Result<usize, InboxError> {
    let kb_root = cfg.memory.kb_root.as_deref().ok_or_else(|| {
        InboxError::Memory("[memory].kb_root is not set (org corpus directory)".to_owned())
    })?;
    if !cfg.memory.enabled {
        return Err(InboxError::Memory(
            "memory is disabled — set [memory].enabled = true and configure embeddings".to_owned(),
        ));
    }
    let root = Path::new(kb_root);
    if !root.is_dir() {
        return Err(InboxError::Memory(format!(
            "[memory].kb_root is not a directory: {kb_root}"
        )));
    }
    let db_path = cfg.memory.db_path.as_ref().map_or_else(
        || cfg.general.attachments_dir.join("memory.grafeo"),
        PathBuf::from,
    );
    let store = MemoryStore::open(&cfg.memory, &db_path).await?;
    index_directory(&store, root).await
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
