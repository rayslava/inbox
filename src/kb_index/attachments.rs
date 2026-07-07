//! Attachment indexing: for each org entry, resolve its `[[attachment:]]` /
//! `[[file:]]` links to confined files, extract their text (OCR/PDF, cached in
//! `:KbSource` so a file shared by several notes is extracted once), and index
//! that text as `source="attachment"` chunks bound to the **owning entry's**
//! `:ID:`. Extraction failures degrade to a warning and skip the file.

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use tracing::warn;

use crate::config::Config;
use crate::memory::{MemoryStore, kb};

use super::attach::{parse_attach_links, resolve_attachments};
use super::chunk;
use super::extract::{ShellExtractor, TextExtractor};
use super::{ATTACH_SOURCE, extract_note_id};

/// Extractor + confinement roots for the attachment-indexing pass.
pub struct AttachContext {
    extractor: Box<dyn TextExtractor>,
    roots: Vec<PathBuf>,
}

impl AttachContext {
    /// Build the context from config, or `None` when attachment extraction is
    /// disabled or the backend is unsupported. Roots are the configured extra
    /// roots plus `attachments_dir` and `[memory].kb_root`.
    #[must_use]
    pub fn from_config(cfg: &Config) -> Option<Self> {
        let ec = &cfg.pipeline.attachment_extract;
        if !ec.enabled {
            return None;
        }
        let extractor: Box<dyn TextExtractor> = match ec.backend.as_str() {
            "shell" => Box::new(ShellExtractor::new(
                ec.languages.clone(),
                ec.min_chars,
                ec.timeout_secs,
            )),
            other => {
                warn!("attachment_extract: unsupported backend '{other}', skipping");
                return None;
            }
        };

        let mut roots: Vec<PathBuf> = ec.roots.iter().map(PathBuf::from).collect();
        roots.push(cfg.general.attachments_dir.clone());
        if let Some(kb_root) = &cfg.memory.kb_root {
            roots.push(PathBuf::from(kb_root));
        }
        Some(Self { extractor, roots })
    }
}

/// Index the attachments referenced by one org file, **replacing** this file's
/// entire `source="attachment"` chunk set (keyed by the org-file path) so links
/// that were removed/moved leave no orphan chunks. Returns the number of chunks
/// written.
///
/// Any attachment that fails to extract (unsupported, empty, or a hard OCR
/// error) is simply omitted from the replacement set, so its chunks are removed
/// and re-added on a later clean run — the `:KbSource` cache means an unchanged
/// file is served from cache and never re-fails, so only a *new/changed* file
/// touched during an outage is affected, and it self-heals. This keeps the
/// attachment index consistent with the (already-replaced) body index rather
/// than freezing a stale generation.
pub async fn index_file_attachments(
    store: &MemoryStore,
    ctx: &AttachContext,
    file_path: &Path,
    content: &str,
) -> usize {
    // UTF-8 path identity only (lossless): a lossy key could alias two distinct
    // non-UTF-8 paths and delete the wrong file's chunks.
    let Some(org_path) = file_path.to_str() else {
        warn!(
            "skipping attachments for non-UTF-8 path {}",
            file_path.display()
        );
        return 0;
    };
    let note_dir = file_path.parent().unwrap_or_else(|| Path::new("."));

    // Map each resolved file to the owning entry ids that reference it, so a
    // file is extracted once but indexed under every owning note. Cheap scan
    // first: most notes carry no attachment links.
    let mut by_file: BTreeMap<PathBuf, BTreeSet<String>> = BTreeMap::new();
    if content.contains("[[attachment:") || content.contains("[[file:") {
        let file_note = extract_note_id(content, file_path);
        for c in chunk::chunk_org(content) {
            let links = parse_attach_links(&c.text);
            if links.is_empty() {
                continue;
            }
            let owning = c.note_id.unwrap_or_else(|| file_note.clone());
            for path in resolve_attachments(Some(&owning), &links, note_dir, &ctx.roots) {
                by_file.entry(path).or_default().insert(owning.clone());
            }
        }
    }

    let mut inputs: Vec<(String, String, String)> = Vec::new();
    for (path, owners) in by_file {
        let (Some(text), Some(apath)) = (
            get_or_extract(store, ctx.extractor.as_ref(), &path).await,
            path.to_str(),
        ) else {
            continue;
        };
        for owner in &owners {
            // Fold the attachment path into the chunk-id hash so two files under
            // one owner with identical text get distinct ids.
            for c in chunk::chunk_org(&text) {
                let hash = chunk::stable_hash(&format!("{apath}\u{0}{}", c.text));
                let id = kb::kb_id(ATTACH_SOURCE, owner, chunk::CHUNKER_VERSION, &hash);
                inputs.push((id, owner.clone(), c.text));
            }
        }
    }

    // Whole-org-file replace: delete all prior attachment chunks for this org
    // file and insert the current set atomically (removes orphans of removed
    // links). `path` is the org file; `source` isolates it from body chunks.
    let count = inputs.len();
    match store
        .kb_reindex(ATTACH_SOURCE, org_path, None, inputs)
        .await
    {
        Ok(_) => count,
        Err(e) => {
            warn!("attachment reindex {org_path}: {e}");
            0
        }
    }
}

/// Return the extracted text for `path`, from the `:KbSource` cache when the
/// file bytes and extractor fingerprint are unchanged, else by running the
/// extractor and caching the result. `None` = unsupported / empty / failed /
/// non-UTF-8 path — the caller omits it from the replacement set.
async fn get_or_extract(
    store: &MemoryStore,
    extractor: &dyn TextExtractor,
    path: &Path,
) -> Option<String> {
    let canonical = std::fs::canonicalize(path).ok()?;
    let cpath = canonical.to_str()?.to_owned();

    // Hash the file bytes off the async worker; key on `len:hash` so a byte
    // change that collided the 64-bit hash still differs by length.
    let read_path = canonical.clone();
    let file_hash = tokio::task::spawn_blocking(move || {
        std::fs::read(&read_path).map(|b| format!("{}:{}", b.len(), chunk::stable_hash_bytes(&b)))
    })
    .await
    .ok()?
    .ok()?;
    let fp = extractor.fingerprint().tag();

    if let Some(text) = store.kb_extract_cache_get(&cpath, &file_hash, &fp).await {
        return Some(text);
    }
    let text = match extractor.extract(&canonical).await {
        Ok(Some(t)) if !t.trim().is_empty() => t,
        Ok(_) => return None,
        Err(e) => {
            warn!("extract {}: {e}", canonical.display());
            return None;
        }
    };
    let now = chrono::Utc::now().to_rfc3339();
    if let Err(e) = store
        .kb_extract_cache_put(&cpath, &file_hash, &fp, &text, &now)
        .await
    {
        warn!("cache extraction {}: {e}", canonical.display());
    }
    Some(text)
}

#[cfg(test)]
mod tests;
