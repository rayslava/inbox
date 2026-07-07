//! Resolve org attachment links (`[[attachment:name]]`, `[[file:path]]`) to
//! physical files, **confined to configured roots** (deny-by-default: `..` and
//! symlink escapes are rejected because every candidate is `canonicalize`d and
//! required to stay under an allowlisted root). org-attach maps an entry id to
//! `id[0:2]/id[2:]` under each root; ancestor `data/` dirs are a fallback.

use std::path::{Component, Path, PathBuf};

/// How far up the note's directory tree to look for inherited `data/` dirs.
const MAX_ANCESTORS: usize = 6;

/// An attachment reference parsed from an entry's text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttachLink {
    /// `[[attachment:name]]` — resolved via org-attach id→dir under a root.
    Named(String),
    /// `[[file:path]]` — a direct path (absolute, or relative to the note dir).
    File(String),
}

/// Parse `[[attachment:…]]` and `[[file:…]]` links from `text`. The target ends
/// at the link's `]]` or its `][` description separator.
#[must_use]
pub fn parse_attach_links(text: &str) -> Vec<AttachLink> {
    let mut out = Vec::new();
    collect_links(text, "[[attachment:", &AttachLink::Named, &mut out);
    collect_links(text, "[[file:", &AttachLink::File, &mut out);
    out
}

fn collect_links(
    text: &str,
    prefix: &str,
    make: &dyn Fn(String) -> AttachLink,
    out: &mut Vec<AttachLink>,
) {
    let mut rest = text;
    while let Some(i) = rest.find(prefix) {
        let after = &rest[i + prefix.len()..];
        // The target ends at whichever comes first: the link close `]]` or the
        // description separator `][`.
        let end = [after.find("]]"), after.find("][")]
            .into_iter()
            .flatten()
            .min()
            .unwrap_or(after.len());
        let target = after[..end].trim();
        if !target.is_empty() {
            out.push(make(target.to_owned()));
        }
        rest = &after[end..];
    }
}

/// True only for a single `Normal` path component (a bare filename) — rejects
/// absolute paths, `..`, `.`, and any embedded separator.
fn is_plain_filename(name: &str) -> bool {
    let mut comps = Path::new(name).components();
    matches!(comps.next(), Some(Component::Normal(_))) && comps.next().is_none()
}

/// org-attach `id[0:2]/id[2:]` split; `None` for ids under 3 bytes or whose
/// 2-byte prefix isn't a char boundary (guards against a panic on a non-ASCII
/// `:ID:` — real org-roam ids are ASCII UUIDs).
fn id_split(id: &str) -> Option<(String, String)> {
    (id.len() > 2 && id.is_char_boundary(2)).then(|| (id[0..2].to_owned(), id[2..].to_owned()))
}

/// Candidate attachment base directories: `root/id[0:2]/id[2:]` for each root,
/// plus `data/id[0:2]/id[2:]` under the note dir and a bounded set of ancestors.
fn candidate_bases(owning_id: Option<&str>, note_dir: &Path, roots: &[PathBuf]) -> Vec<PathBuf> {
    let Some((a, b)) = owning_id.and_then(id_split) else {
        return Vec::new();
    };
    let mut bases: Vec<PathBuf> = roots.iter().map(|r| r.join(&a).join(&b)).collect();
    let mut dir = Some(note_dir);
    for _ in 0..MAX_ANCESTORS {
        let Some(cur) = dir else { break };
        bases.push(cur.join("data").join(&a).join(&b));
        dir = cur.parent();
    }
    bases
}

/// Canonicalize `p` and return it only if it is a regular file confined under
/// one of `canon_roots`. `canonicalize` resolves `..`/symlinks and requires
/// existence, so this covers "exists + is a file + confined + no traversal /
/// symlink escape". NOTE: a path-time check — a caller shipping bytes off-box
/// (the HTTP OCR path) must re-confine at open time (TOCTOU).
fn confine(p: &Path, canon_roots: &[PathBuf]) -> Option<PathBuf> {
    let c = std::fs::canonicalize(p).ok()?;
    let is_file = std::fs::metadata(&c).is_ok_and(|m| m.is_file());
    (is_file && canon_roots.iter().any(|r| c.starts_with(r))).then_some(c)
}

fn canon_roots(roots: &[PathBuf]) -> Vec<PathBuf> {
    roots
        .iter()
        .filter_map(|r| std::fs::canonicalize(r).ok())
        .collect()
}

/// List every regular file in the owning entry's org-attach directories
/// (`root/id[0:2]/id[2:]` under each root, plus ancestor `data/` dirs),
/// **root-confined**. This is how org-attach attaches files via the `:ATTACH:`
/// tag — present in the id-dir with **no inline link** — so it complements
/// [`resolve_attachments`]. Returns canonical, deduped paths.
#[must_use]
pub fn list_attachment_dir_files(
    owning_id: &str,
    note_dir: &Path,
    roots: &[PathBuf],
) -> Vec<PathBuf> {
    let canon_roots = canon_roots(roots);
    let mut out: Vec<PathBuf> = Vec::new();
    for base in candidate_bases(Some(owning_id), note_dir, roots) {
        let Ok(entries) = std::fs::read_dir(&base) else {
            continue;
        };
        for entry in entries.flatten() {
            if let Some(c) = confine(&entry.path(), &canon_roots)
                && !out.contains(&c)
            {
                out.push(c);
            }
        }
    }
    out
}

/// Resolve `links` for an entry whose owning org id is `owning_id` to existing,
/// **root-confined** files. `roots` is the allowlist (org-attach id-dir,
/// `attachments_dir`, KB root); `note_dir` anchors relative `[[file:]]` links and
/// the `data/` fallback. Returned paths are canonical and deduped.
#[must_use]
pub fn resolve_attachments(
    owning_id: Option<&str>,
    links: &[AttachLink],
    note_dir: &Path,
    roots: &[PathBuf],
) -> Vec<PathBuf> {
    let canon_roots = canon_roots(roots);
    let confined = |p: &Path| -> Option<PathBuf> { confine(p, &canon_roots) };

    let bases = candidate_bases(owning_id, note_dir, roots);
    let mut out: Vec<PathBuf> = Vec::new();
    for link in links {
        let candidates: Vec<PathBuf> = match link {
            // An attachment name must be a bare filename, so it can never point
            // outside its owning org-attach id directory (`join` would otherwise
            // honour an absolute or `../` name and reach another id/root file).
            AttachLink::Named(name) if is_plain_filename(name) => {
                bases.iter().map(|b| b.join(name)).collect()
            }
            AttachLink::Named(_) => Vec::new(),
            AttachLink::File(p) => {
                let pp = Path::new(p);
                let full = if pp.is_absolute() {
                    pp.to_path_buf()
                } else {
                    note_dir.join(pp)
                };
                vec![full]
            }
        };
        for cand in candidates {
            if let Some(c) = confined(&cand)
                && !out.contains(&c)
            {
                out.push(c);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests;
