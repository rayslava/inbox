//! Whole-note chunking with heading splits: an org note is split at heading
//! boundaries, so each chunk is a heading and its body (plus a leading preamble
//! chunk for content before the first heading). Pure and dependency-light.

/// Chunker version — part of the namespaced id and the embedding fingerprint, so
/// changing the chunking strategy forces a re-index.
pub const CHUNKER_VERSION: &str = "v1";

/// One retrievable chunk of an org note.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrgChunk {
    /// Heading text of this chunk (`""` for the pre-first-heading preamble).
    pub heading: String,
    /// The chunk body (heading line + text up to the next heading).
    pub text: String,
    /// Stable content hash, used in the chunk id and for dedupe.
    pub hash: String,
    /// The owning entry's org id, resolved by inheritance: this heading's `:ID:`,
    /// else the nearest ancestor heading's, else the file-level id. `None` when no
    /// `:ID:` is in scope (caller falls back to the file stem). This is what binds
    /// a chunk — and any `[[attachment:]]` in it — to the correct org-roam node.
    pub note_id: Option<String>,
}

/// Split `content` into chunks at org heading boundaries, tagging each with its
/// owning entry id (see [`OrgChunk::note_id`]). Empty/whitespace-only sections
/// are dropped. A note with no headings yields a single chunk.
#[must_use]
pub fn chunk_org(content: &str) -> Vec<OrgChunk> {
    let mut chunks = Vec::new();
    let mut heading = String::new();
    let mut buf = String::new();
    // Heading path as (level, own-id); index 0 is the file level (level 0).
    let mut stack: Vec<(usize, Option<String>)> = vec![(0, None)];
    let mut in_props = false;
    // The entry's leading metadata region — only here does a `:PROPERTIES:` drawer
    // count as the entry drawer. Closes once body content starts, so a later body
    // drawer can't retag the entry.
    let mut meta_open = true;
    // Inside a `#+begin_…`/`#+end_…` block, drawer-looking lines are literal text.
    let mut in_block = false;

    let flush = |heading: &str, buf: &str, note_id: Option<String>, out: &mut Vec<OrgChunk>| {
        if !buf.trim().is_empty() {
            out.push(OrgChunk {
                heading: heading.to_owned(),
                text: buf.trim_end().to_owned(),
                hash: stable_hash(buf.trim_end()),
                note_id,
            });
        }
    };

    for line in content.lines() {
        let t = line.trim();
        if in_block {
            if strip_ci_prefix(t, "#+end_").is_some() {
                in_block = false;
            }
        } else if strip_ci_prefix(t, "#+begin_").is_some() {
            in_block = true;
            meta_open = false;
        } else if let Some((level, h)) = heading_level_text(line) {
            flush(&heading, &buf, inherited_id(&stack), &mut chunks);
            buf.clear();
            heading = h;
            in_props = false;
            meta_open = true;
            // Pop siblings/deeper, then push this heading (id filled by its drawer).
            while stack.last().is_some_and(|(l, _)| *l >= level) {
                stack.pop();
            }
            stack.push((level, None));
        } else if meta_open {
            if t.eq_ignore_ascii_case(":PROPERTIES:") {
                in_props = true;
            } else if in_props {
                if t.eq_ignore_ascii_case(":END:") {
                    in_props = false;
                    meta_open = false;
                } else if let Some(rest) = strip_ci_prefix(t, ":ID:")
                    && !rest.trim().is_empty()
                    && let Some(top) = stack.last_mut()
                {
                    top.1 = Some(rest.trim().to_owned());
                }
            } else if !t.is_empty() && !is_planning(t) {
                // Substantive body before any drawer → no entry drawer here.
                meta_open = false;
            }
        }
        buf.push_str(line);
        buf.push('\n');
    }
    flush(&heading, &buf, inherited_id(&stack), &mut chunks);
    chunks
}

/// Nearest in-scope id along the heading path (this heading, else an ancestor,
/// else the file level).
fn inherited_id(stack: &[(usize, Option<String>)]) -> Option<String> {
    stack.iter().rev().find_map(|(_, id)| id.clone())
}

/// Case-insensitive **ASCII** prefix strip returning the remainder. Compares by
/// bytes so a non-ASCII line never slices on a non-char-boundary (panic-safe);
/// the returned offset is a boundary because the matched prefix is all ASCII.
fn strip_ci_prefix<'a>(s: &'a str, prefix: &str) -> Option<&'a str> {
    let (p, b) = (prefix.as_bytes(), s.as_bytes());
    (b.len() >= p.len() && b[..p.len()].eq_ignore_ascii_case(p)).then(|| &s[p.len()..])
}

/// Org planning lines that may precede a property drawer without ending the
/// entry's metadata region.
fn is_planning(t: &str) -> bool {
    ["SCHEDULED:", "DEADLINE:", "CLOSED:"]
        .iter()
        .any(|k| strip_ci_prefix(t, k).is_some())
}

/// If `line` is an org heading (`*`+ then a space), return `(level, trimmed text)`.
fn heading_level_text(line: &str) -> Option<(usize, String)> {
    let stars = line.chars().take_while(|&c| c == '*').count();
    if stars > 0 && line[stars..].starts_with(' ') {
        Some((stars, line[stars..].trim().to_owned()))
    } else {
        None
    }
}

/// Deterministic FNV-1a 64-bit content hash (hex). Stable across runs so chunk
/// ids are content-addressed; not cryptographic.
#[must_use]
pub fn stable_hash(s: &str) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{h:016x}")
}

#[cfg(test)]
mod tests {
    use super::{chunk_org, heading_level_text, stable_hash};

    #[test]
    fn splits_at_headings_with_preamble() {
        let org = "intro line\n* First\nbody one\n* Second\nbody two\n";
        let chunks = chunk_org(org);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].heading, "");
        assert!(chunks[0].text.contains("intro line"));
        assert_eq!(chunks[1].heading, "First");
        assert!(chunks[1].text.contains("body one"));
        assert_eq!(chunks[2].heading, "Second");
    }

    #[test]
    fn no_headings_yields_single_chunk() {
        let chunks = chunk_org("just a plain note\nwith two lines\n");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].heading, "");
    }

    #[test]
    fn blank_sections_dropped() {
        let chunks = chunk_org("\n\n* Only\ncontent\n");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].heading, "Only");
    }

    #[test]
    fn heading_detection() {
        assert_eq!(heading_level_text("* Top"), Some((1, "Top".to_owned())));
        assert_eq!(
            heading_level_text("*** Deep heading"),
            Some((3, "Deep heading".to_owned()))
        );
        assert_eq!(heading_level_text("*bold* not a heading"), None);
        assert_eq!(heading_level_text("plain"), None);
    }

    #[test]
    fn hash_is_stable_and_content_sensitive() {
        assert_eq!(stable_hash("abc"), stable_hash("abc"));
        assert_ne!(stable_hash("abc"), stable_hash("abd"));
        assert_eq!(stable_hash("abc").len(), 16);
    }

    #[test]
    fn file_level_id_applies_to_preamble() {
        let org = ":PROPERTIES:\n:ID: file-1\n:END:\n#+title: T\nbody\n";
        let chunks = chunk_org(org);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].note_id.as_deref(), Some("file-1"));
    }

    #[test]
    fn each_subtree_gets_its_own_id() {
        let org = concat!(
            ":PROPERTIES:\n:ID: file-1\n:END:\n#+title: T\n",
            "* Alpha\n:PROPERTIES:\n:ID: alpha-id\n:END:\nalpha body\n",
            "* Beta\n:PROPERTIES:\n:ID: beta-id\n:END:\nbeta body\n",
        );
        let chunks = chunk_org(org);
        let by_heading = |h: &str| {
            chunks
                .iter()
                .find(|c| c.heading == h)
                .and_then(|c| c.note_id.clone())
        };
        assert_eq!(by_heading("Alpha").as_deref(), Some("alpha-id"));
        assert_eq!(by_heading("Beta").as_deref(), Some("beta-id"));
        // The preamble keeps the file id.
        assert_eq!(
            chunks
                .iter()
                .find(|c| c.heading.is_empty())
                .unwrap()
                .note_id
                .as_deref(),
            Some("file-1")
        );
    }

    #[test]
    fn subtree_without_id_inherits_ancestor() {
        let org = concat!(
            "* Parent\n:PROPERTIES:\n:ID: parent-id\n:END:\nparent body\n",
            "** Child\nchild body with an attachment link\n",
            "* Sibling\nsibling body\n",
        );
        let chunks = chunk_org(org);
        let id = |h: &str| {
            chunks
                .iter()
                .find(|c| c.heading == h)
                .and_then(|c| c.note_id.clone())
        };
        assert_eq!(
            id("Child").as_deref(),
            Some("parent-id"),
            "inherits ancestor id"
        );
        // A sibling at the parent's level does NOT inherit the parent's id.
        assert_eq!(id("Sibling"), None);
    }

    #[test]
    fn body_and_src_block_drawers_do_not_retag() {
        let org = concat!(
            "* Real\n:PROPERTIES:\n:ID: real-id\n:END:\nbody line\n",
            ":PROPERTIES:\n:ID: bogus-body\n:END:\n",
            "#+begin_src\n:PROPERTIES:\n:ID: bogus-src\n:END:\n#+end_src\n",
        );
        let chunks = chunk_org(org);
        let real = chunks.iter().find(|c| c.heading == "Real").unwrap();
        assert_eq!(real.note_id.as_deref(), Some("real-id"));
    }

    #[test]
    fn planning_line_before_drawer_is_tolerated() {
        let org = "* T\nSCHEDULED: <2024-01-01>\n:PROPERTIES:\n:ID: t-id\n:END:\nbody\n";
        let chunks = chunk_org(org);
        assert_eq!(chunks[0].note_id.as_deref(), Some("t-id"));
    }

    #[test]
    fn non_ascii_line_in_drawer_does_not_panic() {
        let org = "* T\n:PROPERTIES:\n:ID: ok-id\nПривет мир\n:END:\nbody\n";
        let chunks = chunk_org(org);
        assert_eq!(chunks[0].note_id.as_deref(), Some("ok-id"));
    }
}
