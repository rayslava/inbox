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
}

/// Split `content` into chunks at org heading boundaries. Empty/whitespace-only
/// sections are dropped. A note with no headings yields a single chunk.
#[must_use]
pub fn chunk_org(content: &str) -> Vec<OrgChunk> {
    let mut chunks = Vec::new();
    let mut heading = String::new();
    let mut buf = String::new();

    let flush = |heading: &str, buf: &str, out: &mut Vec<OrgChunk>| {
        if !buf.trim().is_empty() {
            out.push(OrgChunk {
                heading: heading.to_owned(),
                text: buf.trim_end().to_owned(),
                hash: stable_hash(buf.trim_end()),
            });
        }
    };

    for line in content.lines() {
        if let Some(h) = heading_text(line) {
            flush(&heading, &buf, &mut chunks);
            buf.clear();
            heading = h;
        }
        buf.push_str(line);
        buf.push('\n');
    }
    flush(&heading, &buf, &mut chunks);
    chunks
}

/// If `line` is an org heading (`*`+ then a space), return its trimmed text.
fn heading_text(line: &str) -> Option<String> {
    let stars = line.chars().take_while(|&c| c == '*').count();
    if stars > 0 && line[stars..].starts_with(' ') {
        Some(line[stars..].trim().to_owned())
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
    use super::{chunk_org, heading_text, stable_hash};

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
        assert_eq!(heading_text("* Top"), Some("Top".to_owned()));
        assert_eq!(
            heading_text("*** Deep heading"),
            Some("Deep heading".to_owned())
        );
        assert_eq!(heading_text("*bold* not a heading"), None);
        assert_eq!(heading_text("plain"), None);
    }

    #[test]
    fn hash_is_stable_and_content_sensitive() {
        assert_eq!(stable_hash("abc"), stable_hash("abc"));
        assert_ne!(stable_hash("abc"), stable_hash("abd"));
        assert_eq!(stable_hash("abc").len(), 16);
    }
}
