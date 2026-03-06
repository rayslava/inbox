use std::sync::OnceLock;

use regex::Regex;

static HASHTAG_RE: OnceLock<Regex> = OnceLock::new();

fn hashtag_re() -> &'static Regex {
    HASHTAG_RE.get_or_init(|| {
        // Match #tag preceded by start-of-string or whitespace.
        // Group 1: the leading whitespace (kept). Group 2: the tag word (removed).
        Regex::new(r"(^|\s)#([a-zA-Z][a-zA-Z0-9_-]*)").expect("valid regex")
    })
}

/// Extract `#hashtag` tokens from `text`.
///
/// Returns `(cleaned_text, tags)` where `cleaned_text` has the `#tag` tokens
/// removed and excess whitespace collapsed, and `tags` is a deduplicated,
/// lowercased list preserving first-occurrence order.
///
/// Only tokens that appear after start-of-string or whitespace are matched, so
/// URL fragments like `https://example.com#anchor` are left intact.
#[must_use]
pub fn extract_user_tags(text: &str) -> (String, Vec<String>) {
    let re = hashtag_re();
    let mut tags: Vec<String> = Vec::new();

    let cleaned = re.replace_all(text, |caps: &regex::Captures| {
        let tag = caps[2].to_lowercase();
        if !tags.contains(&tag) {
            tags.push(tag);
        }
        // Keep the leading whitespace/newline but drop the `#word`.
        caps[1].to_owned()
    });

    // Normalize whitespace without destroying newlines.
    let cleaned = cleaned
        .lines()
        .map(|line| line.split_whitespace().collect::<Vec<_>>().join(" "))
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_owned();

    (cleaned, tags)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_tags_unchanged() {
        let (text, tags) = extract_user_tags("hello world https://example.com#anchor");
        assert_eq!(text, "hello world https://example.com#anchor");
        assert!(tags.is_empty());
    }

    #[test]
    fn single_tag_extracted() {
        let (text, tags) = extract_user_tags("check this out #rust");
        assert_eq!(text, "check this out");
        assert_eq!(tags, ["rust"]);
    }

    #[test]
    fn multiple_tags_extracted() {
        let (text, tags) = extract_user_tags("some idea #rust #async #tokio");
        assert_eq!(text, "some idea");
        assert_eq!(tags, ["rust", "async", "tokio"]);
    }

    #[test]
    fn tag_at_start_of_string() {
        let (text, tags) = extract_user_tags("#rust is great");
        assert_eq!(text, "is great");
        assert_eq!(tags, ["rust"]);
    }

    #[test]
    fn duplicate_tags_deduplicated() {
        let (text, tags) = extract_user_tags("topic #rust and more #rust");
        assert_eq!(text, "topic and more");
        assert_eq!(tags, ["rust"]);
    }

    #[test]
    fn tags_lowercased() {
        let (text, tags) = extract_user_tags("hello #Rust #ASYNC");
        assert_eq!(text, "hello");
        assert_eq!(tags, ["rust", "async"]);
    }

    #[test]
    fn url_fragment_not_matched() {
        let (text, tags) = extract_user_tags("see https://example.com#section for details");
        assert_eq!(text, "see https://example.com#section for details");
        assert!(tags.is_empty());
    }

    #[test]
    fn numeric_only_hashtag_not_matched() {
        let (text, tags) = extract_user_tags("issue #123 is open");
        assert_eq!(text, "issue #123 is open");
        assert!(tags.is_empty());
    }
}
