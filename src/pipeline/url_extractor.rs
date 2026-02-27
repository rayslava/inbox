use regex::Regex;
use std::sync::OnceLock;
use tracing::{debug, instrument};
use url::Url;

static URL_RE: OnceLock<Regex> = OnceLock::new();

fn url_re() -> &'static Regex {
    URL_RE.get_or_init(|| {
        // Match http(s):// URLs, stopping at whitespace or common sentence punctuation.
        Regex::new(r#"https?://[^\s<>"'\]\[)]+"#).unwrap()
    })
}

/// Extract all HTTP(S) URLs from a text string.
/// Duplicate URLs (by string) are deduplicated while preserving order.
#[must_use]
#[instrument(skip(text), fields(text_len = text.len()))]
pub fn extract_urls(text: &str) -> Vec<Url> {
    let mut seen = std::collections::HashSet::new();
    let urls: Vec<Url> = url_re()
        .find_iter(text)
        .filter_map(|m| {
            // Strip trailing punctuation that was likely not part of the URL.
            let raw = m
                .as_str()
                .trim_end_matches(['.', ',', ')', '>', ';', '\'', '"']);
            if seen.insert(raw.to_owned()) {
                Url::parse(raw).ok()
            } else {
                None
            }
        })
        .collect();
    debug!(count = urls.len(), "URLs extracted from message text");
    urls
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_plain_url() {
        let urls = extract_urls("check out https://example.com/report.pdf for details");
        assert_eq!(urls.len(), 1);
        assert_eq!(urls[0].as_str(), "https://example.com/report.pdf");
    }

    #[test]
    fn strips_trailing_period() {
        let urls = extract_urls("See https://example.com.");
        assert_eq!(urls.len(), 1);
        assert_eq!(urls[0].as_str(), "https://example.com/");
    }

    #[test]
    fn deduplicates() {
        let urls = extract_urls("https://a.com and https://a.com again");
        assert_eq!(urls.len(), 1);
    }

    #[test]
    fn empty_text_returns_empty() {
        assert!(extract_urls("no urls here").is_empty());
    }
}
