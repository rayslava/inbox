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

/// Extract all HTTP(S) URLs from a text string as raw strings.
/// Use this when you need URL strings without URL validation (e.g. for source link deduplication).
/// Duplicate URLs are deduplicated while preserving order.
#[must_use]
pub fn extract_http_url_strings(text: &str) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    url_re()
        .find_iter(text)
        .filter_map(|m| {
            let raw = m
                .as_str()
                .trim_end_matches(['.', ',', ')', '>', ';', '\'', '"'])
                .to_owned();
            if seen.insert(raw.clone()) {
                Some(raw)
            } else {
                None
            }
        })
        .collect()
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

    #[test]
    fn extract_http_url_strings_deduplicates_and_preserves_order() {
        let urls = extract_http_url_strings(
            "a https://b.example/x then https://a.example/y then https://b.example/x again",
        );
        assert_eq!(
            urls,
            vec![
                "https://b.example/x".to_owned(),
                "https://a.example/y".to_owned()
            ]
        );
    }

    #[test]
    fn extract_http_url_strings_strips_trailing_punctuation() {
        let urls = extract_http_url_strings("See https://example.com/a), and https://b.example/.");
        assert_eq!(
            urls,
            vec![
                "https://example.com/a".to_owned(),
                "https://b.example/".to_owned()
            ]
        );
    }

    #[test]
    fn extract_http_url_strings_keeps_http_tokens_without_url_validation() {
        let urls = extract_http_url_strings("raw token https://example.com/space here");
        assert_eq!(urls, vec!["https://example.com/space".to_owned()]);
    }
}
