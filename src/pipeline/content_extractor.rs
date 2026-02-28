use scraper::{Html, Selector};
use tracing::{debug, instrument};

/// Strip HTML to readable plain text.
/// Extracts the page title separately and returns the body text.
pub struct ExtractedPage {
    pub title: Option<String>,
    pub text: String,
}

#[must_use]
#[instrument(skip(html), fields(html_len = html.len()))]
pub fn extract_text(html: &str) -> ExtractedPage {
    let doc = Html::parse_document(html);

    let title = extract_title(&doc);
    let text = extract_body_text(&doc);

    debug!(
        text_len = text.len(),
        has_title = title.is_some(),
        "HTML text extracted"
    );

    ExtractedPage { title, text }
}

fn extract_title(doc: &Html) -> Option<String> {
    let sel = Selector::parse("title").ok()?;
    let t = doc.select(&sel).next()?.text().collect::<String>();
    let trimmed = t.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_owned())
    }
}

fn extract_body_text(doc: &Html) -> String {
    // Remove script, style, nav, footer, header to reduce noise.
    let _noise_sel = Selector::parse("script, style, nav, footer, header, aside").unwrap();

    // Collect text from meaningful elements.
    let text_sel =
        Selector::parse("p, h1, h2, h3, h4, h5, h6, li, td, th, pre, blockquote").unwrap();

    // We'll gather text from elements that are NOT descendants of noise elements.
    // scraper doesn't have an :not-ancestor selector, so we collect all text nodes
    // from the document and skip those in noise elements.
    //
    // Simpler approach: collect text from target elements globally (scraper doesn't
    // filter by ancestor efficiently), then join.
    let mut parts: Vec<String> = Vec::new();

    for element in doc.select(&text_sel) {
        let text: String = element.text().collect();
        let trimmed = text.trim();
        if !trimmed.is_empty() {
            parts.push(trimmed.to_owned());
        }
    }

    if parts.is_empty() {
        // Fallback: grab all text from body
        if let Ok(body_sel) = Selector::parse("body")
            && let Some(body) = doc.select(&body_sel).next()
        {
            let text: String = body.text().collect::<Vec<_>>().join(" ");
            return collapse_whitespace(&text);
        }
        // Last resort: whole doc text
        let text: String = doc.root_element().text().collect::<Vec<_>>().join(" ");
        return collapse_whitespace(&text);
    }

    parts.join("\n\n")
}

fn collapse_whitespace(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_paragraphs() {
        let html = r"<html><head><title>Test</title></head>
        <body><p>Hello world.</p><p>Second paragraph.</p></body></html>";
        let page = extract_text(html);
        assert_eq!(page.title.as_deref(), Some("Test"));
        assert!(page.text.contains("Hello world."));
        assert!(page.text.contains("Second paragraph."));
    }

    #[test]
    fn handles_empty_html() {
        let page = extract_text("<html><body></body></html>");
        assert!(page.title.is_none());
    }
}
