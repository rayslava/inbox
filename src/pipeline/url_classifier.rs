use mime::Mime;
use tracing::{debug, instrument};
use url::Url;

use super::url_fetcher::UrlFetcher;

#[derive(Debug)]
pub enum UrlKind {
    /// HTML page — scrape for text content.
    Page,
    /// Binary/media file — download as attachment.
    File { mime: Mime },
    /// HEAD failed or ambiguous — try scraping as fallback.
    Unknown,
}

/// Classify a URL by issuing a HEAD request and inspecting Content-Type.
/// File extensions are used as a secondary signal.
#[instrument(skip(fetcher), fields(url = %url))]
pub async fn classify_url(url: &Url, fetcher: &UrlFetcher) -> UrlKind {
    // First, check extension before making a network request.
    if let Some(kind) = classify_by_extension(url) {
        debug!(%url, kind = ?kind, "URL classified by path extension");
        return kind;
    }

    // Issue a HEAD request.
    if let Some(content_type) = fetcher.head(url).await {
        let kind = classify_by_content_type(&content_type);
        debug!(%url, %content_type, kind = ?kind, "URL classified via HEAD Content-Type");
        kind
    } else {
        debug!(%url, "HEAD request failed, treating URL as Unknown");
        UrlKind::Unknown
    }
}

fn classify_by_extension(url: &Url) -> Option<UrlKind> {
    let path = url.path();
    let ext = path.rsplit('.').next()?;
    match ext.to_ascii_lowercase().as_str() {
        "pdf" | "doc" | "docx" | "xls" | "xlsx" | "ppt" | "pptx" | "zip" | "tar" | "gz" | "bz2"
        | "7z" | "rar" | "mp3" | "mp4" | "mkv" | "avi" | "mov" | "webm" | "jpg" | "jpeg"
        | "png" | "gif" | "webp" | "svg" | "ogg" | "flac" | "wav" | "opus" => {
            let mime_str = mime_guess::from_ext(ext).first_or_octet_stream();
            Some(UrlKind::File { mime: mime_str })
        }
        "html" | "htm" | "php" | "asp" | "aspx" => Some(UrlKind::Page),
        _ => None,
    }
}

fn classify_by_content_type(ct: &str) -> UrlKind {
    // Parse just the type/subtype, ignore parameters.
    let base = ct.split(';').next().unwrap_or(ct).trim();
    match base.parse::<Mime>() {
        Ok(mime) => {
            let type_ = mime.type_().as_str();
            let subtype = mime.subtype().as_str();
            if type_ == "text" && (subtype == "html" || subtype == "plain") {
                UrlKind::Page
            } else {
                UrlKind::File { mime }
            }
        }
        Err(_) => UrlKind::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pdf_by_extension() {
        let url = Url::parse("https://example.com/report.pdf").unwrap();
        assert!(matches!(
            classify_by_extension(&url),
            Some(UrlKind::File { .. })
        ));
    }

    #[test]
    fn html_by_extension() {
        let url = Url::parse("https://example.com/index.html").unwrap();
        assert!(matches!(classify_by_extension(&url), Some(UrlKind::Page)));
    }

    #[test]
    fn unknown_extension() {
        let url = Url::parse("https://example.com/article").unwrap();
        assert!(classify_by_extension(&url).is_none());
    }

    #[test]
    fn content_type_html() {
        assert!(matches!(
            classify_by_content_type("text/html; charset=utf-8"),
            UrlKind::Page
        ));
    }

    #[test]
    fn content_type_pdf() {
        assert!(matches!(
            classify_by_content_type("application/pdf"),
            UrlKind::File { .. }
        ));
    }

    #[test]
    fn content_type_plain_text_treated_as_page() {
        assert!(matches!(
            classify_by_content_type("text/plain"),
            UrlKind::Page
        ));
    }

    #[test]
    fn content_type_bare_without_parameters() {
        assert!(matches!(
            classify_by_content_type("application/json"),
            UrlKind::File { .. }
        ));
    }

    #[test]
    fn content_type_malformed_returns_unknown() {
        assert!(matches!(
            classify_by_content_type("not-a-mime-type"),
            UrlKind::Unknown
        ));
    }

    #[test]
    fn extension_case_insensitive() {
        let url = Url::parse("https://example.com/Photo.JPG").unwrap();
        assert!(matches!(
            classify_by_extension(&url),
            Some(UrlKind::File { .. })
        ));
    }

    #[test]
    fn extension_zip_archive() {
        let url = Url::parse("https://example.com/bundle.zip").unwrap();
        assert!(matches!(
            classify_by_extension(&url),
            Some(UrlKind::File { .. })
        ));
    }

    #[test]
    fn extension_php_classified_as_page() {
        let url = Url::parse("https://example.com/index.php").unwrap();
        assert!(matches!(classify_by_extension(&url), Some(UrlKind::Page)));
    }

    #[test]
    fn extension_empty_path_returns_none() {
        let url = Url::parse("https://example.com/").unwrap();
        assert!(classify_by_extension(&url).is_none());
    }
}
