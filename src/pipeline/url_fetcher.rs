use std::path::{Path, PathBuf};
use std::time::Duration;

use anodized::spec;
use reqwest::Client;
use tracing::{debug, info, instrument, warn};
use url::Url;
use uuid::Uuid;

use crate::config::UrlFetchConfig;
use crate::message::{Attachment, MediaKind};

use super::content_extractor::{self, ExtractedPage};
use crate::url_content::UrlContent;

pub struct UrlFetcher {
    client: Client,
    cfg: UrlFetchConfig,
}

impl UrlFetcher {
    /// Create a `UrlFetcher` from config.
    ///
    /// # Panics
    /// Panics if the TLS backend cannot be initialised (extremely unlikely in practice).
    #[must_use]
    pub fn new(cfg: &UrlFetchConfig) -> Self {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            reqwest::header::ACCEPT,
            "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
                .parse()
                .expect("static header value"),
        );
        headers.insert(
            reqwest::header::ACCEPT_LANGUAGE,
            "en-US,en;q=0.5".parse().expect("static header value"),
        );

        let client = crate::tls::client_builder()
            .user_agent(&cfg.user_agent)
            .timeout(Duration::from_secs(cfg.timeout_secs))
            .redirect(reqwest::redirect::Policy::limited(
                cfg.max_redirects as usize,
            ))
            .default_headers(headers)
            .build()
            .expect("Failed to build HTTP client");

        Self {
            client,
            cfg: cfg.clone(),
        }
    }

    /// Issue a HEAD request and return the Content-Type header value, if any.
    pub async fn head(&self, url: &Url) -> Option<String> {
        self.client
            .head(url.as_str())
            .send()
            .await
            .ok()
            .and_then(|r| {
                r.headers()
                    .get(reqwest::header::CONTENT_TYPE)
                    .and_then(|v| v.to_str().ok())
                    .map(std::borrow::ToOwned::to_owned)
            })
    }

    /// Fetch a page and extract readable text.
    #[instrument(skip(self), fields(url = %url))]
    pub async fn fetch_page(&self, url: &Url) -> Option<UrlContent> {
        let resp = self
            .client
            .get(url.as_str())
            .send()
            .await
            .map_err(|e| {
                warn!(?e, %url, "Page fetch failed");
                metrics::counter!(crate::telemetry::URL_FETCHES, "status" => "failure")
                    .increment(1);
                e
            })
            .ok()?;

        if !resp.status().is_success() {
            warn!(status = %resp.status(), %url, "Page fetch non-200");
            metrics::counter!(crate::telemetry::URL_FETCHES, "status" => "failure").increment(1);
            return None;
        }

        let bytes = resp
            .bytes()
            .await
            .map_err(|e| {
                warn!(?e, %url, "Page body read failed");
                e
            })
            .ok()?;

        // Respect max_body_bytes
        let body = if bytes.len() > self.cfg.max_body_bytes {
            &bytes[..self.cfg.max_body_bytes]
        } else {
            &bytes
        };

        let html = String::from_utf8_lossy(body).into_owned();
        let ExtractedPage { title, text } = content_extractor::extract_text(&html);

        debug!(
            %url,
            text_len = text.len(),
            has_title = title.is_some(),
            "Page content extracted"
        );
        metrics::counter!(crate::telemetry::URL_FETCHES, "status" => "success").increment(1);

        Some(UrlContent {
            url: url.to_string(),
            text,
            page_title: title,
        })
    }

    /// Download a file URL as an attachment.
    /// Saves to `{attachments_dir}/{id[0..2]}/{id[2..]}/{filename}`.
    #[instrument(skip(self, attachments_dir), fields(url = %url, id = %msg_id))]
    #[spec(requires: !msg_id.is_nil())]
    pub async fn download_file(
        &self,
        url: &Url,
        msg_id: Uuid,
        attachments_dir: &Path,
    ) -> Option<Attachment> {
        let resp = self
            .client
            .get(url.as_str())
            .send()
            .await
            .map_err(|e| {
                warn!(?e, %url, "File download failed");
                e
            })
            .ok()?;

        if !resp.status().is_success() {
            warn!(status = %resp.status(), %url, "File download non-200");
            return None;
        }

        let mime_type = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .map(|s| s.split(';').next().unwrap_or(s).trim().to_owned());

        let filename = filename_from_url(url);
        let save_path = attachment_save_path(attachments_dir, msg_id, &filename);

        if let Some(parent) = save_path.parent()
            && let Err(e) = tokio::fs::create_dir_all(parent).await
        {
            warn!(?e, ?parent, "Failed to create attachment dir");
            return None;
        }

        let bytes = resp
            .bytes()
            .await
            .map_err(|e| {
                warn!(?e, %url, "File body read failed");
                e
            })
            .ok()?;

        tokio::fs::write(&save_path, &bytes)
            .await
            .map_err(|e| {
                warn!(?e, ?save_path, "Failed to write attachment");
                e
            })
            .ok()?;

        let media_kind = mime_type
            .as_deref()
            .map_or(MediaKind::Document, MediaKind::from_mime);

        info!(
            %url,
            filename = filename_from_url(url),
            bytes = bytes.len(),
            "File attachment downloaded"
        );
        metrics::counter!(crate::telemetry::URL_FETCHES, "status" => "success").increment(1);

        Some(Attachment {
            original_name: filename,
            saved_path: save_path,
            mime_type,
            media_kind,
        })
    }
}

/// Derive a filename from the URL path, falling back to "download".
fn filename_from_url(url: &Url) -> String {
    url.path_segments()
        .and_then(|mut seg| seg.next_back())
        .filter(|s| !s.is_empty())
        .map_or_else(|| "download".into(), sanitize_filename)
}

fn sanitize_filename(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_alphanumeric() || matches!(c, '.' | '-' | '_') {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// org-attach-id-dir layout: `{base}/{id[0..2]}/{id[2..]}/{filename}`
#[must_use]
#[spec(requires: !filename.is_empty())]
pub fn attachment_save_path(base: &Path, id: Uuid, filename: &str) -> PathBuf {
    let id_str = id.to_string().replace('-', "");
    let dir1 = &id_str[..2];
    let dir2 = &id_str[2..];
    base.join(dir1).join(dir2).join(filename)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filename_from_path() {
        let url = Url::parse("https://example.com/files/report.pdf").unwrap();
        assert_eq!(filename_from_url(&url), "report.pdf");
    }

    #[test]
    fn filename_fallback() {
        let url = Url::parse("https://example.com/").unwrap();
        assert_eq!(filename_from_url(&url), "download");
    }

    #[test]
    fn sanitize_special_chars() {
        assert_eq!(sanitize_filename("my file (1).pdf"), "my_file__1_.pdf");
    }

    #[test]
    fn attachment_path_layout() {
        let id = Uuid::parse_str("550e8400-e29b-41d4-a716-446655440000").unwrap();
        let base = Path::new("/data/attachments");
        let path = attachment_save_path(base, id, "report.pdf");
        // id without hyphens = 550e8400e29b41d4a716446655440000
        // first 2 chars: "55", rest: "0e8400e29b41d4a716446655440000"
        assert!(path.to_str().unwrap().contains("/55/"));
        assert!(path.ends_with("report.pdf"));
    }
}
