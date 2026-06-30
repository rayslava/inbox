use crate::UrlContent;
use async_trait::async_trait;
use url::Url;

/// Fetches and extracts readable content from web URLs.
///
/// Implemented in the `inbox` binary (reqwest dual-stack + Nitter rewriting);
/// `core` exposes only the async surface over [`UrlContent`]. Methods return
/// `Option` — a failed fetch yields `None` (the adapter logs the cause).
#[async_trait]
pub trait UrlFetcher: Send + Sync {
    /// Fetch `url` and extract its readable text, or `None` on failure.
    async fn fetch_page(&self, url: &Url) -> Option<UrlContent>;

    /// Return the `Content-Type` of `url` via a HEAD request, or `None`.
    async fn head(&self, url: &Url) -> Option<String>;
}
