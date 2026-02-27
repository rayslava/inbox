/// Text extracted from a fetched URL.
#[derive(Debug, Clone)]
pub struct UrlContent {
    pub url: String,
    pub text: String,
    /// Approximate title extracted from the page, if available.
    pub page_title: Option<String>,
}
