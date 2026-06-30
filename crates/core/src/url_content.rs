/// Text extracted from a fetched URL.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct UrlContent {
    pub url: String,
    pub text: String,
    /// Approximate title extracted from the page, if available.
    pub page_title: Option<String>,
    /// h1/h2 headings extracted from the page, in document order.
    pub headings: Vec<String>,
}
