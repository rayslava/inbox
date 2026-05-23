use anodized::spec;

use crate::config::ToolBackendConfig;
use crate::message::Attachment;
use crate::pipeline::url_fetcher::UrlFetcher;

mod builders;
mod runners;

pub use builders::{add_memory_tools, default_tools, from_tooling};

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_runners;
#[cfg(test)]
mod tests_search_live;
#[cfg(test)]
mod tests_search_memory;

// ── Tool definition ───────────────────────────────────────────────────────────

/// A configured tool the LLM can call.
pub struct Tool {
    pub name: String,
    pub description: String,
    pub enabled: bool,
    pub retries: u32,
    pub backend: ToolBackendConfig,
}

impl Tool {
    #[must_use]
    #[spec(requires: !self.name.trim().is_empty() && !self.description.trim().is_empty())]
    pub fn openai_definition(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": tool_parameters(&self.name),
            }
        })
    }
}

#[spec(requires: !name.trim().is_empty())]
fn tool_parameters(name: &str) -> serde_json::Value {
    match name {
        "scrape_page" => serde_json::json!({
            "type": "object",
            "properties": {
                "url": { "type": "string", "description": "The URL to scrape" }
            },
            "required": ["url"]
        }),
        "download_file" => serde_json::json!({
            "type": "object",
            "properties": {
                "url": { "type": "string", "description": "The URL of the file to download" }
            },
            "required": ["url"]
        }),
        "crawl_url" => serde_json::json!({
            "type": "object",
            "properties": {
                "url": { "type": "string", "description": "The URL to crawl" }
            },
            "required": ["url"]
        }),
        "web_search" | "duckduckgo_search" => serde_json::json!({
            "type": "object",
            "properties": {
                "query": { "type": "string", "description": "The web search query" },
                "limit": { "type": "integer", "description": "Optional max number of results (1-20)" }
            },
            "required": ["query"]
        }),
        "memory_save" => serde_json::json!({
            "type": "object",
            "properties": {
                "key": { "type": "string", "description": "Short identifier for the memory" },
                "value": { "type": "string", "description": "Content to remember" }
            },
            "required": ["key", "value"]
        }),
        "memory_recall" => serde_json::json!({
            "type": "object",
            "properties": {
                "query": { "type": "string", "description": "Search query to find relevant memories" }
            },
            "required": ["query"]
        }),
        "memory_link" => serde_json::json!({
            "type": "object",
            "properties": {
                "from_key": { "type": "string", "description": "Key of the source memory" },
                "to_key": { "type": "string", "description": "Key of the target memory" },
                "relation": { "type": "string", "description": "Relationship type (e.g. 'related_to', 'depends_on')" }
            },
            "required": ["from_key", "to_key", "relation"]
        }),
        "memory_context" => serde_json::json!({
            "type": "object",
            "properties": {
                "query": { "type": "string", "description": "Memory key to find connected memories for" },
                "hops": { "type": "integer", "description": "Number of graph traversal hops (default 1, max 3)" }
            },
            "required": ["query"]
        }),
        _ => serde_json::json!({ "type": "object", "properties": {} }),
    }
}

// ── ToolExecutor ──────────────────────────────────────────────────────────────

/// Dispatches LLM tool calls to their configured backend.
pub struct ToolExecutor {
    tools: Vec<Tool>,
    fetcher: UrlFetcher,
    http_client: reqwest::Client,
    pub(super) memory_store: Option<std::sync::Arc<crate::memory::MemoryStore>>,
}

mod executor;

// ── ToolResult ────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum ToolResult {
    Text(String),
    Attachment {
        text: String,
        attachment: Attachment,
    },
}

impl ToolResult {
    #[must_use]
    pub fn text(&self) -> &str {
        match self {
            Self::Text(t) => t,
            Self::Attachment { text, .. } => text,
        }
    }
}
