use crate::config::{ToolBackendConfig, ToolingConfig};
use crate::pipeline::url_fetcher::UrlFetcher;

use super::{Tool, ToolExecutor};

/// Build the default tool list used when tooling config is not customized.
#[must_use]
pub fn default_tools(fetcher: UrlFetcher) -> ToolExecutor {
    let tools = vec![
        Tool {
            name: "scrape_page".into(),
            description: "Fetch and extract readable text from a web page URL".into(),
            enabled: true,
            retries: 3,
            backend: ToolBackendConfig::Internal { timeout_secs: 15 },
        },
        Tool {
            name: "download_file".into(),
            description: "Download a file from a URL and save it as an attachment".into(),
            enabled: true,
            retries: 3,
            backend: ToolBackendConfig::Internal { timeout_secs: 15 },
        },
        Tool {
            name: "crawl_url".into(),
            description: "Crawl a URL and return markdown/html from crawler service".into(),
            enabled: false,
            retries: 1,
            backend: ToolBackendConfig::Crawler {
                endpoint: "http://localhost:11235/crawl".into(),
                auth_header: None,
                timeout_secs: 30,
                priority: 10,
            },
        },
        Tool {
            name: "web_search".into(),
            description: "Search the web and return top results".into(),
            enabled: false,
            retries: 1,
            backend: ToolBackendConfig::KagiSearch {
                endpoint: "https://kagi.com/api/v0/search".into(),
                api_token: None,
                timeout_secs: 15,
                default_limit: 5,
                max_snippet_chars: 320,
            },
        },
        Tool {
            name: "duckduckgo_search".into(),
            description: "Search the web via DuckDuckGo and return top results".into(),
            enabled: false,
            retries: 1,
            backend: ToolBackendConfig::DuckDuckGoSearch {
                endpoint: "https://duckduckgo.com/html/".into(),
                timeout_secs: 15,
                default_limit: 5,
                max_snippet_chars: 320,
            },
        },
    ];
    ToolExecutor::new(tools, fetcher)
}

fn desc_or(cfg: &str, default: &str) -> String {
    if cfg.trim().is_empty() {
        default.to_owned()
    } else {
        cfg.to_owned()
    }
}

#[must_use]
pub fn from_tooling(tooling: &ToolingConfig, fetcher: UrlFetcher) -> ToolExecutor {
    let scrape_desc = desc_or(
        &tooling.scrape_page.description,
        "Fetch and extract readable text from a web page URL",
    );
    let download_desc = desc_or(
        &tooling.download_file.description,
        "Download a file from a URL and save it as an attachment",
    );
    let crawl_desc = desc_or(
        &tooling.crawl_url.description,
        "Crawl a URL and return markdown/html from crawler service",
    );
    let web_search_desc = desc_or(
        &tooling.web_search.description,
        "Search the web via Kagi and return top results",
    );
    let ddg_search_desc = desc_or(
        &tooling.duckduckgo_search.description,
        "Search the web via DuckDuckGo and return top results",
    );

    let tools = vec![
        Tool {
            name: "scrape_page".into(),
            description: scrape_desc,
            enabled: tooling.scrape_page.enabled,
            retries: tooling.scrape_page.retries,
            backend: tooling.scrape_page.backend.clone(),
        },
        Tool {
            name: "download_file".into(),
            description: download_desc,
            enabled: tooling.download_file.enabled,
            retries: tooling.download_file.retries,
            backend: tooling.download_file.backend.clone(),
        },
        Tool {
            name: "crawl_url".into(),
            description: crawl_desc,
            enabled: tooling.crawl_url.enabled,
            retries: tooling.crawl_url.retries,
            backend: ToolBackendConfig::Crawler {
                endpoint: tooling.crawl_url.endpoint.clone(),
                auth_header: tooling.crawl_url.auth_header.clone(),
                timeout_secs: tooling.crawl_url.timeout_secs,
                priority: tooling.crawl_url.priority,
            },
        },
        Tool {
            name: "web_search".into(),
            description: web_search_desc,
            enabled: tooling.web_search.enabled,
            retries: tooling.web_search.retries,
            backend: ToolBackendConfig::KagiSearch {
                endpoint: tooling.web_search.endpoint.clone(),
                api_token: tooling.web_search.api_token.clone(),
                timeout_secs: tooling.web_search.timeout_secs,
                default_limit: tooling.web_search.default_limit,
                max_snippet_chars: tooling.web_search.max_snippet_chars,
            },
        },
        Tool {
            name: "duckduckgo_search".into(),
            description: ddg_search_desc,
            enabled: tooling.duckduckgo_search.enabled,
            retries: tooling.duckduckgo_search.retries,
            backend: ToolBackendConfig::DuckDuckGoSearch {
                endpoint: tooling.duckduckgo_search.endpoint.clone(),
                timeout_secs: tooling.duckduckgo_search.timeout_secs,
                default_limit: tooling.duckduckgo_search.default_limit,
                max_snippet_chars: tooling.duckduckgo_search.max_snippet_chars,
            },
        },
    ];
    ToolExecutor::new(tools, fetcher)
}

/// Register the `memory_save` and `memory_recall` tools on an existing executor.
pub fn add_memory_tools(
    executor: &mut ToolExecutor,
    store: std::sync::Arc<crate::memory::MemoryStore>,
) {
    executor.memory_store = Some(store);
    executor.tools.push(Tool {
        name: "memory_save".into(),
        description: "Save a fact or note as a persistent memory with a short key. \
                      The value can be any text."
            .into(),
        enabled: true,
        retries: 0,
        backend: ToolBackendConfig::Memory,
    });
    executor.tools.push(Tool {
        name: "memory_recall".into(),
        description: "Search persistent memories for information related to a query. \
                      Returns matching key-value entries."
            .into(),
        enabled: true,
        retries: 0,
        backend: ToolBackendConfig::Memory,
    });
}
