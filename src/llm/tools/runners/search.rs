//! Web-search tool runners (Kagi + `DuckDuckGo`).

use std::time::Duration;

use anodized::spec;

use super::super::ToolResult;
use super::{
    DuckDuckGoSearchToolCfg, KagiSearchToolCfg, KeenableSearchToolCfg, resolve_env_vars,
    truncate_chars,
};
use crate::error::InboxError;

pub(crate) async fn run_kagi_search_tool(
    client: &reqwest::Client,
    cfg: KagiSearchToolCfg<'_>,
    query: &str,
    limit: Option<u32>,
) -> Result<ToolResult, InboxError> {
    #[spec(requires: !cfg.endpoint.is_empty() && cfg.timeout_secs > 0)]
    fn validate_kagi_cfg(cfg: &KagiSearchToolCfg<'_>) {
        let _ = cfg;
    }
    validate_kagi_cfg(&cfg);

    let trimmed_query = query.trim();
    if trimmed_query.is_empty() {
        return Err(InboxError::LlmTool(
            "web_search missing non-empty 'query'".into(),
        ));
    }

    let req = build_kagi_request(client, &cfg, trimmed_query, limit)?;
    let timeout = Duration::from_secs(u64::from(cfg.timeout_secs));

    let resp = tokio::time::timeout(timeout, req.send())
        .await
        .map_err(|_| {
            InboxError::LlmTool(format!(
                "Kagi web_search timed out after {}s",
                cfg.timeout_secs
            ))
        })?
        .map_err(|e| InboxError::LlmTool(format!("Kagi web_search request failed: {e}")))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        let hint = if status.as_u16() == 401 {
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&body) {
                if json["meta"]["api_balance"].is_null() {
                    " (check: API credits loaded? Search API token used? Closed-beta access granted?)"
                } else {
                    " (invalid or expired API token)"
                }
            } else {
                ""
            }
        } else {
            ""
        };
        let preview: String = body.chars().take(200).collect();
        return Err(InboxError::LlmTool(format!(
            "Kagi web_search returned status {status}: {preview}{hint}"
        )));
    }

    parse_kagi_response(resp, trimmed_query, cfg.max_snippet_chars).await
}

fn build_kagi_request(
    client: &reqwest::Client,
    cfg: &KagiSearchToolCfg<'_>,
    query: &str,
    limit: Option<u32>,
) -> Result<reqwest::RequestBuilder, InboxError> {
    let result_limit = limit.unwrap_or(cfg.default_limit).clamp(1, 20);
    let mut endpoint = url::Url::parse(cfg.endpoint)
        .map_err(|e| InboxError::LlmTool(format!("Invalid Kagi endpoint URL: {e}")))?;
    {
        let mut qp = endpoint.query_pairs_mut();
        qp.append_pair("q", query);
        qp.append_pair("limit", &result_limit.to_string());
    }

    let mut req = client.get(endpoint);
    if let Some(token) = cfg.api_token {
        let resolved = resolve_env_vars(token);
        let token_value = resolved.trim();
        if token_value.is_empty() {
            return Err(InboxError::LlmTool(
                "Kagi API token is empty (web_search.api_token)".into(),
            ));
        }
        req = req.header(reqwest::header::AUTHORIZATION, format!("Bot {token_value}"));
    }

    Ok(req)
}

async fn parse_kagi_response(
    resp: reqwest::Response,
    query: &str,
    max_snippet_chars: usize,
) -> Result<ToolResult, InboxError> {
    let json: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| InboxError::LlmTool(format!("Kagi web_search JSON parse failed: {e}")))?;

    if let Some(error) = json.get("error").and_then(serde_json::Value::as_array)
        && !error.is_empty()
    {
        return Err(InboxError::LlmTool(format!(
            "Kagi web_search API error: {}",
            error
                .iter()
                .filter_map(serde_json::Value::as_str)
                .collect::<Vec<_>>()
                .join("; ")
        )));
    }

    let results = json
        .get("data")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| InboxError::LlmTool("Kagi web_search returned no data array".into()))?;

    if results.is_empty() {
        return Ok(ToolResult::Text(format!(
            "Kagi web_search results for \"{query}\": no results."
        )));
    }

    let lines = results
        .iter()
        .enumerate()
        .map(|(idx, item)| format_kagi_result_line(idx + 1, item, max_snippet_chars))
        .collect::<Vec<_>>();

    Ok(ToolResult::Text(format!(
        "Kagi web_search results for \"{query}\":\n\n{}",
        lines.join("\n\n")
    )))
}

#[spec(requires: rank > 0)]
fn format_kagi_result_line(
    rank: usize,
    item: &serde_json::Value,
    max_snippet_chars: usize,
) -> String {
    let title = item
        .get("title")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("(untitled)");
    let url = item
        .get("url")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    let snippet = item
        .get("snippet")
        .and_then(serde_json::Value::as_str)
        .map(|s| s.replace('\n', " "))
        .unwrap_or_default();
    let snippet = truncate_chars(&snippet, max_snippet_chars);
    format!(
        "{rank}. {}\nURL: {}\nSnippet: {}",
        title.trim(),
        url.trim(),
        snippet.trim()
    )
}

pub(crate) async fn run_keenable_search_tool(
    client: &reqwest::Client,
    cfg: KeenableSearchToolCfg<'_>,
    query: &str,
    limit: Option<u32>,
) -> Result<ToolResult, InboxError> {
    #[spec(requires: !cfg.endpoint.is_empty() && cfg.timeout_secs > 0)]
    fn validate_keenable_cfg(cfg: &KeenableSearchToolCfg<'_>) {
        let _ = cfg;
    }
    validate_keenable_cfg(&cfg);

    let trimmed_query = query.trim();
    if trimmed_query.is_empty() {
        return Err(InboxError::LlmTool(
            "keenable_search missing non-empty 'query'".into(),
        ));
    }

    let result_limit = limit.unwrap_or(cfg.default_limit).clamp(1, 20);
    let req = build_keenable_request(client, &cfg, trimmed_query)?;
    let timeout = Duration::from_secs(u64::from(cfg.timeout_secs));

    let resp = tokio::time::timeout(timeout, req.send())
        .await
        .map_err(|_| {
            InboxError::LlmTool(format!(
                "Keenable keenable_search timed out after {}s",
                cfg.timeout_secs
            ))
        })?
        .map_err(|e| {
            InboxError::LlmTool(format!("Keenable keenable_search request failed: {e}"))
        })?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        let hint = if status.as_u16() == 401 {
            " (check: keenable_search.api_key set? Get one at keenable.ai/console)"
        } else {
            ""
        };
        let preview: String = body.chars().take(200).collect();
        return Err(InboxError::LlmTool(format!(
            "Keenable keenable_search returned status {status}: {preview}{hint}"
        )));
    }

    parse_keenable_response(
        resp,
        trimmed_query,
        result_limit as usize,
        cfg.max_snippet_chars,
    )
    .await
}

fn build_keenable_request(
    client: &reqwest::Client,
    cfg: &KeenableSearchToolCfg<'_>,
    query: &str,
) -> Result<reqwest::RequestBuilder, InboxError> {
    let api_key = cfg.api_key.ok_or_else(|| {
        InboxError::LlmTool("Keenable API key is not set (keenable_search.api_key)".into())
    })?;
    let resolved = resolve_env_vars(api_key);
    let key_value = resolved.trim();
    if key_value.is_empty() {
        return Err(InboxError::LlmTool(
            "Keenable API key is empty (keenable_search.api_key)".into(),
        ));
    }

    let body = serde_json::json!({ "query": query });
    Ok(client
        .post(cfg.endpoint)
        .header("X-API-Key", key_value)
        .json(&body))
}

async fn parse_keenable_response(
    resp: reqwest::Response,
    query: &str,
    limit: usize,
    max_snippet_chars: usize,
) -> Result<ToolResult, InboxError> {
    let json: serde_json::Value = resp.json().await.map_err(|e| {
        InboxError::LlmTool(format!("Keenable keenable_search JSON parse failed: {e}"))
    })?;

    let results = json
        .get("results")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| {
            InboxError::LlmTool("Keenable keenable_search returned no results array".into())
        })?;

    if results.is_empty() {
        return Ok(ToolResult::Text(format!(
            "Keenable keenable_search results for \"{query}\": no results."
        )));
    }

    let lines = results
        .iter()
        .take(limit)
        .enumerate()
        .map(|(idx, item)| format_keenable_result_line(idx + 1, item, max_snippet_chars))
        .collect::<Vec<_>>();

    Ok(ToolResult::Text(format!(
        "Keenable keenable_search results for \"{query}\":\n\n{}",
        lines.join("\n\n")
    )))
}

#[spec(requires: rank > 0)]
fn format_keenable_result_line(
    rank: usize,
    item: &serde_json::Value,
    max_snippet_chars: usize,
) -> String {
    let title = item
        .get("title")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("(untitled)");
    let url = item
        .get("url")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    let str_field = |key: &str| {
        item.get(key)
            .and_then(serde_json::Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
    };
    let snippet = str_field("snippet")
        .or_else(|| str_field("description"))
        .unwrap_or_default()
        .replace('\n', " ");
    let snippet = truncate_chars(&snippet, max_snippet_chars);
    format!(
        "{rank}. {}\nURL: {}\nSnippet: {}",
        title.trim(),
        url.trim(),
        snippet.trim()
    )
}

pub(crate) async fn run_duckduckgo_search_tool(
    client: &reqwest::Client,
    cfg: DuckDuckGoSearchToolCfg<'_>,
    query: &str,
    limit: Option<u32>,
) -> Result<ToolResult, InboxError> {
    #[spec(requires: !cfg.endpoint.is_empty() && cfg.timeout_secs > 0)]
    fn validate(cfg: &DuckDuckGoSearchToolCfg<'_>) {
        let _ = cfg;
    }
    validate(&cfg);

    let trimmed_query = query.trim();
    if trimmed_query.is_empty() {
        return Err(InboxError::LlmTool(
            "duckduckgo_search missing non-empty 'query'".into(),
        ));
    }

    let result_limit = limit.unwrap_or(cfg.default_limit).clamp(1, 20);
    let mut endpoint = url::Url::parse(cfg.endpoint)
        .map_err(|e| InboxError::LlmTool(format!("Invalid DDG endpoint URL: {e}")))?;
    endpoint.query_pairs_mut().append_pair("q", trimmed_query);

    let timeout = Duration::from_secs(u64::from(cfg.timeout_secs));
    let resp = tokio::time::timeout(
        timeout,
        client
            .get(endpoint)
            .header(
                reqwest::header::USER_AGENT,
                "Mozilla/5.0 (compatible; inbox-search/1.0)",
            )
            .header(reqwest::header::ACCEPT, "text/html")
            .send(),
    )
    .await
    .map_err(|_| {
        InboxError::LlmTool(format!(
            "DuckDuckGo search timed out after {}s",
            cfg.timeout_secs
        ))
    })?
    .map_err(|e| InboxError::LlmTool(format!("DuckDuckGo search request failed: {e}")))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        let preview: String = body.chars().take(200).collect();
        return Err(InboxError::LlmTool(format!(
            "DuckDuckGo search returned status {status}: {preview}"
        )));
    }

    let body = resp
        .text()
        .await
        .map_err(|e| InboxError::LlmTool(format!("DuckDuckGo body read failed: {e}")))?;

    Ok(parse_ddg_html(
        &body,
        trimmed_query,
        result_limit as usize,
        cfg.max_snippet_chars,
    ))
}

fn parse_ddg_html(html: &str, query: &str, limit: usize, max_snippet_chars: usize) -> ToolResult {
    use scraper::{Html, Selector};

    let doc = Html::parse_document(html);
    // Selectors are constant literals; a parse failure is impossible at runtime
    // but is handled gracefully (empty result) rather than panicking.
    let (Ok(container_sel), Ok(title_sel), Ok(snippet_sel)) = (
        Selector::parse("div.results_links_deep"),
        Selector::parse("a.result__a"),
        Selector::parse(".result__snippet"),
    ) else {
        return ToolResult::Text(format!(
            "DuckDuckGo search results for \"{query}\": no results."
        ));
    };

    let mut lines: Vec<String> = Vec::new();

    for container in doc.select(&container_sel).take(limit) {
        let Some(title_node) = container.select(&title_sel).next() else {
            continue;
        };
        let title: String = title_node.text().collect();
        let href = title_node.attr("href").unwrap_or("");

        let url = url::Url::parse(&format!("https://duckduckgo.com{href}"))
            .ok()
            .and_then(|u| {
                u.query_pairs()
                    .find(|(k, _)| k == "uddg")
                    .map(|(_, v)| v.into_owned())
            })
            .unwrap_or_else(|| href.to_owned());

        let snippet: String = container
            .select(&snippet_sel)
            .next()
            .map(|el| el.text().collect())
            .unwrap_or_default();
        let snippet = snippet.replace('\n', " ");
        let snippet = truncate_chars(&snippet, max_snippet_chars);

        lines.push(format!(
            "{}. {}\nURL: {}\nSnippet: {}",
            lines.len() + 1,
            title.trim(),
            url.trim(),
            snippet.trim()
        ));
    }

    if lines.is_empty() {
        return ToolResult::Text(format!(
            "DuckDuckGo search results for \"{query}\": no results."
        ));
    }

    ToolResult::Text(format!(
        "DuckDuckGo search results for \"{query}\":\n\n{}",
        lines.join("\n\n")
    ))
}
