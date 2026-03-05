use std::time::Duration;

use anodized::spec;
use tokio::process::Command;
use tracing::warn;

use crate::error::InboxError;

use super::ToolResult;

/// Execute a shell tool. argv may contain `{url}` and `{filename}` placeholders.
/// Arguments are passed as separate argv entries — no shell interpolation.
pub(super) async fn run_shell_tool(
    argv: &[String],
    url: &str,
    filename: &str,
    timeout_secs: u32,
) -> Result<ToolResult, InboxError> {
    if argv.is_empty() {
        return Err(InboxError::LlmTool("Shell tool has empty argv".into()));
    }

    let program = &argv[0];
    let processed_args: Vec<String> = argv[1..]
        .iter()
        .map(|a| a.replace("{url}", url).replace("{filename}", filename))
        .collect();

    let output = tokio::time::timeout(
        Duration::from_secs(u64::from(timeout_secs)),
        Command::new(program).args(&processed_args).output(),
    )
    .await
    .map_err(|_| InboxError::LlmTool(format!("Shell tool timed out after {timeout_secs}s")))?
    .map_err(|e| InboxError::LlmTool(format!("Shell tool exec error: {e}")))?;

    if !output.status.success() {
        warn!(
            program,
            status = ?output.status,
            stderr = %String::from_utf8_lossy(&output.stderr),
            "Shell tool exited with error"
        );
    }

    Ok(ToolResult::Text(
        String::from_utf8_lossy(&output.stdout).into_owned(),
    ))
}

/// Configuration bundle for an HTTP tool backend call.
pub(super) struct HttpToolCfg<'a> {
    pub endpoint: &'a str,
    pub method: &'a str,
    pub auth_header: Option<&'a str>,
    pub body_template: Option<&'a str>,
    pub response_path: &'a str,
    pub timeout_secs: u32,
}

pub(super) struct CrawlToolCfg<'a> {
    pub endpoint: &'a str,
    pub auth_header: Option<&'a str>,
    pub timeout_secs: u32,
    pub priority: i32,
}

pub(super) struct KagiSearchToolCfg<'a> {
    pub endpoint: &'a str,
    pub api_token: Option<&'a str>,
    pub timeout_secs: u32,
    pub default_limit: u32,
    pub max_snippet_chars: usize,
}

/// Execute an HTTP tool backend.
pub(super) async fn run_http_tool(
    client: &reqwest::Client,
    cfg: HttpToolCfg<'_>,
    url: &str,
    filename: &str,
) -> Result<ToolResult, InboxError> {
    #[spec(requires: !cfg.endpoint.is_empty() && cfg.timeout_secs > 0)]
    fn validate_http_cfg(cfg: &HttpToolCfg<'_>) {
        let _ = cfg;
    }
    validate_http_cfg(&cfg);

    let endpoint_resolved = cfg
        .endpoint
        .replace("{url}", url)
        .replace("{filename}", filename);

    let mut req = match cfg.method.to_uppercase().as_str() {
        "POST" => client.post(&endpoint_resolved),
        _ => client.get(&endpoint_resolved),
    };

    if let Some(auth) = cfg.auth_header {
        let resolved = resolve_env_vars(auth);
        if let Some((name, value)) = resolved.split_once(':') {
            req = req.header(name.trim(), value.trim());
        }
    }

    if let Some(tmpl) = cfg.body_template {
        let body = tmpl.replace("{url}", url).replace("{filename}", filename);
        req = req.header("content-type", "application/json").body(body);
    }

    let timeout = Duration::from_secs(u64::from(cfg.timeout_secs));
    let resp = tokio::time::timeout(timeout, req.send())
        .await
        .map_err(|_| {
            InboxError::LlmTool(format!("HTTP tool timed out after {}s", cfg.timeout_secs))
        })?
        .map_err(|e| InboxError::LlmTool(format!("HTTP tool request failed: {e}")))?;

    if !resp.status().is_success() {
        return Err(InboxError::LlmTool(format!(
            "HTTP tool returned status {}",
            resp.status()
        )));
    }

    let ct = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_owned();

    let body_bytes = resp
        .bytes()
        .await
        .map_err(|e| InboxError::LlmTool(format!("HTTP tool body read failed: {e}")))?;

    if ct.contains("application/json") && !cfg.response_path.is_empty() {
        let json: serde_json::Value = serde_json::from_slice(&body_bytes)
            .map_err(|e| InboxError::LlmTool(format!("HTTP tool JSON parse failed: {e}")))?;
        let mut node = &json;
        for key in cfg.response_path.split('.') {
            node = node.get(key).ok_or_else(|| {
                InboxError::LlmTool(format!("response_path key '{key}' not found"))
            })?;
        }
        return Ok(ToolResult::Text(
            node.as_str().unwrap_or(&node.to_string()).to_owned(),
        ));
    }

    Ok(ToolResult::Text(
        String::from_utf8_lossy(&body_bytes).into_owned(),
    ))
}

pub(super) async fn run_crawler_tool(
    client: &reqwest::Client,
    cfg: CrawlToolCfg<'_>,
    url: &str,
) -> Result<ToolResult, InboxError> {
    let body = serde_json::json!({
        "urls": [url],
        "priority": cfg.priority,
    });

    let timeout = Duration::from_secs(u64::from(cfg.timeout_secs));
    let mut req = client.post(cfg.endpoint).json(&body);

    if let Some(auth) = cfg.auth_header {
        let resolved = resolve_env_vars(auth);
        if let Some((name, value)) = resolved.split_once(':') {
            req = req.header(name.trim(), value.trim());
        }
    }

    let resp = tokio::time::timeout(timeout, req.send())
        .await
        .map_err(|_| {
            InboxError::LlmTool(format!(
                "Crawler tool timed out after {}s",
                cfg.timeout_secs
            ))
        })?
        .map_err(|e| InboxError::LlmTool(format!("Crawler tool request failed: {e}")))?;

    if !resp.status().is_success() {
        return Err(InboxError::LlmTool(format!(
            "Crawler returned status {}",
            resp.status()
        )));
    }

    let json: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| InboxError::LlmTool(format!("Crawler JSON parse failed: {e}")))?;

    let results = json["results"]
        .as_array()
        .and_then(|a| a.first())
        .ok_or_else(|| InboxError::LlmTool("Crawler returned no results".into()))?;

    let title = results["metadata"]["title"].as_str().unwrap_or("").trim();
    let markdown = results["markdown"]["raw_markdown"]
        .as_str()
        .unwrap_or("")
        .trim();
    let html = results["cleaned_html"].as_str().unwrap_or("").trim();

    let mut parts: Vec<String> = Vec::new();
    if !title.is_empty() {
        parts.push(format!("Title: {title}"));
    }
    if !markdown.is_empty() {
        parts.push(markdown.to_owned());
    } else if !html.is_empty() {
        parts.push(format!("HTML fallback: {html}"));
    } else {
        parts.push("(no markdown/html content returned)".into());
    }

    Ok(ToolResult::Text(parts.join("\n\n")))
}

pub(super) async fn run_kagi_search_tool(
    client: &reqwest::Client,
    cfg: KagiSearchToolCfg<'_>,
    query: &str,
    limit: Option<u32>,
) -> Result<ToolResult, InboxError> {
    let trimmed_query = query.trim();
    if trimmed_query.is_empty() {
        return Err(InboxError::LlmTool(
            "web_search missing non-empty 'query'".into(),
        ));
    }

    let result_limit = limit.unwrap_or(cfg.default_limit).clamp(1, 20);
    let mut endpoint = url::Url::parse(cfg.endpoint)
        .map_err(|e| InboxError::LlmTool(format!("Invalid Kagi endpoint URL: {e}")))?;
    {
        let mut qp = endpoint.query_pairs_mut();
        qp.append_pair("q", trimmed_query);
        qp.append_pair("limit", &result_limit.to_string());
    }
    let timeout = Duration::from_secs(u64::from(cfg.timeout_secs));
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
        let preview: String = body.chars().take(200).collect();
        return Err(InboxError::LlmTool(format!(
            "Kagi web_search returned status {status}: {preview}"
        )));
    }

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
            "Kagi web_search results for \"{trimmed_query}\": no results."
        )));
    }

    let mut lines = Vec::with_capacity(results.len());
    for (idx, item) in results.iter().enumerate() {
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
        let snippet = truncate_chars(&snippet, cfg.max_snippet_chars);
        lines.push(format!(
            "{}. {}\nURL: {}\nSnippet: {}",
            idx + 1,
            title.trim(),
            url.trim(),
            snippet.trim()
        ));
    }

    Ok(ToolResult::Text(format!(
        "Kagi web_search results for \"{trimmed_query}\":\n\n{}",
        lines.join("\n\n")
    )))
}

/// Expand `${VAR}` patterns in a string using environment variables.
pub(super) fn resolve_env_vars(s: &str) -> String {
    let re = regex::Regex::new(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}").unwrap();
    re.replace_all(s, |caps: &regex::Captures<'_>| {
        std::env::var(&caps[1]).unwrap_or_default()
    })
    .into_owned()
}

fn truncate_chars(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_owned()
    } else {
        s.chars().take(max).collect()
    }
}
