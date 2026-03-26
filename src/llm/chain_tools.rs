use anodized::spec;
use tracing::{info, warn};

use crate::error::InboxError;
use crate::message::LlmResponse;
use crate::pipeline::url_extractor::extract_http_url_strings;

use super::{LlmClient, LlmCompletion, LlmRequest, ToolCall, tools};

pub(super) async fn retry_inner(
    backend: &(dyn LlmClient + 'static),
    req: &LlmRequest,
    retries: u32,
) -> Result<LlmCompletion, InboxError> {
    let mut last_err = InboxError::Llm("no attempts".into());
    for attempt in 0..=retries {
        if attempt > 0 {
            let delay_ms = 500u64.saturating_mul(2u64.pow(attempt - 1));
            tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
        }
        match backend.complete(req.clone()).await {
            Ok(c) => return Ok(c),
            Err(e) => {
                warn!(
                    ?e,
                    attempt,
                    max_retries = retries,
                    backend = backend.name(),
                    "Inner LLM call retry"
                );
                last_err = e;
            }
        }
    }
    Err(last_err)
}

#[spec(requires: max_chars > 0)]
pub(super) fn truncate_tool_result(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        text.to_owned()
    } else {
        let t: String = text.chars().take(max_chars).collect();
        format!("{t}\n... [truncated to {max_chars} chars]")
    }
}

#[spec(requires: !calls.is_empty())]
pub(super) async fn execute_tool_calls(
    executor: &tools::ToolExecutor,
    calls: &[ToolCall],
    req: &LlmRequest,
    tool_result_max_chars: usize,
) -> ToolExecutionOutput {
    let results = futures::future::join_all(calls.iter().map(|call| {
        executor.execute(
            &call.name,
            &call.arguments,
            req.msg_id,
            req.attachments_dir.as_path(),
            &req.source_name,
        )
    }))
    .await;

    let mut outputs = Vec::with_capacity(calls.len());
    let mut named_results: Vec<(String, String)> = Vec::with_capacity(calls.len());
    let mut source_url_set: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut source_urls: Vec<String> = Vec::new();

    for (call, result) in calls.iter().zip(results) {
        info!(tool = %call.name, "Executing LLM tool call");
        match result {
            Ok(tools::ToolResult::Text(text) | tools::ToolResult::Attachment { text, .. }) => {
                let result_preview: String = text.chars().take(120).collect();
                info!(
                    tool = %call.name,
                    result_len = text.len(),
                    result_preview = %result_preview,
                    "Tool call result"
                );
                for url in extract_http_url_strings(&text) {
                    if source_url_set.insert(url.clone()) {
                        source_urls.push(url);
                    }
                }
                let text = if tool_result_max_chars > 0 {
                    let orig_len = text.len();
                    let truncated = truncate_tool_result(&text, tool_result_max_chars);
                    if truncated.len() < orig_len {
                        info!(
                            tool = %call.name,
                            orig_len,
                            effective_len = truncated.len(),
                            max_chars = tool_result_max_chars,
                            "Tool result truncated"
                        );
                    }
                    truncated
                } else {
                    text
                };
                outputs.push(format!("tool `{}`: {text}", call.name));
                named_results.push((call.name.clone(), text));
            }
            Err(e) => {
                warn!(tool = %call.name, ?e, "Tool call failed");
                outputs.push(format!("tool `{}` error: {e}", call.name));
            }
        }
    }

    ToolExecutionOutput {
        text: outputs.join("\n"),
        named_results,
        source_urls,
    }
}

pub(super) struct ToolExecutionOutput {
    pub text: String,
    pub named_results: Vec<(String, String)>,
    pub source_urls: Vec<String>,
}

pub(super) fn append_missing_source_links(
    mut resp: LlmResponse,
    tool_source_urls: &[String],
) -> LlmResponse {
    if tool_source_urls.is_empty() {
        return resp;
    }

    let mut already_present: std::collections::HashSet<String> =
        extract_http_url_strings(&resp.summary)
            .into_iter()
            .collect();
    if let Some(excerpt) = &resp.excerpt {
        already_present.extend(extract_http_url_strings(excerpt));
    }

    let missing: Vec<&str> = tool_source_urls
        .iter()
        .filter(|url| !already_present.contains(*url))
        .map(String::as_str)
        .collect();

    if missing.is_empty() {
        return resp;
    }

    resp.summary.push_str("\n\nSources:");
    for url in missing {
        resp.summary.push_str("\n- ");
        resp.summary.push_str(url);
    }

    resp
}

#[cfg(test)]
mod chain_tests {
    use super::truncate_tool_result;

    #[test]
    fn truncate_short_text_passes_through() {
        let text = "hello world";
        assert_eq!(truncate_tool_result(text, 100), text);
    }

    #[test]
    fn truncate_long_text_appends_notice() {
        let text = "a".repeat(30);
        let result = truncate_tool_result(&text, 20);
        assert!(result.starts_with(&"a".repeat(20)));
        assert!(result.contains("[truncated to 20 chars]"));
        assert!(!result.contains(&"a".repeat(21)));
    }

    #[test]
    fn truncate_exact_length_passes_through() {
        let text = "x".repeat(50);
        let result = truncate_tool_result(&text, 50);
        assert_eq!(result, text);
    }
}
