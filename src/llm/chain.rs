use std::fmt::Write as _;

use anodized::spec;
use tracing::{debug, info, warn};

use crate::message::LlmResponse;
use crate::pipeline::url_extractor::extract_http_url_strings;

use super::{
    FallbackMode, LlmClient, LlmCompletion, LlmOutcome, LlmRequest, ToolCall, activate_thinking_tool_def,
    llm_call_tool_def, tools,
};

// ── LlmChain ─────────────────────────────────────────────────────────────────

pub struct LlmChain {
    backends: Vec<Box<dyn LlmClient>>,
    fallback: FallbackMode,
    max_tool_turns: usize,
    max_llm_tool_depth: u32,
    tool_executor: Option<tools::ToolExecutor>,
}

impl LlmChain {
    #[must_use]
    #[spec(requires: max_tool_turns > 0)]
    pub fn new(
        backends: Vec<Box<dyn LlmClient>>,
        fallback: FallbackMode,
        max_tool_turns: usize,
        tool_executor: Option<tools::ToolExecutor>,
        max_llm_tool_depth: u32,
    ) -> Self {
        Self {
            backends,
            fallback,
            max_tool_turns,
            max_llm_tool_depth,
            tool_executor,
        }
    }

    /// Try each backend in order with retries. On exhaustion, apply fallback policy.
    #[spec(requires: self.max_tool_turns > 0)]
    pub async fn complete(&self, req: LlmRequest) -> LlmOutcome {
        let thinking_supported = self.backends.iter().any(|b| b.thinking_supported());
        let mut tool_defs = self
            .tool_executor
            .as_ref()
            .map_or_else(Vec::new, tools::ToolExecutor::active_tool_definitions);
        if thinking_supported && !tool_defs.is_empty() {
            tool_defs.push(activate_thinking_tool_def());
        }
        if req.llm_depth < self.max_llm_tool_depth && !tool_defs.is_empty() {
            tool_defs.push(llm_call_tool_def());
        }

        for backend in &self.backends {
            for attempt in 0..backend.retries() {
                let start = std::time::Instant::now();
                let mut req_attempt = req.clone();
                req_attempt.tool_definitions = tool_defs.clone();
                let mut turns = 0usize;
                let mut thinking_activations = 0usize;
                let mut required_tool_prompts = 0usize;
                let mut tool_source_url_set: std::collections::HashSet<String> =
                    std::collections::HashSet::new();
                let mut tool_source_urls: Vec<String> = Vec::new();

                loop {
                    let tool_names: Vec<&str> = req_attempt
                        .tool_definitions
                        .iter()
                        .filter_map(|d| d["function"]["name"].as_str())
                        .collect();
                    let system_preview: String =
                        req_attempt.system_prompt.chars().take(300).collect();
                    let content_preview: String =
                        req_attempt.user_content.chars().take(600).collect();
                    debug!(
                        backend = backend.name(),
                        model = backend.model(),
                        turn = turns + 1,
                        tools = ?tool_names,
                        system_len = req_attempt.system_prompt.len(),
                        content_len = req_attempt.user_content.len(),
                        system_preview = %system_preview,
                        content_preview = %content_preview,
                        "LLM request"
                    );

                    match backend.complete(req_attempt.clone()).await {
                        Ok(LlmCompletion::Message(resp)) => {
                            if req_attempt.require_initial_tool_call
                                && turns == 0
                                && !tool_defs.is_empty()
                            {
                                if required_tool_prompts < 3 {
                                    debug!(
                                        backend = backend.name(),
                                        prompt_attempt = required_tool_prompts + 1,
                                        "Re-prompting model to make required initial tool call"
                                    );
                                    req_attempt.user_content.push_str(
                                        "\n\nA tool call is required before final JSON because URLs are present. First analyze and call exactly one best retrieval tool, then continue.",
                                    );
                                    required_tool_prompts += 1;
                                    continue;
                                }
                                warn!(
                                    backend = backend.name(),
                                    "Required initial tool call was not produced"
                                );
                                break;
                            }
                            metrics::counter!(
                                crate::telemetry::LLM_REQUESTS,
                                "backend" => backend.name().to_owned(),
                                "status" => "success"
                            )
                            .increment(1);
                            metrics::histogram!(
                                crate::telemetry::LLM_DURATION,
                                "backend" => backend.name().to_owned()
                            )
                            .record(start.elapsed().as_secs_f64());
                            return LlmOutcome::Success(append_missing_source_links(
                                resp,
                                &tool_source_urls,
                            ));
                        }
                        Ok(LlmCompletion::ToolCalls(calls)) => {
                            // Partition: activate_thinking is handled internally
                            let (thinking_calls, calls): (Vec<_>, Vec<_>) = calls
                                .into_iter()
                                .partition(|c| c.name == "activate_thinking");

                            if !thinking_calls.is_empty() {
                                if req_attempt.think.is_none() {
                                    info!(backend = backend.name(), "LLM activated thinking mode");
                                    req_attempt.think = Some(true);
                                }
                                thinking_activations += 1;
                                if calls.is_empty() {
                                    if thinking_activations >= self.max_tool_turns {
                                        warn!(
                                            backend = backend.name(),
                                            max = self.max_tool_turns,
                                            "activate_thinking loop limit reached"
                                        );
                                        break;
                                    }
                                    continue;
                                }
                            }

                            // Partition: llm_call is handled internally
                            let (llm_calls, calls): (Vec<_>, Vec<_>) =
                                calls.into_iter().partition(|c| c.name == "llm_call");

                            if !llm_calls.is_empty() {
                                if turns >= self.max_tool_turns {
                                    warn!(
                                        backend = backend.name(),
                                        max_turns = self.max_tool_turns,
                                        "Max tool turns reached during llm_call"
                                    );
                                    break;
                                }
                                for llm_call in &llm_calls {
                                    let result =
                                        self.execute_llm_tool_call(llm_call, &req_attempt).await;
                                    let _ = write!(
                                        req_attempt.user_content,
                                        "\n\ntool `llm_call` result: {result}"
                                    );
                                }
                                turns += 1;
                                if calls.is_empty() {
                                    continue;
                                }
                            }

                            if calls.is_empty() {
                                warn!(
                                    backend = backend.name(),
                                    "LLM returned empty tool call list"
                                );
                                break;
                            }

                            if turns >= self.max_tool_turns {
                                warn!(
                                    backend = backend.name(),
                                    max_turns = self.max_tool_turns,
                                    "Max tool turns reached"
                                );
                                break;
                            }
                            let Some(executor) = &self.tool_executor else {
                                warn!(
                                    backend = backend.name(),
                                    "Tool call requested but no executor configured"
                                );
                                break;
                            };

                            let output =
                                execute_tool_calls(executor, &calls, &req_attempt).await;
                            for url in output.source_urls {
                                if tool_source_url_set.insert(url.clone()) {
                                    tool_source_urls.push(url);
                                }
                            }
                            req_attempt
                                .user_content
                                .push_str("\n\n--- Tool execution results ---\n");
                            req_attempt.user_content.push_str(&output.text);
                            req_attempt.require_initial_tool_call = false;
                            turns += 1;
                        }
                        Err(e) => {
                            let elapsed_ms = start.elapsed().as_millis();
                            warn!(
                                ?e,
                                backend = backend.name(),
                                model = backend.model(),
                                attempt = attempt + 1,
                                total_attempts = backend.retries(),
                                elapsed_ms,
                                "LLM attempt failed"
                            );
                            break;
                        }
                    }
                }
                metrics::counter!(
                    crate::telemetry::LLM_REQUESTS,
                    "backend" => backend.name().to_owned(),
                    "status" => "failure"
                )
                .increment(1);
            }
            warn!(
                backend = backend.name(),
                model = backend.model(),
                retries = backend.retries(),
                "LLM backend exhausted all retries"
            );
        }

        warn!(
            backend_count = self.backends.len(),
            "All LLM backends failed, applying fallback"
        );
        match self.fallback {
            FallbackMode::Raw => LlmOutcome::RawFallback,
            FallbackMode::Discard => LlmOutcome::Discard,
        }
    }

    #[must_use]
    pub fn max_tool_turns(&self) -> usize {
        self.max_tool_turns
    }

    async fn execute_llm_tool_call(&self, call: &ToolCall, parent_req: &LlmRequest) -> String {
        let system_prompt = call.arguments["system_prompt"]
            .as_str()
            .unwrap_or("You are a helpful assistant.")
            .to_owned();
        let content = call.arguments["content"].as_str().unwrap_or("").to_owned();

        let sub_req = LlmRequest {
            system_prompt,
            user_content: content,
            msg_id: parent_req.msg_id,
            attachments_dir: parent_req.attachments_dir.clone(),
            tool_definitions: vec![],
            require_initial_tool_call: false,
            images: vec![],
            think: None,
            llm_depth: parent_req.llm_depth + 1,
        };

        for backend in &self.backends {
            match backend.complete_raw(sub_req.clone()).await {
                Ok(text) => return text,
                Err(e) => {
                    warn!(?e, backend = backend.name(), "llm_call sub-request failed");
                }
            }
        }

        "llm_call failed: all backends exhausted".into()
    }
}

// ── Tool execution helpers ────────────────────────────────────────────────────

#[spec(requires: !calls.is_empty())]
async fn execute_tool_calls(
    executor: &tools::ToolExecutor,
    calls: &[ToolCall],
    req: &LlmRequest,
) -> ToolExecutionOutput {
    let results = futures::future::join_all(calls.iter().map(|call| {
        executor.execute(
            &call.name,
            &call.arguments,
            req.msg_id,
            req.attachments_dir.as_path(),
        )
    }))
    .await;

    let mut outputs = Vec::with_capacity(calls.len());
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
                outputs.push(format!("tool `{}`: {text}", call.name));
            }
            Err(e) => {
                warn!(tool = %call.name, ?e, "Tool call failed");
                outputs.push(format!("tool `{}` error: {e}", call.name));
            }
        }
    }

    ToolExecutionOutput {
        text: outputs.join("\n"),
        source_urls,
    }
}

struct ToolExecutionOutput {
    text: String,
    source_urls: Vec<String>,
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
