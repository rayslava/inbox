use std::fmt::Write as _;

use anodized::spec;
use tracing::{debug, info, warn};

use super::chain_tools::{append_missing_source_links, execute_tool_calls, retry_inner};
use super::{
    FallbackMode, LlmClient, LlmCompletion, LlmOutcome, LlmRequest, LlmTurnProgress, ToolCall,
    activate_thinking_tool_def, llm_call_tool_def, tools,
};

// ── LlmChain ─────────────────────────────────────────────────────────────────

pub struct LlmChain {
    backends: Vec<Box<dyn LlmClient>>,
    fallback: FallbackMode,
    max_tool_turns: usize,
    max_llm_tool_depth: u32,
    tool_executor: Option<tools::ToolExecutor>,
    inner_retries: u32,
    tool_result_max_chars: usize,
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
        inner_retries: u32,
        tool_result_max_chars: usize,
    ) -> Self {
        Self {
            backends,
            fallback,
            max_tool_turns,
            max_llm_tool_depth,
            tool_executor,
            inner_retries,
            tool_result_max_chars,
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

        // Fallback state persists across all backend+attempt loops.
        let mut fallback_source_urls: Vec<String> = Vec::new();
        let mut fallback_tool_results: Vec<(String, String)> = Vec::new();
        // Helper models consulted via llm_call sub-requests (deduped, ordered by first use).
        let mut helpers: Vec<String> = Vec::new();
        // Total tool-call executions across every backend attempt in this run.
        let mut tool_calls_made: usize = 0;

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
                let mut accumulated_tool_results: Vec<(String, String)> = Vec::new();

                loop {
                    let tool_names_debug: Vec<&str> = req_attempt
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
                        tools = ?tool_names_debug,
                        system_len = req_attempt.system_prompt.len(),
                        content_len = req_attempt.user_content.len(),
                        system_preview = %system_preview,
                        content_preview = %content_preview,
                        "LLM request"
                    );

                    match retry_inner(backend.as_ref(), &req_attempt, self.inner_retries).await {
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
                                fallback_source_urls.clone_from(&tool_source_urls);
                                fallback_tool_results.clone_from(&accumulated_tool_results);
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
                            return LlmOutcome::Success {
                                response: append_missing_source_links(resp, &tool_source_urls),
                                helpers,
                                tool_calls_made,
                            };
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
                                        fallback_source_urls.clone_from(&tool_source_urls);
                                        fallback_tool_results.clone_from(&accumulated_tool_results);
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
                                    warn!(
                                        backend = backend.name(),
                                        max_turns = self.max_tool_turns,
                                        "Max tool turns reached, attempting forced summary"
                                    );
                                    let mut force_req = req_attempt.clone();
                                    force_req.tool_definitions = vec![];
                                    let _ = write!(
                                        force_req.user_content,
                                        "\n\n[Tool call limit reached. Based on all information gathered above, produce your final JSON response now without calling any more tools.]"
                                    );
                                    match retry_inner(
                                        backend.as_ref(),
                                        &force_req,
                                        self.inner_retries,
                                    )
                                    .await
                                    {
                                        Ok(LlmCompletion::Message(resp)) => {
                                            info!(
                                                backend = backend.name(),
                                                turns,
                                                "Forced summary pass succeeded after max tool turns"
                                            );
                                            metrics::counter!(crate::telemetry::LLM_REQUESTS, "backend" => backend.name().to_owned(), "status" => "success").increment(1);
                                            metrics::histogram!(crate::telemetry::LLM_DURATION, "backend" => backend.name().to_owned()).record(start.elapsed().as_secs_f64());
                                            return LlmOutcome::Success {
                                                response: append_missing_source_links(
                                                    resp,
                                                    &tool_source_urls,
                                                ),
                                                helpers,
                                                tool_calls_made,
                                            };
                                        }
                                        _ => {
                                            warn!(
                                                backend = backend.name(),
                                                "Forced summary pass failed, falling through to next attempt"
                                            );
                                        }
                                    }
                                    fallback_source_urls.clone_from(&tool_source_urls);
                                    fallback_tool_results.clone_from(&accumulated_tool_results);
                                    break;
                                }
                                let llm_call_names: Vec<String> =
                                    llm_calls.iter().map(|c| c.name.clone()).collect();
                                for llm_call in &llm_calls {
                                    let (result, sub_produced_by) =
                                        self.execute_llm_tool_call(llm_call, &req_attempt).await;
                                    let _ = write!(
                                        req_attempt.user_content,
                                        "\n\ntool `llm_call` result: {result}"
                                    );
                                    accumulated_tool_results.push(("llm_call".to_owned(), result));
                                    if !sub_produced_by.is_empty()
                                        && !helpers.contains(&sub_produced_by)
                                    {
                                        helpers.push(sub_produced_by);
                                    }
                                    tool_calls_made += 1;
                                }
                                turns += 1;
                                if let Some(tx) = &req_attempt.progress_tx {
                                    let _ = tx.send(LlmTurnProgress {
                                        turn: turns,
                                        max_turns: self.max_tool_turns,
                                        tools_called: llm_call_names,
                                    });
                                }
                                let remaining = self.max_tool_turns.saturating_sub(turns);
                                if remaining > 0 && remaining <= self.max_tool_turns / 2 {
                                    let _ = write!(
                                        req_attempt.user_content,
                                        "\n\n[Tool budget: {remaining} turn(s) remaining. Prefer to consolidate and produce a final answer if you have enough information.]"
                                    );
                                }
                                if calls.is_empty() {
                                    continue;
                                }
                            }

                            if calls.is_empty() {
                                warn!(
                                    backend = backend.name(),
                                    "LLM returned empty tool call list"
                                );
                                fallback_source_urls.clone_from(&tool_source_urls);
                                fallback_tool_results.clone_from(&accumulated_tool_results);
                                break;
                            }

                            if turns >= self.max_tool_turns {
                                warn!(
                                    backend = backend.name(),
                                    max_turns = self.max_tool_turns,
                                    "Max tool turns reached, attempting forced summary"
                                );
                                let mut force_req = req_attempt.clone();
                                force_req.tool_definitions = vec![];
                                let _ = write!(
                                    force_req.user_content,
                                    "\n\n[Tool call limit reached. Based on all information gathered above, produce your final JSON response now without calling any more tools.]"
                                );
                                match retry_inner(backend.as_ref(), &force_req, self.inner_retries)
                                    .await
                                {
                                    Ok(LlmCompletion::Message(resp)) => {
                                        info!(
                                            backend = backend.name(),
                                            turns,
                                            "Forced summary pass succeeded after max tool turns"
                                        );
                                        metrics::counter!(crate::telemetry::LLM_REQUESTS, "backend" => backend.name().to_owned(), "status" => "success").increment(1);
                                        metrics::histogram!(crate::telemetry::LLM_DURATION, "backend" => backend.name().to_owned()).record(start.elapsed().as_secs_f64());
                                        return LlmOutcome::Success {
                                            response: append_missing_source_links(
                                                resp,
                                                &tool_source_urls,
                                            ),
                                            helpers,
                                            tool_calls_made,
                                        };
                                    }
                                    _ => {
                                        warn!(
                                            backend = backend.name(),
                                            "Forced summary pass failed, falling through to next attempt"
                                        );
                                    }
                                }
                                // Update fallback before breaking
                                fallback_source_urls.clone_from(&tool_source_urls);
                                fallback_tool_results.clone_from(&accumulated_tool_results);
                                break;
                            }
                            let Some(executor) = &self.tool_executor else {
                                warn!(
                                    backend = backend.name(),
                                    "Tool call requested but no executor configured"
                                );
                                fallback_source_urls.clone_from(&tool_source_urls);
                                fallback_tool_results.clone_from(&accumulated_tool_results);
                                break;
                            };

                            let tool_names: Vec<String> =
                                calls.iter().map(|c| c.name.clone()).collect();
                            tool_calls_made += calls.len();
                            let output = execute_tool_calls(
                                executor,
                                &calls,
                                &req_attempt,
                                self.tool_result_max_chars,
                            )
                            .await;
                            for url in output.source_urls {
                                if tool_source_url_set.insert(url.clone()) {
                                    tool_source_urls.push(url);
                                }
                            }
                            req_attempt
                                .user_content
                                .push_str("\n\n--- Tool execution results ---\n");
                            req_attempt.user_content.push_str(&output.text);
                            accumulated_tool_results.extend(output.named_results);
                            req_attempt.require_initial_tool_call = false;
                            turns += 1;
                            if let Some(tx) = &req_attempt.progress_tx {
                                let _ = tx.send(LlmTurnProgress {
                                    turn: turns,
                                    max_turns: self.max_tool_turns,
                                    tools_called: tool_names,
                                });
                            }
                            let remaining = self.max_tool_turns.saturating_sub(turns);
                            if remaining > 0 && remaining <= self.max_tool_turns / 2 {
                                let _ = write!(
                                    req_attempt.user_content,
                                    "\n\n[Tool budget: {remaining} turn(s) remaining. Prefer to consolidate and produce a final answer if you have enough information.]"
                                );
                            }
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
                            fallback_source_urls.clone_from(&tool_source_urls);
                            fallback_tool_results.clone_from(&accumulated_tool_results);
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
            FallbackMode::Raw => LlmOutcome::RawFallback {
                source_urls: fallback_source_urls,
                tool_results: fallback_tool_results,
                helpers,
                tool_calls_made,
            },
            FallbackMode::Discard => LlmOutcome::Discard,
        }
    }

    #[must_use]
    pub fn max_tool_turns(&self) -> usize {
        self.max_tool_turns
    }

    /// One-shot text completion with no tools and no JSON structure.
    /// Returns the text and the `backend:model` identifier that produced it,
    /// or `None` if all backends fail or return empty text.
    pub async fn complete_text(&self, system: &str, user: &str) -> Option<(String, String)> {
        let req = LlmRequest::simple(system, user);
        for backend in &self.backends {
            match backend.complete_raw(req.clone()).await {
                Ok((text, produced_by)) => {
                    let trimmed = text.trim().to_owned();
                    if !trimmed.is_empty() {
                        return Some((trimmed, produced_by));
                    }
                }
                Err(e) => {
                    warn!(?e, backend = backend.name(), "complete_text backend failed");
                }
            }
        }
        None
    }

    /// Returns the sub-call's textual result together with the `backend:model`
    /// that produced it (empty string when all backends failed).
    async fn execute_llm_tool_call(
        &self,
        call: &ToolCall,
        parent_req: &LlmRequest,
    ) -> (String, String) {
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
            progress_tx: None,
            source_name: parent_req.source_name.clone(),
        };

        for backend in &self.backends {
            for attempt in 0..=self.inner_retries {
                if attempt > 0 {
                    let delay_ms = 500u64.saturating_mul(2u64.pow(attempt - 1));
                    tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
                }
                match backend.complete_raw(sub_req.clone()).await {
                    Ok(pair) => return pair,
                    Err(e) => {
                        warn!(
                            ?e,
                            backend = backend.name(),
                            attempt,
                            "llm_call sub-request retry"
                        );
                    }
                }
            }
        }

        (
            "llm_call failed: all backends exhausted".into(),
            String::new(),
        )
    }
}
