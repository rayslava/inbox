use std::collections::HashSet;
use std::fmt::Write as _;

use anodized::spec;
use tracing::{debug, info, warn};

use crate::error::InboxError;

use super::chain_tools::{append_missing_source_links, execute_tool_calls, retry_inner};

mod methods;
#[cfg(test)]
mod methods_tests;

use super::{
    FallbackMode, LlmClient, LlmCompletion, LlmOutcome, LlmRequest, LlmTurnProgress,
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

/// Cross-attempt state that survives until the chain produces a final outcome.
/// The fallback URLs / tool results carry over from the most recent attempt
/// because they are the best context the chain has if every backend gives up.
#[derive(Default)]
struct ChainRunState {
    fallback_source_urls: Vec<String>,
    fallback_tool_results: Vec<(String, String)>,
    helpers: Vec<String>,
    tool_calls_made: usize,
}

/// Per-attempt scratch state — turn counter, thinking-mode counter, accumulated
/// tool results within this single attempt at this single backend.
#[derive(Default)]
struct AttemptState {
    turns: usize,
    thinking_activations: usize,
    required_tool_prompts: usize,
    tool_source_url_set: HashSet<String>,
    tool_source_urls: Vec<String>,
    accumulated_tool_results: Vec<(String, String)>,
}

/// Outcome of a single attempt at a single backend.
enum AttemptOutcome {
    /// Bubble straight up — the chain is done.
    Success(LlmOutcome),
    /// Try the next attempt on the same backend.
    Soft,
    /// Skip the rest of this backend's retry budget — same input would fail
    /// the same way (e.g. JSON parse error against a deterministic model).
    Deterministic,
}

/// Decision after the chain handles one turn's response from a backend.
enum TurnAction {
    /// Loop again on the same attempt (more tools to dispatch, or thinking
    /// activated without follow-up tools).
    Continue,
    /// Final response is ready; bubble up.
    Done(crate::message::LlmResponse),
    /// Stop the inner turn loop and roll into the next attempt / backend.
    Break,
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
        let tool_defs = self.build_tool_definitions(&req);
        let mut state = ChainRunState::default();

        for backend in &self.backends {
            let mut backend_giving_up = false;
            for attempt in 0..backend.retries() {
                let outcome = self
                    .run_attempt(backend.as_ref(), &req, &tool_defs, attempt, &mut state)
                    .await;
                match outcome {
                    AttemptOutcome::Success(o) => return o,
                    AttemptOutcome::Deterministic => {
                        backend_giving_up = true;
                        break;
                    }
                    AttemptOutcome::Soft => {}
                }
            }
            if backend_giving_up {
                warn!(
                    backend = backend.name(),
                    model = backend.model(),
                    "Deterministic LLM failure — skipping remaining retries for this backend"
                );
            } else {
                warn!(
                    backend = backend.name(),
                    model = backend.model(),
                    retries = backend.retries(),
                    "LLM backend exhausted all retries"
                );
            }
        }

        warn!(
            backend_count = self.backends.len(),
            "All LLM backends failed, applying fallback"
        );
        self.apply_fallback(state)
    }

    /// Build the tool-definitions list that the request will carry into every
    /// attempt. Conditionally includes `activate_thinking` and `llm_call` based
    /// on backend capability and the recursion budget.
    fn build_tool_definitions(&self, req: &LlmRequest) -> Vec<serde_json::Value> {
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
        tool_defs
    }

    /// One attempt = the inner turn loop against a single backend with a single
    /// retry slot. Mutates `run_state` so the chain-level fallback context
    /// reflects the most recent attempt.
    async fn run_attempt(
        &self,
        backend: &(dyn LlmClient + 'static),
        req: &LlmRequest,
        tool_defs: &[serde_json::Value],
        attempt: u32,
        run_state: &mut ChainRunState,
    ) -> AttemptOutcome {
        let start = std::time::Instant::now();
        let mut req_attempt = req.clone();
        req_attempt.tool_definitions = tool_defs.to_vec();
        let mut state = AttemptState::default();

        loop {
            log_turn(&req_attempt, backend, state.turns);

            match retry_inner(backend, &req_attempt, self.inner_retries).await {
                Ok(LlmCompletion::Message(resp)) => {
                    if let Some(action) =
                        handle_message_turn(&mut req_attempt, tool_defs, &mut state, backend)
                    {
                        match action {
                            TurnAction::Continue => continue,
                            TurnAction::Break => break,
                            TurnAction::Done(_) => {
                                unreachable!("handle_message_turn never returns Done")
                            }
                        }
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
                    return AttemptOutcome::Success(LlmOutcome::Success {
                        response: append_missing_source_links(resp, &state.tool_source_urls),
                        helpers: std::mem::take(&mut run_state.helpers),
                        tool_calls_made: run_state.tool_calls_made,
                    });
                }
                Ok(LlmCompletion::ToolCalls(calls)) => {
                    match self
                        .handle_tool_calls_turn(
                            calls,
                            &mut req_attempt,
                            &mut state,
                            run_state,
                            backend,
                            start,
                        )
                        .await
                    {
                        TurnAction::Continue => {}
                        TurnAction::Done(resp) => {
                            return AttemptOutcome::Success(LlmOutcome::Success {
                                response: append_missing_source_links(
                                    resp,
                                    &state.tool_source_urls,
                                ),
                                helpers: std::mem::take(&mut run_state.helpers),
                                tool_calls_made: run_state.tool_calls_made,
                            });
                        }
                        TurnAction::Break => break,
                    }
                }
                Err(e) => {
                    let elapsed_ms = start.elapsed().as_millis();
                    let deterministic = is_deterministic_error(&e);
                    warn!(
                        ?e,
                        backend = backend.name(),
                        model = backend.model(),
                        attempt = attempt + 1,
                        total_attempts = backend.retries(),
                        elapsed_ms,
                        deterministic,
                        "LLM attempt failed"
                    );
                    commit_fallback_state(&state, run_state);
                    metrics::counter!(
                        crate::telemetry::LLM_REQUESTS,
                        "backend" => backend.name().to_owned(),
                        "status" => "failure"
                    )
                    .increment(1);
                    return if deterministic {
                        AttemptOutcome::Deterministic
                    } else {
                        AttemptOutcome::Soft
                    };
                }
            }
        }

        // Inner loop broke without producing a result.
        commit_fallback_state(&state, run_state);
        metrics::counter!(
            crate::telemetry::LLM_REQUESTS,
            "backend" => backend.name().to_owned(),
            "status" => "failure"
        )
        .increment(1);
        AttemptOutcome::Soft
    }

    /// Drive a tool-call turn end-to-end: handle `activate_thinking` /
    /// `llm_call` partitions, dispatch the remaining tool calls through the
    /// executor, and either continue the loop or surface a forced-summary
    /// success.
    async fn handle_tool_calls_turn(
        &self,
        calls: Vec<super::ToolCall>,
        req_attempt: &mut LlmRequest,
        state: &mut AttemptState,
        run_state: &mut ChainRunState,
        backend: &(dyn LlmClient + 'static),
        start: std::time::Instant,
    ) -> TurnAction {
        let (thinking_calls, calls): (Vec<_>, Vec<_>) = calls
            .into_iter()
            .partition(|c| c.name == "activate_thinking");

        if !thinking_calls.is_empty() {
            if req_attempt.think.is_none() {
                info!(backend = backend.name(), "LLM activated thinking mode");
                req_attempt.think = Some(true);
            }
            state.thinking_activations += 1;
            if calls.is_empty() {
                if state.thinking_activations >= self.max_tool_turns {
                    warn!(
                        backend = backend.name(),
                        max = self.max_tool_turns,
                        "activate_thinking loop limit reached"
                    );
                    return TurnAction::Break;
                }
                return TurnAction::Continue;
            }
        }

        let (llm_calls, calls): (Vec<_>, Vec<_>) =
            calls.into_iter().partition(|c| c.name == "llm_call");

        if !llm_calls.is_empty() {
            if state.turns >= self.max_tool_turns {
                warn!(
                    backend = backend.name(),
                    max_turns = self.max_tool_turns,
                    "Max tool turns reached during llm_call"
                );
                if let Some(resp) = self
                    .force_summary_pass(backend, req_attempt, state.turns, start)
                    .await
                {
                    return TurnAction::Done(resp);
                }
                return TurnAction::Break;
            }
            self.dispatch_llm_subcalls(&llm_calls, req_attempt, state, run_state)
                .await;
            if calls.is_empty() {
                return TurnAction::Continue;
            }
        }

        if calls.is_empty() {
            warn!(
                backend = backend.name(),
                "LLM returned empty tool call list"
            );
            return TurnAction::Break;
        }

        if state.turns >= self.max_tool_turns {
            if let Some(resp) = self
                .force_summary_pass(backend, req_attempt, state.turns, start)
                .await
            {
                return TurnAction::Done(resp);
            }
            return TurnAction::Break;
        }

        let Some(executor) = &self.tool_executor else {
            warn!(
                backend = backend.name(),
                "Tool call requested but no executor configured"
            );
            return TurnAction::Break;
        };

        let tool_names: Vec<String> = calls.iter().map(|c| c.name.clone()).collect();
        run_state.tool_calls_made += calls.len();
        let output =
            execute_tool_calls(executor, &calls, req_attempt, self.tool_result_max_chars).await;
        for url in output.source_urls {
            if state.tool_source_url_set.insert(url.clone()) {
                state.tool_source_urls.push(url);
            }
        }
        req_attempt
            .user_content
            .push_str("\n\n--- Tool execution results ---\n");
        req_attempt.user_content.push_str(&output.text);
        state.accumulated_tool_results.extend(output.named_results);
        req_attempt.require_initial_tool_call = false;
        state.turns += 1;
        if let Some(tx) = &req_attempt.progress_tx {
            let _ = tx.send(LlmTurnProgress {
                turn: state.turns,
                max_turns: self.max_tool_turns,
                tools_called: tool_names,
            });
        }
        self.append_budget_hint(req_attempt, state.turns);
        TurnAction::Continue
    }

    /// Run every `llm_call` in the batch and append its result back into the
    /// parent request's user content. Updates helper-model deduplication, the
    /// progress channel, and the tool-budget hint.
    async fn dispatch_llm_subcalls(
        &self,
        llm_calls: &[super::ToolCall],
        req_attempt: &mut LlmRequest,
        state: &mut AttemptState,
        run_state: &mut ChainRunState,
    ) {
        let llm_call_names: Vec<String> = llm_calls.iter().map(|c| c.name.clone()).collect();
        for llm_call in llm_calls {
            let (result, sub_produced_by) = self.execute_llm_tool_call(llm_call, req_attempt).await;
            let _ = write!(
                req_attempt.user_content,
                "\n\ntool `llm_call` result: {result}"
            );
            state
                .accumulated_tool_results
                .push(("llm_call".to_owned(), result));
            if !sub_produced_by.is_empty() && !run_state.helpers.contains(&sub_produced_by) {
                run_state.helpers.push(sub_produced_by);
            }
            run_state.tool_calls_made += 1;
        }
        state.turns += 1;
        if let Some(tx) = &req_attempt.progress_tx {
            let _ = tx.send(LlmTurnProgress {
                turn: state.turns,
                max_turns: self.max_tool_turns,
                tools_called: llm_call_names,
            });
        }
        self.append_budget_hint(req_attempt, state.turns);
    }

    /// Pick the configured fallback policy and consume the run-level state into
    /// the matching `LlmOutcome` variant.
    fn apply_fallback(&self, state: ChainRunState) -> LlmOutcome {
        match self.fallback {
            FallbackMode::Raw => LlmOutcome::RawFallback {
                source_urls: state.fallback_source_urls,
                tool_results: state.fallback_tool_results,
                helpers: state.helpers,
                tool_calls_made: state.tool_calls_made,
            },
            FallbackMode::Discard => LlmOutcome::Discard,
        }
    }

    /// Final pass once the tool-turn budget is exhausted: re-issue the request
    /// with tools disabled and instruct the model to emit its final JSON from
    /// the context gathered so far. Returns the parsed response on success, or
    /// `None` if it still fails (the caller then applies the fallback policy).
    async fn force_summary_pass(
        &self,
        backend: &(dyn LlmClient + 'static),
        req_attempt: &LlmRequest,
        turns: usize,
        start: std::time::Instant,
    ) -> Option<crate::message::LlmResponse> {
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
        if let Ok(LlmCompletion::Message(resp)) =
            retry_inner(backend, &force_req, self.inner_retries).await
        {
            info!(
                backend = backend.name(),
                turns, "Forced summary pass succeeded after max tool turns"
            );
            metrics::counter!(crate::telemetry::LLM_REQUESTS, "backend" => backend.name().to_owned(), "status" => "success").increment(1);
            metrics::histogram!(crate::telemetry::LLM_DURATION, "backend" => backend.name().to_owned()).record(start.elapsed().as_secs_f64());
            Some(resp)
        } else {
            warn!(
                backend = backend.name(),
                "Forced summary pass failed, falling through to next attempt"
            );
            None
        }
    }

    /// Append a "tool budget remaining" nudge once the loop is at or past the
    /// halfway point of `max_tool_turns`, steering the model toward consolidating
    /// rather than spending its last turns on more tool calls.
    fn append_budget_hint(&self, req_attempt: &mut LlmRequest, turns: usize) {
        let remaining = self.max_tool_turns.saturating_sub(turns);
        if remaining > 0 && remaining <= self.max_tool_turns / 2 {
            let _ = write!(
                req_attempt.user_content,
                "\n\n[Tool budget: {remaining} turn(s) remaining. Prefer to consolidate and produce a final answer if you have enough information.]"
            );
        }
    }

    #[must_use]
    pub fn max_tool_turns(&self) -> usize {
        self.max_tool_turns
    }
}

/// Decide what to do with a `LlmCompletion::Message` mid-attempt. Returns
/// `Some(action)` if the chain should keep looping or stop early; `None`
/// means the caller should treat the message as the final response.
fn handle_message_turn(
    req_attempt: &mut LlmRequest,
    tool_defs: &[serde_json::Value],
    state: &mut AttemptState,
    backend: &(dyn LlmClient + 'static),
) -> Option<TurnAction> {
    if !(req_attempt.require_initial_tool_call && state.turns == 0 && !tool_defs.is_empty()) {
        return None;
    }
    if state.required_tool_prompts < 3 {
        debug!(
            backend = backend.name(),
            prompt_attempt = state.required_tool_prompts + 1,
            "Re-prompting model to make required initial tool call"
        );
        req_attempt.user_content.push_str(
            "\n\nA tool call is required before final JSON because URLs are present. First analyze and call exactly one best retrieval tool, then continue.",
        );
        state.required_tool_prompts += 1;
        return Some(TurnAction::Continue);
    }
    warn!(
        backend = backend.name(),
        "Required initial tool call was not produced"
    );
    Some(TurnAction::Break)
}

/// Snapshot the attempt's source URLs and tool results into the run-level
/// fallback context, overwriting whatever was there from a previous attempt.
fn commit_fallback_state(state: &AttemptState, run_state: &mut ChainRunState) {
    run_state
        .fallback_source_urls
        .clone_from(&state.tool_source_urls);
    run_state
        .fallback_tool_results
        .clone_from(&state.accumulated_tool_results);
}

/// Emit a debug record describing the about-to-be-sent request. Pulled out so
/// `run_attempt`'s control flow stays uncluttered.
fn log_turn(req: &LlmRequest, backend: &(dyn LlmClient + 'static), turns: usize) {
    let tool_names_debug: Vec<&str> = req
        .tool_definitions
        .iter()
        .filter_map(|d| d["function"]["name"].as_str())
        .collect();
    let system_preview: String = req.system_prompt.chars().take(300).collect();
    let content_preview: String = req.user_content.chars().take(600).collect();
    debug!(
        backend = backend.name(),
        model = backend.model(),
        turn = turns + 1,
        tools = ?tool_names_debug,
        system_len = req.system_prompt.len(),
        content_len = req.user_content.len(),
        system_preview = %system_preview,
        content_preview = %content_preview,
        "LLM request"
    );
}

/// Errors that will recur identically on a retry of the *same* backend, so
/// burning the remaining retry budget on them is wasted time. A JSON parse
/// failure means the model produced unparseable output (e.g. Markdown prose);
/// re-running the same model with the same prompt yields the same result, often
/// after minutes of slow local inference each time.
pub(super) fn is_deterministic_error(err: &InboxError) -> bool {
    let InboxError::Llm(msg) = err else {
        return false;
    };
    msg.contains("JSON parse error")
}
