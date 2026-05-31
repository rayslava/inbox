//! Tool-call turn handling for `LlmChain`: partitioning a batch into
//! `activate_thinking` / `llm_call` / regular tools, dispatching each, and
//! folding results back into the attempt. Split out of `chain.rs` to keep that
//! file focused on the backend/attempt loop.

use std::fmt::Write as _;

use tracing::{info, warn};

use super::super::chain_tools::execute_tool_calls;
use super::super::{LlmRequest, LlmTurnProgress, ToolCall};
use super::{AttemptState, BackendCtx, ChainRunState, LlmChain, TurnAction};

impl LlmChain {
    pub(super) async fn handle_tool_calls_turn(
        &self,
        calls: Vec<ToolCall>,
        req_attempt: &mut LlmRequest,
        state: &mut AttemptState,
        run_state: &mut ChainRunState,
        ctx: BackendCtx<'_>,
    ) -> TurnAction {
        let (thinking_calls, calls): (Vec<_>, Vec<_>) = calls
            .into_iter()
            .partition(|c| c.name == "activate_thinking");
        if !thinking_calls.is_empty()
            && let Some(action) = self.handle_thinking_activation(&calls, req_attempt, state, ctx)
        {
            return action;
        }

        let (llm_calls, calls): (Vec<_>, Vec<_>) =
            calls.into_iter().partition(|c| c.name == "llm_call");
        if !llm_calls.is_empty()
            && let Some(action) = self
                .handle_llm_subcalls(&llm_calls, &calls, req_attempt, state, run_state, ctx)
                .await
        {
            return action;
        }

        if calls.is_empty() {
            warn!(
                backend = ctx.backend.name(),
                "LLM returned empty tool call list"
            );
            return TurnAction::Break;
        }

        if state.turns >= self.max_tool_turns {
            return match self
                .force_summary_pass(ctx.backend, req_attempt, state.turns, ctx.start)
                .await
            {
                Some(resp) => TurnAction::Done(resp),
                None => TurnAction::Break,
            };
        }

        let Some(executor) = &self.tool_executor else {
            warn!(
                backend = ctx.backend.name(),
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

    /// React to one or more `activate_thinking` calls in the batch. Returns
    /// `Some` only when the chain should short-circuit the rest of this turn
    /// (regular tool calls also pending → `None` so the caller falls through).
    fn handle_thinking_activation(
        &self,
        remaining_calls: &[ToolCall],
        req_attempt: &mut LlmRequest,
        state: &mut AttemptState,
        ctx: BackendCtx<'_>,
    ) -> Option<TurnAction> {
        if req_attempt.think.is_none() {
            info!(backend = ctx.backend.name(), "LLM activated thinking mode");
            req_attempt.think = Some(true);
        }
        state.thinking_activations += 1;
        if !remaining_calls.is_empty() {
            return None;
        }
        if state.thinking_activations >= self.max_tool_turns {
            warn!(
                backend = ctx.backend.name(),
                max = self.max_tool_turns,
                "activate_thinking loop limit reached"
            );
            return Some(TurnAction::Break);
        }
        Some(TurnAction::Continue)
    }

    /// Drive `llm_call` sub-requests. Returns `Some` only when the chain
    /// should short-circuit (turn budget already exhausted, or every call in
    /// the batch was `llm_call` and there are no regular tool calls left).
    async fn handle_llm_subcalls(
        &self,
        llm_calls: &[ToolCall],
        remaining_calls: &[ToolCall],
        req_attempt: &mut LlmRequest,
        state: &mut AttemptState,
        run_state: &mut ChainRunState,
        ctx: BackendCtx<'_>,
    ) -> Option<TurnAction> {
        if state.turns >= self.max_tool_turns {
            warn!(
                backend = ctx.backend.name(),
                max_turns = self.max_tool_turns,
                "Max tool turns reached during llm_call"
            );
            return Some(
                match self
                    .force_summary_pass(ctx.backend, req_attempt, state.turns, ctx.start)
                    .await
                {
                    Some(resp) => TurnAction::Done(resp),
                    None => TurnAction::Break,
                },
            );
        }
        self.dispatch_llm_subcalls(llm_calls, req_attempt, state, run_state)
            .await;
        if remaining_calls.is_empty() {
            return Some(TurnAction::Continue);
        }
        None
    }

    /// Run every `llm_call` in the batch and append its result back into the
    /// parent request's user content. Updates helper-model deduplication, the
    /// progress channel, and the tool-budget hint.
    async fn dispatch_llm_subcalls(
        &self,
        llm_calls: &[ToolCall],
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
}
