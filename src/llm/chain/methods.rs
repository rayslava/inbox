//! Auxiliary `LlmChain` methods: one-shot text completion and recursive
//! `llm_call` sub-requests. Split out of `chain.rs` to keep that file focused on
//! the main tool-loop driver.

use tracing::warn;

use super::super::{LlmRequest, ToolCall};
use super::LlmChain;

impl LlmChain {
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
    pub(super) async fn execute_llm_tool_call(
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
            has_image_text: false,
            think: None,
            llm_depth: parent_req.llm_depth + 1,
            progress_tx: None,
            source_name: parent_req.source_name.clone(),
        };

        for backend in &self.backends {
            for attempt in 0..=self.inner_retries {
                if attempt > 0 {
                    tokio::time::sleep(super::super::retry_backoff(attempt)).await;
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
