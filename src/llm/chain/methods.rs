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

    /// One-shot vision completion: send `images` with `system`/`user` to the
    /// first vision-capable backend that *responds*, returning the (trimmed)
    /// text and the `backend:model` that produced it. An empty response is a
    /// valid answer (e.g. an image with no readable text), so it is returned
    /// rather than triggering fallback to other backends. Non-vision backends
    /// are skipped.
    ///
    /// `Err(AllUnavailable)` when at least one vision backend was eligible but
    /// every one errored/was in cooldown (a transient outage — the caller should
    /// hold the node pending and retry). `Err(NoVisionBackend)` when no vision
    /// backend was eligible at all (nothing to retry).
    pub(crate) async fn complete_vision_text(
        &self,
        system: &str,
        user: &str,
        images: Vec<(String, String)>,
    ) -> Result<(String, String), super::VisionUnavailable> {
        if images.is_empty() {
            return Err(super::VisionUnavailable::NoVisionBackend);
        }
        let mut req = LlmRequest::simple(system, user);
        req.images = images;
        let mut attempted = false;
        for backend in &self.backends {
            if super::vision::decide(&req, backend.as_ref()) != super::vision::VisionDecision::Run {
                continue;
            }
            attempted = true;
            match backend.complete_raw(req.clone()).await {
                Ok((text, produced_by)) => return Ok((text.trim().to_owned(), produced_by)),
                Err(e) => {
                    // This path bypasses the chain's `record_attempt_error`, so
                    // trip the cooldown here on a transient outage.
                    if !super::is_service_available(&e) {
                        backend.mark_unavailable();
                    }
                    warn!(
                        ?e,
                        backend = backend.name(),
                        "complete_vision_text backend failed"
                    );
                }
            }
        }
        if attempted {
            Err(super::VisionUnavailable::AllUnavailable)
        } else {
            Err(super::VisionUnavailable::NoVisionBackend)
        }
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
