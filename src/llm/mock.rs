use std::sync::Mutex;

use async_trait::async_trait;

use crate::error::InboxError;
use crate::message::LlmResponse;

use super::{LlmClient, LlmCompletion, LlmRequest};

pub enum MockLlmBehavior {
    Success(LlmResponse),
    Fail(String),
    /// Programmable sequence of results returned in order. The last entry is
    /// repeated indefinitely once the script is exhausted, so callers do not
    /// need to over-allocate the script for tests that run extra turns.
    Script(Mutex<Vec<Result<LlmCompletion, String>>>),
}

pub struct MockLlm {
    pub behavior: MockLlmBehavior,
    pub name: String,
    pub retries: u32,
}

impl MockLlm {
    #[must_use]
    pub fn new(response: LlmResponse) -> Self {
        Self {
            behavior: MockLlmBehavior::Success(response),
            name: "mock".into(),
            retries: 1,
        }
    }

    #[must_use]
    pub fn failing(message: impl Into<String>) -> Self {
        Self {
            behavior: MockLlmBehavior::Fail(message.into()),
            name: "mock-failing".into(),
            retries: 1,
        }
    }

    /// Return a scripted sequence of completions. Each call to `complete()`
    /// pops one entry; once the script is empty, the last entry repeats.
    ///
    /// # Panics
    /// Panics if `script` is empty.
    #[must_use]
    pub fn scripted(script: Vec<Result<LlmCompletion, String>>) -> Self {
        assert!(
            !script.is_empty(),
            "scripted MockLlm needs at least one entry"
        );
        Self {
            behavior: MockLlmBehavior::Script(Mutex::new(script)),
            name: "mock-scripted".into(),
            retries: 1,
        }
    }
}

#[async_trait]
impl LlmClient for MockLlm {
    fn name(&self) -> &str {
        &self.name
    }
    fn model(&self) -> &'static str {
        "mock"
    }
    fn retries(&self) -> u32 {
        self.retries
    }
    async fn complete(&self, _req: LlmRequest) -> Result<LlmCompletion, InboxError> {
        match &self.behavior {
            MockLlmBehavior::Success(resp) => Ok(LlmCompletion::Message(resp.clone())),
            MockLlmBehavior::Fail(msg) => Err(InboxError::Llm(msg.clone())),
            MockLlmBehavior::Script(slot) => {
                let mut guard = slot.lock().expect("script lock");
                let entry = if guard.len() > 1 {
                    guard.remove(0)
                } else {
                    guard[0].clone()
                };
                entry.map_err(InboxError::Llm)
            }
        }
    }
}
