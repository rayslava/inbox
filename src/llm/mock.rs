use async_trait::async_trait;

use crate::error::InboxError;
use crate::message::LlmResponse;

use super::{LlmClient, LlmCompletion, LlmRequest};

pub enum MockLlmBehavior {
    Success(LlmResponse),
    Fail(String),
}

pub struct MockLlm {
    pub behavior: MockLlmBehavior,
    pub name: String,
}

impl MockLlm {
    #[must_use]
    pub fn new(response: LlmResponse) -> Self {
        Self {
            behavior: MockLlmBehavior::Success(response),
            name: "mock".into(),
        }
    }

    #[must_use]
    pub fn failing(message: impl Into<String>) -> Self {
        Self {
            behavior: MockLlmBehavior::Fail(message.into()),
            name: "mock-failing".into(),
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
        1
    }
    async fn complete(&self, _req: LlmRequest) -> Result<LlmCompletion, InboxError> {
        match &self.behavior {
            MockLlmBehavior::Success(resp) => Ok(LlmCompletion::Message(resp.clone())),
            MockLlmBehavior::Fail(msg) => Err(InboxError::Llm(msg.clone())),
        }
    }
}
