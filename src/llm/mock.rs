use async_trait::async_trait;

use crate::error::InboxError;
use crate::message::LlmResponse;

use super::{LlmClient, LlmCompletion, LlmRequest};

pub struct MockLlm {
    pub response: LlmResponse,
    pub name: String,
}

impl MockLlm {
    #[must_use]
    pub fn new(response: LlmResponse) -> Self {
        Self {
            response,
            name: "mock".into(),
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
        Ok(LlmCompletion::Message(self.response.clone()))
    }
}
