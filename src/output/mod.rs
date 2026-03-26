use async_trait::async_trait;

use crate::config::Config;
use crate::error::InboxError;
use crate::message::ProcessedMessage;

pub mod org_file;

#[async_trait]
pub trait OutputWriter: Send + Sync + 'static {
    async fn write(&self, msg: &ProcessedMessage, cfg: &Config) -> Result<(), InboxError>;
}

/// A no-op writer for tests that never writes output.
#[cfg(any(test, feature = "test-helpers"))]
pub struct NullWriter;

#[cfg(any(test, feature = "test-helpers"))]
#[async_trait]
impl OutputWriter for NullWriter {
    async fn write(&self, _msg: &ProcessedMessage, _cfg: &Config) -> Result<(), InboxError> {
        Ok(())
    }
}
