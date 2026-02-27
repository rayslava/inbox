use async_trait::async_trait;

use crate::config::Config;
use crate::error::InboxError;
use crate::message::ProcessedMessage;

pub mod org_file;

#[async_trait]
pub trait OutputWriter: Send + Sync + 'static {
    async fn write(&self, msg: &ProcessedMessage, cfg: &Config) -> Result<(), InboxError>;
}
