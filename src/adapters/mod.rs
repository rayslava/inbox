use async_trait::async_trait;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::error::InboxError;
use crate::message::IncomingMessage;

pub mod email;
pub mod http;
pub mod telegram;
pub(crate) mod telegram_media_group;
pub mod telegram_notifier;

#[async_trait]
pub trait InputAdapter: Send + Sync + 'static {
    fn name(&self) -> &'static str;

    async fn run(
        self: Box<Self>,
        tx: mpsc::Sender<IncomingMessage>,
        shutdown: CancellationToken,
    ) -> Result<(), InboxError>;
}
