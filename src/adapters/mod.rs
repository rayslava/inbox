use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::warn;

use crate::error::InboxError;
use crate::message::IncomingMessage;

pub mod email;
pub mod http;
pub mod reconnect;
pub mod telegram;
pub(crate) mod telegram_media_group;
pub mod telegram_notifier;

#[cfg(test)]
mod tests;

#[async_trait]
pub trait InputAdapter: Send + Sync + 'static {
    fn name(&self) -> &'static str;

    async fn run(
        self: Box<Self>,
        tx: mpsc::Sender<IncomingMessage>,
        shutdown: CancellationToken,
    ) -> Result<(), InboxError>;
}

/// Build the list of `InputAdapter` instances enabled by `cfg`.
///
/// `telegram_shared` is injected so the resume notifier and the live adapter
/// share the same retry/feedback maps.
#[must_use]
pub fn build_enabled(
    cfg: &crate::config::Config,
    memory_store: Option<&Arc<crate::memory::MemoryStore>>,
    telegram_shared: telegram::TelegramShared,
) -> Vec<Box<dyn InputAdapter>> {
    let mut adapters: Vec<Box<dyn InputAdapter>> = Vec::new();
    if cfg.adapters.http.enabled {
        adapters.push(Box::new(http::HttpAdapter {
            cfg: cfg.adapters.http.clone(),
            attachments_dir: cfg.general.attachments_dir.clone(),
        }));
    }
    if cfg.adapters.telegram.enabled {
        adapters.push(Box::new(telegram::TelegramAdapter {
            cfg: cfg.adapters.telegram.clone(),
            attachments_dir: cfg.general.attachments_dir.clone(),
            memory_store: memory_store.cloned(),
            shared: telegram_shared,
        }));
    }
    if cfg.adapters.email.enabled {
        adapters.push(Box::new(email::EmailAdapter {
            cfg: cfg.adapters.email.clone(),
            attachments_dir: cfg.general.attachments_dir.clone(),
        }));
    }
    adapters
}

/// Fan every adapter in `adapters` onto its own `tokio::spawn`, logging any
/// terminal error per adapter. The function returns immediately; each
/// adapter runs until its own internal shutdown logic completes.
pub fn spawn_all(
    adapters: Vec<Box<dyn InputAdapter>>,
    tx: &mpsc::Sender<IncomingMessage>,
    shutdown: &CancellationToken,
) {
    for adapter in adapters {
        let tx = tx.clone();
        let sd = shutdown.clone();
        let name = adapter.name();
        tokio::spawn(async move {
            if let Err(e) = adapter.run(tx, sd).await {
                warn!(?e, adapter = name, "adapter exited with error");
            }
        });
    }
}
