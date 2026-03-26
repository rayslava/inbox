use std::net::{IpAddr, Ipv4Addr};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};
use uuid::Uuid;

const RECONNECT_BACKOFF_INIT: Duration = Duration::from_secs(1);
const RECONNECT_BACKOFF_MAX: Duration = Duration::from_secs(60);
const STABLE_THRESHOLD: Duration = Duration::from_secs(30);

use crate::adapters::telegram_media_group::{self, MediaGroupMap};
use crate::config::TelegramConfig;
use crate::error::InboxError;
use crate::memory::MemoryStore;
use crate::message::{IncomingMessage, RetryableMessage};

use super::InputAdapter;

mod files;
mod handlers;

pub(crate) type FeedbackMessageMap = Arc<DashMap<i32, Uuid>>;

pub struct TelegramAdapter {
    pub cfg: TelegramConfig,
    pub attachments_dir: std::path::PathBuf,
    pub memory_store: Option<Arc<MemoryStore>>,
}

#[async_trait::async_trait]
impl InputAdapter for TelegramAdapter {
    fn name(&self) -> &'static str {
        "telegram"
    }

    async fn run(
        self: Box<Self>,
        tx: mpsc::Sender<IncomingMessage>,
        shutdown: CancellationToken,
    ) -> Result<(), InboxError> {
        use teloxide::prelude::*;

        if self.cfg.bot_token.is_empty() {
            return Err(InboxError::Adapter("Telegram bot_token is empty".into()));
        }

        info!("Telegram adapter starting");

        let retry_store: Arc<DashMap<Uuid, RetryableMessage>> = Arc::new(DashMap::new());
        let feedback_msg_map: FeedbackMessageMap = Arc::new(DashMap::new());
        let mut backoff = RECONNECT_BACKOFF_INIT;

        loop {
            let client = crate::tls::client_builder()
                .local_address(IpAddr::V4(Ipv4Addr::UNSPECIFIED))
                .build()
                .expect("Failed to build IPv4-only Telegram client");
            let bot = Bot::with_client(&self.cfg.bot_token, client);
            let handler = build_handler(HandlerConfig {
                allowed_user_ids: self.cfg.allowed_user_ids.clone(),
                attachments_dir: self.attachments_dir.clone(),
                file_download_timeout_secs: self.cfg.file_download_timeout_secs,
                file_download_retries: self.cfg.file_download_retries,
                media_group_timeout_ms: self.cfg.media_group_timeout_ms,
                notify_cfg: crate::adapters::telegram_notifier::NotifyConfig {
                    retries: self.cfg.status_notify_retries,
                    retry_base_ms: self.cfg.status_notify_retry_base_ms,
                },
                retry_store: retry_store.clone(),
                memory_store: self.memory_store.clone(),
                feedback_msg_map: feedback_msg_map.clone(),
            });

            let mut dispatcher = Dispatcher::builder(bot, handler)
                .dependencies(dptree::deps![tx.clone()])
                .build();

            let started = Instant::now();

            let mut dispatch_task = tokio::task::spawn(async move {
                dispatcher.dispatch().await;
            });

            tokio::select! {
                () = shutdown.cancelled() => {
                    dispatch_task.abort();
                    info!("Telegram adapter shutdown");
                    return Ok(());
                }
                result = &mut dispatch_task => {
                    if let Err(ref e) = result {
                        warn!(panic = ?e, "Telegram dispatcher panicked, will reconnect");
                    }
                }
            }

            if shutdown.is_cancelled() {
                return Ok(());
            }

            if started.elapsed() >= STABLE_THRESHOLD {
                backoff = RECONNECT_BACKOFF_INIT;
            }

            metrics::counter!(
                crate::telemetry::ADAPTER_RECONNECTS,
                "adapter" => "telegram"
            )
            .increment(1);

            warn!(
                delay_secs = backoff.as_secs(),
                "Telegram dispatcher exited unexpectedly, reconnecting"
            );

            tokio::select! {
                () = shutdown.cancelled() => return Ok(()),
                () = tokio::time::sleep(backoff) => {}
            }

            backoff = (backoff * 2).min(RECONNECT_BACKOFF_MAX);
        }
    }
}

pub struct HandlerConfig {
    pub allowed_user_ids: Vec<i64>,
    pub attachments_dir: PathBuf,
    pub file_download_timeout_secs: u64,
    pub file_download_retries: u32,
    pub media_group_timeout_ms: u64,
    pub notify_cfg: crate::adapters::telegram_notifier::NotifyConfig,
    pub retry_store: Arc<DashMap<Uuid, RetryableMessage>>,
    pub memory_store: Option<Arc<MemoryStore>>,
    pub feedback_msg_map: FeedbackMessageMap,
}

#[must_use]
pub fn build_handler(
    hc: HandlerConfig,
) -> teloxide::dispatching::UpdateHandler<teloxide::RequestError> {
    use teloxide::prelude::*;

    let retry_store_msg = hc.retry_store.clone();
    let retry_store_cb = hc.retry_store;
    let media_groups: MediaGroupMap = telegram_media_group::new_map();
    let memory_store_msg = hc.memory_store.clone();
    let memory_store_cb = hc.memory_store;
    let feedback_msg_map_msg = hc.feedback_msg_map.clone();
    let feedback_msg_map_cb = hc.feedback_msg_map;
    let notify_cfg = hc.notify_cfg;

    let message_handler = Update::filter_message().endpoint(
        move |bot: Bot, msg: Message, tx: mpsc::Sender<IncomingMessage>| {
            let allowed = hc.allowed_user_ids.clone();
            let attachments_dir = hc.attachments_dir.clone();
            let retry_store = retry_store_msg.clone();
            let media_groups = media_groups.clone();
            let memory_store = memory_store_msg.clone();
            let feedback_msg_map = feedback_msg_map_msg.clone();
            async move {
                let user_id = msg
                    .from
                    .as_ref()
                    .map_or(0_i64, |u| i64::try_from(u.id.0).unwrap_or(i64::MAX));

                if !allowed.is_empty() && !allowed.contains(&user_id) {
                    return Ok(());
                }

                handlers::handle_message(
                    bot,
                    msg,
                    tx,
                    handlers::MessageContext {
                        attachments_dir,
                        dl_cfg: files::DownloadConfig {
                            timeout_secs: hc.file_download_timeout_secs,
                            retries: hc.file_download_retries,
                        },
                        media_group_timeout_ms: hc.media_group_timeout_ms,
                        notify_cfg,
                        retry_store,
                        media_groups,
                        memory_store,
                        feedback_msg_map,
                    },
                )
                .await;

                Ok::<_, teloxide::RequestError>(())
            }
        },
    );

    let callback_handler = Update::filter_callback_query().endpoint(
        move |bot: Bot,
              query: teloxide::types::CallbackQuery,
              tx: mpsc::Sender<IncomingMessage>| {
            let retry_store = retry_store_cb.clone();
            let memory_store = memory_store_cb.clone();
            let feedback_msg_map = feedback_msg_map_cb.clone();
            async move {
                handlers::handle_callback_query(
                    bot,
                    query,
                    tx,
                    retry_store,
                    notify_cfg,
                    memory_store,
                    feedback_msg_map,
                )
                .await
            }
        },
    );

    dptree::entry()
        .branch(message_handler)
        .branch(callback_handler)
}
