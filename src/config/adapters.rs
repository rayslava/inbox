use std::net::SocketAddr;

use serde::Deserialize;

use super::infra::bool_true;

// ── Adapters ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Default, Deserialize)]
pub struct AdaptersConfig {
    #[serde(default)]
    pub telegram: TelegramConfig,
    #[serde(default)]
    pub http: HttpAdapterConfig,
    #[serde(default)]
    pub email: EmailConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TelegramConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub bot_token: String,
    #[serde(default)]
    pub allowed_user_ids: Vec<i64>,
    #[serde(default = "default_tg_file_download_timeout_secs")]
    pub file_download_timeout_secs: u64,
    #[serde(default = "default_tg_file_download_retries")]
    pub file_download_retries: u32,
    #[serde(default = "default_tg_media_group_timeout_ms")]
    pub media_group_timeout_ms: u64,
    /// Maximum attempts to send/edit a Telegram status message (non-terminal stages).
    /// Terminal stages (Done/Failed) use twice this value.
    #[serde(default = "default_tg_status_notify_retries")]
    pub status_notify_retries: u32,
    /// Base delay (ms) for exponential backoff between status notification retries.
    /// Capped internally at 30 seconds.
    #[serde(default = "default_tg_status_notify_retry_base_ms")]
    pub status_notify_retry_base_ms: u64,
}

impl Default for TelegramConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            bot_token: String::new(),
            allowed_user_ids: Vec::new(),
            file_download_timeout_secs: default_tg_file_download_timeout_secs(),
            file_download_retries: default_tg_file_download_retries(),
            media_group_timeout_ms: default_tg_media_group_timeout_ms(),
            status_notify_retries: default_tg_status_notify_retries(),
            status_notify_retry_base_ms: default_tg_status_notify_retry_base_ms(),
        }
    }
}

fn default_tg_file_download_timeout_secs() -> u64 {
    60
}
fn default_tg_file_download_retries() -> u32 {
    3
}
fn default_tg_media_group_timeout_ms() -> u64 {
    1500
}
fn default_tg_status_notify_retries() -> u32 {
    5
}
fn default_tg_status_notify_retry_base_ms() -> u64 {
    1000
}

#[derive(Debug, Clone, Deserialize)]
pub struct HttpAdapterConfig {
    #[serde(default = "bool_true")]
    pub enabled: bool,
    #[serde(default = "default_http_bind")]
    pub bind_addr: SocketAddr,
    #[serde(default)]
    pub auth_token: String,
}

impl Default for HttpAdapterConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            bind_addr: default_http_bind(),
            auth_token: String::new(),
        }
    }
}

fn default_http_bind() -> SocketAddr {
    "0.0.0.0:8080".parse().unwrap()
}

#[derive(Debug, Clone, Deserialize)]
pub struct EmailConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub host: String,
    #[serde(default = "default_imap_port")]
    pub port: u16,
    #[serde(default)]
    pub username: String,
    #[serde(default)]
    pub password: String,
    #[serde(default = "default_mailbox")]
    pub mailbox: String,
    #[serde(default = "bool_true")]
    pub mark_as_seen: bool,
    #[serde(default)]
    pub processed_mailbox: String,
    /// Use TLS for the IMAP connection. Set `false` only for local/test servers.
    #[serde(default = "bool_true")]
    pub tls: bool,
}

impl Default for EmailConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            host: String::new(),
            port: default_imap_port(),
            username: String::new(),
            password: String::new(),
            mailbox: default_mailbox(),
            mark_as_seen: true,
            processed_mailbox: String::new(),
            tls: true,
        }
    }
}

fn default_imap_port() -> u16 {
    993
}
fn default_mailbox() -> String {
    "INBOX".into()
}
