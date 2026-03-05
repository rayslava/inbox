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

#[derive(Debug, Clone, Deserialize, Default)]
pub struct TelegramConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub bot_token: String,
    #[serde(default)]
    pub allowed_user_ids: Vec<i64>,
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
        }
    }
}

fn default_imap_port() -> u16 {
    993
}
fn default_mailbox() -> String {
    "INBOX".into()
}
