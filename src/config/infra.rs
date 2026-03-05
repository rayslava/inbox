use std::net::SocketAddr;
use std::path::PathBuf;

use serde::Deserialize;

// ── General ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct GeneralConfig {
    pub output_file: PathBuf,
    pub attachments_dir: PathBuf,
    #[serde(default = "default_log_level")]
    pub log_level: String,
    #[serde(default = "default_log_format")]
    pub log_format: String,
}

fn default_log_level() -> String {
    "info".into()
}
fn default_log_format() -> String {
    "pretty".into()
}

// ── Admin ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct AdminConfig {
    #[serde(default = "default_admin_bind")]
    pub bind_addr: SocketAddr,
    #[serde(default = "default_admin_user")]
    pub username: String,
    /// Argon2id hash generated via `inbox hash-password`.
    #[serde(default)]
    pub password_hash: String,
    #[serde(default = "default_session_ttl")]
    pub session_ttl_days: u64,
    #[serde(default = "default_drain_secs")]
    pub shutdown_drain_secs: u64,
}

impl Default for AdminConfig {
    fn default() -> Self {
        Self {
            bind_addr: default_admin_bind(),
            username: default_admin_user(),
            password_hash: String::new(),
            session_ttl_days: default_session_ttl(),
            shutdown_drain_secs: default_drain_secs(),
        }
    }
}

fn default_admin_bind() -> SocketAddr {
    "0.0.0.0:9090".parse().unwrap()
}
fn default_admin_user() -> String {
    "admin".into()
}
fn default_session_ttl() -> u64 {
    7
}
fn default_drain_secs() -> u64 {
    5
}

// ── Web UI ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct WebUiConfig {
    #[serde(default = "bool_true")]
    pub enabled: bool,
}

impl Default for WebUiConfig {
    fn default() -> Self {
        Self { enabled: true }
    }
}

pub(super) fn bool_true() -> bool {
    true
}

// ── URL Fetch ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct UrlFetchConfig {
    #[serde(default = "bool_true")]
    pub enabled: bool,
    #[serde(default = "default_fetch_timeout")]
    pub timeout_secs: u64,
    #[serde(default = "default_max_redirects")]
    pub max_redirects: u32,
    #[serde(default = "default_max_body_bytes")]
    pub max_body_bytes: usize,
    #[serde(default = "default_user_agent")]
    pub user_agent: String,
    #[serde(default)]
    pub skip_domains: Vec<String>,
}

impl Default for UrlFetchConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            timeout_secs: default_fetch_timeout(),
            max_redirects: default_max_redirects(),
            max_body_bytes: default_max_body_bytes(),
            user_agent: default_user_agent(),
            skip_domains: Vec::new(),
        }
    }
}

fn default_fetch_timeout() -> u64 {
    10
}
fn default_max_redirects() -> u32 {
    5
}
fn default_max_body_bytes() -> usize {
    5 * 1024 * 1024
}
fn default_user_agent() -> String {
    "inbox-bot/1.0".into()
}

// ── Syncthing ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Default, Deserialize)]
pub struct SyncthingConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_syncthing_url")]
    pub api_url: String,
    #[serde(default)]
    pub api_key: String,
    /// Syncthing folder ID that contains the org output file.
    #[serde(default)]
    pub org_folder_id: String,
    /// Syncthing folder ID for attachments (may differ from org folder).
    #[serde(default)]
    pub attachments_folder_id: Option<String>,
    #[serde(default = "bool_true")]
    pub rescan_on_write: bool,
}

fn default_syncthing_url() -> String {
    "http://localhost:8384".into()
}
