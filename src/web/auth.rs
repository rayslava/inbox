use std::sync::Arc;

use anodized::contract;
use axum::http::{HeaderMap, header};
use chrono::{DateTime, Duration, Utc};
use dashmap::DashMap;
use serde::Deserialize;

pub type SessionStore = DashMap<String, DateTime<Utc>>;

#[must_use]
pub fn new_session_store() -> Arc<SessionStore> {
    Arc::new(DashMap::new())
}

/// Extract the raw session token from the Cookie header.
pub fn extract_session_token(headers: &HeaderMap) -> Option<String> {
    let cookie = headers.get(header::COOKIE)?.to_str().ok()?;
    for part in cookie.split(';') {
        let part = part.trim();
        if let Some(token) = part.strip_prefix("session=") {
            return Some(token.to_owned());
        }
    }
    None
}

/// Return true if the session cookie is valid and not expired.
#[must_use]
#[contract(requires: ttl_days > 0)]
pub fn is_authenticated(headers: &HeaderMap, store: &SessionStore, ttl_days: u64) -> bool {
    let Some(token) = extract_session_token(headers) else {
        return false;
    };
    let Some(entry) = store.get(&token) else {
        return false;
    };
    let age = Utc::now() - *entry;
    let ttl = Duration::days(i64::try_from(ttl_days).unwrap_or(365 * 10));
    let valid = age < ttl;
    if !valid {
        drop(entry);
        store.remove(&token);
    }
    valid
}

/// Verify a plain-text password against a stored Argon2id hash.
#[must_use]
pub fn verify_password(stored_hash: &str, password: &str) -> bool {
    use argon2::{Argon2, PasswordHash, PasswordVerifier};
    let Ok(parsed) = PasswordHash::new(stored_hash) else {
        return false;
    };
    Argon2::default()
        .verify_password(password.as_bytes(), &parsed)
        .is_ok()
}

/// Generate a cryptographically random 32-byte hex session token.
#[must_use]
pub fn generate_session_token() -> String {
    use rand::RngExt;
    let bytes: [u8; 32] = rand::rng().random();
    bytes.iter().fold(String::with_capacity(64), |mut s, b| {
        use std::fmt::Write as _;
        let _ = write!(s, "{b:02x}");
        s
    })
}

#[derive(Deserialize)]
pub struct LoginForm {
    pub username: String,
    pub password: String,
}
