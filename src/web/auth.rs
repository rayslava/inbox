use std::sync::Arc;

use anodized::spec;
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
#[spec(requires: ttl_days > 0)]
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

/// Produce an Argon2id hash for `password` suitable for `admin.password_hash`.
///
/// Salt is generated automatically by the underlying `argon2` crate's
/// `getrandom` integration. The returned PHC string round-trips with
/// `verify_password`.
///
/// # Errors
/// Returns an error if Argon2 hashing fails (e.g. allocation failure).
pub fn hash_password(password: &str) -> Result<String, crate::error::InboxError> {
    use argon2::{Argon2, PasswordHasher};
    Argon2::default()
        .hash_password(password.as_bytes())
        .map(|h| h.to_string())
        .map_err(|e| crate::error::InboxError::Auth(format!("argon2 hash: {e}")))
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

#[cfg(test)]
mod tests {
    use super::{
        extract_session_token, hash_password, is_authenticated, new_session_store, verify_password,
    };
    use axum::http::{HeaderMap, HeaderValue, header};
    use chrono::{Duration, Utc};

    #[test]
    fn extract_session_token_from_cookie_header() {
        let mut headers = HeaderMap::new();
        headers.insert(
            header::COOKIE,
            HeaderValue::from_str("a=1; session=abc123; z=9").expect("header"),
        );
        assert_eq!(extract_session_token(&headers).as_deref(), Some("abc123"));
    }

    #[test]
    fn is_authenticated_removes_expired_session() {
        let store = new_session_store();
        store.insert("deadbeef".into(), Utc::now() - Duration::days(8));
        let mut headers = HeaderMap::new();
        headers.insert(
            header::COOKIE,
            HeaderValue::from_str("session=deadbeef").expect("header"),
        );
        assert!(!is_authenticated(&headers, &store, 7));
        assert!(!store.contains_key("deadbeef"));
    }

    #[test]
    fn hash_password_round_trips_through_verify() {
        let hash = hash_password("hunter2").expect("hash ok");
        assert!(verify_password(&hash, "hunter2"));
        assert!(!verify_password(&hash, "hunter3"));
    }

    #[test]
    fn hash_password_emits_phc_format() {
        let hash = hash_password("anything").expect("hash ok");
        // PHC strings start with $argon2id$ (the variant Argon2::default() picks).
        assert!(
            hash.starts_with("$argon2id$"),
            "expected argon2id PHC string, got: {hash}"
        );
    }

    #[test]
    fn hash_password_distinct_salts_per_call() {
        let a = hash_password("same").expect("hash ok");
        let b = hash_password("same").expect("hash ok");
        assert_ne!(a, b, "salts should differ per call");
    }
}
