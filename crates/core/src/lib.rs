//! `inbox-core` — dependency-light domain types and trait boundaries shared
//! across the inbox daemon and downstream crates (`kb-web`, `omi-bridge`).
//!
//! This crate must stay free of heavy transport/storage deps (reqwest, grafeo,
//! axum, teloxide, sqlx, …). Adapters live in the `inbox` binary and implement
//! the traits declared here, mapping their concrete errors into [`CoreError`].

pub mod error;
pub mod url_content;

pub use error::CoreError;
pub use url_content::UrlContent;

#[cfg(test)]
mod tests;

/// Identity tag for the shared core API surface, exercised by downstream crates
/// to anchor the dependency on `inbox-core` (and bump per Phase).
#[must_use]
pub fn api_tag() -> &'static str {
    concat!("inbox-core/", env!("CARGO_PKG_VERSION"), " (phase0)")
}
