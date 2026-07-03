//! `inbox-core` — dependency-light domain types and trait boundaries shared
//! across the inbox daemon and downstream crates (`kb-web`, `omi-bridge`).
//!
//! This crate must stay free of heavy transport/storage deps (reqwest, grafeo,
//! axum, teloxide, sqlx, …). Adapters live in the `inbox` binary and implement
//! the traits declared here, mapping their concrete errors into [`CoreError`].

pub mod brain;
pub mod embedding;
pub mod error;
pub mod fetch;
pub mod llm;
pub mod message;
pub mod output;
pub mod status;
pub mod url_content;
pub mod vector;

pub use embedding::EmbeddingProvider;
pub use error::CoreError;
pub use fetch::UrlFetcher;
pub use llm::LlmBackend;
pub use message::{IncomingMessage, ProcessedMessage};
pub use output::{OutputTarget, OutputWriter};
pub use status::{NoopNotifier, ProcessingStage, StatusNotifier};
pub use url_content::UrlContent;
pub use vector::{MemoryEntry, SourceEntry, VectorStore};

#[cfg(test)]
mod tests;

/// Identity tag for the shared core API surface, exercised by downstream crates
/// to anchor the dependency on `inbox-core` (and bump per Phase).
#[must_use]
pub fn api_tag() -> &'static str {
    concat!("inbox-core/", env!("CARGO_PKG_VERSION"), " (phase0)")
}
