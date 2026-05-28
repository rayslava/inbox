pub mod store;

#[cfg(test)]
mod tests;

pub use store::{PendingItem, PendingStats, PendingStore};

use std::sync::Arc;

use tracing::{info, warn};

/// Open the pending-items store at the configured path, or return `None` when
/// resume is disabled or the open call fails. Logs a warning on failure and
/// continues — resume is best-effort, never fatal to startup.
pub async fn open_optional(cfg: &crate::config::Config) -> Option<Arc<PendingStore>> {
    if !cfg.pipeline.resume.enabled {
        return None;
    }
    let db_path = cfg
        .pipeline
        .resume
        .db_path
        .clone()
        .unwrap_or_else(|| cfg.general.attachments_dir.join("pending.db"));
    match PendingStore::open(&db_path).await {
        Ok(s) => {
            info!(?db_path, "Pending store opened");
            Some(Arc::new(s))
        }
        Err(e) => {
            warn!(?e, "Failed to open pending store — resume disabled");
            None
        }
    }
}
