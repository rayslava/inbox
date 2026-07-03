use std::path::Path;

use async_trait::async_trait;

use crate::CoreError;
use crate::message::ProcessedMessage;

/// Where a written note lands. Narrow replacement for threading the whole
/// daemon `Config` through the write boundary.
pub struct OutputTarget<'a> {
    /// Org file the rendered node is appended to.
    pub output_file: &'a Path,
    /// Directory attachments were saved under (used to resolve links on render).
    pub attachments_dir: &'a Path,
}

/// Persists a fully-processed message as an org note.
///
/// Implemented in the `inbox` binary (askama render + Syncthing rescan); `core`
/// exposes only the narrow surface so alternative writers (and a future
/// `core/curate`) can depend on the trait, not the `Config` god-object.
#[async_trait]
pub trait OutputWriter: Send + Sync + 'static {
    /// Render and persist `msg` to `target`.
    ///
    /// # Errors
    /// Returns [`CoreError`] if rendering or the write fails.
    async fn write(
        &self,
        msg: &ProcessedMessage,
        target: &OutputTarget<'_>,
    ) -> Result<(), CoreError>;
}
