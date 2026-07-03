use async_trait::async_trait;
use inbox_core::CoreError;

use crate::message::ProcessedMessage;

pub mod org_file;
pub mod org_patcher;
#[cfg(test)]
mod tests_org_patcher;

// `OutputWriter`/`OutputTarget` now live in `inbox-core`; re-exported so existing
// `crate::output::*` paths keep working.
pub use inbox_core::output::{OutputTarget, OutputWriter};

/// A no-op writer for tests that never writes output.
#[cfg(any(test, feature = "test-helpers"))]
pub struct NullWriter;

#[cfg(any(test, feature = "test-helpers"))]
#[async_trait]
impl OutputWriter for NullWriter {
    async fn write(
        &self,
        _msg: &ProcessedMessage,
        _target: &OutputTarget<'_>,
    ) -> Result<(), CoreError> {
        Ok(())
    }
}
