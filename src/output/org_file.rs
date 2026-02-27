use std::path::Path;

use async_trait::async_trait;
use tokio::fs::OpenOptions;
use tokio::io::AsyncWriteExt;
use tracing::{instrument, warn};

use crate::config::{Config, SyncthingConfig};
use crate::error::InboxError;
use crate::message::ProcessedMessage;
use crate::render;

use super::OutputWriter;

pub struct OrgFileWriter;

#[async_trait]
impl OutputWriter for OrgFileWriter {
    #[instrument(skip(self, msg, cfg))]
    async fn write(&self, msg: &ProcessedMessage, cfg: &Config) -> Result<(), InboxError> {
        let node = render::render_org_node(msg, &cfg.general.attachments_dir)?;

        append_to_file(&cfg.general.output_file, &node)
            .await
            .map_err(|e| {
                metrics::counter!(crate::telemetry::WRITE_ERRORS).increment(1);
                InboxError::Output(format!("Failed to append org node: {e}"))
            })?;

        if cfg.syncthing.enabled && cfg.syncthing.rescan_on_write {
            trigger_syncthing_rescans(&cfg.syncthing).await;
        }

        Ok(())
    }
}

async fn append_to_file(path: &Path, content: &str) -> Result<(), std::io::Error> {
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .await?;

    file.write_all(content.as_bytes()).await?;
    file.write_all(b"\n").await?;
    file.flush().await?;
    Ok(())
}

async fn trigger_syncthing_rescans(cfg: &SyncthingConfig) {
    let client = reqwest::Client::new();
    rescan_folder(&client, cfg, &cfg.org_folder_id).await;

    if let Some(att_folder) = &cfg.attachments_folder_id
        && att_folder != &cfg.org_folder_id
    {
        rescan_folder(&client, cfg, att_folder).await;
    }
}

async fn rescan_folder(client: &reqwest::Client, cfg: &SyncthingConfig, folder_id: &str) {
    if folder_id.is_empty() {
        return;
    }
    let url = format!(
        "{}/rest/db/scan?folder={}",
        cfg.api_url,
        urlencoding::encode(folder_id)
    );
    let result = client
        .post(url)
        .header("X-API-Key", &cfg.api_key)
        .send()
        .await;

    if let Err(e) = result {
        warn!(?e, folder_id, "Syncthing rescan trigger failed (non-fatal)");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn append_creates_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.org");
        append_to_file(&path, "* Test node\n").await.unwrap();
        let content = tokio::fs::read_to_string(&path).await.unwrap();
        assert!(content.contains("* Test node"));
    }

    #[tokio::test]
    async fn append_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.org");
        append_to_file(&path, "line1\n").await.unwrap();
        append_to_file(&path, "line2\n").await.unwrap();
        let content = tokio::fs::read_to_string(&path).await.unwrap();
        assert!(content.contains("line1"));
        assert!(content.contains("line2"));
    }
}
