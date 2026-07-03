use std::path::Path;

use async_trait::async_trait;
use inbox_core::{CoreError, OutputTarget};
use tokio::fs::OpenOptions;
use tokio::io::AsyncWriteExt;
use tracing::{instrument, warn};

use crate::config::SyncthingConfig;
use crate::message::ProcessedMessage;
use crate::render;

use super::OutputWriter;

/// Writes enriched messages as org nodes, then optionally triggers a Syncthing
/// rescan. Holds its own [`SyncthingConfig`] so the write boundary stays narrow
/// (paths only, via [`OutputTarget`]) instead of taking the whole `Config`.
pub struct OrgFileWriter {
    syncthing: SyncthingConfig,
}

impl OrgFileWriter {
    #[must_use]
    pub fn new(syncthing: SyncthingConfig) -> Self {
        Self { syncthing }
    }
}

#[async_trait]
impl OutputWriter for OrgFileWriter {
    #[instrument(skip(self, msg, target))]
    async fn write(
        &self,
        msg: &ProcessedMessage,
        target: &OutputTarget<'_>,
    ) -> Result<(), CoreError> {
        let start = std::time::Instant::now();
        let node = render::render_org_node(msg, target.attachments_dir)?;

        append_to_file(target.output_file, &node)
            .await
            .map_err(|e| {
                metrics::counter!(crate::telemetry::WRITE_ERRORS).increment(1);
                metrics::counter!(crate::telemetry::WRITES_TOTAL, "status" => "failure")
                    .increment(1);
                CoreError::Output(format!("Failed to append org node: {e}"))
            })?;

        if self.syncthing.enabled && self.syncthing.rescan_on_write {
            trigger_syncthing_rescans(&self.syncthing).await;
        }

        metrics::counter!(crate::telemetry::WRITES_TOTAL, "status" => "success").increment(1);
        metrics::histogram!(crate::telemetry::WRITE_DURATION).record(start.elapsed().as_secs_f64());
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

pub(crate) async fn trigger_syncthing_rescans(cfg: &SyncthingConfig) {
    let client = match crate::tls::client_builder().build() {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!(?e, "Failed to build Syncthing HTTP client; skipping rescan");
            return;
        }
    };
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
    use crate::config::Config;
    use crate::message::{
        EnrichedMessage, IncomingMessage, MessageSource, ProcessedMessage, SourceMetadata,
    };
    use inbox_core::{CoreError, OutputTarget};
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn target_for(cfg: &Config) -> OutputTarget<'_> {
        OutputTarget {
            output_file: &cfg.general.output_file,
            attachments_dir: &cfg.general.attachments_dir,
        }
    }

    fn test_processed_message(text: &str) -> ProcessedMessage {
        ProcessedMessage {
            enriched: EnrichedMessage {
                original: IncomingMessage::new(
                    MessageSource::Http,
                    text.to_owned(),
                    SourceMetadata::Http {
                        remote_addr: None,
                        user_agent: None,
                    },
                ),
                urls: Vec::new(),
                url_contents: Vec::new(),
            },
            llm_response: None,
            incomplete: crate::message::ProcessingCompleteness::Complete,
            fallback_source_urls: vec![],
            fallback_tool_results: vec![],
            fallback_title: None,
            enrichment: crate::message::EnrichmentMetadata::default(),
        }
    }

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

    #[tokio::test]
    async fn rescan_folder_skips_empty_folder_id() {
        let server = MockServer::start().await;
        let client = reqwest::Client::new();
        let cfg = SyncthingConfig {
            enabled: true,
            api_url: server.uri(),
            api_key: "k".into(),
            org_folder_id: String::new(),
            attachments_folder_id: None,
            rescan_on_write: true,
        };

        rescan_folder(&client, &cfg, "").await;

        let requests = server.received_requests().await.expect("requests");
        assert!(requests.is_empty());
    }

    #[tokio::test]
    async fn trigger_syncthing_rescans_posts_for_org_and_attachments() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/rest/db/scan"))
            .respond_with(ResponseTemplate::new(200))
            .expect(2)
            .mount(&server)
            .await;

        let cfg = SyncthingConfig {
            enabled: true,
            api_url: server.uri(),
            api_key: "k".into(),
            org_folder_id: "org-folder".into(),
            attachments_folder_id: Some("att-folder".into()),
            rescan_on_write: true,
        };

        trigger_syncthing_rescans(&cfg).await;
    }

    #[tokio::test]
    async fn trigger_syncthing_rescans_does_not_duplicate_same_folder() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/rest/db/scan"))
            .respond_with(ResponseTemplate::new(200))
            .expect(1)
            .mount(&server)
            .await;

        let cfg = SyncthingConfig {
            enabled: true,
            api_url: server.uri(),
            api_key: "k".into(),
            org_folder_id: "shared".into(),
            attachments_folder_id: Some("shared".into()),
            rescan_on_write: true,
        };

        trigger_syncthing_rescans(&cfg).await;
    }

    #[tokio::test]
    async fn writer_write_appends_and_triggers_syncthing() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/rest/db/scan"))
            .respond_with(ResponseTemplate::new(200))
            .expect(1)
            .mount(&server)
            .await;

        let dir = tempfile::tempdir().expect("tempdir");
        let output_file = dir.path().join("inbox.org");
        let attachments_dir = dir.path().join("attachments");

        let cfg_toml = format!(
            r#"
[general]
output_file = "{}"
attachments_dir = "{}"

[llm]

[syncthing]
enabled = true
rescan_on_write = true
api_url = "{}"
api_key = "k"
org_folder_id = "org-folder"
"#,
            output_file.display(),
            attachments_dir.display(),
            server.uri()
        );
        let cfg: Config = toml::from_str(&cfg_toml).expect("config");

        let writer = OrgFileWriter::new(cfg.syncthing.clone());
        let msg = test_processed_message("test message");
        writer
            .write(&msg, &target_for(&cfg))
            .await
            .expect("write ok");

        let content = tokio::fs::read_to_string(&output_file)
            .await
            .expect("output read");
        assert!(content.contains("* test message"));
    }

    #[tokio::test]
    async fn writer_write_maps_append_error() {
        let dir = tempfile::tempdir().expect("tempdir");
        let output_file = dir.path().to_path_buf(); // directory path, not a file
        let attachments_dir = dir.path().join("attachments");

        let cfg_toml = format!(
            r#"
[general]
output_file = "{}"
attachments_dir = "{}"

[llm]
"#,
            output_file.display(),
            attachments_dir.display(),
        );
        let cfg: Config = toml::from_str(&cfg_toml).expect("config");

        let writer = OrgFileWriter::new(cfg.syncthing.clone());
        let msg = test_processed_message("test message");
        let err = writer
            .write(&msg, &target_for(&cfg))
            .await
            .expect_err("must fail");
        assert!(matches!(err, CoreError::Output(_)));
    }

    #[tokio::test]
    async fn rescan_folder_network_error_is_non_fatal() {
        let client = reqwest::Client::new();
        let cfg = SyncthingConfig {
            enabled: true,
            api_url: "http://127.0.0.1:1".into(),
            api_key: "k".into(),
            org_folder_id: "org".into(),
            attachments_folder_id: None,
            rescan_on_write: true,
        };
        rescan_folder(&client, &cfg, "org").await;
    }
}
