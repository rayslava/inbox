/// Integration tests for the processing pipeline.
use std::sync::Arc;

use inbox::{
    config::{
        AdaptersConfig, AdminConfig, Config, GeneralConfig, PipelineConfig, SyncthingConfig,
        ToolingConfig, UrlFetchConfig, WebUiConfig,
    },
    message::{IncomingMessage, MessageSource, SourceMetadata},
    output::{OutputWriter, org_file::OrgFileWriter},
    pipeline::Pipeline,
    test_helpers as helpers,
};

fn minimal_config(attachments_dir: std::path::PathBuf, output_file: std::path::PathBuf) -> Config {
    Config {
        general: GeneralConfig {
            output_file,
            attachments_dir,
            log_level: "debug".into(),
            log_format: "pretty".into(),
        },
        admin: AdminConfig::default(),
        web_ui: WebUiConfig::default(),
        pipeline: PipelineConfig::default(),
        llm: helpers::no_llm_config(),
        adapters: AdaptersConfig::default(),
        url_fetch: UrlFetchConfig {
            enabled: false,
            ..UrlFetchConfig::default()
        },
        syncthing: SyncthingConfig::default(),
        tooling: ToolingConfig::default(),
    }
}

#[tokio::test]
async fn pipeline_writes_org_node_for_plain_text_message() {
    let (_tmp, dir) = helpers::temp_dir();
    let output_file = dir.join("inbox.org");
    let cfg = Arc::new(minimal_config(dir.clone(), output_file.clone()));

    let llm = helpers::mock_llm_chain(helpers::default_llm_response());
    let writer = Arc::new(OrgFileWriter) as Arc<dyn OutputWriter>;
    let pipeline = Arc::new(Pipeline::new(Arc::clone(&cfg), llm, writer));

    let msg = IncomingMessage::new(
        MessageSource::Http,
        "Hello from test".into(),
        SourceMetadata::Http {
            remote_addr: None,
            user_agent: None,
        },
    );

    pipeline
        .process(msg)
        .await
        .expect("pipeline should succeed");

    let content = tokio::fs::read_to_string(&output_file)
        .await
        .expect("output file should exist");

    assert!(content.contains("* Test title"), "missing headline");
    assert!(content.contains(":SOURCE:"), "missing SOURCE property");
    assert!(content.contains("A test summary"), "missing summary");
}

#[tokio::test]
async fn pipeline_handles_empty_text_gracefully() {
    let (_tmp, dir) = helpers::temp_dir();
    let output_file = dir.join("inbox.org");
    let cfg = Arc::new(minimal_config(dir.clone(), output_file.clone()));

    let llm = helpers::mock_llm_chain(helpers::default_llm_response());
    let writer = Arc::new(OrgFileWriter) as Arc<dyn OutputWriter>;
    let pipeline = Arc::new(Pipeline::new(Arc::clone(&cfg), llm, writer));

    let msg = IncomingMessage::new(
        MessageSource::Http,
        String::new(),
        SourceMetadata::Http {
            remote_addr: None,
            user_agent: None,
        },
    );

    pipeline
        .process(msg)
        .await
        .expect("pipeline should not error on empty text");
    assert!(output_file.exists(), "output file should be created");
}

#[tokio::test]
async fn pipeline_appends_multiple_messages() {
    let (_tmp, dir) = helpers::temp_dir();
    let output_file = dir.join("inbox.org");
    let cfg = Arc::new(minimal_config(dir.clone(), output_file.clone()));

    let llm = helpers::mock_llm_chain(helpers::default_llm_response());
    let writer = Arc::new(OrgFileWriter) as Arc<dyn OutputWriter>;
    let pipeline = Arc::new(Pipeline::new(Arc::clone(&cfg), llm, writer));

    for i in 0..3_u8 {
        let msg = IncomingMessage::new(
            MessageSource::Http,
            format!("Message number {i}"),
            SourceMetadata::Http {
                remote_addr: None,
                user_agent: None,
            },
        );
        pipeline
            .process(msg)
            .await
            .expect("pipeline should succeed");
    }

    let content = tokio::fs::read_to_string(&output_file).await.unwrap();
    let headline_count = content.matches("* Test title").count();
    assert_eq!(headline_count, 3, "expected 3 org headlines");
}
