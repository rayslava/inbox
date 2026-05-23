//! Chain tests covering real tool execution (scrape + truncation) via `ToolExecutor`.

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;

use crate::config::FallbackMode;
use crate::error::InboxError;

use super::{LlmChain, LlmClient, LlmCompletion, LlmOutcome, LlmRequest, ToolCall};

struct CaptureTurn2Llm {
    turn: Arc<AtomicUsize>,
    scrape_url: String,
    captured: Arc<Mutex<Option<String>>>,
    response: crate::message::LlmResponse,
}

#[async_trait]
impl LlmClient for CaptureTurn2Llm {
    fn name(&self) -> &'static str {
        "capture_turn2"
    }
    fn model(&self) -> &'static str {
        "test"
    }
    fn retries(&self) -> u32 {
        1
    }
    async fn complete(&self, req: LlmRequest) -> Result<LlmCompletion, InboxError> {
        let n = self.turn.fetch_add(1, Ordering::SeqCst);
        if n == 0 {
            Ok(LlmCompletion::ToolCalls(vec![ToolCall {
                id: "t1".into(),
                name: "scrape_page".into(),
                arguments: serde_json::json!({"url": self.scrape_url}),
            }]))
        } else {
            *self.captured.lock().unwrap() = Some(req.user_content.clone());
            Ok(LlmCompletion::Message(self.response.clone()))
        }
    }
}

#[tokio::test]
async fn tool_result_truncated_in_chain() {
    use crate::config::{ToolBackendConfig, UrlFetchConfig};
    use crate::llm::tools::{Tool, ToolExecutor};
    use crate::pipeline::url_fetcher::UrlFetcher;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let content_server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/page"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/html")
                .set_body_string(format!(
                    "<html><body><p>{}</p></body></html>",
                    "x".repeat(200)
                )),
        )
        .mount(&content_server)
        .await;

    let scrape_url = format!("{}/page", content_server.uri());
    let captured: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));

    let turn_count = Arc::new(AtomicUsize::new(0));
    let llm = CaptureTurn2Llm {
        turn: Arc::clone(&turn_count),
        scrape_url,
        captured: Arc::clone(&captured),
        response: crate::test_helpers::default_llm_response(),
    };

    let fetcher = UrlFetcher::new(&UrlFetchConfig {
        enabled: true,
        user_agent: "test/1.0".into(),
        timeout_secs: 5,
        max_redirects: 3,
        max_body_bytes: 1024 * 1024,
        skip_domains: vec![],
        nitter_base_url: None,
    })
    .expect("build fetcher");

    let tools = vec![Tool {
        name: "scrape_page".into(),
        description: "scrape".into(),
        enabled: true,
        retries: 0,
        backend: ToolBackendConfig::Internal { timeout_secs: 5 },
    }];
    let executor = ToolExecutor::new(tools, fetcher).expect("build executor");

    let chain = LlmChain::new(
        vec![Box::new(llm) as Box<dyn LlmClient>],
        FallbackMode::Raw,
        5,
        Some(executor),
        1,
        0,
        50,
    );

    let req = LlmRequest::simple("s", "u");
    let outcome = chain.complete(req).await;
    assert!(matches!(outcome, LlmOutcome::Success { .. }));

    let guard = captured.lock().unwrap();
    let content = guard.as_deref().unwrap_or("");
    assert!(
        content.contains("[truncated to 50 chars]"),
        "expected truncation notice in turn-2 content, got: {content}"
    );
}
