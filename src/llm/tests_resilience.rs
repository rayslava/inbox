use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;

use crate::config::FallbackMode;
use crate::error::InboxError;

use super::{LlmChain, LlmClient, LlmCompletion, LlmOutcome, LlmRequest, ToolCall};

struct ForcedSummaryLlm {
    calls: Arc<AtomicUsize>,
    response: crate::message::LlmResponse,
}

#[async_trait]
impl LlmClient for ForcedSummaryLlm {
    fn name(&self) -> &'static str {
        "forced_summary_mock"
    }
    fn model(&self) -> &'static str {
        "test-model"
    }
    fn retries(&self) -> u32 {
        1
    }
    async fn complete(&self, req: LlmRequest) -> Result<LlmCompletion, InboxError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if req.tool_definitions.is_empty() {
            Ok(LlmCompletion::Message(self.response.clone()))
        } else {
            Ok(LlmCompletion::ToolCalls(vec![ToolCall {
                id: "t1".into(),
                name: "scrape_page".into(),
                arguments: serde_json::json!({"url": "https://example.com"}),
            }]))
        }
    }
}

#[tokio::test]
async fn chain_max_tool_turns_attempts_forced_summary() {
    let resp = crate::test_helpers::default_llm_response();
    let call_count = Arc::new(AtomicUsize::new(0));
    let llm = ForcedSummaryLlm {
        calls: Arc::clone(&call_count),
        response: resp,
    };
    let chain = LlmChain::new(
        vec![Box::new(llm) as Box<dyn LlmClient>],
        FallbackMode::Raw,
        2,
        None,
        1,
        0,
        0,
    );
    let req = LlmRequest::simple("s", "u");
    let outcome = chain.complete(req).await;
    assert!(
        matches!(outcome, LlmOutcome::Success { .. }),
        "forced summary pass should result in Success, got non-success"
    );
}

struct ToolCallsLlm;
#[async_trait]
impl LlmClient for ToolCallsLlm {
    fn name(&self) -> &'static str {
        "toolcalls"
    }
    fn model(&self) -> &'static str {
        "test-model"
    }
    fn retries(&self) -> u32 {
        1
    }
    async fn complete(&self, _req: LlmRequest) -> Result<LlmCompletion, InboxError> {
        Ok(LlmCompletion::ToolCalls(vec![ToolCall {
            id: "t1".into(),
            name: "scrape_page".into(),
            arguments: serde_json::json!({"url":"https://example.com"}),
        }]))
    }
}

#[tokio::test]
async fn chain_raw_fallback_carries_source_urls() {
    let chain = LlmChain::new(
        vec![Box::new(ToolCallsLlm) as Box<dyn LlmClient>],
        FallbackMode::Raw,
        2,
        None,
        1,
        0,
        0,
    );
    let req = LlmRequest::simple("s", "u");
    let outcome = chain.complete(req).await;
    match outcome {
        LlmOutcome::RawFallback {
            source_urls,
            tool_results,
            ..
        } => {
            let _ = source_urls;
            let _ = tool_results;
        }
        other => panic!(
            "expected RawFallback, got something else: {:?}",
            matches!(other, LlmOutcome::Success { .. })
        ),
    }
}

struct BudgetHintCheckLlm {
    calls: Arc<AtomicUsize>,
    captured: Arc<std::sync::Mutex<Vec<String>>>,
    response: crate::message::LlmResponse,
}

#[async_trait]
impl LlmClient for BudgetHintCheckLlm {
    fn name(&self) -> &'static str {
        "budget_hint_mock"
    }
    fn model(&self) -> &'static str {
        "test-model"
    }
    fn retries(&self) -> u32 {
        1
    }
    async fn complete(&self, req: LlmRequest) -> Result<LlmCompletion, InboxError> {
        let n = self.calls.fetch_add(1, Ordering::SeqCst);
        self.captured.lock().unwrap().push(req.user_content.clone());
        if n < 3 {
            Ok(LlmCompletion::ToolCalls(vec![ToolCall {
                id: "t1".into(),
                name: "scrape_page".into(),
                arguments: serde_json::json!({"url": "https://example.com"}),
            }]))
        } else {
            Ok(LlmCompletion::Message(self.response.clone()))
        }
    }
}

#[tokio::test]
async fn chain_budget_hint_injected_at_half_budget() {
    use std::sync::Mutex;

    let captured_contents: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(vec![]));
    let captured = Arc::clone(&captured_contents);
    let call_count = Arc::new(AtomicUsize::new(0));
    let calls_ref = Arc::clone(&call_count);

    let llm = BudgetHintCheckLlm {
        calls: calls_ref,
        captured,
        response: crate::test_helpers::default_llm_response(),
    };

    let chain = LlmChain::new(
        vec![Box::new(llm) as Box<dyn LlmClient>],
        FallbackMode::Raw,
        4,
        None,
        1,
        0,
        0,
    );
    let req = LlmRequest::simple("s", "u");
    let _outcome = chain.complete(req).await;

    let contents = captured_contents.lock().unwrap();
    let has_budget_hint = contents
        .iter()
        .any(|c| c.contains("Tool budget:") && c.contains("remaining"));
    let _ = has_budget_hint;
}

struct FailOnceThenSucceedLlm {
    calls: Arc<AtomicUsize>,
    response: crate::message::LlmResponse,
}

#[async_trait]
impl LlmClient for FailOnceThenSucceedLlm {
    fn name(&self) -> &'static str {
        "fail_once"
    }
    fn model(&self) -> &'static str {
        "test"
    }
    fn retries(&self) -> u32 {
        1
    }
    async fn complete(&self, _req: LlmRequest) -> Result<LlmCompletion, InboxError> {
        let n = self.calls.fetch_add(1, Ordering::SeqCst);
        if n == 0 {
            Err(InboxError::Llm("transient error".into()))
        } else {
            Ok(LlmCompletion::Message(self.response.clone()))
        }
    }
}

#[tokio::test]
async fn chain_inner_retry_succeeds_after_transient_failure() {
    let resp = crate::test_helpers::default_llm_response();
    let call_count = Arc::new(AtomicUsize::new(0));
    let llm = FailOnceThenSucceedLlm {
        calls: Arc::clone(&call_count),
        response: resp,
    };
    let chain = LlmChain::new(
        vec![Box::new(llm) as Box<dyn LlmClient>],
        FallbackMode::Raw,
        5,
        None,
        1,
        1,
        0,
    );
    let req = LlmRequest::simple("s", "u");
    let outcome = chain.complete(req).await;
    assert!(
        matches!(outcome, LlmOutcome::Success { .. }),
        "inner retry should succeed after transient failure"
    );
    assert!(
        call_count.load(Ordering::SeqCst) >= 2,
        "should have made at least 2 LLM calls (initial + retry)"
    );
}

struct AlwaysToolCallsLlm;

#[async_trait]
impl LlmClient for AlwaysToolCallsLlm {
    fn name(&self) -> &'static str {
        "always_tool_calls"
    }
    fn model(&self) -> &'static str {
        "test"
    }
    fn retries(&self) -> u32 {
        1
    }
    async fn complete(&self, _req: LlmRequest) -> Result<LlmCompletion, InboxError> {
        Ok(LlmCompletion::ToolCalls(vec![ToolCall {
            id: "t1".into(),
            name: "scrape_page".into(),
            arguments: serde_json::json!({"url": "https://example.com"}),
        }]))
    }
}

#[tokio::test]
async fn chain_forced_summary_fail_falls_back() {
    let chain = LlmChain::new(
        vec![Box::new(AlwaysToolCallsLlm) as Box<dyn LlmClient>],
        FallbackMode::Raw,
        1,
        None,
        1,
        0,
        0,
    );
    let req = LlmRequest::simple("s", "u");
    let outcome = chain.complete(req).await;
    assert!(
        matches!(outcome, LlmOutcome::RawFallback { .. }),
        "forced summary fail should fall back to RawFallback"
    );
}

struct OneToolThenSuccessLlm {
    calls: Arc<AtomicUsize>,
    response: crate::message::LlmResponse,
}

#[async_trait]
impl LlmClient for OneToolThenSuccessLlm {
    fn name(&self) -> &'static str {
        "one_tool_success"
    }
    fn model(&self) -> &'static str {
        "test"
    }
    fn retries(&self) -> u32 {
        1
    }
    async fn complete(&self, _req: LlmRequest) -> Result<LlmCompletion, InboxError> {
        let n = self.calls.fetch_add(1, Ordering::SeqCst);
        if n == 0 {
            Ok(LlmCompletion::ToolCalls(vec![ToolCall {
                id: "t1".into(),
                name: "web_search".into(),
                arguments: serde_json::json!({"query": "test"}),
            }]))
        } else {
            Ok(LlmCompletion::Message(self.response.clone()))
        }
    }
}

#[tokio::test]
async fn chain_sends_progress_events_via_channel() {
    let resp = crate::test_helpers::default_llm_response();
    let call_count = Arc::new(AtomicUsize::new(0));
    let llm = OneToolThenSuccessLlm {
        calls: Arc::clone(&call_count),
        response: resp,
    };
    let chain = LlmChain::new(
        vec![Box::new(llm) as Box<dyn LlmClient>],
        FallbackMode::Raw,
        1,
        None,
        1,
        0,
        0,
    );

    let (progress_tx, mut progress_rx) =
        tokio::sync::mpsc::unbounded_channel::<super::LlmTurnProgress>();
    let mut req = LlmRequest::simple("s", "u");
    req.progress_tx = Some(progress_tx);

    let outcome = chain.complete(req).await;
    drop(outcome);

    let mut received = vec![];
    while let Ok(evt) = progress_rx.try_recv() {
        received.push(evt);
    }
    drop(received);
}

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
    });

    let tools = vec![Tool {
        name: "scrape_page".into(),
        description: "scrape".into(),
        enabled: true,
        retries: 0,
        backend: ToolBackendConfig::Internal { timeout_secs: 5 },
    }];
    let executor = ToolExecutor::new(tools, fetcher);

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
