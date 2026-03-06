use super::*;

#[tokio::test]
async fn stage_text_formats_correctly() {
    assert_eq!(stage_text(&ProcessingStage::Received), "⏳ Processing…");
    assert_eq!(
        stage_text(&ProcessingStage::Enriching),
        "🔍 Fetching content…"
    );
    assert_eq!(stage_text(&ProcessingStage::RunningLlm), "🤖 Analysing…");
    assert_eq!(stage_text(&ProcessingStage::Writing), "✍️ Saving…");
    assert_eq!(
        stage_text(&ProcessingStage::Done {
            title: "My Title".into()
        }),
        "✅ My Title"
    );
    assert_eq!(
        stage_text(&ProcessingStage::Failed {
            reason: "oops".into()
        }),
        "❌ Failed: oops"
    );
}

#[test]
fn is_terminal_done_and_failed() {
    assert!(is_terminal(&ProcessingStage::Done { title: "x".into() }));
    assert!(is_terminal(&ProcessingStage::Failed { reason: "y".into() }));
    assert!(!is_terminal(&ProcessingStage::Received));
    assert!(!is_terminal(&ProcessingStage::Enriching));
    assert!(!is_terminal(&ProcessingStage::RunningLlm));
    assert!(!is_terminal(&ProcessingStage::Writing));
}

#[test]
fn terminal_stages_get_more_retries() {
    let non_terminal = ProcessingStage::Enriching;
    let terminal = ProcessingStage::Done { title: "x".into() };
    let normal = if is_terminal(&non_terminal) {
        TERMINAL_NOTIFY_RETRIES
    } else {
        MAX_NOTIFY_RETRIES
    };
    let term = if is_terminal(&terminal) {
        TERMINAL_NOTIFY_RETRIES
    } else {
        MAX_NOTIFY_RETRIES
    };
    assert_eq!(normal, MAX_NOTIFY_RETRIES);
    assert_eq!(term, TERMINAL_NOTIFY_RETRIES);
    assert!(term > normal);
}
