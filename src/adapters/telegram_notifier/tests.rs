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
