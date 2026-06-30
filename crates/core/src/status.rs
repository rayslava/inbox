use async_trait::async_trait;

/// Coarse lifecycle stage of a message moving through the pipeline. Serialized
/// (tagged by `stage`) for the admin/web status view.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "snake_case", tag = "stage")]
pub enum ProcessingStage {
    Received,
    AnalyzingImages,
    Enriching,
    RunningLlm {
        turn: usize,
        max_turns: usize,
        last_tools: Vec<String>,
    },
    Writing,
    Done {
        title: String,
    },
    /// Written but not final — held for retry (e.g. image text unread because
    /// vision was unavailable). Re-OCR'd on resume; never reported as success.
    Pending {
        title: String,
    },
    Failed {
        reason: String,
    },
}

/// Per-message status sink. Adapters set a concrete notifier on the incoming
/// message; the pipeline extracts and drives it as the message advances.
#[async_trait]
pub trait StatusNotifier: Send + Sync {
    async fn advance(&mut self, stage: ProcessingStage);

    /// Returns the Telegram message ID of the status message, if this is a
    /// Telegram notifier. Used by the pending store to enable resume notifications.
    fn telegram_status_msg_id(&self) -> Option<i32> {
        None
    }
}

/// No-op notifier for sources without live status reporting.
pub struct NoopNotifier;

#[async_trait]
impl StatusNotifier for NoopNotifier {
    async fn advance(&mut self, _stage: ProcessingStage) {}
}

#[cfg(test)]
mod tests {
    use super::{NoopNotifier, ProcessingStage, StatusNotifier};

    #[tokio::test]
    async fn noop_notifier_advances_and_has_no_telegram_id() {
        let mut n = NoopNotifier;
        n.advance(ProcessingStage::Received).await;
        assert!(n.telegram_status_msg_id().is_none());
    }

    #[test]
    fn processing_stage_serializes_with_stage_tag() {
        let json = serde_json::to_string(&ProcessingStage::Done { title: "t".into() })
            .expect("serialize stage");
        assert!(json.contains("\"stage\":\"done\""));
        assert!(json.contains("\"title\":\"t\""));
    }
}
