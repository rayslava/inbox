use std::net::IpAddr;
use std::path::PathBuf;

use chrono::{DateTime, Utc};
use uuid::Uuid;

use crate::processing_status::StatusNotifier;
use crate::url_content::UrlContent;

pub struct IncomingMessage {
    pub id: Uuid,
    pub source: MessageSource,
    pub received_at: DateTime<Utc>,
    pub text: String,
    pub metadata: SourceMetadata,
    pub attachments: Vec<Attachment>,
    /// Hashtags extracted from the message text (e.g. `#rust` → `"rust"`).
    /// Populated by the pipeline before enrichment; empty if none were found.
    pub user_tags: Vec<String>,
    /// Hints produced by the pre-processing stage.
    /// Populated before enrichment and consumed by LLM guidance and rendering.
    pub preprocessing_hints: ProcessingHints,
    /// Per-message status notifier. Adapters set this; pipeline extracts and drives it.
    pub status_notifier: Option<Box<dyn StatusNotifier>>,
}

/// Hints derived from the pre-processing stage that guide subsequent pipeline stages.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ProcessingHints {
    /// If `true`, instruct the LLM to call the web search tool before summarizing.
    pub force_web_search: bool,
    /// Additional natural-language hints appended to the LLM system prompt.
    pub extra_llm_hints: Vec<String>,
    /// Tags suggested by pre-processing rules; merged with LLM tags in the org output.
    pub suggested_tags: Vec<String>,
}

impl std::fmt::Debug for IncomingMessage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IncomingMessage")
            .field("id", &self.id)
            .field("source", &self.source)
            .field("received_at", &self.received_at)
            .field("text", &self.text)
            .field("metadata", &self.metadata)
            .field("attachments", &self.attachments)
            .field("user_tags", &self.user_tags)
            .field("preprocessing_hints", &self.preprocessing_hints)
            .finish_non_exhaustive()
    }
}

impl IncomingMessage {
    #[must_use]
    pub fn new(source: MessageSource, text: String, metadata: SourceMetadata) -> Self {
        Self::with_id(Uuid::new_v4(), source, text, metadata)
    }

    #[must_use]
    pub fn with_id(
        id: Uuid,
        source: MessageSource,
        text: String,
        metadata: SourceMetadata,
    ) -> Self {
        Self {
            id,
            source,
            received_at: Utc::now(),
            text,
            metadata,
            attachments: Vec::new(),
            user_tags: Vec::new(),
            preprocessing_hints: ProcessingHints::default(),
            status_notifier: None,
        }
    }

    #[must_use]
    pub fn source_name(&self) -> &'static str {
        self.source.as_str()
    }
}

/// A clone-able snapshot of an [`IncomingMessage`] used to re-enqueue failed messages.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RetryableMessage {
    pub text: String,
    pub metadata: SourceMetadata,
    pub attachments: Vec<Attachment>,
    pub user_tags: Vec<String>,
    pub preprocessing_hints: ProcessingHints,
    pub received_at: DateTime<Utc>,
}

impl From<&IncomingMessage> for RetryableMessage {
    fn from(msg: &IncomingMessage) -> Self {
        Self {
            text: msg.text.clone(),
            metadata: msg.metadata.clone(),
            attachments: msg.attachments.clone(),
            user_tags: msg.user_tags.clone(),
            preprocessing_hints: msg.preprocessing_hints.clone(),
            received_at: msg.received_at,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum MessageSource {
    Telegram,
    Http,
    Email,
}

impl MessageSource {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Telegram => "telegram",
            Self::Http => "http",
            Self::Email => "email",
        }
    }
}

impl std::fmt::Display for MessageSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum SourceMetadata {
    Telegram {
        chat_id: i64,
        message_id: i32,
        username: Option<String>,
        /// Display name of the original sender when the message was forwarded.
        forwarded_from: Option<String>,
    },
    Http {
        remote_addr: Option<IpAddr>,
        user_agent: Option<String>,
    },
    Email {
        subject: String,
        from: String,
        message_id: Option<String>,
    },
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Attachment {
    pub original_name: String,
    pub saved_path: PathBuf,
    pub mime_type: Option<String>,
    pub media_kind: MediaKind,
}

/// Semantic classification independent of mime string.
/// Used by web UI inline preview and org template.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum MediaKind {
    Image,
    Audio,
    Video,
    Document,
    Sticker,
    Animation,
    VoiceMessage,
    Other,
}

impl MediaKind {
    #[must_use]
    pub fn from_mime(mime: &str) -> Self {
        if mime.starts_with("image/") {
            Self::Image
        } else if mime.starts_with("audio/") {
            Self::Audio
        } else if mime.starts_with("video/") {
            Self::Video
        } else {
            Self::Document
        }
    }

    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Image => "image",
            Self::Audio => "audio",
            Self::Video => "video",
            Self::Document => "document",
            Self::Sticker => "sticker",
            Self::Animation => "animation",
            Self::VoiceMessage => "voice_message",
            Self::Other => "other",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_source_as_str() {
        assert_eq!(MessageSource::Telegram.as_str(), "telegram");
        assert_eq!(MessageSource::Http.as_str(), "http");
        assert_eq!(MessageSource::Email.as_str(), "email");
    }

    #[test]
    fn message_source_display() {
        assert_eq!(MessageSource::Http.to_string(), "http");
    }

    #[test]
    fn incoming_message_new() {
        let msg = IncomingMessage::new(
            MessageSource::Http,
            "hello".into(),
            SourceMetadata::Http {
                remote_addr: None,
                user_agent: None,
            },
        );
        assert_eq!(msg.text, "hello");
        assert_eq!(msg.source_name(), "http");
        assert!(msg.attachments.is_empty());
    }

    #[test]
    fn media_kind_from_mime() {
        assert_eq!(MediaKind::from_mime("image/png"), MediaKind::Image);
        assert_eq!(MediaKind::from_mime("audio/mpeg"), MediaKind::Audio);
        assert_eq!(MediaKind::from_mime("video/mp4"), MediaKind::Video);
        assert_eq!(MediaKind::from_mime("application/pdf"), MediaKind::Document);
    }

    #[test]
    fn media_kind_as_str() {
        assert_eq!(MediaKind::Image.as_str(), "image");
        assert_eq!(MediaKind::Audio.as_str(), "audio");
        assert_eq!(MediaKind::Video.as_str(), "video");
        assert_eq!(MediaKind::Document.as_str(), "document");
        assert_eq!(MediaKind::Sticker.as_str(), "sticker");
        assert_eq!(MediaKind::Animation.as_str(), "animation");
        assert_eq!(MediaKind::VoiceMessage.as_str(), "voice_message");
        assert_eq!(MediaKind::Other.as_str(), "other");
    }
}

#[derive(Debug)]
pub struct EnrichedMessage {
    pub original: IncomingMessage,
    pub urls: Vec<url::Url>,
    pub url_contents: Vec<UrlContent>,
}

#[derive(Debug)]
pub struct ProcessedMessage {
    pub enriched: EnrichedMessage,
    /// None means raw fallback (LLM unavailable or all backends failed).
    pub llm_response: Option<LlmResponse>,
    /// URLs gathered by tools during LLM processing; populated when LLM falls back.
    pub fallback_source_urls: Vec<String>,
    /// Structured tool results as `(tool_name, text)` pairs; used as summary in raw-fallback rendering.
    pub fallback_tool_results: Vec<(String, String)>,
    /// Title generated by a post-fallback LLM call; `None` if that also failed or was not attempted.
    pub fallback_title: Option<String>,
    /// Diagnostic metadata about the enrichment run: helper models consulted,
    /// memory-recall stats, URL/tool counts. Rendered into the org entry drawer.
    pub enrichment: EnrichmentMetadata,
}

/// Observability trail for one enrichment run. Rendered into the org entry
/// drawer so a reader can see *what* produced the node: which models answered,
/// how much memory was recalled, how many URLs/tool calls were involved.
#[derive(Debug, Default, Clone)]
pub struct EnrichmentMetadata {
    /// `backend:model` identifiers of any additional models consulted during
    /// the run (`llm_call` sub-calls, fallback-title generation). Deduplicated,
    /// ordered by first use. The primary model lives in `LlmResponse.produced_by`.
    pub helpers: Vec<String>,
    /// Number of memories retrieved by `preload_memory_context` before the
    /// primary LLM call. `0` if memory is disabled or no matches found.
    pub memories_recalled: usize,
    /// Number of URLs extracted from the original message (post-preprocessing).
    pub urls_fetched: usize,
    /// Total tool invocations executed during the LLM turn loop.
    pub tool_calls_made: usize,
}

#[derive(Debug, Clone)]
pub struct LlmResponse {
    pub title: String,
    pub tags: Vec<String>,
    pub summary: String,
    pub excerpt: Option<String>,
    pub produced_by: String,
}
