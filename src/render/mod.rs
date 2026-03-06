use askama::Template;

use crate::error::InboxError;
use crate::message::ProcessedMessage;

pub struct AttachmentRef<'a> {
    pub name: &'a str,
    pub path: String,
    pub path_rel: String, // relative to attachments_dir, for /attachments/* URL
}

#[derive(Template)]
#[template(path = "node.org", escape = "none")]
pub struct OrgNodeTemplate<'a> {
    pub title: &'a str,
    pub tags: &'a [String],
    pub id: &'a str,
    pub created: &'a str,
    pub source: &'a str,
    pub urls: &'a [String],
    pub attachments: &'a [AttachmentRef<'a>],
    pub llm_backend: &'a str,
    pub summary: &'a str,
    pub excerpt: Option<&'a str>,
    pub raw_text: &'a str,
    /// Original sender of a forwarded message, if any.
    pub forwarded_from: Option<&'a str>,
    /// Media kinds of non-document attachments (audio/video/voice).
    pub media_kinds: &'a [&'static str],
}

impl OrgNodeTemplate<'_> {
    #[must_use]
    pub fn attachment_names(&self) -> String {
        self.attachments
            .iter()
            .map(|a| a.name)
            .collect::<Vec<_>>()
            .join(" ")
    }

    #[must_use]
    pub fn media_kinds_str(&self) -> String {
        self.media_kinds.join(" ")
    }
}

/// Render a `ProcessedMessage` to an org-mode node string.
///
/// # Errors
/// Returns an error if the Askama template fails to render.
pub fn render_org_node(
    msg: &ProcessedMessage,
    attachments_dir: &std::path::Path,
) -> Result<String, InboxError> {
    use crate::message::{MediaKind, SourceMetadata};

    let enriched = &msg.enriched;
    let original = &enriched.original;

    let id_str = original.id.to_string();
    let created = original
        .received_at
        .format("[%Y-%m-%d %a %H:%M]")
        .to_string();
    let source = original.source.as_str().to_owned();
    let urls: Vec<String> = enriched
        .urls
        .iter()
        .map(std::string::ToString::to_string)
        .collect();

    let att_refs: Vec<AttachmentRef<'_>> = original
        .attachments
        .iter()
        .map(|a| {
            let path_str = a.saved_path.to_string_lossy();
            let rel = relative_path(&a.saved_path, attachments_dir);
            AttachmentRef {
                name: &a.original_name,
                path: path_str.into_owned(),
                path_rel: rel,
            }
        })
        .collect();

    let forwarded_from = match &original.metadata {
        SourceMetadata::Telegram { forwarded_from, .. } => forwarded_from.as_deref(),
        _ => None,
    };

    let media_kinds: Vec<&'static str> = original
        .attachments
        .iter()
        .filter_map(|a| match a.media_kind {
            MediaKind::Audio => Some("audio"),
            MediaKind::Video => Some("video"),
            MediaKind::VoiceMessage => Some("voice_message"),
            _ => None,
        })
        .collect();

    // Merge user-supplied tags, pre-processing suggested tags, and LLM tags (in that
    // priority order) into a single deduplicated list.
    let merged_tags: Vec<String> = {
        let mut all = original.user_tags.clone();
        for t in &original.preprocessing_hints.suggested_tags {
            if !all.iter().any(|x| x.eq_ignore_ascii_case(t)) {
                all.push(t.clone());
            }
        }
        if let Some(r) = &msg.llm_response {
            for t in &r.tags {
                if !all.iter().any(|x| x.eq_ignore_ascii_case(t)) {
                    all.push(t.clone());
                }
            }
        }
        all
    };

    let (title, summary, excerpt, backend) = match &msg.llm_response {
        Some(r) => (
            r.title.as_str(),
            r.summary.as_str(),
            r.excerpt.as_deref(),
            r.produced_by.as_str(),
        ),
        None => (
            original.text.lines().next().unwrap_or("(untitled)"),
            original.text.as_str(),
            None,
            "none",
        ),
    };

    let tmpl = OrgNodeTemplate {
        title,
        tags: &merged_tags,
        id: &id_str,
        created: &created,
        source: &source,
        urls: &urls,
        attachments: &att_refs,
        llm_backend: backend,
        summary,
        excerpt,
        raw_text: &original.text,
        forwarded_from,
        media_kinds: &media_kinds,
    };

    tmpl.render().map_err(InboxError::Template)
}

fn relative_path(path: &std::path::Path, base: &std::path::Path) -> String {
    path.strip_prefix(base).map_or_else(
        |_| path.to_string_lossy().into_owned(),
        |p| p.to_string_lossy().into_owned(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{
        EnrichedMessage, IncomingMessage, LlmResponse, MessageSource, ProcessedMessage,
        SourceMetadata,
    };

    fn make_processed(text: &str, llm_response: Option<LlmResponse>) -> ProcessedMessage {
        let msg = IncomingMessage::new(
            MessageSource::Http,
            text.into(),
            SourceMetadata::Http {
                remote_addr: None,
                user_agent: None,
            },
        );
        ProcessedMessage {
            enriched: EnrichedMessage {
                original: msg,
                urls: vec![],
                url_contents: vec![],
            },
            llm_response,
        }
    }

    #[test]
    fn render_with_llm_response() {
        let resp = LlmResponse {
            title: "My Title".into(),
            tags: vec!["rust".into(), "test".into()],
            summary: "A summary.".into(),
            excerpt: Some("Key quote".into()),
            produced_by: "mock".into(),
        };
        let msg = make_processed("raw text", Some(resp));
        let result = render_org_node(&msg, std::path::Path::new("/tmp")).unwrap();
        assert!(result.contains("* My Title"));
        assert!(result.contains(":rust:test:"));
        assert!(result.contains("A summary."));
        assert!(result.contains("Key quote"));
        assert!(result.contains(":ENRICHED_BY: mock"));
    }

    #[test]
    fn render_without_llm_response_raw_fallback() {
        let msg = make_processed("First line\nSecond line", None);
        let result = render_org_node(&msg, std::path::Path::new("/tmp")).unwrap();
        assert!(result.contains("* First line"));
        assert!(result.contains(":ENRICHED_BY: none"));
        assert!(result.contains("First line"));
    }

    #[test]
    fn render_empty_text_untitled() {
        let msg = make_processed("", None);
        let result = render_org_node(&msg, std::path::Path::new("/tmp")).unwrap();
        assert!(result.contains("(untitled)"));
    }

    #[test]
    fn attachment_names_joined() {
        let tmpl = OrgNodeTemplate {
            title: "t",
            tags: &[],
            id: "id",
            created: "now",
            source: "http",
            urls: &[],
            attachments: &[
                AttachmentRef {
                    name: "a.pdf",
                    path: "/p/a.pdf".to_owned(),
                    path_rel: "a.pdf".to_owned(),
                },
                AttachmentRef {
                    name: "b.jpg",
                    path: "/p/b.jpg".to_owned(),
                    path_rel: "b.jpg".to_owned(),
                },
            ],
            llm_backend: "mock",
            summary: "s",
            excerpt: None,
            raw_text: "",
            forwarded_from: None,
            media_kinds: &[],
        };
        assert_eq!(tmpl.attachment_names(), "a.pdf b.jpg");
    }

    #[test]
    fn render_with_url_in_enriched() {
        let msg_inner = IncomingMessage::new(
            MessageSource::Http,
            "text".into(),
            SourceMetadata::Http {
                remote_addr: None,
                user_agent: None,
            },
        );
        let url: url::Url = "https://example.com/page".parse().unwrap();
        let msg = ProcessedMessage {
            enriched: EnrichedMessage {
                original: msg_inner,
                urls: vec![url],
                url_contents: vec![],
            },
            llm_response: None,
        };
        let result = render_org_node(&msg, std::path::Path::new("/tmp")).unwrap();
        assert!(result.contains("https://example.com/page"));
    }

    #[test]
    fn render_forwarded_from_appears_in_drawer() {
        let msg = IncomingMessage::new(
            MessageSource::Telegram,
            "forwarded content".into(),
            SourceMetadata::Telegram {
                chat_id: 1,
                message_id: 1,
                username: None,
                forwarded_from: Some("@bob".into()),
            },
        );
        let processed = ProcessedMessage {
            enriched: EnrichedMessage {
                original: msg,
                urls: vec![],
                url_contents: vec![],
            },
            llm_response: None,
        };
        let result = render_org_node(&processed, std::path::Path::new("/tmp")).unwrap();
        assert!(
            result.contains(":FORWARDED_FROM: @bob"),
            "drawer should contain FORWARDED_FROM: {result}"
        );
    }

    #[test]
    fn render_no_forwarded_property_when_absent() {
        let msg = make_processed("plain", None);
        let result = render_org_node(&msg, std::path::Path::new("/tmp")).unwrap();
        assert!(
            !result.contains("FORWARDED_FROM"),
            "FORWARDED_FROM should not appear when absent: {result}"
        );
    }

    #[test]
    fn render_voice_message_media_kind_in_drawer() {
        use crate::message::Attachment;

        let mut msg = IncomingMessage::new(
            MessageSource::Telegram,
            "voice note".into(),
            SourceMetadata::Telegram {
                chat_id: 1,
                message_id: 2,
                username: None,
                forwarded_from: None,
            },
        );
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("voice.ogg");
        std::fs::write(&path, b"ogg").unwrap();
        msg.attachments.push(Attachment {
            original_name: "voice.ogg".into(),
            saved_path: path,
            mime_type: Some("audio/ogg".into()),
            media_kind: crate::message::MediaKind::VoiceMessage,
        });
        let processed = ProcessedMessage {
            enriched: EnrichedMessage {
                original: msg,
                urls: vec![],
                url_contents: vec![],
            },
            llm_response: None,
        };
        let result = render_org_node(&processed, tmp.path()).unwrap();
        assert!(
            result.contains(":MEDIA_KIND: voice_message"),
            "drawer should contain MEDIA_KIND: {result}"
        );
    }

    #[test]
    fn render_no_media_kind_for_documents() {
        use crate::message::Attachment;

        let mut msg = IncomingMessage::new(
            MessageSource::Http,
            "doc".into(),
            SourceMetadata::Http {
                remote_addr: None,
                user_agent: None,
            },
        );
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("file.pdf");
        std::fs::write(&path, b"pdf").unwrap();
        msg.attachments.push(Attachment {
            original_name: "file.pdf".into(),
            saved_path: path,
            mime_type: Some("application/pdf".into()),
            media_kind: crate::message::MediaKind::Document,
        });
        let processed = ProcessedMessage {
            enriched: EnrichedMessage {
                original: msg,
                urls: vec![],
                url_contents: vec![],
            },
            llm_response: None,
        };
        let result = render_org_node(&processed, tmp.path()).unwrap();
        assert!(
            !result.contains("MEDIA_KIND"),
            "MEDIA_KIND should not appear for document attachments: {result}"
        );
    }
}
