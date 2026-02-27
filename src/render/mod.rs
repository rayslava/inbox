use askama::Template;

use crate::error::InboxError;
use crate::message::ProcessedMessage;

pub struct AttachmentRef<'a> {
    pub name: &'a str,
    pub path: &'a str,
    pub path_rel: &'a str, // relative to attachments_dir, for /attachments/* URL
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
}

/// Render a `ProcessedMessage` to an org-mode node string.
///
/// # Errors
/// Returns an error if the Askama template fails to render.
pub fn render_org_node(
    msg: &ProcessedMessage,
    attachments_dir: &std::path::Path,
) -> Result<String, InboxError> {
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
                path: Box::leak(path_str.into_owned().into_boxed_str()),
                path_rel: Box::leak(rel.into_boxed_str()),
            }
        })
        .collect();

    let (title, tags, summary, excerpt, backend) = match &msg.llm_response {
        Some(r) => (
            r.title.as_str(),
            r.tags.as_slice(),
            r.summary.as_str(),
            r.excerpt.as_deref(),
            r.produced_by.as_str(),
        ),
        None => (
            original.text.lines().next().unwrap_or("(untitled)"),
            &[][..],
            original.text.as_str(),
            None,
            "none",
        ),
    };

    let tmpl = OrgNodeTemplate {
        title,
        tags,
        id: &id_str,
        created: &created,
        source: &source,
        urls: &urls,
        attachments: &att_refs,
        llm_backend: backend,
        summary,
        excerpt,
        raw_text: &original.text,
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
                AttachmentRef { name: "a.pdf", path: "/p/a.pdf", path_rel: "a.pdf" },
                AttachmentRef { name: "b.jpg", path: "/p/b.jpg", path_rel: "b.jpg" },
            ],
            llm_backend: "mock",
            summary: "s",
            excerpt: None,
            raw_text: "",
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
}
