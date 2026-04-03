use askama::Template;

use crate::error::InboxError;
use crate::message::ProcessedMessage;
use crate::pipeline::url_extractor::extract_http_url_strings;

pub struct AttachmentRef<'a> {
    pub name: &'a str,
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
    pub roam_refs: &'a [String],
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

/// Tag added to org entries when LLM processing fell back to raw mode.
pub const PENDING_TAG: &str = "inbox_pending";

/// Merge user tags, pre-processing suggested tags, and LLM tags in priority order,
/// deduplicating case-insensitively. Appends [`PENDING_TAG`] when there is no LLM response.
fn merge_tags(msg: &ProcessedMessage) -> Vec<String> {
    let original = &msg.enriched.original;
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
    } else {
        all.push(PENDING_TAG.to_owned());
    }
    all
}

/// Resolve the fallback heading title using a 4-level priority chain.
fn fallback_title<'a>(
    msg: &'a ProcessedMessage,
    original: &'a crate::message::IncomingMessage,
) -> &'a str {
    use crate::message::MediaKind;
    msg.fallback_title
        .as_deref()
        .or_else(|| original.text.lines().find(|l| !l.trim().is_empty()))
        .or_else(|| {
            original
                .attachments
                .first()
                .and_then(|a| match a.media_kind {
                    MediaKind::Image => Some("Image"),
                    MediaKind::Audio => Some("Audio"),
                    MediaKind::Video => Some("Video"),
                    MediaKind::VoiceMessage => Some("Voice Message"),
                    MediaKind::Sticker => Some("Sticker"),
                    MediaKind::Animation => Some("Animation"),
                    _ => None,
                })
        })
        .unwrap_or("(untitled)")
}

/// Collect all URLs for the `:ROAM_REFS:` property from multiple sources,
/// deduplicating across the set.
fn build_roam_refs(
    urls: &[String],
    summary: &str,
    excerpt: Option<&str>,
    fallback_source_urls: &[String],
) -> Vec<String> {
    let mut roam_refs = urls.to_vec();
    let mut seen: std::collections::HashSet<String> = urls.iter().cloned().collect();
    let summary_urls = extract_http_url_strings(summary);
    let excerpt_urls = excerpt.map(extract_http_url_strings).unwrap_or_default();
    for url in summary_urls.into_iter().chain(excerpt_urls) {
        if seen.insert(url.clone()) {
            roam_refs.push(url);
        }
    }
    for url in fallback_source_urls {
        if seen.insert(url.clone()) {
            roam_refs.push(url.clone());
        }
    }
    roam_refs
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
        .map(|a| AttachmentRef {
            name: &a.original_name,
            path_rel: relative_path(&a.saved_path, attachments_dir),
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

    let merged_tags = merge_tags(msg);

    let summary_owned: String;
    let (title, summary, excerpt, backend) = if let Some(r) = &msg.llm_response {
        (
            r.title.as_str(),
            r.summary.as_str(),
            r.excerpt.as_deref(),
            r.produced_by.as_str(),
        )
    } else {
        let summary = if msg.fallback_tool_results.is_empty() {
            original.text.as_str()
        } else {
            summary_owned = msg
                .fallback_tool_results
                .iter()
                .map(|(_, text)| text.as_str())
                .collect::<Vec<_>>()
                .join("\n\n");
            &summary_owned
        };
        let title = fallback_title(msg, original);
        (title, summary, None, "none")
    };

    let roam_refs = build_roam_refs(&urls, summary, excerpt, &msg.fallback_source_urls);

    let tmpl = OrgNodeTemplate {
        title,
        tags: &merged_tags,
        id: &id_str,
        created: &created,
        source: &source,
        urls: &urls,
        roam_refs: &roam_refs,
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
mod tests;
#[cfg(test)]
mod tests_pending;
