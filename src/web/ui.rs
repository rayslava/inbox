use std::collections::HashMap;
use std::path::Path;

use askama::Template;

// ── Template types ────────────────────────────────────────────────────────────

#[derive(Template)]
#[template(path = "login.html")]
pub struct LoginTemplate {
    pub error: Option<String>,
}

#[derive(Template)]
#[template(path = "inbox_ui.html")]
pub struct InboxUiTemplate {
    pub nodes: Vec<UiNode>,
    /// Full URL of the HTTP adapter inbox endpoint, e.g. `http://localhost:8080/inbox`.
    /// Empty if the HTTP adapter is disabled.
    pub inbox_url: String,
    /// Bearer token required by the HTTP adapter, or empty if auth is disabled.
    pub auth_token: String,
}

// ── Data types ────────────────────────────────────────────────────────────────

pub struct UiAttachment {
    pub html: String,
}

pub struct UiNode {
    pub title: String,
    pub created: String,
    pub source: String,
    pub tags: Vec<String>,
    pub summary: String,
    pub excerpt: Option<String>,
    pub attachments: Vec<UiAttachment>,
    pub search_text: String,
}

// ── Org parser ────────────────────────────────────────────────────────────────

/// Parse an org-mode file (as written by our template) into UI nodes.
#[must_use]
pub fn parse_org_nodes(content: &str, attachments_dir: &Path) -> Vec<UiNode> {
    let mut nodes = Vec::new();
    let mut current: Vec<&str> = Vec::new();

    for line in content.lines() {
        if line.starts_with("* ") && !current.is_empty() {
            if let Some(node) = parse_node(&current, attachments_dir) {
                nodes.push(node);
            }
            current.clear();
        }
        current.push(line);
    }
    if !current.is_empty()
        && let Some(node) = parse_node(&current, attachments_dir)
    {
        nodes.push(node);
    }
    nodes
}

fn parse_node(lines: &[&str], attachments_dir: &Path) -> Option<UiNode> {
    let header = lines.first()?;
    let after_star = header.strip_prefix("* ").unwrap_or(header);
    let (title, tags) = parse_headline(after_star);

    let mut props: HashMap<String, String> = HashMap::new();
    let mut in_props = false;
    let mut body: Vec<&str> = Vec::new();

    for line in lines.iter().skip(1) {
        if line.trim() == ":PROPERTIES:" {
            in_props = true;
            continue;
        }
        if line.trim() == ":END:" {
            in_props = false;
            continue;
        }
        if in_props {
            if let Some((k, v)) = parse_property(line) {
                props.insert(k, v);
            }
        } else {
            body.push(line);
        }
    }

    let created = props.remove("CREATED").unwrap_or_default();
    let source = props.remove("SOURCE").unwrap_or_default();

    let (summary, excerpt, attachments) = parse_body(&body, attachments_dir);
    let search_text = format!("{title} {source} {tags:?} {summary}");

    Some(UiNode {
        title,
        created,
        source,
        tags,
        summary,
        excerpt,
        attachments,
        search_text,
    })
}

fn parse_headline(header: &str) -> (String, Vec<String>) {
    // Tags are at the end: "Title :tag1:tag2:"
    if let Some(tags_start) = header.rfind(" :") {
        let possible_tags = &header[tags_start + 2..];
        if possible_tags.ends_with(':') && !possible_tags[..possible_tags.len() - 1].contains(' ') {
            let tags: Vec<String> = possible_tags[..possible_tags.len() - 1]
                .split(':')
                .filter(|t| !t.is_empty())
                .map(str::to_owned)
                .collect();
            let title = header[..tags_start].trim().to_owned();
            return (title, tags);
        }
    }
    (header.trim().to_owned(), Vec::new())
}

fn parse_property(line: &str) -> Option<(String, String)> {
    let trimmed = line.trim();
    if !trimmed.starts_with(':') {
        return None;
    }
    let rest = &trimmed[1..];
    let colon = rest.find(':')?;
    let key = rest[..colon].trim().to_owned();
    let value = rest[colon + 1..].trim().to_owned();
    Some((key, value))
}

fn parse_body(
    lines: &[&str],
    attachments_dir: &Path,
) -> (String, Option<String>, Vec<UiAttachment>) {
    let mut summary_lines: Vec<&str> = Vec::new();
    let mut in_quote = false;
    let mut quote_lines: Vec<&str> = Vec::new();
    let mut attachments = Vec::new();

    for line in lines {
        if line
            .trim_ascii_start()
            .eq_ignore_ascii_case("#+begin_quote")
        {
            in_quote = true;
            continue;
        }
        if line.trim_ascii_start().eq_ignore_ascii_case("#+end_quote") {
            in_quote = false;
            continue;
        }
        if in_quote {
            quote_lines.push(line);
        } else if let Some(att) = try_parse_org_link(line, attachments_dir) {
            attachments.push(att);
        } else {
            summary_lines.push(line);
        }
    }

    let summary = summary_lines.join("\n").trim().to_owned();
    let excerpt = if quote_lines.is_empty() {
        None
    } else {
        Some(quote_lines.join("\n").trim().to_owned())
    };

    (summary, excerpt, attachments)
}

fn try_parse_org_link(line: &str, attachments_dir: &Path) -> Option<UiAttachment> {
    // Match [[file:path][name]]
    let stripped = line.trim();
    let without_prefix = stripped.strip_prefix("[[file:")?;
    let (path_part, rest) = without_prefix.split_once("][")?;
    let name = rest.strip_suffix("]]")?.to_owned();

    let file_path = Path::new(path_part);
    let rel = file_path.strip_prefix(attachments_dir).map_or_else(
        |_| {
            file_path
                .file_name()
                .map(|f| f.to_string_lossy().into_owned())
                .unwrap_or_default()
        },
        |p| p.to_string_lossy().into_owned(),
    );

    // Prevent path traversal in generated URLs
    if rel.contains("..") {
        return None;
    }

    let url = format!("/attachments/{rel}");
    let mime = mime_guess::from_path(path_part).first_or_octet_stream();
    let mime_str = mime.essence_str();

    let html = if mime_str.starts_with("image/") {
        format!(r#"<img src="{url}" alt="{name}" loading="lazy" />"#)
    } else if mime_str.starts_with("audio/") {
        format!(r#"<audio controls src="{url}"></audio>"#)
    } else if mime_str.starts_with("video/") {
        format!(r#"<video controls src="{url}" style="max-width:100%"></video>"#)
    } else {
        format!(r#"<a href="{url}" class="doc-link">{name}</a>"#)
    };

    Some(UiAttachment { html })
}
