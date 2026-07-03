//! The local "second brain": KB-only RAG over the store traits. Retrieves the
//! top kb chunks for a question, asks the LLM to answer from them, and returns a
//! deterministic list of cited note ids (parsed from the chunk ids, so the model
//! can never invent a citation).

use crate::{CoreError, LlmBackend, VectorStore};

/// An answer plus the note ids it drew from.
#[derive(Debug, Clone)]
pub struct Answer {
    pub text: String,
    /// Deduped org-roam note ids of the retrieved chunks, in retrieval order.
    pub note_ids: Vec<String>,
}

impl Answer {
    /// Render as org: the answer followed by a `Sources:` list of `[[id:…]]`
    /// links. Returns just the text when there are no citations.
    #[must_use]
    pub fn to_org(&self) -> String {
        if self.note_ids.is_empty() {
            return self.text.clone();
        }
        let links = self
            .note_ids
            .iter()
            .map(|id| format!("- [[id:{id}]]"))
            .collect::<Vec<_>>()
            .join("\n");
        format!("{}\n\nSources:\n{links}", self.text)
    }
}

/// Answer `question` from the KB only: retrieve `top_k` chunks, have `llm`
/// answer from them, and cite the source notes.
///
/// # Errors
/// Returns [`CoreError`] if retrieval or the LLM call fails.
pub async fn answer(
    vs: &dyn VectorStore,
    llm: &dyn LlmBackend,
    question: &str,
    top_k: usize,
) -> Result<Answer, CoreError> {
    let chunks = vs.recall_kb(question, top_k).await?;
    if chunks.is_empty() {
        return Ok(Answer {
            text: "I couldn't find any relevant notes.".to_owned(),
            note_ids: Vec::new(),
        });
    }

    let context = chunks
        .iter()
        .map(|c| c.value.as_str())
        .collect::<Vec<_>>()
        .join("\n\n---\n\n");
    let system = "You answer questions using ONLY the provided notes. Be concise \
                  and factual. If the notes do not contain the answer, say so.";
    let user = format!("Notes:\n{context}\n\nQuestion: {question}");
    let (text, _model) = llm.complete_text(system, &user).await?;

    let mut note_ids = Vec::new();
    for c in &chunks {
        if let Some(nid) = note_id_from_kb_id(&c.key)
            && !note_ids.contains(&nid)
        {
            note_ids.push(nid);
        }
    }

    Ok(Answer { text, note_ids })
}

/// Parse the note id from a namespaced chunk id
/// `kb:<source>:<note-id>:<chunker>:<hash>`.
fn note_id_from_kb_id(id: &str) -> Option<String> {
    let mut parts = id.splitn(5, ':');
    if parts.next()? != "kb" {
        return None;
    }
    let _source = parts.next()?;
    let note_id = parts.next()?;
    (!note_id.is_empty()).then(|| note_id.to_owned())
}

#[cfg(test)]
mod tests {
    use super::{Answer, note_id_from_kb_id};

    #[test]
    fn parses_note_id_from_chunk_id() {
        assert_eq!(
            note_id_from_kb_id("kb:org:20230101-note:v1:abcd").as_deref(),
            Some("20230101-note")
        );
        assert_eq!(note_id_from_kb_id("memory:src:key"), None);
        assert_eq!(note_id_from_kb_id("kb:org::v1:h"), None);
    }

    #[test]
    fn to_org_appends_sources_list() {
        let a = Answer {
            text: "Paris.".to_owned(),
            note_ids: vec!["n1".to_owned(), "n2".to_owned()],
        };
        let org = a.to_org();
        assert!(org.starts_with("Paris."));
        assert!(org.contains("Sources:"));
        assert!(org.contains("[[id:n1]]"));
        assert!(org.contains("[[id:n2]]"));
    }

    #[test]
    fn to_org_without_citations_is_just_text() {
        let a = Answer {
            text: "No sources.".to_owned(),
            note_ids: vec![],
        };
        assert_eq!(a.to_org(), "No sources.");
    }
}
