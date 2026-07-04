//! The local "second brain": per-kind RAG over the store traits. Retrieves the
//! top entries for a question (behavioral memory, KB chunks, or a quota'd blend),
//! asks the LLM to answer from them, and returns a deterministic list of cited
//! note ids (parsed from the chunk ids, so the model can never invent a citation).

use crate::{CoreError, LlmBackend, MemoryEntry, VectorStore};

/// Which store(s) a brain query draws from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetrievalMode {
    /// Behavioral `:Memory` nodes only — what the daemon has learned/been told.
    MemoryOnly,
    /// KB chunks only (`:KbChunk`) — the RAG default over the note corpus.
    KbOnly,
    /// Both stores, capping the KB at `kb_quota` slots so behavioral memory is
    /// never fully crowded out. Tuned for vague queries, where blending an
    /// imprecise match from either store beats a single-store miss.
    Hybrid { kb_quota: usize },
}

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

/// Answer `question` under `mode`: retrieve up to `top_k` entries, have `llm`
/// answer from them, and cite any source notes.
///
/// # Errors
/// Returns [`CoreError`] if retrieval or the LLM call fails.
pub async fn answer(
    vs: &dyn VectorStore,
    llm: &dyn LlmBackend,
    question: &str,
    top_k: usize,
    mode: RetrievalMode,
) -> Result<Answer, CoreError> {
    let chunks = retrieve(vs, question, top_k, mode).await?;
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

/// Retrieve the entries feeding an answer under `mode`. For `Hybrid`, the KB is
/// capped at `kb_quota` (never more than `top_k`) and behavioral memory fills the
/// remaining slots, so a large KB can never fully displace memory.
async fn retrieve(
    vs: &dyn VectorStore,
    question: &str,
    top_k: usize,
    mode: RetrievalMode,
) -> Result<Vec<MemoryEntry>, CoreError> {
    match mode {
        RetrievalMode::MemoryOnly => vs.recall(question, top_k).await,
        RetrievalMode::KbOnly => vs.recall_kb(question, top_k).await,
        RetrievalMode::Hybrid { kb_quota } => {
            let cap = kb_quota.min(top_k);
            let mut out: Vec<MemoryEntry> = vs
                .recall_kb(question, top_k)
                .await?
                .into_iter()
                .take(cap)
                .collect();
            let remaining = top_k - out.len();
            if remaining > 0 {
                out.extend(vs.recall(question, remaining).await?);
            }
            Ok(out)
        }
    }
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
