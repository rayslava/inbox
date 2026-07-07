use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;

use super::{AttachContext, get_or_extract, index_file_attachments};
use crate::error::InboxError;
use crate::kb_index::extract::{ExtractionFingerprint, TextExtractor};
use crate::memory::MemoryStore;

/// What a `StubExtractor` yields, so tests can drive each `get_or_extract` arm.
#[derive(Clone)]
enum StubResult {
    Text(String),
    Empty,
    Fail,
}

/// A `TextExtractor` that returns a canned result and counts its calls, so tests
/// can assert extraction happened (or was served from cache) without real OCR.
struct StubExtractor {
    result: StubResult,
    calls: Arc<AtomicUsize>,
}

#[async_trait]
impl TextExtractor for StubExtractor {
    async fn extract(&self, _path: &std::path::Path) -> Result<Option<String>, InboxError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        match &self.result {
            StubResult::Text(t) => Ok(Some(t.clone())),
            StubResult::Empty => Ok(None),
            StubResult::Fail => Err(InboxError::Memory("stub failure".into())),
        }
    }

    fn fingerprint(&self) -> ExtractionFingerprint {
        ExtractionFingerprint {
            backend: "stub".into(),
            languages: "eng".into(),
            version: "v1".into(),
            vision_fallback: false,
        }
    }
}

fn ctx(text: &str, roots: Vec<PathBuf>) -> (AttachContext, Arc<AtomicUsize>) {
    ctx_with(StubResult::Text(text.to_owned()), roots)
}

fn ctx_with(result: StubResult, roots: Vec<PathBuf>) -> (AttachContext, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let extractor = Box::new(StubExtractor {
        result,
        calls: Arc::clone(&calls),
    });
    (AttachContext { extractor, roots }, calls)
}

/// A minimal parseable `Config` with the given extra TOML appended.
fn cfg_toml(extra: &str) -> crate::config::Config {
    let base = format!(
        "[general]\noutput_file = '/tmp/o.org'\nattachments_dir = '/tmp/att'\n\
         [llm]\n[memory]\nkb_root = '/tmp/kb'\n{extra}"
    );
    toml::from_str(&base).expect("valid config")
}

/// Create `<base>/attach-root/<id[0:2]>/<id[2:]>/<name>`; return the attach root.
fn attach_layout(base: &Path, id: &str, name: &str, body: &[u8]) -> PathBuf {
    let root = base.join("attach-root");
    let dir = root.join(&id[0..2]).join(&id[2..]);
    fs::create_dir_all(&dir).expect("mkdir");
    fs::write(dir.join(name), body).expect("write");
    root
}

#[test]
fn from_config_disabled_is_none() {
    let cfg = cfg_toml("[pipeline.attachment_extract]\nenabled = false\n");
    assert!(AttachContext::from_config(&cfg).is_none());
}

#[test]
fn from_config_shell_includes_default_and_extra_roots() {
    let cfg = cfg_toml("[pipeline.attachment_extract]\nenabled = true\nroots = ['/extra']\n");
    let context = AttachContext::from_config(&cfg).expect("enabled");
    assert!(
        context.roots.iter().any(|r| r.ends_with("att")),
        "attachments_dir"
    );
    assert!(context.roots.iter().any(|r| r.ends_with("kb")), "kb_root");
    assert!(
        context.roots.iter().any(|r| r.ends_with("extra")),
        "configured root"
    );
}

#[test]
fn from_config_unsupported_backend_is_none() {
    let cfg = cfg_toml("[pipeline.attachment_extract]\nenabled = true\nbackend = 'grpc'\n");
    assert!(AttachContext::from_config(&cfg).is_none());
}

#[tokio::test]
async fn indexes_attachment_under_owning_entry_id() {
    let id = "7b9b13fe-1440-48a9-b4ce-060d85958aa8";
    let dir = tempfile::tempdir().expect("dir");
    // org-attach id layout: <root>/<id[0:2]>/<id[2:]>/scan.png
    let root = dir.path().join("attach-root");
    let attach_dir = root.join(&id[0..2]).join(&id[2..]);
    fs::create_dir_all(&attach_dir).expect("mkdir");
    fs::write(attach_dir.join("scan.png"), b"\x89PNG fake").expect("write scan");

    let notes = dir.path().join("notes");
    fs::create_dir_all(&notes).expect("mkdir notes");
    let org = notes.join("note.org");
    let content = format!(
        "* Tax 2024\n:PROPERTIES:\n:ID: {id}\n:END:\nSee [[attachment:scan.png]] for the scan.\n"
    );
    fs::write(&org, &content).expect("write org");

    let store = MemoryStore::new_in_memory().expect("store");
    let (att_ctx, calls) = ctx("Zylophorbium quarterly amortization schedule", vec![root]);

    let n = index_file_attachments(&store, &att_ctx, &org, &content).await;
    assert!(n >= 1, "at least one attachment chunk indexed");
    assert_eq!(calls.load(Ordering::SeqCst), 1, "extractor ran once");

    let hits = store.kb_recall("Zylophorbium", 5).await.expect("recall");
    assert!(
        hits.iter()
            .any(|e| e.key.starts_with(&format!("kb:attachment:{id}:"))),
        "attachment chunk must be recallable under its owning id, got {:?}",
        hits.iter().map(|e| &e.key).collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn shared_file_extracted_once_indexed_under_each_note() {
    // Two entries with distinct ids both link the SAME file → extracted once,
    // indexed under both owning ids (cache-as-output behaviour).
    let dir = tempfile::tempdir().expect("dir");
    let notes = dir.path().join("notes");
    fs::create_dir_all(&notes).expect("mkdir");
    fs::write(notes.join("shared.png"), b"receipt").expect("write shared");
    let org = notes.join("note.org");
    let content = "\
* First
:PROPERTIES:
:ID: aaaa1111-0000-0000-0000-000000000000
:END:
Ref [[file:shared.png]].

* Second
:PROPERTIES:
:ID: bbbb2222-0000-0000-0000-000000000000
:END:
Also [[file:shared.png]].
";
    fs::write(&org, content).expect("write org");

    let store = MemoryStore::new_in_memory().expect("store");
    let (att_ctx, calls) = ctx("shared attachment body text", vec![notes.clone()]);

    let n = index_file_attachments(&store, &att_ctx, &org, content).await;
    assert!(n >= 2, "one chunk per owning note");
    assert_eq!(calls.load(Ordering::SeqCst), 1, "file extracted only once");

    let hits = store
        .kb_recall("shared attachment body", 10)
        .await
        .expect("recall");
    let keys: Vec<&String> = hits.iter().map(|e| &e.key).collect();
    assert!(
        keys.iter().any(|k| k.contains("aaaa1111")) && keys.iter().any(|k| k.contains("bbbb2222")),
        "indexed under both owning notes, got {keys:?}"
    );
}

#[tokio::test]
async fn no_attachments_indexes_nothing() {
    let dir = tempfile::tempdir().expect("dir");
    let org = dir.path().join("plain.org");
    let content = "* Just text\n:PROPERTIES:\n:ID: cccc3333\n:END:\nNo attachments here.\n";
    fs::write(&org, content).expect("write");

    let store = MemoryStore::new_in_memory().expect("store");
    let (att_ctx, calls) = ctx("unused", vec![dir.path().to_path_buf()]);

    let n = index_file_attachments(&store, &att_ctx, &org, content).await;
    assert_eq!(n, 0);
    assert_eq!(
        calls.load(Ordering::SeqCst),
        0,
        "no extraction when the entry has no attach-dir and no links"
    );
}

#[tokio::test]
async fn get_or_extract_skips_unsupported_extension_without_reading() {
    let dir = tempfile::tempdir().expect("dir");
    let zip = dir.path().join("archive.zip");
    fs::write(&zip, b"not a document").expect("write");
    let root = std::fs::canonicalize(dir.path()).expect("canon");

    let store = MemoryStore::new_in_memory().expect("store");
    let (att_ctx, calls) = ctx("unused", vec![]);
    let out = get_or_extract(
        &store,
        att_ctx.extractor.as_ref(),
        &zip,
        std::slice::from_ref(&root),
    )
    .await;
    assert!(out.is_none(), "unsupported type must be skipped");
    assert_eq!(
        calls.load(Ordering::SeqCst),
        0,
        "must not even reach the extractor"
    );
}

#[tokio::test]
async fn get_or_extract_rejects_path_outside_roots() {
    let dir = tempfile::tempdir().expect("dir");
    let img = dir.path().join("x.png");
    fs::write(&img, b"png").expect("write");

    let store = MemoryStore::new_in_memory().expect("store");
    let (att_ctx, calls) = ctx("unused", vec![]);
    // Empty canon_roots → the (supported) file is not under any root → rejected.
    let out = get_or_extract(&store, att_ctx.extractor.as_ref(), &img, &[]).await;
    assert!(
        out.is_none(),
        "out-of-root file must be rejected at extract time"
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        0,
        "rejected before extraction"
    );
}

#[tokio::test]
async fn indexes_attach_tag_file_without_inline_link() {
    // The real org-attach case: an `:ATTACH:`-tagged entry whose file sits in its
    // id-dir with NO inline `[[attachment:]]` link. It must still be indexed.
    let id = "cub-parking-approval";
    let dir = tempfile::tempdir().expect("dir");
    let root = attach_layout(dir.path(), id, "doc.pdf", b"%PDF fake");
    let org = dir.path().join("note.org");
    let content =
        format!("* Approval\n:PROPERTIES:\n:ID: {id}\n:END:\nNo inline link, just :ATTACH:.\n");
    fs::write(&org, &content).expect("write");

    let store = MemoryStore::new_in_memory().expect("store");
    let (att_ctx, calls) = ctx("Kinshicho parking application form", vec![root]);

    let n = index_file_attachments(&store, &att_ctx, &org, &content).await;
    assert!(n >= 1, "attach-dir file indexed without an inline link");
    assert_eq!(calls.load(Ordering::SeqCst), 1, "extracted once");

    let hits = store
        .kb_recall("Kinshicho parking", 5)
        .await
        .expect("recall");
    assert!(
        hits.iter()
            .any(|e| e.key.starts_with(&format!("kb:attachment:{id}:"))),
        "tag-attached file must be recallable under its owning id, got {:?}",
        hits.iter().map(|e| &e.key).collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn empty_extraction_indexes_nothing() {
    let id = "7b9b13fe-1440-48a9-b4ce-060d85958aa8";
    let dir = tempfile::tempdir().expect("dir");
    let root = attach_layout(dir.path(), id, "scan.png", b"bytes");
    let org = dir.path().join("note.org");
    let content = format!("* N\n:PROPERTIES:\n:ID: {id}\n:END:\n[[attachment:scan.png]]\n");
    fs::write(&org, &content).expect("write");

    let store = MemoryStore::new_in_memory().expect("store");
    let (att_ctx, calls) = ctx_with(StubResult::Empty, vec![root]);

    let n = index_file_attachments(&store, &att_ctx, &org, &content).await;
    assert_eq!(n, 0, "empty extraction yields no chunks");
    assert_eq!(calls.load(Ordering::SeqCst), 1, "extractor was tried once");
}

#[tokio::test]
async fn failed_extraction_degrades_without_error() {
    let id = "7b9b13fe-1440-48a9-b4ce-060d85958aa8";
    let dir = tempfile::tempdir().expect("dir");
    let root = attach_layout(dir.path(), id, "scan.png", b"bytes");
    let org = dir.path().join("note.org");
    let content = format!("* N\n:PROPERTIES:\n:ID: {id}\n:END:\n[[attachment:scan.png]]\n");
    fs::write(&org, &content).expect("write");

    let store = MemoryStore::new_in_memory().expect("store");
    let (att_ctx, _calls) = ctx_with(StubResult::Fail, vec![root]);

    // A hard extractor error must be swallowed (warn + skip), not propagated.
    let n = index_file_attachments(&store, &att_ctx, &org, &content).await;
    assert_eq!(n, 0, "failed extraction is skipped, file survives");
}

#[tokio::test]
async fn removed_link_cleans_up_orphan_chunks() {
    // A `[[file:]]` target in the note dir (NOT an org-attach id-dir), referenced
    // ONLY by the inline link — removing the link truly un-references it, so the
    // whole-file replace must clean up its chunk.
    let id = "7b9b13fe-1440-48a9-b4ce-060d85958aa8";
    let dir = tempfile::tempdir().expect("dir");
    let notes = dir.path().join("notes");
    fs::create_dir_all(&notes).expect("mkdir");
    fs::write(notes.join("extra.png"), b"data").expect("write");
    let org = notes.join("note.org");
    let with_link = format!("* N\n:PROPERTIES:\n:ID: {id}\n:END:\n[[file:extra.png]]\n");
    fs::write(&org, &with_link).expect("write");

    let store = MemoryStore::new_in_memory().expect("store");
    let (att_ctx, _) = ctx("Zylophorbium orphan check", vec![notes.clone()]);
    let n = index_file_attachments(&store, &att_ctx, &org, &with_link).await;
    assert!(n >= 1, "chunk indexed initially");

    // Re-index the SAME org file with the link removed → whole-file replace must
    // delete the now-orphaned attachment chunk.
    let without_link = format!("* N\n:PROPERTIES:\n:ID: {id}\n:END:\nno attachment now\n");
    let n2 = index_file_attachments(&store, &att_ctx, &org, &without_link).await;
    assert_eq!(n2, 0, "no chunks after link removed");

    let hits = store.kb_recall("Zylophorbium", 5).await.expect("recall");
    assert!(
        !hits.iter().any(|e| e.key.starts_with("kb:attachment:")),
        "removed-link chunk must not linger, got {:?}",
        hits.iter().map(|e| &e.key).collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn two_attachments_same_owner_get_distinct_chunks() {
    let id = "7b9b13fe-1440-48a9-b4ce-060d85958aa8";
    let dir = tempfile::tempdir().expect("dir");
    let root = attach_layout(dir.path(), id, "a.png", b"a");
    let adir = root.join(&id[0..2]).join(&id[2..]);
    fs::write(adir.join("b.png"), b"b").expect("write b");
    let org = dir.path().join("note.org");
    let content = format!(
        "* N\n:PROPERTIES:\n:ID: {id}\n:END:\n[[attachment:a.png]] and [[attachment:b.png]]\n"
    );
    fs::write(&org, &content).expect("write");

    let store = MemoryStore::new_in_memory().expect("store");
    // Same canned text for both files: distinct ids must come from the path.
    let (att_ctx, _) = ctx("identical Frobnitz body", vec![root]);
    let n = index_file_attachments(&store, &att_ctx, &org, &content).await;
    assert_eq!(
        n, 2,
        "two attachments → two distinct chunks, not a collision"
    );

    let hits = store.kb_recall("Frobnitz", 10).await.expect("recall");
    let keys: BTreeSet<&String> = hits
        .iter()
        .filter(|e| e.key.starts_with("kb:attachment:"))
        .map(|e| &e.key)
        .collect();
    assert_eq!(keys.len(), 2, "two distinct chunk ids, got {keys:?}");
}

#[tokio::test]
async fn failed_extraction_of_changed_file_cleans_stale_chunk() {
    // A cached (unchanged) file never re-fails; only a changed file re-extracts.
    // When that re-extraction fails, the whole-file replace omits it, so the
    // stale chunk is removed (self-heals on a later clean run) rather than
    // freezing a mixed generation vs the already-updated body.
    let id = "7b9b13fe-1440-48a9-b4ce-060d85958aa8";
    let dir = tempfile::tempdir().expect("dir");
    let root = attach_layout(dir.path(), id, "scan.png", b"v1");
    let org = dir.path().join("note.org");
    let content = format!("* N\n:PROPERTIES:\n:ID: {id}\n:END:\n[[attachment:scan.png]]\n");
    fs::write(&org, &content).expect("write");

    let store = MemoryStore::new_in_memory().expect("store");
    let (ok_ctx, _) = ctx("Keepsafe indexed body", vec![root.clone()]);
    index_file_attachments(&store, &ok_ctx, &org, &content).await;

    // Change the attachment bytes (cache miss), then re-index with a failing
    // extractor → the stale chunk is cleaned, not frozen.
    let scan = root.join(&id[0..2]).join(&id[2..]).join("scan.png");
    fs::write(&scan, b"v2 changed bytes").expect("rewrite");
    let (fail_ctx, _) = ctx_with(StubResult::Fail, vec![root]);
    let n = index_file_attachments(&store, &fail_ctx, &org, &content).await;
    assert_eq!(n, 0, "failed run indexes nothing");

    let hits = store.kb_recall("Keepsafe", 5).await.expect("recall");
    assert!(
        !hits.iter().any(|e| e.key.starts_with("kb:attachment:")),
        "stale chunk of a changed+failed file must be cleaned up, got {:?}",
        hits.iter().map(|e| &e.key).collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn second_run_uses_cache_no_re_extraction() {
    let id = "7b9b13fe-1440-48a9-b4ce-060d85958aa8";
    let dir = tempfile::tempdir().expect("dir");
    let root = dir.path().join("attach-root");
    let attach_dir = root.join(&id[0..2]).join(&id[2..]);
    fs::create_dir_all(&attach_dir).expect("mkdir");
    fs::write(attach_dir.join("scan.png"), b"bytes").expect("write");
    let org = dir.path().join("note.org");
    let content = format!("* N\n:PROPERTIES:\n:ID: {id}\n:END:\n[[attachment:scan.png]]\n");
    fs::write(&org, &content).expect("write org");

    let store = MemoryStore::new_in_memory().expect("store");
    let (att_ctx, calls) = ctx("cached body", vec![root]);

    index_file_attachments(&store, &att_ctx, &org, &content).await;
    index_file_attachments(&store, &att_ctx, &org, &content).await;
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "second run must hit the :KbSource cache, not re-extract"
    );
}
