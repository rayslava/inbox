use std::path::Path;

use super::{extract_note_id, index_content, index_directory, index_file};
use crate::memory::MemoryStore;

#[test]
fn extract_note_id_from_id_property() {
    let content = ":PROPERTIES:\n:ID:       abc-123\n:END:\n#+title: Note\n* H\nbody\n";
    assert_eq!(
        extract_note_id(content, Path::new("/x/note.org")),
        "abc-123"
    );
}

#[test]
fn extract_note_id_falls_back_to_filename() {
    let content = "#+title: Note\n* H\nbody\n";
    assert_eq!(
        extract_note_id(content, Path::new("/x/my-note.org")),
        "my-note"
    );
}

#[test]
fn extract_note_id_ignores_stray_id_outside_drawer() {
    // A `:ID:` in body text (not inside the file-level drawer) must not win.
    let content = "#+title: Note\n* H\nbody mentioning :ID: not-a-real-id\n";
    assert_eq!(extract_note_id(content, Path::new("/x/stem.org")), "stem");
}

#[tokio::test]
async fn index_content_stores_retrievable_chunks() {
    let store = MemoryStore::new_in_memory().expect("store");
    let content = "preamble about foxes\n* Animals\nthe quick brown fox\n* Plants\ngreen ferns\n";
    let n = index_content(&store, "org", "note1", "/note1.org", content)
        .await
        .expect("index_content");
    assert_eq!(n, 3, "preamble + 2 headings");

    let hits = store.kb_recall("fox", 5).await.expect("kb_recall");
    assert!(hits.iter().any(|e| e.value.contains("quick brown fox")));
    assert!(hits.iter().all(|e| e.key.starts_with("kb:org:note1:")));
}

#[tokio::test]
async fn index_content_reindex_replaces_stale_chunks() {
    let store = MemoryStore::new_in_memory().expect("store");
    index_content(&store, "org", "n", "/n.org", "* H\noldtoken content\n")
        .await
        .expect("index v1");
    // Re-index the same path with edited content → transactional replace.
    index_content(&store, "org", "n", "/n.org", "* H\nnewtoken content\n")
        .await
        .expect("index v2");

    assert!(
        store
            .kb_recall("newtoken", 5)
            .await
            .expect("recall")
            .iter()
            .any(|e| e.value.contains("newtoken")),
        "new content indexed"
    );
    assert!(
        store
            .kb_recall("oldtoken", 5)
            .await
            .expect("recall")
            .is_empty(),
        "stale chunk must be removed by the replace"
    );
}

#[tokio::test]
async fn index_file_reads_and_indexes_with_id() {
    let dir = tempfile::tempdir().expect("dir");
    let path = dir.path().join("note.org");
    tokio::fs::write(
        &path,
        ":PROPERTIES:\n:ID: id-42\n:END:\n* Topic\nunique gamma content\n",
    )
    .await
    .expect("write");

    let store = MemoryStore::new_in_memory().expect("store");
    let n = index_file(&store, &path, None).await.expect("index_file");
    assert!(n >= 1);

    let hits = store.kb_recall("gamma", 5).await.expect("kb_recall");
    assert!(hits.iter().any(|e| e.key.starts_with("kb:org:id-42:")));
}

fn write_cfg(dir: &Path, body: &str) -> std::path::PathBuf {
    let p = dir.join("config.toml");
    let toml = format!(
        "[general]\noutput_file = \"{out}\"\nattachments_dir = \"{att}\"\n\
         log_level = \"info\"\nlog_format = \"pretty\"\n[llm]\n{body}",
        out = dir.join("out.org").display(),
        att = dir.display(),
    );
    std::fs::write(&p, toml).expect("write cfg");
    p
}

#[tokio::test]
async fn index_corpus_indexes_configured_root() {
    let dir = tempfile::tempdir().expect("dir");
    let corpus = dir.path().join("corpus");
    std::fs::create_dir_all(&corpus).expect("mkdir");
    std::fs::write(
        corpus.join("note.org"),
        ":PROPERTIES:\n:ID: n1\n:END:\n* T\nunique kbcorpustoken\n",
    )
    .expect("write note");
    // No db_path → exercises the `{attachments_dir}/memory.grafeo` default.
    let cfg_path = write_cfg(
        dir.path(),
        &format!(
            "[memory]\nenabled = true\nkb_root = \"{root}\"\n",
            root = corpus.display(),
        ),
    );
    let cfg = crate::config::load(&cfg_path).expect("load cfg");
    let n = super::index_corpus(&cfg).await.expect("index_corpus");
    assert!(n >= 1, "should index the corpus note");
}

#[tokio::test]
async fn index_corpus_errors_without_kb_root() {
    let dir = tempfile::tempdir().expect("dir");
    let cfg_path = write_cfg(dir.path(), "[memory]\nenabled = true\n");
    let cfg = crate::config::load(&cfg_path).expect("load cfg");
    assert!(
        super::index_corpus(&cfg).await.is_err(),
        "missing kb_root must error"
    );
}

#[tokio::test]
async fn index_corpus_errors_on_nonexistent_kb_root() {
    let dir = tempfile::tempdir().expect("dir");
    let missing = dir.path().join("does-not-exist");
    let cfg_path = write_cfg(
        dir.path(),
        &format!(
            "[memory]\nenabled = true\ndb_path = \"{db}\"\nkb_root = \"{root}\"\n",
            db = dir.path().join("m.grafeo").display(),
            root = missing.display(),
        ),
    );
    let cfg = crate::config::load(&cfg_path).expect("load cfg");
    assert!(
        super::index_corpus(&cfg).await.is_err(),
        "a nonexistent kb_root must error, not silently index 0"
    );
}

#[tokio::test]
async fn index_corpus_errors_when_memory_disabled() {
    let dir = tempfile::tempdir().expect("dir");
    let cfg_path = write_cfg(
        dir.path(),
        &format!(
            "[memory]\nenabled = false\nkb_root = \"{r}\"\n",
            r = dir.path().display()
        ),
    );
    let cfg = crate::config::load(&cfg_path).expect("load cfg");
    assert!(
        super::index_corpus(&cfg).await.is_err(),
        "disabled memory must error"
    );
}

#[tokio::test]
async fn index_directory_indexes_org_and_skips_others() {
    let dir = tempfile::tempdir().expect("dir");
    tokio::fs::write(dir.path().join("a.org"), "* A\nalpha content\n")
        .await
        .expect("write a");
    tokio::fs::write(dir.path().join("b.org"), "* B\nbeta content\n")
        .await
        .expect("write b");
    // Skipped: encrypted note and a hidden file.
    tokio::fs::write(dir.path().join("secret.org.gpg"), "encrypted")
        .await
        .expect("write gpg");
    tokio::fs::write(dir.path().join(".hidden.org"), "* H\nhidden content\n")
        .await
        .expect("write hidden");

    let store = MemoryStore::new_in_memory().expect("store");
    let n = index_directory(&store, dir.path(), None)
        .await
        .expect("index_directory");
    assert_eq!(n, 2, "only a.org + b.org");

    let alpha = store.kb_recall("alpha", 5).await.expect("recall alpha");
    assert!(alpha.iter().any(|e| e.value.contains("alpha")));
    let hidden = store.kb_recall("hidden", 5).await.expect("recall hidden");
    assert!(hidden.is_empty(), "hidden file must not be indexed");
}

#[tokio::test]
async fn index_directory_skips_syncthing_artifacts() {
    let dir = tempfile::tempdir().expect("dir");
    tokio::fs::write(dir.path().join("real.org"), "* R\nrealtoken\n")
        .await
        .expect("write real");
    // Syncthing version history: a hidden `.stversions/` directory.
    let stver = dir.path().join(".stversions");
    tokio::fs::create_dir(&stver)
        .await
        .expect("mkdir stversions");
    tokio::fs::write(stver.join("real.org"), "* V\nversiontoken\n")
        .await
        .expect("write versioned");
    // Syncthing conflict copy.
    tokio::fs::write(
        dir.path()
            .join("real.sync-conflict-20260602-224449-DBLEJ2Q.org"),
        "* C\nconflicttoken\n",
    )
    .await
    .expect("write conflict");

    let store = MemoryStore::new_in_memory().expect("store");
    let n = index_directory(&store, dir.path(), None)
        .await
        .expect("index_directory");
    assert_eq!(
        n, 1,
        "only real.org — .stversions and sync-conflict excluded"
    );

    assert!(
        store
            .kb_recall("versiontoken", 5)
            .await
            .expect("recall")
            .is_empty(),
        ".stversions copy must not be indexed"
    );
    assert!(
        store
            .kb_recall("conflicttoken", 5)
            .await
            .expect("recall")
            .is_empty(),
        "sync-conflict copy must not be indexed"
    );
}
