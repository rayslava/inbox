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
    let n = index_file(&store, &path).await.expect("index_file");
    assert!(n >= 1);

    let hits = store.kb_recall("gamma", 5).await.expect("kb_recall");
    assert!(hits.iter().any(|e| e.key.starts_with("kb:org:id-42:")));
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
    let n = index_directory(&store, dir.path())
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
    let n = index_directory(&store, dir.path())
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
