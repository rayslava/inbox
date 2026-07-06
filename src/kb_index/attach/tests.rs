use std::fs;
use std::path::PathBuf;

use super::{AttachLink, parse_attach_links, resolve_attachments};

#[test]
fn parses_attachment_and_file_links() {
    let text = "see [[attachment:tax.pdf]] and [[attachment:scan.jpg][a scan]] \
                plus [[file:/abs/note.txt]] and [[file:rel/doc.org]].";
    let links = parse_attach_links(text);
    assert!(links.contains(&AttachLink::Named("tax.pdf".to_owned())));
    assert!(links.contains(&AttachLink::Named("scan.jpg".to_owned())));
    assert!(links.contains(&AttachLink::File("/abs/note.txt".to_owned())));
    assert!(links.contains(&AttachLink::File("rel/doc.org".to_owned())));
}

#[test]
fn parse_ignores_empty_and_non_links() {
    assert!(parse_attach_links("no links here, just [[id:abc]] and text").is_empty());
    assert!(parse_attach_links("[[attachment:]]").is_empty());
}

/// Create `root/id[0:2]/id[2:]/name` with `body`; return (root, `note_dir`).
fn attach_layout(name: &str, id: &str, body: &str) -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().expect("dir");
    let root = dir.path().join("attach-root");
    let sub = root.join(&id[0..2]).join(&id[2..]);
    fs::create_dir_all(&sub).expect("mkdir");
    fs::write(sub.join(name), body).expect("write file");
    (dir, root)
}

#[test]
fn resolves_named_attachment_via_id_dir() {
    let id = "7b9b13fe-1440-48a9-b4ce-060d85958aa8";
    let (dir, root) = attach_layout("Guidelines.pdf", id, "pdf bytes");
    let note_dir = dir.path().join("notes");
    fs::create_dir_all(&note_dir).expect("mkdir notes");

    let links = vec![AttachLink::Named("Guidelines.pdf".to_owned())];
    let resolved = resolve_attachments(Some(id), &links, &note_dir, std::slice::from_ref(&root));
    assert_eq!(resolved.len(), 1);
    assert!(resolved[0].ends_with("Guidelines.pdf"));
}

#[test]
fn resolves_via_ancestor_data_dir() {
    // `<note_dir>/data/id[0:2]/id[2:]/name` (org-attach default id-dir relative).
    let id = "ab12cdef-0000-0000-0000-000000000000";
    let dir = tempfile::tempdir().expect("dir");
    let note_dir = dir.path().join("roam");
    let data = note_dir.join("data").join(&id[0..2]).join(&id[2..]);
    fs::create_dir_all(&data).expect("mkdir");
    fs::write(data.join("receipt.png"), "png").expect("write");

    let links = vec![AttachLink::Named("receipt.png".to_owned())];
    // Root is note_dir itself (so its `data/` subtree is confined).
    let resolved =
        resolve_attachments(Some(id), &links, &note_dir, std::slice::from_ref(&note_dir));
    assert_eq!(resolved.len(), 1, "found via data/ fallback");
}

#[test]
fn resolves_relative_file_link_under_root() {
    let dir = tempfile::tempdir().expect("dir");
    let note_dir = dir.path().join("notes");
    fs::create_dir_all(&note_dir).expect("mkdir");
    fs::write(note_dir.join("local.txt"), "hi").expect("write");

    let links = vec![AttachLink::File("local.txt".to_owned())];
    let resolved = resolve_attachments(None, &links, &note_dir, std::slice::from_ref(&note_dir));
    assert_eq!(resolved.len(), 1);
    assert!(resolved[0].ends_with("local.txt"));
}

#[test]
fn non_ascii_id_does_not_panic() {
    // A non-ASCII :ID: whose 2-byte prefix splits a char must not panic.
    let dir = tempfile::tempdir().expect("dir");
    let root = dir.path().to_path_buf();
    let links = vec![AttachLink::Named("x.pdf".to_owned())];
    let resolved = resolve_attachments(
        Some("Ыдентификатор"),
        &links,
        dir.path(),
        std::slice::from_ref(&root),
    );
    assert!(resolved.is_empty());
}

#[test]
fn named_link_with_path_parts_is_rejected() {
    // A real file exists under the root, but a Named link that tries to reach it
    // via an absolute or `../` path (bypassing the id dir) must not resolve.
    let id = "7b9b13fe-1440-48a9-b4ce-060d85958aa8";
    let (dir, root) = attach_layout("real.pdf", id, "x");
    // A sibling id dir with a secret the crafted name would try to reach.
    let secret = root.join("00").join("secret").join("s.pdf");
    fs::create_dir_all(secret.parent().unwrap()).expect("mkdir");
    fs::write(&secret, "secret").expect("write");

    let abs = format!("{}", secret.display());
    for bad in [
        AttachLink::Named("../secret/s.pdf".to_owned()),
        AttachLink::Named(abs),
        AttachLink::Named("sub/real.pdf".to_owned()),
    ] {
        let r = resolve_attachments(Some(id), &[bad], dir.path(), std::slice::from_ref(&root));
        assert!(r.is_empty(), "named link with path parts must be rejected");
    }
}

#[test]
fn directory_target_is_rejected() {
    // `[[file:.]]` resolves to the note dir (a directory) — must not be returned.
    let dir = tempfile::tempdir().expect("dir");
    let root = dir.path().to_path_buf();
    let links = vec![AttachLink::File(".".to_owned())];
    let resolved = resolve_attachments(None, &links, &root, std::slice::from_ref(&root));
    assert!(
        resolved.is_empty(),
        "a directory must not resolve as an attachment"
    );
}

#[test]
fn rejects_traversal_escape() {
    let dir = tempfile::tempdir().expect("dir");
    let root = dir.path().join("kb");
    fs::create_dir_all(&root).expect("mkdir");
    // A secret outside the root, reached via `..` from a file link.
    fs::write(dir.path().join("secret.txt"), "top secret").expect("write secret");

    let links = vec![AttachLink::File("../secret.txt".to_owned())];
    let resolved = resolve_attachments(None, &links, &root, std::slice::from_ref(&root));
    assert!(
        resolved.is_empty(),
        "path escaping the root must be rejected"
    );
}

#[test]
fn rejects_absolute_file_outside_roots() {
    let dir = tempfile::tempdir().expect("dir");
    let root = dir.path().join("kb");
    fs::create_dir_all(&root).expect("mkdir");
    let outside = dir.path().join("outside.txt");
    fs::write(&outside, "nope").expect("write");

    let links = vec![AttachLink::File(outside.to_string_lossy().into_owned())];
    let resolved = resolve_attachments(None, &links, &root, std::slice::from_ref(&root));
    assert!(
        resolved.is_empty(),
        "absolute path outside roots must be rejected"
    );
}
