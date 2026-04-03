use tempfile::NamedTempFile;
use uuid::Uuid;

use super::org_patcher::patch_entry;

fn write_temp(content: &str) -> NamedTempFile {
    let f = NamedTempFile::new().unwrap();
    std::fs::write(f.path(), content).unwrap();
    f
}

#[tokio::test]
async fn patch_single_entry() {
    let id = Uuid::new_v4();
    let content = format!(
        "* Old Title :inbox_pending:\n:PROPERTIES:\n:ID:       {id}\n:ENRICHED_BY: none\n:END:\n\nSome text.\n"
    );
    let f = write_temp(&content);
    let new = format!(
        "* New Title :tag1:\n:PROPERTIES:\n:ID:       {id}\n:ENRICHED_BY: openrouter\n:END:\n\nNew summary.\n"
    );

    let found = patch_entry(f.path(), id, &new).await.unwrap();
    assert!(found);
    let result = std::fs::read_to_string(f.path()).unwrap();
    assert!(result.contains("New Title"));
    assert!(!result.contains("Old Title"));
}

#[tokio::test]
async fn patch_entry_in_middle() {
    let id = Uuid::new_v4();
    let content = format!(
        "* First Entry\n:PROPERTIES:\n:ID:       aaaaaaaa-0000-0000-0000-000000000001\n:END:\n\nFirst.\n\
         * Target :inbox_pending:\n:PROPERTIES:\n:ID:       {id}\n:END:\n\nOld.\n\
         * Third Entry\n:PROPERTIES:\n:ID:       aaaaaaaa-0000-0000-0000-000000000003\n:END:\n\nThird.\n"
    );
    let f = write_temp(&content);
    let new = format!("* Target Fixed\n:PROPERTIES:\n:ID:       {id}\n:END:\n\nFixed.\n");

    let found = patch_entry(f.path(), id, &new).await.unwrap();
    assert!(found);

    let result = std::fs::read_to_string(f.path()).unwrap();
    assert!(result.contains("First Entry"), "first entry preserved");
    assert!(result.contains("Target Fixed"), "target replaced");
    assert!(!result.contains("Old."), "old content gone");
    assert!(result.contains("Third Entry"), "third entry preserved");
}

#[tokio::test]
async fn patch_entry_at_end() {
    let id = Uuid::new_v4();
    let content = format!(
        "* First\n:PROPERTIES:\n:ID:       aaaaaaaa-0000-0000-0000-000000000001\n:END:\n\nFirst.\n\
         * Last :inbox_pending:\n:PROPERTIES:\n:ID:       {id}\n:END:\n\nOld last.\n"
    );
    let f = write_temp(&content);
    let new = format!("* Last Fixed\n:PROPERTIES:\n:ID:       {id}\n:END:\n\nNew last.\n");

    let found = patch_entry(f.path(), id, &new).await.unwrap();
    assert!(found);
    let result = std::fs::read_to_string(f.path()).unwrap();
    assert!(result.contains("First"), "first preserved");
    assert!(result.contains("Last Fixed"), "last replaced");
    assert!(!result.contains("Old last."), "old content removed");
}

#[tokio::test]
async fn patch_entry_not_found() {
    let content =
        "* Some Entry\n:PROPERTIES:\n:ID:       aaaaaaaa-0000-0000-0000-000000000001\n:END:\n";
    let f = write_temp(content);
    let missing_id = Uuid::new_v4();

    let found = patch_entry(f.path(), missing_id, "replacement")
        .await
        .unwrap();
    assert!(!found);
    // File unchanged
    assert_eq!(std::fs::read_to_string(f.path()).unwrap(), content);
}

#[tokio::test]
async fn patch_entry_file_not_found() {
    let id = Uuid::new_v4();
    let path = std::path::Path::new("/tmp/nonexistent_inbox_test_12345.org");
    let found = patch_entry(path, id, "anything").await.unwrap();
    assert!(!found);
}

#[tokio::test]
async fn patch_empty_file() {
    let f = write_temp("");
    let id = Uuid::new_v4();
    let found = patch_entry(f.path(), id, "anything").await.unwrap();
    assert!(!found);
}
