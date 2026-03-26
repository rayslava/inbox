use super::proxy::sanitize_filename;

#[test]
fn sanitize_filename_passes_safe_chars() {
    assert_eq!(sanitize_filename("report.pdf"), "report.pdf");
    assert_eq!(sanitize_filename("my-file_v2.txt"), "my-file_v2.txt");
}

#[test]
fn sanitize_filename_replaces_spaces_and_special() {
    assert_eq!(sanitize_filename("my file (1).pdf"), "my_file__1_.pdf");
}

#[test]
fn sanitize_filename_replaces_path_separators() {
    let result = sanitize_filename("../../etc/passwd");
    // Dots and dashes are preserved, slashes become underscores.
    assert_eq!(result, ".._.._etc_passwd");
}

#[test]
fn sanitize_filename_unicode_replaced() {
    // Non-ASCII alphanumeric chars are kept, but control chars and symbols are replaced.
    let result = sanitize_filename("café☕.txt");
    assert!(
        std::path::Path::new(&result)
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("txt"))
    );
    assert!(result.contains("caf"));
}
