use super::{CoreError, UrlContent, api_tag};

#[test]
fn api_tag_names_crate_and_phase() {
    let tag = api_tag();
    assert!(tag.starts_with("inbox-core/"));
    assert!(tag.contains("phase0"));
}

#[test]
fn core_error_display_carries_message() {
    let err = CoreError::VectorStore("grafeo down".to_string());
    assert_eq!(err.to_string(), "Vector store error: grafeo down");
}

#[test]
fn core_error_from_json_is_json_variant() {
    let bad = serde_json::from_str::<UrlContent>("not json").unwrap_err();
    let err: CoreError = bad.into();
    assert!(matches!(err, CoreError::Json(_)));
}

#[test]
fn url_content_round_trips_through_json() {
    let uc = UrlContent {
        url: "https://example.com".to_string(),
        text: "body".to_string(),
        page_title: Some("Title".to_string()),
        headings: vec!["H1".to_string(), "H2".to_string()],
    };
    let json = serde_json::to_string(&uc).unwrap();
    let back: UrlContent = serde_json::from_str(&json).unwrap();
    assert_eq!(back.url, uc.url);
    assert_eq!(back.headings, uc.headings);
    assert_eq!(back.page_title.as_deref(), Some("Title"));
}
