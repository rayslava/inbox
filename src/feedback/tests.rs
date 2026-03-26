use super::*;

#[test]
fn rating_valid_range() {
    assert!(FeedbackRating::new(0).is_none());
    assert!(FeedbackRating::new(4).is_none());
    for v in 1..=3 {
        let r = FeedbackRating::new(v).unwrap();
        assert_eq!(r.value(), v);
    }
}

#[test]
fn rating_serde_roundtrip() {
    let r = FeedbackRating::new(2).unwrap();
    let json = serde_json::to_string(&r).unwrap();
    assert_eq!(json, "2");
    let back: FeedbackRating = serde_json::from_str(&json).unwrap();
    assert_eq!(back, r);
}

#[test]
fn rating_serde_rejects_out_of_range() {
    assert!(serde_json::from_str::<FeedbackRating>("0").is_err());
    assert!(serde_json::from_str::<FeedbackRating>("4").is_err());
}

#[test]
fn rating_display() {
    assert_eq!(FeedbackRating::new(1).unwrap().to_string(), "\u{2b50}");
    assert_eq!(
        FeedbackRating::new(3).unwrap().to_string(),
        "\u{2b50}\u{2b50}\u{2b50}"
    );
}

#[test]
fn feedback_request_deserialise() {
    let json = r#"{"message_id":"00000000-0000-0000-0000-000000000001","rating":3}"#;
    let req: FeedbackRequest = serde_json::from_str(json).unwrap();
    assert_eq!(req.rating.value(), 3);
    assert!(req.comment.is_none());
}

#[test]
fn feedback_request_with_comment() {
    let json = r#"{"message_id":"00000000-0000-0000-0000-000000000001","rating":1,"comment":"bad summary"}"#;
    let req: FeedbackRequest = serde_json::from_str(json).unwrap();
    assert_eq!(req.comment.as_deref(), Some("bad summary"));
}

#[test]
fn feedback_stats_default() {
    let stats = FeedbackStats::default();
    assert_eq!(stats.total, 0);
    assert_eq!(stats.by_rating, [0, 0, 0]);
    assert!((stats.average - 0.0).abs() < f64::EPSILON);
}
