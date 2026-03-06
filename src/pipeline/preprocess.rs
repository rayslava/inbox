use tracing::debug;

use crate::config::{PreprocessingConfig, RuleAction, RuleCondition};
use crate::message::{IncomingMessage, MediaKind, ProcessingHints};

/// Run all configured pre-processing rules against `msg` and return the
/// accumulated [`ProcessingHints`].
///
/// Rules are evaluated in declaration order; all matching rules are applied
/// (not first-match-wins).
pub fn run_preprocessing(msg: &IncomingMessage, cfg: &PreprocessingConfig) -> ProcessingHints {
    let mut hints = ProcessingHints::default();

    for rule in &cfg.rules {
        if evaluate_condition(&rule.condition, msg, rule.threshold) {
            debug!(
                rule = %rule.name,
                condition = ?rule.condition,
                "Pre-processing rule matched"
            );
            apply_action(&rule.action, rule, &mut hints);
        }
    }

    hints
}

fn evaluate_condition(
    condition: &RuleCondition,
    msg: &IncomingMessage,
    threshold: Option<usize>,
) -> bool {
    match condition {
        RuleCondition::TextWordCountLt => {
            let count = msg.text.split_whitespace().count();
            threshold.is_none_or(|t| count < t)
        }
        RuleCondition::HasImageAttachment => msg
            .attachments
            .iter()
            .any(|a| a.media_kind == MediaKind::Image),
        RuleCondition::HasAttachment => !msg.attachments.is_empty(),
    }
}

fn apply_action(
    action: &RuleAction,
    rule: &crate::config::PreprocessingRule,
    hints: &mut ProcessingHints,
) {
    match action {
        RuleAction::ForceWebSearch => {
            hints.force_web_search = true;
            if let Some(hint) = &rule.llm_hint {
                hints.extra_llm_hints.push(hint.clone());
            }
        }
        RuleAction::AddTag => {
            if let Some(tag) = &rule.tag {
                let normalized = tag.to_lowercase();
                if !hints.suggested_tags.contains(&normalized) {
                    hints.suggested_tags.push(normalized);
                }
            }
            if let Some(hint) = &rule.llm_hint {
                hints.extra_llm_hints.push(hint.clone());
            }
        }
        RuleAction::AddLlmHint => {
            if let Some(hint) = &rule.llm_hint {
                hints.extra_llm_hints.push(hint.clone());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{PreprocessingConfig, PreprocessingRule, RuleAction, RuleCondition};
    use crate::message::{Attachment, IncomingMessage, MediaKind, MessageSource, SourceMetadata};

    fn make_msg(text: &str) -> IncomingMessage {
        IncomingMessage::new(
            MessageSource::Http,
            text.into(),
            SourceMetadata::Http {
                remote_addr: None,
                user_agent: None,
            },
        )
    }

    fn short_text_rule() -> PreprocessingRule {
        PreprocessingRule {
            name: "short_text".into(),
            condition: RuleCondition::TextWordCountLt,
            threshold: Some(5),
            action: RuleAction::ForceWebSearch,
            tag: None,
            llm_hint: Some("Short text — search the web.".into()),
        }
    }

    fn image_tag_rule() -> PreprocessingRule {
        PreprocessingRule {
            name: "image_tag".into(),
            condition: RuleCondition::HasImageAttachment,
            threshold: None,
            action: RuleAction::AddTag,
            tag: Some("image".into()),
            llm_hint: None,
        }
    }

    #[test]
    fn no_rules_returns_default_hints() {
        let msg = make_msg("hello world");
        let cfg = PreprocessingConfig { rules: vec![] };
        let hints = run_preprocessing(&msg, &cfg);
        assert!(!hints.force_web_search);
        assert!(hints.extra_llm_hints.is_empty());
        assert!(hints.suggested_tags.is_empty());
    }

    #[test]
    fn short_text_triggers_force_web_search() {
        let msg = make_msg("hi");
        let cfg = PreprocessingConfig {
            rules: vec![short_text_rule()],
        };
        let hints = run_preprocessing(&msg, &cfg);
        assert!(hints.force_web_search);
        assert_eq!(hints.extra_llm_hints, ["Short text — search the web."]);
    }

    #[test]
    fn long_text_does_not_trigger_short_text_rule() {
        let msg = make_msg("one two three four five six");
        let cfg = PreprocessingConfig {
            rules: vec![short_text_rule()],
        };
        let hints = run_preprocessing(&msg, &cfg);
        assert!(!hints.force_web_search);
    }

    #[test]
    fn image_attachment_triggers_add_tag() {
        let mut msg = make_msg("look at this");
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("photo.jpg");
        std::fs::write(&path, b"img").unwrap();
        msg.attachments.push(Attachment {
            original_name: "photo.jpg".into(),
            saved_path: path,
            mime_type: Some("image/jpeg".into()),
            media_kind: MediaKind::Image,
        });
        let cfg = PreprocessingConfig {
            rules: vec![image_tag_rule()],
        };
        let hints = run_preprocessing(&msg, &cfg);
        assert_eq!(hints.suggested_tags, ["image"]);
    }

    #[test]
    fn no_image_does_not_trigger_image_rule() {
        let msg = make_msg("text only");
        let cfg = PreprocessingConfig {
            rules: vec![image_tag_rule()],
        };
        let hints = run_preprocessing(&msg, &cfg);
        assert!(hints.suggested_tags.is_empty());
    }

    #[test]
    fn duplicate_suggested_tags_deduplicated() {
        let mut msg = make_msg("hi");
        let tmp = tempfile::tempdir().unwrap();
        for name in ["a.jpg", "b.jpg"] {
            let path = tmp.path().join(name);
            std::fs::write(&path, b"img").unwrap();
            msg.attachments.push(Attachment {
                original_name: name.into(),
                saved_path: path,
                mime_type: Some("image/jpeg".into()),
                media_kind: MediaKind::Image,
            });
        }
        // Two image rules with the same tag
        let rule1 = image_tag_rule();
        let mut rule2 = image_tag_rule();
        rule2.name = "image_tag2".into();
        let cfg = PreprocessingConfig {
            rules: vec![rule1, rule2],
        };
        let hints = run_preprocessing(&msg, &cfg);
        assert_eq!(hints.suggested_tags, ["image"]);
    }

    #[test]
    fn has_attachment_condition_matches_any_attachment() {
        let mut msg = make_msg("doc");
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("file.pdf");
        std::fs::write(&path, b"pdf").unwrap();
        msg.attachments.push(Attachment {
            original_name: "file.pdf".into(),
            saved_path: path,
            mime_type: Some("application/pdf".into()),
            media_kind: MediaKind::Document,
        });
        let rule = PreprocessingRule {
            name: "any_attachment".into(),
            condition: RuleCondition::HasAttachment,
            threshold: None,
            action: RuleAction::AddLlmHint,
            tag: None,
            llm_hint: Some("Message has an attachment.".into()),
        };
        let cfg = PreprocessingConfig { rules: vec![rule] };
        let hints = run_preprocessing(&msg, &cfg);
        assert_eq!(hints.extra_llm_hints, ["Message has an attachment."]);
    }

    #[test]
    fn suggested_tags_normalized_to_lowercase() {
        let mut msg = make_msg("photo");
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("x.png");
        std::fs::write(&path, b"img").unwrap();
        msg.attachments.push(Attachment {
            original_name: "x.png".into(),
            saved_path: path,
            mime_type: Some("image/png".into()),
            media_kind: MediaKind::Image,
        });
        let rule = PreprocessingRule {
            name: "img".into(),
            condition: RuleCondition::HasImageAttachment,
            threshold: None,
            action: RuleAction::AddTag,
            tag: Some("IMAGE".into()),
            llm_hint: None,
        };
        let cfg = PreprocessingConfig { rules: vec![rule] };
        let hints = run_preprocessing(&msg, &cfg);
        assert_eq!(hints.suggested_tags, ["image"]);
    }
}
