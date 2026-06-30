//! `omi-bridge` — Phase 0 stub for the omi capture bridge.
//!
//! Exists to prove the dependency gate: it builds against `inbox-core` alone,
//! never the `inbox` binary. Real STT + durable outbox → inbox `/ingest` lands
//! in a later phase.

fn main() {
    println!("{} [{}]", describe(), inbox_core::api_tag());
}

/// One-line description of what this crate will become.
fn describe() -> String {
    "omi-bridge (Phase 0 stub): STT + durable outbox → inbox /ingest".to_string()
}

#[cfg(test)]
mod tests {
    use super::describe;

    #[test]
    fn describe_mentions_ingest() {
        assert!(describe().contains("/ingest"));
    }
}
