//! `kb-web` — Phase 0 stub for the Second Mind cloud serving crate.
//!
//! Exists to prove the dependency gate: it builds against `inbox-core` alone,
//! never the `inbox` binary. Real public-anon / private-OIDC serving over the
//! read-only Grafeo files lands in a later phase.

fn main() {
    println!("{} [{}]", describe(), inbox_core::api_tag());
}

/// One-line description of what this crate will become.
fn describe() -> String {
    "kb-web (Phase 0 stub): serves read-only Grafeo files behind public/OIDC".to_string()
}

#[cfg(test)]
mod tests {
    use super::describe;

    #[test]
    fn describe_mentions_grafeo() {
        assert!(describe().contains("Grafeo"));
    }
}
