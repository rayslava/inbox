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
    use async_trait::async_trait;
    use inbox_core::{CoreError, EmbeddingProvider};

    struct MockEmbedder;

    #[async_trait]
    impl EmbeddingProvider for MockEmbedder {
        async fn embed(&self, _text: &str) -> Result<Vec<f32>, CoreError> {
            Ok(vec![0.1, 0.2, 0.3])
        }
    }

    /// Embeds through any core `EmbeddingProvider` — the future RAG path drives
    /// the trait, never the concrete inbox client.
    async fn embed_query_dims(provider: &dyn EmbeddingProvider, query: &str) -> usize {
        provider.embed(query).await.map_or(0, |v| v.len())
    }

    #[test]
    fn describe_mentions_grafeo() {
        assert!(describe().contains("Grafeo"));
    }

    #[tokio::test]
    async fn embed_query_drives_core_trait() {
        assert_eq!(embed_query_dims(&MockEmbedder, "hello").await, 3);
    }
}
