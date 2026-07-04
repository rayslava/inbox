//! Opt-in live check of the OpenAI-compatible embeddings path against a real
//! llama.cpp `server` started with `--embeddings`. Skipped unless
//! `LLAMACPP_EMBED_URL` is set (e.g. `http://127.0.0.1:18001/v1`), mirroring the
//! `TEST_WITH_OLLAMA` convention — the default suite makes no real network
//! calls.
//!
//! Run: `LLAMACPP_EMBED_URL=http://127.0.0.1:18001/v1 cargo test --test \
//! test_llama_cpp_embed_live -- --nocapture`

use inbox::config::EmbeddingApi;
use inbox::memory::embed::EmbedClient;

#[tokio::test]
async fn openai_embeddings_against_real_llama_cpp() {
    let Ok(endpoint) = std::env::var("LLAMACPP_EMBED_URL") else {
        eprintln!("skipping: set LLAMACPP_EMBED_URL to run (e.g. http://127.0.0.1:18001/v1)");
        return;
    };

    let client = EmbedClient::new(endpoint, EmbeddingApi::Openai, "local".into(), None)
        .expect("build embed client");

    let a = client
        .embed("the capital of France is Paris")
        .await
        .expect("embed a");
    let b = client
        .embed("a completely unrelated sentence about turtles")
        .await
        .expect("embed b");

    eprintln!("embedding dims: {}", a.len());
    assert!(!a.is_empty(), "embedding must not be empty");
    assert_eq!(
        a.len(),
        b.len(),
        "same model must yield same-dimension vectors"
    );
    assert!(
        a.iter().any(|&x| x != 0.0),
        "embedding must not be all zeros"
    );
}
