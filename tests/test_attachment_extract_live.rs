//! Opt-in end-to-end check of the Slice-B attachment pieces against real local
//! services: `ShellExtractor` OCR (tesseract) → `:KbSource` extraction-output
//! cache → KB index (`embed_document`) → `kb_recall` (`embed_query`) over a real
//! llama.cpp `--embeddings` server. Proves the exact path the index wiring will
//! drive, including cross-lingual retrieval (RU query → EN scanned text).
//!
//! Requires BOTH `ATTACH_OCR_LIVE=1` (tesseract + `ImageMagick`) and
//! `LLAMACPP_EMBED_URL` (the `OpenAI` base). Run:
//!
//! ```text
//! ATTACH_OCR_LIVE=1 LLAMACPP_EMBED_URL=http://127.0.0.1:32002/v1 \
//!   cargo test --test test_attachment_extract_live -- --nocapture
//! ```

use std::hash::{Hash, Hasher};
use std::path::Path;

use inbox::config::{EmbeddingApi, MemoryConfig};
use inbox::kb_index;
use inbox::kb_index::extract::{ShellExtractor, TextExtractor};
use inbox::memory::MemoryStore;

async fn open_store(endpoint: String) -> (MemoryStore, tempfile::TempDir) {
    let cfg = MemoryConfig {
        enabled: true,
        embedding_endpoint: Some(endpoint),
        embedding_api: EmbeddingApi::Openai,
        embedding_model: Some("nomic-embed-text-v2-moe".into()),
        embedding_document_prefix: Some("search_document: ".into()),
        embedding_query_prefix: Some("search_query: ".into()),
        ..MemoryConfig::default()
    };
    let dir = tempfile::tempdir().expect("tempdir");
    let store = MemoryStore::open(&cfg, &dir.path().join("m.grafeo"))
        .await
        .expect("open store (is the embed server up?)");
    (store, dir)
}

/// Render `text` as a black-on-white PNG (a stand-in for a scanned page).
async fn render_scan(path: &Path, text: &str) {
    let status = tokio::process::Command::new("convert")
        .args([
            "-size",
            "900x160",
            "xc:white",
            "-gravity",
            "center",
            "-pointsize",
            "44",
            "-fill",
            "black",
            "-annotate",
            "+0+0",
            text,
            path.to_str().expect("utf8 path"),
        ])
        .status()
        .await
        .expect("convert runs");
    assert!(status.success(), "convert failed");
}

fn cheap_hash(bytes: &[u8]) -> String {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    bytes.hash(&mut h);
    format!("{:016x}", h.finish())
}

/// Assert the top hit is the attachment chunk (right `source` namespace) and
/// carries the OCR-derived text — not merely any chunk with the note id.
fn assert_top_is_ocr_attachment(hits: &[inbox::memory::MemoryEntry], prefix: &str, lang: &str) {
    let top = hits
        .first()
        .unwrap_or_else(|| panic!("{lang}: expected a hit"));
    assert!(
        top.key.starts_with(prefix),
        "{lang} top must be the attachment chunk, got {}",
        top.key
    );
    assert!(
        top.value.to_lowercase().contains("paris"),
        "{lang} recalled value must be the OCR text, got {:?}",
        top.value
    );
}

#[tokio::test]
async fn attachment_ocr_cache_index_and_recall_end_to_end() {
    let (Ok(endpoint), Ok(_)) = (
        std::env::var("LLAMACPP_EMBED_URL"),
        std::env::var("ATTACH_OCR_LIVE"),
    ) else {
        eprintln!(
            "skipping: set ATTACH_OCR_LIVE=1 and LLAMACPP_EMBED_URL \
             (e.g. http://127.0.0.1:32002/v1)"
        );
        return;
    };

    let dir = tempfile::tempdir().expect("tempdir");
    let scan = dir.path().join("scan.png");
    render_scan(&scan, "The capital of France is Paris").await;

    // 1) OCR the "scanned" page.
    let extractor = ShellExtractor::new("eng".into(), 8, 30);
    let text = extractor
        .extract(&scan)
        .await
        .expect("extract ok")
        .expect("image supported");
    eprintln!("OCR text: {text:?}");
    assert!(
        text.to_lowercase().contains("paris"),
        "OCR must read the scanned text, got: {text:?}"
    );

    let (store, _store_dir) = open_store(endpoint).await;

    // 2) Extraction-output cache: round-trip on an exact match, MISS on a changed
    //    file hash or extractor config (proves the key actually discriminates).
    let canonical = scan.canonicalize().expect("canonical");
    let canonical = canonical.to_string_lossy().into_owned();
    let file_hash = cheap_hash(&std::fs::read(&scan).expect("read bytes"));
    let fp = extractor.fingerprint().tag();
    store
        .kb_extract_cache_put(&canonical, &file_hash, &fp, &text, "2026-07-07T00:00:00Z")
        .await
        .expect("cache put");
    assert!(
        store
            .kb_extract_cache_get(&canonical, "different-hash", &fp)
            .await
            .is_none(),
        "changed file bytes must miss"
    );
    assert!(
        store
            .kb_extract_cache_get(&canonical, &file_hash, "shell|rus+jpn+eng|v1|false")
            .await
            .is_none(),
        "changed extractor config must miss"
    );
    let cached = store
        .kb_extract_cache_get(&canonical, &file_hash, &fp)
        .await
        .expect("exact match must hit");
    assert_eq!(
        cached, text,
        "cache must return the extracted text without re-OCR"
    );

    // 3) Index a same-fingerprint distractor plus the attachment chunk — and feed
    //    the CACHED text (not the local var), so recall depends on the cache path
    //    exactly as the future index wiring will.
    kb_index::index_content(
        &store,
        "attachment",
        "note-recipe",
        "/recipe.png",
        "A recipe for chocolate sponge cake with butter, eggs, and sugar.",
    )
    .await
    .expect("index distractor");
    let note_id = "note-tax-2024";
    kb_index::index_content(&store, "attachment", note_id, &canonical, &cached)
        .await
        .expect("index attachment");
    let want_prefix = format!("kb:attachment:{note_id}:");

    // 4a) English recall → the attachment chunk ranks first over the distractor,
    //     is in the `attachment` namespace, and carries the OCR-derived text.
    let en = store.kb_recall("capital of France", 5).await.expect("en");
    eprintln!(
        "EN hits: {:?}",
        en.iter().map(|e| &e.key).collect::<Vec<_>>()
    );
    assert_top_is_ocr_attachment(&en, &want_prefix, "EN");

    // 4b) Cross-lingual: a Russian query ranks the ENGLISH OCR'd attachment first,
    //     over the unrelated distractor — the v2-moe payoff, not a lexical match.
    let ru = store.kb_recall("столица Франции", 5).await.expect("ru");
    eprintln!(
        "RU hits: {:?}",
        ru.iter().map(|e| &e.key).collect::<Vec<_>>()
    );
    assert_top_is_ocr_attachment(&ru, &want_prefix, "RU");
}
