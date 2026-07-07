use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use super::{
    CommandRunner, EXTRACT_VERSION, ExtractionFingerprint, FileKind, RealRunner, ShellExtractor,
    TextExtractor, classify, page_number, path_str, run_capture,
};
use crate::error::InboxError;

/// One recorded subprocess invocation: `(program, args, timeout_secs)`.
type Call = (String, Vec<String>, u64);

/// A canned, *recording* runner. Captures every `(program, args, timeout)` so
/// tests can assert exact argv/timeout, materializes PDF page PNGs on `pdftoppm`
/// (deliberately **unpadded** `pg-1..pg-N` to stress numeric ordering), and can
/// echo the OCR'd page path back so page order is observable.
struct FakeRunner {
    responses: HashMap<String, String>,
    make_pages: usize,
    tesseract_per_path: bool,
    calls: Arc<Mutex<Vec<Call>>>,
}

#[async_trait]
impl CommandRunner for FakeRunner {
    async fn run(
        &self,
        program: &str,
        args: &[&str],
        timeout_secs: u64,
    ) -> Result<String, InboxError> {
        self.calls.lock().expect("lock").push((
            program.to_owned(),
            args.iter().map(|s| (*s).to_owned()).collect(),
            timeout_secs,
        ));
        if program == "pdftoppm" && self.make_pages > 0 {
            // args = ["-png", "-r", "150", <input>, <prefix>]; unpadded on purpose.
            let prefix = args[4];
            for i in 1..=self.make_pages {
                std::fs::write(format!("{prefix}-{i}.png"), b"png").expect("write page");
            }
        }
        if program == "tesseract" && self.tesseract_per_path {
            let name = Path::new(args[0])
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("?");
            return Ok(format!("ocr[{name}]"));
        }
        Ok(self.responses.get(program).cloned().unwrap_or_default())
    }
}

struct Fake {
    extractor: ShellExtractor,
    calls: Arc<Mutex<Vec<Call>>>,
}

impl Fake {
    fn calls(&self) -> Vec<Call> {
        self.calls.lock().expect("lock").clone()
    }
}

fn fake(
    min_chars: usize,
    responses: &[(&str, &str)],
    make_pages: usize,
    tesseract_per_path: bool,
) -> Fake {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let responses = responses
        .iter()
        .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
        .collect();
    let extractor = ShellExtractor {
        languages: "eng".to_owned(),
        min_chars,
        timeout_secs: 7,
        runner: Arc::new(FakeRunner {
            responses,
            make_pages,
            tesseract_per_path,
            calls: Arc::clone(&calls),
        }),
    };
    Fake { extractor, calls }
}

#[test]
fn classify_by_extension() {
    assert_eq!(classify(Path::new("a.pdf")), FileKind::Pdf);
    assert_eq!(classify(Path::new("A.PDF")), FileKind::Pdf);
    for img in [
        "a.jpg", "a.jpeg", "a.png", "a.tif", "a.tiff", "a.webp", "a.bmp", "a.gif",
    ] {
        assert_eq!(classify(Path::new(img)), FileKind::Image, "{img}");
    }
    assert_eq!(classify(Path::new("a.docx")), FileKind::Unsupported);
    assert_eq!(classify(Path::new("noext")), FileKind::Unsupported);
}

#[test]
fn page_number_parses_padded_and_unpadded() {
    assert_eq!(page_number(Path::new("/t/pg-7.png")), 7);
    assert_eq!(page_number(Path::new("/t/pg-07.png")), 7);
    assert_eq!(page_number(Path::new("/t/pg-10.png")), 10);
    assert_eq!(page_number(Path::new("/t/weird.png")), 0);
}

#[test]
fn fingerprint_tag_is_stable_and_language_sensitive() {
    let base = ExtractionFingerprint {
        backend: "shell".to_owned(),
        languages: "rus+jpn+eng".to_owned(),
        version: EXTRACT_VERSION.to_owned(),
        vision_fallback: false,
    };
    assert_eq!(
        base.tag(),
        format!("shell|rus+jpn+eng|{EXTRACT_VERSION}|false")
    );

    let other = ExtractionFingerprint {
        languages: "eng".to_owned(),
        ..base.clone()
    };
    assert_ne!(
        base.tag(),
        other.tag(),
        "language change must alter the tag"
    );
}

#[test]
fn shell_extractor_reports_its_fingerprint() {
    let x = ShellExtractor::new("eng".to_owned(), 10, 5);
    let fp = x.fingerprint();
    assert_eq!(fp.backend, "shell");
    assert_eq!(fp.languages, "eng");
    assert!(!fp.vision_fallback);
}

#[tokio::test]
async fn unsupported_type_yields_none() {
    let f = fake(10, &[], 0, false);
    let out = f
        .extractor
        .extract(Path::new("/tmp/whatever.docx"))
        .await
        .expect("ok");
    assert!(out.is_none(), "unsupported type must not shell out");
    assert!(f.calls().is_empty(), "no subprocess for unsupported type");
}

#[tokio::test]
async fn image_is_ocr_d_with_expected_argv() {
    let f = fake(10, &[("tesseract", "recognized image text")], 0, false);
    let out = f
        .extractor
        .extract(Path::new("scan.png"))
        .await
        .expect("ok");
    assert_eq!(out.as_deref(), Some("recognized image text"));

    let calls = f.calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].0, "tesseract");
    assert_eq!(calls[0].1, ["scan.png", "stdout", "-l", "eng"]);
    assert_eq!(calls[0].2, 7, "configured timeout is propagated");
}

#[tokio::test]
async fn digital_pdf_uses_pdftotext() {
    let f = fake(5, &[("pdftotext", "a real text layer here")], 0, false);
    let out = f.extractor.extract(Path::new("doc.pdf")).await.expect("ok");
    assert_eq!(out.as_deref(), Some("a real text layer here"));
    // Digital → no rasterize/OCR calls.
    assert_eq!(f.calls().len(), 1);
    assert_eq!(f.calls()[0].0, "pdftotext");
}

#[tokio::test]
async fn empty_text_layer_falls_back_to_ocr_even_when_min_chars_zero() {
    // min_chars == 0 must NOT disable fallback for an empty text layer.
    let f = fake(
        0,
        &[("pdftotext", "   \n  "), ("tesseract", "scanned")],
        1,
        false,
    );
    let out = f
        .extractor
        .extract(Path::new("scan.pdf"))
        .await
        .expect("ok")
        .expect("supported");
    assert!(
        out.contains("scanned"),
        "empty layer must trigger OCR: {out:?}"
    );
    assert!(
        f.calls().iter().any(|c| c.0 == "pdftoppm"),
        "must rasterize"
    );
}

#[tokio::test]
async fn scanned_pdf_ocrs_pages_in_numeric_order() {
    use std::fmt::Write as _;
    // 12 unpadded pages: lexical sort would put pg-10/11/12 before pg-2.
    let f = fake(100, &[("pdftotext", "")], 12, true);
    let out = f
        .extractor
        .extract(Path::new("scanned.pdf"))
        .await
        .expect("ok")
        .expect("pdf supported");

    let mut expected = String::new();
    for i in 1..=12 {
        writeln!(expected, "ocr[pg-{i}.png]").expect("write");
    }
    assert_eq!(out, expected, "pages must be OCR'd in reading order");
}

#[tokio::test]
async fn tesseract_rejects_non_utf8_path() {
    use std::os::unix::ffi::OsStrExt;
    let bad = std::ffi::OsStr::from_bytes(&[0xff, 0x66]);
    let f = fake(10, &[], 0, false);
    let err = f
        .extractor
        .tesseract(Path::new(bad))
        .await
        .expect_err("non-utf8");
    assert!(err.to_string().contains("non-UTF-8"), "{err}");
}

#[test]
fn path_str_accepts_utf8() {
    assert_eq!(path_str(Path::new("/a/b.png")).expect("utf8"), "/a/b.png");
}

#[tokio::test]
async fn real_runner_captures_stdout() {
    let out = RealRunner.run("printf", &["ok"], 5).await.expect("printf");
    assert_eq!(out, "ok");
}

#[tokio::test]
async fn run_capture_returns_stdout() {
    let out = run_capture("printf", &["hello"], 5).await.expect("printf");
    assert_eq!(out, "hello");
}

#[tokio::test]
async fn run_capture_errors_on_failure_with_empty_stdout() {
    // `false` exits non-zero and prints nothing → hard error.
    let err = run_capture("false", &[], 5).await.expect_err("false fails");
    assert!(err.to_string().contains("failed"), "{err}");
}

#[tokio::test]
async fn run_capture_keeps_partial_output_on_nonzero_exit() {
    // Non-zero exit that still produced text → kept with a warning.
    let out = run_capture("sh", &["-c", "printf partial; exit 1"], 5)
        .await
        .expect("partial kept");
    assert_eq!(out, "partial");
}

#[tokio::test]
async fn run_capture_times_out() {
    // `sleep 5` under a 1s timeout must return a timeout error, not hang.
    let err = run_capture("sleep", &["5"], 1).await.expect_err("timeout");
    assert!(err.to_string().contains("timed out"), "{err}");
}

#[tokio::test]
async fn run_capture_kills_timed_out_child() {
    // The child would create a marker after 2s; a 1s timeout must SIGKILL it
    // (kill_on_drop) so the marker never appears — proving no leaked process.
    let dir = tempfile::tempdir().expect("dir");
    let marker = dir.path().join("marker");
    let script = format!("sleep 2; touch {}", marker.display());
    let err = run_capture("sh", &["-c", &script], 1)
        .await
        .expect_err("timeout");
    assert!(err.to_string().contains("timed out"), "{err}");

    // Wait past when the child *would* have written the marker.
    tokio::time::sleep(std::time::Duration::from_millis(2500)).await;
    assert!(
        !marker.exists(),
        "timed-out child was not killed (marker leaked)"
    );
}

#[tokio::test]
async fn run_capture_missing_program_errors() {
    let err = run_capture("definitely-not-a-real-binary-xyz", &[], 5)
        .await
        .expect_err("spawn fails");
    assert!(err.to_string().contains("exec error"), "{err}");
}

/// Opt-in end-to-end OCR: generate a text image with `ImageMagick` `convert`, then
/// OCR it via `ShellExtractor`. Skipped unless `ATTACH_OCR_LIVE=1` (CI has no
/// tesseract/imagemagick), like the `LLAMACPP_*` live tests.
#[tokio::test]
async fn ocr_live_reads_text_from_generated_image() {
    if std::env::var("ATTACH_OCR_LIVE").is_err() {
        eprintln!("skipping: set ATTACH_OCR_LIVE=1 (needs tesseract + imagemagick)");
        return;
    }
    let dir = tempfile::tempdir().expect("dir");
    let img = dir.path().join("hello.png");
    let status = tokio::process::Command::new("convert")
        .args([
            "-size",
            "600x120",
            "xc:white",
            "-pointsize",
            "48",
            "-fill",
            "black",
            "-annotate",
            "+20+70",
            "Hello World",
            img.to_str().expect("path"),
        ])
        .status()
        .await
        .expect("convert runs");
    assert!(status.success(), "convert failed");

    let x = ShellExtractor::new("eng".to_owned(), 8, 30);
    let text = x
        .extract(&img)
        .await
        .expect("extract ok")
        .expect("image is supported");
    assert!(
        text.to_lowercase().contains("hello"),
        "OCR should read the text, got: {text:?}"
    );
}
