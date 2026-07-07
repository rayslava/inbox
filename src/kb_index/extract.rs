//! Extract text from attachment files (scanned images, PDFs) for KB indexing.
//! Backend-pluggable behind [`TextExtractor`] — a local `shell` extractor
//! (tesseract / pdftotext / pdftoppm) today, an HTTP OCR service later — keyed by
//! an [`ExtractionFingerprint`] so a config change (languages, backend, pipeline
//! version) invalidates the cached extraction output.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use tracing::warn;

use crate::error::InboxError;

/// Bump when the extraction pipeline changes in a way that alters its output.
pub const EXTRACT_VERSION: &str = "v1";

/// Identity of the extraction configuration; part of the `:KbSource` cache key,
/// so changing OCR languages / backend / pipeline version forces re-extraction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtractionFingerprint {
    pub backend: String,
    pub languages: String,
    pub version: String,
    pub vision_fallback: bool,
}

impl ExtractionFingerprint {
    /// Stable single-line tag stored alongside the cached extracted text.
    #[must_use]
    pub fn tag(&self) -> String {
        format!(
            "{}|{}|{}|{}",
            self.backend, self.languages, self.version, self.vision_fallback
        )
    }
}

/// Turns an attachment file into plain text for indexing.
#[async_trait]
pub trait TextExtractor: Send + Sync {
    /// Extract text from `path`. `Ok(None)` = unsupported file type (skip);
    /// `Err` = a hard failure the caller downgrades to a warning.
    async fn extract(&self, path: &Path) -> Result<Option<String>, InboxError>;

    /// The active extraction fingerprint (for cache invalidation).
    fn fingerprint(&self) -> ExtractionFingerprint;
}

/// Extraction strategy for a file, chosen by extension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FileKind {
    Pdf,
    Image,
    Unsupported,
}

/// True if `path` is a file type the extractor handles (image or PDF), by
/// extension. Lets a caller skip unsupported files **before** reading them.
#[must_use]
pub fn is_extractable(path: &Path) -> bool {
    !matches!(classify(path), FileKind::Unsupported)
}

/// Classify a file by (lowercased) extension.
fn classify(path: &Path) -> FileKind {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .unwrap_or_default();
    match ext.as_str() {
        "pdf" => FileKind::Pdf,
        "jpg" | "jpeg" | "png" | "tif" | "tiff" | "webp" | "bmp" | "gif" => FileKind::Image,
        _ => FileKind::Unsupported,
    }
}

/// Runs an external program with a timeout and captures stdout. The seam that
/// lets tests exercise the extractor's dispatch without real OCR binaries.
#[async_trait]
trait CommandRunner: Send + Sync {
    async fn run(
        &self,
        program: &str,
        args: &[&str],
        timeout_secs: u64,
    ) -> Result<String, InboxError>;
}

/// Real subprocess runner (spawns the program).
struct RealRunner;

#[async_trait]
impl CommandRunner for RealRunner {
    async fn run(
        &self,
        program: &str,
        args: &[&str],
        timeout_secs: u64,
    ) -> Result<String, InboxError> {
        run_capture(program, args, timeout_secs).await
    }
}

/// Local extractor shelling out to `tesseract` / `pdftotext` / `pdftoppm`.
pub struct ShellExtractor {
    /// tesseract language list, e.g. `"rus+jpn+eng"`.
    pub languages: String,
    /// A digital PDF's `pdftotext` output below this many chars is treated as a
    /// scanned page → rasterize + OCR fallback.
    pub min_chars: usize,
    /// Per-subprocess timeout.
    pub timeout_secs: u64,
    runner: Arc<dyn CommandRunner>,
}

impl ShellExtractor {
    /// Build a shell extractor that spawns the real OCR binaries.
    #[must_use]
    pub fn new(languages: String, min_chars: usize, timeout_secs: u64) -> Self {
        Self {
            languages,
            min_chars,
            timeout_secs,
            runner: Arc::new(RealRunner),
        }
    }

    async fn tesseract(&self, path: &Path) -> Result<String, InboxError> {
        let p = path_str(path)?;
        self.runner
            .run(
                "tesseract",
                &[p, "stdout", "-l", &self.languages],
                self.timeout_secs,
            )
            .await
    }

    async fn pdftotext(&self, path: &Path) -> Result<String, InboxError> {
        let p = path_str(path)?;
        self.runner
            .run("pdftotext", &[p, "-"], self.timeout_secs)
            .await
    }

    /// Rasterize a scanned PDF (`pdftoppm`) and OCR each page.
    async fn pdf_ocr(&self, path: &Path) -> Result<String, InboxError> {
        let tmp =
            tempfile::tempdir().map_err(|e| InboxError::Memory(format!("ocr tmpdir: {e}")))?;
        let prefix = tmp.path().join("pg");
        self.runner
            .run(
                "pdftoppm",
                &["-png", "-r", "150", path_str(path)?, path_str(&prefix)?],
                self.timeout_secs,
            )
            .await?;

        let mut pages: Vec<PathBuf> = std::fs::read_dir(tmp.path())
            .map_err(|e| InboxError::Memory(format!("ocr read pages: {e}")))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x.eq_ignore_ascii_case("png")))
            .collect();
        // Order by numeric page suffix, not lexically: even if a poppler build
        // doesn't zero-pad (`pg-2` vs `pg-10`), pages stay in reading order.
        pages.sort_by_key(|p| page_number(p));

        let mut text = String::new();
        for page in pages {
            text.push_str(&self.tesseract(&page).await?);
            text.push('\n');
        }
        Ok(text)
    }
}

#[async_trait]
impl TextExtractor for ShellExtractor {
    async fn extract(&self, path: &Path) -> Result<Option<String>, InboxError> {
        match classify(path) {
            FileKind::Unsupported => Ok(None),
            FileKind::Image => self.tesseract(path).await.map(Some),
            FileKind::Pdf => {
                let text = self.pdftotext(path).await?;
                // Whole-document heuristic: a real text layer of at least
                // `min_chars` → treat as digital; empty/near-empty (incl. the
                // `min_chars == 0` edge) → scanned, so rasterize + OCR. A mixed
                // digital+scanned PDF is OCR'd only if its text layer is thin.
                let trimmed = text.trim().chars().count();
                if trimmed > 0 && trimmed >= self.min_chars {
                    Ok(Some(text))
                } else {
                    self.pdf_ocr(path).await.map(Some)
                }
            }
        }
    }

    fn fingerprint(&self) -> ExtractionFingerprint {
        ExtractionFingerprint {
            backend: "shell".to_owned(),
            languages: self.languages.clone(),
            version: EXTRACT_VERSION.to_owned(),
            vision_fallback: false,
        }
    }
}

fn path_str(path: &Path) -> Result<&str, InboxError> {
    path.to_str()
        .ok_or_else(|| InboxError::Memory(format!("non-UTF-8 path: {}", path.display())))
}

/// Parse the trailing numeric page index from a `pdftoppm` page image name
/// (`pg-7.png`/`pg-07.png` → 7). Unparseable names sort first (0).
fn page_number(path: &Path) -> u32 {
    path.file_stem()
        .and_then(|s| s.to_str())
        .and_then(|s| s.rsplit(['-', '_']).next())
        .and_then(|n| n.parse().ok())
        .unwrap_or(0)
}

/// Run `program args…` with a timeout, returning stdout. A non-zero exit with
/// empty stdout is an error; a non-zero exit that still produced text (partial
/// OCR) is kept with a warning. Uses argv (no shell), so inputs aren't interpolated.
/// `kill_on_drop` ensures a timed-out child is `SIGKILLed` (not left running) when
/// the future is dropped.
async fn run_capture(
    program: &str,
    args: &[&str],
    timeout_secs: u64,
) -> Result<String, InboxError> {
    let child = tokio::process::Command::new(program)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| InboxError::Memory(format!("{program} exec error: {e}")))?;

    // On timeout the `wait_with_output` future (owning `child`) is dropped, and
    // `kill_on_drop(true)` kills+reaps the process — no leaked OCR subprocess.
    let output = tokio::time::timeout(Duration::from_secs(timeout_secs), child.wait_with_output())
        .await
        .map_err(|_| InboxError::Memory(format!("{program} timed out after {timeout_secs}s")))?
        .map_err(|e| InboxError::Memory(format!("{program} exec error: {e}")))?;

    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        if stdout.trim().is_empty() {
            return Err(InboxError::Memory(format!("{program} failed: {stderr}")));
        }
        warn!(
            program,
            "nonzero exit but produced text; keeping partial output"
        );
    }
    Ok(stdout)
}

#[cfg(test)]
mod tests;
