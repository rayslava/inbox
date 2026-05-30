# Robust Image Processing in the Capture Pipeline (vision-LLM OCR)

## Overview
- Forwarded Telegram photos (and any image-only capture) currently fail enrichment and are written as an empty `* Image :inbox_failed:` node with `ENRICHED_BY: none`.
- This plan makes images first-class: a vision-capable LLM classifies the image (interface/screenshot vs photo) and, when interface-like, transcribes its visible text; the recognized text + classification flow into preprocessing/enrichment so a proper org node (title/tags/summary) is produced.
- It also closes the routing gap that caused the failure: the model pool is vision-blind, so an image request could be sent to a rate-limited model, or silently to a model with no vision capability, with no vision-aware fallback.
- Note: the `ollama` and `openrouter` backends already forward base64 `images` to their APIs (`src/llm/ollama.rs:254`, `src/llm/openrouter.rs:146`). The gap is purely a missing *vision-capability signal* — an image request may be sent to a text-only *model* that ignores the image. Tasks 1–3 add that signal; no image-passing code in ollama/openrouter needs fixing.
- **Decision:** vision-LLM OCR only — no local `tesseract`/`leptess`/`ocrs` dependency. "OCR" and classification are done by a vision model via a structured prompt.
- **Safety invariant:** an image-bearing message must NEVER render as a zero-content `:inbox_failed:` node. Worst case is a metadata-rich placeholder that is retried on the existing pending schedule.

## Context (from discovery)
Files/components involved (verified this session + corroborated by Codex):
- `src/llm/mod.rs:50` `LlmRequest::from_enriched()` — already base64-encodes `MediaKind::Image` attachments into `LlmRequest.images`, skips `> vision_max_bytes` (default 5 MiB), appends `vision_prompt_note`.
- `src/llm/openrouter.rs:146` `build_chat_messages()` — already emits OpenAI `image_url` data-URL parts when `req.images` is non-empty. **Vision payload path already exists.**
- `src/llm/free_router/pool.rs:80` / `src/llm/free_router.rs:140` — pool partitions ONLY by tool-call capability (`tool_models` / `general_models`); `candidate_models` selects by `needs_tools`. **Vision-blind.**
- `src/llm/chain.rs:115,146` — `LlmChain` tries every backend in order, then raw fallback. **Already 629 lines (> 500 limit)** — Task 3 must extract, not inline.
- `src/llm/ollama.rs:254` — already forwards base64 `images` to the Ollama API (image passing works; only model capability is the gap).
- `src/pipeline/mod.rs:112-116` — stage order: tags → preprocessing → enrichment → LLM → write; tracker insertion point.
- `src/pipeline/preprocess.rs:11,38-42` — rule engine; `HasImageAttachment`/`HasAttachment` conditions already exist.
- `src/pipeline/llm_stage.rs:148` — `processed_from_raw_fallback`; only generates a fallback title when `text.is_empty() && !tool_results.is_empty()`. Image-only msgs have neither → no title.
- `src/render/mod.rs:203` — already uses `fallback_tool_results` as the summary in raw-fallback rendering.
- `src/message.rs:10,85,153` — `IncomingMessage`, `RetryableMessage`, `Attachment`, `MediaKind`.
- `src/resume_task.rs:195,254` — patches pending node tag to `:inbox_failed:`; rebuilds `RetryableMessage` on resume.
- `src/config/llm.rs` — `LlmConfig` (`vision_max_bytes`, `vision_prompt_note` already present), `LlmBackendConfig`.
- `src/processing_status.rs` — `ProcessingStage` enum (`RunningLlm{..}` etc.).

Related patterns:
- Backend trait `LlmClient` in `src/llm/mod.rs` (`name`, `model`, `retries`, `complete`).
- `wiremock` for HTTP tests; `teloxide_tests::MockBot` for telegram; `TEST_WITH_OLLAMA=1` opt-in.
- `anodized::spec` contracts on non-trivial functions.

Dependencies identified: no new crates required (base64, serde_json, reqwest already present). Vision-LLM-only ⇒ zero new system binaries.

## Development Approach
- **Testing approach**: Regular (code first, then tests) per repo convention; every task ships its tests before the next starts.
- Complete each task fully before moving on; small focused changes.
- **CRITICAL: every task MUST include new/updated tests** (success + error/edge), in separate test files/modules per repo style.
- **CRITICAL: all tests pass before next task.**
- After each code change run the mandatory pipeline:
  - `cargo clippy --fix --all-features --allow-dirty --all-targets --workspace`
  - `cargo fix --all-features --allow-dirty --all-targets --workspace`
  - `cargo fmt --all`
  - then `cargo test --all-features --workspace`
- No `unwrap`/`expect`/`panic!` in production paths; no `#[allow]`; files < 500 lines; atomic modules.
- Keep `cargo tarpaulin` coverage ≥ 80%; newly added functions are never excluded.

## Testing Strategy
- **Unit tests**: required every task. Vision LLM interactions mocked via `wiremock` (OpenRouter `/api/v1/models` metadata + `/chat/completions`). No real API calls.
- **Pipeline/integration tests**: split into small meaningfully-named files under the relevant module's `tests` (e.g. `src/pipeline/image_analysis/tests.rs`), and `tests/` integration files kept small.
- **Telegram path**: `teloxide_tests::MockBot` where a forwarded photo flow is exercised.
- No e2e/UI suite in this project — N/A.

## Progress Tracking
- Mark `[x]` immediately when done; add ➕ for newly discovered tasks; ⚠️ for blockers.
- Update this file if scope changes during implementation.

## Solution Overview
Two layers, sequenced **safety-first**:

**Layer 1 — Never-empty safety (Tasks 1–4).** Independent of OCR quality. Vision-aware routing ensures image requests reach a vision model (or are deferred), and the raw-fallback path always yields a non-empty, metadata-rich node. After this layer the Evgeniya-class bug cannot recur even with zero recognized text.

**Layer 2 — Image understanding (Tasks 5–8).** A dedicated pre-preprocessing stage asks a vision model to classify (interface vs photo) and transcribe text, stored as structured `ImageAnalysisResult` on the message. Preprocessing and `from_enriched` consume it so the final node has a real title/tags/summary.

Key design decisions & rationale:
- **Structured field, not text mutation.** OCR/classification stored on `IncomingMessage`/`RetryableMessage` as `ImageAnalysisResult` with serde defaults → backward-compatible with already-persisted pending items; original capture text stays pristine.
- **Vision capability is a first-class routing dimension.** `candidate_models(needs_tools, needs_vision)` + `LlmClient::vision_supported()` prevent wasting attempts on text-only/non-vision models and prevent silent "image dropped" behavior.
- **Modality source = OpenRouter `/api/v1/models`.** The shir-man index does not expose `input_modalities`; we join by model id against OpenRouter model metadata (`architecture.input_modalities` contains `"image"`).
- **Degrade, don't fail.** Retryable (all vision backends rate-limited) ⇒ keep pending + non-empty placeholder. Terminal (not interface / no text) ⇒ proper node from metadata.

## Technical Details

### New data types (`src/message.rs`) — defined in full in **Task 4** (no later refinement)
```rust
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ImageAnalysisKind { Interface, Photo, Unknown }

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ImageAnalysisResult {
    pub attachment_name: String,   // original_name of the source attachment
    pub kind: ImageAnalysisKind,
    pub recognized_text: String,   // empty when none / not interface
    pub produced_by: String,       // backend:model that analyzed it
}
```
- New field `image_analyses: Vec<ImageAnalysisResult>` on `IncomingMessage` (runtime-only) and `RetryableMessage` (persisted).
- **Serde reality:** `IncomingMessage` is NOT `Serialize`/`Deserialize` (it holds `status_notifier: Option<Box<dyn StatusNotifier>>` and a hand-written `Debug`). So `#[serde(default)]` is inert there and applies only on the `RetryableMessage` field (the persisted path that needs backward-compat for already-stored pending rows). On `IncomingMessage` the field is plain runtime state and MUST be added to the manual `Debug` impl (`src/message.rs:38-50`).
- `From<&IncomingMessage> for RetryableMessage` (`src/message.rs:96`) clones it; `resume_task` round-trips it.

### New config (`src/config/pipeline.rs` + `src/config/llm.rs`)
- `[pipeline.image_analysis]`: `enabled` (default true), `prompt` (vision instruction), `max_attachments` (cap per message), `interface_min_chars` (heuristic threshold for classifying as interface when the model is ambiguous).
- `LlmBackendConfig.vision_supported: bool` (`#[serde(default)]`, default false) for pinned `openrouter`/`ollama`.

### Routing (`src/llm/free_router/pool.rs`, `free_router.rs`, `mod.rs`, `openrouter.rs`, `ollama.rs`)
- `FreeModel.supports_vision: bool`; `PoolState.vision_models: Vec<FreeModel>`.
- Pool build joins shir-man entries with OpenRouter `/api/v1/models` modality (optional fetch; failure ⇒ `supports_vision = false`, never panics).
- `candidate_models(needs_tools: bool, needs_vision: bool)`.
- `LlmClient::vision_supported(&self) -> bool` (default impl `false`; free_router computes per-request; openrouter/ollama from config).
- `LlmRequest` carries `needs_vision` (derived from non-empty `images`). Chain skips non-vision backends when `needs_vision && images_required` (no recognized text); when recognized text exists, non-vision backends may run with `images` stripped.

### Fallback hardening (`src/pipeline/llm_stage.rs`, `src/render/mod.rs`)
- `processed_from_raw_fallback`: if message has `image_analyses`, synthesize `fallback_tool_results` entries `("image_ocr", recognized_text)` and a deterministic `fallback_title` from the first non-empty recognized line; else emit a metadata summary (`Forwarded from …`, filename, MIME, `no text recognized`). Never empty for image-bearing messages.

### Stage wiring (`src/pipeline/mod.rs`, `src/processing_status.rs`)
- Add `ProcessingStage::AnalyzingImages`; insert tracker earlier; run `analyze_images` after tag extraction, before preprocessing; preprocessing may consult `image_analyses` (e.g. tag `interface`/`screenshot`).

## What Goes Where
- **Implementation Steps** (checkboxes): all code, config, tests in this repo.
- **Post-Completion** (no checkboxes): redeploy to k8s `inbox` ns; confirm a vision model is actually present in the live free pool; optional manual re-send of a screenshot to verify end-to-end.

## Implementation Steps

### Task 1: Add `vision_supported` to the backend trait and pinned-backend config ✅ DONE (04c05a2)

> Deviation: `needs_vision` implemented as a derived method (`!images.is_empty()`) instead of a stored field — single source of truth, can't desync when Task 3 strips images. `has_image_text` is the stored flag. Codex-approved.

**Files:**
- Modify: `src/llm/mod.rs` (trait `LlmClient`, `LlmRequest`)
- Modify: `src/llm/openrouter.rs`
- Modify: `src/llm/ollama.rs`
- Modify: `src/config/llm.rs` (`LlmBackendConfig.vision_supported`)
- Modify: `src/llm/chain/methods.rs` (`LlmRequest` literal at `:45` for the `llm_call` sub-request — will not compile without the new field)
- Modify: `src/llm/tests_builder.rs` (`LlmBackendConfig` literals at `:43`, `:69`)
- Modify: `src/llm/free_router/tests/fixtures.rs` (`LlmBackendConfig` literal at `:43`)
- Modify: `src/llm/tests.rs` or new `src/llm/tests_vision.rs`

- [ ] Add `fn vision_supported(&self) -> bool { false }` default to `LlmClient` trait in `src/llm/mod.rs`. (Trait default ⇒ existing impls compile unchanged; `OpenRouterClient`/`OllamaClient` override below. Note: `FreeRouterClient` override lands in Task 2.)
- [ ] Add `needs_vision: bool` to `LlmRequest`; set it in `from_enriched` (= `!images.is_empty()`), `simple` (= false), **and the `llm_call` sub-request literal in `src/llm/chain/methods.rs:45`** (= false). Grep for every `LlmRequest {` literal to be exhaustive.
- [ ] Add an explicit `has_image_text: bool` (or reuse a derived getter) to `LlmRequest` now, default false — set true in Task 7 once recognized text is fed in. This decouples Task 3's "skip non-vision unless text present" decision from data that only exists after Task 4/7. **Without this flag Task 3 cannot be implemented before Task 7.**
- [ ] Add `vision_supported: bool` (`#[serde(default)]`) to `LlmBackendConfig` in `src/config/llm.rs`; **update the struct literals in `src/llm/tests_builder.rs:43,:69` and `src/llm/free_router/tests/fixtures.rs:43`** (serde default does not help direct literals — they must add the field or use `..Default::default()`).
- [ ] Implement `vision_supported()` for `OpenRouterClient` and `OllamaClient` from their config flag.
- [ ] `#[spec]` where non-trivial; no `unwrap/expect`.
- [ ] Write tests: trait default returns false; openrouter/ollama reflect config flag; `LlmRequest.needs_vision` set correctly for image vs no-image; `has_image_text` default false.
- [ ] Run mandatory pipeline + tests — must pass before Task 2.

### Task 2: Vision-aware free_router pool partition ✅ DONE

> Codex review fixes folded in: fail-fast (no force-refresh) when a vision request hits a healthy pool with no vision models; capped the optional OpenRouter modality-fetch timeout (`VISION_FETCH_TIMEOUT_CAP = 5s`); added a `vision_models` info log + a regression test asserting zero refresh calls on the vision-empty path.

**Files:**
- Modify: `src/llm/free_router/pool.rs` (`FreeModel.supports_vision`, `PoolState.vision_models`, fetch/partition)
- Modify: `src/llm/free_router.rs` (`candidate_models` signature + `vision_supported`)
- Create: `src/llm/free_router/tests/vision_tests.rs` (+ register in test module)

- [ ] Add `supports_vision: bool` to `FreeModel`; `vision_models: Vec<FreeModel>` to `PoolState`; include in `is_empty`. **Do NOT seed `vision_models` in `degraded_fallback` (`src/llm/free_router/pool.rs:90`)** — the `openrouter/free` alias has no modality guarantee, so adding it to the vision pool would reintroduce blind image routing. Degraded vision pool stays empty (⇒ caller defers/skips, which is the safe behavior).
- [ ] Add optional OpenRouter `/api/v1/models` metadata fetch; join by model id; set `supports_vision` from `architecture.input_modalities` containing `"image"`. Fetch failure ⇒ all false (non-fatal, logged `warn`).
- [ ] Change `candidate_models(needs_tools)` → `candidate_models(needs_tools, needs_vision)`. **Preserve the existing tool partition:** current routing selects `tool_models` when tools are present (`src/llm/free_router.rs:338`), and `tool_models` require both tool flags (`src/llm/free_router/pool.rs:162`). So `candidate_models(true, true)` must return the **intersection** of vision-capable AND tool/tool_choice-capable models — not just `vision_models`. `(false, true)` ⇒ vision-only; `(true, false)` ⇒ unchanged tool path; `(false, false)` ⇒ unchanged general path.
- [ ] Implement `FreeRouterClient::vision_supported()` (true if pool has any vision model).
- [ ] Write `wiremock` tests: models-metadata join marks vision models; `candidate_models(true, true)` returns only models that are BOTH vision and tool capable; `candidate_models(false, true)` returns vision-capable; empty vision pool ⇒ no candidates; degraded fallback ⇒ empty vision pool (no blind routing); metadata fetch failure ⇒ graceful (no panic).
- [ ] Run mandatory pipeline + tests — must pass before Task 3.

### Task 3: Chain selects vision backends for image requests

**Files:**
- Create: `src/llm/chain/vision.rs` (backend-selection helper — keep `chain.rs` from growing past its already-over-limit 629 lines)
- Modify: `src/llm/chain.rs` (call the helper; do NOT inline the logic)
- Modify: `src/llm/openrouter.rs` / `src/llm/ollama.rs` (strip `images` before send when the invoked backend is non-vision)
- Create: `src/llm/chain/tests_vision.rs`

- [ ] **First** extract the vision backend-selection decision into `src/llm/chain/vision.rs` (e.g. `fn select_backends(...) -> impl Iterator` / `fn should_skip(backend, needs_vision, has_text) -> bool`); `chain.rs` only calls it. This keeps the already-non-conforming `chain.rs` from growing (CLAUDE.md < 500 lines; honor the `always-fix-preexisting-issues` rule by not worsening it).
- [ ] In `LlmChain::complete`, when `req.needs_vision`: skip backends whose `vision_supported()` is false **unless** `req.has_image_text` (the flag added in Task 1; set true in Task 7) — then run them with images stripped. Decision uses only request-level flags that exist as of Task 1, so Task 3 compiles and is testable before the analysis stage (Task 5/7) lands.
- [ ] Ensure both `build_chat_messages` (openrouter) and the ollama request builder omit `image_url`/`images` for a non-vision backend invocation (images stripped before send).
- [ ] Guarantee: if every backend is skipped (all non-vision, no text), chain returns `RawFallback` rather than hanging or sending a doomed request.
- [ ] Write tests: image request with only text-only backends + no text ⇒ raw fallback (no doomed call); with recognized text ⇒ text-only backend runs without images; vision backend present ⇒ used first; **ollama path strips images for a non-vision model** (not just openrouter).
- [ ] Run mandatory pipeline + tests — must pass before Task 4.

### Task 4: Harden raw fallback so image-only messages are never empty

**Files:**
- Modify: `src/message.rs` (define `ImageAnalysisResult`/`ImageAnalysisKind` in FULL here — see Technical Details; add field + Debug + From)
- Modify: `src/pipeline/llm_stage.rs` (`processed_from_raw_fallback`)
- Modify: `src/pipeline/tests.rs` or new `src/pipeline/tests_image_fallback.rs`

- [ ] Define the FINAL `ImageAnalysisResult` (fields `attachment_name`, `kind`, `recognized_text`, `produced_by`) and `ImageAnalysisKind` with the derives shown in Technical Details. **This is the complete type — Task 5 adds only logic, never redefines it** (avoids breaking this task's own tests).
- [ ] Add `image_analyses: Vec<ImageAnalysisResult>` to `RetryableMessage` with `#[serde(default)]` (backward-compat for already-persisted pending rows) and to `IncomingMessage` as runtime state.
- [ ] **Initialize the field in `IncomingMessage::with_id()` (`src/message.rs:60-77`)** (which `new()` delegates to) — adding the field without updating the constructor will not compile. Grep all `IncomingMessage {` literals to be exhaustive.
- [ ] Add the field to `IncomingMessage`'s manual `Debug` impl (`src/message.rs:38-50`) and clone it in `From<&IncomingMessage> for RetryableMessage` (`src/message.rs:96`).
- [ ] In `processed_from_raw_fallback` (`src/pipeline/llm_stage.rs:148`): build `fallback_tool_results` from recognized text (`("image_ocr", text)`) and a deterministic `fallback_title` from the first non-empty line. These flow through render's existing consumption (`fallback_tool_results` → summary at `src/render/mod.rs:203`; `fallback_title` at `:214`), so render needs **no change** in the common path — it already emits `"Image"` for the title (`render/mod.rs:92`); we are replacing the empty *body*.
- [ ] When no recognized text but attachments present: synthesize a metadata summary (forwarded source, filename, MIME, `no text recognized`) via `fallback_tool_results` and a metadata `fallback_title`.
- [ ] Assert invariant in a test: any message with ≥1 image attachment yields non-empty title AND summary after raw fallback.
- [ ] Write tests: image + text ⇒ OCR title/summary; image + no text ⇒ metadata title/summary; no attachments ⇒ unchanged behavior; `RetryableMessage` round-trips `image_analyses`.
- [ ] Run mandatory pipeline + tests — must pass before Task 5.

> **Milestone after Task 4:** the Evgeniya-class empty `:inbox_failed:` node is impossible regardless of OCR success. Layer 2 adds the actual understanding.

### Task 5: Image-analysis module (vision-LLM classify + transcribe)

**Files:**
- Create: `src/pipeline/image_analysis/mod.rs` (analysis orchestration)
- Create: `src/pipeline/image_analysis/classify.rs` (interface heuristic over model output)
- Create: `src/pipeline/image_analysis/tests.rs`
- Modify: `src/llm/chain/methods.rs` (new raw image-bearing call method — see first checkbox)
- Modify: `src/llm/mod.rs` (expose the new method on the `LlmChain` surface used by the stage)
- Modify: `src/config/pipeline.rs` (`[pipeline.image_analysis]` config)

> Note: `ImageAnalysisResult`/`ImageAnalysisKind` are already defined in Task 4. This task adds analysis *logic only* — it does not touch `src/message.rs` type definitions.

- [ ] **Add a chain method for raw image analysis.** Existing `LlmChain::complete_text` builds `LlmRequest::simple` and cannot attach images (`src/llm/chain/methods.rs:14`); `LlmChain::complete` expects the inbox `LlmResponse` JSON contract. Neither fits. Add e.g. `complete_vision_text(images, prompt) -> Option<(String, String)>` (text + producer) that builds an image-bearing `LlmRequest` with `needs_vision = true` and returns raw transcription text, routed through the vision-aware backend selection from Tasks 2–3.
- [ ] Implement `analyze_image(client, cfg, attachment) -> Option<ImageAnalysisResult>`: read+encode the image, call `complete_vision_text` with the structured classify+transcribe prompt, parse into the struct.
- [ ] Implement interface heuristic in `classify.rs` — **keep minimal (YAGNI):** `interface_min_chars` + line count to set/repair `kind` when the model is ambiguous. Defer UI-vocabulary/filename heuristics until a real misclassification is observed.
- [ ] Respect `max_attachments`, `vision_max_bytes`; skip non-image attachments; all failures non-fatal (`None`, logged).
- [ ] No `unwrap/expect`; `#[spec]` on non-trivial fns; keep each file < 500 lines.
- [ ] Write tests (wiremock vision response): interface image ⇒ `Interface` + text; plain photo ⇒ `Photo`, empty text; malformed/empty model output ⇒ `None`/`Unknown` without panic; oversize/non-image skipped.
- [ ] Run mandatory pipeline + tests — must pass before Task 6.

### Task 6: Wire the stage into the pipeline before preprocessing

**Files:**
- Modify: `src/pipeline/mod.rs` (stage ordering, call `analyze_images`)
- Modify: `src/processing_status.rs` (`ProcessingStage::AnalyzingImages`)
- Modify: `src/adapters/telegram_notifier/mod.rs` (**exhaustive `ProcessingStage` match at `:67` — adding the variant breaks compilation until this arm is handled**)
- Modify: `src/pipeline/preprocess.rs` (optional: consult `image_analyses` for `interface`/`screenshot` tags)
- Modify: `src/pipeline/tests.rs` (+ small new file if needed)

- [ ] Add `ProcessingStage::AnalyzingImages`; **grep all `match` sites on `ProcessingStage` and update each exhaustively — at minimum the Telegram notifier at `src/adapters/telegram_notifier/mod.rs:67`** (provide a user-facing status string consistent with the other arms).
- [ ] Move the tracker `insert` (`src/pipeline/mod.rs:130`) earlier (before the new stage) so it is observable, and advance it via the existing `run_stage` wrapper pattern used for `Enriching` (`src/pipeline/mod.rs:136-143`).
- [ ] In `Pipeline::process()` run `analyze_images` after user-tag extraction (currently ~`:116`), before preprocessing (`:123`); populate `image_analyses`; gate on `[pipeline.image_analysis].enabled`.
- [ ] Optional preprocessing rule: tag `interface`/`screenshot` when an analysis is `Interface`.
- [ ] Write tests: stage populates `image_analyses`; disabled config ⇒ skipped; preprocessing tag applied for interface; status tracker advances through `AnalyzingImages`.
- [ ] Run mandatory pipeline + tests — must pass before Task 7.

### Task 7: Feed recognized text into `from_enriched` and the enrichment prompt

**Files:**
- Modify: `src/llm/mod.rs` (`from_enriched`)
- Modify: `src/pipeline/llm_stage.rs` (`build_llm_guidance` if needed)
- Modify: `src/llm/tests.rs` (+ targeted file)

- [ ] In `from_enriched`, append recognized image text to `user_content` (clearly delimited, e.g. `--- Image text: <name> ---`) and set `has_image_text = true` when any recognized text exists (the flag Task 1 added, consumed by Task 3).
- [ ] `from_enriched` continues to populate `images` whenever image attachments exist (it has no backend/chain visibility). **Per-backend image stripping is NOT done here** — it lives in `LlmChain`/the backend invocation (Task 3), which is the only place that knows which backend is being called. Do not add backend logic to `from_enriched`.
- [ ] Keep `vision_prompt_note` added when `images` is non-empty (unchanged).
- [ ] Write tests: recognized text appears in `user_content`; `has_image_text` set when text present; `images` still populated from attachments; forwarded attribution preserved. (Backend-specific stripping is tested in Task 3, not here.)
- [ ] Run mandatory pipeline + tests — must pass before Task 8.

### Task 8: Pending/resume preserves analysis + retryable-vs-terminal distinction

**Files:**
- Modify: `src/message.rs` or `src/pipeline/llm_stage.rs` (carry a terminal/retryable signal on `ProcessedMessage`)
- Modify: `src/pipeline/mod.rs` (`:162` — the `llm_response.is_none()` ⇒ pending decision)
- Modify: `src/render/mod.rs` (`:85` — same is_none ⇒ `:inbox_pending:` tagging)
- Modify: `src/resume_task.rs` (round-trip `image_analyses`; honor the terminal flag)
- Modify: `src/pending/store.rs` if serialization needs the new field (serde default should suffice)
- Modify: `src/resume_task.rs` tests / `src/pending/tests.rs`

- [ ] Ensure `RetryableMessage` persists/restores `image_analyses` (serde default keeps old rows loadable).
- [ ] **Add an explicit terminal/retryable signal** — currently *every* `llm_response.is_none()` is tagged pending and persisted (`src/render/mod.rs:85`, `src/pipeline/mod.rs:162`), so resume-only changes cannot stop a terminal photo/no-text node from being queued forever. Add a field on `ProcessedMessage` (e.g. `fallback_terminal: bool`): terminal when image analysis ran and produced a usable metadata/OCR node; retryable when vision was unavailable/rate-limited. Branch the pending-vs-final decision at both `:162` and `:85` on it.
- [ ] **Set `fallback_terminal` in BOTH `processed_from_success` and `processed_from_raw_fallback`** (`src/pipeline/llm_stage.rs`) — success path = terminal; fallback path = per the rule above. **Grep and update all `ProcessedMessage {` struct literals across tests/helpers** (adding the field breaks them, same as the `LlmRequest` literals in Task 1).
- [ ] Retryable (vision temporarily unavailable / rate-limited) stays `:inbox_pending:` and retries; terminal renders a proper node, tagged final (not pending), not re-queued.
- [ ] Confirm exhausted-retry path renders the metadata-rich node, never zero-content `Image`.
- [ ] Write tests: round-trip preserves analyses; retryable item re-queued with context; terminal item finalized non-empty AND not pending; backward-compat load of a pre-existing pending row without the field.
- [ ] Run mandatory pipeline + tests — must pass before Task 9.

### Task 9: Update `config.example.toml` and docs

**Files:**
- Modify: `config.example.toml`
- Modify: `CLAUDE.md` (note image-analysis stage + vision routing, if a new pattern)

- [ ] Add `[pipeline.image_analysis]` keys with comments and defaults.
- [ ] Add `vision_supported` to the `[[llm.backends]]` openrouter/ollama examples (commented, default false) + note free_router auto-detects vision via OpenRouter model metadata.
- [ ] Note the never-empty image invariant in CLAUDE.md architecture/LLM notes.
- [ ] No test needed (config/docs) — but run `cargo test` to confirm config still parses (add a parse test for the new section).
- [ ] Run mandatory pipeline + tests — must pass before Task 10.

### Task 10: Verify acceptance criteria
- [ ] A forwarded image-only Telegram message produces a non-empty node with real title/tags/summary (interface case) or metadata node (photo/no-text case) — never `:inbox_failed:` empty.
- [ ] Image requests are routed to a vision-capable model when one is available; text-only backends are not sent doomed image requests.
- [ ] All-vision-rate-limited ⇒ pending retry with non-empty placeholder.
- [ ] Run full suite: `cargo test --all-features --workspace`.
- [ ] `cargo tarpaulin --all-features --workspace --out lcov` → coverage ≥ 80%, no new untested functions (`cargo crap --workspace --lcov lcov.info --format markdown`).
- [ ] `cargo clippy --all-features --all-targets --workspace` clean; no `#[allow]`; no `unwrap/expect/panic` in production.

### Task 11: [Final] Documentation & archive
- [ ] Update README/CLAUDE.md if patterns changed.
- [ ] Move this plan to `docs/plans/completed/`.

## Post-Completion
*Manual / external — no checkboxes.*

**Manual verification:**
- Redeploy daemon to k8s `inbox` namespace; confirm the live free_router pool actually contains ≥1 vision model (log line at pool init/refresh).
- Re-send the original screenshot (or a similar UI screenshot) via Telegram forward; confirm the resulting org node has recognized text + sensible title/tags.
- Confirm behavior on a non-interface photo (should classify Photo, metadata node, no spurious OCR).

**External system updates:**
- None — self-contained daemon. Ensure deployment env has OpenRouter API access for the `/api/v1/models` metadata fetch (already required for completions).
