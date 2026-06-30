# Phase 0 — Dependency-Split Spike — Step Log

**Goal** (from `~/second-mind-plan.md`, rev.4): convert `inbox` into a Cargo
workspace with a dependency-light `core` crate (domain types + narrow traits),
leaving adapters/telemetry/render/Syncthing in the `inbox` bin. Prove via
`cargo tree` that downstream crates (`kb-web`, `omi-bridge`) can build against
`core` alone. Record an **extend-inbox vs build-new** decision + future-path
estimate. This is a *spike to estimate the path*, not a production migration.

**Traits to extract:** `LlmBackend`, `EmbeddingProvider`, `VectorStore`,
`Auth/Session`, `UrlFetcher`, `OutputWriter`.

**Rules in force:** rust-dev skill (clippy pedantic, no `unwrap/expect/panic`
in prod, no `#[allow]`, files <500 LOC, post-change pipeline clippy→fix→fmt→test,
anodized specs on non-trivial pub fns), inbox project rules (TBD from explore),
ask-codex review before every commit.

---

## Step 0 — recon (read-only)

- Repo: `/home/ray/projects/inbox`, single crate, branch `master` (default
  `origin/master`). Untracked pre-existing: `.dockerignore`, `improve.md`,
  `tarpaulin-report.html` — left alone.
- `src/` modules: `adapters/ config/ llm/ memory/ output/ pending/ pipeline/
  processing_status/ render/ web/ feedback/` + `message.rs error.rs telemetry.rs
  health.rs resume_task.rs url_content.rs tls.rs lib.rs main.rs`.
- `Makefile` targets: `images` (image-amd64 + image-arm64), `push`, `manifest`,
  `clean` — i.e. `make images` builds the container, no lint/test targets there.
- Branched: `phase0-dependency-split-spike` off `master` @ 030a380.

## Step 1 — synthesized design (from 3 read-only explores)

### Project rules that constrain the split
- edition 2024, rust 1.95; `[lints.clippy] pedantic = "warn"` in Cargo.toml → must
  hoist to `[workspace.lints]` + `lints.workspace = true` in each member.
- `main.rs` has `#![deny(clippy::unwrap_used, expect_used, panic)]`.
- No CI; release = `make images` → `cargo zigbuild --release --target <musl> --bin
  inbox`, `docker import` of the static binary. **Binary crate must stay named
  `inbox`.** No Dockerfile.
- deny.toml: `wildcards="deny"` but `allow-wildcard-paths=true` → intra-workspace
  `path` deps are fine. License allow-list is broad.
- Coverage ≥ 80% (`tarpaulin.toml`, excludes `tests/*`, keeps `main.rs` in math).
- Deps declared **major-only** (`"0"`,`"1"`). anodized specs on non-trivial pub fns.
- Conventional Commits (git-cliff). config.example.toml must mirror config changes.
- Tests: separate files; no real API calls (wiremock / teloxide_tests / Ollama opt-in).
- `lib.rs` is pure module decls; `[lib] name=inbox` + `[[bin]] name=inbox`; dev-dep
  self path `inbox = { path = ".", features=["test-helpers"] }`.

### Coupling — the 3 real blockers (everything else is clean)
1. `IncomingMessage` holds `status_notifier: Option<Box<dyn StatusNotifier>>`
   (`message.rs:24`). → move `StatusNotifier`+`ProcessingStage`+`NoopNotifier` to
   core (drop `ProcessingTracker`/`InFlightEntry`/telemetry gauge — stay in bin).
2. `Config`/`LlmConfig` leak into trait signatures: `OutputWriter::write(.., &Config)`
   (`output/mod.rs:14`), `LlmRequest::from_enriched(.., &LlmConfig)` (`llm/mod.rs:61`).
   → narrow `OutputWriter::write` to a small core param struct; make `from_enriched`
   a **binary-side free constructor** (`inbox::llm::request_from_enriched`).
3. `InboxError` (`error.rs`) appears in every trait `Result`. → move to core (verify
   it's dependency-light first); re-export from `inbox::error`.

### Trait inventory (current → core)
| Trait (core) | Current | Loc | Already trait | Notes |
|---|---|---|---|---|
| LlmBackend | `LlmClient` | llm/mod.rs:240 | yes (async_trait) | move trait + value types; `LlmResponse` lives in message.rs |
| OutputWriter | `OutputWriter` | output/mod.rs:13 | yes (async_trait) | narrow `&Config` param |
| EmbeddingProvider | `EmbedClient` struct | memory/embed.rs:5 | no | 1 method `embed(&str)->Vec<f32>`; reqwest stays in bin |
| VectorStore | `MemoryStore` struct | memory/mod.rs:54 | no | keep **grafeo** out of core; expose MemoryEntry/SourceEntry value types |
| AuthSession | free fns + `type SessionStore=DashMap` | web/auth.rs | no | argon2/dashmap/axum::http stay in bin |
| UrlFetcher | `UrlFetcher` struct | pipeline/url_fetcher.rs:17 | no | returns `UrlContent` (clean); reqwest stays in bin |

### Strategy — facade re-exports to bound churn
Move only **definitions** into `crates/core`; in the binary, re-export them at the
existing paths (`pub use core::message::*;` in `inbox::message`, same for `error`,
`processing_status`, `url_content`, `llm` value types). The ~30 files using
`crate::message::X` keep compiling untouched. Concrete adapters (Ollama/OpenRouter/
FreeRouter, OrgFileWriter+Syncthing, EmbedClient, MemoryStore+grafeo, UrlFetcher,
auth fns) **stay in the bin** and `impl core::Trait`.

### core dependency budget (must stay light)
serde, serde_json, chrono, uuid, url, async-trait, thiserror, anodized, tokio
(sync mpsc only — for `LlmTurnProgress` sender). **NOT** in core: reqwest, grafeo,
axum, teloxide, dashmap, argon2, metrics, tracing-subscriber, sqlx.

### Workspace layout
```
Cargo.toml            # [workspace] members + [workspace.lints] + [workspace.dependencies]
crates/core/          # traits + domain types (light deps)
crates/inbox/         # the existing daemon (bin name stays `inbox`), impls the traits
crates/kb-web/        # STUB — depends on core ONLY (gate proof)
crates/omi-bridge/    # STUB — depends on core ONLY (gate proof)
```
Gate: `cargo tree -p kb-web -p omi-bridge` shows `core` but **not** `inbox`.

### Spike scope (estimate the path, stay green at every step)
Land a green workspace proving the boundary; where a trait's narrowing balloons,
log it as "deferred + estimate" rather than forcing it. Run the rust-dev pipeline
(clippy→fix→fmt→test, then tarpaulin) at each stage. Codex-review before each commit.

## Step 2 — Stage A: workspace skeleton + core + stubs (binary untouched)

**Decisions made while scaffolding:**
- Crate named **`inbox-core`** (lib `inbox_core`), not `core` — avoids shadowing
  std `core`. Conceptual `core/*` modules in the plan map to `inbox_core::*`.
- **Kept `inbox` at the repo root** (not moved to `crates/inbox`). Root `Cargo.toml`
  is both `[package]` (the bin/lib) and `[workspace]`. Rationale: zero disruption to
  `make images` (`--bin inbox`), `tarpaulin.toml`, `.sqlx`, migrations paths. The
  doc's `crates/inbox` layout is a later cosmetic move; not needed to prove the gate.
- Hoisted clippy pedantic to `[workspace.lints.clippy]`; every member sets
  `lints.workspace = true`. `resolver = "3"` (edition 2024).
- **`CoreError`** is a *light* split of `InboxError`: drops the `reqwest`/`askama`
  `#[from]` variants (those stay in the binary's `InboxError`); keeps io/json/url +
  string variants, plus new `Embedding`/`VectorStore`/`Fetch` variants for the traits.
  The binary will gain `From<InboxError> for CoreError` at the boundary (Stage ≥B).
- Stubs `kb-web`/`omi-bridge` are minimal bins exercising `inbox_core::api_tag()` to
  anchor the dep without pulling Result-wrapping noise (pedantic `unnecessary_wraps`).

**Files added:** `crates/core/{Cargo.toml,src/lib.rs,src/error.rs,src/url_content.rs,
src/tests.rs}`, `crates/kb-web/{Cargo.toml,src/main.rs}`,
`crates/omi-bridge/{Cargo.toml,src/main.rs}`. Root `Cargo.toml` gained `[workspace]`.

**Gate (PASS):** `cargo tree -p kb-web -p omi-bridge` → `inbox-core` only
(serde/serde_json/thiserror/url); **never `inbox`**. The downstream crates cannot
see the daemon — the boundary holds.

**Pipeline on new crates:** clippy pedantic clean, fmt clean, 6 tests green
(4 core + 1 kb-web + 1 omi-bridge). Full-workspace clippy (incl. `inbox`) running.

**Full workspace green:** `cargo clippy --workspace` 0 warnings; `cargo test
--workspace` **696 passed / 0 failed** (15 binaries). (Caught a footgun: piping
`cargo test | tail` masks cargo's exit code — must capture full output + `$?`.)

**Codex review of Stage A → 2 MEDIUM, both fixed pre-commit:**
1. No `default-members` → bare root `cargo test/clippy/tarpaulin` only hit the
   `inbox` package, silently skipping the new crates. → added
   `default-members = [".", "crates/core", "crates/kb-web", "crates/omi-bridge"]`
   (verified: workspace_default_members now lists all four).
2. `From<InboxError> for CoreError` would not be category-preserving (InboxError has
   LlmTool/Attachment/Pipeline/Adapter/Memory; CoreError lacked them). → mirrored
   those `String` categories into `CoreError`; only the heavy `Http`/`Template`
   variants degrade to `Fetch`/`Output` strings at the boundary (documented).

→ **Commit #1** (Stage A).

## Step 3 — Stage B: first trait end-to-end through the boundary

Representative vertical slice to measure the real per-trait cost:
- **Facade move**: `UrlContent` physically moved to `inbox_core`; `src/url_content.rs`
  is now `pub use inbox_core::UrlContent;`. **All ~10 consumers compiled unchanged** —
  the re-export facade works exactly as hoped (zero churn). No orphan-rule issue (the
  type had no inbox-side impls).
- **inbox now depends on `inbox-core`** (path dep).
- **`EmbeddingProvider`** trait added to core (`async_trait`); `EmbedClient` in the
  binary `impl`s it, delegating to its inherent `embed` via UFCS `EmbedClient::embed(
  self, text)` (calls inherent, no recursion) and `.map_err(CoreError::from)`.
- **`From<InboxError> for CoreError`** (total, category-preserving; `Http`/`Template`
  degrade to `Fetch`/`Output` strings) — the one real friction point, now closed.
- **`kb-web` consumes `&dyn EmbeddingProvider`** (test-only mock + `#[tokio::test]`),
  proving a downstream crate drives the trait without seeing the daemon. async-trait/
  tokio are **dev-deps only** — normal-dep gate still core-only.

**Codex review of Stage B → 1 MEDIUM + 1 LOW, both fixed pre-commit:**
- MED: trait method lacked `# Errors` + the non-empty contract → documented on the
  core trait (enforcement still via the inherent `#[spec]` it delegates to).
- LOW: new boundary mappings untested → added `From<InboxError>` unit tests (12
  constructible arms, `src/error.rs`) + a wiremock trait-path test (success + 500→
  `CoreError::Memory`) so `EmbeddingProvider for EmbedClient` is covered.

**Pipeline:** clippy `--workspace` 0 warnings; fmt clean; `cargo test --workspace`
**700 passed / 0 failed**; `cargo tarpaulin --workspace` **85.20%** (≥80%; embed.rs
restored to full by the trait-path test). Heavy `Http`/`Template` From-arms (2 lines)
left untested — those error source types aren't cheaply constructible; trivial
`to_string` mapping, logged as accepted.

→ **Commit #2** (Stage B).

### Measured friction (feeds the estimate)
| Concern | Cost observed |
|---|---|
| Workspace + boundary | trivial (root package+workspace, facade re-exports) |
| Move a pure domain type | ~nil (1-line re-export, 0 consumer edits) |
| Error split | **the** real cost: light `CoreError` + total `From<InboxError>`; one-time, mechanical |
| Wire 1 trait (already-trait-shaped) | small: trait decl + delegating impl + error map + tests |
| Config-in-signature traits (OutputWriter, LlmRequest::from_enriched) | **not yet paid** — needs param-struct narrowing; estimated next-largest |

## Step 4 — DECISION: extend `inbox` (do NOT rebuild)

**Decision: EXTEND.** The spike de-risked the split end-to-end and the evidence is
one-sided:
- `lib.rs` is already pure module declarations; the daemon is library-shaped.
- The **facade re-export** pattern moved a domain type with **zero consumer edits**
  (all ~10 `UrlContent` users compiled untouched). This generalizes to the ~30
  `message`-type consumers.
- The coupling flagged by the codex review is **shallow — 3 edges**, not pervasive:
  the `StatusNotifier` field, two `Config`-in-signature methods, and `InboxError`
  placement. The error edge (the only non-trivial one) is **already paid** here.
- 2 of 6 traits (`LlmClient`, `OutputWriter`) are **already trait-shaped**; one
  (`EmbeddingProvider`) is now wired and green.
- A rebuild would discard a working, **700-test**, multi-backend daemon to re-derive
  the same boundary — negative value. There is no structural rot forcing a rewrite.

### Future-path estimate (remaining boundary work, all incremental + green)
Each item is the *same proven move* (trait in core + value-type move via facade +
delegating impl + error map + tests). Sizes relative to the EmbeddingProvider slice:

| Work item | Size | Notes |
|---|---|---|
| `message` types + `StatusNotifier`/`ProcessingStage` → core (facade) | M (1 PR) | unblocks `IncomingMessage`; consumer churn ≈ nil via re-exports |
| `VectorStore` (`MemoryStore` + `MemoryEntry`/`SourceEntry`/`RelatedMemory`) | S–M | grafeo stays in bin; mirror EmbeddingProvider |
| `UrlFetcher` trait | S | `UrlContent` already in core |
| `AuthSession` trait (session store + credential verify) | S–M | argon2/dashmap/axum::http stay in bin; no existing trait |
| `LlmBackend` = move `LlmClient` + value types; `from_enriched` → binary free fn | M | decouples `LlmConfig` from the trait; touches pipeline call sites |
| `OutputWriter::write(&Config)` → narrow core param struct | M | touches `OrgFileWriter` + 2 call sites |
| Cosmetic: move `inbox` under `crates/inbox` | S (optional) | not required; deferred |

**Rough total:** ~4–6 focused, individually-green commits. Risk: low — pattern proven,
churn bounded by facades, error split done, `make images`/tarpaulin/`.sqlx` unaffected.

**Caveats surfaced by the spike (carry forward):**
- `Config` must NOT enter `core` (god-object, 25 consumers); narrow per-trait params.
- `core` stays transport-free; only async-trait/serde-family/url/thiserror landed so far.

### Status
- Gate (cargo tree): **PASS**. Pipeline: clippy/fmt/test/tarpaulin **green**.
- **`make images`: VERIFIED** post-split (rc=0). musl static `--bin inbox` builds
  both arches (amd64 34.8s, arm64 37.6s; 29M stripped static ELF) and `docker import`
  produces `inbox:0.3.1-52-g7435c26-{amd64,arm64}`. The workspace split does not break
  the Dockerfile-less zigbuild release path.
- Spike commits on branch `phase0-dependency-split-spike`: #1 scaffold, #2 first trait,
  #3 decision.
