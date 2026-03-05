# inbox — Universal Inbox Daemon

A personal capture daemon that accepts content from Telegram, HTTP, and email,
enriches it via LLM, and appends structured org-mode nodes to a file.

## Architecture

```
Adapters (producers)       Pipeline                          Output
─────────────────          ─────────────────────────────     ──────────────────────
TelegramAdapter ──┐        1. URL extraction                 OrgFileWriter
  HttpAdapter ───┼──mpsc─► 2. URL classify (HEAD)      ──►  └─► Syncthing rescan
   EmailAdapter ─┘            → Page: scrape
                              → File: download as attachment
                           3. LLM agentic loop (tool calling)
                              tools: scrape_page, download_file
                           4. Askama template → org node

Admin server (port 9090): /health/live  /health/ready  /metrics  /ui  /attachments/*
Inbox server (port 8080): POST /inbox   POST /inbox/upload
```

## Key modules

- `src/config.rs` — all TOML config structs; `${VAR}` env interpolation at load time
- `src/message.rs` — core types: `IncomingMessage`, `EnrichedMessage`, `ProcessedMessage`
- `src/pipeline/` — URL classifier, fetcher (file download + page scrape), content extractor
- `src/llm/` — `LlmChain` with agentic loop, tool calling, OpenRouter + Ollama backends
- `src/llm/tools.rs` — `scrape_page` + `download_file` tools (reuse pipeline internals)
- `src/output/org_file.rs` — atomic org append, org-attach-id-dir layout, Syncthing trigger
- `src/render/` — `OrgNodeTemplate` (Askama, `escape = "none"`)
- `src/web/` — admin axum router: session auth, inbox UI (orgize), attachment serving
- `src/metrics.rs` — `metrics` facade constants; exporter wired in `main.rs`
- `src/health.rs` — readiness state shared across adapters and shutdown handler

## Build & run

```bash
cargo build
cargo test
cargo run -- --config config.toml

# Generate an admin password hash:
cargo run -- hash-password

# Development (pretty logs):
RUST_LOG=inbox=debug cargo run -- --config config.example.toml

# K8s-style (JSON logs):
LOG_FORMAT=json cargo run -- --config config.toml

# Coverage (requires: cargo install cargo-tarpaulin):
cargo tarpaulin --fail-under 80 --out Html

# Integration tests with local Ollama:
TEST_WITH_OLLAMA=1 cargo test
```

## Templates

- `templates/node.org` — Askama template for org-mode nodes (`escape = "none"`).
  Compiled into the binary. To change the template, edit and recompile.
- `templates/inbox_ui.html` — Askama HTML for the web inbox viewer.
- `templates/login.html` — Askama HTML for the login form.

## Config

Config is TOML with `${ENV_VAR}` interpolation. See `config.example.toml`.
All secrets (API keys, bot tokens, passwords) should be passed via environment variables,
never committed.

Key sections: `[general]`, `[admin]`, `[web_ui]`, `[llm]`, `[[llm.backends]]`,
`[adapters.telegram]`, `[adapters.http]`, `[adapters.email]`, `[url_fetch]`, `[syncthing]`

## Org-attach layout

Attachments are saved at:
  `{attachments_dir}/{id[0..2]}/{id[2..]}/{original_filename}`
where `id` is the `IncomingMessage` UUID (also the org `:ID:` property).
This matches `org-attach-id-dir`.

## Crate conventions

- Errors: `thiserror` for library errors in `src/error.rs`; `anyhow` in `main.rs`
- Async: `tokio` runtime, `async-trait` for object-safe async traits
- Logging: `tracing` + `tracing-subscriber`; use `tracing::instrument` on pipeline steps
- Metrics: `metrics::counter!` / `metrics::histogram!` macros everywhere;
  constants defined in `src/metrics.rs`
- Shutdown: `tokio_util::sync::CancellationToken` threaded through all adapters
- Edition: 2024

## LLM tool calling

`LlmChain::run_agentic()` runs up to `MAX_TOOL_TURNS = 5` turns (configurable).
Tool implementations live in `src/llm/tools.rs` but call into `pipeline/url_fetcher`
and `pipeline/content_extractor` — do not duplicate fetch logic.

Each tool (`scrape_page`, `download_file`) has a configurable backend:
- `Internal` — built-in Rust implementation (default)
- `Shell` — local command via `tokio::process::Command` (argv split, no shell injection)
- `Http` — remote service with URL/body template substitution

Configured via `[[tools]]` in config.toml. See `ToolBackend` in `src/llm/tools.rs`.

## Testing rules

- **No real API calls in tests.** Use `MockLlm` in `tests/helpers.rs` for LLM.
- Use `wiremock` for external HTTP (URL fetching, Syncthing, etc.)
- Local Ollama tests: guarded with `if std::env::var("TEST_WITH_OLLAMA").is_ok()`
- Target: **80% line coverage** via `cargo tarpaulin --fail-under 80`

## K8s notes

- Liveness: `GET :9090/health/live` — always 200 unless stuck
- Readiness: `GET :9090/health/ready` — 503 during startup and drain
- Drain period: `admin.shutdown_drain_secs` (default 5) between SIGTERM and shutdown
- Prometheus scrape: `GET :9090/metrics`
- Set `LOG_FORMAT=json` in pod env for structured logging

## Code style

- Try to keep the files smaller that 500 lines, the modules should be atomic
- Move the tests into separate files and modules
- Split integration tests into smaller meaningful files
- When you add new dependencies to `Cargo.toml` only add the major version
- Markup the functions with the `anodized` https://docs.rs/anodized/latest/anodized/ crate
- Always check the `cargo clippy`
- Run the following set after changes:
  - `cargo clippy --fix --all-features --allow-dirty`
  - `cargo fix --all-features --allow-dirty`
  - `cargo fmt --all`
- Always run tests after all the issues reported are fixed
- No `#[allow]` tags must be used, fix issues, not mask them
- After the change is prepared, update the config.example.toml correspondingly
