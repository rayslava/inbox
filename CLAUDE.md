# inbox — Universal Inbox Daemon

A personal capture daemon that ingests content from Telegram, HTTP, and IMAP email,
processes it through a URL + LLM pipeline, and writes structured org-mode nodes.

## Architecture

- Input adapters: `telegram`, `http`, `email`
- Pipeline: URL extraction/classification, fetch/scrape/download, LLM enrichment, template render
- Output: atomic org append + optional Syncthing rescan
- Admin server: health/readiness, metrics, optional web UI + attachments browser

## Runtime endpoints

- Inbox API (from `[adapters.http].bind_addr`): `POST /inbox`, `POST /inbox/upload`
- Admin API (from `[admin].bind_addr`):
  - Always: `GET /health/live`, `GET /health/ready`, `GET /metrics`
  - If `[web_ui].enabled = true`: `GET /login`, `POST /login`, `GET /logout`, `GET /ui`, `GET /logs`, `GET /status`, `GET /attachments/*`

## Build & run

```bash
cargo build
cargo test
cargo run -- --config config.toml
cargo run -- hash-password
```

## Config notes

- Config file is TOML with `${ENV_VAR}` interpolation at load time.
- Reference: `config.example.toml`.
- Core sections: `[general]`, `[admin]`, `[web_ui]`, `[llm]`, `[[llm.backends]]`, `[adapters.*]`, `[url_fetch]`, `[pipeline.web_content]`, `[syncthing]`, `[tooling.*]`.

## LLM/tooling notes

- Backends: OpenRouter and/or Ollama.
- Built-in tools: `scrape_page`, `download_file`, `crawl_url`; optional `web_search`.
- Tool backend modes: `internal`, `shell`, `http` (per tool config).
- Max tool turns controlled by `[llm].max_tool_turns`.

## Testing rules

- No real API calls in tests.
- Use `wiremock` for external HTTP interactions.
- Ollama-dependent tests are opt-in via `TEST_WITH_OLLAMA=1`.

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
