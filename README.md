# inbox

`inbox` is a personal capture daemon.
It accepts messages from Telegram, HTTP, and IMAP email, enriches them with LLM tooling, and appends structured org-mode entries to an output file.

## What it runs

- Inbox adapter server (default `:8080`): `POST /inbox`, `POST /inbox/upload`
- Admin server (default `:9090`): `/health/live`, `/health/ready`, `/metrics`, and optional web UI (`/login`, `/ui`, `/logs`, `/status`, `/attachments/*`)

## Pipeline

1. Ingest message and attachments from enabled adapters.
2. Extract URLs and classify them (page vs file).
3. Fetch/scrape/download URL content.
4. Run LLM chain with tools (`scrape_page`, `download_file`, `crawl_url`,
   optional `web_search`). Supported backends:
   - `free_router` — dynamic pool of free OpenRouter models from the
     shir-man top-models index, hedged parallel dispatch.
   - `openrouter` — pinned paid model.
   - `ollama` — local inference with circuit breaker.
5. Render with Askama template and append atomically to org file.
6. If the LLM fell back to raw mode, stash the item in a SQLite pending
   store; a background resume task retries it later and patches the org
   entry in place on success.

## Quick start

```bash
cp config.example.toml config.toml
cargo run -- --config config.toml
```

Generate admin password hash:

```bash
cargo run -- hash-password
```

Run tests:

```bash
cargo test
```

## Configuration

Use `config.example.toml` as the reference. Config supports `${ENV_VAR}` interpolation at load time.
Store secrets (API keys, tokens, password hash) in environment variables.
