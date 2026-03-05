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
4. Run LLM chain (OpenRouter/Ollama) with tools (`scrape_page`, `download_file`, `crawl_url`, optional `web_search`).
5. Render with Askama template and append atomically to org file.

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
