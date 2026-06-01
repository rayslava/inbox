## [0.3.2] - 2026-06-01

### 🚀 Features

- *(llm)* Classify transient service-unavailable backend errors
- *(llm)* Skip transient-unavailable backends with per-backend cooldown
- *(pipeline)* Hold vision-incomplete messages pending, never false-success
- *(resume)* Re-OCR on retry; exhaustion is failure, never reported as success

### 🚜 Refactor

- *(llm)* Extract reusable CircuitBreaker from ollama
- *(llm)* Rename is_service_unavailable to is_service_available

### 📚 Documentation

- *(config)* Document circuit_open_secs on cloud backends + vision ollama

### 🎨 Styling

- Use coarser Duration units flagged by clippy after the MSRV bump

### 🧪 Testing

- *(pipeline)* Cover vision-unavailable short-circuit and resume re-OCR guard

### ⚙️ Miscellaneous Tasks

- *(deps)* Raise MSRV to 1.95 so `rand = "0"` resolves to rand 0.10
- *(release)* 0.3.2
## [0.3.1] - 2026-05-23

### 🐛 Bug Fixes

- *(deps)* Pin rand to 0.10 to stop resolver picking rand 0.8
## [0.3.0] - 2026-05-23

### 🐛 Bug Fixes

- *(llm)* Fix stuck enrichment, harden backends, split LLM modules

### 🚜 Refactor

- *(error)* Drop unwrap/expect on infallible and local paths
- *(test)* Split test modules over 500 lines
- *(llm)* Dedup forced-summary, fix panics, share retry backoff
- *(free_router)* Defer startup pool fetch off multi-thread runtime

### 🧪 Testing

- *(llm)* Cover chain helpers, memory tools, and tool dispatch

### ⚙️ Miscellaneous Tasks

- *(telegram)* Test the api url before connection
- *(release)* 0.3.0
## [0.2.0] - 2026-04-27

### 🚀 Features

- Support preprocessing
- Semaphore ollama calls
- *(tg)* Accept the media groups
- *(tools)* Support duckduckgo search
- Add web links to ROAM_REFS if used
- *(llm)* Thinking tool
- *(llm)* Memory support
- *(memory)* Migrate from sqlite to grafeo
- Improve metrics
- Add feedback support
- Enforce LLM memory usage
- *(message)* Add serde derives for pending store serialization
- *(pending)* Implement SQLite pending store with migrations
- *(render+config)* Add inbox_pending tag and ResumeConfig
- *(output)* Add atomic org-file entry patcher
- *(adapters)* Add Telegram resume notifier and status msg tracking
- *(resume)* Add background resume task with pending queue metrics
- *(pipeline+main)* Integrate pending store and resume task
- *(llm)* Add free_router backend for dynamic free-model pool
- *(llm)* Record enrichment metadata in org entries

### 🐛 Bug Fixes

- Prevent the networking issues
- *(web)* Update the urls to prevent 404
- *(tg)* Apply timeouts for the first reply
- *(tg)* Support timeouts on attachment download
- *(tg)* Enforce ipv4
- *(output)* Use correct ids for nodes and attachment paths
- *(llm)* Enforce chain and ensure the result is reported
- *(output)* Correct org-mode formatting
- *(reconnect)* Use biased select to fix flaky shutdown-before-first-run test
- *(memory)* Suppress noisy vector index warning on startup
- *(llm+resume)* Add connect_timeout and handle exhausted pending items
- *(llm)* Improve Ollama stability with circuit breaker and robust JSON parsing
- *(llm)* Redact api_key from tracing spans in call_chat_completion
- *(memory)* Retry Grafeo open when DB is briefly locked

### 🚜 Refactor

- *(adapters)* Extract reconnection loop

### 📚 Documentation

- *(CLAUDE.md)* Forbid unwrap/expect/panic in production code
- Refresh CLAUDE.md and README.md for current project state

### 🧪 Testing

- Update the attachment test
- Improve coverage
- Add coverage for pending store, resume, render, and pipeline
- Add targeted unit tests to lift coverage in low-cost modules
- Raise coverage for resume_task and http adapter
- Cover telegram resume notifier and web proxy/ui paths
- Cover pipeline::process and url_fetcher download/rewrite paths

### ⚙️ Miscellaneous Tasks

- Update Cargo.toml docs
- *(web)* Refurbish UI
- Code style cleanup
- *(llm)* Prevent OOM
- Style cleanup
- Colorize errors
- Add SQL workflow to CLAUDE.md and sqlfluff config
- Document [pipeline.resume] in config.example.toml
- Bump deps
- *(lint)* Clear pre-existing clippy warnings and enforce crate-wide deny
## [0.1.0] - 2026-03-05

### 🚀 Features

- Initial commit
- Support crawl4ai
- *(web)* Support logs in ui
- *(ollama)* Support thinking/not thinking modes
- *(http)* Extract the data from html
- *(tool)* Support retries
- *(tools)* Support Kagi search
- *(telegram)* Accept attachments
- Add status reporting

### 🐛 Bug Fixes

- Update the content extraction
- *(telegram)* Reconnect on net problems
- *(http)* Support redirects and compression
- *(llm)* Add debug prints for llm requests
- *(http)* Use built-in CA
- *(http)* Support CORS
- *(telegram)* Prevent dispatcher panic on start
- *(pipeline)* Do not enrich the enriched article

### 🚜 Refactor

- Support LLM prompts configuration

### 🧪 Testing

- Improve coverage
- Improve coverage

### ⚙️ Miscellaneous Tasks

- Cleanup
- Anodize more functions
- Bump deps
- *(http)* Support http retries
- Sync up with the config
- Cleanup docs and build
