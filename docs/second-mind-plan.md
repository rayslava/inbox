# Second Mind — design plan (rev. 4, post user review)

## Context

Ray runs a mature org-mode KB (org-roam v2 zettelkasten + GTD, ~865 files, GPG-encrypted
journals/finance, Syncthing-replicated) edited daily in Emacs. A Rust daemon `~/projects/inbox`
already aggregates Telegram/email/HTTP into org notes with LLM enrichment, a Grafeo graph memory
with Ollama embeddings, multi-backend LLM fallback (free_router/ollama/openrouter), tool-calling,
pending/retry, and an axum admin with session/argon2 auth.

Three goals: (1) ingest a reflashed omi wearable as a capture device (audio and/or text, no omi
cloud); (2) a full second brain — RAG over the whole KB with pluggable LLM, **with Emacs as a
first-class access surface** (org-roam, org-ql, gptel, and self-made extensions querying the data)
and a RAG **designed for vague requests** (relevant extraction even when the query is imprecise),
queryable from Emacs, Telegram, web, and HTTP — reachable **both locally and as an authenticated
web service**; (3) publish a curated subset to `kb.do.rayslava.com` on the existing Flux/Forgejo
`~/projects/new-do-k8s` cluster.

**Hard constraint: non-destructive, not read-only.** Second Mind *may* inspect and **improve
existing notes** via LLMs (different APIs/backends) — e.g. enrich, re-summarize, add links/tags,
fix structure — and **create new notes** when they are well-categorized and aligned with the
existing layout/conventions. The rule is **never destroy or lose existing data**, never corrupt the
Syncthing mesh, and never silently break the daily Emacs workflow. Concretely:
- Every edit to an existing file is **reversible and auditable**: operate via the org AST, make
  minimal diffs, back up before write (Syncthing `.stversions` already gives version history; add an
  explicit pre-edit snapshot for LLM-driven changes), and record provenance
  (`:ENRICHED_BY:`/`:MODIFIED_BY: second-mind`, timestamp, model) like inbox already does.
- **Default to propose-then-apply**: LLM edits land as review-able changes (a captures/staging copy
  or an Emacs-visible diff) that promote into the target note; bulk auto-edits are opt-in per scope.
- The **org-roam DB** is a derived cache — don't hand-edit it; let `org-roam-db-autosync` rebuild
  after file changes. **Calendars** (`*-cal.org`, gcal/gtasks-managed) and **encrypted**
  `journal/*.gpg` are touched only on explicit opt-in (calendars are externally synced; journals
  require decrypt) — excluded from automatic enrichment by default, not permanently read-only.
- New notes follow existing conventions (org-roam `YYYYMMDDHHMMSS-slug.org` / topic files, proper
  `:ID:`/`#+title:`/filetags) so they integrate with agenda, deft, and backlinks.

### Threat model (rev.4 — cloud holds the full KB behind auth)

Rev.4 deliberately **moves the private KB into the cloud** so the second brain is reachable as an
authenticated web service. This **replaces** the earlier "unpublished content never leaves home"
invariant. Concretely:
- The cloud (`do-k8s`, SGP1) holds **two read-only Grafeo files** shipped one-way from home: a
  **published-only file** (published rows only) opened by the public anon site, and a **full file**
  (published + private rows) opened **only** by the private kb-web behind **OIDC** (Dex+LLDAP). The
  public site never opens the full file — isolation is at the **file/process** level.
- **GPG-encrypted notes stay encrypted at rest** everywhere, including the cloud (`journal/*.gpg`,
  finance). The cloud never receives plaintext for those; they remain opt-in/excluded from
  enrichment. Encryption is the backstop now that full content leaves home.
- The trust boundary is now: the DO cluster node at-rest + OIDC correctness. This is a deliberate
  privacy trade for remote full-KB access.

### Locked decisions
- **omi** = capture device → thin `omi-bridge` does STT, forwards to a **new typed durable ingest
  API** on inbox (the existing `/inbox` is text-only and non-durable — do not rely on it).
- **Intelligence**: full second brain; pluggable LLM (already in inbox) + new pluggable embeddings;
  RAG tuned for vague queries; Emacs as a first-class query surface.
- **Two web surfaces.** (a) **Public anon site** — `kb.do.rayslava.com` serves only `:publish:`
  notes + public search, no login. (b) **Authenticated private web** — full-KB browse/search/RAG
  behind **Dex+LLDAP OIDC**, served from the cloud. Private full data is served remotely **only
  behind auth**; GPG-encrypted notes stay encrypted regardless.
- **Rendering runs LOCALLY** (where Ray's Emacs config lives): render `:publish:` notes with
  **`ox-html`** (NOT ox-who, which is a MediaWiki `'mw` backend), babel disabled, on change; ship a
  published artifact (HTML + published vectors) for the public site. A separate **full Grafeo file**
  carries the **private rows for the authed site**. Render is `emacs -Q --batch` (sandboxed, babel off):
  **local by default, or a dedicated cloud-side renderer (sidecar/job)** where a deploy prefers
  in-cluster rendering. The Emacs renderer stays isolated from `kb-web` (which remains lean Rust
  serving) — keeping Emacs in a cloud image is allowed for the renderer only, not the web server.
- **Brain uses ONE unified store**: a single local Grafeo store holds memories + the whole embedded
  KB. The cloud gets **read-only Grafeo files** derived from that store — a published-only file for
  the public anon site, a full file behind OIDC. Same engine (Grafeo) everywhere; not a second
  authoritative store, and **no separate database technology** (no Postgres/pgvector).
- **Store is Syncthing-synced like the org files.** The Grafeo store is a **single file replicated
  over the same Syncthing mesh** as the org corpus (the way inbox already delivers captures), across
  Ray's trusted personal nodes (desktop/laptop). Coupling stays **loose**: Syncthing just moves the
  file. **Single-writer** model, but the design **must tolerate the store file being modified at any
  time** (connectivity gaps, mesh writes, `*.sync-conflict*` copies appearing): use **CAS on
  mtime+content-hash**, **reload-on-change**, and **detect-and-reconcile** Syncthing conflict copies —
  never silently lose data.
  - **Automatic conflict merge** (we own the schema, so reconcile structurally rather than just
    flagging) — feasible **but spec'd to fail closed**, not best-effort:
    - **Per-entry logical revision.** Every mutation stamps a **logical op-id** — a hybrid logical
      clock `(writer-id, counter)` — **not** wall-clock mtime (Syncthing/clock skew makes mtime an
      unreliable arbiter). Merges resolve order by op-id causality.
    - **Transactional, serialized, crash-safe.** The merge runs as a **normal exclusive Grafeo write
      transaction** (same process/file lock as all writes) over a transactionally consistent
      snapshot; pause Syncthing via the REST API when available, else rely on atomic replace. It
      writes the result to a **temp file + merge-manifest carrying a stable merge-id** (hash of the
      two inputs), fsyncs, **atomic-replaces**, then **archives** (never deletes) the conflict copy —
      **only after post-merge validation passes**. A crash mid-merge leaves the original intact; on
      restart the merge-id makes the redo **deterministic and idempotent** (a half-written temp is
      discarded).
    - **Nodes AND edges.** Entries union by **namespaced ID**; **edges** union by a **canonical edge
      key** `(src-id, type, dst-id[, ordinal])`. Duplicate edges collapse; after any node resolution
      edge **endpoints are remapped** to the survivor; **tombstones** propagate (a delete dominates a
      stale re-add unless the re-add has a later op-id). **Post-merge validation fails closed**: every
      edge endpoint must resolve (no dangling/duplicated edges, no edge pointing at a superseded
      node), or the merge **aborts and keeps both files for manual review**.
    - **Collision policy.** A true same-key divergence resolves by **op-id causal order** (deterministic
      `writer-id` tie-break); **genuinely concurrent** divergence (neither dominates) is **marked
      `conflict` for review, not silently auto-picked**. The superseded side is kept as an explicit
      **`conflict`/superseded variant** (provenance-stamped). **Fingerprint-mismatched** entries are
      never merged across vector spaces (re-embed first).
    - **Variants are retrieval-invisible.** `conflict`/superseded variants carry a distinct state that
      is **excluded from default memory recall, kb-only RAG, hybrid retrieval, export to either Grafeo
      file, and public/private search** — surfaced only in an audit/debug mode — so they never pollute
      behavioral recall or RAG citations.
    - After a successful merge, **rebuild the vector/text indexes**. The whole routine runs **only on
      the single writer** and is **idempotent**; an unexpected schema/fingerprint skips it (manual
      review).
  - **Optional** Syncthing REST API hooks (status / pause / rescan) gated
  behind config, with **graceful degradation** where a deploy doesn't permit direct API access. The
  cloud is **not** a node in this personal single-writer mesh: it receives **one-way, read-only
  Grafeo copies** via the artifact path (#publisher), never write access to the synced store file.
- **Auth**: public site = none. **Private web = Dex+LLDAP OIDC (in scope, built)** fronting the
  cloud private endpoint and any local private API; not inbox's in-memory DashMap sessions.

### Resolved by Phase 0 spike
- **Engine = EXTEND `inbox`** (decided; was open). The Phase 0 dependency-split spike
  (`~/projects/inbox`, branch `phase0-dependency-split-spike`, log
  `docs/phase0-spike-log.md`) converted `inbox` into a Cargo workspace with a
  dependency-light `inbox-core` crate + stub `kb-web`/`omi-bridge` (which build against
  `core` alone — gate verified by `cargo tree`), and wired the first trait
  (`EmbeddingProvider`) end-to-end, green (clippy/fmt/test 700-pass/tarpaulin 85%).
  Finding: coupling is **shallow (3 edges)** and the **facade re-export** pattern moves
  domain types with ~zero consumer churn, so extending is low-risk; a rebuild would
  discard a working 700-test daemon for no structural gain. Remaining boundary work
  (`message`/status move, `VectorStore`/`UrlFetcher`/`AuthSession`/`LlmBackend`,
  `OutputWriter` Config-narrowing) is ~4–6 incremental green commits. Heavy redesign
  remains *available* per-module if a later phase needs it, but is **not** the path.

## Architecture

```
        HOME (workstation / fess)                              CLOUD (do-k8s, SGP1)
  omi ─BLE→ omi-bridge ─(typed durable ingest)→ inbox          read-only Grafeo files (mounted RO):
   (own fw)  STT local|cloud (Stt trait)         enrich+write     ├─ published-only.grafeo (public)
                                                  org files        └─ full.grafeo (private, OIDC only)
   ~/orgmode + Grafeo store                                       │  (GPG notes encrypted at rest)
        ╲___ Syncthing mesh (desktop/laptop, single-writer) ___   ▼
        │    [personal nodes only — cloud NOT in this mesh]   kb-web (axum, lean serving)
        ├─ publisher (local): watch :publish:               ├─ public anon: opens published-only.grafeo
        │   emacs --batch ox-html (babel off) → HTML         │    :publish: HTML + /search
        │   + build published-only.grafeo + full.grafeo      └─ private OIDC: opens full.grafeo
        │   ── one-way ship (Syncthing sidecar | DO Spaces) ─▶    full-KB browse/search/ask
        ▼      HTML + both Grafeo files, read-only            (Dex+LLDAP) ingress kb.do.rayslava.com
   local brain: VectorStore(full KB) + LLM chain             [optional] emacs --batch renderer sidecar/job
   Emacs (org-roam/org-ql/gptel/self-made + direct store query)      (in-cluster render, isolated)
        | M-x second-mind-ask | localhost /ask | Telegram
```

Workspace layout (**provisional, pending Phase 0**) (`crates/`): `core` (shared traits +
dependency-light domain types), `inbox` (daemon, keeps adapters/telemetry/syncthing/render glue),
`omi-bridge`, `kb-web` + a `publisher` mode (can live in kb-web or its own bin). If the Phase 0
spike favors a rebuild, this layout is revisited.

## Risk-driven changes (from codex review — all folded in)

1. **`core` is a trait boundary, not a file move** (codex high, conf .93). `message`/`llm`/`output`
   are coupled to `processing_status`, `pipeline::url_fetcher`, `Config`, telemetry, Syncthing.
   → Phase 0 **dependency-split spike**: define narrow traits — `LlmBackend`, `EmbeddingProvider`,
   `VectorStore`, `Auth/Session`, `UrlFetcher`, `OutputWriter` — move only dependency-light domain
   types into `core`; leave adapters/telemetry/render/Syncthing in the `inbox` bin. Gate with
   `cargo tree` + compile checks that `kb-web`/`omi-bridge` build against `core` alone. **This spike
   also decides extend-inbox vs build-new** (engine is unlocked); heavy/non-clean inbox redesign is
   permitted.
2. **Render = `ox-html`, sandboxed** (codex high, conf .97; verified: ox-who defines `'mw` from
   `'html`, exports `.who` wiki markup). → Public site renders with `ox-html` + a custom stylesheet.
   Batch profile: `org-export-use-babel nil`, no local init, pinned `emacs -Q --batch -l render.el`,
   inotify **debounce**, **atomic** cache writes. Fixture test: a `:publish:` note → valid HTML and
   a shell/ditaa `src` block is **not** executed.
3. **Typed durable omi ingest** (codex high, conf .98; verified: `/inbox` reads only `json["text"]`,
   hardcodes `MessageSource::Http`, `202` = mpsc-send). → Add an `Omi` variant to
   `MessageSource`/`SourceMetadata` (`src/message.rs`) + template branch. Add a **new `/ingest`
   route** with a typed schema (source enum, captured_at, speaker, duration, device, **idempotency
   key**). **Do NOT reuse `PendingStore`** for this (re-review high, conf .96: it only holds a
   post-write `ProcessedMessage` for retry; `resume_task` assumes the org node already exists). Add a
   **separate `ingress_events` outbox table**: raw typed payload + attachment blob paths/checksums +
   idempotency key, written with fsync/transaction **before** the 202; a consumer promotes rows into
   the existing processing pipeline; `pending_items` stays for post-write retry only. `omi-bridge`
   also keeps its own durable outbox, retrying until it observes a committed item. Crash-recovery test
   required. Phase 3 text-path prototype may use `/inbox`, but production omi uses `/ingest`.
4. **ONE unified store — common direction.** A single Grafeo store is the source of truth for
   **both** distilled memories (from `inbox`, `omi`, `signals`) **and** the embedded KB itself —
   using all the KB as memories is fine. One shared instance/handle across `inbox`, `omi`, `brain`,
   `curator` (so inbox writes **affect the second mind** and omi **recalls from the same store**).
   No second authoritative store.
   - **One physical DB, kind-partitioned logically** (re-review high, conf .94: today
     `recall_entries`/`graph_context`/`preload_context` scan **all** `:Memory` with no kind filter,
     and `memory_save` upserts by key — so dumping KB chunks in raw would dilute behavioral recall
     and collide identities). Required: a mandatory **`kind`** (`memory` | `kb-chunk`) with a
     **kind index**, **namespaced IDs** (`memory:<source>:<key>`, `kb:<note-id>:<chunker>:<hash>`),
     source-scoped dedupe keys, plus **explicit retrieval APIs** — memory-only for behavioral
     preload/tools, kb-only for RAG, hybrid **with per-kind quotas** for brain answers. **No-regression
     test**: existing `memory_save`/`memory_recall`/preload results are unchanged as KB-chunk volume
     grows. Entries also carry optional **note id + source path** and an **embedding fingerprint
     (model, dims, metric, normalization, chunker version)**; KB chunks add a **`published`** flag
     (publish gate, #5) that **drives the local export split**: published rows go into the
     published-only Grafeo file (opened by the public site), all rows into the full file (OIDC). The
     boundary is **file/process isolation**, not a row filter; per-row ACL/kind filtering remains as
     defense-in-depth.
   - Wrap it behind `EmbeddingProvider` + `VectorStore` traits (the current EmbedClient is hardcoded
     to Ollama `/api/embed`, `MemoryConfig` has no persisted fingerprint/chunker — both real gaps to
     close), but there is **one implementation/instance locally**.
   - **Cloud runs the same embedded Grafeo, read-only** — no Postgres/pgvector, no separate vector
     DB. `publisher` exports **two read-only Grafeo files** from the single store (no re-embedding →
     no export-time drift): a **published-only file** (the public anon kb-web opens only this) and a
     **full file** with published *and* private rows + their precomputed vectors (the private OIDC
     kb-web opens this). GPG notes stay encrypted at rest. **No `apps/cnpg` change** for Second Mind.
   - **Active-fingerprint enforcement** (re-review high, conf .88), now **single-engine**: each
     shipped Grafeo file carries its `embedding_fingerprint` (model, dims, metric, normalization,
     chunker version); **kb-web rejects loading a file whose fingerprint doesn't match** its expected
     one — same dimensions are NOT enough. Changing the embedding model/metric/normalization/chunker
     triggers a **full re-embed** of the store before the next export, never silent mixing (no
     cross-engine drift to reconcile now). Tests: **export consistency** (each file's rows == their
     source rows) **and** fingerprint-mismatch rejection at load.
5. **Deny-by-default publish gate** (codex medium, conf .86; verified: **zero** `:publish:` tags
   exist today). → The gate governs the **public anon path only**: serve publicly **only** notes
   explicitly carrying `:publish:` (filetag or subtree tag, honoring org tag **inheritance**);
   hard-exclude dot paths, `.stversions`, `*.sync-conflict*`, `journal/`, `*.gpg`, and `:noexport:`
   subtrees from the public site. The **authed private path** serves full KB (still excluding `*.gpg`
   plaintext and honoring opt-in for calendars/journals). Attachment-reachability tests so a *public*
   note can't pull in a private attachment. Unpublished content reaches the cloud **only as
   auth-gated private rows**; the public path receives none.
6. **Auth** (codex medium, conf .84). → Public site: no auth. **Internet-facing private login IS
   built**: the private web (full-KB `/ask`/browse/search) is fronted by **Dex+LLDAP OIDC** (already
   in cluster), **not** inbox's in-memory DashMap sessions. Local private `/ask` HTTP may still bind
   `127.0.0.1`/VPN for home use.

## Components

- **inbox extensions** (`crates/inbox`, `crates/core`): `Omi` source + template; typed durable
  `/ingest`; `core/kb_index` (chunk + embed org via `EmbeddingProvider`, ACL flags); `core/brain`
  (embed→ACL-filtered top-k→LLM chain→answer + cited `id:` links, reusing tool-calling); local
  `/ask` HTTP (localhost). **Telegram query defaults to PUBLISHED-only** (Bot API sends answers via a
  3rd party — re-review high, conf .93); full-KB answers stay on Emacs/localhost/VPN/OIDC-web, with
  full-over-Telegram an explicit opt-in flag + privacy warning.
- **KB curator** (`core/curate`, new): the "improve existing data" engine. Reuses the LLM chain +
  `core/brain` retrieval to propose **non-destructive** edits to existing notes — enrich/summarize,
  add `:ROAM_REFS:`/tags, suggest backlinks, merge duplicates, split overgrown files — and to
  **create** new well-categorized notes, stamping provenance (`:MODIFIED_BY: second-mind`, model,
  timestamp). **Editing engine is a spike-first decision** (re-review high, conf .90: the repo has
  **no org parser**; the existing `org_patcher` is byte-range string surgery, and concurrent
  Syncthing/Emacs edits are normal). Preferred: **delegate parse + structural edit to Emacs
  `org-element`/`org-roam`** via batch or an emacsclient/server call (authoritative org semantics,
  ID/property/drawer preservation) — a Rust org parser is allowed only if it **round-trips Ray's
  corpus** in the spike. Writes use **compare-and-swap on mtime+content-hash** (abort if the file
  changed under us), atomic replace, then Syncthing rescan. Modes: `propose` (default — review-able
  change as a staging copy / Emacs ediff) and `apply` (opt-in per scope/tag). Idempotent; skips
  calendars and `*.gpg` unless explicitly opted in. Triggerable from Emacs (`M-x second-mind-curate`),
  the local HTTP API, and as a batch sweep.
- **omi-bridge** (`crates/omi-bridge`): `POST /omi/audio` + `/omi/text`; `Stt` trait (local
  whisper.cpp/`whisper-rs` vs cloud Deepgram/Whisper API, config-selected with fallback); durable
  outbox → inbox `/ingest`. Same binary runs home or cloud.
- **Signal adapters** (`core/signals`, new — extends inbox's adapter pattern): poll **other cluster
  services** for what Ray reads/saves and distill it into the **shared Grafeo memory** (and, when
  worthwhile, propose notes via the curator). First two, both already in the cluster with their own
  **external app databases** (CNPG — these are the *services'* DBs, unrelated to the Second Mind
  Grafeo store): **Wallabag** (saved/archived articles) and **TTRSS** (RSS reads/stars/labels).
  **API-first** (re-review med, conf .86: `pg-rw` is cluster-internal so home can't read it directly,
  and scraping app schemas risks overprivilege/coupling): use each service's HTTP API with a scoped
  token + a persisted **cursor** for incremental pulls. DB mode is allowed **only** with an explicit
  CNPG **read-only role** + grants, a NetworkPolicy/tunnel, and a **pinned app-schema version** with
  contract tests. Secrets via SOPS. Then LLM-extract topics/entities/summary, `memory_save`/
  `memory_link` into Grafeo (kind=`memory`) with provenance (`:SOURCE: wallabag|ttrss`), dedupe by
  item id. Output is **memory by default** (low-noise),
  surfacing as note proposals only when a topic recurs/crosses a threshold. Pluggable so more
  services (forgejo activity, mitra, etc.) can be added later behind the same trait. These run at
  home alongside inbox so memory stays local and full-trust.
- **publisher** (local mode of `kb-web`): watch `:publish:` notes → `ox-html` render (gate #2/#5) →
  HTML + embed published chunks → write to the artifact folder; **build two read-only Grafeo files**
  from the single store — a **published-only file** and a **full file** (published + private rows,
  with their existing vectors) — and **ship them one-way** to the cloud read-only mounts (Syncthing
  sidecar or DO Spaces), alongside the HTML. No pgvector export.
- **kb-web** (`crates/kb-web`, cloud, lean Rust serving — no Emacs *in the server*): **public anon
  mode** serves published HTML from the artifact PVC + `/search` over the **published-only Grafeo
  file** (embedded in-process; it never opens the full file; **no anonymous `/ask`**); **private OIDC
  mode** serves full-KB
  browse + `/search` + `/ask` over the **full Grafeo file**, fronted by Dex+LLDAP. Both files are
  fingerprint-checked at load. Matching CSS.
- **cloud renderer** (optional, `emacs -Q --batch`): a **separate in-cluster renderer** (sidecar or
  Job) where a deploy prefers to render `ox-html` in the cloud instead of locally. Sandboxed (babel
  off), isolated from `kb-web`; same `render.el` profile as the local renderer. Local render stays
  the default; this is the "keep Emacs in a cloud image as backend renderer" option.
- **k8s** (`new-do-k8s`, copy `apps/wallabag/` pattern): namespace `second-mind`; artifact PVC (or
  DO Spaces) holding the HTML + the two read-only **Grafeo files**, plus a Syncthing sidecar/node for
  one-way delivery; `kb-web` deployment (public + private OIDC modes, each mounting its Grafeo file
  read-only); **public ingress** `kb.do.rayslava.com` (`letsencrypt-prod`) + **private OIDC ingress**
  (Dex+LLDAP forward-auth); **no CNPG / no pgvector** for Second Mind; SOPS secrets (STT/LLM keys,
  Syncthing key, **OIDC client secret**); Forgejo Actions + BuildKit → zot; Flux entry
  `clusters/do-sgp1/second-mind.yaml`.
- **Emacs glue** (`~/.emacs.d/lisp/second-mind.el`, additive): `M-x second-mind-ask` → local `/ask`,
  show answer + `[[id:...]]` links; `M-x second-mind-curate` → review/apply proposed enrichment for
  the current note/region (ediff-style); a `:publish:` toggle command. **Direct store/DB access from
  Emacs**: a read-only query interface (`M-x second-mind-query` + an Elisp client and a gptel tool)
  over a small **`core` query API** — local HTTP endpoint by default, with an optional direct store
  reader — so org-roam / org-ql / gptel and self-made commands can query the second-mind store inline
  (search, recall, backlinks, citations) without going through `/ask`. Integrates with org-roam /
  org-ql / gptel and is extensible for self-made commands. No changes to existing capture/sync flows.

## Phasing

0. **Dependency-split spike** — trait boundaries + `core`; **decide extend-inbox vs build-new**
   (heavy redesign allowed); shared Grafeo service with **kind-aware schema/indexes + namespaced IDs**
   and a **no-regression test** that KB-chunk volume doesn't change existing memory recall. (#1, store)
1. **Second brain (local)** — `EmbeddingProvider`/`VectorStore` (fingerprint column), kb_index over
   full KB, `core/brain` with **per-kind retrieval (memory-only / kb-only / quota'd hybrid)** tuned
   for vague queries, localhost `/ask`, `second-mind.el` (org-roam/org-ql/gptel). Immediate value, no
   cloud. (store)
2. **KB curator (local)** — **start with an org-edit spike** (Emacs `org-element` vs Rust parser,
   corpus round-trip + ID/property preservation + concurrent-edit CAS), then `core/curate`
   propose-mode enrichment; `M-x second-mind-curate`. Non-destructive edits + categorized new notes.
   (constraint, curator)
3. **omi-bridge** — text path first, then audio + `Stt`; typed durable `/ingest`. (#3)
4. **Signal adapters** — `core/signals` Wallabag + TTRSS → shared memory (topics/provenance), note
   proposals on recurrence; pluggable for more services later. (shared-memory)
5. **Publishing + private web** — publish gate + `ox-html` render + artifact; build/ship the **two
   read-only Grafeo files** (published-only + full); cloud kb-web (public anon + **private OIDC**) +
   Flux app + public ingress + private OIDC ingress + **public `/search` only (no anon `/ask`)** +
   authed full-KB `/search` `/ask`; Telegram query. (#2,#5,#6)
6. **Hardening** — export-consistency tests, golden ACL tests, OIDC auth tests, curator apply-mode +
   rollback tests, SOPS, metrics, backups.

## Verification

- **Workspace/spike**: `cargo tree` shows `kb-web`/`omi-bridge` depend only on `core` (not the inbox
  bin); `cargo test`/`clippy` green; `make images` builds; the spike records an extend-vs-rebuild
  decision.
- **Shared memory**: a `memory_save` performed during inbox enrichment is later recalled by an omi
  ingest and by `core/brain` (same Grafeo instance, one process/handle — not three copies).
- **Store sync (Syncthing)**: the store file replicates desktop↔laptop alongside org; a write on the
  active node appears on the other after sync. Inject a Syncthing modification / `*.sync-conflict*`
  copy while the brain holds the file → CAS aborts, the brain reloads, and the conflict copy is
  **auto-merged** under the fail-closed spec:
  - **Nodes + edges**: new-ID entries and edges union in; after a same-key resolution, edge endpoints
    remap to the survivor; **post-merge validation finds no dangling/duplicated edge** (else the merge
    aborts and keeps both files).
  - **Op-id arbitration (not mtime)**: a same-key divergence where one side causally dominates resolves
    to it; a **genuinely concurrent** divergence is **marked `conflict` for review**, not auto-picked;
    a clock-skewed mtime does **not** change the outcome.
  - **Variant invisibility**: the superseded/`conflict` variant is preserved but **does not appear** in
    memory recall, kb-only RAG, hybrid answers, export to either Grafeo file, or public/private search
    (audit mode only).
  - **Crash/idempotency**: kill the writer mid-merge → original file intact on restart; the merge
    redoes deterministically by merge-id and re-running is a no-op. **Fingerprint-mismatched** copies
    are not merged (re-embed/manual review).
  With the Syncthing REST API disabled, the brain still operates (graceful degradation); with it
  enabled, status/pause/rescan hooks work.
- **Direct Emacs store query**: `M-x second-mind-query` (and the gptel tool) returns store
  entries/backlinks/citations over the `core` query API without invoking `/ask`.
- **Memory isolation**: loading the full KB as `kb-chunk` entries leaves `memory_save`/`memory_recall`/
  preload results byte-for-byte unchanged (kind-filtered); a kb-only RAG query never returns behavioral
  memories and vice-versa; hybrid honors per-kind quotas. A **`conflict`/superseded variant** (from an
  auto-merge) is absent from recall, RAG, hybrid, export, and search — present only in audit mode.
- **Fingerprint**: rows with a different `embedding_fingerprint` are never co-queried; changing the
  embed model forces a full re-embed before export, not silent mixing; **kb-web refuses to load a
  Grafeo file whose fingerprint doesn't match** its expected one.
- **Signal adapters**: a new Wallabag save and a TTRSS star each produce a Grafeo memory entry with
  `:SOURCE:` provenance, deduped on re-poll; a recurring topic crosses threshold → a note proposal.
- **Brain (local)**: index a known note; `curl 127.0.0.1:PORT/ask` answers citing it; `M-x
  second-mind-ask` shows it; a deliberately **vague** query still retrieves the relevant note.
- **Curator (non-destructive)**: a propose-mode run produces a review-able diff without mutating the
  source; apply-mode writes a minimal AST diff, leaves a pre-edit snapshot + `:MODIFIED_BY:` stamp,
  preserves all prior content (no headings/links/properties lost), and re-running is idempotent;
  calendars and `*.gpg` are untouched unless explicitly opted in; `org-roam-db-autosync` rebuilds the
  cache cleanly after edits. **Concurrency**: if the file changes mid-edit, the CAS (mtime+hash)
  aborts and re-proposes instead of overwriting; org-roam `:ID:`s survive a curator edit.
- **Render safety**: a `:publish:` note with a shell `src` block renders valid HTML and the block is
  **not** executed (fixture test); editing it updates the cache atomically.
- **Publish gate (public path)**: a note without `:publish:`, a `:noexport:` subtree, a
  `.sync-conflict`, and a `.gpg` never appear in the artifact or on the **public anon** site;
  attachment of a private note is unreachable from a public page.
- **Private web (auth)**: the private endpoint requires a valid OIDC session; with it, full-KB
  browse/search/`/ask` reaches unpublished notes; without it, the private path returns 401/redirect
  and never serves private rows; `*.gpg` is never served as plaintext on either path.
- **omi durability**: `/ingest` persists before 202; kill inbox right after ack → on restart the
  capture still lands in `inbox.org` with `:SOURCE: omi:` + metadata; duplicate idempotency key is
  deduped. Sample WAV → local and cloud STT each yield a transcript node.
- **Export consistency**: the **published-only Grafeo file** contains **only** published rows (no
  unpublished leak); the **full file** contains all rows; every row matches its source row (identical
  vector + note id); both files' fingerprints match the local store; a mismatched file is rejected at
  load, not silently wrong.
- **k8s**: `kubeconform` passes; Flux reconciles `second-mind`; `kb.do.rayslava.com` serves
  published notes + search over TLS; the public surface exposes **no `/ask`** (search only) and the
  public process opens **only** the published-only file; the
  private OIDC ingress serves full-KB only to authenticated users; **no CNPG/pgvector** for Second
  Mind. Public site exposes **no** unpublished note.
- **No-regression**: inbox's Telegram/email/HTTP→org flow and the Emacs daily workflow (capture,
  agenda, gcal/gtasks, journal pre-warm) unchanged.

## Rev.4 (user review)

- **Emacs-first access + vague-query RAG**: goal (2) now names Emacs (org-roam/org-ql/gptel/
  self-made) as a first-class query surface and requires RAG that answers imprecise requests.
- **Authenticated private web in the cloud**: the second brain is reachable as an authed web service
  serving the **full KB** behind Dex+LLDAP OIDC. This **replaces** the old "cloud receives only the
  published artifact / unpublished content never leaves home" invariant — see the new threat model.
  GPG-encrypted notes remain encrypted at rest in the cloud.
- **Engine = EXTEND (resolved by Phase 0 spike)**: the spike converted `inbox` into a
  workspace (`inbox-core` + stub `kb-web`/`omi-bridge`, gate verified) and wired the
  first trait green; coupling is shallow and facade re-exports keep churn near-zero, so
  extend beats rebuild. See "Resolved by Phase 0 spike" above. Heavy per-module redesign
  stays available but isn't the path.
- **Syncthing-synced store**: the Grafeo store is a single file replicated over the same Syncthing
  mesh as the org files (desktop/laptop), single-writer, tolerant of mesh writes / conflict copies
  (CAS + reload + **automatic structural merge** of `*.sync-conflict*` copies via namespaced-ID union,
  LWW-with-loser-preserved on true collisions), with optional Syncthing REST API hooks.
- **Emacs in cloud for rendering**: keeping `emacs -Q --batch` in a cloud image is now allowed for an
  optional in-cluster **renderer** (sidecar/Job); `kb-web` stays lean Rust serving.
- **Direct DB access from Emacs**: a read-only `core` query API + Elisp/gptel interface so
  org-roam/org-ql/gptel and custom commands query the store inline, not only via `/ask`.
- **One store tech (Grafeo) everywhere — pgvector/CNPG dropped**: the cloud serves **two read-only
  Grafeo files** (published-only for the public anon site, full behind OIDC), shipped one-way from
  the single local store. No Postgres/pgvector, no CNPG vector extension for Second Mind; public vs
  private is **file/process isolation**, not row filtering.

## Rev.3 operability fixes (from codex re-review)

- **pgvector image** — **SUPERSEDED by rev.4**: Second Mind uses **no pgvector/CNPG**. The cloud
  serves embedded read-only **Grafeo files**, so there is no `apps/cnpg` vector-extension work, no
  `CREATE EXTENSION vector`, and no pgvector smoke job. (CNPG elsewhere in the cluster — Wallabag/
  TTRSS — is unrelated.)
- **Artifact serve topology** (med, conf .84): RWO `do-block-storage` + separate Syncthing and
  kb-web pods race / block rollouts. → Run Syncthing as a **sidecar in the kb-web pod** writing an
  atomic publish dir + manifest (HTML **+ the two Grafeo files**), kb-web mounts it read-only;
  **or** publish the artifact to **DO Spaces** (an object-store option in its own right) and have
  kb-web read from there. Readiness probe checks the artifact manifest **+ both Grafeo file
  fingerprints**, not just the port.
- **render.el org-roam context** (med, conf .82): `emacs -Q` loses id/attachment/transclusion
  context; corpus has many `[[id:...]]` links and `#+transclude`. → `render.el` explicitly sets
  `org-directory`, `org-id-locations`/`org-id-extra-files` limited to **publish-eligible** files,
  `org-attach-id-dir`, link-rewriting rules, and a **deny-by-default transclusion policy** (a
  `#+transclude` to a non-`:publish:` target is dropped, never inlined). Fixtures for id links,
  attachment links, and transclude-to-private.
- **omi audio/STT contract** (med, conf .78): inbox has no audio/whisper stack; WAV fixture ≠ BLE
  reality. → Define the wire format first (e.g. PCM16 mono 16 kHz chunks with sequence numbers +
  finalize, or a named container/codec), the decoder/resampler deps, local model-artifact
  management, queue limits + backpressure, and tests with chunked/non-WAV audio **and** bridge
  restart mid-stream. STT is not "just a trait" until this contract exists.

## Residual notes
- Full KB (minus `*.gpg` plaintext) now **does** reach the trusted cloud node as a **read-only full
  Grafeo file**, shipped one-way; access to private rows is gated by OIDC and GPG-at-rest, which are
  the boundary (rev.4 threat-model change).
- Keep one embedding model/config in `core` to avoid local/cloud vector-space drift.
- ox-who is untouched (still usable for its MediaWiki/wiki target elsewhere).
