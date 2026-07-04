# second-mind-plan.md — rev.4 revisions (revdiff annotations + codex + user review)

## Context

Ray is reviewing `~/second-mind-plan.md` in revdiff across several rounds. Earlier
rounds already shifted direction and **those edits are applied to the doc already**:

- **A1** Emacs-first access (org-roam/org-ql/gptel/self-made) + vague-query RAG.
- **A1 topology** authed private web serves **full KB from the cloud** behind Dex+LLDAP
  OIDC; old "unpublished never leaves home" invariant replaced by a stated threat model.
- **A2/A3** Engine unlocked (Phase 0 spike decides; heavy inbox redesign allowed).
- **A4** keep `emacs -Q --batch` in a cloud image allowed **for an optional renderer**
  (sidecar/Job); `kb-web` stays lean Rust serving.
- **A5** Grafeo store is a **single file Syncthing-synced** alongside org (desktop/
  laptop), single-writer, tolerant of mesh writes/`*.sync-conflict*` (CAS + reload +
  reconcile); optional Syncthing REST API hooks.
- **A6** direct store query from Emacs (read-only `core` query API + Elisp/gptel tool).

This round resolves the **last open annotation** (doc l.337, the pgvector item):

> "do we still plan to use postgres? Will the local version work like that?"

Findings (verified in `~/projects/inbox`): there is **no Postgres today**. Grafeo is an
**embedded single-file** graph DB (`Cargo.toml:60` `grafeo` with `embedded,wal,
grafeo-file,mmap`; `src/memory/mod.rs:68` `Config::persistent`; default
`<attachments_dir>/memory.grafeo` via `src/llm/builder.rs:63`), and **embeddings live
inside Grafeo** as a native vector index (`src/memory/queries.rs:8` `CREATE VECTOR
INDEX`). pgvector/CNPG appeared **only** in this plan's cloud projection — an added tech.

**Decision (user): Grafeo everywhere, drop pgvector/CNPG.** The cloud "projection"
becomes a **read-only Grafeo file**, not a different database.

## Pending edits to `/home/ray/second-mind-plan.md` (this round)

Replace every pgvector/Postgres/CNPG/"projection" reference with the Grafeo-file model.
Cloud serving uses **two read-only Grafeo files** for physical isolation:
- **published-only file** → opened by the **public anon** kb-web (cannot even read
  private rows — process-level isolation, stronger than a query filter);
- **full file** → opened by the **private OIDC** kb-web.
Both are produced locally by `publisher` (filtered exports of the one authoritative
local Grafeo store) and shipped **one-way, read-only** to cloud via the artifact path
(Syncthing sidecar or DO Spaces) — the cloud is **not** a writable node in Ray's
personal single-writer mesh.

1. **Architecture diagram** — `full-KB projection (CNPG/pgvector)` → `read-only Grafeo
   files (published-only + full)`.

2. **Locked decisions — store bullet** — "read-only full projection … pgvector" →
   read-only Grafeo file(s); name the two-file split + one-way shipping.

3. **Store-sync bullet (A5)** — adjust the closing clause: cloud serves from a
   **read-only Grafeo copy shipped via the artifact path**, not by joining the personal
   single-writer Syncthing mesh.

4. **Risk-driven #4 — cloud bullet (doc ~l.116-119)** — rewrite: cloud runs embedded
   Grafeo read-only; public path = published-only file, private path = full file behind
   OIDC; **no pgvector, no CNPG vector ext, no `apps/cnpg` change**.

5. **Risk-driven #4 — fingerprint bullet (~l.120-125)** — keep fingerprint enforcement,
   but now **single-engine**: the shipped Grafeo file carries its
   `embedding_fingerprint`; cloud kb-web **rejects a file whose fingerprint doesn't
   match** its expected one. Drop "per-fingerprint tables / pgvector" framing; no
   cross-engine drift to test anymore.

6. **Components — publisher** — "export full projection → cloud pgvector" → "produce the
   two read-only Grafeo files (published-only + full) and ship via artifact path."

7. **Components — kb-web** — "over the full projection" → "over the read-only Grafeo
   file(s), embedded in-process (published-only for anon, full for OIDC)."

8. **Components — k8s** — **remove** CNPG vector ext + `secondmind` db + pgvector DSN
   secret; cloud just mounts the Grafeo file(s) (PVC or Spaces). Keep namespace,
   ingresses (public + OIDC), artifact/Syncthing, SOPS (STT/LLM keys, OIDC secret).

9. **Phasing 5** — "pgvector full projection" → "ship read-only Grafeo file(s)".

10. **Verification — export consistency** — rewrite around the Grafeo files: the
    published-only file contains **only** published rows (no unpublished leak); the full
    file (OIDC) contains all; both fingerprints match the local store; a fingerprint
    mismatch is rejected by kb-web at load.

11. **Verification — k8s** — drop "CNPG has vector + secondmind".

12. **Rev.3 fixes — "pgvector image" bullet** — mark **superseded by rev.4**: no
    pgvector; cloud serves embedded Grafeo files; the CNPG vector-extension work is
    dropped.

13. **Rev.3 fixes — "Artifact serve topology" bullet** — extend: the artifact now also
    carries the **Grafeo file(s)** (not just HTML); same Syncthing-sidecar-or-DO-Spaces
    one-way shipping; readiness probe checks the Grafeo file manifest + fingerprint.

14. **Rev.4 subsection** — add a bullet: **one store tech (Grafeo) everywhere; pgvector/
    CNPG dropped; cloud serves read-only Grafeo files (published-only public, full OIDC),
    shipped one-way from the single local store.**

### Added after codex review (missed spots + 2 semantic fixes)

15. **Threat-model paragraph (doc ~l.42-49)** — currently names the boundary as "DO
    cluster node + **CNPG at-rest** + OIDC." Rewrite: cloud holds **two read-only Grafeo
    files**; the **published-only file** (public) contains only published rows; the
    **full file** is reachable **only** by the private kb-web behind Dex+LLDAP OIDC; GPG
    plaintext still excluded. Boundary = DO node at-rest + **OIDC** (no CNPG).

16. **Published-flag routing (doc ~l.150-160)** — currently "published flag drives which
    cloud endpoint serves the row" / public path "queries only `published` rows from the
    full projection" = **row-level filtering**, which contradicts the two-file isolation.
    Reword: the `published` flag drives **local export into the published-only Grafeo
    file**; the **public kb-web never opens the full file** (process/file isolation is
    the boundary). Row ACL/kind filtering stays as **defense-in-depth**, not the boundary.

17. **Architecture diagram (doc ~l.92-100)** — redraw so the **personal single-writer
    Syncthing mesh (desktop/laptop)** is **separate** from cloud delivery. Show
    `publisher` producing the **two Grafeo files** and shipping them **one-way** (artifact
    Syncthing sidecar or DO Spaces) to **read-only cloud mounts**. Remove the implication
    that the personal mesh arrow reaches cloud, and the "export full projection" label.

18. **Enumerate remaining "projection" wording** (the broad "replace every reference" made
    concrete): rendering decision "cloud store/projection" (~l.62-63); diagram "export
    full projection" (~l.100); verification "the projection rejects mixed fingerprints"
    (~l.285-286); residual notes "full projection" (~l.359-360). Each → Grafeo-file /
    export-manifest wording.

19. **DO Spaces rationale (rev.3 artifact topology, doc ~l.341-345)** — drop the "already
    configured for **CNPG backups**" justification (CNPG is being removed for Second
    Mind); reframe DO Spaces as an artifact-store option on its own. Readiness probe
    checks the **HTML artifact + both Grafeo file manifests/fingerprints**.

20. **Signal-adapter CNPG refs (doc ~l.205-213) are EXTERNAL app DBs** (Wallabag/TTRSS),
    not the Second Mind store — **keep**, but carve them out explicitly so they're not
    mistaken for the dropped vector store. Adjust the verification grep accordingly
    (below).

## Verification (of the doc edits)

- Re-launch revdiff: `…/launch-revdiff.sh --only=/home/ray/second-mind-plan.md`.
- `grep -ni 'pgvector\|postgres\|projection'` over the doc returns **only** superseded/
  explanatory mentions — no live "use pgvector" instruction remains. `cnpg` matches are
  **only** the external Wallabag/TTRSS signal-adapter app DBs (#20), never the Second
  Mind store.
- Public-anon path is described as opening the **published-only** Grafeo file (process/
  file isolation, not row filtering); private path opens the full file behind OIDC.
- The threat-model paragraph no longer names CNPG; boundary = DO node at-rest + OIDC.
- Single-writer Syncthing mesh stays home; cloud gets one-way read-only Grafeo copies;
  the diagram shows them as separate paths.
- User adds no new annotations on re-review → done.

## Process

After this plan is written: run **`ask-codex`** to review the plan (per user request),
fold in any material findings, then ExitPlanMode for approval before editing the doc.
