# RFC 0009: Federation-First MIND Package Layer

| Field | Value |
|---|---|
| RFC | 0009 |
| Title | Federation-first MIND package layer |
| Status | **Draft** |
| Authors | STARGA Inc. |
| Created | 2026-05-21 |
| Supersedes | — |
| Superseded by | — |
| Related | RFC 0008 (mindc build + Phase E git deps), RFC 0010 (libMLIR FFI), RFC 0011 (async) |

---

## 1. Motivation

RFC 0008 Phase E established the primitives that make git-hosted dependencies
work: the identity triple `(git_url, rev, tree_sha256)`, the `Mind.lock`
mandatory-enforcement model, and the `~/.mindenv/cache/` content-addressed
store. What Phase E does not specify is anything above the client-side fetch:
how a developer finds packages in the first place, how a published package
is served over HTTP, how mirrors are listed and validated, or what happens
when the primary source is unavailable.

RFC 0009 specifies the rest of the package layer. The design principle is
**federation-first**: there is no central registry, no canonical index, and no
upload API operated by STARGA. A MIND package is a git repository with a
`Mind.toml`. Anyone who can serve files over plain HTTPS can host MIND
packages, indefinitely, without registration or permission.

The new surface in this RFC is:

- The **MIND Package Identity Protocol (MPIP)** — the static HTTP serving
  layout that any HTTP server or git host's raw-content endpoint satisfies
  out of the box.
- **Discovery modes** — direct URL, local index cache, and a sketch of
  federation gossip deferred to Phase 2.
- **Mirror fallback** — multiple mirror entries per dependency in
  `Mind.toml`; `mindc fetch` tries them in order; all must agree on
  `tree_sha256`.
- **New CLI subcommands** — `mindc verify`, `mindc index list`,
  `mindc index clear`.

What does not change from RFC 0008 Phase E: the lockfile schema, the
`tree_sha256` algorithm, the `~/.mindenv/cache/` layout, the `Mind.toml`
`[dependencies]` syntax for git deps, and the mandatory-lockfile enforcement
on `mindc build`. RFC 0009 adds discovery and mirror logic on top of those
unchanged foundations.

---

## 2. Non-goals

The following are explicitly outside the scope of RFC 0009 Phase 1.

**No central registry.** STARGA does not operate a "MIND packages" index
server. Packages live on git hosts — GitHub, GitLab, Codeberg, self-hosted
gitea, bare nginx, or any server that can return bytes over HTTPS. A package's
canonical identity is its git URL, not a short name registered in a central
database.

**No semver auto-resolution.** The `[dependencies]` entry for a git dep
requires an explicit `rev`, `tag`, or `branch`. The constraint syntax
(`>=1.2`, `^1.0`) is deferred to RFC 0009 Phase 2 along with the
constraint-satisfaction solver it requires. Phase 1 keeps resolution trivial:
a rev is a rev.

**No authentication or access-control surface.** Federation-first means the
protocol has no concept of a login, a token, or a paid tier. Private packages
are private git repos. Access is controlled at the git transport layer by the
user's own credentials (SSH key, personal access token, VPN) — not by the
MIND package layer.

**No cross-language package interop.** MIND packages are MIND packages. Linking
against C or system libraries is handled through `extern "C"` (RFC 0010), not
through the package layer.

**No `mindc publish` command in Phase 1.** Authors publish by pushing a tag to
their git remote. The package layer has no upload API. A convenience wrapper
(`mindc publish`) that pushes to a configured remote is deferred to Phase 2;
it would be syntactic sugar over `git push --tags`, nothing more.

---

## 3. The MIND Package Identity Protocol (MPIP)

MPIP is the serving layout that makes any HTTP server a valid MIND package
mirror. It requires no server-side software beyond the ability to return file
contents at a URL. A git host's raw-content endpoint (the kind that returns the
bytes of a file at a specific commit) already satisfies MPIP without
configuration.

### 3.1 Required endpoints

For a package at `<origin>/<owner>/<repo>`, pinned to `<rev>` (a full 40-char
commit SHA as stored in `Mind.lock`), a conforming MPIP server must respond to:

```
GET /<owner>/<repo>/<rev>/Mind.toml        → the package manifest (TOML)
GET /<owner>/<repo>/<rev>/.tree-sha256     → the canonical tree hash (hex, single line)
```

Both must return HTTP 200 with the correct content. Any other response code is
treated as a fetch failure and triggers mirror fallback (§5.2).

### 3.2 Optional endpoints

A conforming MPIP server may additionally serve individual source files:

```
GET /<owner>/<repo>/<rev>/<path/to/source.mind>   → source bytes
GET /<owner>/<repo>/<rev>/Mind.lock               → the package's resolved lock
```

`mindc fetch` does not require individual source file endpoints in Phase 1.
It fetches the full archive via the git protocol (using the same `git clone`
primitive RFC 0008 Phase E established) and derives `tree_sha256` locally.
The optional file endpoints are provided for tooling that wants random-access
reads — editors, documentation generators, online viewers — not for the build
pipeline.

### 3.3 The `.tree-sha256` sidecar

The `.tree-sha256` file at `/<owner>/<repo>/<rev>/.tree-sha256` is a single
line containing the lower-hex SHA-256 of the package tree, computed by the
same algorithm that `src/deps/mod.rs:compute_tree_sha256` implements (walk
sorted paths, skip `.git/`, accumulate `"<rel_path>\0<file_sha256_hex>\0"`
into an outer hasher).

A mirror that does not serve `.tree-sha256` at the required path fails the
MPIP conformance check. `mindc fetch` does not fall back silently on a missing
sidecar; it reports an error and tries the next mirror.

### 3.4 Serving with a git host's raw endpoint

GitHub's raw content endpoint, GitLab's raw blob endpoint, and Codeberg's
equivalent all satisfy MPIP for endpoints (§3.1) without any configuration.
For example, a package at `https://github.com/owner/lib_foo` pinned to commit
`abc1234def5678...` (40 chars) serves:

```
https://raw.githubusercontent.com/owner/lib_foo/abc1234.../Mind.toml
https://raw.githubusercontent.com/owner/lib_foo/abc1234.../.tree-sha256
```

The `mirrors` field in `Mind.toml` (§5.1) lists MPIP origins; the path
template is always `/<owner>/<repo>/<rev>/<file>`. The mirror origin URL
does not include the `/<owner>/<repo>` suffix — that is derived from the
primary `git` URL so that mirrors remain unambiguous references to the same
package.

### 3.5 Self-hosted mirrors

A self-hosted nginx, S3-compatible bucket, or static file server satisfies
MPIP by serving a directory tree mirrored from the canonical git source. The
mirror operator pre-populates the tree by running `mindc fetch` locally (which
populates `~/.mindenv/cache/`) and then syncing the cache directory to the
serving root. No server-side software is required beyond static file serving.

---

## 4. Discovery

Discovery is the mechanism by which a developer learns that a package exists
and what its git URL is. RFC 0009 defines three discovery modes. All three are
optional and stackable; a project that uses only direct URLs never needs the
other two.

### 4.1 Direct URL

The primary and default mode. A developer writes the full git URL directly
in `Mind.toml`:

```toml
[dependencies]
lib_foo = { git = "https://github.com/owner/lib_foo", rev = "abc1234" }
```

There is no name-to-URL resolution step. The URL is the identity. `mindc lock`
fetches the repo, computes `tree_sha256`, and writes the triple to `Mind.lock`.
No registry lookup occurs.

This is not a limitation — it is the design. Dependency confusion attacks
(§8.3) are impossible when the package identity is the full URL. A short name
registered in a central index would reintroduce the attack surface that direct
URLs eliminate.

### 4.2 Local index cache

Every time `mindc lock` or `mindc fetch` resolves a git dep, it writes a
discovery record to the local index at:

```
~/.mindenv/index/<hostname>/<owner>/<repo>/meta.toml
```

`meta.toml` contains the package name (from `Mind.toml [package].name`), the
canonical URL, the most recently pinned rev, and the resolved `tree_sha256`.
Example:

```toml
[package]
name     = "lib_foo"
git      = "https://github.com/owner/lib_foo"
rev      = "abc1234def5678901234567890123456789012345678"
sha256   = "e3b0c44298fc1c149afb4c8996fb92427ae41e4649b934ca495991b7852b855"
```

The local index is a build-up-over-time cache, not a bootstrap requirement.
`mindc fetch` works correctly with an empty index. The index is used by:

- **`mindc index list`** (§6.4) — shows all previously resolved packages.
- **Offline fetches** — when `mindc build --offline` is set and a dep is in
  the index cache, the last-known rev is available for diagnostics.
- **Editor tooling** — an LSP or editor plugin may read the index to offer
  URL completion when the user begins typing a `git = "..."` dep entry.

The local index is **per-user** and lives under `$XDG_DATA_HOME/mindenv/index/`
when `XDG_DATA_HOME` is set, otherwise `~/.mindenv/index/`. It is never shared
across users on a multi-user system; each user's index reflects only their own
resolutions.

### 4.3 Federation gossip (Phase 2 sketch)

A MPIP-conforming mirror may optionally serve a `Mind.federation.json` at its
root. In Phase 2, `mindc gossip` (a new subcommand) follows links in these
federation manifests to build a local discovery graph of known mirrors.
Federation gossip does not imply trust — a mirrored package is still validated
against `tree_sha256`. The gossip protocol is read-only and pull-only; there
is no push or registration step.

Phase 1 ships nothing in this section. The sketch is included to confirm that
the Phase 1 index layout (`~/.mindenv/index/`) is forward-compatible with the
Phase 2 gossip-populated variant. Phase 2 gossip entries will land in the same
directory structure under a `federation/` subdirectory and will not collide
with Phase 1 direct-resolution entries.

---

## 5. Mirror protocol

### 5.1 Declaring mirrors in Mind.toml

A package author or consumer may list mirror origins in the `[dependencies]`
entry:

```toml
[dependencies]
lib_foo = {
    git     = "https://github.com/owner/lib_foo",
    rev     = "abc1234def5678901234567890123456789012345678",
    mirrors = [
        "https://mirror.example.com",
        "https://codeberg.org/owner",
    ]
}
```

Each entry in `mirrors` is an MPIP origin (scheme + host, no trailing slash).
`mindc fetch` constructs the MPIP endpoint paths from the primary `git` URL
(extracting `<owner>/<repo>`) and appends them to each mirror origin in turn.

The `mirrors` array is optional. A dep with no `mirrors` array uses only the
primary `git` URL.

### 5.2 Fetch order and fallback

`mindc fetch` attempts sources in this order:

1. The primary `git` URL (git protocol clone, as in RFC 0008 Phase E).
2. Mirror origins in declaration order, using the MPIP HTTP endpoints (§3.1).

On a successful fetch from any source, `mindc fetch` stops. On failure
(non-200 HTTP response, network error, or timeout), it advances to the next
source. Retry within a single source is not performed; retries are the
fallback list's job.

The determinism property: for a given `(git_url, rev, mirrors)` triple, the
fetch order is always the same. It is not randomised or round-robin. This
makes build failures reproducible: a developer who sees a fallback-triggered
fetch sees it consistently, not intermittently.

### 5.3 Tree hash verification on every fetch

Regardless of which source served the content, `mindc fetch` re-computes
`tree_sha256` after download and verifies it against:

1. The `.tree-sha256` sidecar returned by the mirror (for MPIP HTTP fetches).
2. The `tree_sha256` field in `Mind.lock`.

Both checks must pass. A mirror that returns content with the correct sidecar
but where the locally-computed hash does not match `Mind.lock` is a hard error.
There is no silent retry on hash mismatch; the error names the mirror and the
hash discrepancy.

This check is not skippable. The `tree_sha256` is the trust anchor (§8), and
weakening the verification path even for "trusted" mirrors would silently
undermine the entire integrity model.

### 5.4 Content-addressed permanence

Every MPIP URL contains the full 40-char commit SHA as `<rev>`. Because the
SHA is immutable — a given `rev` always resolves to the same content, by
definition — MPIP URLs are permanently cacheable. A mirror's HTTP caches,
CDN layers, and proxy caches may cache MPIP responses with an infinite
`Cache-Control: immutable` header. The `Mind.lock` pin and `tree_sha256`
verification guarantee that a cached response is bit-identical to what the
author published at that rev.

---

## 6. CLI surface

RFC 0009 adds four new subcommands to the `mindc` surface. The existing
`mindc fetch`, `mindc lock`, and `mindc clean` subcommands from RFC 0008 are
unchanged.

### 6.1 `mindc verify`

```
mindc verify [--package <name>] [PATHS...]
```

Re-fetches each dependency (or the named one) from its primary source and
all declared mirrors. Re-computes `tree_sha256` for the fetched content.
Verifies the result against `Mind.lock`. Does not modify `Mind.lock`, the
local index, or the `~/.mindenv/cache/` directory.

Exit codes follow RFC 0008 §6: 0 = all deps verify cleanly; 1 = at least one
verification failed; 2 = invalid usage.

`mindc verify` is the explicit "re-fetch and re-prove" command. It is distinct
from the implicit verification that `mindc build` performs on every run (which
checks the cache against `Mind.lock` without re-fetching). Use `mindc verify`
when you want to confirm that the primary source and all mirrors still agree
— for example, before cutting a release or after a security advisory.

| Flag / arg | Type | Default | Description |
|---|---|---|---|
| `PATHS...` | paths | cwd | Project roots to verify. |
| `--package <name>` | string | (all deps) | Verify only the named dep. |
| `--verbose` | bool | false | Print each fetch attempt and hash comparison. |

### 6.2 `mindc index list`

```
mindc index list [--json]
```

Prints the contents of the local index cache (`~/.mindenv/index/`). Default
output is a human-readable table: package name, canonical URL, most recently
pinned rev (first 12 chars), and the date the entry was written.

`--json` emits a machine-readable JSON array, one object per entry, suitable
for editor tooling and scripts.

Example human output:

```
PACKAGE          GIT URL                                          REV           DATE
lib_foo          https://github.com/owner/lib_foo                 abc1234def56  2026-05-20
lib_matrix       https://codeberg.org/starga/lib_matrix           8f3e2a109c12  2026-05-18
```

### 6.3 `mindc index clear`

```
mindc index clear [--package <name>]
```

Removes entries from the local index cache. Without `--package`, clears the
entire index. With `--package <name>`, removes only the entry for that package.

Clearing the index does not invalidate `Mind.lock` or the `~/.mindenv/cache/`.
It only removes the discovery metadata. The next `mindc lock` or `mindc fetch`
will repopulate any entries that were cleared.

### 6.4 Full subcommand table after RFC 0009

| Subcommand | RFC | Description |
|---|---|---|
| `mindc build` | 0008 | Build a MIND project. |
| `mindc test` | 0008 | Run `#[test]` functions. |
| `mindc check` | 0007 | Static analysis (fmt-check + lint + type-check). |
| `mindc fmt` | 0007 | Canonical format. |
| `mindc lint` | 0007 | Lint diagnostics over the typed AST. |
| `mindc lock` | 0008 | Regenerate `Mind.lock` from `Mind.toml`. |
| `mindc fetch` | 0008 | Populate `~/.mindenv/cache/` from `Mind.lock`. |
| `mindc clean` | 0008 | Remove `target/` and/or cache entries. |
| `mindc verify` | **0009** | Re-fetch and re-verify all deps against `Mind.lock`. |
| `mindc index list` | **0009** | Print the local discovery index. |
| `mindc index clear` | **0009** | Clear the local discovery index. |

---

## 7. Phasing

### Phase 1 (this RFC, target Q3 2026)

- MPIP serving layout specification (§3) — normative.
- `mirrors` array in `Mind.toml [dependencies]` entries — parsed and validated
  by `mindc lock` and `mindc fetch`.
- Mirror fallback in `mindc fetch` (§5.2) — in-order, deterministic.
- `.tree-sha256` sidecar verification on MPIP HTTP fetches (§5.3).
- Local index cache population in `mindc lock` / `mindc fetch` (§4.2).
- `mindc verify` subcommand (§6.1).
- `mindc index list` and `mindc index clear` subcommands (§6.2, §6.3).

### Phase 2 (deferred)

- Semver constraint syntax in `Mind.toml [dependencies]` (`>=1.2`, `^1.0`).
  Requires a constraint-satisfaction solver; the solver design is a separate
  sub-RFC gated on Phase 1 stability.
- Federation gossip protocol: `Mind.federation.json` layout, `mindc gossip`
  subcommand, gossip-populated index entries (§4.3).
- `mindc publish` convenience wrapper: pushes the current tag to a configured
  remote and optionally syncs to declared mirror origins.

### Phase 3 (long-term)

- Content-addressed mirror backend using IPFS as the transport layer. In this
  model, the MPIP `<rev>` maps to a CID (content identifier) rather than a git
  SHA, enabling fully decentralised hosting where no single server needs to
  remain online. Phase 3 is additive — it does not change the Phase 1 lock
  format or the `tree_sha256` algorithm.

---

## 8. Trust model and threat mitigations

### 8.1 Trust anchor

The user is the trust anchor. The entity that runs `mindc lock` and writes the
resulting `Mind.lock` to version control is the entity that decided to trust
those URLs. STARGA operates no trust list, no allowlist service, and no
certificate authority for MIND packages. There is nothing to compromise at the
registry layer because there is no registry.

The lockfile itself is the trust artefact. A `Mind.lock` that has been reviewed
by the project's maintainers and committed to version control is the record
of intent. Subsequent builds verify against that record via `tree_sha256`;
they do not re-evaluate trust.

### 8.2 Threat: supply-chain substitution via mirror

A mirror could, in principle, serve altered content — a malicious maintainer,
a compromised CDN, or a cache-poisoning attack.

**Mitigation:** `tree_sha256` is computed locally after every fetch, before the
content is used. The hash is verified against `Mind.lock`. A mirror that serves
altered content produces a different `tree_sha256` and fails the verification
step with an explicit error naming the mirror origin and the hash discrepancy.
The content is never used. There is no silent fallback on hash mismatch.

### 8.3 Threat: dependency confusion

An attacker publishes a package at a public URL with the same `[package].name`
as a private internal package, hoping that a build system resolves by name and
picks the attacker's version.

**Mitigation:** MIND packages have no name-based resolution. The dep entry in
`Mind.toml` includes the full git URL:

```toml
lib_foo = { git = "https://github.com/owner/lib_foo", rev = "..." }
```

`lib_foo` at `github.com/owner/lib_foo` and `lib_foo` at
`github.com/attacker/lib_foo` are different packages with different identities.
There is no name registry to squatting, and no resolution step that could
silently prefer one over the other. The URL in the manifest is the identity.

### 8.4 Threat: typosquatting

An attacker registers `github.com/owner/lib_fo0` (digit zero, not letter O)
hoping a developer misreads or mistypes the URL.

**Mitigation at the protocol level:** none. URL typosquatting is a social
engineering problem, not a protocol problem. The package layer provides no
"similar name" warning in Phase 1. A developer who types the wrong URL will
get the wrong package.

Future tooling may add a name-similarity warning that fires when `mindc lock`
resolves a URL whose `[package].name` is a near-match (edit distance 1) to an
already-indexed package at a different URL. That warning is not in Phase 1.

### 8.5 Threat: git history rewrite

A repository owner force-pushes a branch, replacing what `rev=abc` resolves to
without changing the SHA.

Note: this is not possible for a full 40-char commit SHA. A SHA is the hash of
the commit object; the same SHA cannot resolve to different content. Force-push
replaces branch tips (symbolic references), not object content.

The threat is real only for `branch = "main"` deps (which RFC 0008 Phase E
permits as a convenience). `Mind.lock` records the resolved full SHA at lock
time, not the branch name. Subsequent `mindc build` runs use the locked SHA;
a force-push to the branch does not affect them. The branch ref is used only
when `mindc lock --update <pkg>` explicitly re-resolves.

A `rev = "<40-char-sha>"` dep is immune to all rewrite attacks: the SHA is
the identity of the content.

---

## 9. What changes from RFC 0008 Phase E

RFC 0008 Phase E is the complete client-side implementation. RFC 0009 adds
three things on top of it without modifying any of the Phase E contracts:

1. **The MPIP serving spec (§3)** — names the URL layout that makes any HTTP
   server a valid mirror. Phase E already validates `tree_sha256` on fetch;
   MPIP specifies what a server must provide so that validation can happen over
   HTTP instead of only via git clone.

2. **The `mirrors` array (§5.1)** — a new optional field in the
   `[dependencies]` inline table. Phase E `Mind.toml` files without `mirrors`
   parse correctly under RFC 0009 (the field defaults to an empty array). No
   existing manifest requires modification.

3. **The local index and new CLI subcommands (§4.2, §6)** — `mindc verify`,
   `mindc index list`, `mindc index clear`. These are additive; they do not
   change the behaviour of `mindc build`, `mindc lock`, or `mindc fetch`.

The lockfile schema (`Mind.lock`), the `tree_sha256` algorithm, the
`~/.mindenv/cache/git/<hostname>/<owner>/<repo>/<sha>/` layout, and the
mandatory-enforcement rule on `mindc build` are all unchanged from Phase E.

---

## 10. Open questions

### Should mirrors be auto-discovered from the primary's served `Mind.toml`?

A package author could declare mirrors directly in their package's own
`Mind.toml` under a `[package.mirrors]` table. When `mindc lock` fetches the
primary, it would read those mirrors and propagate them into the consumer's
`Mind.lock`. This would let authors maintain their own mirror list without
requiring every consumer to configure mirrors manually.

**Proposed answer: yes, but Phase 2.** The consumer must still be able to
override or suppress a mirror declared by the author. The interaction with
security policy (a consumer may not want traffic going to an author-declared
mirror) needs more design. Phase 1 ships the consumer-declared `mirrors` array
only.

### Should `mindc verify` run automatically as part of `mindc build`?

`mindc build` already checks `tree_sha256` against `Mind.lock` at the start of
every build. The check is local (compares the cached tree against the lockfile
hash) and does not re-fetch from the network. `mindc verify` performs a full
network re-fetch and re-derivation.

**Proposed answer: no.** Automatic `mindc verify` on every `mindc build` would
require a network round-trip on every build, breaking offline workflows and
CI pipelines that pre-populate the cache with `mindc fetch`. The existing build-
time local verification is the right default; `mindc verify` is the explicit
opt-in for paranoid or release-gate scenarios.

### Should the `~/.mindenv/index/` cache be shared across users on multi-user systems?

**Proposed answer: no.** The index is per-user, scoped to `$XDG_DATA_HOME` or
`~/.mindenv/index/`. Sharing an index across users creates a write-contention
problem and a trust problem: user A should not be able to influence user B's
resolution decisions through a shared index. System-wide package caches require
a trust model (package signing, admin-controlled allow list) that RFC 0009
Phase 1 explicitly does not include.

### Should `mindc fetch` emit a progress indicator for large mirrors?

Phase 1 defers this to implementation. The `--verbose` flag on `mindc build`
and `mindc verify` will emit per-fetch log lines. A full progress bar
(percentage, bytes transferred) is a UX improvement for Phase 2.

---

## 11. Relation to other RFCs

**RFC 0008 (mindc build + Phase E git deps)** — the prerequisite. RFC 0009
adds the discovery and mirror layer on top of Phase E's client-side fetch
primitives. RFC 0009 must not be considered stable until RFC 0008 Phase E is
fully shipped and proven. RFC 0008 Phase E is shipped as of the date of this
RFC.

**RFC 0010 (libMLIR FFI in pure MIND)** — orthogonal. The package layer does
not depend on the FFI surface. A MIND package that wraps a C library uses
`extern "C"` in its source; the package layer treats it like any other MIND
package. RFC 0010 is not a prerequisite for RFC 0009.

**RFC 0011 (async / concurrency)** — orthogonal. The mirror fallback logic in
`mindc fetch` (§5.2) is sequential and synchronous. When RFC 0011 ships, a
future RFC may specify parallel mirror fetches with a first-winner model.
Phase 1 does not require RFC 0011.

---

## 12. Decision points

This section records the design choices made for this RFC and the rationale
for each. These decisions are final for Phase 1; superseding a decision
requires a new RFC or an explicit backwards-compatibility annotation.

### Mirror retry policy: in-order vs round-robin

**Decision: in-order.**

Round-robin would distribute load across mirrors but makes failure modes
non-deterministic: the same build might succeed or fail depending on which
mirror was chosen. In-order retry means a developer who observes a mirror
fallback sees it consistently and can diagnose it. Determinism is a higher
priority than load distribution for a source-of-truth package layer.

### Subcommand naming: `mindc verify` vs `mindc check-deps`

**Decision: `mindc verify`.**

`verify` is the shorter and more direct term for "re-fetch and confirm
integrity." `check-deps` is longer, hyphenated (inconsistent with the rest of
the CLI surface), and implies a broader scope (it sounds like it might also
check for semver compatibility, which Phase 1 does not do). `mindc verify`
aligns with the existing vocabulary of the codebase (`verify_entry` in
`src/deps/mod.rs`).

### Index location: `~/.mindenv/index/` vs `~/.config/mindc/index/`

**Decision: `~/.mindenv/index/`.**

The `~/.mindenv/` directory is already established by RFC 0008 Phase E as the
MIND environment root (`~/.mindenv/cache/` for fetched archives). Co-locating
the index in `~/.mindenv/index/` keeps all MIND environment state under one
root. `~/.config/mindc/` would scatter the environment across two directories.
`$XDG_DATA_HOME` is honoured for users who set it.

### `mindc index` as a subcommand group vs separate top-level commands

**Decision: subcommand group (`mindc index list`, `mindc index clear`).**

Grouping under `mindc index` keeps the top-level subcommand list short and
signals that these commands operate on the same subsystem. The alternative —
`mindc list-index` and `mindc clear-index` — is flatter but makes the
relationship between the two commands less obvious. The group pattern is
consistent with how `mindc clean --all` and `mindc clean --cache` are already
factored.

### MPIP sidecar filename: `.tree-sha256` vs `MIND_TREE_SHA256`

**Decision: `.tree-sha256`.**

The dotfile convention (`.<name>`) is already used by `src/deps/mod.rs` for
the sentinel written into the local cache (`.mind_tree_sha256`). The HTTP-
served sidecar uses the same convention. An all-caps name would be
conspicuous in a directory listing and suggests a registry-style artifact,
which this is not. The dotfile is a machine-readable sidecar, not a
human-facing artifact.
