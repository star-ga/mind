# RFC 0013: CLI Agent Harness Stack

| Field | Value |
|---|---|
| RFC | 0013 |
| Title | CLI Agent Harness Stack |
| Status | **Draft** — design only; no phase shipped |
| Authors | STARGA Inc. |
| Created | 2026-05-24 |
| Supersedes | — |
| Superseded by | — |
| Related | RFC 0005 (pure-MIND std surface), RFC 0010 (memory safety + C ABI), RFC 0011 (async + structured concurrency), RFC 0008 (mindc build + test) |

---

## 1. Motivation

The credibility test of self-hosting + RFC 0005 std-surface + RFC 0011 async is
whether MIND can host **an agent CLI** — a Claude-Code-class terminal client
(reference shape: the OSS `claw-code` project, `claw-code.codes`) — **written
end-to-end in pure MIND**, with no Python / TS / Rust runtime underneath.

This is the smallest realistic non-tensor application that exercises every
long-pole stdlib surface at once: argv parsing, JSON / TOML / regex,
filesystem and process spawning, HTTPS to a model endpoint, server-sent-event
streaming, JSON-RPC framing (MCP), and a terminal UI with ANSI control. If
MIND ships those surfaces and the resulting binary is byte-identical across
Linux musl / macOS universal / Windows MSVC under `ReplayScheduler` (RFC 0011),
the language has earned a credibility lane that Rust + tokio + reqwest +
ratatui occupies today.

The roadmap surface for this work lives at `docs/roadmap.md` Phase 16; this
RFC is the per-tier design room.

---

## 2. Non-goals

**Not a Rust replacement for system programming generally.** This RFC scopes
*one* class of application — a streaming terminal agent CLI. WebSocket,
HTTP/2, HTTP/3, generic ACME / certificate issuance, full-featured TUI
widget libraries, and HTTP servers are all explicitly deferred.

**Not a tokio clone.** RFC 0011 defines MIND's async model — `Scheduler` is a
first-class value, futures are explicit `GenRef<Future<T>>`, replay-determinism
is the load-bearing property. This RFC consumes that model; it does not
redefine it.

**Not a curses clone.** The terminal layer (`std.tui`) provides termios raw
mode, ANSI escapes, key-event reading, line editing, and a minimal frame
buffer. Rich widget libraries (ratatui-class) are out of scope.

**Not a generic crypto library.** TLS 1.3 is in scope; arbitrary primitives
(AEAD-as-a-service, generic signatures, post-quantum) are out of scope. The
TLS module exposes only what an HTTPS client needs.

---

## 3. Capability baseline (v0.7.0)

The verification pass produced this table — it is normative for this RFC.

| Capability                | v0.7.0 status                                              | Gap |
|---------------------------|------------------------------------------------------------|-----|
| CLI parser                | `std.string` / `std.vec` / `std.map` shipped; no `std.cli` | Build `std.cli` (arg parsing, subcommands, env binding) |
| JSON / TOML / regex       | `std.json` / `std.toml` / `std.regex` shipped (RFC 0005 Phase 2) | None |
| FS / process              | `std.fs` / `std.process` shipped                           | None |
| Cross-platform binary     | Linux musl + macOS universal + Windows MSVC shipped (#225) | None |
| `std.io`                  | Bare stdin/stdout/stderr only                              | No ANSI / TTY detection / colour |
| `std.net`                 | Raw POSIX TCP/UDP only (IPv4/IPv6)                         | **No HTTP, no TLS, no HTTPS** |
| Async I/O                 | RFC 0011 Phase A shipped (sync `ReplayScheduler`)          | Phases B–F deferred; the real long pole |
| Streaming (SSE / chunked) | n/a                                                         | Requires async + HTTP framing first |
| MCP protocol              | Not in tree                                                | Build from scratch in MIND |
| TUI / ANSI / coloring     | Not in tree                                                | Build `std.tui` (sequences + termios) |

---

## 4. Tier design

The work splits into six tiers, sequenced by dependency. Each tier has its
own §4.x design block; everything below it ships as a separate PR series.

### 4.1 Tier 1 — `std.cli` + `std.io` extensions + `std.tui` minimal

Parallelizable; independent of the network stack. Effort: 1–2 weeks.

- **`std.cli`** — flag parsing (long, short, bundled), subcommand dispatch,
  env-var binding, help-text generation. Deterministic argv ordering: the
  parser produces the same `Args` value byte-identically for the same argv
  sequence on every platform.
- **`std.io` extensions** — `isatty(fd) -> bool`, `winsize(fd) -> (rows, cols)`
  via `TIOCGWINSZ` (POSIX) and Console API (Windows), ANSI escape constants
  (foreground / background colour, cursor positioning, screen clearing,
  scroll regions). No string-building of escapes — typed primitives only.
- **`std.tui` minimal** — termios raw-mode toggle (POSIX) / `SetConsoleMode`
  (Windows), key-event reader (single keystroke, escape-sequence-aware), a
  `LineEditor` primitive (readline-class — history, cursor, basic editing),
  a minimal frame buffer (rectangular character cells, no widgets). The bar
  is "enough to draw a streaming response panel and an input prompt," not a
  ratatui clone.

Acceptance: a worked example (`examples/agent_tui/`) draws a two-pane terminal
that reads input and echoes coloured output, byte-identical across all three
target OSes.

### 4.2 Tier 2 — `std.tls` (TLS 1.3 client) — **the long pole**

Effort: 6–10 weeks. Gates Tier 3 entirely.

**Scope decision — RESOLVED (2026-05-25): Option B-hybrid.** Backed by an
architecture stress-test (mind-architect) that confirmed the load-bearing
premise: crypto primitives are **deterministic-by-standard** — `AES(k,p)`,
`SHA256(m)`, `X25519(s,p)` produce byte-identical output on AVX2 / NEON / every
conforming substrate *by definition of the cipher* (AES-NI vs bitsliced
software differ in timing, not bytes). They therefore discharge the §8.4
cross-substrate byte-identity proof obligation **trivially** (the NIST CAVP /
RFC 8439 / RFC 7748 known-answer vectors already prove it), which is exactly
what nondeterministic numeric FFI (NumPy f64, CUDA tensor cores, OpenBLAS
reduction order) can never do. Crypto FFI does **not** fracture the wedge the
way numeric FFI does. The options weighed:

| Option | Verdict |
|---|---|
| **A — pure-MIND TLS 1.3 incl. hand-rolled primitives** | **Deferred, not rejected.** Multi-quarter; hand-rolling constant-time crypto is the #1 security anti-pattern. Re-opened later as a separate long-horizon RFC: a Q16.16/deterministic-integer constant-time bignum *showcase*, graduating primitive-by-primitive behind the same §8.4 gate. Does NOT gate Tier 3. |
| **B-hybrid — MIND protocol + FFI to a named C crypto core** | **CHOSEN.** Ships in weeks; battle-tested constant-time core; bounded reviewer cost. |

**The FFI boundary** ("MIND owns bytes-in/bytes-out + parsing + policy; C owns
every operation whose *internals* must be constant-time; the wall clock and RNG
are explicit MIND-owned inputs threaded from the call site"):

- **In MIND:** handshake state machine, record-layer framing, alert protocol,
  ALPN/SNI encoding, AEAD nonce construction (counter⊕IV, deterministic),
  **X.509 parse + chain path-building + name/SAN/validity/EKU policy checks**
  (byte-deterministic parsing/comparison — genuine language-exercise), trust-store
  *loading* (libc file/registry I/O).
- **In the C core (all constant-time internals):** AEAD seal/**open** (AES-128/256-GCM,
  ChaCha20-Poly1305) — incl. the constant-time tag **verify** (a MIND `==` on a MAC
  tag is a textbook timing oracle and is **banned**; tag checks go through the C
  `*_open`/`*_verify` only); SHA-256/384; **HKDF-Extract/Expand + HKDF-Expand-Label
  + HMAC** (no MIND HMAC around a C hash — that re-introduces a const-time surface
  for zero credibility gain); X25519 + P-256 ECDH; signature **VERIFY-ONLY**:
  Ed25519, ECDSA-P256, RSA-PKCS1v1.5 + PSS. **Signing / keygen / the core's own RNG
  entry points are NOT admitted** — they are nonce-randomized (ECDSA `k`, RSA-PSS
  salt) and out of scope; mutual-TLS client-cert *signing* is a separate future
  §8.4 admission, never a free extension of this one.
- **Cert-validation time** is an explicit `ValidationTime` input to `tls_connect`,
  sourced from the `std.async` clock at the call site and recorded in the
  ReplayScheduler trace — **never** an ambient `clock_gettime` inside a
  `#[deterministic]` body (§8.8).

**RNG / entropy — no mode flag.** `std.tls` owns a seeded CSPRNG (ChaCha20/HKDF-DRBG
built on the admitted C primitives). The client-random + ephemeral keyshare scalar
are drawn from it; the seed is a property of the `&Scheduler` (RFC 0011 §10 —
"the type signature is the audit trail") passed at the call site. There is exactly
**one** code path — no `replay` vs `production` flag (that *is* the opt-in escape
hatch §8 forbids). **Production:** the scheduler seed is drawn once at the
entropy-collection boundary from a newly §5-allowlisted OS CSPRNG syscall
(`getrandom` Linux / `getentropy` macOS / `BCryptGenRandom` Windows), called
outside every `#[deterministic]` body; the trace records the seed-*derivation
event + `H(seed)`, **never the raw seed** (a production seed is a secret — logging
it would be a session-key-disclosure oracle). **Replay** is for fixed-seed test
fixtures only; replaying a real production session is structurally impossible.

The decision is recorded in §7 (Q1, resolved) and §8.5 (admitted-bindings table);
it supersedes the default no-FFI rule for the named aws-lc symbol subset only.

Once the scope decision lands, `std.tls` exposes:
- TLS 1.3 client (server-auth only at first; mutual-TLS deferred)
- ALPN, SNI, session resumption (`session_ticket`)
- Trust-store loading: system CA bundle on Linux/macOS, Windows Certificate
  Store on Windows
- A `TlsStream` that implements the same `Read`/`Write` interface as
  `std.net.TcpStream` so HTTP/1.1 layers cleanly on top

Acceptance: connects to a public HTTPS endpoint, verifies the cert chain
against the system trust store, performs a round-trip, and produces a
byte-identical handshake transcript across runs given the same
`ReplayScheduler` seed.

### 4.3 Tier 3 — `std.http` + `std.http.sse`

Effort: 2–3 weeks after Tier 2.

- **`std.http`** — HTTP/1.1 request builder, response parser, header
  handling, **chunked-transfer-encoding decoder** (this is the load-bearing
  piece; SSE rides on it). Keep-alive, redirects (configurable), basic
  cookies. Built directly on `std.tls.TlsStream` for HTTPS / `std.net` for
  plaintext.
- **`std.http.sse`** — `text/event-stream` parser sitting on chunked
  transfer; deterministic event emission (`event:` / `data:` / `id:` /
  `retry:` framing with line-buffered reassembly, `\n\n` record separation).
  Pure parser over the chunked stream: same bytes in, same record sequence
  out, every run.

Acceptance: streams a chunked SSE response end-to-end without buffering the
full body, with byte-identical record ordering under `ReplayScheduler`.

### 4.4 Tier 4 — RFC 0011 Phases B / C / D async I/O

The actual blocker behind Tier 2's and Tier 3's *deterministic-streaming*
claims. Tracked here for cross-reference; design lives in RFC 0011.

- **Phase B** — real executor (work-stealing or fixed-pool), still under
  the explicit `Scheduler` first-class-value model from Phase A
- **Phase C** — async I/O primitives (epoll / kqueue / IOCP), with
  `Future<T>` allocated via `GenRef<T>` per RFC 0010 §3.3
- **Phase D** — `Sender` / `Receiver` channels, `select` / `race`,
  structured task supervision and cancellation

Acceptance: a `ReplayScheduler` run of the agent reproduces a byte-identical
transcript across machines and OSes.

### 4.5 Tier 5 — `std.mcp` client + server

Effort: 3–4 weeks. Can start after Tier 1; acceptance test needs Tiers 2+3
for a real HTTPS-backed MCP.

- **`std.mcp.client`** — JSON-RPC 2.0 framing (over stdio for local servers,
  over HTTP/SSE for remote servers), capability negotiation, tool listing,
  tool invocation with typed arguments
- **`std.mcp.server`** — same surface from the server side, for MIND
  programs that *expose* MCP tools. mind-mem and mind-nerve are the
  natural first downstream consumers — both already speak the MCP surface
  through their current implementations

Acceptance: a MIND CLI calls a stdio MCP server (Claude-flavoured), invokes
a tool, and parses results — full round-trip in pure MIND.

### 4.6 Tier 6 — `mindcraft-agent` demo

Integration sprint after Tiers 1–5 land.

A `claw-code`-class CLI: streaming LLM client (HTTPS + SSE), MCP client,
tool execution (`std.process` sandbox + `std.fs` scoped to a workdir),
terminal UI, `--replay` flag using `ReplayScheduler` for deterministic
playback. Single static binary on each of Linux musl, macOS universal,
Windows MSVC.

Acceptance: byte-identical transcript across all three target OSes under
`ReplayScheduler`; loads a remote LLM endpoint and an MCP server; runs from
the shipped cross-platform binaries (no external runtime required).

---

## 5. Acceptance gates (whole-RFC)

- Every tier deliverable ships with conformance tests in `tests/` and a
  worked example in `examples/`.
- The `mindcraft-agent` demo runs on Linux musl, macOS universal, and
  Windows MSVC from a single source tree, with byte-identical
  `ReplayScheduler` traces verified in CI across all three OSes.
- **No FFI outside libc syscalls** unless admitted under the §8 FFI Discipline
  table. The Tier 2 scope decision (§4.2) is the first admission-path test
  case: the named, version-pinned **aws-lc** C crypto subset enters §8.5 only
  after the cross-substrate byte-identity CI gate passes.
- **OS CSPRNG seed syscalls** (`getrandom`/`getentropy`/`BCryptGenRandom`) are
  added to the allowlist as the single sanctioned entropy seam for the Tier 2
  `std.tls` production RNG seed — explicitly non-deterministic, fenced outside
  every `#[deterministic]` body, recorded in §8.5 for traceability (not a §8.4
  deterministic binding).
- Frontend µs benchmarks: `<±2%` drift across all Phase 16 work — same
  anti-regression gate as Phase 15.

---

## 6. Out of scope

- Full ratatui-class TUI widget framework (a minimal frame buffer is in
  Tier 1; rich widgets are a separate later effort)
- HTTP server (this RFC is client-only; server is mind-inference's concern
  in Phase 12)
- WebSocket (not required for the SSE-based MCP/LLM demo; future RFC)
- HTTP/2 / HTTP/3 (HTTP/1.1 is sufficient for SSE)
- Generic certificate issuance / ACME / Let's Encrypt automation
- Mutual TLS / client certificates (initial Tier 2 ships server-auth only)

---

## 7. Open questions

1. **Tier 2 TLS scope** (the load-bearing decision). **RESOLVED (2026-05-25):
   Option B-hybrid** — pure-MIND TLS 1.3 protocol/parsing/policy + FFI to the
   **aws-lc** C crypto primitive subset (verify-only signatures), admitted via
   §8.4 because crypto primitives are deterministic-by-standard and discharge
   the byte-identity gate trivially. Full rationale + FFI boundary + RNG design
   in §4.2; admitted-bindings entries in §8.5; OS-CSPRNG seed syscalls added to
   the §5 libc allowlist. Pure-MIND primitives deferred to a separate
   long-horizon RFC (not a Tier-3 blocker).

2. **Tier 4 sequencing.** RFC 0011 Phases B–F are "deferred" today. Phase B
   alone is enough for non-streaming HTTP; Phases C + D are required for
   deterministic streaming. The realistic earliest end-to-end Phase 16
   completion (mid-to-late 2027) assumes Phases B + C + D all land. If
   Phases C / D slip, Phase 16 ships in a deterministic-but-blocking mode
   (single-threaded, no I/O readiness multiplexing) — degraded but
   functional. *Decision: ship Tier 4 as part of this RFC's gate rather
   than wait.*

3. **`std.tui` Windows behaviour.** Termios on POSIX has a clean
   equivalent (`SetConsoleMode`) on Windows; the line-editor primitive's
   key-event byte sequences (arrow keys, function keys) differ. *Resolved
   in Tier 1 design block: the `KeyEvent` type is the cross-platform
   abstraction; per-OS readers normalize into it.*

4. **FFI surface discipline.** See §8 for the normative refuse-list,
   admission path, and CI enforcement. The two questions §8 leaves
   underspecified (MLIR-lowered vendor calls, drift-revocation
   procedure) are resolved inside §8.7.

---

## 8. FFI Discipline

> **The runtime's execution predicate is: "produces the identical trace on
> every run." No per-call override exists. Bindings are admitted to the trace,
> not exempted from it.**

The wedge is cross-substrate bit-identity. Any nondeterministic surface anywhere
inside the runtime kills it — Rust f32 reorders under `-O2`, CUDA tensor cores
are nondeterministic by API contract, NumPy auto-promotes to f64 at the FFI
boundary. This section locks the discipline that keeps the wedge intact.

Backed by two rounds of independent cross-review: a strong consensus against
Python/NumPy/PyTorch FFI bridges and against opt-in throughput-mode escape
hatches. Combined finding: any path through nondeterministic numerics — even an
opt-in flag — fractures the trust property because any undefined behavior can
then be blamed on the optional path.

### 8.1 Forbidden FFI symbols

The libc syscall allowlist (named in §5) is the only default-admitted surface
for `extern "C"` calls. Any other binding is forbidden by default, including
but not limited to:

- Python C-API (`Py_*`, `PyObject_*`)
- NumPy C-API (auto-promotes to f64 at the boundary)
- PyTorch ATen (`at::*`, `torch::*`)
- cuBLAS / cuDNN default mode (nondeterministic by API contract)
- oneDNN, Intel MKL default mode
- OpenBLAS default reduction order
- Any libm transcendental called outside the `std.math.det` wrapper

Catch-all: any `extern "C"` block whose symbol resolves outside the libc
allowlist + the §8.5 admitted-bindings table is forbidden.

### 8.2 Forbidden compiler / runtime flags

- `--unsafe-fast`, `--ffast-math` equivalent
- `-ffp-contract=fast` and equivalent FMA-reorder flags
- Any compile flag mutating IEEE-754 rounding or reduction order
- Runtime feature flags that select a nondeterministic codegen path

### 8.3 Forbidden annotation surfaces

- `#[allow(nondeterministic)]` — does not exist; will not be added
- `unsafe { ... }` numeric pragma allowing reorder — forbidden
- `#[trusted_unverified]` — forbidden (admission goes through §8.4 only)
- Per-call escape attribute — forbidden

The execution predicate is monolithic: deterministic or refused.

### 8.4 Admission path for deterministic-by-contract bindings

A non-libc binding may be admitted to the trace if and only if **all four**
conditions hold:

1. **Named symbol subset, not library.** The admission annotation is
   `#[trusted_deterministic(binding = "<symbol>", version = "<x.y.z>")]`.
   It admits a specific symbol at a specific version. A sibling nondeterministic
   entry point in the same library is independently rejected unless separately
   admitted.

2. **Cross-substrate byte-identity CI gate.** A test in the default
   `cargo test --workspace --no-fail-fast` surface (no feature flags) asserts
   byte equality across every RFC 0015 substrate the wedge claims:
   ```
   assert_byte_identical(avx2_ref, neon_ref, admitted_binding_output);
   ```
   No cosine-similarity gate (`≥ 0.9999`) is accepted — byte-equality only.
   Tolerance is the A15 trap.

3. **Named explicitly in §8.5 admitted-bindings table.** The annotation alone
   is not sufficient. The RFC text records each admission ADR-style: motivation,
   alternatives considered, byte-identity test path, sunset trigger.

4. **Reversible / sunset.** Admission re-validates on every major MIND release
   AND on every upstream patch bump (firmware, driver, library). NVIDIA Hopper+
   deterministic-tensor-core firmware revs are the live threat — admit at
   firmware X, silently drift at firmware X+1; the sunset clause forces
   re-validation as a precondition to continued admission.

### 8.5 Admitted-bindings table

| Binding | Version | Admitted | Sunset trigger | Byte-identity test |
| --- | --- | --- | --- | --- |
| **aws-lc** C crypto core — admitted symbol subset ONLY: `EVP_AEAD_CTX` (AES-128/256-GCM, ChaCha20-Poly1305) incl. constant-time `*_open`/`*_seal` tag verify; SHA-256/SHA-384; `HKDF_extract`/`HKDF_expand` + HKDF-Expand-Label + HMAC; X25519 + P-256 `ECDH`; signature **VERIFY-ONLY**: `ED25519_verify`, `ECDSA_verify` (P-256), `RSA_verify` (PKCS1v1.5 + PSS). Signing / keygen / the core's own RNG entry points are **NOT** admitted. | aws-lc — pin a specific GA/FIPS release tag at admission time; **static-linked**; runtime CPU-dispatch (AES-NI vs bitslice) noted as timing-only, output-invariant | _(date the §8.4 gate first passes — Tier 2 build)_ | (a) any aws-lc patch/minor/major bump; (b) FIPS-module revision; (c) any admitted symbol's CAVP/RFC test-vector set changes upstream → re-run §8.4 from scratch | `tests/cross_substrate_identity/tls_primitives.rs` — byte-identical AEAD/hash/HKDF/X25519/P-256-ECDH/verify output across avx2 + neon vs RFC 8439 / NIST CAVP / RFC 7748 known-answer vectors; byte-equality only, no tolerance |
| **OS CSPRNG seed syscalls** (`getrandom` Linux / `getentropy` macOS / `BCryptGenRandom` Windows) — a §5 libc/platform-allowlist extension recorded here for traceability, **NOT** a §8.4 deterministic binding | platform-versioned | _(Tier 2 build)_ | new platform target; or syscall semantics change (e.g. `getrandom` flag default) | N/A — explicitly non-deterministic; the single sanctioned entropy seam, fenced OUTSIDE every `#[deterministic]` body; §8.8 binary-scan allowlists these three symbols by name; a negative test asserts they are unreachable from any `#[deterministic]` entry point |

### 8.6 CI enforcement

Three layers, all required:

1. **Mindcraft lint rule** (RFC 0007). Scans `extern "C"` blocks; flags any
   symbol not in the libc allowlist + admitted-bindings table. Fast feedback
   on `mindc lint`.

2. **Compile-time call-graph check.** `mindc` resolves every call site
   reachable from a `#[deterministic]` body; any reachable symbol outside
   libc + the admitted set is a hard compile error with the offending symbol
   named.

3. **Cross-substrate byte-identity test** (the §8.4 condition 2 test). Lives
   in the default test surface — never behind a feature flag. The full-
   workspace-no-features standing gate ensures this runs on every CI green
   check.

The Mindcraft rule is convenience. The compile-time check is the contract.
The byte-identity test is the proof.

### 8.7 Resolved sub-questions

- **MLIR-lowered vendor calls** count as FFI by another name. A `#[deterministic]`
  fn whose MLIR lowering links against cuBLAS / cuDNN / oneDNN is rejected
  identically to an explicit `extern "C"` call. RFC 0014 per-substrate lowering
  tier descriptors carry the bindings-set explicitly so this is enforceable at
  the IR level, not only at the source level.
- **Equivalence-proof ownership.** STARGA owns the byte-identity test against
  the AVX2/NEON reference. A binding vendor passes the gate or does not get
  admitted; no vendor is asked to provide the proof itself.
- **Revocation procedure when an admitted binding drifts.** Hard-break next
  release, no deprecation window. Drift is always STARGA's call, not the
  binding vendor's. The discipline that keeps the wedge intact must not bend
  to ecosystem-friendliness arguments — every cosine-tolerance once was
  ecosystem-friendliness too.

### 8.8 Implicit nondeterminism gotchas (third-round convergence)

A third round of independent cross-review named three implicit-nondeterminism
sources the §8.1–§8.4 ban list doesn't cover by itself. These are gotchas the
discipline must explicitly address.

**Address-dependent compute.** `malloc` returns nondeterministic addresses
(ASLR + allocator order). Any computation whose output depends on a pointer
value — `ptr as i64`, `ptr::eq`-style identity comparison, address-as-hash-
key — breaks the wedge even when every numeric op is deterministic. A
`#[deterministic]` body MUST NOT branch on, hash, compare, or otherwise
incorporate a raw address value. Mindcraft lint catches the pattern; the
compile-time call-graph check rejects any reachable `as i64` from a raw
pointer inside a deterministic body.

**Transitive dynamic linking.** Even with no explicit `extern "C"` block,
the final linked binary may dynamic-link `libstdc++` (`std::sort` un-stable
order, `std::hash` randomized seed), `libgomp` (OpenMP scheduling), `libmvec`
(vectorized math whose reduction order varies per arch). The libc allowlist
covers the syscall interface, not transitive runtime closures. **Static link
is mandatory for the determinism boundary.** Dynamic loading is admitted
only when the exact SONAME + every transitively-reachable symbol is named
in §8.5 with the same byte-identity gate.

**Binary symbol-table scan.** The §8.6 CI enforcement adds a third sub-step:
after link, scan the final binary's symbol table (e.g., `nm -D` on ELF,
`dumpbin /exports` on PE) and reject any nondeterministic symbol that is
reachable from a `#[deterministic]` entry point. Examples to ban by default:
`drand48`, `srandom`, `time`, `clock_gettime` outside the std.async clock
abstraction, `__cxa_atexit` initialization-order hazards. The reject set is
maintained as a normative appendix to this RFC, updated when new
nondeterministic libc symbols are encountered.

**Vendor definition drift** (named in the third review round, with broad
agreement on the risk shape): a vendor's "deterministic mode" is not a static
contract. A driver / microcode / library patch can silently change behavior.
A wedge split — node A on old driver, node B on new driver — produces
divergent hashes for the same input with no code change on STARGA's side.
The §8.4 condition-4 sunset clause is the structural response: re-validate
on every upstream patch bump, not only major. CI gates on the
`(binding, vendor-version, driver-version, microcode-rev)` 4-tuple, and any
field change forces re-admission through §8.4 from scratch.

The combined principle: the determinism boundary is not the source code — it
is the final binary's reachable-from-deterministic-body symbol set, statically
linked, with every dependency version-pinned and every behavior-affecting
vendor-side parameter pinned alongside it.

---

## 9. Timeline

Realistic earliest end-to-end: **mid-to-late 2027**, gated by:
- Phase 15 self-host landing first (so the agent CLI compiles under
  self-hosted `mindc`)
- ~~The Tier 2 TLS scope decision~~ **RESOLVED (§4.2): B-hybrid on aws-lc** —
  Tier 2 is now a bounded ~6–10 week build (MIND protocol + aws-lc primitive
  FFI), no longer a multi-quarter pure-crypto unknown
- RFC 0011 Phases B + C + D landing (Tier 4 gate)

Tiers 1 + 5 start can begin against the v0.7.0 baseline; everything else
waits on the prerequisites above.
