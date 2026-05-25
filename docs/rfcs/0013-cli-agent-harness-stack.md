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

**Open scope decision** (the load-bearing call in this RFC):

| Option                  | Pros                                                                                              | Cons |
|-------------------------|---------------------------------------------------------------------------------------------------|------|
| **A — pure-MIND TLS 1.3** | Matches the "exercise the language" credibility target; no FFI exception; full audit chain in MIND | Multi-quarter project; cryptographic primitives are a known-hazardous surface; reviewer-cost is high |
| **B — FFI to a named C crypto core** | Ships in weeks not quarters; uses a battle-tested constant-time core (`ring` / `BoringSSL`); reviewer-cost is bounded | Introduces the first non-libc FFI surface; named and version-pinned in *this* RFC if approved |

The acceptance gate (§5) forbids any FFI outside libc syscalls **unless this
RFC names the exception**. The decision is recorded in §7 and supersedes the
default no-FFI rule for the named library only.

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
- **No FFI outside libc syscalls** unless the Tier 2 scope decision (§4.2)
  explicitly approves a named C crypto library, in which case that library
  is the only exception, named and version-pinned in §7 of this RFC.
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

1. **Tier 2 TLS scope** (the load-bearing decision). Pure-MIND TLS 1.3 vs
   FFI to a named version-pinned C crypto core. The default rule is no FFI
   outside libc; this RFC is the only mechanism that can name an exception.
   *Decision deferred until the Tier 2 design block is written.*

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

---

## 8. Timeline

Realistic earliest end-to-end: **mid-to-late 2027**, gated by:
- Phase 15 self-host landing first (so the agent CLI compiles under
  self-hosted `mindc`)
- The Tier 2 TLS scope decision (the 4–6 week swing between pure-MIND and
  FFI-to-named-C-crypto is the largest unknown in the schedule)
- RFC 0011 Phases B + C + D landing (Tier 4 gate)

Tiers 1 + 5 start can begin against the v0.7.0 baseline; everything else
waits on the prerequisites above.
