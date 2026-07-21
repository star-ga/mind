# RFC 0022: Deterministic I/O Substrate — fastest async I/O with bit-identical replay

| Field | Value |
|---|---|
| RFC | 0022 |
| Title | Deterministic I/O Substrate (canonical completion ordering + evidence-anchored reactor) |
| Status | **Draft** — Phase 0 primitives **SHIPPED** (`std.io_canon` canonical ordering + `canon_drain`, `std.arena` region allocator, `std.ring` FIFO staging); reactor core (Phase 1) next. |
| Authors | STARGA Inc. |
| Created | 2026-06-01 |
| Depends | RFC 0015 (cross-substrate bit-identity), RFC 0016 (evidence-chain emission), RFC 0001 (mic@1 determinism) |

## 1. Problem — the runtime I/O path is the last non-deterministic frontier

MIND's artifact layer is already the most efficient and the most auditable: a
canonical binary IR with an embedded, tamper-evident evidence chain (RFC 0016;
cryptographic signing is opt-in, Phase C) whose
deterministic-integer / Q16.16 lowering is byte-identical across CPU substrates —
x86-`avx2` and ARM-`neon` (RFC 0015; GPU float determinism is on the roadmap).
The remaining gap
to *fastest-of-any-language* I/O is the **runtime** path: today `std.net` is
synchronous, blocking libc syscalls with no async runtime.

Every incumbent fast-I/O design (work-stealing schedulers, GC, lock-free CAS
fast-paths, dynamic core allocation, RSS spray, float reassociation) is
**timing-dependent** — its output order depends on hardware interrupts, OS
scheduling, or address layout. That makes "go fast" and "be deterministic"
look like a forced trade-off. They are not.

## 2. The insight — determinism is an oracle, not a tax

Recent research on deterministic parallel execution shows the trade-off
dissolves when determinism is moved to a single **dispatcher** that is a pure
function of request *content*, while synchronization-free workers run flat-out
in a deterministic partial order — no locks, no barriers. The reported result is
*zero* overhead versus the non-deterministic state of the art. That is the proof
MIND builds on: **physical** I/O happens at hardware speed; the **logical**
order in which the program reacts is a pure function of input content, hence
bit-identical and replayable.

Determinism then pays back as a compiler/runtime oracle, not a cost:

- Proven single-owner regions ⇒ strip the atomics/fences a channel-based runtime
  must emit.
- A content-built dependency order ⇒ lock-free without CAS contention (workers
  never touch shared data).
- Region/arena allocation tied to the move-checker ⇒ no GC; tail latency becomes
  a compile-time property.
- Ownership ⇒ exact `noalias`/`dereferenceable` to LLVM ⇒ reorderings an escape
  analysis cannot reach.
- Record/replay of the completion-ring input ⇒ bit-identical reproduction of any
  network bug, shipped *by design* rather than via a tracer.

## 3. Architecture (layered)

```
L5  Protocol specialization   — compile-time (MLIR comptime) parsers/serializers/
                                routers; no runtime bounds-check, static routing.
L4  Determinism boundary      — evidence-chain anchor: a SHA-256 over the ordered
                                completion sequence read out of the ring, BEFORE
                                any application logic (this RFC, §4).
L3  Storage I/O               — userspace / polled NVMe with registered buffers.
L2  Network I/O               — kernel-bypass-class zero-copy Rx/Tx; registered,
                                provided buffer rings; multishot accept/recv;
                                portable backend abstraction.
L1  I/O dispatch              — a single dispatcher reaps the completion ring,
                                re-orders completions into a canonical total order
                                by a content-assigned key, builds the dependency
                                order, dispatches sync-free workers (this RFC, §4).
L0  CPU substrate             — strict one-thread-per-core pinning; the OS
                                scheduler turned into a deterministic, verified
                                substrate. Thread-per-core, share-nothing.
```

**Canonical completion re-ordering (L1) is the load-bearing mechanism.** Physical
completion order off the hardware (interrupt / queue-arrival) is non-deterministic
and varies run to run and across substrates. The dispatcher decouples it from the
*logical* resume order by sorting completions on a key — `(conn_id, req_id)` —
**assigned from request content at accept/submit time, never from arrival
timing**. The drained sequence is therefore a pure function of the completion
multiset, identical regardless of how completions physically arrived.

## 4. What this RFC specifies (Phase 0 — shipped)

Three pure-MIND `std` primitives, each `std-surface`-gated, allocation-bounded,
and deterministic by construction:

- **`std.io_canon`** — the canonical re-ordering stage. A completion is a 4-field
  record `[ conn_id | req_id | op | result ]`. `canon_sort` orders a per-tick
  batch into the canonical total order (`conn_id` asc, then `req_id` asc) with a
  fixed comparison/swap schedule (selection sort: branch-deterministic,
  allocation-free, bit-identical across substrates). The `(conn_id, req_id)` key
  is unique per in-flight operation, so the order is total — no tie-break
  ambiguity that could differ across runs.
- **`canon_drain`** *(this RFC)* — serializes the canonically-ordered records into
  a caller buffer as contiguous 32-byte records, writing only events in
  `[0, len)` (never the backing buffer, so stale prior-tick bytes cannot taint
  the output). **This drained byte sequence is the deterministic I/O input the
  evidence chain anchors** (L4): a `SHA-256` over it, computed at the reactor
  boundary before any application logic, is stable across runs and substrates
  because physical arrival order never enters it. This closes the L1→L4 seam
  using the existing FIPS-180-4 `std.sha256` (RFC 0016's hash), so the I/O
  evidence anchor is the *same* primitive as the artifact evidence chain.
- **`std.arena`** — bump-pointer region allocator: the coroutine-frame and
  per-connection-buffer allocator the reactor needs; deterministic drop, no GC.
- **`std.ring`** — fixed-capacity FIFO byte ring: the per-connection staging
  buffer; compile-time-knowable footprint; byte sequence out == byte sequence in.

### Determinism gate (hard constraint on every increment)

1. Keystone `phase_g_keystone_bootstrap` 7/7 byte-identity must stay green.
2. **Two runs with the same completion multiset in different physical orders must
   produce an identical drained completion sequence** (`canon_drain` byte-identity
   — enforced by `tests/std_surface_io_canon.rs`).
3. Reactor increments additionally gate on a loopback throughput/latency benchmark
   that must not regress.

Any technique whose output depends on hardware timing, OS scheduling, interrupt
order, address layout, or float reassociation is **disqualified** (it would break
the evidence chain). GC, dynamic interrupt-driven core allocation, work-stealing,
unordered float reduction, lock-free CAS-as-fastpath, JIT, ASLR-dependent hashing,
and 0-RTT payload timestamps are all excluded; the deterministic equivalents
(regions, content-ordered dispatch, fixed-tree integer reduction, AOT, static
layout) are adopted in their place.

## 5. Build order

1. **Phase 0 — substrate (shipped):** `io_canon` + `canon_drain`, `arena`, `ring`.
2. **Phase 1 — reactor core:** single ring/thread setup with deterministic flags,
   one reap point, canonical sort, evidence anchor, recv into registered region
   buffers. The first benchmarkable reactor; beats the synchronous path while
   staying replayable.
3. **Phase 2+** — zero-copy registered/provided buffers, multishot socket
   lifecycle, ops-substrate core pinning, storage, transport features, and the
   GPU/cross-substrate tie-in — each landed behind the determinism gate above.

## 6. Cross-substrate & cross-repo

The evidence anchor over the drained completion sequence is, by construction, the
I/O analogue of the cross-substrate bit-identity invariant (RFC 0015): the same
completion multiset yields the same anchor on x86, ARM, and GPU substrates. The
runtime that ships this path consumes these `std` primitives unchanged; the
canonical-ordering and evidence-anchor contracts defined here are the single
source of truth the downstream runtime and substrate backends derive from.
