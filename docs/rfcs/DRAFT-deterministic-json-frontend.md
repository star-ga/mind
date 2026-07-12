# RFC DRAFT: Deterministic Streaming SIMD JSON Structural Front-End

- **Start Date**: 2026-07-11
- **RFC PR**: [#NNNN](https://github.com/star-ga/mind/pull/NNNN) (draft — not yet filed)
- **Status**: **Superseded** — consolidated into
  [`DRAFT-deterministic-format-frontend.md`](DRAFT-deterministic-format-frontend.md)
  (the multi-format generalisation; review THAT draft). The successor also
  carries two corrections this draft lacks: (1) the shipped fold hard-caps at
  **4096 elements** (`PINNED_FOLD_ELEM_CAP`, `src/mlir/lowering.rs:909-911`),
  so bulk columns require the tiled radix-4096 fold specced there (§2e);
  (2) the `tensor<f64[N]>` notation below is aspirational — the memref C-ABI
  admits **concrete static dims only** (`tensor_dims_all_static`,
  `src/eval/abi_gate.rs:124`), so read every `[N]` here as concrete
  `[4096]` tiles. Kept for the JSON-specific determinism argument's history.
- **Owns codegen lane**: `mind-det-gemm` (SIMD emitters), `mind-mlir-lowering`
  (fold pinning), `mind-cross-substrate` (canary custody). This RFC is a
  **design doc to review**, not code to ship.

## Summary

A **second, fast JSON path** for the MIND compiler that turns a large,
columnar/numeric-heavy JSON file into `tensor<f64[N]>` columns that feed the
**already-shipped** Jul-4 deterministic tensor-reduction back-end
(`emit_tensor_reduce_pinned`, `src/mlir/lowering.rs:864`). It follows the
two-stage simdjson pattern (SIMD structural-character index → on-demand
columnar materialisation), streams at record boundaries with bounded memory,
and — the only property incumbents structurally lack — produces
**byte-identical column output across x86 / ARM with a hash-anchored
evidence trace**. simdjson and pandas cannot claim cross-substrate byte-identity.

This is a **separate fast path**. The existing scalar tree-builder
`std/json.mind` (1358 lines) stays exactly as it is for config/evidence
files. This RFC does not touch it, and does not touch the byte-identity,
oracle-parity, or fuzzer gates.

## Motivation

`std/json.mind::jv_parse(buf, len)` is a real, correct, deterministic,
**exact-number** parser — numbers are decomposed into `jv_n_int` /
`jv_n_frac` / `jv_n_frac_d` (`std/json.mind:48-50`, `769`), never a lossy
f64 — but it is a **scalar recursive-descent tree-builder**: byte-by-byte
`jv_load_byte` in `while` loops (`std/json.mind:219`, `508`, `637`), a
136-byte (17×8) heap record **per value**, `jv_max_depth` recursion cap
(`std/json.mind:68`). That is the right shape for a 4 KB config file and the
wrong shape for 300 GB of numeric records: ~100 MB/s single-core and
heap-infeasible (300 GB × record overhead ≫ RAM).

The Jul-4 work shipped the **back half** of a bulk numeric pipeline:

- `4591c00` — `Instr::Sum` / `Instr::Mean` lower on the native
  `mindc build --emit cdylib` path via `emit_tensor_reduce_pinned`: a
  **fully-unrolled fixed left-to-right `arith.addf` chain**, no
  fastmath/reassoc/contract flag, never a `tensor.reduce` /
  `vector.reduction` (`src/mlir/lowering.rs:845-851`, `1020`). objdump-
  confirmed 4×`addsd` / 0 FMA. Bit-identical across x86 avx2 / ARM neon **by
  construction** (a fixed-order IEEE-754 RNE chain has no reassociation
  freedom — the `scalar-float-f64` canary logic).
- `0840a24` — `param_non_i64` lets a **static-shape tensor parameter** lower
  to a real memref C-ABI `{alloc_ptr, aligned_ptr, offset, size, stride}`, so
  reductions are **runtime-fed**: `fn ksum4(t: tensor<f64[4]>) -> f64
  { t.sum() }` builds and runs exact. keystone 7/7 green; the cross_substrate
  identity suite (13 canaries at ship time; 20 canary dirs today) stayed
  byte-identical.

What is missing is the **front half**: getting the bytes off disk into those
`tensor<f64[N]>` columns *deterministically*. That is what this RFC specs.

## Guide-level explanation

Two named concepts:

- **The structural index (stage 1).** A SIMD scan classifies every byte of a
  chunk as one of the JSON structural characters `{ } [ ] , : "` /
  whitespace / other, producing a **structural bitmap**. This is a
  *classification*, not a reduction — and that distinction is the whole
  determinism story (see below).
- **Columnar materialisation (stage 2).** An on-demand traversal walks the
  bitmap and writes numeric values straight into flat, per-column tensor
  buffers (`tensor<f64[N]>` tiles). No 136-byte-per-value tree. The buffers
  are exactly the shape `param_non_i64` already accepts.

A MIND program would look like:

```mind
// EXISTING (config/evidence) — unchanged, scalar exact tree:
let root: i64 = jv_parse(buf, len);

// NEW (bulk numeric) — streaming columnar front-end feeding the Jul-4 fold:
let col: tensor<f64[N]> = jc_column_f64(chunk_ptr, chunk_len, "price");
let total: f64 = col.sum();   // lowers via emit_tensor_reduce_pinned, exact
```

The MIND programmer should think of `jc_column_f64` as "the fast path that
only exists for arrays/tables of numbers, and gives me the *same bytes* on my
laptop and on an ARM server — with a trace I can verify",
and `jv_parse` as "the correct, exact, general path for everything else".

## Reference-level explanation

### 1. Architecture — two-stage, streaming, separate fast path

```
                 300 GB JSON on NVMe
                        │
        ┌───────────────┴───────────────┐  CPU splits byte-range chunks
        │  CHUNKER: record-boundary cut │  at record boundaries
        └───────────────┬───────────────┘
                        │  chunk = [start, end) aligned to a top-level
                        │          record boundary; bounded (e.g. 16-64 MB)
        ┌───────────────▼───────────────┐
        │ STAGE 1: SIMD structural scan │  { } [ ] , : "  + whitespace
        │  → structural bitmap (per     │  classify-only, NO reduction
        │    chunk, bounded)            │  (emitters like the GEMM kernels)
        └───────────────┬───────────────┘
        ┌───────────────▼───────────────┐
        │ STAGE 2: on-demand traversal  │  walk bitmap; parse numbers via
        │  → columnar tensor buffers    │  EXACT int/frac decomp; write into
        │    tensor<f64[N]> tiles       │  flat per-column f64 tiles
        └───────────────┬───────────────┘
                        │  tiles are static-shape tensor params
        ┌───────────────▼───────────────┐
        │ JUL-4 BACK-END (SHIPPED)      │  emit_tensor_reduce_pinned:
        │  t.sum() / t.mean()           │  fixed L→R arith.addf, no fastmath
        └───────────────┬───────────────┘  → byte-identical x86/ARM
                        ▼
              deterministic scalar result + evidence trace
```

**Contrast with the scalar tree-builder (explicit):** `jv_parse` builds a
**pointer-linked heap tree** and returns a root handle; every value is a
136-byte record and traversal is recursive-descent. The new path builds **no
tree** — stage 1 emits a bitmap, stage 2 streams values into contiguous
columns. They share the language surface (both are `.mind`-callable) and the
exact-number decomposition; they share **nothing** in data structure or
throughput class. `std/json.mind` is not modified, deprecated, or re-routed.

**Streaming / chunking.** The chunker cuts the input into bounded byte ranges
aligned to **top-level record boundaries**. Memory is bounded by
`chunk_size + column_tile_size`, independent of the 300 GB total. Chunk
boundaries are the primary determinism hazard (§2).

### 2. The determinism problem — the actual alg-invent target

A SIMD/vectorised scan is **reduction-order-sensitive** if written naively.
The hard, defensible invention is a **reduction-order-INVARIANT structural
scan** that stays bit-identical across the x86 and ARM SIMD ISAs — the *same
discipline already proven* for the int-dot GEMM ladder. The three places
nondeterminism sneaks in, and how each is pinned:

**(a) SIMD lane reduction order — PINNED by "classify, don't reduce".**
Stage 1 must NOT compute a cross-lane reduction whose grouping differs by
ISA. It computes a per-byte **classification** and packs the result into a
bitmap — a lane-local, position-preserving operation (compare-against-
structural-char-set → movemask/`vpmovmskb`-class pack on x86, the
`shrn`/`addv`-free bit-narrow idiom on ARM). Because each output bit depends
only on its own input byte and the *fixed* structural-char set, there is **no
cross-lane accumulation order to disagree on**. This is the direct analogue
of why the int-dot ladder is safe: `__mind_blas_matmul_mm_i8_v`'s inner loop
legalises to AVX2 `vpmaddwd` on x86 and `SDOT`/`SMMLA` on aarch64, and *both
produce the identical exact int32 sum* (`src/mlir/lowering.rs:160-165`) — the
value, not the instruction, is pinned. Stage 1's bitmap is even stronger:
it's an exact bit-pack with zero accumulation, so ISA choice cannot change it.

> deferred: the ARM bitmap-pack idiom (NEON has no direct 1-byte-per-lane
> movemask) needs a pinned lowering that provably reproduces the x86
> `vpmovmskb` bit order — upgrade path: a `jc_structural_scan` MLIR emitter
> mirroring `emit_i8_microkernel_avx2` / its neon sibling, blessed against a
> `json-structural-scan` cross-substrate canary on real aarch64 before the
> neon line is committed (same DEFER discipline as `dot-f32-v-4093`:
> `tests/cross_substrate_identity/dot-f32-v-4093/reference_hashes.toml:19`).

**(b) Float accumulation — NOT in the front-end at all.** The front-end does
**no float reduction**. It parses numbers and writes them into a column; the
*only* reduction is `t.sum()` / `t.mean()`, which is the shipped
`emit_tensor_reduce_pinned` fixed L→R `arith.addf` chain with no fastmath
(`src/mlir/lowering.rs:845-851`). So float determinism is inherited from an
already-proven, canary-covered back-end, not re-invented. The front-end's job
is only to place each value at a **deterministic column index** (see (c)).

**(c) Number parsing — EXACT int/frac, no lossy f64 mid-stream.** Number
parsing must preserve the existing exact decomposition (`jv_n_int` /
`jv_n_frac` / `jv_n_frac_d`, `std/json.mind:769`) so the parsed value is a
function of the *bytes only*, identical on every substrate. The f64 column
element is produced by a **single, fully-specified IEEE-754 RNE conversion**
from that exact integer/frac decomposition — the same "one rounding, no
accumulation" property that makes the `scalar-float-f64` canary architecture-
independent (`tests/cross_substrate_identity/scalar-float-f64/reference_hashes.toml`:
"round-to-nearest-even with a single, fully-specified result — avx2 == neon by
IEEE construction"). No intermediate f64 arithmetic, no `strtod` locale/impl
divergence.

**(d) Chunk-boundary handling — PINNED by boundary determinism, not scan
determinism.** Two hazards: (i) a value straddling a chunk edge, (ii) the
**global column order** — element *k* of the output column must be the same
regardless of how the input was chunked or how many threads ran. Pin both by:
- **Deterministic boundary rule** — a chunk always ends at the byte after a
  complete top-level record; a straddling record is fully owned by the
  *lower* chunk (a total, position-only rule; no heuristic).
- **Stable column indexing** — element index = **document record order**, not
  completion order. Threads write into pre-assigned index ranges derived from
  the deterministic chunk cut, so multi-threaded materialisation is
  `MT == ST == reference`, exactly the invariant the `gemm-i8-mt` canary
  proves for thread-band GEMM (`tests/cross_substrate_identity/gemm-i8-mt-64x64x64/`).
  Thread count must never be observable in the output.

> deferred: cross-chunk straddling-value reassembly is the fiddliest correctness
> case — stubbed in Phase 1 (single in-memory array, no chunk edges) — upgrade
> path: a carry-buffer at each chunk tail re-scanned by the next chunk's stage 1,
> with a boundary canary asserting a value split across the edge yields the same
> column bytes as the unsplit input.

### 3. Where it plugs into the compiler

Realistic layer map (what is genuinely new vs. reuse):

| Piece | Layer | New or reuse |
|---|---|---|
| SIMD structural scan | `src/mlir/lowering.rs` emitter, sibling to `emit_i8_microkernel_avx2` / the neon path | **NEW emitter**, reuses the ISA-ladder *pattern* + fail-loud canary discipline |
| Exact number decode | reuse `std/json.mind` number logic (`jv_parse_number`, `:769`) lifted into a column-writing helper | **Reuse** (algorithm), new call-site |
| Column → tensor param | `param_non_i64` memref C-ABI (`0840a24`) + one-shot-bufferize tensor→memref boundary (`src/mlir/lowering.rs:3484`, `10518`) | **Reuse, unchanged** |
| `t.sum()` / `t.mean()` fold | `emit_tensor_reduce_pinned` (`:864`) | **Reuse, unchanged** |
| New std surface | `jc_structural_scan`, `jc_column_f64(ptr, len, field) -> tensor<f64[N]>` intrinsics/externs | **NEW surface** |
| Evidence trace | existing `trace_hash = mini_sha256(emit_mic3(ir))` seam (`src/ir/evidence.rs`) | **Reuse, unchanged** — the mic@3 hash already covers the emitted IR |

Genuinely new: (1) the stage-1 SIMD classify-and-pack emitter and its ARM
sibling; (2) the `jc_*` std surface; (3) the streaming chunker + stable
column-index assignment. Everything downstream of "a filled `tensor<f64[N]>`
tile" is shipped machinery, untouched.

The **new intrinsic surface** mirrors the `__mind_blas_*` externs
(`src/mlir/lowering.rs:30-198`): a `VEC_JSON_SCAN_INTRINSIC` recognised by the
lowering, with a Track-A scalar oracle (`std/json.mind`'s own byte loop) as
the byte-identity reference the SIMD path must match at every input — the
exact fail-closed pattern the int-dot intrinsics already use ("byte-identical
to Track A's scalar `__mind_blas_dot_q16`", `:2495`).

### 4. The I/O floor — honesty rail

At 300 GB, **NVMe sequential read (~4 GB/s) is a 60-100 s floor**. A parser
that is not I/O-bound at this scale is not worth building — and simdjson-class
stage-1 scan (~3-7 GB/s/core, a few cores) *is* faster than one NVMe stream,
so the pipeline **should be I/O-bound**.

State it plainly: **the speed race is a wash at this scale.** We do not claim
a throughput win over simdjson and must not put a MIND-vs-X GB/s
number on any public surface. The **only** differentiated, defensible claim
is: *deterministic + cross-substrate byte-identical + evidence-anchored* — the
three things simdjson (SIMD-lane-order-dependent) **cannot** offer. The value
is not "faster"; it is "the same bytes everywhere, provably, at streaming
speed you'd pay for anyway."

### 5. Phased plan (each phase gated on keystone 7/7 + cross_substrate)

**Phase 0 — single numeric-array column, in-memory, no SIMD (smallest
end-to-end slice).** Input: one JSON numeric array `[1.0, 2.0, ...]` already
in a buffer. Reuse `std/json.mind` number decode in a straight loop to fill a
`tensor<f64[N]>`, then `t.sum()` via the shipped fold. Proves the *pipeline*
(bytes → column → deterministic sum, byte-identical x86/ARM) with **zero new
SIMD**. Gate: a new `json-col-sum` cross-substrate canary + keystone still
7/7. This is the runnable proof the whole RFC rests on.

**Phase 1 — SIMD stage-1 scan (x86), scalar-oracle-checked.** Add the
`jc_structural_scan` AVX2 emitter; assert byte-identical to the
`std/json.mind` scalar classifier at every input (Track-A oracle pattern).
neon line **DEFERRED** pending real-aarch64 bless. Gate: oracle-parity holds,
x86 canary blessed, keystone 7/7.

**Phase 2 — stage-2 columnar materialisation + multi-column.** On-demand
traversal into multiple `tensor<f64[N]>` tiles; stable document-order column
indexing. Gate: `json-multicol` canary, MT==ST index-order proof (gemm-i8-mt
pattern).

**Phase 3 — streaming chunker + cross-chunk straddle.** Bounded-memory
chunking at record boundaries; carry-buffer straddle reassembly. Gate: a
boundary canary (split == unsplit column bytes) + full run on a large file
I/O-bound.

**Phase 4 — ARM bless.** Bless the neon stage-1 line on real
aarch64 (Ampere Altra, the `dot-f32-v` unblock path). A GPU stage-1 rung is
commercial-runtime work behind the open-core `GPUBackend` contract, tracked
privately — NOT this RFC.

> deferred: Phases 1-4 are DESIGN ONLY in this draft. Only Phase 0 is claimed
> runnable-today (it is pure reuse of shipped machinery). Every SIMD emitter,
> the chunker, and the ARM bless are unbuilt — upgrade path is the phase
> gates above, each blocked on its own canary + keystone 7/7.

### What it is NOT (scope fence)

- **NOT a GPU-parse-first engine.** CPU does stage 1. Any GPU rung is behind the
  open-core `GPUBackend` contract and must reproduce the CPU bytes, not
  reassociate floats — commercial-runtime work, tracked privately.
- **NOT a general fast-JSON-lib race.** No strings/objects/nesting fast path
  beyond what columnar numeric extraction needs; `jv_parse` remains the
  general parser. No public throughput claim (§4).
- **NOT touching the gates.** The byte-identity canary suite, oracle-parity
  linter, and differential fuzzer are *consumed* (new canaries added), never
  bypassed or weakened. `trace_hash`/mic@3 seam unchanged.
- **AOT vs runnable honesty.** Phase 0 is runnable on the shipped
  `mindc build --emit cdylib` path today. Phases 1+ are AOT-emitter work that
  does not exist yet and is marked deferred throughout.

## Drawbacks

- Two JSON paths to maintain; risk of drift between the scalar oracle and the
  SIMD path (mitigated by mandatory oracle-parity at every input).
- The differentiator (determinism) is invisible to a benchmark-only reader who
  only sees "not faster than simdjson." The pitch is governance/reproducibility,
  a narrower audience than raw-speed JSON.
- Real ARM blesses are hardware-gated; the neon stage-1 line stays DEFERRED
  until Ampere silicon time, exactly like `dot-f32-v`.

## Rationale and alternatives

- **Why classify-not-reduce for stage 1:** it removes the cross-lane
  accumulation-order hazard entirely, rather than trying to pin a specific
  reduction tree across two ISAs. Strictly safer than the int-dot ladder
  (which does pin a reduction) because a bitmap pack has no accumulation.
- **Alternative — just make `jv_parse` faster:** rejected; a 136-byte-per-value
  heap tree cannot reach streaming throughput or bounded memory at 300 GB
  regardless of scan speed. The data structure is the ceiling.
- **Alternative — bind simdjson as an extern:** rejected; it cannot
  produce cross-substrate byte-identical output or feed the evidence seam, so
  it would forfeit the only differentiated claim.
- **Impact of not doing this:** MIND has a deterministic *reduction* back-end
  with no deterministic *bulk ingest* to feed it; the wedge story stops at
  "config files."

## Prior art

- **simdjson** — the two-stage (structural index → on-demand) pattern this
  reuses; its stage-1 reduction is SIMD-lane-order-dependent, so it cannot
  claim cross-substrate byte-identity.
- **MIND int-dot GEMM ladder** (`src/mlir/lowering.rs:160-259`) — the proof
  that "pin the value, not the instruction" holds across `vpmaddwd` /
  `SDOT` / `SMMLA`; stage 1 applies the same discipline to classification.

## Unresolved questions

- The exact ARM stage-1 bit-pack idiom that reproduces x86 `vpmovmskb` bit
  order (owned by `mind-det-gemm`; resolved at Phase 1/4 with a real-aarch64
  canary bless).
- Column dtype coverage beyond f64 (i64 columns are associative-add, already
  bit-identical — the deferred integer-reduce tier at `:860`).
- Whether the chunker lives in `.mind` std or as a compiler-side driver
  (leaning std, to keep it in the governed language surface).

## Future possibilities

- A GPU stage-1 that reproduces CPU bytes, behind the open-core `GPUBackend`
  contract — commercial-runtime work, tracked privately.
- Extend the columnar front-end to feed the int-dot GEMM path (numeric tables →
  `tensor` → deterministic matmul), not just reductions.
- A `mindc verify`-visible column-hash so a downstream party can attest a 300 GB
  ingest produced identical columns on their own substrate.
