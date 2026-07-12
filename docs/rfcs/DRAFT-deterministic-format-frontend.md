# RFC DRAFT: Deterministic Multi-Format Ingest Front-End (JSON / TOON / CSV / TSV / NDJSON / TOML)

- **Start Date**: 2026-07-11
- **RFC PR**: [#NNNN](https://github.com/star-ga/mind/pull/NNNN) (draft — not yet filed)
- **Status**: Draft
- **Consolidates & supersedes**: `docs/rfcs/DRAFT-deterministic-json-frontend.md`
  (JSON-only, now Status: Superseded). This RFC absorbs its determinism
  argument and I/O-floor honesty rails, generalises to a **format-agnostic**
  ingest pipeline (per-format stage-1 / shared stage-2+), and corrects two
  claims the JSON draft got wrong: the shipped fold's 4096-element cap and the
  `tensor<f64[N]>` notation (the ABI admits concrete static dims only) — both
  specced in §2e.
- **Owns codegen lane**: `mind-det-gemm` (SIMD emitters), `mind-mlir-lowering`
  (fold pinning), `mind-cross-substrate` (canary
  custody). This RFC is a **design doc to review**, not code to ship.

## Summary

A **format-agnostic bulk ingest front-end** for the MIND compiler that turns a
large, columnar/numeric-heavy file — in any of a family of line-oriented /
low-overhead formats (**JSON, TOON, CSV, TSV, NDJSON/JSONL, TOML**) — into
`tensor<f64[4096]>` column tiles that feed the **already-shipped** Jul-4
deterministic tensor-reduction back-end (`emit_tensor_reduce_pinned`,
`src/mlir/lowering.rs:864`; the 4096 tile is the fold's own
`PINNED_FOLD_ELEM_CAP`, `:909` — see §2e for bulk scale). It follows the
two-stage simdjson pattern
(SIMD structural index → on-demand columnar materialisation), streams at
record boundaries with bounded memory, and — the only property incumbents
structurally lack — produces **byte-identical column output across x86 / ARM
with a hash-anchored evidence trace**. simdjson and pandas
cannot claim cross-substrate byte-identity for any format.

The central design insight is that **all of these formats share one target**.
A CSV row, an NDJSON record, and a TOON tabular array all decode into the same
column/tensor shape; only the **stage-1 structural scanner** is per-format.
Everything downstream — indexing, columnar materialisation, `map` → `mic@x` —
is **format-independent and shared**.

This is a **separate fast path**. The existing scalar tree-builders
`std/json.mind` (1358 lines) and `std/toml.mind` (1405 lines) stay exactly as
they are for config/evidence files. This RFC does not modify them, and does
not touch the byte-identity, oracle-parity, or fuzzer gates.

## Motivation

MIND already has two real, correct, deterministic **scalar** parsers, and they
are the wrong shape for bulk:

- `std/json.mind::jv_parse(buf, len)` — an **exact-number** recursive-descent
  tree-builder. Numbers decompose into `jv_n_int` / `jv_n_frac` / `jv_n_frac_d`
  (`std/json.mind:48-50`, `769`), never a lossy f64. But it is **scalar**:
  byte-by-byte `jv_load_byte` in `while` loops (`std/json.mind:219`, `508`,
  `637`), a **136-byte (17×8) heap record per value** (`std/json.mind:15`,
  `149`), a `jv_max_depth` recursion cap (`:68`). Right for a 4 KB config,
  wrong for 300 GB: ~100 MB/s single-core, heap-infeasible.
- `std/toml.mind::toml_parse` — a TOML 1.0 **subset** parser (bare/quoted keys,
  int/hex/bool, arrays, standard/inline tables), also scalar: a `load_byte`
  helper (`std/toml.mind:391` = `__mind_load_i8(buf + i)`) in `while` loops
  (`:176`, `:258`, `:282`, `:334`), a **104-byte (13×8) heap record per value**
  (`std/toml.mind:22`), a `MAX_DEPTH` 512 recursion cap (`:50`). Same
  ~100 MB/s class; explicitly **not** built for 100s of GB. Note it **defers
  float, datetime, multi-line strings** (`std/toml.mind:16`) — so as a *bulk
  numeric* source it is currently integer/bool-only until that Phase-2 float
  work lands.

There is **no CSV, TSV, NDJSON/JSONL, or TOON parser in `std` today** (verified:
`ls std/` shows `json.mind` and `toml.mind` only; no `csv`/`tsv`/`ndjson`/
`toon`). This RFC's stage-1 scanners for those formats do not exist yet and are
marked design-only throughout.

The Jul-4 work shipped the **back half** of a bulk numeric pipeline — this is
the load-bearing fact the whole RFC rests on:

- `4591c00` — `Instr::Sum` / `Instr::Mean` lower on the native
  `mindc build --emit cdylib` path via `emit_tensor_reduce_pinned`: a
  **fully-unrolled fixed left-to-right `arith.addf` chain**, no
  fastmath/reassoc/contract flag, never a `tensor.reduce` /
  `vector.reduction` (`src/mlir/lowering.rs:864`, dispatched at `:4012`/`:4020`).
  objdump-confirmed 4×`addsd` / 0 FMA. Bit-identical across x86 avx2 / ARM neon
  **by construction** (a fixed-order IEEE-754 RNE chain has no reassociation
  freedom — the `scalar-float-f64` canary logic).
- `0840a24` — a **static-shape tensor parameter** lowers to a real memref
  C-ABI `{alloc_ptr, aligned_ptr, offset, size, stride}` (the `param_non_i64`
  gate, `src/eval/abi_gate.rs:112`), so reductions are **runtime-fed**:
  `fn ksum4(t: tensor<f64[4]>) -> f64 { t.sum() }` builds and runs exact.

The deterministic reduction kernel over parsed columns **EXISTS and SHIPS**.
The fast bulk parser front-end **does NOT**. What is missing is getting the
bytes off disk into those `tensor<f64[4096]>` column tiles *deterministically*, from
*any* of these formats. That is what this RFC specs.

### Why one front-end for many formats (TOON context)

- **TOON and JSON share a target — losslessly.** TOON v1.3.3 is a *lossless*
  encoding of the JSON data model: indentation for nesting plus CSV-style
  tabular arrays with explicit `[N]` lengths and `{fields}` headers. A TOON
  document round-trips through JSON with no information loss, so TOON → the
  **same** column/tensor target as JSON; only the stage-1 structural scanner
  differs (indentation + `[N]`/`{fields}` framing instead of `{}[],:"`). The
  same holds for CSV/TSV (delimiter + newline framing) and NDJSON (one JSON
  value per line). This is why the pipeline is format-*agnostic*, not a bag of
  N separate parsers.
- **The differentiator is determinism, not throughput** (§ I/O floor): the
  cross-substrate byte-identity + the `map`/`mic@x` evidence anchor, which the
  incumbent parsers structurally lack.

## Guide-level explanation

Three named concepts — one per-format, two shared:

- **The structural classifier (stage 1, PER-FORMAT).** A SIMD scan classifies
  every byte of a chunk into that format's structural set and packs the result
  into a **structural bitmap**. JSON classifies `{ } [ ] , : "` / whitespace;
  CSV/TSV classify delimiter + quote + newline; NDJSON classifies record-
  separator newlines; TOON classifies indentation + `[N]`/`{fields}` framing.
  This is a *classification*, not a reduction — the whole determinism story
  (§ "classify, don't reduce"). It is the **only** per-format stage.
- **Structural indexing (stage 2, SHARED).** A format-independent step walks
  the bitmap and produces a **record/field index** (offsets of where each
  logical value begins). Because stage 1 already normalised every format into
  the same bitmap shape, this step has no format-specific branches.
- **Columnar materialisation (stage 3, SHARED).** An on-demand traversal reads
  the index and writes numeric values straight into flat, per-column tensor
  buffers (`tensor<f64[4096]>` tiles — the pinned-fold cap is the tile unit,
  §2e). No per-value heap tree. The buffers are exactly the shape
  `param_non_i64` already accepts, and feed `t.sum()` per tile + the tiled
  fold across tiles (§2e).

A MIND program would look like:

```mind
// EXISTING (config/evidence) — unchanged, scalar exact trees:
let root: i64 = jv_parse(buf, len);      // std/json.mind
let cfg:  i64 = toml_parse(buf, len);    // std/toml.mind

// NEW (bulk numeric) — one front-end, format chosen by the stage-1 scanner.
// NOTE: dims are CONCRETE static — the memref C-ABI rejects symbolic `[N]`
// (`tensor_dims_all_static`, src/eval/abi_gate.rs:124) and the pinned fold
// caps at 4096 elements (§2e). A bulk column is a sequence of 4096-element
// tiles combined by the tiled fold.
let tile: tensor<f64[4096]> = mfc_column_f64(chunk_ptr, chunk_len, FMT_NDJSON, "price");
let total: f64 = tile.sum(); // lowers via emit_tensor_reduce_pinned, exact
```

The MIND programmer should think of `mfc_column_f64` as "the fast path that
exists for arrays/tables of numbers *in any supported format*, and gives me the
*same bytes* on my laptop and on an ARM server — with a trace
I can verify", and `jv_parse` / `toml_parse` as "the correct, exact, general
paths for everything else".

## Reference-level explanation

### 1. Architecture — format-agnostic pipeline (per-format stage-1, shared stage-2+)

```
              100s of GB (JSON | TOON | CSV | TSV | NDJSON | TOML) on NVMe
                                    │
        ┌───────────────────────────┴───────────────────────────┐
        │  CHUNKER (SHARED): record-boundary cut, bounded chunk  │
        └───────────────────────────┬───────────────────────────┘
                                    │  chunk = [start, end) aligned to a
                                    │          top-level record boundary
        ┌───────────────────────────▼───────────────────────────┐
        │  STAGE 1  ── PER-FORMAT structural SIMD classifier ──   │  classify-only,
        │  ┌──────────┬──────────┬──────────┬──────────┬───────┐ │  NO reduction
        │  │  JSON    │  TOON    │  CSV/TSV │  NDJSON  │ TOML  │ │  (pluggable;
        │  │ {}[],:"  │ indent + │ delim +  │ record   │ key=  │ │   sibling to
        │  │  + ws    │ [N]{...} │ quote+NL │ NL       │ table │ │   GEMM kernels)
        │  └──────────┴──────────┴──────────┴──────────┴───────┘ │
        │                    → structural bitmap                  │  <── the only
        └───────────────────────────┬───────────────────────────┘      format branch
        ┌───────────────────────────▼───────────────────────────┐
        │  STAGE 2  ── SHARED structural indexing ──             │  bitmap → record/
        │            → record/field offset index                 │  field offsets;
        └───────────────────────────┬───────────────────────────┘  no format branch
        ┌───────────────────────────▼───────────────────────────┐
        │  STAGE 3  ── SHARED columnar materialisation ──        │  EXACT int/frac
        │            → tensor<f64[4096]> column tiles            │  number decode →
        └───────────────────────────┬───────────────────────────┘  flat f64 tiles
                                    │  tiles are static-shape tensor params
        ┌───────────────────────────▼───────────────────────────┐
        │  JUL-4 BACK-END (SHIPPED) ── map → mic@x anchor ──     │  emit_tensor_reduce_
        │  t.sum() / t.mean() per tile                           │  pinned: fixed L→R
        │  + radix-4096 tiled fold across tiles (§2e)            │  arith.addf, no
        └───────────────────────────┬───────────────────────────┘  fastmath
                                    ▼
              deterministic scalar result + mic@3 evidence trace
              (byte-identical x86 / ARM)
```

The **format boundary is stage 1 and stage 1 only**. Adding a new format
(say Parquet-lite, later) is "write one more stage-1 classifier + its ARM
sibling + its canary" — stages 2, 3, and the Jul-4 back-end are reused
unchanged. This is the reuse leverage the whole RFC buys.

**Contrast with the scalar tree-builders (explicit):** `jv_parse` /
`toml_parse` build **pointer-linked heap trees** and return root handles; every
value is a 136-byte / 104-byte record and traversal is recursive-descent. The
new path builds **no tree** — stage 1 emits a bitmap, stage 2 indexes it, stage
3 streams values into contiguous columns. They share the language surface (all
`.mind`-callable) and the exact-number decomposition; they share **nothing** in
data structure or throughput class. `std/json.mind` and `std/toml.mind` are not
modified, deprecated, or re-routed.

**Streaming / chunking (shared).** The chunker cuts input into bounded byte
ranges aligned to **top-level record boundaries**.
Memory is bounded by `chunk_size + column_tile_size`, independent of the total.
The record-boundary rule is per-format only in *what* counts as a boundary
(a `}` at depth 0 for JSON; a newline for NDJSON/CSV; a dedent for TOON) — the
mechanism is shared. Chunk boundaries are the primary determinism hazard (§2d).

### 2. The determinism problem — "classify, don't reduce"

A SIMD/vectorised scan is **reduction-order-sensitive** if written naively.
The defensible invention is a **reduction-order-INVARIANT structural scan**
that stays bit-identical across x86 and ARM SIMD ISAs — the *same discipline
already proven* for the int-dot GEMM ladder. The places nondeterminism sneaks
in, and how each is pinned:

**(a) SIMD lane reduction order — PINNED by "classify, don't reduce".**
Stage 1 must NOT compute a cross-lane reduction whose grouping differs by ISA.
It computes a per-byte **classification** and packs the result into a bitmap —
a **lane-local, position-preserving** operation with **zero cross-lane float
accumulation** (compare-against-structural-set → `vpmovmskb`-class pack on x86,
the `shrn`/narrow bit-pack idiom on ARM). Because each output bit depends only
on its own input byte and the *fixed, per-format* structural set, there is **no
cross-lane accumulation order to disagree on**. This is the direct analogue of
why the int-dot ladder is safe: the inner loop legalises to AVX2 `vpmaddwd` on
x86 and `SDOT`/`SMMLA` on aarch64 and *both produce the identical exact int32
sum* — the value, not the instruction, is pinned. Stage 1's bitmap is
*stronger*: an exact bit-pack with zero accumulation, so ISA choice cannot
change it. **This property is format-independent** — it holds for every
stage-1 classifier because none of them accumulate across lanes.

**(b) Float accumulation — NOT in the front-end at all.** The front-end does
**no float reduction** for any format. It classifies bytes and writes parsed
numbers into a column; the *only* reduction is `t.sum()` / `t.mean()`, the
shipped `emit_tensor_reduce_pinned` fixed L→R `arith.addf` chain with no
fastmath (`src/mlir/lowering.rs:864`). Float determinism is **inherited** from
an already-proven, canary-covered back-end, not re-invented per format. The
front-end's only job is to place each value at a **deterministic column index**
(see (d)). One honesty caveat at bulk scale: combining tile partials is *also*
a float reduction — it is owned by the back-half's tiled fold (§2e), still
never by stages 1–3.

**(c) Number parsing — EXACT int/frac, no lossy f64 mid-stream.** Number
parsing preserves the existing exact decomposition (`jv_n_int` / `jv_n_frac` /
`jv_n_frac_d`, `std/json.mind:769`) — lifted into a **shared** column-writing
helper so CSV, NDJSON, TOON, and JSON all decode numbers identically — so the
parsed value is a function of the *bytes only*, identical on every substrate.
The f64 column element is produced by a **single, fully-specified IEEE-754 RNE
conversion** from that exact integer/frac decomposition — the "one rounding, no
accumulation" property that makes the `scalar-float-f64` canary
architecture-independent
(`tests/cross_substrate_identity/scalar-float-f64/reference_hashes.toml`). No
intermediate f64 arithmetic, no `strtod` locale/impl divergence.

**(d) Chunk-boundary handling — PINNED by boundary determinism, not scan
determinism.** Two hazards, both shared across formats: (i) a value straddling
a chunk edge, (ii) the **global column order** — element *k* of the output
column must be the same regardless of chunking or thread count. Pin both by:
- **Deterministic boundary rule** — a chunk always ends at the byte after a
  complete top-level record; a straddling record is fully owned by the *lower*
  chunk (a total, position-only rule; no heuristic).
- **Stable column indexing** — element index = **document record order**, not
  completion order. Threads write into pre-assigned index ranges derived from
  the deterministic chunk cut, so multi-threaded materialisation is
  `MT == ST == reference`, exactly the invariant the `gemm-i8-mt` canary
  proves for thread-band GEMM
  (`tests/cross_substrate_identity/gemm-i8-mt-64x64x64/`). Thread count must
  never be observable in the output.

> deferred: cross-chunk straddling-value reassembly is the fiddliest
> correctness case, and it is format-shaped (a number split across a CSV cell
> boundary vs a JSON token boundary) — stubbed in Phase 0/1 (single in-memory
> buffer, no chunk edges). upgrade path: a shared carry-buffer at each chunk
> tail re-scanned by the next chunk's stage 1, with a per-format boundary
> canary asserting a value split across the edge yields the same column bytes
> as the unsplit input.

**(e) Bulk scale — the 4096-element pinned-fold cap and the tiled fold.**
The shipped fold is not unbounded: `emit_tensor_reduce_pinned` hard-rejects
any reduction whose element count exceeds `PINNED_FOLD_ELEM_CAP = 4096`
(`src/mlir/lowering.rs:909-911`, applied *after* the f32/f64 gate at `:885`),
and the memref C-ABI admits only **concrete static dims**
(`tensor_dims_all_static`, `src/eval/abi_gate.rs:112`/`:124` — a symbolic
`tensor<f64[N]>` does not compile). A bulk column therefore **cannot** be one
`t.sum()`; it must be tiled, and combining the tile partials **is** a float
reduction someone must own. This RFC owns it here — not by inventing new float
machinery, but by **recursion on the shipped fold**:

- The column is materialised as fixed **`tensor<f64[4096]>` tiles** (4096 is
  not arbitrary — it *is* the cap, which makes the tile the natural ABI unit).
  Each tile reduces through the shipped pinned fold, unchanged.
- Tile partials are collected, in document order, into a partials tensor and
  reduced by the **same shipped pinned fold** (≤4096 partials per level): a
  **fixed radix-4096 left-to-right hierarchical fold**. Level 2 covers
  4096² ≈ 16.8 M elements; level 3 covers ≈ 6.9 × 10¹⁰ (~550 GB of f64) —
  the full bulk motivation, all through shipped machinery.
- The fold-tree shape is a **pure function of the static element count N**
  (the ragged last tile keeps its exact length — no zero-padding, so no
  `-0.0 + 0.0 = +0.0` edge case), never a function of data, thread count, or
  substrate. Since N is static, the whole tree is baked into the emitted IR
  and covered by `trace_hash`.
- **Honesty:** a radix-4096 hierarchical fold is a *different pinned
  association* than one flat left-to-right fold over N elements. The pinned
  value is therefore "the radix-4096 L→R hierarchical fold of N elements" —
  a documented, substrate-invariant reduction order, stated rather than
  hidden.
- **Gate:** a `format-col-sum-tiled` cross-substrate canary (a column larger
  than one tile, e.g. 3 × 4096 + a ragged tail) proving tiled == pinned
  reference and avx2 == neon.
- The in-tree deferred marker already names the alternative upgrade path
  (`src/mlir/lowering.rs:860-862`: a pinned sequential `scf.for` fold for
  over-cap/dynamic shapes) — that dynamic-N tier is a later ergonomic
  upgrade; the static radix-4096 tree is the Phase-0.5 shape because it is
  pure reuse of already-canaried machinery.

### 3. The bit-order crux — RESOLVED-CONSTRUCTIBLE (pin the index, not the mask)

**The former primary open question — does ARM NEON reproduce x86 `vpmovmskb`
structural-bitmap bit-order byte-identically? — is resolved as constructible.**
The answer is yes, with objdump-level arguments. What remains hardware-gated is
the real-silicon bless, not the design. (The GPU rung of the same crux is
tracked in the commercial runtime.)

**x86 reference semantics.** `vpmovmskb` packs byte lane *i*'s MSB into result
bit *i* — **LSB-first, lane 0 → bit 0**. Note the raw-mask width is
substrate-dependent before bit-order even enters: SSE 16 B → 16 bits, AVX2
32 B → 32 bits, a NEON chunk is 16 B.

**NEON — two idioms, different guarantees:**

- **Idiom B — shift-right-accumulate cascade (bit-order-IDENTICAL).** The
  sse2neon `_mm_movemask_epi8` construction: isolate each lane's MSB
  (`vshrq_n_u8(v, 7)`), then fold pairs/quads/octets with
  `vsraq_n_u16/_u32/_u64` at immediate shifts 7/14/28, and extract two bytes
  (`umov`). Every shift amount is a compile-time immediate; `vsra` is a
  per-lane integer add of a shifted copy, and the fold tree computes exactly
  `Σᵢ bitᵢ << i` — the defining equation of `vpmovmskb`. Integer add is exact
  and grouping-independent (the same license as the int-dot ladder), and each
  output bit depends on exactly one input lane — classify, don't reduce. An
  `objdump -d` of the aarch64 artifact shows `ushr/usra/usra/usra/umov` with
  immediate operands and zero `fadd/fmul/fcvt` — the same proof class as the
  Jul-4 fold's "4×`addsd`, 0 FMA". Cost: ~5–6 instructions per 16 B vs 1 on
  x86 — the "NEON tax", invisible at streaming bandwidth (§6) because stage 1
  runs at the I/O floor either way. Two 16 B halves + `mask_lo | (mask_hi <<
  16)` reproduce the AVX2 32-bit mask by the same argument.
- **Idiom A — `vshrn` nibble mask (FAST, not raw-mask-identical).** The
  simdjson-ARM idiom (`vshrn_n_u16(…, 4)`) yields a 64-bit nibble-expanded
  mask (4 bits per input byte) in 1–2 instructions. It does **not** reproduce
  the x86 bit layout — it is legal only under the index-pinned canary below.

**Design decision — pin the DERIVED STRUCTURAL INDEX, not the raw mask.**
The raw bitmap width differs per substrate (16 or 32 bits per chunk), so
pinning the raw packed mask would force every substrate onto Idiom B plus a
chunk-width normalisation — workable but brittle. The doctrine is "pin the
value, not the instruction", and the value here sits one level higher: the
**stage-2 derived structural index** — the monotonically ascending list of
byte offsets of structural characters. Idiom A's nibble mask and Idiom B's
bitmap both identify the **same position set**; a
`ctz`-style scan over either of them emits the **same ordered offset list**. That
hashed list is the cross-substrate canary line, and each substrate then uses
its cheapest legal pack — `vpmovmskb` on x86, Idiom A on ARM — exactly the
`vpmaddwd`/`SDOT` freedom the int-dot ladder proved.

- **Mandatory canary:** `format-structural-index` — hash of the derived
  offset list, asserted `avx2 == neon` (the neon line DEFERRED until a
  real-aarch64 bless, `mind-cross-substrate` custody — the `dot-f32-v`
  discipline, `tests/cross_substrate_identity/dot-f32-v-4093/reference_hashes.toml`).
- **Optional strict canary:** a normalised 64-bit raw bitmap via Idiom B —
  belt-and-suspenders, same defer discipline.
- **No scalar fallback is required for correctness.** The scalar structural
  oracle remains the *fitness oracle* for alg-invent (§8, Phase 2) and the
  fail-closed reference. If a future target lacks any cheap movemask idiom,
  it falls back to scalar `ctz`-per-byte with a deferred-marker
  (`// deferred: <arch> lacks movemask idiom, scalar bit-scan fallback —
  upgrade path: port Idiom B`).

Ownership unchanged: emitters `mind-det-gemm`; blessing `mind-cross-substrate`
(never self-blessed). The property is format-independent — it belongs to the
bit-pack primitive, so proving it once covers every stage-1 classifier.

### 4. Where it plugs into the compiler

Realistic layer map (what is genuinely new vs. reuse). Note the **per-format**
row is the only one that multiplies by format count:

| Piece | Layer | New or reuse |
|---|---|---|
| Stage-1 SIMD classifier (×format) | `src/mlir/lowering.rs` emitter, sibling to the int-dot AVX2 emitter / neon path | **NEW emitter per format**, reuses the ISA-ladder *pattern* + fail-loud canary discipline. The bit-pack primitive is shared. |
| Stage-2 structural indexing | new shared traversal over the bitmap | **NEW, shared** (format-independent, written once) |
| Exact number decode | reuse `std/json.mind` number logic (`jv_parse_number`, `:769`) lifted into a **shared** column-writing helper | **Reuse** (algorithm), new shared call-site |
| Stage-3 column → tensor param | `param_non_i64` memref C-ABI (`0840a24`, `src/eval/abi_gate.rs:112`; **concrete static dims only**, `:124`) | **Reuse, unchanged** |
| `t.sum()` / `t.mean()` fold (≤4096 elems) | `emit_tensor_reduce_pinned` (`:864`; cap `:909-911`) | **Reuse, unchanged** |
| Radix-4096 tiled fold across tiles (§2e) | recursion on the shipped fold; tree shape a pure function of static N | **NEW composition, zero new float semantics** (`format-col-sum-tiled` canary) |
| New std surface | `mfc_structural_scan`, `mfc_column_f64(ptr, len, fmt, field) -> tensor<f64[4096]>` tile intrinsics/externs | **NEW surface** (format selected by the `fmt` tag) |
| Evidence trace | existing `trace_hash = mini_sha256(emit_mic3(ir))` seam (`ir_trace_hash`, `src/ir/evidence.rs:72-74`) | **Reuse, unchanged** — the mic@3 hash covers the emitted IR (provenance, not output identity — §5) |

Genuinely new: (1) the stage-1 SIMD classify-and-pack emitter **per format**
(sharing one bit-pack primitive) and its ARM sibling; (2) the shared stage-2
index + stage-3 materialisation; (3) the `mfc_*` std surface; (4) the streaming
chunker + stable column-index assignment. Everything downstream of "a filled
`tensor<f64[4096]>` tile" is shipped machinery, untouched (plus the §2e
tiled-fold composition, which is recursion on that same machinery).

The **new intrinsic surface** mirrors the `__mind_blas_*` externs: an
`MFC_STRUCTURAL_SCAN_INTRINSIC` recognised by the lowering, with a Track-A
scalar oracle (the format's own scalar byte loop — `std/json.mind`'s for JSON,
a trivial split-on-newline for NDJSON/CSV) as the byte-identity reference the
SIMD path must match at every input — the exact fail-closed pattern the int-dot
intrinsics already use ("byte-identical to Track A's scalar reference").

### 5. CPU-SIMD path — and the GPU rung boundary

**CPU-SIMD (the primary target).** Stage 1 is an AVX2 (x86) / NEON (aarch64)
classify-and-pack emitter, oracle-checked against the scalar classifier at
every input. This is where Phases 1+ live. The NEON derived-index line stays
DEFERRED until a real-aarch64 bless (§3), exactly like `dot-f32-v`.

**GPU rung — behind the open-core contract, filled in the commercial runtime.**
Open-core ships only the **contract**: the `GPUBackend` trait
(`allocate`/`run_op`/`synchronize`, `src/runtime/gpu.rs:15`), whose own doc
states no concrete impl ships in this crate. The concrete kernel and the
device runtime are **commercial-runtime work, tracked privately** — see the
commercial runtime roadmap (`mind-runtime`). This RFC
does not cross that boundary.

**Evidence-identity vs output-identity (do not conflate).** The mic@3 anchor
`trace_hash = SHA-256(emit_mic3(ir))` (`ir_trace_hash`,
`src/ir/evidence.rs:72-74`) is computed over pre-backend IR, so it is
backend-agnostic **by construction**. That attests *provenance*, not numeric
output identity. Output identity is a separate per-substrate-pair proof: shipped
for CPU x86/ARM (the cross_substrate gate). Same evidence hash ≠ same output
bytes; the wedge claim needs both legs, and this RFC keeps them separate.

### 6. The I/O floor — honesty rail

At 100s of GB, **NVMe sequential read is the floor**: reading 272 GiB off NVMe
is ~60-100 s of pure disk time (a raw ~4 GB/s stream), and *nobody* — simdjson,
pandas, or this front-end — reads it faster. A simdjson-class stage-1
scan (~3-7 GB/s/core, a few cores) is faster than one NVMe stream, so the
pipeline **should be I/O-bound** — as it should be.

State it plainly: **a raw rec/s or GB/s race is a wash at this scale, entered
from behind.** We do not claim a throughput win over simdjson or pandas,
and **must not put a MIND-vs-X GB/s or rec/s number on any public surface**
(this RFC contains none). The **only** differentiated, defensible claim is:
*deterministic multi-format → `map`/`mic@x`, byte-identical x86 / ARM,
evidence-anchored* — the property simdjson (SIMD-lane-order-dependent) and
pandas **structurally cannot** offer. The
value is not "faster"; it is "the same bytes everywhere, provably, at streaming
speed you'd pay for anyway — for whatever format the data happens to arrive in."

### 7. Phase 0 — the runnable slice (today, zero new SIMD)

**Phase 0 — single numeric-array column, in-memory, no SIMD, simplest format.**
Input: numbers in the **simplest possible framing** — an NDJSON stream of bare
numbers (`1.0\n2.0\n...`) or a single-column CSV of numbers, whichever is
smaller to scan scalar. Reuse the `std/json.mind` number decode in a straight
loop (format-agnostic: NDJSON/CSV-of-numbers both reduce to "split on newline,
decode each") to fill a `tensor<f64[4096]>` — concrete static dims within the
pinned-fold cap (§2e); the literal end-to-end compile target is

```mind
fn colsum(t: tensor<f64[4096]>) -> f64 { t.sum() }
```

— then the shipped fold → `map` → `mic@x`. This proves the **whole pipeline
shape** (bytes → column → deterministic sum → evidence anchor, byte-identical
x86/ARM) with **zero new SIMD** and **pure reuse of the shipped Jul-4
machinery**. (A symbolic `tensor<f64[N]>` does not compile — §2e; "runnable
today" is claimed only for this concrete form.)

- **Runnable TODAY on `mindc build --emit cdylib`** — it is only shipped
  primitives (scalar number decode + `param_non_i64` tensor param +
  `emit_tensor_reduce_pinned`).
- **It is the fitness harness.** This Phase-0 slice is exactly the scored
  target that alg-invent (AB-MCTS on kernels) runs stage-1 SIMD candidates
  against later: a candidate kernel is *correct* iff its column bytes match the
  Phase-0 scalar reference, and *admissible* iff byte-identical across
  substrates. The harness exists before the kernels do.
- **Gate:** a new `format-col-sum` cross-substrate canary + keystone still 7/7.

### 8. Phased roadmap — the honest critical path

Ordering principle: **the fitness harness before the search; the oracle before
the SIMD.** Phase 0 is runnable today;
everything after it is DESIGN-ONLY until its own gate lands. Every phase gates
on keystone 7/7 + cross_substrate + criterion (one-sided); blessing is always
`mind-cross-substrate`, never self-blessed.

**Phase 0 — CPU fitness harness (runnable, pure reuse, zero new SIMD) (§7).**
Scalar number decode (lifted from `std/json.mind:769`) fills a
`tensor<f64[4096]>` tile; `fn colsum(t: tensor<f64[4096]>) -> f64 { t.sum() }`
through the shipped fold → `map` → mic@3 anchor, byte-identical x86/ARM.
*This is the alg-invent fitness harness — it exists before any kernel does.*
Gate: `format-col-sum` canary. Owner: `mind-dev` (+ `mind-mlir-lowering` if a
fold seam is needed).

**Phase 0.5 — tiled radix-4096 fold (§2e).** Bulk columns beyond one tile;
fold-tree shape a pure function of static N. Without this the bulk premise is
false — it precedes any Phase-1+ claim. Gate: `format-col-sum-tiled` canary,
avx2 == neon. Owner: `mind-mlir-lowering`. *DESIGN-ONLY.*

**Phase 1 — scalar structural oracle (all target formats).** A `ctz`-per-byte
scalar classifier per format emitting the derived structural index (§3) — the
fail-closed reference and the alg-invent Gate-1 oracle. Corpus: a
differential-fuzzer set (the `mindfuzz_cross_runner_identity` pattern) with
adversarial tails — chunk-edge structural chars, UTF-8 multibyte, quoted
structural chars. Gate: oracle parity vs the `std/json.mind` tree-walk on the
JSON subset; corpus committed. Owner: `mind-dev`. *DESIGN-ONLY.*

**Phase 2 — alg-invent (AB-MCTS) on the ONE hard kernel.** The
reduction-order-invariant structural pack (classified chunk → derived-index
representation). Search space: NEON `{vshrq, vsraq fold trees, vshrn nibble,
vqtbl pack, vceqq compare forms, tail-predication schemes}`; x86
`{vpcmpeqb + vpmovmskb variants}`. Fitness — lexicographic hard gates
× throughput:

```
if derived_index(cand) != oracle_index:      REJECT   # Gate 1: Phase-1 oracle
if hash(index_x86)     != hash(index_arm):   REJECT   # Gate 2: cross-substrate
if not objdump_pure(cand):                   REJECT   # Gate 3: no fadd/fmul/fcvt,
                                                      #   immediate shifts only,
                                                      #   no cross-lane reduce
score = measured GB/s                        # INTERNAL ONLY — never public (§6)
```

Because stage 1 is lane-local classify (zero cross-lane accumulation), any
candidate passing Gate 1 is deterministic by construction — the search cannot
invent a nondeterministic winner. Gate: winning kernel blessed as the x86
`format-structural-index` line (neon line DEFERRED to Phase 6). Owner:
`mind-det-gemm` (emitter) + the alg-invent loop. *DESIGN-ONLY.*

**Phase 3 — shared stage-2 index + stage-3 materialisation + multi-column.**
Format-independent, written once; stable document-order column indexing
(`gemm-i8-mt` MT == ST pattern). Gate: `format-multicol` canary + MT==ST.
Owner: `mind-dev` / `mind-mlir-lowering`. *DESIGN-ONLY.*

**Phase 4 — remaining stage-1 classifiers.** JSON `{}[],:"`; TOON indentation
+ `[N]`/`{fields}`; TOML (integer-only until `std/toml.mind:16` float lands).
Each = 1 classifier + ARM sibling + 1 canary; stages 2–3 + back-half reused
unchanged — the format-agnostic thesis proven. Gate: per-format canary +
oracle parity. Owner: `mind-det-gemm`. *DESIGN-ONLY.*

**Phase 5 — streaming chunker + cross-chunk straddle (shared).**
Bounded-memory chunking at record boundaries; carry-buffer straddle
reassembly. Gate: per-format boundary canary (split == unsplit column bytes) +
an I/O-bound full-file run (numbers internal only). Owner: `mind-dev`.
*DESIGN-ONLY.*

**Phase 6 — ARM bless.** The neon `format-structural-index` (and optional
Idiom-B raw-bitmap) lines blessed on real aarch64 (Ampere Altra — the
`dot-f32-v` unblock path). Owner: `mind-cross-substrate`. *DESIGN-ONLY.*

**Phase 7 — GPU rung (commercial runtime, behind the open-core contract).**
The GPU rung is filled behind the open-core `GPUBackend` contract
(`src/runtime/gpu.rs:15`); the concrete kernel and device
runtime are commercial-runtime work, tracked privately (see the commercial
runtime roadmap, `mind-runtime`). *DESIGN-ONLY.*

> deferred: Phases 0.5–7 are DESIGN ONLY in this draft. Only Phase 0 is
> claimed runnable-today (pure reuse of shipped machinery, concrete
> `tensor<f64[4096]>`). upgrade path: the phase gates above, each blocked on
> its own canary + keystone 7/7 + criterion.

### What it is NOT (scope fence)

- **NOT a GPU-parse-first engine.** CPU does stage 1. Any GPU rung is behind the
  open-core `GPUBackend` contract (`src/runtime/gpu.rs:15`) and must reproduce
  the CPU bytes, not reassociate floats; the concrete GPU work is
  commercial-runtime, tracked privately.
- **NOT a general fast-parser race for any format.** No strings/objects/nesting
  fast path beyond what columnar numeric extraction needs; `jv_parse` /
  `toml_parse` remain the general parsers. No public throughput claim (§6).
- **NOT touching the gates.** The byte-identity canary suite, oracle-parity
  linter, and differential fuzzer are *consumed* (new canaries added), never
  bypassed, weakened, or redefined. `trace_hash`/mic@3 seam unchanged. This RFC
  does not touch the *definition* of any gate.
- **NOT signed evidence.** The `map`/`mic@x` anchor is a **hash-anchored** mic@3
  trace, not a cryptographically signed one — Ed25519/ML-DSA signing is RFC 0016
  Phase C, pending. No "signed" language here.
- **AOT vs runnable honesty.** Phase 0 is runnable on the shipped
  `mindc build --emit cdylib` path today. Phases 1+ are AOT-emitter work that
  does not exist yet and is marked DESIGN-ONLY throughout.

## Drawbacks

- Two categories of path to maintain (scalar trees vs bulk front-end) plus one
  stage-1 emitter per format; drift risk between each scalar oracle and its SIMD
  path (mitigated by mandatory oracle-parity at every input).
- The differentiator (determinism) is invisible to a benchmark-only reader who
  only sees "not faster than simdjson." The pitch is
  governance/reproducibility, a narrower audience than raw-speed ingest.
- Real ARM blesses are hardware-gated; the neon stage-1 line stays DEFERRED
  until Ampere silicon time, exactly like `dot-f32-v`. (The GPU bless is
  commercial-runtime work, tracked privately.)
- The ARM raw-bitmap idiom (Idiom B, §3) costs ~5–6 immediate-shift
  instructions per 16 B vs 1 on x86 — a known constant-factor "NEON tax",
  invisible at the I/O floor (§6); the cheap Idiom A path avoids even that
  under the index-pinned canary. The residual risk is the real-silicon bless,
  not the construction.
- The tiled fold (§2e) pins a *documented* radix-4096 association, not the
  flat L→R order — a reviewer comparing against a naive serial sum will see
  different (but each substrate-invariant) bits; the spec states this rather
  than hiding it.

## Rationale and alternatives

- **Why one front-end for many formats:** TOON/CSV/NDJSON all decode to the JSON
  data model's column shape, so only stage-1 differs. N separate parsers would
  duplicate the index + materialisation + reduction wiring N times and risk N
  independent determinism bugs; one shared back-end with pluggable stage-1
  classifiers proves determinism *once*.
- **Why classify-not-reduce for stage 1:** it removes the cross-lane
  accumulation-order hazard entirely, rather than pinning a specific reduction
  tree across two ISAs. Strictly safer than the int-dot ladder (which does pin a
  reduction) because a bitmap pack has no accumulation.
- **Alternative — just make `jv_parse`/`toml_parse` faster:** rejected; a
  per-value heap tree cannot reach streaming throughput or bounded memory
  regardless of scan speed. The data structure is the ceiling.
- **Alternative — bind simdjson/pandas as an extern:** rejected; none can
  produce cross-substrate byte-identical output or feed the evidence seam, so it
  would forfeit the only differentiated claim.
- **Impact of not doing this:** MIND has a deterministic *reduction* back-end
  with no deterministic *bulk ingest* to feed it, for *any* format; the wedge
  story stops at "config files."

## Prior art

- **simdjson** — the two-stage (structural index → on-demand) pattern this
  reuses; its stage-1 pack is SIMD-lane-order-dependent, so no cross-substrate
  byte-identity.
- **TOON (v1.3.3)** — lossless JSON-model encoding (indentation + tabular arrays
  with `[N]`/`{fields}`); the reason a TOON scanner targets the same columns as
  JSON.
- **MIND int-dot GEMM ladder** — the proof that "pin the value, not the
  instruction" holds across `vpmaddwd` / `SDOT` / `SMMLA`; stage 1 applies the
  same discipline to classification.
- **MIND Jul-4 fold** (`4591c00` + `0840a24`) — the shipped deterministic
  reduction back-end this front-end feeds.

## Unresolved questions

- ~~PRIMARY: the ARM NEON bit-order crux~~ — **resolved-constructible**
  (§3): Idiom B reproduces `vpmovmskb` bit-order exactly; the pinned canary
  artifact is the derived structural index, letting each substrate use its
  cheapest legal pack. What remains open is only the **real-silicon bless**
  (real aarch64 for the neon line — `mind-cross-substrate` custody).
- Column dtype coverage beyond f64 (i64 columns are associative-add, already
  bit-identical).
- Whether TOML is worth a bulk stage-1 classifier at all, given `std/toml.mind`
  defers floats (`:16`) — bulk-numeric TOML is integer-only until that lands.
- Whether the chunker lives in `.mind` std or as a compiler-side driver (leaning
  std, to keep it in the governed language surface).

## Future possibilities

- A GPU stage-1 that reproduces the CPU bytes, behind the already-defined
  open-core `GPUBackend` contract (`src/runtime/gpu.rs:15`) — commercial-runtime
  work, tracked privately.
- Extend the columnar front-end to feed the int-dot GEMM path (numeric tables →
  `tensor` → deterministic matmul), not just reductions.
- A `mindc verify`-visible column-hash so a downstream party can attest a large
  multi-format ingest produced identical columns on their own substrate.
- Additional stage-1 classifiers (Parquet-lite, Arrow-IPC) — each is "one more
  stage-1 + canary," reusing the shared back-end.
