# RFC 0015: Cross-Substrate Bit-Identity Proof Obligation

| Field | Value |
|---|---|
| RFC | 0015 |
| Title | Cross-Substrate Bit-Identity Proof Obligation |
| Status | **Accepted — enforced by CI** for the committed Q16.16, exact-integer, and scoped strict f32/f64 workloads on `avx2` ↔ `neon` (§5A). General floating-point coverage, CUDA/Cerebras, and the per-target framework remain outside this shipped claim. |
| Authors | STARGA Inc. |
| Created | 2026-05-25 |
| Supersedes | — |
| Superseded by | — |
| Related | RFC 0006 §5.2 (mind-blas + Q16.16 cross-arch task #57), RFC 0012 §8.4 (Q16.16 compatibility) and §10.5 (floating-point reduction-order question), RFC 0014 (per-substrate lowering tier system — pair RFC) |

> **Status note (2026-09-13).** The internal gate enforces computational-output
> bit-identity across **x86-`avx2` ↔ ARM-`neon`** for the committed workloads,
> including the strict f32 dot and matrix-vector kernels and scalar f64 chain
> in §5A.3. Their references record real ARM verification: 2026-07-14 for the
> two f32 fixtures and 2026-07-05 for scalar f64. The earlier blanket exclusion
> of floating point understated this evidence. It is replaced by the precise
> scope in §5A.4; arbitrary floating-point programs and additional chips still
> require their own coverage and execution evidence. §1 records the original
> motivation; §5 and §8–§9 retain the broader infrastructure roadmap. Neither
> native machine-code bytes nor target-specific MLIR must match across ISAs.

## 1. Motivation

`BackendTarget::Cerebras`'s docstring at `src/runtime/types.rs:67-72`
claims:

> Q16.16 fixed-point reductions emit the same byte-identical hash on
> x86, CUDA, and the Cerebras wafer.

This is the strongest single differentiator MIND has against IREE, OpenXLA,
Triton, and Mojo — none of those compilers make cross-substrate
bit-identity claims at the dialect level. JAX/XLA explicitly disclaim
cross-device determinism. EigenAI (arXiv 2602.00182) and TBIK
(arXiv 2511.17826) claim bit-identity but only single-substrate.

**At this RFC's creation (2026-05-25):** no test in `tests/` cross-checked Q16.16 byte-identity
across two targets. The CPU↔CPU half of task #57 closed at `mind@19e4028`
(Linux gcc ↔ Windows clang), but the multi-substrate matrix has zero
coverage. This historical matrix is superseded for the x86/ARM pair by §5A:

| Pair | Status |
|---|---|
| x86-Linux ↔ x86-Windows (CPU↔CPU) | ✅ Closed mind@19e4028 |
| x86 ↔ CUDA | ❌ No test |
| x86 ↔ Cerebras WSE | ❌ No test |
| CUDA ↔ Cerebras | ❌ No test |
| Any pair involving Tpu/Npu/Lpu/Pim/Fpga | ❌ Targets are Tier 0 (per RFC 0014) |

A regulatory auditor or sceptical reviewer reading the `BackendTarget::Cerebras`
docstring and finding no oracle would correctly conclude the claim is
unverified.

This RFC defines the **proof obligation** every Tier 2+ substrate (per RFC
0014) carries for bit-identity, the **oracle infrastructure** that produces
the proofs, and the **normative form** for reduction order that makes the
proofs possible.

## 2. Non-Goals

- **Substrate tier definitions** — RFC 0014 (pair RFC).
- **Fusion-induced byte drift** — Phase D fusion's bit-identity gate is a
  separate concern; this RFC scopes pre-fusion identity.
- **Universal floating-point coverage** — the three strict f32/f64 output
  fixtures in §5A are in scope. They do not establish identity for arbitrary
  reductions, transcendental functions, all floating-point inputs, or GPU
  execution. RFC 0012 §10.5's general reduction-order question remains open.
- **Runtime-side determinism** — EigenAI/TBIK address runtime determinism;
  this RFC requires the compiled workload to produce the same canonical
  output bytes on each covered target. Target-specific MLIR may differ
  (§3.1); runtime behavior must also meet the workload's declared contract.

## 3. The Proof Obligation

Every Tier 2+ substrate (per RFC 0014 §3) MUST satisfy:

### 3.1 Definition: Bit-Identical Output

For a fixed `.mind` source program S, fixed input dataset I, and fixed
substrate target T:
- `mindc build --target=T S` produces compiler output `C_T(S)`.
- Running `C_T(S)` on input I produces output `O(S, I, T)`.
- The 32-byte SHA-256 hash `H_T = sha256(serialize(O(S, I, T)))` is the
  **substrate output fingerprint**.

The cross-substrate bit-identity contract is:

> For all Tier 2+ substrates T1, T2 in scope:
> `H_T1 == H_T2` for the same (S, I).

This is checked at the **output hash** level, not at intermediate
representations. Intermediate MLIR may differ across substrates (it MUST, to
enable target-specific lowering); only the final tensor-output byte
serialization must match.

### 3.2 Baseline Workload

The baseline Q16.16 workload for bit-identity verification is `blas_vec_q16`
(already used in `tests/blas_vec_q16_smoke.rs` for CPU↔CPU verification, per
RFC 0006). Tier 2 graduation requires this workload's hash matches the CPU
baseline.

Tier 3 graduation requires hash matches across the broader Q16.16 test
matrix:
- `blas_vec_q16` (dot product, axpy)
- `blas_mat_q16` (gemv, gemm)
- `reduce_q16` (sum, mean, future max per `.max` work)
- `conv_q16_2d` (basic 2D convolution)
- BitNet ternary primitives (per RFC 0001)

The exact workload manifest is maintained at
per-workload `tests/cross_substrate_identity/<id>/manifest.toml` files.

### 3.3 What "in scope" means

The bit-identity contract MUST be explicitly scoped on a per-target basis.
A target's RFC 0014 capability descriptor (§4.3 of that RFC) declares which
workloads it covers. The cross-substrate matrix is computed over the
**intersection** of all participating targets' declared coverage.

Example: if CPU declares Q16.16 + f16, and Cerebras declares Q16.16 only,
the bit-identity matrix for {CPU, Cerebras} covers Q16.16 workloads only.

## 4. Reduction-Order Normative Form

Q16.16 integer arithmetic is associative — reduction order doesn't change
the result mathematically. But the **bit-serialized output** depends on
intermediate accumulator width, overflow handling, and (for matrix
products) tiling order.

This RFC defines the normative form every Tier 2+ substrate MUST follow:

### 4.1 Accumulator Width
Q16.16 reductions accumulate in **i64** intermediate registers, regardless
of substrate. The final result is saturating-cast back to Q16.16 (i32) at
the end of the reduction, not at intermediate steps.

### 4.2 Reduction Tree Shape
Q16.16 sum-reductions over an axis are performed as a **left-fold from
index 0 to index N-1**, not as a balanced binary tree. This is slower than
the typical GPU/wafer parallel reduction but is the only way to guarantee
byte-identical output across substrates with different parallel-reduce
implementations.

**Performance escape hatch:** a substrate MAY implement a parallel
reduction tree if and only if it proves equivalence — at compile time,
through the type system, or at test time, through the bit-identity matrix.
For Q16.16 the equivalence is trivial (integer associativity); the proof
obligation is on the substrate to demonstrate it.

### 4.3 Overflow Semantics
Q16.16 overflow MUST saturate (clamp to `i32::MIN..=i32::MAX` after
sub-fixed-point conversion), not wrap. This is the contract whether the
substrate's native integer ops wrap (most CPUs) or saturate (some DSPs).

### 4.4 Matrix Multiplication Order
gemm `C = A @ B` is computed as iterating the K-axis innermost, with the
M-axis outermost. Tiling for memory locality is allowed; cross-tile
accumulator state MUST follow the left-fold rule (§4.2).

## 5. Oracle Infrastructure

### 5.1 Original infrastructure proposal (partially superseded by §5A)

The original proposal called for a test crate `tests/cross_substrate_identity/`
with the following files. These are proposed names, not an inventory of
shipped paths; §5A.1 identifies their implemented counterparts:
- `manifest.toml` — workload list, per-target coverage declarations
- `oracle.rs` — runs the manifest across all locally-available Tier 1+
  substrates, collects output hashes, compares pairwise
- `expected.toml` — golden hashes (auto-regenerated by `mindc tools update-cross-identity` on the canonical CI runner)

### 5.2 CI Gate
Tier 2+ substrate graduation requires the matrix to pass in CI. The CI
runner that produces the canonical golden hashes is documented at
`docs/backends/canonical-runner.md` (file does not yet exist; creation is
part of this RFC's implementation).

### 5.3 Toolchain-Free CI Caveat
`tests/phase_g_keystone_bootstrap.rs:345-374` currently downgrades the
byte-identity oracle to a non-asserting stub when the MLIR toolchain is
absent. This RFC's matrix MUST run on an ELF-capable runner with full
MLIR toolchain present. CI MUST distinguish "matrix passed" from "matrix
skipped due to missing toolchain" and reject merge on the latter for any
PR touching a Tier 2+ lowering path.

### 5.4 Cerebras Substrate Access
The Cerebras wafer is not a commodity test environment. The matrix MUST
support a "deferred verification" mode for substrates without local
access: emit the MLIR, hash the canonicalized MLIR bytes, and treat that
hash as a proxy. This is **weaker** than output-hash identity (production
graduation requires the real thing) but allows iterative development
without a wafer in hand.

This proxy SHOULD be documented in the per-target test as
`#[ignore = "real-substrate verification deferred to canonical runner"]`
with a tracking comment, NOT silently degraded.

## 5A. Conformance Evidence

The proof obligation in §3 and the reduction-order normative form in §4 are
**enforced on every push** by an in-tree merge gate. This section pins the
exact target, CI job, and committed hashes so an auditor can re-derive the
claim from the repository alone.

### 5A.1 Test target

- **Harness:** `tests/cross_substrate_identity.rs` (the file §5.1 sketches as
  `oracle.rs`; the shipped name is `cross_substrate_identity.rs`).
- **Workload specs:** `tests/cross_substrate_identity/<id>/manifest.toml`
  (deterministic seed, length, kernel symbol, output encoding — the
  single-source-of-truth §3.2 references; the shipped file is `manifest.toml`,
  the §5.1 `expected.toml` role is filled by `reference_hashes.toml`).
- **Committed references:** `tests/cross_substrate_identity/<id>/reference_hashes.toml`,
  one identical hash per covered substrate (`avx2 = …`, `neon = …`). Per §3.1
  a workload in the declared identity scope MUST yield the **same** content hash on every
  substrate, so the two lines carrying one value *are* the cross-substrate
  bit-identity claim made inspectable.
- **Run (asserting):** `MIND_BENCH_REQUIRE=1 cargo test --no-default-features
  --features "mlir-build std-surface cross-module-imports"
  --test cross_substrate_identity` with `MIND_BENCH_BLESS` unset.

The native output fixtures build kernels with `mindc --emit-shared`, supply
the specified seeded or fixed inputs, and compare outputs with independent
oracles **within the run**. Each fixture defines its canonical byte encoding
before SHA-256 comparison with the committed reference (byte-identity
**across builds, machines, and time**). The separate `bimap-phf` construction
fixture pins a canonical mic@3 trace hash; it is not a floating-point output
test or a comparison of native binaries.

### 5A.2 CI job (the cross-ISA gate)

`.github/workflows/ci.yml` job **`cross_substrate_identity`** runs the harness
on a dual-arch matrix:

| Runner | `target_arch` | Substrate verified |
|---|---|---|
| `ubuntu-24.04` | `x86_64` | `avx2` |
| `ubuntu-24.04-arm` | `aarch64` | `neon` |

Each runner verifies **its own** substrate against the committed hash and marks
the other `deferred` (never silently `pass` — §5.3 / §5.4). Because both lines
in `reference_hashes.toml` carry the *same* hash, the avx2 runner and the neon
runner independently re-deriving that one value **is** the `H_avx2 == H_neon`
assertion of §3.1, proven on real ARM hardware rather than an unverified copy
of the x86 hash.

The job sets **`MIND_BENCH_REQUIRE=1`**, which turns a missing MLIR toolchain
(`mlir-opt` / `mlir-translate` / `clang`) into a **hard failure** instead of a
self-skip. This closes the §5.3 toolchain-free downgrade vulnerability for this
gate: it cannot pass vacuously — it either runs the kernel and matches the hash
or it fails the build.

CI also runs `tests/cross_substrate_receipts.rs` against the measured/deferred
receipts and manifest inventory. The subsequent bless-mode hash harvest is
an evidence collection step, **not** an asserting gate. Unsupported workload
coverage, including the hardware-specific VNNI case, must remain explicit;
an inventory count is not a claim that every host executes every fixture.

### 5A.3 Committed reference hashes (the load-bearing constants)

The original five Q16.16/int8 workloads below are a baseline subset, not the
complete inventory. The current tree contains 25 workload manifests and 26
tests in `cross_substrate_identity.rs`; per-workload manifests and receipts
define coverage.

| Workload id | Shape / dtype | Reference hash (`avx2` == `neon`) |
|---|---|---|
| `dot-l2-q16` | Q16.16 dot, len 65536 | `1d7f272b85e5f0fd7cf473086fb1da558a723134ff02ef30a4323eb757209823` |
| `dot-l1-q16` | Q16.16 L1, len 65536 | `ce7e2a80515e123f5d4fbb77d841f0d6c56fcbc690bba2e2ff81e45765843b34` |
| `gemv-q16-256x256` | Q16.16 gemv | `dfdf890874472ee369da524955995889c39bc6da770e4e2b1d0d69315e17611a` |
| `gemm-q16-64x64x64` | Q16.16 gemm | `92e2cb75d74d83a4a398d78d9ac560f195279c31814972c892f856f675faea0f` |
| `gemm-i8-64x64x64` | int8 `det.igemm`, i32 out | `917d353b18fd7f5ea4dab7dd02b786f5ccc4a2d954f695084ca0a88214d699c7` |

The three floating-point output fixtures also pin `avx2 == neon`:

| Workload id | Operation / canonical output encoding | Reference hash (`avx2` == `neon`) |
|---|---|---|
| `scalar-float-f64` | Fixed scalar `a + b - c * d / a`; result's IEEE-754 bits in 8 little-endian bytes | `7592a52a5e10a2f24469765f71ce1f9f8ebd9efb51904cf9a18f310d33b3c92d` |
| `dot-f32-v-4093` | Strict f32 dot, length 4093; result bits packed into an i64, 8 little-endian bytes | `a132f7b970b647cd158f591d764c19ec41a8cf27c398c87758f74efb5a8a22c0` |
| `matmul-f32-v-64x64` | Strict f32 64×64 matrix-vector product; 64 result bit patterns, 4 little-endian bytes each | `ec5adb991372fcfc16b964ba566f05fb44701fcf8bbde2a5453fed294e1d0175` |

The f32 kernels use separate multiply/add operations, eight accumulation
lanes and a fixed left-to-right lane fold. The length-4093 dot also exercises
the scalar tail; the 64-column matrix-vector fixture has no remainder. The within-run
checks compare `to_bits()` exactly; no tolerance substitutes for bit-identity.
The f64 fixture pins a fixed strict operation sequence, including a rounded
division. See each fixture's `reference_hashes.toml` for its recorded hardware
provenance. Those executions establish the listed fixture coverage, not a
proof over every floating-point input or machine.

A re-bless (`MIND_BENCH_BLESS=1`) is permitted **only** on an intentional
lowering change (RFC 0020 §13) and must be documented in the release notes.

### 5A.4 Scope of the shipped claim (no over-claim)

What §5A enforces today, stated precisely:

- **In scope (Accepted, gated):** the manifest-defined Q16.16 and exact-integer
  native output workloads, plus the three strict floating-point output fixtures in §5A.3,
  on **`avx2` ↔ `neon`**. For the float fixtures, identity depends on strict
  operations and the specified fold order; floating-point associativity is
  not assumed. Canonical output bytes must match exactly.
- **Separate evidence classes:** `bimap-phf` compares canonical compiler
  construction hashes. The VNNI-specific fixture requires the corresponding
  hardware capability and can be explicitly deferred; its presence in the
  inventory does not prove native execution on every runner. Neither case
  extends the three floating-point output fixtures' scope.
- **Out of scope (still Draft / forward-looking):** unlisted floating-point
  programs or inputs, arbitrary reductions, transcendental functions, and
  general NaN-payload/subnormal/rounding-mode coverage; CUDA, Cerebras, and
  all Tier 0 targets (§1 matrix, RFC 0014 §3); BitNet
  ternary as a *gated* workload (the §6 sub-contract is specified but not yet
  in the manifest); and the per-substrate lowering **framework** itself
  (RFC 0014 §4 — `LoweringRegistry`, unified target enum, 3-way diagnostic —
  not yet shipped). These remain open and gate the *broader* matrix, not this
  RFC's accepted core.

This maps directly onto RFC 0020 §10 (the internal-gate slice that produces the
hashes the public `mind-bench` manifest will publish) and the manifests' own
`RFC 0015 §3.1` citations.


## 6. BitNet Ternary Sub-Contract

RFC 0001 (BitNet native support) defines ternary primitives that operate
on `{-1, 0, +1}`. Bit-identity for BitNet is **categorically stronger**
than Q16.16:
- The primitives are exact integer operations.
- There is no accumulator-width ambiguity (ternary additions sum to ints).
- Reduction order matters only for overflow semantics, which BitNet bounds
  by design.

Therefore: any Tier 2+ substrate that declares BitNet support MUST achieve
bit-identical output across substrates **on the entire BitNet workload
class**, not just the baseline. The exception clauses (§3.3 per-target
scoping) apply if a substrate cannot run BitNet at all, but a substrate
that runs BitNet MUST do so bit-identically.

## 7. Relationship to RFC 0012

RFC 0012 §8.4 preserves the existing Q16.16 gate when tensor syntax lowers
to the same operations. Its §10.5 discusses the general floating-point
reduction-order problem and proposes annotation behavior; that proposal is
not a blanket exclusion of the strict kernels now covered here.

- **Manifest-covered Q16.16 / exact-integer workloads:** cross-substrate
  bit-identity is gated as specified in §5A.
- **The three strict f32/f64 output fixtures:** cross-substrate bit-identity
  is also gated (§5A.3); this supersedes this RFC's earlier blanket f32 carve-out.
- **Other floating-point functions:** require a defined strict computation,
  output encoding and real-target evidence before claiming identity.
- **BitNet functions:** the broader sub-contract remains specified in §6;
  it must not be represented as a shipped workload gate without evidence.

Any annotation conflict (a function annotated with both `#[q16]` and
implicit f32 ops) MUST be a compile error per RFC 0012's
implicit-determinism predicate.

## 8. Open Questions

1. **Mixed-precision functions**: a function may legitimately use both
   Q16.16 and f32 (e.g., Q16.16 model with f32 input normalization). What's
   the bit-identity contract for such a function? Tentative answer:
   coverage of the Q16.16 portion or a strict f32 fixture alone does not prove
   the composed function. Its conversions, reduction order and final output
   need their own cross-target gate. Boundary handling remains open.

2. **Floating-point intermediates in Q16.16 paths**: some substrates may
   convert Q16.16 to f32 internally for performance (e.g., GPU tensor
   cores), then convert back. Is this admissible under §4.2 "left-fold from
   index 0"? Tentative answer: no — the conversion changes the
   accumulator's bit-width and the equivalence proof obligation falls
   entirely on the substrate.

3. **Stochastic operations**: dropout, random sampling, MoE routing with
   noise. These are inherently non-deterministic and cannot have
   cross-substrate bit-identity. Need a `#[stochastic]` annotation that
   explicitly opts out of the proof obligation. RFC 0012 may be the right
   home for this annotation.

4. **Hardware-supported Q16.16**: some accelerators (DSPs, possibly some
   NPUs) have native Q16.16 ops with different overflow semantics. Are
   those substrates eligible for Tier 2 if they can't follow §4.3 saturation
   without runtime overhead? Tentative answer: yes, but they must declare a
   capability flag `q16_native_saturation: false` and the compiler must
   insert saturation guards.

5. **Cerebras-specific reduction-tree dialect**: `mind.cerebras.stencil_tile`
   may benefit from a substrate-native reduction order. Is the
   `equivalence proof` escape hatch (§4.2) sufficient to admit it, or does
   the wafer's parallel-reduce shape make per-substrate output divergence
   inevitable? Pair this with RFC 0014's Cerebras tier graduation work.

## 9. Phasing

- **Phase A — Oracle infrastructure (independent)**
  - `tests/cross_substrate_identity/` scaffolding
  - Manifest format and `mindc tools update-cross-identity` command
  - First test: x86-Linux ↔ x86-Windows Q16.16 (already passing per
    task #57, just formalize in the matrix)

- **Phase B — First cross-substrate pair (blocks Cerebras Tier 2)**
  - CPU↔Cerebras Q16.16 baseline (`blas_vec_q16`)
  - Document canonical CI runner
  - Promote Cerebras to Tier 2 per RFC 0014

- **Phase C — Reduction-order normative form enforcement**
  - Compiler lint: detect parallel reductions in Q16.16 code without an
    equivalence proof
  - RFC 0012 implicit-determinism predicate extended

- **Phase D — Tier 3 first**
  - First substrate graduates to Tier 3 (likely CPU when ≥2 other Tier 2
    targets exist)

## 10. Risks

- **Performance cost**: §4.2 left-fold normative form may cost meaningful
  perf on substrates that natively do balanced parallel reduce. The
  equivalence-proof escape hatch is the mitigation, but proving equivalence
  for Q16.16 across multiple substrates is itself non-trivial work.
- **Cerebras wafer test access**: §5.4 deferred verification mode is a
  weaker guarantee. There is a window during which Cerebras bit-identity
  is provisional. This window MUST close before Cerebras can be cited as
  Tier 3.
- **Toolchain-free CI gap**: §5.3 closes the existing oracle-downgrade
  vulnerability in `phase_g_keystone_bootstrap.rs:345-374`. Implementing
  this requires runner topology changes.
- **Coverage matrix combinatorial explosion**: with 8 substrates and N
  workloads, the cross-substrate matrix grows as 8C2 × N. The manifest
  format (§5.1) must support coverage subsetting per pair to keep CI time
  bounded.

## 11. References

- `src/runtime/types.rs:67-72` — Cerebras docstring asserting cross-substrate
  Q16.16 hash identity (the claim this RFC backs)
- `tests/blas_vec_q16_smoke.rs` — current CPU↔CPU Q16.16 verification
  (task #57 half-closed)
- `mind@19e4028` — task #57 closed for Linux gcc ↔ Windows clang
- `tests/phase_g_keystone_bootstrap.rs:345-374` — toolchain-free CI
  downgrade vulnerability this RFC closes
- RFC 0001 — BitNet ternary primitives (stricter sub-contract, §6)
- RFC 0006 §5.2 — mind-blas Q16.16 cross-arch baseline
- RFC 0012 §8.4 — Q16.16 compatibility; §10.5 — general floating-point
  reduction-order question (scoped strict-float evidence is in §5A here)
- RFC 0014 (pair) — per-substrate lowering tier system
- EigenAI bit-exact inference: arXiv 2602.00182 (single-substrate
  precedent)
- TBIK deterministic inference: arXiv 2511.17826 (single-substrate
  precedent)
- JAX issue #26795 — XLA explicit cross-device non-determinism (the
  contrast this RFC defines MIND against)
- Attribution-as-trace-scoring (Khan, Jun 2026) — external *evaluation-side*
  corroboration of this RFC's precondition: a trace-level attribution score is
  well-defined only against a deterministic substrate ("without bit-identical
  computation across runs, the attribution score is measuring partly noise").
  Independent confirmation that cross-substrate bit-identity is the precondition
  for trustworthy evaluation, not only for reproducibility. (STARGA-internal
  signal `SIG-20260603-011`.)
