# RFC 0015: Cross-Substrate Bit-Identity Proof Obligation

| Field | Value |
|---|---|
| RFC | 0015 |
| Title | Cross-Substrate Bit-Identity Proof Obligation |
| Status | **Draft** |
| Authors | STARGA Inc. |
| Created | 2026-05-25 |
| Supersedes | — |
| Superseded by | — |
| Related | RFC 0006 §5.2 (mind-blas + Q16.16 cross-arch task #57), RFC 0012 §8.4 (cross-substrate f32 carve-out), RFC 0014 (per-substrate lowering tier system — pair RFC) |

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

**But:** no test in `/home/n/mind/tests/` cross-checks Q16.16 byte-identity
across two targets. The CPU↔CPU half of task #57 closed at `mind@19e4028`
(Linux gcc ↔ Windows clang), but the multi-substrate matrix has zero
coverage:

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
- **f32 / floating-point bit-identity** — RFC 0012 §8.4 carves f32 out as
  "within-substrate reproducible, not cross-substrate byte-identical." This
  RFC's scope is **Q16.16 fixed-point only** (and BitNet ternary, per RFC
  0001, via a stricter sub-contract — see §6).
- **Runtime-side determinism** — EigenAI/TBIK address runtime determinism;
  this RFC addresses **compile-target** bit-identity (the compiler emits
  the same MLIR that produces the same bytes on every substrate). The
  runtime then executes that MLIR deterministically per its own contract.

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
`tests/cross_substrate_identity/manifest.toml`.

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

### 5.1 Bit-Identity Test Harness
A new test crate `tests/cross_substrate_identity/` contains:
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

## 7. Relationship to RFC 0012 §8.4

RFC 0012 §8.4 carves f32 functions out of cross-substrate bit-identity:

> Functions annotated f32 are within-substrate reproducible, not
> cross-substrate byte-identical.

That carve-out is an **exception clause without a base clause**. This RFC
provides the base clause:

- **Q16.16 functions**: cross-substrate bit-identical (this RFC).
- **f32 functions**: within-substrate reproducible (RFC 0012 §8.4).
- **BitNet functions**: cross-substrate bit-identical (this RFC §6).

Any annotation conflict (a function annotated with both `#[q16]` and
implicit f32 ops) MUST be a compile error per RFC 0012's
implicit-determinism predicate.

## 8. Open Questions

1. **Mixed-precision functions**: a function may legitimately use both
   Q16.16 and f32 (e.g., Q16.16 model with f32 input normalization). What's
   the bit-identity contract for such a function? Tentative answer:
   identity holds only on the Q16.16 portion; f32 portions fall under RFC
   0012 §8.4. Boundary handling is the open question.

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
- RFC 0012 §8.4 — f32 carve-out (exception clause this RFC pairs with a
  base clause)
- RFC 0014 (pair) — per-substrate lowering tier system
- EigenAI bit-exact inference: arXiv 2602.00182 (single-substrate
  precedent)
- TBIK deterministic inference: arXiv 2511.17826 (single-substrate
  precedent)
- JAX issue #26795 — XLA explicit cross-device non-determinism (the
  contrast this RFC defines MIND against)
