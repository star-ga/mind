# RFC 0014: Per-Substrate MLIR Lowering Pipeline Contracts

| Field | Value |
|---|---|
| RFC | 0014 |
| Title | Per-Substrate MLIR Lowering Pipeline Contracts |
| Status | **Draft** |
| Authors | STARGA Inc. |
| Created | 2026-05-25 |
| Supersedes | — |
| Superseded by | — |
| Related | RFC 0006 (mind-blas + Q16.16 cross-arch), RFC 0008 §4.7 (target enum at CLI), RFC 0010 (memory safety + C ABI), RFC 0012 (tensor-native syntax + determinism annotations), RFC 0015 (cross-substrate bit-identity proof obligation — pair RFC) |

## 1. Motivation

`BackendTarget` declares 8 substrates (`Cpu`, `Gpu`, `Tpu`, `Npu`, `Lpu`, `Dpu`,
`Fpga`, `Cerebras`) at `src/runtime/types.rs:27`, but the compiler has hard
divergence between what it **declares** and what it can **lower**:

- `src/pipeline.rs:189` rejects every non-`Cpu` target with `BackendUnavailable`
  before IR lowering starts.
- `src/build/mod.rs:461-475` `validate_target()` whitelists `Cpu` and
  `Cerebras` only — the other 6 are rejected as "not yet supported in Phase A"
  despite being CLI-accepted at `src/bin/mindc.rs:1212-1230`.
- `src/mlir/lowering.rs` is a 2409-LOC monolith with zero target-specific
  branching — every new target either edits the monolith or duplicates the
  file.
- Cerebras emits MLIR through a **separate** string-template path
  (`src/ops/cerebras.rs:209-237`), not via the main lowering pipeline. Two
  patterns, no unified contract.
- `BuildTarget` (`src/project/mod.rs:297`, 9 variants — adds `Wasm`) and
  `BackendTarget` (8 variants) drift independently; aliases live in three
  string-match sites (`mindc.rs:1212`, `project/mod.rs:328`, `project/mod.rs:778`).

The result: every new substrate currently requires hand-edits in 5+ files
across compiler, project, and runtime layers, with no graduation lifecycle
distinguishing "CLI-accepted placeholder" from "production-grade lowering."

This RFC defines **what a substrate target means at the compiler level**,
the graduation lifecycle from placeholder to first-class, and the contract
each substrate must satisfy.

This RFC is **deliberately a framework, not a per-substrate engineering
spec.** Substrate-specific blockers, dialect choices, vendor SDK
integrations, and detailed lowering strategies are scoped to **per-target
RFCs** that land as each substrate begins Tier 0 → Tier 1 graduation work
(see §3.5 for current per-target sketches and §11 for the graduation-RFC
naming convention).

## 2. Non-Goals

- **Backend plugin ABI** — out-of-tree substrate registration is RFC 0015's
  scope (deferred). This RFC assumes in-tree substrate definitions.
- **Cross-substrate bit-identity proof obligation** — RFC 0015 (pair RFC).
- **Runtime backend implementation** — `mind-runtime` is a separate crate.
  This RFC scopes only the compiler-side contract.
- **Fusion × target interaction** — Phase D fusion's per-target capabilities
  are RFC 0017's scope (deferred).

## 3. Substrate Tier Definitions

Each `BackendTarget` variant resides in one of four tiers. Graduation between
tiers requires meeting the contract for the destination tier.

### Tier 0 — Reserved
Declared in `BackendTarget` enum but the compiler explicitly rejects with a
`NotYetSupported` diagnostic citing this RFC. CLI alias may exist for
forward-compat parsing, but `mindc build --target=<reserved>` fails with a
clear "tracking in <issue/RFC>" message rather than an opaque
`BackendUnavailable`.

**Currently in Tier 0** *(verify before graduation)*: `Tpu`, `Npu`, `Lpu`,
`Dpu`, `Fpga`.

### Tier 1 — Experimental
Compiler emits MLIR for the target (potentially via a stub dialect), but:
- No Q16.16 bit-identity test against the CPU reference.
- May be gated behind a `--features experimental-<target>` cargo flag.
- No production warranty; output may not execute on real hardware.
- Required test: at least one `tests/target_<name>.rs` round-trip parse-emit
  smoke test.

### Tier 2 — Stable
- Q16.16 bit-identity test passes against the CPU reference for at least the
  baseline `blas_vec_q16_smoke` workload.
- Lowering goes through the main `src/mlir/lowering.rs` pipeline via a
  target-specific dispatcher, not a side-channel string template.
- CI runs the target's test matrix on every PR (cross-target isolation per
  RFC 0008 Phase F applies).
- Cross-substrate hash identity tested vs at least one other Stable target.

**Current Tier 2 candidates** *(verify before claiming)*: `Cpu`. `Cerebras`
**does not** currently meet Tier 2 — it uses a side-channel string template,
not the main lowering pipeline, and lacks a cross-substrate bit-identity
test. Cerebras is **Tier 1 today**.

### Tier 3 — Production
- Tier 2 conditions plus:
- Bit-identity matrix coverage per RFC 0015 (cross-substrate hash identity
  against ≥2 other Stable targets).
- Production runtime backend exists in `mind-runtime` and is tested in CI
  end-to-end.
- Documented in `docs/backends/<target>.md`.

**Currently in Tier 3**: none. `Cpu` does not yet have cross-substrate hash
identity coverage (the matrix has nothing to compare it against, since no
other target is Stable).

## 3.5 Substrate-Specific Considerations

Each non-CPU substrate has substrate-specific architecture, vendor-SDK, and
MLIR-ecosystem blockers that this framework RFC does not resolve. The
following sketches enumerate the key blocker per target so the graduation
process (§5) has a known starting point. **Full per-target specs land as
RFC 0014-A through 0014-F** at the time each target enters Tier 0 → Tier 1
graduation work (per §11 naming convention).

### 3.5.1 GPU

`BackendTarget::Gpu` is a single variant but the runtime
(`mind-runtime/src/backend/{cuda,rocm,metal,webgpu}/`) is **four
incompatible paths**. The compiler today treats them as one and defers
selection to runtime; that choice deserves explicit scoping.

Key blocker: per-vendor MLIR dialect selection. CUDA via LLVM NVPTX,
ROCm via AMDGPU, Metal via MetalIR / SPIR-V, WebGPU via SPIR-V is the
likely shape but no RFC binds it.

Per-target RFC required (RFC 0014-A) before GPU graduates Tier 0 → Tier 1.

### 3.5.2 TPU

`BackendTarget::Tpu` targets Google TPUs via StableHLO (mature dialect,
`openxla/stablehlo`, >90% coverage, FP8 + collective_broadcast as of
2025-26). The canonical access path is the **PJRT plugin ABI** (OpenXLA's
C/C++ plugin contract), not direct dialect emission.

Key blocker: this RFC does not scope a PJRT-equivalent plugin ABI. Without
one, MIND must either (a) ship CUDA/TPU/etc. integrations in-tree
forever, or (b) wait on RFC 0016 (backend plugin ABI) before TPU can be
real.

Per-target RFC required (RFC 0014-B). Likely blocked on RFC 0016 first.

### 3.5.3 NPU

`BackendTarget::Npu` aliases parse to `npu | ane | hexagon` per
`src/bin/mindc.rs:1217`. This collapses **three incompatible vendor
toolchains**:

- **Apple Neural Engine (ANE)**: no public MLIR dialect. The public path is
  CoreML/MIL, which silently rejects ops like `concat`. Practical access
  requires the private `_ANEClient` / `_ANECompiler` Obj-C frameworks
  (Orion paper, arXiv 2603.06728). **There is no Apple-published lowering
  path MIND can wire MLIR into.**
- **Qualcomm Hexagon**: QNN SDK; no public MLIR dialect.
- **Intel NPU**: OpenVINO IR; the OpenVINO MLIR translator is
  research-stage as of 2026.
- **AMD XDNA (Ryzen AI NPU)**: closest to a real public dialect via
  IRON / MLIR-AIE (`Xilinx/mlir-aie`).

The honest 2026 NPU story is: route through ONNX → vendor SDK, not MLIR.
The single `BackendTarget::Npu` variant cannot represent these
incompatible toolchains. Splitting into `Npu { vendor: NpuVendor }` (or
parallel `Ane`, `Hexagon`, `IntelNpu`, `Xdna` variants) is a likely
outcome.

Per-target RFC required (RFC 0014-C). This is the highest-fragmentation
substrate and may need to be **multiple** RFCs (one per vendor) rather
than one.

### 3.5.4 LPU

`BackendTarget::Lpu` targets Groq's Language Processing Unit. **GroqFlow
is MLIR-based internally** (per Groq's blog) but there is no published
dialect, IR spec, or plugin SDK. The public access path is **Groq's
hosted API only**.

Key blocker: a "compile from mindc to LPU" path is **not externally
implementable in 2026 without a vendor partnership**. This is a business
blocker, not a technical one.

Per-target RFC required (RFC 0014-D), but it must be paired with vendor
engagement work outside the RFC corpus.

### 3.5.5 Pim (formerly Dpu)

Per §6, `BackendTarget::Dpu` is renamed to `BackendTarget::Pim` with
vendor variant `Pim::Upmem`. UPMEM DRAM Processing Units have published
ML-compiler targets via **TVM (IMTP, arXiv 2412.19630; DCC, arXiv
2511.15503)** — but these are TVM-Relax-based, not MLIR-native.

Key blocker: TVM and MLIR are independent compiler stacks. Targeting
UPMEM requires either (a) emitting LLVM IR for the DPU RISC cores and
linking through UPMEM's runtime, or (b) writing a bespoke MLIR dialect
that mirrors IMTP's primitives. Neither is a small piece of work.

Per-target RFC required (RFC 0014-E). Lower priority than GPU/TPU
because UPMEM hardware deployment is narrow.

### 3.5.6 FPGA

`BackendTarget::Fpga` targets reconfigurable arrays. The MLIR ecosystem
for FPGA has **multiple research-grade options with no canonical winner**:

- **CIRCT** (`llvm/circt`): LLVM-incubator MLIR-to-HDL, emits
  Calyx/FIRRTL/Verilog. Closest to mainline.
- **Vitis HLS**: closed-source, reached from MLIR via `emitc` → C++.
- **MLIR-AIE** (Xilinx/mlir-aie): for AMD AI Engine arrays.
- **Intel oneAPI for FPGA**: SYCL/HLS, not MLIR-native.
- **Academic**: HIR (arXiv 2103.00194), POLSCA, Stencil-HMLS.

Key blocker: choice of HDL/HLS toolchain. CIRCT is the most ecosystem-
aligned but the FPGA vendor's downstream synthesis (Xilinx Vivado, Intel
Quartus) still owns the place-and-route stage.

Per-target RFC required (RFC 0014-F). This RFC must scope toolchain
selection before any FPGA work begins.

### 3.5.7 Cerebras

Already covered in §3 (Tier 1 candidate today) and the Open Questions
(§7.3). The dialect lifecycle decision (`mind.cerebras.*` vs generic
`mind.wafer.*` parameterized by vendor) is the substrate-specific
question that distinguishes Cerebras from the others — it is the only
target where MIND already ships its own dialect rather than borrowing
StableHLO / CIRCT / vendor SDK.

No separate per-target RFC needed; this RFC's §3 + §7.3 + RFC 0015's §5.4
(deferred wafer verification) cover the Cerebras specifics.

## 4. Compiler-Side Contract

A substrate at Tier 1+ must satisfy:

### 4.1 Lowering Dispatcher
The lowerer (`src/mlir/lowering.rs` or its replacement) MUST dispatch on
`target: BackendTarget` and MUST NOT contain hard `if target != Cpu` gates
outside the per-target dispatch table.

The current `pipeline.rs:189` CPU-gate is the explicit anti-pattern this RFC
replaces. Migration plan: extract dispatch to a `LoweringRegistry` keyed on
`BackendTarget`, with `Cpu` as the default registered backend. Other tiers
register their own lowering paths in the same registry.

### 4.2 Unified Target Enum
`BuildTarget` (`src/project/mod.rs:297`) and `BackendTarget`
(`src/runtime/types.rs:27`) MUST be unified into a single source of truth
before any new substrate graduates to Tier 1. Possible approaches:
- Make `BuildTarget` a strict alias of `BackendTarget` + `Wasm` (current
  divergence point).
- Move the alias-parsing logic from 3 sites (`mindc.rs:1212`,
  `project/mod.rs:328`, `project/mod.rs:778`) into a single `impl FromStr`.

### 4.3 Capability Descriptor
Each substrate MUST declare a `SubstrateCapability` struct enumerating:
- Numeric types supported (Q16.16, f32, f16, BitNet ternary, etc.)
- Reduction-order normative form (left-fold? Cerebras CSL stencil tile?)
- Maximum tensor rank
- Required runtime backend version

This descriptor is consulted by the lowerer to choose vocabulary and by RFC
0015's bit-identity oracle to know what to test.

### 4.4 Diagnostic Quality
`validate_target` (`src/build/mod.rs:461-475`) MUST distinguish three failure
modes:
1. `NotYetSupported(target, tier, tracking_issue)` — Tier 0 placeholder; not
   an error in the compiler, an expected gate.
2. `BackendUnavailable(target, reason)` — Tier 1+ but runtime missing on
   this build (e.g., no CUDA libraries). Recoverable by installing toolchain.
3. `BackendInternalError(target, details)` — Tier 1+, runtime present, but
   emit/lowering failed. Compiler bug.

The current single `BackendUnavailable` conflates all three and gives users
no actionable signal.

## 5. Graduation Process

Promoting a substrate one tier requires:

### Tier 0 → Tier 1
1. Add target-specific dispatch path in `LoweringRegistry` (no string-template
   side channel).
2. At least one `tests/target_<name>.rs` smoke test passes.
3. PR review by RFC 0014 authors or designated reviewers.

### Tier 1 → Tier 2
1. Q16.16 bit-identity test against CPU baseline (per RFC 0015 §3.2).
2. Mainline pipeline path (no side-channel template).
3. Per-target test entry in `Mind.toml` matrix (per RFC 0008 Phase F isolation).
4. Documentation in `docs/backends/<name>.md`.

### Tier 2 → Tier 3
1. RFC 0015 cross-substrate hash identity matrix coverage (≥2 other Tier 2+
   targets).
2. End-to-end `mind-runtime` backend test in CI.
3. Production warranty review.

## 6. Substrate Naming Correction: `Dpu` → `Pim`

The current `BackendTarget::Dpu` variant docstrings name "NVIDIA BlueField,
AMD Pensando, Intel IPU" as targets. These are **SmartNICs** — they exist for
network/storage offload, not ML compute. No published MLIR dialect targets
these chips for ML workloads, and they are not competitive with even
consumer GPUs for inference.

The MLIR/ML-compiler research community uses "DPU" for **UPMEM's DRAM
Processing Unit** — processing-in-memory RISC cores with published compiler
targets (IMTP, DCC, PIM-ML benchmark). This is the substrate that has actual
ML compute potential at memory-bandwidth scales.

**Action:** Rename `BackendTarget::Dpu` → `BackendTarget::Pim` with vendor
variants `Pim::Upmem` (and forward-compat for future PIM vendors). Drop
SmartNIC docstrings. The CLI alias `dpu` may remain for backward-compat
parsing but emits a deprecation diagnostic pointing at `pim` or `upmem`.

This is a Tier 0 cleanup, can land independent of the rest of this RFC.

## 7. Open Questions

1. **WebAssembly as a substrate**: `BuildTarget::Wasm` exists in
   `project/mod.rs:297` but `BackendTarget` doesn't include it. Is Wasm a
   target tier or a compilation profile? Decision needed before unification
   in §4.2.

2. **`BackendTarget::Gpu` granularity**: today `Gpu` is a single variant but
   the runtime backends (`mind-runtime/src/backend/{cuda,rocm,metal,webgpu}/`)
   are 4 different paths. Does the compiler enum need
   `Gpu { kind: GpuKind }`, or does the runtime resolve it?

3. **Cerebras-specific dialect lifecycle**: `mind.cerebras.stencil_tile` is
   currently a string template. If Cerebras graduates to Tier 2 via §4.1
   refactor, does the `mind.cerebras.*` dialect become canonical, or is it
   replaced by a generic `mind.wafer.*` dialect parameterized by vendor?

4. **Reduction-order normative form**: capability descriptors in §4.3 declare
   "reduction-order normative form" — but RFC 0012 §8.4 has a cross-substrate
   f32 carve-out without specifying *what the normative order is* on each
   substrate. Pair RFC 0015 must resolve this before Tier 2 graduation.

## 8. Phasing

This RFC has no fixed timeline. Recommended sequencing relative to other
active work:

- **Phase A — Cleanup (independent, can ship anytime)**
  - `Dpu` → `Pim` rename (§6)
  - `validate_target` 3-way diagnostic (§4.4)
  - Unified target enum (§4.2)

- **Phase B — Pipeline extraction (blocks Phase D self-host risk)**
  - `LoweringRegistry` extraction from `pipeline.rs:189` (§4.1)
  - Cerebras migration from side-channel to mainline pipeline
  - Cerebras graduates Tier 1 → Tier 2 candidate

- **Phase C — Tier graduation matrix (parallel to RFC 0015)**
  - GPU graduates Tier 0 → Tier 1
  - Q16.16 cross-substrate test infrastructure
  - First Tier 3 target

**Sequencing note vs `.max → Phase D → self-host #[`:** Phase B of this RFC
should land **before** self-host `#[`. Otherwise the bootstrap fixed-point
will freeze the current CPU-gated architecture into the self-hosted compiler,
creating permanent architectural debt that survives every future release.

## 9. Risks

- **Bootstrap fixed-point fragility**: if `LoweringRegistry` extraction
  changes any MLIR output bytes for the CPU path, the bootstrap fixed-point
  breaks. The refactor MUST preserve byte-identical CPU output (this is
  testable: existing fixed-point test must pass before/after).
- **`mind-runtime` coupling**: capability descriptors (§4.3) overlap with
  runtime concerns. Need to decide whether descriptors live in the compiler
  crate, the runtime crate, or a shared types crate.
- **Cerebras docstring vs reality**: `BackendTarget::Cerebras` docstring
  promises cross-substrate Q16.16 hash identity (`types.rs:67-72`). RFC 0015
  defines *how* to verify this; this RFC defines *when* a target qualifies
  to claim it. Documentation must align after both RFCs ship.

## 10. Per-Target RFC Naming Convention

Substrate-specific RFCs land as **RFC 0014-X** where X is a letter
denoting the substrate. This pattern (suffix-letter for sub-RFCs of a
framework RFC) keeps the substrate sub-RFCs grouped with their parent
without monopolizing top-level RFC numbers for substrates that may never
graduate from Tier 0.

| Substrate | RFC name | Status |
|---|---|---|
| GPU | RFC 0014-A: GPU sub-target structure | Not yet drafted |
| TPU | RFC 0014-B: TPU via StableHLO + PJRT | Not yet drafted (likely blocked on RFC 0016) |
| NPU | RFC 0014-C: NPU vendor fragmentation strategy | Not yet drafted (may split per vendor) |
| LPU | RFC 0014-D: LPU access (vendor-partnership-gated) | Not yet drafted |
| Pim | RFC 0014-E: UPMEM PIM integration | Not yet drafted |
| FPGA | RFC 0014-F: FPGA HDL/HLS toolchain selection | Not yet drafted |
| Cerebras | Folded into RFC 0014 + RFC 0015 | (no separate RFC needed) |

Each per-target RFC scopes:
- Specific MLIR dialect / lowering path
- Runtime backend coupling
- Vendor SDK or partnership requirements (where applicable)
- Tier 0 → Tier 1 graduation acceptance criteria

These RFCs are written **on demand** — when active implementation work
on a specific substrate begins. Drafting all six in advance would be
speculative; the framework here (§3 tiers, §4 contract, §5 graduation) is
sufficient until then.

## 11. References

- `src/pipeline.rs:189` — CPU-gate anti-pattern this RFC replaces
- `src/build/mod.rs:461-475` — current `validate_target` whitelist
- `src/mlir/lowering.rs` — 2409-LOC monolith requiring per-target extraction
- `src/ops/cerebras.rs:209-237` — side-channel string-template pattern
- `src/runtime/types.rs:27` — `BackendTarget` enum, source of truth
- `src/project/mod.rs:297` — `BuildTarget` divergent enum
- `src/bin/mindc.rs:1212-1230` — CLI alias parsing site
- `tests/target_cerebras.rs`, `tests/cerebras_stencil_tile.rs` — existing
  pattern for target-specific tests
- RFC 0006 §5.2 — Q16.16 cross-arch bit-identity gate (task #57, x86↔x86
  half closed at mind@19e4028)
- RFC 0008 §4.7 — CLI target enum (text claims support, code rejects)
- RFC 0012 §8.4 — cross-substrate f32 carve-out (exception without base
  clause until RFC 0015)
- RFC 0015 (pair) — cross-substrate bit-identity proof obligation
