# RFC 0018: Bare-Metal Substrate Lowering Tier

| Field | Value |
|---|---|
| RFC | 0018 |
| Title | Bare-Metal Substrate Lowering Tier |
| Status | **Draft** |
| Authors | STARGA Inc. |
| Created | 2026-05-29 |
| Related | RFC 0014 (per-substrate lowering pipeline contracts — the framework this RFC extends), RFC 0015 (cross-substrate bit-identity — determinism obligations apply here too), RFC 0016 (evidence-chain emission — must carry the bare-metal substrate ID in `evidence_chain.substrate`) |

---

## 1. Summary

This RFC defines a **bare-metal lowering tier** for MIND: a no-OS execution target
where compiled MIND artifacts run directly on hardware without a POSIX kernel,
dynamic linker, or heap allocator. It specifies the scope of the tier, the
determinism obligations it inherits from RFC 0015, what is explicitly out of scope,
and the graduation lifecycle from RFC 0014.

The bare-metal tier is a **large commitment**. This RFC intentionally prioritises
explicit scope, non-goals, and open questions over a prescriptive implementation
plan. The design must be validated before any implementation begins.

## 2. Motivation

MIND targets regulated-industry deployments (deterministic AI in safety-critical
systems, edge inference, embedded control). A non-trivial fraction of these use cases
run on hardware with no OS — microcontrollers, FPGAs, specialised inference ASICs,
and safety-certified environments that prohibit a general-purpose kernel (DO-178C,
IEC 61508, ISO 26262, MISRA). The RFC 0014 substrate tier framework does not yet define a bare-metal tier — its
§3.5 Tier 0 list names only the accelerator substrates (Tpu, Npu, Lpu, Dpu, Fpga).
This RFC registers bare-metal as a first-class substrate tier and specifies it.

Without a defined bare-metal tier:
- MIND cannot be used in the regulated embedded markets where its determinism
  guarantee is *most* valuable.
- RFC 0016 evidence chains cannot carry a `bare_metal` substrate ID, so artifacts
  produced for such targets have no in-band provenance record.
- The `evidence_chain.substrate` vocabulary is incomplete.

## 3. Scope

### 3.1 What the bare-metal tier covers

The bare-metal tier defines a MIND lowering target that:

1. **Emits a self-contained binary.** The output artifact makes no runtime
   assumptions about a kernel, dynamic linker, or OS-managed heap. The artifact
   runs at a fixed entry point after hardware reset / bootloader handoff.
2. **Uses static memory only.** All tensor storage, stack frames, and intermediate
   buffers are resolved at compile time. No `malloc`/`free`, no garbage collector.
   The `region { }` memory model (RFC 0010 Phase J-A/J-B) is the MIND surface for
   expressing this: regions with compile-time-known extent map cleanly to bare-metal
   static allocation.
3. **Carries RFC 0015 determinism obligations.** A Q16.16 graph compiled for a
   bare-metal substrate must produce byte-identical output to the same graph on any
   other declared substrate (AVX2, NEON, the claimed set). The `trace_hash` in the
   evidence chain is the same value regardless of substrate — that is the wedge
   invariant, and the bare-metal tier does not relax it.
4. **Uses the RFC 0016 evidence-chain carrier.** `mindc` emits
   `evidence_chain.substrate = "bare_metal_<arch>_<variant>"` in the mic@3 MAP
   epilogue. The substrate ID vocabulary follows RFC 0014's naming convention
   (`arm_cortex_m4`, `riscv32_imac`, etc.).
5. **Integrates with `mindc build`.** The bare-metal tier is a first-class
   `--target` in the `mindc build` / `Mind.toml [targets.bare_metal]` surface
   (RFC 0008 §4.7), not a separate toolchain.

### 3.2 Target hardware classes

| Class | Examples | Priority |
|---|---|---|
| ARM Cortex-M (no MMU) | Cortex-M4, M33, M55 | Tier 1 candidate (widest regulated embedded use) |
| RISC-V (32/64, no OS) | RV32IMAC, RV64GC microcontrollers | Tier 1 candidate |
| Custom inference ASICs | Safety-critical NPUs, FPGAs | Tier 2 (vendor-specific; gated on per-vendor RFC) |

The initial scope targets ARM Cortex-M and RISC-V 32. Custom ASICs require
per-vendor RFCs and are explicitly deferred (§4).

## 4. Non-goals (explicit)

This RFC is a large commitment. The following are explicitly **out of scope**:

1. **Dynamic memory allocation.** No heap, no arena allocator, no bump allocator.
   Graphs that require dynamic shape inference at runtime are not supported on
   the bare-metal tier in v1. Static-shape-only restriction is intentional and
   safety-critical (no unbounded allocation at runtime).
2. **OS-managed I/O.** No file system, no network stack, no POSIX I/O. The
   `std.io`, `std.fs`, `std.net` stdlib modules are unavailable on this target.
   I/O is the application's responsibility via FFI to hardware registers.
3. **CUDA / GPU targets.** The bare-metal tier is CPU-only (cores with local
   SRAM). GPU substrates on bare-metal-class hardware (e.g. NPU co-processors)
   require separate per-vendor RFCs.
4. **RTOS integration.** FreeRTOS, Zephyr, and similar RTOSes are not bare-metal
   (they provide a kernel abstraction). They may be addressed in a separate
   "RTOS substrate" RFC; this RFC covers only the no-OS case.
5. **Floating-point where Q16.16 is required.** For regulated paths, IEEE 754 float
   is forbidden (same as the rest of MIND's deterministic-compute surface). This is
   not a relaxation — the bare-metal tier inherits the `#[nondeterministic]`
   annotation requirement from RFC 0012 Phase C for any float operation.
6. **Linker script generation.** The RFC defines the *compiler* contract. Linker
   script, startup code (`crt0.s`), interrupt vector table, and BSS/data section
   placement are application responsibilities. `mindc` emits ELF or raw binary;
   the toolchain provides the linker.
7. **SMT-proven bare-metal identity.** Formal verification of the bare-metal
   lowering pipeline is a future extension (see RFC 0017 §8).
8. **Self-hosting on bare metal.** The MIND self-host compiler (`libmindc_mind.so`)
   requires POSIX. The bare-metal tier is a *target*, not a *host*.

## 5. Guide-level explanation

A developer targeting ARM Cortex-M4 would write:

```toml
# Mind.toml
[targets.cortex_m4]
backend    = "bare_metal"
arch       = "arm_cortex_m4"
sources    = ["src/inference.mind"]
static-mem = "64kB"          # compile-time tensor buffer budget
```

And compile with:

```
mindc build --target cortex_m4 --emit-evidence out/inference.mic3
```

The output is a self-contained ELF (or flat binary via `--emit-obj`) with no
dynamic dependencies. `mindc verify out/inference.mic3` confirms the embedded
`trace_hash` and reports `substrate: arm_cortex_m4`.

The `static-mem` budget is a compile-time gate: `mindc` rejects any graph whose
total static tensor allocation exceeds the declared budget, preventing
stack-overflow surprises at runtime.

## 6. Reference-level explanation

### 6.1 Lowering pipeline changes

The bare-metal tier adds a new lowering stage after standard IR lowering
(RFC 0014 §3.3):

1. **Static allocation pass.** Replace all `tensor.alloc` ops with static buffer
   references. Fail compilation if the total exceeds the `static-mem` budget.
2. **No-libc pass.** Strip or replace all `std.io` / `std.fs` / `std.net`
   call sites (fail if present and not conditionally compiled out).
3. **Entry-point generation.** Emit a `__mind_bare_entry` symbol at a
   configurable address (default `0x00000000`; overridable via `Mind.toml`).
4. **Intrinsics selection.** Use hardware-native SIMD intrinsics for the target
   arch (NEON for Cortex-M55 with MVE, `zvfh` extension for RISC-V) only if
   the determinism gate passes; otherwise scalar fallback.

These stages are additive to the RFC 0014 lowering pipeline; the standard CPU
(AVX2/NEON OS-hosted) path is unaffected.

### 6.2 Determinism obligations

Bare-metal targets must satisfy the same RFC 0015 Q16.16 cross-substrate identity
gate as any other substrate:

- A Q16.16 graph compiled for `arm_cortex_m4` and for `x86_avx2` must produce
  identical `trace_hash` values in their evidence chains.
- The `tests/cross_substrate_identity/` manifest (RFC 0015 §5) adds a
  `bare_metal_arm_cortex_m4` column when the bare-metal tier ships.
- The mind-bench workload suite (RFC 0020) adds a `bare_metal` substrate column
  at the same time.

The scalar-fallback intrinsic path (no MVE, no NEON) is the identity-proving anchor:
it must produce results byte-identical to the AVX2 scalar path. Vector-accelerated
paths on bare-metal hardware are only admitted after a reference-hash matrix
confirms identity.

### 6.3 Evidence chain integration

`evidence_chain.substrate` uses the canonical RFC 0014 bare-metal tier IDs:

| Substrate ID | Hardware |
|---|---|
| `bare_metal_arm_cortex_m4` | ARM Cortex-M4, FPU optional |
| `bare_metal_arm_cortex_m33` | ARM Cortex-M33 (TrustZone) |
| `bare_metal_riscv32_imac` | RISC-V RV32IMAC |
| `bare_metal_riscv64_gc` | RISC-V RV64GC |

Additional IDs are registered via RFC amendment when new hardware classes graduate.

### 6.4 Stdlib availability on bare-metal

| Module | Bare-metal availability |
|---|---|
| `std.blas` | Available (Q16.16 path; subject to static-mem budget) |
| `std.vec` / `std.map` | Static-size variants only (no dynamic resize) |
| `std.string` | Read-only string table; no heap allocation |
| `std.sha256` | Available (used by evidence-chain trace_hash — §6.3) |
| `std.io` / `std.fs` / `std.net` | Unavailable (non-goal §4.2) |
| `std.async` | Unavailable (scheduler requires OS context) |
| `std.process` | Unavailable |
| `std.mlir` / `std.llvm` | Unavailable (host toolchain only) |

Attempting to import an unavailable module in a bare-metal target is a compile-time
error with a dedicated error code (E_BARE_001).

### 6.5 RFC 0014 graduation lifecycle

The bare-metal tier enters the RFC 0014 graduation lifecycle at Tier 0 (declared,
not yet lowering). The graduation criteria for Tier 1 (first-class lowering) are:

- Scalar fallback path passes the RFC 0015 cross-substrate identity gate on both
  ARM and RISC-V.
- `static-mem` budget gate is enforced at compile time.
- `mindc verify` passes on a bare-metal artifact.
- At least one end-to-end test runs on real hardware (JTAG/SWD-attached or QEMU
  emulation for CI).

Tier 2 (production-grade): adds vector-accelerated paths with full reference-hash
matrix coverage, and `mind-bench` bare-metal column.

## 7. Drawbacks

- **Scope is large.** A no-OS LLVM lowering target, a static allocation pass, a
  no-libc stdlib variant, and hardware-in-the-loop CI infrastructure is significant
  work. This RFC de-risks it by separating design from commitment: implementation
  does not begin until a follow-on session scopes the engineering plan.
- **Hardware heterogeneity.** Bare-metal targets diverge more than OS-hosted targets
  (memory maps, startup sequences, interrupt controllers). The RFC 0014 framework
  handles per-target variation, but the testing matrix is larger.
- **Determinism on constrained hardware.** FPU availability, SIMD extensions, and
  linker-section layout all affect bit-identity. The scalar-fallback anchor mitigates
  this but not all FPU interactions.

## 8. Rationale and alternatives

**Alternative: use an RTOS as the thin OS layer.** FreeRTOS + newlib is a common
approach for embedded MIND-like workloads. Rejected for the initial tier: RTOS
introduces scheduling nondeterminism and heap fragmentation that conflict with the
RFC 0015 determinism requirement. A separate RTOS RFC can layer on top of the
bare-metal tier.

**Alternative: target WebAssembly as the embedded transport.** WASM runtimes exist
for embedded (wasm3, wasmtime on Cortex-M). Rejected: WASM interpreters introduce
runtime overhead and nondeterminism at the interpreter boundary; the embedded tier
must emit native code to satisfy the latency and determinism bar.

**Alternative: defer entirely.** The bare-metal tier is not required for the
v0.7.x release. Deferral is the current state; this RFC establishes the design
baseline so that implementation can begin in a focused session without re-litigating
scope.

## 9. Prior art

- **Rust `#![no_std]`**: Rust's mechanism for bare-metal targets, splitting the
  standard library into `core` (always available) and `std` (OS-dependent). The
  bare-metal MIND stdlib split (§6.4) mirrors this discipline.
- **LLVM embedded targets** (`arm-none-eabi`, `riscv32-unknown-none-elf`): MIND
  can reuse LLVM's bare-metal target triples directly since MIND lowers through
  MLIR→LLVM.
- **TensorFlow Lite Micro**: a precedent for running neural network inference on
  Cortex-M with static allocation only. MIND's static-mem budget gate (§6.1) is
  a compile-time equivalent to TFLM's arena allocator approach.
- **MISRA C 2012 / DO-178C**: the regulatory context motivating no-heap, no-dynamic
  dispatch requirements.

## 10. Unresolved questions

1. **Entry-point protocol.** What does `__mind_bare_entry` receive on function call?
   A pointer to a static input buffer? A hardware register set? Needs a per-arch ABI
   spec (follow-on RFC per §8 of RFC 0014).
2. **Static-mem accounting.** Does the budget cover only tensor buffers, or also
   stack frames and intermediate lowering temporaries? The conservative answer
   (everything) needs a cost model.
3. **QEMU CI path.** Can the cross-substrate identity CI job (RFC 0015 §5) run
   Cortex-M4 binaries via `qemu-system-arm` (Cortex-M4 emulation)? If yes, the
   CI matrix is feasible without physical hardware for initial Tier 1 graduation.
4. **Float soft-ABI vs hardware FPU.** ARM Cortex-M4 with and without FPU may
   produce different float results even with `#[nondeterministic]` annotations.
   The scalar Q16.16 path is not affected, but the FPU interaction needs a policy.
5. **`std.sha256` on bare-metal.** The FIPS 180-4 implementation must run without
   `std.io` or heap. Confirm the pure-MIND `std.sha256` (shipped in v0.7.x) has
   no hidden OS dependencies.

## 11. Future possibilities

- **RTOS substrate tier** (FreeRTOS, Zephyr) layered on the bare-metal foundation.
- **Bare-metal GPU co-processor tier** (FPGA inference core, safety NPU) — requires
  per-vendor RFCs.
- **Formal verification of bare-metal lowering** via SMT-proven identity (RFC 0017
  §7, §8).
- **`mindc build --emit-hex`** for Intel HEX format, the standard embedded
  programming format.
- **512-mind DIFC integration**: governance modules that run on bare-metal
  microcontrollers at safety-critical nodes (industrial control, financial terminal).

## 12. References

RFC 0014 §3.5 (substrate tier definitions, bare-metal placeholder);
RFC 0015 §3.1 (cross-substrate bit-identity — Q16.16 gate that bare-metal must pass);
RFC 0016 §3 (`evidence_chain.substrate` — bare-metal tier adds IDs to the vocabulary);
RFC 0010 Phase J-A/J-B (region memory model — the MIND surface for static allocation);
RFC 0012 Phase C (`#[nondeterministic]` annotation — float ops on bare-metal still
require explicit opt-out);
LLVM bare-metal target triples (`arm-none-eabi`, `riscv32-unknown-none-elf`);
Rust `#![no_std]` / `core` crate (prior art for OS-free stdlib split).
