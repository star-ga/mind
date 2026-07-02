# Benchmarking methodology — tiers and comparable metrics

> **Audience:** anyone reading, publishing, or comparing a MIND performance number.
>
> This document is the single rule-set for **how MIND numbers are measured and
> reported**. [`docs/benchmarks.md`](benchmarks.md) holds the baseline *values*;
> this file defines the **tiers** those values belong to and the discipline that
> keeps them honest. If a number is quoted without its tier, it is meaningless.

---

## 1. The tiers

A performance number is only interpretable inside a **tier** — a fixed definition
of *what the clock started and stopped on*. MIND recognises four tiers. They are
not interchangeable, and the cross-tier comparison rule (§2) is absolute.

| Tier | Clock spans | What it measures | Where it lives |
|------|-------------|------------------|----------------|
| **T0 — Kernel-only** | one in-process call into an already-loaded compiled kernel; no process spawn, no dlopen, no I/O | steady-state execution throughput of the emitted machine code | `benches/det_matmul_*.rs` (criterion `iter` loop) |
| **T1 — Frontend-only** | source string → parse → typecheck → IR (`compile_source`), in-process | compiler **frontend** latency only — **no codegen, no MLIR, no LLVM, no link** | `benches/compiler.rs::compiler_pipeline` |
| **T2 — Process wall-clock** | `mindc src --emit-shared out.so` as a spawned process: frontend + MLIR-text + `mlir-opt` + LLVM + `clang` link, including process startup | end-to-end compile latency a user actually feels | `benches/compiler.rs::e2e_compile_shared` (T2 sweep) |
| **T3 — Persistent cold / steady** | first request after a cold start vs. steady-state after warm-up, for a long-lived process (the mind-ray discipline) | service latency separated into cold-start and warm paths | service harnesses (out of tree) |

### The mind-ray discipline (T3)

A long-lived process has **two** latencies, never one:

- **cold** — the first request after start, paying one-time costs (page-in,
  dlopen, JIT warm-up, cache population);
- **steady** — every request after warm-up, paying none of them.

Reporting a single number for a persistent service is a category error. T3
**always** reports the pair `(cold, steady)`. A steady number presented as "the"
latency hides the cold cost; a cold number presented as "the" latency slanders
the steady path. Both, labelled, or neither.

---

## 2. The rule: never compare numbers across tiers

**A number from one tier may never be compared, divided, or ratio'd against a
number from another tier.** This is the single most common way performance claims
become lies, and it is forbidden here.

- A **T1 frontend** microsecond is not comparable to another toolchain's **T2
  whole-pipeline** millisecond. They time different work.
- A **T0 kernel** GMAC/s is not comparable to a **T2 compile** latency. One is
  execution, the other is compilation.
- A **T3 steady** number is not comparable to a **T3 cold** number — even within
  the same tier the cold/steady split must be preserved.

When a ratio against an external framework is published, the comparison must be
**tier-matched** (both T2, both T0, …) or it must carry an explicit, prominent
scope note stating the tiers differ and that the ratio reflects that difference,
not a like-for-like speedup.

### Load-bearing correction: the published 1.8–15.5 µs numbers are FRONTEND-ONLY

The headline **1.8–15.5 µs** compilation figures quoted on the README and in
[`docs/benchmarks.md`](benchmarks.md) are **Tier T1 — frontend-only**
(`parse + typecheck + IR`, in-process, via `compile_source`). They are **not**
end-to-end codegen and **not** a process wall-clock:

- they do **not** include MLIR text emission,
- they do **not** include `mlir-opt` / LLVM / `llc`,
- they do **not** include the `clang` link step,
- they do **not** include process startup.

Any external comparison against a tool's full `build` time (PyTorch
`torch.compile`, JAX `jax.jit` cold-start, `mojo build`, …) is **T1 vs T2** and
must carry the scope note already present on those tables. The end-to-end MIND
number — the tier-matched figure — comes from the **T2** `e2e_compile_shared`
sweep (`benches/compiler.rs`), which spawns `mindc --emit-shared` and times the
whole frontend + MLIR + LLVM + link pipeline. Quote **that** against another
toolchain's build time; quote the 1.8–15.5 µs T1 number only against another
tool's *frontend/parse* stage, or with the scope difference spelled out.

> The 1.8–15.5 µs frontend latency is a **protected compile-speed budget** that
> defends the moat (cross-substrate bit-identity + embedded evidence chain), not
> the moat itself. See [`docs/roadmap.md`](roadmap.md) §"Compile-speed invariant".

---

## 3. Comparable execution metrics — GMAC/s and % of ISA peak

Execution-throughput benches (T0, the deterministic GEMM/GEMV kernels) report a
**self-contained roofline axis** so results are comparable across shapes,
machines, and over time **without** pulling in an external BLAS:

- **GMAC/s** — billions of multiply-accumulates per second, computed directly
  from the workload size and the measured wall time:

  ```text
  MACs   = M · N · K   (GEMM)   or   rows · cols   (GEMV)
  GMAC/s = MACs / elapsed_seconds / 1e9
  ```

  This is derived from first principles (the arithmetic the kernel provably
  performs), not from any reference library, so the bench stays dependency-free.

- **% of ISA peak (roofline)** — GMAC/s as a fraction of a *documented,
  conservative* per-core integer-MAC ceiling for the host ISA. The ceiling is a
  fixed constant in the bench (e.g. the AVX2 `vpmaddwd` / widen-multiply-add
  issue rate at the reference clock), labelled as an estimate. It answers "how
  close to the metal is this kernel?" without claiming a vendor-blessed peak.

Both axes are **derived**, so adding them changed **no** kernel bytes and pulled
in **no** new dependency. The roofline percentage is explicitly marked as an
*estimate against a documented constant* — it is a comparability aid, not a
certified hardware-peak claim.

### Why GMAC/s, not "Melem/s"

Criterion's built-in `Throughput::Elements` reporting (still emitted) labels the
rate "elem/s", which reads as elements *moved*, not MACs *performed*. For a
reduction kernel the meaningful unit is multiply-accumulates. GMAC/s is the
unit a roofline analysis uses, so the kernels print it explicitly alongside
criterion's native number.

### What is deliberately NOT here

- **No OpenBLAS / cuBLAS / any BLAS dependency.** A float BLAS cannot reproduce
  MIND's load-bearing property (byte-identical output across substrates — today
  gated x86 == ARM on the integer/Q16.16 tier — f32 add
  is non-associative), so a head-to-head "MIND vs BLAS GFLOP/s" race would compare
  a deterministic integer kernel against a non-deterministic float one: not
  tier-matched, and not the claim. The roofline % gives an absolute "fraction of
  the machine" read instead, with zero external code.

---

## 4. Reporting checklist

Before a MIND performance number leaves the repo:

- [ ] **Tier is stated** (T0 / T1 / T2 / T3). No bare numbers.
- [ ] **Cross-tier comparisons carry a scope note** or are tier-matched (§2).
- [ ] **Frontend (T1) numbers are labelled frontend-only** — never presented as
      end-to-end compile time.
- [ ] **Persistent-service (T3) numbers report `(cold, steady)`**, both labelled.
- [ ] **Execution (T0) numbers report GMAC/s + roofline %**, with the ISA-peak
      constant cited as an estimate.
- [ ] **The environment is recorded** (CPU, clock, ISA, OS, rustc, sample count).
