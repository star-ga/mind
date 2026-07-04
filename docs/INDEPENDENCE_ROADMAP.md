# MIND Rust-Independence Roadmap

> **Living source of truth for the path to a 100% Rust-independent, SOTA MIND toolchain.**
> Update status markers as phases land. Grounded in the 2026-07-04 independence audit
> (native-ELF + MLIR + gap agents) and the native-ELF self-host feasibility probe.
> **No fake wins:** every claim in this file is gated on a runnable, byte-verifiable
> artifact — not a byte-comparison that was never executed.

## Two axes (do not conflate)
Independence is **two separate axes**, and a public claim must be honest about both:
1. **Rust-independence** — is `mindc` itself a pure-MIND program, not Rust?
2. **Toolchain-independence** — does codegen avoid an external LLVM/MLIR/clang?

The **native-ELF** path can win both. The **MLIR** path can only ever win axis 1
(it hard-requires LLVM 17/18 + clang).

## Baseline (2026-07-04)
- Pure-MIND compiler: `examples/mindc_mind/main.mind` = **24,508 LOC** (+ 38 `std/*.mind`
  ≈ 18,876 LOC it links). It emits a **byte-identical** native x86-64 ELF of the whole
  compiler (1,156,476 B, `testdata/native_elf_oracle/`).
- **Honest correction:** that emitted ELF is a **byte-oracle, not a running self-host.**
  `main.mind` has **no `fn main`** (pure library of 532 fns), so the emitted entry is a
  9-byte `exit(0)` stub and all compiler code in `.text` is unreachable. It has **never
  been executed** (`MANIFEST.txt: exit_code None`). Public wording MUST be *"emits a
  byte-identical native ELF of itself"* — never unqualified *"self-hosts"* / *"Rust-independent"*.
- Rust still load-bearing: **80,268 LOC across 119 `src/*.rs`** (parser 5,109; type_checker
  5,125; eval/lower 7,818; mlir/lowering 9,831; ir/compact; build orchestration).
- MLIR path: **12,753 LOC Rust** + external **LLVM 17/18 + clang** (`src/eval/mlir_build.rs`
  shells `mlir-opt → mlir-translate → clang`); only backend for floats/tensors/GPU.
- Eliminated so far: `src/native` (2,441 LOC, **~3%**), deleted 2026-07-01 (frozen as oracle).

---

## PHASE A — Close the scalar self-host loop  ·  **IN PROGRESS**  ·  no deadline; correctness-gated
**Unlocks:** *"MIND's compiler self-hosts — compiles itself to a native ELF and reproduces
itself byte-identically with zero Rust & zero LLVM in the loop"* (integer/control-flow subset).
*Gate is a runnable stage1==stage2, not a byte-comparison. Land it right, not fast.*

- [ ] **A1** Add a pure-MIND `fn main()` driver to `main.mind` (stdin→compile→stdout-ELF→exit); flips entry from exit-stub to real compiler *(blocker B1)*
- [ ] **A2** Stdin seed protocol `[user_lo][21 std/*.mind][main.mind]` — only fd read/write exist, no argv/open *(B2/B3)*
- [ ] **A3** Use the compiler's own **deterministic** self-computed trace-hash for the loop *(B4 — loop needs determinism, not Rust-parity)*
- [ ] **A4** ⚠️ **First-ever execution of the 1.1 MB image**; fix runtime bugs (1 GiB arena under copy-realloc, recursion depth, stdin EOF) — *the schedule risk*
- [ ] **A5** Prove `stage1-output == stage2` byte-identical (Rust+LLVM out of the stage1→stage2 step)
- [ ] **A6** Ship the frozen `stage1.elf` as a real bootstrap binary (repo ships none today)
- [ ] **A7** Replace the determinism-only keystone with a **real independence gate** (stage1==stage2 in CI)
- [ ] **A8** Fix stale docs (`README.md:35` still calls deleted `src/native` "normative")

## PHASE B — Full-surface front-end / middle-end self-host  ·  3–6 months
`main.mind` covers only the subset its own source uses. To compile **arbitrary** MIND programs:
- [ ] **B1** Full type-checker in pure MIND (Rust `type_checker/mod.rs`, 5,125 LOC) — floats, tensors, narrow ints, enums, shapes
- [ ] **B2** Full parser + AST→IR lowering (`parser` 5,109 + `eval/lower.rs` 7,818) for every construct
- [ ] **B3** Autodiff in pure MIND
- [ ] **B4** Optimizer / analysis passes in pure MIND

## PHASE C — Full-surface native backend (drop MLIR/LLVM)  ·  6–18 months → multi-year
Native-ELF covers only scalar i64/ptr/struct/control-flow. To drop the 12,753-LOC MLIR path + LLVM:
- [ ] **C1** Float codegen (f32/f64 strict-FP tier) in native-ELF — carries the determinism/ISA-selection work
- [ ] **C2** Narrow ints (i8/i16/i32/u*)   ·   **C3** division / shift / compare
- [ ] **C4** Tensor/linalg lowering — matmul, reductions, broadcast, indexing *(currently MLIR-only)*
- [ ] **C5** Vectorization (AVX2/NEON SIMD) for performance parity
- [ ] **C6** ⚠️ **Optimizing backend** — register allocation + instruction scheduling (today LLVM `-O3`). *The multi-year item*; without it native codegen is correct but slow
- [ ] **C7** GPU codegen (CUDA/ROCm/Metal) — *commercial mind-runtime territory, hardest*
- [ ] **C8** Linker + fuller syscall surface (open/close/mmap) for general/multi-object programs

## PHASE D — Remove the external toolchain
- [ ] **D1** Delete the 12,753-LOC MLIR Rust + the LLVM 17/18 + clang dependency (after C)
- [ ] **D2** Confirm zero external binaries (`as`/`ld`/`clang`) in any build path

## PHASE E — Delete the Rust compiler
- [ ] **E1** Archive the 80,268-LOC Rust `mindc` (as `src/native` was)
- [ ] **E2** Shipped toolchain = the self-hosted pure-MIND binary. **Now "100% Rust-independent" is true.**

---

## Claim gate (what each milestone lets us say — honestly)
| Milestone | Truthfully claimable | ETA |
|---|---|---|
| **Phase A** | "compiler self-hosts, Rust+LLVM out of the loop (integer subset)" | today, **if the image runs** |
| A → real bootstrap, MLIR external | "self-hosting compiler" (LLVM caveat) | 3–6 mo |
| Phase B+C | "Rust-free & LLVM-free for the full language" | 6–18 mo |
| Phase C6/C7 + E | **literal "100% Rust-independent, SOTA"** | multi-year (C6/GPU are the tail) |

Until Phase A closes with a **runnable** stage1==stage2, the only honest public claim is:
*"the pure-MIND compiler emits a byte-identical native ELF of itself."*
