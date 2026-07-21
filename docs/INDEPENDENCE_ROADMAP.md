# MIND Rust-Independence Roadmap

> **Living source of truth for the path to a 100% Rust-independent, SOTA MIND toolchain.**
> Update status markers as phases land. Grounded in the 2026-07-04 independence audit
> (native-ELF + MLIR + gap agents) and the native-ELF self-host feasibility probe.
> **No fake wins:** every claim in this file is gated on a runnable, byte-verifiable
> artifact — not a byte-comparison that was never executed.

## The bigger goal: SOTA, not just independent
Rust-independence is **one** pillar. The north star is a **SOTA compiler — fastest, most
innovative, AND deterministic**. Independence must never be bought by sacrificing speed:
- **Determinism + tamper-evident evidence chain = the innovative moat** no gcc/clang/rustc has.
  Bit-identical cross-substrate output with an artifact-embedded, tamper-evident evidence chain
  (opt-in Ed25519/ML-DSA/PQC signing, RFC 0016 Phase C) is the differentiation, not a side-constraint.
- **"Fastest" is a hard requirement, not an afterthought.** The native-ELF independence path
  is currently *unoptimized* (correct but slow); LLVM's `-O3` is what makes the MLIR path fast.
  So Phase C6 (an optimizing backend in MIND) is not optional — it's how independence and SOTA
  speed coexist.
- **Method — autoresearch + alg-inv evolution.** The SOTA-fast, still-deterministic codegen is
  a *search/invention* problem, and we own the machinery: `autoresearch` (overnight experiment
  loop) + `alg-inv` (AB-MCTS algorithm invention on MIND kernels). Point them at the
  performance-critical parts — deterministic reduction schedules (fastest bit-identical fold,
  task #58), GEMM/int-dot kernel tiling + instruction selection (beat asm, #47), register
  allocation / scheduling (C6), vectorization — **with fitness = speed GATED by keystone
  byte-identity + correctness**. Nothing "invented" ships unless it stays bit-identical. This is
  the genuinely novel edge: *a compiler whose optimizations are discovered by evolutionary search
  under a hard determinism constraint.* Applicable **per-part** (one kernel/schedule at a time)
  or, longer-term, to **whole** codegen strategy (pass ordering, lowering choices).
  INVARIANT: parallel trajectories must preserve deterministic resume + a single deterministic
  keep/discard decision; U1 net-verifies every invented deliverable (no fake wins).

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

## PHASE A — Close the scalar self-host loop  ·  ✅ **COMPLETE** (2026-07-04) — loop closed, gated, bootstrap shipped
**Unlocks:** *"MIND's compiler self-hosts — compiles itself to a native ELF and reproduces
itself byte-identically with zero Rust & zero LLVM in the loop"* (integer/control-flow subset).
*Gate is a runnable stage1==stage2, not a byte-comparison. Land it right, not fast.*

- [x] **A1** `nb_main()` driver added to `main.mind` (stdin→compile→stdout-ELF→exit); entry flipped from exit-stub to real compiler ✅ *(named `nb_main`, not `main`, to dodge mindc's `@main` cdylib synthesis)*
- [x] **A2** Stdin seed protocol `[8B user_lo][8B src_len][21 std/*.mind][main.mind]` via `nb_read_fully` ✅
- [x] **A3** Compiler's own deterministic self-computed trace-hash used for the loop ✅ *(PT_NOTE `60725af3…`, non-zero, stable across stages)*
- [x] **A4** First-ever execution of the 1.16 MB image — **ran clean, ZERO runtime bugs** ✅ *(1 GiB arena held, recursion held, short-read loop worked)*
- [x] **A5** `stage1 == stage2 == stage3` byte-identical (sha256 `5a45f7c5…`, 1,159,233 B) — **independently re-verified by U1**: `./stage1.elf < src > stage2` → `cmp` IDENTICAL; `strace` = only `execve`(self)/`read`/`write`/`exit`; `ldd` = not dynamic ✅✅
- [x] **A6** Frozen `stage1.elf` bootstrap shipped (`3b1a060`, testdata/selfhost_loop/) ✅
- [x] **A7** Fail-closed self-host loop gate (asserts stage1==stage2==stage3; not skippable) wired into CI (`self_host_loop_smoke.py`, `3b1a060`) ✅
- [x] **A8** README backend claims corrected — self-host landmark + src/native deleted (`c508923`) ✅
- [x] **A-rebless** NOT NEEDED — the separated-driver design (`selfhost_driver.mind`, fed only as source) left `main.mind` byte-identical to HEAD, so mic@1 + mic@3 gates stayed green with nothing to re-bless ✅
- [x] **A9 — mic@3 canonicality: CLOSED (2026-07-04), it HOLDS.** Investigation found the pure-MIND `emit_mic3` is **already byte-identical to Rust's** on the combined-pruned IR — proven by direct `cmp` + matching `ir_trace_hash` on all 6 fixtures + main.mind, and gated green by mic@3-flip. Compiler-independence of `trace_hash = SHA-256(canonical mic@3 bytes)` holds exactly. The `0x1d2` "divergence" was a **stale frozen oracle**: the bundled stdlib grew 2 top-level items since capture, so both Rust *and* pure-MIND now emit 5461 B (oracle had 5447 B). Refreshed the 6 oracle PT_NOTEs from the Rust `emit_mic3` reference (`510a988`; test-data only, emitter/codec untouched). No `MIC3_VERSION` bump (no layout moved).
- [x] **A9b — oracle notes test-time-derived (DONE 2026-07-04, 927f2eb).** The frozen native-ELF oracle re-stales on any `std/*.mind` top-level edit. Derive the 6 oracle PT_NOTEs at test time from the Rust `emit_mic3` reference instead of freezing them, so stdlib drift can't reopen A9. *(mind-compiler / CI lane.)*
- [x] **RI-E1 — seed the self-host loop from the checked-in pure-MIND ELF; Rust `.so` demoted to drift oracle (DONE 2026-07-13).** `self_host_loop_smoke.py` now runs the **standard checked-in-stage0 bootstrap model** (as gcc/rustc do): the PRIMARY, always-run path seeds `stage1` by **executing the frozen `testdata/selfhost_loop/stage1.elf`** on the seeded stdin (`stage1 = run_elf(FROZEN)`), then `stage2 = run_elf(stage1)`, `stage3 = run_elf(stage2)`, and asserts `stage1 == stage2 == stage3 == frozen` — **zero Rust, zero LLVM, zero clang, zero `.so` in the reproduction chain** (only `execve(self)`/`read`/`write`/`exit`). The Rust `.so` is **DEMOTED from seed to oracle**: an ORACLE step asserts the fresh `.so` output `== frozen` to catch `std/*.mind`/`main.mind` **source drift** (soft-skips when the `.so` is absent — the primary path never depends on it). Re-freeze is `--reseed`. **Net: strictly MORE coverage** — reproduction-independence (new, always) PLUS the retained source-drift detection. **Honest scope — what RI-E1 does NOT claim:** *(i)* the FIRST frozen `stage1.elf` was originally minted by the Rust `.so` (chicken-and-egg — residual trusting-trust, orthogonal and universal to every bootstrapped toolchain); *(ii)* a **harness-free standalone `mindc`** (its own file-IO/argv/CLI, no Python driver) is **NOT** delivered here — that is the separate **C8 + argv/CLI** track. The seed chain is now Rust-free; the `.so` is only an oracle; a from-scratch, LLVM-free, standalone `mindc` remains **pending C8**. *(harness/fixture + doc only; `main.mind`, the native encoders, mic@1/mic@3 gates untouched.)*

## PHASE B — Full-surface front-end / middle-end self-host  ·  3–6 months
`main.mind` covers only the subset its own source uses. To compile **arbitrary** MIND programs:
- [ ] **B1** Full type-checker in pure MIND (Rust `type_checker/mod.rs`, 5,125 LOC) — floats, tensors, narrow ints, enums, shapes
- [ ] **B2** Full parser + AST→IR lowering (`parser` 5,109 + `eval/lower.rs` 7,818) for every construct
- [ ] **B3** Autodiff in pure MIND
- [ ] **B4** Optimizer / analysis passes in pure MIND

## PHASE C — Full-surface native backend (drop MLIR/LLVM)  ·  6–18 months → multi-year
Native-ELF covers only scalar i64/ptr/struct/control-flow. To drop the 12,753-LOC MLIR path + LLVM:
- [~] **C1** Float codegen (f32/f64 strict-FP tier) in native-ELF — carries the determinism/ISA-selection work.
  **Partial (in progress):** scalar `f64` emission landed in pure MIND (zero MLIR/LLVM) — reg-form SSE2 encoders
  `addsd`/`subsd`/`mulsd`/`divsd` + `cvttsd2si` (`6179153`), mem-operand `[rbp+disp32]` stack-slot encoders
  (`c914529`), and a lexer `tk_float` literal token (`720570e`), gated by `self_host_native_fp_smoke.py`
  (CPU-as-oracle execution correctness — no float byte-oracle can exist since the deleted Rust native backend
  rejected `ConstF64`). **Still open:** the f32/strict-FP tier, ISA-selection, and a byte-identity oracle for the
  float path. *(Commit/CHANGELOG history labels this increment **"RI-B1"** — that is roadmap **C1** here, NOT
  roadmap B1 (pure-MIND type-checker below); the "RI-B" tag was a work-tracking name, not a roadmap phase.)*
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
