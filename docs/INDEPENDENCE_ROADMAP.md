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
- [~] **B1** Full type-checker in pure MIND (Rust `type_checker/mod.rs`, 5,125 LOC) — floats, tensors, narrow ints, enums, shapes.
  **First slice LANDED (2026-07-21, `1372c4c`):** the E2004 i64→i32 implicit-narrowing rule ported to pure MIND
  (`i32` arm in `resolve_type_ident` + width fn), byte-for-byte matching the Rust oracle over all {i32,i64,f64,bool}
  pairs, gated by `self_host_tc_narrowing_smoke.py` (isolated `selftest_tc_*` export). **Extended (2026-07-21):**
  the scalar-class rules E2010/E2011/E2013/E2016 (`self_host_tc_class_rules_smoke.py`), the shape rules E2005
  (call-arity) / E2101 (broadcast) / E2102 (matmul-rank) / E2103 (matmul-inner-dim), and E2023 (reserved `__mind_`
  prefix — `name.starts_with("__mind_")`) — each an additive `selftest_tc_*` export byte-for-byte matching its Rust
  oracle over positive+negative cases, all gated in `fast_keystone.sh` (`tc_class_rules`, `tc_shape_rules`). The
  remaining ~5,000 LOC (float/tensor/enum + AST-context-dependent rules) is the bulk still open.
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
  rejected `ConstF64`). **f32 scalar tier LANDED (2026-07-21, `1372c4c`):** general single-precision emission —
  `addss`/`subss`/`mulss`/`divss`, `cvtss2sd`/`cvtsd2ss` round-trip, `cvttss2si`, `movss` load/store — via the
  general nb_expr path (not just kernel emitters), gated by `self_host_native_scalar_f32_smoke.py` (CPU-as-oracle,
  exact-bit vs single-precision reference). The f64 canary stays byte-identical. **Still open:** ISA-selection,
  a byte-identity oracle for the float path, and wiring f32 into the general dtype registry (currently a
  dedicated selftest export, not the default nb_expr dtype). *(Commit/CHANGELOG history labels the earlier f64
  increment **"RI-B1"** — that is roadmap **C1** here, NOT roadmap B1 (pure-MIND type-checker below).)*
- [~] **C2** Narrow ints (i8/i16/i32/u*) — **store/load LANDED (2026-07-21, `1372c4c`):** user-reachable
  `__mind_{store,load}_{i8,i16,i32}` (truncating stores, zero-extend loads) via the general nb_emit_intrinsic
  path, gated by `self_host_native_narrowint_smoke.py` (non-fakeable neighbour/high-bit probes). **Still open:**
  narrow-width wrap-around ARITHMETIC (needs a narrow-int type + per-op width masking) and unsigned surface types.
  · **C3** division / shift / compare — **COMPLETE:** `idiv`+`cqo` with zero & INT_MIN/-1 guards, `sar`/`shl`,
  all 6 signed `setcc`; 16 edge-case tests added (`div_shift_cmp_edge_smoke.py`). Logical-`shr`/unsigned-`setcc`
  unreachable until C2 unsigned types (correctly deferred, no oracle).
- [~] **C4** Tensor/linalg lowering — matmul, reductions, broadcast, indexing. **Native-ELF i64 ops LANDED
  (2026-07-21, zero MLIR/LLVM):** elementwise-add (`selftest_native_elf_tensor_ewadd_i64`, C4-T1), dot/MAC
  reduction (`_dot_i64`, C4-T2), and 2-D matmul (`_matmul_i64`, C4-T3, `f468e02`/`24699d0`) — each emits a runnable
  native x86-64 ELF with 2-D row-major addressing + a fail-closed frame-bound guard (dims bounded before products,
  so no i64-overflow shape overruns), verified against an independent Python reference AND the native-ELF byte-oracle.
  **Still open:** broadcast/transpose/general N-D indexing, f-typed tensors (needs C1 float-in-registry), and the
  optimizing backend (C6, MLIR still owns performance today).
- [ ] **C5** Vectorization (AVX2/NEON SIMD) for performance parity
- [ ] **C6** ⚠️ **Optimizing backend** — register allocation + instruction scheduling (today LLVM `-O3`). *The multi-year item*; without it native codegen is correct but slow
- [ ] **C7** GPU codegen (CUDA/ROCm/Metal) — *commercial mind-runtime territory, hardest*
- [ ] **C8** Linker + fuller syscall surface (open/close/mmap) for general/multi-object programs

### Open follow-ups (2026-07-21, from the param/return-wrap `6e4c809` audit)
- [ ] **Param MUTATION via assign reads the original home slot** — `nb_expr`'s ident arm (~main.mind:22050) resolves params via `resolve_param` (home slot) BEFORE consulting the let-env, so `fn f(x: i32) { x = x + 1; ... }` silently drops the mutation (verified byte-identical fail-behavior on both `6e4c809` and its parent — pre-existing, not introduced by the param/return-wrap). Fable audit flagged that `6e4c809`'s commit message calls this "source-marked" but NO `// deferred:` marker exists in main.mind yet — ADD one at nb_expr's ident arm when the next main.mind change reseeds (honesty: the disclosure currently lives only in the commit message + here). Fix path: consult the let-env before `resolve_param` for reassigned params, or spill mutated params into let-env slots.
- [ ] Minor: `frt_scan_retw` duplicates `frt_scan_ret`'s O(tokens) forward scan (compile-time cost only, zero emission effect) + inherits the pre-existing unguarded `pos+2`-at-EOF read shape shared with the dtype scan — fold into a single scan when convenient.

### Open follow-ups (2026-07-21, from the width-driver `ae3bfdf` audit)
- [ ] **Latent narrow-only count/emit asymmetry** (Fable-flagged, currently UNREACHABLE — fails-closed today, zero i64 impact): `nb_count_bind_merged` (main.mind ~23071) hardcodes width **64** for if-merged rebinds, while emit-side `nb_rebind_merges` inherits the true declared width. A narrow var assigned in an `if` branch then reassigned later would count +0 but emit +1 → frame undercount by one slot. The if-statement-with-assign-branches shape fails closed in `selftest_native_elf` on both this commit and its parent, so it's not emittable yet. FIX before that shape becomes emittable: inherit width via a clets lookup in `nb_count_bind_merged`. (While-carry is already safe.)
- [ ] **Narrow-int params + return types don't auto-wrap yet** — the `ae3bfdf` width-driver covers `let` + assign bindings only; params/returns default to width 64. Add explicit `// deferred:` markers at `nb_lets_lookup_width` and the param/return sites when extending.
- [ ] **Top-level straight-line assign fails-closed even for i64** — `let w: i64 = 100; w = w + 100;` (no loop) returns empty EmitState in the no-feed `selftest_native_elf` path; pre-existing, not width-driver-specific. See [[reference_native_elf_toplevel_assign_gap_2026_07_21]]. Fix in the nb assign-statement lowering.

### Open follow-ups (2026-07-21, from the `1372c4c` audit)
- [ ] Wire the new native-ELF smokes (`self_host_native_narrowint_smoke.py`, `self_host_native_scalar_f32_smoke.py`,
  `self_host_tc_narrowing_smoke.py`, `div_shift_cmp_edge_smoke.py`) into `fast_keystone.sh` / CI — today they only
  guard when run manually (consistent with sibling rung smokes, but they should be enforced).
- [ ] Port `self_host_native_scalar_f32_smoke.py` + `div_shift_cmp_edge_smoke.py` to `resolve_so()` — they currently
  SKIP (exit 0) when `MINDC_SO` is unset instead of building the `.so`, a vacuity risk if wired into CI bare.
- [ ] SOTA-speed method (this is where **mind-lab on s1** comes in): point the autoresearch + alg-inv evolutionary
  search at the perf-critical parts (deterministic reduction schedule #58, GEMM/int-dot tiling #47, register
  allocation for C6) with fitness = speed GATED by keystone byte-identity — the novel edge is *codegen optimized by
  evolutionary search under a hard determinism constraint*. Compile-speed campaign is live on s1 from mind-lab-latest.

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
