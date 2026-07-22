# MIND Rust-Independence Roadmap

> **Living source of truth for the path to a 100% Rust-independent, SOTA MIND toolchain.**
> Update status markers as phases land. Grounded in the 2026-07-04 independence audit
> (native-ELF + MLIR + gap agents) and the native-ELF self-host feasibility probe.
> **No fake wins:** every claim in this file is gated on a runnable, byte-verifiable
> artifact — not a byte-comparison that was never executed.
>
> **Measured state (2026-07-21 gap-map):** B-frontend ~30% (type-checker 10 of 22 diagnostic
> rule families ported as pure-MIND decision cores; parser ~40% by construct), C-backend ~35%
> for the static x86-64 product (~15–20% for the full axis). True long poles, in order: C6
> optimizing backend (measured ~9.6× vs MLIR+clang `-O3` — deleting LLVM waits behind it, multi-year),
> B2 `infer_expr` + heavy constructs (quarters), the aarch64 second encoder family (months, cheaply
> de-risked now), C7 GPU. Three distinct finish lines, not to be conflated: (a) 100% Rust-independent
> static x86-64 mindc (~2–3 quarters of serial reseed-gated rungs); (b) + cross-substrate (aarch64,
> +months); (c) + *fastest* (delete LLVM, gated on C6, multi-year). Recent correctness-first rungs:
> param-mutation stale-slot read, non-dyadic float-literal silent-bits, carry/loop-frame 256-cap
> overflow — all silent-miscompile → loud-refusal, keystone byte-identity held.

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
  oracle over positive+negative cases, all gated in `fast_keystone.sh` (`tc_class_rules`, `tc_shape_rules`).
  **Batch (2026-07-21, parallel-ported):** E2015 (let/assign class mismatch), E2006 (fixed-`bytes[N]` into growable
  `bytes`/Vec), the tensor shape-annotation dtype/rank/dim compat guard, and the order-sensitive
  `classify_error_code` router — four more additive `selftest_tc_*` exports, each byte-identical to its Rust oracle
  and gated (`tc_let_class`/`tc_fixed_bytes`/`tc_shape_annot`/`tc_classify`). The
  remaining ~5,000 LOC (float/tensor/enum + AST-context-dependent rules) is the bulk still open.
- [ ] **B2** Full parser + AST→IR lowering (`parser` ~5,563 portable of 7,782 total — the `#[bimap]` + trivia
  ~2,219 LOC are descoped from the self-host target — + `eval/lower.rs` 9,966) for every construct. The self-host
  front-end already lexes+parses+lowers the scalar/i64/control-flow subset (that is what the keystone loop
  compiles); the open 60% is strings, collections, enums, generics, the tensor family, and real type annotations.
  Incrementally portable construct-by-construct (the 67-fixture gap corpus + per-construct byte-oracle prove no
  big-bang needed), but every rung is SERIAL through the RI-E1 reseed, so wall-clock is bounded by rung cadence.
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
  exact-bit vs single-precision reference). The f64 canary stays byte-identical. **Exactness guard LANDED
  (2026-07-21):** the integer-only decimal->IEEE-754 path (`nb_float_lit_bits` -> `nb_strip_pow5`) is exact only
  for a DYADIC literal (5^fk divides num); a non-dyadic literal (`0.1`, `3.14`) truncated in `nb_strip_pow5`'s
  integer `/5` and emitted SILENT wrong bits (`0.1` -> `0.0`). `nb_expr`'s float arm now calls
  `nb_float_lit_is_dyadic` and FAILS CLOSED on a non-dyadic literal instead of miscompiling, gated by
  `self_host_float_lit_exact_smoke.py` (10 dyadic accept + 10 non-dyadic reject, vs the `mod 5^fk` oracle — a
  trunc-based test can't catch it since `trunc(3.0)==trunc(3.14)`). Full correctly-rounded dec2flt (arbitrary-
  precision round-to-nearest-even) stays the deferred upgrade. **Still open:** ISA-selection,
  a byte-identity oracle for the float path, and wiring f32 into the general dtype registry (currently a
  dedicated selftest export, not the default nb_expr dtype). *(Commit/CHANGELOG history labels the earlier f64
  increment **"RI-B1"** — that is roadmap **C1** here, NOT roadmap B1 (pure-MIND type-checker below).)*
- [~] **C2** Narrow ints (i8/i16/i32/u*) — **store/load LANDED (2026-07-21, `1372c4c`):** user-reachable
  `__mind_{store,load}_{i8,i16,i32}` (truncating stores, zero-extend loads) via the general nb_emit_intrinsic
  path, gated by `self_host_native_narrowint_smoke.py` (non-fakeable neighbour/high-bit probes). **Still open:**
  narrow-width wrap-around ARITHMETIC — first slice landed as an additive selftest (`selftest_native_elf_narrow_add_i8`, i8 two's-complement wrap 100+100->-56 via nb_wrap_rax_w after the add); the op x width matrix is now COMPLETE — `{add,sub,mul}x{i8,i16,i32}` (9 drivers) via the parametric `selftest_native_elf_narrow_arith(a,b,op,w)` driver (op in {add,sub,mul}, w in {8,16,32}; host-side width wrap `nb_narrow_wrap_w`; width-aware fail-closed guard `nb_narrow_guard_hi` = [-1e6,1e6] for i8/i16, widened to [-3e9,3e9] for i32 so the wrap past ~2^31 is actually exercised, operands movabs-baked full-width), gated by `self_host_native_narrow_arith_batch_smoke.py` (54 cases, every driver's corpus non-vacuous >=1 genuine wrap, each wrap case proven distinct from the non-wrapped i64 value, width-aware fail-closed guard incl. an in-i32-band/out-of-i8-band ACCEPT proving the guard scales with width; e.g. INT32_MAX+1->INT32_MIN, 65536*65536->0) — all byte-identical to the frozen bootstrap (RI-E1, c3defff9). A narrow-int TYPE + per-op width masking on the *general* (non-selftest) path, and unsigned surface types, remain open.
  · **C3** division / shift / compare — **COMPLETE:** `idiv`+`cqo` with zero & INT_MIN/-1 guards, `sar`/`shl`,
  all 6 signed `setcc`; 16 edge-case tests added (`div_shift_cmp_edge_smoke.py`). Logical-`shr`/unsigned-`setcc`
  unreachable until C2 unsigned types (correctly deferred, no oracle).
- [~] **C4** Tensor/linalg lowering — matmul, reductions, broadcast, indexing. **Native-ELF i64 ops LANDED
  (2026-07-21, zero MLIR/LLVM):** elementwise-add (`selftest_native_elf_tensor_ewadd_i64`, C4-T1), dot/MAC
  reduction (`_dot_i64`, C4-T2), 2-D matmul (`_matmul_i64`, C4-T3, `f468e02`/`24699d0`), 2-D axis reduction
  (`_rowsum_i64`, C4-T4 — squared-row-sum checksum, layout-discriminating) and 2-D transpose (`_transpose_i64`,
  C4-T4 — opposite-stride `i*n+j`->`j*m+i` with a position-weighted checksum), elementwise-multiply (`_ewmul_i64`,
  C4-T5), 2-D column reduction (`_colsum_i64`, C4-T5 — the axis-transpose of rowsum), and row-vector broadcast-add
  (`_bcastadd_i64`, C4-T5 — `C[i,j]=A[i,j]+B[j]`, B stride-0 across rows), row max/min-reduction (`_maxrowmax_i64`/`_rowmin_i64`, C4-T6 — cmp+cmovl running-select), 1-D ReLU (`_relu_i64`, C4-T6 — signed cmp+conditional-zero), and 3-D batched-sum (`_batchsum_i64`, C4-T6 — the first N-D `k*m*n+i*n+j` indexing) — each emits a runnable native x86-64
  ELF with 2-D row-major addressing + a fail-closed frame-bound guard (dims bounded before products, so no
  i64-overflow shape overruns), verified against an independent Python reference AND the native-ELF byte-oracle.
  **Still open:** general N-D indexing, f-typed tensors (needs C1 float-in-registry), and the
  optimizing backend (C6, MLIR still owns performance today).
- [ ] **C5** Vectorization (AVX2/NEON SIMD) for performance parity
- [ ] **C6** ⚠️ **Optimizing backend** — register allocation + instruction scheduling (today LLVM `-O3`). *The multi-year item*; without it native codegen is correct but slow
- [ ] **C7** GPU codegen (CUDA/ROCm/Metal) — *commercial mind-runtime territory, hardest*
- [ ] **C8** Linker + fuller syscall surface (open/close/mmap) for general/multi-object programs

### Open follow-ups (2026-07-21, from the gap-map correctness batch — independent blind-verify of `890f9c2`/`59b1318`/`99ba23a`)
- [x] **narrow-param HANG FIXED as fail-closed (2026-07-21):** a narrow-width (i8/i16/i32) param in a fn that also has a WHILE loop now refuses loudly instead of HANGING / reading a stale carried value. The loop-carry + width interaction desyncs the frame/slot count so `fn f(x: i8){ while c<3 {x=x+1;c=c+1;} return x }` HANGS and `fn f(x: i8){ while.. } return (x as i64)+c` reads c as 0. `selftest_native_elf` scans the pruned fn set (`nb_fns_reassign_narrow_param` = a body-having fn with a narrow param AND a while, via `nb_stmts_have_while`) and returns `es_new()` (empty ELF), gated by `self_host_narrow_param_smoke.py` (4 narrow+loop shapes refused + 3 controls incl. a narrow-param-NO-loop that must still work). TARGETED so it does NOT over-reject: a narrow param that is only read/returned with no loop lowers CORRECTLY via the entry width-wrap driver (autowrap smoke's pw32/wadd pass). Byte-identical: no self-host fn has a narrow param in a loop (the 48 narrow-typed std sigs are extern `unsafe fn`, no body). **Update (`c27a766`):** (a) **FIXED** — narrow-param carry across a TOP-LEVEL while now lowers correctly (`nb_while_carry` mints the twin wrap slot; guard narrowed to allow it, still refuse while-nested-in-if); (a2) narrow-typed LOCAL carried by a loop — **VERIFIED WORKING (`330822c`)**: `let mut y: i8` carried by a top-level while emits+runs correct incl. two's-complement overflow wrap (permanent fixtures added); the `nb_while_carry` twin-slot fix keys on a narrow carried assign target, which covers a local, not just a param; (b) **FIXED (`e69d5b1`)** — `(x as i64)` widening cast now composes in a binop. Root cause was that `as` was never consumed in main.mind's parser, so `(x as i64)` parsed to `paren(x)` leaving position at `as` → the trailing `+expr` became dead code after `ret` (`(x as i64)+3` of f(10) returned 10). Fix: `parse_postfix_rest` consumes `as i64` in postfix position (binds tighter than any infix op) as an exact widening passthrough. Scope confirmed self-host-only (the shipped Rust mindc was already correct — no src/ change). NARROWING casts (`as i8/i16/i32`) in a binop — **FIXED (`330822c`)**: a real `ast_cast` node (kind 25) + `parse_postfix_rest` wrap + the `nb_expr` cast arm emitting the width-matched `movsx` (al/ax/eax) two's-complement truncation (the same wrap the `__mind_wrap_iN` intrinsics use), composing as a binop operand (`(300 as i8)+1`→45, `(200 as i8)`→-56). Count/emit stay balanced (in-place movsx, one result slot — symmetric like `ast_neg`, no twin-slot desync); `ast_cast` arms added to the 4 other native-path walkers for the unary `child1=0` recursion. **C2 narrow-int tier COMPLETE.** Narrow-param carry by a while NESTED IN AN IF — **FIXED (`44a9203`)**: empirical boundary-mapping (guards off, ELF RUN + value-checked vs an independent Python ref) proved the shape ALREADY lowered correctly (the if-region merge threads the nested carry out at any if-depth/then/else); the two guards were merely over-conservative and were replaced by one `nb_region_while_carries_narrow` region-walk that permits top-level OR if-nested while-carry but does NOT descend into a `while` body. **Only genuinely-broken shape still refused:** `while {…while {x=x+1}…}` (nested LOOPS, F2 outer-var region carry). **Boundary precisely mapped (empirical, ELF-run + value-checked):** the miscompile triggers exactly when the OUTER var is reassigned ONLY inside the INNER while and the outer loop iterates >1 (the outer back-edge resets it each iteration → 2, want 4); it is **WIDTH-INDEPENDENT** (i64 miscompiles identically to i8). **Mechanism:** `nb_while_carry` (main.mind:23317) walks only the outer body's top-level statements — a nested `while` falls to the else-arm that just advances `next_id`, never recording an outer-live var reassigned inside it (the deferred marker at 23312-23316 names this). The if-nested case worked because the if-region merge collects the nested while's live_writes; a while-in-while has no merge region — the outer back-edge is the only carry point and never learns x is carried. The real fix is a native-emit analogue of lower.rs's `last_region_exit_rebindings` (descend into nested regions, record the outer-live var at the region's EXIT incarnation, coalesce the inner-exit slot with the outer carry slot, keep the count-pass ↔ emit-pass frame-slot minting in lockstep) — NOT additive/byte-identical, a deliberate mind-ssa + mind-self-host PHASE C follow-up. **Design (panel + Opus consult):** (1) carry detection via a scope-map DIFF (var whose id changed loop-entry→exit is carried, post_id = the id now in the map — so the inner loop's exit incarnation threads up automatically, nested descent for free) rather than the current top-level-Assign pattern-match; (2) enforce count↔emit lockstep with a single shared traversal under a `Mode::{Count,Emit}` flag + one `mint_ssa_id()` chokepoint + an `ssa_id→slot` map built in Count / read-only in Emit (retires the desync hazard class that caused `c27a766`); (3) one slot per unique id, no mint-time aliasing (compaction is a separate Mode-gated pass). **Byte-identity constraint (Opus):** the native emit must reproduce lower.rs `last_region_exit_rebindings` (4737-4777) EXACT id-order + slot-layout, not merely a correct scheme. Falsifiable test for the patch: Count-mode frame-slot count + full SSA-id sequence == Emit-mode on the `while{while{}}` case. **SEV (latent, documented):** the identical bug silently miscompiles **i64** and is UNGUARDED (`nb_fns_reassign_narrow_param` fires only for narrow params) — but it is reachable ONLY via the native-ELF selftest path (main.mind's own recursion has no nested-while-inner-reassign of an outer var, so the self-host loop stays byte-identical), so it is a coverage gap for PHASE D, not a shipped-compiler bug. Narrow-int loop-carry surface fully mapped. Related: [[reference_native_elf_toplevel_assign_gap_2026_07_21]], C2 narrow-arith.
- [ ] **mindc (Rust) scoping bug: a param-shadowing `let` in a block LEAKS** (found while net-verifying the blind-verify's "gap #2", which was a FALSE POSITIVE). The verifier claimed a self-host divergence (`if x>0 { let x=5; s=x; } return s+x` self-host 10 vs "oracle 15") — but running the real mindc MLIR path returns **10, matching the self-host**: no self-host gap, it faithfully reproduces mindc. The verifier assumed proper block-scoping (15) without running mindc. The REAL finding is in `src/`: a NEW-name `let y` in a block is correctly block-scoped (reading it after the block is a compile error), yet a param-SHADOWING `let x` leaks — `if x>0 { let x=5; } return x` for f(10) returns **5 (inner)** not 10 (param). This is a Rust-compiler scoping asymmetry (src/eval/lower.rs / type_checker), NOT a self-host bug; fixing it is a `src/` change (criterion-relevant) + a reseed to keep the self-host matching. Needs a semantics decision (proper block-scoping -> 15) before the fix. Net-verify note: audit findings themselves get net-verified — this one dissolved on contact with the actual oracle.

### Open follow-ups (2026-07-21, from the param/return-wrap `6e4c809` audit)
- [x] **FIXED (2026-07-21):** Param MUTATION via assign no longer reads the stale home slot. `nb_expr`'s ident arm now consults the let-env BEFORE `resolve_param`, so a reassigned param (loop-carried `x = x + 1` / if-branch merge — bound into the let-env at its live post-slot by `nb_lets_bind`) reads that live slot; a name with no let-env binding falls through to the param home slot unchanged. Was a silent WRONG VALUE with no fail-closed signal (`fn f(x){ while c<3 { x=x+1 } return x }` for f(10) emitted a clean ELF exiting 10, not 13). Gated by `self_host_param_mutation_smoke.py` (CPU-oracle: exit == arg+iters over 5 pairs; the stale-slot bug returns arg) in `fast_keystone.sh` + CI. Byte-identity held: the self-host loop reseeded and the ORACLE leg confirms self-host output == fresh Rust output (no semantic divergence — main.mind reassigns/shadows no param).
- [ ] Minor: `frt_scan_retw` duplicates `frt_scan_ret`'s O(tokens) forward scan (compile-time cost only, zero emission effect) + inherits the pre-existing unguarded `pos+2`-at-EOF read shape shared with the dtype scan — fold into a single scan when convenient.

### Open follow-ups (2026-07-21, from the width-driver `ae3bfdf` audit)
- [x] **If-merge count/emit width asymmetry — FIXED (`6e4c809`, same commit as param/return wrap).** `nb_count_bind_merged` now inherits the true declared width (clets lookup) to match emit-side `nb_rebind_merges`, closing the count-vs-emit frame-slot asymmetry for narrow if-merged rebinds before that shape became emittable.
- [x] **Narrow-int params + return types auto-wrap — DONE (`6e4c809`).** `nb_wrap_params_w` (main.mind ~22219) is called at `nb_lower_fn:25565` right after the SysV spill, computing the real declared width via `nb_width_of_ann` and wrapping in place with `nb_wrap_rax_w`; return wrap flows `frt_scan_retw` (`-> iN {`) → `frt_lookup_width` → `lcell+72` → applied by `nb_finish_body:25392` (implicit tail) + the explicit-return arms (23632/23655) via `nb_wrap_ret_w`. Additive (main.mind's kept fns have no narrow param/return — all 7 i8/i16/i32 occurrences are comments), so self-compile byte-identical. Evidence smoke `self_host_native_narrow_paramret_smoke.py` (i8 param div-isolate, i64 controls) net-verified ALL PASS. **Narrow param carried by a TOP-LEVEL loop — now FIXED (`c27a766`):** the count-vs-emit frame desync was in the carry pre-walk — `nb_while_carry` recorded `post_id = arhs-1` and returned `arhs` (no +1) while emit mints the wrap slot at `arhs` and returns `arhs+1`; the fix mints the twin slot for narrow carried targets (i64 unchanged → byte-identical). **Still refused (kept fail-closed):** a while nested in an if (F2 nested-region carry, unrecorded), and the separate `(x as i64)+c` cast-in-binop bug below.
- [x] **Top-level straight-line i64 assign — FIXED (`b93158f`, 2026-07-22).** `let w: i64 = 100; w = w + 100;` (no loop) now emits a runnable native-ELF and runs correct (single 200 / chained 144 / selfref 18 / many 126, byte-exact vs an independent Python ref). A 35-line additive arm in `flatten_stmt_seq` mirrors eval/lower.rs's body_env rebind — `NAME = VALUE` flattens the RHS against the prior env then adds a shadow env entry re-using NAME's span (letenv_lookup last-match-wins); no dedicated Assign instr, so the mic@3 body is byte-identical to the equivalent `let w2 = w + 100`. Additive (main.mind has no top-level reassign in a kept fn — native_elf + mic3_flip byte-identical); reseeded 752495e1, PRIMARY+ORACLE green. Gated by `self_host_native_toplevel_assign_smoke.py`. **Still open (`deferred:` marker):** a value-if RHS (`w = if c {A} else {B}`) falls through to the bare-expr arm and fails closed — mirror the type-7 value-if let sub-case to close it. See [[reference_native_elf_toplevel_assign_gap_2026_07_21]].

### Open follow-ups (2026-07-21, from the `1372c4c` audit)
- [x] **DONE (2026-07-21):** all 14 additive selftest smokes (the C4 tensor ops + type-checker rule ports +
  narrow-int/f32/div/mod/arena-headroom) are now enforced in **BOTH** `fast_keystone.sh` (18 `chk` lines) AND the
  CI `mindcraft_self_host` job (new "Self-host ADDITIVE selftest gates" step, `MINDC_SO` set → fail-closed, never
  SKIPs). Closes the gap where these additive `selftest_*` exports (not reached during self-compile, so uncovered by
  the LOOP / native-ELF byte-identity gates) could regress CI-green.
- [ ] Nice-to-have: port `self_host_native_scalar_f32_smoke.py` + `div_shift_cmp_edge_smoke.py` to `resolve_so()` so
  they BUILD the `.so` when `MINDC_SO` is unset instead of soft-SKIPping — no longer a live vacuity (both CI and
  fast_keystone always set `MINDC_SO`), just robustness for a bare manual run.
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
