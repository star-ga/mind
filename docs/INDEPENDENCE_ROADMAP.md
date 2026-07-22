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
- [x] **narrow-param HANG FIXED as fail-closed (2026-07-21):** a narrow-width (i8/i16/i32) param in a fn that also has a WHILE loop now refuses loudly instead of HANGING / reading a stale carried value. The loop-carry + width interaction desyncs the frame/slot count so `fn f(x: i8){ while c<3 {x=x+1;c=c+1;} return x }` HANGS and `fn f(x: i8){ while.. } return (x as i64)+c` reads c as 0. `selftest_native_elf` scans the pruned fn set (`nb_fns_reassign_narrow_param` = a body-having fn with a narrow param AND a while, via `nb_stmts_have_while`) and returns `es_new()` (empty ELF), gated by `self_host_narrow_param_smoke.py` (4 narrow+loop shapes refused + 3 controls incl. a narrow-param-NO-loop that must still work). TARGETED so it does NOT over-reject: a narrow param that is only read/returned with no loop lowers CORRECTLY via the entry width-wrap driver (autowrap smoke's pw32/wadd pass). Byte-identical: no self-host fn has a narrow param in a loop (the 48 narrow-typed std sigs are extern `unsafe fn`, no body). **Still open (roadmap C2):** (a) actually SUPPORTING narrow-param carry (fix the spill/wrap frame-count desync so `fn f(x:i8){while..}` lowers correctly); (a2) a narrow-typed LOCAL reassigned in a general loop (`let mut y: i8=0; while c<3 {y=y+1;..}`) ALSO hangs via the same desync (i8/i16/i32 all hang) — NOT guarded, because the autowrap smoke's single-iteration `while i<1` narrow-local case works, so a blanket narrow-local-in-loop guard would over-reject it; the real fix is supporting the narrow-local carry, same as (a); (b) a SEPARATE narrow-param `x as i64` CAST bug — `fn f(x: i8){ return (x as i64)+1 }` for f(10) drops the +1 (returns 10) and f(200) fails to wrap to -56 — the cast reads the param before the entry wrap; NOT caught by this guard (no loop), documented for the C2 cast fix. Related: [[reference_native_elf_toplevel_assign_gap_2026_07_21]], C2 narrow-arith.
- [ ] **mindc (Rust) scoping bug: a param-shadowing `let` in a block LEAKS** (found while net-verifying the blind-verify's "gap #2", which was a FALSE POSITIVE). The verifier claimed a self-host divergence (`if x>0 { let x=5; s=x; } return s+x` self-host 10 vs "oracle 15") — but running the real mindc MLIR path returns **10, matching the self-host**: no self-host gap, it faithfully reproduces mindc. The verifier assumed proper block-scoping (15) without running mindc. The REAL finding is in `src/`: a NEW-name `let y` in a block is correctly block-scoped (reading it after the block is a compile error), yet a param-SHADOWING `let x` leaks — `if x>0 { let x=5; } return x` for f(10) returns **5 (inner)** not 10 (param). This is a Rust-compiler scoping asymmetry (src/eval/lower.rs / type_checker), NOT a self-host bug; fixing it is a `src/` change (criterion-relevant) + a reseed to keep the self-host matching. Needs a semantics decision (proper block-scoping -> 15) before the fix. Net-verify note: audit findings themselves get net-verified — this one dissolved on contact with the actual oracle.

### Open follow-ups (2026-07-21, from the param/return-wrap `6e4c809` audit)
- [x] **FIXED (2026-07-21):** Param MUTATION via assign no longer reads the stale home slot. `nb_expr`'s ident arm now consults the let-env BEFORE `resolve_param`, so a reassigned param (loop-carried `x = x + 1` / if-branch merge — bound into the let-env at its live post-slot by `nb_lets_bind`) reads that live slot; a name with no let-env binding falls through to the param home slot unchanged. Was a silent WRONG VALUE with no fail-closed signal (`fn f(x){ while c<3 { x=x+1 } return x }` for f(10) emitted a clean ELF exiting 10, not 13). Gated by `self_host_param_mutation_smoke.py` (CPU-oracle: exit == arg+iters over 5 pairs; the stale-slot bug returns arg) in `fast_keystone.sh` + CI. Byte-identity held: the self-host loop reseeded and the ORACLE leg confirms self-host output == fresh Rust output (no semantic divergence — main.mind reassigns/shadows no param).
- [ ] Minor: `frt_scan_retw` duplicates `frt_scan_ret`'s O(tokens) forward scan (compile-time cost only, zero emission effect) + inherits the pre-existing unguarded `pos+2`-at-EOF read shape shared with the dtype scan — fold into a single scan when convenient.

### Open follow-ups (2026-07-21, from the width-driver `ae3bfdf` audit)
- [ ] **Latent narrow-only count/emit asymmetry** (audit-flagged, currently UNREACHABLE — fails-closed today, zero i64 impact): `nb_count_bind_merged` (main.mind ~23071) hardcodes width **64** for if-merged rebinds, while emit-side `nb_rebind_merges` inherits the true declared width. A narrow var assigned in an `if` branch then reassigned later would count +0 but emit +1 → frame undercount by one slot. The if-statement-with-assign-branches shape fails closed in `selftest_native_elf` on both this commit and its parent, so it's not emittable yet. FIX before that shape becomes emittable: inherit width via a clets lookup in `nb_count_bind_merged`. (While-carry is already safe.)
- [ ] **Narrow-int params + return types don't auto-wrap yet** — the `ae3bfdf` width-driver covers `let` + assign bindings only; params/returns default to width 64. Add explicit `// deferred:` markers at `nb_lets_lookup_width` and the param/return sites when extending.
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
