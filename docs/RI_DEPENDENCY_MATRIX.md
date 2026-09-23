# Rust-Independence (RI) Dependency Matrix

> **Authoritative status of the MIND compiler's remaining Rust / LLVM / MLIR / toolchain dependencies.**
> Replaces the old, unmeasured "~50%" headline. The headline **only moves when an
> architectural dependency actually changes state** — not when semantic coverage
> grows. Every RI report distinguishes `COVERAGE_MOVED` / `DEPENDENCY_MOVED` /
> `RUST_INDEPENDENCE_MOVED`.

Legend: **PASS** = dependency cut / property holds · **PARTIAL** = holds for an
opt-in / subset · **NO** = still fully depended-on. Verified on
`mindc build --backend=native` at `main` (82cf8e87), x86-64, via strace execve
tracing + byte-diff + run-parity.

| # | Field | Status | Exact consumer | Exact blocker | Exact test | Dependency cut when |
|---|-------|--------|----------------|---------------|------------|---------------------|
| 1 | SELF_REPRO_CLOSURE | **PASS** | self-host loop | — | `self_host_loop_smoke.py` — BOTH legs: PRIMARY (stage1==stage2==stage3==**frozen seed**) and ORACLE (a **freshly built** Rust `.so` emits those same bytes) | done (scalar subset) |
| 2 | PURE_MIND_NATIVE_ELF_COMPILER | **PASS** | `testdata/selfhost_loop/stage1.elf` | — | any `--backend=native` build | done |
| 3 | PRODUCTION_NATIVE_DISPATCH | **PARTIAL** | `mindc build --backend=native` (opt-in) | default backend still MLIR | RI-D0 E2E (below) | RI-D1: flip default for supported subset |
| 4 | MLIR_OPT_DEPENDENCY | **PARTIAL** | default `mindc build` | opt-in native path already 0 | `strace -e execve … --backend=native` → 0 mlir-opt | RI-D1 default flip |
| 5 | MLIR_TRANSLATE_DEPENDENCY | **PARTIAL** | default `mindc build` | opt-in native path already 0 | same strace → 0 mlir-translate | RI-D1 default flip |
| 6 | CLANG_DEPENDENCY | **PARTIAL** | default `mindc build` cdylib/exe | opt-in native path already 0 | same strace → 0 clang | RI-D1 default flip |
| 7 | LINKER_DEPENDENCY | **PARTIAL** | default path (clang→ld) | native path emits static ELF directly | `file <native.bin>` → "statically linked, no ld in tree" | RI-D1 default flip |
| 8 | RUST_DRIVER_DEPENDENCY | **NO** | `mindc` (Rust binary) orchestrates + spawns stage1.elf | pure-MIND CLI driver not the shipping entrypoint | `which mindc` is an ELF built by cargo | RI-G: pure-MIND `mindc` replaces Rust driver |
| 9 | SCALAR_LANGUAGE_COVERAGE | **PASS** | native backend | — | int `7+35`→42, struct-return `3+4`→7 via `--backend=native` | done |
| 10 | FLOAT_LANGUAGE_COVERAGE | **PARTIAL** | native backend | scalar f64 `+ − ×` and f64 COMPARE (`< <= > >= == !=`, IEEE-ordered, native == MLIR incl. NaN / ±0.0 / ±inf); f32 / non-dyadic literal / mixed int-float / `/` / `%` / tensor-float all refused | `2.5+4.0 as i64`→6 native; NaN compare native 0 == MLIR 0 (was native 3); 14-pair f64 compare truth table native == MLIR == IEEE in `ri_d1_frozen_profile_gate.py` | RI-E: native tensor float + f32 compare proof + `(op, operand_type)` keying (#313) |
| 11 | TENSOR_LANGUAGE_COVERAGE | **NO** | native backend | tensor lowering not in stage1.elf subset | tensor prog `--backend=native` → FAIL-CLOSED `error[backend-native]` | RI-E: native tensor lowering |
| 12 | AGGREGATE_ENUM_COVERAGE | **PARTIAL** | native backend | aggregate ABI is intrinsic-call-shaped; array WRITE + enum values unported | fixed-array READ native OK; struct/enum → FAIL-CLOSED (see row-12 measured map) | RI-F: native trait/enum-payload dispatch |
| 13 | REGALLOC | **PARTIAL** | native emitter | production-grade allocation | DTK (Deterministic Top-K) first slice landed (#254) | RI-C-REGALLOC: production allocator |
| 14 | STDLIB_LINKING | **PARTIAL** | native backend | std-blob linked via self-host image, not general std resolution; 19 of 42 `std/*.mind` resolve on NEITHER path | `stdlib_manifest_lint.py` (membership now a checked contract, both directions); scalar builds link the seeded std blob | RI-E stdlib native link |
| 15 | CROSS_MACHINE_DETERMINISM | **PASS** | the wedge | — | `cross_substrate_identity` (avx2==neon canaries) + native build byte-identical across runs | done |

## RI-D0 — E2E GATE (LANDED, `main` 82cf8e87)

`mindc build --backend=native <scalar_project>.mind` compiles end-to-end with **zero
MLIR/LLVM/clang/linker in the process tree**, deterministically, fail-closed on
unsupported constructs, **no native→MLIR fallback**.

```
MINDC_NATIVE_BUILD_E2E              = PASS   (int 42 · float-as-i64 6 · struct-return 7)
MLIR_OPT_IN_PROCESS_TREE            = 0      (strace execve: 2 total, none mlir/clang/ld)
MLIR_TRANSLATE_IN_PROCESS_TREE      = 0
LLVM_CODEGEN_IN_PROCESS_TREE        = 0
NATIVE_ELF_RUNS                     = PASS
REFERENCE_VALUE_PARITY             = PASS
DETERMINISTIC_BYTES_OR_DECLARED_LINK_VARIANCE = PASS (byte-identical across two builds)
NO_NATIVE_TO_MLIR_FALLBACK          = TRUE   (tensor/trait → error[backend-native], refuses to write)
MLIR_UNLINKABLE_NATIVE_STILL_WORKS  = PASS   (mindc built --no-default-features (mlir-build OFF)
                                              → MLIR physically absent from the 6.5MB binary, yet
                                              --backend native emits a running ELF (exit 42) with 0
                                              toolchain; a silent MLIR fallback CANNOT LINK, let alone
                                              run. gate: scripts/ri_d1_mlir_free_gate.sh)
```

The `MLIR_UNLINKABLE_*` row is the strongest leg of rows #4/#5/#6: the earlier
`*_IN_PROCESS_TREE = 0` proves MLIR is not *invoked* at runtime; the compiled-out
build proves it is not *present* — fail-closed BY CONSTRUCTION (gate-assertion #1 (council-decided)).

That gate is EXECUTED, not merely cited: `.github/workflows/ci.yml` job
`ri_d1_mlir_free` ("RI-D1 MLIR-free native path (MLIR compiled out)") runs
`scripts/ri_d1_mlir_free_gate.sh` on every push, and the job is listed in
`.github/required-ci-jobs.tsv`, so a release cannot be cut from a commit it did not
pass. The step maps the script's exit 2 (BLOCKED — a missing prerequisite, i.e. the
claim was NOT measured) to a hard failure; in CI an unmeasured claim is never a pass.
Until that job existed the row above was backed by prose alone: the script was
referenced by this file and ANATOMY.md and invoked by no workflow, no preflight step
and no hook.

Consumer: `src/bin/mindc.rs::run_native_backend_bridge` (bridge, commit 52bd6d3b) →
spawns the frozen pure-MIND `stage1.elf`, captures its stdout ELF, writes it. It does
NOT call `resolve_tools()` / `build_all()` (the MLIR path), so the MLIR/clang absence is
architectural, not incidental.

## NATIVE_PRODUCTION_PROFILE + CUTOVER GATE

`--backend=native` is **PRODUCTION_EXPERIMENTAL**; **`DEFAULT_BACKEND=MLIR`**. The global
default MUST NOT flip while native coverage is partial — flipping at 20.6% would fake rows
3–7 PASS by breaking supported builds (**dependency removal must not be claimed through
capability regression**). RI-D1 = prove native is ready to be the default *for a frozen
production profile with zero supported-semantics loss*, behind this gate:

```
GLOBAL_DEFAULT_NATIVE flips only when ALL hold:
  NATIVE_PROFILE_CORPUS_PASS      = 100%
  NO_SILENT_MLIR_FALLBACK         = TRUE
  VALUE_PARITY                    = PASS
  DETERMINISTIC_OUTPUT            = PASS
  SUPPORTED_FEATURE_REGRESSION    = 0
  UNSUPPORTED_FEATURE_DIAGNOSTICS = EXACT
  CI_NATIVE_PROFILE               = GREEN
```

**Corpus baseline** (52 real `examples/` programs via `--backend=native`, measured):
- **14/52 (27%) PASS** native ELF · **38/52 FAIL-CLOSED**.
- 38 fails bucket (dominant construct): **tensor/ML 16** (long-term) · **float-heavy 14**
  (native-float completion) · **field/method 4**.
- **`UNSUPPORTED_FEATURE_DIAGNOSTICS` — non-zero + named-when-confident, but HEURISTIC (RI-D1a,
  #313; NOT yet truly EXACT)**: a native fail-close is no longer a silent 0-byte-exit-0 — the
  pure-MIND driver (`selfhost_driver.mind`) exits non-zero and writes a stderr diagnostic (the
  Rust bridge surfaces it via path #1). BUT the construct name comes from a **lexical substring
  scan of the raw source** (`Tensor<` / `trait ` / `impl `), which is fragile and known-unsound
  (multi-CLI audit 2026-08-14, net-verified): (a) **false-positive** — a program that fails for
  another reason but contains the substring is MISLABELED (confirmed: `let portrait = 5` +
  `f32` → "trait/impl unsupported"); (b) **missed spellings** — `tensor<` lowercase, `Tensor <`,
  `trait\nGreet` fall to the generic message; (c) most real corpus failures (23/26 single-file
  examples) hit the generic bucket anyway. The **truly-exact** fix is to have
  `selftest_native_elf_u` return `{error_kind, source_lo, source_hi}` recorded at its fail-close
  sites and render THAT (tracked separately) — a FORTRESS main.mind slice. Also deferred: the
  scan is per-byte tail recursion (no TCO in the dialect) so a very large fail-closing program
  can exhaust the stack — bound it. Gate today: `self_host_native_diag_smoke.py` (CI-wired) —
  itself weak (exact-spelling only; audit #8).

## ROW 10 — FLOAT_LANGUAGE_COVERAGE, MEASURED MAP (2026-08-29)

The true float boundary of `--backend native`, measured on the shipped `stage1.elf` and
`target/{release,debug}/mindc` built from this tree. Three buckets — the third is the one
that matters.

**Byte-identical / value-correct (PROVEN).** f64 literals (dyadic, ≤18 digits of integer
content), `+ − ×`, float params, float-returning user calls, `as i64` with saturating
Rust-`as` semantics, `as f64` from an int source. Strict-FP holds **by construction**: the
native emitter has no FMA encoder at all (`grep -ci 'fmadd\|vfma' examples/mindc_mind/main.mind`
→ 0), and `objdump -D -b binary -m i386:x86-64` of a native `a*b + c` artifact shows a
separate `mulsd` then `addsd` — no contraction, so the `-ffp-contract=off` contract the MLIR
path enforces with a flag the native path satisfies by having nothing to contract with.

**Refused, loud (fail-closed, correct).** `f32`, non-dyadic literals (`0.1`), literals ≥2^63,
mixed int/float binops, `float % float`, narrowing float casts, `as f64` from a float source,
bare float if/while conditions, all tensor/SIMD float. Note *which* fence catches each:
`binop.div` and (today) every `__mind_conv_*` cast are stopped by the FIRST fence, while
`f32` / non-dyadic literal / mixed binop pass the first fence and are stopped only by the
frozen ELF. `Instr::ConstF64` is admitted for **every** f64 bit pattern; the dyadic
restriction lives solely in the second fence.

**Admitted but NOT proven — the wedge-breaking one (FOUND + CLOSED).** The frozen profile
admitted `BinOp::{Lt,Le,Gt,Ge,Eq,Ne}` keyed on the OPERATOR alone, so a **float** comparison
was in profile. The two backends do not agree on it:

| | native (`main.mind::nb_fp_setcc_opcode`) | MLIR (`src/mlir/lowering.rs`) | NaN result |
|---|---|---|---|
| `<` | `ucomisd` + `setb` (CF) | `arith.cmpf "olt"` | **native true, MLIR false** |
| `<=` | `ucomisd` + `setbe` (CF\|ZF) | `arith.cmpf "ole"` | **native true, MLIR false** |
| `==` | `ucomisd` + `sete` (ZF) | `arith.cmpf "oeq"` | **native true, MLIR false** |
| `!=` | `ucomisd` + `setne` (ZF) | `arith.cmpf "une"` | **native false, MLIR true** |
| `>` / `>=` | `seta` / `setae` (CF=0) | `"ogt"` / `"oge"` | agree (both false) |

An unordered `ucomisd` sets CF=ZF=PF=1, so the unsigned `setcc` family reads the NaN case
backwards for four of the six operators. Reachable with profile constructs only — a f64
literal plus `*` and `-`: six squarings of `65536.0` overflow to `+inf`, and `+inf - +inf`
is NaN. Measured, one source file, both backends built from this tree:

```
fn main()->i64{let a:f64=65536.0; let b:f64=a*a; let c:f64=b*b; let d:f64=c*c;
  let e:f64=d*d; let g:f64=e*e; let h:f64=g*g; let n:f64=h-h;
  if n == 0.0 { if n < 0.0 { return 3; } return 1; } if n < 0.0 { return 2; } return 0;}

  mindc build --backend native  → exit 3   ("n == 0.0" AND "n < 0.0" BOTH true —
                                            impossible for any real number)
  mindc build --backend mlir    → exit 0
```

Both fences admitted it and the native artifact ran with a silently wrong value — the exact
failure the allowlist exists to prevent, one operator family over from the `Call` and `BinOp`
findings, and the same root shape: **admission keyed on the constructor rather than on what
the construct denotes.** `BinOp::Lt` denotes `cmpi slt` on i64 and `ucomisd`+`setb` on f64;
one is corpus-proven and one is not IEEE.

**Closed first by** `src/ir/frozen_profile.rs` (now superseded — see below): a module
carrying any `Instr::ConstF64` rejected every comparison as `binop.compare_in_float_module`. The taint was module-scoped and rejected
integer comparisons too, because `Instr::Param { dst, name, index }` carries **no type** — a
callee `fn lt(x:f64,y:f64)->i64{ if x<y {…} }` holds a float compare with no float literal in
its own body. Over-rejection is the loud, recoverable direction. Regression pinned as
`ri_d1_frozen_profile_gate.py::OUT_PROFILE["float_nan_compare"]` (`PINNED_OUT_PROFILE` 2→3);
that entry FAILS against the pre-fix binary (`rc=0`, 1038B artifact written) and passes
after — it has teeth.

**Dependency cut when:** `nb_fp_setcc_opcode` grows a NaN-aware predicate (parity-flag
masking for `==`/`!=`, swapped-operand `seta`/`setae` for `<`/`<=`), the taint is replaced by
real `(op, operand_type)` keying (task #313), and a NaN program moves to `IN_PROFILE` proving
native == MLIR. **Row 10 stays PARTIAL** — this removed an unsound admission, it did not add
coverage.

**Native fix landed (supersedes the taint rejection):** `nb_fp_setcc_opcode` now emits the
IEEE-ordered forms — `a < b` / `a <= b` as `ucomisd b,a` + `seta` / `setae` (operands
swapped, false when unordered), `a == b` as `sete AND setnp`, `a != b` as `setne OR setp`.
The float-compare taint is gone and comparisons are admitted in float modules;
`float_nan_compare` moved to `IN_PROFILE` (expected exit 0) alongside a 14-pair f64
truth-table corpus (NaN left / right / both, NaN vs ±inf, finite incl. two negatives,
-0.0 vs +0.0 both ways, -inf vs +inf, inf vs inf, inf vs finite). Each builds through the
frozen stage1.elf and must match both the MLIR backend and an independent IEEE-754 oracle.
Still open: f32 compare (unproven; unreachable today because `__mind_conv_f32` is refused by
`admit_call`) and the u64 ordered-compare gap, both behind `(op, operand_type)` keying (#313).

**Residual, NOT closed (INFERRED from source, not proven).** The float-returning-`main`
epilogue is a raw `cvttsd2si` (`nb_fp_trunc_rax_xmm0` = `F2 48 0F 2C C0`, no clamp), whereas
the MLIR path routes float→int through `emit_saturating_fp_to_i64`. Raw `cvttsd2si` yields
`INT64_MIN` for NaN and for out-of-range, where the saturating contract wants `0` and
`INT64_MAX`. `fn main()->f64` returning `2^128` gives native exit 0 vs MLIR exit 232 — but
MLIR returned 232 for *every* float tested, so its float-`main` path looks value-lossy and
the exit-code channel cannot adjudicate this. Needs a non-exit-code oracle before anyone
calls it a bug in either backend. This is also why the raw-`cvttsd2si` epilogue must not be
reused for the `as i64` cast path, which IS saturating and IS proven
(`general_float_netverify.py`).

## ROW 12 — AGGREGATE_ENUM_COVERAGE, MEASURED MAP (2026-08-29)

Measured black-box against the SHIPPED artifacts (`target/release/mindc` carrying the
RI-D1 first fence + the frozen `stage1.elf`), one program per construct: build with
`--backend native`, then run the ELF and compare its exit code to the expected value.
No rebuild — this is what the seam does today.

**Load-bearing structural fact:** MIND has no aggregate/enum `Instr` variants. A struct
literal and an enum payload record both lower to intrinsic `Instr::Call`s —
`__mind_alloc` + `__mind_store_i64` (`lower.rs::emit_boxed_enum_record`, and the
`StructLit` arm), field reads to `__mind_load_i64`, a full-width `as` cast to
`__mind_conv_i64`. So row 12 is decided almost entirely by `admit_call`, NOT by any
tensor-style named rejection.

| Construct | Fence #1 (allowlist) | Frozen native path | Verdict |
|---|---|---|---|
| fixed-array literal + const index `a[2]` | admit | runs, `3` | **PROVEN in-profile** |
| fixed-array literal + loop index | admit | runs, `10` | **PROVEN in-profile** |
| `let mut` array, no store | admit | runs, `1` | **PROVEN in-profile** |
| scalar `match` | admit | runs, `20` | **PROVEN in-profile** |
| `a[i] = v` (`Instr::ArrayStore`) | **admitted** → now rejected | refuses | **OVER-ADMISSION (fixed)** |
| C-like enum value `let e = E.A` | **admitted** | refuses | **OVER-ADMISSION (open)** |
| struct literal / field read / struct return | reject `call.undefined_or_builtin` | — | **OVER-REJECTION** |
| struct param, nested struct, f64 field, field assign | reject `call.undefined_or_builtin` | — | **OVER-REJECTION** |
| enum tuple-payload / struct-variant + match | reject `call.undefined_or_builtin` | — | over-rejection (unproven anyway) |
| trait dispatch | reject `cannot determine` (parse) | — | out-of-profile (row 12 blocker) |

Two defect directions, both instances of admission keyed on the CONSTRUCTOR rather than
on what the construct DENOTES — the same shape as the `admit_binop` (2026-08-21) and
`admit_call` findings:

1. **Over-admission — `ArrayStore` (CLOSED).** `Instr::ArrayStore` was admitted in one
   arm with `ConstArray | ArrayLoad` purely for sharing their constructor family.
   Reading a fixed array is corpus-proven; writing one is not ported to the frozen
   `stage1.elf`, which refuses it as the SECOND fence. Now rejected as
   `aggregate.array_store`. Capability-neutral and byte-neutral (the build already
   exited 3 with no artifact); it only moves the refusal to where it can name the
   construct. Corroborated independently by `rh_f64_aggregate_canary_smoke.py`.
2. **Over-admission — C-like enum value (OPEN, NOT fixable at IR level).**
   `let e = E.A` lowers to allowlisted scalar IR yet the frozen ELF refuses it, so an
   IR-construct allowlist provably CANNOT separate it from the scalar `match` that does
   work. Honest bound on the mechanism: fence #1 is a sound over-approximation, not a
   complete predictor, and fence #2 stays load-bearing. Safe today (refusal is loud);
   the risk to track is a future `stage1.elf` reseed that ACCEPTS such a construct
   without a corpus program proving it.

**Blocking, not row 12's to fix — the first-fence wiring over-rejects the readiness
corpus.** `admit_call` admits only callees DEFINED in the module, so every aggregate
intrinsic is refused. Measured: **3 of the 5 `IN_PROFILE` programs in
`ri_d1_frozen_profile_gate.py` now fail** (`float_as_i64`, `struct_return`,
`narrow_u8_wrap`), and that gate exits 1. `self_host_native_diag_smoke.py` also exits 1
because the fence intercepts the `trait` program with a parse-level "cannot determine"
before the RI-D1a construct-naming diagnostic can run. Both are fail-closed (no
miscompile risk) but they break the allowlist⇔corpus bijection the cutover gate requires.

Slice plan to restore it — **do NOT name-key the intrinsics**: the corpus proves
`__mind_store_i64` only for an *i64* struct field, and an f64 field is a different
native store path, so a bare `{__mind_alloc, __mind_store_i64, __mind_load_i64,
__mind_conv_i64}` allowlist would re-commit the constructor-vs-denotation defect one
level down. The sound form needs the field/operand type — the same `FnDef.value_types`
threading `admit_binop` already defers for `(op, operand_type)` keying. Sequence: (a)
thread `value_types` into the predicate; (b) admit each intrinsic keyed on its
*operand type*; (c) extend `IN_PROFILE` with an f64-field struct program so the
admission is corpus-derived rather than asserted; (d) add the bijection test that fails
when the allowlist and the corpus disagree in EITHER direction — which is what would
have caught both defects above.

## ROW 14 — STDLIB_LINKING, MEASURED MAP (2026-08-29)

Row 14 stays **PARTIAL**. What changed is not the status but the *shape* of the
dependency: std membership was an unchecked hand-copied fact and is now a checked
contract, `examples/mindc_mind/testdata/stdlib_manifest.txt`.

**Measured, three different sets — none of which agreed and none of which were checked:**

| Set | Count | Who consumes it |
|---|---|---|
| `std/*.mind` on disk | 42 | — |
| `STDLIB_MIND_SOURCES` (`src/project/stdlib.rs`, `include_str!` bundle) | 23 | general `use std.<m>` resolution, `feature = "cross-module-imports"` |
| native seed blob | 21 | `mindc.rs::run_native_backend_bridge` → frozen `stage1.elf` |

`seed(21) ⊂ bundled(23) ⊂ disk(42)`. The two extra bundled modules are `http` and
`sha512`. **19 modules resolve on NEITHER path** — the whole TLS 1.3 / crypto surface
(`aes_gcm`, `chacha20_poly1305`, `ecdsa_p256`, `hkdf`, `hpack`, `http2_frame`,
`keccak`, `mlkem768`, `rsa_pss`, `tls13_*`, `x25519`, `x25519mlkem768`, `x509`) plus
`detmath`, `llvm`, `mlir`. `use std.x25519` resolves nowhere today.

**Structural fact this shares with row 12:** the defect was the same one the
2026-08-21 audit found in `admit_binop` and the tensor-builtin fix found in
`admit_call` — a fact asserted by its *duplicate spelling* rather than derived from
what it *denotes*. Membership was spelled out in 17 hand-copied literal lists (the
bridge + 16 `examples/mindc_mind/*.py` copies) plus a 4th, differently-spelled list
in `stdlib.rs`, with nothing executable asserting they agreed.

**Landed:** one committed manifest, exhaustive over `std/*.mind` with **no wildcard
row** — a new std module is a hard lint failure until someone records a deliberate
in/out decision per consumer, exactly as a new `Instr` variant is a compile error in
`profile_frozen_admits`. Both named readers consume it: `mindc.rs` via `include_str!`
(compile-time, so the shipping binary gains no runtime file dependency) and the
Python smokes via `_stdlib_manifest.py`. The seed blob's sha256 is pinned in the
manifest header, so a reseed — which changes the compiled bytes of every native
build — cannot land as a quiet one-line edit.

**Still open for row 14 → PASS (general std resolution):**

1. **There is no std *resolver* on the native path at all.** The bridge does not
   resolve `use std.<m>`; it unconditionally concatenates all 21 seed modules into
   every compile, whether the program imports them or not. General resolution means
   import-driven selection, which changes the compiled bytes → gated behind a reseed.
2. **The 19 unreachable modules** must become reachable by *some* path before "the
   stdlib links" is true in the ordinary sense.
3. **`STDLIB_MIND_SOURCES` is still a hand-written list** — now drift-checked against
   the manifest in both directions, but not yet *generated* from it. Making it
   data-driven needs `build.rs` codegen (`include_str!` needs literal paths), which is
   a separate change on the `cross-module-imports` surface.
4. Until 1–3, "scalar builds link the seeded std blob" remains the honest claim.

## PRIORITY ORDER (by production-profile dependency impact, NOT patch size)

1. **RI-D1a — EXACT native diagnostics — ✅ LANDED (#313)**: the native backend fail-closes with a
   stderr diagnostic naming the construct (tensor / trait-impl EXACT; else honest generic) and a
   non-zero exit, via a pure-MIND capability scan in `selfhost_driver.mind` (self-host closure
   stayed byte-identical — no capability regression). Gate `self_host_native_diag_smoke.py`
   (CI-wired, RI-D seam gates step). deferred: line/column + float-specific naming.
2. **RI-D1b — rank + close the top production-profile blocker** (once diagnostics are exact):
   for each missing construct compute BLOCKED_REAL_PROGRAMS / BLOCKED_FUNCTIONS /
   UNLOCKED_DEPENDENCIES / IMPLEMENTATION_RISK; take the max-coverage-unlocked one. Candidates
   today (approx, grep-bucketed, unverified vs exact diagnostic): native-float completion (14),
   field/method (4). Verify against the corpus before choosing.
3. **RI-D1 cutover**: flip the frozen production profile to native once the gate is 100% green.
4. Native **tensor / aggregate** coverage (rows 11–12) — unblocks the 16 tensor/ML programs.
5. **Register allocation** sufficient for production (row 13).
6. Remaining **linker / toolchain** dependency removal (rows 6–7 default path).
7. **RI-G — pure-MIND shipping/bootstrap** replacement of the Rust driver (row 8;
   `RUST_DRIVER_DEPENDENCY=NO` today — native path is Rust orchestration spawning a pure-MIND
   compiler). Only after native approaches full supported-language parity does the GLOBAL
   default change.

Diagnostic (NOT the cutover gate): isolated single-fn native byte-exactness =
402/1954 = 20.6% (`cutover_coverage_measure.py`). Whole-program profile pass = 27% (14/52).

## SCHEDULER RULE

A small semantic slice ships **only if** it (a) blocks RI-D1 or another dependency-cut
milestone, (b) is required by a representative production corpus, or (c) fixes a
correctness bug on an already-supported native feature. Otherwise → backlog. Characterized
gaps (narrow-arith-result mask, signed narrow, enum payload-binding) are **not** automatically
next — first ask: `DOES_THIS_BLOCK_THE_NEXT_DEPENDENCY_CUT?`

## Coverage vs dependency (why "50→50" looked stalled)

`cutover_coverage_measure.py` (in-isolation, single-fn native-ELF byte-exactness vs the Rust
oracle over main.mind's 1954 fns): **402/1954 = 20.6%** byte-exact. This is a *coverage*
metric — it grew slice-by-slice (#309–#312) but did **not** move a *dependency*. The
dependency that moved is **RI-D0** (rows 3–7 → PARTIAL, the native seam works E2E with zero
MLIR). Reporting rule going forward:

```
#309..#312 note-emit slices:  COVERAGE_MOVED=YES  DEPENDENCY_MOVED=NO   RUST_INDEPENDENCE_MOVED=NO
RI-D0 native shipping seam:   COVERAGE_MOVED=MAYBE DEPENDENCY_MOVED=YES  RUST_INDEPENDENCE_MOVED=YES
```
