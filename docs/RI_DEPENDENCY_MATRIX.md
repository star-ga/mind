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
| 1 | SELF_REPRO_CLOSURE | **PASS** | self-host loop | — | `self_host_loop_smoke.py` (stage1==stage2==stage3) | done (scalar subset) |
| 2 | PURE_MIND_NATIVE_ELF_COMPILER | **PASS** | `testdata/selfhost_loop/stage1.elf` | — | any `--backend=native` build | done |
| 3 | PRODUCTION_NATIVE_DISPATCH | **PARTIAL** | `mindc build --backend=native` (opt-in) | default backend still MLIR | RI-D0 E2E (below) | RI-D1: flip default for supported subset |
| 4 | MLIR_OPT_DEPENDENCY | **PARTIAL** | default `mindc build` | opt-in native path already 0 | `strace -e execve … --backend=native` → 0 mlir-opt | RI-D1 default flip |
| 5 | MLIR_TRANSLATE_DEPENDENCY | **PARTIAL** | default `mindc build` | opt-in native path already 0 | same strace → 0 mlir-translate | RI-D1 default flip |
| 6 | CLANG_DEPENDENCY | **PARTIAL** | default `mindc build` cdylib/exe | opt-in native path already 0 | same strace → 0 clang | RI-D1 default flip |
| 7 | LINKER_DEPENDENCY | **PARTIAL** | default path (clang→ld) | native path emits static ELF directly | `file <native.bin>` → "statically linked, no ld in tree" | RI-D1 default flip |
| 8 | RUST_DRIVER_DEPENDENCY | **NO** | `mindc` (Rust binary) orchestrates + spawns stage1.elf | pure-MIND CLI driver not the shipping entrypoint | `which mindc` is an ELF built by cargo | RI-G: pure-MIND `mindc` replaces Rust driver |
| 9 | SCALAR_LANGUAGE_COVERAGE | **PASS** | native backend | — | int `7+35`→42, struct-return `3+4`→7 via `--backend=native` | done |
| 10 | FLOAT_LANGUAGE_COVERAGE | **PARTIAL** | native backend | tensor-float / full f32 vector surface | `2.5+4.0 as i64`→6 native (scalar OK); tensor float NO | RI-E: native tensor float |
| 11 | TENSOR_LANGUAGE_COVERAGE | **NO** | native backend | tensor lowering not in stage1.elf subset | tensor prog `--backend=native` → FAIL-CLOSED `error[backend-native]` | RI-E: native tensor lowering |
| 12 | AGGREGATE_ENUM_COVERAGE | **PARTIAL** | native backend | full enum-payload / trait dispatch | struct-return native OK; trait → FAIL-CLOSED | RI-F: native trait/enum-payload dispatch |
| 13 | REGALLOC | **PARTIAL** | native emitter | production-grade allocation | DTK (Deterministic Top-K) first slice landed (#254) | RI-C-REGALLOC: production allocator |
| 14 | STDLIB_LINKING | **PARTIAL** | native backend | std-blob linked via self-host image, not general std resolution | scalar builds link the seeded std blob | RI-E stdlib native link |
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
```

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
- **`UNSUPPORTED_FEATURE_DIAGNOSTICS` — tensor/trait now EXACT (RI-D1a, #313)**: a native
  fail-close is no longer a silent 0-byte-exit-0. The pure-MIND self-host driver
  (`selfhost_driver.mind`) scans the user region and writes
  `error[backend-native]: tensor type unsupported` / `… trait/impl unsupported` to stderr and
  exits non-zero (the Rust bridge surfaces it via path #1); any other construct gets an honest
  generic `unsupported construct` (still non-zero — never a silent 0-byte). Gate:
  `self_host_native_diag_smoke.py` (CI-wired). deferred: line/column (no int→decimal helper in
  the self-host dialect yet) + float-specific naming.

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
