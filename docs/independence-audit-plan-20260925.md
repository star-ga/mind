# Independence audit and plan — 2026-09-25

Audit of `star-ga/mind` at `9c231f5` (2026-09-23, version 0.10.2), its branches, its roadmaps
and the downstream code that compiles with it, followed by the plan that comes out of it.
Every number below was measured on that commit unless a row says otherwise. Where this document
and an older status page disagree, this document is the newer measurement; the stale pages are
listed in §8.

Direction set by the owner:

1. **100% Rust independence.** `mindc` becomes a self-hosted pure-MIND program; the Rust compiler
   is the bootstrap oracle and is archived at the end (INDEPENDENCE_ROADMAP Phase E).
2. **MIND's own native backend instead of MLIR / LLVM / clang / system linkers.** The pure-MIND
   native-ELF emitter that already exists is the base (Phases C and D).
3. **Every OS family.** Linux, macOS, Windows, the BSDs, illumos, Android, iOS, WASI, UEFI and bare
   metal, on x86-64, AArch64, RISC-V, wasm32 and small embedded ISAs.
4. **State of the art, measured.** Better than the mainstream languages and the top compilers on
   the dimensions MIND chooses to compete on, each claim backed by a reproducible, signed
   benchmark receipt. The plan does not port the Rust compiler's architecture; it replaces it.

## 1. Ownership

| Workstream | Owns | Does not touch |
|---|---|---|
| Rust compiler and tooling | `src/**/*.rs`; `mindc fmt` / `check` / reporters; the native bridge (`src/build/native_bridge.rs`); packaging and releases; the ecosystem `mindc check` pass rate | `examples/mindc_mind/*.mind`, the frozen `stage1.elf` seed |
| Pure-MIND self-host | `examples/mindc_mind/*.mind`, `testdata/selfhost_loop/stage1.elf`, self-host gates, loop-carry miscompile fixes and self-host refusals | `src/**/*.rs` |

A change that moves an oracle the other workstream depends on (for example a Rust semantics fix
the self-host must then match) lands with a note in the commit message and a report to the other
workstream before the self-host gate is re-anchored.

## 2. Measured state

| Area | Value | Notes |
|---|---|---|
| Rust in `src/` | 148,924 lines in 240 files | 80,268 / 119 on 2026-07-04; the Rust side grew while independence work progressed |
| Pure-MIND compiler | `examples/mindc_mind/main.mind` 57,181 lines, 2,179 functions (618 native-backend `nb_*`) | single file |
| Frozen seed | `stage1.elf` 2,450,978 B; stage1 == stage2 == stage3 | code fills 58% of the fixed 4 MiB R+X window |
| Self-compile speed | ~3.0K lines/s, 4.0 GB peak RSS (68,673 lines in 22–23 s) | single thread |
| Native code quality | Collatz 1..3M: 4.45–4.72 s vs gcc -O2 0.70–0.76 s, clang -O2 0.50–0.53 s, gcc -O0 1.35 s | ~6.5x slower than gcc -O2 |
| Native targets | x86-64 Linux static ELF only | `--backend native --target <other>` currently emits x86-64 Linux and exits 0 |
| Native production reach | the Rust admission check (`src/ir/frozen_profile.rs`) refuses struct returns, casts, strings, narrow ints and signed `/`, `%` | the self-host already implements most of these |
| RI-D1 readiness gate | exits 1 (3 of 5 in-profile programs fail) | runs in preflight only, not CI |
| Native build of example programs | 10 of 111 programs with `fn main` build and run via `--backend native` | 85 refusals name an unresolved intrinsic, mostly `__mind_alloc` |
| Pure-MIND front end vs Rust | 188/223 corpus files parse; 75/223 mic@3 byte-exact; 11 files emit wrong bytes | the pure-MIND parser has no syntax-error path |
| Cross-substrate identity | 35/35 fixtures, x86 AVX2 == ARM NEON | MLIR path only; the native emitter has never run on ARM |
| Downstream compile health | 82 of 534 `.mind` files in 12 downstream repositories pass `mindc check` (15%); 79/534 with `cross-module-imports` | first errors: parse 70%, unknown qualified type, unsupported call |
| Type checker port | ported rules run only in self-test exports; 11 newer error codes (E2025–E2036) have no port | |
| Optimizer | `optimize_mic3` is Rust-only, opt-in, and never reaches native output | the native bridge sends source, not optimized IR |
| Build, lint, tests | fmt, build, clippy and the full CI test matrix pass locally; CI green on `9c231f5` | `Cargo.lock` is not tracked |
| Releases | last tag v0.10.2 (2026-08-07); 548 commits since | release workflow builds linux-gnu x64/arm64, macOS x64/arm64, windows-msvc x64 with cargo + LLVM |

## 3. Correctness findings

Ordered by blast radius. "Fail-open" means a wrong program is accepted or wrong output is produced
without an error. Every row was reproduced on `9c231f5` except those marked *(reported)*, which
come from downstream reports against `9c231f5` and are confirmed before they are fixed.

| # | Finding | Kind | Fix location |
|---|---|---|---|
| 1 | `mindc fmt` deletes `module X { }` wrappers and strips module qualifiers (`fp.g(x)` → `g(x)`) while reporting success | data loss | `src/parser`, `src/fmt` (fix exists on `fix/fmt-lossy-desugars-20260912`) |
| 2 | `cross-module-imports` builds skip arity checking (E2005) everywhere, including cross-module calls | fail-open | `src/type_checker` |
| 3 | `--reporter json` drops project-level errors: empty JSON with exit 1, the message only on stderr | fail-open for CI consumers | `src/check`, reporters |
| 4 | An unknown bare type name is accepted by `check` and `test` in every position (param, return, `let`, struct field, `as`, generic argument); only `a.B` names are existence-checked | fail-open | `src/type_checker/qualified_enums.rs` |
| 5 | Module-qualified types are rejected unless `cross-module-imports` is enabled, which is not a default feature; under it, qualified types inside `module X { }` files do not resolve | language depends on a Cargo feature | `Cargo.toml`, resolver |
| 6 | MLIR `linalg.matmul` / `conv_2d` read an uninitialised output buffer (one call returned 14.0 and 1.78e47) | nondeterministic output | `src/mlir` (fix exists on `fix/mlir-accumulating-linalg-zero-init-20260916`) |
| 7 | Native u64 ordered compares use signed instructions | miscompile | admission + native lowering (fix exists on `feat/rh-native-divmod-main-20260916`) |
| 8 | Lowering panics on another module's `Enum::Variant`, which type-checks (`src/eval/lower.rs:5497`); `let x = null;` panics at the same site | crash after successful check | `src/eval/lower.rs` |
| 9 | Non-`pub` functions are global link symbols: two modules with the same private name fail to link | link failure | MLIR emission / symbol naming |
| 10 | Bundled `std.fs`, `std.net` and `std.regex` are undefined in native executables while bundled `std.sha256` links | link failure | std bundling |
| 11 | `std.time` lowers to `__mind_now_ns`, which native executables do not register; `__mind_argc` / `__mind_argv` are not registered in every backend | missing intrinsic (E2024) | intrinsic table, native bridge |
| 12 | A `let` that shadows a parameter inside a block leaks out of the block (`if x>0 { let x=5; } return x` returns 5 for `f(10)`) | wrong scoping | `src/eval/lower.rs`, type checker |
| 13 | `return` inside `while` produces MLIR that fails with "func.return must be the last operation" | rejected valid program | MLIR lowering |
| 14 | *(reported)* A failed `mindc build` can exit 0 with a runtime-JIT object, an embedded-source object or a shell launcher; some paths already refuse with E5003 / E5005 | fail-open | `src/build` |
| 15 | `mindc test` takes the then-branch of every `if` that compares f64 values | false test passes | interpreter |
| 16 | `const PI: f32 = 3.14;` fails E2001 while `let x: f32 = 1.5;` passes | rejected valid program | type checker |
| 17 | The native bridge rejects `src/`-relative imports, `/ % << >>` and `__mind_*` intrinsics | narrow native reach | `src/build/native_bridge.rs`, `src/ir/frozen_profile.rs` |
| 18 | *(reported)* KAT packages cannot use `../` sources; path dependencies do not resolve `import` of their modules | packaging | `src/project`, `src/deps` |

## 4. Branches

| Branch | Disposition |
|---|---|
| `fix/mlir-accumulating-linalg-zero-init-20260916` | land on main first (finding 6) |
| `fix/fmt-lossy-desugars-20260912` and its superset `fix/type-check-e2016-bench-harness-20260912` | salvage onto main (finding 1; E2016 "check passes, build panics"); drop the commit already on main |
| `feat/rh-native-divmod-main-20260916` | rebase and land (native `/` `%`, unsigned tracking, finding 7, RI-D1 gate green) |
| `fix/d4-finish-20260915` | rebase and review (deref-assign `*r = v`) |
| `feat/native-float-narrow-codegen` | salvage the array pointer-add guard and fixed-array store, then delete |
| nine branches already merged into main | delete |

## 5. Plan

Each item lands on main with a gate in the repository's style: byte-identity against an oracle,
a fail-closed refusal of what is not supported, a negative control for every new check, and the
existing CI matrix green. "Never worse": a landing may not turn a green gate red.

### Track 0 — correctness first (Rust compiler and tooling)

1. Findings 1–3 (formatter data loss, arity checking, JSON reporter).
2. Findings 6, 8, 9, 10, 11 (matmul init, enum-variant panic, private link names, bundled std,
   intrinsics).
3. Findings 4, 5 (bare types; make `cross-module-imports` a default feature so one binary has one
   language), 12, 13, 14, 15, 16, 18.
4. Track `Cargo.lock`; pin release tooling; re-enable the MSRV job.
5. Downstream acceptance in CI: the downstream fail-closed build gate and the standard-library
   prototypes run against every compiler change; the downstream `check` pass rate is measured on
   pinned commits and reported per landing (baseline 82/534).

### Track 1 — dependency cuts (Rust independence)

Only items that remove a dependency move the headline; coverage is reported separately
(RI_DEPENDENCY_MATRIX "coverage vs dependency").

1. Pre-dispatch routing: the compiler chooses native or MLIR per program before compiling, prints
   the route and records it in the evidence. No global default flip.
2. Admission keyed on (operation, operand type) from `FnDef.value_types`, so features already
   implemented in the self-host reach `mindc build`. RI-D1 gate in CI.
3. Import-driven standard-library resolution for the native path (RI row 14).
4. Native trait / enum-payload dispatch (RI-F).
5. The pure-MIND driver replaces the Rust driver (RI-G, C8): file write, argv, `build` / `run` /
   `test` / `check` first.
6. Native keystone; then Phase E — archive the Rust compiler only after an independent
   diverse-double-compiling check and cross-host reproducibility exist, and retire the two-emitter
   agreement checks in the same change (Gate lifecycle, set A).
7. Ratchet: Rust, Python and shell line counts may only go down from the day this plan lands.

### Track 2 — MIND's own backend in place of MLIR / LLVM

1. One cross-ISA semantics table (division by zero, `INT_MIN / -1`, shift counts, saturating
   float-to-int, NaN bit patterns, no FMA contraction, pinned reduction order). Every ISA
   reproduces it with explicit guard sequences.
2. An ISA-neutral machine IR extracted from the `nb_*` lowering, proven by the x86 byte-identity
   gate, before a second ISA is added.
3. Code layout v2: lift the 4 MiB code window, position-independent mode, aligned note, room for
   per-OS notes.
4. Optimizing tier on mic@3: linear-scan register allocation, then inlining, then SCCP / GVN /
   LICM, with the optimized IR feeding the native path ("compile what you hash").
5. Native tensors and deterministic float reductions.
6. Port `runtime-support/*.c` to MIND; no new C in `runtime-support`.
7. Native shared libraries and `extern "C"`: ELF `.so` (PIC, versioned exports, hidden-by-default
   visibility), then Mach-O `.dylib`, then PE `.dll`; full C ABI for SysV, AAPCS64 and Win64.
8. Delete MLIR / LLVM / clang only after the optimizing tier meets its speed gate (Phase D).

### Track 3 — every OS family

| Tier | Targets | Guarantee |
|---|---|---|
| 1 | x86-64 Linux, AArch64 Linux, x86-64 Windows, AArch64 macOS | self-hosting host; byte-identical output from every host; run-parity on every change |
| 2 | RISC-V 64 Linux, wasm32-wasip1, x86-64 FreeBSD, AArch64 Android, AArch64 Windows, x86-64 macOS / universal2, x86-64 UEFI, Cortex-M and RV32 bare metal | emitted and run-tested in emulation or a VM |
| 3 | AArch64 FreeBSD, NetBSD, OpenBSD, illumos, AArch64 UEFI, iOS (libraries only) | emitted, validated, weekly smoke |

Order: target description record and an OS-services layer for `std` (raw syscalls on Linux,
Android, FreeBSD; libc or system-library imports on macOS, iOS, OpenBSD, NetBSD, illumos;
kernel32 on Windows; WASI imports; none on bare metal) → FreeBSD (new OS, no new ISA) → AArch64
encoder and Linux AArch64 host → PE32+ (one all-or-nothing change) and Windows host → Mach-O with
libSystem imports and deterministic ad-hoc signing, macOS host → Tier 2 and 3. Rule: no
source-level `cfg(target_os)` in the portable profile, so `trace_hash` stays target-independent;
OS variation lives below mic@3.

### Track 4 — state of the art, measured

A claim is published only with a signed receipt (hardware id, compiler `trace_hash`, input hash,
command). Targets:

| Metric | Today | Target |
|---|---|---|
| Edit-to-diagnostic latency | full recompile | p95 ≤ 20 ms at 100K lines, identical to a clean check |
| Incremental rebuild | none | ≤ 50 ms at 100K lines; byte-identical to a clean build |
| Cold compile throughput | ~3.0K lines/s, 4.0 GB | ≥ 150K lines/s/core end to end, debug tier |
| Optimized code, strict FP | ~6.5x slower than gcc -O2 | ≥ best of gcc / clang -O3 geomean on 4 hardware classes, bit-identical |
| Cross-substrate identity | 35 fixtures, MLIR path | 100% of ≥ 10,000 cells over the flag × target product, including float reductions and libm |
| Target reach | 1 triple | ≥ 20 triples, all run in CI |
| Toolchain | 148,924 lines of Rust + LLVM | 0 lines of Rust, no external tool in any build path |
| Diagnostics | stable codes | every diagnostic has a code, a span and SARIF; most carry a machine-applicable fix |
| Optimizer evidence | none | every optimizing pass emits a certificate checked by a small pure-MIND checker |

Design rules: one canonical IR end to end; content-addressed, query-based incremental compilation
whose keys double as evidence; deterministic parallel compilation (`-j1` and `-jN` byte-identical);
targets described as data; numerics fixed by the spec (no implicit FMA, explicit reduction order);
an integrated linker per OS; split the single `main.mind` into modules before incremental work.

### Track 5 — language and IR contracts

1. One machine-checked grammar at the 0.10 surface as the source for both parsers, the spec and
   the editor grammar; settle attribute form, `::` vs `.`, bitwise precedence and trait `;`.
2. One error-code registry shared by the spec, the Rust compiler and the pure-MIND compiler.
3. A normative, versioned mic@3 wire specification and a `trace_hash` stability policy across
   compiler versions (today a patch release changes the hash of unchanged source).
4. The spec's conformance corpus runs in `mindc conformance` and in CI.

### Track 6 — downstream compatibility

1. Pinned, reproducible `mindc` releases (tag, digest, tracked lockfile) and enforced
   `mindc-min` / `mindc-max`.
2. Syntax gap triage, ranked by downstream use: `[v; N]`, `for (a, b) in`, `if let`, `..base`,
   enum `: u8` repr, static globals, reductions — implement each or reject it with a fix-it.
3. The runtime boundary consumes mic@3 instead of linking the compiler crate's parser and
   evaluator; freeze the launcher ABI and the C header.

### Track 7 — release, documents, hygiene

1. Cut 0.11.0 after Track 0 (breaking: Ed25519 seeds refused, `mindc test` fails on zero tests,
   new error codes).
2. Rewrite `STATUS.md`, the version matrix and RFC status headers; add an RFC index; extend
   `scripts/check_claims.py` to the new counts.
3. Delete the merged branches and stale release drafts; remove stray root files; fix `.gitignore`
   overlaps; make `ANATOMY.md` generation locale-stable; scope the global `jobs = 2`; declare
   Python ≥ 3.12 for the gate scripts.

## 6. Rules that hold throughout

- One semantics engine; no construct gets a third hand-written implementation.
- Fail closed with a located diagnostic; never a silent fallback.
- Never worse: a red gate is a defect, not a baseline.
- Determinism over speed: a faster result that is not bit-identical is a regression.
- Claims come from CI-verified facts, never ahead of the code.

## 7. Decisions for the owner

1. Parameter-shadowing scope: adopt proper block scoping (finding 12). The self-host currently
   reproduces the leaking behaviour and must follow.
2. WASM and RISC-V were deferred in an earlier plan; this plan puts them in Tier 2.
3. Tier-1 hosts as in §5 Track 3.
4. Machine-IR extraction before AArch64, rather than porting `nb_*` per ISA.
5. `cfg(target_os)` only outside the portable profile.

## 8. Stale public statements to correct

- `STATUS.md` (dated 2026-07-21, 0.10.1): PT_NOTE wiring and `src/native` removal are done; 42 std
  modules, not 13; keystone 7/7; RFC 0021 steps 4 and 6 are done; Ed25519 is retired.
- `README.md` badge 0.10.1; `docs/version-matrix.md` 0.10.0 and `src/native`.
- RFC headers that understate shipped work (0002, 0003, 0005, 0006, 0013, 0021, 0022) and texts
  that describe retired signing schemes (0016, 0017, 0021); RFC 0024 lists error codes the
  compiler does not emit.
- `docs/install.md`: release targets are linux-gnu and separate macOS builds, not musl or universal.
