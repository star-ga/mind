# RFC 0010 Phase G/H/I — Migration Plan (corrected against real architecture)

> Strangler-pattern, byte-identity-gated migration of mindc's codegen
> off the Rust MLIR/LLVM crates. **This plan corrects a false premise
> in RFC 0010 §5**: the heavy Rust FFI binding it assumed does not
> exist. The real picture is documented below.

## 0. Corrected current-state inventory (the load-bearing finding)

RFC 0010 §5 assumed mindc calls libMLIR / libLLVM through the Rust
`mlir-sys` / `inkwell` crates, and that Phase I = "rip out that FFI."
**That is not how mindc works.** Verified 2026-05-21:

- `Cargo.toml`: `default = []`. `melior` (MLIR Rust bindings) and
  `inkwell` (LLVM Rust bindings) are **optional** deps gated behind
  the experimental `mlir_backend = ["melior"]` and `llvm = ["inkwell"]`
  features. Neither is in `default`, `mlir-build`, `std-surface`, or
  any feature the production binary or CI builds with.
- `grep -rln 'use melior|melior::' src/` → **zero hits**.
  `grep -rln 'use inkwell|inkwell::' src/` → **zero hits**. The crates
  are declared but never imported by compiled code. They are dead
  optional dependencies.
- The real codegen path is `mlir-build = ["mlir-subprocess"]` →
  `mlir-subprocess = ["which"]`. mindc emits **MLIR text** from
  `src/mlir/lowering.rs` (2,326 LOC of pure string construction — no
  FFI) and shells out to the external `mlir-opt` / `mlir-translate` /
  `llc` / `clang` binaries as subprocesses to lower that text to an
  object file.

### What this means

1. **Phase I (drop melior + inkwell from Cargo.toml) is nearly free.**
   They are vestigial. Removing them changes no compiled behaviour
   because no compiled code imports them. The only gate is confirming
   the `mlir_backend` / `llvm` experimental features are not relied on
   by any test or downstream — and they are not (no `--features
   mlir_backend` in CI).

2. **The real Rust in the codegen path is `lowering.rs`** — the
   MIND-IR → MLIR-text emitter. And here is the crucial fact: the
   pure-MIND compiler (`examples/mindc_mind/main.mind`) **already
   emits byte-identical MLIR text** (Phase 6.5 apex, v0.6.1
   fixed-point). The logic that lowering.rs performs in Rust is
   *already mirrored* in pure MIND and proven byte-identical.

3. **The subprocess tools (`mlir-opt`/`llc`/`clang`) are C++**, not
   Rust. They are the codegen backend. No language self-hosts those
   (Rust itself shells to / links LLVM). They are out of scope for
   "Rust independence" exactly as MLIR/LLVM-the-backend is out of
   scope.

So "full Rust independence for the compile path" decomposes very
differently from RFC 0010 §5's framing:

| Layer | Today | Rust-independent? |
|-------|-------|-------------------|
| Lex / parse / typecheck / IR | Rust *and* pure-MIND (byte-identical, Phase 6.5) | pure-MIND path exists |
| IR → MLIR text | `lowering.rs` (Rust) *and* `main.mind` (MIND, byte-identical) | pure-MIND path exists |
| MLIR text → object | subprocess `mlir-opt`/`llc`/`clang` (C++) | out of scope (backend) |
| Build orchestration | `mindc build` (Rust crate) | RFC 0008 Phase G shipped; pure-MIND port deferred |
| melior / inkwell | declared, **unused** | trivially removable |

## 1. Revised phase definitions

### Phase G (was: migrate MLIR-glue Rust→MIND) → **redefined**
Two independent workstreams:

- **G1 (trivial, ship now):** Delete `melior` + `inkwell` from
  `Cargo.toml` and the `mlir_backend` / `llvm` feature rows. Confirm
  CI green (they're unused). This is the literal "Phase I keystone"
  from RFC 0010 §5 and it costs ~10 lines + one CI run. It does NOT
  achieve Rust independence by itself — it just removes dead weight.

- **G2 (the real work):** Make the pure-MIND MLIR-text emitter
  (`main.mind`) the *primary* path that `mindc build` invokes, with
  the Rust `lowering.rs` demoted to a differential oracle. Since both
  already produce byte-identical text, this is a wiring change: route
  `mindc build` through the pure-MIND `libmindc_mind.so` for the
  IR→MLIR-text step, keep `lowering.rs` compiled only under a
  `differential-oracle` feature for the byte-identity gate.

### Phase H (was: migrate LLVM-glue) → **mostly N/A**
There is no LLVM Rust glue in the production path — `llc`/`clang`
subprocess does LLVM. The only LLVM-Rust is the unused `inkwell`,
removed in G1. The std.llvm bindings (shipped at `34dd39d`) become
relevant only for a *future* in-process codegen path that replaces
the subprocess shell-out — an optimization/portability play (RFC
0010 §5 Phase H reframed as optional, post-1.0).

### Phase I (the keystone) → **redefined as "Rust crate optional"**
The condition for "the Rust mind crate is no longer load-bearing":
`mindc build` drives the full compile (lex→…→MLIR-text→subprocess)
using only the pure-MIND `libmindc_mind.so` for every MIND-logic
step, with the Rust crate reduced to: (a) the thin CLI binary that
dlopen's `libmindc_mind.so`, (b) subprocess invocation of the C++
backend tools. Whether even (a) can be pure-MIND depends on RFC 0011
(needs process/exec — `std.process` shipped at `7e65212` already
provides spawn/exec, so this is now feasible).

## 2. Strangler increments (G2 — the real migration)

| # | Increment | Differential test | Risk |
|---|-----------|--------------------|------|
| G2.1 | Build `libmindc_mind.so` IR→MLIR-text entrypoint callable from `mindc build` via dlopen | Both paths emit byte-identical MLIR text for every `tests/conformance/**` fixture | Med |
| G2.2 | Route `mindc build --emit-mlir` through the pure-MIND path behind a `--frontend=mind` flag (default stays Rust) | Same fixtures, byte-identical, plus the bootstrap fixed-point oracle | Med |
| G2.3 | Flip the default to `--frontend=mind`; Rust `lowering.rs` compiled only under `differential-oracle` feature | Full test suite + bootstrap fixed-point + bench-gate | High |
| G2.4 | Move `lowering.rs` behind `#[cfg(feature = "differential-oracle")]`; default build no longer compiles it | CI green without the feature | Med |

## 3. The bootstrapping paradox (already solved)

"mindc's own MLIR-emission written in MIND needs a working mindc to
compile it" — this is the *same* paradox Phase 6.5 solved for the
front-end. The resolution: the Rust mindc is the stage-0 bootstrap
compiler that compiles `main.mind` → `libmindc_mind.so` once; from
then the pure-MIND artifact is byte-identical and self-perpetuating
(the v0.6.1 fixed-point). The backend migration rides the *same*
fixed-point — `main.mind` already contains the MLIR-text emitter and
already produces byte-identical output. There is no new paradox; G2
is wiring, not new logic.

## 4. Risk register (top 5)

1. **(High) Default-flip in G2.3 regresses a fixture not covered by
   the conformance set.** Mitigation: before flipping, run the
   pure-MIND path over *every* `.mind` file in the repo (std/,
   examples/, tests/) and diff MLIR text against the Rust path. Flip
   only at 100% match.
2. **(Med) dlopen ABI drift between the Rust orchestrator and
   `libmindc_mind.so`.** Mitigation: the std-surface cdylib path
   already exercises this boundary (std.vec etc.); reuse its calling
   convention + the existing `blas_smoke`-style symbol-presence gate.
3. **(Med) Removing melior/inkwell breaks an experimental test.**
   Mitigation: grep for `--features mlir_backend|llvm` in all
   workflows + tests first; there are none, but verify.
4. **(Low) Bench-gate regression from the dlopen indirection.**
   Mitigation: the IR→MLIR-text step is a tiny fraction of total
   build time (subprocess `llc` dominates); measure, expect noise.
5. **(Low) The differential-oracle feature bit-rots.** Mitigation:
   keep it in the CI matrix so it compiles every push even after the
   default flips.

## 5. v1.0.0 gate (post Phase I)

After G1 (drop dead deps) + G2 (pure-MIND frontend default), the
remaining v1.0.0 gates from task #278:
- Memory-safety model implemented (#263 — RFC 0010 §3 region/GenRef) — **the largest remaining piece**
- Downstream consumers migrated off the Rust crate (#266)
- #57 cross-arch CUDA/ARM/RISC-V green or documented hardware-blocked
- std surface ≥ {vec,string,map,io,fs,net,process,json,regex,toml,async} — async (RFC 0011) is the gap

Sequence: G1 (now) → memory-safety #263 (multi-week) → G2 (weeks) →
RFC 0011 async impl (weeks) → #266 + cross-arch doc → v1.0.0.

## 6. First concrete step (ship next)

**G1: delete the vestigial melior + inkwell deps.** Edit `Cargo.toml`:
remove the `melior` and `inkwell` dependency lines + the
`mlir_backend = ["melior"]` and `llvm = ["inkwell"]` feature rows.
Run `cargo build --no-default-features` + the full feature matrix CI
(`std-surface`, `cross-module-imports`, `mlir-build`, `autodiff`,
`mlir-lowering`, `cpu-buffers`) to confirm nothing referenced them.
Differential test: the bootstrap fixed-point oracle hash is unchanged
(these crates never touched codegen). This is a ~10-line diff, one CI
run, zero behaviour change — and it closes the literal RFC 0010 §5
"Phase I delete mlir-sys + inkwell" line item, with the honest caveat
(documented above) that it does NOT by itself achieve Rust
independence — G2 does.

## 7. Honest timeline

- **G1**: one session (this is mechanical).
- **G2 (pure-MIND frontend as default)**: 2–4 weeks — mostly the
  exhaustive byte-identity differential sweep + the dlopen wiring.
- **Memory safety #263**: 4–8 weeks (region inference + GenRef + the
  diagnostic surface).
- **RFC 0011 async impl**: 4–8 weeks.
- **v1.0.0**: realistically Q4 2026 / Q1 2027, gated on memory safety
  + async, not on the (now-known-trivial) dead-dep removal.

The headline correction for any future reader: **the credibility-
ladder rung-3 graduation (toolchain self-hosts) shipped in v0.7.0.
"Full Rust independence" was always a smaller delta than RFC 0010 §5
implied, because the heavy MLIR/LLVM Rust FFI it assumed never
existed — mindc emits text and shells to the C++ backend, and the
pure-MIND compiler already mirrors the text emitter byte-identically.**
