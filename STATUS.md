# MIND Compiler Status

> **Last Updated:** 2026-07-21
> **Version:** 0.10.1

## Overview

MIND is a deterministic AI compiler and statically-typed tensor programming language designed for certified AI systems in regulated industries. The compiler self-hosts: the pure-MIND front-end reproduces (a) the canonical `mic@3` binary IR of its own source byte-for-byte, (b) the `mic@1` IR-text bootstrap fixed point, and (c) the NATIVE x86-64/ELF of the entire seeded module (1 055 777 B) byte-identically against the Rust reference — the native-ELF self-host fixed point is closed (2026-06-25). The native-ELF emit path — implemented in the pure-MIND self-host front-end (`examples/mindc_mind/main.mind`) — is the normative self-host target; MLIR-text is a downstream-interchange backend. v0.7.0 was the credibility-ladder rung 3 graduation marker: Mindcraft fully shipped (RFC 0007), RFC 0008 KEYSTONE (cargo retired from the pure-MIND compile loop), RFC 0010 extern "C" foundations (Phases A/B/C shipped), 13 stdlib modules, `mindc doc`, and standalone binary releases. v0.10.0 adds full enum/payload lowering, integer-determinism cluster, tensor-tier-1, and the native-ELF self-host fixed-point closure.

## Feature Status

| Feature Group | Status | Version | Canonical Ref |
|---------------|--------|---------|---------------|
| Core language (lexer, parser, type system, shape inference) | ✅ Complete | v0.2.x | [`spec/v1.0/language.md`](https://github.com/star-ga/mind-spec/blob/main/spec/v1.0/language.md) |
| Core IR + verifier | ✅ Complete | v0.2.x | [`docs/ir.md`](docs/ir.md) |
| Autodiff engine (reverse-mode, Core v1 tensor ops, single-output `main`; non-Core-v1 ops error rather than emit a silent zero gradient) | ✅ Complete | v0.2.x | [`docs/autodiff.md`](docs/autodiff.md) |
| MLIR lowering (arith, tensor, linalg, func, scf, vector dialects) | ✅ Complete | v0.2.x | [`docs/ir-mlir.md`](docs/ir-mlir.md) |
| CPU execution | ✅ Complete | v0.2.x | [`docs/gpu.md`](docs/gpu.md) |
| GPU / multi-backend (CUDA, ROCm, Metal, WebGPU, WebNN) | ⏳ Not implemented — CPU-only today; `mindc` returns a structured "no backend for target gpu" error. Native deterministic GPU backend (NVIDIA SASS / AMD) roadmapped. | — | [`docs/gpu.md`](docs/gpu.md) |
| Systems programming (enum, struct, while, bitwise, u32) | ✅ Complete | v0.5.0 | [`docs/roadmap.md`](docs/roadmap.md) |
| Library output + C ABI (`--emit-shared`, Mind.toml exports) | ✅ Complete | v0.2.6–v0.2.11 | [`docs/roadmap.md`](docs/roadmap.md) |
| Language profiles (`default` / `systems` / `embedded`) | ✅ Complete | v0.2.8 | [`docs/roadmap.md`](docs/roadmap.md) |
| Match expressions, `&expr`, REAP MoE, sparse tensors | ✅ Complete | v0.2.11 | [`docs/roadmap.md`](docs/roadmap.md) |
| Pure-MIND standard library RFC 0005 (std.vec/string/map/io) | ✅ Complete | v0.4.0–v0.4.4 | [`docs/rfcs/0005-pure-mind-stdlib.md`](docs/rfcs/) |
| Self-hosted compiler — **native-ELF fixed-point closed** (mic@3 binary-IR flip + mic@1 text fixed point + native x86-64/ELF of full seeded module 1 055 777 B, byte-identical to Rust reference; front-end byte-exact on 66/66 gap corpus; CI-gated; what remains: ir_trace_hash PT\_NOTE wiring + Rust src/native removal) | ✅ Complete | v0.10.0 | `examples/mindc_mind/` |
| mind-blas (RFC 0006) Track A + Track B inc 1–4 | ✅ Complete | v0.6.3–v0.6.7 | [`docs/rfcs/0006-mind-blas.md`](docs/rfcs/0006-mind-blas.md) |
| **Mindcraft RFC 0007 — all 6 phases + MINDCRAFT-001** | ✅ fmt/lint/check behavior shipped; ⚠️ self-hosted-front-end + `.mind`-rule-file + rule-hash claims in the RFC do not match the implementation — see RFC §0 | **v0.6.8** | [`docs/rfcs/0007-mindcraft.md`](docs/rfcs/0007-mindcraft.md) |
| **RFC 0008 — all 7 phases shipped (`mindc build` + `mindc test` + KEYSTONE)** | ✅ **7/7 phases** | **v0.7.0** | [`docs/rfcs/0008-mindc-build.md`](docs/rfcs/0008-mindc-build.md) |
| Rust edition | ✅ 2024 | v0.6.8 | `Cargo.toml` |
| Windows-MSVC SIMD port (RFC 0006 #225) | ✅ Complete | v0.6.8 | `runtime-support/mind_intrinsics.c` |
| **RFC 0010 extern "C" + SysV/Win64 ABI (Phases A/B/C shipped; E/F scaffolded)** | ✅ **A/B/C** + scaffold E/F | **v0.7.0** | [`docs/rfcs/0010-memory-safety-and-c-abi.md`](docs/rfcs/0010-memory-safety-and-c-abi.md) |
| **13 stdlib modules** (vec/string/map/io/blas/toml/json/regex/net/fs/process/mlir/llvm) | ✅ Complete | **v0.7.0** | `std/` |
| **`mindc doc`** — rustdoc-style HTML documentation generator | ✅ Phase 1 shipped | **v0.7.0** | `src/doc/` |
| **Standalone binary release pipeline** (linux-musl + macos-universal + windows-msvc) | ✅ Complete | **v0.7.0** | `.github/workflows/release.yml` |
| RFCs 0009/0011 specifications | ✅ Drafted | **v0.7.0** | `docs/rfcs/` |

## Native-ELF backend — Phase-C increments (unreleased, post-0.10.1)

The pure-MIND native-ELF backend (`examples/mindc_mind/main.mind`) is being
extended toward a full-surface, MLIR/LLVM-free code path (INDEPENDENCE_ROADMAP
Phase C). These are **increments on the native backend, not full-language
Rust-independence** — the self-compile itself still runs on the integer/
control-flow subset, and the MLIR/LLVM path still carries float/tensor/GPU
codegen. Landed 2026-07-21 (`1372c4c`):

| Rung | Status | Gate |
|------|--------|------|
| C1-remainder — scalar `f32` tier (`addss`/`subss`/`mulss`/`divss`, `cvtss2sd`/`cvtsd2ss`, `cvttss2si`, `movss`), zero MLIR/LLVM | ⏳ landed (CPU-as-oracle; f64 canary stays byte-identical) | `self_host_native_scalar_f32_smoke.py` |
| C2 — narrow-int store/load (`__mind_{store,load}_{i8,i16,i32}`, truncating stores / zero-extend loads) | ⏳ store/load landed; wrap-around arithmetic + unsigned types still open | `self_host_native_narrowint_smoke.py` |
| C3 — division / shift / signed compare (`idiv`+`cqo` with zero & INT_MIN/-1 guards, `sar`/`shl`, all 6 signed `setcc`) | ✅ complete (+16 edge tests) | `div_shift_cmp_edge_smoke.py` |
| B1 — first pure-MIND type-checker slice (E2004 i64→i32 implicit narrowing) | ⏳ 1 rule of ~5,100 LOC; byte-for-byte vs Rust oracle | `self_host_tc_narrowing_smoke.py` |

Keystone stays byte-identical (6/6, 798944 B) and the native-ELF self-host loop
is byte-identical at the new fixed point. Detail: [`docs/INDEPENDENCE_ROADMAP.md`](docs/INDEPENDENCE_ROADMAP.md) Phase C.

## Mindcraft (RFC 0007) — fmt/check shipped in v0.6.8

`mindc fmt` and `mindc check` are first-party subcommands in the `mindc` binary. No external dependencies. There is no standalone `mindc lint` subcommand — lint runs embedded inside `mindc check` (`--no-fmt --no-typecheck` to isolate it); see [`docs/rfcs/0007-mindcraft.md`](docs/rfcs/0007-mindcraft.md) §0 for this and the self-hosted-front-end / `.mind`-rule-file / rule-hash corrections.

| Phase | Description | Commit |
|-------|-------------|--------|
| 1 | `MindcraftConfig` manifest types in `Mind.toml` | `6526029` |
| 2A | `mindc fmt` (`--check` / `--diff` / `--stdin` / `--fix`) | `6e36fa3` |
| 3 | Lint rule infrastructure + `RuleRegistry` + glob overrides | `ccbaba9` |
| 4 | 5 named lint rules (`q16_overflow`, `unused_import`, `naming_convention`, `shadowing`, `trailing_whitespace`) | `5ff5367` |
| 5 | `mindc check` project driver — VCS-aware, JSON + LSP reporters | `1442a31` |
| 6 | `--fix` pipeline + CI integration + `.github/workflows/mindcraft.yml` | `15f9960` |
| MINDCRAFT-001 | `pub` keyword preserved through AST and formatter | `1d988bd` |

Bench-gate +7% cap held: `mindc fmt vec.mind` ~46 us, `mindc fmt mindc_mind/main.mind` ~1.8 ms, `mindc check std/` (98 files) ~23 ms.

Spec: [`docs/rfcs/0007-mindcraft.md`](docs/rfcs/0007-mindcraft.md).

## RFC 0008 (`mindc build` / `mindc test`) — All 7 Phases Shipped

| Phase | Description | Commit | Status |
|-------|-------------|--------|--------|
| Spec | 850-line RFC | `20c3c1c` | ✅ Complete |
| A | `mindc build` single-crate orchestrator | `d5bb605` | ✅ Shipped |
| B | `mindc test` discovery + parallel runner | `9c8fb6f` | ✅ Shipped |
| C | Workspace support, topo sort, cycle detection | `267a9a6` | ✅ Shipped |
| D | Path deps + content-hash drift detection | `7117b2a` | ✅ Shipped |
| E | Git deps + `Mind.lock` mandatory enforcement | `f27789f` | ✅ Shipped |
| F | Incremental compilation cache (cold ~188 ms, warm ~3 ms) | `01fc039` | ✅ Shipped |
| G | **KEYSTONE** — `mindc build` bootstraps mind itself | `faa6027` | ✅ **Shipped** |

**Cargo retirement claim (Phase G)**: `mindc build` produces
`libmindc_mind.so` byte-identical to the v0.6.1 fixed-point oracle,
driven entirely by the pure-MIND build orchestrator. Cargo is no longer
load-bearing for the pure-MIND compile loop. The Rust crate hosts
`mindc` until RFC 0010 lands a pure-MIND libMLIR FFI.

Spec: [`docs/rfcs/0008-mindc-build.md`](docs/rfcs/0008-mindc-build.md).

## Roadmap Phases

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Core lexer and parser | ✅ Complete |
| Phase 2 | Type system and inference | ✅ Complete |
| Phase 3 | Shape inference engine | ✅ Complete |
| Phase 4 | Core IR design | ✅ Complete |
| Phase 5 | IR verification | ✅ Complete |
| Phase 6 | Autodiff engine | ✅ Complete |
| Phase 7 | MLIR lowering | ✅ Complete |
| Phase 8 | MLIR optimization | ✅ Complete |
| Phase 9 | CPU runtime integration | ✅ Complete |
| Phase 10.5 | Systems programming foundation | ✅ Complete |
| Phase 10.6 | Library output and C ABI (mindc 0.2.6 → 0.2.11) | ✅ Complete |
| Phase 10.7 | Match expressions + `&expr` reference-taking | ✅ Complete |
| RFC 0005 | Pure-MIND standard library | ✅ Complete |
| RFC 0006 | mind-blas dense-vector surface | ✅ Complete (Track A + B) |
| RFC 0007 | Mindcraft: `mindc fmt` / `mindc check` (lint embedded in `check`, no standalone `lint` subcommand) | ✅ fmt/check shipped, doc caveats — see RFC §0 |
| RFC 0008 — all 7 phases | `mindc build` + `mindc test` + workspace + deps + cache + KEYSTONE | ✅ **7/7 shipped** |
| RFC 0010 — Phases A/B/C | extern "C" + SysV + Win64 ABI + `#[repr(C)]` | ✅ **Shipped** |
| Phase 13 | BCI / Neuroscience runtime | 📋 Roadmap |

See [docs/roadmap.md](docs/roadmap.md) for full phase descriptions.

## Documentation Status

| Document | Status |
|----------|--------|
| [Architecture](docs/architecture.md) | ✅ Complete |
| [Type System](docs/type-system.md) | ✅ Complete |
| [Shape Inference](docs/shapes.md) | ✅ Complete |
| [Autodiff](docs/autodiff.md) | ✅ Complete |
| [IR Specification](docs/ir.md) | ✅ Complete |
| [MLIR Lowering](docs/mlir-lowering.md) | ✅ Complete |
| [GPU Support](docs/gpu.md) | ✅ Contract doc (open-core ships no GPU impl) |
| [FFI/Runtime](docs/ffi-runtime.md) | ✅ Complete |
| [Error Catalog](docs/errors.md) | ✅ Complete |
| [Benchmarks](docs/benchmarks.md) | ✅ Complete |
| [Roadmap](docs/roadmap.md) | ✅ Complete |
| [Optimization Frontier](docs/optimization-frontier.md) | ✅ Roadmap — Pareto-optimal deterministic compiler |
| [RUNS Burndown Roadmap](docs/runs-burndown-roadmap.md) | ✅ Roadmap — 62%→100% executable-subset path |
| [RFC 0007 Mindcraft](docs/rfcs/0007-mindcraft.md) | ✅ Normative |
| [RFC 0008 mindc build](docs/rfcs/0008-mindc-build.md) | ✅ Spec complete — impl 7/7 |
| [RFC 0010 memory safety + C ABI](docs/rfcs/0010-memory-safety-and-c-abi.md) | ✅ Phases A/B/C shipped; E/F scaffolded |

## CI Status

Live status is rendered by the dynamic badges at the top of
[`README.md`](README.md). They reflect the most recent run of `cargo
build`, `cargo test`, `cargo clippy`, `cargo fmt --check`, `cargo deny
check`, and the link checker against `main`.

(This section deliberately does not pin a static "all green" claim;
those go stale on the next flake.)

## Known Issues

See [GitHub Issues](https://github.com/star-ga/mind/issues) for tracked bugs and enhancements.

### Code TODOs

There are TODO items in `src/exec/cpu.rs`, `src/eval/mod.rs`, and
`src/exec/conv.rs` relating to runtime stub implementations. These are
**intentional design markers** indicating functionality provided by the
proprietary `mind-runtime` backend. They are not missing features but
architectural boundary markers.

## IR Canon

The canonical compiled-artifact IR is `IRModule` serialized as **mic@1** text
(`ir::save`) — RFC-0001 deterministic, the stable public contract documented in
`docs/ir-stability.md`. The binary shipping format is **mic@3**: a binary encoding
of the full `IRModule` (all 37 `Instr` variants including control flow, functions,
and SIMD) with an embedded mic@2.1 MAP epilogue carrying the `evidence_chain.*`
and `verify.*` namespaces.

The `trace_hash` is anchored on the canonical **mic@3** binary bytes (re-anchored
from mic@1 text on 2026-05-31 after a collision audit — mic@1 text can drop
function-body semantics; mic@3 binary commits the full `IRModule`; supersedes RFC
0016 GAP-1 mic@1-text rule).

| Layer | Format | Status |
|---|---|---|
| mic@1 text (human-readable, round-trip stable) | `ir::save` / `ir::load` | ✅ Stable — documented public contract |
| mic@3 binary `IRModule` codec | `src/ir/compact/v3/` | ✅ Shipped (RFC 0021 step 1, mind@5c29f0d) |
| Evidence MAP epilogue on mic@3 | `src/ir/compact/v3/evidence.rs` | ✅ Shipped (RFC 0021 step 2, mind@c64bd0b) |
| `mindc --emit-mic3` / `--emit-evidence` CLI | `src/bin/mindc.rs` | ✅ Shipped (RFC 0021 step 3, mind@7fc10d2) |
| `trace_hash` anchored on canonical mic@3 binary bytes (re-anchored 2026-05-31, supersedes RFC 0016 GAP-1 mic@1-text rule) | `src/ir/evidence.rs::ir_trace_hash` | ✅ Shipped |
| RFC 0016 Phase A evidence emit (inert, unsigned) | `src/ir/compact/v2/evidence.rs` | ✅ Shipped (mind@e7c8c28 / cadca87) |
| RFC 0016 Phase B verifier core | `src/ir/compact/v2/evidence.rs::verify_evidence_chain` | ✅ Shipped (mind@cadca87) |
| `mindc verify` CLI subcommand (RFC 0017) | pending | 🚧 In progress (RFC 0021 step 4) |
| mic@2.x → `mind-model@2` demotion | cross-repo migration | 🚧 In progress (RFC 0021 step 5) |
| Cross-substrate CI oracle gate | `tests/cross_substrate_identity/` + `oracle.rs` | 🚧 In progress (RFC 0021 step 6, #307) |
| Ed25519 signing on release artifacts (RFC 0016 Phase C) | release CI runner | ⏳ Pending |
| RFC 0016 Phase D — determinism-default (#289) | pending `Tensor<f32>` gate | ⏳ Pending |
| RFC 0016 Phase E — agent + governance links (RFC 0019) | pending scheduler integration | ⏳ Pending |
| #306 std byte-store `__mind_store_i8` migration (string/sha256/toml) | `45049a9`; verified via SHA-256 FIPS 3/3 + std-surface string/toml suites (real `--emit-shared` ELF) | ✅ Landed |
| #306 keystone byte-identity re-bless | oracle is a post-migration ELF; `mindc build --emit=cdylib` emits a stub in the open toolchain | 🚧 Pending (ELF-capable cdylib build) |

### Evidence chain

A compiled MIND artifact carries a `evidence_chain.*` MAP block that attests:
- **`trace_hash`**: SHA-256 (FIPS 180-4) of the canonical mic@3 binary bytes — the
  artifact's deterministic production hash, reproducible by any verifier. (Re-anchored
  from mic@1 text on 2026-05-31; mic@3 binary commits the full `IRModule`.)
- **`substrate`**: the RFC 0014 canonical lowering-tier ID (`x86_avx2`, `arm_neon`, …).
- **`parent`**: `trace_hash` of the predecessor step in the transformation DAG.
- **`determinism`**: `"deterministic"` or `"nondeterministic"` (never a bare artifact).
- **`toolchain`**: `mindc` version that produced the artifact.

For a Q16.16 graph, two artifacts compiled for different substrates from the same
source have equal `trace_hash` values — cross-substrate bit-identity made inspectable
in-band (RFC 0015). The `verify.*` MAP block (RFC 0017) carries a signed receipt
from `mindc verify`; the `agent.*` MAP block (RFC 0019) links a runtime agent step's
`ReplayScheduler` trace to the artifact's `trace_hash` (the fold:
`H(mic1_ir_hash || replay_trace_hash)`).

### mic@3 binary codec

`mic@3` is the binary shipping format: the mic@2.1 MAP+Ed25519 container extended
to carry the full `IRModule` data model. Key properties:
- Canonical encoding: sorted map keys, fixed-width length prefixes, f64 via `to_bits`,
  deterministic depth-first pre-order traversal of control-flow regions.
- `save→load→save` fixed-point verified by differential-fuzz test suite.
- Additive alongside mic@1 — mic@1 text output is unchanged; `mic@3` is a new emit
  target (`--emit-mic3`, `--emit-evidence`).
- `trace_hash` in the MAP epilogue is SHA-256 of the canonical mic@3 binary bytes,
  so the same hash appears whether the artifact is inspected via mic@3 binary or
  reproduced from source. (Re-anchored from mic@1 text 2026-05-31.)

### RFC 0021 migration progress

| Step | Description | Status |
|---|---|---|
| 1 | mic@3 binary `IRModule` codec | ✅ Done (mind@5c29f0d) |
| 2 | Evidence MAP epilogue on mic@3 | ✅ Done (mind@c64bd0b) |
| 3 | `mindc --emit-mic3` / `--emit-evidence` CLI flags | ✅ Done (mind@7fc10d2) |
| 4 | `mindc verify --evidence` CLI subcommand | 🚧 In progress |
| 5 | Demote mic@2.x → `mind-model@2` (cross-repo migration) | 🚧 In progress |
| 6 | `oracle.rs` + CI bit-identity gate (#307) | 🚧 In progress |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## License

Dual-licensed under Apache 2.0 and Commercial. See [LICENSE](LICENSE) and [LICENSE-COMMERCIAL](LICENSE-COMMERCIAL).
