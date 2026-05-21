# MIND Compiler Status

> **Last Updated:** 2026-05-21
> **Version:** 0.7.0

## Overview

MIND is a deterministic AI compiler and statically-typed tensor programming language designed for certified AI systems in regulated industries. The compiler self-hosts: the pure-MIND `libmindc_mind.so` compiles its own source byte-identically to the Rust reference implementation (bootstrap fixed-point, v0.6.1). v0.7.0 is the credibility-ladder rung 3 graduation marker: Mindcraft fully shipped (RFC 0007), RFC 0008 KEYSTONE (cargo retired from the pure-MIND compile loop), RFC 0010 extern "C" foundations (Phases A/B/C shipped), 13 stdlib modules, `mindc doc`, and standalone binary releases.

## Feature Status

| Feature Group | Status | Version | Canonical Ref |
|---------------|--------|---------|---------------|
| Core language (lexer, parser, type system, shape inference) | ✅ Complete | v0.2.x | [`spec/v1.0/language.md`](https://github.com/star-ga/mind-spec/blob/main/spec/v1.0/language.md) |
| Core IR + verifier | ✅ Complete | v0.2.x | [`docs/ir.md`](docs/ir.md) |
| Autodiff engine (reverse-mode, all Core v1 ops) | ✅ Complete | v0.2.x | [`docs/autodiff.md`](docs/autodiff.md) |
| MLIR lowering (arith, tensor, linalg, func, scf, vector dialects) | ✅ Complete | v0.2.x | [`docs/ir-mlir.md`](docs/ir-mlir.md) |
| CPU execution | ✅ Complete | v0.2.x | [`docs/gpu.md`](docs/gpu.md) |
| GPU / multi-backend (CUDA, ROCm, Metal, WebGPU, WebNN) | ✅ Complete | enterprise | [`docs/gpu.md`](docs/gpu.md) |
| Systems programming (enum, struct, while, bitwise, u32) | ✅ Complete | v0.5.0 | [`docs/roadmap.md`](docs/roadmap.md) |
| Library output + C ABI (`--emit-shared`, Mind.toml exports) | ✅ Complete | v0.2.6–v0.2.11 | [`docs/roadmap.md`](docs/roadmap.md) |
| Language profiles (`default` / `systems` / `embedded`) | ✅ Complete | v0.2.8 | [`docs/roadmap.md`](docs/roadmap.md) |
| Match expressions, `&expr`, REAP MoE, sparse tensors | ✅ Complete | v0.2.11 | [`docs/roadmap.md`](docs/roadmap.md) |
| Pure-MIND standard library RFC 0005 (std.vec/string/map/io) | ✅ Complete | v0.4.0–v0.4.4 | [`docs/rfcs/0005-pure-mind-stdlib.md`](docs/rfcs/) |
| Self-hosted compiler — bootstrap fixed-point | ✅ Complete | v0.6.1 | `examples/mindc_mind/` |
| mind-blas (RFC 0006) Track A + Track B inc 1–4 | ✅ Complete | v0.6.3–v0.6.7 | [`docs/rfcs/0006-mind-blas.md`](docs/rfcs/0006-mind-blas.md) |
| **Mindcraft RFC 0007 — all 6 phases + MINDCRAFT-001** | ✅ **Fully Shipped** | **v0.6.8** | [`docs/rfcs/0007-mindcraft.md`](docs/rfcs/0007-mindcraft.md) |
| **RFC 0008 — all 7 phases shipped (`mindc build` + `mindc test` + KEYSTONE)** | ✅ **7/7 phases** | **v0.7.0** | [`docs/rfcs/0008-mindc-build.md`](docs/rfcs/0008-mindc-build.md) |
| Rust edition | ✅ 2024 | v0.6.8 | `Cargo.toml` |
| Windows-MSVC SIMD port (RFC 0006 #225) | ✅ Complete | v0.6.8 | `runtime-support/mind_intrinsics.c` |
| **RFC 0010 extern "C" + SysV/Win64 ABI (Phases A/B/C shipped; E/F scaffolded)** | ✅ **A/B/C** + scaffold E/F | **v0.7.0** | [`docs/rfcs/0010-memory-safety-and-c-abi.md`](docs/rfcs/0010-memory-safety-and-c-abi.md) |
| **13 stdlib modules** (vec/string/map/io/blas/toml/json/regex/net/fs/process/mlir/llvm) | ✅ Complete | **v0.7.0** | `std/` |
| **`mindc doc`** — rustdoc-style HTML documentation generator | ✅ Phase 1 shipped | **v0.7.0** | `src/doc/` |
| **Standalone binary release pipeline** (linux-musl + macos-universal + windows-msvc) | ✅ Complete | **v0.7.0** | `.github/workflows/release-binary.yml` |
| RFCs 0009/0011 specifications | ✅ Drafted | **v0.7.0** | `docs/rfcs/` |

## Mindcraft (RFC 0007) — Fully Shipped in v0.6.8

`mindc fmt`, `mindc lint`, and `mindc check` are first-party subcommands in the `mindc` binary. No external dependencies.

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
| RFC 0007 | Mindcraft: `mindc fmt` / `mindc lint` / `mindc check` | ✅ Fully Shipped |
| RFC 0008 — all 7 phases | `mindc build` + `mindc test` + workspace + deps + cache + KEYSTONE | ✅ **7/7 shipped** |
| RFC 0010 — Phases A/B/C | extern "C" + SysV + Win64 ABI + `#[repr(C)]` | ✅ **Shipped** |
| Phase 13 | BCI / Neuroscience runtime | ✅ Complete |

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
| [GPU Support](docs/gpu.md) | ✅ Complete |
| [FFI/Runtime](docs/ffi-runtime.md) | ✅ Complete |
| [Error Catalog](docs/errors.md) | ✅ Complete |
| [Benchmarks](docs/benchmarks.md) | ✅ Complete |
| [Roadmap](docs/roadmap.md) | ✅ Complete |
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

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## License

Dual-licensed under Apache 2.0 and Commercial. See [LICENSE](LICENSE) and [LICENSE-COMMERCIAL](LICENSE-COMMERCIAL).
