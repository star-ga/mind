<p align="center">
  <img src="assets/logo/mind-logo.svg" alt="MIND logo" width="512" />
</p>

# MIND — Machine Intelligence Native Design

[![CI](https://github.com/star-ga/mind/actions/workflows/ci.yml/badge.svg)](https://github.com/star-ga/mind/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue?style=flat-square)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.10.1-blue?style=flat-square)](CHANGELOG.md)
[![Cross-substrate](https://img.shields.io/badge/deterministic-byte--identical-brightgreen?style=flat-square)](docs/roadmap.md)

## Install

```sh
curl -sSL https://mindlang.dev/install.sh | sh
```

Pre-built binaries for Linux, macOS, and Windows — no Rust toolchain required.
See [`docs/install.md`](docs/install.md) for all install options including manual download, checksum verification, and build from source.

## Overview

**MIND** is a deterministic compiler: its deterministic-integer and Q16.16 fixed-point lowering produces output that is bit-identical across CPU substrates (x86-`avx2` == ARM-`neon`), with a tamper-evident evidence chain embedded in the artifact itself (opt-in Ed25519 / ML-DSA cryptographic signing). Scalar IEEE-754 `float64` now compiles on the strict deterministic path — run-to-run bit-identical, and verified byte-identical across x86_64 (`avx2`) and ARM64 (`neon`) on real hardware (2026-07-05). Vector-reduction, transcendental, and GPU float determinism are on the roadmap.

It is also a Rust-first language and runtime for building intelligent systems with auditable foundations. It blends declarative tensor algebra, static shape inference, automatic differentiation, and MLIR/LLVM lowering in a compact toolchain that scales from research prototypes to production.

The compiler produces deterministic binaries that execute inside the [Cognitive Kernel](https://mindlang.dev/docs/cognitive-kernel), MIND's microkernel runtime architecture with Control, Memory, and Verification planes.

**MIND self-hosts the full native-ELF fixed point.** The pure-MIND front-end (a) reproduces the
canonical `mic@3` binary IR of its own source byte-for-byte (the layer the evidence chain's `trace_hash`
anchors on), (b) reproduces the `mic@1` IR-text bootstrap fixed point, and (c) emits the NATIVE x86-64/ELF
of the entire seeded module (21 stdlib modules + main.mind, 1 055 777 B) byte-identically against the Rust
reference — the native-ELF self-host fixed point is closed. This is the core of Rust-independence.

**Backend architecture:** the pure-MIND NATIVE-ELF backend (`examples/mindc_mind/main.mind`,
`nb_write_elf`) emits a static x86-64 ELF directly — zero LLVM — and is the normative
Rust-independence target. The compiler **self-hosts** on it: run as a native ELF using only
`read`/`write`/`exit`, a pure-MIND compiler reproduces its own compiler binary byte-identically
three stages deep, with Rust **and** LLVM out of the loop (integer/control-flow subset; gated by
`examples/mindc_mind/self_host_loop_smoke.py`). The Rust `src/native` backend it replaced was
**deleted 2026-07-01** (recoverable in git history). MLIR-text is a downstream-interchange and
exotic-chip-reach backend — demoted from the self-host path, still load-bearing for float/tensor/GPU.
"Target any chip" is implemented via a pluggable backend trait plus commercial backends in the private
`mind-runtime`.

What remains toward full Rust-independence: (1) **mic@3 canonicality** — the pure-MIND `emit_mic3`
self-computed `trace_hash` still diverges from the Rust `--emit-mic3` oracle on the pruned-combined IR,
so byte-parity is needed for a compiler-independent anchor; (2) the **full-surface native backend**
(floats/tensors/GPU) that would let MLIR/LLVM be dropped. See
[`docs/INDEPENDENCE_ROADMAP.md`](docs/INDEPENDENCE_ROADMAP.md) and [`docs/roadmap.md`](docs/roadmap.md).

## Open-core vs proprietary runtime

This repository contains the open-core stack: the MIND language, type system, compiler front-end, IR, and MLIR lowering passes. The open `src/exec/cpu.rs` ships a **reference CPU interpreter** — naive, unoptimized implementations that produce correct results for learning, prototyping, and small workloads (gated behind the `cpu-exec` feature). Production-grade runtime backends for CPU (SIMD, tiled matmul), GPU, and accelerators live in the private [`mind-runtime`](https://github.com/star-ga/mind-runtime) repository. A few operations the open interpreter does not cover (for example Conv2D in `src/exec/conv.rs`) return `ExecError::Unsupported`; these are architectural boundary markers that the proprietary backend fulfills.

## System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Rust | 1.82+ stable | Latest stable |
| OS | Linux, macOS, Windows | Linux x86_64 |
| Memory | 4 GB RAM | 8 GB RAM |
| Disk | 500 MB | 2 GB (with MLIR) |

### Platform Support

| Platform | Status | CI Tested |
|----------|--------|-----------|
| Linux x86_64 | Fully supported | Yes |
| macOS x86_64 | Fully supported | Yes |
| macOS ARM64 (Apple Silicon) | Fully supported | Yes |
| Windows x86_64 | Fully supported | Yes |

### Optional Dependencies

- **LLVM 17+**: Required for `mlir-lowering` feature
- **MLIR tools**: Required for `mlir-exec` feature
- **C compiler**: Required for FFI examples

## Quick Start

```bash
git clone https://github.com/star-ga/mind.git
cd mind
cargo run -- eval "let x: Tensor[f32,(2,3)] = 0; x + 1"
```

Explore the full language tour and runtime guides in [`/docs`](docs/README.md).

## CLI / Compiler Driver

The `mindc` binary provides a deterministic source→IR→MLIR pipeline suitable
for demos and snapshot tests:

```bash
# Basic compilation to IR
cargo run --bin mindc -- examples/hello_tensor.mind --emit-ir

# Autodiff gradient IR
cargo run --bin mindc -- examples/hello_tensor.mind --func main --autodiff --emit-grad-ir

# MLIR output (requires mlir-lowering feature)
cargo run --features "mlir-lowering autodiff" --bin mindc -- examples/hello_tensor.mind --func main --autodiff --emit-mlir

# Verify without emitting output
cargo run --bin mindc -- examples/hello_tensor.mind --verify-only

# Build object file (requires aot feature)
cargo run --features aot --bin mindc -- examples/hello_tensor.mind --emit-obj output.o
```

### Project Commands

```bash
# Build a MIND project (reads Mind.toml)
mindc build
mindc build --release                # optimized (-O3 -flto)
mindc build --release --target cuda  # CUDA backend

# Build and run
mindc run
mindc run --target cuda              # run with CUDA backend

# Test (discovers #[test]-annotated functions, runs in parallel)
mindc test
mindc test --filter kv_cache         # run matching suites
mindc test --target cuda             # GPU tests

# Benchmark (discovers bench/*.mind, builds with --release)
mindc bench
mindc bench --target cuda            # GPU benchmarks
mindc bench --filter throughput      # specific benchmark
```

### Mindcraft Source Toolchain (RFC 0007 — fmt/check shipped in v0.6.8; see note)

`mindc fmt` and `mindc check` are first-party source-quality subcommands
shipping in the same `mindc` binary. No external dependencies. **There is no
standalone `mindc lint` subcommand** — `mindc --help` does not list one, and
`mindc lint` errors `failed to read lint: No such file or directory` (clap
parses `lint` as the `[FILE]` positional). Lint runs embedded inside
`mindc check`; use `--no-fmt --no-typecheck` to isolate it.

```bash
# Format: rewrite .mind files to canonical form (idempotent, deterministic)
mindc fmt src/                       # rewrite in place
mindc fmt --check src/               # CI gate: exit non-zero if any file would change
mindc fmt --diff src/                # show unified diff without writing
mindc fmt --stdin < file.mind        # read from stdin

# Check: fmt idempotence + lint + typecheck in one pass (lint has no separate subcommand)
mindc check                          # full project check (VCS-aware, only dirty files)
mindc check --no-fmt --no-typecheck  # lint only
mindc check --fix                    # auto-fix all fixable lint suggestions
mindc check --reporter=json          # machine-readable output
```

Named lint rules in v0.6.8 (all implemented in Rust, `src/lint/rules/*.rs` —
not `.mind` files, see `docs/rfcs/0007-mindcraft.md` §0): `q16_overflow`,
`unused_import`, `naming_convention`, `shadowing`, `trailing_whitespace`.

CI integration ships as a reusable GitHub Actions workflow at
`.github/workflows/mindcraft.yml`. Spec: [`docs/rfcs/0007-mindcraft.md`](docs/rfcs/0007-mindcraft.md).

### Feature Flags

By default, `mindc` builds with minimal features for fast compilation. Enable
additional features as needed:

```bash
cargo build --features aot        # AOT compilation (--emit-obj, project builds)
cargo build --features autodiff   # Autodiff support
cargo build --features full       # All features
```

MLIR emission requires the `mlir-lowering` feature. Reverse-mode autodiff
covers the Core v1 tensor ops (see [`docs/autodiff.md`](docs/autodiff.md)) and
differentiates a single-output `main` entry point; non-Core-v1 ops (functions,
control flow, std-surface, modulo, bitwise/shift) are non-differentiable and
return a structured error rather than a silent zero gradient.

## Pure-MIND standard library (RFC 0005)

`std/vec.mind`, `std/string.mind`, `std/map.mind`, `std/io.mind` —
four small collections + an I/O surface, written entirely in MIND on
top of the seven i64-ABI intrinsics
(`__mind_alloc` / `__mind_free` / `__mind_realloc` /
`__mind_load_i64` / `__mind_store_i64` / `__mind_read` /
`__mind_write`). No built-in pointer type; every aggregate is an i64
base-address into the heap.

```mind
use std.vec
use std.io

fn main() {
    let v = vec_new()
    let v = vec_push(v, 42)
    let v = vec_push(v, 99)
    let n = print_bytes(vec_addr(v), 16)
}
```

Two feature flags gate the surface (default build is byte-identical
without them — the parser/typecheck/IR hot path is untouched):

```bash
# Compile the std-surface intrinsics + std/*.mind modules.
cargo build --features std-surface

# Add cross-module symbol resolution for `use std.foo`.
cargo build --features std-surface,cross-module-imports
```

Phase B (per-arg signature matching on imported `pub fn`s) validates
arity + per-arg types against the imported declaration and returns
the declared return type; an `export { ... }`-block donor falls back
to Phase-A loose typing.

## Compilation cache (`libmind::cache`)

Content-addressed caching layer in `src/cache/` keyed by compiler version,
profile tag, source SHA-256, and imports SHA-256. Re-invocations on the
same input bypass parse + typecheck + IR build and return the cached IR
directly. Foundation for the sub-µs warm-start frontend latency target.

```rust
use libmind::cache::{CompilationCache, CacheKey, ProfileTag};

let mut cache = CompilationCache::in_memory();
let key = CacheKey::new(env!("CARGO_PKG_VERSION"), ProfileTag::Default, source_hash, imports_hash);
if let Some(entry) = cache.lookup(&key) {
    return entry.ir_bytes;
}
```

See [`tests/cache_smoke.rs`](tests/) and the 17 unit tests under
`src/cache/`.

## Python bridge tooling (`tools/pytorch_bridge/`)

Pure-Python transpiler that lowers PyTorch (via ONNX) and JAX (via XLA
HLO) graphs into MIND source. Pure-Python — no torch / jax import at
module load — so it runs on a CI machine that doesn't have either
framework installed.

```python
from pytorch_bridge import pytorch_to_mind, jax_to_mind

result = pytorch_to_mind("model.onnx", module_name="net")
print(result.module.emit())          # canonical .mind text
print(result.unsupported)             # ops routed to AI-assist proof pass
```

Includes `build_unsat_prompt()` for AI-assisted resolution of UNSAT
typecheck failures. 11 unit tests under `tools/pytorch_bridge/tests/`.

## GPU backend profile

The crate exposes the Core v1 GPU profile and a `--target=gpu` flag in `mindc`.
CPU remains the only implemented target, but the GPU contract (enums, error
model, and `GPUBackend` trait) is treated as stable for downstream runtimes.
Selecting the GPU target returns a structured "no backend available for target gpu" error.
See [`docs/gpu.md`](docs/gpu.md) for the device/target model and current
status.

## Core Concepts

* [Type System](docs/type-system.md) — ranks, shapes, polymorphism, and effect tracking.
* [Shapes](docs/shapes.md) — broadcasting, reductions, and shape-preserving tensor transforms.
* [Autodiff](docs/autodiff.md) — reverse-mode differentiation on the SSA IR.
* [IR core](docs/ir.md) — deterministic IR pipeline with verifier and printer.
* [IR & MLIR](docs/ir-mlir.md) — compiler pipeline from parser to MLIR dialects.

## Applications

### Neuroscience & Brain-Computer Interfaces

MIND's combination of static shape inference, reverse-mode autodiff, ultra-low-latency compilation, and deterministic execution makes it uniquely suited for real-time neural signal processing and brain-computer interface (BCI) applications:

* **Real-time Neural Decoding** — Sub-millisecond inference for invasive BCI systems (Neuralink-style implants, ECoG arrays)
* **Multi-channel Time-Series** — Native tensor operations for Channel × Time × Batch neural data
* **On-device Adaptation** — Gradient-based decoder optimization directly on implanted devices using autodiff
* **Reproducible Research** — Deterministic builds critical for FDA-regulated medical devices and neuroscience studies
* **Edge Deployment** — Deploy to resource-constrained BCI hardware (ARM Cortex-M, RISC-V) with minimal runtime overhead

See [Phase 13 in the roadmap](docs/roadmap.md#phase-13--neuroscience--brain-computer-interfaces) for planned neuroscience-specific standard library modules, data format support, and benchmarks.

### Other Applications

MIND's design also supports machine learning research, embedded AI, and safety-critical intelligent systems requiring auditable execution and reproducible builds.

## Stability & Versioning

MIND Core v1 follows the public contract in mind-spec Core v1. The stability
model, SemVer policy, and CLI guarantees are documented in
[`docs/versioning.md`](docs/versioning.md).

### IR canon (mindc 0.7.x, RFC 0021)

The `IRModule` data shape has two canonical serialisations:

- **`mic@1`** — text form (`libmind::ir::save` / `load`). The documented stable
  contract for the `IRModule` data shape.
- **`mic@3`** — binary form (magic `MIC3`, `src/ir/compact/v3/`). Round-trip
  equivalent to `mic@1`; emit via `mindc --emit-mic3`. Carries the evidence
  MAP epilogue via a `0x4D`-sentinel form (RFC 0021 step 2). The load-bearing
  anchor for the evidence-chain `trace_hash`: `trace_hash = SHA-256(canonical
  mic@3 bytes)` (re-anchored from mic@1 text on 2026-05-31 after a collision
  audit — mic@1 text can drop function-body semantics; mic@3 binary commits the
  full `IRModule`; supersedes the original RFC 0016 GAP-1 mic@1-text rule).

Compile-time evidence-chain attestation ships via `mindc --emit-evidence`
(RFC 0016 Phase A + B, opt-in). `mic@2`/`mic@2.1` are preserved back-compat
lanes pending RFC 0021 step 5 demotion to `mind-model@2`. See
[`docs/ir-stability.md`](docs/ir-stability.md).

## Architecture

* [Runtime & Compiler Architecture](docs/architecture.md)
* [FFI & Runtime Embedding](docs/ffi-runtime.md)

## Testing

The MIND compiler includes a comprehensive test suite with ~1,390 tests across 174 test files covering parsing, type checking, IR generation, MLIR lowering, and execution.

### Running Tests

```bash
# Run all tests
cargo test

# Run tests with output
cargo test -- --nocapture

# Run specific test file
cargo test --test smoke

# Run tests matching a pattern
cargo test tensor

# Run tests with specific features
cargo test --features "mlir-lowering autodiff"
```

### Test Categories

| Category | Files | Description |
|----------|-------|-------------|
| Smoke tests | `smoke.rs` | Quick sanity checks |
| Type system | `type_*.rs`, `typecheck_*.rs` | Type inference and checking |
| Shapes | `shapes*.rs`, `tensor_*.rs` | Shape inference and broadcasting |
| IR/MLIR | `ir_*.rs`, `mlir_*.rs` | IR generation and MLIR lowering |
| Autodiff | `autodiff*.rs`, `*_grad.rs` | Reverse-mode differentiation |
| CLI | `cli_*.rs`, `mindc.rs` | Command-line interface |
| Execution | `exec_*.rs`, `relu_*.rs`, `conv2d_*.rs` | Reference CPU interpreter execution |

### Continuous Integration

All tests run on every pull request via GitHub Actions. See [`.github/workflows/ci.yml`](.github/workflows/ci.yml) for the full CI configuration.

## Benchmarks

The [`/docs/benchmarks.md`](docs/benchmarks.md) report covers baseline compiler/runtime performance, regression tracking, and methodology.

### Compilation Speed

#### Verified Benchmarks (frozen bench-gate baseline, carried through v0.7.1)

*The frontend bench-gate baseline was locked in April 2026 and carries
forward unchanged to v0.7.1; later parser additions ship within the
bench-gate threshold documented at [`.bench-baseline-2026-05-17-phase10-6.txt`](./.bench-baseline-2026-05-17-phase10-6.txt).*


| vs Framework | Compilation Time | MIND Ratio |
|--------------|-----------------|------------|
| **MIND v0.7.1** | **1.8-15.5 µs** | **1× (baseline)** |
| PyTorch 2.10 GPU torch.compile | 99-878 ms | **35,000-176,000× faster** |
| JAX 0.9 cold-start XLA (jax.jit) | 37.5-360.5 ms | **21,200-95,100× faster** |
| Mojo 0.26.1 (mojo build) | 810-829 ms | **135,000-458,000× faster** |

**Scope note:** these ratios compare MIND's **compile-time frontend** (parse + typecheck + IR) against the other frameworks' GPU runtime / cold-start compilation — i.e. different operations, not a runtime-speed comparison. The defensible head-to-head speed comparison is MIND's frontend vs. the prior Rust-combinator frontend (the ~15× speedup below).

| Benchmark | MIND v0.7.1 | Compilations/sec |
|-----------|-------------|------------------|
| scalar_math | **1.77 µs** | 565K cps |
| small_matmul | **2.95 µs** | 339K cps |
| medium_matmul | **2.95 µs** | 339K cps |
| large_matmul | **2.95 µs** | 339K cps |
| tensor_ops | **4.87 µs** | 205K cps |
| reductions | **3.17 µs** | 315K cps |
| reshape_ops | **2.83 µs** | 353K cps |
| medium_mlp | **6.15 µs** | 163K cps |
| large_network | **15.49 µs** | 65K cps |

*Measured via Rust Criterion (100 samples, 95% CI). Environment: Ubuntu 24.04, Ampere-class GPU, CUDA 12.8. See [benchmarks/BENCHMARK_RESULTS.md](benchmarks/BENCHMARK_RESULTS.md) for full methodology.*

### MIC/MAP Format Efficiency

| Format | Tokens | vs JSON | Parse Speed | Annual Cost (1M IRs) |
|--------|--------|---------|-------------|----------------------|
| JSON | 278 | baseline | 5.31 us | $8,340 |
| TOML | 151 | 1.8x | 137.06 us | $4,530 |
| TOON | 67 | 4.1x | 2.67 us | $2,010 |
| mic@1 | 52 | 5.3x | 2.26 us | $1,560 |
| **mic@2** | **27** | **10.3x** | **—** | **$810** |

> **mic@3** (binary IRModule): 90 bytes &mdash; 12.4x smaller than JSON's 1,117 bytes.

| Protocol | Tokens | vs JSON-RPC |
|----------|--------|-------------|
| JSON-RPC | 251 | baseline |
| **MAP** | **58** | **4.3x fewer** |

**MIC saves $6,780/year per million IR operations vs JSON.** See [`benchmarks/BENCHMARK_RESULTS.md`](benchmarks/BENCHMARK_RESULTS.md) for full methodology.

## Proof of Systems

MIND powers real-world applications demonstrating its capabilities. Note: the
GPU-accelerated showcases below run on the commercial `mind-runtime`; the
open-source `mindc` compiler in this repo emits for the **CPU**. The runtime's
GPU and accelerator backends are available to consumers under a commercial
license (see the [Roadmap](docs/roadmap.md) for the determinism work).

| Project | Description | Highlights |
|---------|-------------|------------|
| [Mind-Ray](https://github.com/star-ga/mind-ray) | GPU path tracer | 10-50x faster than Mitsuba 3, Cycles, Falcor |
| [NikolaChess](https://github.com/star-ga/NikolaChess) | NNUE chess engine | GPU-accelerated search, +600 Elo with NNUE |
| [Fractal Voyager](https://github.com/star-ga/fractal-voyager) | Real-time fractal explorer | WebGPU/WebGL2, audio-reactive, infinite zoom |
| [mind-mem](https://github.com/star-ga/mind-mem) | Persistent memory for AI coding agents | 83 MCP tools, hybrid BM25+vector+RRF, governed memory (propose → review → apply) |
| [Swarm Brain](https://mindlang.dev/demo/swarm-brain/) | Cognitive runtime demo (live, WebGPU) | 5-invariant cognitive cycle verified every frame at 60 fps |

## Roadmap

Upcoming milestones and release planning live in [`/docs/roadmap.md`](docs/roadmap.md).

## Claude Code Plugin

This repo is a [Claude Code plugin](https://docs.anthropic.com/en/docs/claude-code/plugins). Install it to give Claude the ability to write correct `.mind` files:

```bash
# From ClawHub
clawhub install mind

# Or directly from this repo (add to .claude/settings.json)
```

**Included:**

| Component | Path | Description |
|-----------|------|-------------|
| `write-mind` skill | `skills/write-mind/SKILL.md` | Full language reference: keywords, types, operators, EBNF grammars, std library, 4 annotated examples |
| `mind-developer` agent | `agents/mind-developer.md` | Expert agent for writing `.mind` code — knows tensor syntax, autodiff, policy kernels |

The skill contains only public Apache 2.0 content from `star-ga/mind` and `star-ga/mind-spec`. No proprietary runtime content.

## Links

* [Architecture diagram](assets/diagrams/architecture.svg)
* [Brand assets](assets/logo/mind-logo.svg) · [Social cover](assets/social/og-cover.svg)
* [Contributing guidelines](CONTRIBUTING.md)
* [Security policy](SECURITY.md)

## Licensing

MIND follows an open-core dual licensing model maintained by STARGA Inc.

- **Community Edition (this repository)**  \
  The language, core compiler, and runtime found here are provided under the Apache License, Version 2.0.  \
  See [`LICENSE`](./LICENSE) for the full text.

- **Enterprise & SaaS Offerings**  \
  Enterprise-only features, hosted “MIND Cloud” services, and proprietary extensions are available under a separate commercial license from STARGA Inc. These components are not covered by the Apache License and are not present in this repository.  \
  Commercial and trademark terms are summarized in [`LICENSE-COMMERCIAL`](./LICENSE-COMMERCIAL) and governed by separate agreements with STARGA Inc.

For commercial licensing, OEM partnerships, or large-scale deployments, please contact:
`info@star.ga`.

* [Licensing overview](LICENSE)

---

Looking for implementation details? Start in [`/docs`](docs/README.md) and join the conversation in [mind-runtime](https://github.com/star-ga/mind-runtime) and [mind-spec](https://github.com/star-ga/mind-spec).
