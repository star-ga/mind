<p align="center">
  <img src="assets/logo/mind-logo.svg" alt="MIND logo" width="512" />
</p>

# MIND — Machine Intelligence Native Design

[![CI](https://github.com/star-ga/mind/actions/workflows/ci.yml/badge.svg)](https://github.com/star-ga/mind/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

## Overview

MIND is a Rust-first language and runtime for building intelligent systems with auditable foundations. It blends declarative tensor algebra, static shape inference, automatic differentiation, and MLIR/LLVM lowering in a compact toolchain that scales from research prototypes to production.

## Open-core vs proprietary runtime

This repository contains the open-core stack: the MIND language, type system, compiler front-end, IR, and MLIR lowering passes. Production-grade runtime backends for CPU, GPU, and accelerators live in the private [`mind-runtime`](https://github.com/star-ga/mind-runtime) repository. Functions in `src/exec/*` marked with `todo!()` or `unimplemented!()` are runtime hooks that the proprietary backend fulfills.

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

# Build and run a MIND project
mindc run
```

### Feature Flags

By default, `mindc` builds with minimal features for fast compilation. Enable
additional features as needed:

```bash
cargo build --features aot        # AOT compilation (--emit-obj, project builds)
cargo build --features autodiff   # Autodiff support
cargo build --features full       # All features
```

MLIR emission requires the `mlir-lowering` feature. Autodiff support is
experimental and currently focused on single-output entry points.

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

## Architecture

* [Runtime & Compiler Architecture](docs/architecture.md)
* [FFI & Runtime Embedding](docs/ffi-runtime.md)

## Testing

The MIND compiler includes a comprehensive test suite with 169+ tests across 69 test files covering parsing, type checking, IR generation, MLIR lowering, and execution.

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
| Execution | `exec_*.rs`, `relu_*.rs`, `conv2d_*.rs` | Runtime execution stubs |

### Continuous Integration

All tests run on every pull request via GitHub Actions. See [`.github/workflows/ci.yml`](.github/workflows/ci.yml) for the full CI configuration.

## Benchmarks

The [`/docs/benchmarks.md`](docs/benchmarks.md) report covers baseline compiler/runtime performance, regression tracking, and methodology.

### Compilation Speed

#### Reference Benchmarks (v0.1.9)

| vs Framework | MIND Speedup |
|--------------|--------------|
| PyTorch 2.0 torch.compile (inductor) | **1,000-2,400× faster** |
| Mojo (mojo build) | **20,000-35,000× faster** |

| Benchmark | PyTorch (inductor) | Mojo (build) | MIND (in-process) |
|-----------|-------------------|--------------|-------------------|
| scalar_math | 43 ms | 908 ms | **26 µs** |
| small_matmul | 62 ms | 928 ms | **45 µs** |
| medium_matmul | - | - | **46 µs** |
| large_matmul | - | - | **45 µs** |

*In-process compilation benchmarks. See [docs/benchmarks/compiler_performance.md](docs/benchmarks/compiler_performance.md) for methodology.*

### MIC/MAP Format Efficiency

| Format | Tokens | vs JSON | Parse Speed | Annual Cost (1M IRs) |
|--------|--------|---------|-------------|----------------------|
| JSON | 278 | baseline | 5.31 us | $8,340 |
| TOML | 151 | 1.8x | 137.06 us | $4,530 |
| TOON | 67 | 4.1x | 2.67 us | $2,010 |
| **MIC** | **52** | **5.3x** | **2.26 us** | **$1,560** |

| Protocol | Tokens | vs JSON-RPC |
|----------|--------|-------------|
| JSON-RPC | 251 | baseline |
| **MAP** | **58** | **4.3x fewer** |

**MIC saves $6,780/year per million IR operations vs JSON.** See [`benchmarks/BENCHMARK_RESULTS.md`](benchmarks/BENCHMARK_RESULTS.md) for full methodology.

## Proof of Systems

MIND powers real-world applications demonstrating its capabilities:

| Project | Description | Highlights |
|---------|-------------|------------|
| [Mind-Ray](https://github.com/star-ga/mind-ray) | GPU path tracer | 10-50x faster than Mitsuba 3, Cycles, Falcor |
| [NikolaChess](https://github.com/star-ga/NikolaChess) | NNUE chess engine | GPU-accelerated search, +600 Elo with NNUE |
| [Fractal Voyager](https://github.com/star-ga/fractal-voyager) | Real-time fractal explorer | WebGPU/WebGL2, audio-reactive, infinite zoom |

## Roadmap

Upcoming milestones and release planning live in [`/docs/roadmap.md`](docs/roadmap.md).

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
