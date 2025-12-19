<p align="center">
  <img src="assets/logo/mind-logo.svg" alt="MIND logo" width="512" />
</p>

# MIND — Native Language for Intelligent Systems

[![CI](https://github.com/cputer/mind/actions/workflows/ci.yml/badge.svg)](https://github.com/cputer/mind/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

## Overview

MIND is a Rust-first language and runtime for building intelligent systems with auditable foundations. It blends declarative tensor algebra, static shape inference, automatic differentiation, and MLIR/LLVM lowering in a compact toolchain that scales from research prototypes to production.

## Open-core vs proprietary runtime

This repository contains the open-core stack: the MIND language, type system, compiler front-end, IR, and MLIR lowering passes. Production-grade runtime backends for CPU, GPU, and accelerators live in the private [`mind-runtime`](https://github.com/cputer/mind-runtime) repository. Functions in `src/exec/*` marked with `todo!()` or `unimplemented!()` are runtime hooks that the proprietary backend fulfills.

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
git clone https://github.com/cputer/mind.git
cd mind
cargo run -- eval "let x: Tensor[f32,(2,3)] = 0; x + 1"
```

Explore the full language tour and runtime guides in [`/docs`](docs/README.md).

## CLI / Compiler Driver

The `mindc` binary provides a deterministic source→IR→MLIR pipeline suitable
for demos and snapshot tests:

```bash
cargo run --bin mindc -- examples/hello_tensor.mind --emit-ir
cargo run --bin mindc -- examples/hello_tensor.mind --func main --autodiff --emit-grad-ir
cargo run --features "mlir-lowering autodiff" --bin mindc -- examples/hello_tensor.mind --func main --autodiff --emit-mlir
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

## Benchmarks

The [`/docs/benchmarks.md`](docs/benchmarks.md) report covers baseline compiler/runtime performance, regression tracking, and methodology.

## Roadmap

Upcoming milestones and release planning live in [`/docs/roadmap.md`](docs/roadmap.md).

## Links

* [Architecture diagram](assets/diagrams/architecture.svg)
* [Brand assets](assets/logo/mind.svg) · [Social cover](assets/social/og-cover.svg)
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

Looking for implementation details? Start in [`/docs`](docs/README.md) and join the conversation in [mind-runtime](https://github.com/cputer/mind-runtime) and [mind-spec](https://github.com/cputer/mind-spec).
