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
cargo run --bin mindc -- examples/simple.mind --emit-ir
cargo run --bin mindc -- examples/simple.mind --func main --autodiff --emit-grad-ir
cargo run --features "mlir-lowering autodiff" --bin mindc -- examples/simple.mind --func main --autodiff --emit-mlir
```

MLIR emission requires the `mlir-lowering` feature. Autodiff support is
experimental and currently focused on single-output entry points.

## Core Concepts

* [Type System](docs/type-system.md) — ranks, shapes, polymorphism, and effect tracking.
* [Autodiff](docs/autodiff.md) — reverse-mode differentiation on the SSA IR.
* [IR core](docs/ir.md) — deterministic IR pipeline with verifier and printer.
* [IR & MLIR](docs/ir-mlir.md) — compiler pipeline from parser to MLIR dialects.

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
`info@star.ga` or `legal@star.ga`.

* [Licensing overview](LICENSE)

---

Looking for implementation details? Start in [`/docs`](docs/README.md) and join the conversation in [mind-runtime](https://github.com/cputer/mind-runtime) and [mind-spec](https://github.com/cputer/mind-spec).
