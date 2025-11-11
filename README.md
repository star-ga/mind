# MIND — The Native Language for Intelligent Systems

[![CI](https://github.com/cputer/mind/actions/workflows/ci.yml/badge.svg)](https://github.com/cputer/mind/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-online-green.svg)](https://cputer.github.io/mind-spec/)
[![Build](https://img.shields.io/github/actions/workflow/status/cputer/mind/ci.yml?branch=main)](https://github.com/cputer/mind/actions)

> **Status:** ✅ Complete (v0.9 baseline)
>  
> Companion repos: [**mind-runtime**](https://github.com/cputer/mind-runtime) · [**mind-spec**](https://github.com/cputer/mind-spec)

---

### Overview

**MIND** is a Rust-based language and runtime designed for intelligent systems — combining declarative tensor algebra, shape inference, automatic differentiation, and MLIR/LLVM backends in a compact, auditable core.

It now ships with:
- ✴️ Full parser → type checker → IR → MLIR pipeline  
- ⚙️ CPU execution, autodiff, and JIT/GPU scaffolding  
- 🧩 Optional FFI & packaging support (`ffi-c`, `pkg` features)  
- 🧠 Comprehensive test suite and CI/CD pipelines  
- 📚 Docsify-based specification site and cargo-deny license checks  

---

### Quick Start

```bash
git clone https://github.com/cputer/mind.git
cd mind
cargo build --release
cargo run -- eval "let x: Tensor[f32,(2,3)] = 0; x + 1"
```

Expected output:

```
Tensor[F32,(2,3)] fill=1
```

See [`mind-spec`](https://github.com/cputer/mind-spec) for full language documentation.

---

### Development

MIND follows a feature-gated architecture:

| Feature       | Description                 |
| ------------- | --------------------------- |
| `cpu-buffers` | Materialized tensor buffers |
| `cpu-exec`    | CPU backend execution       |
| `cpu-conv`    | Conv2D kernels              |
| `mlir-exec`   | MLIR JIT execution          |
| `mlir-build`  | MLIR AOT compiler           |
| `ffi-c`       | C ABI bindings              |
| `pkg`         | Package tooling             |

To test all:

```bash
cargo test --all-features
```

---

### Architecture & Docs

* [ARCHITECTURE.md](ARCHITECTURE.md) — module layout and runtime flow
* [STATUS.md](STATUS.md) — feature roadmap (now **100 %** complete)
* [RELEASING.md](RELEASING.md) — tagging and publishing workflow
* [SECURITY.md](SECURITY.md) — vulnerability disclosure policy

---

### License

Licensed under [MIT](LICENSE).
Copyright © 2025 MIND Language Contributors.

---

## ✅ Current Release

| Component                   | Version | Status            |
| --------------------------- | ------- | ----------------- |
| Compiler Core (`mind`)      | v0.9.0  | ✅ Stable baseline |
| Runtime (`mind-runtime`)    | v0.9.0  | ✅ Complete        |
| Specification (`mind-spec`) | v0.9.0  | ✅ Published       |
