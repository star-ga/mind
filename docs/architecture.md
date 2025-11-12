# Architecture

MIND is structured as a modular compiler-runtime stack. This document captures the top-level components and how data flows between them.

## High-Level Overview

1. **Frontend** – Lexer, parser, and surface-level validations produce a typed abstract syntax tree (AST).
2. **Type & Shape System** – A constraint solver assigns concrete ranks, shapes, and element types while validating effect capabilities.
3. **Intermediate Representation (IR)** – The typed AST lowers into a static single assignment (SSA) IR purpose-built for tensor programs.
4. **Lowering Pipelines** – Dedicated passes perform canonicalization, fusion, layout selection, and eventually emit MLIR.
5. **Execution Runtimes** – Backends convert MLIR to CPU or GPU executables, or interpret the IR directly for debugging.
6. **Tooling** – Packaging, FFI, benchmarking, and developer tooling live alongside the compiler in feature-gated crates.

The architecture diagram in [`../assets/diagrams/architecture.svg`](../assets/diagrams/architecture.svg) mirrors this flow.

## Crate Layout

| Crate/Module        | Responsibility                                                  |
| ------------------- | --------------------------------------------------------------- |
| `mind-syntax`       | Lexer, parser, AST definitions, surface diagnostics             |
| `mind-types`        | Type lattice, constraint solver, effect tracking                |
| `mind-ir`           | Core SSA structures, pattern-matching utilities, graph rewrites |
| `mind-lowering`     | High-level → mid-level lowering, canonicalization passes        |
| `mind-mlir`         | Emission of MLIR dialects, translation to LLVM                  |
| `mind-runtime`      | Tensor buffer management, host/device executors                 |
| `mind-cli`          | Command-line interface, REPL, and package tooling               |

The root crate enables features to bring specific components into the final binary. For example `--features cpu-exec,mlir-exec` compiles both the native interpreter and MLIR JIT.

## Data Flow

```
Source → AST → Typed AST → MIND IR → MLIR → LLVM IR / Runtime Calls → Execution
```

Key invariants:

- Types and shapes must be fully resolved before lowering to MLIR.
- Autodiff annotations expand into explicit IR functions prior to optimization.
- Runtime backends operate on host-device descriptors generated during lowering.

## Extensibility Hooks

- **Dialect Extensions** – New operators are introduced via pattern definitions in `mind-ir` and mirrored in MLIR dialect extensions.
- **Backend Plugins** – Trait-based executors allow embedding custom accelerators via the `Runtime` trait.
- **Pass Pipelines** – Pass managers can be configured per target, making it safe to ship experimental transformations behind feature flags.

For detailed discussions of the intermediate representation see [`ir-mlir.md`](ir-mlir.md); for runtime integration details refer to [`ffi-runtime.md`](ffi-runtime.md).
