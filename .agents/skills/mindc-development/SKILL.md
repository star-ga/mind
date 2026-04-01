---
name: mindc-development
description: MIND language compiler development
tags: [compiler, mind, rust, mlir]
---

# MIND Compiler (mindc) Development

## Architecture
- Language: Rust
- Pipeline: Source → Lexer → Parser → AST → IR → MLIR → Native (ELF)
- Targets: CPU (x86, ARM), CUDA (via mind-runtime)

## Current Parser Status
- Tensor DSL working (matmul, transpose, reshape, etc.)
- Missing: struct, enum, trait, [protection], [invariant], module-qualified paths
- Parser extension is a tracked task (T-20260318-007)

## Key Directories
- `src/` — compiler source (Rust)
- `examples/` — example .mind files
- `tests/` — compiler test suite

## Build
```bash
cargo build --release
./target/release/mindc build <file.mind>
```

## Conventions
- Rust idioms (Result types, no unwrap in production code)
- Test every parser rule
- Benchmark compilation speed
- Binary distribution: mindc-<version>-<platform>.tar.gz
