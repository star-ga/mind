# Changelog

All notable changes to the MIND compiler project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive documentation for `TODO(runtime)` markers in `src/exec/mod.rs`
- Test execution examples in README.md
- Performance baseline documentation with concrete metrics

### Changed
- Updated benchmarks.md with detailed performance baselines

## [0.1.0] - 2025-01-01

### Added

#### Core Language
- MIND language parser with Logos lexer and Chumsky parser combinator
- Static type system with rank/shape polymorphism
- Tensor type annotations: `Tensor[dtype, shape]` syntax
- Shape inference engine with broadcasting support
- Effect tracking system for side-effect analysis

#### Compiler Pipeline
- SSA-based intermediate representation (IR)
- IR verifier and deterministic printer
- MLIR lowering passes (`mlir-lowering` feature)
- MLIR dialect support for tensor operations
- LLVM backend integration (`llvm` feature, optional)

#### Autodiff
- Reverse-mode automatic differentiation on SSA IR
- Gradient generation for arithmetic, reductions, and linear algebra
- Autodiff preview mode for debugging gradient graphs

#### CLI Tools
- `mind` REPL and evaluator binary
- `mindc` compiler driver with `--emit-ir`, `--emit-mlir` flags
- GPU target profile (`--target=gpu`) with structured error handling

#### Execution Backends
- CPU execution stubs (`cpu-exec` feature)
- Convolution stubs (`cpu-conv` feature)
- Runtime backend interface for proprietary `mind-runtime`

#### Tensor Operations
- Elementwise: add, sub, mul, div with broadcasting
- Reductions: sum, mean, max, min (all axes or specified)
- Linear algebra: matmul, dot, transpose
- Activations: ReLU, with gradient support
- Indexing: gather, slice with shape preservation
- Conv2D with padding modes (Same, Valid)

#### Developer Experience
- 69 integration tests covering all subsystems
- GitHub Actions CI for Linux, macOS, Windows
- Clippy and rustfmt enforcement
- Comprehensive documentation in `/docs`

#### Documentation
- Language tour and quick start guide
- Type system specification
- Shape algebra documentation
- IR and MLIR pipeline guides
- GPU backend design document
- Versioning and stability policy
- Roadmap with 13 phases including neuroscience/BCI applications

### Fixed
- Rank mismatch classification in type checker (#128)
- Type mismatches and autodiff test issues (#130)
- `/tmp` race condition in MLIR demo script (#133)

### Security
- Secure temporary file handling in MLIR subprocess integration

## Links

- Repository: https://github.com/star-ga/mind
- Documentation: https://github.com/star-ga/mind/tree/main/docs
- Issues: https://github.com/star-ga/mind/issues

[Unreleased]: https://github.com/star-ga/mind/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/star-ga/mind/releases/tag/v0.1.0
