# Changelog

All notable changes to the MIND compiler project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.1] - 2026-02-17

### Added
- IR verifier audit coverage: 14 new tests for conv2d stride validation,
  reduction axis checks, FnDef body scoping, and duplicate definition detection
- Determinism proof results for v0.2.0-hardened (4/4 DETERMINISTIC)
- Criterion benchmark results for hardened pipeline (338K compilations/sec)

### Fixed
- **C1**: Conv2d IR verifier now rejects zero strides and negative axes
- **C6**: FnDef body verifier enforces SSA scope (use-before-def in body blocks)
- **C2**: String interning DoS protection (MAX_INTERNED_STRINGS = 100,000)
- **C3**: IR printer determinism via sorted function iteration
- **C4**: Constant folding bounds checking for division-by-zero and overflow
- **C5**: Type checker array bounds validation
- **C7**: Hardened eval NaN/Inf propagation
- **A1**: Cargo Deny supply-chain audit configuration

### Security
- String interning rate limiting prevents memory exhaustion attacks
- Constant folding rejects division by zero and integer overflow at compile time

## [0.2.0] - 2026-02-07

### Added
- IR-first compilation pipeline with shape ops and MIC emission
- Remizov universal ODE solver (`std::ode` module)
- Real tensor compute backend with benchmarks
- Open-core reference interpreter for public compiler

### Changed
- **BREAKING**: Replaced Chumsky parser combinator with hand-written recursive descent parser (15x speedup)
- Parser now achieves ~347,000 compilations/sec (up from ~22,700 with Chumsky)
- Removed `chumsky` dependency entirely — zero unnecessary allocations, direct byte-level parsing
- CI skips builds for docs-only changes

### Fixed
- Keyword argument disambiguation in `tensor.gather()` calls (positional `idx` vs `idx=` prefix)
- Clippy lint: `map_or` → `is_some_and` for modern Rust idiom
- Formatting consistency across parser, eval, and exec modules
- Removed unfair NumPy comparisons from benchmarks

### Documentation
- Framework comparison and GPU projections
- Runtime execution benchmarks for v0.1.9
- ODE solver examples and usage guide

## [0.1.9] - 2026-02-05

### Changed
- **BREAKING**: Renamed library crate from `mind` to `libmind` to avoid Windows PDB filename collision
- Updated all imports: `use mind::` → `use libmind::`
- Benchmark numbers aligned across README and docs

### Fixed
- Windows LNK1318 PDB linker errors by limiting parallel build jobs
- Format check CI failures (CRLF → LF line endings)

### Documentation
- Updated test counts to 169+ tests across 70 test files
- Aligned benchmark numbers between README and docs/benchmarks/
- Added v0.1.8 and v0.1.9 to compiler_performance.md version history

## [0.1.8] - 2026-01-15

### Added
- Comprehensive documentation for `TODO(runtime)` markers in `src/exec/mod.rs`
- Test execution examples in README.md
- Performance baseline documentation with concrete metrics

### Changed
- Updated benchmarks.md with detailed performance baselines

## [0.1.0] - 2025-01-01

### Added

#### Core Language
- MIND language parser with Logos lexer (originally Chumsky, replaced by recursive descent in v0.2.0)
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
- 70 integration test files covering all subsystems
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

[Unreleased]: https://github.com/star-ga/mind/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/star-ga/mind/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/star-ga/mind/compare/v0.1.9...v0.2.0
[0.1.9]: https://github.com/star-ga/mind/releases/tag/v0.1.9
[0.1.8]: https://github.com/star-ga/mind/releases/tag/v0.1.8
[0.1.0]: https://github.com/star-ga/mind/releases/tag/v0.1.0
