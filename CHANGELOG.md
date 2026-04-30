# Changelog

All notable changes to the MIND compiler project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `libmind::cache` — content-addressed compilation cache with
  `CompilationCache`, `CacheKey`, `CacheEntry`, `CacheStats`,
  `ProfileTag`, and an in-memory `MemoryStore` backend. Cache key
  includes compiler version + profile tag + source hash + imports
  hash so cross-mode rebuilds never hit a stale entry. Foundation
  for sub-µs warm-start frontend latency. 17 unit tests, all passing.
- `tools/pytorch_bridge/` — PyTorch/JAX → MIND transpiler tooling.
  ONNX-driven PyTorch path, XLA-HLO-driven JAX path, and a
  deterministic prompt builder for AI-assisted UNSAT proof
  resolution. Pure Python, no torch / jax import at module load.
  11 unit tests, all passing.
- `libmind::distributed` — IR-layer primitives for tensor and pipeline
  parallelism: `ShardSpec` / `ShardLayout` (replicated / split /
  split-2D), `AllReduceOp` with lexicographic / tree / arrival
  reduction orders, `AllGatherOp` with lexicographic / arrival gather
  orders, `PipelineGraph` / `PipelineStage` / `StageBoundary`, and
  `DistributedInvariant` enforcement (`deterministic_all_reduce`,
  `reduction_order_lexicographic`, `gather_order_lexicographic`,
  `evidence_chain_continuous`). 31 unit tests, all passing.
  See `docs/roadmap.md` Phase 13.6 for the design rationale and the
  speed-preservation discipline that keeps the 1.8–15.5 µs frontend
  baseline locked when these primitives are not imported.

## [0.2.5] - 2026-04-28

### Added
- **Pratt operator-precedence parser** for the expression layer. Replaces
  the recursive-descent chain `parse_logical_or → parse_logical_and →
  parse_comparison → parse_additive → parse_bitwise → parse_multiplicative`
  with a single dispatch function driven by a binding-power table. Phase 10.5
  operators (`||`, `&&`, `|`, `&`, `^`, `<<`, `>>`, `as`) become table
  entries, and future operators (Phase 11/12) become O(1) inserts.
- **Stable IR public API**: `libmind::ir::load(bytes) -> IRModule` and
  `libmind::ir::save(module) -> String` for the `mic@1` textual format.
  Plus `libmind::compile_to_mic_text(src, opts)` as the AOT pipeline.
- `docs/ir-stability.md` — formalises the IR contract used by
  `mind-runtime` and other downstream backends to consume pre-compiled IR
  instead of re-running the surface parser per inference.
- `.github/workflows/bench-gate.yml` + `tools/bench_gate.py` — CI gate that
  fails on >2% mean regression on `small_matmul`, `medium_mlp`, or
  `large_network` vs the frozen baseline at
  `.bench-baseline-2026-04-28-pratt.txt`.
- Phase 10.5 conformance programs in `tests/conformance/cpu_baseline/`
  (`phase_10_5_const.mind`, `..._logical.mind`, `..._struct.mind`,
  `..._module.mind`).
- `tests/ir_load_save.rs` — pins the round-trip and determinism contract
  for the new IR API.

### Changed
- **Parser is faster than the pre-Phase-10.5 baseline** on `medium_mlp`
  (-9.0%) and `large_network` (-4.6%); within +3.9% on `small_matmul`.
  The Pratt rewrite recovered the +9% regression introduced when Phase 10.5
  dispatch arms were added in 0.2.4.
- Removed the `peek_skip_ws` helper (only used by the recursive-descent
  fast-paths it was added to support; Pratt makes it unnecessary).
- Removed dead `parse_logical_or` forwarder (deprecated since Phase 10.5
  inlined logical-or into `parse_expr`).

### Architecture
- mindc's parser is no longer a runtime dependency for `mind-runtime`
  consumers. The supported pattern is to AOT-compile to `mic@1` once at
  build time and call `ir::load` at runtime. This decouples parser
  performance from per-inference latency across all 12+ planned backends
  (CPU, CUDA, Metal, ROCm, WebGPU, WebNN, ARM, TPU, NPU, LPU, DPU, FPGA,
  Quantum).

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
