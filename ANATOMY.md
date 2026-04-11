# ANATOMY.md — Project File Index

> **For coding agents.** Read this before opening files. Use descriptions and token
> estimates to decide whether you need the full file or the summary is enough.
> Re-generate with: `anatomy .`

**Project:** `mind`
**Files:** 326 | **Est. tokens:** ~415,870
**Generated:** 2026-04-11 07:02 UTC

## Token Budget Guide

| Size | Tokens | Read strategy |
|------|--------|---------------|
| tiny | <50 | Always safe to read |
| small | 50-200 | Read freely |
| medium | 200-500 | Read if relevant |
| large | 500-1500 | Use summary first, read specific sections |
| huge | >1500 | Avoid full read — use grep or read specific lines |

## Directory Overview

| Directory | Files | Est. tokens |
|-----------|-------|-------------|
| `./` | 24 | ~15,969 |
| `agents/` | 1 | ~436 |
| `.agents/skills/mindc-development/` | 1 | ~235 |
| `benches/` | 5 | ~6,036 |
| `benchmarks/` | 11 | ~19,918 |
| `benchmarks/autograd_comparison/` | 8 | ~9,411 |
| `benchmarks/determinism/` | 3 | ~4,601 |
| `benchmarks/inference/` | 4 | ~4,008 |
| `benchmarks/jax_comparison/` | 5 | ~4,642 |
| `benchmarks/mojo/` | 8 | ~4,301 |
| `benchmarks/pytorch_comparison/` | 5 | ~4,828 |
| `.cargo/` | 1 | ~130 |
| `docs/` | 19 | ~16,294 |
| `docs/benchmarks/` | 2 | ~7,027 |
| `docs/design/` | 2 | ~136 |
| `docs/rfcs/` | 4 | ~1,081 |
| `docs/specs/` | 2 | ~976 |
| `examples/` | 14 | ~31,974 |
| `examples/c/` | 2 | ~400 |
| `examples/compliance/` | 3 | ~5,294 |
| `examples/zoo/` | 6 | ~12,885 |
| `.github/` | 3 | ~148 |
| `.github/ISSUE_TEMPLATE/` | 3 | ~440 |
| `.github/workflows/` | 5 | ~3,471 |
| `mind/std/cognitive/` | 4 | ~3,529 |
| `scripts/` | 2 | ~2,268 |
| `skills/write-mind/` | 1 | ~6,002 |
| `src/` | 7 | ~14,738 |
| `src/ast/` | 1 | ~1,871 |
| `src/autodiff/` | 3 | ~4,224 |
| `src/bin/` | 2 | ~10,458 |
| `src/diagnostics/` | 1 | ~2,230 |
| `src/eval/` | 12 | ~54,405 |
| `src/eval/stdlib/` | 2 | ~8,515 |
| `src/exec/` | 3 | ~4,591 |
| `src/ffi/` | 3 | ~3,910 |
| `src/ir/` | 3 | ~5,639 |
| `src/ir/compact/` | 3 | ~13,890 |
| `src/ir/compact/v2/` | 6 | ~15,796 |
| `src/mlir/` | 2 | ~4,569 |
| `src/ops/` | 2 | ~2,047 |
| `src/opt/` | 4 | ~6,469 |
| `src/package/` | 2 | ~1,668 |
| `src/parser/` | 1 | ~15,118 |
| `src/project/` | 1 | ~9,268 |
| `src/runtime/` | 3 | ~968 |
| `src/shapes/` | 2 | ~6,051 |
| `src/stdlib/` | 2 | ~560 |
| `src/type_checker/` | 1 | ~15,239 |
| `src/types/` | 4 | ~3,179 |
| `tests/` | 74 | ~40,918 |
| `tests/autodiff/` | 2 | ~247 |
| `tests/backend/` | 2 | ~125 |
| `tests/conformance/cpu_baseline/` | 9 | ~123 |
| `tests/conformance/gpu_profile/` | 2 | ~11 |
| `tests/fixtures/` | 4 | ~44 |
| `tests/ir_verification/` | 2 | ~108 |
| `tests/lexical/` | 3 | ~191 |
| `tests/runtime/` | 2 | ~135 |
| `tests/shapes/` | 3 | ~260 |
| `tests/type_checker/` | 2 | ~140 |
| `tools/` | 2 | ~1,704 |
| `.wrangler/cache/` | 1 | ~21 |

## Files

### `./`

- `ARCHITECTURE.md` (~182 tok, small) — MIND Architecture (high level)
- `AUDIT_REPORT.md` (~1151 tok, large) — Audit Report
- `bounties.md` (~888 tok, large) — MIND Bounty Board
- `build.rs` (~234 tok, medium) — Copyright 2025 STARGA Inc.
- `Cargo.toml` (~695 tok, large) — [package]
- `CHANGELOG.md` (~1485 tok, large) — Changelog
- `clippy.toml` (~25 tok, tiny)
- `CODE_OF_CONDUCT.md` (~29 tok, tiny) — Code of Conduct
- `COMPLETE_FILE_STRUCTURE.md` (~26 tok, tiny) — Repository Structure (Snapshot)
- `CONTRIBUTING.md` (~1348 tok, large) — Contributing to MIND
- `deny.toml` (~89 tok, small) — [advisories]
- `.editorconfig` (~51 tok, small) — root = true
- `GITHUB_SETUP_INSTRUCTIONS.md` (~240 tok, medium) — GitHub Setup (Quick)
- `.gitignore` (~110 tok, small) — # Rust
- `LICENSE` (~2573 tok, huge) —                                  Apache License
- `LICENSE-COMMERCIAL` (~399 tok, medium) — COMMERCIAL LICENSE NOTICE – MIND (Enterprise & SaaS)
- `plugin.json` (~62 tok, small) — Keys: name, description, version, skills, agents
- `PR_DESCRIPTION.md` (~1272 tok, large) — Complete Patent Benchmark Suite with Python Bindings
- `README.md` (~3203 tok, huge) — MIND — Machine Intelligence Native Design
- `RELEASING.md` (~131 tok, small) — Release checklist (as of v0.2.1)
- `rustfmt.toml` (~23 tok, tiny) — max_width = 100
- `SECURITY.md` (~614 tok, large) — Security Policy
- `STATUS.md` (~874 tok, large) — MIND Compiler Status
- `test_real_compile_time.py` (~265 tok, medium) — Quick test of real MIND compilation time using Python bindings."""
### `agents/`

- `mind-developer.md` (~436 tok, medium) — MIND Developer Agent
### `.agents/skills/mindc-development/`

- `SKILL.md` (~235 tok, medium) — MIND Compiler (mindc) Development
### `benches/`

- `autodiff.rs` (~1661 tok, huge) — Simple linear function
- `compiler.rs` (~1322 tok, large) — Small program: Simple matrix multiplication
- `operations.rs` (~1143 tok, large) — Element-wise operations
- `shapes.rs` (~1267 tok, large) — Simple broadcasting scenarios
- `simple_benchmarks.rs` (~643 tok, large) — Benchmark source programs that are known to work
### `benchmarks/autograd_comparison/`

- `autograd_results.json` (~424 tok, medium) — Keys: system_info, benchmarks
- `benchmark_autograd.py` (~2444 tok, huge)
- `benchmark_python_bindings.py` (~1566 tok, huge)
- `benchmark_real_autograd.py` (~2304 tok, huge)
- `README.md` (~1153 tok, large) — Autograd Comparison: MIND vs PyTorch
- `README_REAL.md` (~1185 tok, large) — Real Autograd Comparison: MIND vs PyTorch
- `real_autograd_results.json` (~328 tok, medium) — Keys: system_info, methodology, benchmarks
- `requirements.txt` (~7 tok, tiny) — torch>=1.0.0
### `benchmarks/`

- `BENCHMARK_RESULTS.md` (~4281 tok, huge) — MIND Benchmark Results
- `COPILOT_REVIEW_FIXES.md` (~1722 tok, huge) — Copilot Review Fixes - PR #172
### `benchmarks/determinism/`

- `benchmark_determinism.py` (~2187 tok, huge)
- `determinism_results.json` (~1103 tok, large) — Keys: system_info, num_runs, tests, all_deterministic
- `README.md` (~1311 tok, large) — MIND Determinism Proof Benchmark
### `benchmarks/`

- `format_benchmark.py` (~2617 tok, huge)
### `benchmarks/inference/`

- `benchmark_inference.py` (~2423 tok, huge)
- `inference_results.json` (~473 tok, medium) — Keys: system_info, benchmarks
- `README.md` (~1108 tok, large) — Inference Speed Benchmark
- `requirements.txt` (~4 tok, tiny) — torch>=1.0.0
### `benchmarks/jax_comparison/`

- `benchmark_jax_compile.py` (~2719 tok, huge)
- `jax_coldstart_results.json` (~376 tok, medium) — Keys: environment, results
- `jax_results.json` (~478 tok, medium) — Keys: system_info, benchmarks
- `README.md` (~1062 tok, large) — JAX Compilation Benchmark
- `requirements.txt` (~7 tok, tiny) — jax>=0.4.0
### `benchmarks/`

- `mic_benchmark.py` (~1473 tok, large)
### `benchmarks/mojo/`

- `benchmark_mojo_compilation.py` (~1534 tok, huge)
- `large_matmul.mojo` (~205 tok, medium) — """
- `medium_matmul.mojo` (~205 tok, medium) — """
- `mojo_results.json` (~216 tok, medium) — Keys: scalar_math, small_matmul, medium_matmul, large_matmul
- `README.md` (~1295 tok, large) — Mojo Compilation Benchmarks
- `run_benchmarks.sh` (~581 tok, large) — Mojo Compilation Benchmark Runner
- `scalar_math.mojo` (~58 tok, small) — """
- `small_matmul.mojo` (~207 tok, medium) — """
### `benchmarks/pytorch_comparison/`

- `=2.0` (~0 tok, tiny)
- `benchmark_pytorch_compile.py` (~3420 tok, huge)
- `pytorch_results.json` (~590 tok, large) — Keys: system_info, benchmarks
- `README.md` (~814 tok, large) — PyTorch Compilation Benchmark
- `requirements.txt` (~4 tok, tiny) — torch>=2.0.0
### `benchmarks/`

- `README.md` (~1189 tok, large) — MIND Performance Benchmarks
- `resnet.md` (~73 tok, small) — ResNet Benchmarks (Preliminary)
- `run_all_benchmarks.sh` (~824 tok, large) — Master script to run all MIND patent benchmarks
- `RUN_GUIDE.md` (~1465 tok, large) — MIND Patent Benchmarks - Environment Guide
- `SAME_MACHINE_BENCHMARKS.md` (~2320 tok, huge) — Same-Machine Benchmarks - Addressing Copilot Concerns
- `scientific_benchmark.py` (~1603 tok, huge)
- `scientific_benchmark_raw.py` (~2351 tok, huge)
### `.cargo/`

- `config.toml` (~130 tok, small) — [registries]
### `docs/`

- `architecture.md` (~755 tok, large) — Architecture
- `autodiff.md` (~467 tok, medium) — Static autodiff (public)
### `docs/benchmarks/`

- `compiler_performance.md` (~4607 tok, huge) — MIND Compiler Performance Benchmarks
### `docs/`

- `benchmarks.md` (~807 tok, large) — Benchmarks
### `docs/benchmarks/`

- `mojo_comparison.md` (~2420 tok, huge) — MIND vs Mojo: Compilation Performance Comparison
### `docs/`

- `cli.md` (~582 tok, large) — MIND CLI Reference
### `docs/design/`

- `README.md` (~26 tok, tiny) — Design Docs
- `v0.3.md` (~110 tok, small) — MIND Design v0.3 (Draft)
### `docs/`

- `errors.md` (~574 tok, large) — MIND Core Error Model
- `ffi-runtime.md` (~529 tok, large) — FFI & Runtime Integration
- `gpu.md` (~387 tok, medium) — GPU backend profile
- `ir.md` (~451 tok, medium) — MIND IR core
- `ir-mlir.md` (~480 tok, medium) — IR & MLIR Integration
- `mlir-lowering.md` (~210 tok, medium) — MLIR lowering pipeline (public)
- `ops.md` (~604 tok, large) — Core v1 operator coverage
- `performance.md` (~742 tok, large) — Performance Guide
- `README.md` (~162 tok, small) — MIND Documentation
### `docs/rfcs/`

- `0000-template.md` (~627 tok, large) — RFC 0000: [Title]
- `000-template.md` (~1 tok, tiny)
- `odc-language-primitives.md` (~422 tok, medium) — RFC: Observer-Dependent Cognition — Language Primitives
- `README.md` (~31 tok, tiny) — RFCs
### `docs/`

- `roadmap.md` (~4234 tok, huge) — Roadmap
- `security.md` (~753 tok, large) — Security Guide
- `shapes.md` (~478 tok, medium) — Tensor shape semantics
### `docs/specs/`

- `README.md` (~23 tok, tiny) — Specifications
- `v1.0.md` (~953 tok, large) — MIND Language Specification v1.0 (Working Draft)
### `docs/`

- `type-system.md` (~605 tok, large) — Type System
- `versioning.md` (~630 tok, large) — MIND Core Stability & Versioning
- `whitepaper.md` (~2844 tok, huge) — MIND: The Native Language for Intelligent Systems
### `examples/`

- `autodiff_demo.mind` (~1846 tok, huge) — Autodiff Demonstration
### `examples/c/`

- `min.c` (~82 tok, small)
- `mind.h` (~318 tok, medium) — Copyright 2025 STARGA Inc.
### `examples/`

- `cnn_classifier.mind` (~1143 tok, large) — CNN Classifier Example
### `examples/compliance/`

- `auditable_model.mind` (~1932 tok, huge) — auditable_model.mind -- Compliance-Ready MLP with Provenance Metadata
- `audit_report.mind` (~2289 tok, huge) — audit_report.mind -- Compliance Artifact Generation
- `README.md` (~1073 tok, large) — Compliance Example
### `examples/`

- `fft_signal.mind` (~533 tok, large) — FFT Signal Processing Example for MIND
- `hello_tensor.mind` (~142 tok, small) — Hello World example for MIND
- `mlir_pipeline_demo.sh` (~1647 tok, huge) — MLIR/LLVM Pipeline Demonstration
- `policy.mind` (~1221 tok, large) — policy.mind — v0.1 Execution Boundary Kernel
- `README.md` (~2066 tok, huge) — MIND Examples
- `remizov_benchmark.mind` (~6297 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `remizov_feynman.mind` (~2792 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `remizov_gpu.mind` (~2559 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `remizov_inverse.mind` (~2511 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `remizov_solver.mind` (~3699 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `remizov_verify.mind` (~3618 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `tiny_edge_model.mind` (~1900 tok, huge) — Tiny Edge Model Example
### `examples/zoo/`

- `conv_classifier.mind` (~2518 tok, huge) — CNN Classifier — MIND Model Zoo
- `linear_regression.mind` (~1320 tok, large) — Linear Regression — MIND Model Zoo
- `logistic_classifier.mind` (~1521 tok, huge) — Logistic Classifier — MIND Model Zoo
- `mlp_mnist.mind` (~2368 tok, huge) — MLP MNIST Classifier — MIND Model Zoo
- `README.md` (~1191 tok, large) — MIND Model Zoo
- `transformer_block.mind` (~3967 tok, huge) — Transformer Block — MIND Model Zoo
### `.github/`

- `CODEOWNERS` (~8 tok, tiny) — *       @cputer
### `.github/ISSUE_TEMPLATE/`

- `bounty_claim.md` (~56 tok, small)
- `bug_report.md` (~213 tok, medium) — Describe the bug
- `feature_request.md` (~171 tok, small) — Problem Statement
### `.github/`

- `PULL_REQUEST_TEMPLATE.md` (~55 tok, small) — Summary
- `release-drafter.yml` (~85 tok, small) — name-template: 'v$NEXT_PATCH_VERSION'
### `.github/workflows/`

- `cargo-deny.yml` (~222 tok, medium) — name: Cargo Deny
- `ci.yml` (~1358 tok, large) — name: CI
- `link-check.yml` (~221 tok, medium) — name: Link Check
- `release-drafter.yml` (~91 tok, small) — name: Release Drafter
- `release.yml` (~1579 tok, huge) — name: Release
### `mind/std/cognitive/`

- `batch_scheduler.mind` (~850 tok, large) — Batch scheduling for inference workloads
- `kv_cache.mind` (~840 tok, large) — KV-Cache for transformer inference
- `speculative.mind` (~891 tok, large) — Speculative decoding with rejection sampling
- `verification.mind` (~948 tok, large) — Verification plane for inference consistency (LCU)
### `scripts/`

- `anatomy-hook.sh` (~258 tok, medium) — anatomy-hook.sh — Git pre-commit hook to refresh ANATOMY.md
- `anatomy.sh` (~2010 tok, huge) — anatomy — Generate ANATOMY.md for any repo
### `skills/write-mind/`

- `SKILL.md` (~6002 tok, huge) — Write MIND Code
### `src/ast/`

- `mod.rs` (~1871 tok, huge) — Copyright 2025 STARGA Inc.
### `src/autodiff/`

- `engine.rs` (~2684 tok, huge) — Copyright 2025 STARGA Inc.
- `mod.rs` (~342 tok, medium) — Copyright 2025 STARGA Inc.
- `rules.rs` (~1198 tok, large) — Copyright 2025 STARGA Inc.
### `src/bin/`

- `mind-ai.rs` (~6045 tok, huge) — Copyright 2025 STARGA Inc.
- `mindc.rs` (~4413 tok, huge) — Copyright 2025 STARGA Inc.
### `src/`

- `conformance.rs` (~1871 tok, huge)
### `src/diagnostics/`

- `mod.rs` (~2230 tok, huge) — Copyright 2025 STARGA Inc.
### `src/eval/`

- `autodiff.rs` (~13879 tok, huge) — Copyright 2025 STARGA Inc.
- `conv2d_grad.rs` (~2397 tok, huge) — Copyright 2025 STARGA Inc.
- `ir_interp.rs` (~2823 tok, huge) — Copyright 2025 STARGA Inc.
- `lower.rs` (~3782 tok, huge) — Copyright 2025 STARGA Inc.
- `mlir_build.rs` (~2563 tok, huge) — Copyright 2025 STARGA Inc.
- `mlir_export.rs` (~8194 tok, huge) — Copyright 2025 STARGA Inc.
- `mlir_gpu.rs` (~301 tok, medium) — Copyright 2025 STARGA Inc.
- `mlir_jit.rs` (~501 tok, large) — Copyright 2025 STARGA Inc.
- `mlir_opt.rs` (~902 tok, large) — Copyright 2025 STARGA Inc.
- `mlir_run.rs` (~1535 tok, huge) — Copyright 2025 STARGA Inc.
- `mod.rs` (~15758 tok, huge) — Copyright 2025 STARGA Inc.
### `src/eval/stdlib/`

- `mod.rs` (~169 tok, small) — Copyright 2025 STARGA Inc.
- `tensor.rs` (~8346 tok, huge) — Copyright 2025 STARGA Inc.
### `src/eval/`

- `value.rs` (~1770 tok, huge) — Copyright 2025 STARGA Inc.
### `src/exec/`

- `conv.rs` (~435 tok, medium) — Copyright 2025 STARGA Inc.
- `cpu.rs` (~3569 tok, huge) — Copyright 2025 STARGA Inc.
- `mod.rs` (~587 tok, large) — Copyright 2025 STARGA Inc.
### `src/ffi/`

- `header.rs` (~413 tok, medium) — Copyright 2025 STARGA Inc.
- `mod.rs` (~1725 tok, huge) — Copyright 2025 STARGA Inc.
- `sys.rs` (~1772 tok, huge) — Copyright 2025-2026 STARGA Inc.
### `src/ir/compact/`

- `emit.rs` (~4112 tok, huge) — Copyright 2025 STARGA Inc.
- `mod.rs` (~2332 tok, huge) — Copyright 2025 STARGA Inc.
- `parse.rs` (~7446 tok, huge) — Copyright 2025 STARGA Inc.
### `src/ir/compact/v2/`

- `binary.rs` (~4502 tok, huge) — Copyright 2025 STARGA Inc.
- `emit.rs` (~1581 tok, huge) — Copyright 2025 STARGA Inc.
- `mod.rs` (~1208 tok, large) — Copyright 2025 STARGA Inc.
- `parse.rs` (~3132 tok, huge) — Copyright 2025 STARGA Inc.
- `types.rs` (~3774 tok, huge) — Copyright 2025 STARGA Inc.
- `varint.rs` (~1599 tok, huge) — Copyright 2025 STARGA Inc.
### `src/ir/`

- `mod.rs` (~1670 tok, huge) — Copyright 2025 STARGA Inc.
- `print.rs` (~2022 tok, huge) — Copyright 2025 STARGA Inc.
- `verify.rs` (~1947 tok, huge) — Copyright 2025 STARGA Inc.
### `src/`

- `lib.rs` (~872 tok, large) — Copyright 2025 STARGA Inc.
- `linalg.rs` (~2025 tok, huge) — Copyright 2025 STARGA Inc.
- `main.rs` (~6507 tok, huge) — Copyright 2025 STARGA Inc.
### `src/mlir/`

- `lowering.rs` (~4299 tok, huge) — Copyright 2025 STARGA Inc.
- `mod.rs` (~270 tok, medium) — Copyright 2025 STARGA Inc.
### `src/ops/`

- `core_v1.rs` (~1823 tok, huge) — Copyright 2025 STARGA Inc.
- `mod.rs` (~224 tok, medium) — Copyright 2025 STARGA Inc.
### `src/opt/`

- `fold.rs` (~824 tok, large) — Copyright 2025 STARGA Inc.
- `ir_canonical.rs` (~1415 tok, large) — Copyright 2025 STARGA Inc.
- `memory_layout.rs` (~4050 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `mod.rs` (~180 tok, small) — Copyright 2025 STARGA Inc.
### `src/package/`

- `manifest.rs` (~310 tok, medium) — Copyright 2025 STARGA Inc.
- `mod.rs` (~1358 tok, large) — Copyright 2025 STARGA Inc.
### `src/parser/`

- `mod.rs` (~15118 tok, huge) — Copyright 2025 STARGA Inc.
### `src/`

- `pipeline.rs` (~1853 tok, huge) — Copyright 2025 STARGA Inc.
### `src/project/`

- `mod.rs` (~9268 tok, huge) — Copyright 2025 STARGA Inc.
### `src/`

- `python.rs` (~1037 tok, large) — Copyright 2025 STARGA Inc.
### `src/runtime/`

- `gpu.rs` (~288 tok, medium) — Experimental GPU backend contract for MIND.
### `src/`

- `runtime_interface.rs` (~573 tok, large) — Describes a tensor visible to the runtime.
### `src/runtime/`

- `mod.rs` (~92 tok, small) — Runtime abstractions for execution backends.
- `types.rs` (~588 tok, large) — Shared runtime surface types for execution backends.
### `src/shapes/`

- `engine.rs` (~1882 tok, huge) — A rank-N tensor shape represented as a list of extents.
- `mod.rs` (~4169 tok, huge) — Copyright 2025 STARGA Inc.
### `src/stdlib/`

- `mod.rs` (~169 tok, small) — Copyright 2025 STARGA Inc.
- `tensor.rs` (~391 tok, medium) — Copyright 2025 STARGA Inc.
### `src/type_checker/`

- `mod.rs` (~15239 tok, huge) — Copyright 2025 STARGA Inc.
### `src/types/`

- `infer.rs` (~448 tok, medium) — Copyright 2025 STARGA Inc.
- `intern.rs` (~1554 tok, huge) — Copyright 2025 STARGA Inc.
- `mod.rs` (~880 tok, large) — Copyright 2025 STARGA Inc.
- `value.rs` (~297 tok, medium) — Copyright 2025 STARGA Inc.
### `tests/autodiff/`

- `matmul_gradient.mind` (~167 tok, small) — Autodiff test: MatMul gradient computation
### `tests/`

- `autodiff_preview.rs` (~398 tok, medium) — Copyright 2025 STARGA Inc.
- `autodiff.rs` (~1112 tok, large) — Gradient for x*x accumulates two paths: d/dx (x*x) = x + x.
### `tests/autodiff/`

- `simple_gradient.mind` (~80 tok, small) — Autodiff test: Simple scalar gradient
### `tests/backend/`

- `cpu_available.mind` (~52 tok, small) — Backend test: CPU backend availability
- `gpu_graceful_failure.mind` (~73 tok, small) — Backend test: GPU backend graceful failure
### `tests/`

- `cli_buffers.rs` (~443 tok, medium) — Copyright 2025 STARGA Inc.
- `cli_build.rs` (~648 tok, large) — Copyright 2025 STARGA Inc.
- `cli_eval.rs` (~523 tok, large) — Copyright 2025 STARGA Inc.
- `cli_exec.rs` (~543 tok, large) — Copyright 2025 STARGA Inc.
- `cli_tensor.rs` (~469 tok, medium) — Copyright 2025 STARGA Inc.
### `tests/conformance/cpu_baseline/`

- `autodiff_pairwise.grad.ir` (~20 tok, tiny) — module {
- `autodiff_pairwise.ir` (~15 tok, tiny) — module {
- `autodiff_pairwise.mind` (~18 tok, tiny)
- `autodiff_pairwise.mlir` (~25 tok, tiny) — module {
- `autodiff_pairwise.runtime` (~1 tok, tiny) — 0
- `simple_arith.ir` (~15 tok, tiny) — module {
- `simple_arith.mind` (~3 tok, tiny)
- `simple_arith.mlir` (~25 tok, tiny) — module {
- `simple_arith.runtime` (~1 tok, tiny) — 7
### `tests/conformance/gpu_profile/`

- `backend_unavailable.error` (~9 tok, tiny) — no backend available for target gpu
- `backend_unavailable.mind` (~2 tok, tiny)
### `tests/`

- `conformance.rs` (~149 tok, small)
- `CONFORMANCE_TESTS.md` (~1225 tok, large) — MIND Conformance Test Corpus
- `const_folding.rs` (~246 tok, medium) — Copyright 2025 STARGA Inc.
- `conv2d_exec.rs` (~578 tok, large) — Copyright 2025 STARGA Inc.
- `conv2d_grad.rs` (~3194 tok, huge) — Copyright 2025 STARGA Inc.
- `conv2d_types.rs` (~366 tok, medium) — Copyright 2025 STARGA Inc.
- `diagnostics_parse.rs` (~359 tok, medium) — Copyright 2025 STARGA Inc.
- `diagnostics.rs` (~688 tok, large) — Copyright 2025 STARGA Inc.
- `dot_variants.rs` (~284 tok, medium) — Copyright 2025 STARGA Inc.
- `exec_basic.rs` (~785 tok, large) — Copyright 2025 STARGA Inc.
- `expr_parser.rs` (~307 tok, medium) — Copyright 2025 STARGA Inc.
- `ffi_header.rs` (~221 tok, medium) — Copyright 2025 STARGA Inc.
### `tests/fixtures/`

- `autodiff.mind` (~18 tok, tiny)
- `invalid_broadcast.mind` (~17 tok, tiny)
- `invalid.mind` (~6 tok, tiny)
- `simple.mind` (~3 tok, tiny)
### `tests/`

- `gather_preview.rs` (~288 tok, medium) — Copyright 2025 STARGA Inc.
- `if_expr.rs` (~429 tok, medium) — Copyright 2025 STARGA Inc.
- `index_slice_grad.rs` (~289 tok, medium) — Copyright 2025 STARGA Inc.
- `index_slice_preview.rs` (~376 tok, medium) — Copyright 2025 STARGA Inc.
- `index_slice_types.rs` (~248 tok, medium) — Copyright 2025 STARGA Inc.
- `ir_core.rs` (~847 tok, large) — Ensure the unused const is kept alive in the SSA namespace but removed from code.
- `ir_lower.rs` (~325 tok, medium) — Copyright 2025 STARGA Inc.
- `ir_stub.rs` (~219 tok, medium) — Copyright 2025 STARGA Inc.
### `tests/ir_verification/`

- `ssa_single_assignment.mind` (~46 tok, tiny) — IR verification test: SSA property validation
- `undefined_operand.mind` (~62 tok, small) — IR verification test: Undefined operand detection
### `tests/lexical/`

- `invalid_keywords_as_identifiers.mind` (~45 tok, tiny) — Lexical test: Keywords cannot be used as identifiers
- `numeric_literals.mind` (~74 tok, small) — Lexical test: Numeric literal formats
- `valid_identifiers.mind` (~72 tok, small) — Lexical test: Valid identifier formats
### `tests/`

- `linalg_grad.rs` (~315 tok, medium) — Copyright 2025 STARGA Inc.
- `linalg_preview.rs` (~291 tok, medium) — Copyright 2025 STARGA Inc.
- `method_call.rs` (~397 tok, medium) — Copyright 2025 STARGA Inc.
- `mindc.rs` (~1949 tok, huge) — Copyright 2025 STARGA Inc.
- `mlir_build.rs` (~1272 tok, large) — Copyright 2025 STARGA Inc.
- `mlir_exec.rs` (~818 tok, large) — Copyright 2025 STARGA Inc.
- `mlir_export_indexing.rs` (~414 tok, medium) — Copyright 2025 STARGA Inc.
- `mlir_export_linalg.rs` (~338 tok, medium) — Copyright 2025 STARGA Inc.
- `mlir_export_reductions.rs` (~335 tok, medium) — Copyright 2025 STARGA Inc.
- `mlir_export.rs` (~328 tok, medium) — Copyright 2025 STARGA Inc.
- `mlir_export_shape.rs` (~348 tok, medium) — Copyright 2025 STARGA Inc.
- `mlir_file_and_lower.rs` (~639 tok, large) — Copyright 2025 STARGA Inc.
- `mlir_gpu.rs` (~314 tok, medium) — Copyright 2025 STARGA Inc.
- `mlir_jit.rs` (~285 tok, medium) — Copyright 2025 STARGA Inc.
- `mlir_lowering.rs` (~1491 tok, large)
- `mlir_opt.rs` (~424 tok, medium) — Copyright 2025 STARGA Inc.
- `ops_registry.rs` (~114 tok, small)
- `package_basic.rs` (~483 tok, medium) — Copyright 2025 STARGA Inc.
- `pipeline.rs` (~1351 tok, large) — Copyright 2025 STARGA Inc.
- `reductions_grad.rs` (~390 tok, medium) — Copyright 2025 STARGA Inc.
- `reductions_preview.rs` (~390 tok, medium) — Copyright 2025 STARGA Inc.
- `relu_exec.rs` (~472 tok, medium) — Copyright 2025 STARGA Inc.
- `relu_preview.rs` (~279 tok, medium) — Copyright 2025 STARGA Inc.
- `repl_basic.rs` (~543 tok, large) — Copyright 2025 STARGA Inc.
### `tests/runtime/`

- `elementwise_add.mind` (~68 tok, small) — Runtime test: Element-wise addition execution
- `reduction_sum.mind` (~67 tok, small) — Runtime test: Reduction sum operation
### `tests/`

- `shape_integration.rs` (~409 tok, medium)
- `shape_ops_preview.rs` (~302 tok, medium) — Copyright 2025 STARGA Inc.
### `tests/shapes/`

- `broadcast_compatible.mind` (~77 tok, small) — Shape test: Compatible broadcasting
- `broadcast_incompatible.mind` (~76 tok, small) — Shape test: Incompatible broadcasting
### `tests/`

- `shapes_engine.rs` (~699 tok, large) — Rank-0 scalar represented as an empty shape.
### `tests/shapes/`

- `matmul_shapes.mind` (~107 tok, small) — Shape test: MatMul shape inference
### `tests/`

- `shapes.rs` (~1132 tok, large) — Copyright 2025 STARGA Inc.
- `smoke.rs` (~259 tok, medium) — Copyright 2025 STARGA Inc.
- `stdlib_tensor.rs` (~256 tok, medium) — Copyright 2025 STARGA Inc.
- `stride_gather_grad.rs` (~312 tok, medium) — Copyright 2025 STARGA Inc.
- `stride_preview.rs` (~279 tok, medium) — Copyright 2025 STARGA Inc.
- `stride_types.rs` (~250 tok, medium) — Copyright 2025 STARGA Inc.
- `tensor_broadcast.rs` (~1004 tok, large) — Copyright 2025 STARGA Inc.
- `tensor_buffers.rs` (~517 tok, large) — Copyright 2025 STARGA Inc.
- `tensor_eval.rs` (~457 tok, medium) — Copyright 2025 STARGA Inc.
- `tensor_stdlib.rs` (~549 tok, large) — Copyright 2025 STARGA Inc.
- `tensor_symbolic.rs` (~550 tok, large) — Copyright 2025 STARGA Inc.
- `transpose_preview.rs` (~269 tok, medium) — Copyright 2025 STARGA Inc.
- `type_ann_check.rs` (~330 tok, medium) — Copyright 2025 STARGA Inc.
- `type_ann_parse.rs` (~243 tok, medium) — Copyright 2025 STARGA Inc.
- `typecheck_binary.rs` (~327 tok, medium) — Copyright 2025 STARGA Inc.
- `typecheck_env.rs` (~246 tok, medium) — Copyright 2025 STARGA Inc.
### `tests/type_checker/`

- `basic_type_inference.mind` (~66 tok, small) — Type checker test: Basic type inference
- `dtype_mismatch.mind` (~74 tok, small) — Type checker test: Dtype mismatch detection
### `tests/`

- `type_error_spans.rs` (~390 tok, medium) — Copyright 2025 STARGA Inc.
- `type_infer.rs` (~344 tok, medium) — Copyright 2025 STARGA Inc.
- `vars_assign.rs` (~260 tok, medium) — Copyright 2025 STARGA Inc.
- `verify_audit.rs` (~1995 tok, huge) — Audit coverage tests for the IR verifier (C1: SSA verification, conv2d stride/axis validation).
### `tools/`

- `add_copyright_headers.py` (~1132 tok, large) — # Copyright 2025 STARGA Inc.
- `cargo-deny-sanitize.sh` (~572 tok, large) — Run cargo-deny but sanitize advisory entries that older cargo-deny versions
### `.wrangler/cache/`

- `wrangler-account.json` (~21 tok, tiny) — Keys: account

---
*Generated by `anatomy 1.0.0`. Edit descriptions manually — re-run preserves structure.*
