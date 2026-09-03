# ANATOMY.md — Project File Index

> **For coding agents.** Read this before opening files. Use descriptions and token
> estimates to decide whether you need the full file or the summary is enough.
> Re-generate with: `anatomy .`

**Project:** `mind`
**Files:** 1438 | **Est. tokens:** ~2,723,500
**Generated:** 2026-09-03 08:21 UTC

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
| `./` | 35 | ~32,388 |
| `agents/` | 1 | ~436 |
| `.agents/skills/mindc-development/` | 1 | ~235 |
| `.arch-mind/` | 2 | ~644 |
| `assets/logo/` | 1 | ~453 |
| `audits/` | 6 | ~607 |
| `bench/` | 5 | ~7,574 |
| `benches/` | 28 | ~83,676 |
| `bench/fft/` | 8 | ~8,060 |
| `benchmarks/` | 12 | ~20,412 |
| `benchmarks/autograd_comparison/` | 8 | ~9,411 |
| `benchmarks/cupy_comparison/` | 6 | ~7,733 |
| `benchmarks/determinism/` | 3 | ~4,601 |
| `benchmarks/inference/` | 4 | ~4,008 |
| `benchmarks/jax_comparison/` | 5 | ~4,642 |
| `benchmarks/mojo/` | 8 | ~4,300 |
| `benchmarks/pytorch_comparison/` | 5 | ~4,828 |
| `.cargo/` | 1 | ~130 |
| `config/` | 1 | ~1,465 |
| `docs/` | 36 | ~93,037 |
| `docs/backends/` | 1 | ~1,482 |
| `docs/benchmarks/` | 3 | ~9,315 |
| `docs/design/` | 4 | ~11,498 |
| `docs/mindcraft/` | 3 | ~7,086 |
| `docs/rfcs/` | 35 | ~166,772 |
| `docs/specs/` | 2 | ~976 |
| `examples/` | 28 | ~49,880 |
| `examples/bimap_currency/` | 3 | ~780 |
| `examples/bimap_pairs/` | 2 | ~801 |
| `examples/c/` | 2 | ~400 |
| `examples/columnar/` | 4 | ~7,585 |
| `examples/compliance/` | 3 | ~5,294 |
| `examples/detmath_kat/` | 2 | ~4,435 |
| `examples/distribution-crossisa/` | 6 | ~6,336 |
| `examples/emit_ir/` | 5 | ~13,648 |
| `examples/grammar_mask/` | 2 | ~4,636 |
| `examples/halbach_q16/` | 2 | ~7,856 |
| `examples/lexer/` | 6 | ~8,888 |
| `examples/mindc_mind/` | 165 | ~403,111 |
| `examples/mindc_mind/testdata/` | 3 | ~4,159 |
| `examples/mindc_mind/testdata/backend_native_bridge/` | 2 | ~415 |
| `examples/mindc_mind/testdata/native_elf_oracle/` | 6 | ~913 |
| `examples/mindc_mind/testdata/selfhost_loop/` | 2 | ~1,928 |
| `examples/native/` | 4 | ~527 |
| `examples/parser/` | 5 | ~17,923 |
| `examples/typecheck/` | 5 | ~14,553 |
| `examples/zoo/` | 6 | ~12,518 |
| `experiments/global-vs-local/` | 7 | ~6,487 |
| `.githooks/` | 1 | ~434 |
| `.github/` | 4 | ~601 |
| `.github/ISSUE_TEMPLATE/` | 3 | ~440 |
| `.github/workflows/` | 9 | ~27,333 |
| `mind/std/cognitive/` | 4 | ~3,529 |
| `runtime-support/` | 1 | ~21,838 |
| `scripts/` | 22 | ~48,807 |
| `scripts/mind-vs-rust/` | 3 | ~933 |
| `scripts/mind-vs-rust/src/` | 1 | ~2,372 |
| `scripts/sdlc/` | 2 | ~2,559 |
| `sdk/ts/mic-map/` | 6 | ~16,202 |
| `sdk/ts/mic-map/dist/` | 36 | ~29,044 |
| `sdk/ts/mic-map/scripts/` | 1 | ~499 |
| `sdk/ts/mic-map/src/` | 9 | ~12,181 |
| `sdk/ts/mic-map/test/` | 4 | ~7,843 |
| `sdk/ts/mic-map/test/fixtures/` | 2 | ~96 |
| `skills/write-mind/` | 1 | ~6,002 |
| `src/` | 9 | ~34,565 |
| `src/ast/` | 1 | ~10,930 |
| `src/autodiff/` | 3 | ~6,624 |
| `src/bin/` | 1 | ~9,878 |
| `src/build/` | 2 | ~18,366 |
| `src/cache/` | 4 | ~3,682 |
| `src/check/` | 3 | ~10,829 |
| `src/deps/` | 1 | ~9,345 |
| `src/diagnostics/` | 1 | ~3,719 |
| `src/distributed/` | 6 | ~7,725 |
| `src/doc/` | 3 | ~10,987 |
| `src/eval/` | 16 | ~87,339 |
| `src/eval/stdlib/` | 2 | ~8,586 |
| `src/exec/` | 3 | ~4,592 |
| `src/ffi/` | 3 | ~5,541 |
| `src/fmt/` | 3 | ~21,737 |
| `src/ir/` | 6 | ~78,361 |
| `src/ir/compact/` | 3 | ~15,292 |
| `src/ir/compact/v2/` | 9 | ~45,364 |
| `src/ir/compact/v3/` | 7 | ~63,340 |
| `src/lint/` | 2 | ~4,001 |
| `src/lint/rules/` | 6 | ~9,880 |
| `src/mlir/` | 3 | ~5,905 |
| `src/ops/` | 3 | ~4,764 |
| `src/opt/` | 9 | ~52,964 |
| `src/package/` | 2 | ~1,877 |
| `src/parser/` | 2 | ~20,355 |
| `src/phf/` | 1 | ~4,955 |
| `src/project/` | 4 | ~14,409 |
| `src/runtime/` | 3 | ~1,485 |
| `src/shapes/` | 2 | ~6,052 |
| `src/stdlib/` | 2 | ~560 |
| `src/test/` | 1 | ~8,383 |
| `src/type_checker/` | 1 | ~15,148 |
| `src/types/` | 4 | ~3,336 |
| `src/workspace/` | 1 | ~4,906 |
| `std/` | 42 | ~202,387 |
| `tests/` | 336 | ~588,459 |
| `tests/autodiff/` | 2 | ~247 |
| `tests/backend/` | 2 | ~125 |
| `tests/common/` | 1 | ~668 |
| `tests/conformance/cpu_baseline/` | 9 | ~171 |
| `tests/conformance/gpu_profile/` | 2 | ~11 |
| `tests/cross_substrate_identity/` | 2 | ~4,113 |
| `tests/cross_substrate_identity/array-store-branch/` | 2 | ~974 |
| `tests/cross_substrate_identity/array-store-loop/` | 2 | ~1,081 |
| `tests/cross_substrate_identity/bimap-phf/` | 2 | ~1,665 |
| `tests/cross_substrate_identity/collatz/` | 2 | ~962 |
| `tests/cross_substrate_identity/dot-f32-v-4093/` | 2 | ~1,222 |
| `tests/cross_substrate_identity/dot-i16-4096/` | 2 | ~648 |
| `tests/cross_substrate_identity/dot-l1-q16/` | 2 | ~363 |
| `tests/cross_substrate_identity/dot-l2-q16/` | 2 | ~813 |
| `tests/cross_substrate_identity/galperin-pi/` | 2 | ~1,004 |
| `tests/cross_substrate_identity/gemm-i8-64x64x64/` | 2 | ~707 |
| `tests/cross_substrate_identity/gemm-i8-mt-64x64x64/` | 2 | ~872 |
| `tests/cross_substrate_identity/gemm-i8-vnni-64x64x64/` | 2 | ~921 |
| `tests/cross_substrate_identity/gemm-q16-64x64x64/` | 2 | ~616 |
| `tests/cross_substrate_identity/gemm-q16-fused-64x64x64/` | 2 | ~896 |
| `tests/cross_substrate_identity/gemv-i16-256x256/` | 2 | ~594 |
| `tests/cross_substrate_identity/gemv-q16-256x256/` | 2 | ~519 |
| `tests/cross_substrate_identity/grammar-mask/` | 2 | ~916 |
| `tests/cross_substrate_identity/lorenz-q16/` | 2 | ~1,243 |
| `tests/cross_substrate_identity/matmul-f32-v-64x64/` | 2 | ~1,115 |
| `tests/cross_substrate_identity/q16-arith-chain/` | 2 | ~788 |
| `tests/cross_substrate_identity/scalar-cast-conv/` | 2 | ~1,644 |
| `tests/cross_substrate_identity/scalar-cast-conv-narrow/` | 2 | ~1,790 |
| `tests/cross_substrate_identity/scalar-float-f64/` | 2 | ~1,310 |
| `tests/cross_substrate_identity/struct-handle-roundtrip/` | 2 | ~746 |
| `tests/cross_substrate_identity/u64-ops/` | 2 | ~1,054 |
| `tests/fixtures/` | 6 | ~228 |
| `tests/fixtures/f64_abi_negative/` | 3 | ~192 |
| `tests/ir_verification/` | 2 | ~108 |
| `tests/lexical/` | 3 | ~191 |
| `tests/mindcraft/` | 1 | ~408 |
| `tests/mindcraft/check/` | 4 | ~48 |
| `tests/mindcraft/check/subdir/` | 1 | ~9 |
| `tests/mindcraft/fmt/` | 14 | ~474 |
| `tests/mindcraft/lint/` | 2 | ~21 |
| `tests/mindcraft/lint/naming_convention/` | 4 | ~176 |
| `tests/mindcraft/lint/q16_overflow/` | 3 | ~191 |
| `tests/mindcraft/lint/shadowing/` | 2 | ~87 |
| `tests/mindcraft/lint/unused_import/` | 2 | ~99 |
| `tests/mindfuzz_cross_substrate/known_environmental/` | 2 | ~556 |
| `tests/mindfuzz_cross_substrate/reproducers/` | 2 | ~375 |
| `tests/mindfuzz_cross_substrate/staged/` | 15 | ~2,832 |
| `tests/runtime/` | 2 | ~135 |
| `tests/selfhost_gaps/` | 148 | ~9,289 |
| `tests/selfhost_gaps/never_wrong/` | 9 | ~184 |
| `tests/shapes/` | 3 | ~260 |
| `tests/type_checker/` | 2 | ~140 |
| `tools/` | 5 | ~7,506 |
| `tools/mindfuzz/` | 7 | ~16,763 |
| `tools/mindfuzz/seeds/` | 6 | ~1,330 |
| `tools/mindfuzz/violations/` | 1 | ~0 |
| `tools/pytorch_bridge/` | 6 | ~4,673 |
| `tools/pytorch_bridge/tests/` | 2 | ~1,244 |

## Files

### `./`

- `ARCHITECTURE.md` (~300 tok, medium) — MIND Architecture (high level)
- `AUDIT_REPORT.md` (~1151 tok, large) — Audit Report
- `.bench-baseline-2026-04-27.txt` (~531 tok, large) —    Compiling mind v0.2.3 (.)
- `.bench-baseline-2026-04-28-pratt.txt` (~185 tok, small) — === Pratt parser baseline (mindc 0.2.5, 2026-04-28) ===
- `.bench-baseline-2026-05-17-phase10-6.txt` (~408 tok, medium) — === Phase 10.6 surface-syntax baseline (mindc 0.2.10, 2026-05-17) ===
- `.bench-baseline-2026-05-17-phase10-7.txt` (~565 tok, large) — === Phase 10.7 surface baseline (mindc 0.2.11, 2026-05-17) ===
- `.bench-baseline-2026-05-18-rfc0005.txt` (~781 tok, large) — === RFC 0005 Phase 2 baseline (mindc 0.4.0, 2026-05-18) ===
- `.bench-baseline-2026-06-01-correctness.txt` (~784 tok, large) — === Correctness-milestone baseline (mindc 0.7.0, 2026-06-01) ===
- `.bench-pre-pratt.txt` (~32 tok, tiny) — === captured pre-Pratt baseline (Phase 10.5 in main) ===
- `bounties.md` (~888 tok, large) — MIND Bounty Board
- `build.rs` (~234 tok, medium) — Copyright 2025 STARGA Inc.
- `Cargo.toml` (~3137 tok, huge) — [package]
- `clippy.toml` (~25 tok, tiny)
- `CODE_OF_CONDUCT.md` (~29 tok, tiny) — Code of Conduct
- `COMPLETE_FILE_STRUCTURE.md` (~26 tok, tiny) — Repository Structure (Snapshot)
- `CONTRIBUTING.md` (~1891 tok, huge) — Contributing to MIND
- `deny.toml` (~89 tok, small) — [advisories]
- `.editorconfig` (~51 tok, small) — root = true
- `.gitattributes` (~130 tok, small) — # Enforce LF line endings for all text so byte-exact tests (fmt idempotence,
- `GITHUB_SETUP_INSTRUCTIONS.md` (~240 tok, medium) — GitHub Setup (Quick)
- `.gitignore` (~813 tok, large) — # Rust
- `HANDOFF-2026-09-01.md` (~3795 tok, huge) — Handoff — MIND compiler, branch `fix/tc-let-null-annotation`
- `HANDOFF-2026-09-02-S1-GLIBC.md` (~501 tok, large) — Handoff for CLAUDE (compiler owner) — S1 Pure-MIND ABI blocker
- `incompatible` (~0 tok, tiny)
- `LICENSE` (~2573 tok, huge) —                                  Apache License
- `LICENSE-COMMERCIAL` (~399 tok, medium) — COMMERCIAL LICENSE NOTICE – MIND (Enterprise & SaaS)
- `Mind.toml` (~108 tok, small) — [package]
- `plugin.json` (~62 tok, small) — Keys: name, description, version, skills, agents
- `README.md` (~6174 tok, huge) — MIND — Machine Intelligence Native Design
- `RELEASING.md` (~131 tok, small) — Release checklist (as of v0.2.1)
- `rustfmt.toml` (~23 tok, tiny) — max_width = 100
- `SECURITY.md` (~1538 tok, huge) — Security Policy
- `.sembleignore` (~72 tok, small) — # semble code-search ignore list
- `STATUS.md` (~4457 tok, huge) — MIND Compiler Status
- `test_real_compile_time.py` (~265 tok, medium) — Quick test of real MIND compilation time using Python bindings."""
### `agents/`

- `mind-developer.md` (~436 tok, medium) — MIND Developer Agent
### `.agents/skills/mindc-development/`

- `SKILL.md` (~235 tok, medium) — MIND Compiler (mindc) Development
### `.arch-mind/`

- `rules.mind` (~557 tok, large) — mind (language compiler / runtime root) architectural-governance rules
- `scan.json` (~87 tok, small) — Keys: _fixture, acyclicity_q16, depth_q16, equality_q16, evidence_chain_density
### `assets/logo/`

- `README.md` (~453 tok, medium) — MIND logo assets
### `audits/`

- `arch-mind-2026-05-18-post-phase-6-1.json` (~169 tok, small) — Keys: _fixture, acyclicity_q16, depth_q16, equality_q16, evidence_chain_density
- `arch-mind-v0.4.0.json` (~86 tok, small) — Keys: _fixture, acyclicity_q16, depth_q16, equality_q16, evidence_chain_density
- `arch-mind-v0.4.1.json` (~88 tok, small) — Keys: _fixture, acyclicity_q16, depth_q16, equality_q16, evidence_chain_density
- `arch-mind-v0.4.2.json` (~88 tok, small) — Keys: _fixture, acyclicity_q16, depth_q16, equality_q16, evidence_chain_density
- `arch-mind-v0.4.3.json` (~88 tok, small) — Keys: _fixture, acyclicity_q16, depth_q16, equality_q16, evidence_chain_density
- `arch-mind-v0.4.4.json` (~88 tok, small) — Keys: _fixture, acyclicity_q16, depth_q16, equality_q16, evidence_chain_density
### `bench/`

- `beat_clang_igemm_driver.c` (~494 tok, medium)
### `benches/`

- `autodiff.rs` (~1661 tok, huge) — Simple linear function
- `bench_aes_gcm.rs` (~2590 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `bench_ecdsa_p256.rs` (~2786 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `bench_hkdf.rs` (~4424 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `bench_hpack.rs` (~3927 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `bench_http2_frame.rs` (~5035 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `bench_keccak.rs` (~2576 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `bench_mlkem768.rs` (~3700 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `bench_rsa_pss.rs` (~3317 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `bench_sha256.rs` (~2594 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `bench_tls13_record.rs` (~6121 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `bench_x25519.rs` (~2469 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `bench_x509.rs` (~3990 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `cerebras_stencil.rs` (~831 tok, large) — Copyright 2025-2026 STARGA Inc.
- `compiler.rs` (~3782 tok, huge) — Small program: Simple matrix multiplication
- `cross_module.rs` (~609 tok, large) — Copyright 2025 STARGA Inc.
- `det_gemv_q16_mt.rs` (~3294 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `det_matmul_i16.rs` (~4621 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `det_matmul_i8.rs` (~5094 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `det_matmul_q16_mt.rs` (~4049 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `det_matmul_q16.rs` (~4972 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `fft_q16.rs` (~5352 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `mindcraft_fmt.rs` (~908 tok, large) — File readers
- `operations.rs` (~1076 tok, large) — Element-wise operations
- `parser_throughput.rs` (~916 tok, large) — Copyright 2025 STARGA Inc.
- `shapes.rs` (~1208 tok, large) — Simple broadcasting scenarios
- `simple_benchmarks.rs` (~707 tok, large) — Mirror mindc's allocator so this compile-speed bench measures the same heap
- `std_surface.rs` (~1067 tok, large) — Copyright 2025 STARGA Inc.
### `bench/fft/`

- `build.sh` (~986 tok, large) — build.sh — self-contained build for the deterministic Q16.16 N=256 FFT bench.
- `fft_driver.c` (~1205 tok, large) — Standalone correctness + timing driver for the C reference Q16.16 FFT.
- `fft_ref.c` (~473 tok, medium) — Q16.16 deterministic radix-2 DIT FFT, N=256 — BYTE-IDENTICAL algorithm to
- `fft_verify.c` (~889 tok, large) — Cross-check harness: load the MIND-compiled fft256 from a .so and assert its
- `.gitignore` (~38 tok, tiny) — # Build artifacts — regenerated by build.sh, never committed.
- `harness.c` (~1485 tok, large) — Self-contained benchmark harness for the deterministic Q16.16 N=256 FFT.
- `README.md` (~1677 tok, huge) — Deterministic Q16.16 N=256 FFT — MIND vs gcc / clang / nvcc
- `RESULTS-fft-2026-06-15.md` (~1307 tok, large) — RESULTS — Deterministic Q16.16 N=256 FFT (MIND vs gcc / clang / nvcc)
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

- `BENCHMARK_RESULTS.md` (~4311 tok, huge) — MIND Benchmark Results
### `benchmarks/cupy_comparison/`

- `leg1_determinism.py` (~2586 tok, huge)
- `leg1_determinism_results.json` (~1451 tok, large) — Keys: leg, host, mind, cupy
- `leg2_perf.py` (~1510 tok, huge)
- `leg2_perf_results.json` (~473 tok, medium) — Keys: leg, host, config, mind, status
- `README.md` (~1619 tok, huge) — CuPy Comparison Benchmark
- `requirements.txt` (~94 tok, small) — # Leg 1 (determinism) + Leg 2 (perf) foil dependencies.
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
- `MIC_MAP_BENCHMARK_README.md` (~335 tok, medium) — MIC/MAP Patent Reference Benchmark
- `mic_map_benchmark_results.json` (~851 tok, large) — Keys: metadata, measurements, paper_figures_verified, claim_checks, all_claims_verified
- `mic_map_benchmark_v2.py` (~3150 tok, huge)
### `benchmarks/mojo/`

- `benchmark_mojo_compilation.py` (~1533 tok, huge)
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

- `README.md` (~1188 tok, large) — MIND Performance Benchmarks
- `resnet.md` (~74 tok, small) — ResNet Benchmarks (Preliminary)
- `run_all_benchmarks.sh` (~824 tok, large) — Master script to run all MIND patent benchmarks
- `RUN_GUIDE.md` (~1465 tok, large) — MIND Patent Benchmarks - Environment Guide
- `scientific_benchmark.py` (~1639 tok, huge)
- `scientific_benchmark_raw.py` (~2485 tok, huge)
### `bench/`

- `matmul_det_bench.mind` (~1079 tok, large) — bench/matmul_det_bench.mind — first pure-MIND runtime benchmark for the
- `RESULTS-beat-clang-igemm-2026-08-21.md` (~839 tok, large) — MIND vs clang -O3 — int8 GEMM head-to-head (2026-08-21)
- `RESULTS-int8-2026-06-08.md` (~693 tok, large) — MIND int8 VNNI GEMM — single-core vs OpenBLAS f32 (2026-06-08)
- `turboquant.mind` (~4469 tok, huge) — bench/turboquant.mind — KV-cache quantization pipeline in pure MIND.
### `.cargo/`

- `config.toml` (~130 tok, small) — [registries]
### `config/`

- `capabilities.toml` (~1465 tok, large) — [ir]
### `docs/`

- `architecture.md` (~965 tok, large) — Architecture
- `ARRAY_SEMANTICS.md` (~7509 tok, huge) — MIND array semantics — normative architecture record
- `autodiff.md` (~595 tok, large) — Static autodiff (public)
### `docs/backends/`

- `cerebras-stencil.md` (~1482 tok, large) — `mind.cerebras.stencil_tile` — Op Surface and Lowering Contract
### `docs/`

- `benchmarking.md` (~1917 tok, huge) — Benchmarking methodology — tiers and comparable metrics
### `docs/benchmarks/`

- `compiler_performance.md` (~4721 tok, huge) — MIND Compiler Performance Benchmarks
### `docs/`

- `benchmarks.md` (~896 tok, large) — Benchmarks
### `docs/benchmarks/`

- `mojo_comparison.md` (~2420 tok, huge) — MIND vs Mojo: Compilation Performance Comparison
- `RESULTS-mind-vs-rust-2026-06-09.md` (~2174 tok, huge) — MIND vs Rust — integer-GEMM, apples-to-apples (2026-06-09)
### `docs/`

- `byte-store-migration.md` (~3357 tok, huge) — Byte-Store Migration — closing `#306`
- `cli.md` (~747 tok, large) — MIND CLI Reference
### `docs/design/`

- `execution-plan-performance-mode.md` (~8045 tok, huge) — Design: PerformanceMode + ExecutionPlan + ExecutionProvider
- `README.md` (~26 tok, tiny) — Design Docs
- `rfc0012-b2-shape-threading-scope.md` (~3317 tok, huge) — RFC 0012 Phase B.2 — shape-dim threading: execution scope
- `v0.3.md` (~110 tok, small) — MIND Design v0.3 (Draft)
### `docs/`

- `determinism.md` (~4368 tok, huge) — The Determinism Contract
- `errors.md` (~701 tok, large) — MIND Core Error Model
- `ffi-runtime.md` (~529 tok, large) — FFI & Runtime Integration
- `gpu.md` (~387 tok, medium) — GPU backend profile
- `INDEPENDENCE_ROADMAP.md` (~18728 tok, huge) — MIND Rust-Independence Roadmap
- `install.md` (~1012 tok, large) — Installing mindc
- `ir.md` (~451 tok, medium) — MIND IR core
- `ir-mlir.md` (~480 tok, medium) — IR & MLIR Integration
- `ir-stability.md` (~1485 tok, large) — IR stability contract
- `MHS_ROADMAP.md` (~2657 tok, huge) — MHS / Governed Device I/O Roadmap — SUPERSEDED
- `migration-roadmap.md` (~1896 tok, huge) — MIND Migration Roadmap — Any Language → Pure Executing MIND
### `docs/mindcraft/`

- `fmt.md` (~2302 tok, huge) — `mindc fmt` — Canonical Formatter Reference
- `phase2-implementation-plan.md` (~2209 tok, huge) — Mindcraft Phase 2 — Implementation Plan
- `rfc0010-phase-ghi-migration-plan.md` (~2575 tok, huge) — RFC 0010 Phase G/H/I — Migration Plan (corrected against real architecture)
### `docs/`

- `mlir-lowering.md` (~286 tok, medium) — MLIR lowering pipeline (public)
- `ops.md` (~604 tok, large) — Core v1 operator coverage
- `optimization-frontier.md` (~11775 tok, huge) — MIND Optimization Frontier
- `performance.md` (~880 tok, large) — Performance Guide
- `README.md` (~162 tok, small) — MIND Documentation
- `reap-pruning.md` (~901 tok, large) — REAP Expert Pruning
### `docs/rfcs/`

- `0000-template.md` (~627 tok, large) — RFC 0000: [Title]
- `0001-bitnet-native-support.md` (~3254 tok, huge) — RFC 0001: Native BitNet Support — `tri` and `q16_16` Types
- `0002-pub-fn-c-exports.md` (~2084 tok, huge) — RFC 0002: `pub fn` → C ABI Symbol Export
- `0003-cdylib-aot-emit.md` (~3195 tok, huge) — RFC 0003: cdylib AOT emit + symbol versioning
- `0004-evidence-token-types.md` (~1912 tok, huge) — RFC 0004: Compile-Time Evidence Token Types
- `0005-phase-6-2-mindc-gaps.md` (~3356 tok, huge) — RFC 0005 Phase 6.2 — mindc Feature Gaps (Design Note)
- `0005-phase-d2b-design-note.md` (~1518 tok, huge) — RFC 0005 Phase D₂b — Cross-arg Named-struct identity matching
- `0005-pure-mind-std-surface.md` (~5516 tok, huge) — RFC 0005: Pure-MIND Standard Surface
- `0006-mind-blas.md` (~5743 tok, huge) — RFC 0006: mind-blas — native BLAS surface for MIND
- `0007-mindcraft.md` (~4499 tok, huge) — RFC 0007: Mindcraft — the pure-MIND format / lint / check toolchain
- `0008-mindc-build.md` (~10964 tok, huge) — RFC 0008: mindc build + mindc test — retiring cargo from the build path
- `0009-federation-package-layer.md` (~6976 tok, huge) — RFC 0009: Federation-First MIND Package Layer
- `000-template.md` (~1 tok, tiny)
- `0010-memory-safety-and-c-abi.md` (~7359 tok, huge) — RFC 0010: Memory Safety Model + C ABI in Pure MIND
- `0011-async-and-structured-concurrency.md` (~4891 tok, huge) — RFC 0011: Async + Structured Concurrency Model
- `0012-tensor-native-syntax.md` (~11504 tok, huge) — RFC 0012: Tensor-Native Surface Syntax — the Differentiation Layer
- `0013-cli-agent-harness-stack.md` (~6776 tok, huge) — RFC 0013: CLI Agent Harness Stack
- `0014-per-substrate-mlir-lowering-contracts.md` (~5412 tok, huge) — RFC 0014: Per-Substrate MLIR Lowering Pipeline Contracts
- `0015-cross-substrate-bit-identity.md` (~5174 tok, huge) — RFC 0015: Cross-Substrate Bit-Identity Proof Obligation
- `0016-evidence-chain-emission.md` (~6998 tok, huge) — RFC 0016: Compile-Time Evidence-Chain Emission
- `0017-mindc-verify.md` (~4360 tok, huge) — RFC 0017: `mindc verify` — Artifact Verification Surface
- `0018-bare-metal-substrate.md` (~3799 tok, huge) — RFC 0018: Bare-Metal Substrate Lowering Tier
- `0019-deterministic-agent-substrate.md` (~4131 tok, huge) — RFC 0019: Deterministic Agent Substrate
- `0020-mind-bench-reproducibility-harness.md` (~4083 tok, huge) — RFC 0020: mind-bench Public Reproducibility Harness
- `0021-canonical-ir-unification.md` (~4487 tok, huge) — RFC 0021: Canonical IR Unification — one IR, provenance as a versioned epilogue
- `0022-deterministic-io-substrate.md` (~2120 tok, huge) — RFC 0022: Deterministic I/O Substrate — fastest async I/O with bit-identical replay
- `0024-loop-collapse.md` (~7579 tok, huge) — RFC 0024: Loop Collapse — prove-or-fail closed-form replacement of counted loops (`#[collapse]`)
- `0025-mind-intent-contracts.md` (~3605 tok, huge) — RFC 0025: MIND Intent — Intent Contracts (goal + constraints → verifiable Contract IR)
- `0027-governed-physical-device-plane.md` (~2790 tok, huge) — RFC 0027: Governed Physical-Device Plane — evidence-bound actuation (`device_receipts`)
- `0028-deterministic-field-calculus.md` (~10282 tok, huge) — RFC 0028: Deterministic Field Calculus
- `DRAFT-deterministic-format-frontend.md` (~10522 tok, huge) — RFC DRAFT: Deterministic Multi-Format Ingest Front-End (JSON / TOON / CSV / TSV / NDJSON / TOML)
- `DRAFT-deterministic-json-frontend.md` (~5177 tok, huge) — RFC DRAFT: Deterministic Streaming SIMD JSON Structural Front-End
- `DRAFT-governed-device-io-mhs.md` (~5629 tok, huge) — Draft RFC: Governed Device I/O and MHS Compatibility — SUPERSEDED
- `odc-language-primitives.md` (~418 tok, medium) — RFC: Observer-Dependent Cognition — Language Primitives
- `README.md` (~31 tok, tiny) — RFCs
### `docs/`

- `RI_DEPENDENCY_MATRIX.md` (~6115 tok, huge) — Rust-Independence (RI) Dependency Matrix
- `runs-burndown-roadmap.md` (~3203 tok, huge) — MIND RUNS Burndown Roadmap
- `security.md` (~1492 tok, large) — Security Guide
- `self-evolution.md` (~2062 tok, huge) — Self-Evolution in MIND
- `self-host-trace-hash-port.md` (~1406 tok, large) — #17 — Self-compute the native PT_NOTE (pure-MIND trace-hash port)
- `shapes.md` (~478 tok, medium) — Tensor shape semantics
- `sparse-tensor-types.md` (~740 tok, large) — Sparse Tensor Types
### `docs/specs/`

- `README.md` (~23 tok, tiny) — Specifications
- `v1.0.md` (~953 tok, large) — MIND Language Specification v1.0 (Working Draft)
### `docs/`

- `type-system.md` (~1082 tok, large) — Type System
- `VERIFICATION_APPARATUS.md` (~7783 tok, huge) — Self-Host Port Verification Apparatus & SOTA Roadmap
- `versioning.md` (~804 tok, large) — MIND Core Stability & Versioning
- `version-matrix.md` (~1796 tok, huge) — MIND Ecosystem — Version Matrix
- `whitepaper.md` (~2788 tok, huge) — MIND: The Native Language for Intelligent Systems
### `examples/`

- `anthropobrot.mind` (~3257 tok, huge) — Anthropobrot: depth-selected orbit-density multisets of the Fatou-Julia iteral.
- `autodiff_demo.mind` (~1715 tok, huge) — Autodiff Demonstration
### `examples/bimap_currency/`

- `.gitignore` (~2 tok, tiny) — target/
- `main.mind` (~716 tok, large) — Single-source bijective map over a "nice set" — ONE declaration, both
- `Mind.toml` (~62 tok, small) — [package]
### `examples/bimap_pairs/`

- `main.mind` (~732 tok, large) — Single-source bijective const pair-tables — ONE declaration, both directions
- `Mind.toml` (~69 tok, small) — [package]
### `examples/c/`

- `min.c` (~82 tok, small)
- `mind.h` (~318 tok, medium) — Copyright 2025 STARGA Inc.
### `examples/`

- `cnn_classifier.mind` (~1060 tok, large) — CNN Classifier Example
- `collatz.mind` (~495 tok, medium) — Deterministic integer Collatz (3n+1) iterator — the integer sibling of the
### `examples/columnar/`

- `structural_scan_json.mind` (~1703 tok, huge) — examples/columnar/structural_scan_json.mind
- `structural_scan_test.py` (~1541 tok, huge) — Runnable verification for examples/columnar/structural_scan_json.mind.
- `tiled_fold.mind` (~1784 tok, huge) — examples/columnar/tiled_fold.mind
- `tiled_fold_test.py` (~2557 tok, huge) — Runnable verification for examples/columnar/tiled_fold.mind.
### `examples/compliance/`

- `auditable_model.mind` (~1932 tok, huge) — auditable_model.mind -- Compliance-Ready MLP with Provenance Metadata
- `audit_report.mind` (~2289 tok, huge) — audit_report.mind -- Compliance Artifact Generation
- `README.md` (~1073 tok, large) — Compliance Example
### `examples/`

- `cos_dottie.mind` (~781 tok, large) — Cosine-map iteration toward the Dottie fixed point (x* ≈ 0.7390851332151607),
### `examples/detmath_kat/`

- `main.mind` (~4415 tok, huge) — examples/detmath_kat — value-oracle known-answer tests for std.detmath.
- `Mind.toml` (~20 tok, tiny) — [package]
### `examples/distribution-crossisa/`

- `afterkelly.cpp` (~2278 tok, huge) — Command line arguments. ____________________________________________
- `data1.txt` (~212 tok, medium) — 45.96
- `data2.txt` (~223 tok, medium) — 107.50
- `distribution.cpp` (~1217 tok, large)
- `distribution_interp_f64.mind` (~1231 tok, large) — Deterministic IEEE-754 float64 piecewise-LINEAR density interpolation kernel,
- `README.md` (~1175 tok, large) — Cross-ISA determinism: a piecewise-linear density kernel
### `examples/`

- `dottie_collapse.mind` (~1144 tok, large) — Salov loop-collapse — Q16.16 fixed-point ITERATION collapse (Slice S3).
### `examples/emit_ir/`

- `bootstrap_smoke.py` (~2890 tok, huge)
- `EXPECTED.md` (~1942 tok, huge) — Phase 6.4 — Expected IR Text
- `fixture.mind` (~183 tok, small) — Phase 6.4 emit_ir smoke fixture.
- `main.mind` (~6419 tok, huge) — examples/emit_ir/main.mind — RFC 0005 Phase 6.4 self-host MLIR text emitter.
- `README.md` (~2214 tok, huge) — RFC 0005 Phase 6.4 — Self-Host MLIR Text Emitter
### `examples/`

- `fft_q16.mind` (~1248 tok, large) — Deterministic Q16.16 fixed-point radix-2 DIT FFT, N=256 (complex).
- `fft_signal.mind` (~533 tok, large) — FFT Signal Processing Example for MIND
- `galperin_pi.mind` (~1486 tok, large) — Galperin's billiard-π: count elastic collisions of two balls + a wall to
- `gauss_collapse.mind` (~672 tok, large) — Salov loop-collapse — closed-form affine sums (Slice S1).
- `geometric_collapse.mind` (~807 tok, large) — Salov loop-collapse — geometric powering closed forms (Slice S2).
### `examples/grammar_mask/`

- `main.mind` (~4577 tok, huge) — examples/grammar_mask/main.mind — structured / grammar-constrained decoding,
- `Mind.toml` (~59 tok, small) — [package]
### `examples/halbach_q16/`

- `main.mind` (~7793 tok, huge) — examples/halbach_q16/main.mind — standalone, SELF-VERIFYING project build of
### `examples/`

- `halbach_q16.mind` (~3965 tok, huge) — Deterministic Q16.16 2D Halbach-vs-uniform magnet-array field model.
### `examples/halbach_q16/`

- `Mind.toml` (~63 tok, small) — [package]
### `examples/`

- `hello_stdlib.mind` (~271 tok, medium) — Hello, std.vec — minimal RFC 0005 cookbook example.
- `hello_tensor.mind` (~141 tok, small) — Hello, MIND — scalar smoke that flows through every stage of the
### `examples/lexer/`

- `bootstrap_smoke.py` (~2367 tok, huge)
- `BOOTSTRAP_SMOKE_REPORT.md` (~1931 tok, huge) — Phase 6.5 Stage 1 — Bootstrap Smoke Report
- `EXPECTED.md` (~1093 tok, large) — Phase 6.1 — Expected Token Stream
- `fixture.mind` (~67 tok, small) — Phase 6.1 lexer smoke fixture.
- `main.mind` (~2461 tok, huge) — examples/lexer/main.mind — RFC 0005 Phase 6.1 self-host smoke
- `README.md` (~969 tok, large) — RFC 0005 Phase 6.1 — Self-Host Lexer Seed
### `examples/`

- `lorenz_f64.mind` (~230 tok, medium) — Deterministic IEEE-754 float64 Lorenz-attractor integrator (forward Euler).
- `lorenz_q16.mind` (~1091 tok, large) — Deterministic Q16.16 fixed-point Lorenz-attractor integrator (forward Euler).
- `mandelbrot.mind` (~1019 tok, large) — Deterministic IEEE-754 float64 Mandelbrot escape-count renderer.
- `mandelbrot_strict.mind` (~982 tok, large) — Strict-f64 Mandelbrot escape-count checksum — a determinism-wedge demo.
### `examples/mindc_mind/`

- `backend_native_bridge_smoke.py` (~3142 tok, huge) — RI-D slice 1 gate (task #110): `mindc build --backend native` is a byte-faithful
- `bootstrap_smoke.py` (~2473 tok, huge)
- `branch_shadow_crossbackend_smoke.py` (~2039 tok, huge) — #318 — branch-local `let` shadow must not leak across backends (the WEDGE gate).
- `check_driver.mind` (~9782 tok, huge) — ===========================================================================
- `closure_netverify.py` (~1423 tok, large) — # Canonical independent net-verify harness for CLOSURES / FN-VALUES / UNRESOLVED
- `collect_field_strings_smoke.py` (~1161 tok, large)
- `cutover_coverage_measure.py` (~2238 tok, huge)
- `div_shift_cmp_edge_smoke.py` (~1846 tok, huge)
- `enum_netverify.py` (~1957 tok, huge) — # Canonical independent net-verify harness for C-LIKE ENUMS in the native-ELF backend.
- `EXPECTED.md` (~773 tok, large) — Phase 6.5 Stage 5 — Expected IR Text (APEX)
- `fast_keystone.sh` (~3764 tok, huge) — fast_keystone.sh — fast LOCAL front-end keystone gate for the pure-MIND self-host
- `field_store_netverify.py` (~1413 tok, large) — # Canonical independent value harness for struct field STORES (`p.x = v`) in the
- `FIXED_POINT_REPORT.md` (~1770 tok, huge) — Phase 6.5 — Bootstrap Fixed-Point Report
- `fixed_point_smoke.py` (~3275 tok, huge)
- `fixture.mind` (~183 tok, small) — Phase 6.4 emit_ir smoke fixture.
- `full_strtab_smoke.py` (~1663 tok, huge)
- `gap_corpus_smoke.py` (~2682 tok, huge)
- `general_float_netverify.py` (~1986 tok, huge) — general_float_netverify.py — GENERAL-path f64 value battery (B0 gate lift).
- `.gitignore` (~5 tok, tiny) — __pycache__/
- `lockstep_lint.py` (~4127 tok, huge) — lockstep_lint.py -- native-ELF walker lockstep linter for the pure-MIND self-host compiler.
- `match_struct_smoke.py` (~1311 tok, large)
- `method_callee_smoke.py` (~1350 tok, large)
- `method_calls_smoke.py` (~1294 tok, large)
- `mic3_flip_smoke.py` (~1150 tok, large)
- `mic3_oracle_smoke.py` (~764 tok, large) — mic@3 self-host convergence — Phase 0 gate: the Rust oracle.
- `mic3_primitives_smoke.py` (~22537 tok, huge) — mic@3 self-host convergence — Phase 1 gate: pure-MIND ULEB128 / zigzag.
- `mindfuzz_self_host.py` (~11539 tok, huge)
- `mod_operator_smoke.py` (~2871 tok, huge)
- `multi_let_smoke.py` (~1499 tok, large)
- `now_ns_smoke.py` (~678 tok, large) — # Copyright 2025 STARGA Inc.
- `option_netverify.py` (~2321 tok, huge) — # Canonical independent net-verify harness for SINGLE-PAYLOAD ENUMS (the
- `oracle_parity_lint.py` (~4516 tok, huge)
- `param_types_smoke.py` (~1273 tok, large)
- `_ref_add.note` (~16 tok, tiny) — 6fa59a74687e6bac38c983655d4d93ab1873299f130f68cbe481cf92041f6610
- `_ref_if_ret.note` (~16 tok, tiny) — 3fdc70390e9e12d8030552d11b2194078e8579fbcfac19b14d1beed77174cb07
- `_ref_main.note` (~16 tok, tiny) — 5c09107874e19b9fd1063f23561d7635b1bf622fa16b3bca50dca3e3de4138a9
- `ref_netverify.py` (~1473 tok, large) — # Canonical independent net-verify harness for i64 references in the native-ELF backend.
- `_ref_recursion.note` (~16 tok, tiny) — 6d125a946243b0550700d9aa6bc2058b51a3b8bf6e536a8b0b3f545d79b7346f
- `_ref_struct_field.note` (~16 tok, tiny) — 2f7f2ea32ee47138e1b0162bd51f418b4e5005b4569054f5f3c392ebd2258d96
- `_ref_value_if.note` (~16 tok, tiny) — ae565a5154b76ee1f44a32c16db2e9a8387f2f60c5fca92bba2dc9bb70afb599
- `rh_f64_aggregate_canary_smoke.py` (~1443 tok, large) — RH f64-aggregate surface canary (RH_REQUIRED_F64_AGGREGATE_SURFACE).
- `ri_d1_frozen_profile_gate.py` (~3278 tok, huge) — RI-D1 readiness gate (task #313): prove `mindc build --backend native` is READY to be
- `self_host_andor_precedence_smoke.py` (~1680 tok, huge) — Front-end PARITY battery for `&&` / `||` precedence + short-circuit.
- `self_host_andor_smoke.py` (~2142 tok, huge) — Permanent battery for the self-host `&&` / `||` short-circuit operators.
- `self_host_arena_growth_smoke.py` (~1382 tok, large)
- `self_host_args_from_os_smoke.py` (~1356 tok, large)
- `selfhost_argv_driver.mind` (~1187 tok, large) — ===========================================================================
- `self_host_argv_smoke.py` (~1136 tok, large)
- `self_host_array_smoke.py` (~1367 tok, large)
- `self_host_arridx_mic3_smoke.py` (~1864 tok, huge)
- `self_host_body_smoke.py` (~3055 tok, huge)
- `self_host_carry_cap_smoke.py` (~1088 tok, large) — Cap-guard smoke for the self-host loop-carry / loop-frame scratch tables.
- `self_host_cast_mic3_smoke.py` (~1776 tok, huge)
- `self_host_check_driver_smoke.py` (~2879 tok, huge)
- `self_host_continue_smoke.py` (~1384 tok, large) — Regression gate for #308 (closes #286's break/continue fixtures): the pure-MIND
- `selfhost_driver.mind` (~4121 tok, huge) — ===========================================================================
- `self_host_dtype_tag_smoke.py` (~780 tok, large) — RI-B1 per-SSA dtype-tag gate (parser <-> nb_fp_* encoder connecting construct).
- `self_host_else_if_smoke.py` (~1715 tok, huge)
- `self_host_failclosed_smoke.py` (~9331 tok, huge) — self_host_failclosed_smoke.py — the fail-closed boundary of the pure-MIND
- `self_host_float_lit_exact_smoke.py` (~1069 tok, large) — CPU-as-oracle smoke for the C1 float-literal exactness guard.
- `self_host_for_smoke.py` (~3191 tok, huge) — Permanent battery for the self-host range-`for` loop.
- `self_host_if_region_carry_smoke.py` (~4552 tok, huge) — Native-ELF smoke: i64 loop-carry through BRANCHED regions (Sub-step C).
- `self_host_ifret_chain_mic3_smoke.py` (~1725 tok, huge)
- `self_host_letpath_failclose_smoke.py` (~2202 tok, huge) — Self-host LET-PATH fail-closed smoke — pins the S1-collapse fail-open wall.
- `self_host_lockstep_smoke.py` (~2161 tok, huge) — SUB-STEP A lockstep smoke: the loop-carry frame COUNT and the loop-carry EMIT are
- `self_host_loop_smoke.py` (~4361 tok, huge)
- `self_host_matchscalar_mic3_smoke.py` (~1725 tok, huge)
- `self_host_match_smoke.py` (~1893 tok, huge)
- `self_host_mic3_float_valueif_smoke.py` (~1768 tok, huge)
- `self_host_mlir_smoke.py` (~1954 tok, huge)
- `self_host_narrow_let_mic3_smoke.py` (~1612 tok, huge)
- `self_host_narrow_param_smoke.py` (~11195 tok, huge) — Native-ELF smoke for narrow-width (i8/i16/i32) function PARAMETERS carried by a loop.
- `self_host_native_autowrap_smoke.py` (~2442 tok, huge) — Roadmap C2 declared-width AUTO-WRAP driver — narrow-int (i8/i16/i32) `let` and
- `self_host_native_avx2_dot_f32_smoke.py` (~1559 tok, huge) — RI-B2-S13 (#108) — native-ELF PACKED-f32 SIMD via 256-bit AVX2 (VEX/YMM) STRICT-FP DOT.
- `self_host_native_blas_dot_i16_smoke.py` (~2879 tok, huge)
- `self_host_native_blas_dot_q16_smoke.py` (~1774 tok, huge)
- `self_host_native_cast_conv_smoke.py` (~1267 tok, large) — RI-B2 scalar-cast-conv rung (#108) — native-ELF int<->float `as`-cast chain.
- `self_host_native_diag_smoke.py` (~1100 tok, large) — RI-D1a gate (task #313): `mindc build --backend=native` must FAIL-CLOSED with an
- `self_host_native_dot_f32_smoke.py` (~1334 tok, large) — RI-B2-S8 STEP C (#108) — native-ELF scalar STRICT-FP f32 DOT-PRODUCT.
- `self_host_native_dot_l1_q16_smoke.py` (~1088 tok, large) — RI-B2 L1-Q16 rung (#108) — native-ELF Q16.16 L1 distance.
- `self_host_native_elf_alignment_smoke.py` (~1902 tok, huge) — Self-host NATIVE-ELF ABI-alignment smoke (FIX #169).
- `self_host_native_elf_smoke.py` (~10885 tok, huge)
- `self_host_native_fp_binop_smoke.py` (~1095 tok, large) — RI-B1 nb_expr FLOAT-op-FLOAT arithmetic routing gate (zero MLIR/LLVM).
- `self_host_native_fp_call_smoke.py` (~5469 tok, huge) — RI-D2 S-C1: FLOAT call-RETURN dtype through the native-ELF general nb_expr lowering.
- `self_host_native_fp_expr_smoke.py` (~1020 tok, large) — RI-B1 nb_expr float-scalar routing gate (zero MLIR/LLVM).
- `self_host_native_fp_field_smoke.py` (~1832 tok, huge) — RI-D2 S-D FLOAT struct-FIELD READ dtype through native-ELF general lowering (zero MLIR).
- `self_host_native_fp_let_smoke.py` (~1208 tok, large) — RI-B1 (#107 follow-up) FLOAT dtype propagation ACROSS a LET binding (zero MLIR/LLVM).
- `self_host_native_fp_param_smoke.py` (~1403 tok, large) — RI-D2 S-B: FLOAT fn-param dtype classification + SysV SSE-spill ABI (zero MLIR/LLVM).
- `self_host_native_fp_smoke.py` (~1142 tok, large) — RI-B1 native-ELF scalar-f64 gate (zero MLIR/LLVM).
- `self_host_native_gemm_i8_smoke.py` (~1051 tok, large) — RI-B2-S7 (#108) — native-ELF scalar int8 GEMM (matrix x matrix), byte-identity rung.
- `self_host_native_gemm_q16_smoke.py` (~1053 tok, large) — RI-B2-S7 (#108) — native-ELF scalar GEMM Q16.16 (matrix x matrix), byte-identity rung.
- `self_host_native_gemv_i16_smoke.py` (~1044 tok, large) — RI-B2-S6 (#108) — native-ELF scalar GEMV int16 (matrix x vector), byte-identity rung.
- `self_host_native_gemv_q16_smoke.py` (~1037 tok, large) — RI-B2-S6 (#108) — native-ELF scalar GEMV Q16.16 (matrix x vector), byte-identity rung.
- `self_host_native_genf32_smoke.py` (~1066 tok, large) — RI-B2-S8 STEP B (#108) — isolate the LCG f32 rounding BEFORE the dot.
- `self_host_native_intdot_i16_smoke.py` (~1018 tok, large) — RI-B2-S2 (#108) — native-ELF scalar int16 DOT-PRODUCT, FIRST byte-identity rung.
- `self_host_native_intdot_q16_smoke.py` (~1008 tok, large) — RI-B2-S4 (#108) — native-ELF scalar Q16.16 DOT-PRODUCT, byte-identity rung.
- `self_host_native_intdot_smoke.py` (~1165 tok, large) — RI-B2-S1 (#108) scalar i64 DOT-PRODUCT reduction native-ELF (zero MLIR/LLVM).
- `self_host_native_matmul_f32_v_smoke.py` (~1611 tok, huge) — RI-B2-S9 (#108) — native-ELF scalar STRICT-FP f32 GEMV (matmul-f32-v).
- `self_host_native_narrow_add_i8_smoke.py` (~2260 tok, huge) — C2 — native-ELF NARROW-INT (i8) WRAP ARITHMETIC, zero MLIR/LLVM.
- `self_host_native_narrow_arith_batch_smoke.py` (~2554 tok, huge) — C2 — native-ELF NARROW-INT WRAP ARITHMETIC batch: {sub,mul}xi8 + {add,mul}xi16.
- `self_host_native_narrowint_smoke.py` (~1656 tok, huge) — Roadmap C2 narrow-int native-ELF rung — user-reachable i8/i16/i32 truncating
- `self_host_native_narrow_paramret_smoke.py` (~1949 tok, huge) — Byte-behavior smoke for narrow-int (i8/i16/i32) PARAM + RETURN auto-wrap in the
- `self_host_native_narrowwrap_smoke.py` (~1673 tok, huge) — Roadmap C2 narrow-int native-ELF rung — user-reachable i8/i16/i32 two's-complement
- `self_host_native_scalar_f32_smoke.py` (~1342 tok, large) — Phase C1-remainder f32 rung — native-ELF scalar SINGLE-precision chain.
- `self_host_native_scalar_f64_smoke.py` (~1201 tok, large) — RI-B2 f64 rung (#108) — native-ELF scalar STRICT-FP f64 CHAIN.
- `self_host_native_scalar_narrow_smoke.py` (~1800 tok, huge) — RI-D / #10 native-float SLICE 1a — native-ELF SATURATING f64 -> signed NARROW
- `self_host_native_scast_smoke.py` (~1694 tok, huge)
- `self_host_native_simd_dot_f32_smoke.py` (~1502 tok, huge) — RI-B2-S11 (#108) — native-ELF PACKED-f32 SIMD (SSE, 128-bit) STRICT-FP DOT-PRODUCT.
- `self_host_native_simd_dot_i16_smoke.py` (~1130 tok, large) — RI-B2-S12 (#108) — native-ELF PACKED-int16 SIMD DOT-PRODUCT, byte-identity rung.
- `self_host_native_simd_dot_q16_smoke.py` (~1486 tok, large) — RI-B2-S10 (#108) — native-ELF PACKED-SIMD Q16.16 DOT-PRODUCT, byte-identity rung.
- `self_host_native_string_smoke.py` (~2275 tok, huge)
- `self_host_native_tensor_batchsum_smoke.py` (~2372 tok, huge) — C4-T6 — native-ELF 3-D BATCHED SUM (i64), zero MLIR/LLVM. The FIRST N-D
- `self_host_native_tensor_bcastadd_smoke.py` (~1700 tok, huge) — C4-T5 — native-ELF tensor ROW-VECTOR BROADCAST ADD (i64), zero MLIR/LLVM.
- `self_host_native_tensor_colsum_smoke.py` (~2094 tok, huge) — C4-T5 — native-ELF tensor COLUMN REDUCTION (i64), zero MLIR/LLVM.
- `self_host_native_tensor_dot_smoke.py` (~1197 tok, large) — C4-T2 — native-ELF tensor DOT PRODUCT (i64), zero MLIR/LLVM.
- `self_host_native_tensor_ewadd_f64_smoke.py` (~1794 tok, huge) — C4-T4 — native-ELF float64 TENSOR element-wise-add + STRICT-SEQUENTIAL reduce.
- `self_host_native_tensor_ewadd_smoke.py` (~1118 tok, large) — C4-T1 — native-ELF tensor ELEMENT-WISE ADD (i64), zero MLIR/LLVM.
- `self_host_native_tensor_ewmul_smoke.py` (~2047 tok, huge) — C4-T5 — native-ELF tensor ELEMENT-WISE MULTIPLY (i64), zero MLIR/LLVM.
- `self_host_native_tensor_matmul_smoke.py` (~1723 tok, huge) — C4-T3 — native-ELF tensor MATMUL (i64), zero MLIR/LLVM.
- `self_host_native_tensor_maxrowmax_smoke.py` (~2453 tok, huge) — C4-T6 — native-ELF tensor ROW MAX-REDUCTION (i64), zero MLIR/LLVM.
- `self_host_native_tensor_relu_smoke.py` (~2194 tok, huge) — C4-T6 — native-ELF tensor ELEMENTWISE RELU (max(x,0)), zero MLIR/LLVM.
- `self_host_native_tensor_rowmin_smoke.py` (~2353 tok, huge) — C4-T6 — native-ELF tensor ROW MIN-REDUCTION (i64), zero MLIR/LLVM.
- `self_host_native_tensor_rowsum_smoke.py` (~1958 tok, huge) — C4-T4 — native-ELF tensor ROW REDUCTION (i64), zero MLIR/LLVM.
- `self_host_native_tensor_transpose_smoke.py` (~2224 tok, huge) — C4-T4 — native-ELF tensor TRANSPOSE (i64), zero MLIR/LLVM.
- `self_host_native_toplevel_assign_smoke.py` (~5550 tok, huge) — INDEPENDENCE_ROADMAP Phase-C follow-up — TOP-LEVEL STRAIGHT-LINE i64 REASSIGN
- `self_host_native_ucast_smoke.py` (~1444 tok, large)
- `self_host_native_write8_smoke.py` (~929 tok, large) — RI-B2-S2 STEP A (#108) — de-risk the C1 "emit 8 LE bytes to stdout + hash" gate.
- `self_host_native_write_f32_smoke.py` (~815 tok, large) — RI-B2-S8 STEP A (#108) — de-risk the raw-f32-bytes harness on a KNOWN f32.
- `self_host_not_smoke.py` (~1441 tok, large)
- `self_host_open_smoke.py` (~1310 tok, large)
- `self_host_param_mutation_smoke.py` (~1243 tok, large) — CPU-as-oracle smoke for the param-mutation fix (nb_expr ident arm: consult the
- `_selfhost_so.py` (~1672 tok, huge) — Shared self-host `.so` resolver for the examples/mindc_mind smokes.
- `self_host_standalone_driver_smoke.py` (~3253 tok, huge)
- `self_host_struct_return_smoke.py` (~4122 tok, huge) — self_host_struct_return_smoke.py — regression lock for STRUCT-BY-VALUE
- `self_host_tc_classify_error_code_smoke.py` (~2158 tok, huge) — CPU-as-oracle smoke for the pure-MIND classify_error_code router.
- `self_host_tc_class_mismatch_smoke.py` (~840 tok, large) — CPU-as-oracle smoke for the pure-MIND E2015 int<->float class-mismatch rule.
- `self_host_tc_class_rules_smoke.py` (~1560 tok, huge) — CPU-as-oracle smoke for the pure-MIND E2010/E2011/E2013/E2016 class rules.
- `self_host_tc_decl_names_smoke.py` (~3329 tok, huge) — CPU-as-oracle smoke for the pure-MIND D1 module DECL-NAME SET.
- `self_host_tc_fixed_bytes_into_vec_smoke.py` (~1110 tok, large) — CPU-as-oracle smoke for the pure-MIND E2006 FIXED_BYTES_INTO_VEC rule (Bug #38).
- `self_host_tc_fn_value_call_smoke.py` (~6982 tok, huge) — CPU-as-oracle smoke for the pure-MIND B1/D4 CAPSTONE — the FULL E2012 rule.
- `self_host_tc_let_class_mismatch_smoke.py` (~991 tok, large) — CPU-as-oracle smoke for the pure-MIND E2015 LET_CLASS_MISMATCH let/assign rule.
- `self_host_tc_let_infer_ident_smoke.py` (~2802 tok, huge) — self_host_tc_let_infer_ident_smoke — three-leg gate for the T2 port.
- `self_host_tc_let_infer_smoke.py` (~2052 tok, huge) — self_host_tc_let_infer_smoke — three-leg gate for the T1 type-inference port.
- `self_host_tc_narrowing_smoke.py` (~1042 tok, large)
- `self_host_tc_scope_frame_smoke.py` (~5434 tok, huge) — CPU-as-oracle smoke for the pure-MIND B1/D2 NESTED SCOPE-FRAME WALK.
- `self_host_tc_self_host_only_call_smoke.py` (~1833 tok, huge) — CPU-as-oracle smoke for the pure-MIND E2024 self-host-only-call rule.
- `self_host_tc_shape_annot_compat_smoke.py` (~1615 tok, huge) — CPU-as-oracle smoke for the pure-MIND shape annotation-compat rule.
- `self_host_tc_shape_rules_smoke.py` (~1820 tok, huge) — CPU-as-oracle smoke for the pure-MIND E2005/E2101/E2102/E2103 shape rules + E2023 reserved-prefix rule.
- `self_host_tc_std_export_smoke.py` (~3178 tok, huge) — CPU-as-oracle smoke for the pure-MIND D3 STD-SURFACE EXPORT NAME SET.
- `self_host_tc_undeclared_assign_smoke.py` (~6919 tok, huge) — CPU-as-oracle smoke for the pure-MIND B1 E2009 rule — undeclared assign.
- `self_host_tc_unknown_call_smoke.py` (~8459 tok, huge) — CPU-as-oracle smoke for the pure-MIND B1 E2003 rule — unknown call.
- `self_host_tc_unknown_ident_smoke.py` (~11052 tok, huge) — CPU-as-oracle smoke for the pure-MIND B1 E2002 rule — unknown identifier.
- `self_host_tc_unknown_variant_smoke.py` (~1911 tok, huge) — CPU-as-oracle smoke for the pure-MIND E2008 unknown-enum-variant rule.
- `self_host_value_if_expr_smoke.py` (~1557 tok, huge)
- `sha256_hash_smoke.py` (~1166 tok, large) — # Copyright 2025 STARGA Inc.
- `smoke_wiring_lint.py` (~2910 tok, huge) — smoke_wiring_lint.py — machine-checked contract for WHERE each
- `SMOKE_WIRING.tsv` (~3294 tok, huge) — # SMOKE_WIRING.tsv — the CHECKED contract for where each examples/mindc_mind s
- `stdlib_manifest_lint.py` (~2418 tok, huge) — stdlib_manifest_lint.py — machine-checked contract for WHICH std/*.mind
- `_stdlib_manifest.py` (~1129 tok, large) — Shared reader for the committed std/*.mind manifest.
- `struct_fields_smoke.py` (~1076 tok, large)
- `struct_lit_smoke.py` (~2240 tok, huge)
- `tc_differential_fuzz.py` (~13250 tok, huge) — tcdiff — differential fuzzer for the self-host source-position tc-rule ports.
### `examples/mindc_mind/testdata/backend_native_bridge/`

- `add.elf` (~123 tok, small) — ELF> @@@8
- `MANIFEST.txt` (~292 tok, medium) — # Pure-MIND native-ELF oracle (RI-D slice 1, task #110). Frozen from the pure-MI
### `examples/mindc_mind/testdata/`

- `dtk_plan_parity_smoke.py` (~2452 tok, huge)
### `examples/mindc_mind/testdata/native_elf_oracle/`

- `add.elf` (~123 tok, small) — ELF> @@@8
- `if_ret.elf` (~150 tok, small) — ELF> @@@8
- `MANIFEST.txt` (~152 tok, small) — # Frozen native-ELF oracle references, captured before #15 deletes src/native.
- `recursion.elf` (~175 tok, small) — ELF> @@@8
- `struct_field.elf` (~166 tok, small) — ELF> @@@8
- `value_if.elf` (~147 tok, small) — ELF> @@@8
### `examples/mindc_mind/testdata/`

- `rh_f64_aggregate_canary.mind` (~876 tok, large) — RH f64-aggregate self-host/native canary (RH_REQUIRED_F64_AGGREGATE_SURFACE).
### `examples/mindc_mind/testdata/selfhost_loop/`

- `MANIFEST.txt` (~102 tok, small) — # Frozen self-host bootstrap ELF (A6/RI-E1): the checked-in pure-MIND stage0
- `PROVENANCE_298_f64_selfhost_mlir.md` (~1826 tok, huge) — #298 self-host f64 MLIR emit — self-host loop seed refreeze provenance
### `examples/mindc_mind/testdata/`

- `stdlib_manifest.txt` (~831 tok, large) — # std/*.mind MANIFEST — the single committed source of truth for "which std
### `examples/mindc_mind/`

- `unified_dispatch_smoke.py` (~1553 tok, huge)
- `validate_real_fns_smoke.py` (~2674 tok, huge)
- `while_struct_smoke.py` (~1031 tok, large)
### `examples/`

- `mlir_pipeline_demo.sh` (~1647 tok, huge) — MLIR/LLVM Pipeline Demonstration
### `examples/native/`

- `ci_kernel.mind` (~39 tok, tiny)
- `ci_kernel_smoke.py` (~417 tok, medium)
- `fib.mind` (~34 tok, tiny) — fn fib(n: i64) -> i64 {
- `loop.mind` (~37 tok, tiny) — fn main() -> i64 {
### `examples/parser/`

- `bootstrap_smoke.py` (~5544 tok, huge)
- `EXPECTED.md` (~2140 tok, huge) — Phase 6.2 — Expected AST Tree
- `fixture.mind` (~160 tok, small) — Phase 6.2 parser smoke fixture.
- `main.mind` (~7825 tok, huge) — examples/parser/main.mind — RFC 0005 Phase 6.2 self-host parser seed.
- `README.md` (~2254 tok, huge) — RFC 0005 Phase 6.2 — Self-Host Parser Seed
### `examples/`

- `policy.mind` (~1301 tok, large) — policy.mind — v0.1 Execution Boundary Kernel
- `README.md` (~2066 tok, huge) — MIND Examples
- `remizov_benchmark.mind` (~6400 tok, huge) — ASPIRATIONAL DEMO — not yet buildable with the open mindc.
- `remizov_feynman.mind` (~2894 tok, huge) — ASPIRATIONAL DEMO — not yet buildable with the open mindc.
- `remizov_gpu.mind` (~2662 tok, huge) — ASPIRATIONAL DEMO — not yet buildable with the open mindc.
- `remizov_inverse.mind` (~2614 tok, huge) — ASPIRATIONAL DEMO — not yet buildable with the open mindc.
- `remizov_solver.mind` (~3802 tok, huge) — ASPIRATIONAL DEMO — not yet buildable with the open mindc.
- `remizov_verify.mind` (~3721 tok, huge) — ASPIRATIONAL DEMO — not yet buildable with the open mindc.
- `tiny_edge_model.mind` (~1876 tok, huge) — Tiny Edge Model Example
### `examples/typecheck/`

- `bootstrap_smoke.py` (~2608 tok, huge)
- `EXPECTED.md` (~2015 tok, huge) — Phase 6.3 — Expected Type-Check Report
- `fixture.mind` (~198 tok, small) — Phase 6.3 type-checker smoke fixture.
- `main.mind` (~7120 tok, huge) — examples/typecheck/main.mind — RFC 0005 Phase 6.3 self-host
- `README.md` (~2612 tok, huge) — RFC 0005 Phase 6.3 — Self-Host Type-Checker Seed
### `examples/zoo/`

- `conv_classifier.mind` (~2407 tok, huge) — ASPIRATIONAL DEMO — not yet buildable with the open mindc.
- `linear_regression.mind` (~1347 tok, large) — ASPIRATIONAL DEMO — not yet buildable with the open mindc.
- `logistic_classifier.mind` (~1517 tok, huge) — ASPIRATIONAL DEMO — not yet buildable with the open mindc.
- `mlp_mnist.mind` (~2275 tok, huge) — ASPIRATIONAL DEMO — not yet buildable with the open mindc.
- `README.md` (~1191 tok, large) — MIND Model Zoo
- `transformer_block.mind` (~3781 tok, huge) — ASPIRATIONAL DEMO — not yet buildable with the open mindc.
### `experiments/global-vs-local/`

- `chern.py` (~1244 tok, large)
- `exp2.py` (~1338 tok, large)
- `exp3_universal.py` (~997 tok, large)
- `plot_chern.py` (~469 tok, medium)
- `plot.py` (~904 tok, large)
- `README.md` (~839 tok, large) — Global vs Local — "Closed-form whole-field invariant" experiments
- `topo.py` (~696 tok, large)
### `.githooks/`

- `pre-commit` (~434 tok, medium) — #!/usr/bin/env bash
### `.github/`

- `CODEOWNERS` (~9 tok, tiny) — *       @star-ga
### `.github/ISSUE_TEMPLATE/`

- `bounty_claim.md` (~56 tok, small)
- `bug_report.md` (~213 tok, medium) — Describe the bug
- `feature_request.md` (~171 tok, small) — Problem Statement
### `.github/`

- `PULL_REQUEST_TEMPLATE.md` (~55 tok, small) — Summary
- `release-drafter.yml` (~85 tok, small) — name-template: 'v$NEXT_PATCH_VERSION'
- `required-ci-jobs.tsv` (~452 tok, medium) — # required-ci-jobs.tsv — the CI jobs that MUST have passed on the exact commit
### `.github/workflows/`

- `bench-gate.yml` (~2168 tok, huge) — name: Bench gate
- `cargo-deny.yml` (~229 tok, medium) — name: Cargo Deny
- `ci.yml` (~17109 tok, huge) — name: CI
- `crypto-vectors.yml` (~1454 tok, large) — name: Crypto Vectors
- `docs-claims.yml` (~1571 tok, huge) — name: Docs Claims
- `link-check.yml` (~229 tok, medium) — name: Link Check
- `mindcraft.yml` (~910 tok, large) — name: Mindcraft Check
- `release-drafter.yml` (~91 tok, small) — name: Release Drafter
- `release.yml` (~3572 tok, huge) — name: Release
### `mind/std/cognitive/`

- `batch_scheduler.mind` (~850 tok, large) — Batch scheduling for inference workloads
- `kv_cache.mind` (~840 tok, large) — KV-Cache for transformer inference
- `speculative.mind` (~891 tok, large) — Speculative decoding with rejection sampling
- `verification.mind` (~948 tok, large) — Verification plane for inference consistency (LCU)
### `runtime-support/`

- `mind_intrinsics.c` (~21838 tok, huge) — Copyright 2025 STARGA Inc.
### `scripts/`

- `ai_attribution_patterns.sh` (~1536 tok, huge) — shellcheck shell=bash
- `anatomy-hook.sh` (~858 tok, large) — STARGA author guard (chained first: a wrong-identity commit must never be created).
- `anatomy.sh` (~2011 tok, huge) — anatomy — Generate ANATOMY.md for any repo
- `cfg_gate_wiring_lint.py` (~3358 tok, huge) — Fail closed when a feature-gated integration test is wired to no CI selector.
- `check_claims.py` (~2779 tok, huge) — Docs-claim CI gate — fail if any public surface drifts from config/capabilities.toml.
- `check_commit_messages.sh` (~1428 tok, large) — check_commit_messages.sh <rev-range> — no model or tool credit in commit
- `check_gate_wiring.py` (~2310 tok, huge) — Wiring lint: a whole-tree gate's CI trigger must cover every path it scans.
- `check_json_not_evidence.sh` (~813 tok, large) — Wedge-integrity gate: JSON is never an evidence-hash preimage.
- `check_no_ai_attribution.sh` (~1221 tok, large) — Public-artifact hygiene gate: no AI tool/model named as having AUTHORED or
- `check_release_gating.py` (~3441 tok, huge) — check_release_gating.py — machine-checked supply-chain contract for the
- `commit-msg-hook.sh` (~803 tok, large) — commit-msg-hook.sh — local commit-msg hook: refuse a commit whose MESSAGE
- `enumeration_drift_lint.py` (~2544 tok, huge) — Fail closed when a comment's ENUMERATION drifts from the code it describes.
- `exec_semantics_gate.sh` (~6635 tok, huge) — exec_semantics_gate.sh — run the test tiers CI could not reach, and PROVE they ran.
- `install.ps1` (~1856 tok, huge) — # install.ps1 - mindc one-line installer for Windows (PowerShell)
- `install.sh` (~1054 tok, large) — MIND compiler (mindc) installer — downloads a pre-built binary from the
### `scripts/mind-vs-rust/`

- `Cargo.toml` (~271 tok, medium) — [package]
- `.gitignore` (~3 tok, tiny) — */target-*/
- `run.sh` (~659 tok, large) — Copyright 2026 STARGA Inc. Licensed under the Apache License, Version 2.0.
### `scripts/mind-vs-rust/src/`

- `main.rs` (~2372 tok, huge) — Copyright 2026 STARGA Inc.
### `scripts/`

- `operand_visitor_parity_lint.py` (~1198 tok, large) — Fail closed when the two IR operand visitors drift apart.
- `preflight.sh` (~6654 tok, huge) — preflight.sh — local CI-parity gate. Run before pushing to avoid red CI.
- `quick_perf.sh` (~734 tok, large) — quick_perf.sh — FAST one-sided compile-speed criterion gate.
- `ri_d1_mlir_free_gate.sh` (~1376 tok, large) — RI-D1 gate assertion #1 — MLIR is un-linkable on the native path (fail-closed
- `run_crypto_vectors.sh` (~1932 tok, huge) — Build every pure-MIND crypto/TLS std module to a shared object and run its
### `scripts/sdlc/`

- `enforcement_bijection.py` (~1314 tok, large) — A test must never outlive the rule it asserts.
- `lost_by_merge.py` (~1245 tok, large) — Fail a merge that silently DROPPED a line both parents agreed on.
### `scripts/`

- `verify_ci_green.py` (~3197 tok, huge) — verify_ci_green.py — refuse to release a commit that CI has not proven green.
- `workflow_scan.py` (~1069 tok, large) — workflow_scan.py — dependency-free structural reader for the GitHub Actions
### `sdk/ts/mic-map/dist/`

- `errors.d.ts` (~209 tok, medium)
- `errors.d.ts.map` (~147 tok, small) — {"version":3,"file":"errors.d.ts","sourceRoot":"","sources":["../src/errors.ts"]
- `errors.js` (~350 tok, medium) — Copyright 2026 STARGA Inc.
- `errors.js.map` (~279 tok, medium) — {"version":3,"file":"errors.js","sourceRoot":"","sources":["../src/errors.ts"],"
- `framing.d.ts` (~190 tok, small)
- `framing.d.ts.map` (~82 tok, small) — {"version":3,"file":"framing.d.ts","sourceRoot":"","sources":["../src/framing.ts
- `framing.js` (~757 tok, large) — Copyright 2026 STARGA Inc.
- `framing.js.map` (~627 tok, large) — {"version":3,"file":"framing.js","sourceRoot":"","sources":["../src/framing.ts"]
- `index.d.ts` (~272 tok, medium)
- `index.d.ts.map` (~244 tok, medium) — {"version":3,"file":"index.d.ts","sourceRoot":"","sources":["../src/index.ts"],"
- `index.js` (~459 tok, medium) — Copyright 2026 STARGA Inc.
- `index.js.map` (~393 tok, medium) — {"version":3,"file":"index.js","sourceRoot":"","sources":["../src/index.ts"],"na
- `map.d.ts` (~355 tok, medium)
- `map.d.ts.map` (~226 tok, medium) — {"version":3,"file":"map.d.ts","sourceRoot":"","sources":["../src/map.ts"],"name
- `map.js` (~2064 tok, huge) — Copyright 2026 STARGA Inc.
- `map.js.map` (~2167 tok, huge) — {"version":3,"file":"map.js","sourceRoot":"","sources":["../src/map.ts"],"names"
- `mic2_emit.d.ts` (~117 tok, small)
- `mic2_emit.d.ts.map` (~58 tok, small) — {"version":3,"file":"mic2_emit.d.ts","sourceRoot":"","sources":["../src/mic2_emi
- `mic2_emit.js` (~594 tok, large) — Copyright 2026 STARGA Inc.
- `mic2_emit.js.map` (~655 tok, large) — {"version":3,"file":"mic2_emit.js","sourceRoot":"","sources":["../src/mic2_emit.
- `mic2_parse.d.ts` (~70 tok, small)
- `mic2_parse.d.ts.map` (~64 tok, small) — {"version":3,"file":"mic2_parse.d.ts","sourceRoot":"","sources":["../src/mic2_pa
- `mic2_parse.js` (~2396 tok, huge) — Copyright 2026 STARGA Inc.
- `mic2_parse.js.map` (~2613 tok, huge) — {"version":3,"file":"mic2_parse.js","sourceRoot":"","sources":["../src/mic2_pars
- `micb.d.ts` (~171 tok, small)
- `micb.d.ts.map` (~93 tok, small) — {"version":3,"file":"micb.d.ts","sourceRoot":"","sources":["../src/micb.ts"],"na
- `micb.js` (~3256 tok, huge) — Copyright 2026 STARGA Inc.
- `micb.js.map` (~3469 tok, huge) — {"version":3,"file":"micb.js","sourceRoot":"","sources":["../src/micb.ts"],"name
- `types.d.ts` (~946 tok, large)
- `types.d.ts.map` (~772 tok, large) — {"version":3,"file":"types.d.ts","sourceRoot":"","sources":["../src/types.ts"],"
- `types.js` (~1458 tok, large) — Copyright 2026 STARGA Inc.
- `types.js.map` (~1822 tok, huge) — {"version":3,"file":"types.js","sourceRoot":"","sources":["../src/types.ts"],"na
- `varint.d.ts` (~318 tok, medium)
- `varint.d.ts.map` (~185 tok, small) — {"version":3,"file":"varint.d.ts","sourceRoot":"","sources":["../src/varint.ts"]
- `varint.js` (~612 tok, large) — Copyright 2026 STARGA Inc.
- `varint.js.map` (~554 tok, large) — {"version":3,"file":"varint.js","sourceRoot":"","sources":["../src/varint.ts"],"
### `sdk/ts/mic-map/`

- `LICENSE` (~2573 tok, huge) —                                  Apache License
- `package.json` (~210 tok, medium) — Keys: name, version, description, type, private
- `package-lock.json` (~13043 tok, huge) — Keys: name, version, lockfileVersion, requires, packages
- `README.md` (~152 tok, small) — @mind/mic-map
### `sdk/ts/mic-map/scripts/`

- `regen_fixtures.sh` (~499 tok, medium) — Copyright 2026 STARGA Inc.
### `sdk/ts/mic-map/src/`

- `errors.ts` (~351 tok, medium) — Copyright 2026 STARGA Inc.
- `framing.ts` (~726 tok, large) — Copyright 2026 STARGA Inc.
- `index.ts` (~497 tok, medium) — Copyright 2026 STARGA Inc.
- `map.ts` (~2187 tok, huge) — Copyright 2026 STARGA Inc.
- `mic2_emit.ts` (~586 tok, large) — Copyright 2026 STARGA Inc.
- `mic2_parse.ts` (~2242 tok, huge) — Copyright 2026 STARGA Inc.
- `micb.ts` (~2959 tok, huge) — Copyright 2026 STARGA Inc.
- `types.ts` (~2016 tok, huge) — Copyright 2026 STARGA Inc.
- `varint.ts` (~617 tok, large) — Copyright 2026 STARGA Inc.
### `sdk/ts/mic-map/test/fixtures/`

- `map_examples.txt` (~76 tok, small) — # MAP protocol frame examples
- `residual_block.mic2.txt` (~20 tok, tiny) — mic@2
### `sdk/ts/mic-map/test/`

- `framing.test.ts` (~1257 tok, large) — Copyright 2026 STARGA Inc.
- `map.test.ts` (~2328 tok, huge) — Copyright 2026 STARGA Inc.
- `mic2.test.ts` (~2637 tok, huge) — Copyright 2026 STARGA Inc.
- `micb.test.ts` (~1621 tok, huge) — Copyright 2026 STARGA Inc.
### `sdk/ts/mic-map/`

- `tsconfig.json` (~152 tok, small) — Keys: compilerOptions, include, exclude
- `vitest.config.ts` (~72 tok, small)
### `skills/write-mind/`

- `SKILL.md` (~6002 tok, huge) — Write MIND Code
### `src/ast/`

- `mod.rs` (~10930 tok, huge) — Copyright 2025 STARGA Inc.
### `src/autodiff/`

- `engine.rs` (~3890 tok, huge) — Copyright 2025 STARGA Inc.
- `mod.rs` (~342 tok, medium) — Copyright 2025 STARGA Inc.
- `rules.rs` (~2392 tok, huge) — Copyright 2025 STARGA Inc.
### `src/bin/`

- `mind-ai.rs` (~9878 tok, huge) — Copyright 2025 STARGA Inc.
### `src/build/`

- `cache.rs` (~6766 tok, huge) — Copyright 2025 STARGA Inc.
- `mod.rs` (~11600 tok, huge) — Copyright 2025 STARGA Inc.
### `src/cache/`

- `entry.rs` (~977 tok, large) — Copyright 2025-2026 STARGA Inc.
- `fingerprint.rs` (~629 tok, large) — Copyright 2025-2026 STARGA Inc.
- `mod.rs` (~964 tok, large) — Copyright 2025-2026 STARGA Inc.
- `store.rs` (~1112 tok, large) — Copyright 2025-2026 STARGA Inc.
### `src/check/`

- `gitignore.rs` (~2345 tok, huge) — Copyright 2025 STARGA Inc.
- `mod.rs` (~8110 tok, huge) — Copyright 2025 STARGA Inc.
- `reporter.rs` (~374 tok, medium) — Copyright 2025 STARGA Inc.
### `src/`

- `conformance.rs` (~1847 tok, huge) — The autodiff_pairwise conformance entry was removed 2026-05-20 — its
### `src/deps/`

- `mod.rs` (~9345 tok, huge) — Copyright 2025 STARGA Inc.
### `src/diagnostics/`

- `mod.rs` (~3719 tok, huge) — Copyright 2025 STARGA Inc.
### `src/distributed/`

- `allgather.rs` (~813 tok, large) — Copyright 2025-2026 STARGA Inc.
- `allreduce.rs` (~1011 tok, large) — Copyright 2025-2026 STARGA Inc.
- `invariants.rs` (~1416 tok, large) — Copyright 2025-2026 STARGA Inc.
- `mod.rs` (~872 tok, large) — Copyright 2025-2026 STARGA Inc.
- `pipeline.rs` (~1973 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `shard.rs` (~1640 tok, huge) — Copyright 2025-2026 STARGA Inc.
### `src/doc/`

- `html.rs` (~956 tok, large) — Copyright 2025 STARGA Inc.
- `markdown.rs` (~2644 tok, huge) — Copyright 2025 STARGA Inc.
- `mod.rs` (~7387 tok, huge) — Copyright 2025 STARGA Inc.
### `src/eval/`

- `abi_gate.rs` (~12049 tok, huge) — Runnable-artifact ABI gate (release-readiness P1.1).
- `autodiff.rs` (~14268 tok, huge) — Copyright 2025 STARGA Inc.
- `closures.rs` (~9284 tok, huge) — Copyright 2025 STARGA Inc.
- `conv2d_grad.rs` (~2397 tok, huge) — Copyright 2025 STARGA Inc.
- `interp_mem.rs` (~2993 tok, huge) — Copyright 2025 STARGA Inc.
- `ir_interp.rs` (~3818 tok, huge) — Copyright 2025 STARGA Inc.
- `mlir_build.rs` (~10824 tok, huge) — Copyright 2025 STARGA Inc.
- `mlir_export.rs` (~12220 tok, huge) — Copyright 2025 STARGA Inc.
- `mlir_gpu.rs` (~301 tok, medium) — Copyright 2025 STARGA Inc.
- `mlir_jit.rs` (~501 tok, large) — Copyright 2025 STARGA Inc.
- `mlir_opt.rs` (~995 tok, large) — Copyright 2025 STARGA Inc.
- `mlir_run.rs` (~1535 tok, huge) — Copyright 2025 STARGA Inc.
- `narrow_scan.rs` (~3009 tok, huge) — Module-wide narrow-int SURFACE prescan (compile-speed early-skip).
### `src/eval/stdlib/`

- `mod.rs` (~169 tok, small) — Copyright 2025 STARGA Inc.
- `tensor.rs` (~8417 tok, huge) — Copyright 2025 STARGA Inc.
### `src/eval/`

- `struct_resolver.rs` (~6767 tok, huge) — Copyright 2025 STARGA Inc.
- `traits.rs` (~4150 tok, huge) — Copyright 2025 STARGA Inc.
- `value.rs` (~2228 tok, huge) — Copyright 2025 STARGA Inc.
### `src/exec/`

- `conv.rs` (~435 tok, medium) — Copyright 2025 STARGA Inc.
- `cpu.rs` (~3570 tok, huge) — Copyright 2025 STARGA Inc.
- `mod.rs` (~587 tok, large) — Copyright 2025 STARGA Inc.
### `src/ffi/`

- `header.rs` (~413 tok, medium) — Copyright 2025 STARGA Inc.
- `mod.rs` (~1780 tok, huge) — Copyright 2025 STARGA Inc.
- `sys.rs` (~3348 tok, huge) — Copyright 2025-2026 STARGA Inc.
### `src/fmt/`

- `cli.rs` (~3873 tok, huge) — Copyright 2025 STARGA Inc.
- `mod.rs` (~594 tok, large) — Copyright 2025 STARGA Inc.
- `printer.rs` (~17270 tok, huge) — Copyright 2025 STARGA Inc.
### `src/`

- `intrinsics.rs` (~10399 tok, huge) — Copyright 2025 STARGA Inc.
### `src/ir/compact/`

- `emit.rs` (~4693 tok, huge) — Copyright 2025 STARGA Inc.
- `mod.rs` (~2475 tok, huge) — Copyright 2025 STARGA Inc.
- `parse.rs` (~8124 tok, huge) — Copyright 2025 STARGA Inc.
### `src/ir/compact/v2/`

- `binary.rs` (~8845 tok, huge) — Copyright 2025 STARGA Inc.
- `emit.rs` (~2445 tok, huge) — Copyright 2025 STARGA Inc.
- `evidence.rs` (~10501 tok, huge) — Copyright 2025 STARGA Inc.
- `map_limits.rs` (~1823 tok, huge) — Copyright 2025 STARGA Inc.
- `map_tests.rs` (~8117 tok, huge) — Copyright 2025 STARGA Inc.
- `mod.rs` (~1343 tok, large) — Copyright 2025 STARGA Inc.
- `parse.rs` (~5388 tok, huge) — Copyright 2025 STARGA Inc.
- `types.rs` (~4764 tok, huge) — Copyright 2025 STARGA Inc.
- `varint.rs` (~2138 tok, huge) — Copyright 2025 STARGA Inc.
### `src/ir/compact/v3/`

- `collapse_receipt.rs` (~5109 tok, huge) — Copyright 2025 STARGA Inc.
- `ed25519.rs` (~6051 tok, huge) — Copyright 2025 STARGA Inc.
- `emit.rs` (~12430 tok, huge) — Copyright 2025 STARGA Inc.
- `mldsa.rs` (~2688 tok, huge) — Copyright 2025 STARGA Inc.
- `mod.rs` (~20794 tok, huge) — Copyright 2025 STARGA Inc.
- `parse.rs` (~13067 tok, huge) — Copyright 2025 STARGA Inc.
- `slhdsa.rs` (~3201 tok, huge) — Copyright 2025 STARGA Inc.
### `src/ir/`

- `evidence.rs` (~13353 tok, huge) — Copyright 2025 STARGA Inc.
- `fp_mode.rs` (~13711 tok, huge) — FP-contract mode — the strict-vs-relaxed floating-point determinism state of
- `frozen_profile.rs` (~7689 tok, huge) — Copyright 2025 STARGA Inc.
- `mod.rs` (~17810 tok, huge) — Copyright 2025 STARGA Inc.
- `print.rs` (~4413 tok, huge) — Copyright 2025 STARGA Inc.
- `verify.rs` (~21385 tok, huge) — Copyright 2025 STARGA Inc.
### `src/`

- `lib.rs` (~1071 tok, large) — Copyright 2025 STARGA Inc.
- `linalg.rs` (~2025 tok, huge) — Copyright 2025 STARGA Inc.
### `src/lint/`

- `mod.rs` (~1311 tok, large) — Copyright 2025 STARGA Inc.
- `rule.rs` (~2690 tok, huge) — Copyright 2025 STARGA Inc.
### `src/lint/rules/`

- `mod.rs` (~454 tok, medium) — Copyright 2025 STARGA Inc.
- `naming_convention.rs` (~2021 tok, huge) — Copyright 2025 STARGA Inc.
- `q16_overflow.rs` (~2910 tok, huge) — Copyright 2025 STARGA Inc.
- `shadowing.rs` (~1934 tok, huge) — Copyright 2025 STARGA Inc.
- `trailing_whitespace.rs` (~869 tok, large) — Copyright 2025 STARGA Inc.
- `unused_import.rs` (~1692 tok, huge) — Copyright 2025 STARGA Inc.
### `src/`

- `main.rs` (~6787 tok, huge) — Copyright 2025 STARGA Inc.
### `src/mlir/`

- `c_export.rs` (~1931 tok, huge) — Copyright 2025 STARGA Inc.
- `gemm_tuning.rs` (~3639 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `mod.rs` (~335 tok, medium) — Copyright 2025 STARGA Inc.
### `src/ops/`

- `cerebras.rs` (~2713 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `core_v1.rs` (~1823 tok, huge) — Copyright 2025 STARGA Inc.
- `mod.rs` (~228 tok, medium) — Copyright 2025 STARGA Inc.
### `src/opt/`

- `collapse.rs` (~11442 tok, huge) — Copyright 2025 STARGA Inc.
- `comptime.rs` (~5868 tok, huge) — Copyright 2025 STARGA Inc.
- `fold.rs` (~2046 tok, huge) — Copyright 2025 STARGA Inc.
- `ir_canonical.rs` (~5268 tok, huge) — Copyright 2025 STARGA Inc.
- `memory_layout.rs` (~4110 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `mod.rs` (~203 tok, medium) — Copyright 2025 STARGA Inc.
- `native_opt.rs` (~11171 tok, huge) — Copyright 2025 STARGA Inc.
- `regalloc_dtk.rs` (~4282 tok, huge) — Copyright 2025 STARGA Inc.
- `scev.rs` (~8574 tok, huge) — Copyright 2025 STARGA Inc.
### `src/package/`

- `manifest.rs` (~310 tok, medium) — Copyright 2025 STARGA Inc.
- `mod.rs` (~1567 tok, huge) — Copyright 2025 STARGA Inc.
### `src/parser/`

- `expand_bimap.rs` (~16544 tok, huge) — An `#[bimap]` attribute on an `enum` declares a one-to-one correspondence
- `trivia.rs` (~3811 tok, huge) — Copyright 2025 STARGA Inc.
### `src/phf/`

- `mod.rs` (~4955 tok, huge) — Copyright 2025 STARGA Inc.
### `src/`

- `pipeline.rs` (~5750 tok, huge) — Copyright 2025 STARGA Inc.
### `src/project/`

- `link.rs` (~2603 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `module_table.rs` (~5820 tok, huge) — Copyright 2025 STARGA Inc.
- `stdlib.rs` (~3656 tok, huge) — Copyright 2025 STARGA Inc.
- `substrate_link.rs` (~2330 tok, huge) — Copyright 2025 STARGA Inc.
### `src/`

- `python.rs` (~1082 tok, large) — Copyright 2025 STARGA Inc.
### `src/runtime/`

- `gpu.rs` (~288 tok, medium) — Experimental GPU backend contract for MIND.
### `src/`

- `runtime_interface.rs` (~573 tok, large) — Describes a tensor visible to the runtime.
### `src/runtime/`

- `mod.rs` (~92 tok, small) — Runtime abstractions for execution backends.
- `types.rs` (~1105 tok, large) — Shared runtime surface types for execution backends.
### `src/shapes/`

- `engine.rs` (~1882 tok, huge) — A rank-N tensor shape represented as a list of extents.
- `mod.rs` (~4170 tok, huge) — Copyright 2025 STARGA Inc.
### `src/stdlib/`

- `mod.rs` (~169 tok, small) — Copyright 2025 STARGA Inc.
- `tensor.rs` (~391 tok, medium) — Copyright 2025 STARGA Inc.
### `src/`

- `target.rs` (~5031 tok, huge) — Copyright 2025-2026 STARGA Inc.
### `src/test/`

- `mod.rs` (~8383 tok, huge) — Copyright 2025 STARGA Inc.
### `src/type_checker/`

- `resolve.rs` (~15148 tok, huge) — Copyright 2025 STARGA Inc.
### `src/types/`

- `infer.rs` (~448 tok, medium) — Copyright 2025 STARGA Inc.
- `intern.rs` (~1554 tok, huge) — Copyright 2025 STARGA Inc.
- `mod.rs` (~1037 tok, large) — Copyright 2025 STARGA Inc.
- `value.rs` (~297 tok, medium) — Copyright 2025 STARGA Inc.
### `src/workspace/`

- `mod.rs` (~4906 tok, huge) — Copyright 2025 STARGA Inc.
### `std/`

- `aes_gcm.mind` (~5400 tok, huge) — std/aes_gcm.mind — AES-128 (FIPS 197) + AES-128-GCM (NIST SP 800-38D) in
- `arena.mind` (~1323 tok, large) — std.arena — bump-pointer region allocator.
- `async.mind` (~2460 tok, huge) — std/async.mind -- RFC 0011 Phase A: Scheduler injection + Sender/Receiver
- `blas.mind` (~2518 tok, huge) — std/blas.mind — RFC 0006 Track A: pure-MIND surface over the six
- `chacha20_poly1305.mind` (~3972 tok, huge) — std/chacha20_poly1305.mind — ChaCha20-Poly1305 AEAD (RFC 8439) in pure MIND.
- `cli.mind` (~3211 tok, huge) — std/cli.mind — RFC 0013 Tier 1 Phase 1: argv-parsing surface.
- `detmath.mind` (~4059 tok, huge) — std.detmath — deterministic transcendental tier (roadmap Phase 17.6).
- `ecdsa_p256.mind` (~6517 tok, huge) — std/ecdsa_p256.mind — ECDSA signature VERIFICATION on NIST P-256
- `fs.mind` (~5043 tok, huge) — std/fs.mind — Task #268: POSIX filesystem surface in pure MIND.
- `hkdf.mind` (~1547 tok, huge) — std/hkdf.mind — HMAC-SHA256 (RFC 2104) + HKDF (RFC 5869) in pure MIND.
- `hpack.mind` (~9694 tok, huge) — std/hpack.mind — HPACK header-compression DECODING (RFC 7541) in pure MIND.
- `http2_frame.mind` (~4191 tok, huge) — std/http2_frame.mind — HTTP/2 framing layer (RFC 9113 §3.4, §4.1, §6) in
- `http.mind` (~6682 tok, huge) — std/http.mind — HTTP/1.1 client over std.net (task #XXX).
- `io_canon.mind` (~2624 tok, huge) — std.io_canon — canonical completion ordering for deterministic I/O.
- `io.mind` (~1719 tok, huge) — std/io.mind — RFC 0005 Phase 2: pure-MIND I/O surface.
- `iouring.mind` (~18767 tok, huge) — std.iouring — minimal io_uring binding (Linux). The physical-I/O reap source
- `json.mind` (~15391 tok, huge) — std/json.mind -- RFC 8259 / ECMA-404 subset parser (task #269, cargo-retirement track).
- `keccak.mind` (~3926 tok, huge) — std/keccak.mind — Keccak / SHA-3 + SHAKE (FIPS 202) in pure MIND.
- `llvm.mind` (~11108 tok, huge) — std/llvm.mind — RFC 0010 Phase F: hand-written MIND extern "C" bindings
- `map.mind` (~1538 tok, huge) — std/map.mind — RFC 0005 Phase 2: pure-MIND insertion-ordered map.
- `mlir.mind` (~11056 tok, huge) — std/mlir.mind — RFC 0010 Phase E: hand-written MIND extern "C" bindings
- `mlkem768.mind` (~6449 tok, huge) — std/mlkem768.mind — ML-KEM-768 (FIPS 203, "Kyber") in pure MIND.
- `net.mind` (~2381 tok, huge) — std/net.mind — Task #268: POSIX socket surface in pure MIND.
- `process.mind` (~3084 tok, huge) — std/process.mind — Task #268: subprocess + process environment in pure MIND.
- `reactor.mind` (~1420 tok, large) — std.reactor — deterministic per-connection request-id allocation.
- `regex.mind` (~9536 tok, huge) — std/regex.mind -- POSIX ERE subset NFA engine (task #269, cargo-retirement track).
- `ring.mind` (~1407 tok, large) — std.ring — fixed-capacity byte ring buffer (FIFO).
- `rsa_pss.mind` (~2453 tok, huge) — std/rsa_pss.mind — RSASSA-PSS signature VERIFICATION (RFC 8017 §8.1.2) with
- `sha256.mind` (~3643 tok, huge) — std/sha256.mind — FIPS 180-4 SHA-256 in pure MIND.
- `sha512.mind` (~5100 tok, huge) — std/sha512.mind — FIPS 180-4 SHA-512 and SHA-384 in pure MIND.
- `string.mind` (~2725 tok, huge) — std/string.mind — RFC 0005 Phase 2: pure-MIND String.
- `time.mind` (~257 tok, medium) — std.time — wall-clock access for evidence / audit timestamps.
- `tls13_finished.mind` (~1409 tok, large) — std/tls13_finished.mind — TLS 1.3 Finished-message MAC + transcript hash
- `tls13_handshake.mind` (~5466 tok, huge) — std/tls13_handshake.mind — TLS 1.3 handshake CRYPTO ORCHESTRATION in pure
- `tls13_keyschedule.mind` (~3220 tok, huge) — std/tls13_keyschedule.mind — TLS 1.3 key schedule (RFC 8446 §7.1) in pure MIND.
- `tls13_record.mind` (~2171 tok, huge) — std/tls13_record.mind — TLS 1.3 record-layer protection (RFC 8446 §5.1-5.3)
- `toml.mind` (~10301 tok, huge) — std/toml.mind -- TOML 1.0 subset parser (task #258, cargo-retirement track).
- `tui.mind` (~4815 tok, huge) — std/tui.mind — RFC 0013 Tier 1 (c): minimal pure-MIND TUI surface.
- `vec.mind` (~1100 tok, large) — std/vec.mind — RFC 0005 Phase 2: pure-MIND growable vector.
- `x25519.mind` (~4019 tok, huge) — std/x25519.mind — X25519 (RFC 7748 §5) Curve25519 Montgomery-ladder ECDH in
- `x25519mlkem768.mind` (~1554 tok, huge) — std/x25519mlkem768.mind — X25519MLKEM768 post-quantum hybrid key exchange in
- `x509.mind` (~7131 tok, huge) — std/x509.mind — minimal X.509v3 DER parsing + RSA PKCS#1 v1.5 (SHA-256)
### `tests/`

- `aggshape_reject.rs` (~1842 tok, huge) — Copyright 2025 STARGA Inc.
- `alias_miscompile_run.rs` (~1338 tok, large) — Copyright 2025 STARGA Inc.
- `array_ctor_push_get_run.rs` (~907 tok, large) — Copyright 2025 STARGA Inc.
- `array_load_bounds_and_dtype.rs` (~2007 tok, huge) — Copyright 2025 STARGA Inc.
- `array_oob_trap_run.rs` (~2030 tok, huge) — Copyright 2025 STARGA Inc.
- `array_store_run.rs` (~2173 tok, huge) — Copyright 2026 STARGA Inc.
- `array_surface_run.rs` (~875 tok, large) — Copyright 2025 STARGA Inc.
- `array_u64_element_shift_run.rs` (~1034 tok, large) — Copyright 2025 STARGA Inc.
### `tests/autodiff/`

- `matmul_gradient.mind` (~167 tok, small) — Autodiff test: MatMul gradient computation
### `tests/`

- `autodiff_preview.rs` (~398 tok, medium) — Copyright 2025 STARGA Inc.
- `autodiff.rs` (~2672 tok, huge) — Gradient for x*x accumulates two paths: d/dx (x*x) = x + x.
### `tests/autodiff/`

- `simple_gradient.mind` (~80 tok, small) — Autodiff test: Simple scalar gradient
### `tests/backend/`

- `cpu_available.mind` (~52 tok, small) — Backend test: CPU backend availability
- `gpu_graceful_failure.mind` (~73 tok, small) — Backend test: GPU backend graceful failure
### `tests/`

- `bare_variant_ambiguity_run.rs` (~1944 tok, huge) — Copyright 2025 STARGA Inc.
- `bare_variant_ctor_run.rs` (~1000 tok, large) — Copyright 2025 STARGA Inc.
- `bimap_derive.rs` (~5697 tok, huge) — Copyright 2025 STARGA Inc.
- `bitwise_no_panic_any_feature.rs` (~1848 tok, huge) — Copyright 2025 STARGA Inc.
- `blas_smoke.rs` (~6266 tok, huge) — Copyright 2025 STARGA Inc.
- `blas_vec_q16_smoke.rs` (~5639 tok, huge) — Copyright 2025 STARGA Inc.
- `blas_vec_smoke.rs` (~2130 tok, huge) — Copyright 2025 STARGA Inc.
- `bool_literal_value_run.rs` (~785 tok, large) — Copyright 2025 STARGA Inc.
- `bug6_tuple_destructure_u64_run.rs` (~1405 tok, large) — Copyright 2025 STARGA Inc.
- `bug_f4_closure_shadow_run.rs` (~1612 tok, huge) — Copyright 2025 STARGA Inc.
- `build_run_runnable_blocker_gate.rs` (~1123 tok, large) — Copyright 2025 STARGA Inc. Licensed under the Apache License, Version 2.0.
- `bytes_buffer_run.rs` (~715 tok, large) — Copyright 2025 STARGA Inc.
- `bytes_fixed_into_vec_run.rs` (~1660 tok, huge) — Copyright 2025 STARGA Inc.
- `bytes_zero_run.rs` (~889 tok, large) — Copyright 2025 STARGA Inc.
- `cerebras_stencil_tile.rs` (~1929 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `chacha20_poly1305_smoke.rs` (~2222 tok, huge) — Copyright 2025 STARGA Inc.
- `char_literal_run.rs` (~995 tok, large) — Copyright 2025 STARGA Inc.
- `cli_buffers.rs` (~459 tok, medium) — Copyright 2025 STARGA Inc.
- `cli_build.rs` (~648 tok, large) — Copyright 2025 STARGA Inc.
- `cli_eval.rs` (~502 tok, large) — Copyright 2025 STARGA Inc.
- `cli_exec.rs` (~558 tok, large) — Copyright 2025 STARGA Inc.
- `cli_tensor.rs` (~469 tok, medium) — Copyright 2025 STARGA Inc.
- `closure_capture_reject.rs` (~982 tok, large) — Copyright 2025 STARGA Inc.
- `closure_i64_capture.rs` (~1645 tok, huge) — Copyright 2025 STARGA Inc.
- `collection_ctor_run.rs` (~560 tok, large) — Copyright 2025 STARGA Inc.
- `collection_mutation_expr_position_run.rs` (~1003 tok, large) — Copyright 2025 STARGA Inc.
### `tests/common/`

- `mod.rs` (~668 tok, large) — Copyright 2025 STARGA Inc.
### `tests/`

- `compound_assign.rs` (~1034 tok, large) — Copyright 2025 STARGA Inc.
- `cond_truthiness.rs` (~891 tok, large) — Copyright 2025 STARGA Inc.
### `tests/conformance/cpu_baseline/`

- `autodiff_pairwise.runtime` (~1 tok, tiny) — 0
- `phase_10_5_const.mind` (~34 tok, tiny) — fn main() {
- `phase_10_5_logical.mind` (~32 tok, tiny) — fn main() {
- `phase_10_5_module.mind` (~31 tok, tiny) — module governance {
- `phase_10_5_struct.mind` (~29 tok, tiny) — fn main() {
- `simple_arith.ir` (~15 tok, tiny) — module {
- `simple_arith.mind` (~3 tok, tiny)
- `simple_arith.mlir` (~25 tok, tiny) — module {
- `simple_arith.runtime` (~1 tok, tiny) — 7
### `tests/conformance/gpu_profile/`

- `backend_unavailable.error` (~9 tok, tiny) — no backend available for target gpu
- `backend_unavailable.mind` (~2 tok, tiny)
### `tests/`

- `conformance.rs` (~129 tok, small)
- `CONFORMANCE_TESTS.md` (~1225 tok, large) — MIND Conformance Test Corpus
- `const_array_run.rs` (~884 tok, large) — Copyright 2025 STARGA Inc.
- `const_f64_array_run.rs` (~1148 tok, large) — Copyright 2025 STARGA Inc.
- `const_folding.rs` (~246 tok, medium) — Copyright 2025 STARGA Inc.
- `continue_in_match_arm_run.rs` (~1222 tok, large) — Copyright 2025 STARGA Inc.
- `conv2d_exec.rs` (~620 tok, large) — Copyright 2025 STARGA Inc.
- `conv2d_grad.rs` (~3194 tok, huge) — Copyright 2025 STARGA Inc.
- `conv2d_types.rs` (~360 tok, medium) — Copyright 2025 STARGA Inc.
- `cross_module_cdylib_compose.rs` (~4112 tok, huge) — Copyright 2025 STARGA Inc.
- `cross_module_enum_run.rs` (~1132 tok, large) — Copyright 2025 STARGA Inc.
- `cross_module_field_access_run.rs` (~1504 tok, huge) — Copyright 2025 STARGA Inc.
- `cross_module.rs` (~1332 tok, large) — Copyright 2025 STARGA Inc.
### `tests/cross_substrate_identity/array-store-branch/`

- `manifest.toml` (~564 tok, large) — version = "1"
- `reference_hashes.toml` (~410 tok, medium) — avx2 = "8a50c515a5de1786b0102fecbb692054299acaa4878a9c2978c7b69cb647561f"
### `tests/cross_substrate_identity/array-store-loop/`

- `manifest.toml` (~636 tok, large) — version = "1"
- `reference_hashes.toml` (~445 tok, medium) — avx2 = "50b3efb2da5eb89764302f3781d390e9121c12cb8f87faaa69f5d0f46bf39f31"
### `tests/cross_substrate_identity/bimap-phf/`

- `manifest.toml` (~780 tok, large) — version = "1"
- `reference_hashes.toml` (~885 tok, large) — avx2 = "33483ee02a0b062becb8c8f7c1078e0bc17c543298971514e37e371b2b86bc01"
### `tests/cross_substrate_identity/collatz/`

- `manifest.toml` (~511 tok, large) — version = "1"
- `reference_hashes.toml` (~451 tok, medium) — avx2 = "87fbc45b398c9d752a438cbcffd4a34076fce3d37556db30eb18fbbc55cc849a"
### `tests/cross_substrate_identity/dot-f32-v-4093/`

- `manifest.toml` (~638 tok, large) — version = "1"
- `reference_hashes.toml` (~584 tok, large) — avx2 = "a132f7b970b647cd158f591d764c19ec41a8cf27c398c87758f74efb5a8a22c0"
### `tests/cross_substrate_identity/dot-i16-4096/`

- `manifest.toml` (~384 tok, medium) — version = "1"
- `reference_hashes.toml` (~264 tok, medium) — avx2 = "af0fc3cf1b510f8f7306a5d7250ae25a52b35281a7cefff2a0ac94b0cd80a127"
### `tests/cross_substrate_identity/dot-l1-q16/`

- `manifest.toml` (~157 tok, small) — version = "1"
- `reference_hashes.toml` (~206 tok, medium) — avx2 = "ce7e2a80515e123f5d4fbb77d841f0d6c56fcbc690bba2e2ff81e45765843b34"
### `tests/cross_substrate_identity/dot-l2-q16/`

- `manifest.toml` (~436 tok, medium) — version = "1"
- `reference_hashes.toml` (~377 tok, medium) — avx2 = "1d7f272b85e5f0fd7cf473086fb1da558a723134ff02ef30a4323eb757209823"
### `tests/cross_substrate_identity/galperin-pi/`

- `manifest.toml` (~516 tok, large) — version = "1"
- `reference_hashes.toml` (~488 tok, medium) — avx2 = "59dfa3f2abe4d71fdc42a54b49da2526f002c8e84a777184fe6a08d25adeca04"
### `tests/cross_substrate_identity/gemm-i8-64x64x64/`

- `manifest.toml` (~437 tok, medium) — version = "1"
- `reference_hashes.toml` (~270 tok, medium) — avx2 = "917d353b18fd7f5ea4dab7dd02b786f5ccc4a2d954f695084ca0a88214d699c7"
### `tests/cross_substrate_identity/gemm-i8-mt-64x64x64/`

- `manifest.toml` (~481 tok, medium) — version = "1"
- `reference_hashes.toml` (~391 tok, medium) — avx2 = "917d353b18fd7f5ea4dab7dd02b786f5ccc4a2d954f695084ca0a88214d699c7"
### `tests/cross_substrate_identity/gemm-i8-vnni-64x64x64/`

- `manifest.toml` (~536 tok, large) — version = "1"
- `reference_hashes.toml` (~385 tok, medium) — avx2 = "917d353b18fd7f5ea4dab7dd02b786f5ccc4a2d954f695084ca0a88214d699c7"
### `tests/cross_substrate_identity/gemm-q16-64x64x64/`

- `manifest.toml` (~391 tok, medium) — version = "1"
- `reference_hashes.toml` (~225 tok, medium) — avx2 = "92e2cb75d74d83a4a398d78d9ac560f195279c31814972c892f856f675faea0f"
### `tests/cross_substrate_identity/gemm-q16-fused-64x64x64/`

- `manifest.toml` (~529 tok, large) — version = "1"
- `reference_hashes.toml` (~367 tok, medium) — avx2 = "92e2cb75d74d83a4a398d78d9ac560f195279c31814972c892f856f675faea0f"
### `tests/cross_substrate_identity/gemv-i16-256x256/`

- `manifest.toml` (~375 tok, medium) — version = "1"
- `reference_hashes.toml` (~219 tok, medium) — avx2 = "3238e8c7e1e9ee9937503700f63eda350fcd10e7db28d470c3dbc26592d0a936"
### `tests/cross_substrate_identity/gemv-q16-256x256/`

- `manifest.toml` (~310 tok, medium) — version = "1"
- `reference_hashes.toml` (~209 tok, medium) — avx2 = "dfdf890874472ee369da524955995889c39bc6da770e4e2b1d0d69315e17611a"
### `tests/cross_substrate_identity/grammar-mask/`

- `manifest.toml` (~575 tok, large) — version = "1"
- `reference_hashes.toml` (~341 tok, medium) — avx2 = "4d46a747338253886a91b02f4b832dc8675dd315b9500490493b0b977295627b"
### `tests/cross_substrate_identity/lorenz-q16/`

- `manifest.toml` (~681 tok, large) — version = "1"
- `reference_hashes.toml` (~562 tok, large) — avx2 = "04da6abc69e63314331e88a7a9670ce5c9e90ddaa2bf5f5dc53526f56477de80"
### `tests/cross_substrate_identity/matmul-f32-v-64x64/`

- `manifest.toml` (~523 tok, large) — version = "1"
- `reference_hashes.toml` (~592 tok, large) — avx2 = "ec5adb991372fcfc16b964ba566f05fb44701fcf8bbde2a5453fed294e1d0175"
### `tests/cross_substrate_identity/q16-arith-chain/`

- `manifest.toml` (~432 tok, medium) — version = "1"
- `reference_hashes.toml` (~356 tok, medium) — avx2 = "ce93cdeb0e650c1c8e0cd05687ed986bbdbac691b6a8742e155b8ffd65997d78"
### `tests/cross_substrate_identity/`

- `README.md` (~1280 tok, large) — cross_substrate_identity — the internal mind-bench reproducibility gate
### `tests/cross_substrate_identity/scalar-cast-conv/`

- `manifest.toml` (~1041 tok, large) — version = "1"
### `tests/cross_substrate_identity/scalar-cast-conv-narrow/`

- `manifest.toml` (~1199 tok, large) — version = "1"
- `reference_hashes.toml` (~591 tok, large) — avx2 = "9e8e4278dbb52705a12c06c2e5ec59f8f994c7f0227617f5f0884cba0608976b"
### `tests/cross_substrate_identity/scalar-cast-conv/`

- `reference_hashes.toml` (~603 tok, large) — avx2 = "a38aaa5196baad698f60edc9d2ffc44aac43540ae74aa3bcaf2687fd37a0b8c2"
### `tests/cross_substrate_identity/scalar-float-f64/`

- `manifest.toml` (~778 tok, large) — version = "1"
- `reference_hashes.toml` (~532 tok, large) — avx2 = "7592a52a5e10a2f24469765f71ce1f9f8ebd9efb51904cf9a18f310d33b3c92d"
### `tests/cross_substrate_identity/struct-handle-roundtrip/`

- `manifest.toml` (~403 tok, medium) — version = "1"
- `reference_hashes.toml` (~343 tok, medium) — avx2 = "018a335a0e9fc397c6f41cba4fc2617f0cf8d1326c5dbf77d53e27feacaeb64c"
### `tests/cross_substrate_identity/u64-ops/`

- `manifest.toml` (~633 tok, large) — version = "1"
- `reference_hashes.toml` (~421 tok, medium) — avx2 = "133eefad053de51b9ca57c8802f60814c8489e1acb21b74c3e549358199af7f3"
### `tests/cross_substrate_identity/`

- `xnode_driver.c` (~2833 tok, huge)
### `tests/`

- `crypto_vectors_driver.py` (~2653 tok, huge) — # Official-vector driver for std/aes_gcm.mind + std/hkdf.mind (pure-MIND crypto).
- `diagnostics_parse.rs` (~1112 tok, large) — Copyright 2025 STARGA Inc.
- `diagnostics.rs` (~688 tok, large) — Copyright 2025 STARGA Inc.
- `digit_separator_run.rs` (~651 tok, large) — Copyright 2025 STARGA Inc.
- `dot_enum_variant_run.rs` (~806 tok, large) — Copyright 2025 STARGA Inc.
- `dot_variants.rs` (~284 tok, medium) — Copyright 2025 STARGA Inc.
- `ecdsa_p256_driver.py` (~2752 tok, huge) — # Ground-truth driver for std/ecdsa_p256.mind (pure-MIND ECDSA P-256/SHA-256
- `emit_ir_for_loop.rs` (~369 tok, medium) — Regression test for #4: lowering a `for` loop to IR (the path `mindc --emit-ir`
- `enum_match_collision_run.rs` (~1033 tok, large) — Copyright 2025 STARGA Inc.
- `enum_match_run.rs` (~2361 tok, huge) — Copyright 2025 STARGA Inc.
- `enum_soundness.rs` (~1411 tok, large) — Copyright 2025 STARGA Inc.
- `enum_struct_variant_run.rs` (~1331 tok, large) — Copyright 2025 STARGA Inc.
- `exec_basic.rs` (~785 tok, large) — Copyright 2025 STARGA Inc.
- `expr_parser.rs` (~307 tok, medium) — Copyright 2025 STARGA Inc.
- `extern_c_phase_a.rs` (~2678 tok, huge) — Copyright 2025 STARGA Inc.
- `extern_c_phase_b.rs` (~5956 tok, huge) — Copyright 2025 STARGA Inc.
- `extern_c_phase_c.rs` (~3413 tok, huge) — Copyright 2025 STARGA Inc.
- `extern_c_safety_tag_informational.rs` (~1427 tok, large) — Copyright 2025 STARGA Inc.
- `extern_narrow_ret_run.rs` (~1576 tok, huge) — Copyright 2025 STARGA Inc.
- `f3_bare_enum_collision_run.rs` (~1392 tok, large) — Copyright 2025 STARGA Inc.
- `f64_abi_negative_control.rs` (~1870 tok, huge) — Copyright 2025 STARGA Inc.
- `f64_activation_lowering.rs` (~972 tok, large) — Copyright 2025 STARGA Inc.
- `f64_call_arg_run.rs` (~1028 tok, large) — Copyright 2025 STARGA Inc.
- `f64_literal_envelope.rs` (~1047 tok, large) — Each source literal is compiled via `mindc --emit-shared`; the exported
- `f64_loop_run.rs` (~1122 tok, large) — Copyright 2025 STARGA Inc.
- `fail_closed_cli_run.rs` (~1203 tok, large) — Regression tests for the two fail-closed CLI guards that shipped WITHOUT one.
- `ffi_header.rs` (~221 tok, medium) — Copyright 2025 STARGA Inc.
### `tests/fixtures/`

- `autodiff.mind` (~55 tok, small) — Minimal differentiable program for the --emit-grad-ir CLI test.
### `tests/fixtures/f64_abi_negative/`

- `README.md` (~146 tok, small) — f64 call-ABI negative control (paired with the #298 self-host f64 MLIR surface)
- `scale_neg.mind` (~23 tok, tiny)
- `scale_pos.mind` (~23 tok, tiny)
### `tests/fixtures/`

- `invalid_broadcast.mind` (~17 tok, tiny)
- `invalid.mind` (~6 tok, tiny)
- `simple.mind` (~3 tok, tiny)
- `test_phase_b_all_pass.mind` (~67 tok, small) — RFC 0008 Phase B test fixture — both tests pass.
- `test_phase_b_one_fail.mind` (~80 tok, small) — RFC 0008 Phase B test fixture — one pass, one fail.
### `tests/`

- `fmt_comment_placement.rs` (~2772 tok, huge) — Copyright 2025 STARGA Inc.
- `fmt_idempotence.rs` (~3658 tok, huge) — Copyright 2025 STARGA Inc.
- `fmt_ir_preservation.rs` (~2030 tok, huge) — Copyright 2025 STARGA Inc.
- `fmt_module_block_item_preserved.rs` (~926 tok, large) — Regression: `mindc fmt` must not drop item declarations nested inside a
- `fmt_stdlib_stability.rs` (~2754 tok, huge) — Copyright 2025 STARGA Inc.
- `fn_value_call_reject.rs` (~761 tok, large) — Copyright 2025 STARGA Inc.
- `for_continue_advances_run.rs` (~1549 tok, huge) — Copyright 2025 STARGA Inc. Licensed under the Apache License, Version 2.0.
- `for_continue_step_injection.rs` (~1302 tok, large) — Copyright 2025 STARGA Inc. Licensed under the Apache License, Version 2.0.
- `for_each_run.rs` (~859 tok, large) — Copyright 2025 STARGA Inc.
- `for_hygiene_run.rs` (~1442 tok, large) — Copyright 2025 STARGA Inc. Licensed under the Apache License, Version 2.0.
- `g2_differential_mlir.rs` (~10959 tok, huge) — Copyright 2025 STARGA Inc.
- `gather_preview.rs` (~288 tok, medium) — Copyright 2025 STARGA Inc.
- `generics_lowering.rs` (~1384 tok, large) — Copyright 2026 STARGA Inc. Licensed under the Apache License, Version 2.0.
- `genref_phase_jb.rs` (~3678 tok, huge) — Copyright 2025 STARGA Inc.
- `grad_wrt_resolve.rs` (~730 tok, large) — Copyright 2025 STARGA Inc.
- `hpack_driver.py` (~3027 tok, huge) — # Official-vector driver for std/hpack.mind (pure-MIND HPACK decoding,
- `http2_frame_driver.py` (~4195 tok, huge) — # Reference-vector driver for std/http2_frame.mind (pure-MIND HTTP/2 framing,
- `if_expr.rs` (~429 tok, medium) — Copyright 2025 STARGA Inc.
- `if_merge_shadow_318.rs` (~1435 tok, large) — Copyright 2025 STARGA Inc.
- `index_slice_grad.rs` (~289 tok, medium) — Copyright 2025 STARGA Inc.
- `index_slice_preview.rs` (~376 tok, medium) — Copyright 2025 STARGA Inc.
- `index_slice_types.rs` (~250 tok, medium) — Copyright 2025 STARGA Inc.
- `int_determinism.rs` (~1226 tok, large) — Copyright 2025 STARGA Inc.
- `intra_module_call_arity.rs` (~3085 tok, huge) — Copyright 2025 STARGA Inc.
- `int_suffix_literal.rs` (~930 tok, large) — Copyright 2025 STARGA Inc.
- `invariant_block_run.rs` (~720 tok, large) — Copyright 2025 STARGA Inc.
- `invariant_check_run.rs` (~695 tok, large) — Copyright 2025 STARGA Inc.
- `ir_core.rs` (~1689 tok, huge) — Ensure the unused const is kept alive in the SSA namespace but removed from code.
- `ir_load_save.rs` (~1257 tok, large) — Copyright 2025 STARGA Inc.
- `ir_lower.rs` (~1331 tok, large) — Copyright 2025 STARGA Inc.
- `ir_negative_literals.rs` (~1722 tok, huge) — Copyright 2025 STARGA Inc.
- `ir_stub.rs` (~219 tok, medium) — Copyright 2025 STARGA Inc.
### `tests/ir_verification/`

- `ssa_single_assignment.mind` (~46 tok, tiny) — IR verification test: SSA property validation
- `undefined_operand.mind` (~62 tok, small) — IR verification test: Undefined operand detection
### `tests/`

- `issue_201_202_unary_not_const_ctx.rs` (~1132 tok, large) — Copyright 2025 STARGA Inc.
- `issue_246_bitnot_operator.rs` (~571 tok, large) — Copyright 2025 STARGA Inc.
- `issue_246_tensor_named_dtype.rs` (~525 tok, large) — Copyright 2025 STARGA Inc.
- `issue_263_method_chain_newline.rs` (~869 tok, large) — Copyright 2025 STARGA Inc.
- `issue_263_use_path_import.rs` (~749 tok, large) — Copyright 2025 STARGA Inc.
- `issue_263_usize_suffix_literal.rs` (~770 tok, large) — Copyright 2025 STARGA Inc.
- `keccak_driver.py` (~1507 tok, huge) — # Official-vector driver for std/keccak.mind (pure-MIND FIPS 202).
### `tests/lexical/`

- `invalid_keywords_as_identifiers.mind` (~45 tok, tiny) — Lexical test: Keywords cannot be used as identifiers
- `numeric_literals.mind` (~74 tok, small) — Lexical test: Numeric literal formats
- `valid_identifiers.mind` (~72 tok, small) — Lexical test: Valid identifier formats
### `tests/`

- `linalg_grad.rs` (~315 tok, medium) — Copyright 2025 STARGA Inc.
- `linalg_preview.rs` (~291 tok, medium) — Copyright 2025 STARGA Inc.
- `lint_infrastructure.rs` (~2540 tok, huge) — Copyright 2025 STARGA Inc.
- `loop_run.rs` (~764 tok, large) — Copyright 2025 STARGA Inc.
- `loud_fail_non_i64.rs` (~852 tok, large) — Release-readiness P1.1 — the runnable-artifact ABI gate.
- `map_get_inference_run.rs` (~713 tok, large) — Copyright 2025 STARGA Inc.
- `map_runtime_run.rs` (~843 tok, large) — Copyright 2025 STARGA Inc.
- `map_surface_run.rs` (~949 tok, large) — Copyright 2025 STARGA Inc.
- `match_arm_stmt_run.rs` (~846 tok, large) — Copyright 2025 STARGA Inc.
- `match_fallback_fail_closed.rs` (~1750 tok, huge) — Copyright 2025 STARGA Inc.
- `match_scrutinee_once.rs` (~1521 tok, huge) — Copyright 2025 STARGA Inc.
- `merge_kind_order_symmetry.rs` (~994 tok, large) — A merge's signedness must not depend on ARM ORDER.
- `method_call.rs` (~397 tok, medium) — Copyright 2025 STARGA Inc.
- `mic3_array_store_roundtrip.rs` (~1054 tok, large) — Copyright 2026 STARGA Inc.
- `mic3_break_continue_string_roundtrip.rs` (~1668 tok, huge) — Copyright 2026 STARGA Inc.
- `mic3_cli_emit.rs` (~2431 tok, huge) — Copyright 2025 STARGA Inc.
- `mic3_const_dense_tensor_roundtrip.rs` (~866 tok, large) — Copyright 2026 STARGA Inc.
- `mic3_parser_dos.rs` (~1891 tok, huge) — Copyright 2025 STARGA Inc.
- `micb_dos_reject.rs` (~2341 tok, huge) — Copyright 2025 STARGA Inc.
- `mindc_build_phase_a.rs` (~5576 tok, huge) — Copyright 2025 STARGA Inc.
- `mindc_cache_phase_f.rs` (~6231 tok, huge) — Copyright 2025 STARGA Inc.
- `mindc_deps_phase_de.rs` (~6823 tok, huge) — Copyright 2025 STARGA Inc.
- `mindc_doc_phase1.rs` (~2957 tok, huge) — Copyright 2025 STARGA Inc.
- `mindc_inspect.rs` (~1135 tok, large) — Integration test for `mindc inspect` — the mic@3 artifact decoder/differ.
### `tests/mindcraft/check/`

- `clean.mind` (~11 tok, tiny) — fn add(a: i64, b: i64) -> i64 {
### `tests/`

- `mindcraft_check_cli.rs` (~3370 tok, huge) — Copyright 2025 STARGA Inc.
### `tests/mindcraft/check/`

- `drifted.mind` (~12 tok, tiny) — fn add(a: i64,  b: i64) -> i64 {
### `tests/`

- `mindcraft_check_fix.rs` (~1667 tok, huge) — Copyright 2025 STARGA Inc.
### `tests/mindcraft/check/`

- `ignored.mind` (~9 tok, tiny) — fn ignored_fn() -> i64 {
### `tests/`

- `mindcraft_check_lsp_reporter.rs` (~1968 tok, huge) — Copyright 2025 STARGA Inc.
### `tests/mindcraft/check/subdir/`

- `nested.mind` (~9 tok, tiny) — fn nested(x: i64) -> i64 {
### `tests/mindcraft/check/`

- `with_lint.mind` (~16 tok, tiny) — fn add(a: i64, b: i64) -> i64 {
### `tests/mindcraft/fmt/`

- `01_indent_if_else.in.mind` (~46 tok, tiny) — fn classify(x: i64) -> i64 {
- `01_indent_if_else.out.mind` (~48 tok, tiny) — fn classify(x: i64) -> i64 {
- `02_struct_literal_multiline.in.mind` (~28 tok, tiny) — fn make_point(a: i64, b: i64) -> Point {
- `02_struct_literal_multiline.out.mind` (~28 tok, tiny) — fn make_point(a: i64, b: i64) -> Point {
- `03_fn_args_multiline.in.mind` (~34 tok, tiny) — fn add(a: i64, b: i64) -> i64 {
- `03_fn_args_multiline.out.mind` (~35 tok, tiny) — fn add(a: i64, b: i64) -> i64 {
- `04_trailing_comma_toggle.in.mind` (~33 tok, tiny) — fn make_config(w: i64, h: i64) -> Config {
- `04_trailing_comma_toggle.out.mind` (~33 tok, tiny) — fn make_config(w: i64, h: i64) -> Config {
- `05_internal_whitespace.in.mind` (~32 tok, tiny) — fn calc(a: i64, b: i64, c: i64) -> i64 {
- `05_internal_whitespace.out.mind` (~33 tok, tiny) — fn calc(a: i64, b: i64, c: i64) -> i64 {
- `06_comment_attachment.in.mind` (~48 tok, tiny) — Copyright 2025 STARGA Inc.
- `06_comment_attachment.out.mind` (~48 tok, tiny) — Copyright 2025 STARGA Inc.
- `07_string_literal_passthrough.in.mind` (~14 tok, tiny) — fn get_message() -> i64 {
- `07_string_literal_passthrough.out.mind` (~14 tok, tiny) — fn get_message() -> i64 {
### `tests/`

- `mindcraft_fmt_cli.rs` (~2925 tok, huge) — Copyright 2025 STARGA Inc.
- `mindcraft_fmt_fix.rs` (~1445 tok, large) — Copyright 2025 STARGA Inc.
- `mindcraft_fmt_fixtures.rs` (~1300 tok, large) — Copyright 2025 STARGA Inc.
### `tests/mindcraft/lint/naming_convention/`

- `negative.mind` (~61 tok, small) — Negative fixture: all names follow canonical conventions.
- `positive_bad_const.mind` (~39 tok, tiny) — Positive fixture: const name violates SCREAMING_SNAKE_CASE.
- `positive_bad_fn.mind` (~37 tok, tiny) — Positive fixture: function name violates lower_snake_case.
- `positive_bad_struct.mind` (~39 tok, tiny) — Positive fixture: struct name violates UpperCamelCase.
### `tests/`

- `mindcraft_lint_naming_convention.rs` (~1112 tok, large) — Copyright 2025 STARGA Inc.
### `tests/mindcraft/lint/q16_overflow/`

- `edge_constant.mind` (~54 tok, small) — Edge case: i32 * literal constant still triggers if no >>16 shift.
- `negative.mind` (~67 tok, small) — Negative fixture: proper Q16.16 multiply with >>16 narrowing.
- `positive.mind` (~70 tok, small) — Positive fixture: bare i32 * i32 without >>16 narrowing.
### `tests/`

- `mindcraft_lint_q16_overflow.rs` (~867 tok, large) — Copyright 2025 STARGA Inc.
### `tests/mindcraft/lint/shadowing/`

- `negative.mind` (~34 tok, tiny) — Negative fixture: two different names — no shadowing.
- `positive.mind` (~53 tok, small) — Positive fixture: two `let x` bindings in the same function body.
### `tests/`

- `mindcraft_lint_shadowing.rs` (~980 tok, large) — Copyright 2025 STARGA Inc.
### `tests/mindcraft/lint/`

- `trailing_ws_clean.mind` (~10 tok, tiny) — fn foo() -> i64 {
- `trailing_ws_dirty.mind` (~11 tok, tiny) — fn foo() -> i64 {
### `tests/mindcraft/lint/unused_import/`

- `negative.mind` (~53 tok, small) — Negative fixture: `use std.vec` is declared AND the `vec` identifier
- `positive.mind` (~46 tok, tiny) — Positive fixture: `use std.vec` is declared but no symbol from vec
### `tests/`

- `mindcraft_lint_unused_import.rs` (~708 tok, large) — Copyright 2025 STARGA Inc.
- `mindcraft_lint_vec_check.rs` (~549 tok, large) — Copyright 2025 STARGA Inc.
### `tests/mindcraft/`

- `STABILITY_SKIP_LIST.md` (~408 tok, medium) — Formatter Stability Skip List
### `tests/`

- `mindc.rs` (~1851 tok, huge) — Copyright 2025 STARGA Inc.
- `mindc_test_phase_b.rs` (~3572 tok, huge) — Copyright 2025 STARGA Inc.
- `mindc_workspace_phase_c.rs` (~4166 tok, huge) — Copyright 2025 STARGA Inc.
### `tests/mindfuzz_cross_substrate/known_environmental/`

- `README.md` (~314 tok, medium) — Known-environmental fuzzer artifacts
- `seed_deadbeef_prog000_e1001_bitwise.mind` (~242 tok, medium) — NOT A DIVERGENCE — misfiled build-capability refusal (issue #72 fuzzer).
### `tests/mindfuzz_cross_substrate/reproducers/`

- `fuzz_oracle_failure_seed_deadbeef_prog000.mind` (~218 tok, medium) — ORACLE FAILURE — a stage did not complete; NOT a proven cross-substrate divergence (issue #72 fuzzer)
- `fuzz_oracle_failure_seed_deadbeef_prog006.mind` (~157 tok, small) — DIVERGENCE REPRODUCER (issue #72 fuzzer)
### `tests/`

- `mindfuzz_cross_substrate.rs` (~16542 tok, huge) — Copyright 2025 STARGA Inc.
### `tests/mindfuzz_cross_substrate/staged/`

- `manifest.tsv` (~357 tok, medium) — scalar_arith_step000	f	3735928559	64	5e39820a2a8325417e39057f19ba9bceec01bd2068c
- `scalar_accum_step000.mind` (~154 tok, small) — MIND-Fuzz seed: scalar accumulator with a return-feeding literal.
- `scalar_accum_step001.mind` (~162 tok, small) — MIND-Fuzz seed: scalar accumulator with a return-feeding literal.
- `scalar_accum_step002.mind` (~169 tok, small) — MIND-Fuzz seed: scalar accumulator with a return-feeding literal.
- `scalar_accum_step003.mind` (~179 tok, small) — MIND-Fuzz seed: scalar accumulator with a return-feeding literal.
- `scalar_accum_step004.mind` (~182 tok, small) — MIND-Fuzz seed: scalar accumulator with a return-feeding literal.
- `scalar_accum_step005.mind` (~186 tok, small) — MIND-Fuzz seed: scalar accumulator with a return-feeding literal.
- `scalar_accum_step006.mind` (~186 tok, small) — MIND-Fuzz seed: scalar accumulator with a return-feeding literal.
- `scalar_arith_step000.mind` (~159 tok, small) — MIND-Fuzz seed: small pure-integer scalar function.
- `scalar_arith_step001.mind` (~167 tok, small) — MIND-Fuzz seed: small pure-integer scalar function.
- `scalar_arith_step002.mind` (~175 tok, small) — MIND-Fuzz seed: small pure-integer scalar function.
- `scalar_arith_step003.mind` (~184 tok, small) — MIND-Fuzz seed: small pure-integer scalar function.
- `scalar_arith_step004.mind` (~188 tok, small) — MIND-Fuzz seed: small pure-integer scalar function.
- `scalar_arith_step005.mind` (~192 tok, small) — MIND-Fuzz seed: small pure-integer scalar function.
- `scalar_arith_step006.mind` (~192 tok, small) — MIND-Fuzz seed: small pure-integer scalar function.
### `tests/`

- `mlir_broadcast.rs` (~1479 tok, large) — Copyright 2025 STARGA Inc.
- `mlir_build.rs` (~1412 tok, large) — Copyright 2025 STARGA Inc.
- `mlir_exec.rs` (~833 tok, large) — Copyright 2025 STARGA Inc.
- `mlir_export_indexing.rs` (~414 tok, medium) — Copyright 2025 STARGA Inc.
- `mlir_export_linalg.rs` (~863 tok, large) — Copyright 2025 STARGA Inc.
- `mlir_export_reductions.rs` (~1691 tok, huge) — Copyright 2025 STARGA Inc.
- `mlir_export.rs` (~573 tok, large) — Copyright 2025 STARGA Inc.
- `mlir_export_shape.rs` (~348 tok, medium) — Copyright 2025 STARGA Inc.
- `mlir_file_and_lower.rs` (~639 tok, large) — Copyright 2025 STARGA Inc.
- `mlir_gpu.rs` (~314 tok, medium) — Copyright 2025 STARGA Inc.
- `mlir_jit.rs` (~285 tok, medium) — Copyright 2025 STARGA Inc.
- `mlir_lowering.rs` (~1490 tok, large)
- `mlir_opt.rs` (~424 tok, medium) — Copyright 2025 STARGA Inc.
- `mlkem768_driver.py` (~1699 tok, huge) — # Reference-vector driver for std/mlkem768.mind (pure-MIND ML-KEM-768,
- `module_const_run.rs` (~774 tok, large) — Copyright 2025 STARGA Inc.
- `module_decl_run.rs` (~643 tok, large) — Copyright 2025 STARGA Inc.
- `module_enum_match_run.rs` (~1034 tok, large) — Copyright 2025 STARGA Inc.
- `module_non_fn_call_reject.rs` (~1215 tok, large) — Copyright 2025 STARGA Inc.
- `narrow_call_abi.rs` (~1374 tok, large) — Copyright 2025 STARGA Inc.
- `narrowing_check.rs` (~438 tok, medium) — Regression test for the silent i64->i32 narrowing miscompile found by MIND-Fuzz
- `narrow_local_mask_run.rs` (~838 tok, large) — Copyright 2025 STARGA Inc.
- `narrow_locals_leak_run.rs` (~1642 tok, huge) — Copyright 2025 STARGA Inc.
- `narrow_reassign_mask_run.rs` (~1342 tok, large) — Copyright 2025 STARGA Inc.
- `narrow_reassign_run.rs` (~733 tok, large) — Copyright 2026 STARGA Inc.
- `narrow_sig_abi_run.rs` (~1211 tok, large) — Copyright 2025 STARGA Inc.
- `narrow_signedness_batch2_run.rs` (~2281 tok, huge) — Copyright 2025 STARGA Inc.
- `narrow_signedness_batch_run.rs` (~1674 tok, huge) — Copyright 2025 STARGA Inc.
- `narrow_tuple_pr216_run.rs` (~1454 tok, large) — Copyright 2025 STARGA Inc.
- `narrow_unsigned_div_zero_run.rs` (~1228 tok, large) — Copyright 2025 STARGA Inc.
- `native_opt_wiring.rs` (~1953 tok, huge) — Copyright 2025 STARGA Inc.
- `nested_block_surface_run.rs` (~1036 tok, large) — Copyright 2025 STARGA Inc.
- `nested_collection_run.rs` (~818 tok, large) — Copyright 2025 STARGA Inc.
- `nested_mut_thread_run.rs` (~1550 tok, huge) — Copyright 2025 STARGA Inc.
- `non_final_catch_all_match_run.rs` (~907 tok, large) — Copyright 2025 STARGA Inc. Licensed under the Apache License, Version 2.0.
- `ops_registry.rs` (~114 tok, small)
- `package_basic.rs` (~491 tok, medium) — Copyright 2025 STARGA Inc.
- `package_traversal.rs` (~905 tok, large) — Copyright 2025 STARGA Inc.
- `parse_match_and_ref.rs` (~3031 tok, huge) — Copyright 2025 STARGA Inc.
- `parse_phase10_surface.rs` (~4815 tok, huge) — Parse-target tests for Phase 10.5 / 10.6 surface-syntax acceptance.
- `parser_trivia.rs` (~2706 tok, huge) — Copyright 2025 STARGA Inc.
- `parser_unsigned_i64_literals.rs` (~1544 tok, huge) — Copyright 2025 STARGA Inc.
- `parse_try_operator.rs` (~1194 tok, large) — Copyright 2025 STARGA Inc.
- `pattern_guard_run.rs` (~945 tok, large) — Copyright 2025 STARGA Inc. Licensed under the Apache License, Version 2.0.
- `phase_g_keystone_bootstrap.rs` (~6353 tok, huge) — Copyright 2025 STARGA Inc.
- `pipeline.rs` (~1476 tok, large) — Copyright 2025 STARGA Inc.
- `reap_threshold.rs` (~2089 tok, huge) — Copyright 2025 STARGA Inc.
- `reductions_grad.rs` (~390 tok, medium) — Copyright 2025 STARGA Inc.
- `reductions_preview.rs` (~390 tok, medium) — Copyright 2025 STARGA Inc.
- `_ref_mic3_dump.rs` (~1816 tok, huge) — Committed self-host reference generator (A9b): reconstruct
- `regalloc_dtk_parity.rs` (~3134 tok, huge) — DTK slice 1 (#254) — parity consumer for src/opt/regalloc_dtk.rs.
- `region_phase_ja.rs` (~4239 tok, huge) — Copyright 2025 STARGA Inc.
- `relu_exec.rs` (~472 tok, medium) — Copyright 2025 STARGA Inc.
- `relu_preview.rs` (~279 tok, medium) — Copyright 2025 STARGA Inc.
- `repl_basic.rs` (~523 tok, large) — Copyright 2025 STARGA Inc.
- `reshape_runtime_dim_273.rs` (~797 tok, large) — Copyright 2025 STARGA Inc.
- `resolve_fn_body.rs` (~1450 tok, large) — Copyright 2025 STARGA Inc.
- `result_option_prelude_run.rs` (~910 tok, large) — Copyright 2025 STARGA Inc.
- `return_cond_type_reject.rs` (~8751 tok, huge) — Copyright 2025 STARGA Inc.
- `return_flow_tree_eval_run.rs` (~1517 tok, huge) — Copyright 2025 STARGA Inc.
- `rfc0012_attribute_syntax.rs` (~1182 tok, large) — Copyright 2025 STARGA Inc.
- `rfc0012_phase_a_shape_types.rs` (~6711 tok, huge) — Copyright 2025 STARGA Inc.
- `rfc0012_phase_b_operators.rs` (~4732 tok, huge) — Copyright 2025 STARGA Inc.
- `rfc0012_phase_c_annotations.rs` (~2405 tok, huge) — Copyright 2025 STARGA Inc.
- `rsa_pss_driver.py` (~2692 tok, huge) — # Ground-truth driver for std/rsa_pss.mind (pure-MIND RSASSA-PSS-VERIFY,
### `tests/runtime/`

- `elementwise_add.mind` (~68 tok, small) — Runtime test: Element-wise addition execution
- `reduction_sum.mind` (~67 tok, small) — Runtime test: Reduction sum operation
### `tests/`

- `scalar_cast_call_run.rs` (~719 tok, large) — Copyright 2025 STARGA Inc.
- `scalar_cast_unsigned_narrow_run.rs` (~1092 tok, large) — Copyright 2025 STARGA Inc.
### `tests/selfhost_gaps/`

- `andor_and_direct_1.mind` (~12 tok, tiny) — fn f(a: bool, b: bool) -> bool {
- `andor_and_ifcond_1.mind` (~18 tok, tiny) — fn f(a: i64, b: i64) -> i64 {
- `andor_and_letinit_1.mind` (~23 tok, tiny) — fn f(a: i64, b: i64) -> i64 {
- `andor_and_ret_1.mind` (~13 tok, tiny) — fn f(a: i64, b: i64) -> bool {
- `andor_or_ifcond_1.mind` (~18 tok, tiny) — fn f(a: i64, b: i64) -> i64 {
- `andor_or_ret_1.mind` (~13 tok, tiny) — fn f(a: i64, b: i64) -> bool {
- `array-nested_1.mind` (~19 tok, tiny) — fn f(a: i64, b: i64) -> i64 {
- `array-nested_2.mind` (~19 tok, tiny) — fn f(a: i64, b: i64) -> i64 {
- `array-nested_3.mind` (~23 tok, tiny) — fn f(a: i64, b: i64, c: i64) -> i64 {
- `array-nonconst_1.mind` (~16 tok, tiny) — fn f(a: i64, b: i64) -> i64 {
- `array-nonconst_2.mind` (~18 tok, tiny) — fn f(a: i64, b: i64) -> i64 {
- `array-nonconst_3.mind` (~18 tok, tiny) — fn g(a: i64, b: i64, c: i64) -> i64 {
- `array-nonconst_4.mind` (~17 tok, tiny) — fn h(a: i64, b: i64) -> i64 {
- `array_of_struct.mind` (~29 tok, tiny) — fn f(a: i64, b: i64) -> i64 {
- `builtin_option_none_1.mind` (~10 tok, tiny)
- `builtin_option_some_1.mind` (~11 tok, tiny)
- `builtin_option_some_return_1.mind` (~10 tok, tiny)
- `builtin_result_err_1.mind` (~12 tok, tiny)
- `builtin_result_ok_1.mind` (~10 tok, tiny)
- `call-arg-nesting_1.mind` (~25 tok, tiny)
- `call-arg-nesting_2.mind` (~35 tok, tiny)
- `call-arg-nesting_3.mind` (~41 tok, tiny)
- `call-arg-nesting_4.mind` (~48 tok, tiny)
- `call-arg-nesting_5.mind` (~20 tok, tiny)
- `call-arg-nesting_6.mind` (~28 tok, tiny)
- `call-arg-nesting_7.mind` (~28 tok, tiny)
- `call-arg-nesting_8.mind` (~50 tok, small)
- `call-arg-nesting_9.mind` (~10 tok, tiny)
- `callarg_then_field_recv.mind` (~40 tok, tiny)
- `chained_field_pqz.mind` (~37 tok, tiny)
- `const_false_lone_1.mind` (~8 tok, tiny) — fn f() -> bool {
- `const_ref_lone_1.mind` (~11 tok, tiny) — fn f() -> i64 {
- `const_true_lone_1.mind` (~7 tok, tiny) — fn f() -> bool {
- `deep-combos_1.mind` (~42 tok, tiny)
- `deep-combos_2.mind` (~42 tok, tiny)
- `deep-combos_3.mind` (~35 tok, tiny)
- `deep-combos_4.mind` (~35 tok, tiny)
- `deep-combos_5.mind` (~69 tok, small)
- `deep-combos_6.mind` (~128 tok, small)
- `deep-combos_7.mind` (~87 tok, small)
- `deep-combos_8.mind` (~14 tok, tiny)
- `discarded-stmt_1.mind` (~12 tok, tiny) — fn main() -> i64 {
- `discarded-stmt_2.mind` (~16 tok, tiny) — fn f(a: i64, b: i64) -> i64 {
- `discarded-stmt_3.mind` (~22 tok, tiny) — fn main() -> i64 {
- `discarded-stmt_4.mind` (~12 tok, tiny) — fn main() -> i64 {
- `discarded-stmt_5.mind` (~17 tok, tiny) — fn main() -> i64 {
- `enum-explicit-disc_1.mind` (~11 tok, tiny) — fn f() -> i64 { E::A }
- `enum-explicit-disc_2.mind` (~11 tok, tiny) — fn f() -> i64 { E::B }
- `enum-explicit-disc_3.mind` (~12 tok, tiny) — fn f() -> i64 { E::C }
- `enum-explicit-disc_4.mind` (~10 tok, tiny) — fn f() -> i64 { E::B }
- `enum_path_lone_1.mind` (~15 tok, tiny) — fn f() -> i64 {
- `fallthrough-shadow_1.mind` (~28 tok, tiny)
- `fallthrough-shadow_2.mind` (~42 tok, tiny)
- `fallthrough-shadow_3.mind` (~55 tok, small)
- `fallthrough-shadow_4.mind` (~29 tok, tiny)
- `fallthrough-shadow_5.mind` (~57 tok, small)
- `fallthrough-shadow_6.mind` (~38 tok, tiny)
- `fallthrough-shadow_7.mind` (~33 tok, tiny)
- `fallthrough-shadow_8.mind` (~25 tok, tiny)
- `field-read_1.mind` (~63 tok, small)
- `field-read_2.mind` (~41 tok, tiny)
- `field-read_3.mind` (~27 tok, tiny)
- `field-read_4.mind` (~29 tok, tiny)
- `field-read_5.mind` (~61 tok, small)
- `for_range_bound10_1.mind` (~24 tok, tiny) — fn f(a: i64) -> i64 {
- `for_range_continue_1.mind` (~35 tok, tiny) — fn f() -> i64 {
- `for_range_multistmt_1.mind` (~29 tok, tiny) — fn f(a: i64) -> i64 {
- `for_range_simple_1.mind` (~24 tok, tiny) — fn f(a: i64) -> i64 {
- `for_range_zero_1.mind` (~24 tok, tiny) — fn f(a: i64) -> i64 {
- `GAPS.md` (~4982 tok, huge) — Self-host nfn driver — gap inventory (fuzz-discovered)
- `index-in-if_1.mind` (~19 tok, tiny)
- `int_underscore_group_1.mind` (~7 tok, tiny) — fn f() -> i64 {
- `int_underscore_million_1.mind` (~8 tok, tiny) — fn f() -> i64 {
- `int_underscore_pair_1.mind` (~7 tok, tiny) — fn f() -> i64 {
- `let-ifexpr-seq_1.mind` (~23 tok, tiny)
- `let-ifexpr-seq_2.mind` (~34 tok, tiny)
- `let-ifexpr-seq_3.mind` (~23 tok, tiny)
- `let-ifexpr-seq_4.mind` (~21 tok, tiny)
- `let-ifexpr-seq_5.mind` (~19 tok, tiny)
- `let-ifexpr-seq_6.mind` (~27 tok, tiny)
- `let-ifexpr-seq_7.mind` (~46 tok, tiny)
- `matchcall_letinit_1.mind` (~34 tok, tiny) — fn g(a: i64) -> i64 {
- `matchcall-scrutinee_0arg.mind` (~24 tok, tiny) — fn g() -> i64 {
- `matchcall-scrutinee_1arg.mind` (~33 tok, tiny) — fn g(a: i64) -> i64 {
- `matchcall-scrutinee_arith.mind` (~21 tok, tiny) — fn f(a: i64) -> i64 {
- `mixed-prefix_10.mind` (~32 tok, tiny)
- `mixed-prefix_11.mind` (~27 tok, tiny)
- `mixed-prefix_12.mind` (~40 tok, tiny)
- `mixed-prefix_1.mind` (~28 tok, tiny)
- `mixed-prefix_2.mind` (~38 tok, tiny)
- `mixed-prefix_3.mind` (~65 tok, small)
- `mixed-prefix_4.mind` (~59 tok, small)
- `mixed-prefix_5.mind` (~41 tok, tiny)
- `mixed-prefix_6.mind` (~38 tok, tiny)
- `mixed-prefix_7.mind` (~29 tok, tiny)
- `mixed-prefix_8.mind` (~49 tok, tiny)
- `mixed-prefix_9.mind` (~50 tok, small)
- `neg_call_branch_both_return_1.mind` (~25 tok, tiny) — fn g(a: i64) -> i64 { a + 1 }
- `neg_call_branch_cond_1.mind` (~21 tok, tiny) — fn g(a: i64) -> i64 { a + 1 }
- `neg_call_branch_else_1.mind` (~21 tok, tiny) — fn g(a: i64) -> i64 { a + 1 }
- `neg_call_branch_then_1.mind` (~21 tok, tiny) — fn g(a: i64) -> i64 { a + 1 }
- `neg_double_1.mind` (~8 tok, tiny) — fn f(a: i64) -> i64 {
- `neg_lit_add_1.mind` (~9 tok, tiny) — fn f(a: i64) -> i64 {
- `neg_lit_cmp_1.mind` (~14 tok, tiny) — fn f(a: i64) -> i64 {
- `neg_lit_div_1.mind` (~9 tok, tiny) — fn f(a: i64) -> i64 {
- `neg_lit_lone_1.mind` (~7 tok, tiny) — fn n() -> i64 {
- `neg_lit_paren_1.mind` (~7 tok, tiny) — fn f() -> i64 {
- `neg_nonleaf_add_1.mind` (~10 tok, tiny) — fn f(a: i64) -> i64 {
- `neg_nonleaf_call_1.mind` (~17 tok, tiny) — fn g(x: i64) -> i64 {
- `neg_nonleaf_mul_1.mind` (~10 tok, tiny) — fn f(a: i64) -> i64 {
- `neg_nonleaf_paren_sub_1.mind` (~10 tok, tiny) — fn f(a: i64) -> i64 {
- `nested_array_typed.mind` (~23 tok, tiny) — fn f(a: i64, b: i64) -> i64 {
- `nested_slit_field.mind` (~30 tok, tiny)
### `tests/selfhost_gaps/never_wrong/`

- `array_lit_in_if_const_1.mind` (~19 tok, tiny)
- `array_lit_in_if_var_1.mind` (~20 tok, tiny)
- `index_call_binop_1.mind` (~28 tok, tiny) — fn g(a: i64) -> i64 { a + 1 }
- `index_call_in_if_1.mind` (~27 tok, tiny) — fn g(a: i64) -> i64 { a + 1 }
- `index_call_nested_1.mind` (~35 tok, tiny) — fn h(a: i64) -> i64 { a + 1 }
- `neg_float_1.mind` (~9 tok, tiny)
- `neg_float_in_if_1.mind` (~15 tok, tiny)
- `neg_index_in_if_1.mind` (~20 tok, tiny)
- `neg_index_trailing_1.mind` (~11 tok, tiny)
### `tests/selfhost_gaps/`

- `not_call_branch_then_1.mind` (~21 tok, tiny) — fn g(a: i64) -> i64 { a + 1 }
- `operator-edges_1.mind` (~24 tok, tiny)
- `operator-edges_2.mind` (~26 tok, tiny)
- `operator-edges_3.mind` (~26 tok, tiny)
- `operator-edges_4.mind` (~8 tok, tiny)
- `operator-edges_5.mind` (~16 tok, tiny)
- `operator-edges_6.mind` (~14 tok, tiny)
- `prec-add-bitand.mind` (~16 tok, tiny) — fn f(a: i64, b: i64, c: i64) -> i64 {
- `prec-add-bitor.mind` (~16 tok, tiny) — fn f(a: i64, b: i64, c: i64) -> i64 {
- `prec-add-shift.mind` (~16 tok, tiny) — fn f(a: i64, b: i64, c: i64) -> i64 {
- `prec-bitand-eq.mind` (~16 tok, tiny) — fn f(a: i64, b: i64, c: i64) -> i64 {
- `prec-mul-bitand.mind` (~16 tok, tiny) — fn f(a: i64, b: i64, c: i64) -> i64 {
- `prec-samebit-or-and.mind` (~16 tok, tiny) — fn f(a: i64, b: i64, c: i64) -> i64 {
- `prec-shift-mul.mind` (~16 tok, tiny) — fn f(a: i64, b: i64, c: i64) -> i64 {
- `prec-xor-cmp.mind` (~16 tok, tiny) — fn f(a: i64, b: i64, c: i64) -> i64 {
- `print_int_discarded.mind` (~11 tok, tiny) — fn main() -> i64 {
- `print_lone.mind` (~9 tok, tiny) — fn main() -> i64 {
- `print_str_discarded.mind` (~11 tok, tiny) — fn main() -> i64 {
- `prior_let_then_field_recv.mind` (~31 tok, tiny)
- `qfield_nested.mind` (~34 tok, tiny)
- `struct-lit_1.mind` (~63 tok, small)
- `struct-lit_2.mind` (~56 tok, small)
- `struct-lit_3.mind` (~60 tok, small)
- `struct-lit-field-recv_1.mind` (~23 tok, tiny)
- `struct-lit-field-recv_2.mind` (~23 tok, tiny)
- `struct-lit-field-recv_3.mind` (~25 tok, tiny)
- `two_scalar_prior.mind` (~38 tok, tiny)
- `value-ifexpr_1.mind` (~98 tok, small) — MISMATCH: a `let`-block in a NESTED (else-if) branch of a value if-expr.
- `value-ifexpr_2.mind` (~79 tok, small) — MISMATCH: same-named `let` in two SIBLING branches of a value if-expr.
- `value-ifexpr_3.mind` (~71 tok, small) — MISMATCH: a `let` inside a NESTED if-expr that sits in the THEN-side of an
- `value-ifexpr_4.mind` (~75 tok, small) — MISMATCH: let-block then-side whose trailing value is a nested if-expr that
- `value-ifexpr_5.mind` (~83 tok, small) — FAIL_CLOSED (in-subset): value if-expr whose else-branch is
- `value-ifexpr_6.mind` (~77 tok, small) — FAIL_CLOSED (in-subset): `let outer; if .. { use outer } else { use outer }`
- `value-ifexpr_7.mind` (~78 tok, small) — FAIL_CLOSED (in-subset): struct-lit construction as a value if-expr branch.
- `value-ifexpr_8.mind` (~64 tok, small) — FAIL_CLOSED (in-subset): field-read `recv.field` as a value if-expr branch.
### `tests/`

- `set_surface_run.rs` (~931 tok, large) — Copyright 2025 STARGA Inc.
- `sha256_smoke.rs` (~1555 tok, huge) — Copyright 2025 STARGA Inc.
- `sha512_smoke.rs` (~1689 tok, huge) — Copyright 2025 STARGA Inc.
- `shape_integration.rs` (~416 tok, medium)
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
- `smoke.rs` (~261 tok, medium) — Copyright 2025 STARGA Inc.
- `sparse_tensor_types.rs` (~1997 tok, huge) — Copyright 2025 STARGA Inc.
- `statement_mutation_run.rs` (~957 tok, large) — Copyright 2025 STARGA Inc.
- `std_import_standalone_run.rs` (~1151 tok, large) — Copyright 2025 STARGA Inc.
- `stdlib_tensor.rs` (~256 tok, medium) — Copyright 2025 STARGA Inc.
- `std_llvm_bindings_smoke.rs` (~2606 tok, huge) — Copyright 2025 STARGA Inc.
- `std_mlir_bindings_smoke.rs` (~4774 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_arena.rs` (~1338 tok, large) — Copyright 2025 STARGA Inc.
- `std_surface_array_literals.rs` (~1334 tok, large) — Copyright 2025 STARGA Inc.
- `std_surface_async.rs` (~4230 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_bitwise_binops.rs` (~2391 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_bool_return.rs` (~1271 tok, large) — Copyright 2026 STARGA Inc.
- `std_surface_break_continue.rs` (~1310 tok, large) — Copyright 2026 STARGA Inc.
- `std_surface_call_lowering.rs` (~834 tok, large) — Copyright 2025 STARGA Inc.
- `std_surface_cdylib_link.rs` (~2900 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_cli_equals_form.rs` (~874 tok, large) — Copyright 2025 STARGA Inc.
- `std_surface_cli.rs` (~1007 tok, large) — Copyright 2025 STARGA Inc.
- `std_surface_cli_subcommand.rs` (~799 tok, large) — Copyright 2025 STARGA Inc.
- `std_surface_field_access.rs` (~2943 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_field_access_step2.rs` (~3489 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_fndef_lowering.rs` (~1484 tok, large) — Copyright 2025 STARGA Inc.
- `std_surface_http.rs` (~3981 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_i32_intrinsics.rs` (~808 tok, large) — Copyright 2025 STARGA Inc.
- `std_surface_if_statement.rs` (~3414 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_intrinsics.rs` (~2438 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_io_ansi.rs` (~793 tok, large) — Copyright 2025 STARGA Inc.
- `std_surface_io_canon.rs` (~4854 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_io_module.rs` (~1540 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_iouring.rs` (~3542 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_json.rs` (~7041 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_logical_ops.rs` (~1183 tok, large) — Copyright 2026 STARGA Inc.
- `std_surface_map_module.rs` (~2015 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_method_call.rs` (~2939 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_net_fs_process.rs` (~8364 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_phase_c_stdlib_bundle.rs` (~1466 tok, large) — Copyright 2025 STARGA Inc.
- `std_surface_phase_d_env_override.rs` (~1597 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_promotion_compose.rs` (~5551 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_reactor.rs` (~1353 tok, large) — Copyright 2025 STARGA Inc.
- `std_surface_regex.rs` (~5282 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_ring.rs` (~1306 tok, large) — Copyright 2025 STARGA Inc.
- `std_surface_self_emit_shared.rs` (~1807 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_string_itoa.rs` (~934 tok, large) — Copyright 2025 STARGA Inc.
- `std_surface_string_module.rs` (~2284 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_string_push_str.rs` (~864 tok, large) — Copyright 2025 STARGA Inc.
- `std_surface_struct_lowering.rs` (~2712 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_toml.rs` (~4144 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_tui.rs` (~2261 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_use_import_phase_b.rs` (~2303 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_use_import.rs` (~1757 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_vec_module.rs` (~1805 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_vec_zeroed.rs` (~1073 tok, large) — Copyright 2025 STARGA Inc.
- `std_surface_while_statement.rs` (~3224 tok, huge) — Copyright 2025 STARGA Inc.
- `stmt_keyword_recognizer.rs` (~2473 tok, huge) — Copyright 2025 STARGA Inc.
- `stride_gather_grad.rs` (~312 tok, medium) — Copyright 2025 STARGA Inc.
- `stride_preview.rs` (~279 tok, medium) — Copyright 2025 STARGA Inc.
- `stride_types.rs` (~244 tok, medium) — Copyright 2025 STARGA Inc.
- `string_escape_decode_run.rs` (~1209 tok, large) — Copyright 2025 STARGA Inc.
- `string_escape_parse.rs` (~483 tok, medium) — Copyright 2025 STARGA Inc.
- `string_from_bytes_run.rs` (~709 tok, large) — Copyright 2025 STARGA Inc.
- `string_pattern_escape_decode.rs` (~1069 tok, large) — Copyright 2025 STARGA Inc. Licensed under the Apache License, Version 2.0.
- `string_runtime_shim_run.rs` (~1027 tok, large) — Copyright 2025 STARGA Inc.
- `string_split_run.rs` (~861 tok, large) — Copyright 2025 STARGA Inc.
- `struct_array_field_run.rs` (~847 tok, large) — Copyright 2025 STARGA Inc.
- `struct_field_collection_run.rs` (~854 tok, large) — Copyright 2025 STARGA Inc.
- `struct_field_in_loop_run.rs` (~1089 tok, large) — Copyright 2025 STARGA Inc.
- `struct_narrow_field.rs` (~821 tok, large) — Copyright 2025 STARGA Inc.
- `substrate_nonentry_import_link.rs` (~1491 tok, large) — Copyright 2025 STARGA Inc.
- `target_cerebras.rs` (~340 tok, medium) — Cerebras backend target — first-class surface tests.
- `tensor_broadcast.rs` (~995 tok, large) — Copyright 2025 STARGA Inc.
- `tensor_buffers.rs` (~517 tok, large) — Copyright 2025 STARGA Inc.
- `tensor_eval.rs` (~457 tok, medium) — Copyright 2025 STARGA Inc.
- `tensor_param_2d_run.rs` (~998 tok, large) — Copyright 2025 STARGA Inc.
- `tensor_param_fail_loud_run.rs` (~2524 tok, huge) — Copyright 2025 STARGA Inc.
- `tensor_stdlib.rs` (~549 tok, large) — Copyright 2025 STARGA Inc.
- `tensor_symbolic.rs` (~542 tok, large) — Copyright 2025 STARGA Inc.
- `tls13_finished_driver.py` (~3085 tok, huge) — # Official-vector driver for std/tls13_finished.mind (pure-MIND TLS 1.3
- `tls13_handshake_driver.py` (~6662 tok, huge) — # Official-vector driver for std/tls13_handshake.mind (pure-MIND TLS 1.3
- `tls13_keyschedule_driver.py` (~2863 tok, huge) — # Official-vector driver for std/tls13_keyschedule.mind (pure-MIND TLS 1.3 key
- `tls13_record_driver.py` (~3051 tok, huge) — # Official-vector driver for std/tls13_record.mind (pure-MIND TLS 1.3 record
- `trait_static_dispatch_run.rs` (~1105 tok, large) — Copyright 2025 STARGA Inc.
- `transpose_preview.rs` (~269 tok, medium) — Copyright 2025 STARGA Inc.
- `try_operator_run.rs` (~1108 tok, large) — Copyright 2025 STARGA Inc.
- `tuple_destructure_run.rs` (~1162 tok, large) — Copyright 2025 STARGA Inc.
- `turboquant_kernel_run.rs` (~1294 tok, large) — Copyright 2025 STARGA Inc.
- `type_ann_check.rs` (~330 tok, medium) — Copyright 2025 STARGA Inc.
- `type_ann_parse.rs` (~583 tok, large) — Copyright 2025 STARGA Inc.
- `typecheck_binary.rs` (~310 tok, medium) — Copyright 2025 STARGA Inc.
- `typecheck_env.rs` (~246 tok, medium) — Copyright 2025 STARGA Inc.
### `tests/type_checker/`

- `basic_type_inference.mind` (~66 tok, small) — Type checker test: Basic type inference
- `dtype_mismatch.mind` (~74 tok, small) — Type checker test: Dtype mismatch detection
### `tests/`

- `typed_literal_match_pattern_run.rs` (~764 tok, large) — Copyright 2025 STARGA Inc. Licensed under the Apache License, Version 2.0.
- `type_error_spans.rs` (~1058 tok, large) — Copyright 2025 STARGA Inc.
- `type_infer.rs` (~344 tok, medium) — Copyright 2025 STARGA Inc.
- `type_struct_run.rs` (~628 tok, large) — Copyright 2025 STARGA Inc.
- `typo_reject.rs` (~1127 tok, large) — Copyright 2025 STARGA Inc.
- `u64_cast_signed_compare_run.rs` (~877 tok, large) — Copyright 2025 STARGA Inc.
- `value_if_comparison.rs` (~813 tok, large) — Copyright 2025 STARGA Inc.
- `value_if_f64_let.rs` (~1148 tok, large) — Copyright 2025 STARGA Inc.
- `vars_assign.rs` (~260 tok, medium) — Copyright 2025 STARGA Inc.
- `verify_audit.rs` (~2058 tok, huge) — Audit coverage tests for the IR verifier (C1: SSA verification, conv2d stride/axis validation).
- `verify_canonical_bytes.rs` (~3803 tok, huge) — Copyright 2025 STARGA Inc.
- `verify_cli.rs` (~3821 tok, huge) — Copyright 2025 STARGA Inc.
- `verify_holes.rs` (~2696 tok, huge) — Copyright 2025 STARGA Inc.
- `verify_pinned_signer.rs` (~993 tok, large) — Copyright 2025 STARGA Inc.
- `verify_require_signed.rs` (~1966 tok, huge) — Copyright 2025 STARGA Inc.
- `verify_ssa.rs` (~7509 tok, huge) — Copyright 2025 STARGA Inc.
- `x25519mlkem768_driver.py` (~2378 tok, huge) — # Known-answer driver for std/x25519mlkem768.mind (pure-MIND X25519MLKEM768
- `x25519_vectors_driver.py` (~1525 tok, huge) — # Official-vector driver for std/x25519.mind (pure-MIND Curve25519 ECDH).
- `x509_vectors_driver.py` (~3593 tok, huge) — # Real-certificate driver for std/x509.mind (pure-MIND X.509 DER parsing + RSA
### `tools/`

- `add_copyright_headers.py` (~1132 tok, large) — # Copyright 2025 STARGA Inc.
- `bench_gate.py` (~3939 tok, huge) — # Copyright 2025 STARGA Inc.
- `cargo-deny-sanitize.sh` (~572 tok, large) — Run cargo-deny but sanitize advisory entries that older cargo-deny versions
### `tools/mindfuzz/`

- `ci_batch.py` (~1482 tok, large) — MIND-Fuzz deterministic batch GENERATOR -> cross-substrate reference corpus.
- `fuzz_loop.py` (~3529 tok, huge) — MIND-Fuzz loop -- LLM-mutation differential testing for the MIND compiler.
- `.gitignore` (~7 tok, tiny) — tools/mindfuzz/__pycache__/
- `mutate.py` (~3631 tok, huge) — MIND-Fuzz mutation engine.
- `mutations.txt` (~1765 tok, huge) — # MIND-Fuzz mutation instructions (adapted from arXiv:2501.00655 Table 1).
- `oracles.py` (~4619 tok, huge) — MIND-Fuzz differential oracles.
- `README.md` (~1730 tok, huge) — MIND-Fuzz
### `tools/mindfuzz/seeds/`

- `cast_edge.mind` (~241 tok, medium) — MIND-Fuzz seed: type-conversion + edge-value scalar entry (reference-checkable).
- `control_flow.mind` (~312 tok, medium) — MIND-Fuzz seed: control-flow-heavy program (loops, nested if, early return,
- `dot_q16.mind` (~221 tok, medium) — MIND-Fuzz seed: Q16.16 dot / L1 / gemv kernels.
- `multi_fn.mind` (~243 tok, medium) — MIND-Fuzz seed: multi-function program with a reference-checkable scalar entry.
- `scalar_accum.mind` (~154 tok, small) — MIND-Fuzz seed: scalar accumulator with a return-feeding literal.
- `scalar_arith.mind` (~159 tok, small) — MIND-Fuzz seed: small pure-integer scalar function.
### `tools/mindfuzz/violations/`

- `.gitkeep` (~0 tok, tiny)
### `tools/pytorch_bridge/`

- `ai_proof.py` (~640 tok, large) — # Copyright 2025-2026 STARGA Inc.
- `.gitignore` (~4 tok, tiny) — __pycache__/
- `__init__.py` (~384 tok, medium) — # Copyright 2025-2026 STARGA Inc.
- `ir.py` (~920 tok, large) — # Copyright 2025-2026 STARGA Inc.
- `jax.py` (~1007 tok, large) — # Copyright 2025-2026 STARGA Inc.
- `pytorch.py` (~1718 tok, huge) — # Copyright 2025-2026 STARGA Inc.
### `tools/pytorch_bridge/tests/`

- `__init__.py` (~0 tok, tiny)
- `test_bridge.py` (~1244 tok, large) — # Copyright 2025-2026 STARGA Inc.
### `tools/`

- `run_bench_gate.sh` (~530 tok, large) — Copyright 2025 STARGA Inc. Licensed under the Apache License, Version 2.0.
- `test_bench_gate.py` (~1333 tok, large) — Regression test for the bench-gate fail-closed contract (tools/bench_gate.py).

---
*Generated by `anatomy 1.0.0`. Edit descriptions manually — re-run preserves structure.*
