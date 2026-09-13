# ANATOMY.md — Project File Index

> **For coding agents.** Read this before opening files. Use descriptions and token
> estimates to decide whether you need the full file or the summary is enough.
> Re-generate with: `anatomy .`

**Project:** `mind`
**Files:** 1817 | **Est. tokens:** ~3482012
**Generated:** 2026-09-13 07:02 UTC

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
| `./` | 33 | ~30884 |
| `.agents/skills/mindc-development/` | 1 | ~235 |
| `.arch-mind/` | 2 | ~644 |
| `.cargo/` | 1 | ~130 |
| `.githooks/` | 4 | ~1755 |
| `.github/` | 4 | ~894 |
| `.github/ISSUE_TEMPLATE/` | 3 | ~440 |
| `.github/workflows/` | 9 | ~37428 |
| `agents/` | 1 | ~436 |
| `assets/logo/` | 1 | ~453 |
| `audits/` | 6 | ~607 |
| `bench/` | 5 | ~7574 |
| `bench/fft/` | 8 | ~8060 |
| `benches/` | 28 | ~83855 |
| `benches/common/` | 2 | ~2995 |
| `benchmarks/` | 13 | ~28066 |
| `benchmarks/autograd_comparison/` | 8 | ~9411 |
| `benchmarks/cupy_comparison/` | 6 | ~7733 |
| `benchmarks/determinism/` | 3 | ~4601 |
| `benchmarks/inference/` | 4 | ~4049 |
| `benchmarks/jax_comparison/` | 5 | ~4689 |
| `benchmarks/mojo/` | 8 | ~4276 |
| `benchmarks/pytorch_comparison/` | 5 | ~4880 |
| `config/` | 2 | ~3347 |
| `docs/` | 37 | ~104810 |
| `docs/backends/` | 1 | ~1482 |
| `docs/benchmarks/` | 3 | ~9042 |
| `docs/design/` | 4 | ~11498 |
| `docs/gates/` | 1 | ~3910 |
| `docs/mindcraft/` | 3 | ~7086 |
| `docs/rfcs/` | 35 | ~171213 |
| `docs/specs/` | 2 | ~976 |
| `examples/` | 28 | ~49880 |
| `examples/bimap_currency/` | 3 | ~780 |
| `examples/bimap_pairs/` | 2 | ~801 |
| `examples/c/` | 2 | ~400 |
| `examples/columnar/` | 4 | ~7585 |
| `examples/compliance/` | 3 | ~5294 |
| `examples/detmath_kat/` | 2 | ~4435 |
| `examples/distribution-crossisa/` | 6 | ~6336 |
| `examples/emit_ir/` | 5 | ~13648 |
| `examples/grammar_mask/` | 2 | ~4636 |
| `examples/halbach_q16/` | 2 | ~7856 |
| `examples/lexer/` | 6 | ~8888 |
| `examples/mind_mirror_v04/` | 2 | ~3836 |
| `examples/mind_mirror_v04/testdata/` | 2 | ~3862 |
| `examples/mindc_mind/` | 174 | ~450655 |
| `examples/mindc_mind/testdata/` | 3 | ~4159 |
| `examples/mindc_mind/testdata/backend_native_bridge/` | 9 | ~1078 |
| `examples/mindc_mind/testdata/native_elf_oracle/` | 6 | ~913 |
| `examples/mindc_mind/testdata/native_record_array/` | 42 | ~2926 |
| `examples/mindc_mind/testdata/native_record_trace/` | 13 | ~2127 |
| `examples/mindc_mind/testdata/selfhost_loop/` | 2 | ~2138 |
| `examples/native/` | 4 | ~1440 |
| `examples/parser/` | 5 | ~17923 |
| `examples/typecheck/` | 5 | ~14553 |
| `examples/zoo/` | 6 | ~12518 |
| `experiments/global-vs-local/` | 7 | ~6487 |
| `mind/std/cognitive/` | 4 | ~3529 |
| `runtime-support/` | 1 | ~21930 |
| `scripts/` | 31 | ~116920 |
| `scripts/mind-vs-rust/` | 3 | ~933 |
| `scripts/mind-vs-rust/src/` | 1 | ~2372 |
| `scripts/sdlc/` | 2 | ~2845 |
| `sdk/ts/mic-map/` | 6 | ~15498 |
| `sdk/ts/mic-map/dist/` | 36 | ~29044 |
| `sdk/ts/mic-map/scripts/` | 1 | ~499 |
| `sdk/ts/mic-map/src/` | 9 | ~12181 |
| `sdk/ts/mic-map/test/` | 4 | ~7843 |
| `sdk/ts/mic-map/test/fixtures/` | 2 | ~96 |
| `skills/write-mind/` | 1 | ~6002 |
| `src/` | 11 | ~51968 |
| `src/ast/` | 1 | ~10960 |
| `src/autodiff/` | 3 | ~6624 |
| `src/bin/` | 1 | ~9878 |
| `src/bin/mindc/` | 1 | ~204 |
| `src/build/` | 13 | ~48384 |
| `src/build/cache/` | 3 | ~3150 |
| `src/cache/` | 4 | ~3682 |
| `src/check/` | 3 | ~10613 |
| `src/deps/` | 1 | ~9388 |
| `src/diagnostics/` | 3 | ~14085 |
| `src/distributed/` | 6 | ~7725 |
| `src/doc/` | 3 | ~11002 |
| `src/eval/` | 36 | ~139475 |
| `src/eval/stdlib/` | 2 | ~8586 |
| `src/eval/struct_resolver/` | 1 | ~2555 |
| `src/exec/` | 3 | ~5522 |
| `src/ffi/` | 3 | ~5541 |
| `src/fmt/` | 3 | ~24052 |
| `src/ir/` | 9 | ~95624 |
| `src/ir/compact/` | 3 | ~15351 |
| `src/ir/compact/v2/` | 9 | ~45363 |
| `src/ir/compact/v3/` | 15 | ~76625 |
| `src/ir/compact/v3/v04/` | 3 | ~13696 |
| `src/lint/` | 2 | ~4001 |
| `src/lint/rules/` | 6 | ~10017 |
| `src/mlir/` | 3 | ~6625 |
| `src/ops/` | 3 | ~4764 |
| `src/opt/` | 9 | ~53642 |
| `src/package/` | 2 | ~1877 |
| `src/parser/` | 4 | ~25256 |
| `src/phf/` | 1 | ~4955 |
| `src/project/` | 18 | ~59056 |
| `src/runtime/` | 3 | ~1485 |
| `src/shapes/` | 2 | ~6052 |
| `src/stdlib/` | 2 | ~560 |
| `src/test/` | 4 | ~14244 |
| `src/type_checker/` | 18 | ~61407 |
| `src/type_checker/slice_abi/` | 3 | ~11245 |
| `src/types/` | 10 | ~20740 |
| `src/workspace/` | 1 | ~4906 |
| `std/` | 42 | ~202387 |
| `tests/` | 399 | ~818690 |
| `tests/autodiff/` | 2 | ~247 |
| `tests/backend/` | 2 | ~125 |
| `tests/common/` | 3 | ~12308 |
| `tests/conformance/cpu_baseline/` | 12 | ~390 |
| `tests/conformance/gpu_profile/` | 2 | ~11 |
| `tests/cross_substrate_identity/` | 2 | ~4371 |
| `tests/cross_substrate_identity/array-store-branch/` | 2 | ~974 |
| `tests/cross_substrate_identity/array-store-loop/` | 2 | ~1081 |
| `tests/cross_substrate_identity/bimap-phf/` | 2 | ~1675 |
| `tests/cross_substrate_identity/collatz/` | 2 | ~962 |
| `tests/cross_substrate_identity/dot-f32-v-4093/` | 2 | ~1222 |
| `tests/cross_substrate_identity/dot-i16-4096/` | 2 | ~648 |
| `tests/cross_substrate_identity/dot-l1-q16/` | 2 | ~363 |
| `tests/cross_substrate_identity/dot-l2-q16/` | 2 | ~813 |
| `tests/cross_substrate_identity/galperin-pi/` | 2 | ~1004 |
| `tests/cross_substrate_identity/gemm-i8-64x64x64/` | 2 | ~707 |
| `tests/cross_substrate_identity/gemm-i8-mt-64x64x64/` | 2 | ~872 |
| `tests/cross_substrate_identity/gemm-i8-vnni-64x64x64/` | 2 | ~943 |
| `tests/cross_substrate_identity/gemm-q16-64x64x64/` | 2 | ~616 |
| `tests/cross_substrate_identity/gemm-q16-fused-64x64x64/` | 2 | ~896 |
| `tests/cross_substrate_identity/gemv-i16-256x256/` | 2 | ~594 |
| `tests/cross_substrate_identity/gemv-q16-256x256/` | 2 | ~519 |
| `tests/cross_substrate_identity/grammar-mask/` | 2 | ~916 |
| `tests/cross_substrate_identity/lorenz-q16/` | 2 | ~1243 |
| `tests/cross_substrate_identity/matmul-f32-v-64x64/` | 2 | ~1115 |
| `tests/cross_substrate_identity/q16-arith-chain/` | 2 | ~788 |
| `tests/cross_substrate_identity/scalar-cast-conv/` | 2 | ~1644 |
| `tests/cross_substrate_identity/scalar-cast-conv-narrow/` | 2 | ~1790 |
| `tests/cross_substrate_identity/scalar-float-f64/` | 2 | ~1310 |
| `tests/cross_substrate_identity/struct-handle-roundtrip/` | 2 | ~746 |
| `tests/cross_substrate_identity/u64-ops/` | 2 | ~1054 |
| `tests/fixtures/` | 7 | ~5820 |
| `tests/fixtures/f64_abi_negative/` | 3 | ~192 |
| `tests/fixtures/nerve_numerics/` | 11 | ~968 |
| `tests/fixtures/selfhost_policy/` | 13 | ~315 |
| `tests/fixtures/selfhost_policy/reject/` | 58 | ~616 |
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
| `tests/mindfuzz_corpus_floor/` | 1 | ~855 |
| `tests/mindfuzz_cross_substrate/known_environmental/` | 2 | ~556 |
| `tests/mindfuzz_cross_substrate/reproducers/` | 2 | ~375 |
| `tests/mindfuzz_cross_substrate/staged/` | 15 | ~2832 |
| `tests/native_bridge_support/` | 2 | ~2574 |
| `tests/runtime/` | 2 | ~135 |
| `tests/selfhost_gaps/` | 163 | ~10627 |
| `tests/selfhost_gaps/never_wrong/` | 18 | ~338 |
| `tests/shapes/` | 3 | ~260 |
| `tests/skip_shape_scan/` | 2 | ~10401 |
| `tests/support/` | 9 | ~19407 |
| `tests/type_checker/` | 2 | ~140 |
| `tools/` | 5 | ~13565 |
| `tools/mindfuzz/` | 7 | ~16763 |
| `tools/mindfuzz/seeds/` | 6 | ~1330 |
| `tools/mindfuzz/violations/` | 1 | ~0 |
| `tools/pytorch_bridge/` | 6 | ~4673 |
| `tools/pytorch_bridge/tests/` | 2 | ~1244 |

## Files

### `./`

- `.bench-baseline-2026-04-27.txt` (~531 tok, large) —    Compiling mind v0.2.3 (.)
- `.bench-baseline-2026-04-28-pratt.txt` (~185 tok, small) — === Pratt parser baseline (mindc 0.2.5, 2026-04-28) ===
- `.bench-baseline-2026-05-17-phase10-6.txt` (~408 tok, medium) — === Phase 10.6 surface-syntax baseline (mindc 0.2.10, 2026-05-17) ===
- `.bench-baseline-2026-05-17-phase10-7.txt` (~565 tok, large) — === Phase 10.7 surface baseline (mindc 0.2.11, 2026-05-17) ===
- `.bench-baseline-2026-05-18-rfc0005.txt` (~781 tok, large) — === RFC 0005 Phase 2 baseline (mindc 0.4.0, 2026-05-18) ===
- `.bench-baseline-2026-06-01-correctness.txt` (~844 tok, large) — === Correctness-milestone baseline (mindc 0.7.0, 2026-06-01) ===
- `.bench-pre-pratt.txt` (~32 tok, tiny) — === captured pre-Pratt baseline (Phase 10.5 in main) ===
- `.editorconfig` (~51 tok, small) — root = true
- `.gitattributes` (~130 tok, small) — # Enforce LF line endings for all text so byte-exact tests (fmt idempotence,
- `.gitignore` (~950 tok, large) — # Rust
- `.sembleignore` (~72 tok, small) — # semble code-search ignore list
- `ARCHITECTURE.md` (~300 tok, medium) — MIND Architecture (high level)
- `AUDIT_REPORT.md` (~1151 tok, large) — Audit Report
- `CODE_OF_CONDUCT.md` (~29 tok, tiny) — Code of Conduct
- `COMPLETE_FILE_STRUCTURE.md` (~26 tok, tiny) — Repository Structure (Snapshot)
- `CONTRIBUTING.md` (~2260 tok, huge) — Contributing to MIND
- `Cargo.toml` (~3947 tok, huge) — [package]
- `GITHUB_SETUP_INSTRUCTIONS.md` (~240 tok, medium) — GitHub Setup (Quick)
- `LICENSE` (~2573 tok, huge) —                                  Apache License
- `LICENSE-COMMERCIAL` (~399 tok, medium) — COMMERCIAL LICENSE NOTICE – MIND (Enterprise & SaaS)
- `Mind.toml` (~108 tok, small) — [package]
- `README.md` (~7225 tok, huge) — MIND — Machine Intelligence Native Design
- `RELEASING.md` (~131 tok, small) — Release checklist (as of v0.2.1)
- `SECURITY.md` (~1903 tok, huge) — Security Policy
- `STATUS.md` (~4457 tok, huge) — MIND Compiler Status
- `bounties.md` (~888 tok, large) — MIND Bounty Board
- `build.rs` (~234 tok, medium) — Copyright 2025 STARGA Inc.
- `clippy.toml` (~25 tok, tiny)
- `deny.toml` (~89 tok, small) — [advisories]
- `incompatible` (~0 tok, tiny)
- `plugin.json` (~62 tok, small) — Keys: name, description, version, skills, agents
- `rustfmt.toml` (~23 tok, tiny) — max_width = 100
- `test_real_compile_time.py` (~265 tok, medium) — Quick test of real MIND compilation time using Python bindings."""
### `.agents/skills/mindc-development/`

- `SKILL.md` (~235 tok, medium) — MIND Compiler (mindc) Development
### `.arch-mind/`

- `rules.mind` (~557 tok, large) — mind (language compiler / runtime root) architectural-governance rules
- `scan.json` (~87 tok, small) — Keys: _fixture, acyclicity_q16, depth_q16, equality_q16, evidence_chain_density
### `.cargo/`

- `config.toml` (~130 tok, small) — [registries]
### `.githooks/`

- `commit-msg` (~239 tok, medium) — #!/usr/bin/env bash
- `post-commit` (~410 tok, medium) — #!/usr/bin/env bash
- `post-merge` (~408 tok, medium) — #!/usr/bin/env bash
- `pre-commit` (~698 tok, large) — #!/usr/bin/env bash
### `.github/`

- `CODEOWNERS` (~9 tok, tiny) — *       @star-ga
### `.github/ISSUE_TEMPLATE/`

- `bounty_claim.md` (~56 tok, small)
- `bug_report.md` (~213 tok, medium) — Describe the bug
- `feature_request.md` (~171 tok, small) — Problem Statement
### `.github/`

- `PULL_REQUEST_TEMPLATE.md` (~55 tok, small) — Summary
- `release-drafter.yml` (~85 tok, small) — name-template: 'v$NEXT_PATCH_VERSION'
- `required-ci-jobs.tsv` (~745 tok, large) — # required-ci-jobs.tsv — the CI jobs that MUST have passed on the exact commit
### `.github/workflows/`

- `bench-gate.yml` (~2417 tok, huge) — name: Bench gate
- `cargo-deny.yml` (~293 tok, medium) — name: Cargo Deny
- `ci.yml` (~25572 tok, huge) — name: CI
- `crypto-vectors.yml` (~1590 tok, huge) — name: Crypto Vectors
- `docs-claims.yml` (~2645 tok, huge) — name: Docs Claims
- `link-check.yml` (~229 tok, medium) — name: Link Check
- `mindcraft.yml` (~910 tok, large) — name: Mindcraft Check
- `release-drafter.yml` (~91 tok, small) — name: Release Drafter
- `release.yml` (~3681 tok, huge) — name: Release
### `agents/`

- `mind-developer.md` (~436 tok, medium) — MIND Developer Agent
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

- `RESULTS-beat-clang-igemm-2026-08-21.md` (~839 tok, large) — MIND vs clang -O3 — int8 GEMM head-to-head (2026-08-21)
- `RESULTS-int8-2026-06-08.md` (~693 tok, large) — MIND int8 VNNI GEMM — single-core vs OpenBLAS f32 (2026-06-08)
- `beat_clang_igemm_driver.c` (~494 tok, medium)
### `bench/fft/`

- `.gitignore` (~38 tok, tiny) — # Build artifacts — regenerated by build.sh, never committed.
- `README.md` (~1677 tok, huge) — Deterministic Q16.16 N=256 FFT — MIND vs gcc / clang / nvcc
- `RESULTS-fft-2026-06-15.md` (~1307 tok, large) — RESULTS — Deterministic Q16.16 N=256 FFT (MIND vs gcc / clang / nvcc)
- `build.sh` (~986 tok, large) — build.sh — self-contained build for the deterministic Q16.16 N=256 FFT bench.
- `fft_driver.c` (~1205 tok, large) — Standalone correctness + timing driver for the C reference Q16.16 FFT.
- `fft_ref.c` (~473 tok, medium) — Q16.16 deterministic radix-2 DIT FFT, N=256 — BYTE-IDENTICAL algorithm to
- `fft_verify.c` (~889 tok, large) — Cross-check harness: load the MIND-compiled fft256 from a .so and assert its
- `harness.c` (~1485 tok, large) — Self-contained benchmark harness for the deterministic Q16.16 N=256 FFT.
### `bench/`

- `matmul_det_bench.mind` (~1079 tok, large) — bench/matmul_det_bench.mind — first pure-MIND runtime benchmark for the
- `turboquant.mind` (~4469 tok, huge) — bench/turboquant.mind — KV-cache quantization pipeline in pure MIND.
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
### `benches/common/`

- `mod.rs` (~668 tok, large) — Shared self-skip probe for criterion bench targets.
- `roofline_peak.rs` (~2327 tok, huge) — Shared roofline denominator for the deterministic GEMM / int-dot benches.
### `benches/`

- `compiler.rs` (~3786 tok, huge) — Small program: Simple matrix multiplication
- `cross_module.rs` (~605 tok, large) — Copyright 2025 STARGA Inc.
- `det_gemv_q16_mt.rs` (~3229 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `det_matmul_i16.rs` (~4549 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `det_matmul_i8.rs` (~4988 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `det_matmul_q16.rs` (~4829 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `det_matmul_q16_mt.rs` (~4049 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `fft_q16.rs` (~5352 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `mindcraft_fmt.rs` (~1079 tok, large) — File readers
- `operations.rs` (~1200 tok, large) — Element-wise operations
- `parser_throughput.rs` (~916 tok, large) — Copyright 2025 STARGA Inc.
- `shapes.rs` (~1368 tok, large) — Simple broadcasting scenarios
- `simple_benchmarks.rs` (~794 tok, large) — Mirror mindc's allocator so this compile-speed bench measures the same heap
- `std_surface.rs` (~1090 tok, large) — Copyright 2025 STARGA Inc.
### `benchmarks/`

- `BENCHMARK_RESULTS.md` (~4559 tok, huge) — MIND Benchmark Results
- `MIC_MAP_BENCHMARK_README.md` (~675 tok, large) — MIC/MAP Patent Reference Benchmark
- `README.md` (~1168 tok, large) — MIND Performance Benchmarks
- `RUN_GUIDE.md` (~1563 tok, huge) — MIND Patent Benchmarks - Environment Guide
### `benchmarks/autograd_comparison/`

- `README.md` (~1153 tok, large) — Autograd Comparison: MIND vs PyTorch
- `README_REAL.md` (~1185 tok, large) — Real Autograd Comparison: MIND vs PyTorch
- `autograd_results.json` (~424 tok, medium) — Keys: system_info, benchmarks
- `benchmark_autograd.py` (~2444 tok, huge)
- `benchmark_python_bindings.py` (~1566 tok, huge)
- `benchmark_real_autograd.py` (~2304 tok, huge)
- `real_autograd_results.json` (~328 tok, medium) — Keys: system_info, methodology, benchmarks
- `requirements.txt` (~7 tok, tiny) — torch>=1.0.0
### `benchmarks/`

- `criterion_ci_sweep.txt` (~4095 tok, huge) — CRITERION CI-EQUIVALENT SWEEP — recorded output
### `benchmarks/cupy_comparison/`

- `README.md` (~1619 tok, huge) — CuPy Comparison Benchmark
- `leg1_determinism.py` (~2586 tok, huge)
- `leg1_determinism_results.json` (~1451 tok, large) — Keys: leg, host, mind, cupy
- `leg2_perf.py` (~1510 tok, huge)
- `leg2_perf_results.json` (~473 tok, medium) — Keys: leg, host, config, mind, status
- `requirements.txt` (~94 tok, small) — # Leg 1 (determinism) + Leg 2 (perf) foil dependencies.
### `benchmarks/determinism/`

- `README.md` (~1311 tok, large) — MIND Determinism Proof Benchmark
- `benchmark_determinism.py` (~2187 tok, huge)
- `determinism_results.json` (~1103 tok, large) — Keys: system_info, num_runs, tests, all_deterministic
### `benchmarks/`

- `format_benchmark.py` (~2434 tok, huge)
### `benchmarks/inference/`

- `README.md` (~1149 tok, large) — Inference Speed Benchmark
- `benchmark_inference.py` (~2423 tok, huge)
- `inference_results.json` (~473 tok, medium) — Keys: system_info, benchmarks
- `requirements.txt` (~4 tok, tiny) — torch>=1.0.0
### `benchmarks/jax_comparison/`

- `README.md` (~1109 tok, large) — JAX Compilation Benchmark
- `benchmark_jax_compile.py` (~2719 tok, huge)
- `jax_coldstart_results.json` (~376 tok, medium) — Keys: environment, results
- `jax_results.json` (~478 tok, medium) — Keys: system_info, benchmarks
- `requirements.txt` (~7 tok, tiny) — jax>=0.4.0
### `benchmarks/`

- `mic_benchmark.py` (~1473 tok, large)
- `mic_map_benchmark_results.json` (~1506 tok, huge) — Keys: metadata, measurements, paper_figures_verified, claim_checks, cost_model
- `mic_map_benchmark_v2.py` (~5563 tok, huge)
### `benchmarks/mojo/`

- `README.md` (~1271 tok, large) — Mojo Compilation Benchmarks
- `benchmark_mojo_compilation.py` (~1533 tok, huge)
- `large_matmul.mojo` (~205 tok, medium) — """
- `medium_matmul.mojo` (~205 tok, medium) — """
- `mojo_results.json` (~216 tok, medium) — Keys: scalar_math, small_matmul, medium_matmul, large_matmul
- `run_benchmarks.sh` (~581 tok, large) — Mojo Compilation Benchmark Runner
- `scalar_math.mojo` (~58 tok, small) — """
- `small_matmul.mojo` (~207 tok, medium) — """
### `benchmarks/pytorch_comparison/`

- `=2.0` (~0 tok, tiny)
- `README.md` (~866 tok, large) — PyTorch Compilation Benchmark
- `benchmark_pytorch_compile.py` (~3420 tok, huge)
- `pytorch_results.json` (~590 tok, large) — Keys: system_info, benchmarks
- `requirements.txt` (~4 tok, tiny) — torch>=2.0.0
### `benchmarks/`

- `resnet.md` (~74 tok, small) — ResNet Benchmarks (Preliminary)
- `run_all_benchmarks.sh` (~824 tok, large) — Master script to run all MIND patent benchmarks
- `scientific_benchmark.py` (~1647 tok, huge)
- `scientific_benchmark_raw.py` (~2485 tok, huge)
### `config/`

- `capabilities.toml` (~2375 tok, huge) — [ir]
- `token_pricing.toml` (~972 tok, large) — [pricing]
### `docs/`

- `ARRAY_SEMANTICS.md` (~8111 tok, huge) — MIND array semantics — normative architecture record
- `ENGINE_CONSUMER_LIST.md` (~9442 tok, huge) — Engine consumer list — the evidence pack for every hand-written semantics engine
- `INDEPENDENCE_ROADMAP.md` (~18828 tok, huge) — MIND Rust-Independence Roadmap
- `MHS_ROADMAP.md` (~2657 tok, huge) — MHS / Governed Device I/O Roadmap — SUPERSEDED
- `README.md` (~162 tok, small) — MIND Documentation
- `RI_DEPENDENCY_MATRIX.md` (~6279 tok, huge) — Rust-Independence (RI) Dependency Matrix
- `VERIFICATION_APPARATUS.md` (~7783 tok, huge) — Self-Host Port Verification Apparatus & SOTA Roadmap
- `architecture.md` (~965 tok, large) — Architecture
- `autodiff.md` (~595 tok, large) — Static autodiff (public)
### `docs/backends/`

- `cerebras-stencil.md` (~1482 tok, large) — `mind.cerebras.stencil_tile` — Op Surface and Lowering Contract
### `docs/`

- `benchmarking.md` (~1917 tok, huge) — Benchmarking methodology — tiers and comparable metrics
- `benchmarks.md` (~896 tok, large) — Benchmarks
### `docs/benchmarks/`

- `RESULTS-mind-vs-rust-2026-06-09.md` (~2174 tok, huge) — MIND vs Rust — integer-GEMM, apples-to-apples (2026-06-09)
- `compiler_performance.md` (~4363 tok, huge) — MIND Compiler Performance Benchmarks
- `mojo_comparison.md` (~2505 tok, huge) — MIND vs Mojo: Compilation Performance Comparison
### `docs/`

- `byte-store-migration.md` (~3357 tok, huge) — Byte-Store Migration — closing `#306`
- `cli.md` (~747 tok, large) — MIND CLI Reference
### `docs/design/`

- `README.md` (~26 tok, tiny) — Design Docs
- `execution-plan-performance-mode.md` (~8045 tok, huge) — Design: PerformanceMode + ExecutionPlan + ExecutionProvider
- `rfc0012-b2-shape-threading-scope.md` (~3317 tok, huge) — RFC 0012 Phase B.2 — shape-dim threading: execution scope
- `v0.3.md` (~110 tok, small) — MIND Design v0.3 (Draft)
### `docs/`

- `determinism.md` (~4368 tok, huge) — The Determinism Contract
- `errors.md` (~1338 tok, large) — MIND Core Error Model
- `ffi-runtime.md` (~529 tok, large) — FFI & Runtime Integration
### `docs/gates/`

- `exec-semantics-tiers.md` (~3910 tok, huge) — Executable-semantics tier gate — doctrine and measurement history
### `docs/`

- `gpu.md` (~387 tok, medium) — GPU backend profile
- `install.md` (~1012 tok, large) — Installing mindc
- `ir-mlir.md` (~480 tok, medium) — IR & MLIR Integration
- `ir-stability.md` (~1763 tok, huge) — IR stability contract
- `ir.md` (~451 tok, medium) — MIND IR core
- `mic3-v04-draft.md` (~1928 tok, huge) — MIC3 `0x04` core codec — implementation draft
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
- `reap-pruning.md` (~901 tok, large) — REAP Expert Pruning
### `docs/rfcs/`

- `000-template.md` (~1 tok, tiny)
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
- `0010-memory-safety-and-c-abi.md` (~7359 tok, huge) — RFC 0010: Memory Safety Model + C ABI in Pure MIND
- `0011-async-and-structured-concurrency.md` (~4891 tok, huge) — RFC 0011: Async + Structured Concurrency Model
- `0012-tensor-native-syntax.md` (~12947 tok, huge) — RFC 0012: Tensor-Native Surface Syntax — the Differentiation Layer
- `0013-cli-agent-harness-stack.md` (~6776 tok, huge) — RFC 0013: CLI Agent Harness Stack
- `0014-per-substrate-mlir-lowering-contracts.md` (~5412 tok, huge) — RFC 0014: Per-Substrate MLIR Lowering Pipeline Contracts
- `0015-cross-substrate-bit-identity.md` (~5971 tok, huge) — RFC 0015: Cross-Substrate Bit-Identity Proof Obligation
- `0016-evidence-chain-emission.md` (~6998 tok, huge) — RFC 0016: Compile-Time Evidence-Chain Emission
- `0017-mindc-verify.md` (~4360 tok, huge) — RFC 0017: `mindc verify` — Artifact Verification Surface
- `0018-bare-metal-substrate.md` (~3799 tok, huge) — RFC 0018: Bare-Metal Substrate Lowering Tier
- `0019-deterministic-agent-substrate.md` (~4131 tok, huge) — RFC 0019: Deterministic Agent Substrate
- `0020-mind-bench-reproducibility-harness.md` (~5774 tok, huge) — RFC 0020: mind-bench Public Reproducibility Harness
- `0021-canonical-ir-unification.md` (~4997 tok, huge) — RFC 0021: Canonical IR Unification — one IR, provenance as a versioned epilogue
- `0022-deterministic-io-substrate.md` (~2120 tok, huge) — RFC 0022: Deterministic I/O Substrate — fastest async I/O with bit-identical replay
- `0024-loop-collapse.md` (~7579 tok, huge) — RFC 0024: Loop Collapse — prove-or-fail closed-form replacement of counted loops (`#[collapse]`)
- `0025-mind-intent-contracts.md` (~3605 tok, huge) — RFC 0025: MIND Intent — Intent Contracts (goal + constraints → verifiable Contract IR)
- `0027-governed-physical-device-plane.md` (~2790 tok, huge) — RFC 0027: Governed Physical-Device Plane — evidence-bound actuation (`device_receipts`)
- `0028-deterministic-field-calculus.md` (~10282 tok, huge) — RFC 0028: Deterministic Field Calculus
- `DRAFT-deterministic-format-frontend.md` (~10522 tok, huge) — RFC DRAFT: Deterministic Multi-Format Ingest Front-End (JSON / TOON / CSV / TSV / NDJSON / TOML)
- `DRAFT-deterministic-json-frontend.md` (~5177 tok, huge) — RFC DRAFT: Deterministic Streaming SIMD JSON Structural Front-End
- `DRAFT-governed-device-io-mhs.md` (~5629 tok, huge) — Draft RFC: Governed Device I/O and MHS Compatibility — SUPERSEDED
- `README.md` (~31 tok, tiny) — RFCs
- `odc-language-primitives.md` (~418 tok, medium) — RFC: Observer-Dependent Cognition — Language Primitives
### `docs/`

- `runs-burndown-roadmap.md` (~3203 tok, huge) — MIND RUNS Burndown Roadmap
- `security.md` (~1492 tok, large) — Security Guide
- `self-host-trace-hash-port.md` (~1406 tok, large) — #17 — Self-compute the native PT_NOTE (pure-MIND trace-hash port)
- `shapes.md` (~478 tok, medium) — Tensor shape semantics
- `sparse-tensor-types.md` (~740 tok, large) — Sparse Tensor Types
### `docs/specs/`

- `README.md` (~23 tok, tiny) — Specifications
- `v1.0.md` (~953 tok, large) — MIND Language Specification v1.0 (Working Draft)
### `docs/`

- `type-system.md` (~1520 tok, huge) — Type System
- `version-matrix.md` (~1828 tok, huge) — MIND Ecosystem — Version Matrix
- `versioning.md` (~1018 tok, large) — MIND Core Stability & Versioning
- `whitepaper.md` (~2788 tok, huge) — MIND: The Native Language for Intelligent Systems
### `examples/`

- `README.md` (~2066 tok, huge) — MIND Examples
- `anthropobrot.mind` (~3257 tok, huge) — Anthropobrot: depth-selected orbit-density multisets of the Fatou-Julia iteral.
- `autodiff_demo.mind` (~1715 tok, huge) — Autodiff Demonstration
### `examples/bimap_currency/`

- `.gitignore` (~2 tok, tiny) — target/
- `Mind.toml` (~62 tok, small) — [package]
- `main.mind` (~716 tok, large) — Single-source bijective map over a "nice set" — ONE declaration, both
### `examples/bimap_pairs/`

- `Mind.toml` (~69 tok, small) — [package]
- `main.mind` (~732 tok, large) — Single-source bijective const pair-tables — ONE declaration, both directions
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

- `README.md` (~1073 tok, large) — Compliance Example
- `audit_report.mind` (~2289 tok, huge) — audit_report.mind -- Compliance Artifact Generation
- `auditable_model.mind` (~1932 tok, huge) — auditable_model.mind -- Compliance-Ready MLP with Provenance Metadata
### `examples/`

- `cos_dottie.mind` (~781 tok, large) — Cosine-map iteration toward the Dottie fixed point (x* ≈ 0.7390851332151607),
### `examples/detmath_kat/`

- `Mind.toml` (~20 tok, tiny) — [package]
- `main.mind` (~4415 tok, huge) — examples/detmath_kat — value-oracle known-answer tests for std.detmath.
### `examples/distribution-crossisa/`

- `README.md` (~1175 tok, large) — Cross-ISA determinism: a piecewise-linear density kernel
- `afterkelly.cpp` (~2278 tok, huge) — Command line arguments. ____________________________________________
- `data1.txt` (~212 tok, medium) — 45.96
- `data2.txt` (~223 tok, medium) — 107.50
- `distribution.cpp` (~1217 tok, large)
- `distribution_interp_f64.mind` (~1231 tok, large) — Deterministic IEEE-754 float64 piecewise-LINEAR density interpolation kernel,
### `examples/`

- `dottie_collapse.mind` (~1144 tok, large) — Salov loop-collapse — Q16.16 fixed-point ITERATION collapse (Slice S3).
### `examples/emit_ir/`

- `EXPECTED.md` (~1942 tok, huge) — Phase 6.4 — Expected IR Text
- `README.md` (~2214 tok, huge) — RFC 0005 Phase 6.4 — Self-Host MLIR Text Emitter
- `bootstrap_smoke.py` (~2890 tok, huge)
- `fixture.mind` (~183 tok, small) — Phase 6.4 emit_ir smoke fixture.
- `main.mind` (~6419 tok, huge) — examples/emit_ir/main.mind — RFC 0005 Phase 6.4 self-host MLIR text emitter.
### `examples/`

- `fft_q16.mind` (~1248 tok, large) — Deterministic Q16.16 fixed-point radix-2 DIT FFT, N=256 (complex).
- `fft_signal.mind` (~533 tok, large) — FFT Signal Processing Example for MIND
- `galperin_pi.mind` (~1486 tok, large) — Galperin's billiard-π: count elastic collisions of two balls + a wall to
- `gauss_collapse.mind` (~672 tok, large) — Salov loop-collapse — closed-form affine sums (Slice S1).
- `geometric_collapse.mind` (~807 tok, large) — Salov loop-collapse — geometric powering closed forms (Slice S2).
### `examples/grammar_mask/`

- `Mind.toml` (~59 tok, small) — [package]
- `main.mind` (~4577 tok, huge) — examples/grammar_mask/main.mind — structured / grammar-constrained decoding,
### `examples/`

- `halbach_q16.mind` (~3965 tok, huge) — Deterministic Q16.16 2D Halbach-vs-uniform magnet-array field model.
### `examples/halbach_q16/`

- `Mind.toml` (~63 tok, small) — [package]
- `main.mind` (~7793 tok, huge) — examples/halbach_q16/main.mind — standalone, SELF-VERIFYING project build of
### `examples/`

- `hello_stdlib.mind` (~271 tok, medium) — Hello, std.vec — minimal RFC 0005 cookbook example.
- `hello_tensor.mind` (~141 tok, small) — Hello, MIND — scalar smoke that flows through every stage of the
### `examples/lexer/`

- `BOOTSTRAP_SMOKE_REPORT.md` (~1931 tok, huge) — Phase 6.5 Stage 1 — Bootstrap Smoke Report
- `EXPECTED.md` (~1093 tok, large) — Phase 6.1 — Expected Token Stream
- `README.md` (~969 tok, large) — RFC 0005 Phase 6.1 — Self-Host Lexer Seed
- `bootstrap_smoke.py` (~2367 tok, huge)
- `fixture.mind` (~67 tok, small) — Phase 6.1 lexer smoke fixture.
- `main.mind` (~2461 tok, huge) — examples/lexer/main.mind — RFC 0005 Phase 6.1 self-host smoke
### `examples/`

- `lorenz_f64.mind` (~230 tok, medium) — Deterministic IEEE-754 float64 Lorenz-attractor integrator (forward Euler).
- `lorenz_q16.mind` (~1091 tok, large) — Deterministic Q16.16 fixed-point Lorenz-attractor integrator (forward Euler).
- `mandelbrot.mind` (~1019 tok, large) — Deterministic IEEE-754 float64 Mandelbrot escape-count renderer.
- `mandelbrot_strict.mind` (~982 tok, large) — Strict-f64 Mandelbrot escape-count checksum — a determinism-wedge demo.
### `examples/mind_mirror_v04/`

- `RUNNER_REPLACEMENT.md` (~784 tok, large) — `mirror_gate.py` — replacement checklist
- `mirror_gate.py` (~3052 tok, huge) — Gate for the pure-MIND MIC3 v0x04 declared-prefix mirror.
### `examples/mind_mirror_v04/testdata/`

- `MANIFEST.tsv` (~2715 tok, huge) — # name	expected_exit	bytes	sha256	consumed	note
- `SEMANTIC_MANIFEST.tsv` (~1147 tok, large) — # name	expected_exit	bytes	sha256	consumed	note
### `examples/mindc_mind/`

- `.gitignore` (~5 tok, tiny) — __pycache__/
- `EXPECTED.md` (~773 tok, large) — Phase 6.5 Stage 5 — Expected IR Text (APEX)
- `FIXED_POINT_REPORT.md` (~1770 tok, huge) — Phase 6.5 — Bootstrap Fixed-Point Report
- `SMOKE_WIRING.tsv` (~3822 tok, huge) — # SMOKE_WIRING.tsv — the CHECKED contract for where each examples/mindc_mind s
- `VERDICT_SHAPE_RESIDUAL.txt` (~1100 tok, large) — # VERDICT_SHAPE_RESIDUAL.txt — the DECLARED, shrink-only residual of the
- `_diagnostic_project.py` (~376 tok, medium) — Owned CLI projects for diagnostic smoke fixtures."""
- `_frozen_native_oracle.py` (~384 tok, medium) — Integrity-checked frozen references for the native ELF smoke."""
- `_ref_add.note` (~16 tok, tiny) — 6fa59a74687e6bac38c983655d4d93ab1873299f130f68cbe481cf92041f6610
- `_ref_if_ret.note` (~16 tok, tiny) — 3fdc70390e9e12d8030552d11b2194078e8579fbcfac19b14d1beed77174cb07
- `_ref_main.note` (~16 tok, tiny) — 641c06d594084dc29762eaee24589dfb484dfea00ddcad6575e14335e1df8004
- `_ref_recursion.note` (~16 tok, tiny) — 6d125a946243b0550700d9aa6bc2058b51a3b8bf6e536a8b0b3f545d79b7346f
- `_ref_struct_field.note` (~16 tok, tiny) — 2f7f2ea32ee47138e1b0162bd51f418b4e5005b4569054f5f3c392ebd2258d96
- `_ref_value_if.note` (~16 tok, tiny) — ae565a5154b76ee1f44a32c16db2e9a8387f2f60c5fca92bba2dc9bb70afb599
- `_selfhost_loop_reseed.py` (~1550 tok, huge) — Legacy Rust-seeded re-freeze implementation for the LOOP harness.
- `_selfhost_so.py` (~3780 tok, huge) — Shared self-host `.so` resolver for the examples/mindc_mind smokes.
- `_stdlib_manifest.py` (~1129 tok, large) — Shared reader for the committed std/*.mind manifest.
- `backend_native_bridge_smoke.py` (~3985 tok, huge) — RI-D slice 1 gate (task #110): `mindc build --backend native` is a byte-faithful
- `bootstrap_smoke.py` (~2473 tok, huge)
- `branch_shadow_crossbackend_smoke.py` (~2039 tok, huge) — #318 — branch-local `let` shadow must not leak across backends (the WEDGE gate).
- `check_driver.mind` (~9782 tok, huge) — ===========================================================================
- `closure_netverify.py` (~1423 tok, large) — # Canonical independent net-verify harness for CLOSURES / FN-VALUES / UNRESOLVED
- `collect_field_strings_smoke.py` (~1215 tok, large)
- `cutover_coverage_measure.py` (~3255 tok, huge)
- `div_shift_cmp_edge_smoke.py` (~1846 tok, huge)
- `enum_netverify.py` (~1957 tok, huge) — # Canonical independent net-verify harness for C-LIKE ENUMS in the native-ELF backend.
- `fast_keystone.sh` (~3876 tok, huge) — fast_keystone.sh — fast LOCAL front-end keystone gate for the pure-MIND self-host
- `field_store_netverify.py` (~1413 tok, large) — # Canonical independent value harness for struct field STORES (`p.x = v`) in the
- `fixed_point_smoke.py` (~3275 tok, huge)
- `fixture.mind` (~183 tok, small) — Phase 6.4 emit_ir smoke fixture.
- `full_strtab_smoke.py` (~1663 tok, huge)
- `gap_corpus_smoke.py` (~5694 tok, huge)
- `general_float_netverify.py` (~1986 tok, huge) — general_float_netverify.py — GENERAL-path f64 value battery (B0 gate lift).
- `lockstep_lint.py` (~4127 tok, huge) — lockstep_lint.py -- native-ELF walker lockstep linter for the pure-MIND self-host compiler.
- `match_struct_smoke.py` (~1350 tok, large)
- `method_callee_smoke.py` (~1390 tok, large)
- `method_calls_smoke.py` (~1334 tok, large)
- `mic3_flip_smoke.py` (~1150 tok, large)
- `mic3_oracle_smoke.py` (~1060 tok, large) — mic@3 self-host convergence — Phase 0 gate: the Rust oracle.
- `mic3_primitives_smoke.py` (~22604 tok, huge) — mic@3 self-host convergence — Phase 1 gate: pure-MIND ULEB128 / zigzag.
- `mindfuzz_self_host.py` (~11539 tok, huge)
- `mod_operator_smoke.py` (~2871 tok, huge)
- `multi_let_smoke.py` (~1539 tok, huge)
- `now_ns_smoke.py` (~1008 tok, large) — # Copyright 2025 STARGA Inc.
- `option_netverify.py` (~2321 tok, huge) — # Canonical independent net-verify harness for SINGLE-PAYLOAD ENUMS (the
- `oracle_parity_lint.py` (~4516 tok, huge)
- `param_types_smoke.py` (~1273 tok, large)
- `ref_netverify.py` (~1473 tok, large) — # Canonical independent net-verify harness for i64 references in the native-ELF backend.
- `rh_f64_aggregate_canary_smoke.py` (~1597 tok, huge) — RH f64-aggregate surface canary (RH_REQUIRED_F64_AGGREGATE_SURFACE).
- `ri_d1_frozen_profile_gate.py` (~3278 tok, huge) — RI-D1 readiness gate (task #313): prove `mindc build --backend native` is READY to be
- `self_host_alias_controls_smoke.py` (~1447 tok, large) — Focused native alias-environment controls.
- `self_host_andor_precedence_smoke.py` (~1680 tok, huge) — Front-end PARITY battery for `&&` / `||` precedence + short-circuit.
- `self_host_andor_smoke.py` (~2142 tok, huge) — Permanent battery for the self-host `&&` / `||` short-circuit operators.
- `self_host_arena_growth_smoke.py` (~1382 tok, large)
- `self_host_args_from_os_smoke.py` (~1356 tok, large)
- `self_host_argv_smoke.py` (~1136 tok, large)
- `self_host_array_smoke.py` (~1425 tok, large)
- `self_host_arridx_mic3_smoke.py` (~1864 tok, huge)
- `self_host_body_smoke.py` (~3055 tok, huge)
- `self_host_carry_cap_smoke.py` (~1169 tok, large) — Cap-guard smoke for the self-host loop-carry / loop-frame scratch tables.
- `self_host_cast_mic3_smoke.py` (~4680 tok, huge)
- `self_host_check_driver_smoke.py` (~2879 tok, huge)
- `self_host_continue_smoke.py` (~1384 tok, large) — Regression gate for #308 (closes #286's break/continue fixtures): the pure-MIND
- `self_host_dtype_tag_smoke.py` (~780 tok, large) — RI-B1 per-SSA dtype-tag gate (parser <-> nb_fp_* encoder connecting construct).
- `self_host_else_if_smoke.py` (~1715 tok, huge)
- `self_host_failclosed_smoke.py` (~9902 tok, huge) — self_host_failclosed_smoke.py — the fail-closed boundary of the pure-MIND
- `self_host_float_lit_exact_smoke.py` (~1368 tok, large) — CPU-as-oracle smoke for the C1 float-literal exactness guard.
- `self_host_for_smoke.py` (~3191 tok, huge) — Permanent battery for the self-host range-`for` loop.
- `self_host_if_region_carry_smoke.py` (~4632 tok, huge) — Native-ELF smoke: i64 loop-carry through BRANCHED regions (Sub-step C).
- `self_host_ifret_chain_mic3_smoke.py` (~1725 tok, huge)
- `self_host_letpath_failclose_smoke.py` (~2243 tok, huge) — Self-host LET-PATH fail-closed smoke — pins the S1-collapse fail-open wall.
- `self_host_lockstep_smoke.py` (~2241 tok, huge) — SUB-STEP A lockstep smoke: the loop-carry frame COUNT and the loop-carry EMIT are
- `self_host_loop_smoke.md` (~1543 tok, huge) — Self-host LOOP gate contract
- `self_host_loop_smoke.py` (~8636 tok, huge) — Self-host LOOP reproduction and advancement gate.
- `self_host_match_smoke.py` (~1893 tok, huge)
- `self_host_matchscalar_mic3_smoke.py` (~1725 tok, huge)
- `self_host_mic3_float_valueif_smoke.py` (~1768 tok, huge)
- `self_host_mlir_smoke.py` (~1954 tok, huge)
- `self_host_narrow_let_mic3_smoke.py` (~1612 tok, huge)
- `self_host_narrow_param_smoke.py` (~11275 tok, huge) — Native-ELF smoke for narrow-width (i8/i16/i32) function PARAMETERS carried by a loop.
- `self_host_native_autowrap_smoke.py` (~2442 tok, huge) — Roadmap C2 declared-width AUTO-WRAP driver — narrow-int (i8/i16/i32) `let` and
- `self_host_native_avx2_dot_f32_smoke.py` (~1663 tok, huge) — RI-B2-S13 (#108) — native-ELF PACKED-f32 SIMD via 256-bit AVX2 (VEX/YMM) STRICT-FP DOT.
- `self_host_native_blas_dot_i16_smoke.py` (~2879 tok, huge)
- `self_host_native_blas_dot_q16_smoke.py` (~1774 tok, huge)
- `self_host_native_cast_conv_smoke.py` (~1373 tok, large) — RI-B2 scalar-cast-conv rung (#108) — native-ELF int<->float `as`-cast chain.
- `self_host_native_diag_smoke.py` (~1326 tok, large) — RI-D1a gate (task #313): `mindc build --backend=native` must FAIL-CLOSED with an
- `self_host_native_dot_f32_smoke.py` (~1436 tok, large) — RI-B2-S8 STEP C (#108) — native-ELF scalar STRICT-FP f32 DOT-PRODUCT.
- `self_host_native_dot_l1_q16_smoke.py` (~1188 tok, large) — RI-B2 L1-Q16 rung (#108) — native-ELF Q16.16 L1 distance.
- `self_host_native_elf_alignment_smoke.py` (~1902 tok, huge) — Self-host NATIVE-ELF ABI-alignment smoke (FIX #169).
- `self_host_native_elf_smoke.py` (~10999 tok, huge)
- `self_host_native_fp_binop_smoke.py` (~1095 tok, large) — RI-B1 nb_expr FLOAT-op-FLOAT arithmetic routing gate (zero MLIR/LLVM).
- `self_host_native_fp_call_smoke.py` (~5469 tok, huge) — RI-D2 S-C1: FLOAT call-RETURN dtype through the native-ELF general nb_expr lowering.
- `self_host_native_fp_expr_smoke.py` (~1020 tok, large) — RI-B1 nb_expr float-scalar routing gate (zero MLIR/LLVM).
- `self_host_native_fp_field_smoke.py` (~1891 tok, huge) — RI-D2 S-D FLOAT struct-FIELD READ dtype through native-ELF general lowering (zero MLIR).
- `self_host_native_fp_let_smoke.py` (~1208 tok, large) — RI-B1 (#107 follow-up) FLOAT dtype propagation ACROSS a LET binding (zero MLIR/LLVM).
- `self_host_native_fp_param_smoke.py` (~1403 tok, large) — RI-D2 S-B: FLOAT fn-param dtype classification + SysV SSE-spill ABI (zero MLIR/LLVM).
- `self_host_native_fp_smoke.py` (~1142 tok, large) — RI-B1 native-ELF scalar-f64 gate (zero MLIR/LLVM).
- `self_host_native_gemm_i8_smoke.py` (~1154 tok, large) — RI-B2-S7 (#108) — native-ELF scalar int8 GEMM (matrix x matrix), byte-identity rung.
- `self_host_native_gemm_q16_smoke.py` (~1157 tok, large) — RI-B2-S7 (#108) — native-ELF scalar GEMM Q16.16 (matrix x matrix), byte-identity rung.
- `self_host_native_gemv_i16_smoke.py` (~1148 tok, large) — RI-B2-S6 (#108) — native-ELF scalar GEMV int16 (matrix x vector), byte-identity rung.
- `self_host_native_gemv_q16_smoke.py` (~1142 tok, large) — RI-B2-S6 (#108) — native-ELF scalar GEMV Q16.16 (matrix x vector), byte-identity rung.
- `self_host_native_genf32_smoke.py` (~1066 tok, large) — RI-B2-S8 STEP B (#108) — isolate the LCG f32 rounding BEFORE the dot.
- `self_host_native_intdot_i16_smoke.py` (~1121 tok, large) — RI-B2-S2 (#108) — native-ELF scalar int16 DOT-PRODUCT, FIRST byte-identity rung.
- `self_host_native_intdot_q16_smoke.py` (~1111 tok, large) — RI-B2-S4 (#108) — native-ELF scalar Q16.16 DOT-PRODUCT, byte-identity rung.
- `self_host_native_intdot_smoke.py` (~1165 tok, large) — RI-B2-S1 (#108) scalar i64 DOT-PRODUCT reduction native-ELF (zero MLIR/LLVM).
- `self_host_native_matmul_f32_v_smoke.py` (~1718 tok, huge) — RI-B2-S9 (#108) — native-ELF scalar STRICT-FP f32 GEMV (matmul-f32-v).
- `self_host_native_narrow_add_i8_smoke.py` (~2260 tok, huge) — C2 — native-ELF NARROW-INT (i8) WRAP ARITHMETIC, zero MLIR/LLVM.
- `self_host_native_narrow_arith_batch_smoke.py` (~2554 tok, huge) — C2 — native-ELF NARROW-INT WRAP ARITHMETIC batch: {sub,mul}xi8 + {add,mul}xi16.
- `self_host_native_narrow_paramret_smoke.py` (~2029 tok, huge) — Byte-behavior smoke for narrow-int (i8/i16/i32) PARAM + RETURN auto-wrap in the
- `self_host_native_narrowint_smoke.py` (~1656 tok, huge) — Roadmap C2 narrow-int native-ELF rung — user-reachable i8/i16/i32 truncating
- `self_host_native_narrowwrap_smoke.py` (~1673 tok, huge) — Roadmap C2 narrow-int native-ELF rung — user-reachable i8/i16/i32 two's-complement
- `self_host_native_record_array_smoke.py` (~2437 tok, huge) — Focused pure-MIND native fixed-array record-field gate."""
- `self_host_native_record_trace_smoke.py` (~4806 tok, huge) — Canonical-trace contract for record-field reads, incl. the fixed-array case.
- `self_host_native_scalar_f32_smoke.py` (~1548 tok, huge) — Phase C1-remainder f32 rung — native-ELF scalar SINGLE-precision chain.
- `self_host_native_scalar_f64_smoke.py` (~1301 tok, large) — RI-B2 f64 rung (#108) — native-ELF scalar STRICT-FP f64 CHAIN.
- `self_host_native_scalar_narrow_smoke.py` (~1800 tok, huge) — RI-D / #10 native-float SLICE 1a — native-ELF SATURATING f64 -> signed NARROW
- `self_host_native_scast_smoke.py` (~1694 tok, huge)
- `self_host_native_simd_dot_f32_smoke.py` (~1606 tok, huge) — RI-B2-S11 (#108) — native-ELF PACKED-f32 SIMD (SSE, 128-bit) STRICT-FP DOT-PRODUCT.
- `self_host_native_simd_dot_i16_smoke.py` (~1234 tok, large) — RI-B2-S12 (#108) — native-ELF PACKED-int16 SIMD DOT-PRODUCT, byte-identity rung.
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
- `self_host_not_smoke.py` (~1481 tok, large)
- `self_host_open_smoke.py` (~1310 tok, large)
- `self_host_param_mutation_smoke.py` (~1323 tok, large) — CPU-as-oracle smoke for the param-mutation fix (nb_expr ident arm: consult the
- `self_host_standalone_driver_smoke.py` (~3274 tok, huge)
- `self_host_struct_return_smoke.py` (~4122 tok, huge) — self_host_struct_return_smoke.py — regression lock for STRUCT-BY-VALUE
- `self_host_tc_class_mismatch_smoke.py` (~1167 tok, large) — CPU-as-oracle smoke for the pure-MIND E2015 int<->float class-mismatch rule.
- `self_host_tc_class_rules_smoke.py` (~1889 tok, huge) — CPU-as-oracle smoke for the pure-MIND E2010/E2011/E2013/E2016 class rules.
- `self_host_tc_classify_error_code_smoke.py` (~2486 tok, huge) — CPU-as-oracle smoke for the pure-MIND classify_error_code router.
- `self_host_tc_decl_names_smoke.py` (~3651 tok, huge) — CPU-as-oracle smoke for the pure-MIND D1 module DECL-NAME SET.
- `self_host_tc_fixed_bytes_into_vec_smoke.py` (~1438 tok, large) — CPU-as-oracle smoke for the pure-MIND E2006 FIXED_BYTES_INTO_VEC rule (Bug #38).
- `self_host_tc_fn_value_call_smoke.py` (~6992 tok, huge) — CPU-as-oracle smoke for the pure-MIND B1/D4 CAPSTONE — the FULL E2012 rule.
- `self_host_tc_let_class_mismatch_smoke.py` (~1319 tok, large) — CPU-as-oracle smoke for the pure-MIND E2015 LET_CLASS_MISMATCH let/assign rule.
- `self_host_tc_let_infer_ident_smoke.py` (~2882 tok, huge) — self_host_tc_let_infer_ident_smoke — three-leg gate for the T2 port.
- `self_host_tc_let_infer_smoke.py` (~2132 tok, huge) — self_host_tc_let_infer_smoke — three-leg gate for the T1 type-inference port.
- `self_host_tc_narrowing_smoke.py` (~1042 tok, large)
- `self_host_tc_scope_frame_smoke.py` (~5756 tok, huge) — CPU-as-oracle smoke for the pure-MIND B1/D2 NESTED SCOPE-FRAME WALK.
- `self_host_tc_self_host_only_call_smoke.py` (~3492 tok, huge) — CPU-as-oracle smoke for the pure-MIND E2024 self-host-only-call rule.
- `self_host_tc_shape_annot_compat_smoke.py` (~1943 tok, huge) — CPU-as-oracle smoke for the pure-MIND shape annotation-compat rule.
- `self_host_tc_shape_rules_smoke.py` (~2149 tok, huge) — CPU-as-oracle smoke for the pure-MIND E2005/E2101/E2102/E2103 shape rules + E2023 reserved-prefix rule.
- `self_host_tc_std_export_smoke.py` (~3500 tok, huge) — CPU-as-oracle smoke for the pure-MIND D3 STD-SURFACE EXPORT NAME SET.
- `self_host_tc_undeclared_assign_smoke.py` (~7241 tok, huge) — CPU-as-oracle smoke for the pure-MIND B1 E2009 rule — undeclared assign.
- `self_host_tc_unknown_call_smoke.py` (~8549 tok, huge) — CPU-as-oracle smoke for the pure-MIND B1 E2003 rule — unknown call.
- `self_host_tc_unknown_ident_smoke.py` (~11370 tok, huge) — CPU-as-oracle smoke for the pure-MIND B1 E2002 rule — unknown identifier.
- `self_host_tc_unknown_variant_smoke.py` (~2233 tok, huge) — CPU-as-oracle smoke for the pure-MIND E2008 unknown-enum-variant rule.
- `self_host_value_if_expr_smoke.py` (~1557 tok, huge)
- `selfhost_argv_driver.mind` (~1187 tok, large) — ===========================================================================
- `selfhost_driver.mind` (~4121 tok, huge) — ===========================================================================
- `selfhost_so_provenance_smoke.py` (~2391 tok, huge) — Gate: `resolve_so()` must QUALIFY the `.so` it returns and REFUSE a stale one.
- `sha256_hash_smoke.py` (~1284 tok, large) — # Copyright 2025 STARGA Inc.
- `smoke_wiring_lint.py` (~8534 tok, huge) — smoke_wiring_lint.py — machine-checked contract for WHERE each
- `stdlib_manifest_lint.py` (~2431 tok, huge) — stdlib_manifest_lint.py — machine-checked contract for WHICH std/*.mind
- `struct_fields_smoke.py` (~1076 tok, large)
- `struct_lit_smoke.py` (~2240 tok, huge)
- `tc_differential_fuzz.py` (~13332 tok, huge) — tcdiff — differential fuzzer for the self-host source-position tc-rule ports.
### `examples/mindc_mind/testdata/backend_native_bridge/`

- `MANIFEST.txt` (~497 tok, medium) — # Pure-MIND native-ELF oracle (RI-D slice 1, task #110). Frozen from the pure-MI
- `add.elf` (~123 tok, small) — ELF> @@@8
- `add.mind` (~24 tok, tiny) — fn add(a: i64, b: i64) -> i64 {
- `const_return.elf` (~100 tok, small) — ELF> @@@8
- `const_return.mind` (~9 tok, tiny) — fn main() -> i64 {
- `local_call.elf` (~119 tok, small) — ELF> @@@8
- `local_call.mind` (~23 tok, tiny) — fn helper(x: i64) -> i64 {
- `while_sum.elf` (~146 tok, small) — ELF> @@@8
- `while_sum.mind` (~37 tok, tiny) — fn main() -> i64 {
### `examples/mindc_mind/testdata/`

- `dtk_plan_parity_smoke.py` (~2452 tok, huge)
### `examples/mindc_mind/testdata/native_elf_oracle/`

- `MANIFEST.txt` (~152 tok, small) — # Frozen native-ELF oracle references, captured before #15 deletes src/native.
- `add.elf` (~123 tok, small) — ELF> @@@8
- `if_ret.elf` (~150 tok, small) — ELF> @@@8
- `recursion.elf` (~175 tok, small) — ELF> @@@8
- `struct_field.elf` (~166 tok, small) — ELF> @@@8
- `value_if.elf` (~147 tok, small) — ELF> @@@8
### `examples/mindc_mind/testdata/native_record_array/`

- `README.md` (~760 tok, large) — Pure-MIND native record-array fixtures
- `aggregate_refuse.mind` (~33 tok, tiny) — fn main() -> i64 {
- `alias_refuse.mind` (~28 tok, tiny) — fn main() -> i64 {
- `computed_i64.mind` (~50 tok, small) — fn twice(x: i64) -> i64 {
- `computed_i64_bitwise.mind` (~40 tok, tiny) — fn main() -> i64 {
- `constant_oob.mind` (~25 tok, tiny) — fn main() -> i64 {
- `copy_out.mind` (~51 tok, small) — fn main() -> i64 {
- `direct.mind` (~42 tok, tiny) — fn main() -> i64 {
- `duplicate_owner_refuse.mind` (~46 tok, tiny) — fn read(r: A) -> i64 {
- `extent_cap_refuse.mind` (~24 tok, tiny) — fn main() -> i64 {
- `extent_overflow_refuse.mind` (~29 tok, tiny) — fn main() -> i64 {
- `field_store_refuse.mind` (~30 tok, tiny) — fn main() -> i64 {
- `i64_field_aggregate_refuse.mind` (~38 tok, tiny) — fn main() -> i64 {
- `i64_field_comparison_refuse.mind` (~29 tok, tiny) — fn main() -> i64 {
- `i64_field_float_refuse.mind` (~28 tok, tiny) — fn main() -> i64 {
- `i64_field_not_refuse.mind` (~27 tok, tiny) — fn main() -> i64 {
- `narrow_refuse.mind` (~25 tok, tiny) — fn main() -> i64 {
- `nonliteral_refuse.mind` (~32 tok, tiny) — fn main() -> i64 {
- `owners.mind` (~65 tok, small) — fn main() -> i64 {
- `record_param_copy.mind` (~54 tok, small) — fn copied(r: A) -> i64 {
- `record_param_loop.mind` (~68 tok, small) — fn total(r: A) -> i64 {
- `record_return.mind` (~32 tok, tiny) — fn make() -> A {
- `record_transport_owners.mind` (~87 tok, small) — fn read_a(r: A) -> i64 {
- `reference.mind` (~320 tok, medium) — fn direct_reference() -> i64 {
- `reference_u8.mind` (~201 tok, medium) — fn total(p: Packet) -> i64 {
- `return_extent_refuse.mind` (~32 tok, tiny) — fn make() -> A {
- `runtime_oob.mind` (~30 tok, tiny) — fn main() -> i64 {
- `u8_aggregate_refuse.mind` (~42 tok, tiny) — fn main() -> i64 {
- `u8_alias_refuse.mind` (~37 tok, tiny) — fn main() -> i64 {
- `u8_comparison_refuse.mind` (~35 tok, tiny) — fn main() -> i64 {
- `u8_constant_oob_refuse.mind` (~33 tok, tiny) — fn main() -> i64 {
- `u8_copy_out.mind` (~61 tok, small) — fn main() -> i64 {
- `u8_direct_loop.mind` (~95 tok, small) — fn total(p: Packet) -> i64 {
- `u8_float_refuse.mind` (~34 tok, tiny) — fn main() -> i64 {
- `u8_not_refuse.mind` (~33 tok, tiny) — fn main() -> i64 {
- `u8_record_return.mind` (~56 tok, small) — fn make() -> Packet {
- `u8_runtime_oob.mind` (~38 tok, tiny) — fn main() -> i64 {
- `u8_unproven_let_refuse.mind` (~38 tok, tiny) — fn main() -> i64 {
- `u8_wrong_owner_refuse.mind` (~66 tok, small) — fn read_a(p: A) -> i64 {
- `wrong_owner_param_refuse.mind` (~59 tok, small) — fn read_a(r: A) -> i64 {
- `wrong_owner_return_refuse.mind` (~50 tok, small) — fn make() -> A {
- `zero_extent_refuse.mind` (~23 tok, tiny) — fn main() -> i64 {
### `examples/mindc_mind/testdata/native_record_trace/`

- `MANIFEST.txt` (~454 tok, medium) — # Canonical record-trace fixtures — recorded evidence bound to its input.
- `README.md` (~872 tok, large) — Canonical record-trace fixtures
- `constant_oob.mind` (~28 tok, tiny) — fn main() -> i64 {
- `construct_only.mind` (~32 tok, tiny) — fn main() -> i64 {
- `copy_i64.mind` (~35 tok, tiny) — fn main() -> i64 {
- `direct_i64.mind` (~42 tok, tiny) — fn main() -> i64 {
- `direct_i64_prefold.mind` (~158 tok, small) — direct_i64.mind plus the intrinsic name literals, so the field prefold takes
- `direct_u8.mind` (~28 tok, tiny) — fn main() -> i64 {
- `reference.mind` (~212 tok, medium) — fn direct_i64_reference() -> i64 {
- `scalar_owner.mind` (~41 tok, tiny) — fn main() -> i64 {
- `scalar_owner_named.mind` (~157 tok, small) — The prefold passes rewrite field reads into `__mind_load_i64` calls whose
- `wrong_owner.mind` (~40 tok, tiny) — fn main() -> i64 {
- `wrong_type.mind` (~28 tok, tiny) — fn main() -> i64 {
### `examples/mindc_mind/testdata/`

- `rh_f64_aggregate_canary.mind` (~876 tok, large) — RH f64-aggregate self-host/native canary (RH_REQUIRED_F64_AGGREGATE_SURFACE).
### `examples/mindc_mind/testdata/selfhost_loop/`

- `MANIFEST.txt` (~312 tok, medium) — # Frozen self-host bootstrap ELF (A6/RI-E1): the checked-in pure-MIND stage0
- `PROVENANCE_298_f64_selfhost_mlir.md` (~1826 tok, huge) — #298 self-host f64 MLIR emit — self-host loop seed refreeze provenance
### `examples/mindc_mind/testdata/`

- `stdlib_manifest.txt` (~831 tok, large) — # std/*.mind MANIFEST — the single committed source of truth for "which std
### `examples/mindc_mind/`

- `unified_dispatch_smoke.py` (~1598 tok, huge)
- `validate_real_fns_smoke.py` (~2734 tok, huge)
- `while_struct_smoke.py` (~1031 tok, large)
### `examples/`

- `mlir_pipeline_demo.sh` (~1647 tok, huge) — MLIR/LLVM Pipeline Demonstration
### `examples/native/`

- `ci_kernel.mind` (~39 tok, tiny)
- `ci_kernel_smoke.py` (~1330 tok, large)
- `fib.mind` (~34 tok, tiny) — fn fib(n: i64) -> i64 {
- `loop.mind` (~37 tok, tiny) — fn main() -> i64 {
### `examples/parser/`

- `EXPECTED.md` (~2140 tok, huge) — Phase 6.2 — Expected AST Tree
- `README.md` (~2254 tok, huge) — RFC 0005 Phase 6.2 — Self-Host Parser Seed
- `bootstrap_smoke.py` (~5544 tok, huge)
- `fixture.mind` (~160 tok, small) — Phase 6.2 parser smoke fixture.
- `main.mind` (~7825 tok, huge) — examples/parser/main.mind — RFC 0005 Phase 6.2 self-host parser seed.
### `examples/`

- `policy.mind` (~1301 tok, large) — policy.mind — v0.1 Execution Boundary Kernel
- `remizov_benchmark.mind` (~6400 tok, huge) — ASPIRATIONAL DEMO — not yet buildable with the open mindc.
- `remizov_feynman.mind` (~2894 tok, huge) — ASPIRATIONAL DEMO — not yet buildable with the open mindc.
- `remizov_gpu.mind` (~2662 tok, huge) — ASPIRATIONAL DEMO — not yet buildable with the open mindc.
- `remizov_inverse.mind` (~2614 tok, huge) — ASPIRATIONAL DEMO — not yet buildable with the open mindc.
- `remizov_solver.mind` (~3802 tok, huge) — ASPIRATIONAL DEMO — not yet buildable with the open mindc.
- `remizov_verify.mind` (~3721 tok, huge) — ASPIRATIONAL DEMO — not yet buildable with the open mindc.
- `tiny_edge_model.mind` (~1876 tok, huge) — Tiny Edge Model Example
### `examples/typecheck/`

- `EXPECTED.md` (~2015 tok, huge) — Phase 6.3 — Expected Type-Check Report
- `README.md` (~2612 tok, huge) — RFC 0005 Phase 6.3 — Self-Host Type-Checker Seed
- `bootstrap_smoke.py` (~2608 tok, huge)
- `fixture.mind` (~198 tok, small) — Phase 6.3 type-checker smoke fixture.
- `main.mind` (~7120 tok, huge) — examples/typecheck/main.mind — RFC 0005 Phase 6.3 self-host
### `examples/zoo/`

- `README.md` (~1191 tok, large) — MIND Model Zoo
- `conv_classifier.mind` (~2407 tok, huge) — ASPIRATIONAL DEMO — not yet buildable with the open mindc.
- `linear_regression.mind` (~1347 tok, large) — ASPIRATIONAL DEMO — not yet buildable with the open mindc.
- `logistic_classifier.mind` (~1517 tok, huge) — ASPIRATIONAL DEMO — not yet buildable with the open mindc.
- `mlp_mnist.mind` (~2275 tok, huge) — ASPIRATIONAL DEMO — not yet buildable with the open mindc.
- `transformer_block.mind` (~3781 tok, huge) — ASPIRATIONAL DEMO — not yet buildable with the open mindc.
### `experiments/global-vs-local/`

- `README.md` (~839 tok, large) — Global vs Local — "Closed-form whole-field invariant" experiments
- `chern.py` (~1244 tok, large)
- `exp2.py` (~1338 tok, large)
- `exp3_universal.py` (~997 tok, large)
- `plot.py` (~904 tok, large)
- `plot_chern.py` (~469 tok, medium)
- `topo.py` (~696 tok, large)
### `mind/std/cognitive/`

- `batch_scheduler.mind` (~850 tok, large) — Batch scheduling for inference workloads
- `kv_cache.mind` (~840 tok, large) — KV-Cache for transformer inference
- `speculative.mind` (~891 tok, large) — Speculative decoding with rejection sampling
- `verification.mind` (~948 tok, large) — Verification plane for inference consistency (LCU)
### `runtime-support/`

- `mind_intrinsics.c` (~21930 tok, huge) — Copyright 2025 STARGA Inc.
### `scripts/`

- `ai_attribution_patterns.sh` (~1536 tok, huge) — shellcheck shell=bash
- `anatomy-hook.sh` (~1811 tok, huge) — STARGA author guard (chained first: a wrong-identity commit must never be created).
- `anatomy.sh` (~2453 tok, huge) — anatomy — Generate ANATOMY.md for any repo
- `cfg_gate_wiring_lint.py` (~4265 tok, huge) — Fail closed when a feature-gated integration test is wired to no CI selector.
- `check_claims.py` (~7653 tok, huge) — Docs-claim CI gate — fail if any public surface drifts from config/capabilities.toml.
- `check_commit_messages.sh` (~1428 tok, large) — check_commit_messages.sh <rev-range> — no model or tool credit in commit
- `check_gate_wiring.py` (~9399 tok, huge) — Wiring lint: a whole-tree gate's CI trigger must cover every path it scans.
- `check_json_not_evidence.sh` (~813 tok, large) — Wedge-integrity gate: JSON is never an evidence-hash preimage.
- `check_no_ai_attribution.sh` (~2217 tok, huge) — Public-artifact hygiene gate: no AI tool/model named as having AUTHORED or
- `check_release_gating.py` (~4342 tok, huge) — check_release_gating.py — machine-checked supply-chain contract for the
- `commit-msg-hook.sh` (~866 tok, large) — commit-msg-hook.sh — local commit-msg hook: refuse a commit whose MESSAGE
- `enumeration_drift_lint.py` (~2544 tok, huge) — Fail closed when a comment's ENUMERATION drifts from the code it describes.
- `exec_semantics_gate.sh` (~9995 tok, huge) — exec_semantics_gate.sh — run the test tiers CI could not reach, and PROVE they ran.
- `gate_assert.py` (~3627 tok, huge) — gate_assert.py — the ONE definition of "did this gate actually assert anything?"
- `gate_runner_wiring_lint.py` (~6454 tok, huge) — Wiring lint: a workflow may reach a gate ONLY through scripts/run_gate.py.
- `install.ps1` (~1856 tok, huge) — # install.ps1 - mindc one-line installer for Windows (PowerShell)
- `install.sh` (~1054 tok, large) — MIND compiler (mindc) installer — downloads a pre-built binary from the
### `scripts/mind-vs-rust/`

- `.gitignore` (~3 tok, tiny) — */target-*/
- `Cargo.toml` (~271 tok, medium) — [package]
- `run.sh` (~659 tok, large) — Copyright 2026 STARGA Inc. Licensed under the Apache License, Version 2.0.
### `scripts/mind-vs-rust/src/`

- `main.rs` (~2372 tok, huge) — Copyright 2026 STARGA Inc.
### `scripts/`

- `operand_visitor_parity_lint.py` (~1198 tok, large) — Fail closed when the two IR operand visitors drift apart.
- `pqc_hybrid_ci.sh` (~663 tok, large) — Required-CI execution gate for the existing MIC3 two-leg PQC-hybrid controls.
- `pqc_hybrid_cli_ci.sh` (~1160 tok, large) — Required-CI execution gate for the actual-CLI PQC-hybrid signing controls.
- `preflight.sh` (~9404 tok, huge) — preflight.sh — local CI-parity gate. Run before pushing to avoid red CI.
- `quick_perf.sh` (~734 tok, large) — quick_perf.sh — FAST one-sided compile-speed criterion gate.
- `ri_d1_mlir_free_gate.sh` (~2136 tok, huge) — RI-D1 gate assertion #1 — MLIR is un-linkable on the native path (fail-closed
- `run_crypto_vectors.sh` (~3204 tok, huge) — Build every pure-MIND crypto/TLS std module to a shared object and run its
- `run_gate.py` (~8072 tok, huge) — run_gate.py — the shared gate runner that makes a VACUOUS PASS structurally fail.
### `scripts/sdlc/`

- `enforcement_bijection.py` (~1504 tok, huge) — A test must never outlive the rule it asserts.
- `lost_by_merge.py` (~1341 tok, large) — Fail a merge that silently DROPPED a line both parents agreed on.
### `scripts/`

- `test_exec_semantics_gate.py` (~6646 tok, huge) — Regression test for scripts/exec_semantics_gate.sh: a FAILING test must red the tier.
- `test_gate_wiring.py` (~7126 tok, huge) — Regression test for scripts/check_gate_wiring.py.
- `test_no_ai_attribution.py` (~3509 tok, huge) — Self-test for the no-AI-attribution gate and for what feeds it.
- `test_smoke_wiring_lint.py` (~2257 tok, huge) — Regression test for the CI-coverage and branch-awareness contracts of
- `verify_ci_green.py` (~3651 tok, huge) — verify_ci_green.py — refuse to release a commit that CI has not proven green.
- `workflow_scan.py` (~4847 tok, huge) — workflow_scan.py — dependency-free structural reader for the GitHub Actions
### `sdk/ts/mic-map/`

- `LICENSE` (~2573 tok, huge) —                                  Apache License
- `README.md` (~152 tok, small) — @mind/mic-map
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

- `package-lock.json` (~12339 tok, huge) — Keys: name, version, lockfileVersion, requires, packages
- `package.json` (~210 tok, medium) — Keys: name, version, description, type, private
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

- `mod.rs` (~10960 tok, huge) — Copyright 2025 STARGA Inc.
### `src/autodiff/`

- `engine.rs` (~3890 tok, huge) — Copyright 2025 STARGA Inc.
- `mod.rs` (~342 tok, medium) — Copyright 2025 STARGA Inc.
- `rules.rs` (~2392 tok, huge) — Copyright 2025 STARGA Inc.
### `src/bin/`

- `mind-ai.rs` (~9878 tok, huge) — Copyright 2025 STARGA Inc.
### `src/bin/mindc/`

- `mic3_output.rs` (~204 tok, medium) — Copyright 2025-2026 STARGA Inc.
### `src/build/`

- `artifact.rs` (~1149 tok, large) — Copyright 2025 STARGA Inc.
- `cache.rs` (~6905 tok, huge) — Copyright 2025 STARGA Inc.
### `src/build/cache/`

- `cache_integrity.rs` (~1160 tok, large) — Copyright 2025-2026 STARGA Inc.
- `cache_integrity_tests.rs` (~1673 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `compiler_identity.rs` (~317 tok, medium) — Copyright 2025-2026 STARGA Inc.
### `src/build/`

- `driver_error.rs` (~699 tok, large) — Copyright 2025 STARGA Inc.
- `error.rs` (~2022 tok, huge) — Copyright 2025 STARGA Inc.
- `mod.rs` (~9611 tok, huge) — Copyright 2025 STARGA Inc.
- `native_bridge.rs` (~5239 tok, huge) — Copyright 2025 STARGA Inc.
- `native_image.rs` (~2018 tok, huge) — Copyright 2025 STARGA Inc.
- `native_scope.rs` (~6258 tok, huge) — Copyright 2025 STARGA Inc.
- `native_scope_tests.rs` (~5219 tok, huge) — Seam tests for the captured-source snapshot and its module identities.
- `project_transaction.rs` (~1908 tok, huge) — Copyright 2026 STARGA Inc.
- `source_key.rs` (~3219 tok, huge) — Copyright 2025 STARGA Inc.
- `source_snapshot_tests.rs` (~1337 tok, large) — Copyright 2025-2026 STARGA Inc.
- `transaction_identity_tests.rs` (~2800 tok, huge) — Copyright 2026 STARGA Inc.
### `src/cache/`

- `entry.rs` (~977 tok, large) — Copyright 2025-2026 STARGA Inc.
- `fingerprint.rs` (~629 tok, large) — Copyright 2025-2026 STARGA Inc.
- `mod.rs` (~964 tok, large) — Copyright 2025-2026 STARGA Inc.
- `store.rs` (~1112 tok, large) — Copyright 2025-2026 STARGA Inc.
### `src/check/`

- `gitignore.rs` (~2345 tok, huge) — Copyright 2025 STARGA Inc.
- `mod.rs` (~7894 tok, huge) — Copyright 2025 STARGA Inc.
- `reporter.rs` (~374 tok, medium) — Copyright 2025 STARGA Inc.
### `src/`

- `conformance.rs` (~8426 tok, huge) — Per-profile execution counts for one conformance run.
### `src/deps/`

- `mod.rs` (~9388 tok, huge) — Copyright 2025 STARGA Inc.
### `src/diagnostics/`

- `capability.rs` (~9002 tok, huge) — Copyright 2025 STARGA Inc.
- `mod.rs` (~3729 tok, huge) — Copyright 2025 STARGA Inc.
- `refusal.rs` (~1354 tok, large) — Copyright 2025 STARGA Inc.
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
- `mod.rs` (~7402 tok, huge) — Copyright 2025 STARGA Inc.
### `src/eval/`

- `abi_gate.rs` (~11329 tok, huge) — Runnable-artifact ABI gate (release-readiness P1.1).
- `abi_gate_struct_ops.rs` (~2875 tok, huge) — Copyright 2026 STARGA Inc.
- `abi_gate_tensor.rs` (~1232 tok, large) — Tensor-shape admission rules for the runnable artifact ABI gate.
- `assert_check.rs` (~989 tok, large) — Copyright 2025 STARGA Inc.
- `autodiff.rs` (~14268 tok, huge) — Copyright 2025 STARGA Inc.
- `callable_exports.rs` (~596 tok, large) — Copyright 2026 STARGA Inc.
- `canonical_lowering.rs` (~4108 tok, huge) — Copyright 2026 STARGA Inc.
- `canonical_producers.rs` (~734 tok, large) — Copyright 2026 STARGA Inc.
- `closures.rs` (~9338 tok, huge) — Copyright 2025 STARGA Inc.
- `conv2d_grad.rs` (~2397 tok, huge) — Copyright 2025 STARGA Inc.
- `declared_width.rs` (~8263 tok, huge) — Copyright 2025 STARGA Inc.
- `field_access.rs` (~4734 tok, huge) — Copyright 2026 STARGA Inc.
- `field_assign_refusal_tests.rs` (~349 tok, medium) — This test targets the module executor's dispatch. Its hand-built
- `fixed_array.rs` (~1478 tok, large) — Copyright 2026 STARGA Inc.
- `fixed_array_struct.rs` (~5770 tok, huge) — Copyright 2026 STARGA Inc.
- `import_collision_gate.rs` (~427 tok, medium) — Copyright 2026 STARGA Inc.
- `interp_mem.rs` (~3817 tok, huge) — Copyright 2025 STARGA Inc.
- `ir_interp.rs` (~4212 tok, huge) — Copyright 2025 STARGA Inc.
- `lower_entry.rs` (~359 tok, medium) — Copyright 2026 STARGA Inc.
- `materialization.rs` (~5006 tok, huge) — Copyright 2026 STARGA Inc.
- `mlir_build.rs` (~10897 tok, huge) — Copyright 2025 STARGA Inc.
- `mlir_export.rs` (~12220 tok, huge) — Copyright 2025 STARGA Inc.
- `mlir_gpu.rs` (~301 tok, medium) — Copyright 2025 STARGA Inc.
- `mlir_jit.rs` (~501 tok, large) — Copyright 2025 STARGA Inc.
- `mlir_opt.rs` (~995 tok, large) — Copyright 2025 STARGA Inc.
- `mlir_run.rs` (~1535 tok, huge) — Copyright 2025 STARGA Inc.
- `module_bindings.rs` (~1211 tok, large) — Copyright 2025-2026 STARGA Inc.
- `module_globals.rs` (~1160 tok, large) — Copyright 2026 STARGA Inc.
- `narrow_arith.rs` (~3968 tok, huge) — Copyright 2025 STARGA Inc.
- `narrow_scan.rs` (~3042 tok, huge) — Module-wide narrow-int SURFACE prescan (compile-speed early-skip).
- `slice_abi.rs` (~325 tok, medium) — Copyright 2025 STARGA Inc.
### `src/eval/stdlib/`

- `mod.rs` (~169 tok, small) — Copyright 2025 STARGA Inc.
- `tensor.rs` (~8417 tok, huge) — Copyright 2025 STARGA Inc.
### `src/eval/`

- `string_equality.rs` (~5380 tok, huge) — Copyright 2025 STARGA Inc.
- `struct_resolver.rs` (~7167 tok, huge) — Copyright 2025 STARGA Inc.
### `src/eval/struct_resolver/`

- `inference.rs` (~2555 tok, huge) — Copyright 2025 STARGA Inc.
### `src/eval/`

- `traits.rs` (~4150 tok, huge) — Copyright 2025 STARGA Inc.
- `type_aliases.rs` (~2114 tok, huge) — Copyright 2025 STARGA Inc.
- `value.rs` (~2228 tok, huge) — Copyright 2025 STARGA Inc.
### `src/exec/`

- `conv.rs` (~435 tok, medium) — Copyright 2025 STARGA Inc.
- `cpu.rs` (~4500 tok, huge) — Copyright 2025 STARGA Inc.
- `mod.rs` (~587 tok, large) — Copyright 2025 STARGA Inc.
### `src/ffi/`

- `header.rs` (~413 tok, medium) — Copyright 2025 STARGA Inc.
- `mod.rs` (~1780 tok, huge) — Copyright 2025 STARGA Inc.
- `sys.rs` (~3348 tok, huge) — Copyright 2025-2026 STARGA Inc.
### `src/fmt/`

- `cli.rs` (~4013 tok, huge) — Copyright 2025 STARGA Inc.
- `mod.rs` (~2739 tok, huge) — Copyright 2025 STARGA Inc.
- `printer.rs` (~17300 tok, huge) — Copyright 2025 STARGA Inc.
### `src/`

- `intrinsic_contract.rs` (~2968 tok, huge) — Copyright 2025 STARGA Inc.
- `intrinsics.rs` (~10966 tok, huge) — Copyright 2025 STARGA Inc.
### `src/ir/`

- `canonical_verify.rs` (~6429 tok, huge) — Copyright 2025 STARGA Inc.
- `canonical_verify_tests.rs` (~6827 tok, huge) — Copyright 2025 STARGA Inc.
### `src/ir/compact/`

- `emit.rs` (~4693 tok, huge) — Copyright 2025 STARGA Inc.
- `mod.rs` (~2534 tok, huge) — Copyright 2025 STARGA Inc.
- `parse.rs` (~8124 tok, huge) — Copyright 2025 STARGA Inc.
### `src/ir/compact/v2/`

- `binary.rs` (~8844 tok, huge) — Copyright 2025 STARGA Inc.
- `emit.rs` (~2445 tok, huge) — Copyright 2025 STARGA Inc.
- `evidence.rs` (~10501 tok, huge) — Copyright 2025 STARGA Inc.
- `map_limits.rs` (~1823 tok, huge) — Copyright 2025 STARGA Inc.
- `map_tests.rs` (~8117 tok, huge) — Copyright 2025 STARGA Inc.
- `mod.rs` (~1343 tok, large) — Copyright 2025 STARGA Inc.
- `parse.rs` (~5388 tok, huge) — Copyright 2025 STARGA Inc.
- `types.rs` (~4764 tok, huge) — Copyright 2025 STARGA Inc.
- `varint.rs` (~2138 tok, huge) — Copyright 2025 STARGA Inc.
### `src/ir/compact/v3/`

- `boundary.rs` (~910 tok, large) — Copyright 2025 STARGA Inc.
- `boundary_tests.rs` (~2796 tok, huge) — Copyright 2025 STARGA Inc.
- `canonical_boundary.rs` (~409 tok, medium) — Copyright 2025 STARGA Inc.
- `collapse_receipt.rs` (~5109 tok, huge) — Copyright 2025 STARGA Inc.
- `ed25519.rs` (~6051 tok, huge) — Copyright 2025 STARGA Inc.
- `emit.rs` (~12388 tok, huge) — Copyright 2025 STARGA Inc.
- `error.rs` (~989 tok, large) — Copyright 2025 STARGA Inc.
- `evidence_size.rs` (~443 tok, medium) — Copyright 2025-2026 STARGA Inc.
- `format.rs` (~911 tok, large) — Copyright 2025-2026 STARGA Inc.
- `mldsa.rs` (~2688 tok, huge) — Copyright 2025 STARGA Inc.
- `mod.rs` (~20273 tok, huge) — Copyright 2025 STARGA Inc.
- `parse.rs` (~13043 tok, huge) — Copyright 2025 STARGA Inc.
- `slhdsa.rs` (~3201 tok, huge) — Copyright 2025 STARGA Inc.
### `src/ir/compact/v3/v04/`

- `decode.rs` (~6283 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `encode.rs` (~6147 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `mod.rs` (~1266 tok, large) — Copyright 2025-2026 STARGA Inc.
### `src/ir/compact/v3/`

- `v04_test_support.rs` (~1571 tok, huge) — Copyright 2026 STARGA Inc.
- `v04_tests.rs` (~5843 tok, huge) — Copyright 2025-2026 STARGA Inc.
### `src/ir/`

- `evidence.rs` (~13387 tok, huge) — Copyright 2025 STARGA Inc.
- `fp_mode.rs` (~13664 tok, huge) — FP-contract mode — the strict-vs-relaxed floating-point determinism state of
- `frozen_profile.rs` (~7690 tok, huge) — Copyright 2025 STARGA Inc.
- `mod.rs` (~17720 tok, huge) — Copyright 2025 STARGA Inc.
- `native_closure.rs` (~4270 tok, huge) — Copyright 2025 STARGA Inc.
- `print.rs` (~4421 tok, huge) — Copyright 2025 STARGA Inc.
- `verify.rs` (~21216 tok, huge) — Copyright 2025 STARGA Inc.
### `src/`

- `lib.rs` (~1118 tok, large) — Copyright 2025 STARGA Inc.
- `linalg.rs` (~2025 tok, huge) — Copyright 2025 STARGA Inc.
### `src/lint/`

- `mod.rs` (~1311 tok, large) — Copyright 2025 STARGA Inc.
- `rule.rs` (~2690 tok, huge) — Copyright 2025 STARGA Inc.
### `src/lint/rules/`

- `mod.rs` (~454 tok, medium) — Copyright 2025 STARGA Inc.
- `naming_convention.rs` (~2031 tok, huge) — Copyright 2025 STARGA Inc.
- `q16_overflow.rs` (~2950 tok, huge) — Copyright 2025 STARGA Inc.
- `shadowing.rs` (~1944 tok, huge) — Copyright 2025 STARGA Inc.
- `trailing_whitespace.rs` (~869 tok, large) — Copyright 2025 STARGA Inc.
- `unused_import.rs` (~1769 tok, huge) — Copyright 2025 STARGA Inc.
### `src/`

- `main.rs` (~6847 tok, huge) — Copyright 2025 STARGA Inc.
### `src/mlir/`

- `c_export.rs` (~2651 tok, huge) — Copyright 2025 STARGA Inc.
- `gemm_tuning.rs` (~3639 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `mod.rs` (~335 tok, medium) — Copyright 2025 STARGA Inc.
### `src/ops/`

- `cerebras.rs` (~2713 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `core_v1.rs` (~1823 tok, huge) — Copyright 2025 STARGA Inc.
- `mod.rs` (~228 tok, medium) — Copyright 2025 STARGA Inc.
### `src/opt/`

- `collapse.rs` (~11454 tok, huge) — Copyright 2025 STARGA Inc.
- `comptime.rs` (~5806 tok, huge) — Copyright 2025 STARGA Inc.
- `fold.rs` (~2046 tok, huge) — Copyright 2025 STARGA Inc.
- `ir_canonical.rs` (~5647 tok, huge) — Copyright 2025 STARGA Inc.
- `memory_layout.rs` (~4119 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `mod.rs` (~544 tok, large) — Copyright 2025 STARGA Inc.
- `native_opt.rs` (~11140 tok, huge) — Copyright 2025 STARGA Inc.
- `regalloc_dtk.rs` (~4282 tok, huge) — Copyright 2025 STARGA Inc.
- `scev.rs` (~8604 tok, huge) — Copyright 2025 STARGA Inc.
### `src/package/`

- `manifest.rs` (~310 tok, medium) — Copyright 2025 STARGA Inc.
- `mod.rs` (~1567 tok, huge) — Copyright 2025 STARGA Inc.
### `src/parser/`

- `eval_imports.rs` (~2137 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `expand_bimap.rs` (~16544 tok, huge) — An `#[bimap]` attribute on an `enum` declares a one-to-one correspondence
- `import_list_tests.rs` (~2764 tok, huge) — Completeness of the parser-owned import list.
- `trivia.rs` (~3811 tok, huge) — Copyright 2025 STARGA Inc.
### `src/phf/`

- `mod.rs` (~4955 tok, huge) — Copyright 2025 STARGA Inc.
### `src/`

- `pipeline.rs` (~7372 tok, huge) — Copyright 2025 STARGA Inc.
### `src/project/`

- `active_module_table.rs` (~3677 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `artifact.rs` (~1045 tok, large) — Copyright 2025 STARGA Inc.
- `build_input_snapshot.rs` (~4632 tok, huge) — Copyright 2025 STARGA Inc.
- `build_lock.rs` (~1630 tok, huge) — Copyright 2026 STARGA Inc.
- `call_bindings.rs` (~5326 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `canonical_bridge.rs` (~5927 tok, huge) — Copyright 2026 STARGA Inc.
- `compiled_sources.rs` (~461 tok, medium) — Copyright 2025 STARGA Inc.
- `embedded_entry.rs` (~670 tok, large) — Copyright 2025-2026 STARGA Inc.
- `identity.rs` (~1719 tok, huge) — Copyright 2026 STARGA Inc.
- `link.rs` (~2598 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `module_table.rs` (~7745 tok, huge) — Copyright 2025 STARGA Inc.
- `runtime_link.rs` (~1621 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `single_file_scope.rs` (~6601 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `source_snapshot.rs` (~608 tok, large) — Copyright 2025-2026 STARGA Inc.
- `sources.rs` (~5153 tok, huge) — Copyright 2025 STARGA Inc.
- `stdlib.rs` (~3656 tok, huge) — Copyright 2025 STARGA Inc.
- `substrate_link.rs` (~2330 tok, huge) — Copyright 2025 STARGA Inc.
- `toolchain_pin.rs` (~3657 tok, huge) — The `[mind]` table is the ONLY *declared* cross-repo compatibility contract
### `src/`

- `python.rs` (~1082 tok, large) — Copyright 2025 STARGA Inc.
- `qualified_enums.rs` (~5560 tok, huge) — Copyright 2025-2026 STARGA Inc.
### `src/runtime/`

- `gpu.rs` (~288 tok, medium) — Experimental GPU backend contract for MIND.
- `mod.rs` (~92 tok, small) — Runtime abstractions for execution backends.
- `types.rs` (~1105 tok, large) — Shared runtime surface types for execution backends.
### `src/`

- `runtime_interface.rs` (~573 tok, large) — Describes a tensor visible to the runtime.
### `src/shapes/`

- `engine.rs` (~1882 tok, huge) — A rank-N tensor shape represented as a list of extents.
- `mod.rs` (~4170 tok, huge) — Copyright 2025 STARGA Inc.
### `src/stdlib/`

- `mod.rs` (~169 tok, small) — Copyright 2025 STARGA Inc.
- `tensor.rs` (~391 tok, medium) — Copyright 2025 STARGA Inc.
### `src/`

- `target.rs` (~5031 tok, huge) — Copyright 2025-2026 STARGA Inc.
### `src/test/`

- `imports.rs` (~4610 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `mod.rs` (~6606 tok, huge) — Copyright 2025 STARGA Inc.
- `runner.rs` (~2689 tok, huge) — Copyright 2025 STARGA Inc.
- `top_level.rs` (~339 tok, medium) — Copyright 2025-2026 STARGA Inc.
### `src/type_checker/`

- `array_control_flow.rs` (~1096 tok, large) — Copyright 2026 STARGA Inc.
- `array_lengths.rs` (~5788 tok, huge) — Copyright 2026 STARGA Inc.
- `canonical_facts.rs` (~2797 tok, huge) — Copyright 2026 STARGA Inc.
- `canonical_macro.rs` (~58 tok, small) — Copyright 2026 STARGA Inc.
- `cross_module_types.rs` (~453 tok, medium) — Copyright 2026 STARGA Inc.
- `duplicate_structs.rs` (~424 tok, medium) — Copyright 2026 STARGA Inc.
- `lowering_refusals.rs` (~6770 tok, huge) — Copyright 2026 STARGA Inc.
- `lowering_refusals_tests.rs` (~4372 tok, huge) — Copyright 2026 STARGA Inc.
- `nerve_lint.rs` (~7669 tok, huge) — Copyright 2025 STARGA Inc.
- `nerve_walk.rs` (~2232 tok, huge) — Copyright 2025 STARGA Inc.
- `qualified_enums.rs` (~3239 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `qualified_imports.rs` (~268 tok, medium) — Feature-neutral span-scoped qualified import predicates for name resolution.
- `resolve.rs` (~15095 tok, huge) — Copyright 2025 STARGA Inc.
- `return_checks.rs` (~1244 tok, large) — Copyright 2025 STARGA Inc.
- `slice_abi.rs` (~7584 tok, huge) — Copyright 2025 STARGA Inc.
### `src/type_checker/slice_abi/`

- `borrow_flow.rs` (~7081 tok, huge) — Copyright 2025 STARGA Inc.
- `owner_assignment.rs` (~1145 tok, large) — Copyright 2026 STARGA Inc.
- `provenance.rs` (~3019 tok, huge) — Copyright 2026 STARGA Inc.
### `src/type_checker/`

- `stdlib_signatures.rs` (~620 tok, large) — Copyright 2025-2026 STARGA Inc.
- `struct_bindings.rs` (~1070 tok, large) — Copyright 2026 STARGA Inc.
- `type_display.rs` (~628 tok, large) — Copyright 2026 STARGA Inc.
### `src/types/`

- `canonical.rs` (~256 tok, medium) — Copyright 2025 STARGA Inc.
- `canonical_intrinsic_tests.rs` (~1418 tok, large) — Copyright 2025 STARGA Inc.
- `canonical_intrinsic_validation.rs` (~446 tok, medium) — Copyright 2025 STARGA Inc.
- `canonical_registry.rs` (~5107 tok, huge) — Copyright 2025 STARGA Inc.
- `canonical_tests.rs` (~3544 tok, huge)
- `canonical_types.rs` (~6538 tok, huge) — Copyright 2025 STARGA Inc.
- `infer.rs` (~448 tok, medium) — Copyright 2025 STARGA Inc.
- `intern.rs` (~1554 tok, huge) — Copyright 2025 STARGA Inc.
- `mod.rs` (~1132 tok, large) — Copyright 2025 STARGA Inc.
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
- `http.mind` (~6682 tok, huge) — std/http.mind — HTTP/1.1 client over std.net (task #XXX).
- `http2_frame.mind` (~4191 tok, huge) — std/http2_frame.mind — HTTP/2 framing layer (RFC 9113 §3.4, §4.1, §6) in
- `io.mind` (~1719 tok, huge) — std/io.mind — RFC 0005 Phase 2: pure-MIND I/O surface.
- `io_canon.mind` (~2624 tok, huge) — std.io_canon — canonical completion ordering for deterministic I/O.
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

- `CONFORMANCE_TESTS.md` (~1563 tok, huge) — MIND Conformance Test Corpus
- `_ref_mic3_dump.rs` (~1821 tok, huge) — Committed self-host reference generator (A9b): reconstruct
- `_self_host_loop_advance_support.py` (~1643 tok, huge) — Shared fixtures for the self-host loop advancement controls.
- `aggregate_const_run.rs` (~1765 tok, huge) — Copyright 2026 STARGA Inc.
- `aggshape_reject.rs` (~1842 tok, huge) — Copyright 2025 STARGA Inc.
- `alias_miscompile_run.rs` (~1287 tok, large) — Copyright 2025 STARGA Inc.
- `array_ctor_push_get_run.rs` (~861 tok, large) — Copyright 2025 STARGA Inc.
- `array_load_bounds_and_dtype.rs` (~2116 tok, huge) — Copyright 2025 STARGA Inc.
- `array_oob_trap_run.rs` (~2050 tok, huge) — Copyright 2025 STARGA Inc.
- `array_store_run.rs` (~2201 tok, huge) — Copyright 2026 STARGA Inc.
- `array_surface_run.rs` (~829 tok, large) — Copyright 2025 STARGA Inc.
- `array_u64_element_shift_run.rs` (~989 tok, large) — Copyright 2025 STARGA Inc.
- `autodiff.rs` (~2672 tok, huge) — Gradient for x*x accumulates two paths: d/dx (x*x) = x + x.
### `tests/autodiff/`

- `matmul_gradient.mind` (~167 tok, small) — Autodiff test: MatMul gradient computation
- `simple_gradient.mind` (~80 tok, small) — Autodiff test: Simple scalar gradient
### `tests/`

- `autodiff_preview.rs` (~398 tok, medium) — Copyright 2025 STARGA Inc.
### `tests/backend/`

- `cpu_available.mind` (~52 tok, small) — Backend test: CPU backend availability
- `gpu_graceful_failure.mind` (~73 tok, small) — Backend test: GPU backend graceful failure
### `tests/`

- `bare_variant_ambiguity_run.rs` (~2049 tok, huge) — Copyright 2025 STARGA Inc.
- `bare_variant_ctor_run.rs` (~949 tok, large) — Copyright 2025 STARGA Inc.
- `bimap_derive.rs` (~7107 tok, huge) — Copyright 2025 STARGA Inc.
- `bitwise_no_panic_any_feature.rs` (~1848 tok, huge) — Copyright 2025 STARGA Inc.
- `bitwise_subset_refusal.rs` (~303 tok, medium) — The low-level subset must refuse bitwise syntax at the parser boundary.
- `blas_smoke.rs` (~6312 tok, huge) — Copyright 2025 STARGA Inc.
- `blas_vec_q16_smoke.rs` (~5743 tok, huge) — Copyright 2025 STARGA Inc.
- `blas_vec_smoke.rs` (~2163 tok, huge) — Copyright 2025 STARGA Inc.
- `bool_literal_value_run.rs` (~740 tok, large) — Copyright 2025 STARGA Inc.
- `bug6_tuple_destructure_u64_run.rs` (~1360 tok, large) — Copyright 2025 STARGA Inc.
- `bug_f4_closure_shadow_run.rs` (~1567 tok, huge) — Copyright 2025 STARGA Inc.
- `build_run_runnable_blocker_gate.rs` (~2835 tok, huge) — Copyright 2025 STARGA Inc. Licensed under the Apache License, Version 2.0.
- `bytes_buffer_run.rs` (~669 tok, large) — Copyright 2025 STARGA Inc.
- `bytes_fixed_into_vec_run.rs` (~1764 tok, huge) — Copyright 2025 STARGA Inc.
- `bytes_zero_run.rs` (~843 tok, large) — Copyright 2025 STARGA Inc.
- `canonical_source_bridge.rs` (~3130 tok, huge) — The ordinary evaluator retains its unit-placeholder return convention.
- `capability_refusal_cause_scan.rs` (~3400 tok, huge) — Copyright 2025 STARGA Inc.
- `cerebras_stencil_tile.rs` (~1929 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `chacha20_poly1305_smoke.rs` (~2259 tok, huge) — Copyright 2025 STARGA Inc.
- `char_literal_run.rs` (~944 tok, large) — Copyright 2025 STARGA Inc.
- `check_claims_gate_tests.py` (~6681 tok, huge) — Gate tests for scripts/check_claims.py — its two DERIVED numbers must bite.
- `cli_buffers.rs` (~422 tok, medium) — Copyright 2025 STARGA Inc.
- `cli_build.rs` (~648 tok, large) — Copyright 2025 STARGA Inc.
- `cli_eval.rs` (~434 tok, medium) — Copyright 2025 STARGA Inc.
- `cli_exec.rs` (~519 tok, large) — Copyright 2025 STARGA Inc.
- `cli_tensor.rs` (~401 tok, medium) — Copyright 2025 STARGA Inc.
- `closed_declaration_wiring.rs` (~1636 tok, huge) — Copyright 2025 STARGA Inc.
- `closure_capture_reject.rs` (~982 tok, large) — Copyright 2025 STARGA Inc.
- `closure_i64_capture.rs` (~1679 tok, huge) — Copyright 2025 STARGA Inc.
- `collection_ctor_run.rs` (~580 tok, large) — Copyright 2025 STARGA Inc.
- `collection_mutation_expr_position_run.rs` (~1097 tok, large) — Copyright 2025 STARGA Inc.
- `collection_owner_assignment_run.rs` (~3737 tok, huge) — Copyright 2025 STARGA Inc.
### `tests/common/`

- `gate.rs` (~4999 tok, huge) — Copyright 2025 STARGA Inc.
- `mod.rs` (~2625 tok, huge) — Copyright 2025 STARGA Inc.
- `xsi_gate.rs` (~4684 tok, huge) — Copyright 2025 STARGA Inc.
### `tests/`

- `compound_assign.rs` (~984 tok, large) — Copyright 2025 STARGA Inc.
- `cond_truthiness.rs` (~840 tok, large) — Copyright 2025 STARGA Inc.
- `conformance.rs` (~1493 tok, large) — Used only by the fail-closed test below, which is itself gated on the GPU
### `tests/conformance/cpu_baseline/`

- `autodiff_pairwise.runtime` (~1 tok, tiny) — 0
- `autodiff_seed.grad.ir` (~20 tok, tiny) — module {
- `autodiff_seed.ir` (~15 tok, tiny) — module {
- `autodiff_seed.mind` (~184 tok, small) — Core v1 conformance — the autodiff leg.
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

- `const_array_run.rs` (~834 tok, large) — Copyright 2025 STARGA Inc.
- `const_f64_array_run.rs` (~1097 tok, large) — Copyright 2025 STARGA Inc.
- `const_folding.rs` (~246 tok, medium) — Copyright 2025 STARGA Inc.
- `continue_in_match_arm_run.rs` (~1176 tok, large) — Copyright 2025 STARGA Inc.
- `conv2d_exec.rs` (~583 tok, large) — Copyright 2025 STARGA Inc.
- `conv2d_grad.rs` (~3194 tok, huge) — Copyright 2025 STARGA Inc.
- `conv2d_types.rs` (~360 tok, medium) — Copyright 2025 STARGA Inc.
- `cross_module.rs` (~1332 tok, large) — Copyright 2025 STARGA Inc.
- `cross_module_cdylib_compose.rs` (~4092 tok, huge) — Copyright 2025 STARGA Inc.
- `cross_module_enum_run.rs` (~1089 tok, large) — Copyright 2025 STARGA Inc.
- `cross_module_field_access_run.rs` (~4530 tok, huge) — Copyright 2025 STARGA Inc.
### `tests/cross_substrate_identity/`

- `README.md` (~1538 tok, huge) — cross_substrate_identity — the internal mind-bench reproducibility gate
### `tests/cross_substrate_identity/array-store-branch/`

- `manifest.toml` (~564 tok, large) — version = "1"
- `reference_hashes.toml` (~410 tok, medium) — avx2 = "8a50c515a5de1786b0102fecbb692054299acaa4878a9c2978c7b69cb647561f"
### `tests/cross_substrate_identity/array-store-loop/`

- `manifest.toml` (~636 tok, large) — version = "1"
- `reference_hashes.toml` (~445 tok, medium) — avx2 = "50b3efb2da5eb89764302f3781d390e9121c12cb8f87faaa69f5d0f46bf39f31"
### `tests/cross_substrate_identity/bimap-phf/`

- `manifest.toml` (~790 tok, large) — version = "1"
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

- `manifest.toml` (~558 tok, large) — version = "1"
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
### `tests/cross_substrate_identity/scalar-cast-conv-narrow/`

- `manifest.toml` (~1199 tok, large) — version = "1"
- `reference_hashes.toml` (~591 tok, large) — avx2 = "9e8e4278dbb52705a12c06c2e5ec59f8f994c7f0227617f5f0884cba0608976b"
### `tests/cross_substrate_identity/scalar-cast-conv/`

- `manifest.toml` (~1041 tok, large) — version = "1"
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

- `cross_substrate_receipts.rs` (~2546 tok, huge) — Copyright 2025 STARGA Inc.
- `cross_substrate_reference_consistency.rs` (~3141 tok, huge) — Copyright 2025 STARGA Inc.
- `crypto_vectors_driver.py` (~2653 tok, huge) — # Official-vector driver for std/aes_gcm.mind + std/hkdf.mind (pure-MIND crypto).
- `determinism_veto_control.rs` (~1490 tok, large) — Copyright 2025 STARGA Inc.
- `diagnostic_code_contract.rs` (~558 tok, large) — Copyright 2026 STARGA Inc.
- `diagnostics.rs` (~688 tok, large) — Copyright 2025 STARGA Inc.
- `diagnostics_parse.rs` (~1112 tok, large) — Copyright 2025 STARGA Inc.
- `digit_separator_run.rs` (~606 tok, large) — Copyright 2025 STARGA Inc.
- `dot_enum_variant_run.rs` (~756 tok, large) — Copyright 2025 STARGA Inc.
- `dot_variants.rs` (~284 tok, medium) — Copyright 2025 STARGA Inc.
- `duplicate_struct_declarations.rs` (~576 tok, large) — Copyright 2026 STARGA Inc.
- `ecdsa_p256_driver.py` (~2752 tok, huge) — # Ground-truth driver for std/ecdsa_p256.mind (pure-MIND ECDSA P-256/SHA-256
- `emit_ir_for_loop.rs` (~374 tok, medium) — Regression test for #4: lowering a `for` loop to IR (the path `mindc --emit-ir`
- `enum_match_collision_run.rs` (~987 tok, large) — Copyright 2025 STARGA Inc.
- `enum_match_run.rs` (~2310 tok, huge) — Copyright 2025 STARGA Inc.
- `enum_soundness.rs` (~1411 tok, large) — Copyright 2025 STARGA Inc.
- `enum_struct_variant_run.rs` (~1280 tok, large) — Copyright 2025 STARGA Inc.
- `evaluator_declared_width.rs` (~5277 tok, huge) — Copyright 2025 STARGA Inc.
- `exec_basic.rs` (~785 tok, large) — Copyright 2025 STARGA Inc.
- `expr_parser.rs` (~916 tok, large) — Copyright 2025 STARGA Inc.
- `extern_c_phase_a.rs` (~2686 tok, huge) — Copyright 2025 STARGA Inc.
- `extern_c_phase_b.rs` (~5974 tok, huge) — Copyright 2025 STARGA Inc.
- `extern_c_phase_c.rs` (~3442 tok, huge) — Copyright 2025 STARGA Inc.
- `extern_c_safety_tag_informational.rs` (~1427 tok, large) — Copyright 2025 STARGA Inc.
- `extern_narrow_ret_run.rs` (~1528 tok, huge) — Copyright 2025 STARGA Inc.
- `f3_bare_enum_collision_run.rs` (~1347 tok, large) — Copyright 2025 STARGA Inc.
- `f64_abi_negative_control.rs` (~1819 tok, huge) — Copyright 2025 STARGA Inc.
- `f64_activation_lowering.rs` (~972 tok, large) — Copyright 2025 STARGA Inc.
- `f64_call_arg_run.rs` (~977 tok, large) — Copyright 2025 STARGA Inc.
- `f64_literal_envelope.rs` (~1158 tok, large) — Each source literal is compiled via `mindc --emit-shared`; the exported
- `f64_loop_run.rs` (~1063 tok, large) — Copyright 2025 STARGA Inc.
- `fail_closed_capability_skip.rs` (~6323 tok, huge) — Copyright 2025 STARGA Inc.
- `fail_closed_capability_skip_env.rs` (~3997 tok, huge) — Copyright 2025 STARGA Inc.
- `fail_closed_capability_skip_stub_exec.rs` (~2022 tok, huge) — Copyright 2025 STARGA Inc.
- `fail_closed_cli_run.rs` (~1568 tok, huge) — Regression tests for the two fail-closed CLI guards that shipped WITHOUT one.
- `fail_open_skip_site_ratchet.rs` (~8472 tok, huge) — Copyright 2025 STARGA Inc.
- `ffi_header.rs` (~221 tok, medium) — Copyright 2025 STARGA Inc.
- `fixed_array_binding_lengths.rs` (~462 tok, medium) — Copyright 2026 STARGA Inc.
- `fixed_array_narrow_assignment_types.rs` (~376 tok, medium) — Copyright 2026 STARGA Inc.
- `fixed_array_return_lengths.rs` (~3080 tok, huge) — Copyright 2026 STARGA Inc.
- `fixed_array_struct_field_run.rs` (~6776 tok, huge) — Copyright 2026 STARGA Inc.
### `tests/fixtures/`

- `autodiff.mind` (~55 tok, small) — Minimal differentiable program for the --emit-grad-ir CLI test.
### `tests/fixtures/f64_abi_negative/`

- `README.md` (~146 tok, small) — f64 call-ABI negative control (paired with the #298 self-host f64 MLIR surface)
- `scale_neg.mind` (~23 tok, tiny)
- `scale_pos.mind` (~23 tok, tiny)
### `tests/fixtures/`

- `g2_expected_outcomes.tsv` (~5592 tok, huge) — # Per-fixture G2 capability ratchet; canonical source SHA-256 binds each case.
- `invalid.mind` (~6 tok, tiny)
- `invalid_broadcast.mind` (~17 tok, tiny)
### `tests/fixtures/nerve_numerics/`

- `e_nerve_001_bad.mind` (~89 tok, small) — NEGATIVE fixture for E_NERVE_001 (numerics.md@v1 §6 rule 1).
- `e_nerve_001_good.mind` (~60 tok, small) — POSITIVE fixture for E_NERVE_001: the whole reachable closure is annotated.
- `e_nerve_002_bad.mind` (~88 tok, small) — NEGATIVE fixture for E_NERVE_002 (numerics.md@v1 §6 rule 2).
- `e_nerve_002_good.mind` (~63 tok, small) — POSITIVE fixture for E_NERVE_002: the same fold, schedule declared.
- `e_nerve_003_bad.mind` (~78 tok, small) — NEGATIVE fixture for E_NERVE_003 (numerics.md@v1 §4, §6 rule 3).
- `e_nerve_003_good.mind` (~82 tok, small) — POSITIVE fixture for E_NERVE_003: the one sanctioned multiply owns the
- `e_nerve_004_bad.mind` (~67 tok, small) — NEGATIVE fixture for E_NERVE_004 (numerics.md@v1 §1, §6 rule 4).
- `e_nerve_004_good.mind` (~53 tok, small) — POSITIVE fixture for E_NERVE_004: inference internals stay in Q16.16.
- `e_nerve_005_bad.mind` (~109 tok, small) — NEGATIVE fixture for E_NERVE_005 (numerics.md@v1 §2, §6 rule 5).
- `e_nerve_005_good.mind` (~104 tok, small) — POSITIVE fixture for E_NERVE_005: the fold is pinned in source, so its
- `unarmed_all_violations.mind` (~175 tok, small) — INERTNESS fixture: every violation above, with the opt-in annotations
### `tests/fixtures/selfhost_policy/`

- `bimap_const_country_code.mind` (~18 tok, tiny)
- `bimap_const_distinct_escaped.mind` (~16 tok, tiny)
- `bimap_const_http_acronym.mind` (~16 tok, tiny)
- `bimap_enum_explicit_values.mind` (~17 tok, tiny)
- `bimap_enum_http_acronym.mind` (~13 tok, tiny)
- `bimap_enum_http_status.mind` (~13 tok, tiny)
- `nerve_attr_const.mind` (~12 tok, tiny)
- `nerve_attr_struct.mind` (~15 tok, tiny)
- `nerve_tree_associative_fixed.mind` (~50 tok, small) — fn tree_sum(x: i64) -> i64 {
- `nerve_user_defined_reduce_sum.mind` (~40 tok, tiny) — fn reduce_sum(x: i64) -> i64 {
- `nerve_while_counter_reset.mind` (~34 tok, tiny) — fn clear(x: i64) -> i64 {
- `nerve_while_nonreduction.mind` (~29 tok, tiny) — fn first(x: i64) -> i64 {
- `nerve_while_sequential.mind` (~42 tok, tiny) — fn value() -> i64 {
### `tests/fixtures/selfhost_policy/reject/`

- `bimap_const_duplicate_decoded_key.diagnostic` (~2 tok, tiny) — E2017
- `bimap_const_duplicate_decoded_key.mind` (~14 tok, tiny)
- `bimap_const_duplicate_decoded_value.diagnostic` (~2 tok, tiny) — E2018
- `bimap_const_duplicate_decoded_value.mind` (~14 tok, tiny)
- `bimap_const_duplicate_key.diagnostic` (~2 tok, tiny) — E2017
- `bimap_const_duplicate_key.mind` (~12 tok, tiny)
- `bimap_const_duplicate_value.diagnostic` (~2 tok, tiny) — E2018
- `bimap_const_duplicate_value.mind` (~13 tok, tiny)
- `bimap_const_empty.diagnostic` (~2 tok, tiny) — E2019
- `bimap_const_empty.mind` (~6 tok, tiny)
- `bimap_const_mixed_key_type.diagnostic` (~2 tok, tiny) — E2020
- `bimap_const_mixed_key_type.mind` (~12 tok, tiny)
- `bimap_const_negative_key.diagnostic` (~2 tok, tiny) — E2020
- `bimap_const_negative_key.mind` (~11 tok, tiny)
- `bimap_const_nonstring_value.diagnostic` (~2 tok, tiny) — E2020
- `bimap_const_nonstring_value.mind` (~8 tok, tiny)
- `bimap_const_numeric_duplicate_key.diagnostic` (~2 tok, tiny) — E2017
- `bimap_const_numeric_duplicate_key.mind` (~11 tok, tiny)
- `bimap_const_numeric_duplicate_value.diagnostic` (~2 tok, tiny) — E2018
- `bimap_const_numeric_duplicate_value.mind` (~13 tok, tiny)
- `bimap_const_trailing_expr.diagnostic` (~2 tok, tiny) — E2020
- `bimap_const_trailing_expr.mind` (~10 tok, tiny)
- `bimap_const_typed.diagnostic` (~2 tok, tiny) — E2020
- `bimap_const_typed.mind` (~15 tok, tiny)
- `bimap_const_u32_overflow.diagnostic` (~2 tok, tiny) — E2020
- `bimap_const_u32_overflow.mind` (~15 tok, tiny)
- `bimap_enum_duplicate_key.diagnostic` (~2 tok, tiny) — E1001
- `bimap_enum_duplicate_key.mind` (~7 tok, tiny)
- `bimap_enum_duplicate_value.diagnostic` (~2 tok, tiny) — E2018
- `bimap_enum_duplicate_value.mind` (~12 tok, tiny)
- `bimap_enum_nonstring_discriminant.diagnostic` (~2 tok, tiny) — E2020
- `bimap_enum_nonstring_discriminant.mind` (~10 tok, tiny)
- `bimap_enum_payload_variant.diagnostic` (~2 tok, tiny) — E2019
- `bimap_enum_payload_variant.mind` (~10 tok, tiny)
- `bimap_forged_generated_attr.diagnostic` (~2 tok, tiny) — E2022
- `bimap_forged_generated_attr.mind` (~28 tok, tiny) — fn currency_count() -> i64 {
- `bimap_generated_base_collision.diagnostic` (~2 tok, tiny) — E2021
- `bimap_generated_base_collision.mind` (~18 tok, tiny)
- `bimap_generated_fn_collision.diagnostic` (~2 tok, tiny) — E2021
- `bimap_generated_fn_collision.mind` (~19 tok, tiny) — fn code_count() -> i64 {
- `nerve_attr_on_use.diagnostic` (~2 tok, tiny) — E1001
- `nerve_attr_on_use.mind` (~11 tok, tiny)
- `nerve_bimap_generated_callee.diagnostic` (~3 tok, tiny) — E_NERVE_001
- `nerve_bimap_generated_callee.mind` (~31 tok, tiny) — fn main() -> i64 {
- `nerve_builtin_reduce_sum.diagnostic` (~3 tok, tiny) — E_NERVE_005
- `nerve_builtin_reduce_sum.mind` (~30 tok, tiny) — fn total(x: i64) -> i64 {
- `nerve_global_no_float_signature.diagnostic` (~3 tok, tiny) — E_NERVE_004
- `nerve_global_no_float_signature.mind` (~44 tok, tiny) — fn integer(x: i64) -> i64 {
- `nerve_host_libm.diagnostic` (~3 tok, tiny) — E_NERVE_004
- `nerve_host_libm.mind` (~26 tok, tiny) — fn value(x: i64) -> i64 {
- `nerve_invalid_reduction_strategy.diagnostic` (~3 tok, tiny) — E_NERVE_002
- `nerve_invalid_reduction_strategy.mind` (~45 tok, tiny) — fn bad_tag(x: i64) -> i64 {
- `nerve_signature_float.diagnostic` (~3 tok, tiny) — E_NERVE_004
- `nerve_signature_float.mind` (~25 tok, tiny) — fn value(a: f64) -> f64 {
- `nerve_while_branch_reduction.diagnostic` (~3 tok, tiny) — E_NERVE_002
- `nerve_while_branch_reduction.mind` (~47 tok, tiny) — fn value(x: i64) -> i64 {
- `nerve_while_reduction.diagnostic` (~3 tok, tiny) — E_NERVE_002
- `nerve_while_reduction.mind` (~33 tok, tiny) — fn value() -> i64 {
### `tests/fixtures/`

- `simple.mind` (~3 tok, tiny)
- `test_phase_b_all_pass.mind` (~67 tok, small) — RFC 0008 Phase B test fixture — both tests pass.
- `test_phase_b_one_fail.mind` (~80 tok, small) — RFC 0008 Phase B test fixture — one pass, one fail.
### `tests/`

- `fmt_comment_placement.rs` (~2772 tok, huge) — Copyright 2025 STARGA Inc.
- `fmt_idempotence.rs` (~4886 tok, huge) — Copyright 2025 STARGA Inc.
- `fmt_ir_preservation.rs` (~3956 tok, huge) — Copyright 2025 STARGA Inc.
- `fmt_module_block_item_preserved.rs` (~926 tok, large) — Regression: `mindc fmt` must not drop item declarations nested inside a
- `fmt_stdlib_stability.rs` (~2754 tok, huge) — Copyright 2025 STARGA Inc.
- `fn_value_call_reject.rs` (~761 tok, large) — Copyright 2025 STARGA Inc.
- `for_continue_advances_run.rs` (~1504 tok, huge) — Copyright 2025 STARGA Inc. Licensed under the Apache License, Version 2.0.
- `for_continue_step_injection.rs` (~1306 tok, large) — Copyright 2025 STARGA Inc. Licensed under the Apache License, Version 2.0.
- `for_each_run.rs` (~805 tok, large) — Copyright 2025 STARGA Inc.
- `for_hygiene_run.rs` (~1396 tok, large) — Copyright 2025 STARGA Inc. Licensed under the Apache License, Version 2.0.
- `g2_differential_mlir.rs` (~10566 tok, huge) — Copyright 2025 STARGA Inc.
- `gate_assert_count_contract_test.py` (~7131 tok, huge) — Gate test: the assertion count must be EVIDENCE, never a printed integer.
- `gather_preview.rs` (~288 tok, medium) — Copyright 2025 STARGA Inc.
- `generics_lowering.rs` (~1384 tok, large) — Copyright 2026 STARGA Inc. Licensed under the Apache License, Version 2.0.
- `genref_phase_jb.rs` (~3790 tok, huge) — Copyright 2025 STARGA Inc.
- `grad_wrt_resolve.rs` (~730 tok, large) — Copyright 2025 STARGA Inc.
- `harness_portability.rs` (~1260 tok, large) — Copyright 2025 STARGA Inc.
- `harness_scratch_isolation.rs` (~3404 tok, huge) — Copyright 2025 STARGA Inc.
- `hpack_driver.py` (~3027 tok, huge) — # Official-vector driver for std/hpack.mind (pure-MIND HPACK decoding,
- `http2_frame_driver.py` (~4195 tok, huge) — # Reference-vector driver for std/http2_frame.mind (pure-MIND HTTP/2 framing,
- `if_expr.rs` (~429 tok, medium) — Copyright 2025 STARGA Inc.
- `if_merge_shadow_318.rs` (~1444 tok, large) — Copyright 2025 STARGA Inc.
- `index_slice_grad.rs` (~289 tok, medium) — Copyright 2025 STARGA Inc.
- `index_slice_preview.rs` (~376 tok, medium) — Copyright 2025 STARGA Inc.
- `index_slice_types.rs` (~250 tok, medium) — Copyright 2025 STARGA Inc.
- `int_determinism.rs` (~1234 tok, large) — Copyright 2025 STARGA Inc.
- `int_suffix_literal.rs` (~883 tok, large) — Copyright 2025 STARGA Inc.
- `intra_module_call_arity.rs` (~3085 tok, huge) — Copyright 2025 STARGA Inc.
- `invariant_block_run.rs` (~675 tok, large) — Copyright 2025 STARGA Inc.
- `invariant_check_run.rs` (~650 tok, large) — Copyright 2025 STARGA Inc.
- `ir_core.rs` (~1697 tok, huge) — Ensure the unused const is kept alive in the SSA namespace but removed from code.
- `ir_load_save.rs` (~1257 tok, large) — Copyright 2025 STARGA Inc.
- `ir_lower.rs` (~1894 tok, huge) — Copyright 2025 STARGA Inc.
- `ir_negative_literals.rs` (~1741 tok, huge) — Copyright 2025 STARGA Inc.
- `ir_stub.rs` (~219 tok, medium) — Copyright 2025 STARGA Inc.
### `tests/ir_verification/`

- `ssa_single_assignment.mind` (~46 tok, tiny) — IR verification test: SSA property validation
- `undefined_operand.mind` (~62 tok, small) — IR verification test: Undefined operand detection
### `tests/`

- `issue_201_202_unary_not_const_ctx.rs` (~1132 tok, large) — Copyright 2025 STARGA Inc.
- `issue_235_single_file_import.rs` (~2389 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `issue_246_bitnot_operator.rs` (~571 tok, large) — Copyright 2025 STARGA Inc.
- `issue_246_tensor_named_dtype.rs` (~525 tok, large) — Copyright 2025 STARGA Inc.
- `issue_263_method_chain_newline.rs` (~869 tok, large) — Copyright 2025 STARGA Inc.
- `issue_263_use_path_import.rs` (~808 tok, large) — Copyright 2025 STARGA Inc.
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
- `loop_run.rs` (~705 tok, large) — Copyright 2025 STARGA Inc.
- `loud_fail_non_i64.rs` (~852 tok, large) — Release-readiness P1.1 — the runnable-artifact ABI gate.
- `lowering_refusal_diagnostics_run.rs` (~6425 tok, huge) — Copyright 2026 STARGA Inc.
- `manifest_pin_enforcement.rs` (~3112 tok, huge) — The `[mind]` table (`mindc-min` / `mindc-max` / `ir-format` /
- `map_get_inference_run.rs` (~668 tok, large) — Copyright 2025 STARGA Inc.
- `map_runtime_run.rs` (~797 tok, large) — Copyright 2025 STARGA Inc.
- `map_surface_run.rs` (~903 tok, large) — Copyright 2025 STARGA Inc.
- `match_arm_stmt_run.rs` (~800 tok, large) — Copyright 2025 STARGA Inc.
- `match_fallback_fail_closed.rs` (~1765 tok, huge) — Copyright 2025 STARGA Inc.
- `match_scrutinee_once.rs` (~1476 tok, large) — Copyright 2025 STARGA Inc.
- `merge_kind_order_symmetry.rs` (~993 tok, large) — A merge's signedness must not depend on ARM ORDER.
- `method_call.rs` (~397 tok, medium) — Copyright 2025 STARGA Inc.
- `mic3_array_store_roundtrip.rs` (~1061 tok, large) — Copyright 2026 STARGA Inc.
- `mic3_break_continue_string_roundtrip.rs` (~1685 tok, huge) — Copyright 2026 STARGA Inc.
- `mic3_cli_emit.rs` (~2324 tok, huge) — Copyright 2025 STARGA Inc.
- `mic3_const_dense_tensor_roundtrip.rs` (~874 tok, large) — Copyright 2026 STARGA Inc.
- `mic3_parser_dos.rs` (~1891 tok, huge) — Copyright 2025 STARGA Inc.
- `micb_dos_reject.rs` (~2341 tok, huge) — Copyright 2025 STARGA Inc.
- `mindc.rs` (~2105 tok, huge) — Copyright 2025 STARGA Inc.
- `mindc_artifact_name.rs` (~4118 tok, huge) — Copyright 2025 STARGA Inc.
- `mindc_build_phase_a.rs` (~5557 tok, huge) — Copyright 2025 STARGA Inc.
- `mindc_cache_build_inputs.rs` (~2109 tok, huge) — Copyright 2025 STARGA Inc.
- `mindc_cache_phase_f.rs` (~6283 tok, huge) — Copyright 2025 STARGA Inc.
- `mindc_deps_phase_de.rs` (~6823 tok, huge) — Copyright 2025 STARGA Inc.
- `mindc_doc_phase1.rs` (~3004 tok, huge) — Copyright 2025 STARGA Inc.
- `mindc_inspect.rs` (~1135 tok, large) — Integration test for `mindc inspect` — the mic@3 artifact decoder/differ.
- `mindc_project_lock.rs` (~1069 tok, large) — Copyright 2026 STARGA Inc.
- `mindc_symlinked_project_identity.rs` (~2222 tok, huge) — Copyright 2026 STARGA Inc.
- `mindc_test_evaluator_issues.rs` (~6043 tok, huge) — Copyright 2025 STARGA Inc.
- `mindc_test_imports.rs` (~5926 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `mindc_test_nested_modules.rs` (~1901 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `mindc_test_phase_b.rs` (~3574 tok, huge) — Copyright 2025 STARGA Inc.
- `mindc_workspace_phase_c.rs` (~4166 tok, huge) — Copyright 2025 STARGA Inc.
### `tests/mindcraft/`

- `STABILITY_SKIP_LIST.md` (~408 tok, medium) — Formatter Stability Skip List
### `tests/mindcraft/check/`

- `clean.mind` (~11 tok, tiny) — fn add(a: i64, b: i64) -> i64 {
- `drifted.mind` (~12 tok, tiny) — fn add(a: i64,  b: i64) -> i64 {
- `ignored.mind` (~9 tok, tiny) — fn ignored_fn() -> i64 {
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
### `tests/mindcraft/lint/naming_convention/`

- `negative.mind` (~61 tok, small) — Negative fixture: all names follow canonical conventions.
- `positive_bad_const.mind` (~39 tok, tiny) — Positive fixture: const name violates SCREAMING_SNAKE_CASE.
- `positive_bad_fn.mind` (~37 tok, tiny) — Positive fixture: function name violates lower_snake_case.
- `positive_bad_struct.mind` (~39 tok, tiny) — Positive fixture: struct name violates UpperCamelCase.
### `tests/mindcraft/lint/q16_overflow/`

- `edge_constant.mind` (~54 tok, small) — Edge case: i32 * literal constant still triggers if no >>16 shift.
- `negative.mind` (~67 tok, small) — Negative fixture: proper Q16.16 multiply with >>16 narrowing.
- `positive.mind` (~70 tok, small) — Positive fixture: bare i32 * i32 without >>16 narrowing.
### `tests/mindcraft/lint/shadowing/`

- `negative.mind` (~34 tok, tiny) — Negative fixture: two different names — no shadowing.
- `positive.mind` (~53 tok, small) — Positive fixture: two `let x` bindings in the same function body.
### `tests/mindcraft/lint/`

- `trailing_ws_clean.mind` (~10 tok, tiny) — fn foo() -> i64 {
- `trailing_ws_dirty.mind` (~11 tok, tiny) — fn foo() -> i64 {
### `tests/mindcraft/lint/unused_import/`

- `negative.mind` (~53 tok, small) — Negative fixture: `use std.vec` is declared AND the `vec` identifier
- `positive.mind` (~46 tok, tiny) — Positive fixture: `use std.vec` is declared but no symbol from vec
### `tests/`

- `mindcraft_check_cli.rs` (~3370 tok, huge) — Copyright 2025 STARGA Inc.
- `mindcraft_check_fix.rs` (~1667 tok, huge) — Copyright 2025 STARGA Inc.
- `mindcraft_check_lsp_reporter.rs` (~1968 tok, huge) — Copyright 2025 STARGA Inc.
- `mindcraft_fmt_cli.rs` (~2925 tok, huge) — Copyright 2025 STARGA Inc.
- `mindcraft_fmt_fix.rs` (~1445 tok, large) — Copyright 2025 STARGA Inc.
- `mindcraft_fmt_fixtures.rs` (~1300 tok, large) — Copyright 2025 STARGA Inc.
- `mindcraft_lint_naming_convention.rs` (~1112 tok, large) — Copyright 2025 STARGA Inc.
- `mindcraft_lint_q16_overflow.rs` (~867 tok, large) — Copyright 2025 STARGA Inc.
- `mindcraft_lint_shadowing.rs` (~980 tok, large) — Copyright 2025 STARGA Inc.
- `mindcraft_lint_unused_import.rs` (~708 tok, large) — Copyright 2025 STARGA Inc.
- `mindcraft_lint_vec_check.rs` (~549 tok, large) — Copyright 2025 STARGA Inc.
### `tests/mindfuzz_corpus_floor/`

- `mod.rs` (~855 tok, large) — Corpus-size floor for the differential determinism fuzzer.
### `tests/`

- `mindfuzz_cross_substrate.rs` (~16552 tok, huge) — Copyright 2025 STARGA Inc.
### `tests/mindfuzz_cross_substrate/known_environmental/`

- `README.md` (~314 tok, medium) — Known-environmental fuzzer artifacts
- `seed_deadbeef_prog000_e1001_bitwise.mind` (~242 tok, medium) — NOT A DIVERGENCE — misfiled build-capability refusal (issue #72 fuzzer).
### `tests/mindfuzz_cross_substrate/reproducers/`

- `fuzz_oracle_failure_seed_deadbeef_prog000.mind` (~218 tok, medium) — ORACLE FAILURE — a stage did not complete; NOT a proven cross-substrate divergence (issue #72 fuzzer)
- `fuzz_oracle_failure_seed_deadbeef_prog006.mind` (~157 tok, small) — DIVERGENCE REPRODUCER (issue #72 fuzzer)
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
- `mlir_build.rs` (~1544 tok, huge) — Copyright 2025 STARGA Inc.
- `mlir_exec.rs` (~779 tok, large) — Copyright 2025 STARGA Inc.
- `mlir_export.rs` (~582 tok, large) — Copyright 2025 STARGA Inc.
- `mlir_export_indexing.rs` (~419 tok, medium) — Copyright 2025 STARGA Inc.
- `mlir_export_linalg.rs` (~873 tok, large) — Copyright 2025 STARGA Inc.
- `mlir_export_reductions.rs` (~1710 tok, huge) — Copyright 2025 STARGA Inc.
- `mlir_export_shape.rs` (~353 tok, medium) — Copyright 2025 STARGA Inc.
- `mlir_file_and_lower.rs` (~611 tok, large) — Copyright 2025 STARGA Inc.
- `mlir_gpu.rs` (~314 tok, medium) — Copyright 2025 STARGA Inc.
- `mlir_jit.rs` (~285 tok, medium) — Copyright 2025 STARGA Inc.
- `mlir_lowering.rs` (~1490 tok, large)
- `mlir_opt.rs` (~474 tok, medium) — Copyright 2025 STARGA Inc.
- `mlkem768_driver.py` (~1699 tok, huge) — # Reference-vector driver for std/mlkem768.mind (pure-MIND ML-KEM-768,
- `module_const_run.rs` (~729 tok, large) — Copyright 2025 STARGA Inc.
- `module_decl_run.rs` (~597 tok, large) — Copyright 2025 STARGA Inc.
- `module_enum_match_run.rs` (~984 tok, large) — Copyright 2025 STARGA Inc.
- `module_non_fn_call_reject.rs` (~1215 tok, large) — Copyright 2025 STARGA Inc.
- `module_size_ratchet.rs` (~3984 tok, huge) — Copyright 2025 STARGA Inc.
- `multimodule_determinism_run.rs` (~4275 tok, huge) — Copyright 2025 STARGA Inc.
- `narrow_call_abi.rs` (~1344 tok, large) — Copyright 2025 STARGA Inc.
- `narrow_local_mask_run.rs` (~879 tok, large) — Copyright 2025 STARGA Inc.
- `narrow_locals_leak_run.rs` (~1743 tok, huge) — Copyright 2025 STARGA Inc.
- `narrow_reassign_mask_run.rs` (~1385 tok, large) — Copyright 2025 STARGA Inc.
- `narrow_reassign_run.rs` (~687 tok, large) — Copyright 2026 STARGA Inc.
- `narrow_sig_abi_run.rs` (~1163 tok, large) — Copyright 2025 STARGA Inc.
- `narrow_signedness_batch2_run.rs` (~2236 tok, huge) — Copyright 2025 STARGA Inc.
- `narrow_signedness_batch_run.rs` (~1629 tok, huge) — Copyright 2025 STARGA Inc.
- `narrow_tuple_pr216_run.rs` (~1409 tok, large) — Copyright 2025 STARGA Inc.
- `narrow_unsigned_div_zero_run.rs` (~1247 tok, large) — Copyright 2025 STARGA Inc.
- `narrowing_check.rs` (~438 tok, medium) — Regression test for the silent i64->i32 narrowing miscompile found by MIND-Fuzz
### `tests/native_bridge_support/`

- `draining_native_compiler.rs` (~300 tok, medium) — Host-native compiler fixture for admission-only controls.
- `mod.rs` (~2274 tok, huge) — Shared harness for the native module-bridge integration controls.
### `tests/`

- `native_closure_admission.rs` (~3941 tok, huge) — Copyright 2025 STARGA Inc.
- `native_default_profile_imports.rs` (~3506 tok, huge) — Default-feature-profile controls for the native backend's import admission.
- `native_module_bridge_controls.rs` (~8330 tok, huge) — Native module-bridge controls: SOURCE RESOLUTION and ADMISSION.
- `native_module_bridge_visibility_controls.rs` (~3913 tok, huge) — Native module-bridge controls: VISIBILITY, EMIT KIND, CROSS-INVOCATION
- `native_opt_wiring.rs` (~1953 tok, huge) — Copyright 2025 STARGA Inc.
- `nerve_numerics_lint.rs` (~4079 tok, huge) — Copyright 2025 STARGA Inc.
- `nested_block_surface_run.rs` (~990 tok, large) — Copyright 2025 STARGA Inc.
- `nested_collection_run.rs` (~772 tok, large) — Copyright 2025 STARGA Inc.
- `nested_import_owner_resolution_run.rs` (~2417 tok, huge) — Copyright 2026 STARGA Inc.
- `nested_mut_thread_run.rs` (~1520 tok, huge) — Copyright 2025 STARGA Inc.
- `nested_project_import_scope_run.rs` (~1957 tok, huge) — Copyright 2026 STARGA Inc.
- `non_final_catch_all_match_run.rs` (~861 tok, large) — Copyright 2025 STARGA Inc. Licensed under the Apache License, Version 2.0.
- `ops_registry.rs` (~114 tok, small)
- `package_basic.rs` (~491 tok, medium) — Copyright 2025 STARGA Inc.
- `package_traversal.rs` (~905 tok, large) — Copyright 2025 STARGA Inc.
- `parse_match_and_ref.rs` (~3704 tok, huge) — Copyright 2025 STARGA Inc.
- `parse_phase10_surface.rs` (~4988 tok, huge) — Parse-target tests for Phase 10.5 / 10.6 surface-syntax acceptance.
- `parse_try_operator.rs` (~1194 tok, large) — Copyright 2025 STARGA Inc.
- `parser_trivia.rs` (~2706 tok, huge) — Copyright 2025 STARGA Inc.
- `parser_unsigned_i64_literals.rs` (~1544 tok, huge) — Copyright 2025 STARGA Inc.
- `pattern_guard_run.rs` (~899 tok, large) — Copyright 2025 STARGA Inc. Licensed under the Apache License, Version 2.0.
- `phase_g_keystone_bootstrap.rs` (~7667 tok, huge) — Copyright 2025 STARGA Inc.
- `pipeline.rs` (~1476 tok, large) — Copyright 2025 STARGA Inc.
- `pqc_hybrid_cli_signing.rs` (~7885 tok, huge) — End-to-end CLI controls for post-quantum hybrid artifact signing.
- `project_function_resolution.rs` (~1475 tok, large) — Copyright 2026 STARGA Inc.
- `public_cpu_native_link.rs` (~1528 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `qualified_enum_run.rs` (~3922 tok, huge) — Copyright 2025 STARGA Inc.
- `reap_threshold.rs` (~2108 tok, huge) — Copyright 2025 STARGA Inc.
- `reductions_grad.rs` (~390 tok, medium) — Copyright 2025 STARGA Inc.
- `reductions_preview.rs` (~390 tok, medium) — Copyright 2025 STARGA Inc.
- `regalloc_dtk_parity.rs` (~3134 tok, huge) — DTK slice 1 (#254) — parity consumer for src/opt/regalloc_dtk.rs.
- `region_phase_ja.rs` (~4263 tok, huge) — Copyright 2025 STARGA Inc.
- `relu_exec.rs` (~434 tok, medium) — Copyright 2025 STARGA Inc.
- `relu_preview.rs` (~279 tok, medium) — Copyright 2025 STARGA Inc.
- `repl_basic.rs` (~468 tok, medium) — Copyright 2025 STARGA Inc.
- `reshape_runtime_dim_273.rs` (~797 tok, large) — Copyright 2025 STARGA Inc.
- `resolve_fn_body.rs` (~1450 tok, large) — Copyright 2025 STARGA Inc.
- `result_option_prelude_run.rs` (~859 tok, large) — Copyright 2025 STARGA Inc.
- `return_cond_type_reject.rs` (~8751 tok, huge) — Copyright 2025 STARGA Inc.
- `return_flow_tree_eval_run.rs` (~1466 tok, large) — Copyright 2025 STARGA Inc.
- `rfc0012_attribute_syntax.rs` (~1182 tok, large) — Copyright 2025 STARGA Inc.
- `rfc0012_phase_a_shape_types.rs` (~6711 tok, huge) — Copyright 2025 STARGA Inc.
- `rfc0012_phase_b_operators.rs` (~4737 tok, huge) — Copyright 2025 STARGA Inc.
- `rfc0012_phase_c_annotations.rs` (~2405 tok, huge) — Copyright 2025 STARGA Inc.
- `roofline_peak.rs` (~697 tok, large) — Correctness proof for the CPUID-gated roofline denominator (B4).
- `rsa_pss_driver.py` (~2692 tok, huge) — # Ground-truth driver for std/rsa_pss.mind (pure-MIND RSASSA-PSS-VERIFY,
### `tests/runtime/`

- `elementwise_add.mind` (~68 tok, small) — Runtime test: Element-wise addition execution
- `reduction_sum.mind` (~67 tok, small) — Runtime test: Reduction sum operation
### `tests/`

- `scalar_cast_call_run.rs` (~673 tok, large) — Copyright 2025 STARGA Inc.
- `scalar_cast_unsigned_narrow_run.rs` (~1047 tok, large) — Copyright 2025 STARGA Inc.
- `self_host_loop_advance_contract_test.py` (~5416 tok, huge) — Focused admission, mode, oracle, and receipt controls for --advance."""
- `self_host_loop_advance_publication_test.py` (~5173 tok, huge) — Focused staging, publication, failure, and dispatch controls for --advance."""
- `self_host_loop_advance_test.py` (~335 tok, medium) — Runner for the self-host LOOP advancement harness controls.
### `tests/selfhost_gaps/`

- `GAPS.md` (~5947 tok, huge) — Self-host nfn driver — gap inventory (fuzz-discovered)
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
- `attr_item_dropped_1.mind` (~24 tok, tiny) — fn g(a: i64) -> i64 {
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
- `cast_i32_1.mind` (~12 tok, tiny) — fn f(a: i64) -> i32 {
- `cast_i64_neg_1.mind` (~12 tok, tiny) — fn f(a: i64) -> i64 {
- `cast_i64_pair_1.mind` (~17 tok, tiny) — fn f(a: i64, b: i64) -> i64 {
- `cast_u8_1.mind` (~11 tok, tiny) — fn f(a: i64) -> u8 {
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
- `for_nonliteral_end_1.mind` (~27 tok, tiny) — fn f(n: i64) -> i64 {
- `for_range_bound10_1.mind` (~24 tok, tiny) — fn f(a: i64) -> i64 {
- `for_range_continue_1.mind` (~35 tok, tiny) — fn f() -> i64 {
- `for_range_multistmt_1.mind` (~29 tok, tiny) — fn f(a: i64) -> i64 {
- `for_range_simple_1.mind` (~24 tok, tiny) — fn f(a: i64) -> i64 {
- `for_range_zero_1.mind` (~24 tok, tiny) — fn f(a: i64) -> i64 {
- `for_shadowed_bound_1.mind` (~16 tok, tiny) — fn f(i:i64)->i64 {
- `for_shadowed_outer_let_1.mind` (~12 tok, tiny) — fn f()->i64 {
- `for_shadowed_sequential_1.mind` (~22 tok, tiny) — fn f(i:i64)->i64 {
- `for_shadowed_trivia_1.mind` (~117 tok, small) — fn outer_tab()->i64 {
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
- `let_shadow_param_1.mind` (~16 tok, tiny) — fn f(n: i64) -> i64 {
- `match_enum_payload_1.mind` (~29 tok, tiny) — fn f(e: E) -> i64 {
- `match_enum_payload_reordered_1.mind` (~18 tok, tiny) — fn f(e:E)->i64 {
- `matchcall-scrutinee_0arg.mind` (~24 tok, tiny) — fn g() -> i64 {
- `matchcall-scrutinee_1arg.mind` (~33 tok, tiny) — fn g(a: i64) -> i64 {
- `matchcall-scrutinee_arith.mind` (~21 tok, tiny) — fn f(a: i64) -> i64 {
- `matchcall_letinit_1.mind` (~34 tok, tiny) — fn g(a: i64) -> i64 {
- `mixed-prefix_1.mind` (~28 tok, tiny)
- `mixed-prefix_10.mind` (~32 tok, tiny)
- `mixed-prefix_11.mind` (~27 tok, tiny)
- `mixed-prefix_12.mind` (~40 tok, tiny)
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

- `addr_of_local_1.mind` (~15 tok, tiny) — fn f(a: i64) -> i64 {
- `array_lit_in_if_const_1.mind` (~19 tok, tiny)
- `array_lit_in_if_var_1.mind` (~20 tok, tiny)
- `bin_lit_return_1.mind` (~10 tok, tiny) — fn f() -> i64 {
- `field_assign_1.mind` (~28 tok, tiny) — fn f(a: i64) -> i64 {
- `hex_lit_mask_1.mind` (~12 tok, tiny) — fn f(a: i64) -> i64 {
- `hex_lit_return_1.mind` (~9 tok, tiny) — fn f() -> i64 {
- `index_assign_1.mind` (~23 tok, tiny) — fn f(a: i64) -> i64 {
- `index_call_binop_1.mind` (~28 tok, tiny) — fn g(a: i64) -> i64 { a + 1 }
- `index_call_in_if_1.mind` (~27 tok, tiny) — fn g(a: i64) -> i64 { a + 1 }
- `index_call_nested_1.mind` (~35 tok, tiny) — fn h(a: i64) -> i64 { a + 1 }
- `method_call_1.mind` (~36 tok, tiny) — fn p_get(s: P) -> i64 {
- `neg_float_1.mind` (~9 tok, tiny)
- `neg_float_in_if_1.mind` (~15 tok, tiny)
- `neg_index_in_if_1.mind` (~20 tok, tiny)
- `neg_index_trailing_1.mind` (~11 tok, tiny)
- `oct_lit_return_1.mind` (~9 tok, tiny) — fn f() -> i64 {
- `unknown_item_attr_1.mind` (~12 tok, tiny) — fn main() -> i64 {
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
- `struct-lit-field-recv_1.mind` (~23 tok, tiny)
- `struct-lit-field-recv_2.mind` (~23 tok, tiny)
- `struct-lit-field-recv_3.mind` (~25 tok, tiny)
- `struct-lit_1.mind` (~63 tok, small)
- `struct-lit_2.mind` (~56 tok, small)
- `struct-lit_3.mind` (~60 tok, small)
- `two_scalar_prior.mind` (~38 tok, tiny)
- `use_item_1.mind` (~13 tok, tiny) — fn f(a: i64) -> i64 {
- `value-ifexpr_1.mind` (~98 tok, small) — MISMATCH: a `let`-block in a NESTED (else-if) branch of a value if-expr.
- `value-ifexpr_2.mind` (~79 tok, small) — MISMATCH: same-named `let` in two SIBLING branches of a value if-expr.
- `value-ifexpr_3.mind` (~71 tok, small) — MISMATCH: a `let` inside a NESTED if-expr that sits in the THEN-side of an
- `value-ifexpr_4.mind` (~75 tok, small) — MISMATCH: let-block then-side whose trailing value is a nested if-expr that
- `value-ifexpr_5.mind` (~83 tok, small) — FAIL_CLOSED (in-subset): value if-expr whose else-branch is
- `value-ifexpr_6.mind` (~77 tok, small) — FAIL_CLOSED (in-subset): `let outer; if .. { use outer } else { use outer }`
- `value-ifexpr_7.mind` (~78 tok, small) — FAIL_CLOSED (in-subset): struct-lit construction as a value if-expr branch.
- `value-ifexpr_8.mind` (~64 tok, small) — FAIL_CLOSED (in-subset): field-read `recv.field` as a value if-expr branch.
- `while_counter_1.mind` (~27 tok, tiny) — fn f(n: i64) -> i64 {
### `tests/`

- `set_surface_run.rs` (~886 tok, large) — Copyright 2025 STARGA Inc.
- `sha256_smoke.rs` (~1586 tok, huge) — Copyright 2025 STARGA Inc.
- `sha512_smoke.rs` (~1721 tok, huge) — Copyright 2025 STARGA Inc.
- `shape_integration.rs` (~416 tok, medium)
- `shape_ops_preview.rs` (~302 tok, medium) — Copyright 2025 STARGA Inc.
- `shapes.rs` (~1132 tok, large) — Copyright 2025 STARGA Inc.
### `tests/shapes/`

- `broadcast_compatible.mind` (~77 tok, small) — Shape test: Compatible broadcasting
- `broadcast_incompatible.mind` (~76 tok, small) — Shape test: Incompatible broadcasting
- `matmul_shapes.mind` (~107 tok, small) — Shape test: MatMul shape inference
### `tests/`

- `shapes_engine.rs` (~699 tok, large) — Rank-0 scalar represented as an empty shape.
### `tests/skip_shape_scan/`

- `controls.rs` (~4228 tok, huge) — Copyright 2025 STARGA Inc.
- `mod.rs` (~6173 tok, huge) — Copyright 2025 STARGA Inc.
### `tests/`

- `slice_call_abi_run.rs` (~5559 tok, huge) — Copyright 2025 STARGA Inc.
- `smoke.rs` (~261 tok, medium) — Copyright 2025 STARGA Inc.
- `source_confinement.rs` (~2038 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `sparse_tensor_types.rs` (~1997 tok, huge) — Copyright 2025 STARGA Inc.
- `statement_mutation_run.rs` (~911 tok, large) — Copyright 2025 STARGA Inc.
- `std_import_standalone_run.rs` (~1102 tok, large) — Copyright 2025 STARGA Inc.
- `std_llvm_bindings_smoke.rs` (~2673 tok, huge) — Copyright 2025 STARGA Inc.
- `std_mlir_bindings_smoke.rs` (~4872 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_arena.rs` (~1351 tok, large) — Copyright 2025 STARGA Inc.
- `std_surface_array_literals.rs` (~3383 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_async.rs` (~4342 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_bitwise_binops.rs` (~2409 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_bool_return.rs` (~1279 tok, large) — Copyright 2026 STARGA Inc.
- `std_surface_break_continue.rs` (~1335 tok, large) — Copyright 2026 STARGA Inc.
- `std_surface_call_lowering.rs` (~1236 tok, large) — Copyright 2025 STARGA Inc.
- `std_surface_cdylib_link.rs` (~2900 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_cli.rs` (~1007 tok, large) — Copyright 2025 STARGA Inc.
- `std_surface_cli_equals_form.rs` (~874 tok, large) — Copyright 2025 STARGA Inc.
- `std_surface_cli_subcommand.rs` (~799 tok, large) — Copyright 2025 STARGA Inc.
- `std_surface_field_access.rs` (~3162 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_field_access_step2.rs` (~3856 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_fndef_lowering.rs` (~1522 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_http.rs` (~4060 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_i32_intrinsics.rs` (~824 tok, large) — Copyright 2025 STARGA Inc.
- `std_surface_if_statement.rs` (~3458 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_intrinsics.rs` (~3439 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_io_ansi.rs` (~793 tok, large) — Copyright 2025 STARGA Inc.
- `std_surface_io_canon.rs` (~4912 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_io_module.rs` (~1544 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_iouring.rs` (~3678 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_json.rs` (~7071 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_logical_ops.rs` (~1207 tok, large) — Copyright 2026 STARGA Inc.
- `std_surface_map_module.rs` (~2020 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_method_call.rs` (~2963 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_net_fs_process.rs` (~8341 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_phase_c_stdlib_bundle.rs` (~1466 tok, large) — Copyright 2025 STARGA Inc.
- `std_surface_phase_d_env_override.rs` (~1597 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_promotion_compose.rs` (~5656 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_reactor.rs` (~1378 tok, large) — Copyright 2025 STARGA Inc.
- `std_surface_regex.rs` (~5298 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_ring.rs` (~1318 tok, large) — Copyright 2025 STARGA Inc.
- `std_surface_self_emit_shared.rs` (~1808 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_string_itoa.rs` (~934 tok, large) — Copyright 2025 STARGA Inc.
- `std_surface_string_module.rs` (~2288 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_string_push_str.rs` (~864 tok, large) — Copyright 2025 STARGA Inc.
- `std_surface_struct_lowering.rs` (~2736 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_toml.rs` (~4164 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_tui.rs` (~2261 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_use_import.rs` (~1757 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_use_import_phase_b.rs` (~2303 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_vec_module.rs` (~1809 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_vec_zeroed.rs` (~1097 tok, large) — Copyright 2025 STARGA Inc.
- `std_surface_while_statement.rs` (~3251 tok, huge) — Copyright 2025 STARGA Inc.
- `stdlib_tensor.rs` (~256 tok, medium) — Copyright 2025 STARGA Inc.
- `stmt_keyword_recognizer.rs` (~2527 tok, huge) — Copyright 2025 STARGA Inc.
- `stride_gather_grad.rs` (~312 tok, medium) — Copyright 2025 STARGA Inc.
- `stride_preview.rs` (~279 tok, medium) — Copyright 2025 STARGA Inc.
- `stride_types.rs` (~244 tok, medium) — Copyright 2025 STARGA Inc.
- `string_eq_len_run.rs` (~2598 tok, huge) — Copyright 2025 STARGA Inc.
- `string_escape_decode_run.rs` (~1159 tok, large) — Copyright 2025 STARGA Inc.
- `string_escape_fail_closed.rs` (~959 tok, large) — Copyright 2025 STARGA Inc.
- `string_escape_parse.rs` (~483 tok, medium) — Copyright 2025 STARGA Inc.
- `string_from_bytes_run.rs` (~664 tok, large) — Copyright 2025 STARGA Inc.
- `string_pattern_escape_decode.rs` (~1069 tok, large) — Copyright 2025 STARGA Inc. Licensed under the Apache License, Version 2.0.
- `string_runtime_shim_run.rs` (~1027 tok, large) — Copyright 2025 STARGA Inc.
- `string_split_run.rs` (~1826 tok, huge) — Copyright 2025 STARGA Inc.
- `struct_array_field_run.rs` (~802 tok, large) — Copyright 2025 STARGA Inc.
- `struct_field_collection_run.rs` (~808 tok, large) — Copyright 2025 STARGA Inc.
- `struct_field_in_loop_run.rs` (~1043 tok, large) — Copyright 2025 STARGA Inc.
- `struct_literal_rebind.rs` (~1607 tok, huge) — Copyright 2026 STARGA Inc.
- `struct_narrow_field.rs` (~771 tok, large) — Copyright 2025 STARGA Inc.
- `substrate_nonentry_import_link.rs` (~1449 tok, large) — Copyright 2025 STARGA Inc.
### `tests/support/`

- `g2_capability_ratchet.rs` (~783 tok, large) — Copyright 2026 STARGA Inc.
- `g2_compile_result.rs` (~364 tok, medium) — Copyright 2026 STARGA Inc.
- `g2_policy_rejections.rs` (~755 tok, large) — Copyright 2026 STARGA Inc.
- `pqc_hybrid_cli_support.rs` (~1429 tok, large) — Copyright 2026 STARGA Inc.
- `v04_mirror_body_vectors.rs` (~4124 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `v04_mirror_descriptor_vectors.rs` (~2649 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `v04_mirror_intrinsic_vectors.rs` (~1359 tok, large) — Copyright 2025-2026 STARGA Inc.
- `v04_mirror_modules.rs` (~1936 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `v04_mirror_oracle.rs` (~6008 tok, huge) — Copyright 2025-2026 STARGA Inc.
### `tests/`

- `target_cerebras.rs` (~340 tok, medium) — Cerebras backend target — first-class surface tests.
- `tensor_broadcast.rs` (~995 tok, large) — Copyright 2025 STARGA Inc.
- `tensor_buffers.rs` (~517 tok, large) — Copyright 2025 STARGA Inc.
- `tensor_eval.rs` (~457 tok, medium) — Copyright 2025 STARGA Inc.
- `tensor_param_2d_run.rs` (~954 tok, large) — Copyright 2025 STARGA Inc.
- `tensor_param_descriptor_abi_run.rs` (~2466 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `tensor_param_fail_loud_run.rs` (~3296 tok, huge) — Copyright 2025 STARGA Inc.
- `tensor_stdlib.rs` (~549 tok, large) — Copyright 2025 STARGA Inc.
- `tensor_symbolic.rs` (~542 tok, large) — Copyright 2025 STARGA Inc.
- `test_harness_root_cause_error.rs` (~758 tok, large) — Copyright 2025 STARGA Inc.
- `tls13_finished_driver.py` (~3085 tok, huge) — # Official-vector driver for std/tls13_finished.mind (pure-MIND TLS 1.3
- `tls13_handshake_driver.py` (~6662 tok, huge) — # Official-vector driver for std/tls13_handshake.mind (pure-MIND TLS 1.3
- `tls13_keyschedule_driver.py` (~2863 tok, huge) — # Official-vector driver for std/tls13_keyschedule.mind (pure-MIND TLS 1.3 key
- `tls13_record_driver.py` (~3051 tok, huge) — # Official-vector driver for std/tls13_record.mind (pure-MIND TLS 1.3 record
- `trait_static_dispatch_run.rs` (~1081 tok, large) — Copyright 2025 STARGA Inc.
- `transpose_preview.rs` (~269 tok, medium) — Copyright 2025 STARGA Inc.
- `try_operator_run.rs` (~1058 tok, large) — Copyright 2025 STARGA Inc.
- `tuple_destructure_run.rs` (~1112 tok, large) — Copyright 2025 STARGA Inc.
- `turboquant_kernel_run.rs` (~1318 tok, large) — Copyright 2025 STARGA Inc.
- `type_alias_narrow_lowering_run.rs` (~2249 tok, huge) — Copyright 2025 STARGA Inc.
- `type_ann_check.rs` (~330 tok, medium) — Copyright 2025 STARGA Inc.
- `type_ann_parse.rs` (~583 tok, large) — Copyright 2025 STARGA Inc.
### `tests/type_checker/`

- `basic_type_inference.mind` (~66 tok, small) — Type checker test: Basic type inference
- `dtype_mismatch.mind` (~74 tok, small) — Type checker test: Dtype mismatch detection
### `tests/`

- `type_error_spans.rs` (~1058 tok, large) — Copyright 2025 STARGA Inc.
- `type_infer.rs` (~344 tok, medium) — Copyright 2025 STARGA Inc.
- `type_struct_run.rs` (~582 tok, large) — Copyright 2025 STARGA Inc.
- `typecheck_binary.rs` (~421 tok, medium) — Copyright 2025 STARGA Inc.
- `typecheck_env.rs` (~246 tok, medium) — Copyright 2025 STARGA Inc.
- `typed_literal_match_pattern_run.rs` (~719 tok, large) — Copyright 2025 STARGA Inc. Licensed under the Apache License, Version 2.0.
- `typo_reject.rs` (~1127 tok, large) — Copyright 2025 STARGA Inc.
- `u64_cast_signed_compare_run.rs` (~832 tok, large) — Copyright 2025 STARGA Inc.
- `u64_tag_survival_run.rs` (~2732 tok, huge) — Copyright 2025 STARGA Inc.
- `v04_mirror_vector_dump.rs` (~7866 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `v04_value_id_boundary.rs` (~4189 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `value_if_comparison.rs` (~763 tok, large) — Copyright 2025 STARGA Inc.
- `value_if_f64_let.rs` (~1097 tok, large) — Copyright 2025 STARGA Inc.
- `vars_assign.rs` (~260 tok, medium) — Copyright 2025 STARGA Inc.
- `verify_audit.rs` (~2074 tok, huge) — Audit coverage tests for the IR verifier (C1: SSA verification, conv2d stride/axis validation).
- `verify_canonical_bytes.rs` (~4034 tok, huge) — Copyright 2025 STARGA Inc.
- `verify_cli.rs` (~3865 tok, huge) — Copyright 2025 STARGA Inc.
- `verify_holes.rs` (~3127 tok, huge) — Copyright 2025 STARGA Inc.
- `verify_pinned_signer.rs` (~993 tok, large) — Copyright 2025 STARGA Inc.
- `verify_require_signed.rs` (~1966 tok, huge) — Copyright 2025 STARGA Inc.
- `verify_ssa.rs` (~7389 tok, huge) — Copyright 2025 STARGA Inc.
- `x25519_vectors_driver.py` (~1525 tok, huge) — # Official-vector driver for std/x25519.mind (pure-MIND Curve25519 ECDH).
- `x25519mlkem768_driver.py` (~2378 tok, huge) — # Known-answer driver for std/x25519mlkem768.mind (pure-MIND X25519MLKEM768
- `x509_vectors_driver.py` (~3593 tok, huge) — # Real-certificate driver for std/x509.mind (pure-MIND X.509 DER parsing + RSA
### `tools/`

- `add_copyright_headers.py` (~1132 tok, large) — # Copyright 2025 STARGA Inc.
- `bench_gate.py` (~5201 tok, huge) — # Copyright 2025 STARGA Inc.
- `cargo-deny-sanitize.sh` (~572 tok, large) — Run cargo-deny but sanitize advisory entries that older cargo-deny versions
### `tools/mindfuzz/`

- `.gitignore` (~7 tok, tiny) — tools/mindfuzz/__pycache__/
- `README.md` (~1730 tok, huge) — MIND-Fuzz
- `ci_batch.py` (~1482 tok, large) — MIND-Fuzz deterministic batch GENERATOR -> cross-substrate reference corpus.
- `fuzz_loop.py` (~3529 tok, huge) — MIND-Fuzz loop -- LLM-mutation differential testing for the MIND compiler.
- `mutate.py` (~3631 tok, huge) — MIND-Fuzz mutation engine.
- `mutations.txt` (~1765 tok, huge) — # MIND-Fuzz mutation instructions (adapted from arXiv:2501.00655 Table 1).
- `oracles.py` (~4619 tok, huge) — MIND-Fuzz differential oracles.
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

- `.gitignore` (~4 tok, tiny) — __pycache__/
- `__init__.py` (~384 tok, medium) — # Copyright 2025-2026 STARGA Inc.
- `ai_proof.py` (~640 tok, large) — # Copyright 2025-2026 STARGA Inc.
- `ir.py` (~920 tok, large) — # Copyright 2025-2026 STARGA Inc.
- `jax.py` (~1007 tok, large) — # Copyright 2025-2026 STARGA Inc.
- `pytorch.py` (~1718 tok, huge) — # Copyright 2025-2026 STARGA Inc.
### `tools/pytorch_bridge/tests/`

- `__init__.py` (~0 tok, tiny)
- `test_bridge.py` (~1244 tok, large) — # Copyright 2025-2026 STARGA Inc.
### `tools/`

- `run_bench_gate.sh` (~535 tok, large) — Copyright 2025 STARGA Inc. Licensed under the Apache License, Version 2.0.
- `test_bench_gate.py` (~6125 tok, huge) — Regression test for the bench-gate fail-closed contract (tools/bench_gate.py).

---
*Generated by `anatomy 1.0.0`. Edit descriptions manually — re-run preserves structure.*
