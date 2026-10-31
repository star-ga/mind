# ANATOMY.md — Project File Index

> **For coding agents.** Read this before opening files. Use descriptions and token
> estimates to decide whether you need the full file or the summary is enough.
> Re-generate with: `anatomy .`

**Project:** `mind`
**Files:** 1921 | **Est. tokens:** ~3,744,620
**Generated:** 2026-10-31 03:50 UTC

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
| `./` | 35 | ~32,793 |
| `agents/` | 1 | ~436 |
| `.agents/skills/mindc-development/` | 1 | ~235 |
| `.arch-mind/` | 2 | ~644 |
| `assets/logo/` | 1 | ~453 |
| `audits/` | 6 | ~607 |
| `audits/native-direct-2026-10-20/` | 5 | ~1,394 |
| `bench/` | 5 | ~7,574 |
| `benches/` | 28 | ~87,443 |
| `benches/common/` | 2 | ~2,995 |
| `benches/e2e/` | 1 | ~3,047 |
| `bench/fft/` | 8 | ~8,060 |
| `benchmarks/` | 13 | ~28,163 |
| `benchmarks/autograd_comparison/` | 8 | ~9,411 |
| `benchmarks/cupy_comparison/` | 6 | ~7,733 |
| `benchmarks/determinism/` | 3 | ~4,601 |
| `benchmarks/inference/` | 4 | ~4,049 |
| `benchmarks/jax_comparison/` | 5 | ~4,689 |
| `benchmarks/mojo/` | 8 | ~4,276 |
| `benchmarks/pytorch_comparison/` | 5 | ~4,880 |
| `.cargo/` | 1 | ~130 |
| `config/` | 2 | ~3,347 |
| `docs/` | 39 | ~113,434 |
| `docs/backends/` | 1 | ~1,482 |
| `docs/benchmarks/` | 3 | ~9,042 |
| `docs/design/` | 4 | ~11,498 |
| `docs/gates/` | 1 | ~3,910 |
| `docs/mindcraft/` | 3 | ~7,086 |
| `docs/rfcs/` | 35 | ~173,185 |
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
| `examples/mindc_mind/` | 174 | ~451,653 |
| `examples/mindc_mind/testdata/` | 3 | ~4,159 |
| `examples/mindc_mind/testdata/backend_native_bridge/` | 9 | ~1,078 |
| `examples/mindc_mind/testdata/native_elf_oracle/` | 6 | ~913 |
| `examples/mindc_mind/testdata/native_record_array/` | 42 | ~2,926 |
| `examples/mindc_mind/testdata/native_record_trace/` | 13 | ~2,127 |
| `examples/mindc_mind/testdata/selfhost_loop/` | 2 | ~2,137 |
| `examples/mind_mirror_v04/` | 2 | ~3,836 |
| `examples/mind_mirror_v04/testdata/` | 2 | ~3,862 |
| `examples/native/` | 4 | ~1,440 |
| `examples/parser/` | 5 | ~17,923 |
| `examples/quant/` | 1 | ~3,302 |
| `examples/quant/american_binomial/` | 2 | ~17,234 |
| `examples/quant/barrier_analytic/` | 2 | ~17,440 |
| `examples/quant/black_scholes/` | 2 | ~6,320 |
| `examples/quant/bond_curve/` | 2 | ~18,347 |
| `examples/quant/crr_binomial/` | 2 | ~4,881 |
| `examples/quant/greeks_higher_order/` | 2 | ~12,094 |
| `examples/quant/implied_vol/` | 2 | ~10,987 |
| `examples/quant/implied_vol_surface/` | 2 | ~17,191 |
| `examples/quant/monte_carlo_gbm/` | 2 | ~13,954 |
| `examples/quant/portfolio_risk/` | 2 | ~19,841 |
| `examples/research/` | 2 | ~518 |
| `examples/typecheck/` | 5 | ~14,553 |
| `examples/zoo/` | 6 | ~12,518 |
| `experiments/global-vs-local/` | 7 | ~6,487 |
| `.githooks/` | 4 | ~1,755 |
| `.github/` | 4 | ~910 |
| `.github/ISSUE_TEMPLATE/` | 3 | ~440 |
| `.github/scripts/` | 1 | ~962 |
| `.github/workflows/` | 9 | ~16,367 |
| `mind/std/cognitive/` | 4 | ~3,529 |
| `runtime-support/` | 1 | ~22,327 |
| `scripts/` | 33 | ~118,193 |
| `scripts/mind-vs-rust/` | 3 | ~933 |
| `scripts/mind-vs-rust/src/` | 1 | ~2,372 |
| `scripts/sdlc/` | 2 | ~2,845 |
| `sdk/ts/mic-map/` | 6 | ~15,511 |
| `sdk/ts/mic-map/dist/` | 36 | ~29,044 |
| `sdk/ts/mic-map/scripts/` | 1 | ~499 |
| `sdk/ts/mic-map/src/` | 9 | ~12,181 |
| `sdk/ts/mic-map/test/` | 4 | ~7,843 |
| `sdk/ts/mic-map/test/fixtures/` | 2 | ~96 |
| `skills/write-mind/` | 1 | ~6,002 |
| `src/` | 11 | ~52,927 |
| `src/ast/` | 2 | ~12,305 |
| `src/autodiff/` | 3 | ~6,624 |
| `src/bin/` | 1 | ~9,878 |
| `src/bin/mindc/` | 1 | ~204 |
| `src/build/` | 13 | ~48,441 |
| `src/build/cache/` | 3 | ~3,150 |
| `src/cache/` | 4 | ~3,682 |
| `src/check/` | 5 | ~12,177 |
| `src/deps/` | 1 | ~9,388 |
| `src/diagnostics/` | 4 | ~15,672 |
| `src/distributed/` | 6 | ~7,725 |
| `src/doc/` | 3 | ~11,002 |
| `src/eval/` | 45 | ~152,271 |
| `src/eval/stdlib/` | 2 | ~8,586 |
| `src/eval/struct_resolver/` | 1 | ~2,555 |
| `src/exec/` | 3 | ~5,522 |
| `src/ffi/` | 3 | ~5,541 |
| `src/fmt/` | 3 | ~25,460 |
| `src/fmt/printer/` | 1 | ~1,878 |
| `src/ir/` | 11 | ~101,025 |
| `src/ir/compact/` | 3 | ~15,351 |
| `src/ir/compact/v2/` | 9 | ~45,363 |
| `src/ir/compact/v3/` | 15 | ~76,625 |
| `src/ir/compact/v3/v04/` | 3 | ~13,696 |
| `src/lint/` | 2 | ~4,001 |
| `src/lint/rules/` | 6 | ~10,129 |
| `src/mlir/` | 4 | ~11,670 |
| `src/ops/` | 3 | ~4,764 |
| `src/opt/` | 9 | ~53,642 |
| `src/package/` | 2 | ~1,877 |
| `src/parser/` | 8 | ~29,774 |
| `src/phf/` | 1 | ~4,955 |
| `src/project/` | 21 | ~70,946 |
| `src/runtime/` | 3 | ~1,485 |
| `src/shapes/` | 2 | ~6,052 |
| `src/stdlib/` | 2 | ~560 |
| `src/test/` | 4 | ~15,068 |
| `src/type_checker/` | 18 | ~62,604 |
| `src/type_checker/resolve/` | 1 | ~1,569 |
| `src/type_checker/slice_abi/` | 3 | ~11,245 |
| `src/types/` | 10 | ~20,759 |
| `src/workspace/` | 1 | ~4,906 |
| `std/` | 42 | ~202,387 |
| `tests/` | 419 | ~852,674 |
| `tests/autodiff/` | 2 | ~247 |
| `tests/backend/` | 2 | ~125 |
| `tests/common/` | 3 | ~12,308 |
| `tests/conformance/cpu_baseline/` | 12 | ~390 |
| `tests/conformance/gpu_profile/` | 2 | ~11 |
| `tests/cross_substrate_identity/` | 2 | ~4,706 |
| `tests/cross_substrate_identity/array-store-branch/` | 2 | ~974 |
| `tests/cross_substrate_identity/array-store-loop/` | 2 | ~1,081 |
| `tests/cross_substrate_identity/bimap-phf/` | 2 | ~1,675 |
| `tests/cross_substrate_identity/collatz/` | 2 | ~962 |
| `tests/cross_substrate_identity/dot-f32-v-4093/` | 2 | ~1,222 |
| `tests/cross_substrate_identity/dot-i16-4096/` | 2 | ~648 |
| `tests/cross_substrate_identity/dot-l1-q16/` | 2 | ~363 |
| `tests/cross_substrate_identity/dot-l2-q16/` | 2 | ~813 |
| `tests/cross_substrate_identity/galperin-pi/` | 2 | ~1,004 |
| `tests/cross_substrate_identity/gemm-i8-64x64x64/` | 2 | ~707 |
| `tests/cross_substrate_identity/gemm-i8-mt-64x64x64/` | 2 | ~872 |
| `tests/cross_substrate_identity/gemm-i8-vnni-64x64x64/` | 2 | ~943 |
| `tests/cross_substrate_identity/gemm-q16-64x64x64/` | 2 | ~616 |
| `tests/cross_substrate_identity/gemm-q16-fused-64x64x64/` | 2 | ~896 |
| `tests/cross_substrate_identity/gemv-i16-256x256/` | 2 | ~594 |
| `tests/cross_substrate_identity/gemv-q16-256x256/` | 2 | ~519 |
| `tests/cross_substrate_identity/grammar-mask/` | 2 | ~916 |
| `tests/cross_substrate_identity/lorenz-q16/` | 2 | ~1,243 |
| `tests/cross_substrate_identity/matmul-f32-v-64x64/` | 2 | ~1,115 |
| `tests/cross_substrate_identity/q16-arith-chain/` | 2 | ~788 |
| `tests/cross_substrate_identity/quant-american-binomial/` | 2 | ~957 |
| `tests/cross_substrate_identity/quant-barrier-analytic/` | 2 | ~865 |
| `tests/cross_substrate_identity/quant-black-scholes/` | 2 | ~944 |
| `tests/cross_substrate_identity/quant-bond-curve/` | 2 | ~781 |
| `tests/cross_substrate_identity/quant-crr-binomial/` | 2 | ~763 |
| `tests/cross_substrate_identity/quant-greeks-higher-order/` | 2 | ~649 |
| `tests/cross_substrate_identity/quant-implied-vol/` | 2 | ~1,059 |
| `tests/cross_substrate_identity/quant-implied-vol-surface/` | 2 | ~643 |
| `tests/cross_substrate_identity/quant-monte-carlo-gbm/` | 2 | ~1,083 |
| `tests/cross_substrate_identity/quant-portfolio-risk/` | 2 | ~600 |
| `tests/cross_substrate_identity/scalar-cast-conv/` | 2 | ~1,644 |
| `tests/cross_substrate_identity/scalar-cast-conv-narrow/` | 2 | ~1,790 |
| `tests/cross_substrate_identity/scalar-float-f64/` | 2 | ~1,310 |
| `tests/cross_substrate_identity/struct-handle-roundtrip/` | 2 | ~746 |
| `tests/cross_substrate_identity/u64-ops/` | 2 | ~1,054 |
| `tests/fixtures/` | 7 | ~5,820 |
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
| `tests/mindfuzz_cross_substrate/staged/` | 15 | ~2,832 |
| `tests/native_bridge_support/` | 2 | ~2,574 |
| `tests/runtime/` | 2 | ~135 |
| `tests/selfhost_gaps/` | 163 | ~10,627 |
| `tests/selfhost_gaps/never_wrong/` | 18 | ~338 |
| `tests/shapes/` | 3 | ~260 |
| `tests/skip_shape_scan/` | 2 | ~10,401 |
| `tests/support/` | 10 | ~22,548 |
| `tests/type_checker/` | 2 | ~140 |
| `tests/xsi_quant/` | 3 | ~11,906 |
| `tools/` | 4 | ~20,957 |
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
- `.bench-baseline-2026-06-01-correctness.txt` (~844 tok, large) — === Correctness-milestone baseline (mindc 0.7.0, 2026-06-01) ===
- `.bench-baseline-2026-10-28-e2e.txt` (~1220 tok, large) — === End-to-end floor (T2 warm/cold + T2-native), recorded 2026-10-28 ===
- `.bench-pre-pratt.txt` (~32 tok, tiny) — === captured pre-Pratt baseline (Phase 10.5 in main) ===
- `bounties.md` (~888 tok, large) — MIND Bounty Board
- `build.rs` (~234 tok, medium) — Copyright 2025 STARGA Inc.
- `Cargo.toml` (~4104 tok, huge) — [package]
- `clippy.toml` (~25 tok, tiny)
- `CODE_OF_CONDUCT.md` (~29 tok, tiny) — Code of Conduct
- `COMPLETE_FILE_STRUCTURE.md` (~26 tok, tiny) — Repository Structure (Snapshot)
- `CONTRIBUTING.md` (~2260 tok, huge) — Contributing to MIND
- `deny.toml` (~131 tok, small) — [advisories]
- `.editorconfig` (~51 tok, small) — root = true
- `.gitattributes` (~130 tok, small) — # Enforce LF line endings for all text so byte-exact tests (fmt idempotence,
- `GITHUB_SETUP_INSTRUCTIONS.md` (~240 tok, medium) — GitHub Setup (Quick)
- `.gitignore` (~950 tok, large) — # Rust
- `incompatible` (~0 tok, tiny)
- `LICENSE` (~2573 tok, huge) —                                  Apache License
- `LICENSE-COMMERCIAL` (~399 tok, medium) — COMMERCIAL LICENSE NOTICE – MIND (Enterprise & SaaS)
- `mindci.toml` (~245 tok, medium) — [pipeline]
- `Mind.toml` (~108 tok, small) — [package]
- `plugin.json` (~62 tok, small) — Keys: name, description, version, skills, agents
- `README.md` (~7470 tok, huge) — MIND — Machine Intelligence Native Design
- `RELEASING.md` (~131 tok, small) — Release checklist (as of v0.2.1)
- `rustfmt.toml` (~23 tok, tiny) — max_width = 100
- `SECURITY.md` (~1903 tok, huge) — Security Policy
- `.sembleignore` (~72 tok, small) — # semble code-search ignore list
- `STATUS.md` (~4457 tok, huge) — MIND Compiler Status
