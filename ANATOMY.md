# ANATOMY.md — Project File Index

> **For coding agents.** Read this before opening files. Use descriptions and token
> estimates to decide whether you need the full file or the summary is enough.
> Re-generate with: `anatomy .`

**Project:** `mind`
**Files:** 3533 | **Est. tokens:** ~6,210,960
**Generated:** 2026-11-24 11:21 UTC

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
| `./` | 35 | ~42,397 |
| `agents/` | 1 | ~436 |
| `.agents/skills/mindc-development/` | 1 | ~235 |
| `.arch-mind/` | 2 | ~898 |
| `assets/logo/` | 1 | ~453 |
| `audits/` | 6 | ~607 |
| `audits/native-direct-2026-10-20/` | 5 | ~1,394 |
| `bench/` | 5 | ~7,576 |
| `benches/` | 28 | ~88,231 |
| `benches/common/` | 2 | ~3,025 |
| `benches/e2e/` | 1 | ~3,053 |
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
| `config/` | 2 | ~3,628 |
| `docs/` | 45 | ~130,894 |
| `docs/backends/` | 1 | ~1,482 |
| `docs/benchmarks/` | 3 | ~9,044 |
| `docs/compiler/` | 11 | ~30,424 |
| `docs/design/` | 5 | ~12,746 |
| `docs/gates/` | 1 | ~4,181 |
| `docs/mindcraft/` | 3 | ~7,086 |
| `docs/rfcs/` | 35 | ~174,498 |
| `docs/specs/` | 2 | ~2,205 |
| `examples/` | 28 | ~50,137 |
| `examples/bimap_currency/` | 3 | ~780 |
| `examples/bimap_pairs/` | 3 | ~853 |
| `examples/c/` | 2 | ~400 |
| `examples/columnar/` | 4 | ~7,585 |
| `examples/compliance/` | 3 | ~5,289 |
| `examples/detmath_kat/` | 2 | ~1,981 |
| `examples/distribution-crossisa/` | 6 | ~6,336 |
| `examples/emit_ir/` | 5 | ~13,648 |
| `examples/formal_verification/` | 9 | ~4,735 |
| `examples/grammar_mask/` | 2 | ~4,636 |
| `examples/halbach_q16/` | 2 | ~7,856 |
| `examples/lexer/` | 6 | ~8,888 |
| `examples/mindc_mind/` | 361 | ~1,080,188 |
| `examples/mindc_mind/testdata/` | 8 | ~26,031 |
| `examples/mindc_mind/testdata/backend_native_bridge/` | 9 | ~1,078 |
| `examples/mindc_mind/testdata/compiler_parts/` | 2 | ~4,469 |
| `examples/mindc_mind/testdata/mindfuzz_regress/` | 74 | ~9,283 |
| `examples/mindc_mind/testdata/mindfuzz_regress_failclosed/` | 14 | ~2,935 |
| `examples/mindc_mind/testdata/native_alloc_bounded/` | 14 | ~3,315 |
| `examples/mindc_mind/testdata/native_alloc_bounded/rt/` | 3 | ~7,461 |
| `examples/mindc_mind/testdata/native_alloc_default/` | 47 | ~6,236 |
| `examples/mindc_mind/testdata/native_avx2/` | 124 | ~88,992 |
| `examples/mindc_mind/testdata/native_cf/` | 24 | ~6,169 |
| `examples/mindc_mind/testdata/native_c_variadic_emit/` | 150 | ~278,450 |
| `examples/mindc_mind/testdata/native_c_variadic_plan/` | 147 | ~89,227 |
| `examples/mindc_mind/testdata/native_dynamic_allocator/` | 4 | ~6,741 |
| `examples/mindc_mind/testdata/native_elf_oracle/` | 6 | ~913 |
| `examples/mindc_mind/testdata/native_lane_values/` | 32 | ~6,430 |
| `examples/mindc_mind/testdata/native_libc_binding/` | 11 | ~2,804 |
| `examples/mindc_mind/testdata/native_libc_io/` | 8 | ~15,321 |
| `examples/mindc_mind/testdata/native_libc_rows/` | 20 | ~7,680 |
| `examples/mindc_mind/testdata/native_libc_startup/` | 3 | ~1,539 |
| `examples/mindc_mind/testdata/native_measure/` | 5 | ~1,521 |
| `examples/mindc_mind/testdata/native_misc/` | 47 | ~6,151 |
| `examples/mindc_mind/testdata/native_record_array/` | 42 | ~2,926 |
| `examples/mindc_mind/testdata/native_record_trace/` | 13 | ~2,127 |
| `examples/mindc_mind/testdata/native_scoped_fixed/` | 8 | ~17,608 |
| `examples/mindc_mind/testdata/native_scoped_fixed/observers/` | 7 | ~7,820 |
| `examples/mindc_mind/testdata/native_scoped_fixed/sources/` | 43 | ~7,306 |
| `examples/mindc_mind/testdata/native_shm/` | 2 | ~2,104 |
| `examples/mindc_mind/testdata/native_source_driver/` | 4 | ~3,237 |
| `examples/mindc_mind/testdata/native_string_handle/` | 16 | ~4,157 |
| `examples/mindc_mind/testdata/native_trap/` | 6 | ~960 |
| `examples/mindc_mind/testdata/seed_ledger/` | 2 | ~4,928 |
| `examples/mindc_mind/testdata/selfhost_loop/` | 2 | ~2,414 |
| `examples/mindc_mind/testdata/selfhost_loop_bounded/` | 1 | ~269 |
| `examples/mindc_mind/testdata/string_borrowed/` | 1 | ~869 |
| `examples/mindc_mind/testdata/string_handle_limits/` | 8 | ~385 |
| `examples/mind_mirror_v04/` | 2 | ~3,873 |
| `examples/mind_mirror_v04/testdata/` | 2 | ~3,862 |
| `examples/native/` | 4 | ~1,445 |
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
| `.github/` | 4 | ~944 |
| `.github/actions/avx2-neon-digest/` | 1 | ~807 |
| `.github/actions/glibc-floor-smoke/` | 1 | ~1,395 |
| `.github/actions/install-mlir-toolchain/` | 1 | ~1,746 |
| `.github/actions/native-lane-count/` | 1 | ~341 |
| `.github/actions/nonempty-artifact/` | 1 | ~152 |
| `.github/actions/package-release/` | 1 | ~1,257 |
| `.github/actions/qemu-user/` | 1 | ~278 |
| `.github/actions/release-artifact-smoke/` | 1 | ~3,736 |
| `.github/actions/seed-ledger/` | 1 | ~249 |
| `.github/actions/std-evidence-checks/` | 1 | ~779 |
| `.github/actions/std-example-programs/` | 1 | ~566 |
| `.github/actions/std-fs-checked/` | 1 | ~1,148 |
| `.github/ISSUE_TEMPLATE/` | 3 | ~440 |
| `.github/scripts/` | 1 | ~1,297 |
| `.github/workflows/` | 10 | ~23,560 |
| `mind/std/cognitive/` | 4 | ~3,529 |
| `runtime-support/` | 2 | ~24,954 |
| `scripts/` | 36 | ~170,150 |
| `scripts/mind-vs-rust/` | 3 | ~933 |
| `scripts/mind-vs-rust/src/` | 1 | ~2,372 |
| `scripts/sdlc/` | 2 | ~2,845 |
| `sdk/ts/mic-map/` | 6 | ~15,511 |
| `sdk/ts/mic-map/dist/` | 36 | ~29,044 |
| `sdk/ts/mic-map/scripts/` | 1 | ~501 |
| `sdk/ts/mic-map/src/` | 9 | ~12,181 |
| `sdk/ts/mic-map/test/` | 4 | ~7,843 |
| `sdk/ts/mic-map/test/fixtures/` | 2 | ~96 |
| `skills/write-mind/` | 1 | ~6,002 |
| `src/` | 13 | ~65,786 |
| `src/ast/` | 2 | ~12,334 |
| `src/autodiff/` | 3 | ~6,624 |
| `src/bin/` | 1 | ~9,878 |
| `src/bin/mindc/` | 1 | ~3,438 |
| `src/build/` | 14 | ~57,788 |
| `src/build/cache/` | 3 | ~3,150 |
| `src/build/native_bridge/` | 10 | ~42,630 |
| `src/cache/` | 4 | ~3,682 |
| `src/check/` | 5 | ~13,127 |
| `src/deps/` | 1 | ~9,388 |
| `src/diagnostics/` | 4 | ~17,991 |
| `src/distributed/` | 6 | ~7,725 |
| `src/doc/` | 3 | ~11,002 |
| `src/eval/` | 57 | ~194,682 |
| `src/eval/stdlib/` | 2 | ~8,577 |
| `src/eval/string_equality/` | 12 | ~48,169 |
| `src/eval/string_equality/facts/` | 1 | ~1,401 |
| `src/eval/struct_resolver/` | 1 | ~2,555 |
| `src/exec/` | 3 | ~5,522 |
| `src/ffi/` | 3 | ~5,541 |
| `src/fmt/` | 3 | ~27,552 |
| `src/fmt/printer/` | 1 | ~1,912 |
| `src/intrinsics/` | 1 | ~3,924 |
| `src/ir/` | 16 | ~109,313 |
| `src/ir/compact/` | 3 | ~15,303 |
| `src/ir/compact/v2/` | 9 | ~45,363 |
| `src/ir/compact/v3/` | 15 | ~78,258 |
| `src/ir/compact/v3/v04/` | 3 | ~13,793 |
| `src/ir/evidence/` | 1 | ~3,500 |
| `src/ir/native_c_abi/` | 1 | ~3,811 |
| `src/ir/native_lane_profile/` | 14 | ~46,223 |
| `src/lint/` | 2 | ~4,001 |
| `src/lint/rules/` | 6 | ~10,333 |
| `src/mlir/` | 10 | ~28,705 |
| `src/ops/` | 3 | ~4,764 |
| `src/opt/` | 9 | ~55,444 |
| `src/package/` | 2 | ~1,877 |
| `src/parser/` | 9 | ~31,177 |
| `src/phf/` | 1 | ~4,955 |
| `src/project/` | 25 | ~97,973 |
| `src/project/single_file_scope/` | 2 | ~4,104 |
| `src/project/stdlib/` | 1 | ~4,725 |
| `src/qualified_enums/` | 2 | ~3,391 |
| `src/runtime/` | 3 | ~1,485 |
| `src/shapes/` | 2 | ~6,052 |
| `src/stdlib/` | 2 | ~560 |
| `src/test/` | 6 | ~26,480 |
| `src/type_checker/` | 21 | ~71,991 |
| `src/type_checker/resolve/` | 2 | ~3,526 |
| `src/type_checker/slice_abi/` | 4 | ~17,202 |
| `src/types/` | 10 | ~23,602 |
| `src/workspace/` | 1 | ~4,906 |
| `std/` | 78 | ~323,396 |
| `tests/` | 498 | ~1,230,557 |
| `tests/autodiff/` | 2 | ~247 |
| `tests/backend/` | 2 | ~125 |
| `tests/common/` | 7 | ~18,102 |
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
| `tests/cross_substrate_identity/json-canon-kat/` | 2 | ~610 |
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
| `tests/fixtures/evidence/` | 2 | ~181 |
| `tests/fixtures/f64_abi_negative/` | 3 | ~192 |
| `tests/fixtures/native_lane/` | 1 | ~15,287 |
| `tests/fixtures/native_lane_extern/` | 2 | ~532 |
| `tests/fixtures/native_lane_gpu/` | 12 | ~11,218 |
| `tests/fixtures/native_lane_refusals/` | 63 | ~1,967 |
| `tests/fixtures/native_lane_std/` | 22 | ~7,252 |
| `tests/fixtures/native_lane_string_rows/` | 7 | ~34,863 |
| `tests/fixtures/nerve_numerics/` | 11 | ~968 |
| `tests/fixtures/selfhost_policy/` | 13 | ~315 |
| `tests/fixtures/selfhost_policy/reject/` | 58 | ~616 |
| `tests/fixtures/string_eq_module_scope/` | 2 | ~31 |
| `tests/fixtures/string_eq_module_scope/src/` | 4 | ~380 |
| `tests/fixtures/tls13_client/` | 34 | ~39,914 |
| `tests/fs_checked_listing/` | 13 | ~14,043 |
| `tests/https_loopback/` | 22 | ~24,750 |
| `tests/ir_verification/` | 2 | ~108 |
| `tests/lexical/` | 3 | ~191 |
| `tests/mindc_build_phase_a/` | 1 | ~1,173 |
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
| `tests/native_alloc_std_drivers/` | 1 | ~315 |
| `tests/native_bridge_support/` | 2 | ~2,987 |
| `tests/native_lane_extern_support/` | 1 | ~7,775 |
| `tests/runtime/` | 2 | ~135 |
| `tests/selfhost_gaps/` | 163 | ~10,627 |
| `tests/selfhost_gaps/never_wrong/` | 18 | ~338 |
| `tests/sha256_lifetime/` | 8 | ~11,467 |
| `tests/sha256_lifetime/mutants/` | 8 | ~241 |
| `tests/shapes/` | 3 | ~260 |
| `tests/skip_shape_scan/` | 2 | ~10,401 |
| `tests/std_entropy_dns_support/` | 1 | ~1,599 |
| `tests/std_evidence_kat/` | 2 | ~2,553 |
| `tests/std_evidence_load/` | 2 | ~1,905 |
| `tests/std_evidence_wrappers/` | 2 | ~3,421 |
| `tests/std_process_wait/` | 5 | ~7,294 |
| `tests/support/` | 16 | ~45,962 |
| `tests/type_checker/` | 2 | ~140 |
| `tests/x509/` | 4 | ~4,407 |
| `tests/x509/testdata/` | 86 | ~13,454 |
| `tests/x509/testdata/adv/` | 26 | ~5,523 |
| `tests/xsi_quant/` | 3 | ~11,922 |
| `tools/` | 5 | ~30,162 |
| `tools/mindfuzz/` | 7 | ~16,763 |
| `tools/mindfuzz/seeds/` | 6 | ~1,330 |
| `tools/mindfuzz/violations/` | 1 | ~0 |
| `tools/pytorch_bridge/` | 6 | ~4,673 |
| `tools/pytorch_bridge/tests/` | 2 | ~1,244 |
| `tools/seed_ledger/` | 6 | ~7,118 |

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
- `build.rs` (~1738 tok, huge) — Copyright 2025 STARGA Inc.
- `Cargo.toml` (~4208 tok, huge) — [package]
- `clippy.toml` (~25 tok, tiny)
- `CODE_OF_CONDUCT.md` (~29 tok, tiny) — Code of Conduct
- `COMPLETE_FILE_STRUCTURE.md` (~26 tok, tiny) — Repository Structure (Snapshot)
- `CONTRIBUTING.md` (~2289 tok, huge) — Contributing to MIND
- `deny.toml` (~131 tok, small) — [advisories]
- `.editorconfig` (~51 tok, small) — root = true
- `.gitattributes` (~191 tok, small) — # Enforce LF line endings for all text so byte-exact tests (fmt idempotence,
- `GITHUB_SETUP_INSTRUCTIONS.md` (~240 tok, medium) — GitHub Setup (Quick)
- `.gitignore` (~8460 tok, huge) — # Rust
- `incompatible` (~0 tok, tiny)
- `LICENSE` (~2573 tok, huge) —                                  Apache License
- `LICENSE-COMMERCIAL` (~399 tok, medium) — COMMERCIAL LICENSE NOTICE – MIND (Enterprise & SaaS)
- `mindci.toml` (~245 tok, medium) — [pipeline]
- `Mind.toml` (~108 tok, small) — [package]
- `plugin.json` (~62 tok, small) — Keys: name, description, version, skills, agents
- `README.md` (~7826 tok, huge) — MIND — Machine Intelligence Native Design
- `RELEASING.md` (~171 tok, small) — Release checklist (as of v0.2.1)
- `rustfmt.toml` (~23 tok, tiny) — max_width = 100
- `SECURITY.md` (~1903 tok, huge) — Security Policy
- `.sembleignore` (~72 tok, small) — # semble code-search ignore list
- `STATUS.md` (~4457 tok, huge) — MIND Compiler Status
- `test_real_compile_time.py` (~265 tok, medium) — Quick test of real MIND compilation time using Python bindings."""
### `agents/`

- `mind-developer.md` (~436 tok, medium) — MIND Developer Agent
### `.agents/skills/mindc-development/`

- `SKILL.md` (~235 tok, medium) — MIND Compiler (mindc) Development
### `.arch-mind/`

- `rules.mind` (~811 tok, large) — mind (language compiler / runtime root) architectural-governance rules
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
### `audits/native-direct-2026-10-20/`

- `const_no_semicolon.mind` (~90 tok, small) — A const declared without a trailing `;` is refused by the native emitter
- `duplicate_const.mind` (~47 tok, tiny) — A duplicate const declaration refuses every read of it on the native route.
- `extern_swallow.mind` (~74 tok, small) — An `extern "C" fn` declaration placed before `main` makes the native emitter
- `loop_copy.mind` (~99 tok, small) — Loop-carried plain copy: the native emitter (examples/mindc_mind/main.mind,
- `README.md` (~1084 tok, large) — Native backend, direct route: findings (2026-10-20)
### `bench/`

- `beat_clang_igemm_driver.c` (~494 tok, medium)
### `benches/`

- `autodiff.rs` (~1661 tok, huge) — Simple linear function
- `bench_aes_gcm.rs` (~2649 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `bench_ecdsa_p256.rs` (~3384 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `bench_hkdf.rs` (~4424 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `bench_hpack.rs` (~3927 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `bench_http2_frame.rs` (~5035 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `bench_keccak.rs` (~2576 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `bench_mlkem768.rs` (~3696 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `bench_rsa_pss.rs` (~3282 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `bench_sha256.rs` (~2594 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `bench_tls13_record.rs` (~6121 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `bench_x25519.rs` (~2469 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `bench_x509.rs` (~3933 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `cerebras_stencil.rs` (~831 tok, large) — Copyright 2025-2026 STARGA Inc.
### `benches/common/`

- `mod.rs` (~668 tok, large) — Shared self-skip probe for criterion bench targets.
- `roofline_peak.rs` (~2357 tok, huge) — Shared roofline denominator for the deterministic GEMM / int-dot benches.
### `benches/`

- `compiler.rs` (~7594 tok, huge) — Small program: Simple matrix multiplication
- `cross_module.rs` (~605 tok, large) — Copyright 2025 STARGA Inc.
- `det_gemv_q16_mt.rs` (~3229 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `det_matmul_i16.rs` (~4549 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `det_matmul_i8.rs` (~4988 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `det_matmul_q16_mt.rs` (~4049 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `det_matmul_q16.rs` (~4829 tok, huge) — Copyright 2025-2026 STARGA Inc.
### `benches/e2e/`

- `paired.rs` (~3053 tok, huge) — Same-run paired A/B of two `mindc` binaries: the end-to-end gate's measurement.
### `benches/`

- `fft_q16.rs` (~5359 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `mindcraft_fmt.rs` (~1079 tok, large) — File readers
- `operations.rs` (~1200 tok, large) — Element-wise operations
- `parser_throughput.rs` (~916 tok, large) — Copyright 2025 STARGA Inc.
- `shapes.rs` (~1368 tok, large) — Simple broadcasting scenarios
- `simple_benchmarks.rs` (~794 tok, large) — Mirror mindc's allocator so this compile-speed bench measures the same heap
- `std_surface.rs` (~1090 tok, large) — Copyright 2025 STARGA Inc.
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
- `requirements.txt` (~7 tok, tiny) — torch>=2.13.0
### `benchmarks/`

- `BENCHMARK_RESULTS.md` (~4656 tok, huge) — MIND Benchmark Results
- `criterion_ci_sweep.txt` (~4095 tok, huge) — CRITERION CI-EQUIVALENT SWEEP — recorded output
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

- `format_benchmark.py` (~2434 tok, huge)
### `benchmarks/inference/`

- `benchmark_inference.py` (~2423 tok, huge)
- `inference_results.json` (~473 tok, medium) — Keys: system_info, benchmarks
- `README.md` (~1149 tok, large) — Inference Speed Benchmark
- `requirements.txt` (~4 tok, tiny) — torch>=2.13.0
### `benchmarks/jax_comparison/`

- `benchmark_jax_compile.py` (~2719 tok, huge)
- `jax_coldstart_results.json` (~376 tok, medium) — Keys: environment, results
- `jax_results.json` (~478 tok, medium) — Keys: system_info, benchmarks
- `README.md` (~1109 tok, large) — JAX Compilation Benchmark
- `requirements.txt` (~7 tok, tiny) — jax>=0.4.0
### `benchmarks/`

- `mic_benchmark.py` (~1473 tok, large)
- `MIC_MAP_BENCHMARK_README.md` (~675 tok, large) — MIC/MAP Patent Reference Benchmark
- `mic_map_benchmark_results.json` (~1506 tok, huge) — Keys: metadata, measurements, paper_figures_verified, claim_checks, cost_model
- `mic_map_benchmark_v2.py` (~5563 tok, huge)
### `benchmarks/mojo/`

- `benchmark_mojo_compilation.py` (~1533 tok, huge)
- `large_matmul.mojo` (~205 tok, medium) — """
- `medium_matmul.mojo` (~205 tok, medium) — """
- `mojo_results.json` (~216 tok, medium) — Keys: scalar_math, small_matmul, medium_matmul, large_matmul
- `README.md` (~1271 tok, large) — Mojo Compilation Benchmarks
- `run_benchmarks.sh` (~581 tok, large) — Mojo Compilation Benchmark Runner
- `scalar_math.mojo` (~58 tok, small) — """
- `small_matmul.mojo` (~207 tok, medium) — """
### `benchmarks/pytorch_comparison/`

- `=2.0` (~0 tok, tiny)
- `benchmark_pytorch_compile.py` (~3420 tok, huge)
- `pytorch_results.json` (~590 tok, large) — Keys: system_info, benchmarks
- `README.md` (~866 tok, large) — PyTorch Compilation Benchmark
- `requirements.txt` (~4 tok, tiny) — torch>=2.13.0
### `benchmarks/`

- `README.md` (~1168 tok, large) — MIND Performance Benchmarks
- `resnet.md` (~74 tok, small) — ResNet Benchmarks (Preliminary)
- `run_all_benchmarks.sh` (~824 tok, large) — Master script to run all MIND patent benchmarks
- `RUN_GUIDE.md` (~1563 tok, huge) — MIND Patent Benchmarks - Environment Guide
- `scientific_benchmark.py` (~1647 tok, huge)
- `scientific_benchmark_raw.py` (~2485 tok, huge)
### `bench/`

- `matmul_det_bench.mind` (~1079 tok, large) — bench/matmul_det_bench.mind — first pure-MIND runtime benchmark for the
- `RESULTS-beat-clang-igemm-2026-08-21.md` (~841 tok, large) — MIND vs clang -O3 — int8 GEMM head-to-head (2026-08-21)
- `RESULTS-int8-2026-06-08.md` (~693 tok, large) — MIND int8 VNNI GEMM — single-core vs OpenBLAS f32 (2026-06-08)
- `turboquant.mind` (~4469 tok, huge) — bench/turboquant.mind — KV-cache quantization pipeline in pure MIND.
### `.cargo/`

- `config.toml` (~130 tok, small) — [registries]
### `config/`

- `capabilities.toml` (~2656 tok, huge) — [ir]
- `token_pricing.toml` (~972 tok, large) — [pricing]
### `docs/`

- `architecture.md` (~965 tok, large) — Architecture
- `ARRAY_SEMANTICS.md` (~8111 tok, huge) — MIND array semantics — normative architecture record
- `autodiff.md` (~595 tok, large) — Static autodiff (public)
### `docs/backends/`

- `cerebras-stencil.md` (~1482 tok, large) — `mind.cerebras.stencil_tile` — Op Surface and Lowering Contract
### `docs/`

- `benchmarking.md` (~4285 tok, huge) — Benchmarking methodology — tiers and comparable metrics
### `docs/benchmarks/`

- `compiler_performance.md` (~4363 tok, huge) — MIND Compiler Performance Benchmarks
### `docs/`

- `benchmarks.md` (~896 tok, large) — Benchmarks
### `docs/benchmarks/`

- `mojo_comparison.md` (~2505 tok, huge) — MIND vs Mojo: Compilation Performance Comparison
- `RESULTS-mind-vs-rust-2026-06-09.md` (~2176 tok, huge) — MIND vs Rust — integer-GEMM, apples-to-apples (2026-06-09)
### `docs/`

- `byte-store-migration.md` (~3357 tok, huge) — Byte-Store Migration — closing `#306`
- `cli.md` (~1310 tok, large) — MIND CLI Reference
### `docs/compiler/`

- `NATIVE_C_ABI_CONTRACT.md` (~1832 tok, huge) — Source-derived native C ABI declaration contract
- `NATIVE_C_ABI_LOWERING.md` (~4632 tok, huge) — Native C ABI lowering: fixed-arity `extern "C"` calls
- `NATIVE_C_CALL_EMISSION.md` (~1766 tok, huge) — Fixed scalar C call emission
- `NATIVE_C_CALL_PLAN.md` (~2022 tok, huge) — Fixed scalar C call plans
- `NATIVE_C_VARIADIC_EMIT.md` (~4199 tok, huge) — Scalar variadic call emission
- `NATIVE_C_VARIADIC_PLAN.md` (~5757 tok, huge) — Scalar SysV variadic argument plan, version 2
- `NATIVE_DYNAMIC_WRITER.md` (~2654 tok, huge) — Native dynamic ELF writer candidate
- `NATIVE_EVIDENCE_RECIPE.md` (~1339 tok, large) — Evidence and opt-in signing for native ymm programs
- `NATIVE_LANE_PROFILE.md` (~2924 tok, huge) — Typed scalar delivery profile
- `NATIVE_SOURCE_NOTE.md` (~1057 tok, large) — Native ELF source-note tagging
- `NATIVE_STRING_VALUES.md` (~2242 tok, huge) — Native string value decoding
### `docs/design/`

- `evidence-composed-image.md` (~1248 tok, large) — Design note (0.10.4 candidate, not implemented): record the composed source image in the evidence
- `execution-plan-performance-mode.md` (~8045 tok, huge) — Design: PerformanceMode + ExecutionPlan + ExecutionProvider
- `README.md` (~26 tok, tiny) — Design Docs
- `rfc0012-b2-shape-threading-scope.md` (~3317 tok, huge) — RFC 0012 Phase B.2 — shape-dim threading: execution scope
- `v0.3.md` (~110 tok, small) — MIND Design v0.3 (Draft)
### `docs/`

- `determinism.md` (~4871 tok, huge) — The Determinism Contract
- `ENGINE_CONSUMER_LIST.md` (~9563 tok, huge) — Engine consumer list — the evidence pack for every hand-written semantics engine
- `errors.md` (~5262 tok, huge) — MIND Core Error Model
- `ffi-runtime.md` (~529 tok, large) — FFI & Runtime Integration
### `docs/gates/`

- `exec-semantics-tiers.md` (~4181 tok, huge) — Executable-semantics tier gate — doctrine and measurement history
### `docs/`

- `gpu.md` (~387 tok, medium) — GPU backend profile
- `independence-audit-plan-20260925.md` (~4759 tok, huge) — Independence audit and plan — 2026-09-25
- `INDEPENDENCE_ROADMAP.md` (~19736 tok, huge) — MIND Rust-Independence Roadmap
- `install.md` (~2940 tok, huge) — Installing mindc
- `ir.md` (~451 tok, medium) — MIND IR core
- `ir-mlir.md` (~480 tok, medium) — IR & MLIR Integration
- `ir-stability.md` (~1946 tok, huge) — IR stability contract
- `MHS_ROADMAP.md` (~2657 tok, huge) — MHS / Governed Device I/O Roadmap — SUPERSEDED
- `mic3-v04-draft.md` (~1928 tok, huge) — MIC3 `0x04` core codec — implementation draft
- `migration-roadmap.md` (~1896 tok, huge) — MIND Migration Roadmap — Any Language → Pure Executing MIND
### `docs/mindcraft/`

- `fmt.md` (~2302 tok, huge) — `mindc fmt` — Canonical Formatter Reference
- `phase2-implementation-plan.md` (~2209 tok, huge) — Mindcraft Phase 2 — Implementation Plan
- `rfc0010-phase-ghi-migration-plan.md` (~2575 tok, huge) — RFC 0010 Phase G/H/I — Migration Plan (corrected against real architecture)
### `docs/`

- `mlir-lowering.md` (~286 tok, medium) — MLIR lowering pipeline (public)
- `native-closure-lane.md` (~3627 tok, huge) — Candidate native closure and namespace admission
- `ops.md` (~604 tok, large) — Core v1 operator coverage
- `optimization-frontier.md` (~11775 tok, huge) — MIND Optimization Frontier
- `performance.md` (~880 tok, large) — Performance Guide
- `README.md` (~199 tok, small) — MIND Documentation
- `reap-pruning.md` (~901 tok, large) — REAP Expert Pruning
- `release-evidence-anchor.md` (~1279 tok, large) — Evidence anchor of a native release ELF
- `RESEARCH_ECOSYSTEM_OFFLINE_HYPOTHESES_20261029.md` (~318 tok, medium) — Compiler research consumer of an offline hypothesis service
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
- `0007-mindcraft.md` (~4585 tok, huge) — RFC 0007: Mindcraft — the pure-MIND format / lint / check toolchain
- `0008-mindc-build.md` (~11172 tok, huge) — RFC 0008: mindc build + mindc test — retiring cargo from the build path
- `0009-federation-package-layer.md` (~6976 tok, huge) — RFC 0009: Federation-First MIND Package Layer
- `000-template.md` (~1 tok, tiny)
- `0010-memory-safety-and-c-abi.md` (~7353 tok, huge) — RFC 0010: Memory Safety Model + C ABI in Pure MIND
- `0011-async-and-structured-concurrency.md` (~6670 tok, huge) — RFC 0011: Async + Structured Concurrency Model
- `0012-tensor-native-syntax.md` (~12947 tok, huge) — RFC 0012: Tensor-Native Surface Syntax — the Differentiation Layer
- `0013-cli-agent-harness-stack.md` (~6776 tok, huge) — RFC 0013: CLI Agent Harness Stack
- `0014-per-substrate-mlir-lowering-contracts.md` (~5605 tok, huge) — RFC 0014: Per-Substrate MLIR Lowering Pipeline Contracts
- `0015-cross-substrate-bit-identity.md` (~5971 tok, huge) — RFC 0015: Cross-Substrate Bit-Identity Proof Obligation
- `0016-evidence-chain-emission.md` (~6998 tok, huge) — RFC 0016: Compile-Time Evidence-Chain Emission
- `0017-mindc-verify.md` (~5379 tok, huge) — RFC 0017: `mindc verify` — Artifact Verification Surface
- `0018-bare-metal-substrate.md` (~3799 tok, huge) — RFC 0018: Bare-Metal Substrate Lowering Tier
- `0019-deterministic-agent-substrate.md` (~4131 tok, huge) — RFC 0019: Deterministic Agent Substrate
- `0020-mind-bench-reproducibility-harness.md` (~5774 tok, huge) — RFC 0020: mind-bench Public Reproducibility Harness
- `0021-canonical-ir-unification.md` (~4997 tok, huge) — RFC 0021: Canonical IR Unification — one IR, provenance as a versioned epilogue
- `0022-deterministic-io-substrate.md` (~2120 tok, huge) — RFC 0022: Deterministic I/O Substrate — fastest async I/O with bit-identical replay
- `0024-loop-collapse.md` (~7579 tok, huge) — RFC 0024: Loop Collapse — prove-or-fail closed-form replacement of counted loops (`#[collapse]`)
- `0025-mind-intent-contracts.md` (~3605 tok, huge) — RFC 0025: MIND Intent — Intent Contracts (goal + constraints → verifiable Contract IR)
- `0027-governed-physical-device-plane.md` (~2790 tok, huge) — RFC 0027: Governed Physical-Device Plane — evidence-bound actuation (`device_receipts`)
- `0028-deterministic-field-calculus.md` (~10288 tok, huge) — RFC 0028: Deterministic Field Calculus
- `DRAFT-deterministic-format-frontend.md` (~10522 tok, huge) — RFC DRAFT: Deterministic Multi-Format Ingest Front-End (JSON / TOON / CSV / TSV / NDJSON / TOML)
- `DRAFT-deterministic-json-frontend.md` (~5177 tok, huge) — RFC DRAFT: Deterministic Streaming SIMD JSON Structural Front-End
- `DRAFT-governed-device-io-mhs.md` (~5629 tok, huge) — Draft RFC: Governed Device I/O and MHS Compatibility — SUPERSEDED
- `odc-language-primitives.md` (~418 tok, medium) — RFC: Observer-Dependent Cognition — Language Primitives
- `README.md` (~31 tok, tiny) — RFCs
### `docs/`

- `RI_DEPENDENCY_MATRIX.md` (~7291 tok, huge) — Rust-Independence (RI) Dependency Matrix
- `runs-burndown-roadmap.md` (~3203 tok, huge) — MIND RUNS Burndown Roadmap
- `runtime-open-errors.md` (~831 tok, large) — Runtime file-open errors
- `security.md` (~1619 tok, huge) — Security Guide
- `seed-lineage-ordering-validation.md` (~729 tok, large) — Seed lineage before advancement execution
- `self-host-trace-hash-port.md` (~1406 tok, large) — #17 — Self-compute the native PT_NOTE (pure-MIND trace-hash port)
- `sha256-memory-contract.md` (~1808 tok, huge) — SHA-256 memory and failure contract
- `shapes.md` (~478 tok, medium) — Tensor shape semantics
- `sparse-tensor-types.md` (~740 tok, large) — Sparse Tensor Types
### `docs/specs/`

- `README.md` (~23 tok, tiny) — Specifications
- `v1.0.md` (~2182 tok, huge) — MIND Language Specification v1.0 (Working Draft)
### `docs/`

- `std-procgroup.md` (~1788 tok, huge) — `std.procgroup`: run many child processes and wait for all of them
- `type-system.md` (~1836 tok, huge) — Type System
- `VERIFICATION_APPARATUS.md` (~7783 tok, huge) — Self-Host Port Verification Apparatus & SOTA Roadmap
- `versioning.md` (~1018 tok, large) — MIND Core Stability & Versioning
- `version-matrix.md` (~783 tok, large) — MIND Ecosystem — Version Matrix
- `whitepaper.md` (~2788 tok, huge) — MIND: The Native Language for Intelligent Systems
### `examples/`

- `anthropobrot.mind` (~3257 tok, huge) — Anthropobrot: depth-selected orbit-density multisets of the Fatou-Julia iteral.
- `autodiff_demo.mind` (~1715 tok, huge) — Autodiff Demonstration
### `examples/bimap_currency/`

- `.gitignore` (~2 tok, tiny) — target/
- `main.mind` (~716 tok, large) — Single-source bijective map over a "nice set" — ONE declaration, both
- `Mind.toml` (~62 tok, small) — [package]
### `examples/bimap_pairs/`

- `expected_stdout.txt` (~21 tok, tiny) — US United States
- `main.mind` (~763 tok, large) — Single-source bijective const pair-tables — ONE declaration, both directions
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
- `audit_report.mind` (~2284 tok, huge) — audit_report.mind -- Compliance Artifact Generation
- `README.md` (~1073 tok, large) — Compliance Example
### `examples/`

- `cos_dottie.mind` (~781 tok, large) — Cosine-map iteration toward the Dottie fixed point (x* ≈ 0.7390851332151607),
### `examples/detmath_kat/`

- `main.mind` (~1961 tok, huge) — examples/detmath_kat — value-oracle known-answer tests for std.detmath.
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

- `fft_q16.mind` (~1255 tok, large) — Deterministic Q16.16 fixed-point radix-2 DIT FFT, N=256 (complex).
- `fft_signal.mind` (~533 tok, large) — FFT Signal Processing Example for MIND
### `examples/formal_verification/`

- `divisor_functions.mind` (~481 tok, medium) — Pure MIND exact divisor sum sigma(n), Euler phi(n), and gcd.
- `farey_finite.mind` (~485 tok, medium) — Pure MIND Farey fractions: enumerates each Farey sequence F_n
- `goldbach_bounded.mind` (~469 tok, medium) — Pure MIND finite Goldbach pair search for even n in [4,100].
- `harmonic_exact.mind` (~548 tok, large) — Pure MIND exact finite harmonic numbers H_n, n in 0..10.
- `implication_truth_table.mind` (~293 tok, medium) — Finite propositional truth-table check over two Boolean variables; a teaching example.
- `mobius_finite.mind` (~499 tok, medium) — Pure MIND Mobius mu(n) on integers 1..10000 and finite Mertens sums.
- `prime_counting.mind` (~485 tok, medium) — Pure MIND exact prime counts and the next prime in a FINITE range.
- `README.md` (~857 tok, large) — Bounded formal-mathematics examples
- `run_all.sh` (~618 tok, large) — Builds and runs the seven bounded examples of this directory with the native
### `examples/`

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

- `hello_stdlib.mind` (~279 tok, medium) — Hello, std.vec — minimal RFC 0005 cookbook example.
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

- `backend_native_bridge_smoke.py` (~4013 tok, huge) — RI-D slice 1 gate (task #110): `mindc build --backend native` is a byte-faithful
- `bootstrap_smoke.py` (~2473 tok, huge)
- `branch_shadow_crossbackend_smoke.py` (~2050 tok, huge) — #318 — branch-local `let` shadow must not leak across backends (the WEDGE gate).
- `_cf_corpus.py` (~5962 tok, huge) — Control-flow corpus for self_host_native_cf_smoke.py (helper module, no verdict).
- `_cf_giant.py` (~9117 tok, huge) — Program generators of self_host_native_cf_smoke (census X1 capacity cases).
- `_cf_harness.py` (~2041 tok, huge) — Low-level helpers of self_host_native_cf_smoke.py: the stdin image, the
- `_cf_oracle.py` (~2027 tok, huge) — Oracle .so heap checks of self_host_native_cf_smoke (not a gate).
- `check_driver.mind` (~9782 tok, huge) — ===========================================================================
- `closure_netverify.py` (~1442 tok, large) — # Canonical independent net-verify harness for CLOSURES / FN-VALUES / UNRESOLVED
- `collect_field_strings_smoke.py` (~1215 tok, large)
- `cutover_coverage_measure.py` (~3255 tok, huge)
- `_diagnostic_project.py` (~376 tok, medium) — Owned CLI projects for diagnostic smoke fixtures."""
- `div_shift_cmp_edge_smoke.py` (~2126 tok, huge)
- `enum_netverify.py` (~1977 tok, huge) — # Canonical independent net-verify harness for C-LIKE ENUMS in the native-ELF backend.
- `_exec_retry.py` (~1169 tok, large) — Retry of a busy exec (ETXTBSY) for a smoke that writes a program file and runs
- `EXPECTED.md` (~773 tok, large) — Phase 6.5 Stage 5 — Expected IR Text (APEX)
- `fast_keystone.sh` (~3913 tok, huge) — fast_keystone.sh — fast LOCAL front-end keystone gate for the pure-MIND self-host
- `field_store_netverify.py` (~1433 tok, large) — # Canonical independent value harness for struct field STORES (`p.x = v`) in the
- `FIXED_POINT_REPORT.md` (~1770 tok, huge) — Phase 6.5 — Bootstrap Fixed-Point Report
- `fixed_point_smoke.py` (~3337 tok, huge)
- `fixture.mind` (~183 tok, small) — Phase 6.4 emit_ir smoke fixture.
- `_frozen_native_oracle.py` (~384 tok, medium) — Integrity-checked frozen references for the native ELF smoke."""
- `full_strtab_smoke.py` (~1663 tok, huge)
- `gap_corpus_smoke.py` (~5789 tok, huge)
- `general_float_netverify.py` (~2020 tok, huge) — general_float_netverify.py — GENERAL-path f64 value battery (B0 gate lift).
- `.gitignore` (~5 tok, tiny) — __pycache__/
- `h2b_exclusion_controls_check.py` (~4235 tok, huge) — H2b exclusion positive controls.
- `h2b_fmt_identity_check.py` (~3861 tok, huge) — H2b layout-only format identity: --emit-ir and --emit-mic3 byte-identity.
- `intrinsic_id_table_lint.py` (~1760 tok, huge) — Native intrinsic id table contract for main.mind (pure text analysis).
- `lockstep_lint.py` (~4258 tok, huge) — lockstep_lint.py -- native-ELF walker lockstep linter for the pure-MIND self-host compiler.
- `match_struct_smoke.py` (~1350 tok, large)
- `method_callee_smoke.py` (~1390 tok, large)
- `method_calls_smoke.py` (~1334 tok, large)
- `mic3_flip_smoke.py` (~4048 tok, huge)
- `mic3_oracle_smoke.py` (~1060 tok, large) — mic@3 self-host convergence — Phase 0 gate: the Rust oracle.
- `mic3_primitives_smoke.py` (~22585 tok, huge) — mic@3 self-host convergence — Phase 1 gate: pure-MIND ULEB128 / zigzag.
- `mindfuzz_self_host.py` (~11993 tok, huge)
- `_mindfuzz_twin_emit.py` (~4717 tok, huge) — Twin-emitter half of the self-host differential fuzzer.
- `mod_operator_smoke.py` (~2894 tok, huge)
- `multi_let_smoke.py` (~1539 tok, huge)
- `native_cabi_driver.mind` (~308 tok, medium) — Author: STARGA Inc. Diagnostic stdin wrapper; no C imports or lane admission.
- `native_cabi_registry.md` (~1708 tok, huge) — Typed native C ABI registry (candidate metadata slice)
- `native_cabi_registry_smoke.py` (~1987 tok, huge) — Author: STARGA Inc. Diagnostic host orchestration for the MIND MCABI001 registry.
- `native_c_call_emit_check.py` (~3683 tok, huge) — Fixed scalar call emission: frozen metadata refusals and real ABI execution.
- `native_c_call_emit_driver.mind` (~618 tok, large) — Copyright 2026 STARGA Inc.
- `native_c_call_emit.mind` (~1514 tok, huge) — Copyright 2026 STARGA Inc.
- `native_c_call_plan_check.py` (~2812 tok, huge) — Frozen scalar SysV plan cases, compared with native and fresh MLIR drivers.
- `native_c_call_plan_driver.mind` (~499 tok, medium) — Copyright 2026 STARGA Inc.
- `native_c_call_plan.mind` (~1394 tok, large) — Copyright 2026 STARGA Inc.
- `native_c_variadic_emit_check.py` (~1675 tok, huge) — Capture bytes or delegate transport. All call assertions execute in MIND."""
- `native_c_variadic_emit_driver.mind` (~2145 tok, huge) — Copyright 2026 STARGA Inc.
- `native_c_variadic_emit.mind` (~1736 tok, huge) — Copyright 2026 STARGA Inc.
- `native_c_variadic_plan_check.py` (~1237 tok, large) — Capture bytes or delegate transport. All plan assertions execute in MIND."""
- `native_c_variadic_plan_driver.mind` (~1656 tok, huge) — Copyright 2026 STARGA Inc.
- `native_c_variadic_plan.mind` (~1624 tok, huge) — Copyright 2026 STARGA Inc.
- `native_c_variadic_verify.mind` (~381 tok, medium) — Copyright 2026 STARGA Inc.
- `native_dynamic_allocator_cases.tsv` (~336 tok, medium) — # Frozen before implementation/build; exact expected stdout is empty. exit70 std
- `native_dynamic_allocator_check.py` (~3465 tok, huge) — External diagnostic transport for the MIND allocator emitter/runtime.
- `native_dynamic_allocator_driver.mind` (~176 tok, small) — Copyright 2026 STARGA Inc. Standalone diagnostic, no compiler admission.
- `native_dynamic_allocator.md` (~2366 tok, huge) — Bounded dynamic allocator component
- `native_dynamic_allocator.mind` (~5402 tok, huge) — Copyright 2026 STARGA Inc.
- `native_dynamic_allocator_mutations.tsv` (~119 tok, small) — id	change	case	expected_original_exit
- `native_dynamic_allocator_ownership_cases.tsv` (~122 tok, small) — # Supplemental ownership-before-size contract; original39-case catalogue unchang
- `native_dynamic_fixture.mind` (~1736 tok, huge) — Copyright 2026 STARGA Inc.
- `native_dynamic_image_cases.tsv` (~760 tok, large) — id	name	category	runtime_exit	stdout_hex
- `native_dynamic_image_check.py` (~6704 tok, huge) — Bounded standalone image-extension diagnostics, never compiler qualification."""
- `native_dynamic_image_expectation_corrections.tsv` (~44 tok, tiny) — id	field	original	corrected	reason
- `native_dynamic_image_fixture.mind` (~3446 tok, huge) — Copyright 2026 STARGA Inc.
- `native_dynamic_image.md` (~3173 tok, huge) — Dynamic image writer with checked runtime mappings
- `native_dynamic_image.mind` (~3470 tok, huge) — Copyright 2026 STARGA Inc.
- `native_dynamic_image_mutations.tsv` (~74 tok, small) — name	case	expected_detection
- `native_dynamic_layout_cases.tsv` (~2646 tok, huge) — id	request_i64	expected_exit	output_delta	expected_i64
- `native_dynamic_layout_check.py` (~2975 tok, huge) — Diagnostic transport for the standalone MIND layout planner, not admission.
- `native_dynamic_layout_driver.mind` (~469 tok, medium) — Copyright 2026 STARGA Inc. Diagnostic wrapper, not a compiler entry point.
- `native_dynamic_layout.md` (~2529 tok, huge) — Checked dynamic MIND image layout, version 1
- `native_dynamic_layout.mind` (~1459 tok, large) — Copyright 2026 STARGA Inc.
- `native_dynamic_legacy_outputs.tsv` (~5637 tok, huge) — scope	case	exit	stdout_sha256	stderr_sha256
- `native_dynamic_mic_cases.tsv` (~184 tok, small) — id	name	stdout_hex
- `native_dynamic_mic_compatibility.md` (~1255 tok, large) — Dynamic writer MIC compatibility
- `native_dynamic_mic_fixture.mind` (~1033 tok, large) — Copyright 2026 STARGA Inc.
- `native_dynamic_mic_mutations.tsv` (~75 tok, small) — name	function	case	expected_detection
- `native_dynamic.mind` (~4642 tok, huge) — Copyright 2026 STARGA Inc.
- `native_dynamic_writer_check.py` (~3656 tok, huge) — Native MIND dynamic ELF writer diagnostics; not compiler qualification.
- `native_emission_context.md` (~3874 tok, huge) — Explicit target context and source-lowered module parts
- `native_emission_context.mind` (~1271 tok, large) — Copyright 2026 STARGA Inc.
- `native_fixed_scope_manifest.mind` (~4926 tok, huge) — Copyright 2026 STARGA Inc.
- `native_io_parity.md` (~1400 tok, large) — Four-argument native I/O parity
- `native_lane_op_differential_smoke.py` (~7769 tok, huge) — Seeded three-path differential for the admitted native-lane scalar operators.
- `native_lane_value_shapes_smoke.py` (~14771 tok, huge) — native-lane-v1 value shapes: the bridge path, a marked compiler, and MLIR.
- `native_libc_abi_rows.tsv` (~644 tok, large) — # selfhost column: the native self-host route verdict, the seed's own. accept = 
- `native_libc_binding_check.py` (~7867 tok, huge) — Source-bound libc declaration diagnostics; no call/production authority."""
- `native_libc_binding_contracts.tsv` (~853 tok, large) — id	name	result	params	library_keys	actions	roles	version	kind	fixed_prefix	noret
- `native_libc_binding_fixture.mind` (~1063 tok, large) — Copyright 2026 STARGA Inc. Diagnostic binary transport, never admission.
- `native_libc_binding.md` (~3820 tok, huge) — Source-bound libc declaration metadata
- `native_libc_binding.mind` (~16144 tok, huge) — Copyright 2026 STARGA Inc.
- `native_libc_binding_mutations.tsv` (~147 tok, small) — name	source_change	required_detection
- `native_libc_io_check.py` (~5536 tok, huge) — Source-bound fixed libc I/O thunk diagnostics; no production admission."""
- `native_libc_io_fixture.mind` (~2121 tok, huge) — Copyright 2026 STARGA Inc.
- `native_libc_io.md` (~2089 tok, huge) — Fixed libc intrinsic I/O component
- `native_libc_io.mind` (~3137 tok, huge) — Copyright 2026 STARGA Inc.
- `native_libc_startup_cases.tsv` (~362 tok, medium) — # Predeclared standalone startup contract. stderr empty except documented loader
- `native_libc_startup_check.py` (~6413 tok, huge) — Actual dynamic-loader diagnostics for the standalone MIND startup adapter."""
- `native_libc_startup_fixture.mind` (~2550 tok, huge) — Copyright 2026 STARGA Inc.
- `native_libc_startup.md` (~2589 tok, huge) — Standalone libc startup adapter
- `native_libc_startup.mind` (~1864 tok, huge) — Copyright 2026 STARGA Inc.
- `native_libc_startup_mutations.tsv` (~134 tok, small) — id	change	case	expected_detection
- `native_pointer_origin_cases.json` (~3497 tok, huge) — Keys: version, author, status, scope, source_catalogues
- `native_posix.md` (~2022 tok, huge) — Linux x86-64 POSIX substrate
- `native_scoped_source_driver.mind` (~2228 tok, huge) — STARGA Inc. NSF_STDIN_V2 source compiler transport.
- `native_scoped_source_image.md` (~1925 tok, huge) — Scoped fixed C calls: NSF/1 candidate
- `native_scoped_source_image.mind` (~5023 tok, huge) — Copyright 2026 STARGA Inc.
- `native_source_driver_check.md` (~2601 tok, huge) — Reproducing the source-input adapter diagnostic
- `native_source_driver_check.py` (~5814 tok, huge) — Explicit-instrument NSI_STDIN_V1 diagnostic, not compiler qualification.
- `native_source_driver.md` (~906 tok, large) — Native source-input adapter
- `native_source_driver.mind` (~2228 tok, huge) — STARGA Inc. NSI_STDIN_V1 source compiler transport.
- `native_source_image.md` (~2374 tok, huge) — Source-lowered Linux libc image candidate
- `native_source_image.mind` (~3979 tok, huge) — Copyright 2026 STARGA Inc.
- `native_stream_io_case_corrections.json` (~5416 tok, huge) — Keys: original_contract, original_sha256, corrected_contract, corrected_sha256, changes
- `native_stream_io_cases.json` (~3715 tok, huge) — Keys: contract, cases
- `native_stream_io_cases_v2.json` (~4374 tok, huge) — Keys: contract, cases
- `native_stream_io_cases_v3.json` (~4377 tok, huge) — Keys: contract, cases
- `native_stream_io_check.py` (~1413 tok, large) — Source-lowered MIND stream-I/O diagnostic transport, not compiler admission.
- `native_stream_io.md` (~2162 tok, huge) — Source-lowered stream I/O candidate
- `native_stream_io_mutation_observations.json` (~2000 tok, huge) — Keys: candidate_main_sha256, recipe_sha256, results, limitations
- `native_stream_io_mutations.json` (~459 tok, medium)
- `nb_alloc_select.mind` (~3425 tok, huge) — Copyright 2026 STARGA Inc.
- `nb_arr_fold.mind` (~1181 tok, large) — Copyright 2026 STARGA Inc.
- `nb_ast.mind` (~2689 tok, huge) — Copyright 2026 STARGA Inc.
- `nb_branch_memo.mind` (~3097 tok, huge) — Copyright 2026 STARGA Inc.
- `nb_cpu_features.mind` (~1788 tok, huge) — Copyright 2026 STARGA Inc.
- `nb_emit_core.mind` (~3065 tok, huge) — Copyright 2026 STARGA Inc.
- `nb_exit_carry.mind` (~696 tok, large) — Copyright 2026 STARGA Inc.
- `nb_exit.mind` (~2007 tok, huge) — Copyright 2026 STARGA Inc.
- `nb_extern_abi.mind` (~1119 tok, large) — Copyright 2026 STARGA Inc.
- `nb_float_lit2.mind` (~2941 tok, huge) — Copyright 2026 STARGA Inc.
- `nb_float_lit.mind` (~2520 tok, huge) — Copyright 2026 STARGA Inc.
- `nb_group_bankw.mind` (~1731 tok, huge) — Copyright 2026 STARGA Inc.
- `nb_group_epi.mind` (~2316 tok, huge) — Copyright 2026 STARGA Inc.
- `nb_group_fuse32.mind` (~893 tok, large) — Copyright 2026 STARGA Inc.
- `nb_group_fuse.mind` (~2106 tok, huge) — Copyright 2026 STARGA Inc.
- `nb_group_guard.mind` (~646 tok, large) — Copyright 2026 STARGA Inc.
- `nb_group_i32w.mind` (~3787 tok, huge) — Copyright 2026 STARGA Inc.
- `nb_group_sum_acc.mind` (~2372 tok, huge) — Copyright 2026 STARGA Inc.
- `nb_group_sum_bank.mind` (~2778 tok, huge) — Copyright 2026 STARGA Inc.
- `nb_group_sum_enc.mind` (~3015 tok, huge) — Copyright 2026 STARGA Inc.
- `nb_group_tail32.mind` (~2135 tok, huge) — Copyright 2026 STARGA Inc.
- `nb_group_terms.mind` (~3299 tok, huge) — Copyright 2026 STARGA Inc.
- `nb_group_walk.mind` (~2510 tok, huge) — Copyright 2026 STARGA Inc.
- `nb_intern_names2.mind` (~1569 tok, huge) — Copyright 2026 STARGA Inc.
- `nb_intern_names.mind` (~2059 tok, huge) — Copyright 2026 STARGA Inc.
- `nb_lane_keys.mind` (~2754 tok, huge) — Copyright 2026 STARGA Inc.
- `nb_lane_report.mind` (~2446 tok, huge) — Copyright 2026 STARGA Inc.
- `nb_lane_rules.mind` (~684 tok, large) — Copyright 2026 STARGA Inc.
- `nb_lane_scan.mind` (~3280 tok, huge) — Copyright 2026 STARGA Inc.
- `nb_lane_str2.mind` (~3898 tok, huge) — Copyright 2026 STARGA Inc.
- `nb_lane_str.mind` (~2861 tok, huge) — Copyright 2026 STARGA Inc.
- `nb_lastdef.mind` (~2516 tok, huge) — Copyright 2026 STARGA Inc.
- `nb_lex_scan.mind` (~1989 tok, huge) — Copyright 2026 STARGA Inc.
- `nb_lower_loop.mind` (~934 tok, large) — Copyright 2026 STARGA Inc.
- `nb_measure.mind` (~2774 tok, huge) — Copyright 2026 STARGA Inc.
- `nb_narrow_rebind.mind` (~3320 tok, huge) — Copyright 2026 STARGA Inc.
- `nb_opt_level.mind` (~681 tok, large) — Copyright 2026 STARGA Inc.
- `nb_opt_licm_check.mind` (~2691 tok, huge) — Copyright 2026 STARGA Inc.
- `nb_opt_licm_facts.mind` (~2438 tok, huge) — Copyright 2026 STARGA Inc.
- `nb_opt_licm.mind` (~3219 tok, huge) — Copyright 2026 STARGA Inc.
- `nb_opt_licm_scan.mind` (~1591 tok, huge) — Copyright 2026 STARGA Inc.
- `nb_part_dispatch.mind` (~2458 tok, huge) — Copyright 2026 STARGA Inc.
- `nb_prim_desc.mind` (~1914 tok, huge) — Copyright 2026 STARGA Inc.
- `nb_print.mind` (~1678 tok, huge) — Copyright 2026 STARGA Inc.
- `nb_shm.mind` (~1596 tok, huge) — Copyright 2026 STARGA Inc.
- `nb_stack_budget.mind` (~1194 tok, large) — Copyright 2026 STARGA Inc.
- `nb_strtab_index.mind` (~2059 tok, huge) — Copyright 2026 STARGA Inc.
- `nb_synth_sha2.mind` (~1773 tok, huge) — Copyright 2026 STARGA Inc.
- `nb_synth_sha.mind` (~1547 tok, huge) — Copyright 2026 STARGA Inc.
- `nbt_extern_keys.mind` (~2134 tok, huge) — Copyright 2026 STARGA Inc.
- `nbt_extern_scan.mind` (~2878 tok, huge) — Copyright 2026 STARGA Inc.
- `nbt_fvc_names.mind` (~1066 tok, large) — Copyright 2026 STARGA Inc.
- `nbt_jump_table.mind` (~866 tok, large) — Copyright 2026 STARGA Inc.
- `nb_value_if_memo.mind` (~668 tok, large) — Copyright 2026 STARGA Inc.
- `nb_x86_vex.mind` (~3282 tok, huge) — Copyright 2026 STARGA Inc.
- `nb_x86_ymm_bank_checks.mind` (~1721 tok, huge) — Copyright 2026 STARGA Inc.
- `nb_x86_ymm_bank.mind` (~2624 tok, huge) — Copyright 2026 STARGA Inc.
- `nb_x86_ymm_bank_walk.mind` (~2120 tok, huge) — Copyright 2026 STARGA Inc.
- `nb_x86_ymm_epi.mind` (~972 tok, large) — Copyright 2026 STARGA Inc.
- `nbx_bind.mind` (~3149 tok, huge) — Copyright 2026 STARGA Inc.
- `nbx_cuda_rows2.mind` (~3233 tok, huge) — Copyright 2026 STARGA Inc.
- `nbx_libc_rows.mind` (~736 tok, large) — Copyright 2026 STARGA Inc.
- `now_ns_smoke.py` (~1013 tok, large) — # Copyright 2025 STARGA Inc.
- `option_netverify.py` (~2340 tok, huge) — # Canonical independent net-verify harness for SINGLE-PAYLOAD ENUMS (the
- `oracle_parity_lint.py` (~4516 tok, huge)
- `param_types_smoke.py` (~1273 tok, large)
- `PARTS.md` (~6464 tok, huge) — Compiler part files
- `_process_starts.py` (~9904 tok, huge) — The start rule of the harness, the static half of the one stdin start helper (start_child in _selfhost_sources.py).
- `_ref_add.note` (~16 tok, tiny) — 5de995a99c75ea86bf680d8369bf903c8ced86c144b92eeae4664078225b98e1
- `_ref_if_ret.note` (~16 tok, tiny) — 37fdad11eeb3d8763df53419f13dd9031ada3df2308374e1b62ee547685c74ce
- `_ref_main.note` (~16 tok, tiny) — 9b0cb46e1365e85979120f49caf6fdf5cadcd29a8a7a29f8859e94cc4024e2cc
- `ref_netverify.py` (~1493 tok, large) — # Canonical independent net-verify harness for i64 references in the native-ELF backend.
- `_ref_recursion.note` (~16 tok, tiny) — a7fc331ef6d24dd69ff9c0aa2720e3dbfe1d7f472ef3872ba0d22f9b2fa53f20
- `_ref_struct_field.note` (~16 tok, tiny) — d73ada9d0a64d7fe0b2e8f2ec58bbe469a0a9a94ec493b2ac45b6933e6e03162
- `_ref_value_if.note` (~16 tok, tiny) — 0dcc62ae9e1f2b72eafa8814156504d627d541f2d5feabe12cf2b9a28947787a
- `rh_f64_aggregate_canary_smoke.py` (~1622 tok, huge) — RH f64-aggregate surface canary (RH_REQUIRED_F64_AGGREGATE_SURFACE).
- `ri_d1_frozen_profile_gate.py` (~4909 tok, huge) — RI-D1 readiness gate (task #313): prove `mindc build --backend native` is READY to be
- `self_host_alias_controls_smoke.py` (~1478 tok, large) — Focused native alias-environment controls.
- `_selfhost_alloc_rows.py` (~5636 tok, huge) — Rows of the allocator-class smoke (self_host_native_alloc_bounded_smoke.py): the default-profile
- `self_host_andor_precedence_smoke.py` (~1709 tok, huge) — Front-end PARITY battery for `&&` / `||` precedence + short-circuit.
- `self_host_andor_smoke.py` (~2161 tok, huge) — Permanent battery for the self-host `&&` / `||` short-circuit operators.
- `self_host_arena_growth_smoke.py` (~1413 tok, large)
- `self_host_args_from_os_smoke.py` (~1375 tok, large)
- `selfhost_argv_driver.mind` (~1320 tok, large) — ===========================================================================
- `self_host_argv_smoke.py` (~1156 tok, large)
- `self_host_array_smoke.py` (~1425 tok, large)
- `self_host_arridx_mic3_smoke.py` (~1883 tok, huge)
- `self_host_body_smoke.py` (~3055 tok, huge)
- `self_host_bounded_profile_smoke.py` (~7768 tok, huge) — Bounded native memory profile gate: the compiler seed AND its programs load
- `self_host_carry_cap_smoke.py` (~1226 tok, large) — Cap-guard smoke for the self-host loop-carry / loop-frame scratch tables.
- `self_host_cast_mic3_smoke.py` (~4759 tok, huge)
- `self_host_check_driver_smoke.py` (~2889 tok, huge)
- `self_host_continue_smoke.py` (~4707 tok, huge) — Regression gate for #308 (closes #286's break/continue fixtures): the pure-MIND
- `selfhost_driver.mind` (~7395 tok, huge) — ===========================================================================
- `self_host_dtype_tag_smoke.py` (~780 tok, large) — RI-B1 per-SSA dtype-tag gate (parser <-> nb_fp_* encoder connecting construct).
- `self_host_else_if_smoke.py` (~1735 tok, huge)
- `self_host_failclosed_smoke.py` (~9999 tok, huge) — self_host_failclosed_smoke.py — the fail-closed boundary of the pure-MIND
- `self_host_float_lit_exact_smoke.py` (~1368 tok, large) — CPU-as-oracle smoke for the C1 float-literal exactness guard.
- `self_host_for_smoke.py` (~11141 tok, huge) — Permanent battery for the self-host range-`for` loop.
- `_selfhost_identity.py` (~9099 tok, huge) — Two-compiler identity of the native-lane smokes (`--identity-vs`), its partitions and their self-tests,
- `_selfhost_identity_selftest.py` (~2926 tok, huge) — Self-tests of the identity partition rules (`--identity-vs`): the merged partition and the allocator
- `self_host_if_region_carry_smoke.py` (~5284 tok, huge) — Native-ELF smoke: i64 loop-carry through BRANCHED regions (Sub-step C).
- `self_host_ifret_chain_mic3_smoke.py` (~1749 tok, huge)
- `self_host_letpath_failclose_smoke.py` (~2243 tok, huge) — Self-host LET-PATH fail-closed smoke — pins the S1-collapse fail-open wall.
- `self_host_lockstep_smoke.py` (~2260 tok, huge) — SUB-STEP A lockstep smoke: the loop-carry frame COUNT and the loop-carry EMIT are
- `_selfhost_loop_reseed.py` (~4296 tok, huge) — Seed-provenance support for the LOOP harness: the legacy Rust-seeded
- `self_host_loop_smoke.md` (~2809 tok, huge) — Self-host LOOP gate contract
- `self_host_loop_smoke.py` (~9669 tok, huge) — Self-host LOOP reproduction and advancement gate.
- `self_host_matchscalar_mic3_smoke.py` (~1744 tok, huge)
- `self_host_match_smoke.py` (~1912 tok, huge)
- `self_host_mic3_float_valueif_smoke.py` (~1822 tok, huge)
- `self_host_mlir_smoke.py` (~1954 tok, huge)
- `self_host_narrow_let_mic3_smoke.py` (~1631 tok, huge)
- `self_host_narrow_param_smoke.py` (~12063 tok, huge) — Native-ELF smoke for narrow-width (i8/i16/i32) function PARAMETERS carried by a loop.
- `self_host_native_alloc_bounded_smoke.py` (~9548 tok, huge) — Bounded-profile allocator gate (W11c): grow-on-demand arena, C-runtime
- `self_host_native_autowrap_smoke.py` (~2461 tok, huge) — Roadmap C2 declared-width AUTO-WRAP driver — narrow-int (i8/i16/i32) `let` and
- `self_host_native_avx2_dot_f32_smoke.py` (~1682 tok, huge) — RI-B2-S13 (#108) — native-ELF PACKED-f32 SIMD via 256-bit AVX2 (VEX/YMM) STRICT-FP DOT.
- `self_host_native_blas_dot_i16_smoke.py` (~2899 tok, huge)
- `self_host_native_blas_dot_q16_smoke.py` (~1793 tok, huge)
- `self_host_native_cast_conv_smoke.py` (~1392 tok, large) — RI-B2 scalar-cast-conv rung (#108) — native-ELF int<->float `as`-cast chain.
- `self_host_native_cf_smoke.py` (~8299 tok, huge) — Native stage1: general control flow (HOME lowering) — three-way behavioural gate.
- `self_host_native_diag_smoke.py` (~7870 tok, huge) — RI-D1a gate (task #313): `mindc build --backend=native` must FAIL-CLOSED with an
- `self_host_native_div_trap_smoke.py` (~10002 tok, huge)
- `self_host_native_dot_f32_smoke.py` (~1455 tok, large) — RI-B2-S8 STEP C (#108) — native-ELF scalar STRICT-FP f32 DOT-PRODUCT.
- `self_host_native_dot_l1_q16_smoke.py` (~1207 tok, large) — RI-B2 L1-Q16 rung (#108) — native-ELF Q16.16 L1 distance.
- `self_host_native_elf_alignment_smoke.py` (~1921 tok, huge) — Self-host NATIVE-ELF ABI-alignment smoke (FIX #169).
- `self_host_native_elf_smoke.py` (~11024 tok, huge)
- `self_host_native_emission_context_smoke.py` (~8868 tok, huge) — Independent diagnostic transport for native emission contexts.
- `self_host_native_f64_intrinsics_smoke.py` (~4928 tok, huge) — W8c smoke: float builtins/intrinsics in the pure-MIND native compiler.
- `self_host_native_f64_lit_smoke.py` (~3084 tok, huge) — W8b smoke: correctly rounded decimal -> binary64 float literals in the pure-MIND
- `self_host_native_fp_binop_smoke.py` (~1220 tok, large) — RI-B1 nb_expr FLOAT-op-FLOAT arithmetic routing gate (zero MLIR/LLVM).
- `self_host_native_fp_call_smoke.py` (~5488 tok, huge) — RI-D2 S-C1: FLOAT call-RETURN dtype through the native-ELF general nb_expr lowering.
- `self_host_native_fp_expr_smoke.py` (~1145 tok, large) — RI-B1 nb_expr float-scalar routing gate (zero MLIR/LLVM).
- `self_host_native_fp_field_smoke.py` (~2016 tok, huge) — RI-D2 S-D FLOAT struct-FIELD READ dtype through native-ELF general lowering (zero MLIR).
- `self_host_native_fp_let_smoke.py` (~1333 tok, large) — RI-B1 (#107 follow-up) FLOAT dtype propagation ACROSS a LET binding (zero MLIR/LLVM).
- `self_host_native_fp_param_smoke.py` (~1528 tok, huge) — RI-D2 S-B: FLOAT fn-param dtype classification + SysV SSE-spill ABI (zero MLIR/LLVM).
- `self_host_native_fp_smoke.py` (~1161 tok, large) — RI-B1 native-ELF scalar-f64 gate (zero MLIR/LLVM).
- `self_host_native_gemm_i8_smoke.py` (~1173 tok, large) — RI-B2-S7 (#108) — native-ELF scalar int8 GEMM (matrix x matrix), byte-identity rung.
- `self_host_native_gemm_q16_smoke.py` (~1176 tok, large) — RI-B2-S7 (#108) — native-ELF scalar GEMM Q16.16 (matrix x matrix), byte-identity rung.
- `self_host_native_gemv_i16_smoke.py` (~1168 tok, large) — RI-B2-S6 (#108) — native-ELF scalar GEMV int16 (matrix x vector), byte-identity rung.
- `self_host_native_gemv_q16_smoke.py` (~1161 tok, large) — RI-B2-S6 (#108) — native-ELF scalar GEMV Q16.16 (matrix x vector), byte-identity rung.
- `self_host_native_genf32_smoke.py` (~1085 tok, large) — RI-B2-S8 STEP B (#108) — isolate the LCG f32 rounding BEFORE the dot.
- `self_host_native_intdot_i16_smoke.py` (~1140 tok, large) — RI-B2-S2 (#108) — native-ELF scalar int16 DOT-PRODUCT, FIRST byte-identity rung.
- `self_host_native_intdot_q16_smoke.py` (~1130 tok, large) — RI-B2-S4 (#108) — native-ELF scalar Q16.16 DOT-PRODUCT, byte-identity rung.
- `self_host_native_intdot_smoke.py` (~1184 tok, large) — RI-B2-S1 (#108) scalar i64 DOT-PRODUCT reduction native-ELF (zero MLIR/LLVM).
- `self_host_native_io_smoke.py` (~2582 tok, huge) — Author: STARGA Inc. Exact four-argument I/O parity and raw-ABI controls.
- `self_host_native_main_args_smoke.py` (~2442 tok, huge)
- `self_host_native_matmul_f32_v_smoke.py` (~1737 tok, huge) — RI-B2-S9 (#108) — native-ELF scalar STRICT-FP f32 GEMV (matmul-f32-v).
- `self_host_native_misc_census_smoke.py` (~3332 tok, huge) — Native stage1 census gate: top-level int consts (A8; string consts refuse), pointer casts (A2) and a
- `self_host_native_narrow_add_i8_smoke.py` (~2279 tok, huge) — C2 — native-ELF NARROW-INT (i8) WRAP ARITHMETIC, zero MLIR/LLVM.
- `self_host_native_narrow_arith_batch_smoke.py` (~2574 tok, huge) — C2 — native-ELF NARROW-INT WRAP ARITHMETIC batch: {sub,mul}xi8 + {add,mul}xi16.
- `self_host_native_narrowint_smoke.py` (~1675 tok, huge) — Roadmap C2 narrow-int native-ELF rung — user-reachable i8/i16/i32 truncating
- `self_host_native_narrow_paramret_smoke.py` (~2049 tok, huge) — Byte-behavior smoke for narrow-int (i8/i16/i32) PARAM + RETURN auto-wrap in the
- `self_host_native_narrowwrap_smoke.py` (~1692 tok, huge) — Roadmap C2 narrow-int native-ELF rung — user-reachable i8/i16/i32 two's-complement
- `self_host_native_process_smoke.py` (~2502 tok, huge) — Linux x86-64 process/clock/directory syscall parity, in disposable directories.
- `self_host_native_record_array_smoke.py` (~2456 tok, huge) — Focused pure-MIND native fixed-array record-field gate."""
- `self_host_native_record_trace_smoke.py` (~4825 tok, huge) — Canonical-trace contract for record-field reads, incl. the fixed-array case.
- `self_host_native_scalar_f32_smoke.py` (~1568 tok, huge) — Phase C1-remainder f32 rung — native-ELF scalar SINGLE-precision chain.
- `self_host_native_scalar_f64_smoke.py` (~1320 tok, large) — RI-B2 f64 rung (#108) — native-ELF scalar STRICT-FP f64 CHAIN.
- `self_host_native_scalar_narrow_smoke.py` (~1829 tok, huge) — RI-D / #10 native-float SLICE 1a — native-ELF SATURATING f64 -> signed NARROW
- `self_host_native_scast_smoke.py` (~1722 tok, huge)
- `self_host_native_shadow_scope_smoke.py` (~10698 tok, huge) — Lexical scoping of nested bindings in the native seed (block / loop-body shadows).
- `self_host_native_simd_dot_f32_smoke.py` (~1625 tok, huge) — RI-B2-S11 (#108) — native-ELF PACKED-f32 SIMD (SSE, 128-bit) STRICT-FP DOT-PRODUCT.
- `self_host_native_simd_dot_i16_smoke.py` (~1253 tok, large) — RI-B2-S12 (#108) — native-ELF PACKED-int16 SIMD DOT-PRODUCT, byte-identity rung.
- `self_host_native_simd_dot_q16_smoke.py` (~1506 tok, huge) — RI-B2-S10 (#108) — native-ELF PACKED-SIMD Q16.16 DOT-PRODUCT, byte-identity rung.
- `self_host_native_string_escape_smoke.py` (~3318 tok, huge) — Source-bound native/MLIR string values and bounded decoder diagnostics.
- `self_host_native_string_smoke.py` (~7073 tok, huge)
- `self_host_native_sys_smoke.py` (~4900 tok, huge) — Linux x86-64 filesystem parity: current-source stage1 versus Rust/MLIR.
- `self_host_native_tensor_batchsum_smoke.py` (~2391 tok, huge) — C4-T6 — native-ELF 3-D BATCHED SUM (i64), zero MLIR/LLVM. The FIRST N-D
- `self_host_native_tensor_bcastadd_smoke.py` (~1720 tok, huge) — C4-T5 — native-ELF tensor ROW-VECTOR BROADCAST ADD (i64), zero MLIR/LLVM.
- `self_host_native_tensor_colsum_smoke.py` (~2114 tok, huge) — C4-T5 — native-ELF tensor COLUMN REDUCTION (i64), zero MLIR/LLVM.
- `self_host_native_tensor_dot_smoke.py` (~1216 tok, large) — C4-T2 — native-ELF tensor DOT PRODUCT (i64), zero MLIR/LLVM.
- `self_host_native_tensor_ewadd_f64_smoke.py` (~1814 tok, huge) — C4-T4 — native-ELF float64 TENSOR element-wise-add + STRICT-SEQUENTIAL reduce.
- `self_host_native_tensor_ewadd_smoke.py` (~1137 tok, large) — C4-T1 — native-ELF tensor ELEMENT-WISE ADD (i64), zero MLIR/LLVM.
- `self_host_native_tensor_ewmul_smoke.py` (~2066 tok, huge) — C4-T5 — native-ELF tensor ELEMENT-WISE MULTIPLY (i64), zero MLIR/LLVM.
- `self_host_native_tensor_matmul_smoke.py` (~1742 tok, huge) — C4-T3 — native-ELF tensor MATMUL (i64), zero MLIR/LLVM.
- `self_host_native_tensor_maxrowmax_smoke.py` (~2473 tok, huge) — C4-T6 — native-ELF tensor ROW MAX-REDUCTION (i64), zero MLIR/LLVM.
- `self_host_native_tensor_relu_smoke.py` (~2214 tok, huge) — C4-T6 — native-ELF tensor ELEMENTWISE RELU (max(x,0)), zero MLIR/LLVM.
- `self_host_native_tensor_rowmin_smoke.py` (~2373 tok, huge) — C4-T6 — native-ELF tensor ROW MIN-REDUCTION (i64), zero MLIR/LLVM.
- `self_host_native_tensor_rowsum_smoke.py` (~1977 tok, huge) — C4-T4 — native-ELF tensor ROW REDUCTION (i64), zero MLIR/LLVM.
- `self_host_native_tensor_transpose_smoke.py` (~2243 tok, huge) — C4-T4 — native-ELF tensor TRANSPOSE (i64), zero MLIR/LLVM.
- `self_host_native_toplevel_assign_smoke.py` (~5569 tok, huge) — INDEPENDENCE_ROADMAP Phase-C follow-up — TOP-LEVEL STRAIGHT-LINE i64 REASSIGN
- `self_host_native_ucast_smoke.py` (~1473 tok, large)
- `self_host_native_write8_smoke.py` (~948 tok, large) — RI-B2-S2 STEP A (#108) — de-risk the C1 "emit 8 LE bytes to stdout + hash" gate.
- `self_host_native_write_f32_smoke.py` (~834 tok, large) — RI-B2-S8 STEP A (#108) — de-risk the raw-f32-bytes harness on a KNOWN f32.
- `self_host_not_smoke.py` (~1481 tok, large)
- `self_host_open_smoke.py` (~1353 tok, large)
- `self_host_param_mutation_smoke.py` (~1342 tok, large) — CPU-as-oracle smoke for the param-mutation fix (nb_expr ident arm: consult the
- `selfhost_so_provenance_smoke.py` (~2391 tok, huge) — Gate: `resolve_so()` must QUALIFY the `.so` it returns and REFUSE a stale one.
- `_selfhost_so.py` (~5461 tok, huge) — Shared self-host `.so` resolver for the examples/mindc_mind smokes.
- `selfhost_source_bundle_check.py` (~4228 tok, huge) — Captured compiler-source regression gate. Bootstrap orchestration only.
- `selfhost_source_bundle.md` (~1570 tok, huge) — Captured compiler source bundles
- `self_host_source_note_smoke.py` (~3307 tok, huge) — Native MIND regression for PT_NOTE tagging with trailing string rodata.
- `_selfhost_sources.py` (~10775 tok, huge) — Immutable source views for internal compiler bootstrap orchestration.
- `self_host_standalone_driver_smoke.py` (~3328 tok, huge)
- `_selfhost_string_rows.py` (~12355 tok, huge) — Runner of the string-rule acceptance rows (tests/fixtures/native_lane_string_rows, README.md there):
- `self_host_struct_return_smoke.py` (~4142 tok, huge) — self_host_struct_return_smoke.py — regression lock for STRUCT-BY-VALUE
- `self_host_tc_classify_error_code_smoke.py` (~2486 tok, huge) — CPU-as-oracle smoke for the pure-MIND classify_error_code router.
- `self_host_tc_class_mismatch_smoke.py` (~1167 tok, large) — CPU-as-oracle smoke for the pure-MIND E2015 int<->float class-mismatch rule.
- `self_host_tc_class_rules_smoke.py` (~1889 tok, huge) — CPU-as-oracle smoke for the pure-MIND E2010/E2011/E2013/E2016 class rules.
- `self_host_tc_decl_names_smoke.py` (~3651 tok, huge) — CPU-as-oracle smoke for the pure-MIND D1 module DECL-NAME SET.
- `self_host_tc_fixed_bytes_into_vec_smoke.py` (~1438 tok, large) — CPU-as-oracle smoke for the pure-MIND E2006 FIXED_BYTES_INTO_VEC rule (Bug #38).
- `self_host_tc_fn_value_call_smoke.py` (~7013 tok, huge) — CPU-as-oracle smoke for the pure-MIND B1/D4 CAPSTONE — the FULL E2012 rule.
- `self_host_tc_let_class_mismatch_smoke.py` (~1319 tok, large) — CPU-as-oracle smoke for the pure-MIND E2015 LET_CLASS_MISMATCH let/assign rule.
- `self_host_tc_let_infer_ident_smoke.py` (~2882 tok, huge) — self_host_tc_let_infer_ident_smoke — three-leg gate for the T2 port.
- `self_host_tc_let_infer_smoke.py` (~2132 tok, huge) — self_host_tc_let_infer_smoke — three-leg gate for the T1 type-inference port.
- `self_host_tc_narrowing_smoke.py` (~1042 tok, large)
- `self_host_tc_scope_frame_smoke.py` (~5756 tok, huge) — CPU-as-oracle smoke for the pure-MIND B1/D2 NESTED SCOPE-FRAME WALK.
- `self_host_tc_self_host_only_call_smoke.py` (~5098 tok, huge) — CPU-as-oracle smoke for the pure-MIND E2024 self-host-only-call rule.
- `self_host_tc_shape_annot_compat_smoke.py` (~1943 tok, huge) — CPU-as-oracle smoke for the pure-MIND shape annotation-compat rule.
- `self_host_tc_shape_rules_smoke.py` (~2149 tok, huge) — CPU-as-oracle smoke for the pure-MIND E2005/E2101/E2102/E2103 shape rules + E2023 reserved-prefix rule.
- `self_host_tc_std_export_smoke.py` (~3554 tok, huge) — CPU-as-oracle smoke for the pure-MIND D3 STD-SURFACE EXPORT NAME SET.
- `self_host_tc_undeclared_assign_smoke.py` (~7241 tok, huge) — CPU-as-oracle smoke for the pure-MIND B1 E2009 rule — undeclared assign.
- `self_host_tc_unknown_call_smoke.py` (~8550 tok, huge) — CPU-as-oracle smoke for the pure-MIND B1 E2003 rule — unknown call.
- `self_host_tc_unknown_ident_smoke.py` (~11390 tok, huge) — CPU-as-oracle smoke for the pure-MIND B1 E2002 rule — unknown identifier.
- `self_host_tc_unknown_variant_smoke.py` (~2233 tok, huge) — CPU-as-oracle smoke for the pure-MIND E2008 unknown-enum-variant rule.
- `self_host_value_if_expr_smoke.py` (~1576 tok, huge)
- `sha256_hash_smoke.py` (~1290 tok, large) — # Copyright 2025 STARGA Inc.
- `_shadow_scope_lang.py` (~3624 tok, huge) — Mini-language of the nested-binding scoping smoke (self_host_native_shadow_scope_smoke.py).
- `smoke_wiring_lint.py` (~9182 tok, huge) — smoke_wiring_lint.py — machine-checked contract for WHERE each
- `SMOKE_WIRING.tsv` (~7512 tok, huge) — # SMOKE_WIRING.tsv — the CHECKED contract for where each examples/mindc_mind smo
- `source_note_driver.mind` (~349 tok, medium) — Diagnostic entry point, composed with the real compiler and stdlib by the
- `stdlib_manifest_lint.py` (~9497 tok, huge) — stdlib_manifest_lint.py — machine-checked contract for WHICH std/*.mind
- `_stdlib_manifest.py` (~2226 tok, huge) — Shared reader for the committed std/*.mind manifest.
- `string_borrowed_check.md` (~937 tok, large) — Borrowed String append checks
- `string_borrowed_check.py` (~2794 tok, huge) — Source-bound borrowed String gate. MIND body and C shim are separate legs."""
- `string_escape_driver.mind` (~531 tok, large)
- `struct_fields_smoke.py` (~1076 tok, large)
- `struct_lit_smoke.py` (~2240 tok, huge)
- `tc_differential_fuzz.py` (~13353 tok, huge) — tcdiff — differential fuzzer for the self-host source-position tc-rule ports.
- `tc_std_surface_table.mind` (~4017 tok, huge) — Copyright 2026 STARGA Inc.
### `examples/mindc_mind/testdata/backend_native_bridge/`

- `add.elf` (binary, 492 bytes)
- `add.mind` (~24 tok, tiny) — fn add(a: i64, b: i64) -> i64 {
- `const_return.elf` (binary, 397 bytes)
- `const_return.mind` (~9 tok, tiny) — fn main() -> i64 {
- `local_call.elf` (binary, 474 bytes)
- `local_call.mind` (~23 tok, tiny) — fn helper(x: i64) -> i64 {
- `MANIFEST.txt` (~497 tok, medium) — # Pure-MIND native-ELF oracle (RI-D slice 1, task #110). Frozen from the pure-MI
- `while_sum.elf` (binary, 581 bytes)
- `while_sum.mind` (~37 tok, tiny) — fn main() -> i64 {
### `examples/mindc_mind/testdata/compiler_parts/`

- `part_dispatch_check_driver.mind` (~3816 tok, huge) — Copyright 2026 STARGA Inc.
- `part_dispatch_group_driver.mind` (~653 tok, large) — Copyright 2026 STARGA Inc.
### `examples/mindc_mind/testdata/`

- `dtk_plan_parity_smoke.py` (~2452 tok, huge)
- `h2b_exclusion_pins.json` (~2404 tok, huge) — Keys: comment, entries
- `h2b_fmt_identity_files.txt` (~237 tok, medium) — examples/mindc_mind/native_fixed_scope_manifest.mind
- `lane_keys.txt` (~156 tok, small) — 16 lane.source.string
### `examples/mindc_mind/testdata/mindfuzz_regress/`

- `alias_break.mind` (~364 tok, medium) — Regression fixture: loop-carried ALIAS-BREAK (src/eval/lower.rs While arm).
- `block_local_seg_then_let.mind` (~89 tok, small) — Regression class: block-local names (a then-block `let` shadowing an outer
- `block_local_then_assign.mind` (~68 tok, small) — Regression class: block-local names (a then-block `let` shadowing an outer
- `branch_alias_value_if_then.mind` (~75 tok, small) — Regression class: an if-branch alias value (`b = a`) feeding a later loop
### `examples/mindc_mind/testdata/mindfuzz_regress_failclosed/`

- `rv3_e7_outer_else_nonempty.mind` (~200 tok, medium) — Regression class (twin-emitter review round 3): read of an earlier loop's exit id inside a while nested in an else-branc
- `rv3_e8_no_outer_else.mind` (~189 tok, small) — Regression class (twin-emitter review round 3): read of an earlier loop's exit id inside a while nested in an else-branc
- `rv3_h1_pad1.mind` (~181 tok, small) — Regression class (twin-emitter review round 3): phantom alias-break ids (padding variant).
- `rv3_h2_pad2.mind` (~188 tok, small) — Regression class (twin-emitter review round 3): phantom alias-break ids (padding variant).
- `rv3_pcand_cond_read.mind` (~84 tok, small) — Regression class (twin-emitter review round 3): alias-break over a PARAMETER candidate (cond read).
- `rv3_pcand_min.mind` (~77 tok, small) — Regression class (twin-emitter review round 3): alias-break over a PARAMETER candidate.
- `rv3_pcand_nested.mind` (~110 tok, small) — Regression class (twin-emitter review round 3): alias-break over a PARAMETER candidate (nested loop).
- `rv3_r20_helper_only.mind` (~223 tok, medium) — Regression class (twin-emitter review round 3): minimized blind-review reproducer.
- `rv3_rand_110_20_m0.mind` (~415 tok, medium) — Regression class (twin-emitter review round 3): random-program reproducer (blind review round 2).
- `rv3_rand_412_4_m2.mind` (~149 tok, small) — Regression class (twin-emitter review round 3): random-program reproducer (blind review round 2).
- `rv3_rand_530_61_m0_main_only.mind` (~311 tok, medium) — Regression class (twin-emitter review round 3): minimized blind-review reproducer.
- `rv3_rand_530_61_m0.mind` (~521 tok, large) — Regression class (twin-emitter review round 3): random-program reproducer (blind review round 2).
- `rv3_reg_min.mind` (~189 tok, small) — Regression class (twin-emitter review round 3): read of an earlier loop's exit id inside a while nested in an else-branc
- `rv3_s3_412_helper.mind` (~98 tok, small) — Regression class (twin-emitter review round 3): alias-break over a PARAMETER candidate (+ native shadow value).
### `examples/mindc_mind/testdata/mindfuzz_regress/`

- `idcollide_copy_chain3.mind` (~138 tok, small) — Regression: loop alias-break copy vids (base0 + 2k + 1) collided with param
- `idcollide_copy_vs_param_k1.mind` (~135 tok, small) — Regression: loop alias-break copy vids (base0 + 2k + 1) collided with param
- `idcollide_copy_vs_param_noread.mind` (~119 tok, small) — Regression: loop alias-break copy vids (base0 + 2k + 1) collided with param
- `idcollide_copy_vs_param_p11.mind` (~119 tok, small) — Regression: loop alias-break copy vids (base0 + 2k + 1) collided with param
- `idcollide_nested_while_n_4_3.mind` (~175 tok, small) — Regression: nested while inside a top-level while in a >=4-param fn. The
- `idcollide_nested_while_n_5_3.mind` (~178 tok, small) — Regression: nested while inside a top-level while in a >=4-param fn. The
- `idcollide_nested_while_n_5_4.mind` (~178 tok, small) — Regression: nested while inside a top-level while in a >=4-param fn. The
- `idcollide_nested_while_n_6_3.mind` (~181 tok, small) — Regression: nested while inside a top-level while in a >=4-param fn. The
- `idcollide_nested_while_n_6_4.mind` (~181 tok, small) — Regression: nested while inside a top-level while in a >=4-param fn. The
- `idcollide_nested_while_n_7_3.mind` (~184 tok, small) — Regression: nested while inside a top-level while in a >=4-param fn. The
- `idcollide_nested_while_n_7_4.mind` (~184 tok, small) — Regression: nested while inside a top-level while in a >=4-param fn. The
- `idcollide_nested_while_n_8_3.mind` (~187 tok, small) — Regression: nested while inside a top-level while in a >=4-param fn. The
- `idcollide_nested_while_n_8_4.mind` (~187 tok, small) — Regression: nested while inside a top-level while in a >=4-param fn. The
- `idcollide_triple_nested_while.mind` (~199 tok, small) — Regression: triple-nested while + a post-loop alias in a 5-param fn (body-local
- `merge_binding_if_then_while_read.mind` (~73 tok, small) — Regression class: an if's merge binding read by a later while cond/body.
- `postloop_gap2_freeze.mind` (~64 tok, small) — Regression class: post-loop reads resolve to the while's dominating exit ids.
- `postloop_read_after_two_loops.mind` (~95 tok, small) — Regression class: post-loop reads resolve to the while's dominating exit ids.
- `rv3_loop_in_then_assigns_outer_and_shadow.mind` (~115 tok, small) — Regression class (twin-emitter review round 3): loop in an if-then writes an outer var first seen inside the loop (+ blo
- `rv3_native_shadow.mind` (~78 tok, small) — Regression class (twin-emitter review round 3): native ELF: a loop-body let shadowing an outer name must leave the outer
- `rv3_native_shadow_param.mind` (~82 tok, small) — Regression class (twin-emitter review round 3): native ELF: a loop-body let shadowing a PARAMETER must leave the paramet
- `rv3_prog0093_minimal.mind` (~177 tok, small) — Regression class (twin-emitter review round 3): nested-if merge bindings read by a later sequence-level let (seed 99991)
- `rv3_prog0178_minimal.mind` (~222 tok, medium) — Regression class (twin-emitter review round 3): nested-if merge bindings read by a later sequence-level let (seed 4242).
- `rv3_r20_min.mind` (~87 tok, small) — Regression class (twin-emitter review round 3): minimized blind-review reproducer.
- `rv3_rand_230_52_m0_main_only.mind` (~98 tok, small) — Regression class (twin-emitter review round 3): minimized blind-review reproducer.
- `rv3_rand_230_52_m0.mind` (~332 tok, medium) — Regression class (twin-emitter review round 3): random-program reproducer (blind review round 2).
- `rv3_rt_break_flush_carried.mind` (~101 tok, small) — Regression class (twin-emitter review round 3, runtime leg): native ELF break must flush carried values written before i
- `rv3_rt_carried_ident_assign.mind` (~88 tok, small) — Regression class (twin-emitter review round 3, runtime leg): native ELF carried var assigned from a bare ident after a b
- `rv3_rt_dtk_b1_selfassign_helper.mind` (~60 tok, small) — round-2 blocker: helper top-level bare-ident assign (w = y) under the DTK register plan; notes matched, native ELF exite
- `rv3_rt_dtk_b2_otherassign_helper.mind` (~66 tok, small) — round-2 blocker: helper top-level bare-ident assign (w = y) under the DTK register plan; notes matched, native ELF exite
- `rv3_rt_dtk_b3_selfassign_param.mind` (~63 tok, small) — round-2 blocker: helper top-level bare-ident assign (w = y) under the DTK register plan; notes matched, native ELF exite
- `rv3_rt_dtk_b6_assign_ident_then_add.mind` (~68 tok, small) — round-2 blocker: helper top-level bare-ident assign (w = y) under the DTK register plan; notes matched, native ELF exite
- `rv3_rt_dtk_c3_toplevel_then_use.mind` (~69 tok, small) — round-2 blocker: helper top-level bare-ident assign (w = y) under the DTK register plan; notes matched, native ELF exite
- `rv3_rt_dtk_c5_two_helpers.mind` (~64 tok, small) — round-2 blocker: helper top-level bare-ident assign (w = y) under the DTK register plan; notes matched, native ELF exite
- `rv3_rt_dtk_letcopy_dtk_1_13.mind` (~94 tok, small) — round-3 blocker: bare-ident let of a reassigned source in a DTK-eligible helper (LET copy slot not modelled by the regis
- `rv3_rt_dtk_letcopy_dtk_1_15.mind` (~97 tok, small) — round-3 blocker: bare-ident let of a reassigned source in a DTK-eligible helper (LET copy slot not modelled by the regis
- `rv3_rt_dtk_letcopy_dtk_1_20.mind` (~90 tok, small) — round-3 blocker: bare-ident let of a reassigned source in a DTK-eligible helper (LET copy slot not modelled by the regis
- `rv3_rt_dtk_letcopy_dtk_1_48.mind` (~105 tok, small) — round-3 blocker: bare-ident let of a reassigned source in a DTK-eligible helper (LET copy slot not modelled by the regis
- `rv3_rt_dtk_letcopy_dtk_1_58.mind` (~91 tok, small) — round-3 blocker: bare-ident let of a reassigned source in a DTK-eligible helper (LET copy slot not modelled by the regis
- `rv3_rt_dtk_letcopy_gen1234_p144.mind` (~428 tok, medium) — round-3 blocker, generator-caught (seed 1234 prog 144, three-mode gen_dtk_fn mode 1): bare-ident let of a reassigned sou
- `rv3_rt_dtk_letcopy_m1.mind` (~82 tok, small) — round-3 blocker: bare-ident let of a reassigned source in a DTK-eligible helper (LET copy slot not modelled by the regis
- `rv3_rt_dtk_letcopy_m3.mind` (~90 tok, small) — round-3 blocker: bare-ident let of a reassigned source in a DTK-eligible helper (LET copy slot not modelled by the regis
- `rv3_rt_dtk_letcopy_m4.mind` (~80 tok, small) — round-3 blocker: bare-ident let of a reassigned source in a DTK-eligible helper (LET copy slot not modelled by the regis
- `rv3_rt_dtk_letcopy_m7.mind` (~78 tok, small) — round-3 blocker: bare-ident let of a reassigned source in a DTK-eligible helper (LET copy slot not modelled by the regis
- `rv3_rt_dtk_letcopy_rand_31_72.mind` (~132 tok, small) — round-3 blocker: bare-ident let of a reassigned source in a DTK-eligible helper (LET copy slot not modelled by the regis
- `rv3_rt_dtk_v_101_52.mind` (~91 tok, small) — round-2 blocker: helper top-level bare-ident assign (w = y) under the DTK register plan; notes matched, native ELF exite
- `rv3_rt_dtk_v_32_169.mind` (~120 tok, small) — round-2 blocker: helper top-level bare-ident assign (w = y) under the DTK register plan; notes matched, native ELF exite
- `rv3_rt_let_alias_of_carried.mind` (~78 tok, small) — Regression class (twin-emitter review round 3, runtime leg): native ELF let alias of a carried var must not observe the 
- `rv3_rt_nested_carry_init_s4242_p0231.mind` (~241 tok, medium) — Regression class (twin-emitter review round 3, runtime leg): native ELF nested-loop carry init must come from the pre-lo
- `rv3_rt_shadow_nested_carry_s1234_p0105.mind` (~514 tok, large) — Regression class (twin-emitter review round 3, runtime leg): native ELF nested-region writes to a body-local shadow are 
- `rv3_shadow_then_loop_outer_assigned_in_loop.mind` (~97 tok, small) — Regression class (twin-emitter review round 3): loop in an if-then writes an outer var first seen inside the loop (+ blo
- `rv3_v1_noshadow.mind` (~83 tok, small) — Regression class (twin-emitter review round 3): minimized blind-review reproducer.
- `rv3_v6_let_then_loop_only_outer.mind` (~82 tok, small) — Regression class (twin-emitter review round 3): minimized blind-review reproducer.
- `rv3_v7_fn_param.mind` (~95 tok, small) — Regression class (twin-emitter review round 3): loop in an if-then writes an outer var first seen inside the loop (param
- `rv3_wit_call_after.mind` (~95 tok, small) — Regression class (twin-emitter review round 3): loop in an if-then writes an outer var first seen inside the loop.
- `rv3_wit_nested_if_in_loop_in_then.mind` (~94 tok, small) — Regression class (twin-emitter review round 3): loop in an if-then writes an outer var first seen inside the loop.
- `rv3_wit_nested_while_in_loop_in_then.mind` (~110 tok, small) — Regression class (twin-emitter review round 3): loop in an if-then writes an outer var first seen inside the loop.
- `rv3_wit_outer_counter.mind` (~77 tok, small) — Regression class (twin-emitter review round 3): loop in an if-then writes an outer var first seen inside the loop.
- `rv3_wit_then_seq_after_if.mind` (~112 tok, small) — Regression class (twin-emitter review round 3): loop in an if-then writes an outer var first seen inside the loop.
- `rv3_wit_two_outer_one_prewritten.mind` (~101 tok, small) — Regression class (twin-emitter review round 3): loop in an if-then writes an outer var first seen inside the loop.
- `shadow_assign_before_if.mind` (~100 tok, small) — Regression class: assign-before-shadow (src/eval/lower.rs If arm, GAP 2).
- `shadow_assign_before_inner_block.mind` (~95 tok, small) — Regression class: assign-before-shadow (src/eval/lower.rs If arm, GAP 2).
- `shadow_assign_before_in_while.mind` (~81 tok, small) — Regression class: assign-before-shadow (src/eval/lower.rs If arm, GAP 2).
- `shadow_assign_before_nested_if.mind` (~86 tok, small) — Regression class: assign-before-shadow (src/eval/lower.rs If arm, GAP 2).
- `shadow_assign_before_noassign.mind` (~65 tok, small) — Regression class: assign-before-shadow (src/eval/lower.rs If arm, GAP 2).
- `shadow_assign_before_prog0040.mind` (~95 tok, small) — Regression class: assign-before-shadow (src/eval/lower.rs If arm, GAP 2).
- `shadow_assign_before_reads.mind` (~97 tok, small) — Regression class: assign-before-shadow (src/eval/lower.rs If arm, GAP 2).
- `shadow_assign_before_rmw.mind` (~89 tok, small) — Regression class: assign-before-shadow (src/eval/lower.rs If arm, GAP 2).
- `shadow_assign_before_twice.mind` (~87 tok, small) — Regression class: assign-before-shadow (src/eval/lower.rs If arm, GAP 2).
- `shadow_assign_before_two_names.mind` (~96 tok, small) — Regression class: assign-before-shadow (src/eval/lower.rs If arm, GAP 2).
- `shadow_assign_before_while_after.mind` (~92 tok, small) — Regression class: assign-before-shadow (src/eval/lower.rs If arm, GAP 2).
### `examples/mindc_mind/testdata/native_alloc_bounded/`

- `afl_inc.mind` (~190 tok, small) — Strictly growing alloc/free loop (verifier repro, W11c round 2): each step
- `alloc_nonpos.mind` (~327 tok, medium) — alloc/realloc/free parity with the C runtime (runtime-support/mind_intrinsics.c):
- `big_increasing.mind` (~159 tok, small) — A 300..319 MB scratch buffer allocated, touched at its last word and freed,
- `bin_fit_search.mind` (~338 tok, medium) — A free block that fits must be found even when it is not the head of its
- `dup_struct_name.mind` (~186 tok, small) — Two struct declarations with the same name (W11c). The emitter visits the
- `grow8.mind` (~173 tok, small) — Naive append (verifier repro, W11c round 2): one buffer grown by 8 bytes per
- `grow_big.mind` (~283 tok, medium) — measured-report's synthesize-path footprint: 256+512+4+6+8 MiB = 786 MiB of
- `interleaved_growth.mind` (~306 tok, medium) — Two buffers grown alternately by __mind_realloc, 8 and 16 bytes per step for
- `oom_exit.mind` (~111 tok, small) — Allocation without free past the address-space limit: the native image must
- `r18_free_loop.mind` (~77 tok, small) — fn main() -> i64 {
- `realloc_free_neighbour.mind` (~299 tok, medium) — realloc grows a block in place into its free successor (W11c round 3).
- `realloc_null_zero.mind` (~199 tok, small) — realloc(NULL, 0) (verifier repro edge/e04, W11c round 2). The C runtime's
### `examples/mindc_mind/testdata/native_alloc_bounded/rt/`

- `alloc_rt_harness.c` (~3526 tok, huge) — Standalone stress harness for the bounded-profile allocator blob (rt.bin).
- `alloc_rt.s` (~3446 tok, huge) — .intel_syntax noprefix
- `gen_mind.py` (~489 tok, medium) — rt.bin -> MIND byte-emitter functions (nb_le 7-byte chunks; nb_i64le for
### `examples/mindc_mind/testdata/native_alloc_bounded/`

- `vec_grow.mind` (~326 tok, medium) — std vec_push shape: a buffer grown by doubling __mind_realloc, filled,
- `vec_std.mind` (~341 tok, medium) — std vec_push itself: std/vec.mind vec_new + vec_push growing through
### `examples/mindc_mind/testdata/native_alloc_default/`

- `after_early_return.mind` (~113 tok, small) — A free after an early return in the loop body (the return never runs). 1,100 x 1 MB. Exit status 33.
- `align_zero.mind` (~138 tok, small) — Blocks handed out by the reclaiming allocator are 8-byte aligned and read zero, also after a block was freed and written
- `argv_free_loop.mind` (~119 tok, small) — fn main(argc, argv) with a free loop, run with two arguments: argc is 3, argv reads back. Exit status 33.
- `arm_array_lit.mind` (~121 tok, small) — A free under a array literal element inside a while loop, 1,100 x 1 MB: a bump arena (1 GiB) overflows, so the image mus
- `arm_assign.mind` (~109 tok, small) — A free under a plain assignment inside a while loop, 1,100 x 1 MB: a bump arena (1 GiB) overflows, so the image must rec
- `arm_binop.mind` (~113 tok, small) — A free under a binary operator operand inside a while loop, 1,100 x 1 MB: a bump arena (1 GiB) overflows, so the image m
- `arm_call_arg.mind` (~122 tok, small) — A free under a call argument inside a while loop, 1,100 x 1 MB: a bump arena (1 GiB) overflows, so the image must reclai
- `arm_cast.mind` (~114 tok, small) — A free under a narrowing cast inside a while loop, 1,100 x 1 MB: a bump arena (1 GiB) overflows, so the image must recla
- `arm_enum_ctor.mind` (~123 tok, small) — A free under an enum constructor payload inside a while loop, 1,100 x 1 MB: a bump arena (1 GiB) overflows, so the image
- `arm_expr_stmt.mind` (~109 tok, small) — A free under a bare expression statement inside a while loop, 1,100 x 1 MB: a bump arena (1 GiB) overflows, so the image
- `arm_field_assign.mind` (~128 tok, small) — A free under a field assignment inside a while loop, 1,100 x 1 MB: a bump arena (1 GiB) overflows, so the image must rec
- `arm_field.mind` (~136 tok, small) — A free under a field read of a call inside a while loop, 1,100 x 1 MB: a bump arena (1 GiB) overflows, so the image must
- `arm_for_body.mind` (~87 tok, small) — A free in the body of a for loop, 1,100 x 1 MB. The image must reclaim. Exit status 33.
- `arm_if_cond.mind` (~118 tok, small) — A free under a if condition inside a while loop, 1,100 x 1 MB: a bump arena (1 GiB) overflows, so the image must reclaim
- `arm_if_else.mind` (~125 tok, small) — A free under a else block inside a while loop, 1,100 x 1 MB: a bump arena (1 GiB) overflows, so the image must reclaim. 
- `arm_if_then.mind` (~114 tok, small) — A free under a then block inside a while loop, 1,100 x 1 MB: a bump arena (1 GiB) overflows, so the image must reclaim. 
- `arm_index_assign.mind` (~118 tok, small) — A free under a index assignment inside a while loop, 1,100 x 1 MB: a bump arena (1 GiB) overflows, so the image must rec
- `arm_index.mind` (~120 tok, small) — A free under a index expression inside a while loop, 1,100 x 1 MB: a bump arena (1 GiB) overflows, so the image must rec
- `arm_let.mind` (~110 tok, small) — A free under a let initialiser inside a while loop, 1,100 x 1 MB: a bump arena (1 GiB) overflows, so the image must recl
- `arm_neg.mind` (~109 tok, small) — A free under a unary minus inside a while loop, 1,100 x 1 MB: a bump arena (1 GiB) overflows, so the image must reclaim.
- `arm_not.mind` (~124 tok, small) — A free under a logical not inside an if condition inside a while loop, 1,100 x 1 MB: a bump arena (1 GiB) overflows, so 
- `arm_paren.mind` (~110 tok, small) — A free under a parenthesis inside a while loop, 1,100 x 1 MB: a bump arena (1 GiB) overflows, so the image must reclaim.
- `arm_return.mind` (~131 tok, small) — A free under a return statement (in a fn that the loop calls) inside a while loop, 1,100 x 1 MB: a bump arena (1 GiB) ov
- `arm_struct_lit.mind` (~130 tok, small) — A free under a struct literal field inside a while loop, 1,100 x 1 MB: a bump arena (1 GiB) overflows, so the image must
- `arm_value_if.mind` (~125 tok, small) — A free under a value if inside a while loop, 1,100 x 1 MB: a bump arena (1 GiB) overflows, so the image must reclaim. Ex
- `arm_while_body.mind` (~132 tok, small) — A free under a nested while body inside a while loop, 1,100 x 1 MB: a bump arena (1 GiB) overflows, so the image must re
- `arm_while_cond.mind` (~128 tok, small) — A free under a while condition (an inner loop that runs 0 times) inside a while loop, 1,100 x 1 MB: a bump arena (1 GiB)
- `arm_while_cond_only.mind` (~77 tok, small) — A free in the condition of the only loop: each evaluation allocates and frees 1 MB, 1,101 times, so a bump arena (1 GiB)
- `both_ways.mind` (~124 tok, small) — A function that frees, called once outside every loop and from a loop 1,100 x 1 MB. Exit status 33.
- `calls1.mind` (~113 tok, small) — A free 1 call(s) below the loop; each caller is defined before its callee. 1,100 x 1 MB. Exit status 33.
- `calls2.mind` (~126 tok, small) — A free 2 call(s) below the loop; each caller is defined before its callee. 1,100 x 1 MB. Exit status 33.
- `calls3.mind` (~138 tok, small) — A free 3 call(s) below the loop; each caller is defined before its callee. 1,100 x 1 MB. Exit status 33.
- `ctl_both_directives.mind` (~117 tok, small) — mind: native_opt = full
- `ctl_directive_late.mind` (~114 tok, small) — The directive sits after code, outside the leading comment block: it does not count. RECLAIM. 1,100 x 1 MB. Exit status 
- `ctl_directive.mind` (~115 tok, small) — mind: native_alloc = bump
- `ctl_for_end_bound.mind` (~111 tok, small) — The END bound of a for loop calls a function that frees. The parser evaluates the bound once, before the loop, so it is 
- `ctl_free_outside.mind` (~68 tok, small) — A free outside every loop. BUMP: the image equals the base compiler's. Exit status 33.
- `ctl_no_free.mind` (~52 tok, small) — A program with no free. BUMP: the image equals the base compiler's. Exit status 33.
- `ctl_recursion_no_loop.mind` (~84 tok, small) — Recursion without a loop frees on every level. Not counted: BUMP, and the image equals the base compiler's. Exit status 
- `dyn_free_loop.mind` (~165 tok, small) — A loop that calls an extern fn, allocates 1 MB and frees it, 2,000 times: 2 GB of
- `exhaust_live.mind` (~143 tok, small) — 1,100 live blocks of 1 MB (1.1 GB) in a linked list, then a free loop: the arena (1 GiB) is full before the loop. The im
- `interleaved_free.mind` (~260 tok, medium) — Four blocks of different sizes per iteration, freed in an order that is not the reverse of the allocation order, 2,000 t
- `libc_first_exhaust.mind` (~331 tok, medium) — getaddrinfo (a libc call that allocates with malloc) runs first, then 1 MB blocks stay live in a linked list
- `mutual_recursion.mind` (~150 tok, small) — Two functions that call each other, called from a loop; the free is at the bottom of the recursion. 1,100 x 1 MB. Exit s
- `realloc_free_loop.mind` (~111 tok, small) — realloc from 4 KB to 1 MB, then free, 1,100 times: a bump arena (1 GiB) overflows. Exit status 33.
- `realloc_growth.mind` (~478 tok, medium) — Part 1: a block grows from 64 B to 1 MB with two other blocks live, 1,100 times. Every word of every block is checked af
- `std_sha256_loop.mind` (~133 tok, small) — std.sha256's sha256 frees its scratch. 1,600,000 calls x 712 B (more than 1 GiB) from a user loop: the image must reclai
### `examples/mindc_mind/testdata/native_avx2/`

- `avx2_semantics.mind` (~1559 tok, huge) — Copyright 2026 STARGA Inc.
- `bank_acc8_layer_a.mind` (~326 tok, medium) — Copyright 2026 STARGA Inc.
- `bank_acc8.mind` (~504 tok, large) — Copyright 2026 STARGA Inc.
- `bank_acc8_scalar.mind` (~279 tok, medium) — Copyright 2026 STARGA Inc.
- `bank_canary.mind` (~568 tok, large) — Copyright 2026 STARGA Inc.
- `bank_canary_scalar.mind` (~572 tok, large) — Copyright 2026 STARGA Inc.
- `bank_epi_forms.mind` (~560 tok, large) — Copyright 2026 STARGA Inc.
- `bank_epi_quad.mind` (~1127 tok, large) — Copyright 2026 STARGA Inc.
- `bank_epi_scalar.mind` (~374 tok, medium) — Copyright 2026 STARGA Inc.
- `bank_forms.mind` (~583 tok, large) — Copyright 2026 STARGA Inc.
- `bank_forms_scalar.mind` (~893 tok, large) — Copyright 2026 STARGA Inc.
- `bank_perf_layer_a.mind` (~315 tok, medium) — Copyright 2026 STARGA Inc.
- `bank_perf.mind` (~467 tok, medium) — Copyright 2026 STARGA Inc.
- `bank_refuse_epi_reg15.mind` (~156 tok, small) — Copyright 2026 STARGA Inc.
- `bank_refuse_f64.mind` (~156 tok, small) — Copyright 2026 STARGA Inc.
- `bank_refuse_layer_a.mind` (~146 tok, small) — Copyright 2026 STARGA Inc.
- `bank_refuse_nonliteral_disp.mind` (~143 tok, small) — Copyright 2026 STARGA Inc.
- `bank_refuse_nonliteral_reg.mind` (~144 tok, small) — Copyright 2026 STARGA Inc.
- `bank_refuse_reg15.mind` (~136 tok, small) — Copyright 2026 STARGA Inc.
- `bank_refuse_user_call.mind` (~157 tok, small) — Copyright 2026 STARGA Inc.
- `bank_refuse_value_use.mind` (~138 tok, small) — Copyright 2026 STARGA Inc.
- `bank_residue.mind` (~184 tok, small) — Copyright 2026 STARGA Inc.
- `bank_twin_epi_reg15.mind` (~146 tok, small) — Copyright 2026 STARGA Inc.
- `bank_twin_f64.mind` (~137 tok, small) — Copyright 2026 STARGA Inc.
- `bank_twin_layer_a.mind` (~136 tok, small) — Copyright 2026 STARGA Inc.
- `bank_twin_nonliteral_disp.mind` (~128 tok, small) — Copyright 2026 STARGA Inc.
- `bank_twin_nonliteral_reg.mind` (~127 tok, small) — Copyright 2026 STARGA Inc.
- `bank_twin_reg15.mind` (~126 tok, small) — Copyright 2026 STARGA Inc.
- `bank_twin_user_call.mind` (~139 tok, small) — Copyright 2026 STARGA Inc.
- `bank_twin_value_use.mind` (~126 tok, small) — Copyright 2026 STARGA Inc.
- `cpu_features_bits.mind` (~101 tok, small) — Copyright 2026 STARGA Inc.
- `cpu_features_probe.mind` (~187 tok, small) — Copyright 2026 STARGA Inc.
- `e_i32w.mind` (~284 tok, medium) — i32-walk kind 89 term loop: one row, eight add terms, 64 i32 lanes.
- `encoder_table_driver.mind` (~1460 tok, large) — Copyright 2026 STARGA Inc.
- `e_walk.mind` (~288 tok, medium) — Acceptance shape through kind 89: one row, eight add terms.
- `f_check.mind` (~2263 tok, huge) — Edge rows for the kind 89 epilogue. The scalar twin is epi_nv.
- `f_e_shape.mind` (~606 tok, large) — Epilogue shape row. One call, 32 lanes, empty walk, init bit 0 set.
- `f_fr4.mind` (~2056 tok, huge) — The kind 89 epilogue against one lane of a fixed-point row finisher, operators as written there.
- `f_kfuse32.mind` (~3143 tok, huge) — Fused tail oracle, i32 walks (kind 89, desc[7] bits 2 and 7, W = 64). As f_kfuse.mind: the
- `f_kfuse.mind` (~3271 tok, huge) — Fused tail oracle, i64 walks (kind 89, desc[7] bit 7, W = 32). Each case runs the same rows
- `f_kperf32.mind` (~1047 tok, large) — Tail speed row on the i32-walk walk form: desc[7] bits 2, 3 and 4 (i32 walk, term pairs, u16 lane
- `f_kperf.mind` (~989 tok, large) — Tail speed row for bit 5 and bit 7. The gate sets MODE (5 or 7), FILL (1: the caller fills the 64-byte
- `f_kt32_fuse.mind` (~1401 tok, large) — Fused u32 tail. 7|8 and 2|7|8 must match the bit-5 u32 twin. The out row stays.
- `f_kt32.mind` (~2507 tok, huge) — K89-TAIL32 known answers. Bit 8 selects u32 src and dst. The scalar twin is the bit-5
- `f_perf.mind` (~500 tok, large) — Epilogue speed row. 1024 rows, 2000 calls, dst is the out row.
- `f_rand.mind` (~2626 tok, huge) — Random epilogue lanes (W=32 and W=64, padded out_step, mixed shift counts, trips at random lanes)
- `f_rows.mind` (~1986 tok, huge) — Multi-row epilogue rows with a padded out_step and trips in chosen lanes.
- `f_scalar.mind` (~2239 tok, huge) — Scalar twin of the f_check edge grid. Same epi_nv formula. No native call
- `group_acc88_init23.hex` (~257 tok, medium) — 64000000000000006500000000000000660000000000000067000000000000006800000000000000
- `group_acc88_init23.mind` (~230 tok, medium) — Kind 88 init 2 and init 3 are zero start. Prior lanes are 7. One add term.
- `group_acc88_init23_scalar.mind` (~93 tok, small) — Scalar twin of group_acc88_init23. init 2 and init 3 zero, then add term 0.
- `group_acc88_neg.hex` (~385 tok, medium) — 64000000000000006500000000000000660000000000000067000000000000006800000000000000
- `group_acc88_neg.mind` (~341 tok, medium) — Kind 88 with a negative init, no guarded row, no other probe-skip call. The
- `group_acc88_neg_scalar.mind` (~96 tok, small) — Scalar twin of group_acc88_neg. Three negative inits are zero start, then add term 0.
- `group_accrows_alias.hex` (~385 tok, medium) — e803000000000000e903000000000000ea03000000000000eb03000000000000ec03000000000000
- `group_accrows_alias.mind` (~475 tok, medium) — Kind 89 with out inside the bank. Same bytes as the scalar twin.
- `group_accrows_alias_scalar.mind` (~775 tok, large) — Scalar twin of the alias fixture. out sits inside bank.
- `group_accrows_init2.mind` (~229 tok, medium) — Kind 89 keeps prior lanes only when init == 1.
- `group_accrows_init3.hex` (~129 tok, small) — 64000000000000006500000000000000660000000000000067000000000000006800000000000000
- `group_accrows_init3.mind` (~244 tok, medium) — Kind 89 keeps prior lanes only when init == 1.
- `group_accrows_init3_scalar.mind` (~128 tok, small) — Scalar twin of group_accrows_init3. init == 1 keeps out. init 2 and init 3 zero, then add term 0.
- `group_accrows_perf.mind` (~493 tok, medium) — 8760 calls, 64 rows, 10 add terms and 10 sub terms. One checksum.
- `group_accrows_rows.hex` (~3841 tok, huge) — 00000000000000000000000000000000000000000000000000000000000000000000000000000000
- `group_accrows_rows.mind` (~827 tok, large) — Kind 89 rows fixture. rows = 7, out_step 576, starts_step 72.
- `group_accrows_scalar.mind` (~1150 tok, large) — Scalar twin of group_accrows_rows. Plain loops, same stdout.
- `group_accrows_vs_acc.mind` (~817 tok, large) — Same inputs as group_accrows_rows through one kind-88 call per row.
- `group_bank_wide_group.hex` (~417 tok, medium) — 2a000000000000002a000000000000002a000000000000002a000000000000002a00000000000000
- `group_bank_wide_group.mind` (~628 tok, large) — Kind 87 with a group whose low four bits are 8..15: group 11, then 27 (16 + 11).
- `group_bank_wide_group_scalar.mind` (~74 tok, small) — Scalar twin of group_bank_wide_group. Both calls touch nothing: 832 bytes of 42.
- `group_sum_acc32.hex` (~260 tok, medium) — 080000000000000009000000000000000a000000000000000b00000000000000
- `group_sum_acc32.mind` (~362 tok, medium) — __mind_group_acc32_i64 against the N2 oracle's acc32 rows (the scalar oracle
- `group_sum_alias.hex` (~1170 tok, large) — 030000000000000073000000000000003b00000000000000c8ffffffffffffff
- `group_sum_alias.mind` (~460 tok, medium) — Overlap: every write lands inside one 2304-byte region M that also holds the
- `group_sum_bank3.hex` (~423 tok, medium) — 0800000000000000070000000000000006000000000000000000000000000000
- `group_sum_bank3.mind` (~611 tok, large) — Group 3 over the N2 corpus (gs_a/b/c = n2s_a/b/c of the scalar oracle program
- `group_sum_bank4.hex` (~650 tok, large) — 0800000000000000070000000000000006000000000000000000000000000000
- `group_sum_bank4.mind` (~603 tok, large) — Group 4 over the N2 corpus (gs_a/b/c/x = n2s_a/b/c/x of the scalar oracle
- `group_sum_k87only.mind` (~679 tok, large) — Probe-skip, kind 87 only: group 19 and group 20 (bit 4 set, no probe). No kind 88
- `group_sum_k88only.mind` (~337 tok, medium) — Probe-skip, kind 88 only. K1 = 4611686018427387904 (zero start, no probe) and
- `group_sum_none.hex` (~130 tok, small) — 0700000000000000070000000000000007000000000000000700000000000000
- `group_sum_none.mind` (~197 tok, small) — Calls that must read and write nothing: group 5, pixels 0 and pixels -3 with
- `group_sum_noprobe.hex` (~1313 tok, large) — 080000000000000009000000000000000a000000000000000b000000000000000c00000000000000
- `group_sum_noprobe.mind` (~931 tok, large) — Probe-skip on the acc32, bank3 and bank4 inputs. Kind 88 uses K1 =
- `group_sum_perf.mind` (~438 tok, medium) — Timing variant (not a stdout row): group 3 over 4096 pixels (13 rows, plane
- `group_sum_perf_scalar.mind` (~719 tok, large) — Scalar twin of group_sum_perf.mind: the same rows and taps written as plain
- `group_sum_ptr_driver.mind` (~514 tok, large) — Copyright 2026 STARGA Inc.
- `group_sum_ptr_encoder_table.tsv` (~789 tok, large) — # Kind 87 running-pointer encoder oracle. Columns: tag <TAB> instruction (Intel 
- `group_sum_ptr_shape.mind` (~344 tok, medium) — Copyright 2026 STARGA Inc.
- `group_sum_ptr_twin.mind` (~1155 tok, large) — Copyright 2026 STARGA Inc.
- `group_sum_remainder.hex` (~1073 tok, large) — 0800000000000000070000000000000006000000000000000000000000000000
- `group_sum_remainder.mind` (~658 tok, large) — Remainder paths: group 3 with pixels = 5 (one four-pixel step + one scalar
- `k3_bank.mind` (~1256 tok, large) — Kind 87 with group bit 5. Stdout is the bad-store count. Exit 0 when every
- `k3_bank_tight.mind` (~534 tok, large) — Kind 87 group bit 5 with a tight plane (plane == pixels * 4),
- `k3_bound.mind` (~779 tok, large) — i32-walk bound-field contract (nb_group_i32w.mind). One row, lane 0 is read.
- `k3_check.mind` (~1576 tok, huge) — i32-walk value check. Exit 0 when kind 89 matches the scalar twin and kind 89 with
- `k3_fallback.mind` (~1528 tok, huge) — i32-walk fast path and fallback, 3 rows, out_step, starts_step, add and sub lists,
- `k3_perf.mind` (~465 tok, medium) — 8760 calls, 64 rows, 10 add terms and 10 sub terms, i32 mode. One checksum.
- `k3_rand.mind` (~1561 tok, huge) — i32-walk fast-path oracle (bound field 0, init 4). 10 rows x 10000 terms, plus one edge row.
- `load_store_i64.mind` (~76 tok, small) — Copyright 2026 STARGA Inc.
- `main_params_argv_capture.mind` (~304 tok, medium) — Copyright 2026 STARGA Inc.
- `main_params_vmovdqu.mind` (~326 tok, medium) — Copyright 2026 STARGA Inc.
- `n32_bank.mind` (~995 tok, large) — Kind 87 mode N32 (group bits 5 and 6): i32 code rows from u32 channel rows. The reference
- `t16_alias.mind` (~392 tok, medium) — i32-walk term fallback overlap: bit 4 + bit 2 (mode 20), bound field 2^31 takes nbt_fallback.
- `t16_check.mind` (~2894 tok, huge) — Kind 89 bit 4 value check (u16 offset lists). Exit 0 when every mode matches
- `t16_e_shape.mind` (~462 tok, medium) — Bits 3+4 shape. Two add terms and two sub terms, u16 lists, i64 walk and i32
- `t16_perf32.mind` (~627 tok, large) — 8760 calls, 64 rows, 10 add terms and 10 sub terms. desc[7] is argv[1] (decimal).
- `t16_perf.mind` (~677 tok, large) — 8760 calls, 64 rows, 10 add terms and 10 sub terms. desc[7] is argv[1] (decimal).
- `t16_rand.mind` (~1452 tok, large) — Fixed-seed bit 4 oracle (u16 offset lists). 18 calls, 100000 terms each. Term
- `t16_rows.mind` (~2069 tok, huge) — Kind 89 multi-row value check. Exit 0 when every mode matches the scalar twin
- `t2_check.mind` (~2222 tok, huge) — Kind 89 bit 3 value check. Exit 0 when the paired loop matches the bit-clear
- `t2_e_shape.mind` (~450 tok, medium) — Bit 3 shape. Two add terms and two sub terms, i64 walk and i32 walk.
- `t2_perf32.mind` (~462 tok, medium) — 8760 calls, 64 rows, 10 add terms and 10 sub terms. desc[7] = 13.
- `t2_perf.mind` (~510 tok, large) — 8760 calls, 64 rows, 10 add terms and 10 sub terms. desc[7] = 9 (bit 3, init 1).
- `t2_rand.mind` (~1252 tok, large) — Fixed-seed bit 3 oracle. 12 calls, 100000 terms each. Term count is 1200000.
- `t2_trap.mind` (~404 tok, medium) — Bit 3 or bit 4 with a reserved bit (bit 1 or bits 6..31). Bit 4 traps only
- `t3_trap_sweep.mind` (~1696 tok, huge) — Reserved-bit trap sweep for kinds 87, 88 and 89. One fork per case; a trapped case dies
- `vmovdqu_i32x8.mind` (~71 tok, small) — Copyright 2026 STARGA Inc.
- `vpaddb_i8x32.mind` (~70 tok, small) — Copyright 2026 STARGA Inc.
- `vpaddd_i32x8.mind` (~70 tok, small) — Copyright 2026 STARGA Inc.
- `vpaddq_i64x4.mind` (~72 tok, small) — Copyright 2026 STARGA Inc.
- `vpshufb_i8x32.mind` (~73 tok, small) — Copyright 2026 STARGA Inc.
- `vpsubd_i32x8.mind` (~70 tok, small) — Copyright 2026 STARGA Inc.
- `vpsubq_i64x4.mind` (~72 tok, small) — Copyright 2026 STARGA Inc.
- `ymm_names_inert.mind` (~189 tok, small) — Copyright 2026 STARGA Inc.
### `examples/mindc_mind/testdata/native_cf/`

- `a1_logic_in_cond.mind` (~230 tok, medium) — A1: && / || / ! and a value-if inside if and while conditions, with
- `a6_vif_in_loop.mind` (~202 tok, medium) — A6: a value-if used as a value inside a while body (let init, assignment
- `a7_vif_nested_expr.mind` (~272 tok, medium) — A7: value-ifs nested inside expressions: binop operands, call arguments
- `a9_read_then_branch_assign.mind` (~192 tok, small) — A HOME-lowered read of a variable is a copy: a value-if branch that assigns
- `a9_vif_branch_effects.mind` (~212 tok, medium) — A9: value-if branches that assign an enclosing variable, return from the
- `deep_nesting.mind` (~306 tok, medium) — Arbitrary nesting: six levels of if / while / value-if mixing every exit
- `f64_loop_vif_return.mind` (~206 tok, medium) — f64 values through the HOME lowering: a float accumulator assigned from a
- `l1_return_in_loop.mind` (~306 tok, medium) — L1: `return` inside while bodies, from 1, 2 and 3 levels of nesting, inside
- `l2_loop_if_branches.mind` (~412 tok, medium) — L2: an `if` inside a loop whose branches hold a call statement, a let, a
- `l3_break_continue_params.mind` (~314 tok, medium) — L3: break / continue inside functions that take parameters, including
- `mlir_break_in_cond_vif.mind` (~165 tok, small) — A `break` inside a value-if that sits in an if condition inside a loop. The
- `mlir_f64_loop_carry_continue.mind` (~190 tok, small) — f64 values through the HOME lowering: || in an if condition inside a loop with
- `mlir_f64_loop_carry_vif.mind` (~251 tok, medium) — f64 values through the HOME lowering: && in an if condition inside a loop, a float
- `mlir_lost_branch_assign.mind` (~126 tok, small) — Known MLIR-backend defect, pinned: a value-if passed to a call that is used
- `mlir_lost_cond_assign.mind` (~132 tok, small) — Known MLIR-backend defect, pinned: an assignment inside a value-if branch in
- `mlir_lost_loop_assign.mind` (~217 tok, medium) — Known MLIR-backend defect, pinned: inside a loop, a store to an enclosing
- `mlir_narrow_loop_carry.mind` (~253 tok, medium) — A narrow i32 parameter copied into an i32 let, multiplied and added past
- `mlir_shadow_exit_carry.mind` (~380 tok, medium) — Known MLIR-backend defect, pinned: a let that shadows an outer variable
- `mlir_stale_read_after_vif.mind` (~140 tok, small) — Known MLIR-backend defect, pinned: a variable read AFTER a value-if in the
- `narrow_wrap_loop.mind` (~280 tok, medium) — Narrow integer lets (i8 / i16 / i32) through the HOME lowering: every store
- `note_kinds.mind` (~48 tok, tiny) — A program the mic@3 twin can hash keeps the IR trace-hash note (n_type 1).
- `o_if_else_merges.mind` (~332 tok, medium) — O1 / O2 / O3: if/else outside loops that merges assigned outer variables,
- `shadow_scopes.mind` (~602 tok, large) — Lets that shadow an outer name inside loop bodies and if branches, on the
- `shadow_scopes_ssa.mind` (~401 tok, medium) — The same shadow scoping on the SSA lowering (no return inside a loop, so the
### `examples/mindc_mind/testdata/native_c_variadic_emit/`

- `allocation-zero.patch` (~21 tok, tiny) — --- original.mind
- `BUILD_HASHES.tsv` (~430 tok, medium) — variant	backend	sha256
- `C01.input` (binary, 352 bytes)
- `C01.semantic` (binary, 8368 bytes)
- `C01.verify` (binary, 16736 bytes)
- `C02.input` (binary, 416 bytes)
- `C02.semantic` (binary, 8368 bytes)
- `C02.verify` (binary, 16736 bytes)
- `C03.input` (binary, 544 bytes)
- `C03.semantic` (binary, 8368 bytes)
- `C03.verify` (binary, 16736 bytes)
- `C04.input` (binary, 416 bytes)
- `C04.semantic` (binary, 8368 bytes)
- `C04.verify` (binary, 16736 bytes)
- `C05.input` (binary, 416 bytes)
- `C05.semantic` (binary, 8368 bytes)
- `C05.verify` (binary, 16736 bytes)
- `C06.input` (binary, 416 bytes)
- `C06.semantic` (binary, 8368 bytes)
- `C06.verify` (binary, 16736 bytes)
- `C07.input` (binary, 416 bytes)
- `C07.semantic` (binary, 8368 bytes)
- `C07.verify` (binary, 16736 bytes)
- `C08.input` (binary, 480 bytes)
- `C08.semantic` (binary, 8368 bytes)
- `C08.verify` (binary, 16736 bytes)
- `C09.input` (binary, 736 bytes)
- `C09.semantic` (binary, 8368 bytes)
- `C09.verify` (binary, 16736 bytes)
- `C10.input` (binary, 800 bytes)
- `C10.semantic` (binary, 8368 bytes)
- `C10.verify` (binary, 16736 bytes)
- `C11.input` (binary, 992 bytes)
- `C11.semantic` (binary, 8368 bytes)
- `C11.verify` (binary, 16736 bytes)
- `C12.input` (binary, 1376 bytes)
- `C12.semantic` (binary, 8368 bytes)
- `C12.verify` (binary, 16736 bytes)
- `C13.input` (binary, 1440 bytes)
- `C13.semantic` (binary, 8368 bytes)
- `C13.verify` (binary, 16736 bytes)
- `C14.input` (binary, 544 bytes)
- `C14.semantic` (binary, 8368 bytes)
- `C14.verify` (binary, 16736 bytes)
- `C15.input` (binary, 16672 bytes)
- `C15.semantic` (binary, 8368 bytes)
- `C15.verify` (binary, 16736 bytes)
- `C16.input` (binary, 416 bytes)
- `C16.semantic` (binary, 8368 bytes)
- `C16.verify` (binary, 16736 bytes)
- `C17.input` (binary, 480 bytes)
- `C17.semantic` (binary, 8368 bytes)
- `C17.verify` (binary, 16736 bytes)
- `C18.input` (binary, 544 bytes)
- `C18.semantic` (binary, 8368 bytes)
- `C18.verify` (binary, 16736 bytes)
- `C19.input` (binary, 608 bytes)
- `C19.semantic` (binary, 8368 bytes)
- `C19.verify` (binary, 16736 bytes)
- `C20.input` (binary, 672 bytes)
- `C20.semantic` (binary, 8368 bytes)
- `C20.verify` (binary, 16736 bytes)
- `C21.input` (binary, 736 bytes)
- `C21.semantic` (binary, 8368 bytes)
- `C21.verify` (binary, 16736 bytes)
- `C22.input` (binary, 800 bytes)
- `C22.semantic` (binary, 8368 bytes)
- `C22.verify` (binary, 16736 bytes)
- `C23.input` (binary, 864 bytes)
- `C23.semantic` (binary, 8368 bytes)
- `C23.verify` (binary, 16736 bytes)
- `C24.input` (binary, 928 bytes)
- `C24.semantic` (binary, 8368 bytes)
- `C24.verify` (binary, 16736 bytes)
- `C25.input` (binary, 352 bytes)
- `C25.semantic` (binary, 8368 bytes)
- `C25.verify` (binary, 16736 bytes)
- `C26.input` (binary, 352 bytes)
- `C26.semantic` (binary, 8368 bytes)
- `C26.verify` (binary, 16736 bytes)
- `C27.input` (binary, 352 bytes)
- `C27.semantic` (binary, 8368 bytes)
- `C27.verify` (binary, 16736 bytes)
- `C28.input` (binary, 352 bytes)
- `C28.semantic` (binary, 8368 bytes)
- `C28.verify` (binary, 16736 bytes)
- `C29.input` (binary, 352 bytes)
- `C29.semantic` (binary, 8368 bytes)
- `C29.verify` (binary, 16736 bytes)
- `C30.input` (binary, 352 bytes)
- `C30.semantic` (binary, 8368 bytes)
- `C30.verify` (binary, 16736 bytes)
- `C31.input` (binary, 480 bytes)
- `C31.semantic` (binary, 8368 bytes)
- `C31.verify` (binary, 16736 bytes)
- `C32.input` (binary, 416 bytes)
- `C32.semantic` (binary, 8368 bytes)
- `C32.verify` (binary, 16736 bytes)
- `C33.input` (binary, 416 bytes)
- `C33.semantic` (binary, 8368 bytes)
- `C33.verify` (binary, 16736 bytes)
- `C34.input` (binary, 912 bytes)
- `C34.semantic` (binary, 8368 bytes)
- `C34.verify` (binary, 16736 bytes)
- `C35.input` (binary, 416 bytes)
- `C35.semantic` (binary, 8368 bytes)
- `C35.verify` (binary, 16736 bytes)
- `C36.input` (binary, 16672 bytes)
- `C36.semantic` (binary, 8368 bytes)
- `C36.verify` (binary, 16736 bytes)
- `C37.input` (binary, 352 bytes)
- `C37.semantic` (binary, 8368 bytes)
- `C37.verify` (binary, 16736 bytes)
- `C38.input` (binary, 416 bytes)
- `C38.semantic` (binary, 8368 bytes)
- `C38.verify` (binary, 16736 bytes)
- `capture.c` (~12405 tok, huge) — Copyright 2026 STARGA Inc. Independent observation fixture, not a verdict authority.
- `capture.S` (~1065 tok, large) — # Copyright 2026 STARGA Inc. Independent SysV observation fixture.
- `CASES.tsv` (~480 tok, medium) — C01	baseline	call	no-tail
- `E01.input` (binary, 352 bytes)
- `E02.input` (binary, 352 bytes)
- `E03.input` (binary, 352 bytes)
- `E04.input` (binary, 352 bytes)
- `E05.input` (binary, 352 bytes)
- `E06.input` (binary, 352 bytes)
- `E07.input` (binary, 416 bytes)
- `E08.input` (binary, 352 bytes)
- `E09.input` (binary, 352 bytes)
- `E10.input` (binary, 352 bytes)
- `E11.input` (binary, 352 bytes)
- `E12.input` (binary, 352 bytes)
- `E13.input` (binary, 352 bytes)
- `E14.input` (binary, 304 bytes)
- `E15.input` (binary, 352 bytes)
- `E16.input` (binary, 416 bytes)
- `E17.input` (binary, 352 bytes)
- `E18.input` (binary, 352 bytes)
- `E19.input` (binary, 352 bytes)
- `M01.patch` (~46 tok, tiny) — --- original.mind
- `M02.patch` (~45 tok, tiny) — --- original.mind
- `M03.patch` (~48 tok, tiny) — --- original.mind
- `M04.patch` (~120 tok, small) — --- original.mind
- `M05.patch` (~34 tok, tiny) — --- original.mind
- `M06.patch` (~31 tok, tiny) — --- original.mind
- `M07.patch` (~44 tok, tiny) — --- original.mind
- `M08.patch` (~54 tok, small) — --- original.mind
- `MUTATIONS.json` (~639 tok, large)
- `MUTATIONS.tsv` (~26 tok, tiny) — M01	C02	1
- `OBSERVED.tsv` (~7784 tok, huge) — path	exit	payload_sha256
- `run.sh` (~1434 tok, large) — Copyright 2026 STARGA Inc. Transport only; MIND performs the assertions.
### `examples/mindc_mind/testdata/native_c_variadic_plan/`

- `allocation-zero.patch` (~29 tok, tiny) — --- a/native_c_variadic_plan.mind
- `BUILD_HASHES.tsv` (~327 tok, medium) — variant	backend	exit	executable_sha256
- `C01.expected` (binary, 272 bytes)
- `C01.input` (binary, 496 bytes)
- `C02.expected` (binary, 432 bytes)
- `C02.input` (binary, 704 bytes)
- `C03.expected` (binary, 432 bytes)
- `C03.input` (binary, 704 bytes)
- `C04.expected` (binary, 592 bytes)
- `C04.input` (binary, 912 bytes)
- `C05.expected` (binary, 592 bytes)
- `C05.input` (binary, 912 bytes)
- `C06.expected` (binary, 432 bytes)
- `C06.input` (binary, 704 bytes)
- `C07.expected` (binary, 432 bytes)
- `C07.input` (binary, 704 bytes)
- `C08.expected` (binary, 1232 bytes)
- `C08.input` (binary, 1744 bytes)
- `C09.expected` (binary, 1392 bytes)
- `C09.input` (binary, 1952 bytes)
- `C10.expected` (binary, 1872 bytes)
- `C10.input` (binary, 2576 bytes)
- `C11.expected` (binary, 2832 bytes)
- `C11.input` (binary, 3824 bytes)
- `C12.expected` (binary, 16 bytes)
- `C12.input` (binary, 240 bytes)
- `C13.expected` (binary, 432 bytes)
- `C13.input` (binary, 704 bytes)
- `C14.expected` (binary, 16 bytes)
- `C14.input` (binary, 240 bytes)
- `C15.expected` (binary, 16 bytes)
- `C15.input` (binary, 288 bytes)
- `C16.expected` (binary, 16 bytes)
- `C16.input` (binary, 240 bytes)
- `C17.expected` (binary, 16 bytes)
- `C17.input` (binary, 240 bytes)
- `C18.expected` (binary, 16 bytes)
- `C18.input` (binary, 240 bytes)
- `C19.expected` (binary, 16 bytes)
- `C19.input` (binary, 240 bytes)
- `C20.expected` (binary, 16 bytes)
- `C20.input` (binary, 240 bytes)
- `C21.expected` (binary, 16 bytes)
- `C21.input` (binary, 240 bytes)
- `C22.expected` (binary, 16 bytes)
- `C22.input` (binary, 240 bytes)
- `C23.expected` (binary, 16 bytes)
- `C23.input` (binary, 240 bytes)
- `C24.expected` (binary, 16 bytes)
- `C24.input` (binary, 240 bytes)
- `C25.expected` (binary, 16 bytes)
- `C25.input` (binary, 240 bytes)
- `C26.expected` (binary, 16 bytes)
- `C26.input` (binary, 240 bytes)
- `C27.expected` (binary, 16 bytes)
- `C27.input` (binary, 240 bytes)
- `C28.expected` (binary, 16 bytes)
- `C28.input` (binary, 240 bytes)
- `C29.expected` (binary, 16 bytes)
- `C29.input` (binary, 240 bytes)
- `C30.expected` (binary, 16 bytes)
- `C30.input` (binary, 240 bytes)
- `C31.expected` (binary, 16 bytes)
- `C31.input` (binary, 240 bytes)
- `C32.expected` (binary, 16 bytes)
- `C32.input` (binary, 240 bytes)
- `C33.expected` (binary, 16 bytes)
- `C33.input` (binary, 240 bytes)
- `C34.expected` (binary, 16 bytes)
- `C34.input` (binary, 240 bytes)
- `C35.expected` (binary, 16 bytes)
- `C35.input` (binary, 240 bytes)
- `C36.expected` (binary, 16 bytes)
- `C36.input` (binary, 240 bytes)
- `C37.expected` (binary, 16 bytes)
- `C37.input` (binary, 240 bytes)
- `C38.expected` (binary, 16 bytes)
- `C38.input` (binary, 240 bytes)
- `C39.expected` (binary, 16 bytes)
- `C39.input` (binary, 288 bytes)
- `C40.expected` (binary, 16 bytes)
- `C40.input` (binary, 288 bytes)
- `C41.expected` (binary, 16 bytes)
- `C41.input` (binary, 288 bytes)
- `C42.expected` (binary, 16 bytes)
- `C42.input` (binary, 288 bytes)
- `C43.expected` (binary, 16 bytes)
- `C43.input` (binary, 288 bytes)
- `C44.expected` (binary, 16 bytes)
- `C44.input` (binary, 288 bytes)
- `C45.expected` (binary, 16 bytes)
- `C45.input` (binary, 288 bytes)
- `C46.expected` (binary, 16 bytes)
- `C46.input` (binary, 240 bytes)
- `C47.expected` (binary, 16 bytes)
- `C47.input` (binary, 240 bytes)
- `C48.expected` (binary, 16 bytes)
- `C48.input` (binary, 240 bytes)
- `C49.expected` (binary, 16 bytes)
- `C49.input` (binary, 240 bytes)
- `C50.expected` (binary, 16 bytes)
- `C50.input` (binary, 240 bytes)
- `C51.expected` (binary, 272 bytes)
- `C51.input` (binary, 496 bytes)
- `C52.expected` (binary, 432 bytes)
- `C52.input` (binary, 704 bytes)
- `C53.expected` (binary, 41072 bytes)
- `C53.input` (binary, 53536 bytes)
- `C54.expected` (binary, 41072 bytes)
- `C54.input` (binary, 65776 bytes)
- `C55.expected` (binary, 272 bytes)
- `C55.input` (binary, 496 bytes)
- `C56.expected` (binary, 272 bytes)
- `C56.input` (binary, 496 bytes)
- `C57.expected` (binary, 272 bytes)
- `C57.input` (binary, 496 bytes)
- `C58.expected` (binary, 272 bytes)
- `C58.input` (binary, 496 bytes)
- `C59.expected` (binary, 2672 bytes)
- `C59.input` (binary, 4240 bytes)
- `C60.expected` (binary, 1072 bytes)
- `C60.input` (binary, 1776 bytes)
- `C61.expected` (binary, 432 bytes)
- `C61.input` (binary, 704 bytes)
- `C62.expected` (binary, 16 bytes)
- `C62.input` (binary, 336 bytes)
- `C63.expected` (binary, 16 bytes)
- `C63.input` (binary, 336 bytes)
- `C64.expected` (binary, 16 bytes)
- `C64.input` (binary, 288 bytes)
- `C65.expected` (binary, 16 bytes)
- `C65.input` (binary, 240 bytes)
- `CASES.json` (~6542 tok, huge)
- `CASES.tsv` (~587 tok, large) — C01	baseline	0	proposal01-no-tail
- `COMBINED_BUILD_HASHES.tsv` (~317 tok, medium) — variant	backend	executable_sha256
- `COMBINED_OBSERVED.tsv` (~7674 tok, huge) — variant	case	backend	exit	stdout_sha256	stderr_sha256	receipt_sha256
- `CUSTODY_OBSERVED.tsv` (~204 tok, medium) — control	exit	live_marker	captured_marker	stdout_sha256	stderr_sha256
- `M01.patch` (~92 tok, small) — --- a/native_c_variadic_plan.mind
- `M02.patch` (~31 tok, tiny) — --- a/native_c_variadic_plan.mind
- `M03.patch` (~36 tok, tiny) — --- a/native_c_variadic_plan.mind
- `M04.patch` (~134 tok, small) — --- a/native_c_variadic_plan.mind
- `M05.patch` (~130 tok, small) — --- a/native_c_variadic_plan.mind
- `M06.patch` (~31 tok, tiny) — --- a/native_c_variadic_plan.mind
- `MUTATIONS.json` (~597 tok, large)
- `MUTATIONS.tsv` (~21 tok, tiny) — M01	C12	1
- `OBSERVED.tsv` (~7674 tok, huge) — variant	case	backend	exit	stdout_sha256	stderr_sha256	receipt_sha256
- `run.sh` (~905 tok, large) — Copyright 2026 STARGA Inc. Transport only; MIND performs every assertion.
### `examples/mindc_mind/testdata/native_dynamic_allocator/`

- `allocator.s` (~4044 tok, huge) — .intel_syntax noprefix
- `check.c` (~1899 tok, huge)
- `gen_mind.py` (~502 tok, large) — rt.bin -> MIND byte-emitter functions (nb_le 7-byte chunks; nb_i64le for
- `probe.s` (~296 tok, medium) — # Copyright 2026 STARGA Inc. Diagnostic register canaries only.
### `examples/mindc_mind/testdata/native_elf_oracle/`

- `add.elf` (binary, 492 bytes)
- `if_ret.elf` (binary, 598 bytes)
- `MANIFEST.txt` (~152 tok, small) — # Frozen native-ELF oracle references, captured before #15 deletes src/native.
- `recursion.elf` (binary, 699 bytes)
- `struct_field.elf` (binary, 664 bytes)
- `value_if.elf` (binary, 585 bytes)
### `examples/mindc_mind/testdata/native_lane_values/`

- `casts_edges.mind` (~567 tok, large) — Author: STARGA Inc. Full-width i64 <-> f64 conversion edges (NaN, +-inf, +-2^63,
- `casts_exprs.mind` (~421 tok, medium) — Author: STARGA Inc. i64 <-> f64 conversions of expression-shaped operands:
- `i32_param_range.mind` (~224 tok, medium) — N2: an i32/u32 parameter gets no entry wrap (only 8/16-bit parameters do).
- `licm_addr.mind` (~106 tok, small) — LICM slice 1 must NOT hoist `t` (native-only row: references are native-only):
- `licm_arith_mix.mind` (~211 tok, medium) — LICM slice 1 keeps the narrow code of a hoisted name in plain arithmetic too.
- `licm_basic.mind` (~101 tok, small) — LICM slice 1: `let t` and `let u` hoist out of the loop (names, literals, * + << ^).
- `licm_branch_loop.mind` (~154 tok, small) — LICM slice 1 reaches loops inside an if arm and inside a nested block; a loop
- `licm_chain.mind` (~143 tok, small) — LICM slice 1: a chain of invariant lets after a `continue` hoists as a whole
- `licm_counter.mind` (~100 tok, small) — LICM slice 1 must NOT hoist: `t` reads the loop counter the condition reads and
- `licm_div_zero.mind` (~414 tok, medium) — LICM slice 1 never hoists `/` or `%`: a division by zero in a loop that runs zero
- `licm_float.mind` (~117 tok, small) — LICM slice 1 on f64 lets: `t` (a product of invariant f64 names) hoists with its
- `licm_for.mind` (~76 tok, small) — LICM slice 1 on a desugared range `for`: `t` hoists, `u` reads the loop variable.
- `licm_if_assign.mind` (~120 tok, small) — LICM slice 1 must NOT hoist `t`: its operand is assigned inside an if arm of the
- `licm_narrow.mind` (~129 tok, small) — LICM slice 1 keeps a hoisted let's declared width: `a` wraps to i32 where it is
- `licm_nested.mind` (~157 tok, small) — LICM slice 1, nested While: `c` hoists out of the inner loop and again out of the
- `licm_shadow.mind` (~138 tok, small) — LICM slice 1 with a body-local shadow (native-only row: the MLIR leg refuses
- `licm_shl_u16.mind` (~222 tok, medium) — LICM slice 1 keeps the narrow code of a hoisted name: `<<` on a u16 left masks
- `licm_shl_u8.mind` (~286 tok, medium) — LICM slice 1 keeps the narrow code of a hoisted name. `y` reads `x`, both hoist as a
- `licm_shr_i8.mind` (~263 tok, medium) — LICM slice 1 keeps the narrow code of a hoisted name: `>>` on an i8 left masks
- `licm_store.mind` (~119 tok, small) — LICM slice 1 must NOT hoist across a store: `v` loads what the loop just stored
- `loop_copy_bare.mind` (~146 tok, small) — Loop-carried copy: `b = a; a = n0;` in a while loop, n0 a plain let. The
- `loop_copy_call.mind` (~131 tok, small) — Loop-carried copy with a call in the let: n0 = id(a + b + i); b = a; a = n0.
- `loop_copy_chain4.mind` (~191 tok, small) — A 4-value copy chain: t = id(a + d + i); d = c; c = b; b = id(a + t); a = t.
- `loop_copy_mut_temp.mind` (~128 tok, small) — Loop-carried copy through a mutable temp declared before the loop:
- `loop_copy_round8.mind` (~222 tok, medium) — An 8-value round chain: each trip shifts h <- g <- f <- e and d <- c <- b <- a
- `narrow_wrap_edges.mind` (~255 tok, medium) — W0 width coverage: each narrow primitive wraps at its own boundary, so a
- `nb_opt_level_cases.mind` (~847 tok, large) — Copyright 2026 STARGA Inc.
- `print_bool.mind` (~57 tok, small) — print: a bool writes 1 or 0, then "\n"; `true` and `false` are 1 and 0.
- `print_flow.mind` (~91 tok, small) — print: inside a loop and a branch; an empty literal writes nothing; the
- `print_int_edges.mind` (~105 tok, small) — print: an integer writes its signed decimal, then "\n" in a second write
- `print_mixed.mind` (~105 tok, small) — print: several arguments in one statement, in order; a call as the first
- `print_str_lit.mind` (~84 tok, small) — print: a string literal writes its bytes and no newline; UTF-8 bytes pass
### `examples/mindc_mind/testdata/native_libc_binding/`

- `abi_accept_struct_ptr.mind` (~69 tok, small) — Positive control for the E2040 rows: the pointer form is accepted.
- `abi_callees.c` (~1219 tok, large)
- `abi_refuse_aapcs.mind` (~45 tok, tiny) — Refusal row: AAPCS has no lowering row on x86-64 (E2039).
- `abi_refuse_struct_param.mind` (~55 tok, small) — Refusal row: a by-value repr(C) struct parameter (E2040). Pass a pointer.
- `abi_refuse_struct_return.mind` (~56 tok, small) — Refusal row: a by-value repr(C) struct return (E2040). Pass a pointer.
- `abi_refuse_win64_varargs.mind` (~46 tok, tiny) — Refusal row: Win64 varargs have no lowering row (E2039).
- `abi_sysv_float.mind` (~92 tok, small) — extern "C" ABI row, System V: the f32 and f64 returns, kept apart from
- `abi_sysv.mind` (~489 tok, medium) — extern "C" ABI rows, System V (extern "C" on an x86-64 Linux target) convention: each pub fn calls one C callee of
- `abi_win64_float.mind` (~96 tok, small) — extern "C" ABI row, Win64: the f32 and f64 returns, kept apart from
- `abi_win64.mind` (~488 tok, medium) — extern "C" ABI rows, Win64 (callconv(.win64)) convention: each pub fn calls one C callee of
- `views.c` (~149 tok, small)
### `examples/mindc_mind/testdata/native_libc_io/`

- `generated_source.mind` (~55 tok, small)
- `identity_row.txt` (~46 tok, tiny) — error[native-lane]: the program is outside candidate native-lane-v1 (lane.extern
- `metadata_cases.json` (~1292 tok, large)
- `mutations.json` (~102 tok, small)
- `reference.mind` (~487 tok, medium) — Copyright 2026 STARGA Inc.
- `runtime_cases.json` (~12563 tok, huge)
- `runtime_main.s` (~514 tok, large) — # Copyright 2026 STARGA Inc.
- `thunks.s` (~262 tok, medium) — # Copyright 2026 STARGA Inc.
### `examples/mindc_mind/testdata/native_libc_rows/`

- `creat_probe.mind` (~545 tok, large) — creat on the C ABI path: a descriptor of a new regular file whose mode is the
- `declared_unused.mind` (~447 tok, medium) — A declaration of every row of the libc block and of the seven older network
- `getenv_probe.mind` (~398 tok, medium) — getenv on the C ABI path: the result is the full 64-bit pointer of the
- `older_seven.mind` (~706 tok, large) — The seven older rows the same program family declares: socket, setsockopt,
- `readlink_probe.mind` (~338 tok, medium) — readlink on the C ABI path: a positive 64-bit length, the third argument as a
- `refused_bind_two_args.mind` (~79 tok, small) — fn main() -> i64 {
- `refused_close_const_result.mind` (~72 tok, small) — fn main() -> i64 {
- `refused_getenv_mut_result.mind` (~72 tok, small) — fn main() -> i64 {
- `refused_listen_i64_fd.mind` (~76 tok, small) — fn main() -> i64 {
- `refused_sendto_i32_len.mind` (~106 tok, small) — fn main() -> i64 {
- `refused_unknown_accept.mind` (~87 tok, small) — fn main() -> i64 {
- `ROWS.txt` (~2033 tok, huge) — # Rows 89..98 of the native extern table (nbx_libc_rows.mind, mark nblibc03), on
- `tcp_rows.mind` (~1022 tok, large) — TCP over loopback on the C ABI path: socket, bind to 127.0.0.1 port 0, listen,
- `twin_bind_two_args.mind` (~86 tok, small) — fn main() -> i64 {
- `twin_close_const_result.mind` (~71 tok, small) — fn main() -> i64 {
- `twin_getenv_mut_result.mind` (~72 tok, small) — fn main() -> i64 {
- `twin_listen_i64_fd.mind` (~78 tok, small) — fn main() -> i64 {
- `twin_sendto_i32_len.mind` (~105 tok, small) — fn main() -> i64 {
- `twin_unknown_accept.mind` (~90 tok, small) — fn main() -> i64 {
- `udp_rows.mind` (~1197 tok, large) — UDP over loopback on the C ABI path: sendto and recvfrom (six arguments, a
### `examples/mindc_mind/testdata/native_libc_startup/`

- `gen_mind.py` (~480 tok, medium) — Emit MIND byte constructors from the independently assembled ABI reference.
- `preload.c` (~285 tok, medium)
- `startup.s` (~774 tok, large) — # Copyright 2026 STARGA Inc.
### `examples/mindc_mind/testdata/native_measure/`

- `measure_encoder_table.tsv` (~321 tok, medium) — # M-A3 G-ENC table: each asm is encoded with llvm-mc --triple=x86_64 -x86-asm-sy
- `measure_lfence.mind` (~170 tok, small) — Copyright 2026 STARGA Inc.
- `measure_perf_event_open.mind` (~695 tok, large) — Copyright 2026 STARGA Inc.
- `measure_rdtscp_helper.mind` (~139 tok, small) — Copyright 2026 STARGA Inc.
- `measure_rdtscp.mind` (~196 tok, small) — Copyright 2026 STARGA Inc.
### `examples/mindc_mind/testdata/native_misc/`

- `a2_ptrcast_ok.mind` (~350 tok, medium) — A2: pointer casts between PROVEN 64-bit integers and pointers are
- `a2_refuse_ptr_let_annotation.mind` (~68 tok, small) — A2 boundary: a pointer-TYPED let annotation is not parsed by the native port
- `a2_refuse_ptr_to_i32.mind` (~56 tok, small) — A2 fail-closed: a pointer narrowed to i32 is a truncation, not a 64-bit
- `a2_refuse_ptr_to_u8.mind` (~35 tok, tiny) — A2 fail-closed: a pointer narrowed to u8.
- `a2_refuse_src_cmp.mind` (~69 tok, small) — A2 fail-closed: a pointer cast whose source is not PROVEN to be a 64-bit
- `a2_refuse_src_f64let.mind` (~69 tok, small) — A2 fail-closed: a pointer cast whose source is not PROVEN to be a 64-bit
- `a2_refuse_src_i32let.mind` (~69 tok, small) — A2 fail-closed: a pointer cast whose source is not PROVEN to be a 64-bit
- `a2_refuse_src_narrowcall.mind` (~46 tok, tiny) — A2 fail-closed: a pointer cast of a call whose declared return type is i32.
- `a2_refuse_src_narrowparam.mind` (~34 tok, tiny) — A2 fail-closed: a pointer cast of an i32 PARAMETER.
- `a2_refuse_src_u32let.mind` (~71 tok, small) — A2 fail-closed: a pointer cast whose source is not PROVEN to be a 64-bit
- `a2_refuse_src_unannotated.mind` (~69 tok, small) — A2 fail-closed: a pointer cast whose source is not PROVEN to be a 64-bit
- `a8_const_mixed.mind` (~91 tok, small) — A8: `;`-terminated and `;`-less consts interleaved with fns, a const
- `a8_const_narrow_types.mind` (~151 tok, small) — A8: consts declared with each narrow, unsigned and pointer-sized integer
- `a8_const_narrow_u8.mind` (~72 tok, small) — A8: a narrow-typed const inlines exactly like an i64 one. mindc-Rust neither
- `a8_const_nosemi.mind` (~331 tok, medium) — A8: top-level `const` items WITHOUT a trailing `;` (the `SX_*` table
- `a8_const_usize.mind` (~30 tok, tiny) — A8: the common `const N: usize = ...;` shape. Expected exit 8.
- `a8_refuse_bool_const.mind` (~59 tok, small) — A8 fail-closed: `bool` shares the integer scalar class in the type checker,
- `a8_refuse_const_assign.mind` (~78 tok, small) — A8 refusal: assignment to a top-level const. mindc-Rust assigns a
- `a8_refuse_const_compound.mind` (~59 tok, small) — A8 refusal: compound assignment to a top-level const. MLIR: K += 2 on the
- `a8_refuse_constexpr.mind` (~64 tok, small) — A8 fail-closed: a `;`-less const whose initializer is an EXPRESSION. The
- `a8_refuse_float_const.mind` (~61 tok, small) — A8 fail-closed: an int initializer on an f64 const is a type error in
- `a8_refuse_u64_const_div.mind` (~121 tok, small) — A8 x W2-trap: a fn that reads a u64 (or usize) const can hold a u64 value
- `a8_str_const.mind` (~300 tok, medium) — A8 refusal (deferred): string consts `const NAME: str|&str|String = "..."`,
- `a8_str_refuse_i64type.mind` (~74 tok, small) — A8 refusal: a string initializer under an `i64` declared type. mindc-Rust
- `a9_arr_fold.mind` (~118 tok, small) — Array literal elements that fold by spec 3.2 (0 - 1, 2 * 3, (9 - 2) / 2, -(4)),
- `n2_assign_valueif.mind` (~104 tok, small) — N2: the twin has no wrap for a value-if right side of a narrow assignment, so it
- `n2_assign_widths.mind` (~134 tok, small) — N2: an assignment to a narrow local wraps the new value at the declared width,
- `n2_call_arith.mind` (~77 tok, small) — N2: arithmetic on a u8 call result in a single-expression body is u8 arithmetic,
- `n2_let_fn_body.mind` (~97 tok, small) — N2: a typed `let` in a helper fn body with a param keeps its declared width in the
- `n2_let_valueif_widths.mind` (~154 tok, small) — N2: a narrow let bound to a value-if wraps the merged value to its declared width,
- `n2_let_widths.mind` (~216 tok, medium) — N2: a typed `let` keeps its declared width in the trace note (Rust mask_narrow_let).
- `n2_param_widths.mind` (~130 tok, small) — N2: an 8/16-bit parameter is wrapped once at entry to its declared width, in the
- `n2_u64_tag.mind` (~76 tok, small) — N2: a u64 binding keeps its unsigned type on the let and on each assignment (an
- `n2_u64_valueif.mind` (~76 tok, small) — N2: the twin has no u64 tag after a value-if initializer, so it refuses the function
- `st_cond_use.mind` (~237 tok, medium) — Store intrinsic used as a VALUE: the C runtime (MLIR build) returns 0 for
- `st_let_init.mind` (~161 tok, small) — Store intrinsic used as a VALUE: the C runtime (MLIR build) returns 0 for
- `st_main_tail.mind` (~157 tok, small) — Store intrinsic used as a VALUE: the C runtime (MLIR build) returns 0 for
- `st_operand_i16.mind` (~167 tok, small) — Store intrinsic used as a VALUE: the C runtime (MLIR build) returns 0 for
- `st_pos_assign.mind` (~198 tok, small) — Store intrinsic used as a VALUE: the C runtime (MLIR build) returns 0 for
- `st_pos_call_arg.mind` (~196 tok, small) — Store intrinsic used as a VALUE: the C runtime (MLIR build) returns 0 for
- `st_pos_if_cond.mind` (~201 tok, medium) — Store intrinsic used as a VALUE: the C runtime (MLIR build) returns 0 for
- `st_pos_while_cond.mind` (~205 tok, medium) — Store intrinsic used as a VALUE: the C runtime (MLIR build) returns 0 for
- `st_return_i32.mind` (~177 tok, small) — Store intrinsic used as a VALUE: the C runtime (MLIR build) returns 0 for
- `st_tail_chain.mind` (~284 tok, medium) — Store intrinsic used as a VALUE: the C runtime (MLIR build) returns 0 for
- `st_tail_valueif_main.mind` (~198 tok, small) — Store intrinsic used as a VALUE: the C runtime (MLIR build) returns 0 for
- `st_tail_w_set.mind` (~182 tok, small) — Store intrinsic used as a VALUE: the C runtime (MLIR build) returns 0 for
- `st_valueif_i8.mind` (~179 tok, small) — Store intrinsic used as a VALUE: the C runtime (MLIR build) returns 0 for
### `examples/mindc_mind/testdata/native_record_array/`

- `aggregate_refuse.mind` (~33 tok, tiny) — fn main() -> i64 {
- `alias_refuse.mind` (~28 tok, tiny) — fn main() -> i64 {
- `computed_i64_bitwise.mind` (~40 tok, tiny) — fn main() -> i64 {
- `computed_i64.mind` (~50 tok, small) — fn twice(x: i64) -> i64 {
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
- `README.md` (~760 tok, large) — Pure-MIND native record-array fixtures
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

- `constant_oob.mind` (~28 tok, tiny) — fn main() -> i64 {
- `construct_only.mind` (~32 tok, tiny) — fn main() -> i64 {
- `copy_i64.mind` (~35 tok, tiny) — fn main() -> i64 {
- `direct_i64.mind` (~42 tok, tiny) — fn main() -> i64 {
- `direct_i64_prefold.mind` (~158 tok, small) — direct_i64.mind plus the intrinsic name literals, so the field prefold takes
- `direct_u8.mind` (~28 tok, tiny) — fn main() -> i64 {
- `MANIFEST.txt` (~454 tok, medium) — # Canonical record-trace fixtures — recorded evidence bound to its input.
- `README.md` (~872 tok, large) — Canonical record-trace fixtures
- `reference.mind` (~212 tok, medium) — fn direct_i64_reference() -> i64 {
- `scalar_owner.mind` (~41 tok, tiny) — fn main() -> i64 {
- `scalar_owner_named.mind` (~157 tok, small) — The prefold passes rewrite field reads into `__mind_load_i64` calls whose
- `wrong_owner.mind` (~40 tok, tiny) — fn main() -> i64 {
- `wrong_type.mind` (~28 tok, tiny) — fn main() -> i64 {
### `examples/mindc_mind/testdata/native_scoped_fixed/`

- `independent-CASES.json` (~5969 tok, huge) — Keys: schema, status, source_base, prefix, cases
- `independent-EARLY_SOURCE_REVIEW.md` (~1656 tok, huge) — Early scoped fixed-call source review
- `independent-PACKETS.json` (~2920 tok, huge) — Keys: schema, status, wire, HOME_requirement, packets
- `independent-SCOPE_CASES.json` (~1048 tok, large) — Keys: schema, status, cases
- `independent-SUPPLEMENT.json` (~972 tok, large) — Keys: schema, status, cases, amendment
- `independent-VISIBILITY_SUPPLEMENT.json` (~238 tok, medium) — Keys: schema, status, id, purpose, source
### `examples/mindc_mind/testdata/native_scoped_fixed/observers/`

- `geometry.mind` (~714 tok, large) — STARGA Inc. Geometry ownership observer, not a production context validator.
- `geometry-owner-mutant.mind` (~714 tok, large) — STARGA Inc. Geometry ownership observer, not a production context validator.
- `staged-frame-range-mutant.mind` (~1273 tok, large) — STARGA Inc. Exact source-extracted staged-state helper controls.
- `staged-rebase-mutant.mind` (~1278 tok, large) — STARGA Inc. Exact source-extracted staged-state helper controls.
- `staged-rollback-mutant.mind` (~1280 tok, large) — STARGA Inc. Exact source-extracted staged-state helper controls.
- `staged-scope-end-mutant.mind` (~1281 tok, large) — STARGA Inc. Exact source-extracted staged-state helper controls.
- `staged-state.mind` (~1280 tok, large) — STARGA Inc. Exact source-extracted staged-state helper controls.
### `examples/mindc_mind/testdata/native_scoped_fixed/`

- `README.md` (~179 tok, small) — NSF/1 frozen diagnostic catalogue
### `examples/mindc_mind/testdata/native_scoped_fixed/sources/`

- `A01-branch-local-call-targets.mind` (~85 tok, small) — fn af_helper(x:i64)->i64 {x+7}
- `A02-nested-foreign-call.mind` (~33 tok, tiny) — fn main()->i64 {let x:i64=close(close((0-1) as i32)) as i64;if x!=0-1{return 91;}0}
- `A03-i64-argument.mind` (~23 tok, tiny) — fn main()->i64 {let x:i64=0-1;close(x) as i64}
- `A04-u32-argument.mind` (~23 tok, tiny) — fn main()->i64 {let x:u32=1;close(x) as i64}
- `A05-f64-argument.mind` (~23 tok, tiny) — fn main()->i64 {let x:f64=1.0;close(x) as i64}
- `A06-parameter-callee-shadow.mind` (~27 tok, tiny) — fn main()->i64 {let close:i64=3;close((0-1) as i32) as i64}
- `A07-unknown-library-symbol.mind` (~25 tok, tiny) — fn main()->i64 {af_unknown((0-1) as i32) as i64}
- `A08-reserved-intrinsic-declaration.mind` (~21 tok, tiny) — fn main()->i64 {__mind_alloc(8)}
- `A09-alias-import.mind` (~11 tok, tiny) — fn main()->i64 {0}
- `A10-primitive-type-shadow.mind` (~13 tok, tiny) — fn main()->i64 {0}
- `A11-calls-256.mind` (~2716 tok, huge) — fn main()->i64 {let mut total:i64=0;
- `A12-calls-257.mind` (~2727 tok, huge) — fn main()->i64 {let mut total:i64=0;
- `C01-no-c-reference.mind` (~6 tok, tiny) — fn main()->i64 { 0 }
- `C02-dormant-declaration-reference.mind` (~17 tok, tiny) — fn main()->i64 { 0 }
- `P01-close-i32-result.mind` (~36 tok, tiny) — fn main()->i64 {
- `P02-close-observed-tls-errno.mind` (~91 tok, small) — fn main()->i64 {
- `P03-read-ebadf.mind` (~103 tok, small) — fn main()->i64 {
- `P04-write-ebadf.mind` (~103 tok, small) — fn main()->i64 {
- `P05-empty-nonblocking-pipe.mind` (~96 tok, small) — fn main()->i64 {
- `P06-positioned-pipe-espipe.mind` (~96 tok, small) — fn main()->i64 {
- `P07-nio-shortcuts-preserve-errno.mind` (~92 tok, small) — fn main()->i64 {
- `P08-close-owned-inherited-fd.mind` (~158 tok, small) — fn main()->i64 {
- `P09-typed-param-loop-live-values.mind` (~109 tok, small) — fn scfc_apply(fd:i32)->i64 {
- `P10-nested-argument-once.mind` (~104 tok, small) — fn scfc_argument(p:i64)->i32 {
- `P11-public-std-and-user-close.mind` (~73 tok, small) — fn main()->i64 {
- `R01-undeclared-close.mind` (~12 tok, tiny) — fn main()->i64 { close((0-1) as i32) as i64 }
- `R02-undeclared-errno.mind` (~12 tok, tiny) — fn main()->i64 { __errno_location() as i64 }
- `R03-duplicate-same-unit.mind` (~34 tok, tiny) — fn main()->i64 { close((0-1) as i32) as i64 }
- `R04-wrong-close-parameter.mind` (~21 tok, tiny) — fn main()->i64 { close(0-1) as i64 }
- `R05-wrong-close-result.mind` (~21 tok, tiny) — fn main()->i64 { close((0-1) as i32) }
- `R06-errno-wrong-qualifier.mind` (~26 tok, tiny) — fn main()->i64 { __errno_location() as i64 }
- `R07-errno-wrong-depth.mind` (~27 tok, tiny) — fn main()->i64 { __errno_location() as i64 }
- `R08-local-callee-shadow.mind` (~27 tok, tiny) — fn main()->i64 { let close:i64=7; close((0-1) as i32) as i64 }
- `R09-typed-bool-shadow.mind` (~39 tok, tiny) — fn scfc_bad(fd:i32)->i64 {
- `R10-extra-argument.mind` (~25 tok, tiny) — fn main()->i64 { close((0-1) as i32,0 as i32) as i64 }
- `R11-local-body-and-extern.mind` (~30 tok, tiny) — fn close(fd:i32)->i32 { fd }
- `R12-fixed-looking-variadic.mind` (~34 tok, tiny) — fn main()->i64 { open(0 as *const u8,0 as i32,0 as i32) as i64 }
- `R13-nonreturning-call.mind` (~21 tok, tiny) — fn main()->i64 { _exit(0 as i32); 0 }
- `R14-untyped-close-literal.mind` (~21 tok, tiny) — fn main()->i64 { close(0-1) as i64 }
- `R15-private-std-local-edge.mind` (~8 tok, tiny) — fn main()->i64 { proc_errno() }
- `R16-unresolved-user-import.mind` (~28 tok, tiny) — fn main()->i64 { close((0-1) as i32) as i64 }
- `R17-public-std-without-import.mind` (~69 tok, small) — fn main()->i64 {
- `R18-foreign-local-name-collision.mind` (~40 tok, tiny) — fn f(x: i64) -> i64 {
### `examples/mindc_mind/testdata/native_scoped_fixed/`

- `SUPPLEMENT.json` (~4626 tok, huge) — Keys: schema, status, cases, observer_contract, mutations
### `examples/mindc_mind/testdata/native_shm/`

- `shm_fork_rounds.mind` (~1209 tok, large) — Author: STARGA Inc. A5 fork + MAP_SHARED arena + pipe barrier, 10^6 rounds.
- `shm_mmap_probe.mind` (~895 tok, large) — Author: STARGA Inc. A5 shared-memory rows probe: the 6-argument syscall ABI.
### `examples/mindc_mind/testdata/native_source_driver/`

- `catalog.py` (~1414 tok, large) — Original 38 adapter controls and four MIND source mutants; enumeration only."""
- `hash_refusal_catalog.py` (~418 tok, medium) — Additive normal SHA-return fault seams; original adapter cases unchanged."""
- `note_catalog.py` (~929 tok, large) — Independent 76-input note supplement and two actual MIND mutations."""
- `note_fixture_tail.mind` (~476 tok, medium)
### `examples/mindc_mind/testdata/native_string_handle/`

- `addr_write.mind` (~151 tok, small) — fn show(s: String) -> i64 {
- `empty_literal.mind` (~24 tok, tiny) — fn len_of(s: String) -> i64 {
- `escapes.mind` (~64 tok, small) — fn byte_sum(s: String) -> i64 {
- `forwarded.mind` (~52 tok, small) — fn sink(s: String, k: i64) -> i64 {
- `handle_driver.mind` (~862 tok, large) — Copyright 2026 STARGA Inc.
- `len_edges.mind` (~65 tok, small) — fn len_of(s: String) -> i64 {
- `literal_args_loop.mind` (~79 tok, small) — fn take(s: String, k: i64) -> i64 {
- `loop_and_second.mind` (~56 tok, small) — fn len_of(s: String) -> i64 {
- `loop_print_text.mind` (~32 tok, tiny) — fn main() -> i64 {
- `many_args.mind` (~64 tok, small) — fn mix(a: String, b: i64, c: String, d: i64, e: String, f: i64, g: String, h: i64) -> i64 {
- `many_literals_300.mind` (~2474 tok, huge) — fn len_of(s: String) -> i64 {
- `nul_inside.mind` (~53 tok, small) — fn len_of(s: String) -> i64 {
- `one_literal.mind` (~25 tok, tiny) — fn len_of(s: String) -> i64 {
- `print_arg.mind` (~13 tok, tiny) — fn main() -> i64 {
- `same_text_two_fns.mind` (~51 tok, small) — fn len_of(s: String) -> i64 {
- `through_two_calls.mind` (~92 tok, small) — fn byte_sum(s: String) -> i64 {
### `examples/mindc_mind/testdata/native_trap/`

- `assert_arity3.mind` (~71 tok, small) — Copyright 2026 STARGA Inc.
- `assert_direct.mind` (~118 tok, small) — Copyright 2026 STARGA Inc.
- `assert_near_name.mind` (~70 tok, small) — Copyright 2026 STARGA Inc.
- `sha256_fail.mind` (~268 tok, medium) — Copyright 2026 STARGA Inc.
- `sha256_fips.mind` (~364 tok, medium) — Copyright 2026 STARGA Inc.
- `sha256_nocall.mind` (~69 tok, small) — Copyright 2026 STARGA Inc.
### `examples/mindc_mind/testdata/`

- `process_start_plants.tsv` (~8639 tok, huge) — name	flagged	row	form	code
- `process_starts.tsv` (~10198 tok, huge) — file	function	reason
- `rh_f64_aggregate_canary.mind` (~876 tok, large) — RH f64-aggregate self-host/native canary (RH_REQUIRED_F64_AGGREGATE_SURFACE).
### `examples/mindc_mind/testdata/seed_ledger/`

- `SEED_DRIFT.txt` (~77 tok, small) — # Seed drift: seed-bound files that differ from their SEED_LEDGER.tsv row.
- `SEED_LEDGER.tsv` (~4851 tok, huge) — # Seed drift ledger: one sha256 row per seed-bound file, as of the last re-pin.
### `examples/mindc_mind/testdata/selfhost_loop_bounded/`

- `MANIFEST.txt` (~269 tok, medium) — # Frozen BOUNDED-profile self-host compiler seed: the pure-MIND compiler
### `examples/mindc_mind/testdata/selfhost_loop/`

- `MANIFEST.txt` (~588 tok, large) — # Frozen self-host bootstrap ELF (A6/RI-E1): the checked-in pure-MIND stage0
- `PROVENANCE_298_f64_selfhost_mlir.md` (~1826 tok, huge) — #298 self-host f64 MLIR emit — self-host loop seed refreeze provenance
### `examples/mindc_mind/testdata/`

- `stdlib_manifest.txt` (~1069 tok, large) — # std/*.mind MANIFEST — the single committed source of truth for "which std
### `examples/mindc_mind/testdata/string_borrowed/`

- `reference.c` (~869 tok, large) — Copyright STARGA Inc. Reference infrastructure only, not MIND implementation.
### `examples/mindc_mind/testdata/string_handle_limits/`

- `loop_k_45m.mind` (~58 tok, small) — fn take(s: String, k: i64) -> i64 {
- `loop_k_50m.mind` (~58 tok, small) — fn take(s: String, k: i64) -> i64 {
- `loop_over_2p25.mind` (~57 tok, small) — fn len_of(s: String) -> i64 {
- `loop_print_empty_50m.mind` (~32 tok, tiny) — fn main() -> i64 {
- `loop_take_50m.mind` (~60 tok, small) — fn take(s: String, k: i64) -> i64 {
- `loop_under_2p25.mind` (~57 tok, small) — fn len_of(s: String) -> i64 {
- `store_bytes_faults.mind` (~31 tok, tiny) — fn poke(s: String) -> i64 {
- `store_handle_faults.mind` (~32 tok, tiny) — fn poke(s: String) -> i64 {
### `examples/mindc_mind/`

- `unified_dispatch_smoke.py` (~1598 tok, huge)
- `validate_real_fns_smoke.py` (~2734 tok, huge)
- `VERDICT_SHAPE_RESIDUAL.txt` (~1052 tok, large) — # VERDICT_SHAPE_RESIDUAL.txt — the DECLARED, shrink-only residual of the
- `_w8_native.py` (~13038 tok, huge) — Shared harness for the W8 float smokes (self_host_native_f64_*_smoke.py).
- `while_struct_smoke.py` (~1031 tok, large)
### `examples/mind_mirror_v04/`

- `mirror_gate.py` (~3089 tok, huge) — Gate for the pure-MIND MIC3 v0x04 declared-prefix mirror.
- `RUNNER_REPLACEMENT.md` (~784 tok, large) — `mirror_gate.py` — replacement checklist
### `examples/mind_mirror_v04/testdata/`

- `MANIFEST.tsv` (~2715 tok, huge) — # name	expected_exit	bytes	sha256	consumed	note
- `SEMANTIC_MANIFEST.tsv` (~1147 tok, large) — # name	expected_exit	bytes	sha256	consumed	note
### `examples/`

- `mlir_pipeline_demo.sh` (~1647 tok, huge) — MLIR/LLVM Pipeline Demonstration
### `examples/native/`

- `ci_kernel.mind` (~39 tok, tiny)
- `ci_kernel_smoke.py` (~1335 tok, large)
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
### `examples/quant/american_binomial/`

- `main.mind` (~17213 tok, huge) — examples/quant/american_binomial — deterministic American option pricing:
- `Mind.toml` (~21 tok, tiny) — [package]
### `examples/quant/barrier_analytic/`

- `main.mind` (~17419 tok, huge) — examples/quant/barrier_analytic — deterministic closed-form barrier option
- `Mind.toml` (~21 tok, tiny) — [package]
### `examples/quant/black_scholes/`

- `main.mind` (~6300 tok, huge) — examples/quant/black_scholes — deterministic European option pricing.
- `Mind.toml` (~20 tok, tiny) — [package]
### `examples/quant/bond_curve/`

- `main.mind` (~18327 tok, huge) — examples/quant/bond_curve — deterministic fixed-income analytics.
- `Mind.toml` (~20 tok, tiny) — [package]
### `examples/quant/crr_binomial/`

- `main.mind` (~4861 tok, huge) — examples/quant/crr_binomial — deterministic Cox-Ross-Rubinstein lattice.
- `Mind.toml` (~20 tok, tiny) — [package]
### `examples/quant/greeks_higher_order/`

- `main.mind` (~12072 tok, huge) — examples/quant/greeks_higher_order — deterministic second- and third-order
- `Mind.toml` (~22 tok, tiny) — [package]
### `examples/quant/implied_vol/`

- `main.mind` (~10967 tok, huge) — examples/quant/implied_vol — deterministic implied volatility by a bracketed
- `Mind.toml` (~20 tok, tiny) — [package]
### `examples/quant/implied_vol_surface/`

- `main.mind` (~17169 tok, huge) — examples/quant/implied_vol_surface — the raw SVI volatility surface and the
- `Mind.toml` (~22 tok, tiny) — [package]
### `examples/quant/monte_carlo_gbm/`

- `main.mind` (~13933 tok, huge) — examples/quant/monte_carlo_gbm — reproducible Monte-Carlo option pricing.
- `Mind.toml` (~21 tok, tiny) — [package]
### `examples/quant/portfolio_risk/`

- `main.mind` (~19820 tok, huge) — examples/quant/portfolio_risk — deterministic Value-at-Risk, Expected
- `Mind.toml` (~21 tok, tiny) — [package]
### `examples/quant/`

- `README.md` (~3302 tok, huge) — examples/quant — deterministic derivative pricing
### `examples/`

- `README.md` (~2308 tok, huge) — MIND Examples
- `remizov_benchmark.mind` (~6400 tok, huge) — ASPIRATIONAL DEMO — not yet buildable with the open mindc.
- `remizov_feynman.mind` (~2894 tok, huge) — ASPIRATIONAL DEMO — not yet buildable with the open mindc.
- `remizov_gpu.mind` (~2662 tok, huge) — ASPIRATIONAL DEMO — not yet buildable with the open mindc.
- `remizov_inverse.mind` (~2614 tok, huge) — ASPIRATIONAL DEMO — not yet buildable with the open mindc.
- `remizov_solver.mind` (~3802 tok, huge) — ASPIRATIONAL DEMO — not yet buildable with the open mindc.
- `remizov_verify.mind` (~3721 tok, huge) — ASPIRATIONAL DEMO — not yet buildable with the open mindc.
### `examples/research/`

- `kakeya_field_diagnostic.mind` (~124 tok, small) — Kakeya-inspired deterministic directional-field diagnostic.
- `README.md` (~394 tok, medium) — Scientific-computing research examples (not certified theorems)
### `examples/`

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

- `commit-msg` (~239 tok, medium) — #!/usr/bin/env bash
- `post-commit` (~410 tok, medium) — #!/usr/bin/env bash
- `post-merge` (~408 tok, medium) — #!/usr/bin/env bash
- `pre-commit` (~698 tok, large) — #!/usr/bin/env bash
### `.github/actions/avx2-neon-digest/`

- `action.yml` (~807 tok, large) — name: avx2-neon-digest
### `.github/actions/glibc-floor-smoke/`

- `action.yml` (~1395 tok, large) — name: Glibc-floor smoke
### `.github/actions/install-mlir-toolchain/`

- `action.yml` (~1746 tok, huge) — name: Install MLIR toolchain
### `.github/actions/native-lane-count/`

- `action.yml` (~341 tok, medium) — name: native-lane-count
### `.github/actions/nonempty-artifact/`

- `action.yml` (~152 tok, small) — name: nonempty-artifact
### `.github/actions/package-release/`

- `action.yml` (~1257 tok, large) — name: Package release archive
### `.github/actions/qemu-user/`

- `action.yml` (~278 tok, medium) — name: qemu-user
### `.github/actions/release-artifact-smoke/`

- `action.yml` (~3736 tok, huge) — name: Release artifact smoke
### `.github/actions/seed-ledger/`

- `action.yml` (~249 tok, medium) — name: seed-ledger
### `.github/actions/std-evidence-checks/`

- `action.yml` (~779 tok, large) — name: std-evidence-checks
### `.github/actions/std-example-programs/`

- `action.yml` (~566 tok, large) — name: std-example-programs
### `.github/actions/std-fs-checked/`

- `action.yml` (~1148 tok, large) — name: std-fs-checked
### `.github/`

- `CODEOWNERS` (~9 tok, tiny) — *       @star-ga
### `.github/ISSUE_TEMPLATE/`

- `bounty_claim.md` (~56 tok, small)
- `bug_report.md` (~213 tok, medium) — Describe the bug
- `feature_request.md` (~171 tok, small) — Problem Statement
### `.github/`

- `PULL_REQUEST_TEMPLATE.md` (~55 tok, small) — Summary
- `release-drafter.yml` (~94 tok, small) — name-template: 'v$NEXT_PATCH_VERSION'
- `required-ci-jobs.tsv` (~786 tok, large) — # required-ci-jobs.tsv — the CI jobs that MUST have passed on the exact commit
### `.github/scripts/`

- `pr_review.py` (~1297 tok, large) — PR review + public-leak check for this repository (comment only; never pushes or merges)."""
### `.github/workflows/`

- `bench-gate.yml` (~5476 tok, huge) — name: Bench gate
- `cargo-deny.yml` (~333 tok, medium) — name: Cargo Deny
- `crypto-vectors.yml` (~1616 tok, huge) — name: Crypto Vectors
- `docs-claims.yml` (~2937 tok, huge) — name: Docs Claims
- `installers.yml` (~4370 tok, huge) — name: Installers
- `link-check.yml` (~327 tok, medium) — name: Link Check
- `mindcraft.yml` (~938 tok, large) — name: Mindcraft Check
- `pr-review.yml` (~1482 tok, large) — name: pr-review
- `release-drafter.yml` (~205 tok, medium) — name: Release Drafter
- `release.yml` (~5876 tok, huge) — name: Release
### `mind/std/cognitive/`

- `batch_scheduler.mind` (~850 tok, large) — Batch scheduling for inference workloads
- `kv_cache.mind` (~840 tok, large) — KV-Cache for transformer inference
- `speculative.mind` (~891 tok, large) — Speculative decoding with rejection sampling
- `verification.mind` (~948 tok, large) — Verification plane for inference consistency (LCU)
### `runtime-support/`

- `mind_fs_read.h` (~1396 tok, large) — Copyright 2025-2026 STARGA Inc.
- `mind_intrinsics.c` (~23558 tok, huge) — Copyright 2025 STARGA Inc.
### `scripts/`

- `ai_attribution_patterns.sh` (~1628 tok, huge) — shellcheck shell=bash
- `anatomy-hook.sh` (~2508 tok, huge) — STARGA author guard (chained first: a wrong-identity commit must never be created).
- `anatomy.sh` (~3505 tok, huge) — anatomy — Generate ANATOMY.md for any repo
- `arch_gate.sh` (~2218 tok, huge) — arch_gate.sh - architectural-governance gate (arch-mind).
- `cfg_gate_wiring_lint.py` (~4265 tok, huge) — Fail closed when a feature-gated integration test is wired to no CI selector.
- `check_claims.py` (~9033 tok, huge) — Docs-claim CI gate — fail if any public surface drifts from config/capabilities.toml.
- `check_commit_messages.sh` (~1867 tok, huge) — check_commit_messages.sh <rev-range> — no model or tool credit in commit
- `check_gate_wiring_preflight.py` (~7817 tok, huge) — Preflight cargo-test parity checks extracted from check_gate_wiring.
- `check_gate_wiring.py` (~9610 tok, huge) — Wiring lint: a whole-tree gate's CI trigger must cover every path it scans.
- `check_json_not_evidence.sh` (~813 tok, large) — Wedge-integrity gate: JSON is never an evidence-hash preimage.
- `check_no_ai_attribution.sh` (~2217 tok, huge) — Public-artifact hygiene gate: no AI tool/model named as having AUTHORED or
- `check_release_gating.py` (~9316 tok, huge) — check_release_gating.py — machine-checked supply-chain contract for the
- `commit-msg-hook.sh` (~1066 tok, large) — commit-msg-hook.sh — local commit-msg hook: refuse a commit whose MESSAGE
- `enumeration_drift_lint.py` (~2544 tok, huge) — Fail closed when a comment's ENUMERATION drifts from the code it describes.
- `exec_semantics_gate.sh` (~10211 tok, huge) — exec_semantics_gate.sh — run the test tiers CI could not reach, and PROVE they ran.
- `gate_assert.py` (~3627 tok, huge) — gate_assert.py — the ONE definition of "did this gate actually assert anything?"
- `gate_runner_wiring_lint.py` (~7629 tok, huge) — Wiring lint: a workflow may reach a gate ONLY through scripts/run_gate.py.
- `install.ps1` (~2625 tok, huge) — # install.ps1 - mindc one-line installer for Windows (PowerShell)
- `install.sh` (~1112 tok, large) — MIND compiler (mindc) installer — downloads a pre-built binary from the
### `scripts/mind-vs-rust/`

- `Cargo.toml` (~271 tok, medium) — [package]
- `.gitignore` (~3 tok, tiny) — */target-*/
- `run.sh` (~659 tok, large) — Copyright 2026 STARGA Inc. Licensed under the Apache License, Version 2.0.
### `scripts/mind-vs-rust/src/`

- `main.rs` (~2372 tok, huge) — Copyright 2026 STARGA Inc.
### `scripts/`

- `open_error_regression.sh` (~546 tok, large) — Author: STARGA Inc. Build/transport only; executable assertions are MIND.
- `operand_visitor_parity_lint.py` (~1198 tok, large) — Fail closed when the two IR operand visitors drift apart.
- `pqc_hybrid_ci.sh` (~663 tok, large) — Required-CI execution gate for the existing MIC3 two-leg PQC-hybrid controls.
- `pqc_hybrid_cli_ci.sh` (~1160 tok, large) — Required-CI execution gate for the actual-CLI PQC-hybrid signing controls.
- `preflight_bench_pairs.sh` (~1949 tok, huge) — Pair HEAD with the merge-base after the absolute bench floor misses.
- `preflight.sh` (~11689 tok, huge) — preflight.sh — local CI-parity gate. Run before pushing to avoid red CI.
- `quick_perf.sh` (~734 tok, large) — quick_perf.sh — FAST one-sided compile-speed criterion gate.
- `ri_d1_mlir_free_gate.sh` (~2224 tok, huge) — RI-D1 gate assertion #1 — MLIR is un-linkable on the native path (fail-closed
- `run_crypto_vectors.sh` (~7635 tok, huge) — Build every pure-MIND crypto/TLS std module to a shared object and run its
- `run_gate.py` (~9128 tok, huge) — run_gate.py — the shared gate runner that makes a VACUOUS PASS structurally fail.
### `scripts/sdlc/`

- `enforcement_bijection.py` (~1504 tok, huge) — A test must never outlive the rule it asserts.
- `lost_by_merge.py` (~1341 tok, large) — Fail a merge that silently DROPPED a line both parents agreed on.
### `scripts/`

- `sha256_memory_regression.sh` (~1431 tok, large) — Author: STARGA Inc. Build/transport harness; all executable checks are MIND.
- `test_exec_semantics_gate.py` (~7922 tok, huge) — Regression test for scripts/exec_semantics_gate.sh: a FAILING test must red the tier.
- `test_gate_wiring.py` (~10748 tok, huge) — Regression test for scripts/check_gate_wiring.py.
- `test_no_ai_attribution.py` (~8877 tok, huge) — Self-test for the no-AI-attribution gate and for what feeds it.
- `test_smoke_wiring_lint.py` (~10111 tok, huge) — Regression test for the CI-coverage and branch-awareness contracts of
- `verify_ci_green.py` (~3651 tok, huge) — verify_ci_green.py — refuse to release a commit that CI has not proven green.
- `workflow_scan.py` (~6873 tok, huge) — workflow_scan.py — dependency-free structural reader for the GitHub Actions
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
- `package-lock.json` (~12352 tok, huge) — Keys: name, version, lockfileVersion, requires, packages
- `README.md` (~152 tok, small) — @mind/mic-map
### `sdk/ts/mic-map/scripts/`

- `regen_fixtures.sh` (~501 tok, large) — Copyright 2026 STARGA Inc.
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

- `mod.rs` (~11083 tok, huge) — Copyright 2025 STARGA Inc.
- `spelling.rs` (~1251 tok, large) — Copyright 2025 STARGA Inc.
### `src/autodiff/`

- `engine.rs` (~3890 tok, huge) — Copyright 2025 STARGA Inc.
- `mod.rs` (~342 tok, medium) — Copyright 2025 STARGA Inc.
- `rules.rs` (~2392 tok, huge) — Copyright 2025 STARGA Inc.
### `src/bin/`

- `mind-ai.rs` (~9878 tok, huge) — Copyright 2025 STARGA Inc.
### `src/bin/mindc/`

- `mic3_output.rs` (~3438 tok, huge) — Copyright 2025-2026 STARGA Inc.
### `src/build/`

- `artifact.rs` (~1373 tok, large) — Copyright 2025 STARGA Inc.
- `atomic_publish.rs` (~413 tok, medium) — Copyright 2026 STARGA Inc.
### `src/build/cache/`

- `cache_integrity.rs` (~1160 tok, large) — Copyright 2025-2026 STARGA Inc.
- `cache_integrity_tests.rs` (~1673 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `compiler_identity.rs` (~317 tok, medium) — Copyright 2025-2026 STARGA Inc.
### `src/build/`

- `cache.rs` (~6859 tok, huge) — Copyright 2025 STARGA Inc.
- `driver_error.rs` (~699 tok, large) — Copyright 2025 STARGA Inc.
- `error.rs` (~2022 tok, huge) — Copyright 2025 STARGA Inc.
- `mod.rs` (~9306 tok, huge) — Copyright 2025 STARGA Inc.
### `src/build/native_bridge/`

- `admission.rs` (~8804 tok, huge) — Copyright 2025 STARGA Inc.
- `alloc_guard.rs` (~3656 tok, huge) — Copyright 2026 STARGA Inc.
- `child.rs` (~4117 tok, huge) — Copyright 2025 STARGA Inc.
- `child_tests.rs` (~1700 tok, huge) — Copyright 2025 STARGA Inc.
- `entry.rs` (~5184 tok, huge) — Copyright 2025 STARGA Inc.
- `namespace.rs` (~6810 tok, huge) — Copyright 2025 STARGA Inc.
- `open_host.rs` (~1383 tok, large) — Copyright 2026 STARGA Inc.
- `outcome.rs` (~1026 tok, large) — Copyright 2025 STARGA Inc.
### `src/build/`

- `native_bridge.rs` (~8499 tok, huge) — Copyright 2025 STARGA Inc.
### `src/build/native_bridge/`

- `seed.rs` (~6165 tok, huge) — Copyright 2025 STARGA Inc.
- `shadow.rs` (~3785 tok, huge) — Copyright 2025 STARGA Inc.
### `src/build/`

- `native_image.rs` (~4296 tok, huge) — Copyright 2025 STARGA Inc.
- `native_scope.rs` (~7541 tok, huge) — Copyright 2025 STARGA Inc.
- `native_scope_tests.rs` (~5219 tok, huge) — Seam tests for the captured-source snapshot and its module identities.
- `project_transaction.rs` (~2908 tok, huge) — Copyright 2026 STARGA Inc.
- `source_key.rs` (~3219 tok, huge) — Copyright 2025 STARGA Inc.
- `source_snapshot_tests.rs` (~1337 tok, large) — Copyright 2025-2026 STARGA Inc.
- `transaction_identity_tests.rs` (~4097 tok, huge) — Copyright 2026 STARGA Inc.
### `src/cache/`

- `entry.rs` (~977 tok, large) — Copyright 2025-2026 STARGA Inc.
- `fingerprint.rs` (~629 tok, large) — Copyright 2025-2026 STARGA Inc.
- `mod.rs` (~964 tok, large) — Copyright 2025-2026 STARGA Inc.
- `store.rs` (~1112 tok, large) — Copyright 2025-2026 STARGA Inc.
### `src/`

- `c_export_reach.rs` (~1659 tok, huge) — Copyright 2025-2026 STARGA Inc.
### `src/check/`

- `early_failure.rs` (~1137 tok, large) — Copyright 2025 STARGA Inc.
- `fmt_pass.rs` (~730 tok, large) — Copyright 2026 STARGA Inc.
- `gitignore.rs` (~3581 tok, huge) — Copyright 2025 STARGA Inc.
- `mod.rs` (~6790 tok, huge) — Copyright 2025 STARGA Inc.
- `reporter.rs` (~889 tok, large) — Copyright 2025 STARGA Inc.
### `src/`

- `conformance.rs` (~8426 tok, huge) — Per-profile execution counts for one conformance run.
### `src/deps/`

- `mod.rs` (~9388 tok, huge) — Copyright 2025 STARGA Inc.
### `src/diagnostics/`

- `capability.rs` (~9002 tok, huge) — Copyright 2025 STARGA Inc.
- `mod.rs` (~3733 tok, huge) — Copyright 2025 STARGA Inc.
- `refusal.rs` (~1354 tok, large) — Copyright 2025 STARGA Inc.
- `toolchain.rs` (~3902 tok, huge) — Copyright 2026 STARGA Inc.
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
- `abi_gate_struct_ops.rs` (~2866 tok, huge) — Copyright 2026 STARGA Inc.
- `abi_gate_tensor.rs` (~1232 tok, large) — Tensor-shape admission rules for the runnable artifact ABI gate.
- `assert_check.rs` (~1001 tok, large) — Copyright 2025 STARGA Inc.
- `assert_trap.rs` (~421 tok, medium) — Copyright 2026 STARGA Inc.
- `autodiff.rs` (~14268 tok, huge) — Copyright 2025 STARGA Inc.
- `callable_exports.rs` (~596 tok, large) — Copyright 2026 STARGA Inc.
- `canonical_lowering.rs` (~4108 tok, huge) — Copyright 2026 STARGA Inc.
- `canonical_producers.rs` (~734 tok, large) — Copyright 2026 STARGA Inc.
- `closures.rs` (~6164 tok, huge) — Copyright 2025 STARGA Inc.
- `conv2d_grad.rs` (~2397 tok, huge) — Copyright 2025 STARGA Inc.
- `declared_width.rs` (~8280 tok, huge) — Copyright 2025 STARGA Inc.
- `enum_cmp.rs` (~1907 tok, huge) — Copyright 2026 STARGA Inc.
- `enum_payload.rs` (~815 tok, large) — Copyright 2026 STARGA Inc.
- `extern_c_intrinsics.rs` (~6736 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `field_access.rs` (~6663 tok, huge) — Copyright 2026 STARGA Inc.
- `field_assign_refusal_tests.rs` (~695 tok, large) — The store must land in the record: a later read sees it. (The evaluator
- `fixed_array.rs` (~2039 tok, huge) — Copyright 2026 STARGA Inc.
- `fixed_array_struct.rs` (~5770 tok, huge) — Copyright 2026 STARGA Inc.
- `float_ops.rs` (~1055 tok, large) — Copyright 2025-2026 STARGA Inc.
- `for_hygiene.rs` (~1119 tok, large) — Copyright 2026 STARGA Inc.
- `import_collision_gate.rs` (~939 tok, large) — Copyright 2026 STARGA Inc.
- `imported_extern_decls.rs` (~1178 tok, large) — Copyright 2025-2026 STARGA Inc.
- `interp_mem.rs` (~5911 tok, huge) — Copyright 2025 STARGA Inc.
- `ir_interp.rs` (~4212 tok, huge) — Copyright 2025 STARGA Inc.
- `loops.rs` (~2354 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `lower_entry.rs` (~2832 tok, huge) — Copyright 2026 STARGA Inc.
- `lower_import_consts.rs` (~497 tok, medium) — Copyright 2026 STARGA Inc.
- `match_stmt.rs` (~970 tok, large) — Copyright 2026 STARGA Inc.
- `materialization.rs` (~5848 tok, huge) — Copyright 2026 STARGA Inc.
- `mlir_build_overlap.rs` (~1173 tok, large) — Copyright 2026 STARGA Inc.
- `mlir_build_overlap_tests.rs` (~1411 tok, large) — Copyright 2026 STARGA Inc.
- `mlir_build_pipe.rs` (~2922 tok, huge) — Copyright 2025 STARGA Inc.
- `mlir_build_proc.rs` (~608 tok, large) — Copyright 2026 STARGA Inc.
- `mlir_build.rs` (~10088 tok, huge) — Copyright 2025 STARGA Inc.
- `mlir_build_runtime_src.rs` (~1499 tok, large) — Copyright 2025-2026 STARGA Inc.
- `mlir_build_tests.rs` (~4266 tok, huge) — Copyright 2025 STARGA Inc.
- `mlir_export.rs` (~12272 tok, huge) — Copyright 2025 STARGA Inc.
- `mlir_gpu.rs` (~301 tok, medium) — Copyright 2025 STARGA Inc.
- `mlir_jit.rs` (~501 tok, large) — Copyright 2025 STARGA Inc.
- `mlir_opt.rs` (~995 tok, large) — Copyright 2025 STARGA Inc.
- `mlir_run.rs` (~2243 tok, huge) — Copyright 2025 STARGA Inc.
- `module_bindings.rs` (~4659 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `module_globals.rs` (~1160 tok, large) — Copyright 2026 STARGA Inc.
- `narrow_arith.rs` (~3977 tok, huge) — Copyright 2025 STARGA Inc.
- `narrow_scan.rs` (~3930 tok, huge) — Module-wide narrow-int SURFACE prescan (compile-speed early-skip).
- `node_children.rs` (~3450 tok, huge) — Copyright 2025 STARGA Inc.
- `record_store.rs` (~2457 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `result_variants.rs` (~602 tok, large) — Copyright 2026 STARGA Inc.
- `slice_abi.rs` (~325 tok, medium) — Copyright 2025 STARGA Inc.
### `src/eval/stdlib/`

- `mod.rs` (~169 tok, small) — Copyright 2025 STARGA Inc.
- `tensor.rs` (~8408 tok, huge) — Copyright 2025 STARGA Inc.
### `src/eval/string_equality/`

- `classify.rs` (~6020 tok, huge) — Copyright 2026 STARGA Inc.
### `src/eval/string_equality/facts/`

- `payloads.rs` (~1401 tok, large) — Copyright 2026 STARGA Inc.
### `src/eval/string_equality/`

- `facts.rs` (~7897 tok, huge) — Copyright 2026 STARGA Inc.
- `flows.rs` (~4174 tok, huge) — Copyright 2026 STARGA Inc.
- `flow_tests.rs` (~3654 tok, huge) — Copyright 2026 STARGA Inc.
- `project.rs` (~5274 tok, huge) — Copyright 2026 STARGA Inc.
### `src/eval/`

- `string_equality.rs` (~6564 tok, huge) — Copyright 2025 STARGA Inc.
### `src/eval/string_equality/`

- `segments.rs` (~3341 tok, huge) — Copyright 2026 STARGA Inc.
- `segment_tests.rs` (~3515 tok, huge) — Copyright 2026 STARGA Inc.
- `slots.rs` (~2198 tok, huge) — Copyright 2026 STARGA Inc.
- `slot_tests.rs` (~2705 tok, huge) — Copyright 2026 STARGA Inc.
- `sums.rs` (~772 tok, large) — Copyright 2026 STARGA Inc.
- `tests.rs` (~4214 tok, huge) — Copyright 2025 STARGA Inc.
- `walk.rs` (~4405 tok, huge) — Copyright 2026 STARGA Inc.
### `src/eval/`

- `string_record.rs` (~3664 tok, huge) — Copyright 2026 STARGA Inc.
### `src/eval/struct_resolver/`

- `inference.rs` (~2555 tok, huge) — Copyright 2025 STARGA Inc.
### `src/eval/`

- `struct_resolver.rs` (~7173 tok, huge) — Copyright 2025 STARGA Inc.
- `traits.rs` (~4150 tok, huge) — Copyright 2025 STARGA Inc.
- `type_aliases.rs` (~2302 tok, huge) — Copyright 2025 STARGA Inc.
- `value.rs` (~4142 tok, huge) — Copyright 2025 STARGA Inc.
- `zeroed_array.rs` (~2912 tok, huge) — Copyright 2026 STARGA Inc.
### `src/exec/`

- `conv.rs` (~435 tok, medium) — Copyright 2025 STARGA Inc.
- `cpu.rs` (~4500 tok, huge) — Copyright 2025 STARGA Inc.
- `mod.rs` (~587 tok, large) — Copyright 2025 STARGA Inc.
### `src/ffi/`

- `header.rs` (~413 tok, medium) — Copyright 2025 STARGA Inc.
- `mod.rs` (~1780 tok, huge) — Copyright 2025 STARGA Inc.
- `sys.rs` (~3348 tok, huge) — Copyright 2025-2026 STARGA Inc.
### `src/fmt/`

- `cli.rs` (~5733 tok, huge) — Copyright 2025 STARGA Inc.
- `mod.rs` (~4605 tok, huge) — Copyright 2025 STARGA Inc.
- `printer.rs` (~17214 tok, huge) — Copyright 2025 STARGA Inc.
### `src/fmt/printer/`

- `spelling.rs` (~1912 tok, huge) — Copyright 2025 STARGA Inc.
### `src/`

- `intrinsic_contract.rs` (~4987 tok, huge) — Copyright 2025 STARGA Inc.
- `intrinsics.rs` (~11096 tok, huge) — The intrinsic registry: the ONE table naming every `__mind_*` primitive the
### `src/intrinsics/`

- `tests.rs` (~3924 tok, huge) — Copyright 2026 STARGA Inc.
### `src/ir/`

- `canonical_verify.rs` (~6429 tok, huge) — Copyright 2025 STARGA Inc.
- `canonical_verify_tests.rs` (~6827 tok, huge) — Copyright 2025 STARGA Inc.
### `src/ir/compact/`

- `emit.rs` (~4645 tok, huge) — Copyright 2025 STARGA Inc.
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
- `emit.rs` (~12325 tok, huge) — Copyright 2025 STARGA Inc.
- `error.rs` (~989 tok, large) — Copyright 2025 STARGA Inc.
- `evidence_size.rs` (~443 tok, medium) — Copyright 2025-2026 STARGA Inc.
- `format.rs` (~2787 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `mldsa.rs` (~2688 tok, huge) — Copyright 2025 STARGA Inc.
- `mod.rs` (~20174 tok, huge) — Copyright 2025 STARGA Inc.
- `parse.rs` (~12962 tok, huge) — Copyright 2025 STARGA Inc.
- `slhdsa.rs` (~3201 tok, huge) — Copyright 2025 STARGA Inc.
### `src/ir/compact/v3/v04/`

- `decode.rs` (~6283 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `encode.rs` (~6244 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `mod.rs` (~1266 tok, large) — Copyright 2025-2026 STARGA Inc.
### `src/ir/compact/v3/`

- `v04_tests.rs` (~5843 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `v04_test_support.rs` (~1571 tok, huge) — Copyright 2026 STARGA Inc.
### `src/ir/evidence/`

- `reach.rs` (~3500 tok, huge) — Copyright 2026 STARGA Inc.
### `src/ir/`

- `evidence.rs` (~12583 tok, huge) — Copyright 2025 STARGA Inc.
- `fp_mode.rs` (~13714 tok, huge) — FP-contract mode — the strict-vs-relaxed floating-point determinism state of
- `frozen_profile.rs` (~7312 tok, huge) — Copyright 2025 STARGA Inc.
- `frozen_profile_tests.rs` (~5205 tok, huge) — Copyright 2025 STARGA Inc.
- `global_enums.rs` (~789 tok, large) — Copyright 2025 STARGA Inc.
- `mic_load.rs` (~996 tok, large) — Copyright 2025 STARGA Inc.
- `mod.rs` (~16264 tok, huge) — Copyright 2025 STARGA Inc.
- `native_c_abi.rs` (~3388 tok, huge) — Copyright 2026 STARGA Inc.
### `src/ir/native_c_abi/`

- `tests.rs` (~3811 tok, huge) — Copyright 2026 STARGA Inc.
### `src/ir/`

- `native_closure.rs` (~5242 tok, huge) — Copyright 2025 STARGA Inc.
### `src/ir/native_lane_profile/`

- `admit_extern.rs` (~3137 tok, huge) — Copyright 2026 STARGA Inc.
- `admit_ir.rs` (~3464 tok, huge) — Copyright 2026 STARGA Inc.
- `admit.rs` (~5590 tok, huge) — Copyright 2026 STARGA Inc.
- `alloc_size.rs` (~1558 tok, huge) — Copyright 2026 STARGA Inc.
- `alloc_size_tests.rs` (~1105 tok, large) — Copyright 2026 STARGA Inc.
- `arrays.rs` (~1016 tok, large) — Copyright 2026 STARGA Inc.
- `arrays_tests.rs` (~1988 tok, huge) — Copyright 2026 STARGA Inc.
- `mod.rs` (~425 tok, medium) — Copyright 2026 STARGA Inc.
- `names.rs` (~1322 tok, large) — Copyright 2026 STARGA Inc.
- `seed.rs` (~4477 tok, huge) — Copyright 2026 STARGA Inc.
- `seed_tests.rs` (~3838 tok, huge) — Copyright 2026 STARGA Inc.
- `tests_more.rs` (~5805 tok, huge) — Copyright 2026 STARGA Inc.
- `tests.rs` (~6433 tok, huge) — Copyright 2026 STARGA Inc.
- `types.rs` (~6065 tok, huge) — Copyright 2026 STARGA Inc.
### `src/ir/`

- `print.rs` (~5543 tok, huge) — Copyright 2025 STARGA Inc.
- `project_consts.rs` (~127 tok, small) — Copyright 2026 STARGA Inc.
- `verify_calls.rs` (~1946 tok, huge) — Copyright 2026 STARGA Inc.
- `verify_ids.rs` (~1720 tok, huge) — Copyright 2025 STARGA Inc.
- `verify.rs` (~21228 tok, huge) — Copyright 2025 STARGA Inc.
### `src/`

- `lib.rs` (~1133 tok, large) — Copyright 2025 STARGA Inc.
- `linalg.rs` (~2025 tok, huge) — Copyright 2025 STARGA Inc.
### `src/lint/`

- `mod.rs` (~1311 tok, large) — Copyright 2025 STARGA Inc.
- `rule.rs` (~2690 tok, huge) — Copyright 2025 STARGA Inc.
### `src/lint/rules/`

- `mod.rs` (~454 tok, medium) — Copyright 2025 STARGA Inc.
- `naming_convention.rs` (~2031 tok, huge) — Copyright 2025 STARGA Inc.
- `q16_overflow.rs` (~2950 tok, huge) — Copyright 2025 STARGA Inc.
- `shadowing.rs` (~2025 tok, huge) — Copyright 2025 STARGA Inc.
- `trailing_whitespace.rs` (~869 tok, large) — Copyright 2025 STARGA Inc.
- `unused_import.rs` (~2004 tok, huge) — Copyright 2025 STARGA Inc.
### `src/`

- `main.rs` (~6847 tok, huge) — Copyright 2025 STARGA Inc.
### `src/mlir/`

- `c_export.rs` (~3770 tok, huge) — Copyright 2025 STARGA Inc.
- `const_globals.rs` (~4744 tok, huge) — Copyright 2025 STARGA Inc.
- `div_trap.rs` (~1875 tok, huge) — Copyright 2025 STARGA Inc.
- `extern_narrow_args.rs` (~1841 tok, huge) — Copyright 2026 STARGA Inc.
- `gemm_tuning.rs` (~6147 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `id_subst.rs` (~747 tok, large) — Copyright 2025 STARGA Inc.
- `memory_address.rs` (~1838 tok, huge) — Copyright 2026 STARGA Inc.
- `mod.rs` (~619 tok, large) — Copyright 2025 STARGA Inc.
- `nobuiltin.rs` (~4674 tok, huge) — Copyright 2025 STARGA Inc.
- `staged_opt.rs` (~2450 tok, huge) — Copyright 2025 STARGA Inc.
### `src/ops/`

- `cerebras.rs` (~2713 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `core_v1.rs` (~1823 tok, huge) — Copyright 2025 STARGA Inc.
- `mod.rs` (~228 tok, medium) — Copyright 2025 STARGA Inc.
### `src/opt/`

- `collapse.rs` (~11454 tok, huge) — Copyright 2025 STARGA Inc.
- `comptime.rs` (~5806 tok, huge) — Copyright 2025 STARGA Inc.
- `fold.rs` (~3848 tok, huge) — Copyright 2025 STARGA Inc.
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

- `assert.rs` (~844 tok, large) — Copyright 2026 STARGA Inc.
- `dotted_field.rs` (~480 tok, medium) — Copyright 2026 STARGA Inc.
- `eval_imports.rs` (~2137 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `expand_bimap.rs` (~16544 tok, huge) — An `#[bimap]` attribute on an `enum` declares a one-to-one correspondence
- `import_list_tests.rs` (~2764 tok, huge) — Completeness of the parser-owned import list.
- `lexeme.rs` (~1679 tok, huge) — Copyright 2025 STARGA Inc.
- `paths.rs` (~695 tok, large) — Copyright 2026 STARGA Inc.
- `spelling.rs` (~1973 tok, huge) — Copyright 2025 STARGA Inc.
- `trivia.rs` (~4061 tok, huge) — Copyright 2025 STARGA Inc.
### `src/phf/`

- `mod.rs` (~4955 tok, huge) — Copyright 2025 STARGA Inc.
### `src/`

- `pipeline.rs` (~7559 tok, huge) — Copyright 2025 STARGA Inc.
### `src/project/`

- `active_module_table_extern_abi_tests.rs` (~1945 tok, huge) — Copyright 2025 STARGA Inc.
- `active_module_table.rs` (~7515 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `artifact.rs` (~1639 tok, huge) — Copyright 2025 STARGA Inc.
- `build_input_snapshot.rs` (~4689 tok, huge) — Copyright 2025 STARGA Inc.
- `build_lock.rs` (~4456 tok, huge) — Copyright 2026 STARGA Inc.
- `call_bindings.rs` (~5326 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `canonical_bridge.rs` (~5927 tok, huge) — Copyright 2026 STARGA Inc.
- `compiled_sources.rs` (~530 tok, large) — Copyright 2025 STARGA Inc.
- `embedded_entry.rs` (~670 tok, large) — Copyright 2025-2026 STARGA Inc.
- `exported_extern.rs` (~1250 tok, large) — Copyright 2025-2026 STARGA Inc.
- `host_triple.rs` (~992 tok, large) — Copyright 2025-2026 STARGA Inc.
- `identity.rs` (~1902 tok, huge) — Copyright 2026 STARGA Inc.
- `link.rs` (~4438 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `module_table.rs` (~8278 tok, huge) — Copyright 2025 STARGA Inc.
- `native_sources.rs` (~718 tok, large) — Copyright 2026 STARGA Inc.
- `private_linkage.rs` (~7899 tok, huge) — Copyright 2026 STARGA Inc.
- `runtime_link.rs` (~2184 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `single_file_scope.rs` (~7131 tok, huge) — Copyright 2025-2026 STARGA Inc.
### `src/project/single_file_scope/`

- `tests.rs` (~2078 tok, huge) — Copyright 2026 STARGA Inc.
- `world_reach.rs` (~2026 tok, huge) — Copyright 2026 STARGA Inc.
### `src/project/`

- `source_snapshot.rs` (~608 tok, large) — Copyright 2025-2026 STARGA Inc.
- `sources.rs` (~8417 tok, huge) — Copyright 2025 STARGA Inc.
### `src/project/stdlib/`

- `load.rs` (~4725 tok, huge) — Copyright 2026 STARGA Inc.
### `src/project/`

- `stdlib.rs` (~5368 tok, huge) — Copyright 2025 STARGA Inc.
- `std_link_set.rs` (~2506 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `std_struct_schema.rs` (~840 tok, large) — Copyright 2026 STARGA Inc.
- `substrate_link.rs` (~5202 tok, huge) — Copyright 2025 STARGA Inc.
- `toolchain_pin.rs` (~7543 tok, huge) — route rule (see [`Backend`]).
### `src/`

- `python.rs` (~1082 tok, large) — Copyright 2025 STARGA Inc.
### `src/qualified_enums/`

- `alias_closure.rs` (~1462 tok, large) — Copyright 2026 STARGA Inc.
- `canonical_types.rs` (~1929 tok, huge) — Copyright 2026 STARGA Inc.
### `src/`

- `qualified_enums.rs` (~7098 tok, huge) — Copyright 2025-2026 STARGA Inc.
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

- `std_qualified.rs` (~7258 tok, huge) — Copyright 2026 STARGA Inc.
- `target.rs` (~6043 tok, huge) — Copyright 2025-2026 STARGA Inc.
### `src/test/`

- `imports.rs` (~5979 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `mod.rs` (~6247 tok, huge) — Copyright 2025 STARGA Inc.
- `runner.rs` (~3348 tok, huge) — Copyright 2025 STARGA Inc.
- `std_import_fns.rs` (~4990 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `top_level.rs` (~339 tok, medium) — Copyright 2025-2026 STARGA Inc.
- `worker.rs` (~5577 tok, huge) — Copyright 2026 STARGA Inc.
### `src/type_checker/`

- `array_control_flow.rs` (~1096 tok, large) — Copyright 2026 STARGA Inc.
- `array_lengths.rs` (~5788 tok, huge) — Copyright 2026 STARGA Inc.
- `canonical_facts.rs` (~2797 tok, huge) — Copyright 2026 STARGA Inc.
- `canonical_macro.rs` (~58 tok, small) — Copyright 2026 STARGA Inc.
- `cross_module_types.rs` (~453 tok, medium) — Copyright 2026 STARGA Inc.
- `duplicate_structs.rs` (~424 tok, medium) — Copyright 2026 STARGA Inc.
- `extern_declarations.rs` (~5745 tok, huge) — Copyright 2026 STARGA Inc.
- `intrinsic_call_arity.rs` (~2798 tok, huge) — Copyright 2026 STARGA Inc.
- `lowering_refusals.rs` (~7104 tok, huge) — Copyright 2026 STARGA Inc.
- `lowering_refusals_tests.rs` (~4372 tok, huge) — Copyright 2026 STARGA Inc.
- `nerve_lint.rs` (~7669 tok, huge) — Copyright 2025 STARGA Inc.
- `nerve_walk.rs` (~2232 tok, huge) — Copyright 2025 STARGA Inc.
- `qualified_enums.rs` (~4405 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `qualified_imports.rs` (~268 tok, medium) — Feature-neutral span-scoped qualified import predicates for name resolution.
### `src/type_checker/resolve/`

- `fn_values.rs` (~1376 tok, large) — Copyright 2026 STARGA Inc.
### `src/type_checker/`

- `resolve.rs` (~14678 tok, huge) — Copyright 2025 STARGA Inc.
### `src/type_checker/resolve/`

- `std_names.rs` (~2150 tok, huge) — Copyright 2026 STARGA Inc.
### `src/type_checker/`

- `return_checks.rs` (~1411 tok, large) — Copyright 2025 STARGA Inc.
### `src/type_checker/slice_abi/`

- `aliases.rs` (~3970 tok, huge) — Copyright 2026 STARGA Inc.
- `borrow_flow.rs` (~6717 tok, huge) — Copyright 2025 STARGA Inc.
- `owner_assignment.rs` (~1145 tok, large) — Copyright 2026 STARGA Inc.
- `provenance.rs` (~5370 tok, huge) — Copyright 2026 STARGA Inc.
### `src/type_checker/`

- `slice_abi.rs` (~7581 tok, huge) — Copyright 2025 STARGA Inc.
- `std_imports.rs` (~632 tok, large) — Copyright 2026 STARGA Inc.
- `stdlib_signatures.rs` (~782 tok, large) — Copyright 2025-2026 STARGA Inc.
- `struct_bindings.rs` (~1070 tok, large) — Copyright 2026 STARGA Inc.
- `type_display.rs` (~628 tok, large) — Copyright 2026 STARGA Inc.
### `src/types/`

- `canonical_intrinsic_tests.rs` (~1565 tok, huge) — Copyright 2025 STARGA Inc.
- `canonical_intrinsic_validation.rs` (~446 tok, medium) — Copyright 2025 STARGA Inc.
- `canonical_registry.rs` (~5107 tok, huge) — Copyright 2025 STARGA Inc.
- `canonical.rs` (~256 tok, medium) — Copyright 2025 STARGA Inc.
- `canonical_tests.rs` (~3544 tok, huge)
- `canonical_types.rs` (~6538 tok, huge) — Copyright 2025 STARGA Inc.
- `infer.rs` (~448 tok, medium) — Copyright 2025 STARGA Inc.
- `intern.rs` (~1554 tok, huge) — Copyright 2025 STARGA Inc.
- `mod.rs` (~1151 tok, large) — Copyright 2025 STARGA Inc.
- `value.rs` (~2993 tok, huge) — Copyright 2025 STARGA Inc.
### `src/workspace/`

- `mod.rs` (~4906 tok, huge) — Copyright 2025 STARGA Inc.
### `std/`

- `aes_gcm.mind` (~5400 tok, huge) — std/aes_gcm.mind — AES-128 (FIPS 197) + AES-128-GCM (NIST SP 800-38D) in
- `arena.mind` (~1323 tok, large) — std.arena — bump-pointer region allocator.
- `async.mind` (~2471 tok, huge) — std/async.mind -- RFC 0011 Phase A: Scheduler injection + Sender/Receiver
- `blas.mind` (~2522 tok, huge) — std/blas.mind — RFC 0006 Track A: pure-MIND surface over the six
- `chacha20_poly1305.mind` (~3973 tok, huge) — std/chacha20_poly1305.mind — ChaCha20-Poly1305 AEAD (RFC 8439) in pure MIND.
- `cli.mind` (~3223 tok, huge) — std/cli.mind — RFC 0013 Tier 1 Phase 1: argv-parsing surface.
- `cpu_affinity.mind` (~1234 tok, large) — std/cpu_affinity.mind — CPU affinity mask of CPU_SETSIZE 1024 bits.
- `cpu_affinity_sys.mind` (~570 tok, large) — std.cpu_affinity_sys — the OS calls of std.cpu_affinity, one straight-line wrapper each.
- `detmath.mind` (~4180 tok, huge) — std.detmath — deterministic transcendental tier (roadmap Phase 17.6).
- `dns.mind` (~6150 tok, huge) — std/dns.mind — host name resolution, connect-by-name, and the stream options
- `ecdsa_p256.mind` (~7346 tok, huge) — std/ecdsa_p256.mind — ECDSA signature VERIFICATION on NIST P-256
- `entropy.mind` (~860 tok, large) — std/entropy.mind — operating-system randomness (keys, nonces, TLS randoms).
- `evidence_mic.mind` (~3080 tok, huge) — std/evidence_mic.mind — write a std.evidence sink as a mic@3 MAP record and read it back.
- `evidence.mind` (~2507 tok, huge) — std/evidence.mind — record governance verdicts of std calls in a caller-owned sink.
- `fs_checked.mind` (~2600 tok, huge) — Checked Linux x86-64 directory enumeration. Author: STARGA Inc.
- `fs.mind` (~5376 tok, huge) — std/fs.mind — Task #268: POSIX filesystem surface in pure MIND.
- `hkdf.mind` (~1547 tok, huge) — std/hkdf.mind — HMAC-SHA256 (RFC 2104) + HKDF (RFC 5869) in pure MIND.
- `hpack.mind` (~9694 tok, huge) — std/hpack.mind — HPACK header-compression DECODING (RFC 7541) in pure MIND.
- `http2_frame.mind` (~4191 tok, huge) — std/http2_frame.mind — HTTP/2 framing layer (RFC 9113 §3.4, §4.1, §6) in
- `http.mind` (~6682 tok, huge) — std/http.mind — HTTP/1.1 client over std.net (task #XXX).
- `https_http1.mind` (~6266 tok, huge) — std/https_http1.mind — the HTTP/1.1 response side of std.https: the
- `https_io.mind` (~1340 tok, large) — std/https_io.mind — what the std.https driver reads from the world around one
- `https.mind` (~7777 tok, huge) — std/https.mind — authenticated HTTPS GET and POST over the bounded TLS 1.3
- `https_probe.mind` (~1693 tok, huge) — std/https_probe.mind — the std.https network probe: resolve the host of an
- `https_record.mind` (~1979 tok, huge) — std/https_record.mind — the TLS 1.3 client's record layer: validate one inbound
- `https_url.mind` (~5267 tok, huge) — std/https_url.mind — the std.https error codes and bounds, URL parsing and
- `io_canon.mind` (~2624 tok, huge) — std.io_canon — canonical completion ordering for deterministic I/O.
- `io.mind` (~1719 tok, huge) — std/io.mind — RFC 0005 Phase 2: pure-MIND I/O surface.
- `iouring.mind` (~23114 tok, huge) — std.iouring — minimal io_uring binding (Linux). The physical-I/O reap source
- `json_canon.mind` (~4335 tok, huge) — std/json_canon.mind -- CRX-04 canonical JSON dump (WP-MC-47b).
- `json.mind` (~15391 tok, huge) — std/json.mind -- RFC 8259 / ECMA-404 subset parser (task #269, cargo-retirement track).
- `keccak.mind` (~3926 tok, huge) — std/keccak.mind — Keccak / SHA-3 + SHAKE (FIPS 202) in pure MIND.
- `llvm.mind` (~11108 tok, huge) — std/llvm.mind — RFC 0010 Phase F: hand-written MIND extern "C" bindings
- `map.mind` (~1538 tok, huge) — std/map.mind — RFC 0005 Phase 2: pure-MIND insertion-ordered map.
- `mic_record.mind` (~3332 tok, huge) — std/mic_record.mind — write and read program results as mic@3 MAP records.
- `mic_record_table.mind` (~2681 tok, huge) — std/mic_record_table.mind — the key rules and the sorted entry table of
- `mic_record_wire.mind` (~2145 tok, huge) — std/mic_record_wire.mind — the limits, value tags, named errors and byte
- `mic_results.mind` (~2954 tok, huge) — std/mic_results.mind — a program's results as a std.mic_record record.
- `mlir.mind` (~11056 tok, huge) — std/mlir.mind — RFC 0010 Phase E: hand-written MIND extern "C" bindings
- `mlkem768.mind` (~6450 tok, huge) — std/mlkem768.mind — ML-KEM-768 (FIPS 203, "Kyber") in pure MIND.
- `net.mind` (~2412 tok, huge) — std/net.mind — Task #268: POSIX socket surface in pure MIND.
- `os_evidence.mind` (~1426 tok, large) — std/os_evidence.mind — std.evidence records of std.fs_checked, std.procgroup and
- `posix.mind` (~1553 tok, huge) — std.posix — Linux x86-64 filesystem surface, with no libc in native builds.
- `posix_sys.mind` (~1098 tok, large) — std.posix_sys — the raw Linux x86-64 system calls of std.posix and
- `process.mind` (~6149 tok, huge) — std/process.mind — Task #268: subprocess + process environment in pure MIND.
- `procgroup.mind` (~5876 tok, huge) — std.procgroup — run many child processes at once and wait for all of them.
- `procgroup_sys.mind` (~1458 tok, large) — std.procgroup_sys — the OS calls of std.procgroup, one straight-line wrapper each.
- `procgroup_text.mind` (~810 tok, large) — std.procgroup_text — the C-string and byte helpers of std.procgroup.
- `reactor.mind` (~1420 tok, large) — std.reactor — deterministic per-connection request-id allocation.
- `regex.mind` (~9534 tok, huge) — std/regex.mind -- POSIX ERE subset NFA engine (task #269, cargo-retirement track).
- `ring.mind` (~1407 tok, large) — std.ring — fixed-capacity byte ring buffer (FIFO).
- `rsa_pss.mind` (~3150 tok, huge) — std/rsa_pss.mind — RSASSA-PSS signature VERIFICATION (RFC 8017 §8.1.2) with
- `sha256.mind` (~4448 tok, huge) — std/sha256.mind — FIPS 180-4 SHA-256 in MIND.
- `sha512.mind` (~5100 tok, huge) — std/sha512.mind — FIPS 180-4 SHA-512 and SHA-384 in pure MIND.
- `string.mind` (~2948 tok, huge) — std/string.mind — RFC 0005 Phase 2: pure-MIND String.
- `time.mind` (~257 tok, medium) — std.time — wall-clock access for evidence / audit timestamps.
- `tls13_client_api.mind` (~5414 tok, huge) — std/tls13_client_api.mind
- `tls13_client_core.mind` (~4575 tok, huge) — std/tls13_client_core.mind
- `tls13_client_hello.mind` (~5621 tok, huge) — std/tls13_client_hello.mind
- `tls13_finished.mind` (~2712 tok, huge) — std/tls13_finished.mind — TLS 1.3 Finished-message MAC + transcript hash
- `tls13_handshake.mind` (~2986 tok, huge) — std/tls13_handshake.mind — TLS 1.3 handshake CRYPTO ORCHESTRATION in pure
- `tls13_keyschedule.mind` (~3220 tok, huge) — std/tls13_keyschedule.mind — TLS 1.3 key schedule (RFC 8446 §7.1) in pure MIND.
- `tls13_record.mind` (~2244 tok, huge) — std/tls13_record.mind — TLS 1.3 record-layer protection (RFC 8446 §5.1-5.3)
- `toml.mind` (~10604 tok, huge) — std/toml.mind -- TOML 1.0 subset parser (task #258, cargo-retirement track).
- `tui.mind` (~4869 tok, huge) — std/tui.mind — RFC 0013 Tier 1 (c): minimal pure-MIND TUI surface.
- `vec.mind` (~1100 tok, large) — std/vec.mind — RFC 0005 Phase 2: pure-MIND growable vector.
- `x25519.mind` (~4477 tok, huge) — std/x25519.mind — X25519 (RFC 7748 §5) Curve25519 Montgomery-ladder ECDH in
- `x25519mlkem768.mind` (~1554 tok, huge) — std/x25519mlkem768.mind — X25519MLKEM768 post-quantum hybrid key exchange in
- `x509_bignum.mind` (~1704 tok, huge) — std/x509_bignum.mind — fixed-width big-integer arithmetic (radix 2^16 limbs,
- `x509_bundle.mind` (~3780 tok, huge) — std/x509_bundle.mind — in-memory DER and PEM trust-anchor bundle loader.
- `x509_der.mind` (~4658 tok, huge) — std/x509_der.mind — DER TLV decoding, OID recognition, byte utilities, and
- `x509_evidence.mind` (~659 tok, large) — std/x509_evidence.mind — std.evidence records of the std.x509_path chain verdicts.
- `x509_ext.mind` (~2786 tok, huge) — std/x509_ext.mind — X.509v3 validity date parsing and extension processing
- `x509.mind` (~1080 tok, large) — std/x509.mind — X.509v3 DER parsing, RSA PKCS#1 v1.5 (SHA-256) signature
- `x509_name.mind` (~4230 tok, huge) — std/x509_name.mind — RFC 6125 hostname matching, IPv4/IPv6 literal parsing,
- `x509_p384.mind` (~4342 tok, huge) — std/x509_p384.mind — ECDSA signature verification on NIST P-384
- `x509_path.mind` (~6041 tok, huge) — std/x509_path.mind — X.509 certificate path building, parent policy validation,
- `x509_sig.mind` (~4070 tok, huge) — std/x509_sig.mind — RSA PKCS#1 v1.5, RSASSA-PSS and ECDSA signature
### `tests/`

- `aggregate_const_run.rs` (~1765 tok, huge) — Copyright 2026 STARGA Inc.
- `aggshape_reject.rs` (~1842 tok, huge) — Copyright 2025 STARGA Inc.
- `alias_miscompile_run.rs` (~1287 tok, large) — Copyright 2025 STARGA Inc.
- `array_ctor_push_get_run.rs` (~861 tok, large) — Copyright 2025 STARGA Inc.
- `array_load_bounds_and_dtype.rs` (~2116 tok, huge) — Copyright 2025 STARGA Inc.
- `array_oob_trap_run.rs` (~4980 tok, huge) — Copyright 2025 STARGA Inc.
- `array_store_run.rs` (~2207 tok, huge) — Copyright 2026 STARGA Inc.
- `array_surface_run.rs` (~829 tok, large) — Copyright 2025 STARGA Inc.
- `array_u64_element_shift_run.rs` (~989 tok, large) — Copyright 2025 STARGA Inc.
- `assert_parenthesised_message.rs` (~1952 tok, huge) — Copyright 2026 STARGA Inc.
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
- `build_run_runnable_blocker_gate.rs` (~5850 tok, huge) — Copyright 2025 STARGA Inc. Licensed under the Apache License, Version 2.0.
- `bytes_buffer_run.rs` (~669 tok, large) — Copyright 2025 STARGA Inc.
- `bytes_fixed_into_vec_run.rs` (~1764 tok, huge) — Copyright 2025 STARGA Inc.
- `bytes_zero_run.rs` (~843 tok, large) — Copyright 2025 STARGA Inc.
- `call_arity_check_cli.rs` (~1642 tok, huge) — Copyright 2025 STARGA Inc.
- `canonical_source_bridge.rs` (~3130 tok, huge) — The ordinary evaluator retains its unit-placeholder return convention.
- `capability_refusal_cause_scan.rs` (~3400 tok, huge) — Copyright 2025 STARGA Inc.
- `cerebras_stencil_tile.rs` (~1929 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `chacha20_poly1305_smoke.rs` (~2259 tok, huge) — Copyright 2025 STARGA Inc.
- `char_literal_run.rs` (~944 tok, large) — Copyright 2025 STARGA Inc.
- `check_claims_gate_tests.py` (~9142 tok, huge) — Gate tests for scripts/check_claims.py — its two DERIVED numbers must bite.
- `check_fmt_drift_reported.rs` (~880 tok, large) — Copyright 2026 STARGA Inc.
- `check_reporter_early_failure.rs` (~1577 tok, huge) — Copyright 2025 STARGA Inc.
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
- `collection_owner_assignment_run.rs` (~3759 tok, huge) — Copyright 2025 STARGA Inc.
### `tests/common/`

- `gate.rs` (~4999 tok, huge) — Copyright 2025 STARGA Inc.
- `json_canon_kat.rs` (~1057 tok, large) — Copyright 2025 STARGA Inc.
- `mic_record.rs` (~2694 tok, huge) — Copyright 2026 STARGA Inc.
- `mod.rs` (~2711 tok, huge) — Copyright 2025 STARGA Inc.
- `x509_so.rs` (~1351 tok, large) — Copyright 2025-2026 STARGA Inc.
- `xsi_gate.rs` (~4684 tok, huge) — Copyright 2025 STARGA Inc.
- `xsi_workloads.rs` (~606 tok, large) — Copyright 2025 STARGA Inc.
### `tests/`

- `compound_assign.rs` (~984 tok, large) — Copyright 2025 STARGA Inc.
- `cond_truthiness.rs` (~840 tok, large) — Copyright 2025 STARGA Inc.
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

- `conformance.rs` (~1493 tok, large) — Used only by the fail-closed test below, which is itself gated on the GPU
- `CONFORMANCE_TESTS.md` (~1563 tok, huge) — MIND Conformance Test Corpus
- `const_array_run.rs` (~834 tok, large) — Copyright 2025 STARGA Inc.
- `const_f64_array_run.rs` (~1097 tok, large) — Copyright 2025 STARGA Inc.
- `const_folding.rs` (~246 tok, medium) — Copyright 2025 STARGA Inc.
- `const_table_build_determinism.rs` (~5377 tok, huge) — Copyright 2025 STARGA Inc.
- `continue_in_match_arm_run.rs` (~1176 tok, large) — Copyright 2025 STARGA Inc.
- `conv2d_exec.rs` (~583 tok, large) — Copyright 2025 STARGA Inc.
- `conv2d_grad.rs` (~3194 tok, huge) — Copyright 2025 STARGA Inc.
- `conv2d_types.rs` (~360 tok, medium) — Copyright 2025 STARGA Inc.
- `cross_module_cdylib_compose.rs` (~4098 tok, huge) — Copyright 2025 STARGA Inc.
- `cross_module_enum_run.rs` (~1095 tok, large) — Copyright 2025 STARGA Inc.
- `cross_module_enum_variant_lowering.rs` (~3202 tok, huge) — Copyright 2026 STARGA Inc.
- `cross_module_field_access_run.rs` (~4567 tok, huge) — Copyright 2025 STARGA Inc.
- `cross_module.rs` (~1332 tok, large) — Copyright 2025 STARGA Inc.
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
### `tests/cross_substrate_identity/json-canon-kat/`

- `manifest.toml` (~401 tok, medium) — version = "1"
- `reference_hashes.toml` (~209 tok, medium) — avx2 = "56757a5d72b626954435be6b7026bf11d26986c17413098d73502c3373a42114"
### `tests/cross_substrate_identity/lorenz-q16/`

- `manifest.toml` (~681 tok, large) — version = "1"
- `reference_hashes.toml` (~562 tok, large) — avx2 = "04da6abc69e63314331e88a7a9670ce5c9e90ddaa2bf5f5dc53526f56477de80"
### `tests/cross_substrate_identity/matmul-f32-v-64x64/`

- `manifest.toml` (~523 tok, large) — version = "1"
- `reference_hashes.toml` (~592 tok, large) — avx2 = "ec5adb991372fcfc16b964ba566f05fb44701fcf8bbde2a5453fed294e1d0175"
### `tests/cross_substrate_identity/q16-arith-chain/`

- `manifest.toml` (~432 tok, medium) — version = "1"
- `reference_hashes.toml` (~356 tok, medium) — avx2 = "ce93cdeb0e650c1c8e0cd05687ed986bbdbac691b6a8742e155b8ffd65997d78"
### `tests/cross_substrate_identity/quant-american-binomial/`

- `manifest.toml` (~783 tok, large) — version = "1"
- `reference_hashes.toml` (~174 tok, small) — avx2 = "88a57cd53930a73b38ad474101fceaa523e5ba8a9069bf4debdd18ef8b60bb25"
### `tests/cross_substrate_identity/quant-barrier-analytic/`

- `manifest.toml` (~691 tok, large) — version = "1"
- `reference_hashes.toml` (~174 tok, small) — avx2 = "8a6f29261a5869a71e3bd792e6091fe644388b62b2bf035f229127ba1f6f3dd3"
### `tests/cross_substrate_identity/quant-black-scholes/`

- `manifest.toml` (~770 tok, large) — version = "1"
- `reference_hashes.toml` (~174 tok, small) — avx2 = "e28d6f8d6caf0f5f5265f9ec6f300b4f57640338914e0d00fa5c7e6c2a2a11e2"
### `tests/cross_substrate_identity/quant-bond-curve/`

- `manifest.toml` (~607 tok, large) — version = "1"
- `reference_hashes.toml` (~174 tok, small) — avx2 = "b63f8f3f4491bb768b4a960536546a8e7ca17f9e5bbddf84a3a0789d85f77ef4"
### `tests/cross_substrate_identity/quant-crr-binomial/`

- `manifest.toml` (~589 tok, large) — version = "1"
- `reference_hashes.toml` (~174 tok, small) — avx2 = "ccad4237c9a725bc41967840d6b54d939e4b134bbfcb0a99b2a7358008d68b96"
### `tests/cross_substrate_identity/quant-greeks-higher-order/`

- `manifest.toml` (~475 tok, medium) — version = "1"
- `reference_hashes.toml` (~174 tok, small) — avx2 = "0e239922bb97639c207c0c06ed7e812bcad3e42dc0c38edd757b422c2437a18f"
### `tests/cross_substrate_identity/quant-implied-vol/`

- `manifest.toml` (~885 tok, large) — version = "1"
- `reference_hashes.toml` (~174 tok, small) — avx2 = "4340ca307df3f929c13ff1d244714eada0c9683f68b9d1273f03c6b064cc6152"
### `tests/cross_substrate_identity/quant-implied-vol-surface/`

- `manifest.toml` (~469 tok, medium) — version = "1"
- `reference_hashes.toml` (~174 tok, small) — avx2 = "a5e46ad874cb759980cd139a821bd4ebc5855176b37f25b33e58296e4e71754e"
### `tests/cross_substrate_identity/quant-monte-carlo-gbm/`

- `manifest.toml` (~909 tok, large) — version = "1"
- `reference_hashes.toml` (~174 tok, small) — avx2 = "f2a16351198d49224e7b388576cda7780acefabe019cc2fefdcadbddd9981109"
### `tests/cross_substrate_identity/quant-portfolio-risk/`

- `manifest.toml` (~426 tok, medium) — version = "1"
- `reference_hashes.toml` (~174 tok, small) — avx2 = "85c8e65fdfd6e73d318d28788516ef14ab1da3f5bf9a7ea687edd1df13564582"
### `tests/cross_substrate_identity/`

- `README.md` (~1873 tok, huge) — cross_substrate_identity — the internal mind-bench reproducibility gate
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

- `cross_substrate_receipts.rs` (~2546 tok, huge) — Copyright 2025 STARGA Inc.
- `cross_substrate_reference_consistency.rs` (~3141 tok, huge) — Copyright 2025 STARGA Inc.
- `crypto_vectors_driver.py` (~2653 tok, huge) — # Official-vector driver for std/aes_gcm.mind + std/hkdf.mind (pure-MIND crypto).
- `determinism_veto_affinity_sys.rs` (~3081 tok, huge) — Copyright 2026 STARGA Inc.
- `determinism_veto_bare_std.rs` (~4047 tok, huge) — Copyright 2026 STARGA Inc.
- `determinism_veto_control.rs` (~7838 tok, huge) — Copyright 2025 STARGA Inc.
- `determinism_veto_std_callers.rs` (~3263 tok, huge) — Copyright 2026 STARGA Inc.
- `diagnostic_code_contract.rs` (~558 tok, large) — Copyright 2026 STARGA Inc.
- `diagnostics_parse.rs` (~1112 tok, large) — Copyright 2025 STARGA Inc.
- `diagnostics.rs` (~688 tok, large) — Copyright 2025 STARGA Inc.
- `digit_separator_run.rs` (~606 tok, large) — Copyright 2025 STARGA Inc.
- `dot_enum_variant_run.rs` (~756 tok, large) — Copyright 2025 STARGA Inc.
- `dot_variants.rs` (~284 tok, medium) — Copyright 2025 STARGA Inc.
- `duplicate_struct_declarations.rs` (~576 tok, large) — Copyright 2026 STARGA Inc.
- `ecdsa_p256_driver.py` (~2777 tok, huge) — # Ground-truth driver for std/ecdsa_p256.mind (pure-MIND ECDSA P-256/SHA-256
- `emit_ir_for_loop.rs` (~374 tok, medium) — Regression test for #4: lowering a `for` loop to IR (the path `mindc --emit-ir`
- `emit_object_relocatable.rs` (~987 tok, large) — Copyright 2026 STARGA Inc.
- `enum_match_collision_run.rs` (~987 tok, large) — Copyright 2025 STARGA Inc.
- `enum_match_run.rs` (~2310 tok, huge) — Copyright 2025 STARGA Inc.
- `enum_soundness.rs` (~1411 tok, large) — Copyright 2025 STARGA Inc.
- `enum_struct_variant_run.rs` (~1280 tok, large) — Copyright 2025 STARGA Inc.
- `evaluator_declared_width.rs` (~5277 tok, huge) — Copyright 2025 STARGA Inc.
- `evidence_release_smoke.rs` (~5482 tok, huge) — Copyright 2026 STARGA Inc.
- `exec_basic.rs` (~785 tok, large) — Copyright 2025 STARGA Inc.
- `expr_parser.rs` (~916 tok, large) — Copyright 2025 STARGA Inc.
- `extern_c_phase_a.rs` (~2722 tok, huge) — Copyright 2025 STARGA Inc.
- `extern_c_phase_b.rs` (~6230 tok, huge) — Copyright 2025 STARGA Inc.
- `extern_c_phase_c.rs` (~6145 tok, huge) — Copyright 2025 STARGA Inc.
- `extern_c_safety_tag_informational.rs` (~1427 tok, large) — Copyright 2025 STARGA Inc.
- `extern_declaration_conflicts.rs` (~2114 tok, huge) — Copyright 2026 STARGA Inc.
- `extern_narrow_ret_run.rs` (~1528 tok, huge) — Copyright 2025 STARGA Inc.
- `f3_bare_enum_collision_run.rs` (~1347 tok, large) — Copyright 2025 STARGA Inc.
- `f64_abi_negative_control.rs` (~1819 tok, huge) — Copyright 2025 STARGA Inc.
- `f64_activation_lowering.rs` (~972 tok, large) — Copyright 2025 STARGA Inc.
- `f64_call_arg_run.rs` (~977 tok, large) — Copyright 2025 STARGA Inc.
- `f64_literal_envelope.rs` (~1158 tok, large) — Each source literal is compiled via `mindc --emit-shared`; the exported
- `f64_loop_run.rs` (~3720 tok, huge) — Copyright 2025 STARGA Inc.
- `fail_closed_capability_skip_env.rs` (~3997 tok, huge) — Copyright 2025 STARGA Inc.
- `fail_closed_capability_skip.rs` (~6329 tok, huge) — Copyright 2025 STARGA Inc.
- `fail_closed_capability_skip_stub_exec.rs` (~2022 tok, huge) — Copyright 2025 STARGA Inc.
- `fail_closed_cli_run.rs` (~1585 tok, huge) — Regression tests for the two fail-closed CLI guards that shipped WITHOUT one.
- `fail_open_skip_site_ratchet.rs` (~9057 tok, huge) — Copyright 2025 STARGA Inc.
- `ffi_header.rs` (~221 tok, medium) — Copyright 2025 STARGA Inc.
- `file_read_under_test.rs` (~3659 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `fixed_array_binding_lengths.rs` (~462 tok, medium) — Copyright 2026 STARGA Inc.
- `fixed_array_narrow_assignment_types.rs` (~376 tok, medium) — Copyright 2026 STARGA Inc.
- `fixed_array_return_lengths.rs` (~3080 tok, huge) — Copyright 2026 STARGA Inc.
- `fixed_array_struct_field_run.rs` (~6786 tok, huge) — Copyright 2026 STARGA Inc.
### `tests/fixtures/`

- `autodiff.mind` (~55 tok, small) — Minimal differentiable program for the --emit-grad-ir CLI test.
### `tests/fixtures/evidence/`

- `cpu_features_probe.mind` (~59 tok, small) — Reaches __mind_cpu_features_v1 (Det::World): attested nondeterministic.
- `ymm_vpaddd_i32x8.mind` (~122 tok, small) — A program that calls the native-only AVX2 row __mind_vpaddd_i32x8.
### `tests/fixtures/f64_abi_negative/`

- `README.md` (~146 tok, small) — f64 call-ABI negative control (paired with the #298 self-host f64 MLIR surface)
- `scale_neg.mind` (~23 tok, tiny)
- `scale_pos.mind` (~23 tok, tiny)
### `tests/fixtures/`

- `g2_expected_outcomes.tsv` (~5592 tok, huge) — # Per-fixture G2 capability ratchet; canonical source SHA-256 binds each case.
- `invalid_broadcast.mind` (~17 tok, tiny)
- `invalid.mind` (~6 tok, tiny)
### `tests/fixtures/native_lane_extern/`

- `l1_extern_close_flip.mind` (~203 tok, medium) — l1_extern_close_flip.mind — L1's extern-refusal fixture, flipped at L2a.
- `native_c_abi_fixed.mind` (~329 tok, medium) — native_c_abi_fixed.mind — the fixed-arity extern corpus of lane L2a.
### `tests/fixtures/native_lane_gpu/`

- `align_driver.c` (~559 tok, large)
- `align_probe_coop.mind` (~685 tok, large) — The 10-argument cuLaunchCooperativeKernel row next to the 11-argument
- `align_probe.mind` (~1269 tok, large) — Two cuLaunchKernel calls in one frame. The odd overflow count needs the
- `fake_driver.c` (~2183 tok, huge)
- `fake_rows2.mind` (~2278 tok, huge) — The MC-6 driver rows (the cuda.h 12.6 macro targets, legacy default stream)
- `fake_rows.mind` (~903 tok, large) — The driver-row program run against the test-only fake libcuda.so.1
- `gpu_only.mind` (~129 tok, small) — The libcuda-only import: both called rows are unversioned driver rows, so
- `mc6_cases.tsv` (~630 tok, large) — # MC-6 cases of tests/native_lane_gpu_loader.rs, tab-separated; a literal \n in 
- `trap_after_init.mind` (~111 tok, small) — A runtime bounds trap after the driver started a helper thread (the fake
- `trap_declared_driver.mind` (~123 tok, small) — A libc image that DECLARES a driver row it never calls. The exit decision
- `trap_libc_only.mind` (~92 tok, small) — The libc-only control of trap_after_init.mind: no driver row is declared,
- `vec_add.mind` (~2256 tok, huge) — A native-lane program that drives the real CUDA driver API through the
### `tests/fixtures/native_lane_refusals/`

- `lane_alloc_size_positive.mind` (~61 tok, small) — lane.alloc.size_positive: an image that already names __lane_realloc is not respelled, so this unproven size still refus
- `lane_binop_non_scalar.mind` (~32 tok, tiny) — lane.binop.non_scalar: a binary operator over two unit values.
- `lane_bool_width.mind` (~35 tok, tiny) — lane.bool.width: bool arithmetic has i1 width in MLIR but word width in the seed.
- `lane_call_ambiguous.mind` (~58 tok, small) — lane.call.ambiguous: the user image and an imported seed module both define the name.
- `lane_call_arity.mind` (~32 tok, tiny) — lane.call.arity: an intrinsic called with the wrong argument count.
- `lane_call_unknown.mind` (~33 tok, tiny) — lane.call.unknown: a callee that is neither a user function nor a lane intrinsic.
- `lane_entry_f64_result.mind` (~23 tok, tiny) — lane.entry.f64_result: the entry function returns f64.
- `lane_extern_arity.mind` (~36 tok, tiny) — lane.extern.arity: a table name declared with the wrong parameter count.
- `lane_extern_bind_clause.mind` (~35 tok, tiny) — lane.extern.bind_clause: an extern block that names a library to bind.
- `lane_extern_callconv.mind` (~36 tok, tiny) — lane.extern.callconv: a callconv other than C/SysV.
- `lane_extern_compiler_unmarked.mind` (~38 tok, tiny) — lane.extern.compiler_unmarked: an extern block under a compiler without nbextc01.
- `lane_extern_nested.mind` (~34 tok, tiny) — lane.extern.nested: an extern block inside a function body.
- `lane_extern_signature_mismatch.mind` (~40 tok, tiny) — lane.extern.signature_mismatch: a table name with a width its row does not carry.
- `lane_extern_unknown_symbol.mind` (~36 tok, tiny) — lane.extern.unknown_symbol: a name outside the closed table.
- `lane_source_array_element.mind` (~32 tok, tiny) — lane.source.array_element: a non-literal array element.
- `lane_source_array_literal.mind` (~33 tok, tiny) — lane.source.array_literal: an array literal argument.
- `lane_source_array_value.mind` (~27 tok, tiny) — lane.source.array_value: an array returned as a value.
- `lane_source_assert.mind` (~27 tok, tiny) — lane.source.assert: an assert statement.
- `lane_source_binding_type.mind` (~29 tok, tiny) — lane.source.binding_type: a let bound to a unit value.
- `lane_source_cast.mind` (~30 tok, tiny) — lane.source.cast: a cast the lane does not lower (f64 to i32).
- `lane_source_closure.mind` (~30 tok, tiny) — lane.source.closure: a closure value.
- `lane_source_const_attributes.mind` (~29 tok, tiny) — lane.source.const_attributes: an attributed const.
- `lane_source_const_type.mind` (~31 tok, tiny) — lane.source.const_type: a const initialised from a unit call.
- `lane_source_duplicate_parameter.mind` (~37 tok, tiny) — lane.source.duplicate_parameter: two parameters with one name.
- `lane_source_enum.mind` (~27 tok, tiny) — lane.source.enum: an enum declared in a function body.
- `lane_source_extern_const.mind` (~30 tok, tiny) — lane.source.extern_const: an extern constant in a function body.
- `lane_source_float_neg_lowering.mind` (~32 tok, tiny) — lane.source.float_neg_lowering: runtime negation of an f64 value.
- `lane_source_for_each.mind` (~36 tok, tiny) — lane.source.for_each: iteration over an array.
- `lane_source_function_contract.mind` (~32 tok, tiny) — lane.source.function_contract: a generic function.
- `lane_source_index.mind` (~24 tok, tiny) — lane.source.index: an index into a scalar.
- `lane_source_index_store.mind` (~27 tok, tiny) — lane.source.index_store: an element store.
- `lane_source_loop_attributes.mind` (~37 tok, tiny) — lane.source.loop_attributes: an attributed range loop.
- `lane_source_loop_exit_scope.mind` (~23 tok, tiny) — lane.source.loop_exit_scope: break outside a loop.
- `lane_source_map.mind` (~21 tok, tiny) — lane.source.map: a map literal.
- `lane_source_match.mind` (~32 tok, tiny) — lane.source.match: a match expression.
- `lane_source_method_call.mind` (~27 tok, tiny) — lane.source.method_call: a method call on a scalar.
- `lane_source_neg_type.mind` (~27 tok, tiny) — lane.source.neg_type: negation of a unit value.
- `lane_source_nested_function.mind` (~35 tok, tiny) — lane.source.nested_function: a function defined inside a function body.
- `lane_source_nested_item.mind` (~45 tok, tiny) — lane.source.nested_item: a const item in a function body (a nested std
- `lane_source_print.mind` (~33 tok, tiny) — lane.source.print: print lowers to runtime calls no native-lane compiler implements.
- `lane_source_reference.mind` (~28 tok, tiny) — lane.source.reference: a reference to a local.
- `lane_source_region.mind` (~25 tok, tiny) — lane.source.region: a region block.
- `lane_source_set.mind` (~21 tok, tiny) — lane.source.set: a set literal.
- `lane_source_struct.mind` (~28 tok, tiny) — lane.source.struct: a struct declared in a function body.
- `lane_source_tensor.mind` (~22 tok, tiny) — lane.source.tensor: a gradient call.
- `lane_source_top_level.mind` (~25 tok, tiny) — lane.source.top_level: a struct declaration.
- `lane_source_trait.mind` (~30 tok, tiny) — lane.source.trait: a trait declared in a function body.
- `lane_source_tuple.mind` (~21 tok, tiny) — lane.source.tuple: a tuple value.
- `lane_source_type_alias.mind` (~27 tok, tiny) — lane.source.type_alias: a type alias in a function body.
- `lane_source_unbound.mind` (~31 tok, tiny) — lane.source.unbound: assignment to a function name.
- `lane_type_array.mind` (~29 tok, tiny) — lane.type.array: an array-typed parameter.
- `lane_type_branch_mismatch.mind` (~39 tok, tiny) — lane.type.branch_mismatch: an if expression whose branches differ in type.
- `lane_type_fn_pointer.mind` (~36 tok, tiny) — lane.type.fn_pointer: a C function pointer parameter.
- `lane_type_generic.mind` (~30 tok, tiny) — lane.type.generic: a generic type parameter.
- `lane_type_int_width.mind` (~22 tok, tiny) — lane.type.int_width: an i32 binding.
- `lane_type_mismatch.mind` (~29 tok, tiny) — lane.type.mismatch: an f64 binding initialised with an integer.
- `lane_type_named.mind` (~29 tok, tiny) — lane.type.named: a parameter of a named type.
- `lane_type_non_f64_float.mind` (~24 tok, tiny) — lane.type.non_f64_float: an f32 binding.
- `lane_type_raw_pointer.mind` (~31 tok, tiny) — lane.type.raw_pointer: a raw pointer parameter.
- `lane_type_reference.mind` (~29 tok, tiny) — lane.type.reference: a reference parameter.
- `lane_type_slice.mind` (~28 tok, tiny) — lane.type.slice: a slice-typed parameter.
- `lane_type_tensor.mind` (~31 tok, tiny) — lane.type.tensor: a tensor parameter.
- `lane_type_tuple.mind` (~30 tok, tiny) — lane.type.tuple: a tuple-typed parameter.
### `tests/fixtures/native_lane/`

- `request_corpus.txt` (~15287 tok, huge) — # Native-lane request corpus: the programs of the native-wave2 / native-wave3
### `tests/fixtures/native_lane_std/`

- `cli_args_get.mind` (~22 tok, tiny) — fn main() -> i64 {
- `import_not_seed_unused.mind` (~14 tok, tiny) — fn main() -> i64 {
- `import_not_seed_used.mind` (~37 tok, tiny) — fn main() -> i64 {
- `import_seed_unused.mind` (~14 tok, tiny) — fn main() -> i64 {
- `io_canon_anchor.mind` (~66 tok, small) — fn main() -> i64 {
- `io_print_bytes.mind` (~55 tok, small) — fn main() -> i64 {
- `naes_shape_closed_vendored.mind` (~3137 tok, huge) — An external unit shape: a closed, primitive image (no import, no string
- `naes_shape_open_divergent.mind` (~3043 tok, huge) — naes_shape_closed_vendored plus a call into the std seed (`sha256`) outside
- `naes_shape_public_name_closed.mind` (~25 tok, tiny) — fn decode(a: i64) -> i64 {
- `naes_shape_public_name_open.mind` (~120 tok, small) — naes_shape_public_name_closed plus one call into the std seed outside every
- `private_collision_with_seed_call.mind` (~66 tok, small) — fn arena_align8(n: i64) -> i64 {
- `private_seed_helper.mind` (~47 tok, tiny) — fn main() -> i64 {
- `ring_roundtrip.mind` (~61 tok, small) — fn main() -> i64 {
- `sha256_1000.mind` (~95 tok, small) — fn main() -> i64 {
- `sha256_empty.mind` (~57 tok, small) — fn main() -> i64 {
- `sha256_without_import.mind` (~52 tok, small) — fn main() -> i64 {
- `std_sha256_in_loop.mind` (~146 tok, small) — std.sha256's `sha256` reached from a loop. Its scratch is freed on every
- `string_push_capacity.mind` (~32 tok, tiny) — fn main() -> i64 {
- `user_defines_imported_seed_function.mind` (~38 tok, tiny) — fn string_push_capacity(len: i64, cap: i64) -> i64 {
- `user_defines_sha256.mind` (~30 tok, tiny) — fn sha256(a: i64, b: i64, c: i64) -> i64 {
- `user_shadows_seed_helper.mind` (~69 tok, small) — fn sha_ch(x: i64, y: i64, z: i64) -> i64 {
- `vec_zeroed.mind` (~26 tok, tiny) — fn main() -> i64 {
### `tests/fixtures/native_lane_string_rows/`

- `baseline_frozen_seed.tsv` (~8115 tok, huge) — # string-rule acceptance rows: what compiler examples/mindc_mind/testdata/selfho
- `keys.tsv` (~237 tok, medium) — # The refusal keys the rows may name (verdict refuse, or a recorded key of a by_
- `mutants.tsv` (~2347 tok, huge) — arm	clause	off	rows
- `README.md` (~4867 tok, huge) — String-rule acceptance rows
- `rows.tsv` (~17944 tok, huge) — id	verdict	key	line	col	exit	profile	status	clause	note
- `selftest_judge.tsv` (~606 tok, large) — # Cases of the runner self-test: a row (verdict exit key line col) and what a co
- `selftest_plants.tsv` (~747 tok, large) — # Planted defects of the runner self-test: in a scratch copy of this directory, 
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
### `tests/fixtures/string_eq_module_scope/`

- `main.mind` (~12 tok, tiny) — fn main() -> i64 {
- `Mind.toml` (~19 tok, tiny) — [package]
### `tests/fixtures/string_eq_module_scope/src/`

- `j.mind` (~94 tok, small) — Positive control: this module imports `z`, which stores a value of
- `k.mind` (~221 tok, medium) — Typed element compares. No module this one shares an importer with
- `m.mind` (~9 tok, tiny) — fn one() -> i64 {
- `z.mind` (~56 tok, small) — No module of `k`'s import closure imports this one. It stores a value of
### `tests/fixtures/`

- `test_phase_b_all_pass.mind` (~67 tok, small) — RFC 0008 Phase B test fixture — both tests pass.
- `test_phase_b_one_fail.mind` (~80 tok, small) — RFC 0008 Phase B test fixture — one pass, one fail.
### `tests/fixtures/tls13_client/`

- `1301-001d-hrr0.bin` (binary, 1869 bytes)
- `1301-001d-hrr0-tamper1.bin` (binary, 1869 bytes)
- `1301-001d-hrr0-tamper2.bin` (binary, 1869 bytes)
- `1301-001d-hrr1.bin` (binary, 2116 bytes)
- `1301-001d-hrr1-tamper1.bin` (binary, 2116 bytes)
- `1301-001d-hrr1-tamper2.bin` (binary, 2116 bytes)
- `1301-11ec-hrr0.bin` (binary, 4179 bytes)
- `1301-11ec-hrr0-tamper1.bin` (binary, 4179 bytes)
- `1301-11ec-hrr0-tamper2.bin` (binary, 4179 bytes)
- `1301-11ec-hrr1.bin` (binary, 4392 bytes)
- `1301-11ec-hrr1-tamper1.bin` (binary, 4392 bytes)
- `1301-11ec-hrr1-tamper2.bin` (binary, 4392 bytes)
- `1303-001d-hrr0.bin` (binary, 1869 bytes)
- `1303-001d-hrr0-tamper1.bin` (binary, 1869 bytes)
- `1303-001d-hrr0-tamper2.bin` (binary, 1869 bytes)
- `1303-001d-hrr1.bin` (binary, 2116 bytes)
- `1303-001d-hrr1-tamper1.bin` (binary, 2116 bytes)
- `1303-001d-hrr1-tamper2.bin` (binary, 2116 bytes)
- `1303-11ec-hrr0.bin` (binary, 4179 bytes)
- `1303-11ec-hrr0-tamper1.bin` (binary, 4179 bytes)
- `1303-11ec-hrr0-tamper2.bin` (binary, 4179 bytes)
- `1303-11ec-hrr1.bin` (binary, 4392 bytes)
- `1303-11ec-hrr1-tamper1.bin` (binary, 4392 bytes)
- `1303-11ec-hrr1-tamper2.bin` (binary, 4392 bytes)
- `cvsigs.mind` (~3774 tok, huge) — Pinned deterministic TEST-ONLY RSA-PSS/SHA256 signatures.
- `fixture_keys.mind` (~4049 tok, huge) — GENERATED by make_fixture_keys.py (deterministic seed 20261014). Dev-time
- `fixture_private.mind` (~930 tok, large) — TEST-ONLY RSA private key, fixed seed 20261014. NEVER use for real traffic.
- `generate.mind` (~4008 tok, huge) — Deterministic server-flight generator (test fixture module; the production
- `native_alloc_probe.mind` (~102 tok, small) — Minimal reproduction of the native gate refusal on this checkpoint.
- `native_hkdf_loop_probe.mind` (~127 tok, small) — Reduced from std/hkdf.mind: hkdf_expand's value-if inside a loop.
- `README.md` (~2213 tok, huge) — TLS 1.3 client recorded flights
- `rfc8448.mind` (~2400 tok, huge) — Fixed RFC 8448 section 3 bytes, copied from tls13_handshake_driver.py.
- `SHA256SUMS` (~794 tok, large) — 8308fcebf9a2513ed0f9e5285271a2f9c8423cf2acd233aad8c8845bd509f6df  tests/fixtures
- `test_x509_verify.inc.mind` (~2677 tok, huge) — === TLS-CLIENT TEST-ONLY VERIFIER =====================================
### `tests/`

- `flat_emit_sibling_const.rs` (~542 tok, large) — Copyright 2026 STARGA Inc.
- `flat_string_field_schema.rs` (~4984 tok, huge) — Copyright 2026 STARGA Inc.
- `fmt_bool_literal_roundtrip.rs` (~1958 tok, huge) — Copyright 2025 STARGA Inc.
- `fmt_comment_placement.rs` (~2772 tok, huge) — Copyright 2025 STARGA Inc.
- `fmt_idempotence.rs` (~4287 tok, huge) — Copyright 2025 STARGA Inc.
- `fmt_ir_preservation.rs` (~3956 tok, huge) — Copyright 2025 STARGA Inc.
- `fmt_module_block_item_preserved.rs` (~926 tok, large) — Regression: `mindc fmt` must not drop item declarations nested inside a
- `fmt_stdlib_stability.rs` (~2756 tok, huge) — Copyright 2025 STARGA Inc.
- `fmt_surface_spelling_roundtrip.rs` (~3023 tok, huge) — Copyright 2025 STARGA Inc.
- `fn_value_call_reject.rs` (~761 tok, large) — Copyright 2025 STARGA Inc.
- `for_body_redeclares_var_run.rs` (~1327 tok, large) — Copyright 2026 STARGA Inc.
- `for_continue_advances_run.rs` (~2771 tok, huge) — Copyright 2025 STARGA Inc. Licensed under the Apache License, Version 2.0.
- `for_continue_step_injection.rs` (~1306 tok, large) — Copyright 2025 STARGA Inc. Licensed under the Apache License, Version 2.0.
- `for_each_run.rs` (~805 tok, large) — Copyright 2025 STARGA Inc.
- `for_hygiene_run.rs` (~1424 tok, large) — Copyright 2025 STARGA Inc. Licensed under the Apache License, Version 2.0.
### `tests/fs_checked_listing/`

- `CONTRACT.md` (~1575 tok, huge) — Checked directory listing development contract
- `EVIDENCE.md` (~1631 tok, huge) — Checked directory-listing qualification
- `FAULT_CASES.json` (~1457 tok, large) — Keys: schema, status, baseline, cases, observation
- `fault_driver.mind` (~1328 tok, large) — Author: STARGA Inc. Explicit expected syscall and allocation observations.
- `fault_main.mind` (~31 tok, tiny) — Author: STARGA Inc. All predeclared synthetic cases, no real syscall authority.
- `fault_support.mind` (~1371 tok, large) — Author: STARGA Inc. Test-only syscall and allocation observations.
- `import.mind` (~143 tok, small) — Author: STARGA Inc. Exercise the embedded module through its public import.
- `INSTRUMENTATION.json` (~996 tok, large) — Keys: schema, scope, ordered_exact_replacements
- `Mind.toml` (~38 tok, tiny) — [package]
- `MUTATIONS.json` (~933 tok, large) — Keys: schema, status, cases
- `real.mind` (~1534 tok, huge) — Author: STARGA Inc. Original API and real Linux raw syscalls.
- `TEST_PLAN.md` (~1244 tok, large) — Checked-listing acceptance plan
- `unit.mind` (~1762 tok, huge) — Author: STARGA Inc. Independent expected fields; no parser result as oracle.
### `tests/`

- `g2_differential_mlir.rs` (~10655 tok, huge) — Copyright 2025 STARGA Inc.
- `g2_diff_tensor_param_grammar.rs` (~1908 tok, huge) — Copyright 2026 STARGA Inc.
- `gate_assert_count_contract_test.py` (~8635 tok, huge) — Gate test: the assertion count must be EVIDENCE, never a printed integer.
- `_gate_floor_cases.py` (~1172 tok, large) — Floor cases of the gate-assert count contract.
- `gather_preview.rs` (~288 tok, medium) — Copyright 2025 STARGA Inc.
- `generics_lowering.rs` (~1384 tok, large) — Copyright 2026 STARGA Inc. Licensed under the Apache License, Version 2.0.
- `genref_phase_jb.rs` (~3790 tok, huge) — Copyright 2025 STARGA Inc.
- `grad_wrt_resolve.rs` (~730 tok, large) — Copyright 2025 STARGA Inc.
- `harness_portability.rs` (~1260 tok, large) — Copyright 2025 STARGA Inc.
- `harness_scratch_isolation.rs` (~3343 tok, huge) — Copyright 2025 STARGA Inc.
- `hpack_driver.py` (~3027 tok, huge) — # Official-vector driver for std/hpack.mind (pure-MIND HPACK decoding,
- `http2_frame_driver.py` (~4195 tok, huge) — # Reference-vector driver for std/http2_frame.mind (pure-MIND HTTP/2 framing,
### `tests/https_loopback/`

- `ca_bad.pem` (~83 tok, small) — -----BEGIN CERTIFICATE-----
- `ca_bundle.pem` (~6180 tok, huge) — # MIND std.https loopback test bundle (TEST-ONLY anchors).
- `ca_other.pem` (~277 tok, medium) — -----BEGIN CERTIFICATE-----
- `client.mind` (~3088 tok, huge) — Client program for tests/std_https_loopback.rs (TEST-ONLY): every case
- `inter.der` (binary, 811 bytes)
- `leaf_ec.der` (binary, 636 bytes)
- `leaf_expired.der` (binary, 839 bytes)
- `leaf_good.der` (binary, 839 bytes)
- `leaf_key_test_only.cnf` (~701 tok, large) — # TEST-ONLY RSA-2048 key of leaf_good.der, leaf_expired.der and leaf_wronghost.d
- `leaf_rsa3072.der` (binary, 967 bytes)
- `leaf_rsa3072_key_test_only.cnf` (~981 tok, large) — # TEST-ONLY RSA-3072 key of leaf_rsa3072.der (tests/https_loopback/README.md).
- `leaf_wronghost.der` (binary, 843 bytes)
- `README.md` (~1029 tok, large) — std.https loopback fixtures (TEST-ONLY)
- `request.mind` (~1500 tok, huge) — One std.https request for tests/std_https_openssl.rs (TEST-ONLY).
- `root.der` (binary, 767 bytes)
- `server_conn.mind` (~1995 tok, huge) — Loopback TLS 1.3 server, part: one connection (TEST-ONLY). The harness appends
- `server_http.mind` (~1920 tok, huge) — Loopback TLS 1.3 server, part: request routing and responses (TEST-ONLY). The harness appends
- `server_key_test_only.txt` (~293 tok, medium) — # TEST-ONLY RSA-2048 key of tests/https_loopback/leaf_*.der (localhost).
- `server.mind` (~1226 tok, large) — Loopback TLS 1.3 server for tests/std_https_loopback.rs and
- `server_rsa.mind` (~1817 tok, huge) — Loopback TLS 1.3 server, part: the RSA-2048 signer (TEST-ONLY). The harness appends
- `server_tls.mind` (~864 tok, large) — Loopback TLS 1.3 server, part: handshake helpers (TEST-ONLY). The harness appends
- `steps.mind` (~1369 tok, large) — Step client for tests/std_https_network_safe.rs (TEST-ONLY): one std.https
### `tests/`

- `if_expr.rs` (~429 tok, medium) — Copyright 2025 STARGA Inc.
- `if_merge_shadow_318.rs` (~1444 tok, large) — Copyright 2025 STARGA Inc.
- `imported_enum_eval.rs` (~1957 tok, huge) — Copyright 2026 STARGA Inc.
- `imported_extern_narrow_ret_run.rs` (~3696 tok, huge) — Copyright 2026 STARGA Inc.
- `index_slice_grad.rs` (~289 tok, medium) — Copyright 2025 STARGA Inc.
- `index_slice_preview.rs` (~376 tok, medium) — Copyright 2025 STARGA Inc.
- `index_slice_types.rs` (~250 tok, medium) — Copyright 2025 STARGA Inc.
- `int_determinism.rs` (~1716 tok, huge) — Copyright 2025 STARGA Inc.
- `intra_module_call_arity.rs` (~3085 tok, huge) — Copyright 2025 STARGA Inc.
- `intrinsic_call_arity_build.rs` (~3267 tok, huge) — Copyright 2026 STARGA Inc.
- `intrinsic_call_arity.rs` (~3490 tok, huge) — Copyright 2026 STARGA Inc.
- `int_suffix_literal.rs` (~883 tok, large) — Copyright 2025 STARGA Inc.
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
- `issue_235_single_file_import.rs` (~2468 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `issue_246_bitnot_operator.rs` (~571 tok, large) — Copyright 2025 STARGA Inc.
- `issue_246_tensor_named_dtype.rs` (~525 tok, large) — Copyright 2025 STARGA Inc.
- `issue_263_method_chain_newline.rs` (~869 tok, large) — Copyright 2025 STARGA Inc.
- `issue_263_use_path_import.rs` (~984 tok, large) — Copyright 2025 STARGA Inc.
- `issue_263_usize_suffix_literal.rs` (~770 tok, large) — Copyright 2025 STARGA Inc.
- `keccak_driver.py` (~1507 tok, huge) — # Official-vector driver for std/keccak.mind (pure-MIND FIPS 202).
### `tests/lexical/`

- `invalid_keywords_as_identifiers.mind` (~45 tok, tiny) — Lexical test: Keywords cannot be used as identifiers
- `numeric_literals.mind` (~74 tok, small) — Lexical test: Numeric literal formats
- `valid_identifiers.mind` (~72 tok, small) — Lexical test: Valid identifier formats
### `tests/`

- `libm_name_capture_run.rs` (~2977 tok, huge) — Copyright 2025 STARGA Inc.
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
- `mic3_cli_emit.rs` (~4682 tok, huge) — Copyright 2025 STARGA Inc.
- `mic3_const_dense_tensor_roundtrip.rs` (~2592 tok, huge) — Copyright 2026 STARGA Inc.
- `mic3_parser_dos.rs` (~1891 tok, huge) — Copyright 2025 STARGA Inc.
- `micb_dos_reject.rs` (~2341 tok, huge) — Copyright 2025 STARGA Inc.
- `mindc_artifact_name.rs` (~4118 tok, huge) — Copyright 2025 STARGA Inc.
### `tests/mindc_build_phase_a/`

- `backend.rs` (~1173 tok, large) — Copyright 2025 STARGA Inc.
### `tests/`

- `mindc_build_phase_a.rs` (~5935 tok, huge) — Copyright 2025 STARGA Inc.
- `mindc_cache_build_inputs.rs` (~2147 tok, huge) — Copyright 2025 STARGA Inc.
- `mindc_cache_phase_f.rs` (~6289 tok, huge) — Copyright 2025 STARGA Inc.
- `mindc_deps_phase_de.rs` (~6880 tok, huge) — Copyright 2025 STARGA Inc.
- `mindc_doc_phase1.rs` (~3004 tok, huge) — Copyright 2025 STARGA Inc.
- `mindc_inspect.rs` (~2191 tok, huge) — Integration test for `mindc inspect` — the mic@3 artifact decoder/differ.
- `mindc_project_lock.rs` (~6202 tok, huge) — Copyright 2026 STARGA Inc.
### `tests/mindcraft/check/`

- `clean.mind` (~11 tok, tiny) — fn add(a: i64, b: i64) -> i64 {
### `tests/`

- `mindcraft_check_cli.rs` (~6518 tok, huge) — Copyright 2025 STARGA Inc.
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

- `mindcraft_fmt_cli.rs` (~3283 tok, huge) — Copyright 2025 STARGA Inc.
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

- `mindcraft_lint_shadowing.rs` (~1256 tok, large) — Copyright 2025 STARGA Inc.
### `tests/mindcraft/lint/`

- `trailing_ws_clean.mind` (~10 tok, tiny) — fn foo() -> i64 {
- `trailing_ws_dirty.mind` (~11 tok, tiny) — fn foo() -> i64 {
### `tests/mindcraft/lint/unused_import/`

- `negative.mind` (~53 tok, small) — Negative fixture: `use std.vec` is declared AND the `vec` identifier
- `positive.mind` (~46 tok, tiny) — Positive fixture: `use std.vec` is declared but no symbol from vec
### `tests/`

- `mindcraft_lint_unused_import.rs` (~882 tok, large) — Copyright 2025 STARGA Inc.
- `mindcraft_lint_vec_check.rs` (~549 tok, large) — Copyright 2025 STARGA Inc.
### `tests/mindcraft/`

- `STABILITY_SKIP_LIST.md` (~408 tok, medium) — Formatter Stability Skip List
### `tests/`

- `mindc.rs` (~2105 tok, huge) — Copyright 2025 STARGA Inc.
- `mindc_symlinked_project_identity.rs` (~2240 tok, huge) — Copyright 2026 STARGA Inc.
- `mindc_test_eval_arrays_results_parity.rs` (~3996 tok, huge) — Copyright 2026 STARGA Inc.
- `mindc_test_eval_blockers_parity.rs` (~3025 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `mindc_test_eval_float_compare_parity.rs` (~2455 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `mindc_test_eval_parity.rs` (~3505 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `mindc_test_evaluator_issues.rs` (~6042 tok, huge) — Copyright 2025 STARGA Inc.
- `mindc_test_imports.rs` (~6333 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `mindc_test_nested_modules.rs` (~1901 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `mindc_test_phase_b.rs` (~3574 tok, huge) — Copyright 2025 STARGA Inc.
- `mindc_workspace_phase_c.rs` (~4166 tok, huge) — Copyright 2025 STARGA Inc.
### `tests/mindfuzz_corpus_floor/`

- `mod.rs` (~855 tok, large) — Corpus-size floor for the differential determinism fuzzer.
### `tests/mindfuzz_cross_substrate/known_environmental/`

- `README.md` (~314 tok, medium) — Known-environmental fuzzer artifacts
- `seed_deadbeef_prog000_e1001_bitwise.mind` (~242 tok, medium) — NOT A DIVERGENCE — misfiled build-capability refusal (issue #72 fuzzer).
### `tests/mindfuzz_cross_substrate/reproducers/`

- `fuzz_oracle_failure_seed_deadbeef_prog000.mind` (~218 tok, medium) — ORACLE FAILURE — a stage did not complete; NOT a proven cross-substrate divergence (issue #72 fuzzer)
- `fuzz_oracle_failure_seed_deadbeef_prog006.mind` (~157 tok, small) — DIVERGENCE REPRODUCER (issue #72 fuzzer)
### `tests/`

- `mindfuzz_cross_substrate.rs` (~16552 tok, huge) — Copyright 2025 STARGA Inc.
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
- `mlir_build.rs` (~2759 tok, huge) — Copyright 2025 STARGA Inc.
- `mlir_exec.rs` (~862 tok, large) — Copyright 2025 STARGA Inc.
- `mlir_export_indexing.rs` (~419 tok, medium) — Copyright 2025 STARGA Inc.
- `mlir_export_linalg.rs` (~873 tok, large) — Copyright 2025 STARGA Inc.
- `mlir_export_reductions.rs` (~1833 tok, huge) — Copyright 2025 STARGA Inc.
- `mlir_export.rs` (~582 tok, large) — Copyright 2025 STARGA Inc.
- `mlir_export_shape.rs` (~353 tok, medium) — Copyright 2025 STARGA Inc.
- `mlir_file_and_lower.rs` (~611 tok, large) — Copyright 2025 STARGA Inc.
- `mlir_gpu.rs` (~314 tok, medium) — Copyright 2025 STARGA Inc.
- `mlir_jit.rs` (~285 tok, medium) — Copyright 2025 STARGA Inc.
- `mlir_lowering.rs` (~1588 tok, huge)
- `mlir_memory_addresses.rs` (~3673 tok, huge) — Copyright 2026 STARGA Inc.
- `mlir_opt.rs` (~474 tok, medium) — Copyright 2025 STARGA Inc.
- `mlir_terminator_last_op.rs` (~1397 tok, large) — Copyright 2026 STARGA Inc.
- `mlir_toolchain_diagnostics.rs` (~1776 tok, huge) — Copyright 2026 STARGA Inc.
- `mlkem768_driver.py` (~1699 tok, huge) — # Reference-vector driver for std/mlkem768.mind (pure-MIND ML-KEM-768,
- `module_const_run.rs` (~729 tok, large) — Copyright 2025 STARGA Inc.
- `module_decl_run.rs` (~597 tok, large) — Copyright 2025 STARGA Inc.
- `module_enum_match_run.rs` (~984 tok, large) — Copyright 2025 STARGA Inc.
- `module_non_fn_call_reject.rs` (~1215 tok, large) — Copyright 2025 STARGA Inc.
- `module_path_reference.rs` (~619 tok, large) — Copyright 2026 STARGA Inc.
- `module_size_ratchet.rs` (~4363 tok, huge) — Copyright 2025 STARGA Inc.
- `multimodule_determinism_run.rs` (~4292 tok, huge) — Copyright 2025 STARGA Inc.
- `narrow_call_abi.rs` (~1344 tok, large) — Copyright 2025 STARGA Inc.
- `narrowing_check.rs` (~438 tok, medium) — Regression test for the silent i64->i32 narrowing miscompile found by MIND-Fuzz
- `narrow_local_mask_run.rs` (~879 tok, large) — Copyright 2025 STARGA Inc.
- `narrow_locals_leak_run.rs` (~1743 tok, huge) — Copyright 2025 STARGA Inc.
- `narrow_reassign_mask_run.rs` (~1385 tok, large) — Copyright 2025 STARGA Inc.
- `narrow_reassign_run.rs` (~687 tok, large) — Copyright 2026 STARGA Inc.
- `narrow_sig_abi_run.rs` (~1163 tok, large) — Copyright 2025 STARGA Inc.
- `narrow_signedness_batch2_run.rs` (~2236 tok, huge) — Copyright 2025 STARGA Inc.
- `narrow_signedness_batch_run.rs` (~1629 tok, huge) — Copyright 2025 STARGA Inc.
- `narrow_tuple_pr216_run.rs` (~1409 tok, large) — Copyright 2025 STARGA Inc.
- `narrow_unsigned_div_zero_run.rs` (~1413 tok, large) — Copyright 2025 STARGA Inc.
### `tests/native_alloc_std_drivers/`

- `x25519_chain_driver.mind` (~315 tok, medium) — Driver of the allocator row default/x25519_chain (self_host_native_alloc_bounded_smoke.py).
### `tests/`

- `native_array_binop_refusal.rs` (~1914 tok, huge) — Copyright 2025 STARGA Inc.
- `native_bridge_child_status.rs` (~5926 tok, huge) — Copyright 2025 STARGA Inc.
- `native_bridge_refusals.rs` (~6436 tok, huge) — Copyright 2025 STARGA Inc.
### `tests/native_bridge_support/`

- `draining_native_compiler.rs` (~391 tok, medium) — Host-native compiler fixture for admission-only controls.
- `mod.rs` (~2596 tok, huge) — Shared harness for the native module-bridge integration controls.
### `tests/`

- `native_closure_admission.rs` (~4567 tok, huge) — Copyright 2025 STARGA Inc.
- `native_default_profile_imports.rs` (~3723 tok, huge) — Default-feature-profile controls for the native backend's import admission.
- `native_embedded_seed.rs` (~4634 tok, huge) — Copyright 2025 STARGA Inc.
- `native_historical_intrinsic_refusal.rs` (~1463 tok, large) — Copyright 2025 STARGA Inc.
- `native_import_terminator_controls.rs` (~1558 tok, huge) — Native bridge controls: an `import`/`use` item the frozen compiler would
- `native_lane_bridge_controls.rs` (~2466 tok, huge) — Copyright 2025 STARGA Inc.
- `native_lane_extern_fixed.rs` (~7448 tok, huge) — Copyright 2026 STARGA Inc.
- `native_lane_extern_parity.rs` (~8718 tok, huge) — Copyright 2026 STARGA Inc.
### `tests/native_lane_extern_support/`

- `mod.rs` (~7775 tok, huge) — Copyright 2026 STARGA Inc.
### `tests/`

- `native_lane_gpu_loader.rs` (~8251 tok, huge) — Copyright 2026 STARGA Inc.
- `native_lane_open_parity.rs` (~3403 tok, huge) — Copyright 2026 STARGA Inc.
- `native_lane_refusal_keys.rs` (~5357 tok, huge) — Copyright 2026 STARGA Inc.
- `native_lane_request_fixtures.rs` (~4516 tok, huge) — Copyright 2026 STARGA Inc.
- `native_lane_std_seed_calls.rs` (~2236 tok, huge) — Copyright 2026 STARGA Inc.
- `native_module_bridge_controls.rs` (~8507 tok, huge) — Native module-bridge controls: SOURCE RESOLUTION and ADMISSION.
- `native_module_bridge_visibility_controls.rs` (~3913 tok, huge) — Native module-bridge controls: VISIBILITY, EMIT KIND, CROSS-INVOCATION
- `native_opt_wiring.rs` (~1953 tok, huge) — Copyright 2025 STARGA Inc.
- `native_seed_copy_admission_controls.rs` (~2081 tok, huge) — Historical native admission of user copies of private std seed helpers.
- `nerve_numerics_lint.rs` (~4079 tok, huge) — Copyright 2025 STARGA Inc.
- `nested_block_surface_run.rs` (~990 tok, large) — Copyright 2025 STARGA Inc.
- `nested_collection_run.rs` (~772 tok, large) — Copyright 2025 STARGA Inc.
- `nested_import_owner_resolution_run.rs` (~6563 tok, huge) — Copyright 2026 STARGA Inc.
- `nested_mut_thread_run.rs` (~1520 tok, huge) — Copyright 2025 STARGA Inc.
- `nested_project_import_scope_run.rs` (~3562 tok, huge) — Copyright 2026 STARGA Inc.
- `non_final_catch_all_match_run.rs` (~861 tok, large) — Copyright 2025 STARGA Inc. Licensed under the Apache License, Version 2.0.
- `non_test_rust_line_ratchet.rs` (~5387 tok, huge) — Copyright 2026 STARGA Inc.
- `open_error_kat.mind` (~344 tok, medium) — Author: STARGA Inc.
- `open_error_kat.rs` (~1163 tok, large) — Copyright 2026 STARGA Inc.
- `ops_registry.rs` (~114 tok, small)
- `package_basic.rs` (~491 tok, medium) — Copyright 2025 STARGA Inc.
- `package_traversal.rs` (~905 tok, large) — Copyright 2025 STARGA Inc.
- `parse_match_and_ref.rs` (~3704 tok, huge) — Copyright 2025 STARGA Inc.
- `parse_phase10_surface.rs` (~4988 tok, huge) — Parse-target tests for Phase 10.5 / 10.6 surface-syntax acceptance.
- `parser_trivia.rs` (~2706 tok, huge) — Copyright 2025 STARGA Inc.
- `parser_unsigned_i64_literals.rs` (~1544 tok, huge) — Copyright 2025 STARGA Inc.
- `parse_try_operator.rs` (~1194 tok, large) — Copyright 2025 STARGA Inc.
- `pattern_guard_run.rs` (~899 tok, large) — Copyright 2025 STARGA Inc. Licensed under the Apache License, Version 2.0.
- `phase_g_keystone_bootstrap.rs` (~7976 tok, huge) — Copyright 2025 STARGA Inc.
- `pipeline.rs` (~1476 tok, large) — Copyright 2025 STARGA Inc.
- `pqc_hybrid_cli_signing.rs` (~7885 tok, huge) — End-to-end CLI controls for post-quantum hybrid artifact signing.
- `private_fn_link_collision_run.rs` (~992 tok, large) — Copyright 2026 STARGA Inc.
- `project_function_resolution.rs` (~1475 tok, large) — Copyright 2026 STARGA Inc.
- `public_cpu_native_link.rs` (~3345 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `qualified_enum_explicit_sources.rs` (~3767 tok, huge) — Copyright 2026 STARGA Inc.
- `qualified_enum_run.rs` (~4729 tok, huge) — Copyright 2025 STARGA Inc.
- `reap_threshold.rs` (~2108 tok, huge) — Copyright 2025 STARGA Inc.
- `reductions_grad.rs` (~390 tok, medium) — Copyright 2025 STARGA Inc.
- `reductions_preview.rs` (~390 tok, medium) — Copyright 2025 STARGA Inc.
- `_ref_mic3_dump.rs` (~3651 tok, huge) — Committed self-host reference generator (A9b): reconstruct
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
- `rfc0012_phase_a_shape_types.rs` (~6812 tok, huge) — Copyright 2025 STARGA Inc.
- `rfc0012_phase_b_operators.rs` (~4737 tok, huge) — Copyright 2025 STARGA Inc.
- `rfc0012_phase_c_annotations.rs` (~2405 tok, huge) — Copyright 2025 STARGA Inc.
- `roofline_peak.rs` (~697 tok, large) — Correctness proof for the CPUID-gated roofline denominator (B4).
- `rsa_pss_driver.py` (~2693 tok, huge) — # Ground-truth driver for std/rsa_pss.mind (pure-MIND RSASSA-PSS-VERIFY,
### `tests/runtime/`

- `elementwise_add.mind` (~68 tok, small) — Runtime test: Element-wise addition execution
### `tests/`

- `runtime_obj_overlap.rs` (~1918 tok, huge) — Copyright 2026 STARGA Inc.
### `tests/runtime/`

- `reduction_sum.mind` (~67 tok, small) — Runtime test: Reduction sum operation
### `tests/`

- `scalar_cast_call_run.rs` (~673 tok, large) — Copyright 2025 STARGA Inc.
- `scalar_cast_unsigned_narrow_run.rs` (~1047 tok, large) — Copyright 2025 STARGA Inc.
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
- `GAPS.md` (~5947 tok, huge) — Self-host nfn driver — gap inventory (fuzz-discovered)
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
- `matchcall_letinit_1.mind` (~34 tok, tiny) — fn g(a: i64) -> i64 {
- `matchcall-scrutinee_0arg.mind` (~24 tok, tiny) — fn g() -> i64 {
- `matchcall-scrutinee_1arg.mind` (~33 tok, tiny) — fn g(a: i64) -> i64 {
- `matchcall-scrutinee_arith.mind` (~21 tok, tiny) — fn f(a: i64) -> i64 {
- `match_enum_payload_1.mind` (~29 tok, tiny) — fn f(e: E) -> i64 {
- `match_enum_payload_reordered_1.mind` (~18 tok, tiny) — fn f(e:E)->i64 {
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
- `struct-lit_1.mind` (~63 tok, small)
- `struct-lit_2.mind` (~56 tok, small)
- `struct-lit_3.mind` (~60 tok, small)
- `struct-lit-field-recv_1.mind` (~23 tok, tiny)
- `struct-lit-field-recv_2.mind` (~23 tok, tiny)
- `struct-lit-field-recv_3.mind` (~25 tok, tiny)
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

- `self_host_loop_advance_contract_test.py` (~7048 tok, huge) — Focused admission, mode, oracle, and receipt controls for --advance."""
- `self_host_loop_advance_publication_test.py` (~7808 tok, huge) — Focused staging, publication, failure, and dispatch controls for --advance."""
- `_self_host_loop_advance_support.py` (~1910 tok, huge) — Shared fixtures for the self-host loop advancement controls.
- `self_host_loop_advance_test.py` (~3484 tok, huge) — Runner for the self-host LOOP advancement harness controls.
- `set_surface_run.rs` (~886 tok, large) — Copyright 2025 STARGA Inc.
- `sha256_allocation_kat.mind` (~409 tok, medium) — Author: STARGA Inc.
- `sha256_legacy_kat.mind` (~605 tok, large) — Author: STARGA Inc.
### `tests/sha256_lifetime/`

- `assemble.sh` (~888 tok, large) — Author: STARGA Inc. Source assembly only; executable assertions are MIND.
- `cases.json` (~4371 tok, huge) — Keys: cases, real_count, observer_count, mutations, builds
- `driver.mind` (~1555 tok, huge) — Author: STARGA Inc. All digest, ownership and refusal verdicts execute in MIND.
### `tests/sha256_lifetime/mutants/`

- `M01.patch` (~30 tok, tiny) — --- sha256.mind
- `M02.patch` (~31 tok, tiny) — --- sha256.mind
- `M03.patch` (~31 tok, tiny) — --- sha256.mind
- `M04.patch` (~31 tok, tiny) — --- sha256.mind
- `M05.patch` (~31 tok, tiny) — --- sha256.mind
- `M06.patch` (~32 tok, tiny) — --- sha256.mind
- `M07.patch` (~22 tok, tiny) — --- sha256.mind
- `M08.patch` (~33 tok, tiny) — --- sha256.mind
### `tests/sha256_lifetime/`

- `observer.mind` (~569 tok, large) — Author: STARGA Inc. Test-only allocation observer; never a product allocator.
- `raw_driver.mind` (~386 tok, medium) — Author: STARGA Inc. Additive raw-only diagnostic; does not replace full suite.
- `raw_observed_driver.mind` (~507 tok, large) — Author: STARGA Inc. Additive raw allocation-observer diagnostic; does not replace full suite.
- `README.md` (~1379 tok, large) — SHA-256 lifetime and refusal controls
- `vectors.mind` (~1812 tok, huge) — Author: STARGA Inc. Frozen known answers; preparation is not subject execution.
### `tests/`

- `sha256_memory_kat.mind` (~2659 tok, huge) — Author: STARGA Inc.
- `sha256_repeated_kat.mind` (~300 tok, medium) — Author: STARGA Inc.
- `sha256_smoke.rs` (~1586 tok, huge) — Copyright 2025 STARGA Inc.
- `sha512_smoke.rs` (~1721 tok, huge) — Copyright 2025 STARGA Inc.
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
### `tests/skip_shape_scan/`

- `controls.rs` (~4228 tok, huge) — Copyright 2025 STARGA Inc.
- `mod.rs` (~6173 tok, huge) — Copyright 2025 STARGA Inc.
### `tests/`

- `slice_argument_admission_run.rs` (~5519 tok, huge) — Copyright 2026 STARGA Inc.
- `slice_call_abi_run.rs` (~6870 tok, huge) — Copyright 2025 STARGA Inc.
- `slice_literal_element_gaps_run.rs` (~3216 tok, huge) — Copyright 2026 STARGA Inc.
- `smoke.rs` (~261 tok, medium) — Copyright 2025 STARGA Inc.
- `source_confinement.rs` (~4985 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `sparse_tensor_types.rs` (~1997 tok, huge) — Copyright 2025 STARGA Inc.
- `statement_mutation_run.rs` (~911 tok, large) — Copyright 2025 STARGA Inc.
- `std_bundled_module_table.rs` (~8119 tok, huge) — Copyright 2026 STARGA Inc.
- `std_cpu_affinity.rs` (~5408 tok, huge) — Copyright 2026 STARGA Inc.
- `std_cpu_affinity_split.rs` (~3034 tok, huge) — Copyright 2026 STARGA Inc.
- `std_dns_free_run.rs` (~1732 tok, huge) — Copyright 2026 STARGA Inc.
- `std_dns_layout_run.rs` (~2220 tok, huge) — Copyright 2026 STARGA Inc.
- `std_entropy_dns_run.rs` (~5013 tok, huge) — Copyright 2026 STARGA Inc.
### `tests/std_entropy_dns_support/`

- `mod.rs` (~1599 tok, huge) — Copyright 2026 STARGA Inc.
### `tests/std_evidence_kat/`

- `main.mind` (~2518 tok, huge) — Known-answer checks for std.evidence and std.evidence_mic (the load-side
- `Mind.toml` (~35 tok, tiny) — [package]
### `tests/std_evidence_load/`

- `main.mind` (~1870 tok, huge) — Load-side checks for std.evidence_mic: evd_load refuses a changed, added or
- `Mind.toml` (~35 tok, tiny) — [package]
### `tests/std_evidence_wrappers/`

- `main.mind` (~3385 tok, huge) — Checks for the `_ev` std variants (std.os_evidence, std.x509_evidence) and the
- `Mind.toml` (~36 tok, tiny) — [package]
### `tests/`

- `std_fs_append_run.rs` (~879 tok, large) — Copyright 2026 STARGA Inc.
- `std_https_deadline.rs` (~643 tok, large) — Copyright 2026 STARGA Inc.
- `std_https_loopback.rs` (~1559 tok, huge) — Copyright 2026 STARGA Inc.
- `std_https_network_safe.rs` (~5337 tok, huge) — Copyright 2026 STARGA Inc.
- `std_https_openssl.rs` (~5427 tok, huge) — Copyright 2026 STARGA Inc.
- `std_import_admission_parity.rs` (~3299 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `std_import_position.rs` (~2687 tok, huge) — Copyright 2026 STARGA Inc.
- `std_import_standalone_run.rs` (~1102 tok, large) — Copyright 2025 STARGA Inc.
- `std_json_arc_payloads.rs` (~5995 tok, huge) — Copyright 2025 STARGA Inc.
- `stdlib_tensor.rs` (~256 tok, medium) — Copyright 2025 STARGA Inc.
- `std_llvm_bindings_smoke.rs` (~2673 tok, huge) — Copyright 2025 STARGA Inc.
- `std_mlir_bindings_smoke.rs` (~4872 tok, huge) — Copyright 2025 STARGA Inc.
### `tests/std_process_wait/`

- `accept.py` (~707 tok, large) — # Author: STARGA Inc.
- `cases_spawn.py` (~2574 tok, huge) — # Author: STARGA Inc.
- `cases_status.py` (~2392 tok, huge) — # Author: STARGA Inc.
- `_harness.py` (~1026 tok, large) — # Author: STARGA Inc.
- `run_cases.sh` (~595 tok, large) — Author: STARGA Inc.
### `tests/`

- `std_process_wait_surface.rs` (~545 tok, large) — Copyright 2026 STARGA Inc.
- `std_procgroup_run.mind` (~3771 tok, huge) — Program for tests/std_procgroup_run.rs: one executable that is both the
- `std_procgroup_run.rs` (~7393 tok, huge) — Copyright 2026 STARGA Inc.
- `std_substrate_native_link.rs` (~8348 tok, huge) — Copyright 2026 STARGA Inc.
- `std_surface_arena.rs` (~1351 tok, large) — Copyright 2025 STARGA Inc.
- `std_surface_array_literals.rs` (~3383 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_async.rs` (~4357 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_bitwise_binops.rs` (~2409 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_bool_return.rs` (~1279 tok, large) — Copyright 2026 STARGA Inc.
- `std_surface_break_continue.rs` (~1335 tok, large) — Copyright 2026 STARGA Inc.
- `std_surface_call_lowering.rs` (~1236 tok, large) — Copyright 2025 STARGA Inc.
- `std_surface_cdylib_link.rs` (~3183 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_cli_equals_form.rs` (~874 tok, large) — Copyright 2025 STARGA Inc.
- `std_surface_cli.rs` (~1007 tok, large) — Copyright 2025 STARGA Inc.
- `std_surface_cli_subcommand.rs` (~799 tok, large) — Copyright 2025 STARGA Inc.
- `std_surface_field_access.rs` (~3202 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_field_access_step2.rs` (~3893 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_fndef_lowering.rs` (~1522 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_fs_mind_open.rs` (~1868 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_http.rs` (~4060 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_i32_intrinsics.rs` (~824 tok, large) — Copyright 2025 STARGA Inc.
- `std_surface_if_statement.rs` (~3458 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_intrinsics.rs` (~3450 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_io_ansi.rs` (~793 tok, large) — Copyright 2025 STARGA Inc.
- `std_surface_io_canon.rs` (~4918 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_io_module.rs` (~1544 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_iouring.rs` (~6829 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_json_canon.rs` (~4864 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_json.rs` (~7071 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_logical_ops.rs` (~1207 tok, large) — Copyright 2026 STARGA Inc.
- `std_surface_map_module.rs` (~2020 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_method_call.rs` (~2963 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_mic_record_refusals.rs` (~3524 tok, huge) — Copyright 2026 STARGA Inc.
- `std_surface_mic_record.rs` (~7195 tok, huge) — Copyright 2026 STARGA Inc.
- `std_surface_net_fs_process.rs` (~8355 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_phase_c_stdlib_bundle.rs` (~1466 tok, large) — Copyright 2025 STARGA Inc.
- `std_surface_phase_d_env_override.rs` (~2354 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_promotion_compose.rs` (~5656 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_reactor.rs` (~1378 tok, large) — Copyright 2025 STARGA Inc.
- `std_surface_regex.rs` (~5298 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_ring.rs` (~1318 tok, large) — Copyright 2025 STARGA Inc.
- `std_surface_self_emit_shared.rs` (~1808 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_string_itoa.rs` (~934 tok, large) — Copyright 2025 STARGA Inc.
- `std_surface_string_module.rs` (~2288 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_string_push_str.rs` (~864 tok, large) — Copyright 2025 STARGA Inc.
- `std_surface_struct_lowering.rs` (~2772 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_toml.rs` (~4173 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_tui.rs` (~2261 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_use_import_phase_b.rs` (~2303 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_use_import.rs` (~1757 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_vec_module.rs` (~1809 tok, huge) — Copyright 2025 STARGA Inc.
- `std_surface_vec_zeroed.rs` (~1097 tok, large) — Copyright 2025 STARGA Inc.
- `std_surface_while_statement.rs` (~3251 tok, huge) — Copyright 2025 STARGA Inc.
- `std_tls_x509_import.rs` (~8225 tok, huge) — Copyright 2026 STARGA Inc.
- `stmt_keyword_recognizer.rs` (~2527 tok, huge) — Copyright 2025 STARGA Inc.
- `stride_gather_grad.rs` (~312 tok, medium) — Copyright 2025 STARGA Inc.
- `stride_preview.rs` (~279 tok, medium) — Copyright 2025 STARGA Inc.
- `stride_types.rs` (~244 tok, medium) — Copyright 2025 STARGA Inc.
- `string_edges_backend_parity.rs` (~2621 tok, huge) — Copyright 2026 STARGA Inc.
- `string_eq_len_run.rs` (~2613 tok, huge) — Copyright 2025 STARGA Inc.
- `string_escape_decode_run.rs` (~1159 tok, large) — Copyright 2025 STARGA Inc.
- `string_escape_fail_closed.rs` (~959 tok, large) — Copyright 2025 STARGA Inc.
- `string_escape_parse.rs` (~483 tok, medium) — Copyright 2025 STARGA Inc.
- `string_flows_backend_parity.rs` (~6856 tok, huge) — Copyright 2026 STARGA Inc.
- `string_from_bytes_run.rs` (~664 tok, large) — Copyright 2025 STARGA Inc.
- `string_ops_backend_parity.rs` (~5537 tok, huge) — Copyright 2026 STARGA Inc.
- `string_ops_native_r5.rs` (~3929 tok, huge) — Copyright 2026 STARGA Inc.
- `string_pattern_escape_decode.rs` (~1069 tok, large) — Copyright 2025 STARGA Inc. Licensed under the Apache License, Version 2.0.
- `string_runtime_shim_run.rs` (~1027 tok, large) — Copyright 2025 STARGA Inc.
- `string_slots_backend_parity.rs` (~3295 tok, huge) — Copyright 2026 STARGA Inc.
- `string_split_run.rs` (~1826 tok, huge) — Copyright 2025 STARGA Inc.
- `struct_array_field_run.rs` (~802 tok, large) — Copyright 2025 STARGA Inc.
- `struct_field_collection_run.rs` (~808 tok, large) — Copyright 2025 STARGA Inc.
- `struct_field_in_loop_run.rs` (~1043 tok, large) — Copyright 2025 STARGA Inc.
- `struct_literal_rebind.rs` (~1607 tok, huge) — Copyright 2026 STARGA Inc.
- `struct_narrow_field.rs` (~771 tok, large) — Copyright 2025 STARGA Inc.
- `substrate_nonentry_import_link.rs` (~1455 tok, large) — Copyright 2025 STARGA Inc.
### `tests/support/`

- `file_read_directory.rs` (~2522 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `fmt_idempotence_bare_lists.rs` (~8004 tok, huge) — Copyright 2026 STARGA Inc.
- `g2_capability_ratchet.rs` (~883 tok, large) — Copyright 2026 STARGA Inc.
- `g2_compile_result.rs` (~364 tok, medium) — Copyright 2026 STARGA Inc.
- `g2_policy_rejections.rs` (~755 tok, large) — Copyright 2026 STARGA Inc.
- `https_loopback_harness.rs` (~2025 tok, huge) — Copyright 2026 STARGA Inc.
- `https_openssl_harness.rs` (~5141 tok, huge) — Copyright 2026 STARGA Inc.
- `mindc_test_bare_imports.rs` (~3717 tok, huge) — Copyright 2026 STARGA Inc.
- `pqc_hybrid_cli_support.rs` (~1429 tok, large) — Copyright 2026 STARGA Inc.
- `tls13_leaf_key_program.rs` (~4103 tok, huge) — Copyright 2026 STARGA Inc.
- `v04_mirror_body_vectors.rs` (~4124 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `v04_mirror_descriptor_vectors.rs` (~2649 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `v04_mirror_intrinsic_vectors.rs` (~1359 tok, large) — Copyright 2025-2026 STARGA Inc.
- `v04_mirror_modules.rs` (~1936 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `v04_mirror_oracle.rs` (~6008 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `x509_literals_program.rs` (~943 tok, large) — Copyright 2026 STARGA Inc.
### `tests/`

- `target_cerebras.rs` (~340 tok, medium) — Cerebras backend target — first-class surface tests.
- `tensor_broadcast.rs` (~995 tok, large) — Copyright 2025 STARGA Inc.
- `tensor_buffers.rs` (~517 tok, large) — Copyright 2025 STARGA Inc.
- `tensor_eval.rs` (~457 tok, medium) — Copyright 2025 STARGA Inc.
- `tensor_param_2d_run.rs` (~954 tok, large) — Copyright 2025 STARGA Inc.
- `tensor_param_descriptor_abi_run.rs` (~3280 tok, huge) — Copyright 2025-2026 STARGA Inc.
- `tensor_param_fail_loud_run.rs` (~3296 tok, huge) — Copyright 2025 STARGA Inc.
- `tensor_stdlib.rs` (~549 tok, large) — Copyright 2025 STARGA Inc.
- `tensor_symbolic.rs` (~542 tok, large) — Copyright 2025 STARGA Inc.
- `test_harness_root_cause_error.rs` (~758 tok, large) — Copyright 2025 STARGA Inc.
- `tls13_client_cases.mind` (~5652 tok, huge) — Pure-MIND regression entry point; identical assertions on both backends.
- `_tls13_client_modes.py` (~4922 tok, huge) — # Copyright 2025 STARGA Inc.
- `tls13_client_runner.mind` (~449 tok, medium) — One authoritative MIND entry point for both backend transports.
- `tls13_client_rx.mind` (~1349 tok, large) — Application receive buffer of the TLS 1.3 client: it holds one record
- `tls13_client_shipped.mind` (~1114 tok, large) — Shipped-verifier refusal: the real std/x509 verifier, an anchor set that does
- `tls13_finished_driver.py` (~3085 tok, huge) — # Official-vector driver for std/tls13_finished.mind (pure-MIND TLS 1.3
- `tls13_handshake_driver.py` (~6913 tok, huge) — # Official-vector driver for std/tls13_handshake.mind (pure-MIND TLS 1.3
- `tls13_keyschedule_driver.py` (~2863 tok, huge) — # Official-vector driver for std/tls13_keyschedule.mind (pure-MIND TLS 1.3 key
- `tls13_record_driver.py` (~3051 tok, huge) — # Official-vector driver for std/tls13_record.mind (pure-MIND TLS 1.3 record
- `trait_static_dispatch_run.rs` (~1081 tok, large) — Copyright 2025 STARGA Inc.
- `transpose_preview.rs` (~269 tok, medium) — Copyright 2025 STARGA Inc.
- `try_operator_run.rs` (~1058 tok, large) — Copyright 2025 STARGA Inc.
- `tuple_destructure_run.rs` (~1112 tok, large) — Copyright 2025 STARGA Inc.
- `turboquant_kernel_run.rs` (~1326 tok, large) — Copyright 2025 STARGA Inc.
- `type_alias_narrow_lowering_run.rs` (~2255 tok, huge) — Copyright 2025 STARGA Inc.
- `type_ann_check.rs` (~330 tok, medium) — Copyright 2025 STARGA Inc.
- `type_ann_parse.rs` (~583 tok, large) — Copyright 2025 STARGA Inc.
- `typecheck_binary.rs` (~421 tok, medium) — Copyright 2025 STARGA Inc.
- `typecheck_env.rs` (~246 tok, medium) — Copyright 2025 STARGA Inc.
### `tests/type_checker/`

- `basic_type_inference.mind` (~66 tok, small) — Type checker test: Basic type inference
- `dtype_mismatch.mind` (~74 tok, small) — Type checker test: Dtype mismatch detection
### `tests/`

- `typed_literal_match_pattern_run.rs` (~719 tok, large) — Copyright 2025 STARGA Inc. Licensed under the Apache License, Version 2.0.
- `type_error_spans.rs` (~1058 tok, large) — Copyright 2025 STARGA Inc.
- `type_infer.rs` (~344 tok, medium) — Copyright 2025 STARGA Inc.
- `type_struct_run.rs` (~582 tok, large) — Copyright 2025 STARGA Inc.
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
- `verify_require_signed.rs` (~1968 tok, huge) — Copyright 2025 STARGA Inc.
- `verify_scope.rs` (~5566 tok, huge) — Copyright 2026 STARGA Inc.
- `verify_ssa.rs` (~7389 tok, huge) — Copyright 2025 STARGA Inc.
- `vnni_resolved_toolchain.rs` (~2605 tok, huge) — Copyright 2026 STARGA Inc.
- `wide_integer_types.rs` (~517 tok, large) — Copyright 2026 STARGA Inc.
- `x25519mlkem768_driver.py` (~2378 tok, huge) — # Known-answer driver for std/x25519mlkem768.mind (pure-MIND X25519MLKEM768
- `x25519_vectors_driver.py` (~2450 tok, huge) — # Official-vector driver for std/x25519.mind (pure-MIND Curve25519 ECDH).
- `x509_chain_smoke.rs` (~6314 tok, huge) — Copyright 2025-2026 STARGA Inc.
### `tests/x509/`

- `native_body.mind` (~547 tok, large) — native_body.mind — statements of main() for the native harness, appended by
- `native_head.mind` (~501 tok, large) — tests/x509/native_main_head.mind + native_main_body.mind — NATIVE-backend
- `native_main_body.mind` (~348 tok, medium) — Body of the native main() — appended inside a `fn main() -> i64 { ... }`
### `tests/`

- `x509_pss_salt.rs` (~784 tok, large) — Copyright 2025-2026 STARGA Inc.
### `tests/x509/`

- `README.md` (~3011 tok, huge) — X.509 chain and trust-anchor tests
### `tests/x509/testdata/adv/`

- `a_inter_pss384_root.der` (binary, 524 bytes)
- `a_inter_rsa384_root.der` (binary, 524 bytes)
- `a_leaf_pss384_root.der` (binary, 380 bytes)
- `a_leaf_rsa384_root.der` (binary, 380 bytes)
- `a_root_pss384.der` (binary, 828 bytes)
- `a_root_rsa384.der` (binary, 724 bytes)
- `bundle_ec_plus_sha1.pem` (~450 tok, medium) — -----BEGIN CERTIFICATE-----
- `bundle_sha1_plus_ec.der` (binary, 1246 bytes)
- `pss3072_long_leaf.der` (binary, 1076 bytes)
- `pss3072_long_leaf_salt351.der` (binary, 1076 bytes)
- `pss3072_long_root.der` (binary, 1061 bytes)
- `pss4096_inter2048.der` (binary, 1086 bytes)
- `pss4096_leaf.der` (binary, 665 bytes)
- `pss4096_long_leaf.der` (binary, 1204 bytes)
- `pss4096_long_root.der` (binary, 1317 bytes)
- `pss4096_root.der` (binary, 1338 bytes)
- `rsa3072_leaf.der` (binary, 688 bytes)
- `rsa3072_root.der` (binary, 978 bytes)
- `rsa4096_inter2048.der` (binary, 982 bytes)
- `rsa4096_leaf.der` (binary, 561 bytes)
- `rsa4096_root.der` (binary, 1234 bytes)
- `sha1root_sha256_leaf.der` (binary, 569 bytes)
- `sha1_selfsigned_root.der` (binary, 821 bytes)
- `xs_inter.der` (binary, 319 bytes)
- `xs_leaf.der` (binary, 362 bytes)
- `xs_rootA.der` (binary, 316 bytes)
### `tests/x509/testdata/`

- `bundle_anchors.der` (binary, 4929 bytes)
- `bundle_anchors.pem` (~1780 tok, huge) — -----BEGIN CERTIFICATE-----
- `bundle_bad.pem` (~171 tok, small) — -----BEGIN CERTIFICATE-----
- `der_negative_ecdsa.der` (binary, 491 bytes)
- `der_negative_ecdsa_host.txt` (~5 tok, tiny) — www.wild.example.com
- `der_sig_bits.der` (binary, 455 bytes)
- `der_trailing.der` (binary, 459 bytes)
- `dns_ip_leaf.der` (binary, 441 bytes)
- `ec_inter.der` (binary, 422 bytes)
- `ec_leaf_badku.der` (binary, 461 bytes)
- `ec_leaf_badsig.der` (binary, 455 bytes)
- `ec_leaf_crit.der` (binary, 475 bytes)
- `ec_leaf.der` (binary, 455 bytes)
- `ec_leaf_ip.der` (binary, 449 bytes)
- `ec_leaf_midwild.der` (binary, 465 bytes)
- `ec_leaf_noeku.der` (binary, 435 bytes)
- `ec_leaf_nosan.der` (binary, 433 bytes)
- `ec_leaf_trunc.der` (binary, 100 bytes)
- `ec_leaf_wild.der` (binary, 492 bytes)
- `ec_root.der` (binary, 425 bytes)
- `eku_client_leaf.der` (binary, 449 bytes)
- `eku_root.der` (binary, 374 bytes)
- `eku_server_leaf.der` (binary, 451 bytes)
- `kp_leaf_badkey.der` (binary, 860 bytes)
- `kp_leaf.der` (binary, 857 bytes)
- `kp_root.der` (binary, 815 bytes)
- `kp_rsa3072_leaf.der` (binary, 993 bytes)
- `nc_inter.der` (binary, 472 bytes)
- `nc_leaf_bad.der` (binary, 460 bytes)
- `nc_leaf_ok.der` (binary, 461 bytes)
- `nc_root.der` (binary, 421 bytes)
- `noca_inter.der` (binary, 415 bytes)
- `noca_leaf.der` (binary, 458 bytes)
- `noca_root.der` (binary, 425 bytes)
- `now.txt` (~3 tok, tiny) — 1792140749
- `p384_leaf.der` (binary, 517 bytes)
- `p384_root.der` (binary, 485 bytes)
- `path_anchor_expired.der` (binary, 341 bytes)
- `path_anchor_wrongkey.der` (binary, 328 bytes)
- `path_badeku.der` (binary, 349 bytes)
- `path_badku.der` (binary, 323 bytes)
- `path_ec_inter.der` (binary, 328 bytes)
- `path_ec_leaf.der` (binary, 376 bytes)
- `path_ec_root.der` (binary, 328 bytes)
- `path_exp_inter.der` (binary, 335 bytes)
- `path_expired.der` (binary, 325 bytes)
- `path_exp_leaf.der` (binary, 378 bytes)
- `path_future.der` (binary, 324 bytes)
- `path_ip_bad.der` (binary, 358 bytes)
- `path_ip_good.der` (binary, 360 bytes)
- `path_ip_outside.der` (binary, 361 bytes)
- `path_ipv6.der` (binary, 368 bytes)
- `path_leaf_expired.der` (binary, 377 bytes)
- `path_m01_mid_star_dot.der` (binary, 371 bytes)
- `path_m01_trailing_star.der` (binary, 376 bytes)
- `path_nc_inter.der` (binary, 408 bytes)
- `path_outside.der` (binary, 368 bytes)
- `path_p384_inter.der` (binary, 395 bytes)
- `path_p384_leaf.der` (binary, 441 bytes)
- `path_p384_root.der` (binary, 393 bytes)
- `path_pss_inter.der` (binary, 831 bytes)
- `path_pss_leaf.der` (binary, 878 bytes)
- `path_pss_root.der` (binary, 830 bytes)
- `path_root_len0.der` (binary, 327 bytes)
- `path_rsa_inter.der` (binary, 727 bytes)
- `path_rsa_leaf.der` (binary, 774 bytes)
- `path_rsa_root.der` (binary, 726 bytes)
- `path_wild_double_star.der` (binary, 371 bytes)
- `path_wild_idn_mid_star_dot.der` (binary, 381 bytes)
- `path_wild_idn_star.der` (binary, 378 bytes)
- `path_wild_partial.der` (binary, 375 bytes)
- `path_wild_star_alone.der` (binary, 363 bytes)
- `path_wrongkey.der` (binary, 325 bytes)
- `pl_i1.der` (binary, 421 bytes)
- `pl_i2.der` (binary, 424 bytes)
- `pl_leaf.der` (binary, 455 bytes)
- `pl_root.der` (binary, 424 bytes)
- `pss_bad_trailer.der` (binary, 888 bytes)
- `pss_default_hash.der` (binary, 844 bytes)
- `pss_default_mgf.der` (binary, 818 bytes)
- `pss_leaf.der` (binary, 954 bytes)
- `pss_root.der` (binary, 923 bytes)
- `rsa_leaf.der` (binary, 855 bytes)
- `rsa_root.der` (binary, 819 bytes)
- `sha1_leaf.der` (binary, 853 bytes)
- `sha1_root.der` (binary, 821 bytes)
### `tests/`

- `x509_vectors_driver.py` (~7542 tok, huge) — # Real-certificate driver for std/x509.mind (pure-MIND X.509 DER parsing + RSA
- `_x509_vectors_modes.py` (~7366 tok, huge) — # Copyright 2025 STARGA Inc.
### `tests/xsi_quant/`

- `gates.rs` (~4100 tok, huge) — Copyright 2025 STARGA Inc.
- `mod.rs` (~4381 tok, huge) — Copyright 2025 STARGA Inc.
- `oracle.rs` (~3441 tok, huge) — Copyright 2025 STARGA Inc.
### `tools/`

- `add_copyright_headers.py` (~1132 tok, large) — # Copyright 2025 STARGA Inc.
- `bench_gate.py` (~8618 tok, huge) — # Copyright 2025 STARGA Inc.
- `bench_pairs.py` (~8864 tok, huge) — # Copyright 2025 STARGA Inc.
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

- `run_bench_gate.sh` (~535 tok, large) — Copyright 2025 STARGA Inc. Licensed under the Apache License, Version 2.0.
### `tools/seed_ledger/`

- `Mind.toml` (~57 tok, small) — [package]
- `seed_ledger_bytes.mind` (~1037 tok, large) — tools/seed_ledger/seed_ledger_bytes.mind — byte slices, output and sha256 digests.
- `seed_ledger_check.mind` (~1293 tok, large) — tools/seed_ledger/seed_ledger_check.mind — the check: drift, unlisted, stale, hand edits.
- `seed_ledger.mind` (~1287 tok, large) — tools/seed_ledger/seed_ledger.mind — the seed drift ledger guard (entry).
- `seed_ledger_mint.mind` (~1709 tok, huge) — tools/seed_ledger/seed_ledger_mint.mind — the ledger is bound to the minted seed.
- `seed_ledger_rows.mind` (~1735 tok, huge) — tools/seed_ledger/seed_ledger_rows.mind — ledger rows, G2 rows and the seed source list.
### `tools/`

- `test_bench_gate.py` (~11013 tok, huge) — Regression test for the bench-gate fail-closed contract (tools/bench_gate.py).

---
*Generated by `anatomy 1.0.0`. Edit descriptions manually — re-run preserves structure.*
