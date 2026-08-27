#!/usr/bin/env bash
# preflight.sh — local CI-parity gate. Run before pushing to avoid red CI.
#
# Mirrors the jobs that actually gate main: rustfmt (ci.yml Format Check),
# build+test, mindc check over std/+examples/ (ci.yml mindcraft_check, which
# also flags `mindc fmt` drift), and — with --full — the keystone byte-identity
# gate and the frozen-frontend bench gate (bench-gate.yml, --no-default-features).
#
#   scripts/preflight.sh          # FAST (~seconds-min): fmt + build + mindc check
#   scripts/preflight.sh --full   # + keystone 7/7 + cross-substrate determinism
#                                 #   + self-host LOOP + mic@3 FLIP + bench gate
#
# GATE-EVIDENCE RULE: never accept `exit 0` as proof a gate ran. Every cargo-test
# gate below asserts a POSITIVE test count. A correct `--test <target>` selector is
# NOT sufficient — a test file that is `#![cfg]`d out by a missing feature compiles
# to nothing and cargo still reports `ok. 0 passed` with exit 0 (measured
# 2026-08-27 on cross_substrate_identity with `--features mlir-build` alone, which
# leaves the file's required `cross-module-imports` off).
#
# Exits non-zero if any gate fails; prints exactly what to run to fix it.
set -uo pipefail
cd "$(dirname "$0")/.."

fail=0
step() { printf '\n\033[1m== %s ==\033[0m\n' "$1"; }
bad()  { printf '\033[31mFAIL:\033[0m %s\n' "$1"; fail=1; }

step "rustfmt  [ci.yml Format Check]"
if cargo fmt --check >/dev/null 2>&1; then echo "ok"; else
  bad "rustfmt drift — run: cargo fmt"
  cargo fmt --check 2>&1 | grep "Diff in" | head
fi

step "build (default features = std-surface)  [ci.yml Build & Test]"
if cargo build --quiet 2>/tmp/preflight-build.err; then echo "ok"; else
  bad "build broken"; tail -15 /tmp/preflight-build.err
fi

step "mindc check std/ examples/  [ci.yml mindcraft_check — error-severity incl. fmt::drift]"
if cargo build --release --no-default-features --features "std-surface cross-module-imports" --bin mindc --quiet 2>/dev/null; then
  errs=$(./target/release/mindc check std/ examples/ 2>&1 | grep -E ': error:' || true)
  if [ -z "$errs" ]; then echo "ok (warnings allowed)"; else
    bad "mindc check error-severity diagnostics (fmt::drift → mindc fmt <file>; tuple-return → #[repr(C)] struct):"
    printf '%s\n' "$errs" | head
  fi
else
  bad "could not build mindc"
fi

step "no-features compile parity  [ci.yml 'Check compiles (no features)' + gated feature-combo steps]"
# The keystone/build/mindc gates above ALL use std-surface, so a change that
# compiles there but not under --no-default-features (a mis-placed #[cfg], a
# std-surface-only variant referenced from a non-std-surface path) rides straight
# onto red main. Mirror every feature combo ci.yml compiles.
nf_ok=1
for feats in "" "std-surface" "cross-module-imports" "std-surface,cross-module-imports" \
             "autodiff" "mlir-lowering" "cpu-buffers" "cpu-exec" "pkg"; do
  if [ -z "$feats" ]; then flag=(--no-default-features); label="--no-default-features";
  else flag=(--no-default-features --features "$feats"); label="--no-default-features --features $feats"; fi
  if ! cargo check "${flag[@]}" --quiet 2>/tmp/preflight-nf.err; then
    bad "cargo check $label BROKEN — this is the exact class that slips past the std-surface gates:"
    grep -E '^error' /tmp/preflight-nf.err | head -4
    nf_ok=0
  fi
done
[ "$nf_ok" = 1 ] && echo "ok (all CI feature combos compile)"

if [ "${1:-}" = "--full" ]; then
  step "no-features test parity  [ci.yml Build & Test 'Test' steps — the fail-close regression class]"
  # A test that feeds lower_to_ir free operands / hits a gated path lowered to a
  # silent const-0 pre-#9 now panics loud — and only shows under the non-std-surface
  # test steps the keystone never runs. The `error: test failed` grep is abort-aware
  # (a stack-overflow SIGABRT prints NO `test result: FAILED`, only the cargo line).
  # Two accepted CI-ABSENT local-only failures are excluded: mindfuzz_cross_substrate
  # (needs the MLIR toolchain; soft-skips in CI's build_test job) and g2_differential_mlir
  # (dlopens the gitignored, stale in-tree libmindc_mind.so; CI's fresh checkout rebuilds).
  nft_out=$(cargo test --no-default-features --features std-surface,cross-module-imports \
              --no-fail-fast 2>&1 | grep -E "error: test failed|has overflowed its stack" | \
              grep -viE "mindfuzz_cross_substrate|g2_differential_mlir" || true)
  if [ -z "$nft_out" ]; then echo "ok (no test failures beyond the accepted CI-absent mindfuzz + g2)"; else
    bad "cargo test --no-default-features --features std-surface,cross-module-imports FAILS beyond mindfuzz:"; printf '%s\n' "$nft_out" | head
  fi

  step "keystone byte-identity 7/7  [ci.yml + cross-substrate — the wedge invariant]"
  ks_out=$(MIND_BENCH_REQUIRE=1 cargo test --release \
       --features "mlir-build std-surface cross-module-imports" \
       --test phase_g_keystone_bootstrap -- --test-threads=1 2>&1 || true)
  ks_n=$(printf '%s\n' "$ks_out" | sed -n 's/^test result: ok\. \([0-9]*\) passed.*/\1/p' | tail -1)
  if [ "${ks_n:-0}" -ge 7 ] 2>/dev/null; then
    echo "ok ($ks_n/7 byte-identical)"
  else
    bad "keystone NOT 7/7 (passed='${ks_n:-none}') — 0 passed means the target was cfg'd out or SKIPped, which is NOT a pass; a cross-substrate byte-identity regression; do NOT push"
  fi

  step "cross-substrate determinism 24/24  [ci.yml cross_substrate_identity — THE wedge invariant]"
  # Was ABSENT from preflight until 2026-08-27 while preflight still printed
  # "safe to push". Features must match ci.yml:286 EXACTLY — tests/cross_substrate_identity.rs
  # carries a file-level #![cfg(all(feature="mlir-build", feature="std-surface",
  # feature="cross-module-imports"))]; drop any one and the whole file vanishes and
  # cargo reports `ok. 0 passed` with exit 0. Hence the POSITIVE-count assert.
  xs_out=$(MIND_BENCH_REQUIRE=1 cargo test --no-default-features \
             --features "mlir-build std-surface cross-module-imports" \
             --test cross_substrate_identity -- --nocapture 2>&1 || true)
  xs_n=$(printf '%s\n' "$xs_out" | sed -n 's/^test result: ok\. \([0-9]*\) passed.*/\1/p' | tail -1)
  if [ -n "$xs_n" ] && [ "$xs_n" -ge 24 ]; then
    echo "ok ($xs_n/24+ reproducibility gates byte-identical)"
  else
    bad "cross-substrate determinism gate did NOT prove itself (passed='${xs_n:-none}', need >=24) — 0 tests means the file was cfg'd out, NOT a pass; do NOT push"
  fi

  step "self-host LOOP byte-identity  [examples/mindc_mind/self_host_loop_smoke.py]"
  # Banked lesson (reference_preflight_missing_loop_gate_reseed): a main.mind/std edit
  # can pass the local keystone 7/7 and still RED CI on a stale frozen seed. The loop
  # gate is the one that catches it. Any main.mind/std edit needs --reseed in the SAME change.
  if [ -f examples/mindc_mind/self_host_loop_smoke.py ]; then
    if lp_out=$(python3 examples/mindc_mind/self_host_loop_smoke.py 2>&1); then
      echo "ok (stage1==stage2==stage3 byte-identical)"
    else
      bad "self-host LOOP gate FAILED (stale frozen seed? reseed in the SAME change):"; printf '%s\n' "$lp_out" | tail -5
    fi
  else
    bad "self_host_loop_smoke.py MISSING — the loop gate cannot run; do NOT push"
  fi

  step "whole-module mic@3 FLIP  [examples/mindc_mind/mic3_flip_smoke.py]"
  # Banked lesson (reference_mic3_flip_required_local_gate_2026_08_06): REQUIRED for ANY
  # lower.rs / emit / mic@3 change. Keystone cargo-test + oracle-parity do NOT cover the
  # whole-module FLIP — this is the gate that reverted #287-F2 (#223 -> #224).
  if [ -f examples/mindc_mind/mic3_flip_smoke.py ]; then
    if fl_out=$(python3 examples/mindc_mind/mic3_flip_smoke.py 2>&1); then
      echo "ok (whole-module FLIP byte-identical)"
    else
      bad "mic@3 whole-module FLIP gate FAILED — this is the #287-F2 revert class:"; printf '%s\n' "$fl_out" | tail -5
    fi
  else
    bad "mic3_flip_smoke.py MISSING — the FLIP gate cannot run; do NOT push"
  fi

  step "bench gate (frozen low-level frontend)  [bench-gate.yml, --no-default-features]"
  base=$(ls -t .bench-baseline-*correctness*.txt 2>/dev/null | head -1)
  if [ -n "$base" ] && [ -f tools/bench_gate.py ]; then
    cargo bench --bench compiler --no-default-features -- \
      --warm-up-time 3 --measurement-time 8 --output-format bencher > /tmp/preflight-bench.out 2>/dev/null
    if python3 tools/bench_gate.py --baseline "$base" --current /tmp/preflight-bench.out --threshold 0.10; then
      echo "ok (regression <= +10% vs $base; speedups always pass)"
    else
      bad "frozen-frontend bench regression >+10% vs $base — STOP & decide: revert, or re-bless baseline if a dramatic win elsewhere justifies it"
    fi
  else
    echo "skip (no correctness baseline / bench_gate.py)"
  fi
fi

echo
if [ "$fail" = 0 ]; then
  printf '\033[32m✓ preflight PASS — safe to push\033[0m\n'
else
  printf '\033[31m✗ preflight FAIL — fix the above before pushing\033[0m\n'
fi
exit "$fail"
