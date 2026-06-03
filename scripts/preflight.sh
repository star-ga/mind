#!/usr/bin/env bash
# preflight.sh — local CI-parity gate. Run before pushing to avoid red CI.
#
# Mirrors the jobs that actually gate main: rustfmt (ci.yml Format Check),
# build+test, mindc check over std/+examples/ (ci.yml mindcraft_check, which
# also flags `mindc fmt` drift), and — with --full — the keystone byte-identity
# gate and the frozen-frontend bench gate (bench-gate.yml, --no-default-features).
#
#   scripts/preflight.sh          # FAST (~seconds-min): fmt + build + mindc check
#   scripts/preflight.sh --full   # + keystone 7/7 + bench gate (slow, minutes)
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
if cargo build --release --no-default-features --features std-surface --bin mindc --quiet 2>/dev/null; then
  errs=$(./target/release/mindc check std/ examples/ 2>&1 | grep -E ': error:' || true)
  if [ -z "$errs" ]; then echo "ok (warnings allowed)"; else
    bad "mindc check error-severity diagnostics (fmt::drift → mindc fmt <file>; tuple-return → #[repr(C)] struct):"
    printf '%s\n' "$errs" | head
  fi
else
  bad "could not build mindc"
fi

if [ "${1:-}" = "--full" ]; then
  step "keystone byte-identity 7/7  [ci.yml + cross-substrate — the wedge invariant]"
  if MIND_BENCH_REQUIRE=1 cargo test --release \
       --features "mlir-build std-surface cross-module-imports" \
       --test phase_g_keystone_bootstrap -- --test-threads=1 2>&1 | grep -q "7 passed"; then
    echo "ok (7/7 byte-identical)"
  else
    bad "keystone NOT 7/7 — a cross-substrate byte-identity regression; do NOT push"
  fi

  step "bench gate (frozen low-level frontend)  [bench-gate.yml, --no-default-features]"
  base=$(ls -t .bench-baseline-*correctness*.txt 2>/dev/null | head -1)
  if [ -n "$base" ] && [ -f tools/bench_gate.py ]; then
    cargo bench --bench compiler --no-default-features -- \
      --warm-up-time 3 --measurement-time 8 --output-format bencher > /tmp/preflight-bench.out 2>/dev/null
    if python3 tools/bench_gate.py --baseline "$base" --current /tmp/preflight-bench.out --threshold 0.07; then
      echo "ok (<=7% vs $base)"
    else
      bad "frozen-frontend bench regression >7% vs $base"
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
