#!/usr/bin/env bash
# Copyright 2025 STARGA Inc. Licensed under the Apache License, Version 2.0.
#
# Canonical criterion bench-gate RUNNER. Criterion is load-sensitive in two
# ways, and this guards both:
#   1. BURSTY noise -> a large per-bench `(+/- N)` spread. `bench_gate.py` now
#      flags those as NOISY (inconclusive) instead of FAILing.
#   2. SUSTAINED load -> the whole sample distribution shifts up UNIFORMLY
#      (small internal spread, high absolute median), which the variance check
#      cannot see. The only defence is to refuse to measure on a busy box.
#      That is what this runner adds: a pre-bench load guard + CPU pinning.
#
# Usage:
#   tools/run_bench_gate.sh [baseline_file] [--wait]
#     --wait   block until the box is idle (load < BENCH_MAX_LOAD) instead of
#              aborting; useful locally after kicking off other work.
# Env:
#   BENCH_MAX_LOAD  (default 2.5)  max 1-min loadavg to allow a measurement.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
MAX_LOAD="${BENCH_MAX_LOAD:-2.5}"
BASELINE="${1:-.bench-baseline-2026-06-01-correctness.txt}"
WAIT=0
for a in "$@"; do [ "$a" = "--wait" ] && WAIT=1; done

load1() { cut -d' ' -f1 /proc/loadavg; }
loaded() { awk -v l="$(load1)" -v m="$MAX_LOAD" 'BEGIN{exit !(l>m)}'; }

if loaded; then
  if [ "$WAIT" = 1 ]; then
    echo "[bench] box loaded ($(load1) > $MAX_LOAD) — waiting for idle..."
    while loaded; do sleep 30; done
    sleep 10
  else
    echo "::error::box load $(load1) > $MAX_LOAD — criterion needs an idle host."
    echo "::error::re-run when idle, or pass --wait to block until idle."
    exit 3
  fi
fi

# Pin to the first up-to-4 cores for a stable measurement.
NP="$(nproc)"; HI=$(( NP > 4 ? 3 : NP - 1 )); PIN="0-${HI}"
echo "[bench] load $(load1), pinned to cores ${PIN}, baseline ${BASELINE}"

OUT="$(mktemp)"
taskset -c "${PIN}" cargo bench --bench compiler --no-default-features -- \
  --warm-up-time 3 --measurement-time 8 --output-format bencher | tee "${OUT}"
python3 tools/bench_gate.py --baseline "${BASELINE}" --current "${OUT}" --threshold 0.10
rc=$?
rm -f "${OUT}"
exit "${rc}"
