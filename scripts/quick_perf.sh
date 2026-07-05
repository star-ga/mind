#!/usr/bin/env bash
# quick_perf.sh — FAST one-sided compile-speed criterion gate.
#
# Why this exists: the full `cargo bench` criterion run takes minutes (100 samples +
# warmup per bench), which is enough friction that the compile-speed check gets
# skipped. This runs ONLY the compile_small group with reduced samples so a real
# verdict lands in ~20-40s (once the bench crate is built), removing the excuse.
#
# The gate is ONE-SIDED (standing rule): faster or unchanged = PASS; slower is a
# WORTH-IT decision, not an auto-fail — a correctness fix may justify a few % of
# compile cost, but "mindc compile speed never regresses" is the default, so a real
# slowdown must be investigated + justified, never waved through.
#
# CRITICAL: criterion compile benches are microsecond-scale and HIGHLY sensitive to
# CPU load. Run on a QUIET machine (load avg < ~1, no background builds/agents) or
# the numbers are noise. The script refuses to run above a load threshold.
#
# Usage:
#   scripts/quick_perf.sh                  # gate against criterion's stored baseline
#   SAMPLES=50 scripts/quick_perf.sh       # more samples (slower, tighter CI)
#   FORCE=1 scripts/quick_perf.sh          # run even if load is high (numbers unreliable)
#
# Determinism is a SEPARATE fast gate: examples/mindc_mind/fast_keystone.sh
# (byte-identity + keystone) and `cargo test --features "std-surface mlir-build
# cross-module-imports" --test cross_substrate_identity`. Run BOTH on every
# compiler change: quick_perf.sh (speed) + fast_keystone.sh (determinism).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT" || exit 2

# --- quiet-machine guard (µs benches are load-sensitive) ---
LOAD1="$(awk '{print $1}' /proc/loadavg)"
NPROC="$(nproc 2>/dev/null || echo 4)"
# threshold: half the cores, min 1.5 — above this, criterion numbers are unreliable.
THRESH="$(awk -v n="$NPROC" 'BEGIN{t=n/2; if(t<1.5)t=1.5; print t}')"
if awk -v l="$LOAD1" -v t="$THRESH" 'BEGIN{exit !(l>t)}'; then
  if [ "${FORCE:-0}" != "1" ]; then
    echo "REFUSING: load $LOAD1 > $THRESH — criterion µs benches would be noise."
    echo "  Wait for background builds/agents to finish, or FORCE=1 to override (unreliable)."
    exit 3
  fi
  echo "WARNING: load $LOAD1 > $THRESH but FORCE=1 — treat results as unreliable."
fi

echo "== quick compile-speed gate (compile_small) — load $LOAD1, ${SAMPLES:-25} samples =="
cargo bench --bench simple_benchmarks -- compile_small \
    --sample-size "${SAMPLES:-25}" --warm-up-time "${WARMUP:-1}" --measurement-time "${MEASURE:-2}" 2>&1 \
  | grep -iE 'compile_small/|time:|change:|slower|faster|no change|p = ' \
  | sed 's/^/  /'

echo ""
echo "Verdict guide (one-sided): 'improved'/'No change' = PASS; 'regressed' with"
echo "p < 0.05 = a real slowdown — investigate + justify or fix (compile speed is"
echo "a load-bearing property). Re-run to confirm it is not machine noise."
