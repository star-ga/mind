#!/usr/bin/env bash
# Copyright 2026 STARGA Inc. Licensed under the Apache License, Version 2.0.
# Part of the MIND project (Machine Intelligence Native Design).
#
# Reproduce the apples-to-apples MIND-vs-Rust integer-GEMM benchmark.
#
#   1. Build mindc (the deterministic MIND compiler) with the MLIR toolchain.
#   2. Emit the MIND kernels (gemmi8 / gemmq) to a shared library.
#   3. Build the Rust harness at -O2 and -O3 (target-cpu=native, and generic).
#   4. Run each, timing the dlopen'd MIND .so and the in-binary Rust kernel
#      under the SAME warmup + median-of-N method, on the SAME shapes/seeds,
#      and assert the Rust output byte-matches the MIND output.
#
# Requires: rustc/cargo + mlir-opt / mlir-translate / clang on PATH.
set -euo pipefail

MIND_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HARNESS="$MIND_ROOT/scripts/mind-vs-rust"
SO=/tmp/mvr_kernels.so
SRC=/tmp/mvr_kernels.mind
CORE="${MVR_CORE:-2}"
REPS="${MVR_REPS:-64}"

# 1. mindc (skip if already built)
MINDC="$MIND_ROOT/target/release/mindc"
[ -x "$MINDC" ] || MINDC="$MIND_ROOT/target/debug/mindc"
if [ ! -x "$MINDC" ]; then
  echo "building mindc (~2.5 min) ..."
  ( cd "$MIND_ROOT" && cargo build --release \
      --features "mlir-build std-surface cross-module-imports" --bin mindc )
  MINDC="$MIND_ROOT/target/release/mindc"
fi

# 2. emit the MIND kernels
cat > "$SRC" <<'MIND'
pub fn gemmi8(a: i64, b: i64, c: i64, m: i64, k: i64, n: i64) -> i64 {
    __mind_blas_matmul_mm_i8_v(a, b, c, m, k, n)
}
pub fn gemmq(a: i64, b: i64, c: i64, m: i64, k: i64, n: i64) -> i64 {
    __mind_blas_matmul_mm_q16_v(a, b, c, m, k, n)
}
MIND
"$MINDC" "$SRC" --emit-shared "$SO"

# determinism axis: emit twice, sha256 must match
"$MINDC" "$SRC" --emit-shared /tmp/mvr_a.so >/dev/null
"$MINDC" "$SRC" --emit-shared /tmp/mvr_b.so >/dev/null
echo "## MIND .so build reproducibility (must be identical):"
sha256sum /tmp/mvr_a.so /tmp/mvr_b.so

# 3 + 4. build + run the harness at each opt level
run_variant() {
  local label="$1" flags="$2" optenv="$3" dir="$4"
  ( cd "$HARNESS" && RUSTFLAGS="$flags" MVR_OPT="$optenv" \
      cargo build --release --target-dir "$dir" >/dev/null 2>&1 )
  echo "######## $label ########"
  taskset -c "$CORE" "$HARNESS/$dir/release/mind-vs-rust" "$SO" --reps "$REPS"
  echo
}

run_variant "RUST -O3 (target-cpu=native)" "-C opt-level=3 -C target-cpu=native" 3 target-o3
run_variant "RUST -O2 (target-cpu=native)" "-C opt-level=2 -C target-cpu=native" 2 target-o2
run_variant "RUST -O3 (generic / plain cargo --release)" "-C opt-level=3" 3 target-o3g
run_variant "RUST -O2 (generic)" "-C opt-level=2" 2 target-o2g
