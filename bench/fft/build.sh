#!/usr/bin/env bash
# ============================================================================
# build.sh — self-contained build for the deterministic Q16.16 N=256 FFT bench.
#
# Produces, from a clean checkout:
#   1. mindc            — the MIND compiler (cargo, mlir-build feature) if absent
#   2. fft_test.so      — examples/fft_q16.mind compiled to a cdylib via mindc
#   3. harness_gcc      — harness.c + fft_ref.c built with gcc   -O3 -march=native
#      harness_clang    — harness.c + fft_ref.c built with clang -O3 -march=native
#      harness_nvcc     — harness.c + fft_ref.c built with nvcc  -O3  (if nvcc present)
#
# The harness dlopen()s fft_test.so (the MIND kernel) and also calls fft_ref.c
# (the byte-identical C kernel) compiled INTO the harness. So each harness_<cc>
# binary reports that compiler's -O3 codegen of the reference as the baseline,
# while the MIND number is invariant (it always comes from the same .so).
#
# Run after building:   ./harness_gcc   ./fft_test.so 300000
#                       ./harness_clang ./fft_test.so 300000
#                       ./harness_nvcc  ./fft_test.so 300000
#
# Reproducible: the .so is bit-identical across rebuilds (deterministic codegen);
# the FFT output hash is a5b24cb31a7f2c7f on every run on every substrate.
# ============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"      # .../mind
MIND_SRC="$REPO_ROOT/examples/fft_q16.mind"
SO_OUT="$HERE/fft_test.so"

# ---- 1. locate or build mindc (needs the mlir-build feature for --emit-shared) ----
MINDC="${MINDC:-}"
if [[ -z "$MINDC" ]]; then
  for cand in "$REPO_ROOT/target/release/mindc" "$HOME/.mind/target/release/mindc"; do
    [[ -x "$cand" ]] && MINDC="$cand" && break
  done
fi
if [[ -z "$MINDC" || ! -x "$MINDC" ]]; then
  echo "[build] mindc not found — building (release, mlir-build feature) ..."
  ( cd "$REPO_ROOT" && cargo build --release --features "mlir-build std-surface cross-module-imports" --bin mindc )
  MINDC="$REPO_ROOT/target/release/mindc"
fi
echo "[build] mindc = $MINDC"
"$MINDC" --version | head -1 || true

# ---- 2. compile the MIND FFT kernel to a shared library --------------------
echo "[build] compiling $MIND_SRC -> $SO_OUT (mindc --emit-shared)"
"$MINDC" --emit-shared "$SO_OUT" "$MIND_SRC" >/dev/null
echo "[build] .so sha256: $(sha256sum "$SO_OUT" | cut -d' ' -f1)"
# Sanity: confirm load/store intrinsics are inlined (0 PLT calls in <fft256>).
PLT=$(objdump -d "$SO_OUT" 2>/dev/null \
        | awk '/<fft256>:/{f=1} f&&/call.*__mind_(load|store)/{c++} /^$/{if(f)f=0} END{print c+0}')
echo "[build] __mind_load/store PLT calls in <fft256>: $PLT (expect 0 = inlined)"

# ---- 3. build the harness with each available C compiler -------------------
CFLAGS="-O3 -march=native"
build_cc () {
  local cc="$1" out="$2"; shift 2
  if command -v "$cc" >/dev/null 2>&1 || [[ -x "$cc" ]]; then
    echo "[build] $out  ($cc $*)"
    "$cc" "$@" -o "$HERE/$out" "$HERE/harness.c" "$HERE/fft_ref.c" -ldl -lm
  else
    echo "[build] SKIP $out ($cc not found)"
  fi
}

build_cc gcc   harness_gcc   $CFLAGS
build_cc clang harness_clang $CFLAGS

# nvcc: try PATH, then the standard CUDA 12.6 location. nvcc tunes -O3 for the
# host CPU itself; we do NOT pass -march=native (nvcc host-passthrough differs).
NVCC="${NVCC:-}"
if [[ -z "$NVCC" ]]; then
  if command -v nvcc >/dev/null 2>&1; then NVCC=nvcc
  elif [[ -x /usr/local/cuda-12.6/bin/nvcc ]]; then NVCC=/usr/local/cuda-12.6/bin/nvcc
  elif [[ -x /usr/local/cuda/bin/nvcc ]]; then NVCC=/usr/local/cuda/bin/nvcc
  fi
fi
if [[ -n "$NVCC" ]]; then
  echo "[build] harness_nvcc  ($NVCC -O3)"
  "$NVCC" -O3 -o "$HERE/harness_nvcc" "$HERE/harness.c" "$HERE/fft_ref.c" -ldl -lm
else
  echo "[build] SKIP harness_nvcc (nvcc not found)"
fi

echo "[build] DONE. Run:  ./harness_gcc ./fft_test.so 300000"
