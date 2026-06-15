#!/usr/bin/env bash
# Build the GPU FFT bench: MIND Q16 kernel (nvcc -O3/-O0 codegen race) + cuFFT baseline.
# Requires: nvcc (CUDA 12.x), cufft. Run on a box with an NVIDIA GPU.
set -euo pipefail
cd "$(dirname "$0")"
echo "[1/3] MIND Q16 GPU harness (-O3)"
nvcc -O3 -arch=sm_86 harness_gpu.cu fft_q16_gpu.cu -o harness_gpu_O3
echo "[2/3] MIND Q16 GPU harness (-O0, codegen-race control)"
nvcc -O0 -arch=sm_86 harness_gpu.cu fft_q16_gpu.cu -o harness_gpu_O0
echo "[3/3] cuFFT FP32 baseline"
nvcc -O3 -arch=sm_86 cufft_context.cu -o cufft_context -lcufft
echo "done. run: ./harness_gpu_O3 65536 300   and   ./cufft_context 65536 300"
