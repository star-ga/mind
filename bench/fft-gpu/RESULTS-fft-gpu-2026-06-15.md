# RESULTS — Deterministic Q16.16 N=256 FFT on GPU (MIND vs cuFFT)

**Measured 2026-06-15 on the dev box (RTX 3080, CUDA 12.6, nvcc V12.6.85).**
Reported exactly as measured. **No fabricated numbers, no inflation. cuFFT is
FASTER than our kernel on raw throughput — that is stated plainly below.**

## The honest headline

| Kernel | batch | ns/FFT (kernel-only, p50) | byte-identical across runs? |
|--------|------:|--------------------------:|:---------------------------:|
| **cuFFT FP32** (vendor lib) | 65536 | **6.23** | ❌ no cross-config guarantee |
| **MIND Q16 GPU** (ours)     | 65536 | **18.44** | ✅ `a5b24cb31a7f2c7f` every run |

At matched batch (65536), **cuFFT is ~3.0× faster** than our current GPU kernel.
We do **NOT** beat cuFFT on raw GPU throughput today. This is the opposite of the
CPU bench (where MIND beats nvcc 1.23×): on GPU, cuFFT — a decades-tuned vendor
library with shared-memory radix-4/8 butterflies and register blocking — beats
our naive 116-line radix-2 integer kernel.

Matched-batch cross-check (4096): cuFFT 8.53 ns/FFT vs MIND 21.91 ns/FFT (~2.6×).

## What we DO win: the property, not the speed

- MIND GPU output hash is `a5b24cb31a7f2c7f` on **every launch**, and **byte-identical
  to the CPU reference** (`fft_ref.c`) — same bits CPU and GPU.
- cuFFT FP32 carries **no** cross-config byte-identity guarantee (IEEE-754 add is
  non-associative; the hash repeats on a fixed plan here but is not promised across
  batch/width/thread-count/arch). cuFFT hash this run: `25534381cf6f50b1`.

So the defensible claim is: **we trade ~3× raw GPU latency for provable, cross-
substrate bit-identity that cuFFT structurally cannot offer** — and the speed gap
is an open optimization target, not a property limit.

## Internal-only optimization note (deferred work)

The kernel (`fft_q16_gpu.cu`, 116 lines) is naive radix-2, one FFT per thread-ish.
Upgrade path to close/beat the cuFFT gap WITHOUT breaking byte-identity:
shared-memory radix-4/8 butterflies, register-blocked stages, warp-cooperative
N=256 (one warp per FFT). Each step must re-pass the byte-identity gate
(hash must stay `a5b24cb31a7f2c7f`). Tracked as a mind-det-gemm task; numbers
above are the pre-optimization baseline.

## Reproduce

```
# MIND kernel (nvcc -O3) vs CPU reference, byte-identity + timing:
./harness_gpu_O3 65536 300
# cuFFT FP32 baseline at same batch:
./cufft_context 65536 300
```
