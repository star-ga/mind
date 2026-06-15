# Deterministic Q16.16 N=256 FFT — MIND vs gcc / clang / nvcc

A microbenchmark of a **deterministic fixed-point (Q16.16) radix-2 DIT FFT,
N=256**, comparing the MIND-compiled shared library against the *byte-identical*
C reference kernel compiled `-O3` by gcc, clang, and nvcc.

Two axes are measured:

1. **Speed** — ns/FFT (p50/p95) vs each C compiler's `-O3` codegen of the same
   integer algorithm.
2. **Determinism / bit-identity** — the MIND `.so` output is **byte-for-byte
   identical** to the C reference (FNV-1a hash `a5b24cb31a7f2c7f`), reproducible
   across rebuilds and runs. **This is a property an FP32 FFT (cuFFT / FFTW /
   nvcc float kernels) structurally cannot have** — IEEE-754 add is
   non-associative, so reduction/butterfly order changes the bits; Q16.16
   `(a*b)>>16` then integer add is associative and exact.

---

## Hardware / software

| Component | Value |
|-----------|-------|
| CPU       | Intel Core i7-5930K @ 3.50 GHz (Haswell-E, 6c/12t) |
| GPU       | NVIDIA GeForce RTX 3080 (driver 595.71.05) — *not used in this CPU bench* |
| RAM       | 64 GB |
| OS        | Ubuntu 24.04.4 LTS, kernel 6.17.0-35-generic |
| gcc       | 13.3.0 (`-O3 -march=native`) |
| clang     | 18.1.3 (`-O3 -march=native`) |
| nvcc      | CUDA 12.6, V12.6.85 (`-O3`, host C path) |
| MIND      | mindc 0.8.1, `--emit-shared` (mlir-build) |
| Governor  | `performance` on all cores |

> nvcc here compiles the **host C** reference (`harness.c` + `fft_ref.c`) — this
> is nvcc's `-O3` codegen of the integer FFT on the CPU, the same class of
> codegen-vs-codegen comparison gcc and clang give. No GPU kernel is launched.

---

## Algorithm

Deterministic **Q16.16 fixed-point radix-2 decimation-in-time FFT, N = 256**:

- Input: two `i64` buffers (`re`, `im`), each 256 Q16.16 samples, plus an
  interleaved twiddle table `tw[2k] = round(cos(-2πk/N)·2^16)`,
  `tw[2k+1] = round(sin(-2πk/N)·2^16)` for `k = 0..N/2-1`.
- Bit-reversal permutation, then 8 DIT stages of butterflies.
- Complex multiply is fixed-point: `t = (a*b) >> 16` (arithmetic shift), then
  integer add/sub. `qmul(a,b) = (a*b)>>16` is identical in the MIND kernel
  (`examples/fft_q16.mind`) and the C reference (`fft_ref.c`).

**Algorithm-equivalence note.** The MIND kernel and the C reference implement
the *same* integer arithmetic, butterfly order, and twiddle indexing. They are
not "approximately equal" — they produce the **exact same 4096 output bytes**.
The harness enforces this with a `memcmp` gate before timing; if the outputs
ever differed the benchmark aborts (invalid). MIND's load/store go through the
`__mind_load_i64` / `__mind_store_i64` ABI intrinsics, which the compiler now
lowers to inlined `llvm.load`/`llvm.store` (0 PLT calls in `<fft256>`), so the
inner butterfly is the same mov-based code C emits.

---

## Build

```bash
cd bench/fft
./build.sh          # builds mindc if needed, the .so, and harness_{gcc,clang,nvcc}
```

`build.sh` is self-contained and commented. It:
1. locates or builds `mindc` (release, `mlir-build` feature),
2. compiles `examples/fft_q16.mind` → `bench/fft/fft_test.so` via
   `mindc --emit-shared`, and asserts 0 `__mind_load/store` PLT calls,
3. builds `harness.c` + `fft_ref.c` with gcc, clang, and nvcc (each at `-O3`).

To point at a prebuilt compiler: `MINDC=/path/to/mindc ./build.sh`.

---

## Run

```bash
cd bench/fft
taskset -c 3 ./harness_gcc   ./fft_test.so 300000
taskset -c 3 ./harness_clang ./fft_test.so 300000
taskset -c 3 ./harness_nvcc  ./fft_test.so 300000
```

Each harness:
- dlopen()s the MIND `.so` (`fft256` symbol) — the MIND number is invariant of
  which compiler built the harness,
- compiles the C reference (`fft_ref.c`) *into itself* — that is the baseline,
- runs the **correctness gate** (memcmp MIND vs C, prints `byte_identical` +
  hash),
- then times **only the bare kernel call**, in place on a fixed buffer.

### Methodology (p50/p95)

- **≥ 200k iterations** (300k used for the pinned table below), 2000-call
  warm-up per kernel before timing.
- `clock_gettime(CLOCK_MONOTONIC)` around the bare in-place FFT call only — no
  memcpy / input regeneration inside the timed region.
- The FFT control flow is fully **data-independent** (same butterfly count every
  call regardless of values), so repeated in-place calls are a fair, standard
  microbench.
- Samples sorted; **p50** = median, **p95** = 95th percentile.
- Pin to one core (`taskset -c 3`) to remove migration noise; set the governor
  to `performance`.
- `GFLOP/s = (5·N·log2 N) / ns_p50 = 10240 / ns_p50`.

A busy machine (high load average) inflates p95 and can drag a p50 on the
second-timed kernel; pin the core and re-run on a quiet box for the cleanest
numbers. p50 is the stable statistic to report.

---

## Results (measured 2026-06-15, i7-5930K, pinned core, 300k iters)

| Kernel (compiler) | ns/FFT p50 | ns/FFT p95 | GFLOP/s p50 | MIND vs this |
|-------------------|-----------:|-----------:|------------:|:------------:|
| **MIND `.so`**    | **~3,400** | **~3,600** | **~3.01**   | 1.00× |
| gcc 13.3 `-O3`    | ~4,185     | ~4,330     | ~2.45       | MIND **1.22× faster** |
| clang 18.1 `-O3`  | ~3,760     | ~3,950     | ~2.72       | MIND **1.07× faster** |
| nvcc 12.6 `-O3`   | ~4,205     | ~4,360     | ~2.44       | MIND **1.18× faster** |

- **byte_identical = YES** on every run; hash **`a5b24cb31a7f2c7f`**
  (MIND output == C reference output, exact, all four compilers).
- MIND beats gcc and nvcc by ~1.2× and edges clang by ~1.07× on this kernel.

See `RESULTS-fft-2026-06-15.md` for the full per-run table and verdict.

---

## THE KEY CLAIM — deterministic + bit-identical

The MIND FFT is **deterministic and bit-identical**, verified concretely:

1. **Same artifact, repeated builds** — `mindc --emit-shared` to the same output
   path twice yields a **byte-for-byte identical `.so`** (`sha256sum` matches).
2. **Same output, every run** — the FFT output hash is `a5b24cb31a7f2c7f` on
   every one of 10+ runs.
3. **Same bits as the reference** — MIND output `memcmp`-equals the C reference,
   exactly, on every compiler.

An FP32 FFT (cuFFT, FFTW, nvcc float kernels) **structurally cannot make this
claim**: IEEE-754 floating-point addition is non-associative, so any change in
butterfly/reduction order, SIMD width, or thread count changes the result bits.
The Q16.16 integer kernel here is exact and order-stable — the same artifact
hash and the same output hash hold across rebuilds, runs, and (by construction,
integer-only) across CPU/ARM/GPU substrates. That bit-identity is the property
no FP32 FFT can offer, and it comes here with a *speed win*, not a speed
penalty.
