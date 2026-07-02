# CuPy Comparison Benchmark

This benchmark uses **CuPy** (cuBLAS / CUDA) as the GPU **foil** for MIND's
two load-bearing properties:

1. **Determinism (the wedge)** — MIND's deterministic kernel produces a
   **bit-identical** artifact across substrates (x86 avx2 == ARM neon),
   re-run after re-run, gate-enforced. CuPy's GPU float results are **not**
   guaranteed identical run-to-run for atomic/order-sensitive ops.
2. **Performance (the tax)** — how close MIND's deterministic GEMM is to
   raw cuBLAS GPU throughput.

> **CuPy is the FOIL, not a dependency.** CuPy is MIT-licensed and is used
> **only** by this benchmark. It is never linked into or shipped with any MIND
> artifact. Keep it out of the runtime.

This is roadmap item **3.2b**.

## What We Prove

| Leg | Claim | Evidence |
|-----|-------|----------|
| 1 | CuPy GPU float output can DIFFER run-to-run; MIND CPU Q16.16 is bit-identical across x86/ARM | two CuPy hashes that differ, vs one MIND hash repeated (avx2 == neon) |
| 2 | MIND deterministic GEMM is within X% of cuBLAS throughput AND bit-identical, which CuPy is not | GFLOP/s sweep (VRAM-gated) |

## Honest Framing (READ THIS)

- The bit-identity claim is **CPU x86/ARM bit-identity vs CuPy GPU float
  non-determinism**. We do **NOT** claim MIND's own GPU semantic tier is
  bit-identical to CuPy's GPU. MIND's GPU path *also* permits non-determinism in
  the semantic tier; the moat is the deterministic CPU artifact pinned across
  substrates.
- Divergence is **op-, size-, hardware-, and driver-dependent.** On this box
  (RTX 3080, CUDA 12, CuPy 14.1.1) the simple repeated `cupy.sum` reduction and
  fixed-shape cuBLAS GEMM were **STABLE** run-to-run — cuBLAS picks one
  deterministic kernel and the reduction order was fixed. We report that
  honestly. The op that **does** diverge here is **atomic `cupyx.scatter_add`**:
  many GPU threads atomically accumulate floats into a few bins, and the atomic
  commit order is genuinely nondeterministic, so the output bytes drift every
  run. **No fake hashes** — if an op is deterministic on your hardware, the
  script says so.
- A GPU-float GFLOP/s and a deterministic-CPU GFLOP/s are **different
  substrates**. Leg 2 states this; the perf number is context, the bit-identity
  is the product claim.

## Leg 1 — Determinism Demo (priority; tiny GPU footprint)

Runs the same logical float workloads in CuPy several ways, hashes the **raw
output bytes**, and checks for divergence. The MIND side is **not recomputed** —
it points at the committed, CI-gate-enforced reference in
`tests/cross_substrate_identity/gemm-q16-64x64x64/reference_hashes.toml`
(single source of truth, RFC 0020 §10).

```bash
python leg1_determinism.py
```

### Verified output (RTX 3080, CUDA 12, CuPy 14.1.1 — this box)

```
MIND (committed, CI-gate-enforced — gemm-q16-64x64x64):
  x86 avx2 hash : 92e2cb75d74d83a4a398d78d9ac560f195279c31814972c892f856f675faea0f
  arm neon hash : 92e2cb75d74d83a4a398d78d9ac560f195279c31814972c892f856f675faea0f
  bit-identical x86==arm : True

CuPy (foil) — same input, repeated runs, raw output bytes hashed:
  [  scatter_add_f32_8M] cupyx.scatter_add (atomic) distinct_hashes=8/8  -> DIVERGES
        run-hash #1: 104a530f45f503de92bf6fc6d1b4dbd59b956552d23e6b42e2609242fcedadd7
        run-hash #2: 181de2c3a8e5f15071f65b3cc731f718ac29ba908e676b3a0793c2edde282f29   <-- DIFFERS
  [   reduction_f32_64M] cupy.sum               distinct_hashes=1/8  -> stable (deterministic here)
  [   reduction_f32_16M] cupy.sum               distinct_hashes=1/8  -> stable (deterministic here)
  [       gemm_f32_1024] cupy matmul (cuBLAS)   distinct_hashes=1/8  -> stable (deterministic here)
  [       gemm_f32_2048] cupy matmul (cuBLAS)   distinct_hashes=1/8  -> stable (deterministic here)
```

**The artifact, side by side:**

| Side | Workload | run 1 hash | run 2 hash | identical? |
|------|----------|-----------|-----------|------------|
| CuPy GPU | `scatter_add` f32, 8M atomics | `104a530f…` | `181de2c3…` | **NO** (8/8 distinct) |
| MIND CPU | Q16.16 GEMM 64×64×64 | `92e2cb75…` (avx2) | `92e2cb75…` (neon) | **YES** (gate-enforced) |

The CuPy input bytes are identical every run (input hash pinned); only the output
drifts. The MIND hash is the same value on x86 and ARM and on every re-run.

## Leg 2 — Perf Baseline (VRAM-gated)

Measures cuBLAS GEMM throughput (GFLOP/s) at 512 / 1024 / 2048 / 4096 square,
reported alongside the MIND deterministic GEMM at the same logical workload.

```bash
python leg2_perf.py [--required-free-gb 8]
```

### The VRAM gate

The RTX 3080 here is 10 GB and **ollama holds most of it**. The harness queries
free VRAM and **SKIPS gracefully** if less than `--required-free-gb` (default 8)
is free — it will **never evict** a co-resident tenant. The 8 GB gate is
deliberately conservative: it covers the CUDA context + cuBLAS workspace +
headroom, not just the matrices (the 4096² working set is only ~0.2 GB).

### Verified output (this box, gate skip)

```
VRAM: 4.25 GB free / 10.35 GB total
Gate: need 8.0 GB free (largest GEMM 4096x4096 ~ 0.20 GB working set)
SKIP (graceful): only 4.25 GB free < 8.0 GB required; refusing to evict
  co-resident GPU tenants (e.g. ollama).
```

To run the perf leg, free the GPU (stop ollama or run on a box with headroom)
and re-run. The MIND deterministic-GEMM throughput is reproduced via the in-tree
Criterion bench:

```bash
cargo bench --features 'mlir-build std-surface cross-module-imports' --bench det_matmul_q16
```

## Output Files

- `leg1_determinism_results.json` — raw input/output hashes per probe + the MIND reference.
- `leg2_perf_results.json` — GEMM GFLOP/s sweep, or the skip reason + VRAM snapshot.

## Reproducing the MIND Side Independently

The MIND hash is not magic — it is `sha256(C)` where `C` is the 64×64 Q16.16 GEMM
output, each `i32` little-endian, inputs from an LCG seeded `0xDEADBEEF`
(A before B). The full gate that pins it:

```bash
cargo test --features 'mlir-build std-surface cross-module-imports' \
  --test cross_substrate_identity gemm_q16_reproducibility_gate
```

The encoding and seed contract live in
`tests/cross_substrate_identity/gemm-q16-64x64x64/manifest.toml`; the pinned
per-substrate hashes (avx2 == neon) live in its `reference_hashes.toml`.

## Prerequisites

```bash
pip install -r requirements.txt   # cupy-cuda12x for CUDA 12
```

Python 3.11+ (uses stdlib `tomllib` to read the MIND fixtures).
