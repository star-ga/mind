# Benchmarks

This document summarizes the benchmarking approach for the MIND compiler and runtime. The goal is to track regressions and communicate baseline performance characteristics.

## Methodology

- **Target Hardware** – Default runs target x86-64 AVX2 hosts with 32 GB RAM; GPU runs target CUDA-capable cards when `mlir-exec` is enabled.
- **Datasets** – Synthetic workloads (matrix multiplications, convolutions) plus representative ML kernels sourced from `benchmarks/`.
- **Execution Modes** – Interpreter (`cpu-exec`), ahead-of-time MLIR (`mlir-build`), and JIT (`mlir-exec`).
- **Warmup & Repetitions** – Each benchmark performs 3 warmup runs followed by 10 measured iterations; results report median and 95th percentile.

## Performance Baselines

The following baselines were collected on reference hardware (AMD Ryzen 9 5900X, 64 GB DDR4-3600, Ubuntu 22.04 LTS) using Rust 1.82 stable.

### Compiler Performance

| Operation | Input Size | Time (median) | Memory |
|-----------|------------|---------------|--------|
| Parse | 1K LOC | 2.1 ms | 12 MB |
| Parse | 10K LOC | 18 ms | 45 MB |
| Type check | 1K LOC | 4.3 ms | 18 MB |
| Type check | 10K LOC | 38 ms | 85 MB |
| IR lower | 1K LOC | 1.8 ms | 8 MB |
| IR lower | 10K LOC | 15 ms | 42 MB |
| MLIR emit | 1K ops | 3.2 ms | 15 MB |
| MLIR emit | 10K ops | 28 ms | 95 MB |

### Shape Inference

| Tensor Rank | Broadcast Dims | Time |
|-------------|----------------|------|
| 2D | 0 | 0.8 μs |
| 2D | 2 | 1.2 μs |
| 4D | 0 | 1.5 μs |
| 4D | 4 | 2.8 μs |
| 8D | 4 | 5.1 μs |

### Autodiff

| Function Complexity | Forward Ops | Grad Gen Time |
|---------------------|-------------|---------------|
| Simple (add/mul) | 10 | 0.4 ms |
| Medium (matmul chain) | 100 | 3.2 ms |
| Complex (conv + reduce) | 1000 | 28 ms |

### Test Suite

| Category | Test Count | Total Time |
|----------|------------|------------|
| Unit tests | 80 | ~0.2 s |
| Integration tests | 89 | ~0.5 s |
| Full suite | 169+ | ~1 s |

## Metrics

| Metric        | Description                                      |
| ------------- | ------------------------------------------------ |
| Latency       | Execution time per run (ms)                      |
| Throughput    | Ops or samples per second                        |
| Memory usage  | Peak RSS collected via `procfs` helpers          |
| Compile time  | IR → MLIR → executable duration                  |

Results are exported as JSON into `benchmarks/results/*.json` and visualized with the CLI (`mind bench report`).

## Regression Tracking

Continuous integration runs a smoke subset on every pull request. Nightly jobs execute the full suite and compare against the rolling baseline stored in `benchmarks/baselines/`.

When a regression exceeds thresholds:

1. CI marks the run unstable and attaches artifacts.
2. Engineers inspect IR/MLIR dumps to identify passes responsible for the change.
3. A follow-up issue documents the root cause and mitigation plan.

## Runtime Tensor Compute (Criterion, in-process)

All numbers measured with Criterion.rs 0.5.1 (in-process, 100 samples).
MIND measurements include the full parse + compile + compute pipeline.

| Operation | MIND Time | Notes |
|-----------|----------|-------|
| Parse + eval overhead (baseline) | 33 µs | Empty program parse + eval |
| add 4 elements (materialized) | 33 µs | Dominated by parse overhead |
| add 10K elements | 54 µs | ~21 µs actual compute |
| add 100K elements | 220 µs | ~187 µs actual compute |
| sum 10K elements | 65 µs | add + reduce |
| mean 10K elements | 67 µs | add + reduce + divide |
| relu 10K elements | 62 µs | subtract + clamp |
| matmul (10,20)×(20,30) | 45 µs | 6K multiply-accumulate |
| matmul (32,64)×(64,32) | 70 µs | 65K multiply-accumulate |
| matmul (64,128)×(128,64) | 194 µs | 1M multiply-accumulate |
| matmul (128,256)×(256,128) | 913 µs | 8.4M multiply-accumulate |

See [`benchmarks/compiler_performance.md`](benchmarks/compiler_performance.md) for detailed compilation methodology.

## Methodology

All benchmarks use:
- **Criterion.rs 0.5.1** for MIND (in-process, eliminates subprocess overhead)
- 100 samples minimum, median reported
- MIND measurements include full parse + compile + compute pipeline

## Future Work

- GPU runtime benchmarks on CUDA hardware
- BLAS-accelerated matmul backend
- Full Remizov solver end-to-end benchmarks (CPU and GPU)
- Automated comparison against PyTorch/XLA baselines
- Visualization dashboards for long-term trends

See the [roadmap](roadmap.md) for scheduling details.
