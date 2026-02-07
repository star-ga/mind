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

## Runtime Execution (v0.1.9)

End-to-end execution benchmarks for compiled MIND programs via the `mind_main` FFI entry point.
Programs compiled with `mindc build --release`, CPU backend.

| Workload | Size | Median | Notes |
|----------|------|--------|-------|
| Chernoff step | N=1,024 | 2 ms | 7 elem-wise + reduction |
| Chernoff step | N=262,144 | 3 ms | Constant propagation |
| Matmul | 128x128 | 2 ms | Sum of result |
| Matmul | 256x256 | 3 ms | Sum of result |
| 10-step iteration | N=1,024 | 3 ms | Chained Chernoff |
| 20-step iteration | N=1,024 | 4 ms | Chained Chernoff |
| Matmul + elem-wise | 256x256 | 3 ms | Full solver step |

The evaluator uses constant propagation for uniform-fill tensors. The 2-3 ms floor
is parse + evaluate + output overhead. See [`benchmarks/compiler_performance.md`](benchmarks/compiler_performance.md)
for detailed methodology.

## Comparison: MIND vs NumPy/SciPy

Measured on the same machine. NumPy 1.26.4, SciPy 1.11.4, MIND v0.1.9.

### Startup Time

| Framework | Time | Speedup |
|-----------|------|---------|
| **MIND binary** | **1.1 ms** | **105x** |
| Python + NumPy | 111 ms | 1x |

### Matmul (256x256)

| Framework | Time |
|-----------|------|
| **MIND** | **3 ms** |
| NumPy (BLAS) | 8.9 ms |

### ODE Solving: Remizov vs SciPy

| Method | Time | Variable Coefficients? |
|--------|------|----------------------|
| SciPy solve_bvp (n=200) | 2.9 ms | Limited |
| Remizov (Python, n_iter=50) | 1,226 ms | Yes (any a,b,c) |
| Remizov (MIND native, projected) | ~5 ms | Yes (any a,b,c) |
| Remizov (MIND GPU, projected) | ~0.05 ms | Yes + parallel |

See [`benchmarks/compiler_performance.md`](benchmarks/compiler_performance.md) for full comparison tables.

## GPU Projections (Remizov Solver)

The Remizov shift operator is embarrassingly parallel: each grid point is independent.

| Grid Size | CPU (est.) | GPU (est.) | Speedup |
|-----------|-----------|-----------|---------|
| 1,000 | ~50 ms | ~0.5 ms | ~100x |
| 10,000 | ~500 ms | ~1 ms | ~500x |
| 100,000 | ~5 s | ~5 ms | ~1,000x |

GPU advantage grows with grid size because every x_i in the shift operator
`S(t)f(x)` can be computed independently across CUDA cores.

## Future Work

- GPU runtime benchmarks (validate projections on CUDA hardware)
- Runtime benchmarks with non-uniform (materialized) tensor data
- Direct comparison with Julia DifferentialEquations.jl
- Automated comparison against PyTorch/XLA baselines
- Visualization dashboards for long-term trends

See the [roadmap](roadmap.md) for scheduling details.
