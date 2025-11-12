# Benchmarks

This document summarizes the benchmarking approach for the MIND compiler and runtime. The goal is to track regressions and communicate baseline performance characteristics.

## Methodology

- **Target Hardware** – Default runs target x86-64 AVX2 hosts with 32 GB RAM; GPU runs target CUDA-capable cards when `mlir-exec` is enabled.
- **Datasets** – Synthetic workloads (matrix multiplications, convolutions) plus representative ML kernels sourced from `benchmarks/`.
- **Execution Modes** – Interpreter (`cpu-exec`), ahead-of-time MLIR (`mlir-build`), and JIT (`mlir-exec`).
- **Warmup & Repetitions** – Each benchmark performs 3 warmup runs followed by 10 measured iterations; results report median and 95th percentile.

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

## Future Work

- GPU benchmark coverage for the runtime plugin API
- Automated comparison against PyTorch/XLA baselines
- Visualization dashboards for long-term trends

See the [roadmap](roadmap.md) for scheduling details.
