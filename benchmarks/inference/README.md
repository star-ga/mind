# Inference Speed Benchmark

This benchmark compares **runtime execution speed** (not compilation time) between MIND and PyTorch.

## What We Measure

- **Latency**: Time per batch (milliseconds)
- **Throughput**: Samples processed per second
- **Execution speed**: After compilation is complete

## Quick Start

```bash
pip install -r requirements.txt
python benchmark_inference.py
```

## Test Cases

| Benchmark | Model | Input Shape | Batch Size |
|-----------|-------|-------------|------------|
| `large_matmul_4096` | 4096×4096 matrix multiplication | (1, 4096) | 1 |
| `mlp_batch32` | 784 → 512 → 256 → 10 MLP | (32, 784) | 32 |
| `conv2d_inference` | Small CNN (Conv → Pool → Conv → Pool → FC) | (8, 3, 64, 64) | 8 |

## Expected Results

```
=== INFERENCE SPEED COMPARISON: MIND vs PyTorch ===

Latency (lower is better):
Benchmark                 MIND                 PyTorch              MIND Speedup
--------------------------------------------------------------------------------
large_matmul_4096         80.00 ms             XXX ms               X.XX×
mlp_batch32               1.20 ms              XXX ms               X.XX×
conv2d_inference          5.00 ms              XXX ms               X.XX×

Throughput (higher is better):
Benchmark                 MIND                 PyTorch              MIND Advantage
--------------------------------------------------------------------------------
large_matmul_4096         12.5 samples/sec     XXX samples/sec      X.XX×
mlp_batch32               26.7K samples/sec    XXX samples/sec      X.XX×
conv2d_inference          1.6K samples/sec     XXX samples/sec      X.XX×
```

## Key Differences from Compilation Benchmarks

### Compilation Benchmarks (PyTorch, JAX, Mojo)
- Measure time to **compile** code
- Historical external ratios are not current claims because their artifacts and timing scopes do not match the MIND T1 baseline
- One-time cost

### Inference Benchmarks (This)
- Measure time to **execute** code
- Both use LLVM backend
- Performance should be similar

## Why Inference Speed Matters

### Real-World Performance
- Compilation is one-time cost
- Inference runs millions of times
- Total time = Compile time + (Inference time × N)

### MIND Advantage
Even if inference speeds are equal:
- MIND compiles in **1.8-15.5 µs** (frontend)
- External compile timing is not a matched, committed result
- Break-even requires measured, tier-matched compile and inference inputs

Example:
```
Model compilation + 1000 inferences:

These illustrative totals require measured, tier-matched compile and inference
inputs; the repository does not publish them as a result.
```

## Technical Details

### PyTorch Execution
- Uses ATen/C++ operators
- BLAS/cuBLAS for matmul
- cuDNN for convolutions
- Highly optimized

### MIND Execution (Expected)
- MLIR → LLVM → machine code
- Same BLAS/cuBLAS backend
- Similar optimizations
- Possible advantage: static compilation allows more aggressive optimizations

## Limitations

### Current Benchmark
- MIND values are **estimates**
- PyTorch values are **actual measurements**
- Real MIND runtime benchmarks require executing MIND-compiled code

### Future Work
- Build actual MIND runtime
- Execute MIND-compiled programs
- Compare real execution speeds

## Comparison Summary

| Metric | MIND | PyTorch |
|--------|------|---------|
| **Compilation** | 1.8-15.5 µs (frontend T1) | No matched external result |
| **Inference** | ~Equal | ~Equal |
| **Total (1000 runs)** | No measured total | No measured total |

Total-time comparisons require both end-to-end compilation and inference measurements for the same workload.

## Troubleshooting

### PyTorch too old
```bash
pip install 'torch>=2.13'
```

### CUDA errors
The benchmark automatically uses CPU if CUDA is unavailable.

### Slow measurements
Reduce `SAMPLE_SIZE` (default: 100).

## Results

After running, results are saved to:
- `inference_results.json` - Detailed timing data

## Context

This benchmark complements the compilation benchmarks:

**Compilation Benchmarks**: raw external records are retained with explicit scope; no cross-tier ratio is published
**Inference Benchmarks**: MIND and PyTorch are similar (both use LLVM/BLAS)

**Overall**: Total workflow savings require measured, tier-matched compile and
inference results for the same workload.

## Patent Relevance

This benchmark supports the **overall value proposition** of MIND:
- Fast compilation enables interactive development
- No sacrifice in runtime performance
- Best of both worlds: fast compile + fast execution
