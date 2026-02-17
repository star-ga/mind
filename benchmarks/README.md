# MIND Performance Benchmarks

This directory contains benchmark results and performance analysis for the MIND compiler and runtime.

## Latest Results

**Last Updated**: February 17, 2026
**Reference Platform**: Ubuntu 24.04, Intel Core i7-5930K @ 3.50GHz, 64GB DDR4, RTX 3080 10GB, CUDA 12.8

### Compiler Performance (v0.2.1)

| Metric | Value | Notes |
|--------|-------|-------|
| **scalar_math** | **1.77 µs** | Single arithmetic expression |
| **matmul ops** | **2.6-3.0 µs** | Matrix multiplication programs |
| **medium_mlp** | **6.15 µs** | Multi-layer perceptron (5 ops) |
| **large_network** | **15.49 µs** | Complex network (12 ops) |
| **Throughput** | **~340,000 compiles/sec** | For simple programs |
| **Scaling** | **O(n) with program complexity** | Compile-time scales with number of operations, NOT tensor dimensions |

**Scope**: MIND measures **frontend only** (parse + typecheck + IR lowering). Does not include code generation.

See **[BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md)** for detailed methodology and results.
See **[FINAL_PATENT_RESULTS.md](FINAL_PATENT_RESULTS.md)** for patent benchmark documentation.

## Comparison with PyTorch 2.10 GPU (Verified February 2026)

| Benchmark | MIND v0.2.1 (frontend) | PyTorch 2.10 GPU (full pipeline) | Ratio |
|-----------|------------------------|----------------------------------|-------|
| scalar_math | 1.77 µs | 99 ms | 56,000× |
| small_matmul | 2.95 µs | 162 ms | 55,000× |
| simple_mlp | 6.15 µs | 752 ms | 122,000× |
| conv2d | ~5 µs | 878 ms | 176,000× |

**Note**: Different scopes of work. MIND measures frontend compilation. PyTorch measures full torch.compile() pipeline including Triton/cuBLAS kernel generation. Mojo measures full LLVM compilation to native binary.

## Patent Benchmarks

### Quick Start - Run All Benchmarks

```bash
# Run all patent benchmarks
./run_all_benchmarks.sh
```

### Available Benchmarks

| Benchmark | Status | Patent Claims | Description |
|-----------|--------|---------------|-------------|
| **[PyTorch Comparison](pytorch_comparison/)** | ✅ Ready | Claims 1-5, 11-15 | Compilation time vs torch.compile() |
| **[Determinism Proof](determinism/)** | ✅ Ready | Claims 16-20 | Bit-level reproducibility verification |
| **[Autograd Comparison](autograd_comparison/)** | ✅ Ready | Claims 6-10 | Gradient computation speed & memory |
| **[JAX Comparison](jax_comparison/)** | ✅ Ready | Claims 1-5 | Compilation time vs jax.jit() |
| **[Inference Speed](inference/)** | ✅ Ready | Supporting | Runtime execution comparison |
| **[Mojo Comparison](mojo/)** | ✅ Complete | Claims 1-5 | Compilation time vs Mojo |

### Individual Benchmark Commands

```bash
# MIND Criterion benchmarks (in-process, most accurate)
cargo bench --bench simple_benchmarks
cargo bench --bench compiler

# PyTorch 2.0 compilation comparison
cd pytorch_comparison && python benchmark_pytorch_compile.py

# Scientific benchmark with video-ready output
python scientific_benchmark.py

# Determinism proof (requires MIND CLI)
cd determinism && python benchmark_determinism.py
```

## MIND Internal Benchmarks

### Quick Start

```bash
# Run all working benchmarks
cargo bench --bench simple_benchmarks
cargo bench --bench compiler

# View results
open target/criterion/report/index.html
```

### Methodology

**Record**: Hardware specs, batch sizes, precision, exact commit hash. Prefer reproducible scripts.

**Baseline Requirements**:
- Clean, idle system
- CPU frequency scaling disabled
- Consistent sample sizes (100 for Criterion, 5-10 for PyTorch cold-start)
- Document outliers and system specs

## Benchmark Results Summary

### Compilation Speed Comparisons (v0.2.1, February 2026)

| Framework | Compilation Time | Scope | MIND Ratio |
|-----------|-----------------|-------|------------|
| **MIND v0.2.1** | **1.8-15.5 µs** | Frontend only | **1× (baseline)** |
| PyTorch 2.10 GPU | 99-878 ms | Full pipeline (Inductor + Triton/cuBLAS) | 35,000-176,000× |
| Mojo 0.26.1 | 810-829 ms | Full LLVM compilation (`mojo build`) | 135,000-458,000× |
| JAX 0.9 | 37.5-360.5 ms | Cold-start XLA compilation (`jax.jit()`) | 21,200-95,100× |

**Important**: These frameworks perform different amounts of work. Ratios reflect frontend vs full pipeline/compilation comparison.

### Key Findings

1. **Frontend Speed**: MIND frontend compiles in 1.8-15.5 µs (Criterion verified)
2. **Throughput**: ~340,000 frontend compilations per second sustained
3. **Determinism**: Verified bit-identical outputs across 1000+ runs
4. **Autograd**: Compile-time differentiation vs runtime tape-based
5. **Scaling**: Compile time scales with program complexity (O(n) operations), not tensor dimensions
