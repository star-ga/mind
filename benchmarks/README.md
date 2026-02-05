# MIND Performance Benchmarks

This directory contains benchmark results and performance analysis for the MIND compiler and runtime.

## Latest Results

**Last Updated**: February 5, 2026
**Reference Platform**: Ubuntu 24.04, Intel Core i7-5930K @ 3.50GHz, 64GB DDR4, RTX 3080 10GB, CUDA 13.0

### Compiler Performance

| Metric | Value | Comparison |
|--------|-------|------------|
| **scalar_math** | **25 µs** | 136,000× faster than PyTorch 2.9 |
| **matmul ops** | **52-53 µs** | 17,000-36,000× faster than Mojo 0.25 |
| **Throughput** | **~40,000 compiles/sec** | Interactive development ready |
| **Scaling** | **O(1) with tensor size** | Compile-time independent of data dimensions |

See **[BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md)** for detailed methodology and results.

## Patent Benchmarks

**NEW**: Comprehensive benchmarks for patent application support.

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
# PyTorch 2.0 compilation comparison
cd pytorch_comparison && python benchmark_pytorch_compile.py

# Determinism proof (requires MIND CLI)
cd determinism && python benchmark_determinism.py

# Autograd comparison
cd autograd_comparison && python benchmark_autograd.py

# JAX comparison
cd jax_comparison && python benchmark_jax_compile.py

# Inference speed
cd inference && python benchmark_inference.py

# Mojo comparison (already run)
cd mojo && python benchmark_mojo_compilation.py
```

## MIND Internal Benchmarks

### Quick Start

```bash
# Run all working benchmarks
cargo bench --bench simple_benchmarks

# View results
open target/criterion/report/index.html
```

### Methodology

**Record**: Hardware specs, batch sizes, precision, exact commit hash. Prefer reproducible scripts.

**Baseline Requirements**:
- Clean, idle system
- CPU frequency scaling disabled
- Consistent sample sizes (20 for quick tests, 100 for official reports)
- Document outliers and system specs

## Benchmark Results Summary

### Compilation Speed Comparisons (Verified January 2026)

| Framework | Compilation Time | MIND Speedup |
|-----------|-----------------|--------------|
| **MIND** | **25-53 µs** | **1× (baseline)** |
| PyTorch 2.9 | ~3,400 ms | **136,000× slower** |
| Mojo 0.25 | ~908 ms | **36,000× slower** |
| JAX 0.8 | ~430 ms | **17,000× slower** |

### Key Findings

1. **Compilation Speed**: MIND compiles 17,000-136,000× faster than competing frameworks
2. **Determinism**: Verified bit-identical outputs across 1000+ runs
3. **Autograd**: Compile-time differentiation vs runtime tape-based
4. **O(1) Scaling**: Compile time independent of tensor dimensions

## Roadmap

See `/docs/benchmarks/compiler_performance.md` for:
- Detailed results
- Competitive analysis
- Next steps
- Reproducibility instructions
