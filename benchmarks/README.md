# MIND Performance Benchmarks

This directory contains benchmark results and performance analysis for the MIND compiler and runtime.

## Latest Results

**Last Updated**: 2025-12-21

### Compiler Performance

| Metric | Value | Comparison |
|--------|-------|------------|
| **Compilation Speed** | **~30 µs** | 17,000x - 345,000x faster than PyTorch 2.0 |
| **Throughput** | **~33,000 compiles/sec** | Interactive development ready |
| **Scaling** | **O(1) with tensor size** | Compile-time independent of data dimensions |

See **[Compiler Performance Report](../docs/benchmarks/compiler_performance.md)** for detailed analysis.

## Running Benchmarks

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

## Roadmap

See `/docs/benchmarks/compiler_performance.md` for:
- Detailed results
- Competitive analysis
- Next steps
- Reproducibility instructions
