# Performance Guide

> **Audience:** MIND users optimizing model performance

## Optimization Levels

| Flag | Use Case | Deterministic |
|------|----------|---------------|
| `--debug` | Development, debugging | Yes |
| `--release` | Production deployment | Yes |
| `--release --fast-math` | Maximum speed (non-certified) | No |

```bash
# Development (fast compile, slow run)
mindc model.mind -o model

# Production (optimized, deterministic)
mindc --release model.mind -o model

# Maximum performance (sacrifices reproducibility)
mindc --release --fast-math model.mind -o model
```

## Profiling Your Model

```bash
# Generate profile data
mindc --release --profile model.mind -o model
./model --profile-output=profile.json

# View flamegraph (requires flamegraph tool)
./model --profile=flamegraph > flame.svg
```

## Common Optimizations

### 1. Batch Size

Larger batches improve hardware utilization:

```mind
// Slower: single sample inference
let output = model(input<[1, 224, 224, 3]>);

// Faster: batched inference
let output = model(input<[32, 224, 224, 3]>);
```

### 2. Memory Layout

Ensure tensors are contiguous for optimal cache behavior:

```mind
let t = transpose(x);       // May create strided view
let t_fast = contiguous(t); // Force contiguous layout
```

### 3. MLIR Optimization Level

```bash
# Default optimization
mindc --release model.mind

# Aggressive MLIR optimization
mindc --release --mlir-opt=3 model.mind
```

## Benchmarking

```bash
# Warmup runs + timing
mindc --release model.mind -o model
./model --benchmark --warmup=10 --iterations=100
```

Output includes:
- Mean inference time
- Standard deviation
- Throughput (samples/sec)
- Memory high watermark

## Memory Profiling

```bash
# Track allocations
mindc --release --trace-alloc model.mind -o model
./model --memory-profile=memory.json
```

## CPU vs GPU

| Workload | Recommended Backend |
|----------|---------------------|
| Small models (<1M params) | CPU |
| Large batches | GPU |
| Real-time inference | CPU (lower latency) |
| Training | GPU |

```bash
# Force CPU backend
mindc --release --target=cpu model.mind

# Enable GPU (requires MLIR CUDA)
mindc --release --target=cuda model.mind
```

## Tips for Regulated Deployments

1. **Always use `--release`** (not `--release --fast-math`) for determinism
2. **Profile in production conditions** (same hardware, same batch sizes)
3. **Document hardware specs** in benchmark reports
4. **Use `--release-safe`** to retain bounds checks in production if required

## Performance vs Determinism

MIND defaults to deterministic execution. If you need maximum performance and can sacrifice bit-exact reproducibility:

```bash
# Enable fast math optimizations
mindc --release --fast-math model.mind
```

**Trade-offs:**
- Fused multiply-add (FMA) instructions
- Relaxed floating-point ordering
- SIMD optimizations that may differ across runs

See [docs/benchmarks.md](benchmarks.md) for official performance numbers.
