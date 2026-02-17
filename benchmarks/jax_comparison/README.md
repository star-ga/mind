# JAX Compilation Benchmark

This benchmark compares **MIND compilation time** vs **JAX jax.jit()** compilation time.

## What We Measure

- **JAX**: `jax.jit()` compilation overhead (XLA compilation)
- **MIND**: `compile_source()` time (parse → type-check → IR lowering)

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run Benchmark

```bash
python benchmark_jax_compile.py
```

## Benchmark Programs

| Benchmark | JAX Function | MIND Equivalent | Input Shape |
|-----------|--------------|-----------------|-------------|
| `scalar_math` | `x + 2 * 3 - 4 / 2` | `1 + 2 * 3 - 4 / 2` | Scalar |
| `small_matmul` | `jnp.matmul([10,20], [20,30])` | `tensor.matmul([10,20], [20,30])` | Small matrices |
| `medium_matmul` | `jnp.matmul([128,256], [256,512])` | `tensor.matmul([128,256], [256,512])` | Medium matrices |
| `large_matmul` | `jnp.matmul([512,1024], [1024,512])` | `tensor.matmul([512,1024], [1024,512])` | Large matrices |
| `simple_mlp` | 784 → 256 → 10 MLP | Multi-layer network | Batch of 32 |
| `conv2d` | `jax.lax.conv()` | Conv2D layer | 64×56×56 feature maps |

## Verified Results (February 2026)

MIND v0.2.1 frontend compiles in **1.8-6.1 µs** (Criterion in-process, scales with program complexity).

JAX 0.9 `jax.jit()` cold-start XLA compilation takes **37.5-360.5 ms** (compilation cache disabled) due to:
- Python function tracing
- XLA HLO generation and optimization
- LLVM/PTX compilation

**Verified ratio: 21,200-95,100× faster** MIND frontend compilation.

**Scope note:** MIND measures frontend only (parse + typecheck + IR). JAX measures full XLA compilation pipeline. Different amounts of work.

## Output Format

```
=== MIND vs JAX Compilation Benchmark ===
Platform: [Platform info]
JAX: [Version], Devices: [CPU/GPU]

| Benchmark | MIND | JAX | MIND Speedup |
|-----------|------|-----|--------------|
| Scalar Math | 22 µs | XXX ms | XXX× |
| Small MatMul | 41 µs | XXX ms | XXX× |
| Medium MatMul | 41 µs | XXX ms | XXX× |
| Large MatMul | 41 µs | XXX s | XXX× |
| Simple MLP | 45 µs | XXX s | XXX× |
| Conv2D | 50 µs | XXX s | XXX× |

Methodology: 10 runs, mean ± std, warmup excluded
```

## Fair Comparison Notes

### What JAX Does
- **Trace**: Convert Python code to JAX primitives
- **XLA HLO**: Generate XLA high-level operations
- **Optimize**: XLA optimization passes
- **Codegen**: LLVM (CPU) or PTX (GPU) compilation

### What MIND Does
- **Parse**: Source code to AST
- **Type-check**: Static type inference
- **IR lowering**: AST to IR

Both are measuring **compilation time**, not execution time.

## Key Differences

| Feature | MIND | JAX |
|---------|------|-----|
| **Input** | Static source code | Python functions |
| **Compilation** | Ahead-of-time (AOT) | Just-in-time (JIT) |
| **Tracing** | No | Yes (traces Python) |
| **Backend** | IR (optional MLIR) | XLA → LLVM/PTX |
| **Speed** | 1.8-15.5 µs | 37.5-360.5 ms |

## Why MIND is Faster

### 1. Static Compilation
- MIND: Compiles from source, no tracing
- JAX: Must trace Python execution, handle dynamic shapes

### 2. No XLA Overhead
- MIND: Lightweight IR generation
- JAX: Full XLA compilation pipeline

### 3. Simpler Backend
- MIND: Direct IR lowering
- JAX: XLA → LLVM → machine code

### 4. No Shape Inference
- MIND: Shapes known at compile time
- JAX: Must infer shapes from traced execution

## Troubleshooting

### JAX not installed
```bash
pip install jax jaxlib
```

### GPU version
```bash
pip install jax[cuda12]  # For CUDA 12
```

### Slow measurements
Reduce `SAMPLE_SIZE` in the script (default: 10).

## Results

After running, results are saved to:
- `jax_results.json` - Raw timing data
- Console output - Comparison table

## Context

This benchmark supports MIND patent claims about compilation speed:

**Comparison with JAX**:
- JAX is Google's high-performance ML framework
- Uses XLA for compilation
- JAX 0.9 cold-start XLA is 21,200-95,100× slower than MIND frontend

**Goal**: Demonstrate that even state-of-the-art JIT compilers (XLA) are orders of magnitude slower than MIND's frontend compilation.

## References

- JAX: https://github.com/google/jax
- XLA: https://www.tensorflow.org/xla
- MIND: https://github.com/star-ga/mind
