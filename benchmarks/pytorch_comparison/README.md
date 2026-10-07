# PyTorch Compilation Benchmark

This benchmark compares **MIND compilation time** vs **PyTorch torch.compile()** compilation time.

## What We Measure

- **PyTorch 2.10+**: `torch.compile()` compilation overhead on the selected device (caches cleared)
- **MIND**: CLI subprocess time in this harness. The separately recorded `compile_source()` Criterion baseline measures only parse → type-check → IR lowering.

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run Benchmark

```bash
python benchmark_pytorch_compile.py
```

## Benchmark Programs

| Benchmark | PyTorch Model | MIND Equivalent | Input Shape |
|-----------|---------------|-----------------|-------------|
| `scalar_math` | Simple arithmetic | `1 + 2 * 3 - 4 / 2` | `(1, 10)` |
| `small_matmul` | Linear(20, 30) | `tensor.matmul([10,20], [20,30])` | `(1, 10, 20)` |
| `medium_matmul` | Linear(256, 512) | `tensor.matmul([128,256], [256,512])` | `(1, 128, 256)` |
| `large_matmul` | Linear(1024, 512) | `tensor.matmul([512,1024], [1024,512])` | `(1, 512, 1024)` |
| `simple_mlp` | 784 → 256 → 10 MLP | Multi-layer network | `(1, 784)` |
| `conv2d` | ResNet-50 style Conv2D | Conv2D layer | `(1, 64, 56, 56)` |

## Committed evidence

[`pytorch_results.json`](pytorch_results.json) records PyTorch 2.9.1+cpu with
`cuda_available:false`. Its 10-sample rows record PyTorch means of 42.8–79.3 ms
and MIND CLI subprocess means of 1.31–1.51 ms, with derived wall-clock ratios
of 30.8–52.6×. PyTorch is measured in-process while MIND is spawned as a CLI,
so these are cross-harness observations rather than a tier-matched speedup.

No committed GPU result supports the former PyTorch 2.10 GPU headline. The
standalone MIND 1.8–15.5 µs Criterion values are T1 frontend measurements and
must not be divided by the process measurements above.

## Output Format

```
=== MIND vs PyTorch 2.0 Compilation Benchmark ===
Hardware: [CPU/GPU info]
Software: MIND 0.x.x, PyTorch 2.x.x, Python 3.x

| Benchmark | MIND | PyTorch 2.0 | MIND Speedup |
|-----------|------|-------------|--------------|
| Scalar Math | 22 µs | XXX ms | XXX× |
| Small MatMul | 41 µs | XXX ms | XXX× |
| Medium MatMul | 41 µs | XXX s | XXX× |
| Large MatMul | 41 µs | XXX s | XXX× |
| Simple MLP | 45 µs | XXX s | XXX× |
| Conv2D | 50 µs | XXX s | XXX× |

Methodology: 10 runs, mean ± std, warmup excluded
```

## Fair Comparison Notes

### What PyTorch Does
- Graph capture via TorchDynamo
- Symbolic tracing
- TorchInductor code generation
- Triton kernel generation (GPU)
- LLVM compilation (CPU)

### What MIND Does
- Parse source
- Type-check
- Lower to IR
- (Optional: MLIR lowering for execution)

Both are measuring **compilation time**, not execution time.

## Troubleshooting

### PyTorch too old
```bash
pip install 'torch>=2.13'
```

### CUDA errors
The benchmark automatically uses CPU if CUDA is unavailable.

### Slow measurements
Reduce `SAMPLE_SIZE` in the script (default: 10).

## Results

After running, results are saved to:
- `pytorch_results.json` - Raw timing data
- Console output - Comparison table

## Context

This benchmark supports MIND patent claims about compilation speed advantages:
- **Claims 1-5**: Core compilation system
- **Claims 11-15**: Fast compilation compared to prior art

**Goal**: Record reproducible, explicitly scoped comparisons. A GPU frontend
speedup claim requires a committed GPU artifact with matched timing scope.
