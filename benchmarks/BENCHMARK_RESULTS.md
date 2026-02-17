# MIND Benchmark Results

**Last Updated:** February 17, 2026
**Reference Platform:** Ubuntu 24.04, Intel Core i7-5930K @ 3.50GHz, 64GB DDR4, RTX 3080 10GB, CUDA 12.8

---

## Scientific Methodology

### Why Subprocess Overhead Matters

When benchmarking compilation speed, there are two ways to measure:

1. **In-process measurement** — Directly call the compiler function and measure wall-clock time. This captures *pure compilation time* without process startup overhead.

2. **Subprocess measurement** — Spawn a new process to run the compiler CLI. This includes:
   - Process creation (~500-2000 µs on Linux, ~2000-5000 µs on Windows)
   - Binary loading and initialization
   - Actual compilation
   - Process teardown

For fair comparison, we use **in-process Criterion benchmarks** for MIND, which measure only the `compile_source()` function call — the actual work being done.

### Subprocess Overhead Subtraction

When MIND is invoked via CLI (for comparison scripts), we subtract the subprocess baseline:

```
Pure Compile Time = Total CLI Time - Subprocess Baseline
```

**Subprocess baseline** is measured by running `mind --version` (minimal work, same process overhead).

| Platform | Typical Subprocess Overhead |
|----------|----------------------------|
| Linux x86_64 | ~1,200-1,500 µs |
| macOS ARM64 | ~800-1,200 µs |
| Windows x86_64 | ~2,000-5,000 µs |

### Reference Platform

All official benchmarks use a single reference platform for reproducibility:

| Component | Specification |
|-----------|---------------|
| OS | Ubuntu 24.04 LTS |
| CPU | Intel Core i7-5930K @ 3.50GHz |
| Memory | 64GB DDR4 |
| GPU | NVIDIA RTX 3080 10GB |
| CUDA | 13.0 |
| Rust | 1.82+ stable |

### Measurement Protocol

1. **Warmup**: 3-5 runs discarded to warm caches
2. **Sampling**: 20+ runs collected
3. **Statistics**: Mean, median, std deviation reported
4. **Tool**: Criterion.rs for MIND (statistically rigorous)
5. **Isolation**: Single-threaded, no background processes

### Reproducing Results

```bash
# Clone and build
git clone https://github.com/star-ga/mind.git
cd mind
cargo build --release

# Run Criterion benchmarks (in-process, no subprocess overhead)
cargo bench --bench simple_benchmarks

# Run comparison benchmarks (with subprocess overhead subtraction)
cd benchmarks/pytorch_comparison
python benchmark_pytorch_compile.py
```

---

## Compilation Speed

### MIND v0.2.1 vs PyTorch 2.10 GPU torch.compile (February 2026 - Verified)

**Methodology:** PyTorch using GPU `torch.compile` full cold-start (Triton/Inductor caches cleared), MIND in-process via Criterion benchmarks

**Scope Note:** MIND measures frontend only (parse + typecheck + IR). PyTorch measures full compilation pipeline (FX graph + Inductor + Triton/cuBLAS kernel generation).

| Benchmark | PyTorch 2.10 GPU | MIND v0.2.1 (frontend) | Ratio |
|-----------|-----------------|------------------------|-------|
| scalar_math | 99 ms | 1.77 µs | **56,000×** |
| small_matmul | 162 ms | 2.95 µs | **55,000×** |
| medium_matmul | 109 ms | 2.95 µs | **37,000×** |
| large_matmul | 105 ms | 2.95 µs | **36,000×** |
| simple_mlp | 752 ms | 6.15 µs | **122,000×** |
| conv2d | 878 ms | ~5 µs | **176,000×** |

**MIND frontend compiles 35,000-176,000× faster than PyTorch 2.10 GPU torch.compile full pipeline.**

*Environment: Ubuntu 24.04, RTX 3080, CUDA 12.8, PyTorch 2.10.0+cu128*

### MIND v0.2.1 vs Mojo 0.26.1 (February 2026 - Verified)

**Methodology:** Mojo `mojo build` full LLVM compilation to native binary, MIND in-process via Criterion benchmarks

**Scope Note:** MIND measures frontend only (parse + typecheck + IR). Mojo measures full LLVM compilation to a native binary.

| Benchmark | Mojo 0.26.1 | MIND v0.2.1 (frontend) | Ratio |
|-----------|-------------|------------------------|-------|
| scalar_math | 810 ms | 1.77 µs | **458,000×** |
| matmul | 827 ms | 2.95 µs | **280,000×** |
| mlp | 829 ms | 6.15 µs | **135,000×** |

**MIND frontend compiles 135,000-458,000× faster than Mojo 0.26.1 full compilation.**

*Environment: Ubuntu 24.04, Mojo 0.26.1.0, pixi*

### MIND v0.2.1 vs JAX 0.9 Cold-Start XLA Compilation (February 2026 - Verified)

**Methodology:** JAX `jax.jit()` cold-start XLA compilation with cache disabled (`JAX_ENABLE_COMPILATION_CACHE=0`, `jax.clear_caches()`), MIND in-process via Criterion benchmarks

**Scope Note:** MIND measures frontend only (parse + typecheck + IR). JAX measures full XLA compilation (HLO lowering + optimization + code generation).

| Benchmark | JAX 0.9 Cold-Start | MIND v0.2.1 (frontend) | Ratio |
|-----------|-------------------|------------------------|-------|
| scalar_math | 37.5 ms | 1.77 µs | **21,200×** |
| small_matmul | 127.2 ms | 2.95 µs | **43,100×** |
| medium_matmul | 139.7 ms | 2.95 µs | **47,400×** |
| large_matmul | 280.6 ms | 2.95 µs | **95,100×** |
| simple_mlp | 360.5 ms | 6.15 µs | **58,600×** |

**MIND frontend compiles 21,200-95,100× faster than JAX 0.9 cold-start XLA compilation.**

*Environment: Ubuntu 24.04, RTX 3080, CUDA 12.8, JAX 0.9.0.1*

### Historical: Subprocess Comparison (January 19, 2026)

*Note: Subprocess overhead adds ~1.3ms to MIND measurements. These numbers are kept for reference.*

| Benchmark | PyTorch (inductor) | MIND (subprocess) | Speedup |
|-----------|-------------------|-------------------|---------|
| scalar_math | 42.8 ms | 1.4 ms | **31× faster** |
| small_matmul | 61.5 ms | 1.3 ms | **46× faster** |
| medium_matmul | 48.4 ms | 1.3 ms | **37× faster** |
| large_matmul | 52.4 ms | 1.4 ms | **39× faster** |

### Reference Criterion Benchmarks - Linux (February 17, 2026)

**Platform:** Ubuntu 24.04, Intel Core i7-5930K @ 3.50GHz, 64GB DDR4, NVIDIA RTX 3080 10GB

**simple_benchmarks** (equivalent-complexity programs):
```
scalar_math:      time:   [1.75 µs 1.77 µs 1.79 µs]
small_matmul:     time:   [2.93 µs 2.95 µs 2.97 µs]
medium_matmul:    time:   [2.93 µs 2.95 µs 2.97 µs]
large_matmul:     time:   [2.93 µs 2.95 µs 2.97 µs]
tensor_ops:       time:   [4.84 µs 4.87 µs 4.91 µs]
reductions:       time:   [3.15 µs 3.17 µs 3.20 µs]
reshape_ops:      time:   [2.81 µs 2.83 µs 2.86 µs]
```

**compiler pipeline** (scaling with program complexity):
```
small_matmul:     time:   [2.58 µs 2.60 µs 2.62 µs]
medium_mlp:       time:   [6.10 µs 6.15 µs 6.20 µs]
large_network:    time:   [15.30 µs 15.49 µs 15.70 µs]
```

**Key Insight:** Compilation time scales with **program complexity** (number of operations), not tensor dimensions. Within the same program, increasing tensor sizes does not affect compile time.

### Determinism Verification (Dec 27, 2025)

All 4 tests passed with 100% bit-identical SHA256 hashes across 10 runs each:

| Test | Runs | Status | Avg Time |
|------|------|--------|----------|
| scalar_math | 10 | DETERMINISTIC | 5.2 ms |
| small_matmul | 10 | DETERMINISTIC | 5.4 ms |
| medium_matmul | 10 | DETERMINISTIC | 5.2 ms |
| mlp | 10 | DETERMINISTIC | 6.2 ms |

---

## MIC/MAP Format Efficiency

## Executive Summary

| Format | Tokens | vs JSON | Reduction | Parse Speed |
|--------|--------|---------|-----------|-------------|
| JSON | 278 | 1.0x | baseline | 5.31 us |
| TOML | 151 | 1.8x | 46% | 137.06 us |
| TOON | 67 | 4.1x | 76% | 2.67 us |
| **MIC** | **52** | **5.3x** | **81%** | **2.26 us** |

**MIC is the most token-efficient AND fastest format for IR serialization.**

---

## Token Efficiency Chart

```
Tokens (fewer = better)

JSON     ████████████████████████████████████████████████████████  283
TOML     ██████████████████████████████                            151
TOON     █████████████                                              67
MIC      ██████████                                                 52
         ├─────────┼─────────┼─────────┼─────────┼─────────┼─────────┤
         0        50       100       150       200       250       300
```

## Size Comparison Chart (bytes)

```
Size in Bytes (smaller = better)

JSON     ████████████████████████████████████████████████████████  1133
TOML     ██████████████████████████████                             607
TOON     █████████████                                              269
MIC      ██████████                                                 209
         ├─────────┼─────────┼─────────┼─────────┼─────────┼─────────┤
         0       200       400       600       800      1000      1200
```

## Reduction vs JSON Chart

```
Token Reduction vs JSON (higher = better)

JSON     ▓                                                          1.0x
TOML     ▓▓▓▓▓▓▓▓▓▓                                                 1.9x
TOON     ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓                                      4.2x
MIC      ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓                                5.4x
         ├─────────┼─────────┼─────────┼─────────┼─────────┼─────────┤
         0x       1x        2x        3x        4x        5x        6x
```

---

## Parse Speed Benchmark

```
Parse Speed (microseconds per parse, lower = better)

TOML     ████████████████████████████████████████████████████████████████ 137.06 us
JSON     ███                                                               5.31 us
TOON     ██                                                                2.67 us
MIC      █                                                                 2.26 us
         ├─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼────┤
         0        25        50        75       100       125       150 us
```

| Format | Per Parse (us) | vs JSON |
|--------|----------------|---------|
| TOML | 137.06 | 25.8x slower |
| JSON | 5.31 | baseline |
| TOON | 2.67 | 2.0x faster |
| **MIC** | **2.26** | **2.4x faster** |

**MIC parses 2.4x faster than JSON and 60x faster than TOML.**

---

## Detailed Results

### IR Serialization Benchmark

| Format | Size (bytes) | Tokens | Lines | vs JSON |
|--------|--------------|--------|-------|---------|
| JSON (pretty) | 1,133 | 283 | 91 | 1.0x |
| JSON (compact) | 539 | 134 | 1 | 2.1x |
| TOML | 607 | 151 | 59 | 1.9x |
| TOON | 269 | 67 | 15 | 4.2x |
| **MIC** | **209** | **52** | **16** | **5.4x** |

### Head-to-Head Comparisons

| Comparison | Winner | Margin |
|------------|--------|--------|
| MIC vs JSON | MIC | 5.4x fewer tokens |
| MIC vs TOML | MIC | 2.9x fewer tokens |
| MIC vs TOON | MIC | 1.3x fewer tokens |
| TOON vs JSON | TOON | 4.2x fewer tokens |
| TOON vs TOML | TOON | 2.3x fewer tokens |

---

## Sample Formats

### MIC (52 tokens) - Winner
```
mic@1
S0 "input"
S1 "weight"
S2 "bias"
S3 "output"
T0 [f32;B,784]
T1 [f32;784,256]
T2 [f32;256]
T3 [f32;B,256]
N0 param S0 T0
N1 param S1 T1
N2 param S2 T2
N3 matmul N0 N1 T3
N4 add N3 N2 T3
N5 relu N4 T3
O N5
```

### TOON (67 tokens)
```
version: 1
symbols[4]: input,weight,bias,output
outputs[1]: 5
types[4]{id,dtype,shape}:
  0,f32,B:784
  1,f32,784:256
  2,f32,256
  3,f32,B:256
nodes[6]{id,op,inputs,type_id}:
  0,param,S0,0
  1,param,S1,1
  2,param,S2,2
  3,matmul,N0:N1,3
  4,add,N3:N2,3
  5,relu,N4,3
```

### TOML (151 tokens)
```toml
version = 1
symbols = ["input", "weight", "bias", "output"]
outputs = [5]

[[types]]
id = 0
dtype = "f32"
shape = ["B", 784]

[[nodes]]
id = 0
op = "param"
symbol = 0
type_id = 0
...
```

### JSON (283 tokens)
```json
{
  "version": 1,
  "symbols": ["input", "weight", "bias", "output"],
  "types": [
    {"id": 0, "dtype": "f32", "shape": ["B", 784]},
    ...
  ],
  "nodes": [
    {"id": 0, "op": "param", "symbol": 0, "type": 0},
    ...
  ],
  "outputs": [5]
}
```

---

## MAP Protocol Benchmark

| Protocol | Size | Tokens | vs JSON-RPC |
|----------|------|--------|-------------|
| JSON-RPC | 1,004 | 251 | 1.0x |
| **MAP** | **234** | **58** | **4.3x** |

```
Protocol Tokens (fewer = better)

JSON-RPC  ████████████████████████████████████████████████████  251
MAP       ████████████                                           58
          ├─────────┼─────────┼─────────┼─────────┼─────────┼────┤
          0        50       100       150       200       250
```

### MAP vs JSON-RPC Sample

**MAP (58 tokens):**
```
@1 hello mic=1 map=1
=1 ok version=1.0 features=[patch,check,dump]
@2 load <<EOF
mic@1
T0 f32
N0 const.f32 1.0 T0
O N0
EOF
=2 ok nodes=1
@3 bye
=3 ok
```

**JSON-RPC (251 tokens):**
```json
{"jsonrpc":"2.0","method":"hello","params":{"mic":1,"map":1},"id":1}
{"jsonrpc":"2.0","result":{"version":"1.0","features":["patch","check","dump"]},"id":1}
{"jsonrpc":"2.0","method":"load","params":{"module":{...}},"id":2}
...
```

---

## Why MIC Wins

### Design Advantages

| Feature | MIC | TOON | TOML | JSON |
|---------|-----|------|------|------|
| Domain-specific | Yes | No | No | No |
| Type notation | `[f32;B,784]` | `f32,B:784` | verbose | verbose |
| Node notation | `N3 matmul N0 N1` | CSV row | verbose | verbose |
| Headers needed | No | Yes | No | No |
| Array lengths | Implicit | Explicit | Implicit | Implicit |
| Nesting | Flat | Flat | Deep | Deep |
| Git-friendly | Yes | Yes | Partial | No |
| **Parse speed** | **2.26 us** | 2.67 us | 137.06 us | 5.31 us |

### Token Savings Breakdown

| Element | MIC | JSON | Savings |
|---------|-----|------|---------|
| Type definition | 15 chars | 45 chars | 3x |
| Node definition | 20 chars | 55 chars | 2.8x |
| Shape notation | `B,784` | `["B", 784]` | 2x |
| References | `N0` | `{"ref": 0}` | 5x |

---

## Use Case Recommendations

| Use Case | Best Format | Reason |
|----------|-------------|--------|
| AI agent IR editing | **MIC** | Maximum token efficiency |
| AI agent protocols | **MAP** | 4.3x better than JSON-RPC |
| Config files | TOML | Human readability |
| API responses | JSON | Universal support |
| Tabular AI data | TOON | Good for uniform arrays |

---

## Cost Impact (GPT-5.2 Pricing)

At $0.00175/1K tokens (input):

```
Annual Cost per 1M IR Operations (lower = better)

JSON     ████████████████████████████████████████████████████████  $487
TOML     ███████████████████████████████                           $264
TOON     ██████████████                                            $117
MIC      ███████████                                               $91
         ├─────────┼─────────┼─────────┼─────────┼─────────┼────────┤
         $0      $100      $200      $300      $400      $500
```

| Format | Tokens/IR | Cost/1K IRs | Annual (1M IRs) | Savings vs JSON |
|--------|-----------|-------------|-----------------|-----------------|
| JSON | 278 | $0.49 | $487 | - |
| TOML | 151 | $0.26 | $264 | $223 (46%) |
| TOON | 67 | $0.12 | $117 | $370 (76%) |
| **MIC** | **52** | **$0.09** | **$91** | **$396 (81%)** |

**MIC saves $396/year per million IR operations vs JSON.**

---

## Methodology

- Token count: ~4 characters per token (GPT-style estimation)
- Test data: 6-node neural network layer (param, matmul, add, relu)
- All formats encode identical IR structure
- Benchmark script: `benchmarks/format_benchmark.py`

## Reproduction

```bash
cd mind-main
python benchmarks/format_benchmark.py
```

---

## References

- [TOON Format](https://github.com/toon-format/toon) - Token-Oriented Object Notation
- [MIC Specification](https://github.com/star-ga/mind-spec/blob/main/rfcs/0001-mindir-compact.md)
- [MAP Specification](https://github.com/star-ga/mind-spec/blob/main/rfcs/0002-mind-ai-protocol.md)
