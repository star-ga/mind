# MIND Benchmark Results

**Last Updated:** February 17, 2026
**Reference Platform:** Ubuntu 24.04, a commodity x86 CPU, 64GB DDR4, Ampere-class GPU, CUDA 12.8

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
| CPU | a commodity x86 CPU |
| Memory | 64GB DDR4 |
| GPU | NVIDIA Ampere-class GPU |
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

The standalone MIND values in this report are T1 frontend measurements. The
committed PyTorch record (`pytorch_comparison/pytorch_results.json`) is
PyTorch 2.9.1+cpu with `cuda_available:false`; it compares an in-process
PyTorch call with a MIND CLI subprocess and records 30.8–52.6× wall-clock
ratios. Those are cross-harness observations, not tier-matched speedups.

The former PyTorch GPU table has no matching committed GPU artifact. The JAX
cold-start record is committed and contains JAX `37.523–360.513 ms` mean timings
with T1 MIND `1.77/2.95/6.15 µs` values, but those are cross-tier historical
observations rather than a current speedup claim. The raw Mojo timings lack
matched provenance. No tier-matched external speedup is established by these
records until an artifact records both sides' timing scope and environment.

### Historical: Cross-harness comparison (January 19, 2026)

*Note: Subprocess overhead adds ~1.3ms to MIND measurements. These numbers are kept for reference.*

| Benchmark | PyTorch (inductor) | MIND (subprocess) | Recorded ratio |
|-----------|-------------------|-------------------|---------|
| scalar_math | 42.8 ms | 1.4 ms | **31×** |
| small_matmul | 61.5 ms | 1.3 ms | **46×** |
| medium_matmul | 48.4 ms | 1.3 ms | **37×** |
| large_matmul | 52.4 ms | 1.4 ms | **39×** |

These are recorded wall-clock ratios from different harnesses, not a
tier-matched speedup claim. The current raw artifact is CPU-only; see
`pytorch_comparison/pytorch_results.json`.

### Reference Criterion Benchmarks - Linux (February 17, 2026)

**Platform:** Ubuntu 24.04, a commodity x86 CPU, 64GB DDR4, NVIDIA Ampere-class GPU

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

| Format | Tokens (`cl100k_base`) | vs JSON | Reduction | Parse Speed |
|--------|------------------------|---------|-----------|-------------|
| JSON | 400 | 1.0x | baseline | 5.31 us |
| TOML | 259 | 1.5x | 35% | 137.06 us |
| TOON | 144 | 2.8x | 64% | 2.67 us |
| mic@1 | 119 | 3.4x | 70% | 2.26 us |
| **mic@2** | **71** | **5.6x** | **82%** | **—** |

**mic@2 is the most token-efficient text format. The canonical `mic@3` binary of the same
network is 87 bytes:** `mindc mlp.mind --emit-mic3 mlp.mic3` at compiler `83ed6a11`, where
`mlp.mind` is

```text
let input: Tensor[f32,(B,784)] = 0;
let weight: Tensor[f32,(784,256)] = 0;
let bias: Tensor[f32,(256)] = 0;
tensor.relu(tensor.matmul(input, weight) + bias)
```

(output SHA-256 `73beda1df272f7cb54bdf15270ced149517cad954919d96840461dcdcbac2d28`). No byte
ratio against JSON is stated: the compiler has no JSON encoding of that IR.

Parse speed above is a Python reference-parser micro-benchmark: JSON is parsed by the
C `json` module, MIC/TOON by pure-Python reference parsers. It is indicative of format
shape, not a like-for-like parser comparison, and no claim below rests on it.

---

## Token Efficiency Chart

```
Tokens (fewer = better, cl100k_base)

JSON     ████████████████████████████████████████████████████████  400
TOML     ████████████████████████████████████                      259
TOON     ████████████████████                                      144
MIC      ████████████████                                          119
         ├─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┤
         0        50       100       150       200       250       300       400
```

## Size Comparison Chart (bytes)

```
Size in Bytes (smaller = better)

JSON     ████████████████████████████████████████████████████████  1117
TOML     ██████████████████████████████                             606
TOON     █████████████                                              269
MIC      ██████████                                                 210
         ├─────────┼─────────┼─────────┼─────────┼─────────┼─────────┤
         0       200       400       600       800      1000      1200
```

## Reduction vs JSON Chart

```
Token Reduction vs JSON (higher = better, cl100k_base)

JSON     ▓                                                          1.0x
TOML     ▓▓▓▓▓▓▓                                                    1.5x
TOON     ▓▓▓▓▓▓▓▓▓▓▓▓▓▓                                             2.8x
MIC      ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓                                          3.4x
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

## Cost Impact (LLM Token Pricing)

Every input to this section is committed and the arithmetic is re-derived by a
gate, because it previously was not: README published an annual saving of $6,780
while this page said $396 for the same reference IR — a 17.1x gap that back-solved to
a $0.030/1K token price stated in no file in the tree, and the two numbers rested
on a 4-chars-per-token estimate rather than a tokenizer. Both are now measured
with `cl100k_base` and priced from `config/token_pricing.toml`.

Price input: **$0.00175 per 1K input tokens** (`config/token_pricing.toml`, reviewed
2026-02-17). That value is a modelling assumption carried forward from this page,
not a vendor quote — the published sentence restates it inline so the figure can
never be read without its assumption.

Workload: one reference IR document sent to a model once per operation, one
million operations per year. No output tokens, retries or context repetition are
modelled, so the figure is a floor on the serialization saving alone.

```
Annual Cost per 1M IR Operations (lower = better, cl100k_base tokens)

JSON     ████████████████████████████████████████████████████████  $700
TOML     ████████████████████████████████████                      $453
TOON     ████████████████████                                      $252
MIC      ████████████████                                          $208
         ├─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼────────┤
         $0      $100      $200      $300      $400      $500      $600   $700
```

| Format | Tokens/IR | Cost/1K IRs | Annual (1M IRs) | Savings vs JSON |
|--------|-----------|-------------|-----------------|-----------------|
| JSON | 400 | $0.70 | $700 | - |
| TOML | 259 | $0.45 | $453 | $247 (35%) |
| TOON | 144 | $0.25 | $252 | $448 (64%) |
| **`mic@1`** | **119** | **$0.21** | **$208** | **$492 (70%)** |

**At $0.00175/1K input tokens (2026-02-17), MIC saves $492/year per million IR operations vs JSON.**

---

## Methodology

- Token count: real `cl100k_base` BPE tokenization (`tiktoken`). The older
  `len(text) // 4` estimate is still reported by the benchmark as a secondary
  column, clearly labelled — it must never back a dollar figure, because it
  overstates MIC's advantage (5.3x estimated vs 3.4x measured).
- Price: `config/token_pricing.toml`, with its source and review date.
- Test data: 6-node neural network layer (param, matmul, add, relu)
- All formats encode identical IR structure
- Benchmark script: `benchmarks/mic_map_benchmark_v2.py` (writes
  `benchmarks/mic_map_benchmark_results.json`, including the `cost_model` block
  the published sentence is rendered from)
- Gate: `scripts/check_claims.py` recomputes the saving from the price config and
  the benchmark's token counts and fails the build if any declared surface carries
  a different figure. Proof that the comparison bites:
  `python3 tests/check_claims_gate_tests.py`.

## Reproduction

```bash
pip install 'tiktoken>=0.7.0'
python3 benchmarks/mic_map_benchmark_v2.py   # refreshes the measurement + cost model
python3 scripts/check_claims.py              # fails if a surface disagrees
```

The Rust-Criterion side is recorded separately. The CI-equivalent sweep

```bash
cargo bench --no-default-features -- --output-format bencher --measurement-time 8
```

is captured verbatim in [`criterion_ci_sweep.txt`](criterion_ci_sweep.txt), which
also lists every bench input the run could **not** measure and why — a fixture
that needs a feature this build lacks is skipped by name, never silently dropped
and never allowed to abort the sweep.

---

## References

- [TOON Format](https://github.com/toon-format/toon) - Token-Oriented Object Notation
- [MIC Specification](https://github.com/star-ga/mind-spec/blob/main/rfcs/0001-mindir-compact.md)
- [MAP Specification](https://github.com/star-ga/mind-spec/blob/main/rfcs/0002-mind-ai-protocol.md)
