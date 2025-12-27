# MIC/MAP Format Benchmark Results

**Date:** December 27, 2025

---

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

## Cost Impact (GPT-4 Pricing)

At $0.03/1K tokens (input):

```
Annual Cost per 1M IR Operations (lower = better)

JSON     ████████████████████████████████████████████████████████  $8,340
TOML     ████████████████████████████████                          $4,530
TOON     ██████████████                                            $2,010
MIC      ███████████                                               $1,560
         ├─────────┼─────────┼─────────┼─────────┼─────────┼────────┤
         $0      $2K       $4K       $6K       $8K      $10K
```

| Format | Tokens/IR | Cost/1K IRs | Annual (1M IRs) | Savings vs JSON |
|--------|-----------|-------------|-----------------|-----------------|
| JSON | 278 | $8.34 | $8,340 | - |
| TOML | 151 | $4.53 | $4,530 | $3,810 (46%) |
| TOON | 67 | $2.01 | $2,010 | $6,330 (76%) |
| **MIC** | **52** | **$1.56** | **$1,560** | **$6,780 (81%)** |

**MIC saves $6,780/year per million IR operations vs JSON.**

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
- [MIC Specification](https://github.com/cputer/mind-spec/blob/main/rfcs/0001-mindir-compact.md)
- [MAP Specification](https://github.com/cputer/mind-spec/blob/main/rfcs/0002-mind-ai-protocol.md)
