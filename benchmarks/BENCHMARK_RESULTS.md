# MIC/MAP Format Benchmark Results

**Date:** December 27, 2025

## Summary

| Format/Protocol | vs Baseline | Token Reduction |
|-----------------|-------------|-----------------|
| **MIC** vs JSON | **5.3x** smaller | 226 tokens saved |
| **MAP** vs JSON-RPC | **4.3x** smaller | 193 tokens saved |

---

## MIC vs JSON - IR Serialization

| Format | Size (bytes) | Tokens | Lines |
|--------|--------------|--------|-------|
| JSON (pretty) | 1,115 | 278 | 91 |
| JSON (compact) | 512 | 128 | 1 |
| **MIC** | **209** | **52** | **16** |

**Result: MIC is 5.3x more token-efficient than JSON**

### Sample: MIC Format
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

### Sample: Equivalent JSON
```json
{
  "version": 1,
  "symbols": ["input", "weight", "bias", "output"],
  "types": [
    {"id": 0, "dtype": "f32", "shape": ["B", 784]},
    {"id": 1, "dtype": "f32", "shape": [784, 256]},
    {"id": 2, "dtype": "f32", "shape": [256]},
    {"id": 3, "dtype": "f32", "shape": ["B", 256]}
  ],
  "nodes": [
    {"id": 0, "op": "param", "symbol": 0, "type": 0},
    {"id": 1, "op": "param", "symbol": 1, "type": 1},
    {"id": 2, "op": "param", "symbol": 2, "type": 2},
    {"id": 3, "op": "matmul", "inputs": [0, 1], "type": 3},
    {"id": 4, "op": "add", "inputs": [3, 2], "type": 3},
    {"id": 5, "op": "relu", "inputs": [4], "type": 3}
  ],
  "outputs": [5]
}
```

---

## MAP vs JSON-RPC - Protocol Communication

| Protocol | Size (bytes) | Tokens | Lines |
|----------|--------------|--------|-------|
| **MAP** | **234** | **58** | **17** |
| JSON-RPC | 1,004 | 251 | 63 |

**Result: MAP is 4.3x more token-efficient than JSON-RPC**

### Sample: MAP Session
```
@1 hello mic=1 map=1
=1 ok version=1.0 features=[patch,check,dump]
@2 load <<EOF
mic@1
T0 f32
N0 const.f32 1.0 T0
N1 const.f32 2.0 T0
N2 add N0 N1 T0
O N2
EOF
=2 ok nodes=3
@3 check
=3 ok valid=true
@4 dump
=4 ok mic@1...
@5 bye
=5 ok
```

### Sample: Equivalent JSON-RPC Session
```json
{"jsonrpc": "2.0", "method": "hello", "params": {"mic_version": 1, "map_version": 1}, "id": 1}
{"jsonrpc": "2.0", "result": {"version": "1.0", "features": ["patch", "check", "dump"]}, "id": 1}
{"jsonrpc": "2.0", "method": "load", "params": {"module": {...}}, "id": 2}
{"jsonrpc": "2.0", "result": {"nodes": 3}, "id": 2}
{"jsonrpc": "2.0", "method": "check", "id": 3}
{"jsonrpc": "2.0", "result": {"valid": true}, "id": 3}
...
```

---

## Why Token Efficiency Matters for AI Agents

1. **Cost Reduction**: LLM API costs are per-token. 5x reduction = 5x cost savings.
2. **Context Window**: More IR fits in limited context windows (128k, 200k tokens).
3. **Latency**: Fewer tokens = faster response times.
4. **Accuracy**: Less noise = better model understanding of IR structure.

---

## MIC Design Advantages

| Feature | MIC | JSON |
|---------|-----|------|
| Line-oriented | Yes (git-friendly) | No |
| Stable IDs | Yes (N0, N1, N2...) | No |
| Nested structure | Flat | Deep nesting |
| Human readable | High | Medium |
| Incremental patching | Native support | Requires full replace |
| Comments | Supported (#) | Not supported |

---

## MAP Design Advantages

| Feature | MAP | JSON-RPC |
|---------|-----|----------|
| Request format | `@seq cmd args` | `{"jsonrpc":"2.0",...}` |
| Response format | `=seq result` | `{"jsonrpc":"2.0",...}` |
| Heredoc support | Native (`<<EOF`) | None |
| Overhead per message | ~10 bytes | ~50+ bytes |
| Error format | `=seq err msg="..."` | `{"error":{"code":...}}` |

---

## Methodology

- Token count estimated at ~4 characters per token (GPT-style)
- Tests run on identical IR structures
- Protocol comparison uses equivalent 5-command sessions
- Benchmark script: `benchmarks/format_benchmark.py`

---

## Reproduction

```bash
cd mind-main
python benchmarks/format_benchmark.py
```
