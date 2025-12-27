#!/usr/bin/env python3
"""
MIC vs JSON vs TOML Benchmark
Compares token efficiency, file size, and parse speed
"""

import json
import time
import sys

try:
    import toml
    HAS_TOML = True
except ImportError:
    HAS_TOML = False
    print("Note: toml not installed, skipping TOML benchmarks")

# Sample IR module - simple neural network layer
SAMPLE_IR = {
    "version": 1,
    "symbols": ["input", "weight", "bias", "output"],
    "types": [
        {"id": 0, "dtype": "f32", "shape": [None, 784]},
        {"id": 1, "dtype": "f32", "shape": [784, 256]},
        {"id": 2, "dtype": "f32", "shape": [256]},
        {"id": 3, "dtype": "f32", "shape": [None, 256]},
    ],
    "nodes": [
        {"id": 0, "op": "param", "symbol": 0, "type": 0},
        {"id": 1, "op": "param", "symbol": 1, "type": 1},
        {"id": 2, "op": "param", "symbol": 2, "type": 2},
        {"id": 3, "op": "matmul", "inputs": [0, 1], "type": 3},
        {"id": 4, "op": "add", "inputs": [3, 2], "type": 3},
        {"id": 5, "op": "relu", "inputs": [4], "type": 3},
    ],
    "outputs": [5]
}

# Equivalent MIC format
SAMPLE_MIC = """mic@1
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
"""

def count_tokens(text):
    """Approximate token count (GPT-style tokenization estimate)"""
    # Rough estimate: ~4 chars per token for code
    return len(text) // 4

def benchmark_format(name, serialize_fn, parse_fn, data, iterations=1000):
    """Benchmark a format"""
    # Serialize
    start = time.perf_counter()
    for _ in range(iterations):
        serialized = serialize_fn(data)
    serialize_time = (time.perf_counter() - start) / iterations * 1000  # ms
    
    # Parse
    start = time.perf_counter()
    for _ in range(iterations):
        parsed = parse_fn(serialized)
    parse_time = (time.perf_counter() - start) / iterations * 1000  # ms
    
    size = len(serialized) if isinstance(serialized, (str, bytes)) else len(str(serialized))
    tokens = count_tokens(serialized if isinstance(serialized, str) else serialized.decode())
    
    return {
        "name": name,
        "size_bytes": size,
        "tokens": tokens,
        "serialize_ms": serialize_time,
        "parse_ms": parse_time,
    }

def mic_serialize(data):
    """Serialize to MIC format"""
    lines = ["mic@1"]
    for i, sym in enumerate(data["symbols"]):
        lines.append(f'S{i} "{sym}"')
    for t in data["types"]:
        shape = ",".join("B" if s is None else str(s) for s in t["shape"])
        lines.append(f'T{t["id"]} [{t["dtype"]};{shape}]')
    for n in data["nodes"]:
        if n["op"] == "param":
            lines.append(f'N{n["id"]} param S{n["symbol"]} T{n["type"]}')
        elif n["op"] in ("matmul", "add"):
            inputs = " ".join(f'N{i}' for i in n["inputs"])
            lines.append(f'N{n["id"]} {n["op"]} {inputs} T{n["type"]}')
        elif n["op"] == "relu":
            lines.append(f'N{n["id"]} relu N{n["inputs"][0]} T{n["type"]}')
    for o in data["outputs"]:
        lines.append(f'O N{o}')
    return "\n".join(lines)

def mic_parse(text):
    """Parse MIC format (simplified)"""
    result = {"symbols": [], "types": [], "nodes": [], "outputs": []}
    for line in text.strip().split("\n"):
        if line.startswith("mic@"):
            result["version"] = int(line[4:])
        elif line.startswith("S"):
            result["symbols"].append(line.split('"')[1])
        elif line.startswith("T"):
            result["types"].append(line)
        elif line.startswith("N"):
            result["nodes"].append(line)
        elif line.startswith("O"):
            result["outputs"].append(line)
    return result

print("=" * 60)
print("MIC vs JSON vs TOML Benchmark")
print("=" * 60)
print()

# JSON benchmark
json_result = benchmark_format(
    "JSON",
    lambda d: json.dumps(d),
    lambda s: json.loads(s),
    SAMPLE_IR
)

# JSON compact
json_compact_result = benchmark_format(
    "JSON (compact)",
    lambda d: json.dumps(d, separators=(',', ':')),
    lambda s: json.loads(s),
    SAMPLE_IR
)

# MIC benchmark
mic_result = benchmark_format(
    "MIC",
    mic_serialize,
    mic_parse,
    SAMPLE_IR
)

# TOML benchmark (if available)
results = [json_result, json_compact_result, mic_result]

if HAS_TOML:
    # TOML needs string keys
    toml_data = json.loads(json.dumps(SAMPLE_IR))
    toml_result = benchmark_format(
        "TOML",
        lambda d: toml.dumps(d),
        lambda s: toml.loads(s),
        toml_data
    )
    results.append(toml_result)

# Print results
print(f"{'Format':<15} {'Size (bytes)':<14} {'Tokens':<10} {'Serialize':<12} {'Parse':<10}")
print("-" * 60)

for r in results:
    print(f"{r['name']:<15} {r['size_bytes']:<14} {r['tokens']:<10} {r['serialize_ms']:.3f} ms     {r['parse_ms']:.3f} ms")

print()
print("=" * 60)
print("Comparison vs JSON")
print("=" * 60)

json_size = json_result["size_bytes"]
json_tokens = json_result["tokens"]

for r in results:
    size_ratio = json_size / r["size_bytes"]
    token_ratio = json_tokens / r["tokens"]
    print(f"{r['name']:<15} Size: {size_ratio:.2f}x  Tokens: {token_ratio:.2f}x")

print()
print("=" * 60)
print("Sample Outputs")
print("=" * 60)

print("\n--- MIC Format ---")
print(mic_serialize(SAMPLE_IR))

print("\n--- JSON Format ---")
print(json.dumps(SAMPLE_IR, indent=2)[:500] + "...")

print()
print("=" * 60)
print("MIC Advantages for AI Agents")
print("=" * 60)
print(f"- Token reduction: {json_tokens / mic_result['tokens']:.1f}x fewer tokens than JSON")
print(f"- Size reduction: {json_size / mic_result['size_bytes']:.1f}x smaller than JSON")
print("- Line-oriented: easy git diffs, one node per line")
print("- Stable IDs: safe for incremental patching")
print("- Human readable: no nested brackets/braces")
