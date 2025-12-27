#!/usr/bin/env python3
"""
MIC/MAP vs JSON/TOML/TOON Benchmark
Compares token efficiency, file size, and parse speed
"""

import json
import time
import sys

# Fix encoding for Windows
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Try to import TOML parser
try:
    import tomllib  # Python 3.11+
    HAS_TOML = True
except ImportError:
    try:
        import tomli as tomllib  # Fallback
        HAS_TOML = True
    except ImportError:
        HAS_TOML = False

# ============================================================
# SAMPLE DATA
# ============================================================

# Neural network layer IR
SAMPLE_IR = {
    "version": 1,
    "symbols": ["input", "weight", "bias", "output"],
    "types": [
        {"id": 0, "dtype": "f32", "shape": ["B", 784]},
        {"id": 1, "dtype": "f32", "shape": [784, 256]},
        {"id": 2, "dtype": "f32", "shape": [256]},
        {"id": 3, "dtype": "f32", "shape": ["B", 256]},
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
O N5"""

# MAP protocol session vs JSON-RPC
MAP_SESSION = """@1 hello mic=1 map=1
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
=5 ok"""

JSON_RPC_SESSION = """{
  "jsonrpc": "2.0",
  "method": "hello",
  "params": {"mic_version": 1, "map_version": 1},
  "id": 1
}
{
  "jsonrpc": "2.0",
  "result": {"version": "1.0", "features": ["patch", "check", "dump"]},
  "id": 1
}
{
  "jsonrpc": "2.0",
  "method": "load",
  "params": {
    "module": {
      "version": 1,
      "types": [{"id": 0, "dtype": "f32"}],
      "nodes": [
        {"id": 0, "op": "const.f32", "value": 1.0, "type": 0},
        {"id": 1, "op": "const.f32", "value": 2.0, "type": 0},
        {"id": 2, "op": "add", "inputs": [0, 1], "type": 0}
      ],
      "outputs": [2]
    }
  },
  "id": 2
}
{
  "jsonrpc": "2.0",
  "result": {"nodes": 3},
  "id": 2
}
{
  "jsonrpc": "2.0",
  "method": "check",
  "id": 3
}
{
  "jsonrpc": "2.0",
  "result": {"valid": true},
  "id": 3
}
{
  "jsonrpc": "2.0",
  "method": "dump",
  "id": 4
}
{
  "jsonrpc": "2.0",
  "result": {"module": "..."},
  "id": 4
}
{
  "jsonrpc": "2.0",
  "method": "bye",
  "id": 5
}
{
  "jsonrpc": "2.0",
  "result": "ok",
  "id": 5
}"""

# ============================================================
# UTILITIES
# ============================================================

def count_tokens(text):
    """Approximate token count (GPT-style ~4 chars per token)"""
    return len(text) // 4

def mic_serialize(data):
    """Serialize to MIC format"""
    lines = ["mic@1"]
    for i, sym in enumerate(data["symbols"]):
        lines.append(f'S{i} "{sym}"')
    for t in data["types"]:
        shape = ",".join(str(s) for s in t["shape"])
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
    """Parse MIC format"""
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

def benchmark(name, text):
    """Get stats for a format"""
    return {
        "name": name,
        "size": len(text),
        "tokens": count_tokens(text),
        "lines": len(text.strip().split("\n")),
    }

# TOON format (Token-Oriented Object Notation)
SAMPLE_TOON = """version: 1
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
  5,relu,N4,3"""

# TOML format
SAMPLE_TOML = """version = 1
symbols = ["input", "weight", "bias", "output"]
outputs = [5]

[[types]]
id = 0
dtype = "f32"
shape = ["B", 784]

[[types]]
id = 1
dtype = "f32"
shape = [784, 256]

[[types]]
id = 2
dtype = "f32"
shape = [256]

[[types]]
id = 3
dtype = "f32"
shape = ["B", 256]

[[nodes]]
id = 0
op = "param"
symbol = 0
type_id = 0

[[nodes]]
id = 1
op = "param"
symbol = 1
type_id = 1

[[nodes]]
id = 2
op = "param"
symbol = 2
type_id = 2

[[nodes]]
id = 3
op = "matmul"
inputs = [0, 1]
type_id = 3

[[nodes]]
id = 4
op = "add"
inputs = [3, 2]
type_id = 3

[[nodes]]
id = 5
op = "relu"
inputs = [4]
type_id = 3"""

def toon_parse(text):
    """Parse TOON format (simplified)"""
    result = {"symbols": [], "types": [], "nodes": [], "outputs": []}
    lines = text.strip().split("\n")
    for line in lines:
        line = line.strip()
        if line.startswith("version:"):
            result["version"] = int(line.split(":")[1].strip())
        elif line.startswith("symbols"):
            symbols = line.split(":")[1].strip()
            result["symbols"] = symbols.split(",")
        elif "," in line and not line.startswith("types") and not line.startswith("nodes"):
            result["nodes"].append(line)
    return result

def benchmark_parse_speed(name, text, parse_func, iterations=10000):
    """Benchmark parse speed"""
    start = time.perf_counter()
    for _ in range(iterations):
        parse_func(text)
    elapsed = time.perf_counter() - start
    return {
        "name": name,
        "iterations": iterations,
        "total_ms": elapsed * 1000,
        "per_parse_us": (elapsed * 1_000_000) / iterations,
    }

# ============================================================
# RUN BENCHMARKS
# ============================================================

print("=" * 70)
print("  IR Serialization Benchmark - All Formats")
print("=" * 70)
print()

json_text = json.dumps(SAMPLE_IR, indent=2)
json_compact = json.dumps(SAMPLE_IR, separators=(',', ':'))
mic_text = mic_serialize(SAMPLE_IR)

results = [
    benchmark("JSON (pretty)", json_text),
    benchmark("TOML", SAMPLE_TOML),
    benchmark("TOON", SAMPLE_TOON),
    benchmark("MIC", mic_text),
]

print(f"{'Format':<18} {'Size':<12} {'Tokens':<10} {'Lines':<8} {'vs JSON':<10}")
print("-" * 70)
json_tokens = results[0]["tokens"]
for r in results:
    ratio = f"{json_tokens/r['tokens']:.1f}x" if r['tokens'] > 0 else "N/A"
    print(f"{r['name']:<18} {r['size']:<12} {r['tokens']:<10} {r['lines']:<8} {ratio:<10}")

print()
print("=" * 70)
print("  Parse Speed Benchmark (10,000 iterations)")
print("=" * 70)
print()

speed_results = [
    benchmark_parse_speed("JSON", json_text, json.loads),
    benchmark_parse_speed("MIC", mic_text, mic_parse),
    benchmark_parse_speed("TOON", SAMPLE_TOON, toon_parse),
]

if HAS_TOML:
    def toml_parse(text):
        return tomllib.loads(text)
    speed_results.insert(1, benchmark_parse_speed("TOML", SAMPLE_TOML, toml_parse))

print(f"{'Format':<12} {'Total (ms)':<15} {'Per Parse (us)':<18} {'vs JSON':<12}")
print("-" * 70)
json_speed = speed_results[0]["per_parse_us"]
for r in speed_results:
    ratio = f"{json_speed/r['per_parse_us']:.1f}x faster" if r['per_parse_us'] < json_speed else f"{r['per_parse_us']/json_speed:.1f}x slower" if r['per_parse_us'] > json_speed else "baseline"
    print(f"{r['name']:<12} {r['total_ms']:<15.2f} {r['per_parse_us']:<18.2f} {ratio:<12}")

print()
print("=" * 70)
print("  MAP vs JSON-RPC - Protocol Benchmark")
print("=" * 70)
print()

map_stats = benchmark("MAP", MAP_SESSION)
jsonrpc_stats = benchmark("JSON-RPC", JSON_RPC_SESSION)

print(f"{'Protocol':<18} {'Size':<12} {'Tokens':<10} {'Lines':<8}")
print("-" * 70)
print(f"{map_stats['name']:<18} {map_stats['size']:<12} {map_stats['tokens']:<10} {map_stats['lines']:<8}")
print(f"{jsonrpc_stats['name']:<18} {jsonrpc_stats['size']:<12} {jsonrpc_stats['tokens']:<10} {jsonrpc_stats['lines']:<8}")

print()
print(f"MAP vs JSON-RPC: {jsonrpc_stats['size']/map_stats['size']:.1f}x smaller, {jsonrpc_stats['tokens']/map_stats['tokens']:.1f}x fewer tokens")

mic_stats = results[3]  # MIC is 4th in results list
mic_tokens = mic_stats["tokens"]

print()
print("=" * 70)
print("  Combined Results")
print("=" * 70)
print()
print("| Format/Protocol | vs JSON/JSON-RPC | Token Savings |")
print("|-----------------|------------------|---------------|")
print(f"| MIC             | {json_tokens/mic_tokens:.1f}x fewer tokens | {json_tokens - mic_tokens} tokens saved |")
print(f"| MAP             | {jsonrpc_stats['tokens']/map_stats['tokens']:.1f}x fewer tokens | {jsonrpc_stats['tokens'] - map_stats['tokens']} tokens saved |")

print()
print("=" * 70)
print("  Sample: MIC Format")
print("=" * 70)
print()
print(mic_text)

print()
print("=" * 70)
print("  Sample: MAP Session")
print("=" * 70)
print()
print(MAP_SESSION)

print()
print("=" * 70)
print("  Key Advantages")
print("=" * 70)
print("""
MIC Format:
  - 2.9x fewer tokens than JSON
  - Line-oriented (git-friendly diffs)
  - Stable IDs for safe patching
  - No nested brackets/braces
  - Human readable

MAP Protocol:
  - 4.0x fewer tokens than JSON-RPC
  - Single-line requests/responses
  - Sequence numbers for correlation
  - Heredoc support for large payloads
  - Designed for AI agent interaction
""")
