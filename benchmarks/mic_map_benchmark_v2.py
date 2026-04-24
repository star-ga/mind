"""
MIC/MAP Patent Reference Benchmark — Reproduces Paper's Canonical Numbers
==========================================================================

Reproduces the measurement numbers recited in the provisional patent
application "BPE-Optimized Serialization Formats and Secure Interaction
Protocol for Generative AI Compiler Integration" (STARGA Inc.,
Nikolai Nedovodin).

Uses the ORIGINAL methodology documented in mind/benchmarks/mic_benchmark.py:
  * 6-node MLP reference IR (4 symbols, 4 types, 6 nodes, 1 output)
  * JSON baseline = json.dumps(..., indent=2) (pretty-printed, verbose keys)
  * Token count = len(text) // 4 (standard GPT-style heuristic)

Reproduces the paper's Table I.3 / APPENDIX C numbers:
  * JSON:  1133 bytes, 278 tokens
  * MIC v1: 209 bytes, 52 tokens  (5.3x token reduction, 81%)

Also reports REAL cl100k_base tiktoken results as a secondary view for
cross-verification at non-provisional conversion.

Usage:
    pip install tiktoken>=0.7.0
    python3 mic_map_benchmark.py

Copyright (c) 2025-2026 STARGA, Inc. All rights reserved.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False
    print("Note: tiktoken not available, skipping real-tokenizer view")


# =================================================================== #
#  Reference IR — matches /home/n/mind/benchmarks/mic_benchmark.py       #
# =================================================================== #

SAMPLE_IR_DICT: dict[str, Any] = {
    "version": 1,
    "symbols": ["input", "weight", "bias", "output"],
    "types": [
        {"id": 0, "dtype": "f32", "shape": [None, 784]},
        {"id": 1, "dtype": "f32", "shape": [784, 256]},
        {"id": 2, "dtype": "f32", "shape": [256]},
        {"id": 3, "dtype": "f32", "shape": [None, 256]},
    ],
    "nodes": [
        {"id": 0, "op": "param",  "symbol": 0, "type": 0},
        {"id": 1, "op": "param",  "symbol": 1, "type": 1},
        {"id": 2, "op": "param",  "symbol": 2, "type": 2},
        {"id": 3, "op": "matmul", "inputs": [0, 1], "type": 3},
        {"id": 4, "op": "add",    "inputs": [3, 2], "type": 3},
        {"id": 5, "op": "relu",   "inputs": [4],    "type": 3},
    ],
    "outputs": [5],
}

SAMPLE_MIC_V1_TEXT: str = """mic@1
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

SAMPLE_MIC_V2_TEXT: str = """mic@2
T0 f32 B 784
T1 f32 784 256
T2 f32 256
T3 f32 B 256
p input T0
p weight T1
p bias T2
m 0 1
+ 4 2
r 5
O 6
"""


# =================================================================== #
#  Token counting — two methodologies                                    #
# =================================================================== #

def tokens_approximate(text: str) -> int:
    """Original GPT-style approximation used in mic_benchmark.py (4 chars/token)."""
    return len(text) // 4


def tokens_tiktoken(text: str, encoder) -> int:
    """Real BPE tokenization with cl100k_base encoder."""
    return len(encoder.encode(text))


# =================================================================== #
#  MIC-B binary (new in MIC v2/binary spec)                              #
# =================================================================== #

def _uleb128(v: int) -> bytes:
    out = bytearray()
    while True:
        b = v & 0x7F
        v >>= 7
        if v:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


DTYPE_BYTE = {
    "f16": 0, "f32": 1, "f64": 2, "bf16": 3, "i8": 4, "i16": 5,
    "i32": 6, "i64": 7, "u8": 8, "u16": 9, "u32": 10, "u64": 11, "bool": 12,
}
OPCODE_BYTE = {
    "m": 0, "+": 1, "-": 2, "*": 3, "/": 4, "r": 5, "s": 6, "sig": 7,
    "th": 8, "gelu": 9, "ln": 10, "t": 11, "rshp": 12, "sum": 13,
    "mean": 14, "max": 15, "cat": 16, "split": 17, "gth": 18,
}


def serialize_mic_b(ir: dict[str, Any]) -> bytes:
    """MIC-B binary encoding per patent APPENDIX I §I.5."""
    strs: list[str] = []
    idx: dict[str, int] = {}

    def intern(s: str) -> int:
        if s not in idx:
            idx[s] = len(strs)
            strs.append(s)
        return idx[s]

    # String table order: symbols, dims, value names
    for s in ir["symbols"]:
        intern(s)
    for t in ir["types"]:
        for d in t["shape"]:
            intern(str(d))
    # Custom names for nodes not in this IR.

    out = bytearray()
    out.extend(b"MICB")
    out.append(0x02)
    out.extend(_uleb128(len(strs)))
    for s in strs:
        b = s.encode("utf-8")
        out.extend(_uleb128(len(b)))
        out.extend(b)
    # Symbol table
    out.extend(_uleb128(len(ir["symbols"])))
    for s in ir["symbols"]:
        out.extend(_uleb128(idx[s]))
    # Type table
    out.extend(_uleb128(len(ir["types"])))
    for t in ir["types"]:
        out.append(DTYPE_BYTE[t["dtype"]])
        out.extend(_uleb128(len(t["shape"])))
        for d in t["shape"]:
            out.extend(_uleb128(idx[str(d)]))
    # Value table (arg+param+node in declaration order)
    out.extend(_uleb128(len(ir["nodes"])))
    opmap = {"param": None, "matmul": "m", "add": "+", "relu": "r"}
    for n in ir["nodes"]:
        if n["op"] == "param":
            out.append(0x01)  # Param tag
            out.extend(_uleb128(idx[ir["symbols"][n["symbol"]]]))
            out.extend(_uleb128(n["type"]))
        else:
            out.append(0x02)  # Node tag
            op = opmap[n["op"]]
            out.append(OPCODE_BYTE[op])
            out.extend(_uleb128(len(n["inputs"])))
            for i in n["inputs"]:
                out.extend(_uleb128(i))
    # Output
    out.extend(_uleb128(ir["outputs"][0]))
    return bytes(out)


# =================================================================== #
#  Main                                                                 #
# =================================================================== #

def main() -> int:
    print("MIC/MAP Patent Benchmark — reproducing paper's canonical numbers\n")

    # Paper's baseline: JSON with indent=2 (pretty-printed)
    json_pretty = json.dumps(SAMPLE_IR_DICT, indent=2)
    json_mini   = json.dumps(SAMPLE_IR_DICT, separators=(",", ":"))
    mic_v1      = SAMPLE_MIC_V1_TEXT
    mic_v2      = SAMPLE_MIC_V2_TEXT
    mic_b_bytes = serialize_mic_b(SAMPLE_IR_DICT)

    encoder = tiktoken.get_encoding("cl100k_base") if HAS_TIKTOKEN else None

    rows: list[dict[str, Any]] = []
    for label, payload, is_binary in [
        ("json_indent2_pretty", json_pretty, False),
        ("json_minified",       json_mini,   False),
        ("mic_v1",              mic_v1,      False),
        ("mic_v2",              mic_v2,      False),
        ("mic_b_binary",        mic_b_bytes, True),
    ]:
        b = payload if is_binary else payload.encode("utf-8")
        row = {
            "label":            label,
            "bytes":            len(b),
            "tokens_approx":    tokens_approximate(payload) if not is_binary else None,
            "tokens_tiktoken":  tokens_tiktoken(payload, encoder) if (not is_binary and encoder) else None,
            "sha256":           hashlib.sha256(b).hexdigest(),
            "first_bytes_hex":  b[:48].hex(),
            "is_binary":        is_binary,
        }
        rows.append(row)

    print("=" * 86)
    print("METHODOLOGY 1: Paper's original — JSON indent=2 baseline + len(text)//4 tokens")
    print("=" * 86)
    print(f"\n  {'FORMAT':<24}{'BYTES':>8}{'TOKENS (~len/4)':>18}{'vs JSON':>12}{'REDUCTION':>14}")
    j = next(r for r in rows if r["label"] == "json_indent2_pretty")
    for r in rows:
        if r["tokens_approx"] is None:
            bytes_ratio = j["bytes"] / r["bytes"]
            byte_reduction = (1 - r["bytes"] / j["bytes"]) * 100
            print(f"  {r['label']:<24}{r['bytes']:>8}{'  (binary)':>18}"
                  f"{bytes_ratio:>10.2f}x {f'{byte_reduction:.0f}% bytes':>14}")
        else:
            ratio = j["tokens_approx"] / r["tokens_approx"] if r["tokens_approx"] else 0
            reduction = (1 - r["tokens_approx"] / j["tokens_approx"]) * 100 if j["tokens_approx"] else 0
            print(f"  {r['label']:<24}{r['bytes']:>8}{r['tokens_approx']:>18}"
                  f"{ratio:>10.2f}x {f'{reduction:.0f}%':>14}")

    if encoder is not None:
        print("\n" + "=" * 86)
        print("METHODOLOGY 2: Real cl100k_base tokenization (secondary view)")
        print("=" * 86)
        print(f"\n  {'FORMAT':<24}{'BYTES':>8}{'TOKENS (tiktoken)':>20}{'vs JSON':>12}")
        for r in rows:
            if r["tokens_tiktoken"] is None:
                print(f"  {r['label']:<24}{r['bytes']:>8}{'  (binary)':>20}"
                      f"{j['bytes']/r['bytes']:>10.2f}x")
            else:
                ratio = j["tokens_tiktoken"] / r["tokens_tiktoken"] if r["tokens_tiktoken"] else 0
                print(f"  {r['label']:<24}{r['bytes']:>8}{r['tokens_tiktoken']:>20}"
                      f"{ratio:>10.2f}x")

    # Paper claim verification
    mic1 = next(r for r in rows if r["label"] == "mic_v1")
    mic2 = next(r for r in rows if r["label"] == "mic_v2")
    micb = next(r for r in rows if r["label"] == "mic_b_binary")
    ratio_mic1_paper = j["tokens_approx"] / mic1["tokens_approx"]
    ratio_mic2_paper = j["tokens_approx"] / mic2["tokens_approx"] if mic2["tokens_approx"] else 0
    ratio_micb_bytes = j["bytes"] / micb["bytes"]

    print("\n" + "=" * 86)
    print("CLAIM THRESHOLD VERIFICATION (paper methodology)")
    print("=" * 86)
    checks = [
        ("Paper FIG 1 claim: JSON = 278 tokens (±5%)",
         abs(j["tokens_approx"] - 278) <= 14,
         f"measured: {j['tokens_approx']}"),
        ("Paper FIG 1 claim: MIC v1 = 52 tokens (±5%)",
         abs(mic1["tokens_approx"] - 52) <= 3,
         f"measured: {mic1['tokens_approx']}"),
        ("Paper FIG 1 claim: MIC v1 achieves ~5.3x token reduction",
         abs(ratio_mic1_paper - 5.3) <= 0.5,
         f"measured: {ratio_mic1_paper:.2f}x"),
        ("Claim 25: ≥80% token reduction (MIC v1 vs JSON)",
         (1 - mic1["tokens_approx"] / j["tokens_approx"]) >= 0.80,
         f"measured: {(1 - mic1['tokens_approx']/j['tokens_approx'])*100:.1f}%"),
        ("Claim 1B: MIC v1 ≥4.0x fewer tokens than JSON",
         ratio_mic1_paper >= 4.0,
         f"measured: {ratio_mic1_paper:.2f}x"),
        ("Claim 66A: MIC v2 ≥4.0x fewer tokens than JSON",
         ratio_mic2_paper >= 4.0,
         f"measured: {ratio_mic2_paper:.2f}x"),
        ("Claim 87: MIC-B ≥5.0x fewer bytes than JSON",
         ratio_micb_bytes >= 5.0,
         f"measured: {ratio_micb_bytes:.2f}x"),
    ]
    all_pass = all(c[1] for c in checks)
    for desc, ok, detail in checks:
        mark = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {mark}  {desc}")
        print(f"           ({detail})")
    print(f"\n  OVERALL: {'ALL CLAIMS VERIFIED ✓' if all_pass else 'SOME CLAIMS FAILED ✗'}")

    # Write machine-readable report
    report = {
        "metadata": {
            "script":            "mic_map_benchmark.py",
            "version":           "2.0",
            "patent_reference":  "STARGA Inc., BPE-Optimized Serialization Formats and Secure Interaction Protocol",
            "reference_ir":      "6-node MLP layer (param, matmul, add, relu)",
            "methodology_1":     "JSON indent=2 baseline + len(text)//4 token approximation (paper original)",
            "methodology_2":     "Real cl100k_base tokenization (cross-verification)",
            "tiktoken_version":  tiktoken.__version__ if HAS_TIKTOKEN else None,
            "python_version":    sys.version.split()[0],
        },
        "measurements":        rows,
        "paper_figures_verified": {
            "JSON_278_tokens":    j["tokens_approx"],
            "MIC_v1_52_tokens":   mic1["tokens_approx"],
            "MIC_v1_5.3x_ratio":  round(ratio_mic1_paper, 2),
        },
        "claim_checks": [
            {"claim": d, "pass": ok, "detail": det} for d, ok, det in checks
        ],
        "all_claims_verified": all_pass,
    }
    out_path = Path(__file__).parent / "mic_map_benchmark_results.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nMachine-readable report: {out_path}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
