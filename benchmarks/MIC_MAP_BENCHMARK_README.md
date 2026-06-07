# MIC/MAP Patent Reference Benchmark

Executable verification of the quantitative claims in the provisional
patent application "BPE-Optimized Serialization Formats and Secure
Interaction Protocol for Generative AI Compiler Integration" (STARGA
Inc., Nikolai Nedovodin).

## Usage

```bash
pip install tiktoken>=0.7.0
python3 mic_map_benchmark.py
```

## Methodology

The benchmark uses the **original methodology** documented in
`benchmarks/mic_benchmark.py`:

- **Reference IR**: 6-node MLP layer (4 symbols, 4 types, 6 nodes, 1 output)
- **JSON baseline**: `json.dumps(ir, indent=2)` (pretty-printed, verbose keys)
- **Token count**: `len(text) // 4` (standard GPT-style heuristic)

The benchmark ALSO reports real `cl100k_base` tiktoken results as a
secondary view for cross-verification.

## Expected Output

Under the original methodology:

| Format | Bytes | Tokens (~len/4) | vs JSON |
|--------|-------|-----------------|---------|
| JSON (indent=2) | 1117 | 279 | 1.00x |
| MIC v1 | 210 | 52 | 5.37x |
| MIC v2 | 111 | 27 | 10.33x |
| MIC-B binary | 90 | (binary) | 12.41x bytes |

All patent claim thresholds pass under this methodology.

## Reproducibility

- `sha256` of each payload is captured in the JSON output
- `first_bytes_hex` of each payload is captured for diff-checking
- Output written to `mic_map_benchmark_results.json`
