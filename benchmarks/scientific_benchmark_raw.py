#!/usr/bin/env python3
"""
Scientific Benchmark: MIND Frontend vs PyTorch torch.compile()
==============================================================

IMPORTANT: This comparison measures DIFFERENT pipeline stages:
  - MIND: Frontend only (parse + typecheck + IR lowering) — NO code generation
  - PyTorch: Full compilation (graph capture + optimization + code generation)

These are NOT equivalent operations. MIND does strictly less work.
The comparison shows how fast MIND's frontend is, not that MIND is
a "faster compiler" in the traditional sense.

Methodology:
  - MIND: In-process Criterion benchmarks (most accurate, no subprocess overhead)
  - PyTorch: torch.compile() with inductor backend (CPU), includes first inference
  - Both measure on same machine, sequential execution
  - PyTorch: 3 warmup + 10 measured samples per benchmark
"""

import torch
import time
import statistics
import json
import platform
import subprocess
from pathlib import Path

SAMPLE_SIZE = 10
WARMUP_RUNS = 3

def get_system_info():
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cpu": platform.processor() or "unknown",
        "cuda_available": torch.cuda.is_available(),
    }

def measure_torch_compile(model_fn, input_shape, device="cpu"):
    """Measure torch.compile() time in microseconds."""
    model = model_fn().to(device)
    x = torch.randn(*input_shape, device=device)
    torch.compiler.reset()

    start = time.perf_counter()
    compiled = torch.compile(model, backend="inductor")
    with torch.no_grad():
        _ = compiled(x)
    if device == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()

    return (end - start) * 1_000_000

# ─── PyTorch Models (equivalent tensor operations) ───

class ScalarMath(torch.nn.Module):
    def forward(self, x):
        return x + 2 * 3 - 4 / 2

class SmallMatMul(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(20, 30))
    def forward(self, x):
        return torch.matmul(x, self.weight)

class MediumMatMul(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(256, 512))
    def forward(self, x):
        return torch.matmul(x, self.weight)

class LargeMatMul(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(1024, 512))
    def forward(self, x):
        return torch.matmul(x, self.weight)

class SimpleMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(784, 256)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(256, 10)
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

BENCHMARKS = {
    "scalar_math": (ScalarMath, (1, 10)),
    "small_matmul": (SmallMatMul, (1, 10, 20)),
    "medium_matmul": (MediumMatMul, (1, 128, 256)),
    "large_matmul": (LargeMatMul, (1, 512, 1024)),
    "simple_mlp": (SimpleMLP, (1, 784)),
}

# ─── MIND Criterion reference (from cargo bench on THIS machine) ───
# These are populated from the Criterion run we just did
MIND_CRITERION = {
    "scalar_math": None,     # Will be filled from cargo bench output
    "small_matmul": None,
    "medium_matmul": None,
    "large_matmul": None,
    "simple_mlp": None,
}

def parse_criterion_output():
    """Run cargo bench and parse results."""
    print("Running MIND Criterion benchmarks (in-process, most accurate)...")
    result = subprocess.run(
        ["cargo", "bench", "--bench", "simple_benchmarks"],
        capture_output=True, text=True,
        cwd="/home/n/mind",
        timeout=300
    )

    output = result.stdout + result.stderr
    results = {}

    # Parse "time:   [X.XX µs Y.YY µs Z.ZZ µs]" lines
    import re
    lines = output.split('\n')
    current_bench = None
    for line in lines:
        if 'parse_check_lower/' in line:
            current_bench = line.strip().split('/')[-1]
        elif 'time:' in line and current_bench:
            match = re.search(r'time:\s+\[[\d.]+ [µn]s\s+([\d.]+) ([µn]s)', line)
            if match:
                val = float(match.group(1))
                unit = match.group(2)
                if unit == 'ns':
                    val /= 1000
                results[current_bench] = val
                current_bench = None

    return results

def format_time(us):
    if us < 1000:
        return f"{us:.1f} µs"
    elif us < 1_000_000:
        return f"{us/1000:.1f} ms"
    else:
        return f"{us/1_000_000:.2f} s"

def main():
    print("=" * 80)
    print("SCIENTIFIC BENCHMARK: MIND Frontend vs PyTorch torch.compile()")
    print("=" * 80)
    print()

    sys_info = get_system_info()
    print(f"Platform: {sys_info['platform']}")
    print(f"Python:   {sys_info['python_version']}")
    print(f"PyTorch:  {sys_info['pytorch_version']}")
    print(f"CUDA:     {'Yes' if sys_info['cuda_available'] else 'No (CPU only)'}")
    print()

    # Step 1: Run MIND Criterion benchmarks
    mind_results = parse_criterion_output()
    print(f"\nMIND Criterion results (in-process, parse+typecheck+IR):")
    for name, time_us in sorted(mind_results.items()):
        print(f"  {name}: {format_time(time_us)}")
    print()

    # Step 2: Run PyTorch benchmarks
    print("Running PyTorch torch.compile() benchmarks (inductor, CPU)...")
    print(f"  Warmup: {WARMUP_RUNS} runs, Samples: {SAMPLE_SIZE}")
    print()

    pytorch_results = {}
    for name, (model_fn, input_shape) in BENCHMARKS.items():
        print(f"  Benchmarking {name}...")

        # Warmup
        for _ in range(WARMUP_RUNS):
            try:
                measure_torch_compile(model_fn, input_shape)
            except Exception as e:
                print(f"    Warmup failed: {e}")

        # Measure
        times = []
        for i in range(SAMPLE_SIZE):
            try:
                t = measure_torch_compile(model_fn, input_shape)
                times.append(t)
            except Exception as e:
                print(f"    Sample {i+1} failed: {e}")

        if times:
            pytorch_results[name] = {
                "mean": statistics.mean(times),
                "median": statistics.median(times),
                "stdev": statistics.stdev(times) if len(times) > 1 else 0,
                "min": min(times),
                "max": max(times),
                "samples": len(times),
            }
            print(f"    Result: {format_time(pytorch_results[name]['mean'])} "
                  f"(±{format_time(pytorch_results[name]['stdev'])})")

    # Step 3: Comparison
    print()
    print("=" * 90)
    print("RESULTS: MIND Frontend (parse+typecheck+IR) vs PyTorch torch.compile() (full pipeline)")
    print("=" * 90)
    print()
    print("IMPORTANT: MIND measures FRONTEND ONLY (no code generation).")
    print("           PyTorch measures FULL COMPILATION (graph capture + optimization + codegen).")
    print("           These are NOT equivalent operations.")
    print()
    print(f"{'Benchmark':<18} {'MIND Frontend':<15} {'PyTorch Full':<18} {'Ratio':<12} {'Note'}")
    print("-" * 90)

    all_results = {}
    for name in BENCHMARKS:
        mind_time = mind_results.get(name)
        pytorch_time = pytorch_results.get(name, {}).get("mean")

        if mind_time and pytorch_time:
            ratio = pytorch_time / mind_time
            all_results[name] = {
                "mind_us": mind_time,
                "pytorch_us": pytorch_time,
                "ratio": ratio,
            }
            print(f"{name:<18} {format_time(mind_time):<15} {format_time(pytorch_time):<18} {ratio:>10,.0f}×  (different scope)")
        elif mind_time:
            print(f"{name:<18} {format_time(mind_time):<15} {'FAILED':<18} {'N/A':<12}")

    print("-" * 90)
    print()

    if all_results:
        ratios = [r["ratio"] for r in all_results.values()]
        print(f"Ratio range: {min(ratios):,.0f}× to {max(ratios):,.0f}×")
        print(f"Median ratio: {statistics.median(ratios):,.0f}×")
        print()
        print("METHODOLOGY NOTES:")
        print("  1. MIND: Rust Criterion in-process benchmarks (100 samples, statistical)")
        print("  2. PyTorch: Python perf_counter, torch.compile(backend='inductor') + first inference")
        print("  3. MIND measures: source → AST → typechecked AST → IR (STOPS HERE)")
        print("  4. PyTorch measures: Python → FX graph → optimization → C++/Triton code → compiled binary")
        print("  5. These measure DIFFERENT amounts of work — ratio is NOT a compilation speed comparison")
        print()

    # Save results
    output = {
        "system_info": sys_info,
        "methodology": {
            "mind": "In-process Criterion (parse + typecheck + IR lowering only)",
            "pytorch": "torch.compile(inductor) + first inference (full pipeline)",
            "note": "NOT equivalent operations — MIND does strictly less work",
        },
        "results": all_results,
    }

    results_file = "/tmp/scientific_benchmark_results.json"
    with open(results_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to: {results_file}")

if __name__ == "__main__":
    main()
