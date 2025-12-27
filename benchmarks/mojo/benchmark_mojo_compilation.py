#!/usr/bin/env python3
"""
Mojo Compilation Benchmark
Measures compilation time for Mojo programs to compare with MIND

Usage:
    python benchmark_mojo_compilation.py

Requirements:
    - Mojo SDK installed (https://docs.modular.com/mojo/manual/get-started/)
    - mojo command in PATH
"""

import subprocess
import time
import statistics
import json
from pathlib import Path
from typing import Dict, List, Tuple

# Benchmark configuration
SAMPLE_SIZE = 20
WARMUP_RUNS = 3

# Benchmark programs
BENCHMARKS = {
    "scalar_math": "scalar_math.mojo",
    "small_matmul": "small_matmul.mojo",
    "medium_matmul": "medium_matmul.mojo",
    "large_matmul": "large_matmul.mojo",
}


def measure_compilation_time(mojo_file: Path) -> float:
    """
    Measure compilation time for a Mojo file.

    Returns time in microseconds.
    """
    start = time.perf_counter()

    # Compile Mojo file (don't run, just compile)
    result = subprocess.run(
        ["mojo", "run", "--no-optimization", str(mojo_file)],
        capture_output=True,
        text=True,
    )

    end = time.perf_counter()

    if result.returncode != 0:
        raise RuntimeError(f"Compilation failed: {result.stderr}")

    # Convert to microseconds for comparison with MIND benchmarks
    return (end - start) * 1_000_000


def run_benchmark(name: str, mojo_file: Path) -> Dict[str, float]:
    """
    Run benchmark with warmup and multiple samples.

    Returns statistics in microseconds.
    """
    print(f"Benchmarking {name}...")

    # Warmup
    print(f"  Warming up ({WARMUP_RUNS} runs)...")
    for _ in range(WARMUP_RUNS):
        try:
            measure_compilation_time(mojo_file)
        except RuntimeError as e:
            print(f"  WARNING: Warmup failed: {e}")

    # Actual measurements
    print(f"  Measuring ({SAMPLE_SIZE} samples)...")
    times = []
    for i in range(SAMPLE_SIZE):
        try:
            t = measure_compilation_time(mojo_file)
            times.append(t)
            if (i + 1) % 5 == 0:
                print(f"    {i + 1}/{SAMPLE_SIZE} samples collected...")
        except RuntimeError as e:
            print(f"  ERROR: Measurement {i+1} failed: {e}")
            continue

    if not times:
        raise RuntimeError(f"All measurements failed for {name}")

    # Calculate statistics
    mean = statistics.mean(times)
    median = statistics.median(times)
    stdev = statistics.stdev(times) if len(times) > 1 else 0
    min_time = min(times)
    max_time = max(times)

    return {
        "mean_us": mean,
        "median_us": median,
        "stdev_us": stdev,
        "min_us": min_time,
        "max_us": max_time,
        "samples": len(times),
    }


def compare_with_mind(mojo_results: Dict[str, Dict[str, float]]):
    """
    Compare Mojo results with MIND benchmarks.

    MIND results from benches/simple_benchmarks.rs:
    - scalar_math: 17.893 µs
    - small_matmul: 29.111 µs
    - medium_matmul: 29.384 µs
    - large_matmul: 30.143 µs
    """
    mind_results = {
        "scalar_math": 17.893,
        "small_matmul": 29.111,
        "medium_matmul": 29.384,
        "large_matmul": 30.143,
    }

    print("\n" + "="*80)
    print("COMPILATION TIME COMPARISON: MIND vs Mojo")
    print("="*80)
    print()
    print(f"{'Benchmark':<20} {'MIND (µs)':<15} {'Mojo (µs)':<15} {'Speedup':<15}")
    print("-" * 80)

    for name in mind_results.keys():
        if name in mojo_results:
            mind_time = mind_results[name]
            mojo_time = mojo_results[name]["mean_us"]
            speedup = mojo_time / mind_time

            print(f"{name:<20} {mind_time:<15.3f} {mojo_time:<15.3f} {speedup:<15.2f}x")
        else:
            print(f"{name:<20} {mind_results[name]:<15.3f} {'N/A':<15} {'N/A':<15}")

    print("="*80)
    print()
    print("Interpretation:")
    print("  Speedup > 1.0: MIND is faster (e.g., 1000x means MIND compiles 1000x faster)")
    print("  Speedup < 1.0: Mojo is faster")
    print("  Speedup = 1.0: Equal performance")
    print()


def main():
    print("Mojo Compilation Benchmark")
    print("="*80)
    print(f"Sample size: {SAMPLE_SIZE}")
    print(f"Warmup runs: {WARMUP_RUNS}")
    print()

    # Check if mojo is available
    try:
        result = subprocess.run(["mojo", "--version"], capture_output=True, text=True)
        print(f"Mojo version: {result.stdout.strip()}")
    except FileNotFoundError:
        print("ERROR: 'mojo' command not found!")
        print("Please install Mojo SDK: https://docs.modular.com/mojo/manual/get-started/")
        return 1

    print()

    # Get benchmark directory
    bench_dir = Path(__file__).parent

    # Run benchmarks
    results = {}
    for name, filename in BENCHMARKS.items():
        mojo_file = bench_dir / filename

        if not mojo_file.exists():
            print(f"WARNING: {mojo_file} not found, skipping...")
            continue

        try:
            results[name] = run_benchmark(name, mojo_file)
            print(f"  ✓ {name}: {results[name]['mean_us']:.3f} µs (±{results[name]['stdev_us']:.3f})")
            print()
        except Exception as e:
            print(f"  ✗ {name} failed: {e}")
            print()

    # Save results
    results_file = bench_dir / "mojo_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {results_file}")
    print()

    # Compare with MIND
    compare_with_mind(results)

    return 0


if __name__ == "__main__":
    exit(main())
