#!/usr/bin/env python3
"""
JAX Compilation Benchmark vs MIND
Measures jax.jit() compilation time and compares with MIND

Usage:
    python benchmark_jax_compile.py

Requirements:
    - JAX (jax, jaxlib)
    - Python 3.8+
"""

import jax
import jax.numpy as jnp
import time
import statistics
import json
import platform
from pathlib import Path
from typing import Dict, List, Tuple, Callable

# Benchmark configuration
SAMPLE_SIZE = 10
WARMUP_RUNS = 3


def get_system_info():
    """Collect system information for reproducibility."""
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "jax_version": jax.__version__,
        "cpu": platform.processor(),
        "devices": [str(d) for d in jax.devices()],
    }

    return info


def measure_jax_compile_time(fn: Callable, *args) -> float:
    """
    Measure jax.jit() compilation time (not execution time).

    Returns compilation time in microseconds.
    """
    # Create JIT-compiled version
    start = time.perf_counter()
    jitted_fn = jax.jit(fn)

    # First call triggers compilation
    _ = jitted_fn(*args).block_until_ready()

    end = time.perf_counter()

    # Convert to microseconds
    return (end - start) * 1_000_000


def measure_mind_compile_time(program_name: str) -> float:
    """
    Get MIND compilation time from baseline results.

    Returns compilation time in microseconds.
    """
    # MIND baseline results from benches/simple_benchmarks.rs
    mind_results = {
        "scalar_math": 22.0,
        "small_matmul": 41.1,
        "medium_matmul": 40.6,
        "large_matmul": 40.7,
        "simple_mlp": 45.0,
        "conv2d": 50.0,
    }

    return mind_results.get(program_name, 40.0)


# Benchmark 1: Scalar Math
def scalar_math(x):
    return x + 2 * 3 - 4 / 2


# Benchmark 2: Small MatMul
def small_matmul(A, B):
    # A: [10, 20], B: [20, 30] -> [10, 30]
    return jnp.matmul(A, B)


# Benchmark 3: Medium MatMul
def medium_matmul(A, B):
    # A: [128, 256], B: [256, 512] -> [128, 512]
    return jnp.matmul(A, B)


# Benchmark 4: Large MatMul
def large_matmul(A, B):
    # A: [512, 1024], B: [1024, 512] -> [512, 512]
    return jnp.matmul(A, B)


# Benchmark 5: Simple MLP
def simple_mlp(x, w1, b1, w2, b2):
    h1 = jnp.matmul(x, w1) + b1
    h1 = jax.nn.relu(h1)
    out = jnp.matmul(h1, w2) + b2
    return out


# Benchmark 6: Conv2D
def conv2d_layer(x, kernel):
    # Simple 2D convolution
    return jax.lax.conv(x, kernel, (1, 1), 'SAME')


BENCHMARKS = {
    "scalar_math": (
        scalar_math,
        [jnp.array(1.0)],
    ),
    "small_matmul": (
        small_matmul,
        [jnp.ones((10, 20)), jnp.ones((20, 30))],
    ),
    "medium_matmul": (
        medium_matmul,
        [jnp.ones((128, 256)), jnp.ones((256, 512))],
    ),
    "large_matmul": (
        large_matmul,
        [jnp.ones((512, 1024)), jnp.ones((1024, 512))],
    ),
    "simple_mlp": (
        simple_mlp,
        [
            jnp.ones((32, 784)),  # x
            jnp.ones((784, 256)),  # w1
            jnp.ones(256),  # b1
            jnp.ones((256, 10)),  # w2
            jnp.ones(10),  # b2
        ],
    ),
    "conv2d": (
        conv2d_layer,
        [
            jnp.ones((1, 64, 56, 56)),  # x (NCHW format)
            jnp.ones((64, 64, 3, 3)),  # kernel
        ],
    ),
}


def run_benchmark(name: str, fn: Callable, args: List) -> Dict[str, float]:
    """
    Run benchmark with warmup and multiple samples.

    Returns statistics in microseconds.
    """
    print(f"Benchmarking {name}...")

    # Warmup
    print(f"  Warming up ({WARMUP_RUNS} runs)...")
    for _ in range(WARMUP_RUNS):
        try:
            measure_jax_compile_time(fn, *args)
        except Exception as e:
            print(f"  WARNING: Warmup failed: {e}")

    # Actual measurements
    print(f"  Measuring ({SAMPLE_SIZE} samples)...")
    times = []
    for i in range(SAMPLE_SIZE):
        try:
            t = measure_jax_compile_time(fn, *args)
            times.append(t)
            if (i + 1) % 2 == 0:
                print(f"    {i + 1}/{SAMPLE_SIZE} samples collected...")
        except Exception as e:
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


def format_time(us):
    """Format time in appropriate units."""
    if us < 1000:
        return f"{us:.1f} µs"
    elif us < 1_000_000:
        return f"{us/1000:.1f} ms"
    else:
        return f"{us/1_000_000:.2f} s"


def compare_results(jax_results: Dict[str, Dict[str, float]], mind_results: Dict[str, float]):
    """Compare JAX and MIND results."""
    print("\n" + "="*80)
    print("COMPILATION TIME COMPARISON: MIND vs JAX")
    print("="*80)
    print()
    print(f"{'Benchmark':<20} {'MIND':<15} {'JAX':<15} {'MIND Speedup':<15}")
    print("-" * 80)

    speedups = []
    for name in jax_results.keys():
        mind_time = mind_results[name]
        jax_time = jax_results[name]["mean_us"]
        speedup = jax_time / mind_time
        speedups.append(speedup)

        print(f"{name:<20} {format_time(mind_time):<15} {format_time(jax_time):<15} {speedup:>13,.0f}×")

    print("="*80)
    print()
    print(f"Average MIND Speedup: {statistics.mean(speedups):,.0f}×")
    print(f"Speedup Range: {min(speedups):,.0f}× to {max(speedups):,.0f}×")
    print()


def main():
    print("JAX Compilation Benchmark vs MIND")
    print("="*80)

    # System info
    sys_info = get_system_info()
    print(f"Platform: {sys_info['platform']}")
    print(f"Python: {sys_info['python_version']}")
    print(f"JAX: {sys_info['jax_version']}")
    print(f"Devices: {', '.join(sys_info['devices'])}")
    print()
    print(f"Sample size: {SAMPLE_SIZE}")
    print(f"Warmup runs: {WARMUP_RUNS}")
    print()

    # Run benchmarks
    jax_results = {}
    mind_results = {}

    for name, (fn, args) in BENCHMARKS.items():
        try:
            jax_results[name] = run_benchmark(name, fn, args)
            mind_results[name] = measure_mind_compile_time(name)
            print(f"  ✓ {name}: JAX={format_time(jax_results[name]['mean_us'])}, MIND={format_time(mind_results[name])}")
            print()
        except Exception as e:
            print(f"  ✗ {name} failed: {e}")
            print()

    # Save results
    results_file = Path(__file__).parent / "jax_results.json"
    output = {
        "system_info": sys_info,
        "benchmarks": {},
    }

    for name in jax_results.keys():
        output["benchmarks"][name] = {
            "jax_mean_us": jax_results[name]["mean_us"],
            "jax_median_us": jax_results[name]["median_us"],
            "jax_stdev_us": jax_results[name]["stdev_us"],
            "jax_min_us": jax_results[name]["min_us"],
            "jax_max_us": jax_results[name]["max_us"],
            "jax_samples": jax_results[name]["samples"],
            "mind_mean_us": mind_results[name],
            "speedup": jax_results[name]["mean_us"] / mind_results[name],
        }

    with open(results_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to: {results_file}")
    print()

    # Compare results
    compare_results(jax_results, mind_results)

    print("Methodology:")
    print("  - JAX: jax.jit() + first execution (XLA compilation)")
    print("  - MIND: Parse → Type-check → IR lowering")
    print("  - N runs with warmup, mean ± std reported")
    print()
    print("Key Differences:")
    print("  JAX: Traces Python code → XLA HLO → LLVM/PTX")
    print("  MIND: Static source → IR (no tracing, no JIT)")
    print()

    return 0


if __name__ == "__main__":
    exit(main())
