#!/usr/bin/env python3
"""Quick test of real MIND compilation time using Python bindings."""

import mind
import time
import statistics

WARMUP = 10
SAMPLES = 100

# Simple program
program = """
    let x: Tensor[f32,(100,100)] = 0;
    let y: Tensor[f32,(100,100)] = 1;
    tensor.matmul(x, y)
"""

print("Testing real MIND compilation time via Python bindings...")
print(f"Warmup: {WARMUP}, Samples: {SAMPLES}")
print()

# Warmup
for _ in range(WARMUP):
    mind.compile(program)

# Measure
times = []
for _ in range(SAMPLES):
    start = time.perf_counter()
    mind.compile(program)
    end = time.perf_counter()
    times.append((end - start) * 1_000_000)  # Convert to µs

mean = statistics.mean(times)
stdev = statistics.stdev(times)
minimum = min(times)
maximum = max(times)

print(f"Real MIND Compilation Time (NO subprocess overhead):")
print(f"  Mean:   {mean:.1f} µs")
print(f"  StdDev: {stdev:.1f} µs")
print(f"  Min:    {minimum:.1f} µs")
print(f"  Max:    {maximum:.1f} µs")
print()
print(f"This is the TRUE compilation time for MIND!")
