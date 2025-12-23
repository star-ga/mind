#!/usr/bin/env python3
"""
PyTorch 2.0 Compilation Benchmark vs MIND
Measures torch.compile() compilation time and compares with MIND

Usage:
    python benchmark_pytorch_compile.py

Requirements:
    - PyTorch 2.0+ (torch.compile support)
    - Python 3.8+
"""

import torch
import time
import statistics
import json
import platform
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

# Benchmark configuration
SAMPLE_SIZE = 10
WARMUP_RUNS = 3


def get_system_info():
    """Collect system information for reproducibility."""
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cpu": platform.processor(),
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu"] = torch.cuda.get_device_name(0)

    return info


def measure_torch_compile_time(model_fn, input_shape, device="cpu"):
    """
    Measure torch.compile() compilation time (not execution time).

    Returns compilation time in microseconds.
    """
    # Create model and input
    model = model_fn()
    model = model.to(device)
    x = torch.randn(*input_shape, device=device)

    # Measure compilation time
    start = time.perf_counter()
    compiled_model = torch.compile(model, mode="default")

    # First call triggers compilation
    with torch.no_grad():
        _ = compiled_model(x)

    # Ensure compilation is complete
    if device == "cuda":
        torch.cuda.synchronize()

    end = time.perf_counter()

    # Convert to microseconds
    return (end - start) * 1_000_000


def measure_mind_compile_time(program_name, num_samples=20):
    """
    Measure MIND compilation time on THIS machine (same-machine comparison).

    This function actually runs MIND CLI to compile programs, measuring real
    compilation time on the same system as PyTorch measurements.

    Returns compilation time in microseconds (mean of multiple samples).
    """
    # MIND programs equivalent to PyTorch benchmarks
    mind_programs = {
        "scalar_math": "1 + 2 * 3 - 4 / 2",
        "small_matmul": """
            let a: Tensor[f32,(10,20)] = 1;
            let b: Tensor[f32,(20,30)] = 1;
            tensor.matmul(a, b)
        """,
        "medium_matmul": """
            let a: Tensor[f32,(128,256)] = 1;
            let b: Tensor[f32,(256,512)] = 1;
            tensor.matmul(a, b)
        """,
        "large_matmul": """
            let a: Tensor[f32,(512,1024)] = 1;
            let b: Tensor[f32,(1024,512)] = 1;
            tensor.matmul(a, b)
        """,
        "simple_mlp": """
            let input: Tensor[f32,(32,784)] = 0;
            let w1: Tensor[f32,(784,256)] = 1;
            let b1: Tensor[f32,(256)] = 0;
            let w2: Tensor[f32,(256,10)] = 1;
            let b2: Tensor[f32,(10)] = 0;
            let h1 = tensor.relu(add(tensor.matmul(input, w1), b1));
            add(tensor.matmul(h1, w2), b2)
        """,
    }

    program = mind_programs.get(program_name)
    if not program:
        raise ValueError(f"No MIND program defined for benchmark: {program_name}")

    # Find MIND CLI binary
    mind_binary_base = Path(__file__).parent.parent.parent / "target" / "release" / "mind"

    # Handle Windows .exe extension
    if platform.system().lower().startswith("win"):
        mind_binary = mind_binary_base.with_suffix(".exe")
        if not mind_binary.exists():
            mind_binary = mind_binary_base
    else:
        mind_binary = mind_binary_base

    if not mind_binary.exists():
        raise RuntimeError(f"MIND CLI not found at {mind_binary}. Run: cargo build --release --bin mind")

    # Measure compilation time over multiple samples
    times = []
    for _ in range(num_samples):
        start = time.perf_counter()
        result = subprocess.run(
            [str(mind_binary), "eval", program],
            capture_output=True,
            text=False,
        )
        end = time.perf_counter()

        if result.returncode != 0:
            raise RuntimeError(f"MIND compilation failed: {result.stderr.decode()}")

        times.append((end - start) * 1_000_000)  # Convert to microseconds

    return statistics.mean(times)


# Benchmark 1: Scalar Math Operations
class ScalarMath(torch.nn.Module):
    def forward(self, x):
        return x + 2 * 3 - 4 / 2


# Benchmark 2: Small MatMul
class SmallMatMul(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(20, 30))

    def forward(self, x):
        # x: [batch, 10, 20] @ weight: [20, 30] -> [batch, 10, 30]
        return torch.matmul(x, self.weight)


# Benchmark 3: Medium MatMul
class MediumMatMul(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(256, 512))

    def forward(self, x):
        # x: [batch, 128, 256] @ weight: [256, 512] -> [batch, 128, 512]
        return torch.matmul(x, self.weight)


# Benchmark 4: Large MatMul
class LargeMatMul(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(1024, 512))

    def forward(self, x):
        # x: [batch, 512, 1024] @ weight: [1024, 512] -> [batch, 512, 512]
        return torch.matmul(x, self.weight)


# Benchmark 5: Simple MLP
class SimpleMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(784, 256)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Benchmark 6: Conv2D Layer (ResNet-50 style)
class Conv2DLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn = torch.nn.BatchNorm2d(64)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


BENCHMARKS = {
    "scalar_math": (ScalarMath, (1, 10)),
    "small_matmul": (SmallMatMul, (1, 10, 20)),
    "medium_matmul": (MediumMatMul, (1, 128, 256)),
    "large_matmul": (LargeMatMul, (1, 512, 1024)),
    "simple_mlp": (SimpleMLP, (1, 784)),
}


def run_benchmark(name: str, model_fn, input_shape: Tuple, device: str = "cpu") -> Dict[str, float]:
    """
    Run benchmark with warmup and multiple samples.

    Returns statistics in microseconds.
    """
    print(f"Benchmarking {name}...")

    # Warmup
    print(f"  Warming up ({WARMUP_RUNS} runs)...")
    for _ in range(WARMUP_RUNS):
        try:
            measure_torch_compile_time(model_fn, input_shape, device)
        except Exception as e:
            print(f"  WARNING: Warmup failed: {e}")

    # Actual measurements
    print(f"  Measuring ({SAMPLE_SIZE} samples)...")
    times = []
    for i in range(SAMPLE_SIZE):
        try:
            t = measure_torch_compile_time(model_fn, input_shape, device)
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


def compare_results(pytorch_results: Dict[str, Dict[str, float]], mind_results: Dict[str, float]):
    """Compare PyTorch and MIND results."""
    print("\n" + "="*80)
    print("COMPILATION TIME COMPARISON: MIND vs PyTorch 2.0")
    print("(Both measured on the SAME machine for fair comparison)")
    print("="*80)
    print()
    print(f"{'Benchmark':<20} {'MIND':<15} {'PyTorch 2.0':<15} {'MIND Speedup':<15}")
    print("-" * 80)

    speedups = []
    for name in pytorch_results.keys():
        mind_time = mind_results[name]
        pytorch_time = pytorch_results[name]["mean_us"]
        speedup = pytorch_time / mind_time
        speedups.append(speedup)

        print(f"{name:<20} {format_time(mind_time):<15} {format_time(pytorch_time):<15} {speedup:>13,.0f}×")

    print("="*80)
    print()
    print(f"Average MIND Speedup: {statistics.mean(speedups):,.0f}×")
    print(f"Speedup Range: {min(speedups):,.0f}× to {max(speedups):,.0f}×")
    print()


def main():
    print("PyTorch 2.0 Compilation Benchmark vs MIND")
    print("="*80)

    # Check PyTorch version
    torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
    if torch_version < (2, 0):
        print(f"ERROR: PyTorch 2.0+ required (found {torch.__version__})")
        print("Install with: pip install 'torch>=2.0'")
        return 1

    # System info
    sys_info = get_system_info()
    print(f"Hardware: {sys_info['cpu']}")
    print(f"Platform: {sys_info['platform']}")
    print(f"Python: {sys_info['python_version']}")
    print(f"PyTorch: {sys_info['pytorch_version']}")
    if sys_info['cuda_available']:
        print(f"GPU: {sys_info['gpu']}")
        print(f"CUDA: {sys_info['cuda_version']}")
    print()
    print(f"Sample size: {SAMPLE_SIZE}")
    print(f"Warmup runs: {WARMUP_RUNS}")
    print()

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device.upper()}")
    print()

    # Run benchmarks
    pytorch_results = {}
    mind_results = {}

    for name, (model_fn, input_shape) in BENCHMARKS.items():
        try:
            pytorch_results[name] = run_benchmark(name, model_fn, input_shape, device)
            mind_results[name] = measure_mind_compile_time(name)
            print(f"  ✓ {name}: PyTorch={format_time(pytorch_results[name]['mean_us'])}, MIND={format_time(mind_results[name])}")
            print()
        except Exception as e:
            print(f"  ✗ {name} failed: {e}")
            print()

    # Save results
    results_file = Path(__file__).parent / "pytorch_results.json"
    output = {
        "system_info": sys_info,
        "benchmarks": {},
    }

    for name in pytorch_results.keys():
        output["benchmarks"][name] = {
            "pytorch_mean_us": pytorch_results[name]["mean_us"],
            "pytorch_median_us": pytorch_results[name]["median_us"],
            "pytorch_stdev_us": pytorch_results[name]["stdev_us"],
            "pytorch_min_us": pytorch_results[name]["min_us"],
            "pytorch_max_us": pytorch_results[name]["max_us"],
            "pytorch_samples": pytorch_results[name]["samples"],
            "mind_mean_us": mind_results[name],
            "speedup": pytorch_results[name]["mean_us"] / mind_results[name],
        }

    with open(results_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to: {results_file}")
    print()

    # Compare results
    compare_results(pytorch_results, mind_results)

    print("Methodology:")
    print("  - PyTorch: torch.compile() + first inference (full compilation)")
    print("  - MIND: Parse → Type-check → IR lowering")
    print("  - N runs with warmup, mean ± std reported")
    print()

    return 0


if __name__ == "__main__":
    exit(main())
