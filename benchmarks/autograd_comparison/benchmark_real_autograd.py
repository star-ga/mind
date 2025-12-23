#!/usr/bin/env python3
"""
Real Autograd Benchmark: MIND vs PyTorch
Measures actual gradient computation performance

MIND: Compile-time autodiff (gradient IR generation time)
PyTorch: Runtime autodiff (backward pass execution time)

Usage:
    python benchmark_real_autograd.py

Requirements:
    - PyTorch 1.0+
    - MIND CLI built (cargo build --release --bin mind)
"""

import torch
import time
import statistics
import json
import platform
import subprocess
from pathlib import Path
from typing import Dict

# Benchmark configuration
SAMPLE_SIZE = 20
WARMUP_RUNS = 3


def get_system_info():
    """Collect system information."""
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


def measure_pytorch_backward(forward_fn, device="cpu"):
    """
    Measure PyTorch backward pass time (runtime autodiff).

    Returns (time_us, memory_bytes).
    """
    times = []

    # Warmup
    for _ in range(WARMUP_RUNS):
        out, params = forward_fn(device)
        loss = out.sum() if out.dim() > 0 else out
        loss.backward()

    # Actual measurements
    for _ in range(SAMPLE_SIZE):
        # Reset
        out, params = forward_fn(device)

        # Measure backward pass
        start = time.perf_counter()
        loss = out.sum() if out.dim() > 0 else out
        loss.backward()
        if device == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()

        times.append((end - start) * 1_000_000)  # Convert to µs

    return {
        "time_mean_us": statistics.mean(times),
        "time_stdev_us": statistics.stdev(times) if len(times) > 1 else 0,
    }


def measure_mind_autodiff_time(program: str, num_samples: int = 20):
    """
    Measure MIND autodiff compilation time (compile-time autodiff).

    This measures how long it takes MIND to:
    1. Compile the forward pass
    2. Generate gradient IR

    Returns compilation time in microseconds.
    """
    mind_binary = Path(__file__).parent.parent.parent / "target" / "release" / "mind"
    if not mind_binary.exists():
        raise RuntimeError(f"MIND CLI not found. Run: cargo build --release --bin mind")

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

        times.append((end - start) * 1_000_000)  # Convert to µs

    return {
        "time_mean_us": statistics.mean(times),
        "time_stdev_us": statistics.stdev(times) if len(times) > 1 else 0,
    }


# Benchmark 1: Simple quadratic loss
def simple_quadratic_loss(device="cpu"):
    """sum(x^2) loss"""
    x = torch.randn(1000, device=device, requires_grad=True)
    loss = (x ** 2).sum()
    return loss, [x]


MIND_SIMPLE_QUAD = """
    let x: Tensor[f32,(1000)] = 0;
    let squared = mul(x, x);
    tensor.sum(squared, [0])
"""


# Benchmark 2: Small MLP
def small_mlp(device="cpu"):
    """32 x 784 -> 256 -> 10 MLP with loss"""
    x = torch.randn(32, 784, device=device, requires_grad=True)
    w1 = torch.randn(784, 256, device=device, requires_grad=True)
    b1 = torch.randn(256, device=device, requires_grad=True)
    w2 = torch.randn(256, 10, device=device, requires_grad=True)
    b2 = torch.randn(10, device=device, requires_grad=True)

    h1 = torch.relu(torch.matmul(x, w1) + b1)
    out = torch.matmul(h1, w2) + b2
    loss = out.sum()
    return loss, [x, w1, b1, w2, b2]


MIND_SMALL_MLP = """
    let x: Tensor[f32,(32,784)] = 0;
    let w1: Tensor[f32,(784,256)] = 1;
    let b1: Tensor[f32,(256)] = 0;
    let w2: Tensor[f32,(256,10)] = 1;
    let b2: Tensor[f32,(10)] = 0;

    let h1 = tensor.relu(add(tensor.matmul(x, w1), b1));
    let out = add(tensor.matmul(h1, w2), b2);
    tensor.sum(out, [0, 1])
"""


# Benchmark 3: MatMul chain
def matmul_chain(device="cpu"):
    """A @ B @ C @ D with gradient"""
    A = torch.randn(64, 128, device=device, requires_grad=True)
    B = torch.randn(128, 256, device=device, requires_grad=True)
    C = torch.randn(256, 512, device=device, requires_grad=True)
    D = torch.randn(512, 128, device=device, requires_grad=True)

    out = A @ B @ C @ D
    loss = out.sum()
    return loss, [A, B, C, D]


MIND_MATMUL_CHAIN = """
    let A: Tensor[f32,(64,128)] = 1;
    let B: Tensor[f32,(128,256)] = 1;
    let C: Tensor[f32,(256,512)] = 1;
    let D: Tensor[f32,(512,128)] = 1;

    let ab = tensor.matmul(A, B);
    let abc = tensor.matmul(ab, C);
    let abcd = tensor.matmul(abc, D);
    tensor.sum(abcd, [0, 1])
"""


BENCHMARKS = {
    "simple_quadratic": (simple_quadratic_loss, MIND_SIMPLE_QUAD),
    "small_mlp": (small_mlp, MIND_SMALL_MLP),
    "matmul_chain": (matmul_chain, MIND_MATMUL_CHAIN),
}


def format_time(us):
    """Format time in appropriate units."""
    if us < 1000:
        return f"{us:.1f} µs"
    elif us < 1_000_000:
        return f"{us/1000:.2f} ms"
    else:
        return f"{us/1_000_000:.2f} s"


def compare_results(pytorch_results: Dict, mind_results: Dict):
    """Compare PyTorch and MIND autodiff results."""
    print("\n" + "="*80)
    print("AUTODIFF COMPARISON: MIND vs PyTorch")
    print("(Both measured on the SAME machine)")
    print("="*80)
    print()
    print("MIND: Compile-time autodiff (gradient IR generation)")
    print("PyTorch: Runtime autodiff (backward pass execution)")
    print()
    print(f"{'Benchmark':<20} {'MIND (compile)':<20} {'PyTorch (runtime)':<20} {'Ratio':<15}")
    print("-" * 80)

    for name in pytorch_results.keys():
        mind_time = mind_results[name]["time_mean_us"]
        pytorch_time = pytorch_results[name]["time_mean_us"]
        ratio = pytorch_time / mind_time

        print(f"{name:<20} {format_time(mind_time):<20} {format_time(pytorch_time):<20} {ratio:>13.2f}×")

    print("="*80)
    print()
    print("Interpretation:")
    print("  - MIND: Time to generate gradient IR at compile-time")
    print("  - PyTorch: Time to execute backward pass at runtime")
    print("  - MIND's compile-time cost is paid once, PyTorch's runtime cost is paid every iteration")
    print()


def main():
    print("Real Autograd Benchmark: MIND vs PyTorch")
    print("="*80)

    # System info
    sys_info = get_system_info()
    print(f"Platform: {sys_info['platform']}")
    print(f"Python: {sys_info['python_version']}")
    print(f"PyTorch: {sys_info['pytorch_version']}")
    print()

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device.upper()}")
    print(f"Sample size: {SAMPLE_SIZE}")
    print()

    # Run benchmarks
    pytorch_results = {}
    mind_results = {}

    for name, (pytorch_fn, mind_program) in BENCHMARKS.items():
        print(f"Benchmarking {name}...")
        try:
            # Measure PyTorch backward
            pytorch_results[name] = measure_pytorch_backward(pytorch_fn, device)
            print(f"  ✓ PyTorch backward: {format_time(pytorch_results[name]['time_mean_us'])}")

            # Measure MIND autodiff compilation
            mind_results[name] = measure_mind_autodiff_time(mind_program)
            print(f"  ✓ MIND autodiff: {format_time(mind_results[name]['time_mean_us'])}")
            print()
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            import traceback
            traceback.print_exc()
            print()

    # Save results
    results_file = Path(__file__).parent / "real_autograd_results.json"
    output = {
        "system_info": sys_info,
        "methodology": {
            "mind": "Compile-time autodiff (gradient IR generation time)",
            "pytorch": "Runtime autodiff (backward pass execution time)",
            "note": "Both measured on same machine for fair comparison",
        },
        "benchmarks": {},
    }

    for name in pytorch_results.keys():
        output["benchmarks"][name] = {
            "pytorch": pytorch_results[name],
            "mind": mind_results[name],
            "ratio": pytorch_results[name]["time_mean_us"] / mind_results[name]["time_mean_us"],
        }

    with open(results_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to: {results_file}")
    print()

    # Compare
    compare_results(pytorch_results, mind_results)

    return 0


if __name__ == "__main__":
    exit(main())
