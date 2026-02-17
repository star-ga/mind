#!/usr/bin/env python3
"""
MIND Scientific Benchmark — Clean Recording Script
=====================================================
Run this for a terminal recording/video of the benchmark process.

Output is designed to be clear and educational for viewers.
"""

import torch
import time
import statistics
import subprocess
import re
import sys

SAMPLE_SIZE = 10
WARMUP_RUNS = 3

# ANSI colors
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

def banner(text):
    width = 78
    print(f"\n{CYAN}{'─' * width}{RESET}")
    print(f"{BOLD}{text}{RESET}")
    print(f"{CYAN}{'─' * width}{RESET}\n")

def section(text):
    print(f"\n{YELLOW}▸ {text}{RESET}\n")

def format_time(us):
    if us < 1000:
        return f"{us:.1f} µs"
    elif us < 1_000_000:
        return f"{us/1000:.1f} ms"
    else:
        return f"{us/1_000_000:.2f} s"

# ─── PyTorch Models ───

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
        self.fc2 = torch.nn.Linear(256, 10)
    def forward(self, x):
        return self.fc2(torch.nn.functional.relu(self.fc1(x)))

BENCHMARKS = {
    "scalar_math": (ScalarMath, (1, 10)),
    "small_matmul": (SmallMatMul, (1, 10, 20)),
    "medium_matmul": (MediumMatMul, (1, 128, 256)),
    "large_matmul": (LargeMatMul, (1, 512, 1024)),
    "simple_mlp": (SimpleMLP, (1, 784)),
}

def measure_torch_compile(model_fn, input_shape):
    model = model_fn()
    x = torch.randn(*input_shape)
    torch.compiler.reset()
    start = time.perf_counter()
    compiled = torch.compile(model, backend="inductor")
    with torch.no_grad():
        _ = compiled(x)
    end = time.perf_counter()
    return (end - start) * 1_000_000

def run_criterion():
    """Parse Criterion benchmark output."""
    result = subprocess.run(
        ["cargo", "bench", "--bench", "simple_benchmarks"],
        capture_output=True, text=True,
        cwd="/home/n/mind", timeout=300
    )
    output = result.stdout + result.stderr
    results = {}
    current = None
    for line in output.split('\n'):
        if 'parse_check_lower/' in line:
            current = line.strip().split('/')[-1]
        elif 'time:' in line and current:
            m = re.search(r'time:\s+\[[\d.]+ [µn]s\s+([\d.]+) ([µn]s)', line)
            if m:
                val = float(m.group(1))
                if m.group(2) == 'ns': val /= 1000
                results[current] = val
                current = None
    return results

def main():
    banner("MIND Scientific Benchmark")

    print(f"  {DIM}What this measures:{RESET}")
    print(f"  MIND:    Frontend only — parse + typecheck + IR lowering")
    print(f"  PyTorch: Full pipeline — graph capture + optimization + code generation")
    print(f"  {DIM}(These are NOT equivalent operations — MIND does less work){RESET}")
    print()
    print(f"  Python {sys.version.split()[0]} | PyTorch {torch.__version__} | CPU only")

    # Step 1: MIND
    section("Step 1: MIND Criterion Benchmarks (in-process, 100 samples each)")
    mind = run_criterion()

    print(f"  {'Program':<18} {'Time':<12} {'Pipeline Stage'}")
    print(f"  {'─' * 55}")
    for name in ["scalar_math", "small_matmul", "medium_matmul", "large_matmul", "tensor_ops", "reductions", "reshape_ops"]:
        if name in mind:
            print(f"  {name:<18} {GREEN}{format_time(mind[name]):<12}{RESET} parse → typecheck → IR")

    # Step 2: PyTorch
    section(f"Step 2: PyTorch torch.compile() (inductor, {WARMUP_RUNS} warmup + {SAMPLE_SIZE} samples)")

    pytorch = {}
    for name, (model_fn, shape) in BENCHMARKS.items():
        sys.stdout.write(f"  {name:<18} ")
        sys.stdout.flush()

        # Warmup
        for _ in range(WARMUP_RUNS):
            try: measure_torch_compile(model_fn, shape)
            except: pass

        # Measure
        times = []
        for _ in range(SAMPLE_SIZE):
            try: times.append(measure_torch_compile(model_fn, shape))
            except: pass

        if times:
            mean = statistics.mean(times)
            std = statistics.stdev(times) if len(times) > 1 else 0
            pytorch[name] = mean
            print(f"{format_time(mean):<12} ±{format_time(std):<10} graph → optimize → codegen")
        else:
            print("FAILED")

    # Step 3: Comparison
    banner("Results: Frontend Speed Ratio")

    print(f"  {DIM}MIND = parse + typecheck + IR (frontend only, no code generation)")
    print(f"  PyTorch = torch.compile(inductor) + first inference (full compilation){RESET}")
    print()
    print(f"  {'Benchmark':<18} {'MIND':<12} {'PyTorch':<14} {'Ratio'}")
    print(f"  {'─' * 60}")

    ratios = []
    for name in BENCHMARKS:
        if name in mind and name in pytorch:
            ratio = pytorch[name] / mind[name]
            ratios.append(ratio)
            print(f"  {name:<18} {GREEN}{format_time(mind[name]):<12}{RESET} {format_time(pytorch[name]):<14} {ratio:>10,.0f}×")

    print(f"  {'─' * 60}")
    if ratios:
        print(f"\n  {BOLD}MIND frontend is {statistics.median(ratios):,.0f}× faster{RESET} (median)")
        print(f"  Range: {min(ratios):,.0f}× to {max(ratios):,.0f}×")

    print()
    print(f"  {YELLOW}⚠  IMPORTANT CAVEAT{RESET}")
    print(f"  These tools do different amounts of work:")
    print(f"  • MIND stops at IR — no machine code, no executable")
    print(f"  • PyTorch produces runnable optimized code")
    print(f"  • The ratio reflects scope difference, not just speed")
    print()

if __name__ == "__main__":
    main()
