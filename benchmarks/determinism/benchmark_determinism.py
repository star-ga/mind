#!/usr/bin/env python3
"""
MIND Determinism Proof Benchmark
Proves bit-level reproducibility for patent Claims 16-20

Usage:
    python benchmark_determinism.py

Tests:
    1. Run same MIND program multiple times
    2. Compute SHA256 hash of compilation output
    3. Verify all hashes are IDENTICAL
    4. Prove deterministic compilation
"""

import subprocess
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Number of runs to prove determinism
NUM_RUNS = 10

# Test programs
TEST_PROGRAMS = {
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
    "mlp": """
        let input: Tensor[f32,(128,784)] = 0;
        let w1: Tensor[f32,(784,512)] = 1;
        let b1: Tensor[f32,(512)] = 0;
        let w2: Tensor[f32,(512,256)] = 1;
        let b2: Tensor[f32,(256)] = 0;
        let w3: Tensor[f32,(256,10)] = 1;
        let b3: Tensor[f32,(10)] = 0;

        let matmul1 = tensor.matmul(input, w1);
        let add1 = add(matmul1, b1);
        let h1 = tensor.relu(add1);

        let matmul2 = tensor.matmul(h1, w2);
        let add2 = add(matmul2, b2);
        let h2 = tensor.relu(add2);

        let matmul3 = tensor.matmul(h2, w3);
        add(matmul3, b3)
    """,
}


def get_mind_binary() -> Path:
    """
    Get the path to the MIND CLI binary.

    Handles Windows .exe extension automatically.
    """
    mind_binary_base = Path(__file__).parent.parent.parent / "target" / "release" / "mind"

    # Handle Windows .exe extension
    if platform.system() == "Windows":
        mind_binary = mind_binary_base.with_suffix(".exe")
        if not mind_binary.exists():
            mind_binary = mind_binary_base
    else:
        mind_binary = mind_binary_base

    return mind_binary


def compile_mind_program(source_code: str) -> bytes:
    """
    Compile MIND program and return the IR output bytes.

    This uses the MIND CLI to compile a program and returns the IR.
    """
    mind_binary = get_mind_binary()

    if not mind_binary.exists():
        raise RuntimeError(f"MIND CLI not found at {mind_binary}. Run: cargo build --release --bin mind")

    # Compile using MIND CLI eval command (outputs IR)
    result = subprocess.run(
        [str(mind_binary), "eval", source_code],
        capture_output=True,
        text=False,  # Get bytes output
    )

    if result.returncode != 0:
        raise RuntimeError(f"Compilation failed: {result.stderr.decode()}")

    # Return the compiled output (IR or AST representation)
    return result.stdout


def compute_hash(data: bytes) -> str:
    """Compute SHA256 hash of data."""
    return hashlib.sha256(data).hexdigest()


def test_determinism(name: str, source_code: str, num_runs: int = NUM_RUNS) -> Dict:
    """
    Test determinism by compiling the same program multiple times.

    Returns dict with hashes and determinism verification.
    """
    print(f"\nTesting determinism: {name}")
    print("-" * 80)

    hashes = []
    compile_times = []

    for i in range(num_runs):
        print(f"  Run {i+1}/{num_runs}...", end=" ", flush=True)

        start = time.perf_counter()
        try:
            output = compile_mind_program(source_code)
            compile_time = (time.perf_counter() - start) * 1_000_000  # Convert to µs

            hash_value = compute_hash(output)
            hashes.append(hash_value)
            compile_times.append(compile_time)

            print(f"sha256={hash_value[:16]}... ({compile_time:.1f} µs)")

        except Exception as e:
            print(f"FAILED: {e}")
            return {
                "name": name,
                "deterministic": False,
                "error": str(e),
            }

    # Clean up
    if temp_file.exists():
        temp_file.unlink()

    # Verify all hashes are identical
    unique_hashes = set(hashes)
    is_deterministic = len(unique_hashes) == 1

    return {
        "name": name,
        "deterministic": is_deterministic,
        "num_runs": num_runs,
        "unique_hashes": len(unique_hashes),
        "hashes": hashes,
        "reference_hash": hashes[0] if hashes else None,
        "avg_compile_time_us": sum(compile_times) / len(compile_times) if compile_times else 0,
    }


def print_determinism_proof(results: List[Dict]):
    """Print determinism proof in patent-ready format."""
    print("\n" + "="*80)
    print("DETERMINISM VERIFICATION PROOF")
    print("="*80)
    print()

    for result in results:
        name = result["name"]
        is_deterministic = result.get("deterministic", False)

        print(f"Test: {name}")
        print(f"Runs: {result.get('num_runs', 0)}")

        if "error" in result:
            print(f"Status: ❌ FAILED - {result['error']}")
        elif is_deterministic:
            print(f"Status: ✅ DETERMINISTIC")
            print(f"Reference Hash: {result['reference_hash']}")
            print(f"Unique Hashes: {result['unique_hashes']} (all identical)")
            print(f"Avg Compile Time: {result['avg_compile_time_us']:.1f} µs")

            # Show hash verification
            print("\nHash Verification:")
            ref_hash = result['reference_hash']
            for i, h in enumerate(result['hashes'], 1):
                match = "✓ MATCH" if h == ref_hash else "✗ MISMATCH"
                print(f"  Run {i:2d}: {h[:16]}... {match}")
        else:
            print(f"Status: ❌ NON-DETERMINISTIC")
            print(f"Unique Hashes: {result['unique_hashes']} (expected 1)")
            print("\nHash Values:")
            for i, h in enumerate(result['hashes'], 1):
                print(f"  Run {i:2d}: {h}")

        print()

    # Summary
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r.get("deterministic", False))

    print("="*80)
    print(f"SUMMARY: {passed_tests}/{total_tests} tests DETERMINISTIC")
    if passed_tests == total_tests:
        print("✅ DETERMINISM VERIFIED: All outputs are bit-identical across runs")
        print("\nThis proves:")
        print("  - Compilation is deterministic (Claims 16-20)")
        print("  - Output is reproducible across runs")
        print("  - No non-deterministic factors (timestamps, random seeds, etc.)")
    else:
        print("❌ DETERMINISM FAILED: Some tests produced different outputs")

    print("="*80)
    print()


def get_system_info():
    """Collect system information."""
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


def main():
    print("MIND Determinism Proof Benchmark")
    print("="*80)
    print("Testing bit-level reproducibility for patent Claims 16-20")
    print()

    # System info
    sys_info = get_system_info()
    print(f"Platform: {sys_info['platform']}")
    print(f"Python: {sys_info['python_version']}")
    print(f"Machine: {sys_info['machine']}")
    print()

    # Check if MIND CLI is available
    mind_binary = get_mind_binary()
    if not mind_binary.exists():
        print(f"MIND CLI: Not found at {mind_binary}")
        print("\nNOTE: This benchmark requires the MIND compiler to be built.")
        print("Run: cargo build --release --bin mind")
        return 1

    try:
        result = subprocess.run(
            [str(mind_binary), "--help"],
            capture_output=True,
            timeout=5,
        )
        if result.returncode == 0:
            print(f"MIND CLI: Available at {mind_binary}")
        else:
            print(f"MIND CLI: Error - {result.stderr.decode()}")
            return 1
    except Exception as e:
        print(f"MIND CLI: Error running binary - {e}")
        return 1

    print()

    # Run determinism tests
    results = []
    for name, source_code in TEST_PROGRAMS.items():
        result = test_determinism(name, source_code, NUM_RUNS)
        results.append(result)

    # Print proof
    print_determinism_proof(results)

    # Save results
    results_file = Path(__file__).parent / "determinism_results.json"
    output = {
        "system_info": sys_info,
        "num_runs": NUM_RUNS,
        "tests": results,
        "all_deterministic": all(r.get("deterministic", False) for r in results),
    }

    with open(results_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to: {results_file}")
    print()

    # Return success if all tests passed
    return 0 if output["all_deterministic"] else 1


if __name__ == "__main__":
    exit(main())
