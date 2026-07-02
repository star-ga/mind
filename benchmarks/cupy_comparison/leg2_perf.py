#!/usr/bin/env python3
"""
LEG 2 — Perf baseline (THE TAX), gated on GPU headroom.

Measures CuPy cuBLAS GEMM throughput (GFLOP/s) across square sizes and reports it
next to the MIND deterministic GEMM at the same LOGICAL workload. The goal slide is:

    "MIND is within X% of CuPy throughput AND bit-identical across substrates,
     which CuPy is not."

This leg is GATED on free VRAM. The RTX 3080 here is 10 GB and ollama holds most
of it. If < REQUIRED_FREE_GB is free, we SKIP gracefully with a clear message and
DO NOT evict anything. The harness always emits a JSON results file (skipped or run).

MIND side: the deterministic Q16.16 GEMM is a CPU integer kernel (the
gemm-q16-64x64x64 cross-substrate gate workload). Its throughput is measured by
the in-tree Criterion bench `benches/det_matmul_q16.rs` — we record the command to
reproduce it rather than re-implementing the kernel here. A float-GPU GFLOP/s and a
deterministic-CPU GFLOP/s are NOT the same hardware; the table states that plainly.
The load-bearing MIND claim is bit-identity, not beating cuBLAS on raw GPU FLOPs.

Usage:
    python leg2_perf.py [--required-free-gb 8]
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

REQUIRED_FREE_GB_DEFAULT = 8.0
SIZES = [512, 1024, 2048, 4096]
WARMUP = 3
ITERS = 20


def free_vram_gb():
    """Return (free_gb, total_gb) via CuPy's runtime, or (None, None) on failure."""
    try:
        import cupy as cp

        free_b, total_b = cp.cuda.runtime.memGetInfo()
        return free_b / 1e9, total_b / 1e9
    except Exception:
        return None, None


def estimate_need_gb(n):
    """Three n*n float32 matrices (A, B, C) for the largest size."""
    return (3 * n * n * 4) / 1e9


def bench_cupy_gemm(n):
    """Return GFLOP/s for an n*n float32 cuBLAS GEMM (median of ITERS)."""
    import cupy as cp

    rs = cp.random.RandomState(0xDEADBEEF)
    a = rs.standard_normal((n, n), dtype=cp.float32)
    b = rs.standard_normal((n, n), dtype=cp.float32)
    for _ in range(WARMUP):
        c = a @ b
    cp.cuda.Stream.null.synchronize()

    start = cp.cuda.Event()
    end = cp.cuda.Event()
    times_ms = []
    for _ in range(ITERS):
        start.record()
        c = a @ b
        end.record()
        end.synchronize()
        times_ms.append(cp.cuda.get_elapsed_time(start, end))
    times_ms.sort()
    median_ms = times_ms[len(times_ms) // 2]
    flop = 2.0 * n * n * n  # multiply-add
    gflops = flop / (median_ms / 1e3) / 1e9
    del a, b, c
    cp.get_default_memory_pool().free_all_blocks()
    return {"n": n, "median_ms": median_ms, "gflops": gflops}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--required-free-gb", type=float, default=REQUIRED_FREE_GB_DEFAULT)
    args = ap.parse_args()

    out = {
        "leg": "2_perf",
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "config": {
            "sizes": SIZES,
            "warmup": WARMUP,
            "iters": ITERS,
            "required_free_gb": args.required_free_gb,
        },
        "mind": {
            "workload": "gemm-q16-64x64x64 (deterministic Q16.16, CPU integer)",
            "throughput_command": (
                "cargo bench --features 'mlir-build std-surface cross-module-imports' "
                "--bench det_matmul_q16"
            ),
            "note": (
                "MIND det-GEMM is a CPU integer kernel; its GFLOP/s is on a "
                "different substrate than cuBLAS GPU float. The load-bearing "
                "claim is cross-substrate bit-identity, not raw GPU FLOPs."
            ),
        },
    }

    free_gb, total_gb = free_vram_gb()
    out["host"]["vram_free_gb"] = free_gb
    out["host"]["vram_total_gb"] = total_gb

    print("=" * 72)
    print("LEG 2 — Perf baseline: CuPy cuBLAS GEMM throughput (VRAM-gated)")
    print("=" * 72)

    try:
        import cupy as cp

        out["host"]["cupy_version"] = cp.__version__
    except Exception as e:
        out["status"] = "skipped"
        out["skip_reason"] = f"CuPy import failed: {type(e).__name__}: {e}"
        print(f"\nSKIP: {out['skip_reason']}")
        _write(out)
        return 0

    if free_gb is None:
        out["status"] = "skipped"
        out["skip_reason"] = "could not query VRAM (no CUDA device?)"
        print(f"\nSKIP: {out['skip_reason']}")
        _write(out)
        return 0

    need_gb = estimate_need_gb(max(SIZES))
    print(f"\nVRAM: {free_gb:.2f} GB free / {total_gb:.2f} GB total")
    print(f"Gate: need {args.required_free_gb:.1f} GB free "
          f"(largest GEMM {max(SIZES)}x{max(SIZES)} ~ {need_gb:.2f} GB working set)")

    if free_gb < args.required_free_gb:
        out["status"] = "skipped"
        out["skip_reason"] = (
            f"only {free_gb:.2f} GB free < {args.required_free_gb:.1f} GB required; "
            f"refusing to evict co-resident GPU tenants (e.g. ollama)."
        )
        print(f"\nSKIP (graceful): {out['skip_reason']}")
        print("  Re-run when the GPU has headroom, or lower --required-free-gb at")
        print("  your own risk for the smaller sizes.")
        _write(out)
        return 0

    # --- gate passed: run the GEMM sweep --------------------------------------
    out["status"] = "ran"
    out["cupy_gemm"] = []
    print("\nRunning cuBLAS GEMM sweep (float32):")
    print(f"  {'n':>6} {'median_ms':>12} {'GFLOP/s':>12}")
    for n in SIZES:
        if estimate_need_gb(n) > free_gb * 0.8:
            print(f"  {n:>6}  skipped (working set too large for free VRAM)")
            continue
        r = bench_cupy_gemm(n)
        out["cupy_gemm"].append(r)
        print(f"  {n:>6} {r['median_ms']:>12.4f} {r['gflops']:>12.1f}")

    _write(out)
    return 0


def _write(out):
    p = Path(__file__).parent / "leg2_perf_results.json"
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {p}")


if __name__ == "__main__":
    sys.exit(main())
