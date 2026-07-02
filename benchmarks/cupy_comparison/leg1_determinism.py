#!/usr/bin/env python3
"""
LEG 1 — Determinism demo (THE WEDGE).

Side-by-side proof of the core MIND claim:

    CuPy GPU float result:  hash_run1  MAY DIFFER FROM  hash_run2  (same GPU, same input)
    MIND CPU Q16.16 result: hash_run1  ==  hash_run2  ==  hash_arm  (bit-identical, gate-enforced)

CuPy is the FOIL, not a dependency of MIND. We run the SAME logical reduction/GEMM
in CuPy several ways and hash the raw output bytes to see whether the bits drift.
The MIND side is NOT recomputed here: its hash is the committed, CI-gate-enforced
reference from tests/cross_substrate_identity/gemm-q16-64x64x64/reference_hashes.toml
(avx2 == neon, bit-identical by construction).

HONEST FRAMING (read benchmarks/cupy_comparison/README.md):
  - We claim MIND CPU x86/ARM BIT-IDENTITY vs CuPy GPU FLOAT NON-DETERMINISM.
  - We do NOT claim MIND's GPU semantic tier is bit-identical to CuPy's GPU.
  - If CuPy does NOT diverge on this hardware for an op, we report that honestly
    (some ops are deterministic) and move to one that does, or document why.

Tiny GPU footprint by design: the reduction array and GEMM here are sized to fit
in well under 1 GB so this runs with zero extra pressure beyond a small CuPy alloc.

Usage:
    python leg1_determinism.py
"""

import hashlib
import json
import platform
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
# The committed, gate-enforced MIND reference (single source of truth — RFC 0020 §4.3).
MIND_FIXTURE = (
    REPO_ROOT
    / "tests"
    / "cross_substrate_identity"
    / "gemm-q16-64x64x64"
    / "reference_hashes.toml"
)
MIND_MANIFEST = (
    REPO_ROOT
    / "tests"
    / "cross_substrate_identity"
    / "gemm-q16-64x64x64"
    / "manifest.toml"
)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_mind_reference():
    """Point at the committed MIND hashes; do NOT recompute (instruction)."""
    with open(MIND_FIXTURE, "rb") as f:
        hashes = tomllib.load(f)
    with open(MIND_MANIFEST, "rb") as f:
        manifest = tomllib.load(f)
    return hashes, manifest


def cupy_determinism_probe():
    """
    Run several CuPy float workloads twice each and report whether the raw output
    bytes diverge. Returns a list of probe dicts. Empirically tests divergence —
    never assumes it.
    """
    import cupy as cp
    import cupyx

    probes = []
    seed = 0xDEADBEEF

    # --- Probe 0: atomic scatter-add (THE HEADLINE divergence) ------------------
    # Many GPU threads atomically accumulate floats into a few bins. Float add is
    # not associative and the atomic commit ORDER is genuinely run-to-run
    # nondeterministic -> the output bytes drift every run. This is the canonical
    # GPU float-nondeterminism case and it reproduces on this RTX 3080.
    def probe_scatter_add(n, bins, label):
        rs = cp.random.RandomState(seed)
        idx = rs.randint(0, bins, size=n).astype(cp.int32)
        val = rs.standard_normal(n, dtype=cp.float32)
        idx_hash = sha256_bytes(idx.get().tobytes())
        val_hash = sha256_bytes(val.get().tobytes())
        hashes = []
        for _ in range(8):
            out = cp.zeros(bins, dtype=cp.float32)
            cupyx.scatter_add(out, idx, val)  # the op under test
            cp.cuda.Stream.null.synchronize()
            hashes.append(sha256_bytes(out.get().tobytes()))
        diverged = len(set(hashes)) > 1
        return {
            "probe": label,
            "op": "cupyx.scatter_add (atomic)",
            "n": n,
            "bins": bins,
            "dtype": "float32",
            "input_hashes": [idx_hash, val_hash],
            "output_hashes": hashes,
            "distinct_output_hashes": len(set(hashes)),
            "diverged": diverged,
        }

    probes.append(probe_scatter_add(8 * 1024 * 1024, 257, "scatter_add_f32_8M"))

    # --- Probe A: large float32 sum reduction -----------------------------------
    # cupy.sum over a large array uses a parallel tree/atomic reduction; the order
    # of float adds can vary run-to-run, so the low bits of the sum can drift.
    def probe_reduction(n, dtype, label):
        rs = cp.random.RandomState(seed)
        # Same input both runs (host-pinned canonical bytes so the INPUT is identical).
        x = rs.standard_normal(n, dtype=cp.float32).astype(dtype)
        x_bytes = x.get().tobytes()
        x_hash = sha256_bytes(x_bytes)
        hashes = []
        raw = []
        for _ in range(8):
            cp.cuda.Stream.null.synchronize()
            s = cp.sum(x)  # the op under test
            cp.cuda.Stream.null.synchronize()
            b = s.get().tobytes()
            raw.append(b)
            hashes.append(sha256_bytes(b))
        diverged = len(set(hashes)) > 1
        return {
            "probe": label,
            "op": "cupy.sum",
            "n": n,
            "dtype": str(dtype),
            "input_hash": x_hash,
            "output_hashes": hashes,
            "distinct_output_hashes": len(set(hashes)),
            "diverged": diverged,
        }

    probes.append(probe_reduction(64 * 1024 * 1024, cp.float32, "reduction_f32_64M"))
    probes.append(probe_reduction(16 * 1024 * 1024, cp.float32, "reduction_f32_16M"))

    # --- Probe B: repeated cuBLAS GEMM (float32 accumulation) --------------------
    # cuBLAS may pick different kernels / split-K reductions across calls; with
    # float accumulation the C result bytes can differ run-to-run.
    def probe_gemm(m, dtype, label):
        rs = cp.random.RandomState(seed)
        a = rs.standard_normal((m, m), dtype=cp.float32).astype(dtype)
        b = rs.standard_normal((m, m), dtype=cp.float32).astype(dtype)
        a_hash = sha256_bytes(a.get().tobytes())
        b_hash = sha256_bytes(b.get().tobytes())
        hashes = []
        for _ in range(8):
            cp.cuda.Stream.null.synchronize()
            c = a @ b
            cp.cuda.Stream.null.synchronize()
            hashes.append(sha256_bytes(c.get().tobytes()))
        diverged = len(set(hashes)) > 1
        return {
            "probe": label,
            "op": "cupy matmul (cuBLAS)",
            "m": m,
            "dtype": str(dtype),
            "input_hashes": [a_hash, b_hash],
            "output_hashes": hashes,
            "distinct_output_hashes": len(set(hashes)),
            "diverged": diverged,
        }

    probes.append(probe_gemm(1024, cp.float32, "gemm_f32_1024"))
    probes.append(probe_gemm(2048, cp.float32, "gemm_f32_2048"))

    return probes


def main():
    out = {
        "leg": "1_determinism",
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
    }

    # --- MIND side (committed reference, NOT recomputed) ------------------------
    mind_hashes, mind_manifest = load_mind_reference()
    out["mind"] = {
        "workload": mind_manifest["name"],
        "encoding": mind_manifest["kernel"]["output_encoding"],
        "source": "committed gate reference (reference_hashes.toml) — RFC 0020 §10",
        "hash_avx2_x86": mind_hashes["avx2"],
        "hash_neon_arm": mind_hashes["neon"],
        "bit_identical_x86_arm": mind_hashes["avx2"] == mind_hashes["neon"],
    }

    # --- CuPy side (the foil) --------------------------------------------------
    try:
        import cupy as cp

        out["host"]["cupy_version"] = cp.__version__
        try:
            dev = cp.cuda.Device(0)
            props = cp.cuda.runtime.getDeviceProperties(0)
            out["host"]["gpu"] = props["name"].decode()
        except Exception:
            pass
        out["cupy"] = {"probes": cupy_determinism_probe()}
        out["cupy"]["any_diverged"] = any(
            p["diverged"] for p in out["cupy"]["probes"]
        )
    except Exception as e:
        out["cupy"] = {"error": f"{type(e).__name__}: {e}", "probes": []}
        out["cupy"]["any_diverged"] = None

    # --- Report ----------------------------------------------------------------
    print("=" * 72)
    print("LEG 1 — Determinism demo: CuPy GPU float  vs  MIND CPU Q16.16")
    print("=" * 72)
    print()
    print("MIND (committed, CI-gate-enforced — gemm-q16-64x64x64):")
    print(f"  x86 avx2 hash : {out['mind']['hash_avx2_x86']}")
    print(f"  arm neon hash : {out['mind']['hash_neon_arm']}")
    print(
        f"  bit-identical x86==arm : {out['mind']['bit_identical_x86_arm']}"
    )
    print()
    print("CuPy (foil) — same input, repeated runs, raw output bytes hashed:")
    if "error" in out["cupy"]:
        print(f"  CuPy unavailable / failed: {out['cupy']['error']}")
    else:
        for p in out["cupy"]["probes"]:
            distinct = p["distinct_output_hashes"]
            verdict = "DIVERGES" if p["diverged"] else "stable (deterministic here)"
            print(f"  [{p['probe']:>20}] {p['op']:<22} "
                  f"distinct_hashes={distinct}/8  -> {verdict}")
            if p["diverged"]:
                uniq = sorted(set(p["output_hashes"]))
                print(f"        (input bytes identical every run; output bytes drift)")
                print(f"        run-hash #1: {uniq[0]}")
                print(f"        run-hash #2: {uniq[1]}   <-- DIFFERS")
        print()
        if out["cupy"]["any_diverged"]:
            print("  RESULT: CuPy float output is NON-DETERMINISTIC on this GPU "
                  "(at least one probe diverged).")
        else:
            print("  RESULT: every CuPy probe was STABLE on this GPU this run. "
                  "See README 'Honest framing' — divergence is hardware/driver/"
                  "size dependent; the cross-GPU framing still holds.")
    print()
    print("THE WEDGE:")
    print("  MIND CPU artifact: hash_run1 == hash_run2 == hash_arm  (bit-identical, gated).")
    print("  CuPy GPU artifact: hash_run1 may != hash_run2          (float order non-det).")
    print("  Honest scope: this is CPU x86/ARM bit-identity vs CuPy GPU float; MIND's")
    print("  own GPU semantic tier is NOT claimed bit-identical to CuPy GPU.")

    res_path = Path(__file__).parent / "leg1_determinism_results.json"
    with open(res_path, "w") as f:
        json.dump(out, f, indent=2)
    print()
    print(f"Wrote {res_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
