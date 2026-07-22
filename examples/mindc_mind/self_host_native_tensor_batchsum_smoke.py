#!/usr/bin/env python3
"""C4-T6 — native-ELF 3-D BATCHED SUM (i64), zero MLIR/LLVM. The FIRST N-D
indexing rung of the tensor/linalg ladder.

`selftest_native_elf_tensor_batchsum_i64(b, m, n)` emits a runnable x86-64
ET_EXEC that (1) materializes a b×m×n i64 tensor A row-major in the frame with
deterministic self-seeding A[k*m*n + i*n + j] = k*m*n+i*n+j+1 (the FLAT index
t+1), (2) computes per-batch S[k] = Sum_{i<m} Sum_{j<n} A[k*m*n + i*n + j] via a
TRIPLE nest whose innermost body forms the genuine 3-D flat address
k*m*n + i*n + j at runtime (kk*mn + ri*ndim + cj, then *8 + base — the first
rung composing THREE index terms), (3) folds the batches into a POSITION-
WEIGHTED, SQUARED checksum sum = Sum_{k<b} (k+1)*S[k]*S[k], (4) writes the 8 LE
bytes of `sum` to stdout, and (5) exits (sum == expected)*41 + 1 — 42 only on an
EXACT full-width i64 match against the emit-time-baked expected checksum
(movabs-baked past imm32), 1 otherwise.

Why (k+1)-WEIGHTED and SQUARED (batch-partition-discriminating): the grand sum
of A is layout/partition-invariant, so a plain sum-of-batch-sums would NOT catch
a wrong batch stride. Weighting each S[k] by (k+1) AND squaring it makes the
observable depend on WHICH elements landed in WHICH batch: a transposed batch
stride (reading the k axis as if it were the fastest-varying axis instead of the
slowest), a collapsed batch dimension, or a wrong bound all change the weighted
sum-of-squares. This smoke proves it by also computing a TRANSPOSED-batch-stride
variant and asserting it differs from the correct one on every non-cubic shape.

Two independent full-width gates per shape:
  (a) stdout == struct.pack('<q', S) where S is THIS script's pure-Python
      seed / triple-reduce / weighted-square fold over the same seeds — an
      independent reference (no shared code with the emitter);
  (b) exit == 42 — the in-ELF exact-i64 comparison against the baked checksum.

SHAPE GUARD (the T1..T5 audit hazard): the export FAILS CLOSED — returns an
empty buffer — unless 1 <= b,m,n and b,m,n <= 4096 and m*n <= 4096 and
b*m*n <= 4096 (one 4096-element frame array). This smoke asserts the refusal on
out-of-bounds, degenerate, and i64-overflow shapes (where a product wraps to 0).

ADDITIVITY: a NEW export never reached during self-compile, so the integer
native-ELF oracle stays byte-identical. No frozen byte oracle exists for this
path; the gate is EXECUTION CORRECTNESS.

Usage:
  MINDC_SO=/path/to.so python3 self_host_native_tensor_batchsum_smoke.py
"""
import ctypes
import os
import pathlib
import stat
import struct
import subprocess
import sys
import tempfile

_HERE = pathlib.Path(__file__).parent
_DEFAULT_SO = _HERE / "libmindc_mind.so"
sys.path.insert(0, str(_HERE))
try:
    from _selfhost_so import resolve_so  # noqa: E402
except Exception:  # pragma: no cover - resolver is optional
    resolve_so = None


def mind_batchsum_elf(lib, b: int, m: int, n: int) -> bytes:
    fn = lib.selftest_native_elf_tensor_batchsum_i64
    fn.restype = ctypes.c_int64
    fn.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
    es = fn(b, m, n)
    rd = lambda addr, o=0: ctypes.cast(addr + o, ctypes.POINTER(ctypes.c_int64))[0]
    sh = rd(es, 0)  # buf: String handle (addr/len/cap)
    ln = rd(sh, 8)
    if ln <= 0:
        return b""
    return ctypes.string_at(rd(sh, 0), ln)


def ref_batchsum_wsq(b: int, m: int, n: int) -> int:
    """Independent pure-Python reference: seed the flat 3-D tensor, per-batch
    reduce, then fold with the (k+1)-weight AND the square."""
    # A[k][i][j] = k*m*n + i*n + j + 1 (the flat index + 1).
    total = 0
    for k in range(b):
        s = 0
        for i in range(m):
            for j in range(n):
                s += k * m * n + i * n + j + 1
        total += (k + 1) * s * s
    return total


def ref_batchsum_wsq_transposed(b: int, m: int, n: int) -> int:
    """The WRONG batch-stride value — used only to prove the correct value is
    discriminating. Here the SAME flat buffer is re-partitioned as if the batch
    axis were the FASTEST-varying one: element t = k*m*n+i*n+j is assigned to
    batch (t mod b) instead of batch (t // (m*n)). On a cubic shape (b==m==n)
    this can coincide; on a non-cubic shape it must differ."""
    flat = [k * m * n + i * n + j + 1
            for k in range(b) for i in range(m) for j in range(n)]
    buckets = [0] * b
    for t, v in enumerate(flat):
        buckets[t % b] += v
    total = 0
    for k in range(b):
        total += (k + 1) * buckets[k] * buckets[k]
    return total


def run_elf(elf: bytes, tmp: pathlib.Path):
    p = tmp / "mind_tensor_batchsum.elf"
    p.write_bytes(elf)
    p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    r = subprocess.run([str(p)], capture_output=True)
    return r.returncode, r.stdout


def main() -> int:
    so = os.environ.get("MINDC_SO")
    if not so and resolve_so is not None:
        so = str(resolve_so())
    if not so:
        so = str(_DEFAULT_SO)
    if not os.path.exists(so):
        if os.environ.get("MINDC_SO"):
            print(f"FAIL  MINDC_SO set but missing: {so!r}")
            return 1
        print(f"SKIP  {so} not built")
        return 0
    lib = ctypes.CDLL(so)
    if not hasattr(lib, "selftest_native_elf_tensor_batchsum_i64"):
        print("FAIL  selftest_native_elf_tensor_batchsum_i64: symbol absent (C4-T6 not built)")
        return 1

    # Distinct shapes: unit, cubic, and several NON-CUBIC (b!=m, m!=n, b!=n) that
    # make the transposed-batch-stride variant provably distinct. Single-batch,
    # single-row, single-col edges. The frame cap (b*m*n = 4096: 16x16x16 and
    # 64x8x8) pushes the checksum far past imm32 — exercises the movabs baking.
    shapes = [
        (1, 1, 1), (2, 3, 4), (4, 3, 2), (3, 1, 5), (1, 4, 6), (5, 5, 1),
        (2, 2, 2), (2, 8, 8), (8, 8, 2), (16, 16, 16), (64, 8, 8),
    ]
    all_ok = True
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        for (b, m, n) in shapes:
            expected_sum = ref_batchsum_wsq(b, m, n)
            wrong_sum = ref_batchsum_wsq_transposed(b, m, n)
            # Discriminating-by-construction: with more than one batch (b > 1) and
            # a non-cubic shape, the transposed batch stride gives a different
            # value, so a value == expected_sum proves the correct batch partition
            # was used. (b == 1 has a single trivial batch — no stride ambiguity;
            # a cubic shape can coincide, so both are excluded from the assert.)
            if b > 1 and not (b == m and m == n):
                assert expected_sum != wrong_sum, (
                    f"non-discriminating shape {b}x{m}x{n}: correct==transposed "
                    f"batch-stride checksum"
                )
            elf = mind_batchsum_elf(lib, b, m, n)
            if not (len(elf) > 120 and elf[:4] == b"\x7fELF"):
                print(f"  FAIL  batchsum({b}x{m}x{n}): not a runnable ELF (len={len(elf)})")
                all_ok = False
                continue
            code, out = run_elf(elf, tmp)
            want = struct.pack("<q", expected_sum)
            ok = code == 42 and out == want
            all_ok = all_ok and ok
            got_sum = struct.unpack("<q", out)[0] if len(out) == 8 else None
            print(
                f"  {'PASS' if ok else 'FAIL'}  batchsum({b}x{m}x{n}) -> exit={code} "
                f"(want 42) stdout_sum={got_sum} expected_sum={expected_sum} "
                f"transposed_sum={wrong_sum} (elf {len(elf)}B, seed loop + 3-D "
                f"triple-reduce nest over (k*m*n+i*n+j)*8+base addressing, "
                f"(k+1)-weighted squared batch fold, native x86-64, zero MLIR/LLVM)"
            )

        # Shape guard: out-of-frame, degenerate, and i64-overflow shapes must
        # FAIL CLOSED (empty buffer) — the frame-overrun audit hazard. The last
        # rows are i64-overflow shapes: a product wraps to 0 (mod 2^64), so the
        # product check alone would pass them; the per-dim `> 4096` bound (applied
        # before the product) is what refuses them. 2^32=4294967296.
        for (b, m, n) in [
            (2, 64, 64), (64, 64, 2), (4097, 1, 1), (1, 4097, 1), (1, 1, 4097),
            (0, 1, 1), (1, 0, 1), (1, 1, 0), (1, -3, 1), (-3, 1, 1),
            (4294967296, 4294967296, 1),
            (1, 4294967296, 4294967296),
            (65536, 65536, 65536),
        ]:
            elf = mind_batchsum_elf(lib, b, m, n)
            ok = len(elf) == 0
            all_ok = all_ok and ok
            print(
                f"  {'PASS' if ok else 'FAIL'}  batchsum({b}x{m}x{n}) refused "
                f"(fail-closed shape guard, got {len(elf)}B want 0B)"
            )
    if all_ok:
        print(
            "ALL PASS  3-D batched sum lowers native-ELF end to end — a "
            "self-seeded b×m×n i64 tensor, an emitted seed loop + a triple "
            "reduce nest over genuine 3-D flat (k*m*n+i*n+j)*8+base addressing, "
            "a (k+1)-weighted SQUARED (batch-partition-discriminating) fold, "
            "full-width stdout check + exact-i64 in-ELF comparison (movabs-baked "
            "past imm32), fail-closed frame-bound guard, transposed batch stride "
            "proven distinct on non-cubic shapes, zero MLIR/LLVM (C4-T6)"
        )
        return 0
    print("FAIL  native-ELF tensor batchsum gate")
    return 1


if __name__ == "__main__":
    sys.exit(main())
