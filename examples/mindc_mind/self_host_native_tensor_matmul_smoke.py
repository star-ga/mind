#!/usr/bin/env python3
"""C4-T3 — native-ELF tensor MATMUL (i64), zero MLIR/LLVM.

The nested-loop 2-D addressing rung of the tensor/linalg ladder.
`selftest_native_elf_tensor_matmul_i64(m, k, n)` emits a runnable x86-64
ET_EXEC that (1) materializes A (MxK) and B (KxN) row-major in the frame with
deterministic self-seeding A[r*K+c] = r+c+1, B[p*N+j] = p*j+1, (2) computes
C = A @ B as the canonical triple nest (for i in M: for j in N: acc = 0;
for p in K: acc += A[i*K+p]*B[p*N+j]; C[i*N+j] = acc) with all three 2-D
row-major element addresses formed at runtime as ((a*dim)+b)*8 + base,
(3) reduces sum = Sum C[t] over the M*N result via a fourth counted loop,
(4) writes the 8 LE bytes of `sum` to stdout, and (5) exits with
(sum == expected)*41 + 1 — 42 only on an EXACT full-width i64 match against
the emit-time-baked expected total (movabs-baked past imm32), 1 otherwise.

Two independent full-width gates per shape:
  (a) stdout == struct.pack('<q', S) where S is THIS script's pure-Python
      matmul-and-reduce over the same seeds — an independent reference
      implementation (numpy-style, no shared code with the emitter);
  (b) exit == 42 — the in-ELF exact-i64 comparison against the baked total.
A wrong index stride (i*K+p vs i*N+p), wrong loop nesting, wrong bound, or a
dropped acc reset changes BOTH observables on the non-square shapes
(non-vacuous; multiple distinct shapes so a fluke cannot pass).

SHAPE GUARD (the T1/T2 audit hazard): the export FAILS CLOSED — returns an
empty buffer — unless 1 <= m,k,n and m*k <= 4096 and k*n <= 4096 and
m*n <= 4096 (three 4096-element frame arrays). This smoke asserts the refusal
on out-of-bounds and degenerate shapes too.

ADDITIVITY: a NEW export never reached during self-compile, so the integer
native-ELF oracle stays byte-identical. No frozen byte oracle exists for this
path; the gate is EXECUTION CORRECTNESS.

Usage:
  MINDC_SO=/path/to.so python3 self_host_native_tensor_matmul_smoke.py
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


def mind_matmul_elf(lib, m: int, k: int, n: int) -> bytes:
    fn = lib.selftest_native_elf_tensor_matmul_i64
    fn.restype = ctypes.c_int64
    fn.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
    es = fn(m, k, n)
    rd = lambda addr, o=0: ctypes.cast(addr + o, ctypes.POINTER(ctypes.c_int64))[0]
    sh = rd(es, 0)  # buf: String handle (addr/len/cap)
    ln = rd(sh, 8)
    if ln <= 0:
        return b""
    return ctypes.string_at(rd(sh, 0), ln)


def ref_matmul_sum(m: int, k: int, n: int) -> int:
    """Independent pure-Python reference: seed, matmul, reduce."""
    a = [[r + c + 1 for c in range(k)] for r in range(m)]
    b = [[r * c + 1 for c in range(n)] for r in range(k)]
    total = 0
    for i in range(m):
        for j in range(n):
            acc = 0
            for p in range(k):
                acc += a[i][p] * b[p][j]
            total += acc
    return total


def run_elf(elf: bytes, tmp: pathlib.Path):
    p = tmp / "mind_tensor_matmul.elf"
    p.write_bytes(elf)
    p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    r = subprocess.run([str(p)], capture_output=True)
    return r.returncode, r.stdout


def main() -> int:
    so = os.environ.get("MINDC_SO", str(_DEFAULT_SO))
    if not os.path.exists(so):
        if os.environ.get("MINDC_SO"):
            print(f"FAIL  MINDC_SO set but missing: {so!r}")
            return 1
        print(f"SKIP  {so} not built")
        return 0
    lib = ctypes.CDLL(so)
    if not hasattr(lib, "selftest_native_elf_tensor_matmul_i64"):
        print("FAIL  selftest_native_elf_tensor_matmul_i64: symbol absent (C4-T3 not built)")
        return 1

    # Distinct shapes: 1x1x1, non-square (strides differ, catches i*K vs i*N
    # index bugs), k=1 edge, square mid, deep-K reduction, and the frame cap
    # (64x64x64: m*k = k*n = m*n = 4096 exactly). The larger shapes push the
    # reduction total past imm32 — exercises the movabs baking.
    shapes = [(1, 1, 1), (2, 3, 2), (4, 5, 3), (3, 1, 5), (16, 16, 16), (2, 1024, 2), (64, 64, 64)]
    all_ok = True
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        for (m, k, n) in shapes:
            expected_sum = ref_matmul_sum(m, k, n)
            elf = mind_matmul_elf(lib, m, k, n)
            if not (len(elf) > 120 and elf[:4] == b"\x7fELF"):
                print(f"  FAIL  matmul({m}x{k}x{n}): not a runnable ELF (len={len(elf)})")
                all_ok = False
                continue
            code, out = run_elf(elf, tmp)
            want = struct.pack("<q", expected_sum)
            ok = code == 42 and out == want
            all_ok = all_ok and ok
            got_sum = struct.unpack("<q", out)[0] if len(out) == 8 else None
            print(
                f"  {'PASS' if ok else 'FAIL'}  matmul({m}x{k}x{n}) -> exit={code} "
                f"(want 42) stdout_sum={got_sum} expected_sum={expected_sum} "
                f"(elf {len(elf)}B, triple-nested MAC + 2 seed nests + reduce, "
                f"native x86-64, zero MLIR/LLVM)"
            )

        # Shape guard: out-of-frame and degenerate shapes must FAIL CLOSED
        # (empty buffer) — the T1/T2 frame-overrun audit hazard.
        for (m, k, n) in [(65, 64, 1), (1, 1, 4097), (0, 1, 1), (1, -3, 1)]:
            elf = mind_matmul_elf(lib, m, k, n)
            ok = len(elf) == 0
            all_ok = all_ok and ok
            print(
                f"  {'PASS' if ok else 'FAIL'}  matmul({m}x{k}x{n}) refused "
                f"(fail-closed shape guard, got {len(elf)}B want 0B)"
            )
    if all_ok:
        print(
            "ALL PASS  tensor matmul lowers native-ELF end to end — two "
            "self-seeded 2-D row-major i64 operands, an emitted triple-nested "
            "i/j/p MAC loop over ((a*dim)+b)*8+base addressing, per-element "
            "C store + horizontal reduction, full-width stdout check + "
            "exact-i64 in-ELF comparison (movabs-baked past imm32), "
            "fail-closed frame-bound guard, zero MLIR/LLVM (C4-T3)"
        )
        return 0
    print("FAIL  native-ELF tensor matmul gate")
    return 1


if __name__ == "__main__":
    sys.exit(main())
