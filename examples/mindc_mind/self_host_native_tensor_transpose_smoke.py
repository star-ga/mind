#!/usr/bin/env python3
"""C4-T4 — native-ELF tensor TRANSPOSE (i64), zero MLIR/LLVM.

The layout-permuting rung of the tensor/linalg ladder.
`selftest_native_elf_tensor_transpose_i64(m, n)` emits a runnable x86-64
ET_EXEC that (1) self-seeds the MxN input A row-major in the frame with the
deterministic nonzero pattern A[i*n+j] = i*n+j+1, (2) writes the NxM transpose
B[j*m+i] = A[i*n+j] with an emitted i/j nest whose A read (i*n+j) and B write
(j*m+i) addresses are BOTH formed by the ((a*dim)+b)*8+base 2-D shape — the two
operands use OPPOSITE (dim,stride) pairs, which is the whole point of a
transpose, (3) computes a POSITION-WEIGHTED checksum chk = Sum idx * B[idx] over
the output buffer (a plain grand-sum would be layout-invariant and could NOT
distinguish a real transpose from the identity / a mis-strided copy; the idx
weight makes any wrong output index change chk), (4) writes the 8 LE bytes of
`chk` to stdout, and (5) exits (chk == expected)*41 + 1 — 42 only on an EXACT
full-width i64 match against the emit-time-baked expected checksum (movabs-baked
past imm32), 1 otherwise.

Two independent full-width gates per shape:
  (a) stdout == struct.pack('<q', S) where S is THIS script's pure-Python
      transpose-and-weight over the same seeds — an independent reference
      (no shared code with the emitter);
  (b) exit == 42 — the in-ELF exact-i64 comparison against the baked checksum.
Because the checksum is position-weighted, an identity copy (B=A), a wrong B
stride (i*n+j instead of j*m+i), or a wrong A read stride changes BOTH
observables on the non-square shapes (non-vacuous; multiple distinct shapes so a
fluke cannot pass). This script also cross-checks that a DELIBERATELY WRONG
(identity / transposed-stride) reference disagrees with the emitter on a
non-square shape, proving the checksum is discriminating rather than
layout-invariant.

SHAPE GUARD (mirror of the T3 matmul guard): the export FAILS CLOSED — returns
an empty buffer — unless 1 <= m,n and m <= 4096 and n <= 4096 and m*n <= 4096
(two 4096-element frame arrays). This smoke asserts the refusal on out-of-bounds
and degenerate shapes too, including i64-overflow shapes whose product wraps.

ADDITIVITY: a NEW export never reached during self-compile, so the integer
native-ELF oracle stays byte-identical. No frozen byte oracle exists for this
path; the gate is EXECUTION CORRECTNESS.

Usage:
  MINDC_SO=/path/to.so python3 self_host_native_tensor_transpose_smoke.py
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


def mind_transpose_elf(lib, m: int, n: int) -> bytes:
    fn = lib.selftest_native_elf_tensor_transpose_i64
    fn.restype = ctypes.c_int64
    fn.argtypes = [ctypes.c_int64, ctypes.c_int64]
    es = fn(m, n)
    rd = lambda addr, o=0: ctypes.cast(addr + o, ctypes.POINTER(ctypes.c_int64))[0]
    sh = rd(es, 0)  # buf: String handle (addr/len/cap)
    ln = rd(sh, 8)
    if ln <= 0:
        return b""
    return ctypes.string_at(rd(sh, 0), ln)


def ref_transpose_checksum(m: int, n: int) -> int:
    """Independent pure-Python reference: seed A, transpose into B, then
    position-weight. B is NxM with B[j*m+i] = A[i*n+j] = i*n+j+1; the checksum
    is Sum_{idx < N*M} idx * B[idx]."""
    a = [[i * n + j + 1 for j in range(n)] for i in range(m)]
    b = [0] * (n * m)
    for i in range(m):
        for j in range(n):
            b[j * m + i] = a[i][j]
    return sum(idx * v for idx, v in enumerate(b))


def wrong_identity_checksum(m: int, n: int) -> int:
    """A DELIBERATELY WRONG reference: no transpose at all (B == A flattened).
    Must disagree with the real transpose on a non-square shape — this is what
    proves the position-weighted checksum is layout-discriminating."""
    flat = [i * n + j + 1 for i in range(m) for j in range(n)]
    return sum(idx * v for idx, v in enumerate(flat))


def run_elf(elf: bytes, tmp: pathlib.Path):
    p = tmp / "mind_tensor_transpose.elf"
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
    if not hasattr(lib, "selftest_native_elf_tensor_transpose_i64"):
        print("FAIL  selftest_native_elf_tensor_transpose_i64: symbol absent (C4-T4 not built)")
        return 1

    # Sanity: the position-weighted checksum must actually distinguish a real
    # transpose from an identity copy on a non-square shape (else the whole gate
    # is vacuous). Square 1x1 is invariant by construction — use a non-square.
    if ref_transpose_checksum(4, 5) == wrong_identity_checksum(4, 5):
        print("FAIL  reference checksum is NOT layout-discriminating (bug in test)")
        return 1

    # Distinct shapes: 1x1 (trivial), non-square both orientations (strides
    # differ, catches i*n+j vs j*m+i index bugs), row/col vectors, square mid,
    # and the frame cap (64x64 = 4096, and 2x2048 = 4096). The larger shapes
    # push the checksum past imm32 — exercises the movabs baking.
    shapes = [(1, 1), (2, 3), (4, 5), (5, 4), (3, 1), (1, 7), (16, 16), (2, 2048), (64, 64)]
    all_ok = True
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        for (m, n) in shapes:
            expected = ref_transpose_checksum(m, n)
            elf = mind_transpose_elf(lib, m, n)
            if not (len(elf) > 120 and elf[:4] == b"\x7fELF"):
                print(f"  FAIL  transpose({m}x{n}): not a runnable ELF (len={len(elf)})")
                all_ok = False
                continue
            code, out = run_elf(elf, tmp)
            want = struct.pack("<q", expected)
            ok = code == 42 and out == want
            all_ok = all_ok and ok
            got = struct.unpack("<q", out)[0] if len(out) == 8 else None
            # On genuinely 2-D non-square shapes (m>1 AND n>1) the emitter's
            # value must also DIFFER from the identity-copy checksum — extra
            # proof it really transposed. Vector shapes (m==1 or n==1) are
            # transpose-invariant in flattened layout, so skip them here.
            note = ""
            if m != n and m > 1 and n > 1:
                wid = wrong_identity_checksum(m, n)
                if got == wid:
                    ok = False
                    all_ok = False
                    note = f" [!! matches identity-copy checksum {wid} — NOT transposed]"
                else:
                    note = f" [!= identity-copy {wid}, transpose confirmed]"
            print(
                f"  {'PASS' if ok else 'FAIL'}  transpose({m}x{n}) -> exit={code} "
                f"(want 42) stdout_chk={got} expected_chk={expected}{note} "
                f"(elf {len(elf)}B, i/j nest, opposite 2-D strides, position-"
                f"weighted checksum, native x86-64, zero MLIR/LLVM)"
            )

        # Shape guard: out-of-frame and degenerate shapes must FAIL CLOSED
        # (empty buffer). The last two are i64-overflow shapes: the product m*n
        # wraps to 0 (mod 2^64), so an `m*n > 4096` check alone would pass them
        # through — the per-dim `> 4096` bound (applied before the product) is
        # what refuses them. 2^32=4294967296, 2^33=8589934592.
        for (m, n) in [
            (65, 64), (64, 65), (1, 4097), (4097, 1), (0, 1), (1, 0), (1, -3), (-2, 2),
            (4294967296, 4294967296),
            (8589934592, 8589934592),
        ]:
            elf = mind_transpose_elf(lib, m, n)
            ok = len(elf) == 0
            all_ok = all_ok and ok
            print(
                f"  {'PASS' if ok else 'FAIL'}  transpose({m}x{n}) refused "
                f"(fail-closed shape guard, got {len(elf)}B want 0B)"
            )
    if all_ok:
        print(
            "ALL PASS  tensor transpose lowers native-ELF end to end — a "
            "self-seeded row-major i64 input, an emitted i/j nest writing the "
            "transpose through OPPOSITE 2-D row-major strides "
            "(A i*n+j -> B j*m+i via ((a*dim)+b)*8+base), a position-weighted "
            "checksum that catches any wrong/identity/mis-strided layout, "
            "full-width stdout check + exact-i64 in-ELF comparison "
            "(movabs-baked past imm32), fail-closed frame-bound guard, "
            "zero MLIR/LLVM (C4-T4)"
        )
        return 0
    print("FAIL  native-ELF tensor transpose gate")
    return 1


if __name__ == "__main__":
    sys.exit(main())
