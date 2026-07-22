#!/usr/bin/env python3
"""C4-T5 — native-ELF tensor ROW-VECTOR BROADCAST ADD (i64), zero MLIR/LLVM.

selftest_native_elf_tensor_bcastadd_i64(m, n) emits a runnable x86-64 ET_EXEC that
seeds A (MxN) A[i*n+j]=i*n+j+1 and a length-N row vector B[j]=j*3+1, computes
C[i*n+j]=A[i*n+j]+B[j] with B BROADCAST across all m rows (B addressed by column j
only, stride 0 in i), folds a POSITION-WEIGHTED checksum chk=Sum (i*n+j)*C[i*n+j],
writes 8 LE bytes to stdout, exits (chk==expected)*41+1. Two independent full-width
gates: stdout==pure-Python reference AND exit==42 (in-ELF exact i64, movabs-baked).
Wrong-axis (B[i] down columns) and no-broadcast (C=A) references are proven DISTINCT
on non-square shapes. Fail-closed shape guard (1<=m,n<=4096, m*n<=4096) asserted incl
i64-overflow shapes. ADDITIVE export: integer native-ELF oracle stays byte-identical.

Usage: MINDC_SO=/path/to.so python3 self_host_native_tensor_bcastadd_smoke.py
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


def mind_bcastadd_elf(lib, m: int, n: int) -> bytes:
    fn = lib.selftest_native_elf_tensor_bcastadd_i64
    fn.restype = ctypes.c_int64
    fn.argtypes = [ctypes.c_int64, ctypes.c_int64]
    es = fn(m, n)
    rd = lambda addr, o=0: ctypes.cast(addr + o, ctypes.POINTER(ctypes.c_int64))[0]
    sh = rd(es, 0)  # buf: String handle (addr/len/cap)
    ln = rd(sh, 8)
    if ln <= 0:
        return b""
    return ctypes.string_at(rd(sh, 0), ln)


def ref_bcastadd_chk(m: int, n: int) -> int:
    """Independent pure-Python reference: seed A + row-vector B, broadcast-add
    C[i*n+j] = A[i*n+j] + B[j], position-weighted checksum Sum (i*n+j)*C[i*n+j]."""
    a = [[i * n + j + 1 for j in range(n)] for i in range(m)]
    b = [j * 3 + 1 for j in range(n)]  # length-N row vector
    chk = 0
    for i in range(m):
        for j in range(n):
            idx = i * n + j
            c = a[i][j] + b[j]  # B broadcast across rows (index by column j)
            chk += idx * c
    return chk


def ref_bcastadd_wrong_axis_chk(m: int, n: int) -> int:
    """The WRONG-axis value: broadcast a length-M vector DOWN columns (add B[i]
    instead of B[j]) — used only to prove the correct value is discriminating."""
    a = [[i * n + j + 1 for j in range(n)] for i in range(m)]
    b = [i * 3 + 1 for i in range(m)]  # length-M vector, broadcast down columns
    chk = 0
    for i in range(m):
        for j in range(n):
            idx = i * n + j
            c = a[i][j] + b[i]  # WRONG: index by row i
            chk += idx * c
    return chk


def ref_no_broadcast_chk(m: int, n: int) -> int:
    """The no-broadcast value: C = A (B dropped) — a second wrong variant proven
    distinct, so the correct value could not be a degenerate identity."""
    a = [[i * n + j + 1 for j in range(n)] for i in range(m)]
    chk = 0
    for i in range(m):
        for j in range(n):
            idx = i * n + j
            chk += idx * a[i][j]
    return chk


def run_elf(elf: bytes, tmp: pathlib.Path):
    p = tmp / "mind_tensor_bcastadd.elf"
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
    if not hasattr(lib, "selftest_native_elf_tensor_bcastadd_i64"):
        print("FAIL  selftest_native_elf_tensor_bcastadd_i64: symbol absent (C4-T5 not built)")
        return 1

    shapes = [(1, 1), (2, 3), (3, 2), (1, 16), (16, 1), (8, 8), (2, 64), (64, 64)]
    all_ok = True
    saw_past_imm32 = False
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        for (m, n) in shapes:
            expected_chk = ref_bcastadd_chk(m, n)
            wrong_axis = ref_bcastadd_wrong_axis_chk(m, n)
            no_bcast = ref_no_broadcast_chk(m, n)
            if m != n:
                assert expected_chk != wrong_axis, (
                    f"non-discriminating shape {m}x{n}: correct==wrong-axis checksum"
                )
                assert expected_chk != no_bcast, (
                    f"non-discriminating shape {m}x{n}: correct==no-broadcast checksum"
                )
            if expected_chk > 0x7FFFFFFF:
                saw_past_imm32 = True
            elf = mind_bcastadd_elf(lib, m, n)
            if not (len(elf) > 120 and elf[:4] == b"\x7fELF"):
                print(f"  FAIL  bcastadd({m}x{n}): not a runnable ELF (len={len(elf)})")
                all_ok = False
                continue
            code, out = run_elf(elf, tmp)
            want = struct.pack("<q", expected_chk)
            ok = code == 42 and out == want
            all_ok = all_ok and ok
            got_chk = struct.unpack("<q", out)[0] if len(out) == 8 else None
            print(
                f"  {'PASS' if ok else 'FAIL'}  bcastadd({m}x{n}) -> exit={code} "
                f"(want 42) stdout_chk={got_chk} expected_chk={expected_chk} "
                f"wrong_axis={wrong_axis} no_bcast={no_bcast} (elf {len(elf)}B, "
                f"seed A nest + row-vector B seed + broadcast-add nest (stride-0 "
                f"in i) + position-weighted checksum, native x86-64, zero MLIR/LLVM)"
            )

        if not saw_past_imm32:
            print("  FAIL  no shape pushed the checksum past imm32 (movabs baking untested)")
            all_ok = False

        for (m, n) in [
            (65, 64), (64, 65), (1, 4097), (4097, 1), (0, 1), (1, 0), (1, -3), (-3, 1),
            (4294967296, 4294967296),
            (8589934592, 8589934592),
        ]:
            elf = mind_bcastadd_elf(lib, m, n)
            ok = len(elf) == 0
            all_ok = all_ok and ok
            print(
                f"  {'PASS' if ok else 'FAIL'}  bcastadd({m}x{n}) refused "
                f"(fail-closed shape guard, got {len(elf)}B want 0B)"
            )
    if all_ok:
        print(
            "ALL PASS  tensor row-vector broadcast-add lowers native-ELF end to "
            "end — stride-0 column addressing for B, position-weighted checksum, "
            "full-width stdout + exact-i64 in-ELF comparison (movabs past imm32), "
            "fail-closed frame-bound guard, wrong-axis + no-broadcast proven "
            "distinct, zero MLIR/LLVM (C4-T5)"
        )
        return 0
    print("FAIL  native-ELF tensor broadcast-add gate")
    return 1


if __name__ == "__main__":
    sys.exit(main())