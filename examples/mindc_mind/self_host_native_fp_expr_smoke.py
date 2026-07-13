#!/usr/bin/env python3
"""RI-B1 nb_expr float-scalar routing gate (zero MLIR/LLVM).

The FINAL connecting construct: nb_expr now routes an ast_float_lit's value to the
SSE2 scalar-double encoders (nb_fp_const_xmm + movsd-store) instead of the
GP-integer const path. `selftest_native_elf_fp_expr(src, len)` lexes + parses a REAL
float literal source, lowers it THROUGH nb_expr (not the fp encoders directly), then
loads the slot back to xmm0 and cvttsd2si -> exit(trunc(value)), wrapped in a real
runnable ELF via the same nb_write_elf scaffold the integer path uses.

No frozen float oracle exists (the deleted Rust native backend returned
Unsupported(ConstF64)), so byte-identity is impossible here by construction. The
oracle is EXECUTION CORRECTNESS: emit the ELF, run it, assert exit == trunc(value).
The CPU is the reference. A wrong SSE encoding or wrong IEEE-754 bits either faults
or returns the wrong integer, so this is a genuine, non-fakeable proof that a real
float source program lowers to native-ELF end to end with zero MLIR/LLVM.

Usage:
  MINDC_SO=/path/to.so python3 self_host_native_fp_expr_smoke.py
"""
import ctypes
import os
import pathlib
import stat
import subprocess
import sys
import tempfile

_HERE = pathlib.Path(__file__).parent
_DEFAULT_SO = _HERE / "libmindc_mind.so"  # legacy in-tree path (fallback only)
sys.path.insert(0, str(_HERE))
from _selfhost_so import resolve_so  # noqa: E402


def mind_fp_expr_elf(lib, src: str) -> bytes:
    fn = lib.selftest_native_elf_fp_expr
    fn.restype = ctypes.c_int64
    fn.argtypes = [ctypes.c_int64, ctypes.c_int64]
    buf = ctypes.create_string_buffer(src.encode(), len(src.encode()))
    es = fn(ctypes.cast(buf, ctypes.c_void_p).value, len(src.encode()))
    rd = lambda addr, o=0: ctypes.cast(addr + o, ctypes.POINTER(ctypes.c_int64))[0]
    sh = rd(es, 0)  # buf: String handle (addr/len/cap)
    return ctypes.string_at(rd(sh, 0), rd(sh, 8))


def run_elf(elf: bytes, tmp: pathlib.Path) -> int:
    p = tmp / "mind_fp_expr.elf"
    p.write_bytes(elf)
    p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return subprocess.run([str(p)]).returncode


def main() -> int:
    so = str(resolve_so())  # MINDC_SO verbatim, else fresh build (never stale)
    if not os.path.exists(so):
        if os.environ.get("MINDC_SO"):
            print(f"FAIL  MINDC_SO set but missing: {so!r}")
            return 1
        print(f"SKIP  {so} not built")
        return 0
    lib = ctypes.CDLL(so)
    if not hasattr(lib, "selftest_native_elf_fp_expr"):
        print("FAIL  selftest_native_elf_fp_expr: symbol absent (nb_expr float arm not built)")
        return 1

    # (float source, expected exit = int(trunc(value))). All exactly-representable
    # dyadic literals, so the integer-only decimal->IEEE-754 path is exact.
    cases = [
        ("1.5", 1),
        ("2.5", 2),
        ("3.0", 3),
        ("0.5", 0),
        ("7.5", 7),
        ("12.25", 12),
    ]
    all_ok = True
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        for src, expected in cases:
            elf = mind_fp_expr_elf(lib, src)
            if not (len(elf) > 120 and elf[:4] == b"\x7fELF"):
                print(f"  FAIL  {src!r}: not a runnable ELF (len={len(elf)})")
                all_ok = False
                continue
            got = run_elf(elf, tmp)
            ok = got == expected
            all_ok = all_ok and ok
            print(
                f"  {'PASS' if ok else 'FAIL'}  {src!r:>8} -> nb_expr float arm "
                f"exit={got} expected={expected} "
                f"(elf {len(elf)}B, SSE2 native, zero MLIR/LLVM)"
            )
    if all_ok:
        print(
            "ALL PASS  nb_expr routes ast_float_lit through the SSE2 f64 encoders — "
            "a real float source program lowers to native-ELF end to end with zero "
            "MLIR/LLVM (RI-B1 float-scalar path connected)"
        )
        return 0
    print("FAIL  nb_expr float-scalar routing gate")
    return 1


if __name__ == "__main__":
    sys.exit(main())
