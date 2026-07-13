#!/usr/bin/env python3
"""RI-B1 nb_expr FLOAT-op-FLOAT arithmetic routing gate (zero MLIR/LLVM).

The next connecting construct after float-literal lowering: nb_expr's binop arm now
propagates FLOAT dtype through its operands (nb_node_dtype) and, when BOTH operands
are FLOAT and the operator is scalar arithmetic (+ - * /), routes the SSE2 f64
memory-operand encoders (movsd load lhs -> xmm0 ; <op>sd xmm0,[rhs slot] ; movsd
store xmm0 -> dst slot) instead of the GP-integer path.

`selftest_native_elf_fp_binop(src, len)` lexes + parses a REAL float ARITHMETIC
source (e.g. `1.5 + 2.5`) with parse_expr, lowers the WHOLE binop AST THROUGH
nb_expr (not the fp encoders directly), then loads the result slot back to xmm0 and
cvttsd2si -> exit(trunc(a OP b)), wrapped in a real runnable ELF via the same
nb_write_elf scaffold the integer path uses.

No frozen float oracle exists (the deleted Rust native backend returned
Unsupported(ConstF64)), so byte-identity is impossible here by construction. The
oracle is EXECUTION CORRECTNESS: emit the ELF, run it, assert exit == trunc(a OP b).
The CPU is the reference. A wrong SSE op-selection, wrong operand slot, or wrong
IEEE-754 bits either faults or returns the wrong integer, so this is a genuine,
non-fakeable proof that a real float arithmetic program lowers to native-ELF end to
end with zero MLIR/LLVM.

Usage:
  MINDC_SO=/path/to.so python3 self_host_native_fp_binop_smoke.py
"""
import ctypes
import os
import pathlib
import stat
import subprocess
import sys
import tempfile

_HERE = pathlib.Path(__file__).parent
_DEFAULT_SO = _HERE / "libmindc_mind.so"


def mind_fp_binop_elf(lib, src: str) -> bytes:
    fn = lib.selftest_native_elf_fp_binop
    fn.restype = ctypes.c_int64
    fn.argtypes = [ctypes.c_int64, ctypes.c_int64]
    buf = ctypes.create_string_buffer(src.encode(), len(src.encode()))
    es = fn(ctypes.cast(buf, ctypes.c_void_p).value, len(src.encode()))
    rd = lambda addr, o=0: ctypes.cast(addr + o, ctypes.POINTER(ctypes.c_int64))[0]
    sh = rd(es, 0)  # buf: String handle (addr/len/cap)
    return ctypes.string_at(rd(sh, 0), rd(sh, 8))


def run_elf(elf: bytes, tmp: pathlib.Path) -> int:
    p = tmp / "mind_fp_binop.elf"
    p.write_bytes(elf)
    p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return subprocess.run([str(p)]).returncode


def main() -> int:
    so = os.environ.get("MINDC_SO", str(_DEFAULT_SO))
    if not os.path.exists(so):
        if os.environ.get("MINDC_SO"):
            print(f"FAIL  MINDC_SO set but missing: {so!r}")
            return 1
        print(f"SKIP  {so} not built")
        return 0
    lib = ctypes.CDLL(so)
    if not hasattr(lib, "selftest_native_elf_fp_binop"):
        print("FAIL  selftest_native_elf_fp_binop: symbol absent (nb_expr float-binop arm not built)")
        return 1

    # (float arith source, expected exit = int(trunc(a OP b))). All operands are
    # exactly-representable dyadic literals, so the integer-only decimal->IEEE-754
    # path is exact and the arithmetic result is exact.
    cases = [
        ("1.5 + 2.5", 4),
        ("7.5 - 2.5", 5),
        ("2.0 * 3.0", 6),
        ("7.5 / 2.5", 3),
        ("1.25 + 0.5", 1),
        ("10.0 / 4.0", 2),
    ]
    all_ok = True
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        for src, expected in cases:
            elf = mind_fp_binop_elf(lib, src)
            if not (len(elf) > 120 and elf[:4] == b"\x7fELF"):
                print(f"  FAIL  {src!r}: not a runnable ELF (len={len(elf)})")
                all_ok = False
                continue
            got = run_elf(elf, tmp)
            ok = got == expected
            all_ok = all_ok and ok
            print(
                f"  {'PASS' if ok else 'FAIL'}  {src!r:>12} -> nb_expr float-binop arm "
                f"exit={got} expected={expected} "
                f"(elf {len(elf)}B, SSE2 native, zero MLIR/LLVM)"
            )
    if all_ok:
        print(
            "ALL PASS  nb_expr routes FLOAT-op-FLOAT through the SSE2 f64 binop "
            "encoders — a real float arithmetic program lowers to native-ELF end to "
            "end with zero MLIR/LLVM (RI-B1 float-scalar arithmetic path connected)"
        )
        return 0
    print("FAIL  nb_expr float-binop routing gate")
    return 1


if __name__ == "__main__":
    sys.exit(main())
