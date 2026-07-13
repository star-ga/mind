#!/usr/bin/env python3
"""RI-B1 (#107 follow-up) FLOAT dtype propagation ACROSS a LET binding (zero MLIR/LLVM).

The next connecting construct after float-literal + FLOAT-op-FLOAT lowering: nb_expr's
binop arm now consults a per-fn SSA dtype table (threaded at lcell+8, the same plumbing
nb_lower_fn uses), and the nb_stmt `let` arm tags the bound slot with the init's dtype.
So a float let-bound operand -- `let x: f64 = 1.5; x + 2.5`, where `x` is an ast_ident
the pure-AST classifier calls INT -- is now seen as FLOAT and routes the SSE2 f64
memory-operand encoders.

`selftest_native_elf_fp_let(src, len)` lexes + parses a REAL statement block with
parse_block, lowers the WHOLE block THROUGH nb_block_stmts / nb_stmt / nb_expr (not the
fp encoders directly), tags each `let`'s slot, then loads the tail expr's result slot
back to xmm0 and cvttsd2si -> exit(trunc(value)), wrapped in a real runnable ELF via the
same nb_write_elf scaffold the integer path uses.

The self-host front-end requires typed lets (`let X: T = INIT`), exactly as main.mind's
own source is written; the dtype is taken from the INIT expression, not the annotation.

No frozen float oracle exists (the deleted Rust native backend returned
Unsupported(ConstF64)), so byte-identity is impossible here by construction. The oracle
is EXECUTION CORRECTNESS: emit the ELF, run it, assert exit == trunc(value). WITHOUT the
dtype-table threading `x` would classify INT and the binop would take the GP-integer
path, faulting or returning the wrong integer -- so a correct exit is a non-fakeable
proof the FLOAT dtype crossed the let.

Usage:
  MINDC_SO=/path/to.so python3 self_host_native_fp_let_smoke.py
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


def mind_fp_let_elf(lib, src: str) -> bytes:
    fn = lib.selftest_native_elf_fp_let
    fn.restype = ctypes.c_int64
    fn.argtypes = [ctypes.c_int64, ctypes.c_int64]
    buf = ctypes.create_string_buffer(src.encode(), len(src.encode()))
    es = fn(ctypes.cast(buf, ctypes.c_void_p).value, len(src.encode()))
    rd = lambda addr, o=0: ctypes.cast(addr + o, ctypes.POINTER(ctypes.c_int64))[0]
    sh = rd(es, 0)  # buf: String handle (addr/len/cap)
    return ctypes.string_at(rd(sh, 0), rd(sh, 8))


def run_elf(elf: bytes, tmp: pathlib.Path) -> int:
    p = tmp / "mind_fp_let.elf"
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
    if not hasattr(lib, "selftest_native_elf_fp_let"):
        print("FAIL  selftest_native_elf_fp_let: symbol absent (dtype-table threading not built)")
        return 1

    # (block source, expected exit = int(trunc(tail value))). Typed lets (self-host
    # subset); all operands are exactly-representable dyadic literals so the
    # integer-only decimal->IEEE-754 path is exact and the arithmetic result is exact.
    cases = [
        ("{ let x: f64 = 1.5; x + 2.5 }", 4),
        ("{ let a: f64 = 2.0; let b: f64 = 3.0; a * b }", 6),
        ("{ let y: f64 = 7.5; y - 2.5 }", 5),
        ("{ let p: f64 = 10.0; let q: f64 = 4.0; p / q }", 2),
        ("{ let s: f64 = 1.25; s + 0.5 }", 1),
    ]
    all_ok = True
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        for src, expected in cases:
            elf = mind_fp_let_elf(lib, src)
            if not (len(elf) > 120 and elf[:4] == b"\x7fELF"):
                print(f"  FAIL  {src!r}: not a runnable ELF (len={len(elf)})")
                all_ok = False
                continue
            got = run_elf(elf, tmp)
            ok = got == expected
            all_ok = all_ok and ok
            print(
                f"  {'PASS' if ok else 'FAIL'}  {src!r:>48} -> "
                f"exit={got} expected={expected} "
                f"(elf {len(elf)}B, SSE2 native, zero MLIR/LLVM)"
            )
    if all_ok:
        print(
            "ALL PASS  FLOAT dtype propagates across a `let` binding — the per-fn SSA "
            "dtype table (lcell+8) tags each let's slot and nb_expr's binop arm routes "
            "the SSE2 f64 encoders for a float let-bound operand, native-ELF end to end "
            "with zero MLIR/LLVM (RI-B1 #107 let-binding float propagation connected)"
        )
        return 0
    print("FAIL  nb_expr float let-binding propagation gate")
    return 1


if __name__ == "__main__":
    sys.exit(main())
