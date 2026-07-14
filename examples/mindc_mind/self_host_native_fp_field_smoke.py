#!/usr/bin/env python3
"""RI-D2 S-D FLOAT struct-FIELD READ dtype through native-ELF general lowering (zero MLIR).

A struct field declared f64/f32 READS as a FLOAT-tagged value: nb_expr's ast_field arm
classifies the field via nb_field_dtype (value-node based, off the struct-literal binding
the native backend tracks), and a FLOAT field loads through the SSE register file
(nb_field_read_fp: mov rax,[rbp+addr] ; movsd xmm0,[rax] ; movsd [rbp+dst],xmm0) and tags
its result slot FLOAT. nb_node_dtype_tbl's ast_field arm then sees `p.x` as FLOAT, so an
immediate binop `p.x + 1.0` routes the SSE2 f64 encoders. An INT/pointer field takes the
UNCHANGED integer load, so main.mind's float-free structs are byte-identical.

`selftest_native_elf_fp_field(src, len)` lexes + parses a REAL statement block with
parse_block, lowers the WHOLE block THROUGH nb_block_stmts / nb_stmt / nb_expr, then loads
the tail expr's result slot back to xmm0 and cvttsd2si -> exit(trunc(value)), wrapped in a
real runnable ELF via the same nb_write_elf scaffold the integer path uses.

No frozen float oracle exists (the deleted Rust native backend returned
Unsupported(ConstF64)), so byte-identity is impossible here by construction. The oracle is
EXECUTION CORRECTNESS: emit the ELF, run it, assert exit == trunc(value). WITHOUT the S-D
field-dtype classification `p.x` would classify INT, take the GP-integer field load, and
the trunc of the mis-interpreted bits would fault or return the wrong integer -- so a
correct exit is a non-fakeable proof the FLOAT dtype crossed the struct field. A MIXED
struct (int + float fields) reading its float field proves the per-field dtype + offset.

Usage:
  MINDC_SO=/path/to.so python3 self_host_native_fp_field_smoke.py
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


def mind_fp_field_elf(lib, src: str) -> bytes:
    fn = lib.selftest_native_elf_fp_field
    fn.restype = ctypes.c_int64
    fn.argtypes = [ctypes.c_int64, ctypes.c_int64]
    buf = ctypes.create_string_buffer(src.encode(), len(src.encode()))
    es = fn(ctypes.cast(buf, ctypes.c_void_p).value, len(src.encode()))
    rd = lambda addr, o=0: ctypes.cast(addr + o, ctypes.POINTER(ctypes.c_int64))[0]
    sh = rd(es, 0)  # buf: String handle (addr/len/cap)
    return ctypes.string_at(rd(sh, 0), rd(sh, 8))


def run_elf(elf: bytes, tmp: pathlib.Path) -> int:
    p = tmp / "mind_fp_field.elf"
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
    if not hasattr(lib, "selftest_native_elf_fp_field"):
        print("FAIL  selftest_native_elf_fp_field: symbol absent (S-D field-dtype not built)")
        return 1

    # (block source, expected exit = int(trunc(tail value))). A struct with an f64 field
    # constructed via a struct literal, bound to a typed let, then its FLOAT field read
    # into a float expression. The MIXED case (int + float fields) reads the float field
    # at a NON-zero offset -> proves per-field dtype + offset resolution together. All
    # literals are exactly-representable dyadics so the decimal->IEEE-754 path is exact.
    cases = [
        ("struct P { x: f64 } { let p: P = P { x: 3.5 }; p.x + 1.0 }", 4),
        ("struct P { x: f64 } { let p: P = P { x: 6.5 }; p.x - 2.0 }", 4),
        ("struct Q { a: i64, b: f64 } { let q: Q = Q { a: 5, b: 2.5 }; q.b + 1.0 }", 3),
        ("struct Q { a: i64, b: f64 } { let q: Q = Q { a: 9, b: 4.0 }; q.b * 2.0 }", 8),
    ]
    all_ok = True
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        for src, expected in cases:
            elf = mind_fp_field_elf(lib, src)
            if not (len(elf) > 120 and elf[:4] == b"\x7fELF"):
                print(f"  FAIL  {src!r}: not a runnable ELF (len={len(elf)})")
                all_ok = False
                continue
            got = run_elf(elf, tmp)
            ok = got == expected
            all_ok = all_ok and ok
            print(
                f"  {'PASS' if ok else 'FAIL'}  {src!r:>60} -> "
                f"exit={got} expected={expected} "
                f"(elf {len(elf)}B, SSE2 native, zero MLIR/LLVM)"
            )
    if all_ok:
        print(
            "ALL PASS  FLOAT struct-FIELD READ dtype flows through general nb_expr lowering "
            "— nb_field_dtype classifies an f64/f32 field FLOAT off the struct-literal "
            "binding, nb_field_read_fp loads it via movsd and tags the result slot FLOAT, "
            "and nb_node_dtype_tbl's ast_field arm lets an immediate binop over the field "
            "route the SSE2 f64 encoders; a MIXED struct proves per-field dtype + offset, "
            "native-ELF end to end with zero MLIR/LLVM (RI-D2 S-D)"
        )
        return 0
    print("FAIL  nb_expr float struct-field READ dtype gate")
    return 1


if __name__ == "__main__":
    sys.exit(main())
