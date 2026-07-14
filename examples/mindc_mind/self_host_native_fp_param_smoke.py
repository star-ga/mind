#!/usr/bin/env python3
"""RI-D2 S-B: FLOAT fn-param dtype classification + SysV SSE-spill ABI (zero MLIR/LLVM).

The next connecting construct after float-let propagation: a FLOAT *parameter* now flows
through GENERAL nb_expr lowering, not a hand-emitted fp selftest. Two coupled halves:

  * Classifier: nb_lower_fn tags each f64/f32 param's SSA slot FLOAT in the per-fn dtype
    table (nb_tag_float_params), and nb_node_dtype_tbl resolves a param ident to its
    tagged slot dtype (resolve_param branch) — so `x + 2.5` over `x: f64` classifies
    FLOAT and routes the SSE2 f64 encoders.
  * ABI: nb_emit_params spills params per SysV AMD64 — integer/pointer params from the GP
    pool (rdi/rsi/...), FLOAT params from the SSE pool (xmm0..7), on INDEPENDENT counters.

`selftest_native_elf_fp_param(src, len)` parses a REAL `fn f(x: f64) -> f64 { x + 2.5 }`,
stages known values into the SysV arg registers (the caller side), spills them through the
REAL nb_emit_params two-pool ABI, tags the float params, lowers the body through
nb_block_stmts/nb_expr, then reads the result slot back to xmm0 and cvttsd2si ->
exit(trunc(value)), wrapped in a runnable ELF via the same nb_write_elf scaffold.

Staged arg values (baked in the harness; NOT in the source): SSE index 0 -> 3.5, SSE index
1 -> 2.0, any int param -> 42. The mixed `fn f(n: i64, x: f64)` case is the two-pool proof:
n arrives GP (rdi, slot 0), x arrives SSE (xmm0, slot 1). A single-counter (broken) ABI
would spill x from rsi (GP) and the body would read garbage -> a wrong exit.

No frozen float oracle exists (the deleted Rust native backend returned Unsupported(
ConstF64)), so byte-identity is impossible here by construction. The oracle is EXECUTION
CORRECTNESS: emit the ELF, run it, assert exit == trunc(value). A wrong register-file/pool
choice or a mis-classified param faults or returns the wrong integer, so a correct exit is
non-fakeable.

Usage:
  MINDC_SO=/path/to.so python3 self_host_native_fp_param_smoke.py
"""
import ctypes
import os
import pathlib
import stat
import subprocess
import sys
import tempfile

_HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(_HERE))
from _selfhost_so import resolve_so  # noqa: E402


def mind_fp_param_elf(lib, src: str) -> bytes:
    fn = lib.selftest_native_elf_fp_param
    fn.restype = ctypes.c_int64
    fn.argtypes = [ctypes.c_int64, ctypes.c_int64]
    buf = ctypes.create_string_buffer(src.encode(), len(src.encode()))
    es = fn(ctypes.cast(buf, ctypes.c_void_p).value, len(src.encode()))
    rd = lambda addr, o=0: ctypes.cast(addr + o, ctypes.POINTER(ctypes.c_int64))[0]
    sh = rd(es, 0)  # buf: String handle (addr/len/cap)
    return ctypes.string_at(rd(sh, 0), rd(sh, 8))


def run_elf(elf: bytes, tmp: pathlib.Path) -> int:
    p = tmp / "mind_fp_param.elf"
    p.write_bytes(elf)
    p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return subprocess.run([str(p)]).returncode


def main() -> int:
    so = resolve_so()
    so = str(so)
    if not os.path.exists(so):
        if os.environ.get("MINDC_SO"):
            print(f"FAIL  MINDC_SO set but missing: {so!r}")
            return 1
        print(f"SKIP  {so} not built")
        return 0
    lib = ctypes.CDLL(so)
    if not hasattr(lib, "selftest_native_elf_fp_param"):
        print("FAIL  selftest_native_elf_fp_param: symbol absent (S-B not built)")
        return 1

    # (fn source, expected exit = int(trunc(body value))). Staged args: x(sse0)=3.5,
    # b(sse1)=2.0, any i64 param=42. All operands are exactly-representable dyadic
    # literals so the integer-only decimal->IEEE-754 path is exact.
    cases = [
        # single f64 param through the SSE pool + a float-literal binop.
        ("fn f(x: f64) -> f64 { x + 2.5 }", 6),   # 3.5 + 2.5 = 6.0
        ("fn f(x: f64) -> f64 { x - 1.5 }", 2),   # 3.5 - 1.5 = 2.0
        # two f64 params -> xmm0/xmm1 (independent SSE indices), param*param.
        ("fn f(x: f64, b: f64) -> f64 { x * b }", 7),  # 3.5 * 2.0 = 7.0
        # MIXED int+float: n -> GP (rdi, slot 0), x -> SSE (xmm0, slot 1). Proves the
        # two-pool split — a single-counter ABI would spill x from rsi and fault/miss.
        ("fn f(n: i64, x: f64) -> f64 { x + 2.5 }", 6),  # x=3.5 -> 6.0 (n unused)
    ]
    all_ok = True
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        for src, expected in cases:
            elf = mind_fp_param_elf(lib, src)
            if not (len(elf) > 120 and elf[:4] == b"\x7fELF"):
                print(f"  FAIL  {src!r}: not a runnable ELF (len={len(elf)})")
                all_ok = False
                continue
            got = run_elf(elf, tmp)
            ok = got == expected
            all_ok = all_ok and ok
            print(
                f"  {'PASS' if ok else 'FAIL'}  {src!r:>44} -> "
                f"exit={got} expected={expected} "
                f"(elf {len(elf)}B, SSE2 native, zero MLIR/LLVM)"
            )
    if all_ok:
        print(
            "ALL PASS  FLOAT fn-param flows through general nb_expr lowering — nb_lower_fn "
            "tags each f64/f32 param slot FLOAT, nb_node_dtype_tbl's resolve_param branch "
            "classifies a param ident FLOAT, and nb_emit_params spills float params from "
            "the SSE pool (xmm) on an independent counter while int/ptr params keep the GP "
            "pool (SysV two-pool ABI), native-ELF end to end with zero MLIR/LLVM (RI-D2 S-B)"
        )
        return 0
    print("FAIL  float fn-param dtype/ABI gate")
    return 1


if __name__ == "__main__":
    sys.exit(main())
