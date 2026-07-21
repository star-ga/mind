#!/usr/bin/env python3
"""Roadmap C2 narrow-int native-ELF rung — user-reachable i8/i16/i32 two's-complement
WRAP-TO-WIDTH, lowered THROUGH the general nb_expr / nb_emit_intrinsic path.

All native arithmetic (nb_arith_rax_mem) emits with REX.W = 64-bit add/sub/imul, so a
value computed from narrow operands is held full-width in registers and DOES NOT wrap
at the declared width. This rung adds the missing reg-to-reg wrap primitives and wires
them to three user-reachable intrinsics:

  * __mind_wrap_i8(x)  -> (x as i8)  as i64   nb_movsx_rax_al   = 48 0F BE C0
  * __mind_wrap_i16(x) -> (x as i16) as i64   nb_movsx_rax_ax   = 48 0F BF C0
  * __mind_wrap_i32(x) -> (x as i32) as i64   nb_movsxd_rax_eax = 48 63 C0

Each masks the value to N low bits and re-signs bit N-1, so an overflowed narrow result
wraps two's-complement (INT_N_MAX+1 -> INT_N_MIN) instead of i64-widening. This is the
mechanical foundation of narrow-int wrap arithmetic; the follow-up rung threads a
per-SSA declared width through NbState so a narrow binop auto-emits the wrap.

No frozen byte-oracle exists for this construct (the deleted Rust src/native backend had
no narrow-wrap smoke, and rejected ConstF64/narrow codegen). The oracle is EXECUTION
CORRECTNESS on the CPU: a real user MIND `main` compiled to native ELF via
`selftest_native_elf` exercises every wrap and returns 42 IFF each behaves correctly.
The check is non-fakeable — every `e*` term is exactly 0 iff its wrap is correct, and
`main` returns 42 IFF the FULL 64-bit `err == 0` (compared before the exit-code mod-256
truncation, so a wrong lowering whose error delta happens to be a multiple of 256 — e.g.
a no-op or zero-extend leaving 2^32 — still yields err != 0 -> exit 43, not a masked 42):

  * wrap at overflow    — INT32_MAX+1, INT16_MAX+1, INT8_MAX+1 must yield the NEGATIVE
    INT_N_MIN (a non-wrapping i64 would stay positive: 2147483648 / 32768 / 128).
  * unsigned-max        — 0xFF wrapped i8 must be -1 (all-ones re-signs).
  * in-range identity   — a value that fits the width is returned unchanged (100 -> 100,
    70000 -> low16 0x1170 = 4464).
  * wrap AFTER arith    — a sum computed in 64-bit regs (2147483647 + 1) then wrapped to
    i32 yields INT32_MIN, proving the wrap acts on the register result, not a constant.

`err` sums all independently-zero-iff-correct terms; `main` returns err + 42 so a correct
build exits 42 and ANY miscompiled wrap shifts the exit code. Zero MLIR/LLVM in the path.

Usage:
  MINDC_SO=/path/to.so python3 self_host_native_narrowwrap_smoke.py
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

MARKER = 42

# Every `e*` term below is algebraically zero IFF the wrap is correct.
#   e32 = wrap_i32(2^31)   + 2^31  -> (-2^31) + 2^31 = 0   (INT32_MAX+1 -> INT32_MIN)
#   e16 = wrap_i16(2^15)   + 2^15  -> (-2^15) + 2^15 = 0   (INT16_MAX+1 -> INT16_MIN)
#   e8  = wrap_i8(2^7)     + 2^7   -> (-128)  + 128  = 0   (INT8_MAX+1  -> INT8_MIN)
#   en8 = wrap_i8(255)     + 1     -> (-1)    + 1    = 0   (0xFF -> -1)
#   ep  = wrap_i32(100)    - 100   -> 100 - 100 = 0        (in-range identity)
#   ep16= wrap_i16(70000)  - 4464  -> low16 0x1170 = 4464  (truncation, positive)
#   ea  = wrap_i32(2147483647 + 1) + 2^31 -> wrap acts on the 64-bit REGISTER sum
SRC = f"""fn main() -> i64 {{
    let e32: i64 = __mind_wrap_i32(2147483648) + 2147483648;
    let e16: i64 = __mind_wrap_i16(32768) + 32768;
    let e8: i64 = __mind_wrap_i8(128) + 128;
    let en8: i64 = __mind_wrap_i8(255) + 1;
    let ep: i64 = __mind_wrap_i32(100) - 100;
    let ep16: i64 = __mind_wrap_i16(70000) - 4464;
    let s: i64 = 2147483647 + 1;
    let ea: i64 = __mind_wrap_i32(s) + 2147483648;
    let err: i64 = e32 + e16 + e8 + en8 + ep + ep16 + ea;
    if err == 0 {{
        {MARKER}
    }} else {{
        43
    }}
}}
"""


def mind_narrowwrap_elf(lib, src: str) -> bytes:
    fn = lib.selftest_native_elf
    fn.restype = ctypes.c_int64
    fn.argtypes = [ctypes.c_int64, ctypes.c_int64]
    enc = src.encode()
    buf = ctypes.create_string_buffer(enc, len(enc))
    es = fn(ctypes.cast(buf, ctypes.c_void_p).value, len(enc))
    rd = lambda addr, o=0: ctypes.cast(addr + o, ctypes.POINTER(ctypes.c_int64))[0]
    sh = rd(es, 0)  # buf: String handle (addr/len/cap)
    return ctypes.string_at(rd(sh, 0), rd(sh, 8))


def run_elf(elf: bytes, tmp: pathlib.Path) -> int:
    p = tmp / "mind_narrowwrap.elf"
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
    if not hasattr(lib, "selftest_native_elf"):
        print("FAIL  selftest_native_elf: symbol absent")
        return 1

    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        elf = mind_narrowwrap_elf(lib, SRC)
        if not (len(elf) > 120 and elf[:4] == b"\x7fELF"):
            print(f"  FAIL  narrowwrap: not a runnable ELF (len={len(elf)})")
            return 1
        got = run_elf(elf, tmp)
        ok = got == MARKER
        print(
            f"  {'PASS' if ok else 'FAIL'}  narrow-int wrap exit={got} "
            f"expected={MARKER}  (elf {len(elf)}B, i8/i16/i32 two's-complement "
            f"wrap-to-width via nb_emit_intrinsic movsx, zero MLIR/LLVM)"
        )
        if ok:
            print(
                "ALL PASS  user-reachable __mind_wrap_{i8,i16,i32} lower to native "
                "x86-64 through the general nb_expr / nb_emit_intrinsic path — a narrow "
                "value overflowing its width wraps two's-complement (INT_N_MAX+1 -> "
                "INT_N_MIN, 0xFF i8 -> -1) instead of i64-widening, and an in-range "
                "value round-trips unchanged (roadmap C2 narrow-int wrap rung)"
            )
            return 0
        print(
            "FAIL  narrow-int wrap rung: exit != 42 means a narrow overflow i64-widened "
            "instead of wrapping, a wrap re-signed the wrong bit, or an in-range value "
            "was corrupted — do NOT guess (report the native exit above)."
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
