#!/usr/bin/env python3
"""RI-B2 scalar-cast-conv rung (#108) — native-ELF int<->float `as`-cast chain.

`selftest_native_elf_scalar_cast_conv(inr,povf,novf,nan,pinf,ninf,n)` emits a
runnable x86-64 ET_EXEC that reproduces ref_scalar_cast_conv: six SATURATING
float->i64 edges (9.7->9, 1e30->INT64_MAX, -1e30->INT64_MIN, NaN->0, +inf->
INT64_MAX, -inf->INT64_MIN), int->f64->i64 (b0), int->f32->i64 (b1, f32 rounding
16777219->16777220), bool->f64->i64 (b2=(inr<povf)->1), folded in fixed source
order via acc = acc*1000003 + term (wrapping i64), then the 8 LE bytes written.

Saturating float->i64 is branchless SSE2 (cvttsd2si sentinel + cmplesd/cmpunordsd
masks), so the RESULT is byte-identical to the MLIR canary scalar-cast-conv. Zero
MLIR/LLVM in the native path.

Gate: sha256(the 8 stdout bytes) == the committed canary scalar-cast-conv.

Usage:
  MINDC_SO=/path/to.so python3 self_host_native_cast_conv_smoke.py
"""
import ctypes
import hashlib
import os
import pathlib
import stat
import struct
import subprocess
import sys
import tempfile

_HERE = pathlib.Path(__file__).parent
_DEFAULT_SO = _HERE / "libmindc_mind.so"

# Committed cross-substrate reference: canary scalar-cast-conv (avx2==neon).
CANARY = "a38aaa5196baad698f60edc9d2ffc44aac43540ae74aa3bcaf2687fd37a0b8c2"

# SCALAR_CAST_INPUTS = (9.7, 1e30, -1e30, NaN, +inf, -inf, 16777219).
INPUTS = (9.7, 1e30, -1e30, float("nan"), float("inf"), float("-inf"), 16_777_219)
_MASK64 = (1 << 64) - 1


def _f64_bits(x: float) -> int:
    u = int.from_bytes(struct.pack("<d", x), "little")
    return u - (1 << 64) if u >= (1 << 63) else u


def _sat_f2i64(x: float) -> int:
    import math
    if math.isnan(x):
        return 0
    if x >= 9223372036854775808.0:
        return 9223372036854775807
    if x < -9223372036854775808.0:
        return -9223372036854775808
    return int(x)  # trunc toward zero


def _ref_scalar_cast_conv() -> int:
    import numpy as np
    inr, povf, novf, nan, pinf, ninf, n = INPUTS
    e = [_sat_f2i64(v) for v in (inr, povf, novf, nan, pinf, ninf)]
    b0 = _sat_f2i64(float(n))
    b1 = _sat_f2i64(float(np.float32(n)))
    b2 = 1 if inr < povf else 0
    k = 1000003
    acc = e[0]
    for t in [e[1], e[2], e[3], e[4], e[5], b0, b1, b2]:
        acc = (acc * k + t) & _MASK64
        if acc >= (1 << 63):
            acc -= 1 << 64
    return acc


def mind_cast_conv_elf(lib) -> bytes:
    fn = lib.selftest_native_elf_scalar_cast_conv
    fn.restype = ctypes.c_int64
    fn.argtypes = [ctypes.c_int64] * 7
    inr, povf, novf, nan, pinf, ninf, n = INPUTS
    es = fn(_f64_bits(inr), _f64_bits(povf), _f64_bits(novf), _f64_bits(nan),
            _f64_bits(pinf), _f64_bits(ninf), n)
    rd = lambda addr, o=0: ctypes.cast(addr + o, ctypes.POINTER(ctypes.c_int64))[0]
    sh = rd(es, 0)
    return ctypes.string_at(rd(sh, 0), rd(sh, 8))


def run_elf_capture(elf: bytes, tmp: pathlib.Path) -> bytes:
    p = tmp / "mind_cast_conv.elf"
    p.write_bytes(elf)
    p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return subprocess.run([str(p)], capture_output=True).stdout


def main() -> int:
    so = os.environ.get("MINDC_SO", str(_DEFAULT_SO))
    if not os.path.exists(so):
        if os.environ.get("MINDC_SO"):
            print(f"FAIL  MINDC_SO set but missing: {so!r}")
            return 1
        print(f"SKIP  {so} not built")
        return 0
    lib = ctypes.CDLL(so)
    if not hasattr(lib, "selftest_native_elf_scalar_cast_conv"):
        print("FAIL  selftest_native_elf_scalar_cast_conv: symbol absent (cast-conv rung not built)")
        return 1

    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        elf = mind_cast_conv_elf(lib)
        if not (len(elf) > 120 and elf[:4] == b"\x7fELF"):
            print(f"  FAIL  cast_conv: not a runnable ELF (len={len(elf)})")
            return 1
        out = run_elf_capture(elf, tmp)
        if len(out) != 8:
            print(f"  FAIL  expected 8 stdout bytes, got {len(out)}: {out.hex()}")
            return 1
        val = struct.unpack("<q", out)[0]
        got = hashlib.sha256(out).hexdigest()
        ok = got == CANARY
        print(f"  native result i64     = {val}  (LE8 {out.hex()})")
        print(f"  native sha256         = {got}")
        print(f"  canary scalar-cast    = {CANARY}")
        if ok:
            print("ALL PASS  native-ELF int<->float cast chain is BYTE-IDENTICAL to the "
                  "MLIR canary scalar-cast-conv — branchless SSE2 saturating float->i64 "
                  "(cvttsd2si + cmplesd/cmpunordsd masks), cvtsi2sd/cvtsi2ss int->float, "
                  "f32 rounding, zero MLIR/LLVM (RI-B2 #108, closes the scalar tier)")
            return 0
        ref = _ref_scalar_cast_conv()
        print(f"  python oracle i64     = {ref}")
        print("FAIL  native-ELF cast-conv hash != canary scalar-cast-conv — a saturation "
              "edge or the f32 rounding differs; do NOT guess (report native value/hash above).")
        return 1


if __name__ == "__main__":
    sys.exit(main())
