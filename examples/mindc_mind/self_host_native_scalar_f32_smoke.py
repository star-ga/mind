#!/usr/bin/env python3
"""Phase C1-remainder f32 rung — native-ELF scalar SINGLE-precision chain.

`selftest_native_elf_scalar_f32(a_bits, b_bits, c_bits, d_bits)` (each arg carries a
32-bit f32 pattern zero-extended into i64) emits a runnable x86-64 ET_EXEC that
computes the strict-source-precedence, unfused chain `a + b - c * d / a` entirely in
SINGLE precision via the F3-prefix scalar-single encoders (movss load/store,
addss/subss/mulss/divss), each intermediate rounded to f32. The single-rounded
result is round-tripped f32->f64->f32 (cvtss2sd ; cvtsd2ss — value-preserving, so it
exercises the new narrowing cvtsd2ss without changing the value) and truncated to a
signed i64 (cvttss2si). Output is 12 bytes:
    [0..8]  = (i64) trunc(result)   (LE, cvttss2si rax,xmm0)
    [8..12] = result.to_bits()      (LE, movd eax,xmm0)

CPU-as-oracle only: the deleted Rust native backend rejected ConstF64, so there is NO
byte-identity oracle for scalar float native emission (unlike the int/control-flow
path). This gate is execution-correctness against numpy-style single-precision
reference arithmetic (struct '<f' round-trips every intermediate through f32).

Usage:
  MINDC_SO=/path/to.so python3 self_host_native_scalar_f32_smoke.py
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

# Single-precision inputs (mirror the f64 rung's SCALAR_F64_INPUTS).
INPUTS = (1.5, 2.25, 0.5, 3.125)


def _f32(x: float) -> float:
    """Round a Python float to the nearest IEEE-754 single value."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def _f32_bits(x: float) -> int:
    """f32 -> i64 argument carrying the 32-bit pattern zero-extended."""
    return int.from_bytes(struct.pack("<f", x), "little")


def _ref_scalar_f32_chain(a: float, b: float, c: float, d: float):
    """(a + b) - ((c * d) / a) — fixed source precedence, unfused, every
    intermediate single-rounded. Returns (result_f32, trunc_i64)."""
    a, b, c, d = _f32(a), _f32(b), _f32(c), _f32(d)
    t1 = _f32(a + b)
    t3 = _f32(_f32(c * d) / a)
    t4 = _f32(t1 - t3)
    trunc = int(t4)  # C truncation toward zero
    return t4, trunc


def mind_scalar_f32_elf(lib) -> bytes:
    fn = lib.selftest_native_elf_scalar_f32
    fn.restype = ctypes.c_int64
    fn.argtypes = [ctypes.c_int64] * 4
    a, b, c, d = INPUTS
    es = fn(_f32_bits(a), _f32_bits(b), _f32_bits(c), _f32_bits(d))
    rd = lambda addr, o=0: ctypes.cast(addr + o, ctypes.POINTER(ctypes.c_int64))[0]
    sh = rd(es, 0)
    return ctypes.string_at(rd(sh, 0), rd(sh, 8))


def run_elf_capture(elf: bytes, tmp: pathlib.Path) -> bytes:
    p = tmp / "mind_scalar_f32.elf"
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
    if not hasattr(lib, "selftest_native_elf_scalar_f32"):
        print("FAIL  selftest_native_elf_scalar_f32: symbol absent (C1-remainder f32 rung not built)")
        return 1

    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        elf = mind_scalar_f32_elf(lib)
        if not (len(elf) > 120 and elf[:4] == b"\x7fELF"):
            print(f"  FAIL  scalar_f32: not a runnable ELF (len={len(elf)})")
            return 1
        out = run_elf_capture(elf, tmp)
        if len(out) != 12:
            print(f"  FAIL  expected 12 stdout bytes, got {len(out)}: {out.hex()}")
            return 1
        got_trunc = struct.unpack("<q", out[0:8])[0]
        got_f32 = struct.unpack("<f", out[8:12])[0]
        got_bits = int.from_bytes(out[8:12], "little")

        ref_f32, ref_trunc = _ref_scalar_f32_chain(*INPUTS)
        ref_bits = int.from_bytes(struct.pack("<f", ref_f32), "little")

        print(f"  native result f32  = {got_f32}  (bits {got_bits:#010x}, LE4 {out[8:12].hex()})")
        print(f"  native trunc i64   = {got_trunc}  (LE8 {out[0:8].hex()})")
        print(f"  python oracle f32  = {ref_f32}  (bits {ref_bits:#010x})")
        print(f"  python oracle trunc= {ref_trunc}")

        if got_bits == ref_bits and got_trunc == ref_trunc:
            print("ALL PASS  native-ELF single-precision chain `a + b - c * d / a` matches "
                  "the numpy-style f32 reference — unfused SSE addss/subss/mulss/divss with "
                  "single-rounded intermediates, cvtss2sd/cvtsd2ss round-trip and cvttss2si "
                  "truncation, movd lift, zero MLIR/LLVM (Phase C1-remainder f32 scalar tier)")
            return 0
        print("FAIL  native-ELF f32 chain != single-precision oracle — an op rounded to "
              "double or a conversion/opcode byte is wrong; do NOT guess (report the "
              "native f32/bits/trunc above).")
        return 1


if __name__ == "__main__":
    sys.exit(main())
