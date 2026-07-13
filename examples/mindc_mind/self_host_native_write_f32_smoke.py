#!/usr/bin/env python3
"""RI-B2-S8 STEP A (#108) — de-risk the raw-f32-bytes harness on a KNOWN f32.

`selftest_native_elf_write_f32(bits)` emits a runnable x86-64 ET_EXEC that routes
a KNOWN f32 (its IEEE-754 bit pattern) through xmm0 via movss-load, extracts the
bits with `movd eax, xmm0` (which zeroes rax[63:32]), writes the 8 LITTLE-ENDIAN
bytes of rax to stdout, and exit(0). Proves the movss-load + movd + f32->bits->
stdout path the strict-FP dot gate (Step C) relies on.

Oracle (Python): sha256(the 8 captured bytes) == sha256(struct.pack('<q', bits))
— the `to_bits() as i64` zero-extended encoding the canary hashes.

Usage:
  MINDC_SO=/path/to.so python3 self_host_native_write_f32_smoke.py
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

BITS = 0x3FC00000  # 1.5f


def mind_write_f32_elf(lib, bits: int) -> bytes:
    fn = lib.selftest_native_elf_write_f32
    fn.restype = ctypes.c_int64
    fn.argtypes = [ctypes.c_int64]
    es = fn(bits)
    rd = lambda addr, o=0: ctypes.cast(addr + o, ctypes.POINTER(ctypes.c_int64))[0]
    sh = rd(es, 0)
    return ctypes.string_at(rd(sh, 0), rd(sh, 8))


def run_elf_capture(elf: bytes, tmp: pathlib.Path) -> bytes:
    p = tmp / "mind_write_f32.elf"
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
    if not hasattr(lib, "selftest_native_elf_write_f32"):
        print("FAIL  selftest_native_elf_write_f32: symbol absent (RI-B2-S8 Step A not built)")
        return 1

    expect = hashlib.sha256(struct.pack("<q", BITS)).hexdigest()
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        elf = mind_write_f32_elf(lib, BITS)
        if not (len(elf) > 120 and elf[:4] == b"\x7fELF"):
            print(f"  FAIL  write_f32({BITS:#010x}): not a runnable ELF (len={len(elf)})")
            return 1
        out = run_elf_capture(elf, tmp)
        ok_len = len(out) == 8
        ok_bytes = out == struct.pack("<q", BITS)
        got = hashlib.sha256(out).hexdigest()
        ok_hash = got == expect
        print(f"  {'PASS' if ok_len else 'FAIL'}  emitted exactly 8 bytes (got {len(out)})")
        print(f"  {'PASS' if ok_bytes else 'FAIL'}  bytes == LE i64 of f32 bits {BITS:#010x} "
              f"(1.5f): {out.hex()} vs {struct.pack('<q', BITS).hex()}")
        print(f"  {'PASS' if ok_hash else 'FAIL'}  sha256(stdout) == "
              f"sha256(struct.pack('<q', {BITS:#010x})) = {expect[:12]}...")
    if ok_len and ok_bytes and ok_hash:
        print("ALL PASS  RI-B2-S8 Step A: movss-load + movd + f32->bits->stdout proven on 1.5f")
        return 0
    print("FAIL  RI-B2-S8 Step A raw-f32-bytes harness")
    return 1


if __name__ == "__main__":
    sys.exit(main())
