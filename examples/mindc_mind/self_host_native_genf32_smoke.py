#!/usr/bin/env python3
"""RI-B2-S8 STEP B (#108) — isolate the LCG f32 rounding BEFORE the dot.

`selftest_native_elf_genf32(n)` emits a runnable x86-64 ET_EXEC that draws the
first `n` LCG f32 samples (fresh seed 0xDEADBEEF — the `a`-vector prefix the
canary dot-f32-v uses) and writes each sample's 8 LE bit-bytes to stdout. Each
draw is:
    state = state*1664525 + 1013904223            (u64)
    u32   = (state >> 16) & 0xFFFFFFFF
    f32   = ((f32)u32 / 4294967296.0f) * 2.0f - 1.0f   (cvtsi2ss/divss/mulss/subss)

Oracle (Python, numpy float32 for exact single-precision rounding): recompute the
same n draws with next_f32_unit and compare sha256(stdout). This pins the
u32->f32 conversion + div/mul/sub rounding in isolation so a Step-C mismatch can
be attributed to the fold, not the generator.

Usage:
  MINDC_SO=/path/to.so python3 self_host_native_genf32_smoke.py
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

import numpy as np

_HERE = pathlib.Path(__file__).parent
_DEFAULT_SO = _HERE / "libmindc_mind.so"

N = 16
SEED = 0xDEADBEEF
_MASK64 = (1 << 64) - 1


def ref_bytes(n: int) -> bytes:
    state = SEED
    divisor = np.float32(4294967296.0)  # u32::MAX as f32 rounds to 2^32
    two = np.float32(2.0)
    one = np.float32(1.0)
    out = bytearray()
    for _ in range(n):
        state = (state * 1664525 + 1013904223) & _MASK64
        u = (state >> 16) & 0xFFFFFFFF
        f = np.float32(np.float32(u) / divisor) * two - one
        bits = int.from_bytes(np.float32(f).tobytes(), "little")
        out += struct.pack("<q", bits)
    return bytes(out)


def mind_genf32_elf(lib, n: int) -> bytes:
    fn = lib.selftest_native_elf_genf32
    fn.restype = ctypes.c_int64
    fn.argtypes = [ctypes.c_int64]
    es = fn(n)
    rd = lambda addr, o=0: ctypes.cast(addr + o, ctypes.POINTER(ctypes.c_int64))[0]
    sh = rd(es, 0)
    return ctypes.string_at(rd(sh, 0), rd(sh, 8))


def run_elf_capture(elf: bytes, tmp: pathlib.Path) -> bytes:
    p = tmp / "mind_genf32.elf"
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
    if not hasattr(lib, "selftest_native_elf_genf32"):
        print("FAIL  selftest_native_elf_genf32: symbol absent (RI-B2-S8 Step B not built)")
        return 1

    ref = ref_bytes(N)
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        elf = mind_genf32_elf(lib, N)
        if not (len(elf) > 120 and elf[:4] == b"\x7fELF"):
            print(f"  FAIL  genf32(n={N}): not a runnable ELF (len={len(elf)})")
            return 1
        out = run_elf_capture(elf, tmp)
        ok_len = len(out) == 8 * N
        got = hashlib.sha256(out).hexdigest()
        exp = hashlib.sha256(ref).hexdigest()
        ok_hash = got == exp
        print(f"  {'PASS' if ok_len else 'FAIL'}  emitted {8 * N} bytes (got {len(out)})")
        print(f"  {'PASS' if ok_hash else 'FAIL'}  sha256(stdout) == sha256(numpy f32 ref) "
              f"= {exp[:12]}...  (got {got[:12]}...)")
        if not ok_hash:
            for i in range(min(N, 8)):
                nb = int.from_bytes(out[i * 8:i * 8 + 4], "little")
                rb = int.from_bytes(ref[i * 8:i * 8 + 4], "little")
                nv = struct.unpack("<f", out[i * 8:i * 8 + 4])[0]
                rv = struct.unpack("<f", ref[i * 8:i * 8 + 4])[0]
                mark = "" if nb == rb else "  <-- MISMATCH"
                print(f"    [{i}] native {nb:#010x} ({nv})  ref {rb:#010x} ({rv}){mark}")
    if ok_len and ok_hash:
        print("ALL PASS  RI-B2-S8 Step B: LCG f32 generation is bit-exact to the numpy f32 "
              "reference (u32->f32 / divss / mulss / subss rounding pinned)")
        return 0
    print("FAIL  RI-B2-S8 Step B LCG f32 generation")
    return 1


if __name__ == "__main__":
    sys.exit(main())
