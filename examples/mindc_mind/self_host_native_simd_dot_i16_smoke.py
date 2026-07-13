#!/usr/bin/env python3
"""RI-B2-S12 (#108) — native-ELF PACKED-int16 SIMD DOT-PRODUCT, byte-identity rung.

Proves the pure-MIND native-ELF backend emits VECTOR packed-int16 SIMD (SSE2,
128-bit) via the pmaddwd idiom whose i64 result is byte-identical to the SAME MLIR
canary dot-i16-4096 the scalar S2 rung hit. Deterministic integer SIMD, zero LLVM.

`selftest_native_elf_simd_dot_i16(n)` emits a runnable x86-64 ET_EXEC that:
  (1) LCG-generates a[0..n] then b[0..n] as PACKED int16 (stride 2) via the EXACT
      cross-substrate generator (seed 0xDEADBEEF; state = state*1664525 +
      1013904223; element = (state >> 32) low 16 bits as i16),
  (2) computes the dot with an 8-wide packed pmaddwd accumulate (4 i32 lanes in
      xmm7 via paddd),
  (3) horizontally reduces the 4 lanes (pshufd + paddd), narrows ONCE to i32
      (movsxd — the MLIR canary's `acc as i32`),
  (4) writes the result's 8 LITTLE-ENDIAN bytes to stdout via write(2).

The pmaddwd-i32-pairwise model equals the scalar pure-i64 sum-then-narrow because
every step is linear mod 2^32 and the final `as i32` narrow absorbs the wrapping.

Gate: emit the ELF, run it, capture the 8 stdout bytes, sha256 == the committed
cross-substrate reference for `dot-i16-4096`.

Usage:
  MINDC_SO=/path/to.so python3 self_host_native_simd_dot_i16_smoke.py
"""
import ctypes
import hashlib
import os
import pathlib
import stat
import subprocess
import sys
import tempfile

_HERE = pathlib.Path(__file__).parent
_DEFAULT_SO = _HERE / "libmindc_mind.so"

# Committed cross-substrate reference: canary `dot-i16-4096` =
# canonical_hash(result) = sha256 of the 8 LE bytes of the i64 dot result
# (tests/cross_substrate_identity.rs dot_i16_reproducibility_gate).
CANARY = "af0fc3cf1b510f8f7306a5d7250ae25a52b35281a7cefff2a0ac94b0cd80a127"
LENGTH = 4096


def mind_simd_dot_i16_elf(lib, n: int) -> bytes:
    fn = lib.selftest_native_elf_simd_dot_i16
    fn.restype = ctypes.c_int64
    fn.argtypes = [ctypes.c_int64]
    es = fn(n)
    rd = lambda addr, o=0: ctypes.cast(addr + o, ctypes.POINTER(ctypes.c_int64))[0]
    sh = rd(es, 0)  # buf: String handle (addr/len/cap)
    return ctypes.string_at(rd(sh, 0), rd(sh, 8))


def run_elf_capture(elf: bytes, tmp: pathlib.Path) -> bytes:
    p = tmp / "mind_simd_dot_i16.elf"
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
    if not hasattr(lib, "selftest_native_elf_simd_dot_i16"):
        print("FAIL  selftest_native_elf_simd_dot_i16: symbol absent (RI-B2-S12 not built)")
        return 1

    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        elf = mind_simd_dot_i16_elf(lib, LENGTH)
        if not (len(elf) > 120 and elf[:4] == b"\x7fELF"):
            print(f"  FAIL  simd_dot_i16(n={LENGTH}): not a runnable ELF (len={len(elf)})")
            return 1
        out = run_elf_capture(elf, tmp)
        if len(out) != 8:
            print(f"  FAIL  expected 8 stdout bytes, got {len(out)}: {out.hex()}")
            return 1
        val = int.from_bytes(out, "little", signed=True)
        got = hashlib.sha256(out).hexdigest()
        ok = got == CANARY
        print(f"  native result i64  = {val}  (LE bytes {out.hex()})")
        print(f"  native sha256      = {got}")
        print(f"  canary dot-i16-4096= {CANARY}")
        if ok:
            print("ALL PASS  native-ELF PACKED-int16 SIMD (SSE2, 128-bit) dot is "
                  "BYTE-IDENTICAL to the MLIR canary dot-i16-4096 — 8-wide pmaddwd "
                  "accumulate (4 i32 lanes via paddd), horizontal reduce, narrow-once-"
                  "to-i32, zero MLIR/LLVM. The pmaddwd-i32-pairwise-wrap equals the "
                  "scalar pure-i64 sum-then-narrow (linear mod 2^32). Native-ELF SIMD "
                  "== scalar == MLIR (RI-B2-S12 #108, int16 SIMD rung).")
            return 0
        print("FAIL  native-ELF SIMD int16 dot hash != canary dot-i16-4096 — the "
              "pmaddwd/paddd reduction diverged; report the native value + hash above "
              "and STOP (do NOT guess).")
        return 1


if __name__ == "__main__":
    sys.exit(main())
