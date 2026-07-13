#!/usr/bin/env python3
"""RI-B2 L1-Q16 rung (#108) — native-ELF Q16.16 L1 distance.

`selftest_native_elf_dot_l1_q16(n)` emits a runnable x86-64 ET_EXEC that
generates a[0..n]/b[0..n] via the EXACT Q16.16 LCG (seed 0xDEADBEEF, shared with
the intdot_q16 rung), computes acc = Sum |sext(a[i]) - sext(b[i])| in a full i64
accumulator, narrows ONCE to i32 (`acc as i32`), and writes the 8 LE bytes of the
result to stdout. ABS is branchless integer: d=a-b ; t=d SAR 63 ; abs=(d^t)-t —
reusing the existing sub/sar/xor encoders, no new opcode.

Gate: sha256(the 8 stdout bytes) == the committed canary dot-l1-q16.
Zero MLIR/LLVM in the native path — byte-identical to the MLIR canary.

Usage:
  MINDC_SO=/path/to.so python3 self_host_native_dot_l1_q16_smoke.py
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

# Committed cross-substrate reference: canary dot-l1-q16 =
# canonical_hash((Sum|a-b| as i32) as i64) — dot_l1_q16_reproducibility_gate.
CANARY = "ce7e2a80515e123f5d4fbb77d841f0d6c56fcbc690bba2e2ff81e45765843b34"

N = 65536
SEED = 0xDEADBEEF
_MASK64 = (1 << 64) - 1


def _ref_dot_l1_q16(n: int, seed: int) -> int:
    """Byte-for-byte ref_dot_l1_q16_scalar: LCG -> Q16 i32 a/b, Sum|a-b|, as i32."""
    st = seed & _MASK64

    def nxt_q16() -> int:
        nonlocal st
        st = (st * 1664525 + 1013904223) & _MASK64
        u = (st >> 16) & 0xFFFFFFFF
        v = u - (1 << 32) if u >= (1 << 31) else u  # u32 as i32
        return v >> 12  # arithmetic

    a = [nxt_q16() for _ in range(n)]
    b = [nxt_q16() for _ in range(n)]
    acc = 0
    for i in range(n):
        d = a[i] - b[i]
        acc += -d if d < 0 else d
    r = acc & 0xFFFFFFFF
    return r - (1 << 32) if r >= (1 << 31) else r


def mind_dot_l1_q16_elf(lib) -> bytes:
    fn = lib.selftest_native_elf_dot_l1_q16
    fn.restype = ctypes.c_int64
    fn.argtypes = [ctypes.c_int64]
    es = fn(N)
    rd = lambda addr, o=0: ctypes.cast(addr + o, ctypes.POINTER(ctypes.c_int64))[0]
    sh = rd(es, 0)
    return ctypes.string_at(rd(sh, 0), rd(sh, 8))


def run_elf_capture(elf: bytes, tmp: pathlib.Path) -> bytes:
    p = tmp / "mind_dot_l1_q16.elf"
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
    if not hasattr(lib, "selftest_native_elf_dot_l1_q16"):
        print("FAIL  selftest_native_elf_dot_l1_q16: symbol absent (L1-Q16 rung not built)")
        return 1

    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        elf = mind_dot_l1_q16_elf(lib)
        if not (len(elf) > 120 and elf[:4] == b"\x7fELF"):
            print(f"  FAIL  dot_l1_q16: not a runnable ELF (len={len(elf)})")
            return 1
        out = run_elf_capture(elf, tmp)
        if len(out) != 8:
            print(f"  FAIL  expected 8 stdout bytes, got {len(out)}: {out.hex()}")
            return 1
        val = struct.unpack("<q", out)[0]
        got = hashlib.sha256(out).hexdigest()
        ok = got == CANARY
        print(f"  native result i64  = {val}  (LE8 {out.hex()})")
        print(f"  native sha256      = {got}")
        print(f"  canary dot-l1-q16  = {CANARY}")
        if ok:
            print("ALL PASS  native-ELF Q16.16 L1 distance Sum|a-b| is BYTE-IDENTICAL "
                  "to the MLIR canary dot-l1-q16 — shared LCG generator, branchless "
                  "integer abs (sub/sar-63/xor), narrow-once-to-i32, zero MLIR/LLVM "
                  "(RI-B2 #108, closes the scalar L1 tier)")
            return 0
        ref = _ref_dot_l1_q16(N, SEED)
        print(f"  python oracle i64  = {ref}")
        print("FAIL  native-ELF L1-Q16 hash != canary dot-l1-q16 — the reduction or abs "
              "idiom differs; do NOT guess (report native value/bits/hash above).")
        return 1


if __name__ == "__main__":
    sys.exit(main())
