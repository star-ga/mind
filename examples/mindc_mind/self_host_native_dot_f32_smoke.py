#!/usr/bin/env python3
"""RI-B2-S8 STEP C (#108) — native-ELF scalar STRICT-FP f32 DOT-PRODUCT.

`selftest_native_elf_dot_f32(n)` emits a runnable x86-64 ET_EXEC that:
  (1) LCG-generates a[0..n] then b[0..n] as f32 (seed 0xDEADBEEF, the SAME
      next_f32_unit the cross-substrate canary dot-f32-v uses) into two in-frame
      arrays,
  (2) computes the STRICT-FP dot with the EXACTLY-pinned fold
      (tests/cross_substrate_identity.rs::ref_dot_f32_strict):
        ve = (n/8)*8 ; acc[0..8] = 0.0f
        for i in (0,8,..,ve-8): for lane 0..8: acc[lane] += a[i+lane]*b[i+lane]
        hs = acc[0]; hs += acc[1] .. += acc[7]      (fixed left-to-right fold)
        s = hs; for j in ve..n: s += a[j]*b[j]      (scalar tail)
      mulss THEN addss, UNFUSED (no FMA),
  (3) writes the result f32's 8 LE bit-bytes (`to_bits() as i64`) to stdout.

Gate: sha256(the 8 stdout bytes) == the committed avx2 reference for dot-f32-v.
Zero MLIR/LLVM in the native path — the strict-FP fold matches the AVX2 canary
kernel bit-for-bit (dot-f32-v is an avx2-only canary; neon is deferred).

Usage:
  MINDC_SO=/path/to.so python3 self_host_native_dot_f32_smoke.py
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

# Committed cross-substrate reference: canary dot-f32-v (len 4093, seed
# 0xDEADBEEF) = canonical_hash(result.to_bits() as i64) — reference_hashes.toml
# / dot_f32_v_reproducibility_gate.
CANARY = "a132f7b970b647cd158f591d764c19ec41a8cf27c398c87758f74efb5a8a22c0"
LENGTH = 4093
SEED = 0xDEADBEEF
_MASK64 = (1 << 64) - 1


def _make_pair_f32(n: int):
    state = SEED
    divisor = np.float32(4294967296.0)
    two = np.float32(2.0)
    one = np.float32(1.0)

    def draw():
        nonlocal state
        state = (state * 1664525 + 1013904223) & _MASK64
        u = (state >> 16) & 0xFFFFFFFF
        return np.float32(np.float32(u) / divisor) * two - one

    a = np.array([draw() for _ in range(n)], dtype=np.float32)
    b = np.array([draw() for _ in range(n)], dtype=np.float32)
    return a, b


def _ref_dot_strict(a, b) -> np.float32:
    n = len(a)
    ve = (n // 8) * 8
    acc = [np.float32(0.0)] * 8
    i = 0
    while i < ve:
        for lane in range(8):
            acc[lane] = acc[lane] + a[i + lane] * b[i + lane]
        i += 8
    hs = acc[0]
    for lane in range(1, 8):
        hs = hs + acc[lane]
    s = hs
    j = ve
    while j < n:
        s = s + a[j] * b[j]
        j += 1
    return np.float32(s)


def mind_dot_f32_elf(lib, n: int) -> bytes:
    fn = lib.selftest_native_elf_dot_f32
    fn.restype = ctypes.c_int64
    fn.argtypes = [ctypes.c_int64]
    es = fn(n)
    rd = lambda addr, o=0: ctypes.cast(addr + o, ctypes.POINTER(ctypes.c_int64))[0]
    sh = rd(es, 0)
    return ctypes.string_at(rd(sh, 0), rd(sh, 8))


def run_elf_capture(elf: bytes, tmp: pathlib.Path) -> bytes:
    p = tmp / "mind_dot_f32.elf"
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
    if not hasattr(lib, "selftest_native_elf_dot_f32"):
        print("FAIL  selftest_native_elf_dot_f32: symbol absent (RI-B2-S8 Step C not built)")
        return 1

    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        elf = mind_dot_f32_elf(lib, LENGTH)
        if not (len(elf) > 120 and elf[:4] == b"\x7fELF"):
            print(f"  FAIL  dot_f32(n={LENGTH}): not a runnable ELF (len={len(elf)})")
            return 1
        out = run_elf_capture(elf, tmp)
        if len(out) != 8:
            print(f"  FAIL  expected 8 stdout bytes, got {len(out)}: {out.hex()}")
            return 1
        bits = int.from_bytes(out[:4], "little")
        val = struct.unpack("<f", out[:4])[0]
        got = hashlib.sha256(out).hexdigest()
        ok = got == CANARY
        print(f"  native result f32  = {val}  (bits {bits:#010x}, LE8 {out.hex()})")
        print(f"  native sha256      = {got}")
        print(f"  canary dot-f32-v   = {CANARY}")
        if ok:
            print("ALL PASS  native-ELF strict-FP f32 dot is BYTE-IDENTICAL to the MLIR "
                  "canary dot-f32-v — 8-lane accumulate + pinned left-to-right fold + scalar "
                  "tail, unfused mulss/addss, zero MLIR/LLVM (RI-B2-S8 #108, FLOAT tier opener)")
            return 0
        # Mismatch: report the numpy strict-FP oracle for diagnosis.
        a, b = _make_pair_f32(LENGTH)
        ref = _ref_dot_strict(a, b)
        rbits = int.from_bytes(np.float32(ref).tobytes(), "little")
        print(f"  numpy strict oracle= {float(ref)}  (bits {rbits:#010x})")
        print("FAIL  native-ELF f32 dot hash != canary dot-f32-v — fold order or a rounding "
              "detail differs; do NOT guess (report native f32/bits/hash above).")
        return 1


if __name__ == "__main__":
    sys.exit(main())
