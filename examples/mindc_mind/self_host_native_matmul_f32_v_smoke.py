#!/usr/bin/env python3
"""RI-B2-S9 (#108) — native-ELF scalar STRICT-FP f32 GEMV (matmul-f32-v).

`selftest_native_elf_matmul_f32_v(m, n)` emits a runnable x86-64 ET_EXEC that:
  (1) LCG-generates W[0..m*n] (row-major) then x[0..n] as f32 (seed 0xDEADBEEF,
      the SAME make_matvec_f32 the cross-substrate canary matmul-f32-v uses —
      W generated BEFORE x),
  (2) for each row r<m computes y[r] = the STRICT-FP dot of W[r,:] . x with the
      EXACTLY-pinned S8 fold
      (tests/cross_substrate_identity.rs::ref_dot_f32_strict, applied per row):
        ve = (cols/8)*8 ; acc[0..8] = 0.0f
        for i in (0,8,..,ve-8): for lane 0..8: acc[lane] += W[r,i+lane]*x[i+lane]
        hs = acc[0]; hs += acc[1] .. += acc[7]      (fixed left-to-right fold)
      cols=64 so ve=64 — NO scalar tail. mulss THEN addss, UNFUSED (no FMA),
  (3) writes each y[r]'s 4 LE IEEE-754 bit-bytes (`to_bits().to_le_bytes()`) to
      stdout in row-major order (m rows -> 4*m bytes).

Gate: sha256(the 4*m stdout bytes) == the committed avx2 reference for
matmul-f32-v (each f32 result's bit pattern, LE, concatenated). Zero MLIR/LLVM
in the native path — the per-row strict-FP fold matches the AVX2 canary kernel
bit-for-bit (matmul-f32-v is an avx2-only canary; neon is deferred).

Usage:
  MINDC_SO=/path/to.so python3 self_host_native_matmul_f32_v_smoke.py
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

# Committed cross-substrate reference: canary matmul-f32-v (64x64, seed
# 0xDEADBEEF) = sha256(concat of each y[r].to_bits().to_le_bytes()).
# tests/cross_substrate_identity/matmul-f32-v-64x64/reference_hashes.toml (avx2).
CANARY = "ec5adb991372fcfc16b964ba566f05fb44701fcf8bbde2a5453fed294e1d0175"
ROWS = 64
COLS = 64
SEED = 0xDEADBEEF
_MASK64 = (1 << 64) - 1


def _make_matvec_f32(rows: int, cols: int):
    state = SEED
    divisor = np.float32(4294967296.0)
    two = np.float32(2.0)
    one = np.float32(1.0)

    def draw():
        nonlocal state
        state = (state * 1664525 + 1013904223) & _MASK64
        u = (state >> 16) & 0xFFFFFFFF
        return np.float32(np.float32(u) / divisor) * two - one

    w = np.array([draw() for _ in range(rows * cols)], dtype=np.float32)
    x = np.array([draw() for _ in range(cols)], dtype=np.float32)
    return w, x


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


def mind_matmul_f32_elf(lib, m: int, n: int) -> bytes:
    fn = lib.selftest_native_elf_matmul_f32_v
    fn.restype = ctypes.c_int64
    fn.argtypes = [ctypes.c_int64, ctypes.c_int64]
    es = fn(m, n)
    rd = lambda addr, o=0: ctypes.cast(addr + o, ctypes.POINTER(ctypes.c_int64))[0]
    sh = rd(es, 0)
    return ctypes.string_at(rd(sh, 0), rd(sh, 8))


def run_elf_capture(elf: bytes, tmp: pathlib.Path) -> bytes:
    p = tmp / "mind_matmul_f32.elf"
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
    if not hasattr(lib, "selftest_native_elf_matmul_f32_v"):
        print("FAIL  selftest_native_elf_matmul_f32_v: symbol absent (RI-B2-S9 not built)")
        return 1

    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        elf = mind_matmul_f32_elf(lib, ROWS, COLS)
        if not (len(elf) > 120 and elf[:4] == b"\x7fELF"):
            print(f"  FAIL  matmul_f32_v({ROWS}x{COLS}): not a runnable ELF (len={len(elf)})")
            return 1
        out = run_elf_capture(elf, tmp)
        if len(out) != 4 * ROWS:
            print(f"  FAIL  expected {4 * ROWS} stdout bytes, got {len(out)}: {out.hex()}")
            return 1
        got = hashlib.sha256(out).hexdigest()
        ok = got == CANARY
        y0 = struct.unpack("<f", out[0:4])[0]
        y0_bits = int.from_bytes(out[0:4], "little")
        print(f"  native y[0] f32    = {y0}  (bits {y0_bits:#010x})")
        print(f"  native sha256      = {got}")
        print(f"  canary matmul-f32-v= {CANARY}")
        if ok:
            print("ALL PASS  native-ELF strict-FP f32 GEMV is BYTE-IDENTICAL to the MLIR "
                  "canary matmul-f32-v — outer row loop over the S8 8-lane accumulate + "
                  "pinned left-to-right fold, unfused mulss/addss, zero MLIR/LLVM "
                  "(RI-B2-S9 #108, f32 matrix-vector rung)")
            return 0
        # Mismatch: report the numpy per-row strict-FP oracle for diagnosis.
        w, x = _make_matvec_f32(ROWS, COLS)
        oracle = [_ref_dot_strict(w[r * COLS:(r + 1) * COLS], x) for r in range(ROWS)]
        obits = b"".join(np.float32(v).tobytes() for v in oracle)
        ohash = hashlib.sha256(obits).hexdigest()
        print(f"  numpy oracle y[0]  = {float(oracle[0])}  (bits {int.from_bytes(np.float32(oracle[0]).tobytes(), 'little'):#010x})")
        print(f"  numpy oracle sha256= {ohash}")
        # Show the first few rows where native diverges from the oracle.
        for r in range(ROWS):
            nb = out[r * 4:(r + 1) * 4]
            ob = np.float32(oracle[r]).tobytes()
            if nb != ob:
                print(f"  DIVERGE row {r}: native bits {int.from_bytes(nb, 'little'):#010x} "
                      f"(={struct.unpack('<f', nb)[0]}) vs oracle bits "
                      f"{int.from_bytes(ob, 'little'):#010x} (={float(oracle[r])})")
                if r >= 3:
                    break
        print("FAIL  native-ELF f32 GEMV hash != canary matmul-f32-v — fold order or "
              "generation order differs; do NOT guess (report native/oracle above).")
        return 1


if __name__ == "__main__":
    sys.exit(main())
