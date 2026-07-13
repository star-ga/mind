#!/usr/bin/env python3
"""RI-B2-S6 (#108) — native-ELF scalar GEMV int16 (matrix x vector), byte-identity rung.

`selftest_native_elf_gemv_i16(m, n)` emits a runnable x86-64 ET_EXEC that:
  (1) generates W[0..m*n] then x[0..n] via the EXACT LCG the cross-substrate canary
      gemv-i16 uses (seed 0xDEADBEEF; state = state*1664525 + 1013904223;
      element = (state >> 32) low 16 bits as i16), W row-major then x,
  (2) for each row r computes y[r] = Sum_c sext(W[r,c]) * sext(x[c]) with a FULL
      i64 accumulator (movsx WORD loads), narrows ONCE to i32 (movsxd) — the MLIR
      canary's `acc as i32`,
  (3) writes the 4 LITTLE-ENDIAN bytes of each y[r] to stdout in ROW-MAJOR order.

Gate: emit the ELF, run it, capture the m*4 stdout bytes, sha256 == the committed
cross-substrate reference for `gemv-i16` (canonical_hash_i32s of the y vector).
Zero MLIR/LLVM in the native path — the first native-ELF GEMV with a byte-identity
oracle vs the MLIR/LLVM substrate.

Usage:
  MINDC_SO=/path/to.so python3 self_host_native_gemv_i16_smoke.py
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

# Committed cross-substrate reference: canary `gemv-i16` (256x256, seed 0xDEADBEEF)
# = canonical_hash_i32s(y) = sha256 of each i32 y[r].to_le_bytes() row-major
# (tests/cross_substrate_identity.rs gemv_i16_reproducibility_gate).
CANARY = "3238e8c7e1e9ee9937503700f63eda350fcd10e7db28d470c3dbc26592d0a936"
ROWS = 256
COLS = 256


def mind_gemv_i16_elf(lib, m: int, n: int) -> bytes:
    fn = lib.selftest_native_elf_gemv_i16
    fn.restype = ctypes.c_int64
    fn.argtypes = [ctypes.c_int64, ctypes.c_int64]
    es = fn(m, n)
    rd = lambda addr, o=0: ctypes.cast(addr + o, ctypes.POINTER(ctypes.c_int64))[0]
    sh = rd(es, 0)  # buf: String handle (addr/len/cap)
    return ctypes.string_at(rd(sh, 0), rd(sh, 8))


def run_elf_capture(elf: bytes, tmp: pathlib.Path) -> bytes:
    p = tmp / "mind_gemv_i16.elf"
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
    if not hasattr(lib, "selftest_native_elf_gemv_i16"):
        print("FAIL  selftest_native_elf_gemv_i16: symbol absent (RI-B2-S6 not built)")
        return 1

    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        elf = mind_gemv_i16_elf(lib, ROWS, COLS)
        if not (len(elf) > 120 and elf[:4] == b"\x7fELF"):
            print(f"  FAIL  gemv_i16({ROWS}x{COLS}): not a runnable ELF (len={len(elf)})")
            return 1
        out = run_elf_capture(elf, tmp)
        if len(out) != ROWS * 4:
            print(f"  FAIL  expected {ROWS*4} stdout bytes, got {len(out)}: {out[:64].hex()}")
            return 1
        got = hashlib.sha256(out).hexdigest()
        ok = got == CANARY
        y0 = int.from_bytes(out[0:4], "little", signed=True)
        yl = int.from_bytes(out[-4:], "little", signed=True)
        print(f"  native y[0]={y0}  y[{ROWS-1}]={yl}  ({len(out)} stdout bytes)")
        print(f"  native sha256      = {got}")
        print(f"  canary gemv-i16    = {CANARY}")
        if ok:
            print("ALL PASS  native-ELF int16 GEMV is BYTE-IDENTICAL to the MLIR canary "
                  "gemv-i16 — outer row-loop over the proven i64-MAC/narrow-i32 dot, "
                  "row-major i32 vector output, zero MLIR/LLVM (RI-B2-S6 #108)")
            return 0
        print("FAIL  native-ELF int16 GEMV hash != canary gemv-i16 — reduction/order "
              "differs from the straightforward per-row i64-accumulate/narrow-i32; "
              "report native hash + y vector for mind-det-gemm to pin the model")
        return 1


if __name__ == "__main__":
    sys.exit(main())
