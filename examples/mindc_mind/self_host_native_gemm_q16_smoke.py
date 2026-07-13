#!/usr/bin/env python3
"""RI-B2-S7 (#108) — native-ELF scalar GEMM Q16.16 (matrix x matrix), byte-identity rung.

`selftest_native_elf_gemm_q16(m, n, k)` emits a runnable x86-64 ET_EXEC that:
  (1) generates A[0..m*k] then B[0..k*n] via the EXACT Q16.16 LCG the cross-substrate
      canary gemm-q16 uses (seed 0xDEADBEEF; state = state*1664525 + 1013904223;
      element next_q16 = ((state>>16) as u32 as i32) >> 12), A row-major then B,
  (2) for each output C[i,j] computes Sum_k (sext(A[i,k]) * sext(B[k,j])) >> 16 with a
      FULL i64 accumulator (movsxd DWORD loads, per-product Q16 rescale; B read down
      column j, stride N), narrows ONCE to i32 (movsxd) — the canary's `acc as i32`,
  (3) writes the 4 LITTLE-ENDIAN bytes of each C[i,j] to stdout in ROW-MAJOR order.

Gate: emit the ELF, run it, capture the m*n*4 stdout bytes, sha256 == the committed
cross-substrate reference for `gemm-q16` (canonical_hash_i32s of the M*N matrix).
Zero MLIR/LLVM in the native path.

Usage:
  MINDC_SO=/path/to.so python3 self_host_native_gemm_q16_smoke.py
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

# Committed cross-substrate reference: canary `gemm-q16` (64x64x64, seed 0xDEADBEEF)
# = canonical_hash_i32s(C) = sha256 of each i32 C[i,j].to_le_bytes() row-major
# (tests/cross_substrate_identity.rs gemm_q16_reproducibility_gate).
CANARY = "92e2cb75d74d83a4a398d78d9ac560f195279c31814972c892f856f675faea0f"
M = 64
N = 64
K = 64


def mind_gemm_q16_elf(lib, m: int, n: int, k: int) -> bytes:
    fn = lib.selftest_native_elf_gemm_q16
    fn.restype = ctypes.c_int64
    fn.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
    es = fn(m, n, k)
    rd = lambda addr, o=0: ctypes.cast(addr + o, ctypes.POINTER(ctypes.c_int64))[0]
    sh = rd(es, 0)  # buf: String handle (addr/len/cap)
    return ctypes.string_at(rd(sh, 0), rd(sh, 8))


def run_elf_capture(elf: bytes, tmp: pathlib.Path) -> bytes:
    p = tmp / "mind_gemm_q16.elf"
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
    if not hasattr(lib, "selftest_native_elf_gemm_q16"):
        print("FAIL  selftest_native_elf_gemm_q16: symbol absent (RI-B2-S7 not built)")
        return 1

    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        elf = mind_gemm_q16_elf(lib, M, N, K)
        if not (len(elf) > 120 and elf[:4] == b"\x7fELF"):
            print(f"  FAIL  gemm_q16({M}x{N}x{K}): not a runnable ELF (len={len(elf)})")
            return 1
        out = run_elf_capture(elf, tmp)
        if len(out) != M * N * 4:
            print(f"  FAIL  expected {M*N*4} stdout bytes, got {len(out)}: {out[:64].hex()}")
            return 1
        got = hashlib.sha256(out).hexdigest()
        ok = got == CANARY
        c0 = int.from_bytes(out[0:4], "little", signed=True)
        cl = int.from_bytes(out[-4:], "little", signed=True)
        print(f"  native C[0,0]={c0}  C[{M-1},{N-1}]={cl}  ({len(out)} stdout bytes)")
        print(f"  native sha256      = {got}")
        print(f"  canary gemm-q16    = {CANARY}")
        if ok:
            print("ALL PASS  native-ELF Q16.16 GEMM is BYTE-IDENTICAL to the MLIR canary "
                  "gemm-q16 — triple loop over the proven i64-MAC/>>16/narrow-i32 dot, "
                  "row-major i32 matrix output, zero MLIR/LLVM (RI-B2-S7 #108)")
            return 0
        print("FAIL  native-ELF Q16.16 GEMM hash != canary gemm-q16 — reduction/transpose "
              "differs from the per-output (Q16-rescale) accumulate over B column j; "
              "report native hash for mind-det-gemm to pin the model")
        return 1


if __name__ == "__main__":
    sys.exit(main())
