#!/usr/bin/env python3
"""RI-B2-S4 (#108) — native-ELF scalar Q16.16 DOT-PRODUCT, byte-identity rung.

`selftest_native_elf_intdot_q16(n)` emits a runnable x86-64 ET_EXEC that:
  (1) generates a[0..n] then b[0..n] via the EXACT LCG the cross-substrate canary
      dot-l2-q16 uses (seed 0xDEADBEEF; state = state*1664525 + 1013904223;
      next_u32 = (state >> 16) as u32; next_q16 = (next_u32 as i32) >> 12),
      into two in-frame i32 Q16.16 arrays,
  (2) computes acc = Sum (sext(a[i]) * sext(b[i])) >> 16 with a FULL i64
      accumulator (each i32 element sign-extended from memory via movsxd —
      nb_emit_load_i32_sx; per-product Q16.16 rescale via SAR 16),
  (3) narrows ONCE to i32 (movsxd) — the MLIR canary's `acc as i32`,
  (4) writes the 8 LITTLE-ENDIAN bytes of the result to stdout via write(2).

Gate: emit the ELF, run it, capture the 8 stdout bytes, sha256 == the committed
cross-substrate reference for `dot-l2-q16`. Zero MLIR/LLVM in the native path.

Usage:
  MINDC_SO=/path/to.so python3 self_host_native_intdot_q16_smoke.py
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

# Committed cross-substrate reference: canary `dot-l2-q16` =
# canonical_hash(result) = sha256 of the 8 LE bytes of the i64 dot result
# (tests/cross_substrate_identity.rs dot_l2_q16_reproducibility_gate).
CANARY = "1d7f272b85e5f0fd7cf473086fb1da558a723134ff02ef30a4323eb757209823"
LENGTH = 65536


def mind_intdot_q16_elf(lib, n: int) -> bytes:
    fn = lib.selftest_native_elf_intdot_q16
    fn.restype = ctypes.c_int64
    fn.argtypes = [ctypes.c_int64]
    es = fn(n)
    rd = lambda addr, o=0: ctypes.cast(addr + o, ctypes.POINTER(ctypes.c_int64))[0]
    sh = rd(es, 0)  # buf: String handle (addr/len/cap)
    return ctypes.string_at(rd(sh, 0), rd(sh, 8))


def run_elf_capture(elf: bytes, tmp: pathlib.Path) -> bytes:
    p = tmp / "mind_intdot_q16.elf"
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
    if not hasattr(lib, "selftest_native_elf_intdot_q16"):
        print("FAIL  selftest_native_elf_intdot_q16: symbol absent (RI-B2-S4 not built)")
        return 1

    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        elf = mind_intdot_q16_elf(lib, LENGTH)
        if not (len(elf) > 120 and elf[:4] == b"\x7fELF"):
            print(f"  FAIL  intdot_q16(n={LENGTH}): not a runnable ELF (len={len(elf)})")
            return 1
        out = run_elf_capture(elf, tmp)
        if len(out) != 8:
            print(f"  FAIL  expected 8 stdout bytes, got {len(out)}: {out.hex()}")
            return 1
        val = int.from_bytes(out, "little", signed=True)
        got = hashlib.sha256(out).hexdigest()
        ok = got == CANARY
        print(f"  native result i64 = {val}  (LE bytes {out.hex()})")
        print(f"  native sha256      = {got}")
        print(f"  canary dot-l2-q16  = {CANARY}")
        if ok:
            print("ALL PASS  native-ELF Q16.16 dot-product is BYTE-IDENTICAL to the MLIR "
                  "canary dot-l2-q16 — i64 accumulate, per-product >>16 (SAR), narrow once "
                  "to i32, zero MLIR/LLVM (RI-B2-S4 #108)")
            return 0
        print("FAIL  native-ELF Q16.16 dot hash != canary dot-l2-q16 — reduction model "
              "differs from the i64-accumulate/per-product->>16/narrow-i32; report native "
              "value + hash for mind-det-gemm to pin the exact MLIR reduction model")
        return 1


if __name__ == "__main__":
    sys.exit(main())
