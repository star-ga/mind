#!/usr/bin/env python3
"""RI-B2 f64 rung (#108) — native-ELF scalar STRICT-FP f64 CHAIN.

`selftest_native_elf_scalar_f64(a_bits, b_bits, c_bits, d_bits)` emits a runnable
x86-64 ET_EXEC that reproduces the cross-substrate canary scalar-float-f64 — the
strict-IEEE scalar chain `a + b - c * d / a` (ref_scalar_f64_chain), fixed source
precedence with NO contraction: the operation tree is
    t1 = a + b ; t2 = c * d ; t3 = t2 / a ; t4 = t1 - t3
running entirely through the RI-B1 memory-operand SSE2 f64 encoders (movsd
load/store, addsd/mulsd/divsd/subsd on [rbp+disp]), unfused (no FMA). The result's
64-bit pattern is lifted with `movq rax,xmm0` (66 48 0F 7E C0) and its 8 LE
bit-bytes (`to_bits() as i64`) written to stdout.

Gate: sha256(the 8 stdout bytes) == the committed canary scalar-float-f64.
Zero MLIR/LLVM in the native path — byte-identical to the MLIR canary.

Usage:
  MINDC_SO=/path/to.so python3 self_host_native_scalar_f64_smoke.py
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

# Committed cross-substrate reference: canary scalar-float-f64 =
# canonical_hash(scalar_f64_chain(a,b,c,d).to_bits() as i64) —
# reference_hashes.toml / scalar_float_f64_reproducibility_gate.
CANARY = "7592a52a5e10a2f24469765f71ce1f9f8ebd9efb51904cf9a18f310d33b3c92d"

# SCALAR_F64_INPUTS = (1.5, 2.25, 0.5, 3.125) — cross_substrate_identity.rs:1591.
INPUTS = (1.5, 2.25, 0.5, 3.125)
_MASK64 = (1 << 64) - 1


def _f64_bits(x: float) -> int:
    """f64 -> i64 argument carrying the 64-bit pattern (movabs imm64 is signed)."""
    u = int.from_bytes(struct.pack("<d", x), "little")
    return u - (1 << 64) if u >= (1 << 63) else u


def _ref_scalar_f64_chain(a: float, b: float, c: float, d: float) -> float:
    # (a + b) - ((c * d) / a) — fixed source precedence, unfused.
    return a + b - c * d / a


def mind_scalar_f64_elf(lib) -> bytes:
    fn = lib.selftest_native_elf_scalar_f64
    fn.restype = ctypes.c_int64
    fn.argtypes = [ctypes.c_int64] * 4
    a, b, c, d = INPUTS
    es = fn(_f64_bits(a), _f64_bits(b), _f64_bits(c), _f64_bits(d))
    rd = lambda addr, o=0: ctypes.cast(addr + o, ctypes.POINTER(ctypes.c_int64))[0]
    sh = rd(es, 0)
    return ctypes.string_at(rd(sh, 0), rd(sh, 8))


def run_elf_capture(elf: bytes, tmp: pathlib.Path) -> bytes:
    p = tmp / "mind_scalar_f64.elf"
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
    if not hasattr(lib, "selftest_native_elf_scalar_f64"):
        print("FAIL  selftest_native_elf_scalar_f64: symbol absent (RI-B2 f64 rung not built)")
        return 1

    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        elf = mind_scalar_f64_elf(lib)
        if not (len(elf) > 120 and elf[:4] == b"\x7fELF"):
            print(f"  FAIL  scalar_f64: not a runnable ELF (len={len(elf)})")
            return 1
        out = run_elf_capture(elf, tmp)
        if len(out) != 8:
            print(f"  FAIL  expected 8 stdout bytes, got {len(out)}: {out.hex()}")
            return 1
        val = struct.unpack("<d", out)[0]
        bits = int.from_bytes(out, "little")
        got = hashlib.sha256(out).hexdigest()
        ok = got == CANARY
        print(f"  native result f64  = {val}  (bits {bits:#018x}, LE8 {out.hex()})")
        print(f"  native sha256      = {got}")
        print(f"  canary scalar-f64  = {CANARY}")
        if ok:
            print("ALL PASS  native-ELF strict-FP f64 chain `a + b - c * d / a` is "
                  "BYTE-IDENTICAL to the MLIR canary scalar-float-f64 — fixed source "
                  "precedence, unfused SSE2 addsd/mulsd/divsd/subsd, movq lift, zero "
                  "MLIR/LLVM (RI-B2 #108, upgrades the f64 native path from "
                  "execution-gated to byte-identity)")
            return 0
        a, b, c, d = INPUTS
        ref = _ref_scalar_f64_chain(a, b, c, d)
        rbits = int.from_bytes(struct.pack("<d", ref), "little")
        print(f"  python oracle      = {ref}  (bits {rbits:#018x})")
        print("FAIL  native-ELF f64 chain hash != canary scalar-float-f64 — op order or "
              "a rounding detail differs; do NOT guess (report native f64/bits/hash above).")
        return 1


if __name__ == "__main__":
    sys.exit(main())
