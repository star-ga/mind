#!/usr/bin/env python3
"""C4-T2 — native-ELF tensor DOT PRODUCT (i64), zero MLIR/LLVM.

The multiply-accumulate rung of the tensor/linalg ladder (the core reduction of
matmul — a later T3 nests this loop for 2-D matmul).
`selftest_native_elf_tensor_dot_i64(n)` emits a runnable x86-64 ET_EXEC that
(1) materializes two length-n i64 buffers in the frame with deterministic
self-seeding a[i]=i+1, b[i]=2*i+3 (nonzero dot at EVERY n, incl. n=1, so each
shape discriminates), (2) computes acc = Sum a[i]*b[i] via one emitted counted
loop doing load-load-mul-accumulate over base + i*8 element addressing,
(3) writes the 8 LE bytes of `acc` to stdout, (4) exits with
(acc == expected)*41 + 1 where `expected` is the emit-time-baked closed form
Sum(i+1)(2i+3) = n(n-1)(2n-1)/3 + 5n(n-1)/2 + 3n — 42 only on an EXACT
full-width i64 match, 1 otherwise (NOT a mod-256-maskable residue).

Two independent full-width gates per n:
  (a) stdout == struct.pack('<q', sum((i+1) * (2*i+3) for i in range(n)))  —
      the runtime MAC loop result checked against Python's independent
      reference;
  (b) exit == 42 — the in-ELF full-i64 comparison against the baked closed
      form (movabs, so it stays full-width past imm32 at large n).
A wrong loop bound, wrong element address, wrong multiply, or wrong accumulate
changes BOTH observables (non-vacuous; multiple distinct n so a fluke cannot
pass).

ADDITIVITY: a NEW export never reached during self-compile, so the integer
native-ELF oracle stays byte-identical. No frozen byte oracle exists for this
path; the gate is EXECUTION CORRECTNESS.

Usage:
  MINDC_SO=/path/to.so python3 self_host_native_tensor_dot_smoke.py
"""
import ctypes
import os
import pathlib
import stat
import struct
import subprocess
import sys
import tempfile

_HERE = pathlib.Path(__file__).parent
_DEFAULT_SO = _HERE / "libmindc_mind.so"


def mind_dot_elf(lib, n: int) -> bytes:
    fn = lib.selftest_native_elf_tensor_dot_i64
    fn.restype = ctypes.c_int64
    fn.argtypes = [ctypes.c_int64]
    es = fn(n)
    rd = lambda addr, o=0: ctypes.cast(addr + o, ctypes.POINTER(ctypes.c_int64))[0]
    sh = rd(es, 0)  # buf: String handle (addr/len/cap)
    return ctypes.string_at(rd(sh, 0), rd(sh, 8))


def run_elf(elf: bytes, tmp: pathlib.Path):
    p = tmp / "mind_tensor_dot.elf"
    p.write_bytes(elf)
    p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    r = subprocess.run([str(p)], capture_output=True)
    return r.returncode, r.stdout


def main() -> int:
    so = os.environ.get("MINDC_SO", str(_DEFAULT_SO))
    if not os.path.exists(so):
        if os.environ.get("MINDC_SO"):
            print(f"FAIL  MINDC_SO set but missing: {so!r}")
            return 1
        print(f"SKIP  {so} not built")
        return 0
    lib = ctypes.CDLL(so)
    if not hasattr(lib, "selftest_native_elf_tensor_dot_i64"):
        print("FAIL  selftest_native_elf_tensor_dot_i64: symbol absent (C4-T2 not built)")
        return 1

    # Distinct shapes: length-1, tiny, non-power-of-2, medium, large (cap 4096).
    # At n=4096 the dot (~45.8e9) exceeds imm32 — exercises the movabs baking.
    ns = [1, 2, 3, 7, 100, 1000, 4096]
    all_ok = True
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        for n in ns:
            expected_dot = sum((i + 1) * (2 * i + 3) for i in range(n))
            closed = n * (n - 1) * (2 * n - 1) // 3 + 5 * n * (n - 1) // 2 + 3 * n
            assert expected_dot == closed  # closed form sanity
            elf = mind_dot_elf(lib, n)
            if not (len(elf) > 120 and elf[:4] == b"\x7fELF"):
                print(f"  FAIL  dot(n={n}): not a runnable ELF (len={len(elf)})")
                all_ok = False
                continue
            code, out = run_elf(elf, tmp)
            want = struct.pack("<q", expected_dot)
            ok = code == 42 and out == want
            all_ok = all_ok and ok
            got_dot = struct.unpack("<q", out)[0] if len(out) == 8 else None
            print(
                f"  {'PASS' if ok else 'FAIL'}  dot(n={n}) -> exit={code} "
                f"(want 42) stdout_dot={got_dot} expected_dot={expected_dot} "
                f"(elf {len(elf)}B, 2 counted loops, native x86-64, zero MLIR/LLVM)"
            )
    if all_ok:
        print(
            "ALL PASS  tensor dot product lowers native-ELF end to end — two "
            "self-seeded i64 buffers, an emitted acc += a[i]*b[i] "
            "multiply-accumulate loop over base+i*8 addressing, full-width "
            "stdout check + exact-i64 in-ELF comparison (movabs-baked past "
            "imm32), zero MLIR/LLVM (C4-T2)"
        )
        return 0
    print("FAIL  native-ELF tensor dot gate")
    return 1


if __name__ == "__main__":
    sys.exit(main())
