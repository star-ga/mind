#!/usr/bin/env python3
"""C4-T1 — native-ELF tensor ELEMENT-WISE ADD (i64), zero MLIR/LLVM.

The first tensor/linalg rung that drops MLIR for one tensor op.
`selftest_native_elf_tensor_ewadd_i64(n)` emits a runnable x86-64 ET_EXEC that
(1) materializes two length-n i64 buffers in the frame with deterministic
self-seeding a[i]=i+1, b[i]=2*i, (2) computes c[i] = a[i] + b[i] via an emitted
counted loop over the C2 element load/store primitives (runtime base + i*8
indexing), (3) horizontally reduces sum = Sum c[i] via a second emitted loop,
(4) writes the 8 LE bytes of `sum` to stdout, (5) exits with
(sum == expected)*41 + 1 where `expected` is the emit-time-baked closed form
Sum(3i+1) = 3n(n-1)/2 + n — 42 only on an EXACT full-width i64 match, 1
otherwise (NOT a mod-256-maskable residue).

Two independent full-width gates per n:
  (a) stdout == struct.pack('<q', sum((i+1) + 2*i for i in range(n)))  — the
      runtime loop result checked against Python's independent reference;
  (b) exit == 42 — the in-ELF full-i64 comparison against the baked closed form.
A wrong loop bound, wrong element address, wrong add, or wrong reduction changes
BOTH observables (non-vacuous; multiple distinct n so a fluke cannot pass).

ADDITIVITY: a NEW export never reached during self-compile, so the integer
native-ELF oracle stays byte-identical. No frozen byte oracle exists for this
path; the gate is EXECUTION CORRECTNESS.

Usage:
  MINDC_SO=/path/to.so python3 self_host_native_tensor_ewadd_smoke.py
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


def mind_ewadd_elf(lib, n: int) -> bytes:
    fn = lib.selftest_native_elf_tensor_ewadd_i64
    fn.restype = ctypes.c_int64
    fn.argtypes = [ctypes.c_int64]
    es = fn(n)
    rd = lambda addr, o=0: ctypes.cast(addr + o, ctypes.POINTER(ctypes.c_int64))[0]
    sh = rd(es, 0)  # buf: String handle (addr/len/cap)
    return ctypes.string_at(rd(sh, 0), rd(sh, 8))


def run_elf(elf: bytes, tmp: pathlib.Path):
    p = tmp / "mind_tensor_ewadd.elf"
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
    if not hasattr(lib, "selftest_native_elf_tensor_ewadd_i64"):
        print("FAIL  selftest_native_elf_tensor_ewadd_i64: symbol absent (C4-T1 not built)")
        return 1

    # Distinct shapes: length-1, tiny, non-power-of-2, medium, large (cap 4096).
    ns = [1, 2, 3, 7, 64, 1000, 4096]
    all_ok = True
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        for n in ns:
            expected_sum = sum((i + 1) + 2 * i for i in range(n))
            assert expected_sum == 3 * n * (n - 1) // 2 + n  # closed form sanity
            elf = mind_ewadd_elf(lib, n)
            if not (len(elf) > 120 and elf[:4] == b"\x7fELF"):
                print(f"  FAIL  ewadd(n={n}): not a runnable ELF (len={len(elf)})")
                all_ok = False
                continue
            code, out = run_elf(elf, tmp)
            want = struct.pack("<q", expected_sum)
            ok = code == 42 and out == want
            all_ok = all_ok and ok
            got_sum = struct.unpack("<q", out)[0] if len(out) == 8 else None
            print(
                f"  {'PASS' if ok else 'FAIL'}  ewadd(n={n}) -> exit={code} "
                f"(want 42) stdout_sum={got_sum} expected_sum={expected_sum} "
                f"(elf {len(elf)}B, 3 counted loops, native x86-64, zero MLIR/LLVM)"
            )
    if all_ok:
        print(
            "ALL PASS  tensor element-wise add lowers native-ELF end to end — two "
            "self-seeded i64 buffers, an emitted c[i]=a[i]+b[i] element loop over "
            "base+i*8 addressing, a horizontal reduction loop, full-width stdout "
            "check + exact-i64 in-ELF comparison, zero MLIR/LLVM (C4-T1)"
        )
        return 0
    print("FAIL  native-ELF tensor ewadd gate")
    return 1


if __name__ == "__main__":
    sys.exit(main())
