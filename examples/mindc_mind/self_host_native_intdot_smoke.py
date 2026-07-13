#!/usr/bin/env python3
"""RI-B2-S1 (#108) scalar i64 DOT-PRODUCT reduction native-ELF (zero MLIR/LLVM).

The first rung of the tensor/BLAS native-ELF ladder — the RI-B2 analogue of
RI-B1's float-literal-before-a-float-binop. `selftest_native_elf_intdot(a, b, n)`
emits a runnable x86-64 ET_EXEC that computes `acc = sum a[i]*b[i]` over two
in-memory i64 arrays of length `n`: the fixtures are baked into the entry stack
frame as movabs immediates, then a counted `while i < n` loop RE-DERIVES each
element address at runtime (`base + i*8`), loads a[i] and b[i] through the
register-indirect i64 load, multiplies (imul rax,[mem]) and accumulates (add
rax,[mem]) — so the streaming reduction-over-memory path runs end to end. The
final accumulator is returned as the process exit code.

PURE REUSE: the emitter composes only existing nb_* helpers (movabs/store/
load-i64/arith-mem cmp-je/jmp scaffold), NO new encoder. It is a NEW export never
reached during self-compile, so the integer native-ELF oracle stays byte-
identical (additivity).

No frozen BYTE oracle exists for this path yet (the deleted Rust native backend
never emitted a dot-product entry; byte-identity vs the MLIR canary is the later
RI-B2-S3 slice). The oracle here is EXECUTION CORRECTNESS: emit the ELF, run it,
assert exit == the exact dot. Fixtures are chosen so the dot fits in a byte
(0..255), and at least 4 DISTINCT shapes so a wrong loop cannot fluke one seed.

Usage:
  MINDC_SO=/path/to.so python3 self_host_native_intdot_smoke.py
"""
import ctypes
import os
import pathlib
import stat
import subprocess
import sys
import tempfile

_HERE = pathlib.Path(__file__).parent
_DEFAULT_SO = _HERE / "libmindc_mind.so"


def mind_intdot_elf(lib, a: list[int], b: list[int]) -> bytes:
    assert len(a) == len(b)
    n = len(a)
    a_arr = (ctypes.c_int64 * n)(*a)
    b_arr = (ctypes.c_int64 * n)(*b)
    fn = lib.selftest_native_elf_intdot
    fn.restype = ctypes.c_int64
    fn.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
    es = fn(
        ctypes.cast(a_arr, ctypes.c_void_p).value,
        ctypes.cast(b_arr, ctypes.c_void_p).value,
        n,
    )
    rd = lambda addr, o=0: ctypes.cast(addr + o, ctypes.POINTER(ctypes.c_int64))[0]
    sh = rd(es, 0)  # buf: String handle (addr/len/cap)
    return ctypes.string_at(rd(sh, 0), rd(sh, 8))


def run_elf(elf: bytes, tmp: pathlib.Path) -> int:
    p = tmp / "mind_intdot.elf"
    p.write_bytes(elf)
    p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return subprocess.run([str(p)]).returncode


def main() -> int:
    so = os.environ.get("MINDC_SO", str(_DEFAULT_SO))
    if not os.path.exists(so):
        if os.environ.get("MINDC_SO"):
            print(f"FAIL  MINDC_SO set but missing: {so!r}")
            return 1
        print(f"SKIP  {so} not built")
        return 0
    lib = ctypes.CDLL(so)
    if not hasattr(lib, "selftest_native_elf_intdot"):
        print("FAIL  selftest_native_elf_intdot: symbol absent (RI-B2-S1 not built)")
        return 1

    # (a, b, expected dot). At least 4 DISTINCT shapes; every dot fits in a byte.
    cases = [
        ([1, 2, 3], [4, 5, 6], 32),           # 4+10+18
        ([2, 2, 2, 2], [1, 2, 3, 4], 20),     # 2+4+6+8
        ([10], [7], 70),                      # length-1
        ([1, 1, 1, 1, 1], [1, 1, 1, 1, 1], 5),  # length-5 loop
        ([3, 4], [5, 6], 39),                 # 15+24
    ]
    all_ok = True
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        for a, b, expected in cases:
            elf = mind_intdot_elf(lib, a, b)
            if not (len(elf) > 120 and elf[:4] == b"\x7fELF"):
                print(f"  FAIL  dot({a},{b}): not a runnable ELF (len={len(elf)})")
                all_ok = False
                continue
            got = run_elf(elf, tmp)
            ok = got == expected
            all_ok = all_ok and ok
            print(
                f"  {'PASS' if ok else 'FAIL'}  dot(a={a}, b={b}) -> "
                f"exit={got} expected={expected} "
                f"(elf {len(elf)}B, counted while-loop, native x86-64, zero MLIR/LLVM)"
            )
    if all_ok:
        print(
            "ALL PASS  scalar i64 dot-product reduction lowers native-ELF end to end "
            "— a counted `while i < n` loop indexes two in-memory i64 arrays "
            "(base + i*8), multiplies element-wise and accumulates, exit(acc), with "
            "zero MLIR/LLVM (RI-B2-S1 #108 tensor/BLAS ladder rung 1)"
        )
        return 0
    print("FAIL  native-ELF int dot-product reduction gate")
    return 1


if __name__ == "__main__":
    sys.exit(main())
