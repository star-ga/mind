#!/usr/bin/env python3
"""RI-B1 native-ELF scalar-f64 gate (zero MLIR/LLVM).

The pure-MIND native-ELF backend is otherwise i64-only. `selftest_native_elf_fp`
in main.mind emits x86-64 SSE2 scalar-double machine code directly (movabs/movq
bridge for the const, addsd/subsd/mulsd/divsd, cvttsd2si) and wraps it in a real
runnable ELF via the same nb_write_elf scaffold the integer path uses.

No frozen float oracle exists (the deleted Rust native backend returned
Unsupported(ConstF64)), so byte-identity is impossible here by construction. The
oracle is EXECUTION CORRECTNESS: emit the ELF, run it, assert the exit code equals
trunc(a OP b). The CPU is the reference. A wrong SSE encoding either faults or
returns the wrong integer, so this is a genuine, non-fakeable proof.

Usage:
  MINDC_SO=/path/to.so python3 self_host_native_fp_smoke.py
"""
import ctypes
import os
import pathlib
import stat
import struct
import subprocess
import sys
import tempfile


def dbits(x: float) -> int:
    return struct.unpack("<q", struct.pack("<d", x))[0]


def mind_fp_elf(lib, a: float, b: float, op_sel: int, entry: str = "selftest_native_elf_fp") -> bytes:
    fn = getattr(lib, entry)
    fn.restype = ctypes.c_int64
    fn.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
    es = fn(dbits(a), dbits(b), op_sel)
    rd = lambda addr, o=0: ctypes.cast(addr + o, ctypes.POINTER(ctypes.c_int64))[0]
    sh = rd(es, 0)  # buf: String handle (addr/len/cap)
    return ctypes.string_at(rd(sh, 0), rd(sh, 8))


def run_elf(elf: bytes, tmp: pathlib.Path) -> int:
    p = tmp / "mind_fp.elf"
    p.write_bytes(elf)
    p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return subprocess.run([str(p)]).returncode


def main() -> int:
    so = os.environ.get("MINDC_SO")
    if not so or not os.path.exists(so):
        print(f"FAIL  MINDC_SO not set or missing: {so!r}")
        return 1
    lib = ctypes.CDLL(so)

    # (a, b, op_sel, name) -> expected exit = int(trunc(a OP b)), all small +ints.
    cases = [
        (1.5, 2.5, 0, "addsd", 4),   # 4.0
        (7.5, 2.5, 1, "subsd", 5),   # 5.0
        (2.0, 3.0, 2, "mulsd", 6),   # 6.0
        (7.5, 2.5, 3, "divsd", 3),   # 3.0
        (1.5, 2.5, 2, "mulsd-trunc", 3),  # 3.75 -> 3 (truncation is real)
    ]
    all_ok = True
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        for a, b, op, name, expected in cases:
            elf = mind_fp_elf(lib, a, b, op)
            if not (len(elf) > 120 and elf[:4] == b"\x7fELF"):
                print(f"  FAIL  {name}: not a runnable ELF (len={len(elf)})")
                all_ok = False
                continue
            got = run_elf(elf, tmp)
            ok = got == expected
            all_ok = all_ok and ok
            print(
                f"  {'PASS' if ok else 'FAIL'}  {name}: "
                f"trunc({a} op {b}) exit={got} expected={expected} "
                f"(elf {len(elf)}B, SSE2 native, zero MLIR/LLVM)"
            )

        # Memory-operand (stack-slot) forms: every value round-trips through an
        # [rbp+disp32] slot (movsd-load/store F2 0F 10/11 85 + binop-mem F2 0F <opc> 85).
        if not hasattr(lib, "selftest_native_elf_fp_mem"):
            print("  FAIL  selftest_native_elf_fp_mem: symbol absent (encoder not built)")
            all_ok = False
        else:
            for a, b, op, name, expected in cases:
                elf = mind_fp_elf(lib, a, b, op, entry="selftest_native_elf_fp_mem")
                if not (len(elf) > 120 and elf[:4] == b"\x7fELF"):
                    print(f"  FAIL  mem-{name}: not a runnable ELF (len={len(elf)})")
                    all_ok = False
                    continue
                got = run_elf(elf, tmp)
                ok = got == expected
                all_ok = all_ok and ok
                print(
                    f"  {'PASS' if ok else 'FAIL'}  mem-{name}: "
                    f"trunc({a} op {b}) exit={got} expected={expected} "
                    f"(elf {len(elf)}B, SSE2 mem-operand [rbp+d32], zero MLIR/LLVM)"
                )
    if all_ok:
        print(
            "ALL PASS  native-ELF scalar-f64 "
            "(reg-form + mem-operand stack-slot: const/add/sub/mul/div/trunc) — zero MLIR/LLVM"
        )
        return 0
    print("FAIL  native-ELF scalar-f64 gate")
    return 1


if __name__ == "__main__":
    sys.exit(main())
