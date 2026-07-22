#!/usr/bin/env python3
"""C4-T5 — native-ELF tensor COLUMN REDUCTION (i64), zero MLIR/LLVM.

The axis-transpose of C4-T4 rowsum. `selftest_native_elf_tensor_colsum_i64(m, n)`
emits a runnable x86-64 ET_EXEC that (1) materializes A (MxN) row-major in the
frame with deterministic self-seeding A[i*n+j] = i*n+j+1 (the flat index t+1),
(2) computes colsum[j] = Sum_{i<m} A[i*n+j] for each COLUMN via an inner counted
loop over ROWS (ri) nested under the column loop (cj) — outer=cj, inner=ri, 2-D
row-major addressing ((i*n)+j)*8 + base (a=ri, dim=n, b=cj), the axis-transpose
of rowsum's outer=ri/inner=cj, (3) folds the columns into a SQUARED checksum
sum = Sum_{j<n} colsum[j]*colsum[j], (4) writes the 8 LE bytes of `sum` to
stdout, and (5) exits (sum == expected)*41 + 1 — 42 only on an EXACT full-width
i64 match against the emit-time-baked expected checksum (movabs-baked past
imm32), 1 otherwise.

Why SQUARED (layout/axis-discriminating): the grand sum of A is axis-invariant,
so a plain sum-of-colsums equals the sum-of-rowsums — a wrong reduction axis
would NOT be caught. Squaring each colsum before folding makes the observable
depend on the COLUMN partition: a row reduction, a transposed stride (i*n+j vs
j*m+i), or a wrong bound all yield a different sum-of-squares. This smoke proves
it by also computing the ROW sum-of-squares and asserting it differs from the
column one on every non-square shape (so the ELF's value provably could not have
come from the wrong axis).

Two independent full-width gates per shape:
  (a) stdout == struct.pack('<q', S) where S is THIS script's pure-Python
      seed-reduce-square over the same seeds — an independent reference (no
      shared code with the emitter);
  (b) exit == 42 — the in-ELF exact-i64 comparison against the baked checksum.

SHAPE GUARD (the T1/T2/T3 audit hazard): the export FAILS CLOSED — returns an
empty buffer — unless 1 <= m,n and m <= 4096 and n <= 4096 and m*n <= 4096 (one
4096-element frame array). This smoke asserts the refusal on out-of-bounds and
degenerate shapes too, including i64-overflow shapes where m*n wraps to 0.

ADDITIVITY: a NEW export never reached during self-compile, so the integer
native-ELF oracle stays byte-identical. No frozen byte oracle exists for this
path; the gate is EXECUTION CORRECTNESS.

Usage:
  MINDC_SO=/path/to.so python3 self_host_native_tensor_colsum_smoke.py
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


def mind_colsum_elf(lib, m: int, n: int) -> bytes:
    fn = lib.selftest_native_elf_tensor_colsum_i64
    fn.restype = ctypes.c_int64
    fn.argtypes = [ctypes.c_int64, ctypes.c_int64]
    es = fn(m, n)
    rd = lambda addr, o=0: ctypes.cast(addr + o, ctypes.POINTER(ctypes.c_int64))[0]
    sh = rd(es, 0)  # buf: String handle (addr/len/cap)
    ln = rd(sh, 8)
    if ln <= 0:
        return b""
    return ctypes.string_at(rd(sh, 0), ln)


def ref_colsum_sq(m: int, n: int) -> int:
    """Independent pure-Python reference: seed, COLUMN-reduce, square-fold."""
    a = [[i * n + j + 1 for j in range(n)] for i in range(m)]
    total = 0
    for j in range(n):
        c = 0
        for i in range(m):
            c += a[i][j]
        total += c * c
    return total


def ref_rowsum_sq(m: int, n: int) -> int:
    """The WRONG-axis value — used only to prove the column value is discriminating."""
    a = [[i * n + j + 1 for j in range(n)] for i in range(m)]
    total = 0
    for i in range(m):
        r = 0
        for j in range(n):
            r += a[i][j]
        total += r * r
    return total


def run_elf(elf: bytes, tmp: pathlib.Path):
    p = tmp / "mind_tensor_colsum.elf"
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
    if not hasattr(lib, "selftest_native_elf_tensor_colsum_i64"):
        print("FAIL  selftest_native_elf_tensor_colsum_i64: symbol absent (C4-T5 not built)")
        return 1

    # Distinct shapes: 1x1, non-square both orientations (row!=col sum-of-squares,
    # catches wrong-axis/transposed-stride bugs), single row, single column, square
    # mid, wide, tall, and the frame cap (64x64: m*n = 4096 exactly). The larger
    # shapes push the checksum past imm32 — exercises the movabs baking.
    shapes = [(1, 1), (2, 3), (3, 2), (1, 16), (16, 1), (8, 8), (2, 64), (64, 2), (64, 64)]
    all_ok = True
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        for (m, n) in shapes:
            expected_sum = ref_colsum_sq(m, n)
            row_sum = ref_rowsum_sq(m, n)
            # Discriminating-by-construction: on every non-square shape the wrong
            # axis gives a different value, so a value == expected_sum proves the
            # COLUMN partition was used (not a fluke of a symmetric checksum).
            if m != n:
                assert expected_sum != row_sum, (
                    f"non-discriminating shape {m}x{n}: col==row sum-of-squares"
                )
            elf = mind_colsum_elf(lib, m, n)
            if not (len(elf) > 120 and elf[:4] == b"\x7fELF"):
                print(f"  FAIL  colsum({m}x{n}): not a runnable ELF (len={len(elf)})")
                all_ok = False
                continue
            code, out = run_elf(elf, tmp)
            want = struct.pack("<q", expected_sum)
            ok = code == 42 and out == want
            all_ok = all_ok and ok
            got_sum = struct.unpack("<q", out)[0] if len(out) == 8 else None
            # Non-vacuous: on non-square shapes the ELF value must NOT equal the
            # wrong-axis (row) sum-of-squares — a stronger, per-run distinctness gate.
            if ok and m != n:
                assert got_sum != row_sum, (
                    f"colsum({m}x{n}) produced the ROW-axis value {row_sum} — wrong axis"
                )
            print(
                f"  {'PASS' if ok else 'FAIL'}  colsum({m}x{n}) -> exit={code} "
                f"(want 42) stdout_sum={got_sum} expected_sum={expected_sum} "
                f"row_sum={row_sum} (elf {len(elf)}B, seed nest + col-reduce nest "
                f"+ squared fold, native x86-64, zero MLIR/LLVM)"
            )

        # Shape guard: out-of-frame and degenerate shapes must FAIL CLOSED (empty
        # buffer) — the T1/T2/T3 frame-overrun audit hazard. The last two are
        # i64-overflow shapes: m*n wraps to 0 (mod 2^64), so the `m*n > 4096`
        # product check alone would pass them through — the per-dim `> 4096` bound
        # (applied before the product) is what refuses them. 2^32=4294967296,
        # 2^33=8589934592 (2^33 * 2^33 = 2^66 == 0 mod 2^64).
        for (m, n) in [
            (65, 64), (64, 65), (1, 4097), (4097, 1), (0, 1), (1, 0), (1, -3), (-3, 1),
            (4294967296, 4294967296),
            (8589934592, 8589934592),
        ]:
            elf = mind_colsum_elf(lib, m, n)
            ok = len(elf) == 0
            all_ok = all_ok and ok
            print(
                f"  {'PASS' if ok else 'FAIL'}  colsum({m}x{n}) refused "
                f"(fail-closed shape guard, got {len(elf)}B want 0B)"
            )
    if all_ok:
        print(
            "ALL PASS  tensor column-reduction lowers native-ELF end to end — a "
            "self-seeded 2-D row-major i64 operand, an emitted seed nest + a "
            "col-reduce nest over ((i*n)+j)*8+base addressing (outer=cj/inner=ri, "
            "the axis-transpose of rowsum), a SQUARED (layout-discriminating) "
            "column fold, full-width stdout check + exact-i64 in-ELF comparison "
            "(movabs-baked past imm32), fail-closed frame-bound guard, wrong-axis "
            "value proven distinct, zero MLIR/LLVM (C4-T5)"
        )
        return 0
    print("FAIL  native-ELF tensor colsum gate")
    return 1


if __name__ == "__main__":
    sys.exit(main())
