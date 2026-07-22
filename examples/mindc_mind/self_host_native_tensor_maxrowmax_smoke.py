#!/usr/bin/env python3
"""C4-T6 — native-ELF tensor ROW MAX-REDUCTION (i64), zero MLIR/LLVM.

The argmax-reduction rung of the tensor/linalg ladder.
`selftest_native_elf_tensor_maxrowmax_i64(m, n)` emits a runnable x86-64 ET_EXEC
that (1) materializes A (MxN) row-major in the frame with deterministic
self-seeding A[i*n+j] = ((i*7 + j*3 + 1) mod 97) + 1 (every element in [1,97],
nonzero at every index), (2) computes rowmax[i] = max_{j<n} A[i*n+j] for each row
via an inner counted loop nested under the row loop (2-D row-major addressing
((i*n)+j)*8 + base) whose running max is tracked by a native
`cmp rax,[acc] ; cmovl rax,[aval]` — a real compare + conditional-move, branchless
in the reduction body, (3) folds the rows into a POSITION-WEIGHTED checksum
sum = Sum_{i<m} (i+1)*rowmax[i], (4) writes the 8 LE bytes of `sum` to stdout, and
(5) exits (sum == expected)*41 + 1 — 42 only on an EXACT full-width i64 match
against the emit-time-baked expected checksum (movabs-baked), 1 otherwise.

Why the seed is a genuine ARGMAX (not the last element): the +3 column stride is
taken mod 97, so once a row is wide enough for 3*j to cross 97 (n >= ~34) the row
values WRAP and the maximum lands on an INTERIOR column, not the last. This smoke
proves it on the wide shapes by computing the last-element variant and asserting
it DIFFERS from the true max-reduction there.

Why POSITION-WEIGHTED (discriminating): a wrong reduction that changes even one
rowmax — a MIN, a first/last-element pick, a wrong axis — lands at a different
(i+1)-weighted total. This smoke computes the MIN-reduction and the LAST-element
reductions independently and asserts each is DISTINCT from the true value on the
shapes where they can differ (so a value == expected provably came from the MAX,
not a fluke of a coincident wrong reduction).

Two independent full-width gates per shape:
  (a) stdout == struct.pack('<q', S) where S is THIS script's pure-Python
      seed-maxreduce-weight over the same seeds — an independent reference (no
      shared code with the emitter);
  (b) exit == 42 — the in-ELF exact-i64 comparison against the baked checksum.

SHAPE GUARD (the T1..T5 audit hazard): the export FAILS CLOSED — returns an empty
buffer — unless 1 <= m,n and m <= 4096 and n <= 4096 and m*n <= 4096 (one
4096-element frame array). This smoke asserts the refusal on out-of-bounds and
degenerate shapes too, including i64-overflow shapes where m*n wraps to 0.

ADDITIVITY: a NEW export never reached during self-compile, so the integer
native-ELF oracle stays byte-identical. No frozen byte oracle exists for this
path; the gate is EXECUTION CORRECTNESS.

Usage:
  MINDC_SO=/path/to.so python3 self_host_native_tensor_maxrowmax_smoke.py
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


def mind_maxrowmax_elf(lib, m: int, n: int) -> bytes:
    fn = lib.selftest_native_elf_tensor_maxrowmax_i64
    fn.restype = ctypes.c_int64
    fn.argtypes = [ctypes.c_int64, ctypes.c_int64]
    es = fn(m, n)
    rd = lambda addr, o=0: ctypes.cast(addr + o, ctypes.POINTER(ctypes.c_int64))[0]
    sh = rd(es, 0)  # buf: String handle (addr/len/cap)
    ln = rd(sh, 8)
    if ln <= 0:
        return b""
    return ctypes.string_at(rd(sh, 0), ln)


def _seed(i: int, j: int) -> int:
    return ((i * 7 + j * 3 + 1) % 97) + 1


def ref_maxrowmax(m: int, n: int) -> int:
    """Independent pure-Python reference: seed, ROW-MAX, position-weighted fold."""
    total = 0
    for i in range(m):
        r = max(_seed(i, j) for j in range(n))
        total += (i + 1) * r
    return total


def wrong_minreduce(m: int, n: int) -> int:
    """WRONG variant: a MIN reduction with the same weighting."""
    total = 0
    for i in range(m):
        r = min(_seed(i, j) for j in range(n))
        total += (i + 1) * r
    return total


def wrong_lastelem(m: int, n: int) -> int:
    """WRONG variant: pick the LAST column instead of the max."""
    return sum((i + 1) * _seed(i, n - 1) for i in range(m))


def wrong_firstelem(m: int, n: int) -> int:
    """WRONG variant: pick the FIRST column instead of the max."""
    return sum((i + 1) * _seed(i, 0) for i in range(m))


def run_elf(elf: bytes, tmp: pathlib.Path):
    p = tmp / "mind_tensor_maxrowmax.elf"
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
    if not hasattr(lib, "selftest_native_elf_tensor_maxrowmax_i64"):
        print("FAIL  selftest_native_elf_tensor_maxrowmax_i64: symbol absent (C4-T6 not built)")
        return 1

    # Distinct shapes: 1x1, small non-square (min-reduction distinct), single row,
    # single column, square mid, and the WIDE wrapping shapes (1x64/2x64/64x64)
    # where the seed row wraps mod 97 so the max is a genuine INTERIOR argmax and
    # the last-element variant provably differs. 64x64 = m*n = 4096 (frame cap).
    shapes = [(1, 1), (2, 3), (3, 2), (1, 16), (16, 1), (8, 8), (1, 64), (2, 64), (64, 64)]
    all_ok = True
    proved_interior_argmax = False
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        for (m, n) in shapes:
            expected_sum = ref_maxrowmax(m, n)
            v_min = wrong_minreduce(m, n)
            v_last = wrong_lastelem(m, n)
            v_first = wrong_firstelem(m, n)
            # Discriminating-by-construction: on any shape with a non-degenerate
            # row (n >= 2 and not all-equal) a MIN reduction gives a DIFFERENT
            # weighted total, so a value == expected_sum could not have come from a
            # min. Assert it where it can differ.
            if expected_sum != v_min:
                pass  # genuinely distinct (the common case for n >= 2)
            elif n >= 2:
                # would only be equal if every row were constant — not true here
                assert False, f"non-discriminating vs MIN at {m}x{n}"
            # Interior-argmax proof: on the wide wrapping shapes the true max is an
            # interior column, so the last-element variant MUST differ.
            if n >= 34 and m * n > 1:
                assert expected_sum != v_last, (
                    f"max==last at {m}x{n}: seed not wrapping / not a real argmax"
                )
                assert expected_sum != v_first, f"max==first at {m}x{n}"
                proved_interior_argmax = True

            elf = mind_maxrowmax_elf(lib, m, n)
            if not (len(elf) > 120 and elf[:4] == b"\x7fELF"):
                print(f"  FAIL  maxrowmax({m}x{n}): not a runnable ELF (len={len(elf)})")
                all_ok = False
                continue
            code, out = run_elf(elf, tmp)
            want = struct.pack("<q", expected_sum)
            ok = code == 42 and out == want
            all_ok = all_ok and ok
            got_sum = struct.unpack("<q", out)[0] if len(out) == 8 else None
            print(
                f"  {'PASS' if ok else 'FAIL'}  maxrowmax({m}x{n}) -> exit={code} "
                f"(want 42) stdout_sum={got_sum} expected_sum={expected_sum} "
                f"wrong[min={v_min} last={v_last} first={v_first}] (elf {len(elf)}B, "
                f"seed nest + row-max nest via cmp+cmovl over ((i*n)+j)*8+base, "
                f"position-weighted fold, native x86-64, zero MLIR/LLVM)"
            )

        # The interior-argmax property MUST have been exercised (proves max != last
        # somewhere) — a vacuous suite that only ever hit max==last would be a
        # weak test. Fail loud if no wrapping shape was checked.
        if not proved_interior_argmax:
            print("  FAIL  no wrapping shape exercised — interior-argmax unproven")
            all_ok = False

        # Shape guard: out-of-frame and degenerate shapes must FAIL CLOSED (empty
        # buffer). The last two are i64-overflow shapes: m*n wraps to 0 (mod 2^64),
        # so the `m*n > 4096` product check alone would pass them — the per-dim
        # `> 4096` bound (applied before the product) is what refuses them.
        for (m, n) in [
            (65, 64), (64, 65), (1, 4097), (4097, 1), (0, 1), (1, 0), (1, -3), (-3, 1),
            (4294967296, 4294967296),
            (8589934592, 8589934592),
        ]:
            elf = mind_maxrowmax_elf(lib, m, n)
            ok = len(elf) == 0
            all_ok = all_ok and ok
            print(
                f"  {'PASS' if ok else 'FAIL'}  maxrowmax({m}x{n}) refused "
                f"(fail-closed shape guard, got {len(elf)}B want 0B)"
            )
    if all_ok:
        print(
            "ALL PASS  tensor row max-reduction lowers native-ELF end to end — a "
            "self-seeded 2-D row-major i64 operand, an emitted seed nest + a "
            "row-max nest over ((i*n)+j)*8+base addressing tracked by a real "
            "cmp+cmovl running-max select, a POSITION-WEIGHTED (discriminating) "
            "row fold, full-width stdout check + exact-i64 in-ELF comparison "
            "(movabs-baked), fail-closed frame-bound guard, interior-argmax proven "
            "(max != last on wrapping shapes) and wrong reductions (min/last/first) "
            "proven distinct, zero MLIR/LLVM (C4-T6)"
        )
        return 0
    print("FAIL  native-ELF tensor maxrowmax gate")
    return 1


if __name__ == "__main__":
    sys.exit(main())
