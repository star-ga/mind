#!/usr/bin/env python3
"""C4-T6 — native-ELF tensor ELEMENTWISE RELU (max(x,0)), zero MLIR/LLVM.

The 1-D nonlinearity rung of the tensor/linalg ladder.
`selftest_native_elf_tensor_relu_i64(n)` emits a runnable x86-64 ET_EXEC that
(1) materializes ONE length-n i64 buffer in the frame with deterministic
self-seeding a[i] = (i - n//2)*3 — a ramp that STRADDLES ZERO (indices below
n//2 are strictly negative, above n//2 strictly positive), so the rectifier is
exercised on both sides at every non-trivial shape, (2) computes the element-wise
rectifier c[i] = max(a[i], 0) via an emitted counted loop whose core is a signed
`cmp a[i], 0 ; setg` sign predicate multiplied back into the value (positives
pass, non-positives clamp to 0 — a cmp-driven conditional zero, no MLIR select /
no libm), (3) folds the result into a POSITION-WEIGHTED checksum
sum = Sum_{i<n} (i+1)*c[i] via a third emitted loop, (4) writes the 8 LE bytes of
`sum` to stdout, and (5) exits (sum == expected)*41 + 1 — 42 only on an EXACT
full-width i64 match against the emit-time-baked expected checksum (movabs-baked
past imm32), 1 otherwise.

Why POSITION-WEIGHTED (op- and index-discriminating): dropping the rectifier
(a plain copy that keeps the negatives) changes the value because the clamped-away
negative terms would otherwise SUBTRACT from the weighted sum — so a plain-copy
value is provably distinct from the relu value on every shape that contains a
negative element. The (i+1) weight additionally makes the observable
index-sensitive: a plain grand-sum of c would be invariant under an index
permutation or a mis-strided read. This smoke proves both by ALSO computing the
no-relu (plain copy) weighted sum and the position-blind relu sum, and asserting
each differs from the relu weighted sum on the tested shapes — so a value ==
expected_sum could not have come from a missing rectifier or a position-blind
reduction.

Two independent full-width gates per shape:
  (a) stdout == struct.pack('<q', S) where S is THIS script's pure-Python
      seed-relu-weight over the same seeds — an independent reference (no shared
      code with the emitter);
  (b) exit == 42 — the in-ELF exact-i64 comparison against the baked checksum.

SHAPE GUARD: the export FAILS CLOSED — returns an empty buffer — unless
1 <= n <= 4096 (one 4096-element input array + one output array). This smoke
asserts the refusal on out-of-bounds and degenerate shapes too, including
i64-overflow shapes.

ADDITIVITY: a NEW export never reached during self-compile, so the integer
native-ELF oracle stays byte-identical. No frozen byte oracle exists for this
path; the gate is EXECUTION CORRECTNESS.

Usage:
  MINDC_SO=/path/to.so python3 self_host_native_tensor_relu_smoke.py
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


def mind_relu_elf(lib, n: int) -> bytes:
    fn = lib.selftest_native_elf_tensor_relu_i64
    fn.restype = ctypes.c_int64
    fn.argtypes = [ctypes.c_int64]
    es = fn(n)
    rd = lambda addr, o=0: ctypes.cast(addr + o, ctypes.POINTER(ctypes.c_int64))[0]
    sh = rd(es, 0)  # buf: String handle (addr/len/cap)
    ln = rd(sh, 8)
    if ln <= 0:
        return b""
    return ctypes.string_at(rd(sh, 0), ln)


def _seed(n: int):
    """The zero-straddling ramp a[i] = (i - n//2)*3."""
    half = n // 2
    return [(i - half) * 3 for i in range(n)]


def ref_relu_wsum(n: int) -> int:
    """Independent pure-Python reference: seed, RELU (max(x,0)), (i+1)-weighted fold."""
    a = _seed(n)
    c = [x if x > 0 else 0 for x in a]
    return sum((i + 1) * c[i] for i in range(n))


def ref_copy_wsum(n: int) -> int:
    """The WRONG-OP value (NO relu — plain copy, negatives kept), same weighting.
    Used only to prove the rectifier is what shapes the observable."""
    a = _seed(n)
    return sum((i + 1) * a[i] for i in range(n))


def ref_relu_plainsum(n: int) -> int:
    """The POSITION-BLIND value (unweighted relu sum) — used only to prove the
    (i+1) weighting is what makes the observable index-sensitive."""
    a = _seed(n)
    return sum(x if x > 0 else 0 for x in a)


def run_elf(elf: bytes, tmp: pathlib.Path):
    p = tmp / "mind_tensor_relu.elf"
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
    if not hasattr(lib, "selftest_native_elf_tensor_relu_i64"):
        print("FAIL  selftest_native_elf_tensor_relu_i64: symbol absent (C4-T6 not built)")
        return 1

    # Distinct lengths: small even/odd (straddle boundary lands differently),
    # non-power-of-two, and larger shapes that push the weighted checksum past
    # imm32 (exercises the movabs baking). 4096 is the exact frame cap. n=1 is
    # the degenerate case where a[0]=(0-0)*3=0 -> relu(0)=copy(0), so it does NOT
    # discriminate copy-vs-relu (asserted-around below), but still gates exit+stdout.
    lengths = [1, 2, 3, 4, 5, 17, 64, 1000, 4096]
    all_ok = True
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        for n in lengths:
            expected_sum = ref_relu_wsum(n)
            no_relu = ref_copy_wsum(n)
            blind = ref_relu_plainsum(n)
            # Discriminating-by-construction: for every shape that contains a
            # negative element (n >= 2 -> half >= 1 -> a[0] < 0), dropping the
            # relu (plain copy) MUST change the weighted sum. This proves a value
            # == expected_sum came from the rectifier, not a plain copy.
            has_negative = (n // 2) >= 1
            if has_negative:
                assert expected_sum != no_relu, (
                    f"non-discriminating (relu) shape n={n}: relu==plain-copy weighted sum"
                )
            # The (i+1) weighting must diverge from the position-blind relu sum
            # once more than one positive index contributes.
            if n >= 4:
                assert expected_sum != blind, (
                    f"non-discriminating (weight) shape n={n}: weighted==blind relu sum"
                )
            elf = mind_relu_elf(lib, n)
            if not (len(elf) > 120 and elf[:4] == b"\x7fELF"):
                print(f"  FAIL  relu(n={n}): not a runnable ELF (len={len(elf)})")
                all_ok = False
                continue
            code, out = run_elf(elf, tmp)
            want = struct.pack("<q", expected_sum)
            ok = code == 42 and out == want
            all_ok = all_ok and ok
            got_sum = struct.unpack("<q", out)[0] if len(out) == 8 else None
            print(
                f"  {'PASS' if ok else 'FAIL'}  relu(n={n}) -> exit={code} "
                f"(want 42) stdout_sum={got_sum} expected_sum={expected_sum} "
                f"no_relu_sum={no_relu} blind_sum={blind} (elf {len(elf)}B, "
                f"seed loop + relu loop (cmp+setg conditional-zero) + "
                f"position-weighted fold, native x86-64, zero MLIR/LLVM)"
            )

        # Shape guard: out-of-frame and degenerate shapes must FAIL CLOSED (empty
        # buffer). The last three are i64-overflow lengths (n itself out of
        # range); the `> 4096` / `< 1` bounds refuse them regardless of any wrap.
        for n in [4097, 0, -1, -4096, 4294967296, 8589934592, -8589934592]:
            elf = mind_relu_elf(lib, n)
            ok = len(elf) == 0
            all_ok = all_ok and ok
            print(
                f"  {'PASS' if ok else 'FAIL'}  relu(n={n}) refused "
                f"(fail-closed shape guard, got {len(elf)}B want 0B)"
            )
    if all_ok:
        print(
            "ALL PASS  tensor element-wise RELU lowers native-ELF end to end "
            "— one self-seeded zero-straddling 1-D i64 ramp, an emitted seed loop "
            "+ a relu loop (signed cmp+setg conditional-zero over base+i*8 "
            "addressing) + a POSITION-WEIGHTED (index-discriminating) fold, "
            "full-width stdout check + exact-i64 in-ELF comparison (movabs-baked "
            "past imm32), fail-closed frame-bound guard, plain-copy (no-relu) and "
            "position-blind values proven distinct, zero MLIR/LLVM (C4-T6)"
        )
        return 0
    print("FAIL  native-ELF tensor relu gate")
    return 1


if __name__ == "__main__":
    sys.exit(main())
