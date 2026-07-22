#!/usr/bin/env python3
"""C4-T5 — native-ELF tensor ELEMENT-WISE MULTIPLY (i64), zero MLIR/LLVM.

The 1-D Hadamard rung of the tensor/linalg ladder.
`selftest_native_elf_tensor_ewmul_i64(n)` emits a runnable x86-64 ET_EXEC that
(1) materializes two length-n i64 buffers in the frame with deterministic
self-seeding a[i]=i+1, b[i]=2*i+3 (both nonzero at every index, including n=1),
(2) computes the element-wise product c[i] = a[i]*b[i] via an emitted counted
loop over base + i*8 addressing, (3) folds the result into a POSITION-WEIGHTED
checksum sum = Sum_{i<n} (i+1)*c[i] via a third emitted loop, (4) writes the 8
LE bytes of `sum` to stdout, and (5) exits (sum == expected)*41 + 1 — 42 only on
an EXACT full-width i64 match against the emit-time-baked expected checksum
(movabs-baked past imm32), 1 otherwise.

Why POSITION-WEIGHTED (index-discriminating): a plain grand-sum of c would be
invariant under an index permutation, and it would not distinguish a wrong fuse
op cleanly at every shape. Weighting each c[i] by (i+1) makes the observable
depend on WHICH index produced WHICH product: a wrong op (add not mul), a
mis-strided read, or a swapped index all yield a different weighted sum. This
smoke proves it by ALSO computing the wrong-op weighted sum (element-wise ADD)
and the plain (unweighted) product sum, and asserting each differs from the
ewmul value on the tested shapes — so the ELF's value provably could not have
come from a wrong op or a position-blind reduction.

Two independent full-width gates per shape:
  (a) stdout == struct.pack('<q', S) where S is THIS script's pure-Python
      seed-multiply-weight over the same seeds — an independent reference (no
      shared code with the emitter);
  (b) exit == 42 — the in-ELF exact-i64 comparison against the baked checksum.

SHAPE GUARD: the export FAILS CLOSED — returns an empty buffer — unless
1 <= n <= 4096 (three 4096-element frame arrays). This smoke asserts the refusal
on out-of-bounds and degenerate shapes too, including i64-overflow shapes.

ADDITIVITY: a NEW export never reached during self-compile, so the integer
native-ELF oracle stays byte-identical. No frozen byte oracle exists for this
path; the gate is EXECUTION CORRECTNESS.

Usage:
  MINDC_SO=/path/to.so python3 self_host_native_tensor_ewmul_smoke.py
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


def mind_ewmul_elf(lib, n: int) -> bytes:
    fn = lib.selftest_native_elf_tensor_ewmul_i64
    fn.restype = ctypes.c_int64
    fn.argtypes = [ctypes.c_int64]
    es = fn(n)
    rd = lambda addr, o=0: ctypes.cast(addr + o, ctypes.POINTER(ctypes.c_int64))[0]
    sh = rd(es, 0)  # buf: String handle (addr/len/cap)
    ln = rd(sh, 8)
    if ln <= 0:
        return b""
    return ctypes.string_at(rd(sh, 0), ln)


def ref_ewmul_wsum(n: int) -> int:
    """Independent pure-Python reference: seed, element-wise MUL, (i+1)-weight fold."""
    a = [i + 1 for i in range(n)]
    b = [2 * i + 3 for i in range(n)]
    c = [a[i] * b[i] for i in range(n)]
    return sum((i + 1) * c[i] for i in range(n))


def ref_ewadd_wsum(n: int) -> int:
    """The WRONG-OP value (element-wise ADD, same weighting) — used only to prove
    the ewmul value is op-discriminating."""
    a = [i + 1 for i in range(n)]
    b = [2 * i + 3 for i in range(n)]
    c = [a[i] + b[i] for i in range(n)]
    return sum((i + 1) * c[i] for i in range(n))


def ref_ewmul_plainsum(n: int) -> int:
    """The POSITION-BLIND value (unweighted product sum) — used only to prove the
    weighting is what makes the observable index-sensitive."""
    a = [i + 1 for i in range(n)]
    b = [2 * i + 3 for i in range(n)]
    return sum(a[i] * b[i] for i in range(n))


def run_elf(elf: bytes, tmp: pathlib.Path):
    p = tmp / "mind_tensor_ewmul.elf"
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
    if not hasattr(lib, "selftest_native_elf_tensor_ewmul_i64"):
        print("FAIL  selftest_native_elf_tensor_ewmul_i64: symbol absent (C4-T5 not built)")
        return 1

    # Distinct lengths: n=1 (single nonzero product, discriminates), small,
    # non-power-of-two, and larger shapes that push the weighted checksum past
    # imm32 (exercises the movabs baking). 4096 is the exact frame cap.
    lengths = [1, 2, 3, 5, 17, 64, 1000, 4096]
    all_ok = True
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        for n in lengths:
            expected_sum = ref_ewmul_wsum(n)
            wrong_op = ref_ewadd_wsum(n)
            plain = ref_ewmul_plainsum(n)
            # Discriminating-by-construction: the ewmul weighted sum must differ
            # from both the wrong-op (add) weighted sum and the position-blind
            # product sum on every tested shape, so a value == expected_sum
            # proves MUL + the (i+1) weighting were both used.
            assert expected_sum != wrong_op, (
                f"non-discriminating (op) shape n={n}: ewmul==ewadd weighted sum"
            )
            # The (i+1) weighting only diverges from the plain product sum once
            # there is more than one index (at n=1 the sole weight is 1, so they
            # coincide — expected, not a bug); assert the weight discriminates for
            # every multi-element shape.
            if n >= 2:
                assert expected_sum != plain, (
                    f"non-discriminating (weight) shape n={n}: weighted==plain product sum"
                )
            elf = mind_ewmul_elf(lib, n)
            if not (len(elf) > 120 and elf[:4] == b"\x7fELF"):
                print(f"  FAIL  ewmul(n={n}): not a runnable ELF (len={len(elf)})")
                all_ok = False
                continue
            code, out = run_elf(elf, tmp)
            want = struct.pack("<q", expected_sum)
            ok = code == 42 and out == want
            all_ok = all_ok and ok
            got_sum = struct.unpack("<q", out)[0] if len(out) == 8 else None
            print(
                f"  {'PASS' if ok else 'FAIL'}  ewmul(n={n}) -> exit={code} "
                f"(want 42) stdout_sum={got_sum} expected_sum={expected_sum} "
                f"wrong_op_sum={wrong_op} plain_sum={plain} (elf {len(elf)}B, "
                f"seed loop + ew-mul loop + position-weighted fold, native "
                f"x86-64, zero MLIR/LLVM)"
            )

        # Shape guard: out-of-frame and degenerate shapes must FAIL CLOSED (empty
        # buffer). The last two are i64-overflow lengths (n itself out of range);
        # the per-dim `> 4096` bound refuses them regardless of any wrap.
        for n in [4097, 0, -1, -4096, 4294967296, 8589934592, -8589934592]:
            elf = mind_ewmul_elf(lib, n)
            ok = len(elf) == 0
            all_ok = all_ok and ok
            print(
                f"  {'PASS' if ok else 'FAIL'}  ewmul(n={n}) refused "
                f"(fail-closed shape guard, got {len(elf)}B want 0B)"
            )
    if all_ok:
        print(
            "ALL PASS  tensor element-wise multiply lowers native-ELF end to end "
            "— two self-seeded 1-D i64 operands, an emitted seed loop + an "
            "ew-mul loop over base+i*8 addressing, a POSITION-WEIGHTED "
            "(index-discriminating) fold, full-width stdout check + exact-i64 "
            "in-ELF comparison (movabs-baked past imm32), fail-closed frame-bound "
            "guard, wrong-op and position-blind values proven distinct, zero "
            "MLIR/LLVM (C4-T5)"
        )
        return 0
    print("FAIL  native-ELF tensor ewmul gate")
    return 1


if __name__ == "__main__":
    sys.exit(main())