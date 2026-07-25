#!/usr/bin/env python3
"""C4-T4 — native-ELF float64 TENSOR element-wise-add + STRICT-SEQUENTIAL reduce.

The FIRST float-typed native-ELF tensor kernel (every prior tensor selftest is
_i64). `selftest_native_elf_tensor_ewadd_f64(a, b, n, expected_bits)` emits a
runnable x86-64 ET_EXEC that (1) bakes two length-n f64 fixtures a[]/b[] into the
entry frame as their raw IEEE-754 bit patterns, (2) computes c[i] = a[i] + b[i]
via an emitted counted loop over base+i*8 addressing using the C1 SSE scalar-f64
primitives (movsd/addsd), (3) reduces sum = ((0.0 + c[0]) + c[1]) + ... via a
SECOND emitted loop in STRICT LEFT-TO-RIGHT order (no reassociation, no tree, no
SIMD hadd), (4) writes the 8 IEEE-754 LE bytes of `sum` to stdout, (5) exits with
(sum_bits == expected_bits)*41 + 1 — 42 only on an EXACT 64-bit match.

Two independent full-width gates per case:
  (a) stdout == struct.pack('<d', sequential_reference)  — the ELF's runtime
      reduction checked against a Python f64 reference folded in the SAME order;
  (b) exit == 42 — the in-ELF full 64-bit comparison against expected_bits.

STRICT-ORDER PROOF: at least one multi-element case is crafted so the strict
LEFT-TO-RIGHT fold and a REASSOCIATED fold (right-to-left / pairwise) disagree in
the low mantissa bits. The smoke asserts (i) those two references DIFFER, and
(ii) the emitted ELF matches the SEQUENTIAL one — so a reassociating emitter would
FAIL this case. This holds the strict-FP determinism tier (the f64 wedge).

No frozen byte oracle exists for this path (the deleted Rust native backend
rejected ConstF64); the gate is EXECUTION CORRECTNESS + strict-FP determinism.

Usage:
  MINDC_SO=/path/to.so python3 self_host_native_tensor_ewadd_f64_smoke.py
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


def _bits(x: float) -> int:
    return struct.unpack("<q", struct.pack("<d", x))[0]


def seq_sum(cs: list[float]) -> float:
    """Strict left-to-right f64 fold: ((0.0 + c0) + c1) + ...  (no reassociation)."""
    s = 0.0
    for c in cs:
        s = s + c
    return s


def reassoc_sum(cs: list[float]) -> float:
    """A DIFFERENT association order (right-to-left) used only to prove the ELF is
    strictly left-to-right — never the reference the ELF is checked against."""
    s = 0.0
    for c in reversed(cs):
        s = s + c
    return s


def mind_ewadd_f64_elf(lib, a: list[float], b: list[float], expected_bits: int) -> bytes:
    assert len(a) == len(b)
    n = len(a)
    a_arr = (ctypes.c_double * n)(*a)
    b_arr = (ctypes.c_double * n)(*b)
    fn = lib.selftest_native_elf_tensor_ewadd_f64
    fn.restype = ctypes.c_int64
    fn.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
    es = fn(
        ctypes.cast(a_arr, ctypes.c_void_p).value,
        ctypes.cast(b_arr, ctypes.c_void_p).value,
        n,
        expected_bits,
    )
    rd = lambda addr, o=0: ctypes.cast(addr + o, ctypes.POINTER(ctypes.c_int64))[0]
    sh = rd(es, 0)  # buf: String handle (addr/len/cap)
    return ctypes.string_at(rd(sh, 0), rd(sh, 8))


def run_elf(elf: bytes, tmp: pathlib.Path):
    p = tmp / "mind_tensor_ewadd_f64.elf"
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
    if not hasattr(lib, "selftest_native_elf_tensor_ewadd_f64"):
        print("FAIL  selftest_native_elf_tensor_ewadd_f64: symbol absent (C4-T4 not built)")
        return 1

    # (a, b, strict_order_matters?). The last two cases are crafted so that a
    # LEFT-TO-RIGHT fold and a REASSOCIATED fold differ in the low mantissa bits —
    # classic catastrophic-cancellation orderings after the elementwise add.
    cases = [
        ([1.0, 2.0, 3.0], [0.5, 0.25, 0.125], False),          # exact small
        ([10.5], [0.25], False),                                # length-1
        ([1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0], False),   # uniform
        # c = a+b = [1e16, 1.0, -1e16, 1.0]; left-to-right -> ...+1 = 1.0, a
        # right-to-left fold -> ...+1e16 = 0.0. Asymmetric: the orders disagree.
        ([1e16, 1.0, -1e16, 1.0], [0.0, 0.0, 0.0, 0.0], True),
        # c = [1.0, 1e16, -1e16, 1.0, 1e-8]; order changes the low bits.
        ([1.0, 1e16, -1e16, 1.0, 1e-8], [0.0, 0.0, 0.0, 0.0, 0.0], True),
    ]
    all_ok = True
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        for a, b, strict in cases:
            cs = [a[i] + b[i] for i in range(len(a))]
            ref = seq_sum(cs)
            ref_bits = _bits(ref)
            if strict:
                other = reassoc_sum(cs)
                if _bits(other) == ref_bits:
                    print(
                        f"  FAIL  case a={a}: strict-order case does NOT actually "
                        f"discriminate (seq==reassoc bit-for-bit) — bad fixture"
                    )
                    all_ok = False
                    continue
            elf = mind_ewadd_f64_elf(lib, a, b, ref_bits)
            if not (len(elf) > 120 and elf[:4] == b"\x7fELF"):
                print(f"  FAIL  ewadd_f64(a={a}): not a runnable ELF (len={len(elf)})")
                all_ok = False
                continue
            code, out = run_elf(elf, tmp)
            want = struct.pack("<d", ref)
            ok = code == 42 and out == want
            all_ok = all_ok and ok
            got = struct.unpack("<d", out)[0] if len(out) == 8 else None
            extra = ""
            if strict:
                extra = (
                    f" [strict-order: seq={ref!r} reassoc={reassoc_sum(cs)!r} "
                    f"DIFFER -> a reassociating emitter would fail this]"
                )
            print(
                f"  {'PASS' if ok else 'FAIL'}  ewadd_f64(n={len(a)}) -> exit={code} "
                f"(want 42) stdout_sum={got!r} ref_seq={ref!r} "
                f"(elf {len(elf)}B, 2 counted loops, addsd, native x86-64, "
                f"zero MLIR/LLVM){extra}"
            )
    if all_ok:
        print(
            "ALL PASS  float64 tensor element-wise add + STRICT-SEQUENTIAL reduction "
            "lowers native-ELF end to end — two baked f64 buffers, an emitted "
            "c[i]=a[i]+b[i] addsd loop over base+i*8, a strict left-to-right "
            "reduction loop (no reassociation), 8-byte IEEE-754 stdout check + "
            "exact-64-bit in-ELF exit gate, strict-order-vs-reassoc discrimination, "
            "zero MLIR/LLVM (C4-T4 first float-typed native-ELF tensor kernel)"
        )
        return 0
    print("FAIL  native-ELF float64 tensor ewadd gate")
    return 1


if __name__ == "__main__":
    sys.exit(main())
