#!/usr/bin/env python3
"""C2 — native-ELF NARROW-INT WRAP ARITHMETIC batch: {sub,mul}xi8 + {add,mul}xi16.

Extends the i8-add rung (self_host_native_narrow_add_i8_smoke.py) across the op x
width matrix. Each `selftest_native_elf_narrow_<op>_i<w>(a, b)` export emits a
runnable x86-64 ET_EXEC that:

  (1) bakes host i64 operands a, b full-width via movabs,
  (2) narrows each to width w at RUNTIME with nb_wrap_rax_w(_, w)
      (movsx rax,al / movsx rax,ax — mask low w bits, re-sign top bit),
  (3) applies the op (sub/mul/add) in a full 64-bit register, then
  (4) narrows the RESULT back to width w with the SAME primitive — so an
      overflowing narrow op wraps two's-complement (i8: 100*2 -> -56; i16:
      40000-1 -> the wrapped i16) instead of i64-widening,
  (5) writes the 8 LE bytes of the sign-extended result to stdout, and
  (6) exits (r == baked_expected)*41 + 1 — 42 only on an EXACT full-width i64
      match against the emit-time-baked expected (movabs-baked past imm32).

Two independent full-width gates per case, exactly as the i8-add rung:
  (a) stdout == struct.pack('<q', ref_wrap(ref_wrap(a) OP ref_wrap(b), w)) where
      ref_wrap is THIS script's independent pure-Python width-w two's-complement
      wrap (ctypes.c_int8/c_int16, no shared code with the MIND emitter/host);
  (b) exit == 42 — the in-ELF exact-i64 comparison against the baked expected.

Non-vacuous: every driver's corpus is asserted to contain >=1 genuine WRAP case
(result != the non-wrapped i64 value) and >=1 non-wrap case, so an ELF matching
the wrapped reference provably could not have come from a plain (non-wrapping) op.

FAIL-CLOSED DOMAIN GUARD: the shared driver refuses (empty buffer) unless both
operands are in [-1000000, 1000000]; asserted on out-of-domain operands.

ADDITIVITY: NEW exports never reached during self-compile, so the integer
native-ELF oracle stays byte-identical. The gate is EXECUTION CORRECTNESS.

Usage:
  MINDC_SO=/path/to.so python3 self_host_native_narrow_arith_batch_smoke.py
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

MARKER = 42


def elf_of(lib, sym: str, a: int, b: int) -> bytes:
    fn = getattr(lib, sym)
    fn.restype = ctypes.c_int64
    fn.argtypes = [ctypes.c_int64, ctypes.c_int64]
    es = fn(a, b)
    rd = lambda addr, o=0: ctypes.cast(addr + o, ctypes.POINTER(ctypes.c_int64))[0]
    sh = rd(es, 0)  # buf: String handle (addr/len/cap)
    ln = rd(sh, 8)
    if ln <= 0:
        return b""
    return ctypes.string_at(rd(sh, 0), ln)


def ref_wrap(x: int, w: int) -> int:
    """Independent pure-Python width-w two's-complement wrap: (x as iW) as i64.

    Uses the platform's own signed truncation via ctypes — no shared logic with
    the emitter or the MIND host helper."""
    if w == 8:
        return ctypes.c_int8(x & 0xFF).value
    if w == 16:
        return ctypes.c_int16(x & 0xFFFF).value
    return ctypes.c_int32(x & 0xFFFFFFFF).value


def apply_op(op: str, x: int, y: int) -> int:
    if op == "sub":
        return x - y
    if op == "mul":
        return x * y
    return x + y


def ref_expected(op: str, a: int, b: int, w: int) -> int:
    """Correct WRAPPED result: narrow both operands, op, narrow the result."""
    return ref_wrap(apply_op(op, ref_wrap(a, w), ref_wrap(b, w)), w)


def ref_nonwrapped(op: str, a: int, b: int, w: int) -> int:
    """The WRONG (non-wrapping) value — narrow operands but DO NOT wrap the
    result. Used only to prove the final iW wrap is what the ELF applied."""
    return apply_op(op, ref_wrap(a, w), ref_wrap(b, w))


def run_elf(elf: bytes, tmp: pathlib.Path, tag: str):
    p = tmp / f"mind_narrow_{tag}.elf"
    p.write_bytes(elf)
    p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    r = subprocess.run([str(p)], capture_output=True)
    return r.returncode, r.stdout


# (symbol, op, width, corpus). Every corpus mixes wrapping + non-wrapping cases.
DRIVERS = [
    ("selftest_native_elf_narrow_sub_i8", "sub", 8, [
        (100, -100), (-128, 1), (127, -1), (3, 4), (-5, 2),
        (200, 100), (0, 0), (-128, 0),
    ]),
    ("selftest_native_elf_narrow_mul_i8", "mul", 8, [
        (100, 2), (16, 8), (-16, 8), (3, 4), (2, 5),
        (127, 127), (11, 12), (1, 1),
    ]),
    ("selftest_native_elf_narrow_add_i16", "add", 16, [
        (1000000, 1000000), (32767, 1), (-32768, -1), (100, 200),
        (30000, 30000), (5, 7), (32767, 0), (-32768, 0),
    ]),
    ("selftest_native_elf_narrow_mul_i16", "mul", 16, [
        (1000, 1000), (256, 256), (-256, 256), (7, 8), (100, 3),
        (32767, 32767), (200, 200), (2, 3),
    ]),
]


def main() -> int:
    so = os.environ.get("MINDC_SO", str(_DEFAULT_SO))
    if not os.path.exists(so):
        if os.environ.get("MINDC_SO"):
            print(f"FAIL  MINDC_SO set but missing: {so!r}")
            return 1
        print(f"SKIP  {so} not built")
        return 0
    lib = ctypes.CDLL(so)
    all_ok = True
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        for sym, op, w, cases in DRIVERS:
            if not hasattr(lib, sym):
                print(f"FAIL  {sym}: symbol absent (C2 batch not built)")
                all_ok = False
                continue
            n_wrap = 0
            n_nowrap = 0
            drv_ok = True
            for a, b in cases:
                expected = ref_expected(op, a, b, w)
                nonwrapped = ref_nonwrapped(op, a, b, w)
                wraps = expected != nonwrapped
                if wraps:
                    n_wrap += 1
                else:
                    n_nowrap += 1
                elf = elf_of(lib, sym, a, b)
                if not (len(elf) > 120 and elf[:4] == b"\x7fELF"):
                    print(f"  FAIL  {sym}(a={a},b={b}): not a runnable ELF (len={len(elf)})")
                    drv_ok = False
                    continue
                code, out = run_elf(elf, tmp, f"{op}_{w}")
                want = struct.pack("<q", expected)
                ok = code == MARKER and out == want
                drv_ok = drv_ok and ok
                got = struct.unpack("<q", out)[0] if len(out) == 8 else None
                tag = "wrap  " if wraps else "no-wrap"
                print(
                    f"  {'PASS' if ok else 'FAIL'}  {op}_i{w}(a={a},b={b}) [{tag}] -> "
                    f"exit={code} (want {MARKER}) stdout_iW={got} expected={expected} "
                    f"nonwrapped_i64={nonwrapped} (elf {len(elf)}B, native x86-64, "
                    f"zero MLIR/LLVM)"
                )
            if n_wrap < 1 or n_nowrap < 1:
                print(f"  FAIL  {sym}: vacuous corpus n_wrap={n_wrap} n_nowrap={n_nowrap}")
                drv_ok = False
            else:
                print(f"  PASS  {sym}: non-vacuous ({n_wrap} wrap + {n_nowrap} no-wrap)")
            all_ok = all_ok and drv_ok

        # Fail-closed domain guard (shared driver): out-of-domain operands refused.
        for a, b in [
            (1000001, 0), (0, 1000001), (-1000001, 0), (0, -1000001),
            (2147483648, 1), (9999999999, 0),
        ]:
            elf = elf_of(lib, "selftest_native_elf_narrow_sub_i8", a, b)
            ok = len(elf) == 0
            all_ok = all_ok and ok
            print(
                f"  {'PASS' if ok else 'FAIL'}  sub_i8(a={a},b={b}) refused "
                f"(fail-closed domain guard, got {len(elf)}B want 0B)"
            )

    if all_ok:
        print(
            "ALL PASS  narrow-int wrap arithmetic {sub,mul}xi8 + {add,mul}xi16 lowers "
            "native-ELF end to end — baked operands narrowed via movsx al/ax, op "
            "applied in a full 64-bit register, the RESULT narrowed again so iW "
            "overflow wraps two's-complement instead of i64-widening; full-width "
            "stdout check against an independent width-w wrap reference + exact-i64 "
            "in-ELF comparison (movabs-baked past imm32), non-wrapping value proven "
            "distinct on every wrap case, fail-closed domain guard, zero MLIR/LLVM "
            "(roadmap C2 narrow-int wrap tier)"
        )
        return 0
    print("FAIL  native-ELF narrow-int wrap-arithmetic batch gate")
    return 1


if __name__ == "__main__":
    sys.exit(main())
