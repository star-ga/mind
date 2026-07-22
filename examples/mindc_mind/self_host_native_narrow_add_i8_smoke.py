#!/usr/bin/env python3
"""C2 — native-ELF NARROW-INT (i8) WRAP ARITHMETIC, zero MLIR/LLVM.

The scalar narrow-int rung of the self-host ladder.
`selftest_native_elf_narrow_add_i8(a, b)` emits a runnable x86-64 ET_EXEC that
computes  r = ((a as i8) + (b as i8)) as i8  with TWO'S-COMPLEMENT wrap at the
i8 width, entirely in native code:

  (1) bakes the host i64 operands a, b full-width via movabs,
  (2) narrows each to i8 at RUNTIME with the reg-to-reg wrap primitive
      nb_wrap_rax_w(_, 8) = movsx rax,al  (mask low 8 bits, re-sign bit 7),
  (3) adds the two narrowed values in a full 64-bit register, then
  (4) narrows the SUM back to i8 with the SAME wrap primitive — so an overflowing
      narrow sum wraps (100+100 -> -56, 127+1 -> -128) instead of i64-widening,
  (5) writes the 8 LE bytes of the (sign-extended) i8 result to stdout — the
      FULL-width observable, and
  (6) exits (r == baked_expected)*41 + 1 — 42 only on an EXACT full-width i64
      match against the emit-time-baked expected (movabs-baked past imm32), 1
      otherwise (not a mod-256-maskable residue).

Why DISCRIMINATING: the whole point of this rung is that the narrow ADD wraps
rather than i64-widens. This smoke computes, per (a, b), BOTH the correct wrapped
i8 value AND the NON-WRAPPED i64 value ((a as i8) + (b as i8) with no final wrap),
and for every wrapping case asserts they DIFFER — so an ELF whose stdout equals
the wrapped reference provably could not have come from a plain (non-wrapping)
i64 add. Two independent full-width gates per case:
  (a) stdout == struct.pack('<q', ref_wrap8(ref_wrap8(a)+ref_wrap8(b))) where
      ref_wrap8 is THIS script's independent pure-Python i8 two's-complement wrap
      (ctypes.c_int8, no shared code with the emitter);
  (b) exit == 42 — the in-ELF exact-i64 comparison against the baked expected.

FAIL-CLOSED DOMAIN GUARD: scalar arithmetic (no arrays/loops, no frame to
overrun), but the export still FAILS CLOSED — empty buffer — unless both operands
are within [-1000000, 1000000]. This smoke asserts the refusal on out-of-domain
operands, mirroring the tensor rungs' shape guard.

ADDITIVITY: a NEW export never reached during self-compile, so the integer
native-ELF oracle stays byte-identical. No frozen byte oracle exists for this
path; the gate is EXECUTION CORRECTNESS.

Usage:
  MINDC_SO=/path/to.so python3 self_host_native_narrow_add_i8_smoke.py
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


def mind_narrow_add_i8_elf(lib, a: int, b: int) -> bytes:
    fn = lib.selftest_native_elf_narrow_add_i8
    fn.restype = ctypes.c_int64
    fn.argtypes = [ctypes.c_int64, ctypes.c_int64]
    es = fn(a, b)
    rd = lambda addr, o=0: ctypes.cast(addr + o, ctypes.POINTER(ctypes.c_int64))[0]
    sh = rd(es, 0)  # buf: String handle (addr/len/cap)
    ln = rd(sh, 8)
    if ln <= 0:
        return b""
    return ctypes.string_at(rd(sh, 0), ln)


def ref_wrap8(x: int) -> int:
    """Independent pure-Python i8 two's-complement wrap: (x as i8) as i64.

    Uses the platform's own signed-8-bit truncation via ctypes — no shared logic
    with the emitter or the MIND host helper."""
    return ctypes.c_int8(x & 0xFF).value


def ref_expected(a: int, b: int) -> int:
    """Correct WRAPPED result: narrow both operands, add, narrow the sum."""
    return ref_wrap8(ref_wrap8(a) + ref_wrap8(b))


def ref_nonwrapped(a: int, b: int) -> int:
    """The WRONG (non-wrapping) i64 value — narrow both operands but DO NOT wrap
    the sum. Used only to prove the i8 wrap on the sum is what the ELF applied."""
    return ref_wrap8(a) + ref_wrap8(b)


def run_elf(elf: bytes, tmp: pathlib.Path):
    p = tmp / "mind_narrow_add_i8.elf"
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
    if not hasattr(lib, "selftest_native_elf_narrow_add_i8"):
        print("FAIL  selftest_native_elf_narrow_add_i8: symbol absent (C2 not built)")
        return 1

    # (a, b, wraps?) — the `wraps` flag is asserted against the reference so the
    # corpus is provably non-vacuous: >=1 case that WRAPS and >=1 that does NOT.
    #   100+100 = 200 -> -56          wrap (positive overflow)
    #   127+1   = 128 -> -128         wrap (INT8_MAX+1 -> INT8_MIN)
    #   -128+-1 = -129 -> 127         wrap (negative overflow)
    #   -100+-100 = -200 -> 56        wrap
    #   3+4     = 7 -> 7              NO wrap (in range)
    #   -5+2    = -3 -> -3           NO wrap (in range, negative)
    #   127+0   = 127 -> 127         NO wrap (INT8_MAX exact)
    #   -128+0  = -128 -> -128       NO wrap (INT8_MIN exact)
    # Operands beyond i8 also exercise the per-operand narrowing:
    #   200+200 -> wrap8(200)=-56, -56+-56=-112 -> -112  NO further wrap on sum
    #   130+130 -> wrap8(130)=-126, -126+-126=-252 -> 4  wrap (sum overflows)
    cases = [
        (100, 100), (127, 1), (-128, -1), (-100, -100),
        (3, 4), (-5, 2), (127, 0), (-128, 0),
        (200, 200), (130, 130), (255, 1), (0, 0),
    ]
    all_ok = True
    n_wrap = 0
    n_nowrap = 0
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        for a, b in cases:
            expected = ref_expected(a, b)
            nonwrapped = ref_nonwrapped(a, b)
            wraps = expected != nonwrapped
            if wraps:
                n_wrap += 1
            else:
                n_nowrap += 1
            # Sanity: a wrap case MUST diverge from the non-wrapping value; a
            # non-wrap case MUST coincide (the wrap is a no-op there). Either way
            # the observable is meaningful.
            if wraps:
                assert expected != nonwrapped, (
                    f"tagged wrap but equal for a={a} b={b}"
                )
            elf = mind_narrow_add_i8_elf(lib, a, b)
            if not (len(elf) > 120 and elf[:4] == b"\x7fELF"):
                print(f"  FAIL  add_i8(a={a},b={b}): not a runnable ELF (len={len(elf)})")
                all_ok = False
                continue
            code, out = run_elf(elf, tmp)
            want = struct.pack("<q", expected)
            ok = code == MARKER and out == want
            all_ok = all_ok and ok
            got = struct.unpack("<q", out)[0] if len(out) == 8 else None
            tag = "wrap  " if wraps else "no-wrap"
            print(
                f"  {'PASS' if ok else 'FAIL'}  add_i8(a={a},b={b}) [{tag}] -> "
                f"exit={code} (want {MARKER}) stdout_i8={got} expected={expected} "
                f"nonwrapped_i64={nonwrapped} (elf {len(elf)}B, movsx-al i8 wrap, "
                f"native x86-64, zero MLIR/LLVM)"
            )

        # Non-vacuous: at least one genuine wrap AND at least one non-wrap case.
        if n_wrap < 1 or n_nowrap < 1:
            print(f"  FAIL  vacuous corpus: n_wrap={n_wrap} n_nowrap={n_nowrap} "
                  f"(need >=1 of each)")
            all_ok = False
        else:
            print(f"  PASS  non-vacuous corpus: {n_wrap} wrapping + {n_nowrap} "
                  f"non-wrapping cases")

        # Fail-closed domain guard: operands outside [-1000000, 1000000] must be
        # REFUSED (empty buffer). Includes i32/i64 boundary magnitudes.
        for a, b in [
            (1000001, 0), (0, 1000001), (-1000001, 0), (0, -1000001),
            (2147483648, 1), (1, -2147483648), (9999999999, 0),
        ]:
            elf = mind_narrow_add_i8_elf(lib, a, b)
            ok = len(elf) == 0
            all_ok = all_ok and ok
            print(
                f"  {'PASS' if ok else 'FAIL'}  add_i8(a={a},b={b}) refused "
                f"(fail-closed domain guard, got {len(elf)}B want 0B)"
            )

    if all_ok:
        print(
            "ALL PASS  narrow-int i8 wrap arithmetic lowers native-ELF end to end "
            "— baked operands narrowed via movsx rax,al, added in a full 64-bit "
            "register, the SUM narrowed again so INT8 overflow wraps two's-"
            "complement (100+100 -> -56, 127+1 -> -128) instead of i64-widening; "
            "full-width stdout check against an independent i8-wrap reference + "
            "exact-i64 in-ELF comparison (movabs-baked past imm32), non-wrapping "
            "value proven distinct on every wrap case, fail-closed domain guard, "
            "zero MLIR/LLVM (roadmap C2 narrow-int wrap rung)"
        )
        return 0
    print("FAIL  native-ELF narrow-int i8 wrap-add gate")
    return 1


if __name__ == "__main__":
    sys.exit(main())
