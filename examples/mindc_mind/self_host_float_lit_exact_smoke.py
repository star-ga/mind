#!/usr/bin/env python3
"""CPU-as-oracle smoke for the C1 float-literal exactness guard.

The pure-MIND decimal->IEEE-754 path (nb_float_lit_bits -> nb_strip_pow5) is exact
ONLY for a dyadic literal — one whose num = ipart*10^fk + fpart is divisible by
5^fk. A non-dyadic literal (0.1, 3.14, 0.7, ...) truncates in nb_strip_pow5's
integer /5 and emits SILENT wrong IEEE-754 bits (0.1 -> 0.0, 3.14 -> 3.0). nb_expr's
ast_float_lit arm now calls nb_float_lit_is_dyadic and FAILS CLOSED on a non-dyadic
literal instead of emitting wrong bits; full correctly-rounded dec2flt is the
deferred upgrade (see nb_float_lit_bits header).

A trunc-based float smoke CANNOT catch this (trunc(3.0) == trunc(3.14) == 3), so
this gates the discriminator DIRECTLY: selftest_float_lit_dyadic byte-for-byte
against the Rust-equivalent rule (ipart*10^fk + fpart) mod 5^fk == 0, over dyadic
literals (must accept) AND non-dyadic literals (must reject). Guarded on >=1 of
each so a check that cannot fail cannot pass.

Env: MINDC_SO (prebuilt .so, skips the build) or MINDC_BIN (default mindc).
Template: self_host_tc_shape_rules_smoke.py (E2023 byte-encoding).
"""
import ctypes
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN_MIND = os.path.join(HERE, "main.mind")

# Dyadic (num divisible by 5^fk -> exact) and non-dyadic (inexact) literals, each
# <= 8 bytes for the byte-buffer export. Mix of fractional lengths and integer
# parts to exercise the 5^fk divisibility across k.
DYADIC = ["0.5", "0.25", "0.125", "0.0625", "1.5", "2.5", "3.125", "12.25", "0.375", "7.5"]
NONDYADIC = ["0.1", "0.2", "0.3", "0.7", "1.1", "3.14", "0.35", "0.123", "9.99", "0.01"]


def oracle_dyadic(lit: str) -> int:
    """1 iff 5^fk divides num = ipart*10^fk + fpart (the exact-representability rule)."""
    ip, fp = (lit.split(".") + [""])[:2]
    fk = len(fp)
    num = int(ip or 0) * (10 ** fk) + int(fp or 0)
    return 1 if num % (5 ** fk) == 0 else 0


def encode(lit: str):
    """(first 8 bytes zero-padded, true length) for selftest_float_lit_dyadic."""
    b = [ord(c) for c in lit[:8]] + [0] * 8
    return tuple(b[:8]) + (len(lit),)


def build_so():
    so = os.environ.get("MINDC_SO")
    if so:
        return so
    mindc = os.environ.get("MINDC_BIN", "mindc")
    out = tempfile.NamedTemporaryFile(suffix=".so", delete=False).name
    cmd = [mindc, MAIN_MIND, "--emit-shared", out]
    print("BUILD:", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("BUILD FAILED rc=", r.returncode)
        print(r.stderr[-3000:])
        sys.exit(1)
    return out


def main() -> int:
    so = build_so()
    st = os.stat(so)
    print(f"SO: {so} ({st.st_size} bytes)")
    if st.st_size < 4096:
        print("FAIL: .so too small (stub?)")
        return 1
    lib = ctypes.CDLL(so)
    if not hasattr(lib, "selftest_float_lit_dyadic"):
        print("FAIL: selftest_float_lit_dyadic absent (float exactness guard not built)")
        return 1
    fn = lib.selftest_float_lit_dyadic
    fn.restype = ctypes.c_int64
    fn.argtypes = [ctypes.c_int64] * 9

    total = pos = fails = 0
    for lit in DYADIC + NONDYADIC:
        got = fn(*encode(lit))
        exp = oracle_dyadic(lit)
        total += 1
        if exp == 1:
            pos += 1
        ok = got == exp
        if not ok:
            fails += 1
        print(
            f"  {'ok ' if ok else 'DIFF'} float_lit {lit:8} dyadic got={got} exp={exp}"
        )
    neg = total - pos
    print(f"exactness: literals={total} dyadic={pos} non-dyadic={neg} fails={fails}")
    if pos < 1:
        print("FAIL: vacuous (no dyadic/accept case)")
        return 1
    if neg < 1:
        print("FAIL: vacuous (no non-dyadic/reject control)")
        return 1
    if fails:
        print("FAIL: pure-MIND dyadic-exactness predicate diverges from the oracle")
        return 1
    print(
        "ALL PASS  nb_float_lit_is_dyadic exactly discriminates dyadic (exact) from "
        "non-dyadic (would emit silent wrong bits) decimal literals — the nb_expr "
        "float arm fails closed on the latter instead of silently miscompiling"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
