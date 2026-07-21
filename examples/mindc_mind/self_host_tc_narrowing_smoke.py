"""
Self-host TYPE-CHECKER narrow-int smoke (Rust-independence roadmap Phase B1 slice)
— proves the pure-MIND front-end (examples/mindc_mind/main.mind) models the E2004
implicit-integer-narrowing rule byte-for-byte with the Rust reference type checker
(src/type_checker/mod.rs {int_scalar_bits, is_implicit_narrowing}).

The only oracle-backed narrow-int rule in the Rust checker is E2004: an assignment
`let dst: TO = <value: FROM>` where both TO and FROM are int scalars and TO is
strictly narrower than FROM is an error (silent truncation otherwise). The Rust
checker models exactly two int-scalar widths — i32=32, i64=64 (u32 collapses to
i32; bool/float are NOT int scalars). Widening (i32->i64) and same-width are NOT
flagged. No literal-fits-width or mixed-width result-type rules exist in the Rust
reference (u8/u16/u64/i8/i16 route to opaque Named strings with no width), so they
are intentionally NOT ported — porting them would invent semantics.

This exercises the ADDITIVE `selftest_tc_narrowing(to_sel, from_sel)` export (i32
tag + resolve_type_ident i32 arm + tc_int_scalar_bits + tc_is_implicit_narrowing),
which is ISOLATED from the mic@1 canary / self-host loop (never reached through
mindc_compile). CPU-as-oracle: the expected verdict for every type pair is computed
here from the exact Rust rule and asserted against the pure-MIND result.

Run:  MINDC_SO=<built .so> python3 examples/mindc_mind/self_host_tc_narrowing_smoke.py
"""

import ctypes
import os
import pathlib
import sys

_HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(_HERE))
from _selfhost_so import resolve_so  # noqa: E402

SO = resolve_so()

# Selector -> (type name, int-scalar bit width per the Rust int_scalar_bits;
# None = not an int scalar). This IS the Rust oracle for the widths.
SEL = {
    0: ("i32", 32),
    1: ("i64", 64),
    2: ("f64", None),
    3: ("bool", None),
}


def oracle_narrowing(to_sel: int, from_sel: int) -> int:
    """Rust is_implicit_narrowing: 1 iff both int scalars and TO strictly narrower
    than FROM, else 0."""
    to_bits = SEL[to_sel][1]
    from_bits = SEL[from_sel][1]
    if to_bits is None or from_bits is None:
        return 0
    return 1 if to_bits < from_bits else 0


def main() -> int:
    if not SO.exists():
        if os.environ.get("MINDC_SO"):
            print(f"ERROR: {SO} not found (MINDC_SO is set — refusing to skip)")
            return 1
        print(f"SKIP: {SO} not built")
        return 0

    lib = ctypes.CDLL(str(SO))
    lib.selftest_tc_narrowing.restype = ctypes.c_int64
    lib.selftest_tc_narrowing.argtypes = [ctypes.c_int64, ctypes.c_int64]

    all_ok = True
    saw_positive = False
    for to_sel in SEL:
        for from_sel in SEL:
            got = int(lib.selftest_tc_narrowing(to_sel, from_sel))
            exp = oracle_narrowing(to_sel, from_sel)
            ok = got == exp
            all_ok = all_ok and ok
            if exp == 1:
                saw_positive = True
            to_name = SEL[to_sel][0]
            from_name = SEL[from_sel][0]
            verdict = "NARROW(E2004)" if exp == 1 else "ok"
            print(
                f"  {'PASS' if ok else 'FAIL'}  let dst: {to_name:<4} = <{from_name:<4}>  "
                f"mind={got} rust={exp} [{verdict}]"
            )
            if not ok:
                print(
                    f"        MISMATCH: pure-MIND narrowing verdict diverged from the "
                    f"Rust is_implicit_narrowing oracle for ({to_name}, {from_name})"
                )

    # Guard: the one true positive (i64 -> i32) MUST fire, else the check is vacuous.
    if not saw_positive:
        print("  FAIL  no positive narrowing case exercised — check is vacuous")
        return 1
    if not all_ok:
        print("\nFAIL  pure-MIND E2004 narrowing rule diverged from Rust oracle")
        return 1
    print(
        "\nALL PASS  pure-MIND E2004 implicit-narrowing rule byte-for-byte matches the "
        "Rust type-checker oracle over all {i32,i64,f64,bool} type pairs "
        "(i64->i32 flagged; widening/same-width/non-int NOT flagged)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
