"""
Self-host STRUCT-FIELD smoke — proves the pure-MIND compiler PARSES the
`name: type` field pairs of a struct and RETAINS the ordered field-name spans
in a heap table (field-access groundwork, part 1).

This exercises the additive struct field-name retention path in main.mind
(selftest_struct_field_count / selftest_struct_field_name_{lo,hi,len} exports).
It does NO mic@3 byte-output and is fully isolated from the canary
(mindc_compile -> lower_program -> emit_fn_def stays stubbed; parse_struct_def
still discards field names), so fixed_point_smoke.py stays byte-identical.

These are self-consistency unit goldens (field counts + name lengths verified
against the source bytes), not a byte-identity test.

Run:  python3 examples/mindc_mind/struct_fields_smoke.py
"""

import ctypes
import os
import pathlib
import sys

_DEFAULT_SO = pathlib.Path(__file__).parent / "libmindc_mind.so"
SO = pathlib.Path(os.environ.get("MINDC_SO", str(_DEFAULT_SO)))

# Each case: (source, expected_field_count, [expected field names in order]).
# The field-name length golden is derived from each name string; the absolute
# span offsets are checked to slice back to exactly the field name in the source.
CASES = [
    (b"struct S { a: i64, b: i64 }\n", 2, [b"a", b"b"]),
    (b"struct P { x: i64, y: i64, z: i64 }\n", 3, [b"x", b"y", b"z"]),
    (b"struct E { }\n", 0, []),
    (b"struct Empty {}\n", 0, []),
    # Trailing comma after the last field.
    (b"struct T { one: i64, two: i64, }\n", 2, [b"one", b"two"]),
    # Single field, no trailing comma.
    (b"struct One { only: i64 }\n", 1, [b"only"]),
    # Longer field names + a struct whose name differs from any field.
    (b"struct Node { value: i64, next_id: i64 }\n", 2, [b"value", b"next_id"]),
    # No struct at all -> zero fields (the build-table returns an empty table).
    (b"pub fn f(a: i64) -> i64 { a }\n", 0, []),
]


def main() -> int:
    if not SO.exists():
        if os.environ.get("MINDC_SO"):
            print(f"ERROR: {SO} not found (MINDC_SO is set — refusing to skip)")
            return 1
        print(f"SKIP: {SO} not built")
        return 0

    lib = ctypes.CDLL(str(SO))
    for fn in ("selftest_struct_field_count",
               "selftest_struct_field_name_lo",
               "selftest_struct_field_name_hi",
               "selftest_struct_field_name_len"):
        f = getattr(lib, fn)
        f.restype = ctypes.c_int64
    lib.selftest_struct_field_count.argtypes = [ctypes.c_int64, ctypes.c_int64]
    for fn in ("selftest_struct_field_name_lo",
               "selftest_struct_field_name_hi",
               "selftest_struct_field_name_len"):
        getattr(lib, fn).argtypes = [
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]

    failures = 0
    for src, want_count, want_names in CASES:
        buf = ctypes.create_string_buffer(src, len(src))
        addr = ctypes.cast(buf, ctypes.c_void_p).value
        n = lib.selftest_struct_field_count(addr, len(src))
        ok = n == want_count
        detail = ""
        if ok:
            for i, want_name in enumerate(want_names):
                lo = lib.selftest_struct_field_name_lo(addr, len(src), i)
                hi = lib.selftest_struct_field_name_hi(addr, len(src), i)
                ln = lib.selftest_struct_field_name_len(addr, len(src), i)
                got_name = src[lo:hi]
                if got_name != want_name or ln != len(want_name):
                    ok = False
                    detail = (f"  field[{i}] = {got_name!r} (len {ln}), "
                              f"expected {want_name!r}")
                    break
            # Out-of-range probe returns the -1 sentinel.
            if ok and lib.selftest_struct_field_name_lo(
                    addr, len(src), want_count) != -1:
                ok = False
                detail = "  out-of-range index did not return -1 sentinel"
        else:
            detail = f"  count {n}, expected {want_count}"
        print(f"  {'PASS' if ok else 'FAIL'}  "
              f"{src.strip().decode():<40} -> count={n}{detail}")
        if not ok:
            failures += 1

    print(f"\n{'ALL PASS' if failures == 0 else f'{failures} FAILED'}  "
          f"({len(CASES)} cases)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
