"""
Self-host PARAM-TYPE smoke — proves the pure-MIND compiler PARSES a fn
signature's `name: Type` param list and RETAINS the ordered
(param-name span, declared-type-name span) pairs in a heap table
(field-access groundwork, part 2).

This exercises the additive per-fn param-type table in main.mind
(selftest_param_type_count / selftest_param_type_name_{lo,hi,len} /
selftest_param_name_{lo,hi} exports). It does NO mic@3 byte-output and is fully
isolated from the canary (mindc_compile -> lower_program -> emit_fn_def stays
stubbed; parse_fn_def / parse_params_rest are unchanged), so fixed_point_smoke.py
stays byte-identical.

The (param-name -> declared type-name) binding is the receiver-type info a later
`recv.field` access needs: resolve recv -> its declared struct type (this table)
-> the field index in that struct (part 1's struct field table).

These are self-consistency unit goldens (param counts + type/name spans verified
against the source bytes), not a byte-identity test.

Run:  python3 examples/mindc_mind/param_types_smoke.py
"""

import ctypes
import os
import pathlib
import sys

_DEFAULT_SO = pathlib.Path(__file__).parent / "libmindc_mind.so"
SO = pathlib.Path(os.environ.get("MINDC_SO", str(_DEFAULT_SO)))

# Each case: (source, [(param_name, type_name) in order]).
# The spans are sliced back out of the source and compared exactly, so this
# verifies both the param-name and declared-type-name retention.
CASES = [
    # The motivating receiver-style signature: param `s` declared type `S`.
    (b"pub fn get(s: S) -> i64 { 0 }\n", [(b"s", b"S")]),
    # Two params, mixed builtin + user type.
    (b"pub fn f(a: i64, b: T) -> i64 { 0 }\n", [(b"a", b"i64"), (b"b", b"T")]),
    # Zero-param fn -> empty table.
    (b"pub fn z() -> i64 { 0 }\n", []),
    # Trailing comma after the last param.
    (b"pub fn g(x: i64, y: i64,) -> i64 { 0 }\n",
     [(b"x", b"i64"), (b"y", b"i64")]),
    # Non-pub fn (parse_item dispatches tk_kw_fn directly).
    (b"fn h(p: Point) -> i64 { 0 }\n", [(b"p", b"Point")]),
    # Three params, longer names + a struct-receiver type.
    (b"pub fn m(self_id: Node, k: i64, acc: i64) -> i64 { 0 }\n",
     [(b"self_id", b"Node"), (b"k", b"i64"), (b"acc", b"i64")]),
    # No fn at all -> zero params (empty table).
    (b"struct S { a: i64 }\n", []),
]


def main() -> int:
    if not SO.exists():
        if os.environ.get("MINDC_SO"):
            print(f"ERROR: {SO} not found (MINDC_SO is set — refusing to skip)")
            return 1
        print(f"SKIP: {SO} not built")
        return 0

    lib = ctypes.CDLL(str(SO))
    lib.selftest_param_type_count.restype = ctypes.c_int64
    lib.selftest_param_type_count.argtypes = [ctypes.c_int64, ctypes.c_int64]
    for fn in ("selftest_param_type_name_lo",
               "selftest_param_type_name_hi",
               "selftest_param_type_name_len",
               "selftest_param_name_lo",
               "selftest_param_name_hi"):
        f = getattr(lib, fn)
        f.restype = ctypes.c_int64
        f.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]

    failures = 0
    for src, want_params in CASES:
        buf = ctypes.create_string_buffer(src, len(src))
        addr = ctypes.cast(buf, ctypes.c_void_p).value
        n = lib.selftest_param_type_count(addr, len(src))
        want_count = len(want_params)
        ok = n == want_count
        detail = ""
        if ok:
            for i, (want_name, want_type) in enumerate(want_params):
                nlo = lib.selftest_param_name_lo(addr, len(src), i)
                nhi = lib.selftest_param_name_hi(addr, len(src), i)
                tlo = lib.selftest_param_type_name_lo(addr, len(src), i)
                thi = lib.selftest_param_type_name_hi(addr, len(src), i)
                tln = lib.selftest_param_type_name_len(addr, len(src), i)
                got_name = src[nlo:nhi]
                got_type = src[tlo:thi]
                if got_name != want_name:
                    ok = False
                    detail = (f"  param[{i}] name = {got_name!r}, "
                              f"expected {want_name!r}")
                    break
                if got_type != want_type or tln != len(want_type):
                    ok = False
                    detail = (f"  param[{i}] type = {got_type!r} (len {tln}), "
                              f"expected {want_type!r}")
                    break
            # Out-of-range probe returns the -1 sentinel.
            if ok and lib.selftest_param_type_name_lo(
                    addr, len(src), want_count) != -1:
                ok = False
                detail = "  out-of-range index did not return -1 sentinel"
        else:
            detail = f"  count {n}, expected {want_count}"
        print(f"  {'PASS' if ok else 'FAIL'}  "
              f"{src.strip().decode():<46} -> count={n}{detail}")
        if not ok:
            failures += 1

    print(f"\n{'ALL PASS' if failures == 0 else f'{failures} FAILED'}  "
          f"({len(CASES)} cases)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
