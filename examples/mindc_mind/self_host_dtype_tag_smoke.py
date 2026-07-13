#!/usr/bin/env python3
"""RI-B1 per-SSA dtype-tag gate (parser <-> nb_fp_* encoder connecting construct).

The native-ELF backend is otherwise i64-only. To lower real floats it must know,
per SSA id, whether a value is a 64-bit integer or a scalar f64. This first slice
lands the dtype TAG itself: nb_node_dtype (AST-node -> dtype classifier) + the
SSA-id-keyed dtype table (nb_dtype_table_new/set/get) in main.mind.

`selftest_dtype_tag(src, len)` lexes + parses ONE primary expression, classifies
its SSA value's dtype, round-trips the tag through the table, and returns it:
  `1.5` (a real ast_float_lit from the parser) -> nb_dtype_float() == 1
  `42`  (an ast_int_lit)                        -> nb_dtype_int()   == 0

Pure dtype threading — emits no code — so it cannot perturb any byte-identity
oracle. The nb_expr SSE-selection arm (route FLOAT-tagged values to nb_fp_*) is
the follow-up construct.

Usage:
  MINDC_SO=/path/to.so python3 self_host_dtype_tag_smoke.py
"""
import ctypes
import os
import pathlib
import sys

_HERE = pathlib.Path(__file__).parent
_DEFAULT_SO = _HERE / "libmindc_mind.so"

# nb_dtype_int() / nb_dtype_float() in main.mind.
DT_INT = 0
DT_FLOAT = 1


def dtype_of(lib, src: str) -> int:
    buf = ctypes.create_string_buffer(src.encode(), len(src.encode()))
    return lib.selftest_dtype_tag(
        ctypes.cast(buf, ctypes.c_void_p).value, len(src.encode())
    )


def main() -> int:
    so = os.environ.get("MINDC_SO", str(_DEFAULT_SO))
    if not os.path.exists(so):
        if os.environ.get("MINDC_SO"):
            print(f"FAIL  MINDC_SO set but missing: {so!r}")
            return 1
        print(f"SKIP  {so} not built")
        return 0
    lib = ctypes.CDLL(so)
    if not hasattr(lib, "selftest_dtype_tag"):
        print("FAIL  selftest_dtype_tag: symbol absent (dtype-tag construct not built)")
        return 1
    lib.selftest_dtype_tag.restype = ctypes.c_int64
    lib.selftest_dtype_tag.argtypes = [ctypes.c_int64, ctypes.c_int64]

    # (source, expected dtype tag). The float leaf must classify FLOAT; every
    # integer/other leaf must classify INT (the additive/float-only invariant —
    # main.mind's own integer source must never re-tag to FLOAT).
    cases = [
        ("1.5", DT_FLOAT),
        ("3.14159", DT_FLOAT),
        ("0.0", DT_FLOAT),
        ("42", DT_INT),
        ("0", DT_INT),
    ]
    all_ok = True
    for src, expected in cases:
        got = dtype_of(lib, src)
        ok = got == expected
        all_ok = all_ok and ok
        kind = "FLOAT" if expected == DT_FLOAT else "INT"
        print(
            f"  {'PASS' if ok else 'FAIL'}  {src!r:>10} -> dtype tag {got} "
            f"(expected {expected} = {kind})"
        )
    if all_ok:
        print(
            "ALL PASS  per-SSA dtype tag: ast_float_lit classifies FLOAT, "
            "integer leaves classify INT (additive/float-only) — parser<->encoder "
            "connecting construct landed (nb_expr SSE arm is the follow-up)"
        )
        return 0
    print("FAIL  per-SSA dtype-tag gate")
    return 1


if __name__ == "__main__":
    sys.exit(main())
