"""
Self-host METHOD-CALL smoke — proves the pure-MIND compiler PARSES a method
call `recv.method(args...)` into a distinct method node, retaining the
method-NAME span, the receiver, and the argument count (UFCS groundwork, part 1).

This exercises the additive method-call parser path in main.mind
(selftest_method_present / selftest_method_name_{lo,hi,len} /
selftest_method_arg_count / selftest_method_recv_kind / selftest_method_recv_lo).
It does NO mic@3 byte-output and is fully isolated from the canary
(mindc_compile -> lower_program -> emit_fn_def stays stubbed; parse_postfix_rest
is untouched), so fixed_point_smoke.py stays byte-identical.

These are self-consistency unit goldens (method-name lengths + arg counts
verified against the source bytes), not a byte-identity test.

The Rust oracle (src/eval/lower.rs MethodCall arm) lowers `recv.method(args)` via
UFCS desugar to a plain Instr::Call named `{lowercase(ReceiverType)}_{method}`
with the receiver threaded as the first arg. Capturing the method name + arg
count is the prerequisite for emitting that call in a later byte-output part.

Run:  python3 examples/mindc_mind/method_calls_smoke.py
"""

import ctypes
import os
import pathlib
import sys

_DEFAULT_SO = pathlib.Path(__file__).parent / "libmindc_mind.so"
SO = pathlib.Path(os.environ.get("MINDC_SO", str(_DEFAULT_SO)))

# ast kind tag for an ident leaf (must match ast_ident() in main.mind).
AST_IDENT = 2

# Each case: (source, present(1/0), method_name, arg_count).
# Sources wrap the method call in a trivial fn so the lexer sees real tokens;
# the parser token-scans for the first `ident . ident (` shape.
CASES = [
    (b"pub fn f(p: Point) -> i64 { p.dist(7) }\n", 1, b"dist", 1),
    (b"pub fn f(p: Point) -> i64 { p.area() }\n", 1, b"area", 0),
    (b"pub fn f(p: Point) -> i64 { p.add(a, b) }\n", 1, b"add", 2),
    (b"pub fn f(v: Vec) -> i64 { v.scaled(x, y, z) }\n", 1, b"scaled", 3),
    (b"pub fn f(p: Point) -> i64 { p.combine(1, 2, 3, 4) }\n", 1, b"combine", 4),
    # Longer method name.
    (b"pub fn f(n: Node) -> i64 { n.next_value(k) }\n", 1, b"next_value", 1),
    # No method call at all -> not present.
    (b"pub fn f(a: i64) -> i64 { a + 1 }\n", 0, None, None),
    # Field access (no parens) is NOT a method call.
    (b"pub fn f(p: Point) -> i64 { p.x }\n", 0, None, None),
]


def _slice(src, lo, hi):
    return src[lo:hi]


def main() -> int:
    if not SO.exists():
        if os.environ.get("MINDC_SO"):
            print(f"ERROR: {SO} not found (MINDC_SO is set — refusing to skip)")
            return 1
        print(f"SKIP: {SO} not built")
        return 0

    lib = ctypes.CDLL(str(SO))
    for fn in ("selftest_method_present",
               "selftest_method_name_lo",
               "selftest_method_name_hi",
               "selftest_method_name_len",
               "selftest_method_arg_count",
               "selftest_method_recv_kind",
               "selftest_method_recv_lo"):
        f = getattr(lib, fn)
        f.restype = ctypes.c_int64
        f.argtypes = [ctypes.c_int64, ctypes.c_int64]

    failures = 0
    for idx, (src, present, name, argc) in enumerate(CASES):
        buf = ctypes.create_string_buffer(src, len(src))
        addr = ctypes.cast(buf, ctypes.c_void_p).value
        n = len(src)

        got_present = lib.selftest_method_present(addr, n)
        if got_present != present:
            print(f"FAIL case {idx}: present {got_present} != {present}  src={src!r}")
            failures += 1
            continue

        if present == 0:
            print(f"ok   case {idx}: no method call (as expected)  src={src!r}")
            continue

        name_lo = lib.selftest_method_name_lo(addr, n)
        name_hi = lib.selftest_method_name_hi(addr, n)
        name_len = lib.selftest_method_name_len(addr, n)
        got_argc = lib.selftest_method_arg_count(addr, n)
        recv_kind = lib.selftest_method_recv_kind(addr, n)
        recv_lo = lib.selftest_method_recv_lo(addr, n)

        got_name = _slice(src, name_lo, name_hi)
        ok = True
        if got_name != name:
            print(f"FAIL case {idx}: method name {got_name!r} != {name!r}")
            ok = False
        if name_len != len(name):
            print(f"FAIL case {idx}: name_len {name_len} != {len(name)}")
            ok = False
        if got_argc != argc:
            print(f"FAIL case {idx}: arg_count {got_argc} != {argc}")
            ok = False
        if recv_kind != AST_IDENT:
            print(f"FAIL case {idx}: recv_kind {recv_kind} != AST_IDENT({AST_IDENT})")
            ok = False
        # The receiver lo must point at the receiver ident in the source.
        if recv_lo < 0 or recv_lo >= len(src):
            print(f"FAIL case {idx}: recv_lo {recv_lo} out of range")
            ok = False

        if ok:
            print(f"ok   case {idx}: {got_name.decode()}(argc={got_argc}) "
                  f"recv@{recv_lo}  src={src!r}")
        else:
            failures += 1

    if failures:
        print(f"\n{failures} FAILED")
        return 1
    print(f"\nALL {len(CASES)} method-call parse cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
