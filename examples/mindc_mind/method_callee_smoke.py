"""
Self-host METHOD-CALL RECEIVER-TYPE RESOLUTION smoke (UFCS callee, part 2).

Proves the pure-MIND compiler RESOLVES a method call's receiver to its declared
struct type and computes the UFCS callee name `{lowercase(ReceiverType)}_{method}`
WITHOUT any mic@3 byte-output. It exercises the additive resolution path in
main.mind:

  selftest_method_recv_param_idx   — which fn-param the receiver resolves to
  selftest_method_recv_type_lo/hi  — span of the resolved declared TYPE name
  selftest_method_callee_name_into — writes the callee string into a caller buf

The Rust oracle (src/eval/lower.rs MethodCall arm) desugars `recv.method(args)`
to a plain Instr::Call named `{lowercase(ReceiverType)}_{method}`, receiver
threaded as the first arg. Example: `p: Point`, `p.dist(7)` -> "point_dist".
This part computes that callee string; part 3 emits the OP_CALL with it.

This is fully isolated from the canary (mindc_compile -> lower_program ->
emit_fn_def stays stubbed; no EmitState touched), so fixed_point_smoke.py stays
byte-identical. These are self-consistency unit goldens (callee strings verified
against the source bytes), not a byte-identity test.

Run:  python3 examples/mindc_mind/method_callee_smoke.py
"""

import ctypes
import os
import pathlib
import sys

_DEFAULT_SO = pathlib.Path(__file__).parent / "libmindc_mind.so"
SO = pathlib.Path(os.environ.get("MINDC_SO", str(_DEFAULT_SO)))

# Each case: (source, expected_callee_or_None, expected_param_idx).
# A None callee means resolution must FAIL (returns -1, writes nothing).
CASES = [
    # Receiver param 0 of type Point -> lowercase "point" + "_dist".
    (b"pub fn use_it(p: Point) -> i64 { p.dist(7) }\n", b"point_dist", 0),
    # Single-letter type S, method foo -> "s_foo".
    (b"pub fn g(s: S) -> i64 { s.foo() }\n", b"s_foo", 0),
    # Zero-arg method on a multi-letter type.
    (b"pub fn h(n: Node) -> i64 { n.next_value(k) }\n", b"node_next_value", 0),
    # Receiver is the SECOND param -> resolves to that param's type.
    (b"pub fn use2(k: i64, p: Point) -> i64 { p.area() }\n", b"point_area", 1),
    # Type already lowercase passes through unchanged.
    (b"pub fn r(w: widget) -> i64 { w.tick() }\n", b"widget_tick", 0),
    # All-caps type lowercases fully.
    (b"pub fn r(v: ABC) -> i64 { v.run(1, 2) }\n", b"abc_run", 0),
    # Receiver NOT a known param (no method call -> field access only) -> fail.
    (b"pub fn f(p: Point) -> i64 { p.x }\n", None, -1),
    # No method call at all -> fail.
    (b"pub fn f(a: i64) -> i64 { a + 1 }\n", None, -1),
]

OUTBUF_CAP = 256


def main() -> int:
    if not SO.exists():
        if os.environ.get("MINDC_SO"):
            print(f"ERROR: {SO} not found (MINDC_SO is set — refusing to skip)")
            return 1
        print(f"SKIP: {SO} not built")
        return 0

    lib = ctypes.CDLL(str(SO))
    for fn in ("selftest_method_recv_param_idx",
               "selftest_method_recv_type_lo",
               "selftest_method_recv_type_hi"):
        f = getattr(lib, fn)
        f.restype = ctypes.c_int64
        f.argtypes = [ctypes.c_int64, ctypes.c_int64]

    callee = lib.selftest_method_callee_name_into
    callee.restype = ctypes.c_int64
    callee.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]

    failures = 0
    for idx, (src, want_callee, want_idx) in enumerate(CASES):
        buf = ctypes.create_string_buffer(src, len(src))
        addr = ctypes.cast(buf, ctypes.c_void_p).value
        n = len(src)

        got_idx = lib.selftest_method_recv_param_idx(addr, n)
        if got_idx != want_idx:
            print(f"FAIL case {idx}: recv_param_idx {got_idx} != {want_idx}  src={src!r}")
            failures += 1
            continue

        out = ctypes.create_string_buffer(OUTBUF_CAP)
        out_addr = ctypes.cast(out, ctypes.c_void_p).value
        got_len = callee(addr, n, out_addr)

        if want_callee is None:
            if got_len != -1:
                print(f"FAIL case {idx}: expected fail (-1) got len {got_len}  src={src!r}")
                failures += 1
            else:
                print(f"ok   case {idx}: resolution failed as expected  src={src!r}")
            continue

        if got_len != len(want_callee):
            print(f"FAIL case {idx}: callee len {got_len} != {len(want_callee)}  src={src!r}")
            failures += 1
            continue
        got_callee = out.raw[:got_len]
        if got_callee != want_callee:
            print(f"FAIL case {idx}: callee {got_callee!r} != {want_callee!r}  src={src!r}")
            failures += 1
            continue

        # Cross-check: the resolved TYPE span lowercased + '_' + method == callee.
        ty_lo = lib.selftest_method_recv_type_lo(addr, n)
        ty_hi = lib.selftest_method_recv_type_hi(addr, n)
        ty = src[ty_lo:ty_hi].lower()
        if not got_callee.startswith(ty + b"_"):
            print(f"FAIL case {idx}: callee {got_callee!r} does not start with "
                  f"lowercased type {ty!r}_  src={src!r}")
            failures += 1
            continue

        print(f"ok   case {idx}: callee={got_callee.decode()} "
              f"(param {got_idx}, type {src[ty_lo:ty_hi].decode()})")

    if failures:
        print(f"\n{failures} FAILED")
        return 1
    print(f"\nALL {len(CASES)} method-callee resolution cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
