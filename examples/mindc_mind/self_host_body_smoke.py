"""
Self-host REAL-BODY smoke — proves the pure-MIND compiler lowers function
bodies with parameter resolution (params bound to their %SSA ids), not the
const.i64-0 placeholder of the frozen mic@1 canary path.

This exercises the additive `emit_fn_def_with_body` path (selftest_emit_fn_body
export) in main.mind, which is isolated from the canary
(mindc_compile -> lower_program -> emit_fn_def stays stubbed, so
fixed_point_smoke.py stays byte-identical). Oracle progression rung 1:
SELF-CONSISTENCY golden strings (per the verified self-host roadmap).

Run:  python3 examples/mindc_mind/self_host_body_smoke.py
"""

import ctypes
import pathlib
import sys

SO = pathlib.Path(__file__).parent / "libmindc_mind.so"

CASES = [
    (b"pub fn add(a: i64, b: i64) -> i64 { a + b }\n",
     b"  %2 = add %0, %1\n  output %2\n"),
    (b"pub fn k() -> i64 { 42 }\n",
     b"  %0 = const.i64 42\n  output %0\n"),
    (b"pub fn id(x: i64) -> i64 { x }\n",
     b"  output %0\n"),
    (b"pub fn three(a: i64, b: i64, c: i64) -> i64 { a + b + c }\n",
     b"  %3 = add %0, %1\n  %4 = add %3, %2\n  output %4\n"),
]


def emit_body(lib, src: bytes) -> bytes:
    buf = ctypes.create_string_buffer(src, len(src))
    es = lib.selftest_emit_fn_body(ctypes.cast(buf, ctypes.c_void_p).value, len(src))
    rd = lambda a, o=0: ctypes.cast(a + o, ctypes.POINTER(ctypes.c_int64))[0]
    sh = rd(es, 0)
    return ctypes.string_at(rd(sh, 0), rd(sh, 8))


def main() -> int:
    if not SO.exists():
        print(f"SKIP: {SO} not built")
        return 0
    lib = ctypes.CDLL(str(SO))
    lib.selftest_emit_fn_body.restype = ctypes.c_int64
    lib.selftest_emit_fn_body.argtypes = [ctypes.c_int64, ctypes.c_int64]
    failures = 0
    for src, want in CASES:
        got = emit_body(lib, src)
        ok = got == want
        print(f"  {'PASS' if ok else 'FAIL'}  {src.strip().decode():<48} -> {got!r}")
        if not ok:
            print(f"        expected {want!r}")
            failures += 1
    print(f"\n{'ALL PASS' if failures == 0 else f'{failures} FAILED'}  "
          f"({len(CASES)} cases)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
