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
    # Call body: the result is still a const.i64-0 placeholder (mic@1 surfaces
    # no real call op), but the ARGUMENTS now lower with param resolution.
    # g(a): arg `a` resolves to %0 (no op) -> call placeholder %1 -> output %1.
    (b"pub fn f(a: i64) -> i64 { g(a) }\n",
     b"  %1 = const.i64 0\n  output %1\n"),
    # g(a + 1): arg binop lowers (%0 param, %1 const 1, %2 add) -> placeholder %3.
    (b"pub fn f(a: i64) -> i64 { g(a + 1) }\n",
     b"  %1 = const.i64 1\n  %2 = add %0, %1\n  %3 = const.i64 0\n  output %3\n"),
    # let-binding SSA resolution: `let s: i64 = a + 1; s`. a=%0 (param),
    # 1=%1 (const), add=%2; `s` binds to %2; the trailing `s` resolves to
    # %2 (no op); output %2.  (The bootstrap parser requires the `: i64`
    # type annotation on let — parse_let reads ty at pos+3, init at pos+5.)
    (b"pub fn f(a: i64) -> i64 { let s: i64 = a + 1; s }\n",
     b"  %1 = const.i64 1\n  %2 = add %0, %1\n  output %2\n"),
    # nested let: `let x: i64 = 5; let y: i64 = x + x; y`. 5=%0 (const);
    # x binds %0; x + x -> %1 = add %0, %0; y binds %1; trailing y resolves
    # %1 (no op); output %1.
    (b"pub fn g() -> i64 { let x: i64 = 5; let y: i64 = x + x; y }\n",
     b"  %0 = const.i64 5\n  %1 = add %0, %0\n  output %1\n"),
    # Arithmetic + comparison operators: all are wired through the tokenizer
    # (single-char +,-,*,/,<,>), parse_pratt (tk_to_op + infix_prec), and
    # emit_op_mnemonic (add/sub/mul/div/lt/gt). Previously only `add` was
    # covered here; lock the rest so a regression in any operator mnemonic is
    # caught. (`==`/`<=`/`>=`/`!=` are NOT yet supported — they need multi-char
    # lexer tokens; `==` currently mis-lexes as two `=`.)
    (b"pub fn f(a: i64, b: i64) -> i64 { a - b }\n",
     b"  %2 = sub %0, %1\n  output %2\n"),
    (b"pub fn f(a: i64, b: i64) -> i64 { a * b }\n",
     b"  %2 = mul %0, %1\n  output %2\n"),
    (b"pub fn f(a: i64, b: i64) -> i64 { a / b }\n",
     b"  %2 = div %0, %1\n  output %2\n"),
    (b"pub fn f(a: i64, b: i64) -> i64 { a < b }\n",
     b"  %2 = lt %0, %1\n  output %2\n"),
    (b"pub fn f(a: i64, b: i64) -> i64 { a > b }\n",
     b"  %2 = gt %0, %1\n  output %2\n"),
    # Parens + nesting + precedence: (a + b) * a -> add then mul over the result.
    (b"pub fn f(a: i64, b: i64) -> i64 { (a + b) * a }\n",
     b"  %2 = add %0, %1\n  %3 = mul %2, %0\n  output %3\n"),
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
