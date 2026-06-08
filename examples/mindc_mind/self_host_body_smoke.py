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
import os
import pathlib
import sys

# Default to the locally-built .so next to this script; CI points MINDC_SO at
# the artifact it builds (e.g. /tmp/libmindc_mind_self_host.so) so the real-body
# emitter is gated in CI, not just locally.
_DEFAULT_SO = pathlib.Path(__file__).parent / "libmindc_mind.so"
SO = pathlib.Path(os.environ.get("MINDC_SO", str(_DEFAULT_SO)))

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
    # (single-char +,-,*,/,<,> and two-char ==,<=,>=,!= via scan-loop
    # peek-ahead), parse_pratt (tk_to_op + infix_prec), and emit_op_mnemonic
    # (add/sub/mul/div/lt/gt/eq/le/ge/ne). Lock every operator mnemonic so a
    # regression is caught. The two-char operators (==,<=,>=,!=) mirror the `->`
    # arrow handling; `==` previously mis-lexed as two single `=` tokens.
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
    (b"pub fn f(a: i64, b: i64) -> i64 { a == b }\n",
     b"  %2 = eq %0, %1\n  output %2\n"),
    (b"pub fn f(a: i64, b: i64) -> i64 { a <= b }\n",
     b"  %2 = le %0, %1\n  output %2\n"),
    (b"pub fn f(a: i64, b: i64) -> i64 { a >= b }\n",
     b"  %2 = ge %0, %1\n  output %2\n"),
    (b"pub fn f(a: i64, b: i64) -> i64 { a != b }\n",
     b"  %2 = ne %0, %1\n  output %2\n"),
    # Parens + nesting + precedence: (a + b) * a -> add then mul over the result.
    (b"pub fn f(a: i64, b: i64) -> i64 { (a + b) * a }\n",
     b"  %2 = add %0, %1\n  %3 = mul %2, %0\n  output %3\n"),
    # Unary negation: -a -> 0 - a (const0 then sub).
    (b"pub fn f(a: i64) -> i64 { -a }\n",
     b"  %1 = const.i64 0\n  %2 = sub %1, %0\n  output %2\n"),
    # Unary logical-not: !a -> a == 0 (const0 then eq).
    (b"pub fn f(a: i64) -> i64 { !a }\n",
     b"  %1 = const.i64 0\n  %2 = eq %0, %1\n  output %2\n"),
    # If-expression control-flow lowering. Mirrors the Rust front-end
    # (src/eval/lower.rs ast::Node::If) and its mic@1 text form
    # (src/ir/print.rs Instr::If): the condition, then-branch and else-branch
    # each lower into a HIDDEN sub-stream, so the only VISIBLE op is the flat
    #   %dst = if cond=%c then=%t else=%e
    # line.  Each branch begins with a `const.i64 0` seed (ir.fresh()) that
    # still consumes an SSA id, so dst/then/else match Rust's allocation even
    # though the seeds and branch bodies are not printed.
    #
    # if a { b } else { a }: a=%0,b=%1. cond a=%0; then-seed=%2, b=%1;
    # else-seed=%3, a=%0; dst=%4.
    (b"pub fn f(a: i64, b: i64) -> i64 { if a { b } else { a } }\n",
     b"  %4 = if cond=%0 then=%1 else=%0\n  output %4\n"),
    # if a { b } else { c }: a=%0,b=%1,c=%2. cond a=%0; then-seed=%3, b=%1;
    # else-seed=%4, c=%2; dst=%5.
    (b"pub fn f(a: i64, b: i64, c: i64) -> i64 { if a { b } else { c } }\n",
     b"  %5 = if cond=%0 then=%1 else=%2\n  output %5\n"),
    # No else clause: the else-branch's unit-0 seed IS the else result.
    # a=%0,b=%1. cond a=%0; then-seed=%2, b=%1; else-seed=%3 (result); dst=%4.
    (b"pub fn f(a: i64, b: i64) -> i64 { if a { b } }\n",
     b"  %4 = if cond=%0 then=%1 else=%3\n  output %4\n"),
    # Literal branch bodies: each int-lit lowers to a hidden const in its
    # branch sub-stream, consuming its own id ON TOP of the branch seed.
    # a=%0. cond a=%0; then-seed=%1, 5=%2 -> then=%2; else-seed=%3, 7=%4 ->
    # else=%4; dst=%5.
    (b"pub fn f(a: i64) -> i64 { if a { 5 } else { 7 } }\n",
     b"  %5 = if cond=%0 then=%2 else=%4\n  output %5\n"),
    # Field access (receiver.field). Mirrors the Rust front-end
    # (src/eval/lower.rs ast::Node::FieldAccess) `None` arm: the bootstrap
    # parser tracks no struct field-name order, so every field access is
    # unresolvable and lowers to exactly ONE `const.i64 0` (the receiver is NOT
    # re-lowered on the None arm). Previously `s.buf` MIS-PARSED as two separate
    # primaries (`s` then a dropped `.buf`) yielding two const.i64-0 ops; it now
    # parses as a single ast_field node.
    #
    # s.buf: param s=%0, field access -> fresh %1 = const 0; output %1.
    (b"pub fn f(s: i64) -> i64 { s.buf }\n",
     b"  %1 = const.i64 0\n  output %1\n"),
    # Chained field a.b.c: the OUTER field access lands on the None arm and
    # short-circuits to a single const (the receiver chain is not re-lowered).
    (b"pub fn f(s: i64) -> i64 { s.a.b }\n",
     b"  %1 = const.i64 0\n  output %1\n"),
    # Field as a binop operand: field=%2 (const 0, after params s=%0,x=%1),
    # then add %2, %1 -> %3; output %3. Confirms field binds tighter than `+`
    # (postfix `.field` is applied before parse_pratt).
    (b"pub fn f(s: i64, x: i64) -> i64 { s.buf + x }\n",
     b"  %2 = const.i64 0\n  %3 = add %2, %1\n  output %3\n"),
    # Field as a let initializer: `let n: i64 = s.next_id; n`. field -> %1 =
    # const 0; n binds %1; trailing n resolves %1 (no op); output %1.
    (b"pub fn f(s: i64) -> i64 { let n: i64 = s.next_id; n }\n",
     b"  %1 = const.i64 0\n  output %1\n"),
]


def emit_body(lib, src: bytes) -> bytes:
    buf = ctypes.create_string_buffer(src, len(src))
    es = lib.selftest_emit_fn_body(ctypes.cast(buf, ctypes.c_void_p).value, len(src))
    rd = lambda a, o=0: ctypes.cast(a + o, ctypes.POINTER(ctypes.c_int64))[0]
    sh = rd(es, 0)
    return ctypes.string_at(rd(sh, 0), rd(sh, 8))


def main() -> int:
    if not SO.exists():
        # When MINDC_SO is explicitly set (CI), a missing .so is a hard failure
        # — otherwise the gate would silently pass green. Local runs without the
        # var skip (the .so is an opt-in local build artifact).
        if os.environ.get("MINDC_SO"):
            print(f"ERROR: {SO} not found (MINDC_SO is set — refusing to skip)")
            return 1
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
