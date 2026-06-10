"""
Self-host WHILE-LOOP STRUCTURE smoke (loops part 1) — proves the pure-MIND
compiler can LOCATE a `while` loop in source and READ its structure: presence,
the AST kind of the condition expression, and the statement count of the body.

The bootstrap statement dispatch (parse_stmt / parse_primary) does NOT recognise
the `while` keyword — main.mind uses no while loops itself — so a `while`
spelling lexes as a bare ident. These selftests therefore locate the loop by
scanning the token stream for an ident-token spelling "while" and then reuse the
existing parse_expr (condition) and parse_block (body) helpers to read structure.
They do NO mic@3 byte-output and are fully isolated from the canary
(mindc_compile -> lower_program -> emit_fn_def stays stubbed; the shared
statement dispatch is untouched), so fixed_point_smoke.py stays byte-identical.

These are self-consistency unit goldens (presence / cond-kind / body-stmt-count),
not a byte-identity test.

Run:  python3 examples/mindc_mind/while_struct_smoke.py
"""

import ctypes
import os
import pathlib
import sys

_DEFAULT_SO = pathlib.Path(__file__).parent / "libmindc_mind.so"
SO = pathlib.Path(os.environ.get("MINDC_SO", str(_DEFAULT_SO)))

# ast_binop kind tag (main.mind: ast_binop() -> 3). The condition `i < n` parses
# as a binop, so a comparison-condition while reports cond_kind == AST_BINOP.
AST_BINOP = 3
# When there is no while loop, cond_kind / body_stmt_count return the -1
# sentinel and present returns 0.
NO_WHILE = -1

# Each case: (source, want_present, want_cond_kind, want_body_stmt_count).
# want_cond_kind / want_body_stmt_count are NO_WHILE when want_present == 0.
CASES = [
    # Empty body, comparison condition -> present, binop cond, 0 body stmts.
    (b"pub fn count(n: i64) -> i64 { let i: i64 = 0; while i < n { } i }\n",
     1, AST_BINOP, 0),
    # One-statement body.
    (b"pub fn count(n: i64) -> i64 { let i: i64 = 0; while i < n { i; } i }\n",
     1, AST_BINOP, 1),
    # Two-statement body.
    (b"pub fn run(n: i64) -> i64 { let i: i64 = 0; while i < n { i; i; } i }\n",
     1, AST_BINOP, 2),
    # `>` comparison condition is still a binop.
    (b"pub fn down(n: i64) -> i64 { while n > 0 { n; } n }\n",
     1, AST_BINOP, 1),
    # No while loop at all -> present 0, sentinels.
    (b"pub fn id(x: i64) -> i64 { x }\n",
     0, NO_WHILE, NO_WHILE),
    # An `if` is not a while.
    (b"pub fn pick(c: i64) -> i64 { if c < 1 { 0 } else { 1 } }\n",
     0, NO_WHILE, NO_WHILE),
]


def main() -> int:
    if not SO.exists():
        if os.environ.get("MINDC_SO"):
            print(f"ERROR: {SO} not found (MINDC_SO is set — refusing to skip)")
            return 1
        print(f"SKIP: {SO} not built")
        return 0

    lib = ctypes.CDLL(str(SO))
    for fn in ("selftest_while_present",
               "selftest_while_cond_kind",
               "selftest_while_body_stmt_count"):
        f = getattr(lib, fn)
        f.restype = ctypes.c_int64
        f.argtypes = [ctypes.c_int64, ctypes.c_int64]

    failures = 0
    for src, want_present, want_cond, want_body in CASES:
        buf = ctypes.create_string_buffer(src, len(src))
        addr = ctypes.cast(buf, ctypes.c_void_p).value
        present = lib.selftest_while_present(addr, len(src))
        cond = lib.selftest_while_cond_kind(addr, len(src))
        body = lib.selftest_while_body_stmt_count(addr, len(src))
        ok = (present == want_present and cond == want_cond
              and body == want_body)
        detail = ""
        if not ok:
            detail = (f"  got present={present} cond={cond} body={body}, "
                      f"expected present={want_present} cond={want_cond} "
                      f"body={want_body}")
        print(f"  {'PASS' if ok else 'FAIL'}  "
              f"{src.strip().decode():<58} -> "
              f"present={present} cond={cond} body={body}{detail}")
        if not ok:
            failures += 1

    print(f"\n{'ALL PASS' if failures == 0 else f'{failures} FAILED'}  "
          f"({len(CASES)} cases)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
