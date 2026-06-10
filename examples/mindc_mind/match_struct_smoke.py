"""
Self-host MATCH STRUCTURE smoke (the match frontier, part 1) — proves the
pure-MIND compiler can LOCATE a `match` expression in source and READ its
structure: presence, arm count, the AST kind of the scrutinee, and whether a
wildcard `_` arm exists.

The Rust front-end lowers `match` as a pure AST-level desugar to a nested
if/else chain (src/eval/lower.rs desugar_match_to_if): each test arm becomes a
`<scrutinee> == <pat>` comparison feeding the keystone-protected If lowering,
and a `_` / bare-ident arm becomes the terminal else. There is NO dedicated
match/switch opcode — match and the equivalent if/else chain emit byte-identical
mic@3 (both end in OP_IF 0x1c chains).

The bootstrap statement dispatch (parse_stmt / parse_primary) does NOT recognise
the `match` keyword — main.mind uses no surface `match` itself (it dispatches AST
node kinds with `if kind == X` chains) — so a `match` spelling lexes as a bare
ident. These selftests therefore locate the match by scanning the token stream
for an ident-token spelling "match", parse the scrutinee with the existing
parse_expr (which stops at `{`), then read the arm structure off the token
stream. The fat arrow `=>` is not a bootstrap token, so it lexes as the
two-token sequence tk_eq then tk_gt; each arm has exactly one such pair at brace
depth 0. They do NO mic@3 byte-output and are fully isolated from the canary
(mindc_compile -> lower_program -> emit_fn_def stays stubbed; the shared
statement dispatch is untouched), so fixed_point_smoke.py stays byte-identical.

These are self-consistency unit goldens (presence / arm-count / scrut-kind /
wildcard), not a byte-identity test.

Run:  python3 examples/mindc_mind/match_struct_smoke.py
"""

import ctypes
import os
import pathlib
import sys

_DEFAULT_SO = pathlib.Path(__file__).parent / "libmindc_mind.so"
SO = pathlib.Path(os.environ.get("MINDC_SO", str(_DEFAULT_SO)))

# AST kind tags (main.mind): ast_ident() -> 2, ast_binop() -> 3.
AST_IDENT = 2
AST_BINOP = 3
# When there is no match, arm_count / scrut_kind / has_wildcard return the -1
# sentinel and present returns 0.
NO_MATCH = -1

# Each case: (source, want_present, want_arm_count, want_scrut_kind,
#             want_has_wildcard). The *_kind / *_count / *_wildcard fields are
# NO_MATCH when want_present == 0.
CASES = [
    # Three-arm match with a wildcard, bare-ident scrutinee.
    (b"pub fn classify(k: i64) -> i64 { match k { 0 => 10, 1 => 20, _ => 99 } }\n",
     1, 3, AST_IDENT, 1),
    # Two-arm match with a wildcard.
    (b"pub fn pick(k: i64) -> i64 { match k { 0 => 10, _ => 99 } }\n",
     1, 2, AST_IDENT, 1),
    # Two literal arms, NO wildcard.
    (b"pub fn two(k: i64) -> i64 { match k { 0 => 10, 1 => 20 } }\n",
     1, 2, AST_IDENT, 0),
    # Single wildcard arm.
    (b"pub fn one(k: i64) -> i64 { match k { _ => 99 } }\n",
     1, 1, AST_IDENT, 1),
    # Binop scrutinee `match k + 1 { ... }` -> scrut kind is a binop.
    (b"pub fn shift(k: i64) -> i64 { match k + 1 { 0 => 10, _ => 99 } }\n",
     1, 2, AST_BINOP, 1),
    # Four arms with wildcard.
    (b"pub fn quad(k: i64) -> i64 { match k { 0 => 1, 1 => 2, 2 => 3, _ => 9 } }\n",
     1, 4, AST_IDENT, 1),
    # No match at all -> present 0, sentinels.
    (b"pub fn id(x: i64) -> i64 { x }\n",
     0, NO_MATCH, NO_MATCH, NO_MATCH),
    # An `if` is not a match.
    (b"pub fn pick(c: i64) -> i64 { if c < 1 { 0 } else { 1 } }\n",
     0, NO_MATCH, NO_MATCH, NO_MATCH),
]


def main() -> int:
    if not SO.exists():
        if os.environ.get("MINDC_SO"):
            print(f"ERROR: {SO} not found (MINDC_SO is set — refusing to skip)")
            return 1
        print(f"SKIP: {SO} not built")
        return 0

    lib = ctypes.CDLL(str(SO))
    for fn in ("selftest_match_present",
               "selftest_match_arm_count",
               "selftest_match_scrut_kind",
               "selftest_match_has_wildcard"):
        f = getattr(lib, fn)
        f.restype = ctypes.c_int64
        f.argtypes = [ctypes.c_int64, ctypes.c_int64]

    failures = 0
    for src, want_present, want_arms, want_scrut, want_wild in CASES:
        buf = ctypes.create_string_buffer(src, len(src))
        addr = ctypes.cast(buf, ctypes.c_void_p).value
        present = lib.selftest_match_present(addr, len(src))
        arms = lib.selftest_match_arm_count(addr, len(src))
        scrut = lib.selftest_match_scrut_kind(addr, len(src))
        wild = lib.selftest_match_has_wildcard(addr, len(src))
        label = src.decode().strip()
        ok = (present == want_present and arms == want_arms
              and scrut == want_scrut and wild == want_wild)
        if not ok:
            failures += 1
            print(f"FAIL: {label}")
            print(f"  present  got {present} want {want_present}")
            print(f"  arms     got {arms} want {want_arms}")
            print(f"  scrut    got {scrut} want {want_scrut}")
            print(f"  wildcard got {wild} want {want_wild}")
        else:
            print(f"ok: {label}  (arms={arms}, scrut={scrut}, wild={wild})")

    if failures:
        print(f"\n{failures} FAILED")
        return 1
    print(f"\nall {len(CASES)} match-structure cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
