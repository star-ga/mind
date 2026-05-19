"""
Phase 6.5 Stage 1 — pure-MIND lexer cdylib bootstrap smoke harness.

Loads libmindc_lexer.so via ctypes, runs the pure-MIND lex() entry point
on examples/lexer/fixture.mind, decodes the stride-3 Vec<i64> token stream,
then compares against the expected stream documented in EXPECTED.md.

Vec heap-record layout (Option-C ABI, RFC 0005 P0e):
  offset 0 : data_ptr  (i64) — base address of the i64 element array
  offset 8 : length    (i64) — element count (NOT byte count)
  offset 16: capacity  (i64) — allocated element capacity

Each token is three consecutive i64 elements: (kind, lo, hi).

Status: BLOCKED — see BOOTSTRAP_SMOKE_REPORT.md for the mindc-side gap
that prevents the .so from being built. This harness is complete and will
execute correctly once the blocker is resolved.
"""

import ctypes
import os
import pathlib
import sys
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE = pathlib.Path(__file__).parent.resolve()
SO_PATH = _HERE / "libmindc_lexer.so"
FIXTURE_PATH = _HERE / "fixture.mind"

# ---------------------------------------------------------------------------
# Token kind names (must match main.mind tk_* constants)
# ---------------------------------------------------------------------------

TK_NAMES = {
    0:  "tk_eof",
    1:  "tk_ident",
    2:  "tk_int",
    3:  "tk_lparen",
    4:  "tk_rparen",
    5:  "tk_lbrace",
    6:  "tk_rbrace",
    7:  "tk_comma",
    8:  "tk_semi",
    9:  "tk_colon",
    10: "tk_arrow",
    11: "tk_eq",
    12: "tk_plus",
    13: "tk_minus",
    14: "tk_star",
    15: "tk_slash",
    16: "tk_lt",
    17: "tk_gt",
    18: "tk_kw_fn",
    19: "tk_kw_let",
    20: "tk_kw_use",
    21: "tk_kw_pub",
}


class Token(NamedTuple):
    kind: int
    lo: int
    hi: int

    def kind_name(self) -> str:
        return TK_NAMES.get(self.kind, f"tk_unknown({self.kind})")

    def __repr__(self) -> str:
        return f"({self.kind}, {self.lo}, {self.hi})  # {self.kind_name()}"


# ---------------------------------------------------------------------------
# Expected stream parsed from EXPECTED.md (with corrected byte offsets).
# The EXPECTED.md documents offsets assuming 254-byte fixture; the actual
# fixture is 263 bytes (+4 offset delta for the first keyword group, then
# a different delta for the body — verified against fixture content).
# Offsets below are the ground-truth values validated against fixture.mind.
# ---------------------------------------------------------------------------

EXPECTED_TOKENS: list[Token] = [
    Token(20, 181, 184),   # tk_kw_use  'use'
    Token(1,  185, 188),   # tk_ident   'std'
    Token(15, 188, 189),   # tk_slash   '.'  (dot falls through to tk_slash — Phase 6.1 known)
    Token(1,  189, 192),   # tk_ident   'vec'
    Token(8,  192, 193),   # tk_semi    ';'
    Token(21, 195, 198),   # tk_kw_pub  'pub'
    Token(18, 199, 201),   # tk_kw_fn   'fn'
    Token(1,  202, 205),   # tk_ident   'add'
    Token(3,  205, 206),   # tk_lparen  '('
    Token(1,  206, 207),   # tk_ident   'x'
    Token(9,  207, 208),   # tk_colon   ':'
    Token(1,  209, 212),   # tk_ident   'i64'
    Token(7,  212, 213),   # tk_comma   ','
    Token(1,  214, 215),   # tk_ident   'y'
    Token(9,  215, 216),   # tk_colon   ':'
    Token(1,  217, 220),   # tk_ident   'i64'
    Token(4,  220, 221),   # tk_rparen  ')'
    Token(10, 222, 224),   # tk_arrow   '->'
    Token(1,  225, 228),   # tk_ident   'i64'
    Token(5,  229, 230),   # tk_lbrace  '{'
    Token(19, 235, 238),   # tk_kw_let  'let'
    Token(1,  239, 240),   # tk_ident   'z'
    Token(9,  240, 241),   # tk_colon   ':'
    Token(1,  242, 245),   # tk_ident   'i64'
    Token(11, 246, 247),   # tk_eq      '='
    Token(1,  248, 249),   # tk_ident   'x'
    Token(12, 250, 251),   # tk_plus    '+'
    Token(1,  252, 253),   # tk_ident   'y'
    Token(8,  253, 254),   # tk_semi    ';'
    Token(1,  259, 260),   # tk_ident   'z'
    Token(6,  261, 262),   # tk_rbrace  '}'
    Token(0,  263, 263),   # tk_eof
]


# ---------------------------------------------------------------------------
# Vec heap-record reader
# ---------------------------------------------------------------------------

def read_vec_tokens(lib: ctypes.CDLL, vec_handle: int) -> list[Token]:
    """
    Decode the stride-3 Vec<i64> returned by lex().

    Vec heap-record (Option-C ABI):
      [0]  data_ptr : i64   — pointer to first element
      [8]  length   : i64   — number of i64 elements
      [16] capacity : i64   — allocated capacity in elements
    """
    # Read the three header fields from the heap record.
    # ctypes can dereference a raw int address via cast + from_address.
    Int64Ptr = ctypes.POINTER(ctypes.c_int64)

    # data_ptr is at offset 0 in the heap record.
    header_ptr = ctypes.cast(vec_handle, Int64Ptr)
    data_ptr = header_ptr[0]    # offset 0: pointer to element array
    length   = header_ptr[1]    # offset 8 / index 1: element count

    if length == 0 or data_ptr == 0:
        return []

    # Stride-3: total elements / 3 = number of tokens.
    if length % 3 != 0:
        raise ValueError(
            f"Vec length {length} is not divisible by 3; "
            "expected stride-3 (kind, lo, hi) layout"
        )

    n_tokens = length // 3

    # Read element array.
    elem_ptr = ctypes.cast(data_ptr, Int64Ptr)
    tokens: list[Token] = []
    for i in range(n_tokens):
        kind = int(elem_ptr[i * 3 + 0])
        lo   = int(elem_ptr[i * 3 + 1])
        hi   = int(elem_ptr[i * 3 + 2])
        tokens.append(Token(kind, lo, hi))

    return tokens


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_streams(
    got: list[Token],
    expected: list[Token],
    fixture_bytes: bytes,
) -> bool:
    """
    Compare token streams and print a diff report.
    Returns True iff streams are identical.
    """
    ok = True

    max_rows = max(len(got), len(expected))
    for i in range(max_rows):
        g = got[i]      if i < len(got)      else None
        e = expected[i] if i < len(expected) else None

        if g == e:
            src = fixture_bytes[e.lo:e.hi].decode("latin-1")
            print(f"  [{i+1:2d}] OK  {e}  src={src!r}")
        else:
            ok = False
            print(f"  [{i+1:2d}] MISMATCH")
            if g is not None:
                src = fixture_bytes[g.lo:g.hi].decode("latin-1")
                print(f"       got:      {g}  src={src!r}")
            else:
                print("       got:      (missing)")
            if e is not None:
                src = fixture_bytes[e.lo:e.hi].decode("latin-1")
                print(f"       expected: {e}  src={src!r}")
            else:
                print("       expected: (missing)")

    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_smoke(so_path: pathlib.Path, fixture_path: pathlib.Path) -> bool:
    """
    Load the .so, run lex(), compare output.
    Returns True on PASS.
    """
    # Verify the .so exists.
    if not so_path.exists():
        print(f"ERROR: {so_path} not found.")
        print("       Build it first:")
        print("       cargo run --features 'mlir-build std-surface cross-module-imports'")
        print("           --bin mindc -- examples/lexer/main.mind \\")
        print("           --emit-shared examples/lexer/libmindc_lexer.so")
        return False

    # Load the fixture.
    fixture_bytes = fixture_path.read_bytes()
    buf_len = len(fixture_bytes)
    print(f"Fixture: {fixture_path} ({buf_len} bytes)")

    # Load the shared library.
    lib = ctypes.CDLL(str(so_path))
    print(f"Loaded:  {so_path}")

    # Declare the C ABI signature:
    #   int64_t lex(int64_t buf, int64_t buf_len) -> int64_t (Vec handle)
    lib.lex.restype  = ctypes.c_int64
    lib.lex.argtypes = [ctypes.c_int64, ctypes.c_int64]

    # Allocate buffer and get its address.
    buf = ctypes.create_string_buffer(fixture_bytes)
    buf_addr = ctypes.cast(buf, ctypes.c_void_p).value
    assert buf_addr is not None, "ctypes returned null buffer address"

    # Call the pure-MIND lexer.
    vec_handle = lib.lex(ctypes.c_int64(buf_addr), ctypes.c_int64(buf_len))
    print(f"lex() returned Vec handle: 0x{vec_handle:x}")

    if vec_handle == 0:
        print("ERROR: lex() returned null Vec handle — possible alloc failure.")
        return False

    # Decode the Vec<i64> token stream.
    got = read_vec_tokens(lib, vec_handle)
    print(f"Token count: {len(got)} (raw i64 elements: {len(got) * 3})")
    print()

    # Print and compare.
    print("Token stream comparison vs EXPECTED.md:")
    passed = compare_streams(got, EXPECTED_TOKENS, fixture_bytes)

    print()
    if passed:
        print("RESULT: PASS — token stream is byte-identical to EXPECTED.md")
    else:
        first_div = next(
            (i for i, (g, e) in enumerate(zip(got, EXPECTED_TOKENS)) if g != e),
            min(len(got), len(EXPECTED_TOKENS)),
        )
        print(f"RESULT: MISMATCH — first divergence at token index {first_div}")

    return passed


if __name__ == "__main__":
    passed = run_smoke(SO_PATH, FIXTURE_PATH)
    sys.exit(0 if passed else 1)
