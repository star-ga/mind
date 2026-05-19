"""
Phase 6.5 Stage 3 — pure-MIND type-checker cdylib bootstrap smoke harness.

Loads libmindc_lexer.so + libmindc_parser.so + libmindc_typecheck.so via
ctypes, runs the full pipeline:

  1. lex(buf, len)              -> Vec handle (token stream)
  2. parse(vec_handle, buf_addr) -> i64 AST root (ast_program heap-record)
  3. typecheck(ast_root, buf_addr) -> String handle (type-check report)

Reads the returned String byte-by-byte and compares it to the expected
124-byte report documented in EXPECTED.md.

String heap-record layout (RFC 0005 Option C, 3×i64 = 24 bytes):
  offset  0 : addr  (i64) — byte backing-store base address
  offset  8 : len   (i64) — byte count
  offset 16 : cap   (i64) — allocated capacity in bytes

Each byte is stored as a full i64 at its byte offset (addr + i), so
reading back a character at index i is:
  __mind_load_i64(addr + i) & 0xFF

The backing store is a flat i64[] where each character occupies 8 bytes
at byte-stride 1 — the stores overlap but only the low byte of each slot
carries the character value.  The Python reader mirrors this: it casts
the addr to a c_int64 pointer and reads `c_int64[i] & 0xFF` for each i.
"""

import ctypes
import pathlib
import sys
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE = pathlib.Path(__file__).parent.resolve()
LEXER_SO_PATH      = _HERE.parent / "lexer"  / "libmindc_lexer.so"
PARSER_SO_PATH     = _HERE.parent / "parser" / "libmindc_parser.so"
TYPECHECK_SO_PATH  = _HERE / "libmindc_typecheck.so"
FIXTURE_PATH       = _HERE / "fixture.mind"

# ---------------------------------------------------------------------------
# Expected report (from EXPECTED.md, 124 bytes, 6 LF-terminated lines)
# ---------------------------------------------------------------------------

EXPECTED_REPORT = (
    "fn add : (i64, i64) -> i64\n"
    "let z : i64\n"
    "fn compute : (i64, i64, i64) -> i64\n"
    "let r : i64\n"
    "fn cmp : (i64, i64) -> i64\n"
    "let b : bool\n"
)

assert len(EXPECTED_REPORT) == 127, (
    f"EXPECTED_REPORT must be 127 bytes, got {len(EXPECTED_REPORT)}"
)

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

_Int64Ptr = ctypes.POINTER(ctypes.c_int64)


def read_i64(addr: int) -> int:
    """Read a single i64 from a raw heap address."""
    p = ctypes.cast(addr, _Int64Ptr)
    return int(p[0])


def read_string_handle(handle: int) -> bytes:
    """
    Decode a MIND String heap-record handle into a Python bytes object.

    The String record (3×i64) at `handle` contains:
      [0]  addr — base address of the byte backing store
      [8]  len  — logical byte count
      [16] cap  — capacity (ignored)

    Each character at index i is stored as an i64 at backing-store byte
    offset i (not i*8), so we read c_int64 at the cast address and mask
    the low 8 bits.  This matches the MIND `__mind_load_i64(addr + i) & 255`
    pattern used by load_byte() and string_get_byte().
    """
    if handle == 0:
        return b""

    hdr = ctypes.cast(handle, _Int64Ptr)
    addr = int(hdr[0])   # backing store base (byte offset 0)
    length = int(hdr[1]) # logical byte count (byte offset 8)

    if addr == 0 or length == 0:
        return b""

    # Cast the backing store to an i64 pointer.  Each character is at
    # byte offset i — equivalent to reading p[i//8] and shifting, but
    # since we cast to a byte-stride pointer using c_int8, simpler:
    byte_ptr = ctypes.cast(addr, ctypes.POINTER(ctypes.c_int8))
    result = bytearray()
    for i in range(length):
        # i64 stored at byte offset i: low byte on little-endian x86 is
        # the actual character byte.
        result.append(int(byte_ptr[i]) & 0xFF)
    return bytes(result)


# ---------------------------------------------------------------------------
# Main smoke runner
# ---------------------------------------------------------------------------

def run_smoke(
    lexer_so: pathlib.Path,
    parser_so: pathlib.Path,
    typecheck_so: pathlib.Path,
    fixture_path: pathlib.Path,
) -> bool:
    """
    Run the full lex → parse → typecheck pipeline on fixture_path.
    Returns True on PASS (report byte-identical to EXPECTED_REPORT).
    """
    for p in (lexer_so, parser_so, typecheck_so, fixture_path):
        if not p.exists():
            print(f"ERROR: {p} not found.")
            return False

    fixture_bytes = fixture_path.read_bytes()
    buf_len = len(fixture_bytes)
    print(f"Fixture: {fixture_path} ({buf_len} bytes)")

    # Load all three shared libraries with RTLD_LOCAL so each has its own
    # copy of __mind_alloc / map_new / string_new etc.  Heap pointers cross
    # library boundaries because all three link against the same libc malloc.
    lexer_lib     = ctypes.CDLL(str(lexer_so),     mode=ctypes.RTLD_LOCAL)
    parser_lib    = ctypes.CDLL(str(parser_so),    mode=ctypes.RTLD_LOCAL)
    typecheck_lib = ctypes.CDLL(str(typecheck_so), mode=ctypes.RTLD_LOCAL)
    print(f"Loaded:  {lexer_so.name}")
    print(f"Loaded:  {parser_so.name}")
    print(f"Loaded:  {typecheck_so.name}")

    # Create a ctypes buffer for the fixture bytes so its address is stable.
    buf = ctypes.create_string_buffer(fixture_bytes)
    buf_addr = ctypes.cast(buf, ctypes.c_void_p).value
    assert buf_addr is not None

    # ── Step 1: Lex ──────────────────────────────────────────────────────────
    lexer_lib.lex.restype  = ctypes.c_int64
    lexer_lib.lex.argtypes = [ctypes.c_int64, ctypes.c_int64]
    vec_handle = lexer_lib.lex(ctypes.c_int64(buf_addr), ctypes.c_int64(buf_len))
    print(f"\nlex()   Vec handle:  0x{vec_handle:x}")
    if vec_handle == 0:
        print("ERROR: lex() returned null — alloc failure.")
        return False

    hdr = ctypes.cast(vec_handle, ctypes.POINTER(ctypes.c_int64))
    n_elems  = int(hdr[1])
    n_tokens = n_elems // 3
    print(f"        {n_tokens} tokens ({n_elems} i64 elements)")

    # ── Step 2: Parse ────────────────────────────────────────────────────────
    parser_lib.parse.restype  = ctypes.c_int64
    parser_lib.parse.argtypes = [ctypes.c_int64, ctypes.c_int64]
    ast_root = parser_lib.parse(
        ctypes.c_int64(vec_handle), ctypes.c_int64(buf_addr)
    )
    print(f"parse() AST root:    0x{ast_root:x}")
    if ast_root == 0:
        print("ERROR: parse() returned null — alloc failure.")
        return False

    # Sanity: AST root kind should be ast_program (11) and aux = item count.
    hdr_ast = ctypes.cast(ast_root, ctypes.POINTER(ctypes.c_int64))
    ast_kind = int(hdr_ast[0])
    ast_aux  = int(hdr_ast[6])
    print(f"        kind={ast_kind} (expect 11=ast_program), items={ast_aux}")
    if ast_kind != 11:
        print(f"ERROR: unexpected AST root kind {ast_kind}, expected 11.")
        return False

    # ── Step 3: Typecheck ────────────────────────────────────────────────────
    typecheck_lib.typecheck.restype  = ctypes.c_int64
    typecheck_lib.typecheck.argtypes = [ctypes.c_int64, ctypes.c_int64]
    string_handle = typecheck_lib.typecheck(
        ctypes.c_int64(ast_root), ctypes.c_int64(buf_addr)
    )
    print(f"typecheck() String:  0x{string_handle:x}")
    if string_handle == 0:
        print("ERROR: typecheck() returned null — alloc failure.")
        return False

    # ── Step 4: Decode String ────────────────────────────────────────────────
    try:
        report_bytes = read_string_handle(string_handle)
    except Exception as exc:
        print(f"ERROR reading String handle: {exc}")
        return False

    report_str = report_bytes.decode("ascii", errors="replace")
    print(f"\nType-check report ({len(report_bytes)} bytes):")
    print("---")
    print(report_str, end="")
    print("---")

    # ── Step 5: Compare ──────────────────────────────────────────────────────
    expected_bytes = EXPECTED_REPORT.encode("ascii")

    if report_bytes == expected_bytes:
        print("\nReport comparison vs EXPECTED.md: PASS")
        print(f"  Byte count: {len(report_bytes)} (expected 124)")
        return True

    print("\nReport comparison vs EXPECTED.md: MISMATCH")
    print(f"  Got    {len(report_bytes)} bytes")
    print(f"  Expect {len(expected_bytes)} bytes")

    # Find first divergent byte for diagnosis.
    min_len = min(len(report_bytes), len(expected_bytes))
    for i in range(min_len):
        if report_bytes[i] != expected_bytes[i]:
            ctx_start = max(0, i - 5)
            ctx_end   = min(min_len, i + 10)
            print(f"  First divergence at byte {i}:")
            print(f"    Got:    {list(report_bytes[ctx_start:ctx_end])}")
            print(f"    Expect: {list(expected_bytes[ctx_start:ctx_end])}")
            break
    else:
        if len(report_bytes) < len(expected_bytes):
            print(f"  Report truncated: got {len(report_bytes)}, "
                  f"expect {len(expected_bytes)}")
        else:
            print(f"  Report has extra bytes beyond expected {len(expected_bytes)}")

    # Print expected for comparison.
    print("\nExpected report:")
    print("---")
    print(EXPECTED_REPORT, end="")
    print("---")
    return False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    passed = run_smoke(
        LEXER_SO_PATH,
        PARSER_SO_PATH,
        TYPECHECK_SO_PATH,
        FIXTURE_PATH,
    )
    if passed:
        print("\nRESULT: PASS — report matches EXPECTED.md byte-for-byte")
    else:
        print("\nRESULT: MISMATCH — see first divergence above")
    sys.exit(0 if passed else 1)
