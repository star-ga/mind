"""
Phase 6.5 Stage 4 — pure-MIND MLIR-text emitter cdylib bootstrap smoke harness.

Loads libmindc_lexer.so + libmindc_parser.so + libmindc_typecheck.so +
libmindc_emit_ir.so via ctypes, runs the full pipeline:

  1. lex(buf, len)                 -> Vec handle (token stream)
  2. parse(vec_handle, buf_addr)   -> i64 AST root (ast_program heap-record)
  3. typecheck(ast_root, buf_addr) -> String handle (type-check report, unused)
  4. lower_program(ast_root, buf)  -> EmitState handle (MLIR text accumulator)

Decodes the returned EmitState's String buffer and compares it byte-for-byte
against the 149-byte MLIR text documented in EXPECTED.md.

Heap-record layouts (RFC 0005 Option C, 8-byte stride):

  EmitState (3×i64, 24 bytes at handle):
    offset  0 : buf     (i64) — String heap-record handle
    offset  8 : next_id (i64) — SSA value counter
    offset 16 : last_id (i64) — most-recently-allocated SSA id

  String (3×i64, 24 bytes at handle):
    offset  0 : addr (i64) — byte backing-store base address
    offset  8 : len  (i64) — logical byte count
    offset 16 : cap  (i64) — allocated capacity (ignored)

  Byte backing store: each character at index i occupies one byte at byte
  offset i of the store (not i*8). Matches __mind_load_i64(addr + i) & 255.
"""

import ctypes
import pathlib
import sys

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE = pathlib.Path(__file__).parent.resolve()
LEXER_SO_PATH     = _HERE.parent / "lexer"     / "libmindc_lexer.so"
PARSER_SO_PATH    = _HERE.parent / "parser"    / "libmindc_parser.so"
TYPECHECK_SO_PATH = _HERE.parent / "typecheck" / "libmindc_typecheck.so"
EMIT_IR_SO_PATH   = _HERE / "libmindc_emit_ir.so"
FIXTURE_PATH      = _HERE / "fixture.mind"

# ---------------------------------------------------------------------------
# Expected MLIR text (from EXPECTED.md, 149 bytes)
# ---------------------------------------------------------------------------

EXPECTED_MLIR = (
    "module {\n"
    "  %0 = const.i64 0\n"
    "  output %0\n"
    "  // fn add\n"
    "  %1 = const.i64 0\n"
    "  output %1\n"
    "  // fn compute\n"
    "  %2 = const.i64 0\n"
    "  output %2\n"
    "}  // next_id = 3\n"
)

assert len(EXPECTED_MLIR) == 148, (
    f"EXPECTED_MLIR must be 148 bytes, got {len(EXPECTED_MLIR)}"
)

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

_Int64Ptr = ctypes.POINTER(ctypes.c_int64)


def read_i64_at(addr: int, byte_offset: int = 0) -> int:
    """Read a single i64 from a raw heap address + byte offset."""
    p = ctypes.cast(addr + byte_offset, _Int64Ptr)
    return int(p[0])


def read_string_handle(handle: int) -> bytes:
    """
    Decode a MIND String heap-record handle into a Python bytes object.

    The String record (3×i64) at `handle`:
      [offset  0] addr — byte backing-store base
      [offset  8] len  — logical byte count
      [offset 16] cap  — capacity (ignored)

    Each character at index i is stored as one byte at byte offset i of the
    backing store (stride-1, not stride-8). This matches string_push_byte's
    __mind_store_i64(addr + len, byte) pattern.
    """
    if handle == 0:
        return b""

    addr   = read_i64_at(handle, 0)
    length = read_i64_at(handle, 8)

    if addr == 0 or length == 0:
        return b""

    byte_ptr = ctypes.cast(addr, ctypes.POINTER(ctypes.c_int8))
    result = bytearray()
    for i in range(length):
        result.append(int(byte_ptr[i]) & 0xFF)
    return bytes(result)


def read_emit_state(handle: int) -> tuple[int, int, int]:
    """
    Decode an EmitState heap-record handle.

    Returns (buf_handle, next_id, last_id).
    EmitState layout (3×i64 at 8-byte stride):
      offset  0 : buf     — String handle
      offset  8 : next_id
      offset 16 : last_id
    """
    buf_handle = read_i64_at(handle, 0)
    next_id    = read_i64_at(handle, 8)
    last_id    = read_i64_at(handle, 16)
    return buf_handle, next_id, last_id


# ---------------------------------------------------------------------------
# Main smoke runner
# ---------------------------------------------------------------------------

def run_smoke(
    lexer_so:     pathlib.Path,
    parser_so:    pathlib.Path,
    typecheck_so: pathlib.Path,
    emit_ir_so:   pathlib.Path,
    fixture_path: pathlib.Path,
) -> bool:
    """
    Run the full lex → parse → typecheck → emit_ir pipeline on fixture_path.
    Returns True on PASS (emitted MLIR byte-identical to EXPECTED_MLIR).
    """
    for p in (lexer_so, parser_so, typecheck_so, emit_ir_so, fixture_path):
        if not p.exists():
            print(f"ERROR: {p} not found.")
            return False

    fixture_bytes = fixture_path.read_bytes()
    buf_len = len(fixture_bytes)
    print(f"Fixture: {fixture_path} ({buf_len} bytes)")

    # Load all four shared libraries. RTLD_LOCAL gives each its own symbol
    # namespace; heap pointers cross boundaries because all link to the same
    # libc malloc.
    lexer_lib     = ctypes.CDLL(str(lexer_so),     mode=ctypes.RTLD_LOCAL)
    parser_lib    = ctypes.CDLL(str(parser_so),    mode=ctypes.RTLD_LOCAL)
    typecheck_lib = ctypes.CDLL(str(typecheck_so), mode=ctypes.RTLD_LOCAL)
    emit_ir_lib   = ctypes.CDLL(str(emit_ir_so),   mode=ctypes.RTLD_LOCAL)
    print(f"Loaded:  {lexer_so.name}")
    print(f"Loaded:  {parser_so.name}")
    print(f"Loaded:  {typecheck_so.name}")
    print(f"Loaded:  {emit_ir_so.name}")

    # Stable ctypes buffer for the fixture source bytes.
    buf = ctypes.create_string_buffer(fixture_bytes)
    buf_addr = ctypes.cast(buf, ctypes.c_void_p).value
    assert buf_addr is not None

    # ── Step 1: Lex ──────────────────────────────────────────────────────────
    lexer_lib.lex.restype  = ctypes.c_int64
    lexer_lib.lex.argtypes = [ctypes.c_int64, ctypes.c_int64]
    vec_handle = lexer_lib.lex(ctypes.c_int64(buf_addr), ctypes.c_int64(buf_len))
    print(f"\nlex()          Vec handle:     0x{vec_handle:x}")
    if vec_handle == 0:
        print("ERROR: lex() returned null — alloc failure.")
        return False

    hdr = ctypes.cast(vec_handle, ctypes.POINTER(ctypes.c_int64))
    n_elems  = int(hdr[1])
    n_tokens = n_elems // 3
    print(f"               {n_tokens} tokens ({n_elems} i64 elements)")

    # ── Step 2: Parse ────────────────────────────────────────────────────────
    parser_lib.parse.restype  = ctypes.c_int64
    parser_lib.parse.argtypes = [ctypes.c_int64, ctypes.c_int64]
    ast_root = parser_lib.parse(
        ctypes.c_int64(vec_handle), ctypes.c_int64(buf_addr)
    )
    print(f"parse()        AST root:       0x{ast_root:x}")
    if ast_root == 0:
        print("ERROR: parse() returned null — alloc failure.")
        return False

    hdr_ast  = ctypes.cast(ast_root, ctypes.POINTER(ctypes.c_int64))
    ast_kind = int(hdr_ast[0])
    ast_aux  = int(hdr_ast[6])
    print(f"               kind={ast_kind} (expect 11=ast_program), items={ast_aux}")
    if ast_kind != 11:
        print(f"ERROR: unexpected AST root kind {ast_kind}, expected 11.")
        return False

    # ── Step 3: Typecheck (pipeline continuity — result not used in emit) ────
    typecheck_lib.typecheck.restype  = ctypes.c_int64
    typecheck_lib.typecheck.argtypes = [ctypes.c_int64, ctypes.c_int64]
    tc_handle = typecheck_lib.typecheck(
        ctypes.c_int64(ast_root), ctypes.c_int64(buf_addr)
    )
    print(f"typecheck()    String handle:  0x{tc_handle:x}")
    if tc_handle == 0:
        print("ERROR: typecheck() returned null — alloc failure.")
        return False

    # ── Step 4: Emit IR ──────────────────────────────────────────────────────
    emit_ir_lib.lower_program.restype  = ctypes.c_int64
    emit_ir_lib.lower_program.argtypes = [ctypes.c_int64, ctypes.c_int64]
    es_handle = emit_ir_lib.lower_program(
        ctypes.c_int64(ast_root), ctypes.c_int64(buf_addr)
    )
    print(f"lower_program() EmitState:     0x{es_handle:x}")
    if es_handle == 0:
        print("ERROR: lower_program() returned null — alloc failure.")
        return False

    # ── Step 5: Decode EmitState ─────────────────────────────────────────────
    try:
        buf_handle, next_id, last_id = read_emit_state(es_handle)
    except Exception as exc:
        print(f"ERROR reading EmitState handle: {exc}")
        return False

    print(f"               buf=0x{buf_handle:x}  next_id={next_id}  last_id={last_id}")

    try:
        emitted_bytes = read_string_handle(buf_handle)
    except Exception as exc:
        print(f"ERROR reading String (EmitState.buf) handle: {exc}")
        return False

    emitted_str = emitted_bytes.decode("ascii", errors="replace")
    print(f"\nEmitted MLIR ({len(emitted_bytes)} bytes):")
    print("---")
    print(emitted_str, end="")
    print("---")

    # ── Step 6: Compare ──────────────────────────────────────────────────────
    expected_bytes = EXPECTED_MLIR.encode("ascii")

    if emitted_bytes == expected_bytes:
        print("\nMLIR comparison vs EXPECTED.md: PASS")
        print(f"  Byte count: {len(emitted_bytes)} (expected 148)")
        print(f"  next_id:    {next_id} (expected 3)")
        return True

    print("\nMLIR comparison vs EXPECTED.md: MISMATCH")
    print(f"  Got    {len(emitted_bytes)} bytes")
    print(f"  Expect {len(expected_bytes)} bytes")

    min_len = min(len(emitted_bytes), len(expected_bytes))
    for i in range(min_len):
        if emitted_bytes[i] != expected_bytes[i]:
            ctx_start = max(0, i - 5)
            ctx_end   = min(min_len, i + 10)
            print(f"  First divergence at byte {i}:")
            print(f"    Got:    {list(emitted_bytes[ctx_start:ctx_end])}")
            print(f"    Expect: {list(expected_bytes[ctx_start:ctx_end])}")
            break
    else:
        if len(emitted_bytes) < len(expected_bytes):
            print(f"  Emitted truncated: got {len(emitted_bytes)}, "
                  f"expect {len(expected_bytes)}")
        else:
            print(f"  Emitted has extra bytes beyond expected {len(expected_bytes)}")

    print("\nExpected MLIR:")
    print("---")
    print(EXPECTED_MLIR, end="")
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
        EMIT_IR_SO_PATH,
        FIXTURE_PATH,
    )
    if passed:
        print("\nRESULT: PASS — emitted MLIR byte-identical to EXPECTED.md")
        print("Stage 4 closed. Ready for Stage 5 apex.")
    else:
        print("\nRESULT: MISMATCH — see first divergence above")
    sys.exit(0 if passed else 1)
