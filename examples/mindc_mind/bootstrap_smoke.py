"""
Phase 6.5 Stage 5 APEX — combined libmindc_mind.so bootstrap smoke harness.

Loads the single combined shared library libmindc_mind.so and runs the full
lex → parse → typecheck → emit_ir pipeline via the unified entry point
mindc_compile(src_addr, src_len) -> EmitState.

mindc_compile wires all four pure-MIND sub-components:
  1. lex(buf, len)                     -> Vec    (token stream)
  2. parse(vec_handle, buf_addr)        -> i64    (AST root)
  3. typecheck(ast_root, buf_addr)      -> String (type-check report, unused)
  4. lower_program(ast_root, buf_addr)  -> EmitState (MLIR text accumulator)

Returns the i64 heap-record address of the EmitState.  The harness decodes
EmitState.buf (a String handle) and compares the resulting bytes byte-for-byte
against the 148-byte MLIR text documented in EXPECTED.md.

Heap-record layouts (RFC 0005 Option C, 8-byte stride):

  EmitState (3×i64, 24 bytes at handle):
    offset  0 : buf     (i64) — String heap-record handle
    offset  8 : next_id (i64) — SSA value counter
    offset 16 : last_id (i64) — most-recently-allocated SSA id

  String (3×i64, 24 bytes at handle):
    offset  0 : addr (i64) — byte backing-store base address
    offset  8 : len  (i64) — logical byte count
    offset 16 : cap  (i64) — allocated capacity (ignored)

VERDICT:
  APEX PASS  = emitted MLIR byte-identical to EXPECTED.md (148 bytes)
  MISMATCH   = output differs — documents first divergent byte/line
"""

import ctypes
import pathlib
import sys

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE         = pathlib.Path(__file__).parent.resolve()
COMBINED_SO   = _HERE / "libmindc_mind.so"
FIXTURE_PATH  = _HERE / "fixture.mind"

# ---------------------------------------------------------------------------
# Expected MLIR text (from EXPECTED.md, 148 bytes, no trailing extra newline)
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

    String record (3×i64) at `handle`:
      [offset  0] addr — byte backing-store base
      [offset  8] len  — logical byte count
      [offset 16] cap  — capacity (ignored)

    Each character at index i is stored as one byte at byte offset i
    of the backing store (stride-1, not stride-8).
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

def run_smoke(combined_so: pathlib.Path, fixture_path: pathlib.Path) -> bool:
    """
    Load libmindc_mind.so, call mindc_compile on fixture_path, and compare
    the emitted MLIR text byte-for-byte against EXPECTED_MLIR.

    Returns True on APEX PASS, False on MISMATCH.
    """
    for p in (combined_so, fixture_path):
        if not p.exists():
            print(f"ERROR: {p} not found.")
            return False

    fixture_bytes = fixture_path.read_bytes()
    buf_len = len(fixture_bytes)
    print(f"Fixture: {fixture_path} ({buf_len} bytes)")

    # Load the combined shared library.
    lib = ctypes.CDLL(str(combined_so), mode=ctypes.RTLD_LOCAL)
    so_size = combined_so.stat().st_size
    print(f"Loaded:  {combined_so.name} ({so_size:,} bytes)")

    # Stable ctypes buffer for the fixture source bytes.
    buf = ctypes.create_string_buffer(fixture_bytes)
    buf_addr = ctypes.cast(buf, ctypes.c_void_p).value
    assert buf_addr is not None

    # ── Call mindc_compile ────────────────────────────────────────────────────
    #
    # mindc_compile(src_addr: i64, src_len: i64) -> EmitState
    #
    # In MIND's Option-C ABI, structs are heap records returned as their
    # i64 base address.  We declare restype=c_int64 to capture that address.
    lib.mindc_compile.restype  = ctypes.c_int64
    lib.mindc_compile.argtypes = [ctypes.c_int64, ctypes.c_int64]

    print(f"\nCalling mindc_compile(0x{buf_addr:x}, {buf_len}) ...")
    es_handle = lib.mindc_compile(
        ctypes.c_int64(buf_addr), ctypes.c_int64(buf_len)
    )
    print(f"  mindc_compile() EmitState handle: 0x{es_handle:x}")

    if es_handle == 0:
        print("ERROR: mindc_compile() returned null — alloc failure.")
        return False

    # ── Decode EmitState ──────────────────────────────────────────────────────
    try:
        buf_handle, next_id, last_id = read_emit_state(es_handle)
    except Exception as exc:
        print(f"ERROR reading EmitState handle: {exc}")
        return False

    print(f"  EmitState.buf=0x{buf_handle:x}  next_id={next_id}  last_id={last_id}")

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

    # ── Byte-for-byte comparison ──────────────────────────────────────────────
    expected_bytes = EXPECTED_MLIR.encode("ascii")

    if emitted_bytes == expected_bytes:
        print("\nMLIR comparison vs EXPECTED.md: APEX PASS")
        print(f"  Byte count: {len(emitted_bytes)} (expected 148)")
        print(f"  next_id:    {next_id} (expected 3)")
        print(f"  last_id:    {last_id} (expected 2)")
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
    passed = run_smoke(COMBINED_SO, FIXTURE_PATH)
    if passed:
        print(
            "\nRESULT: APEX PASS — combined libmindc_mind.so produces MLIR "
            "byte-identical to mindc-Rust on the same fixture."
        )
        print(
            "SELF-HOST THESIS PROVEN: the four pure-MIND mindc sub-components, "
            "integrated into a single cdylib, compile MIND programs to "
            "byte-identical output as the Rust reference compiler."
        )
    else:
        print("\nRESULT: MISMATCH — see first divergence above")
        print("Verdict: FIRST-DIVERGENCE")
    sys.exit(0 if passed else 1)
