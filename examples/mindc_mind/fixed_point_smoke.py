"""
Phase 6.5 — Bootstrap fixed-point round-trip harness.

Feeds examples/mindc_mind/main.mind (the full 1,084-LOC combined pure-MIND
mindc source) to libmindc_mind.so via mindc_compile(src_addr, src_len) and
compares the emitted MLIR byte-for-byte against the oracle produced by
mindc-Rust on the same source.

Heap-record layouts (RFC 0005 Option C, 8-byte stride):

  EmitState (3×i64, 24 bytes at handle):
    offset  0 : buf     (i64) — String heap-record handle
    offset  8 : next_id (i64) — SSA value counter
    offset 16 : last_id (i64) — most-recently-allocated SSA id

  String (3×i64, 24 bytes at handle):
    offset  0 : addr (i64) — byte backing-store base address
    offset  8 : len  (i64) — logical byte count
    offset 16 : cap  (i64) — allocated capacity (ignored)

Verdicts:
  PASS             — emitted MLIR byte-identical to oracle (mindc-Rust is decorative)
  FIRST-DIVERGENCE — first differing byte/line + subsystem identified
  BLOCKED          — mindc_compile crashed / returned null / unrecoverable error
"""

import ctypes
import pathlib
import sys

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE       = pathlib.Path(__file__).parent.resolve()
COMBINED_SO = _HERE / "libmindc_mind.so"
SOURCE_PATH = _HERE / "main.mind"
ORACLE_PATH = pathlib.Path("/tmp/oracle_pure.mlir")

# ---------------------------------------------------------------------------
# Low-level heap helpers (same as bootstrap_smoke.py)
# ---------------------------------------------------------------------------

_Int64Ptr = ctypes.POINTER(ctypes.c_int64)


def read_i64_at(addr: int, byte_offset: int = 0) -> int:
    """Read a single i64 from raw heap address + byte offset."""
    p = ctypes.cast(addr + byte_offset, _Int64Ptr)
    return int(p[0])


def read_string_handle(handle: int) -> bytes:
    """
    Decode a MIND String heap-record handle into Python bytes.

    String record (3×i64) at `handle`:
      [+0] addr — byte backing-store base
      [+8] len  — logical byte count
      [+16] cap — capacity (ignored)
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
    """
    buf_handle = read_i64_at(handle, 0)
    next_id    = read_i64_at(handle, 8)
    last_id    = read_i64_at(handle, 16)
    return buf_handle, next_id, last_id


# ---------------------------------------------------------------------------
# Oracle loading
# ---------------------------------------------------------------------------

def load_oracle(oracle_path: pathlib.Path) -> bytes:
    """
    Load the oracle MLIR produced by mindc-Rust --emit-ir on main.mind.

    Strips the cargo invocation header lines (everything before `module {`)
    so we compare pure MLIR text only.
    """
    raw = oracle_path.read_text(encoding="ascii", errors="replace")
    lines = raw.split("\n")
    mlir_lines: list[str] = []
    found = False
    for line in lines:
        if not found and line.strip() == "module {":
            found = True
        if found:
            mlir_lines.append(line)
    if not found:
        raise ValueError(
            f"Oracle at {oracle_path} does not contain 'module {{' — "
            "was it generated with --emit-ir?"
        )
    mlir = "\n".join(mlir_lines).rstrip("\n") + "\n"
    return mlir.encode("ascii")


# ---------------------------------------------------------------------------
# Divergence analysis
# ---------------------------------------------------------------------------

def identify_subsystem(emitted: bytes, oracle: bytes, first_div: int) -> str:
    """
    Heuristic: look at the oracle content up to the first divergence to
    decide which compiler subsystem is responsible.

    The oracle groups output by function.  The comment lines
    '  // fn <name>' mark function boundaries.  We find the last such
    boundary before the divergence point.
    """
    oracle_prefix = oracle[:first_div].decode("ascii", errors="replace")
    # Walk backwards through comment markers
    lines = oracle_prefix.split("\n")
    for line in reversed(lines):
        stripped = line.strip()
        if stripped.startswith("// fn "):
            fn_name = stripped[len("// fn "):]
            # Map function name to subsystem
            if any(k in fn_name for k in (
                "tk_", "load_byte", "is_space", "is_digit", "is_alpha",
                "is_alnum", "skip_whitespace", "skip_line_comment",
                "match_keyword", "lex_ident", "lex_int_lit", "lex_token",
                "lex",
            )):
                return f"LEXER (fn {fn_name})"
            if any(k in fn_name for k in (
                "parse", "ast_", "node_", "expect_", "peek_",
                "advance_", "current_",
            )):
                return f"PARSER (fn {fn_name})"
            if any(k in fn_name for k in (
                "typecheck", "check_", "resolve_", "infer_", "scope_",
                "lookup_", "bind_",
            )):
                return f"TYPECHECK (fn {fn_name})"
            if any(k in fn_name for k in (
                "emit_", "lower_", "alloc_", "str_", "flush_",
                "mindc_compile",
            )):
                return f"EMITTER (fn {fn_name})"
            return f"UNKNOWN-FN ({fn_name})"
    return "UNKNOWN (before first fn comment)"


def report_divergence(emitted: bytes, oracle: bytes) -> None:
    """Print first-divergence details."""
    min_len = min(len(emitted), len(oracle))
    for i in range(min_len):
        if emitted[i] != oracle[i]:
            ctx_start = max(0, i - 20)
            ctx_end   = min(min_len, i + 20)
            subsystem = identify_subsystem(emitted, oracle, i)

            # Find line numbers
            oracle_line = oracle[:i].count(b"\n") + 1
            emitted_line = emitted[:i].count(b"\n") + 1

            print(f"\n  First divergence at byte {i}:")
            print(f"    Oracle  line ~{oracle_line}")
            print(f"    Emitted line ~{emitted_line}")
            print(f"    Subsystem: {subsystem}")
            print(f"    Oracle  context:  {list(oracle[ctx_start:ctx_end])}")
            print(f"    Emitted context:  {list(emitted[ctx_start:ctx_end])}")

            # Show human-readable lines
            oracle_lines  = oracle.decode("ascii", errors="replace").split("\n")
            emitted_lines = emitted.decode("ascii", errors="replace").split("\n")
            oline = oracle_line - 1
            eline = emitted_line - 1
            print(f"\n    Oracle  line {oracle_line}:  "
                  f"{oracle_lines[oline] if oline < len(oracle_lines) else '<EOF>'!r}")
            print(f"    Emitted line {emitted_line}: "
                  f"{emitted_lines[eline] if eline < len(emitted_lines) else '<EOF>'!r}")
            return

    # No byte differs — length mismatch
    if len(emitted) < len(oracle):
        print(f"\n  Emitted is TRUNCATED: got {len(emitted)}, "
              f"oracle has {len(oracle)} bytes")
        print(f"  First missing oracle byte at position {len(emitted)}: "
              f"{oracle[len(emitted)]!r}")
    else:
        print(f"\n  Emitted has EXTRA bytes: got {len(emitted)}, "
              f"oracle has {len(oracle)} bytes")
        print(f"  First extra byte at position {len(oracle)}: "
              f"{emitted[len(oracle)]!r}")


# ---------------------------------------------------------------------------
# Main round-trip runner
# ---------------------------------------------------------------------------

def run_fixed_point(
    combined_so: pathlib.Path,
    source_path: pathlib.Path,
    oracle_path: pathlib.Path,
) -> tuple[str, str]:
    """
    Load libmindc_mind.so, feed it the full main.mind source, compare
    the emitted MLIR against the mindc-Rust oracle.

    Returns (verdict, detail) where verdict is one of:
      'PASS', 'FIRST-DIVERGENCE', 'BLOCKED'
    """
    # Verify required files
    for p in (combined_so, source_path, oracle_path):
        if not p.exists():
            print(f"ERROR: {p} not found.")
            return "BLOCKED", f"Missing file: {p}"

    # Load oracle
    try:
        oracle_bytes = load_oracle(oracle_path)
    except Exception as exc:
        print(f"ERROR loading oracle: {exc}")
        return "BLOCKED", f"Oracle load failure: {exc}"

    print(f"Oracle:  {oracle_path} ({len(oracle_bytes)} bytes, "
          f"{oracle_bytes.count(b'%')} SSA values)")

    # Load source
    source_bytes = source_path.read_bytes()
    print(f"Source:  {source_path.name} ({len(source_bytes)} bytes, "
          f"{source_bytes.count(b'\n')} lines)")

    # Load shared library
    lib = ctypes.CDLL(str(combined_so), mode=ctypes.RTLD_LOCAL)
    so_size = combined_so.stat().st_size
    print(f"Library: {combined_so.name} ({so_size:,} bytes)")

    # Stable ctypes buffer for source bytes
    buf = ctypes.create_string_buffer(source_bytes)
    buf_addr = ctypes.cast(buf, ctypes.c_void_p).value
    assert buf_addr is not None, "ctypes buffer address must not be None"

    # Configure mindc_compile ABI
    lib.mindc_compile.restype  = ctypes.c_int64
    lib.mindc_compile.argtypes = [ctypes.c_int64, ctypes.c_int64]

    print(f"\nCalling mindc_compile(0x{buf_addr:x}, {len(source_bytes)}) ...")

    try:
        es_handle = lib.mindc_compile(
            ctypes.c_int64(buf_addr),
            ctypes.c_int64(len(source_bytes)),
        )
    except Exception as exc:
        print(f"ERROR: mindc_compile raised: {exc}")
        return "BLOCKED", f"mindc_compile exception: {exc}"

    print(f"  EmitState handle: 0x{es_handle:x}")

    if es_handle == 0:
        print("ERROR: mindc_compile() returned null — alloc failure or crash.")
        return "BLOCKED", "mindc_compile() returned null handle"

    # Decode EmitState
    try:
        buf_handle, next_id, last_id = read_emit_state(es_handle)
    except Exception as exc:
        print(f"ERROR reading EmitState: {exc}")
        return "BLOCKED", f"EmitState decode failure: {exc}"

    print(f"  EmitState.buf=0x{buf_handle:x}  next_id={next_id}  last_id={last_id}")

    try:
        emitted_bytes = read_string_handle(buf_handle)
    except Exception as exc:
        print(f"ERROR reading String handle: {exc}")
        return "BLOCKED", f"String handle decode failure: {exc}"

    emitted_str = emitted_bytes.decode("ascii", errors="replace")
    print(f"\nEmitted MLIR ({len(emitted_bytes)} bytes, {next_id} SSA values):")
    print("--- (first 10 lines) ---")
    for line in emitted_str.split("\n")[:10]:
        print(f"  {line}")
    print("--- (last 5 lines) ---")
    lines = emitted_str.split("\n")
    for line in lines[-5:]:
        print(f"  {line}")
    print("---")

    # Byte-for-byte comparison
    if emitted_bytes == oracle_bytes:
        print(f"\nByte comparison: PASS")
        print(f"  Bytes:   {len(emitted_bytes)} (oracle: {len(oracle_bytes)})")
        print(f"  next_id: {next_id}")
        return "PASS", f"byte-identical ({len(emitted_bytes)} bytes, next_id={next_id})"

    print(f"\nByte comparison: MISMATCH")
    print(f"  Emitted: {len(emitted_bytes)} bytes")
    print(f"  Oracle:  {len(oracle_bytes)} bytes")
    report_divergence(emitted_bytes, oracle_bytes)

    return "FIRST-DIVERGENCE", (
        f"emitted={len(emitted_bytes)}B oracle={len(oracle_bytes)}B "
        f"next_id={next_id}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 72)
    print("Phase 6.5 — Bootstrap fixed-point round-trip")
    print("  Source:  examples/mindc_mind/main.mind (combined pure-MIND mindc)")
    print("  Library: examples/mindc_mind/libmindc_mind.so")
    print("  Oracle:  mindc-Rust --emit-ir on the same source")
    print("=" * 72)

    verdict, detail = run_fixed_point(COMBINED_SO, SOURCE_PATH, ORACLE_PATH)

    print("\n" + "=" * 72)
    print(f"VERDICT: {verdict}")
    print(f"DETAIL:  {detail}")
    print("=" * 72)

    if verdict == "PASS":
        print(
            "\nBOOTSTRAP FIXED-POINT REACHED.\n"
            "libmindc_mind.so (pure-MIND mindc) compiles its own source\n"
            "byte-identically to mindc-Rust.  The Rust compiler is now decorative."
        )
        sys.exit(0)
    elif verdict == "FIRST-DIVERGENCE":
        print("\nFixed-point not yet reached. See first divergence above.")
        sys.exit(1)
    else:
        print("\nBLOCKED — see error above for scoped reproducer details.")
        sys.exit(2)
