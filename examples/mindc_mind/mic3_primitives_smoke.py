#!/usr/bin/env python3
"""mic@3 self-host convergence — Phase 1 gate: pure-MIND ULEB128 / zigzag.

Loads the self-host cdylib (libmindc_mind.so, or MINDC_SO) and calls the exported
`selftest_mic3_uleb(n)`, which emits the ULEB128 of `n` into a fresh EmitState
(main.mind `emit_uleb128`). Reads the bytes back and checks them byte-for-byte
against the reference ULEB128 — the same encoding `src/ir/compact/v3/emit.rs
uleb128_write` produces. This proves the pure-MIND mic@3 varint primitive is
byte-exact, the foundation Phase 2's section emitter builds on (design:
mind-ecosystem-audit/SELF-HOST-MIC3-CONVERGENCE-DESIGN-2026-06-09.md).

Isolated from the canary mic@1 path; does not affect the keystone. Exit 0 = PASS.

Usage:  [MINDC_SO=/path/to.so] python3 mic3_primitives_smoke.py
"""
import ctypes
import os
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
_DEFAULT_SO = _HERE / "libmindc_mind.so"
_Int64Ptr = ctypes.POINTER(ctypes.c_int64)


def read_i64_at(addr: int, off: int = 0) -> int:
    return int(ctypes.cast(addr + off, _Int64Ptr)[0])


def read_string_handle(handle: int) -> bytes:
    """MIND String record: [+0] addr, [+8] len, [+16] cap."""
    if handle == 0:
        return b""
    addr = read_i64_at(handle, 0)
    length = read_i64_at(handle, 8)
    if addr == 0 or length == 0:
        return b""
    p = ctypes.cast(addr, ctypes.POINTER(ctypes.c_int8))
    return bytes(int(p[i]) & 0xFF for i in range(length))


def ref_uleb128(n: int) -> bytes:
    """Reference unsigned-LEB128 (matches Rust uleb128_write)."""
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            break
    return bytes(out)


def main() -> int:
    so = os.environ.get("MINDC_SO", str(_DEFAULT_SO))
    if not pathlib.Path(so).exists():
        if "MINDC_SO" in os.environ:
            raise SystemExit(f"FAIL: MINDC_SO set but missing: {so}")
        print(f"SKIP: {so} not found (opt-in local build artifact)")
        return 0

    lib = ctypes.CDLL(so)
    lib.selftest_mic3_uleb.restype = ctypes.c_void_p
    lib.selftest_mic3_uleb.argtypes = [ctypes.c_int64]

    # Spread of values: single-byte, the 0x7F/0x80 boundary, the classic
    # multi-byte cases, and a large value that exercises several continuations.
    cases = [0, 1, 2, 63, 127, 128, 129, 300, 16384, 624485, 1 << 28, 1 << 35]

    print("mic@3 self-host Phase 1 — pure-MIND ULEB128 byte-exactness gate")
    print(f"  .so: {so}")
    failures = 0
    for n in cases:
        es = lib.selftest_mic3_uleb(n)
        got = read_string_handle(read_i64_at(es, 0))
        want = ref_uleb128(n)
        ok = got == want
        failures += not ok
        tag = "OK" if ok else f"FAIL (want {want.hex()})"
        print(f"  uleb128({n}) = {got.hex():<10} {tag}")

    total = len(cases)

    # --- MIC3 header: "MIC3" magic + version byte (mic@3 = 2) ---
    lib.selftest_mic3_header.restype = ctypes.c_void_p
    lib.selftest_mic3_header.argtypes = []
    es = lib.selftest_mic3_header()
    got = read_string_handle(read_i64_at(es, 0))
    want = b"MIC3\x02"
    failures += got != want
    total += 1
    print(f"  header = {got.hex():<10} {'OK' if got == want else 'FAIL (want ' + want.hex() + ')'}")

    # --- string-table entry: uleb128(len) || bytes (the per-string framing) ---
    lib.selftest_mic3_lp_bytes.restype = ctypes.c_void_p
    lib.selftest_mic3_lp_bytes.argtypes = [ctypes.c_int64, ctypes.c_int64]
    for sval in [b"", b"x", b"add", b"compute", b"a" * 200]:
        cbuf = ctypes.create_string_buffer(sval, max(1, len(sval)))
        addr = ctypes.cast(cbuf, ctypes.c_void_p).value
        es = lib.selftest_mic3_lp_bytes(addr, len(sval))
        got = read_string_handle(read_i64_at(es, 0))
        want = ref_uleb128(len(sval)) + sval
        failures += got != want
        total += 1
        tag = "OK" if got == want else f"FAIL (want {want.hex()})"
        print(f"  lp_bytes(len={len(sval)}) = {got.hex():<18} {tag}")

    # --- instruction emitters: operands -> mic@3 bytes (emit_instr arms) ---
    def ref_zigzag(v):
        return 2 * v if v >= 0 else (-v) * 2 - 1

    def ref_const_i64(dst, v):
        return bytes([1]) + ref_uleb128(dst) + ref_uleb128(ref_zigzag(v))

    def ref_binop(dst, op, lhs, rhs):
        return bytes([4]) + ref_uleb128(dst) + bytes([op]) + ref_uleb128(lhs) + ref_uleb128(rhs)

    lib.selftest_mic3_const_i64.restype = ctypes.c_void_p
    lib.selftest_mic3_const_i64.argtypes = [ctypes.c_int64, ctypes.c_int64]
    for dst, v in [(0, 0), (5, 300), (0, -1), (3, -1000), (10, 624485)]:
        es = lib.selftest_mic3_const_i64(dst, v)
        got = read_string_handle(read_i64_at(es, 0))
        want = ref_const_i64(dst, v)
        failures += got != want
        total += 1
        tag = "OK" if got == want else f"FAIL (want {want.hex()})"
        print(f"  const_i64(dst={dst}, v={v}) = {got.hex():<14} {tag}")

    lib.selftest_mic3_binop.restype = ctypes.c_void_p
    lib.selftest_mic3_binop.argtypes = [ctypes.c_int64] * 4
    for dst, op, lhs, rhs in [(2, 0, 0, 1), (5, 7, 3, 4), (100, 12, 50, 99)]:
        es = lib.selftest_mic3_binop(dst, op, lhs, rhs)
        got = read_string_handle(read_i64_at(es, 0))
        want = ref_binop(dst, op, lhs, rhs)
        failures += got != want
        total += 1
        tag = "OK" if got == want else f"FAIL (want {want.hex()})"
        print(f"  binop(dst={dst},op={op},lhs={lhs},rhs={rhs}) = {got.hex():<12} {tag}")

    if failures:
        raise SystemExit(f"FAIL: {failures}/{total} mic@3 primitive mismatches")
    print(f"  PASS — {total}/{total} byte-exact vs reference "
          f"(uleb128 + header + string-table + instr emit)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
