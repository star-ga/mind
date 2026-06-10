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

    # --- terminators: Output (0x13) and Return (0x17, opt vid) ---
    def ref_output(idv):
        return bytes([0x13]) + ref_uleb128(idv)

    def ref_return(has, idv):
        return bytes([0x17]) + (bytes([1]) + ref_uleb128(idv) if has else bytes([0]))

    lib.selftest_mic3_output.restype = ctypes.c_void_p
    lib.selftest_mic3_output.argtypes = [ctypes.c_int64]
    for idv in [0, 5, 300]:
        es = lib.selftest_mic3_output(idv)
        got = read_string_handle(read_i64_at(es, 0))
        want = ref_output(idv)
        failures += got != want
        total += 1
        tag = "OK" if got == want else f"FAIL (want {want.hex()})"
        print(f"  output(id={idv}) = {got.hex():<8} {tag}")

    lib.selftest_mic3_return.restype = ctypes.c_void_p
    lib.selftest_mic3_return.argtypes = [ctypes.c_int64, ctypes.c_int64]
    for has, idv in [(0, 0), (1, 3), (1, 300)]:
        es = lib.selftest_mic3_return(has, idv)
        got = read_string_handle(read_i64_at(es, 0))
        want = ref_return(has, idv)
        failures += got != want
        total += 1
        tag = "OK" if got == want else f"FAIL (want {want.hex()})"
        print(f"  return(has={has},id={idv}) = {got.hex():<8} {tag}")

    # --- full string-table SECTION: uleb128(count) || N length-prefixed entries.
    #     Uses the fixture.mind string set so the output equals the captured mic@3
    #     oracle's string table (Phase 0, b230530) byte-for-byte. ---
    lib.selftest_mic3_strtab.restype = ctypes.c_void_p
    lib.selftest_mic3_strtab.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
    strings = [b"add", b"x", b"y", b"compute", b"z"]
    sbuf = b"".join(strings)
    offsets = [0]
    for sng in strings:
        offsets.append(offsets[-1] + len(sng))
    sbuf_c = ctypes.create_string_buffer(sbuf, len(sbuf))
    offs_c = (ctypes.c_int64 * len(offsets))(*offsets)
    es = lib.selftest_mic3_strtab(
        ctypes.cast(sbuf_c, ctypes.c_void_p).value,
        ctypes.cast(offs_c, ctypes.c_void_p).value,
        len(strings),
    )
    got = read_string_handle(read_i64_at(es, 0))
    want = ref_uleb128(len(strings)) + b"".join(ref_uleb128(len(x)) + x for x in strings)
    # The bytes the fixture.mind mic@3 oracle carries after its MIC3+version header.
    oracle_strtab = bytes.fromhex("0503616464017801790763") + b"ompute" + bytes.fromhex("017a")
    matches_oracle = got == oracle_strtab
    failures += got != want
    total += 1
    tag = "OK" + (" (== fixture oracle string table)" if matches_oracle else "") \
        if got == want else f"FAIL (want {want.hex()})"
    print(f"  strtab[{len(strings)}] = {got.hex()}  {tag}")

    # --- exports section: uleb128(count) || count uleb128 indices ---
    lib.selftest_mic3_exports.restype = ctypes.c_void_p
    lib.selftest_mic3_exports.argtypes = [ctypes.c_int64, ctypes.c_int64]
    exp_idx = [0, 2, 4, 300]
    idx_c = (ctypes.c_int64 * len(exp_idx))(*exp_idx)
    es = lib.selftest_mic3_exports(ctypes.cast(idx_c, ctypes.c_void_p).value, len(exp_idx))
    got = read_string_handle(read_i64_at(es, 0))
    want = ref_uleb128(len(exp_idx)) + b"".join(ref_uleb128(e) for e in exp_idx)
    failures += got != want
    total += 1
    print(f"  exports{exp_idx} = {got.hex():<12} {'OK' if got == want else 'FAIL want ' + want.hex()}")

    # --- empty std-surface registries: three uleb128(0) ---
    lib.selftest_mic3_empty_registries.restype = ctypes.c_void_p
    lib.selftest_mic3_empty_registries.argtypes = []
    es = lib.selftest_mic3_empty_registries()
    got = read_string_handle(read_i64_at(es, 0))
    want = b"\x00\x00\x00"
    failures += got != want
    total += 1
    print(f"  empty_registries = {got.hex():<8} {'OK' if got == want else 'FAIL want ' + want.hex()}")

    # --- COMPLETE bodiless module: header||strtab||next_id||exports||0-instrs||registries ---
    lib.selftest_mic3_module_noinstr.restype = ctypes.c_void_p
    lib.selftest_mic3_module_noinstr.argtypes = [ctypes.c_int64] * 6
    msyms = [b"f"]
    mbuf = b"".join(msyms)
    moffs = [0]
    for sng in msyms:
        moffs.append(moffs[-1] + len(sng))
    mexports = [0]
    next_id = 1
    mbuf_c = ctypes.create_string_buffer(mbuf, len(mbuf))
    moffs_c = (ctypes.c_int64 * len(moffs))(*moffs)
    mexp_c = (ctypes.c_int64 * len(mexports))(*mexports)
    es = lib.selftest_mic3_module_noinstr(
        ctypes.cast(mbuf_c, ctypes.c_void_p).value,
        ctypes.cast(moffs_c, ctypes.c_void_p).value,
        len(msyms), next_id,
        ctypes.cast(mexp_c, ctypes.c_void_p).value, len(mexports),
    )
    got = read_string_handle(read_i64_at(es, 0))
    want = (b"MIC3\x02"
            + ref_uleb128(len(msyms)) + b"".join(ref_uleb128(len(x)) + x for x in msyms)
            + ref_uleb128(next_id)
            + ref_uleb128(len(mexports)) + b"".join(ref_uleb128(e) for e in mexports)
            + ref_uleb128(0)            # 0 instructions
            + b"\x00\x00\x00")          # 3 empty registries
    failures += got != want
    total += 1
    print(f"  module(noinstr, syms={msyms}) = {got.hex()}  "
          f"{'OK (complete valid mic@3 module)' if got == want else 'FAIL want ' + want.hex()}")

    # --- FN_DEF header (OP_FN_DEF 0x15): name_idx || params(0) || opt_vid(ret) ||
    #     opt_f64(None) || body_len.  fn f()->i64 {..}: 15 00 00 01 00 00 01 ---
    lib.selftest_mic3_fn_def_header.restype = ctypes.c_void_p
    lib.selftest_mic3_fn_def_header.argtypes = [ctypes.c_int64] * 4
    es = lib.selftest_mic3_fn_def_header(0, 1, 0, 1)
    got = read_string_handle(read_i64_at(es, 0))
    want = (bytes([0x15]) + ref_uleb128(0) + ref_uleb128(0)
            + bytes([1]) + ref_uleb128(0) + bytes([0]) + ref_uleb128(1))
    failures += got != want
    total += 1
    print(f"  fn_def_header(name0,ret%0,body=1) = {got.hex()}  "
          f"{'OK (== oracle FN_DEF header)' if got == want else 'FAIL want ' + want.hex()}")

    # --- COMPLETE WITH-BODY module: pub fn <name>() -> i64 {{ <val> }} — the
    #     first end-to-end with-body mic@3 emit in pure-MIND. (f(){42}) == the
    #     captured 29-byte Rust oracle byte-for-byte; verified across values. ---
    def ref_const_fn(name, val):
        return (b"MIC3\x02"
                + ref_uleb128(1) + ref_uleb128(len(name)) + name        # strtab
                + ref_uleb128(1)                                        # next_id
                + ref_uleb128(0)                                        # exports
                + ref_uleb128(3)                                        # 3 instrs
                + bytes([0x15]) + ref_uleb128(0) + ref_uleb128(0)       # FN_DEF
                + bytes([1]) + ref_uleb128(0) + bytes([0]) + ref_uleb128(1)
                + bytes([1]) + ref_uleb128(0) + ref_uleb128(ref_zigzag(val))   # body const
                + bytes([1]) + ref_uleb128(0) + ref_uleb128(ref_zigzag(0))     # module const 0
                + bytes([0x13]) + ref_uleb128(0)                        # output
                + b"\x00\x00\x00")                                      # registries

    lib.selftest_mic3_const_fn.restype = ctypes.c_void_p
    lib.selftest_mic3_const_fn.argtypes = [ctypes.c_int64] * 4
    for name, val in [(b"f", 42), (b"f", 100), (b"g", 7), (b"f", -5)]:
        nbuf = ctypes.create_string_buffer(name, len(name))
        noffs = (ctypes.c_int64 * 2)(0, len(name))
        es = lib.selftest_mic3_const_fn(
            ctypes.cast(nbuf, ctypes.c_void_p).value,
            ctypes.cast(noffs, ctypes.c_void_p).value, 0, val)
        got = read_string_handle(read_i64_at(es, 0))
        want = ref_const_fn(name, val)
        failures += got != want
        total += 1
        note = " (== fn f(){42} oracle, 29B)" if (name, val) == (b"f", 42) else ""
        ok = "OK" + note if got == want else f"FAIL want {want.hex()}"
        print(f"  const_fn {name.decode()}()={{{val}}} [{len(got)}B] = {got.hex()}  {ok}")

    # --- OP_PARAM (0x18): vid(dst) || string_idx(name) || uleb(index).
    #     fn id(a): PARAM(%0, "a"=1, 0) = 18 00 01 00 ---
    lib.selftest_mic3_param.restype = ctypes.c_void_p
    lib.selftest_mic3_param.argtypes = [ctypes.c_int64] * 3
    for dst, nidx, index in [(0, 1, 0), (2, 3, 1), (0, 300, 5)]:
        es = lib.selftest_mic3_param(dst, nidx, index)
        got = read_string_handle(read_i64_at(es, 0))
        want = bytes([0x18]) + ref_uleb128(dst) + ref_uleb128(nidx) + ref_uleb128(index)
        failures += got != want
        total += 1
        note = " (== fn id(a) PARAM)" if (dst, nidx, index) == (0, 1, 0) else ""
        print(f"  param(dst={dst},name={nidx},idx={index}) = {got.hex():<10} "
              f"{'OK' + note if got == want else 'FAIL want ' + want.hex()}")

    # --- encode_named_vids: uleb(count) || (string_idx, vid) per pair ---
    lib.selftest_mic3_named_vids.restype = ctypes.c_void_p
    lib.selftest_mic3_named_vids.argtypes = [ctypes.c_int64, ctypes.c_int64]
    for pairs in [[(1, 0)], [(1, 0), (2, 1)], [(1, 0), (2, 1), (3, 2)]]:
        flat = [v for pr in pairs for v in pr]
        pairs_c = (ctypes.c_int64 * len(flat))(*flat)
        es = lib.selftest_mic3_named_vids(ctypes.cast(pairs_c, ctypes.c_void_p).value, len(pairs))
        got = read_string_handle(read_i64_at(es, 0))
        want = ref_uleb128(len(pairs)) + b"".join(ref_uleb128(s) + ref_uleb128(v) for s, v in pairs)
        failures += got != want
        total += 1
        print(f"  named_vids({pairs}) = {got.hex():<12} "
              f"{'OK' if got == want else 'FAIL want ' + want.hex()}")

    # --- COMPLETE WITH-BODY arithmetic fn: pub fn <name>(<pa>,<pb>) -> i64 { pa OP pb }
    #     add(a,b){a+b} (op_byte=0) == the captured 49-byte oracle byte-for-byte. ---
    def ref_arith_fn(name, pa, pb, op_byte):
        strs = [name, pa, pb]
        out = b"MIC3\x02"
        out += ref_uleb128(3) + b"".join(ref_uleb128(len(x)) + x for x in strs)
        out += ref_uleb128(1) + ref_uleb128(0) + ref_uleb128(3)        # next_id, exports, 3 instrs
        out += bytes([0x15]) + ref_uleb128(0)                          # FN_DEF, name_idx=0
        out += ref_uleb128(2) + ref_uleb128(1) + ref_uleb128(0) + ref_uleb128(2) + ref_uleb128(1)
        out += bytes([1]) + ref_uleb128(2) + bytes([0]) + ref_uleb128(3)  # ret Some %2, reap None, body_len 3
        out += bytes([0x18]) + ref_uleb128(0) + ref_uleb128(1) + ref_uleb128(0)  # PARAM %0
        out += bytes([0x18]) + ref_uleb128(1) + ref_uleb128(2) + ref_uleb128(1)  # PARAM %1
        out += bytes([4]) + ref_uleb128(2) + bytes([op_byte]) + ref_uleb128(0) + ref_uleb128(1)  # BINOP
        out += bytes([1]) + ref_uleb128(0) + ref_uleb128(0)            # ConstI64(%0,0)
        out += bytes([0x13]) + ref_uleb128(0)                          # Output(%0)
        out += b"\x00\x00\x00"
        return out

    lib.selftest_mic3_arith_fn.restype = ctypes.c_void_p
    lib.selftest_mic3_arith_fn.argtypes = [ctypes.c_int64] * 3
    for name, pa, pb, op in [(b"add", b"a", b"b", 0), (b"mul", b"x", b"y", 2)]:
        sbuf = name + pa + pb
        soff = [0, len(name), len(name) + len(pa), len(name) + len(pa) + len(pb)]
        sbuf_c = ctypes.create_string_buffer(sbuf, len(sbuf))
        soff_c = (ctypes.c_int64 * len(soff))(*soff)
        es = lib.selftest_mic3_arith_fn(
            ctypes.cast(sbuf_c, ctypes.c_void_p).value,
            ctypes.cast(soff_c, ctypes.c_void_p).value, op)
        got = read_string_handle(read_i64_at(es, 0))
        want = ref_arith_fn(name, pa, pb, op)
        failures += got != want
        total += 1
        note = " (== add(a,b){a+b} oracle, 49B)" if (name, op) == (b"add", 0) else ""
        ok = "OK" + note if got == want else f"FAIL want {want.hex()}"
        print(f"  arith_fn {name.decode()}({pa.decode()},{pb.decode()}) [{len(got)}B] = {got.hex()}  {ok}")

    if failures:
        raise SystemExit(f"FAIL: {failures}/{total} mic@3 primitive mismatches")
    print(f"  PASS — {total}/{total} byte-exact vs reference "
          f"(uleb128 + header + string-table + instr emit)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
