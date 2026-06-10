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

    # --- COMPLETE WITH-BODY identity fn: pub fn <name>(<pa>) -> i64 { <pa> }
    #     The first function whose RETURN is a PARAMETER directly — ret_vid is the
    #     param's own vid (%0) and the body is a single PARAM (no BINOP, no body
    #     const). ident(a){a} == the captured 38-byte oracle byte-for-byte. ---
    def ref_ident_fn(name, pa):
        strs = [name, pa]
        out = b"MIC3\x02"
        out += ref_uleb128(2) + b"".join(ref_uleb128(len(x)) + x for x in strs)
        out += ref_uleb128(1) + ref_uleb128(0) + ref_uleb128(3)        # next_id, exports, 3 instrs
        out += bytes([0x15]) + ref_uleb128(0)                          # FN_DEF, name_idx=0
        out += ref_uleb128(1) + ref_uleb128(1) + ref_uleb128(0)        # 1 param (str=1, vid=0)
        out += bytes([1]) + ref_uleb128(0) + bytes([0]) + ref_uleb128(1)  # ret Some %0, reap None, body_len 1
        out += bytes([0x18]) + ref_uleb128(0) + ref_uleb128(1) + ref_uleb128(0)  # PARAM %0
        out += bytes([1]) + ref_uleb128(0) + ref_uleb128(0)            # ConstI64(%0,0)
        out += bytes([0x13]) + ref_uleb128(0)                          # Output(%0)
        out += b"\x00\x00\x00"
        return out

    lib.selftest_mic3_ident_fn.restype = ctypes.c_void_p
    lib.selftest_mic3_ident_fn.argtypes = [ctypes.c_int64] * 2
    for name, pa in [(b"ident", b"a"), (b"id", b"x")]:
        sbuf = name + pa
        soff = [0, len(name), len(name) + len(pa)]
        sbuf_c = ctypes.create_string_buffer(sbuf, len(sbuf))
        soff_c = (ctypes.c_int64 * len(soff))(*soff)
        es = lib.selftest_mic3_ident_fn(
            ctypes.cast(sbuf_c, ctypes.c_void_p).value,
            ctypes.cast(soff_c, ctypes.c_void_p).value)
        got = read_string_handle(read_i64_at(es, 0))
        want = ref_ident_fn(name, pa)
        failures += got != want
        total += 1
        note = " (== ident(a){a} oracle, 38B)" if (name, pa) == (b"ident", b"a") else ""
        ok = "OK" + note if got == want else f"FAIL want {want.hex()}"
        print(f"  ident_fn {name.decode()}({pa.decode()}) [{len(got)}B] = {got.hex()}  {ok}")

    # --- COMPLETE WITH-BODY N-param chained-binop fn:
    #     pub fn <name>(p0,..,p(n-1)) -> i64 { p0 OP p1 OP .. } — the first body that
    #     emits MORE THAN ONE BINOP where each consumes the previous binop's result.
    #     params take %0..%(n-1); %n = %0 op %1; %(n+k) = %(n+k-1) op %(k+1); the
    #     FN_DEF ret_vid is the FINAL binop (2n-2), body_len = n PARAMs + (n-1) BINOPs.
    #     chain(a,b,c){a+b+c} == captured 64-byte oracle; chain4{a+b+c+d} == 78-byte. ---
    def ref_chain_fn(name, params, op_byte):
        n = len(params)
        strs = [name] + params
        out = b"MIC3\x02"
        out += ref_uleb128(len(strs)) + b"".join(ref_uleb128(len(x)) + x for x in strs)
        out += ref_uleb128(1) + ref_uleb128(0) + ref_uleb128(3)        # next_id, exports, 3 instrs
        out += bytes([0x15]) + ref_uleb128(0)                          # FN_DEF, name_idx=0
        out += ref_uleb128(n)                                          # param-list count
        for i in range(n):
            out += ref_uleb128(i + 1) + ref_uleb128(i)                 # (str_idx i+1, vid i)
        out += bytes([1]) + ref_uleb128(2 * n - 2)                     # ret Some %(2n-2)
        out += bytes([0]) + ref_uleb128(2 * n - 1)                     # reap None, body_len 2n-1
        for i in range(n):
            out += bytes([0x18]) + ref_uleb128(i) + ref_uleb128(i + 1) + ref_uleb128(i)  # PARAM %i
        dst = n
        for k in range(1, n):                                          # n-1 BINOPs
            lhs, rhs = (0, 1) if k == 1 else (dst - 1, k)
            out += bytes([4]) + ref_uleb128(dst) + bytes([op_byte]) + ref_uleb128(lhs) + ref_uleb128(rhs)
            dst += 1
        out += bytes([1]) + ref_uleb128(0) + ref_uleb128(0)           # ConstI64(%0,0)
        out += bytes([0x13]) + ref_uleb128(0)                         # Output(%0)
        out += b"\x00\x00\x00"
        return out

    lib.selftest_mic3_chain_fn.restype = ctypes.c_void_p
    lib.selftest_mic3_chain_fn.argtypes = [ctypes.c_int64] * 4
    for name, params, op in [(b"chain", [b"a", b"b", b"c"], 0),
                             (b"chain4", [b"a", b"b", b"c", b"d"], 0),
                             (b"mchain", [b"x", b"y", b"z"], 2)]:
        sbuf = name + b"".join(params)
        soff = [0, len(name)]
        for p in params:
            soff.append(soff[-1] + len(p))
        sbuf_c = ctypes.create_string_buffer(sbuf, len(sbuf))
        soff_c = (ctypes.c_int64 * len(soff))(*soff)
        es = lib.selftest_mic3_chain_fn(
            ctypes.cast(sbuf_c, ctypes.c_void_p).value,
            ctypes.cast(soff_c, ctypes.c_void_p).value, len(params), op)
        got = read_string_handle(read_i64_at(es, 0))
        want = ref_chain_fn(name, params, op)
        failures += got != want
        total += 1
        note = ""
        if (name, op) == (b"chain", 0):
            note = " (== chain(a,b,c){a+b+c} oracle, 64B)"
        elif (name, op) == (b"chain4", 0):
            note = " (== chain4(a,b,c,d){a+b+c+d} oracle, 78B)"
        ok = "OK" + note if got == want else f"FAIL want {want.hex()}"
        pnames = ",".join(p.decode() for p in params)
        print(f"  chain_fn {name.decode()}({pnames}) [{len(got)}B] = {got.hex()}  {ok}")

    # --- COMPLETE WITH-BODY mixed-operator expression-tree fn:
    #     pub fn <name>(p0,..,p(n-1)) -> i64 { <arbitrary binary expr-tree> } — the
    #     first body that is NOT a left-spine accumulator. The tree is a flat
    #     POST-ORDER node array (4 i64 per node: kind, a, b, c). LEAF (kind 0):
    #     a=param index (vid==index). BINOP (kind 1): a=op_byte, b=left slot,
    #     c=right slot. LEAF vid = param index; BINOP vid allocated n,n+1,.. in
    #     post-order; ret_vid = last node (root); body_len = n + #binops.
    #     mixed(a,b,c,d){a*b+c*d} == captured 77-byte oracle; f(a,b,c){(a+b)*c} == 60B. ---
    def ref_tree_fn(name, params, nodes):
        # Node kinds: LEAF=(0, param_index, 0, 0); BINOP=(1, op_byte, left, right);
        # CONST=(2, raw_literal, 0, 0) [Phase 4h]. PARAM leaves own vids 0..n-1; CONST
        # leaves and BINOPs SHARE the post-order nxt++ counter (the oracle interleaves
        # them, e.g. h(a){a*2+1} -> PARAM %0, CONST %1, BINOP %2, CONST %3, BINOP %4).
        # body_len = total node count (every non-PARAM node emits one body instr, plus
        # the n PARAM instrs). The const value is zigzag-encoded (lit 2 -> CONST 4).
        n = len(params)
        strs = [name] + params
        # Pass 1: resolve slot vids (post-order; CONST + BINOP from n via shared nxt).
        slot_vid = [0] * len(nodes)
        nxt = n
        for j, nd in enumerate(nodes):
            if nd[0] == 0:                       # PARAM LEAF
                slot_vid[j] = nd[1]              # param index == vid
            else:                                # CONST (2) or BINOP (1)
                slot_vid[j] = nxt
                nxt += 1
        root_vid = slot_vid[-1]
        body_len = len(nodes)                    # n PARAMs + every non-param node
        out = b"MIC3\x02"
        out += ref_uleb128(len(strs)) + b"".join(ref_uleb128(len(x)) + x for x in strs)
        out += ref_uleb128(1) + ref_uleb128(0) + ref_uleb128(3)   # next_id, exports, 3 instrs
        out += bytes([0x15]) + ref_uleb128(0)                     # FN_DEF, name_idx=0
        out += ref_uleb128(n)
        for i in range(n):
            out += ref_uleb128(i + 1) + ref_uleb128(i)            # (str_idx i+1, vid i)
        out += bytes([1]) + ref_uleb128(root_vid)                 # ret Some %root
        out += bytes([0]) + ref_uleb128(body_len)                 # reap None, body_len
        for i in range(n):
            out += bytes([0x18]) + ref_uleb128(i) + ref_uleb128(i + 1) + ref_uleb128(i)  # PARAM %i
        for j, nd in enumerate(nodes):
            if nd[0] == 1:                       # BINOP, post-order
                _, ob, ls, rs = nd
                out += bytes([4]) + ref_uleb128(slot_vid[j]) + bytes([ob]) \
                    + ref_uleb128(slot_vid[ls]) + ref_uleb128(slot_vid[rs])
            elif nd[0] == 2:                     # CONST leaf, post-order (zigzag value)
                out += bytes([1]) + ref_uleb128(slot_vid[j]) + ref_uleb128(ref_zigzag(nd[1]))
        out += bytes([1]) + ref_uleb128(0) + ref_uleb128(0)       # ConstI64(%0,0)
        out += bytes([0x13]) + ref_uleb128(0)                     # Output(%0)
        out += b"\x00\x00\x00"
        return out

    lib.selftest_mic3_tree_fn.restype = ctypes.c_void_p
    lib.selftest_mic3_tree_fn.argtypes = [ctypes.c_int64] * 6
    # Each case: (name, params, nodes) where a node is a 4-tuple (kind, a, b, c).
    # LEAF=(0, param_index, 0, 0); BINOP=(1, op_byte, left_slot, right_slot). op0=add, op2=mul.
    tree_cases = [
        # mixed(a,b,c,d){a*b+c*d}: +(*(a,b),*(c,d))
        (b"mixed", [b"a", b"b", b"c", b"d"],
         [(0, 0, 0, 0), (0, 1, 0, 0), (1, 2, 0, 1),
          (0, 2, 0, 0), (0, 3, 0, 0), (1, 2, 3, 4),
          (1, 0, 2, 5)]),
        # f(a,b,c){(a+b)*c}: *(+(a,b),c)
        (b"f", [b"a", b"b", b"c"],
         [(0, 0, 0, 0), (0, 1, 0, 0), (1, 0, 0, 1),
          (0, 2, 0, 0), (1, 2, 2, 3)]),
        # g(a,b,c,d){a+b*c-d}: -(+(a,*(b,c)),d) with op1=sub, op0=add, op2=mul
        (b"g", [b"a", b"b", b"c", b"d"],
         [(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (1, 2, 1, 2),
          (1, 0, 0, 3), (0, 3, 0, 0), (1, 1, 4, 5)]),
    ]
    for name, params, nodes in tree_cases:
        sbuf = name + b"".join(params)
        soff = [0, len(name)]
        for p in params:
            soff.append(soff[-1] + len(p))
        sbuf_c = ctypes.create_string_buffer(sbuf, len(sbuf))
        soff_c = (ctypes.c_int64 * len(soff))(*soff)
        # Flatten nodes into a 4-i64-per-node heap array; scratch vidbuf of n_nodes i64.
        flat = [v for nd in nodes for v in nd]
        nodes_c = (ctypes.c_int64 * len(flat))(*flat)
        vidbuf_c = (ctypes.c_int64 * len(nodes))()
        es = lib.selftest_mic3_tree_fn(
            ctypes.cast(sbuf_c, ctypes.c_void_p).value,
            ctypes.cast(soff_c, ctypes.c_void_p).value,
            len(params),
            ctypes.cast(nodes_c, ctypes.c_void_p).value, len(nodes),
            ctypes.cast(vidbuf_c, ctypes.c_void_p).value)
        got = read_string_handle(read_i64_at(es, 0))
        want = ref_tree_fn(name, params, nodes)
        failures += got != want
        total += 1
        note = ""
        if name == b"mixed":
            note = " (== mixed(a,b,c,d){a*b+c*d} oracle, 77B)"
        elif name == b"f":
            note = " (== f(a,b,c){(a+b)*c} oracle, 60B)"
        ok = "OK" + note if got == want else f"FAIL want {want.hex()}"
        pnames = ",".join(p.decode() for p in params)
        print(f"  tree_fn {name.decode()}({pnames}) [{len(got)}B] = {got.hex()}  {ok}")

    # --- Phase 4g: AST-DRIVEN mic@3 emission (selftest_mic3_ast_fn) -----------
    #   Phase 4f drove emit_mic3_tree_fn_module from a SYNTHETIC hand-built node
    #   descriptor. Phase 4g lex+parses REAL source, walks the LIVE bootstrap AST,
    #   flattens it into the SAME 4-i64 post-order descriptor, and emits via the
    #   UNTOUCHED 4f emitter. So the AST path must be byte-IDENTICAL to the 4f
    #   synthetic-tree path AND to the Rust --emit-mic3 oracle for param-only
    #   bodies. We therefore reuse ref_tree_fn as the golden (ref_ast_fn == it):
    #   each case carries its real source + the expected post-order node list, and
    #   we assert the .so's AST-flattened emit == ref_tree_fn(name, params, nodes).
    #   Phase 4h: CONST LEAVES (ast_int_lit) now lower too — a const leaf is a
    #   kind-2 node (value = raw literal); its vid is allocated INTERLEAVED with the
    #   binops via the shared post-order counter and the value is zigzag-encoded at
    #   emit (lit 2 -> CONST 4). Verified byte-for-byte vs the captured --emit-mic3
    #   oracle (h(a){a*2+1}=50B, k(a,b){a*b+3}=55B, m(a){2*a+1}=50B const-LEFT).
    ref_ast_fn = ref_tree_fn  # AST path matches the synthetic-tree path byte-for-byte
    lib.selftest_mic3_ast_fn.restype = ctypes.c_void_p
    lib.selftest_mic3_ast_fn.argtypes = [ctypes.c_int64] * 7
    # Each case: (name, params, src, nodes). `src` is real MIND lexed/parsed by the
    # .so; `nodes` is the expected post-order descriptor (only feeds the golden).
    ast_cases = [
        # mixed(a,b,c,d){a*b+c*d}: +(*(a,b),*(c,d)) — 77B
        (b"mixed", [b"a", b"b", b"c", b"d"],
         b"fn mixed(a: i64, b: i64, c: i64, d: i64) -> i64 { a * b + c * d }",
         [(0, 0, 0, 0), (0, 1, 0, 0), (1, 2, 0, 1),
          (0, 2, 0, 0), (0, 3, 0, 0), (1, 2, 3, 4),
          (1, 0, 2, 5)]),
        # f(a,b,c){(a+b)*c}: *(+(a,b),c) — 60B
        (b"f", [b"a", b"b", b"c"],
         b"fn f(a: i64, b: i64, c: i64) -> i64 { (a + b) * c }",
         [(0, 0, 0, 0), (0, 1, 0, 0), (1, 0, 0, 1),
          (0, 2, 0, 0), (1, 2, 2, 3)]),
        # g(a,b,c,d){a+b*c-d}: -(+(a,*(b,c)),d) — 73B (op1=sub, op0=add, op2=mul)
        (b"g", [b"a", b"b", b"c", b"d"],
         b"fn g(a: i64, b: i64, c: i64, d: i64) -> i64 { a + b * c - d }",
         [(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (1, 2, 1, 2),
          (1, 0, 0, 3), (0, 3, 0, 0), (1, 1, 4, 5)]),
        # add(a,b,c){a+b+c}: +(+(a,b),c) left-assoc — 49B-class (3-param chain)
        (b"add", [b"a", b"b", b"c"],
         b"fn add(a: i64, b: i64, c: i64) -> i64 { a + b + c }",
         [(0, 0, 0, 0), (0, 1, 0, 0), (1, 0, 0, 1),
          (0, 2, 0, 0), (1, 0, 2, 3)]),
        # --- Phase 4h: CONST-LEAF bodies (ast_int_lit). CONST=(2, raw_literal, 0, 0).
        #     Post-order interleaving confirmed byte-for-byte vs --emit-mic3 oracle.
        # h(a){a*2+1}: +(*(a,2),1) — CONST RIGHT operand — 50B
        (b"h", [b"a"],
         b"fn h(a: i64) -> i64 { a * 2 + 1 }",
         [(0, 0, 0, 0), (2, 2, 0, 0), (1, 2, 0, 1),
          (2, 1, 0, 0), (1, 0, 2, 3)]),
        # k(a,b){a*b+3}: +(*(a,b),3) — CONST as a leaf among two params — 55B
        (b"k", [b"a", b"b"],
         b"fn k(a: i64, b: i64) -> i64 { a * b + 3 }",
         [(0, 0, 0, 0), (0, 1, 0, 0), (1, 2, 0, 1),
          (2, 3, 0, 0), (1, 0, 2, 3)]),
        # m(a){2*a+1}: +(*(2,a),1) — CONST LEFT operand (mul lhs is the const) — 50B
        (b"m", [b"a"],
         b"fn m(a: i64) -> i64 { 2 * a + 1 }",
         [(2, 2, 0, 0), (0, 0, 0, 0), (1, 2, 0, 1),
          (2, 1, 0, 0), (1, 0, 2, 3)]),
    ]
    for name, params, src, nodes in ast_cases:
        n = len(params)
        # Contiguous string-table name buffer: name then p0..p(n-1).
        strs = [name] + params
        sbuf = b"".join(strs)
        soff = [0]
        for s in strs:
            soff.append(soff[-1] + len(s))
        sbuf_c = ctypes.create_string_buffer(sbuf, len(sbuf))
        soff_c = (ctypes.c_int64 * len(soff))(*soff)
        # Real source the .so lexes/parses.
        src_c = ctypes.create_string_buffer(src, len(src))
        # Heap scratch: nodes (n_nodes*4 i64), cursor (1 i64), vidbuf (n_nodes i64).
        n_nodes = len(nodes)
        nodes_c = (ctypes.c_int64 * (n_nodes * 4))()
        cursor_c = (ctypes.c_int64 * 1)()
        vidbuf_c = (ctypes.c_int64 * n_nodes)()
        es = lib.selftest_mic3_ast_fn(
            ctypes.cast(src_c, ctypes.c_void_p).value, len(src),
            ctypes.cast(sbuf_c, ctypes.c_void_p).value,
            ctypes.cast(soff_c, ctypes.c_void_p).value,
            ctypes.cast(nodes_c, ctypes.c_void_p).value,
            ctypes.cast(cursor_c, ctypes.c_void_p).value,
            ctypes.cast(vidbuf_c, ctypes.c_void_p).value)
        got = read_string_handle(read_i64_at(es, 0))
        want = ref_ast_fn(name, params, nodes)
        failures += got != want
        total += 1
        ok = "OK" if got == want else f"FAIL want {want.hex()}"
        pnames = ",".join(p.decode() for p in params)
        print(f"  ast_fn  {name.decode()}({pnames}) [{len(got)}B] = {got.hex()}  {ok}")

    # --- Phase 4i: LET-BOUND LOCALS / multi-statement bodies (selftest_mic3_let_fn)
    #   The body is a STATEMENT SEQUENCE: zero+ `let name: ty = init;` bindings then a
    #   trailing expression. Oracle-decoded rule (verified byte-for-byte vs
    #   --emit-mic3): each let flattens its init post-order into the SAME descriptor;
    #   the init's ROOT slot vid is bound to the let-name (no instr, no strtab entry
    #   for the name). A later ident referencing a let-name returns that bound SLOT
    #   (a pure vid alias — `s*s` reuses one slot twice). The trailing expr's root
    #   slot = ret_vid. We build the post-order descriptor the SAME way the .so does
    #   (let-init nodes, then trailing-expr nodes, with let-names aliasing slots) and
    #   assert byte-for-byte via ref_tree_fn — so this reuses the proven 4f golden.
    #   letf f(a,b){let t=a*b; t+1}=55B, letg g(a){let x=a+1; let y=x*2; y}=50B,
    #   leth h(a,b){let s=a+b; s*s}=52B.
    lib.selftest_mic3_let_fn.restype = ctypes.c_void_p
    lib.selftest_mic3_let_fn.argtypes = [ctypes.c_int64] * 9
    # Each case: (name, params, src, nodes). `nodes` is the FULL post-order descriptor
    # (let-init nodes then trailing-expr nodes); LEAF=(0,param_index,0,0),
    # BINOP=(1,op_byte,left_slot,right_slot), CONST=(2,raw_literal,0,0). Slot indices
    # reference the SHARED post-order array, so a let-name reference is encoded as the
    # binop pointing at the let-init's already-emitted slot.
    let_cases = [
        # f(a,b){let t = a*b; t+1}: init a*b -> PARAM a(s0),PARAM b(s1),mul(s2);
        # trailing t+1 -> t aliases s2, CONST 1(s3), add(s2,s3)(s4). 55B.
        (b"f", [b"a", b"b"],
         b"pub fn f(a: i64, b: i64) -> i64 { let t: i64 = a * b; t + 1 }",
         [(0, 0, 0, 0), (0, 1, 0, 0), (1, 2, 0, 1),
          (2, 1, 0, 0), (1, 0, 2, 3)]),
        # g(a){let x=a+1; let y=x*2; y}: x init -> PARAM a(s0),CONST 1(s1),add(s2);
        # y init -> x aliases s2, CONST 2(s3), mul(s2,s3)(s4); trailing y aliases s4. 50B.
        (b"g", [b"a"],
         b"pub fn g(a: i64) -> i64 { let x: i64 = a + 1; let y: i64 = x * 2; y }",
         [(0, 0, 0, 0), (2, 1, 0, 0), (1, 0, 0, 1),
          (2, 2, 0, 0), (1, 2, 2, 3)]),
        # h(a,b){let s=a+b; s*s}: init a+b -> PARAM a(s0),PARAM b(s1),add(s2);
        # trailing s*s -> mul(s2,s2)(s3). 52B (ret=%3).
        (b"h", [b"a", b"b"],
         b"pub fn h(a: i64, b: i64) -> i64 { let s: i64 = a + b; s * s }",
         [(0, 0, 0, 0), (0, 1, 0, 0), (1, 0, 0, 1),
          (1, 2, 2, 2)]),
    ]
    for name, params, src, nodes in let_cases:
        n = len(params)
        strs = [name] + params
        sbuf = b"".join(strs)
        soff = [0]
        for s in strs:
            soff.append(soff[-1] + len(s))
        sbuf_c = ctypes.create_string_buffer(sbuf, len(sbuf))
        soff_c = (ctypes.c_int64 * len(soff))(*soff)
        src_c = ctypes.create_string_buffer(src, len(src))
        n_nodes = len(nodes)
        # n_lets = #non-final statements; size the let-env generously by node count.
        nodes_c = (ctypes.c_int64 * (n_nodes * 4))()
        cursor_c = (ctypes.c_int64 * 1)()
        vidbuf_c = (ctypes.c_int64 * n_nodes)()
        lenv_c = (ctypes.c_int64 * (n_nodes * 3))()
        lcount_c = (ctypes.c_int64 * 1)()
        es = lib.selftest_mic3_let_fn(
            ctypes.cast(src_c, ctypes.c_void_p).value, len(src),
            ctypes.cast(sbuf_c, ctypes.c_void_p).value,
            ctypes.cast(soff_c, ctypes.c_void_p).value,
            ctypes.cast(nodes_c, ctypes.c_void_p).value,
            ctypes.cast(cursor_c, ctypes.c_void_p).value,
            ctypes.cast(vidbuf_c, ctypes.c_void_p).value,
            ctypes.cast(lenv_c, ctypes.c_void_p).value,
            ctypes.cast(lcount_c, ctypes.c_void_p).value)
        got = read_string_handle(read_i64_at(es, 0))
        want = ref_tree_fn(name, params, nodes)
        failures += got != want
        total += 1
        note = ""
        if name == b"f":
            note = " (== f(a,b){let t=a*b; t+1} oracle, 55B)"
        elif name == b"g":
            note = " (== g(a){let x=a+1; let y=x*2; y} oracle, 50B)"
        elif name == b"h":
            note = " (== h(a,b){let s=a+b; s*s} oracle, 52B)"
        ok = "OK" + note if got == want else f"FAIL want {want.hex()}"
        pnames = ",".join(p.decode() for p in params)
        print(f"  let_fn  {name.decode()}({pnames}) [{len(got)}B] = {got.hex()}  {ok}")

    # --- Phase 4j: IF-EXPRESSION as a value (selftest_mic3_if_fn) --------------
    #   pub fn <name>(p0,..) -> i64 { if cond { then } else { else } } — the first
    #   control-flow body. Lowers to n PARAM instrs then ONE OP_IF (0x1C) instr
    #   (body_len = n + 1); ret_vid = the if's dst. Decoded byte-for-byte from the
    #   Rust --emit-mic3 oracle: cond/then/else are each a param/const/binop tree;
    #   each branch is prefixed by a synthesised ConstI64(synth, 0); the vid counter
    #   threads cond -> then(synth+tree) -> else(synth+tree) -> dst; branch_bindings
    #   and merges are both empty (no branch writes). sel(a,b){if a{b}else{a}}=60B.
    def _flatten_branch(params, expr_nodes, base):
        """Resolve a branch's post-order node list to vids from `base`.
        expr_nodes: LEAF=(0,param_idx,0,0), CONST=(2,raw,0,0), BINOP=(1,op,l,r).
        Returns (slot_vid list, root_vid, next_free_vid)."""
        slot_vid = [0] * len(expr_nodes)
        nxt = base
        for j, nd in enumerate(expr_nodes):
            if nd[0] == 0:
                slot_vid[j] = nd[1]
            else:
                slot_vid[j] = nxt
                nxt += 1
        return slot_vid, (slot_vid[-1] if slot_vid else base), nxt

    def _emit_branch_instrs(expr_nodes, slot_vid):
        out = b""
        for j, nd in enumerate(expr_nodes):
            if nd[0] == 1:
                _, ob, ls, rs = nd
                out += bytes([4]) + ref_uleb128(slot_vid[j]) + bytes([ob]) \
                    + ref_uleb128(slot_vid[ls]) + ref_uleb128(slot_vid[rs])
            elif nd[0] == 2:
                out += bytes([1]) + ref_uleb128(slot_vid[j]) + ref_uleb128(ref_zigzag(nd[1]))
        return out

    def ref_if_fn(name, params, cond, then, els):
        n = len(params)
        strs = [name] + params
        # cond (no synth), then (synth+tree), else (synth+tree), dst.
        c_sv, cond_id, after_cond = _flatten_branch(params, cond, n)
        cond_np = sum(1 for nd in cond if nd[0] != 0)
        then_synth = after_cond
        t_sv, then_root, after_then = _flatten_branch(params, then, after_cond + 1)
        then_np = sum(1 for nd in then if nd[0] != 0)
        else_synth = after_then
        e_sv, else_root, after_else = _flatten_branch(params, els, after_then + 1)
        else_np = sum(1 for nd in els if nd[0] != 0)
        dst = after_else
        op_if = bytes([0x1C]) + ref_uleb128(cond_id)
        op_if += ref_uleb128(cond_np) + _emit_branch_instrs(cond, c_sv)
        op_if += ref_uleb128(then_np + 1)
        op_if += bytes([1]) + ref_uleb128(then_synth) + ref_uleb128(0)
        op_if += _emit_branch_instrs(then, t_sv) + ref_uleb128(then_root)
        op_if += ref_uleb128(else_np + 1)
        op_if += bytes([1]) + ref_uleb128(else_synth) + ref_uleb128(0)
        op_if += _emit_branch_instrs(els, e_sv) + ref_uleb128(else_root)
        op_if += ref_uleb128(dst) + ref_uleb128(0) + ref_uleb128(0)  # dst, bb, merges
        out = b"MIC3\x02"
        out += ref_uleb128(len(strs)) + b"".join(ref_uleb128(len(x)) + x for x in strs)
        out += ref_uleb128(1) + ref_uleb128(0) + ref_uleb128(3)   # next_id, exports, 3 instrs
        out += bytes([0x15]) + ref_uleb128(0) + ref_uleb128(n)    # FN_DEF, name_idx=0
        for i in range(n):
            out += ref_uleb128(i + 1) + ref_uleb128(i)
        out += bytes([1]) + ref_uleb128(dst) + bytes([0]) + ref_uleb128(n + 1)  # ret %dst, body n+1
        for i in range(n):
            out += bytes([0x18]) + ref_uleb128(i) + ref_uleb128(i + 1) + ref_uleb128(i)
        out += op_if
        out += bytes([1]) + ref_uleb128(0) + ref_uleb128(0)       # ConstI64(%0,0)
        out += bytes([0x13]) + ref_uleb128(0)                     # Output(%0)
        out += b"\x00\x00\x00"
        return out

    lib.selftest_mic3_if_fn.restype = ctypes.c_void_p
    lib.selftest_mic3_if_fn.argtypes = [ctypes.c_int64] * 8
    # Each case: (name, params, src, cond, then, else) where cond/then/else are
    # post-order node lists. LEAF=(0,param_idx,0,0), CONST=(2,raw,0,0),
    # BINOP=(1,op_byte,left_slot,right_slot). op0=add, op1=sub, op2=mul.
    if_cases = [
        # sel(a,b){if a {b} else {a}}: cond=a, then=b, else=a — 60B (== captured oracle)
        (b"sel", [b"a", b"b"],
         b"pub fn sel(a: i64, b: i64) -> i64 { if a { b } else { a } }",
         [(0, 0, 0, 0)], [(0, 1, 0, 0)], [(0, 0, 0, 0)]),
        # sel(a,b){if a {1} else {2}}: const branches
        (b"sel", [b"a", b"b"],
         b"pub fn sel(a: i64, b: i64) -> i64 { if a { 1 } else { 2 } }",
         [(0, 0, 0, 0)], [(2, 1, 0, 0)], [(2, 2, 0, 0)]),
        # sel(a,b){if a {b+a} else {a}}: then = b+a (BINOP), else = a
        (b"sel", [b"a", b"b"],
         b"pub fn sel(a: i64, b: i64) -> i64 { if a { b + a } else { a } }",
         [(0, 0, 0, 0)], [(0, 1, 0, 0), (0, 0, 0, 0), (1, 0, 0, 1)], [(0, 0, 0, 0)]),
        # f(a,b,c){if a {b*c} else {c}}: 3 params, then = b*c (mul), else = c
        (b"f", [b"a", b"b", b"c"],
         b"pub fn f(a: i64, b: i64, c: i64) -> i64 { if a { b * c } else { c } }",
         [(0, 0, 0, 0)], [(0, 1, 0, 0), (0, 2, 0, 0), (1, 2, 0, 1)], [(0, 2, 0, 0)]),
    ]
    for name, params, src, cond, then, els in if_cases:
        n = len(params)
        strs = [name] + params
        sbuf = b"".join(strs)
        soff = [0]
        for s in strs:
            soff.append(soff[-1] + len(s))
        sbuf_c = ctypes.create_string_buffer(sbuf, len(sbuf))
        soff_c = (ctypes.c_int64 * len(soff))(*soff)
        src_c = ctypes.create_string_buffer(src, len(src))
        nodes_c = (ctypes.c_int64 * (64 * 4))()
        cursor_c = (ctypes.c_int64 * 1)()
        vidbuf_c = (ctypes.c_int64 * 64)()
        ok_c = (ctypes.c_int64 * 1)()
        es = lib.selftest_mic3_if_fn(
            ctypes.cast(src_c, ctypes.c_void_p).value, len(src),
            ctypes.cast(sbuf_c, ctypes.c_void_p).value,
            ctypes.cast(soff_c, ctypes.c_void_p).value,
            ctypes.cast(nodes_c, ctypes.c_void_p).value,
            ctypes.cast(cursor_c, ctypes.c_void_p).value,
            ctypes.cast(vidbuf_c, ctypes.c_void_p).value,
            ctypes.cast(ok_c, ctypes.c_void_p).value)
        got = read_string_handle(read_i64_at(es, 0))
        want = ref_if_fn(name, params, cond, then, els)
        failures += got != want
        total += 1
        note = " (== sel(a,b){if a {b} else {a}} oracle, 60B)" \
            if (name, src.count(b"{ b }")) == (b"sel", 1) and src.endswith(b"{ a } }") else ""
        ok = "OK" + note if got == want else f"FAIL want {want.hex()}"
        pnames = ",".join(p.decode() for p in params)
        print(f"  if_fn   {name.decode()}({pnames}) [{len(got)}B] = {got.hex()}  {ok}")

    # --- Phase 4j+: IF with MULTI-STATEMENT (let-in-branch) blocks ------------
    #   pub fn <name>(p..) -> i64 { if cond { <stmt-block> } else { <stmt-block> } }
    #   where a branch may be a STATEMENT SEQUENCE `{ let n: ty = init; ..; trail }`.
    #   A branch-local `let` is a WRITE, which the Rust oracle turns into a NON-EMPTY
    #   merge phi set (branch_bindings + merges). Emitted by the pure-MIND
    #   selftest_mic3_if_block_fn (main.mind), cross-checked BYTE-EXACT against the
    #   real `mindc --emit-mic3` oracle (hard-coded golden below, AND regenerated
    #   live when target/release/mindc is available). Isolated from the canary
    #   (AST body emitter still rejects ast_if), so the fixed point is unchanged.
    import subprocess

    _MINDC = _HERE.parent.parent / "target" / "release" / "mindc"

    def _live_oracle(src_text):
        """Regenerate the real mic@3 oracle via `mindc --emit-mic3`, or None."""
        if not _MINDC.exists():
            return None
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            srcp = pathlib.Path(td) / "case.mind"
            outp = pathlib.Path(td) / "case.mic3"
            srcp.write_text(src_text)
            r = subprocess.run(
                [str(_MINDC), "--emit-mic3", str(outp), str(srcp)],
                capture_output=True,
            )
            if r.returncode != 0 or not outp.exists():
                return None
            return outp.read_bytes()

    lib.selftest_mic3_if_block_fn.restype = ctypes.c_void_p
    lib.selftest_mic3_if_block_fn.argtypes = [ctypes.c_int64] * 22

    # Each case: (name, params, then_lets, else_lets, src, golden_hex).
    #   then_lets/else_lets are the branch-local let-names in encounter order; the
    #   contiguous string table is [name, *params, *then_lets, *else-only-lets].
    #   golden_hex = the captured real `mindc --emit-mic3` output (the oracle).
    block_cases = [
        # g(a,b){ if a { let t=b+a; t } else { b } } — then-let only, else=param. 73B.
        (b"g", [b"a", b"b"], [b"t"], [],
         "fn g(a: i64, b: i64) -> i64 { if a { let t: i64 = b + a; t } else { b } }",
         "4d49433302040167016101620174010003150002010002010107000318000100"
         "180102011c000002010200040300010003020104000105000107010306010603"
         "050100001300000000"),
        # f(a,b){ if a { let t=a*b; t+1 } else { let u=b; u } } — both branches let. 91B.
        (b"f", [b"a", b"b"], [b"t"], [b"u"],
         "fn f(a: i64, b: i64) -> i64 { if a { let t: i64 = a * b; t + 1 } "
         "else { let u: i64 = b; u } }",
         "4d49433302050166016101620174017501000315000201000201010b00031800"
         "0100180102011c000005010200040302000101040204050003040109000502010"
         "600010700010b020308040a020803070a09010100001300000000".replace(
             " ", "")),
        # h(a,b){ if a { let t=a*b; t+1 } else { b } } — then 2-stmt let, else=param. 81B.
        (b"h", [b"a", b"b"], [b"t"], [],
         "fn h(a: i64, b: i64) -> i64 { if a { let t: i64 = a * b; t + 1 } "
         "else { b } }",
         "4d49433302040168016101620174010003150002010002010109000318000100"
         "180102011c0000040102000403020001010402040500030405020106000107000"
         "109010308010803070100001300000000".replace(" ", "")),
        # k(a,b){ if a { let x=a+1; let y=x*2; y } else { b } } — two names in then. 94B.
        (b"k", [b"a", b"b"], [b"x", b"y"], [],
         "fn k(a: i64, b: i64) -> i64 { if a { let x: i64 = a + 1; "
         "let y: i64 = x * 2; y } else { b } }",
         "4d4943330205016b016101620178017901000315000201000201010c00031800"
         "0100180102011c000005010200010302040400000301050404060204050603010"
         "700010800010a00010c020309040b020904080b060a0100001300000000".replace(
             " ", "")),
    ]
    for name, params, then_lets, else_lets, src, golden_hex in block_cases:
        # else_lets restricted to names not already in then (first-seen dedup).
        else_only = [u for u in else_lets if u not in then_lets]
        strs = [name] + params + then_lets + else_only
        sbuf = b"".join(strs)
        soff = [0]
        for s in strs:
            soff.append(soff[-1] + len(s))
        strcount = len(strs)
        srcb = src.encode()

        sbuf_c = ctypes.create_string_buffer(sbuf, max(1, len(sbuf)))
        soff_c = (ctypes.c_int64 * len(soff))(*soff)
        src_c = ctypes.create_string_buffer(srcb, len(srcb))

        # Per-branch scratch: nodes 4 i64/node, lenv 3 i64/entry, vidbuf 1 i64/slot.
        t_nodes_c = (ctypes.c_int64 * (64 * 4))()
        t_cur_c = (ctypes.c_int64 * 1)()
        t_vid_c = (ctypes.c_int64 * 64)()
        t_lenv_c = (ctypes.c_int64 * (16 * 3))()
        t_lc_c = (ctypes.c_int64 * 1)()
        e_nodes_c = (ctypes.c_int64 * (64 * 4))()
        e_cur_c = (ctypes.c_int64 * 1)()
        e_vid_c = (ctypes.c_int64 * 64)()
        e_lenv_c = (ctypes.c_int64 * (16 * 3))()
        e_lc_c = (ctypes.c_int64 * 1)()
        c_nodes_c = (ctypes.c_int64 * (64 * 4))()
        c_cur_c = (ctypes.c_int64 * 1)()
        c_vid_c = (ctypes.c_int64 * 64)()
        # Merge scratch: mspans 2 i64/name, extra 5 i64/name.
        mspans_c = (ctypes.c_int64 * (16 * 2))()
        mcount_c = (ctypes.c_int64 * 1)()
        extra_c = (ctypes.c_int64 * (16 * 5))()
        ok_c = (ctypes.c_int64 * 1)()

        def _p(b):
            return ctypes.cast(b, ctypes.c_void_p).value

        es = lib.selftest_mic3_if_block_fn(
            _p(src_c), len(srcb),
            _p(sbuf_c), _p(soff_c), strcount,
            _p(t_nodes_c), _p(t_cur_c), _p(t_vid_c), _p(t_lenv_c), _p(t_lc_c),
            _p(e_nodes_c), _p(e_cur_c), _p(e_vid_c), _p(e_lenv_c), _p(e_lc_c),
            _p(c_nodes_c), _p(c_cur_c), _p(c_vid_c),
            _p(mspans_c), _p(mcount_c), _p(extra_c), _p(ok_c))
        got = read_string_handle(read_i64_at(es, 0))

        # The oracle: prefer a live regeneration; fall back to the captured golden.
        # When both are available they MUST agree (guards golden staleness).
        golden = bytes.fromhex(golden_hex)
        live = _live_oracle(src)
        if live is not None and live != golden:
            raise SystemExit(
                f"FAIL: captured golden for {name.decode()} is stale vs live "
                f"oracle\n  golden {golden.hex()}\n  live   {live.hex()}")
        want = live if live is not None else golden

        failures += got != want
        total += 1
        ok = "OK (real-oracle byte-exact)" if got == want \
            else f"FAIL want {want.hex()}"
        pnames = ",".join(p.decode() for p in params)
        print(f"  if_blk  {name.decode()}({pnames}) [{len(got)}B] = {got.hex()}  {ok}")

    # --- Phase 5: COMPLETE MULTI-FUNCTION module (Strategy A foundation) -------
    #   The first pure-MIND emit of a module with MORE THAN ONE function. The four
    #   multi-function deltas vs the single-fn families above: (1) ONE string table
    #   deduped in first-seen traversal order; (2) ONE flat next_id threaded across
    #   all fns + scaffolds; (3) ONE top-level instr list with a single uleb count;
    #   (4) exports sorted (empty here — export-free fixtures). Each case is
    #   cross-checked BYTE-EXACT against a hard-coded oracle AND, when
    #   target/release/mindc is present, against a LIVE `--emit-mic3` regeneration
    #   (guards golden staleness). Additive: does NOT touch the canary mic@1 path.

    # (5a) two const-return fns: `pub fn one(){1} pub fn two(){2}` — 50B oracle.
    lib.selftest_mic3_module_2fn.restype = ctypes.c_void_p
    lib.selftest_mic3_module_2fn.argtypes = [ctypes.c_int64] * 4
    twofn_cases = [
        # (strings-first-seen, v0, v1, src, golden_hex)
        ([b"one", b"two"], 1, 2,
         "pub fn one() -> i64 { 1 }  pub fn two() -> i64 { 2 }",
         "4d4943330202036f6e650374776f0200061500000100000101000201000013001"
         "50100010000010100040101001301000000"),
        # value-variant (4, 10) proves a genuine narrow lowering, not a byte blob.
        ([b"one", b"two"], 4, 10,
         "pub fn one() -> i64 { 4 }  pub fn two() -> i64 { 10 }",
         None),
    ]
    for strs, v0, v1, src, golden_hex in twofn_cases:
        sbuf = b"".join(strs)
        soff = [0]
        for s in strs:
            soff.append(soff[-1] + len(s))
        sbuf_c = ctypes.create_string_buffer(sbuf, len(sbuf))
        soff_c = (ctypes.c_int64 * len(soff))(*soff)
        es = lib.selftest_mic3_module_2fn(
            ctypes.cast(sbuf_c, ctypes.c_void_p).value,
            ctypes.cast(soff_c, ctypes.c_void_p).value, v0, v1)
        got = read_string_handle(read_i64_at(es, 0))
        golden = bytes.fromhex(golden_hex) if golden_hex else None
        live = _live_oracle(src)
        if golden is not None and live is not None and live != golden:
            raise SystemExit(
                f"FAIL: 2fn golden stale vs live\n  golden {golden.hex()}\n"
                f"  live   {live.hex()}")
        want = live if live is not None else golden
        if want is None:
            raise SystemExit("FAIL: no oracle for 2fn case (no golden, no mindc)")
        failures += got != want
        total += 1
        note = " (== one(){1} two(){2} oracle, 50B)" if (v0, v1) == (1, 2) else ""
        ok = "OK" + note if got == want else f"FAIL want {want.hex()}"
        print(f"  mod_2fn one={v0},two={v1} [{len(got)}B] = {got.hex()}  {ok}")

    # (5b) const fn0 + arith fn1: `pub fn one(){1} pub fn add(a,b){a+b}` — 68B oracle.
    #   The two functions have DIFFERENT body shapes (const vs PARAM/PARAM/BINOP)
    #   under one module header, with flat scaffold vids %0/%1.
    lib.selftest_mic3_module_const_arith.restype = ctypes.c_void_p
    lib.selftest_mic3_module_const_arith.argtypes = [ctypes.c_int64] * 4
    ca_cases = [
        # (strings-first-seen, v0, op_byte, src, golden_hex). op0=add.
        ([b"one", b"add", b"a", b"b"], 1, 0,
         "pub fn one() -> i64 { 1 }  pub fn add(a: i64, b: i64) -> i64 { a + b }",
         "4d4943330204036f6e6503616464016101620200061500000100000101000201"
         "000013001501020200030101020003180002001801030104020000010101001301"
         "000000"),
        # mul variant (v0=7, op2=mul) — genuine lowering across const + op.
        ([b"one", b"mul", b"x", b"y"], 7, 2,
         "pub fn one() -> i64 { 7 }  pub fn mul(x: i64, y: i64) -> i64 { x * y }",
         None),
    ]
    for strs, v0, op_byte, src, golden_hex in ca_cases:
        sbuf = b"".join(strs)
        soff = [0]
        for s in strs:
            soff.append(soff[-1] + len(s))
        sbuf_c = ctypes.create_string_buffer(sbuf, len(sbuf))
        soff_c = (ctypes.c_int64 * len(soff))(*soff)
        es = lib.selftest_mic3_module_const_arith(
            ctypes.cast(sbuf_c, ctypes.c_void_p).value,
            ctypes.cast(soff_c, ctypes.c_void_p).value, v0, op_byte)
        got = read_string_handle(read_i64_at(es, 0))
        golden = bytes.fromhex(golden_hex) if golden_hex else None
        live = _live_oracle(src)
        if golden is not None and live is not None and live != golden:
            raise SystemExit(
                f"FAIL: const_arith golden stale vs live\n  golden {golden.hex()}"
                f"\n  live   {live.hex()}")
        want = live if live is not None else golden
        if want is None:
            raise SystemExit("FAIL: no oracle for const_arith case")
        failures += got != want
        total += 1
        note = " (== one(){1} add(a,b){a+b} oracle, 68B)" \
            if (v0, op_byte) == (1, 0) else ""
        ok = "OK" + note if got == want else f"FAIL want {want.hex()}"
        print(f"  mod_carit v0={v0},op={op_byte} [{len(got)}B] = {got.hex()}  {ok}")

    # --- Phase 5c: AST-DRIVEN full-module COLLECTION PASS (hand-feeding removal) -
    #   selftest_mic3_module_from_ast takes ONLY the source bytes (+ heap scratch:
    #   a contiguous string buffer, an offsets array, and a 1-i64 count cell). It
    #   lexes/parses, then AUTOMATICALLY builds the deduped first-seen-traversal
    #   string table + offsets (the hand-feeding the 5a/5b cases above still rely
    #   on is gone) and extracts the const values / arith op-byte from the live
    #   AST, driving the proven 5a/5b assemblers. Each case is cross-checked
    #   BYTE-EXACT against a hard-coded golden AND a LIVE `--emit-mic3`
    #   regeneration (guards golden staleness). Additive: the canary mic@1 path
    #   is untouched, so the fixed point stays byte-identical.
    lib.selftest_mic3_module_from_ast.restype = ctypes.c_void_p
    lib.selftest_mic3_module_from_ast.argtypes = [ctypes.c_int64] * 5

    ast_cases = [
        # (src, golden_hex). The first two are the known 50B / 68B targets.
        ("pub fn one() -> i64 { 1 }  pub fn two() -> i64 { 2 }",
         "4d4943330202036f6e650374776f0200061500000100000101000201000013001"
         "50100010000010100040101001301000000"),
        ("pub fn one() -> i64 { 1 }  pub fn add(a: i64, b: i64) -> i64 { a + b }",
         "4d4943330204036f6e6503616464016101620200061500000100000101000201"
         "000013001501020200030101020003180002001801030104020000010101001301"
         "000000"),
        # value/name variants — prove a genuine collection pass, not a byte blob.
        ("pub fn one() -> i64 { 4 }  pub fn two() -> i64 { 10 }", None),
        ("pub fn first() -> i64 { 99 }  pub fn second() -> i64 { 100 }", None),
        ("pub fn one() -> i64 { 7 }  pub fn mul(x: i64, y: i64) -> i64 { x * y }",
         None),
        # Strategy A — LET-body coverage. fn0 is a 2-param arith fn written with a
        # `let t = a OP b; t` body; the Rust lowering reuses the binop vid for the
        # let-bound value, so this is BYTE-IDENTICAL to the direct `{ a OP b }`
        # form, i.e. the arith-first + const-second module shape (64B oracle).
        ("pub fn f(a: i64, b: i64) -> i64 { let t: i64 = a + b; t }  "
         "pub fn g() -> i64 { 1 }",
         "4d4943330204016601610162016702000615000201000201010200031800010018"
         "0102010402000001010000130015030001000001010002010100130100" + "0000"),
        # value/op variant of the let-body shape (genuine collection, not a blob).
        ("pub fn h(x: i64, y: i64) -> i64 { let s: i64 = x * y; s }  "
         "pub fn k() -> i64 { 5 }",
         None),
        # Strategy A — CALL-body coverage. fn1 calls fn0 with two int-literal args;
        # each arg lowers to a body ConstI64 (%0,%1) then OP_CALL %2 = add(%0,%1),
        # the callee name DEDUPing to fn0's string index 0 (80B oracle).
        ("pub fn add(a: i64, b: i64) -> i64 { a + b }  "
         "pub fn use_it() -> i64 { add(2, 3) }",
         "4d49433302040361646401610162067573655f697402000615000201000201010200"
         "0318000100180102010402000001010000130015030001020003010004010106160200"
         "0200010101001301" + "000000"),
        # value/op variant of the call-body shape.
        ("pub fn sub(a: i64, b: i64) -> i64 { a - b }  "
         "pub fn run() -> i64 { sub(9, 4) }",
         None),
        # Strategy A — COMPARISON-op body coverage. The six comparison ops lower
        # exactly like an arith binop but with their own mic@3 op-byte (Lt=0x05,
        # Le=0x06, Gt=0x07, Ge=0x08, Eq=0x09, Ne=0x0A — src/ir/compact/v3/emit.rs
        # binop_to_byte). The MIND op-tag order (lt,gt,eq,le,ge,ne) differs from
        # that byte order, so main.mind's binop_to_byte maps each explicitly. Each
        # is the arith-fn0 + const-fn1 shape (65B), cross-checked byte-exact vs a
        # hard-coded golden AND the live --emit-mic3 oracle. The op-byte is the
        # third byte of the BINOP `04 02 XX 00 01` near the body tail.
        ("pub fn lt(a: i64, b: i64) -> i64 { a < b }  pub fn g() -> i64 { 1 }",
         "4d4943330204026c740161016201670200061500020100020101020003180001"
         "001801020104020500010100001300150300010000010100020101001301" + "000000"),
        ("pub fn le(a: i64, b: i64) -> i64 { a <= b }  pub fn g() -> i64 { 1 }",
         "4d4943330204026c650161016201670200061500020100020101020003180001"
         "001801020104020600010100001300150300010000010100020101001301" + "000000"),
        ("pub fn gt(a: i64, b: i64) -> i64 { a > b }  pub fn g() -> i64 { 1 }",
         "4d494333020402677401610162016702000615000201000201010200031800010"
         "01801020104020700010100001300150300010000010100020101001301" + "000000"),
        ("pub fn ge(a: i64, b: i64) -> i64 { a >= b }  pub fn g() -> i64 { 5 }",
         "4d49433302040267650161016201670200061500020100020101020003180001"
         "0018010201040208000101000013001503000100000101000a0101001301" + "000000"),
        ("pub fn eq(a: i64, b: i64) -> i64 { a == b }  pub fn g() -> i64 { 1 }",
         "4d49433302040265710161016201670200061500020100020101020003180001"
         "001801020104020900010100001300150300010000010100020101001301" + "000000"),
        ("pub fn ne(a: i64, b: i64) -> i64 { a != b }  pub fn g() -> i64 { 5 }",
         "4d4943330204026e650161016201670200061500020100020101020003180001"
         "001801020104020a000101000013001503000100000101000a0101001301" + "000000"),
        # let-bound comparison: `{ let t = a > b; t }` is byte-identical to the
        # direct `{ a > b }` form (the let reuses the binop vid), exercising the
        # let-body comparison path (let_arith_fn_opbyte -> binop_to_byte).
        ("pub fn gt(a: i64, b: i64) -> i64 { let t: i64 = a > b; t }  "
         "pub fn g() -> i64 { 2 }", None),
        # comparison fn0 + const-args CALL fn1 (le<=, 76B) — comparison op through
        # the arith+call module shape.
        ("pub fn le(a: i64, b: i64) -> i64 { a <= b }  "
         "pub fn run() -> i64 { le(9, 4) }", None),
    ]
    for src, golden_hex in ast_cases:
        srcb = src.encode()
        srcc = ctypes.create_string_buffer(srcb, len(srcb))
        # Scratch: a contiguous strbuf (+8B pad for the word-granular over-read),
        # an offsets array, and a 1-i64 interned-count cell — all caller-owned.
        strbuf = ctypes.create_string_buffer(128)
        offs = (ctypes.c_int64 * 16)()
        ccell = (ctypes.c_int64 * 1)()
        es = lib.selftest_mic3_module_from_ast(
            ctypes.cast(srcc, ctypes.c_void_p).value, len(srcb),
            ctypes.cast(strbuf, ctypes.c_void_p).value,
            ctypes.cast(offs, ctypes.c_void_p).value,
            ctypes.cast(ccell, ctypes.c_void_p).value)
        got = read_string_handle(read_i64_at(es, 0))
        golden = bytes.fromhex(golden_hex) if golden_hex else None
        live = _live_oracle(src)
        if golden is not None and live is not None and live != golden:
            raise SystemExit(
                f"FAIL: from_ast golden stale vs live\n  golden {golden.hex()}"
                f"\n  live   {live.hex()}")
        want = live if live is not None else golden
        if want is None:
            raise SystemExit("FAIL: no oracle for from_ast case")
        failures += got != want
        total += 1
        ok = "OK (AST-driven, byte-exact vs --emit-mic3)" \
            if got == want else f"FAIL want {want.hex()}"
        print(f"  mod_ast [{len(got)}B] = {got.hex()}  {ok}")

    # --- Phase 5d: N-FUNCTION AST-driven module assembly (Strategy A inc. 5) -----
    #   selftest_mic3_module_nfn generalizes the fixed 2-function module driver to
    #   an ARBITRARY function count (3, 4, .. N). It lexes/parses the source, builds
    #   the deduped first-seen string table, and assembles the whole module —
    #   next_id = N, instr_count = 3*N, per fn k: FN_DEF{body} || ConstI64(%k,0) ||
    #   Output(%k) — byte-exact vs the Rust --emit-mic3 oracle. Each case is
    #   cross-checked against a hard-coded golden AND a LIVE `--emit-mic3`
    #   regeneration (guards golden staleness). Additive: the canary mic@1 path is
    #   untouched, so the fixed point stays byte-identical.
    lib.selftest_mic3_module_nfn.restype = ctypes.c_void_p
    lib.selftest_mic3_module_nfn.argtypes = [ctypes.c_int64] * 5

    nfn_cases = [
        # 3 zero-arg const-return fns — next_id 3, instr_count 9 (71B oracle).
        ("pub fn one() -> i64 { 1 }  pub fn two() -> i64 { 2 }  "
         "pub fn three() -> i64 { 3 }",
         "4d4943330203036f6e650374776f0574687265650300091500000100000101"
         "00020100001300150100010000010100040101001301150200010000010100"
         "060102001302000000"),
        # 4 zero-arg const-return fns — next_id 4, instr_count 12 (91B oracle).
        ("pub fn one() -> i64 { 1 }  pub fn two() -> i64 { 2 }  "
         "pub fn three() -> i64 { 3 }  pub fn four() -> i64 { 4 }",
         "4d4943330204036f6e650374776f05746872656504666f757204000c150000"
         "010000010100020100001300150100010000010100040101001301150200010"
         "000010100060102001302150300010000010100080103001303000000"),
        # 3 MIXED-body fns: arith fn0 + const fn1 + CALL fn2 (the call to fn0
        # DEDUPs to its string index 0). next_id 3, instr_count 9 (99B oracle).
        ("pub fn add(a: i64, b: i64) -> i64 { a + b }  "
         "pub fn one() -> i64 { 1 }  pub fn use_it() -> i64 { add(2, 3) }",
         "4d49433302050361646401610162036f6e65067573655f6974030009150002"
         "0100020101020003180001001801020104020000010100001300150300010000"
         "010100020101001301150400010200030100040101061602000200010102001302"
         "000000"),
        # value/name variants — prove a genuine N-fn collection pass, not a blob.
        ("pub fn first() -> i64 { 99 }  pub fn second() -> i64 { 100 }  "
         "pub fn third() -> i64 { 7 }", None),
        ("pub fn a() -> i64 { 1 }  pub fn b() -> i64 { 2 }  "
         "pub fn c() -> i64 { 3 }  pub fn d() -> i64 { 4 }  "
         "pub fn e() -> i64 { 5 }", None),
        # 4-fn mixed: two arith + const + call (call dedups to sub's index 0).
        ("pub fn sub(a: i64, b: i64) -> i64 { a - b }  "
         "pub fn mul(x: i64, y: i64) -> i64 { x * y }  "
         "pub fn z() -> i64 { 8 }  pub fn run() -> i64 { sub(9, 4) }", None),
        # 2-fn regression THROUGH the N-fn path (must still match the 50B/68B
        # oracles the fixed 2-fn driver produces).
        ("pub fn one() -> i64 { 1 }  pub fn two() -> i64 { 2 }", None),
        ("pub fn one() -> i64 { 1 }  pub fn add(a: i64, b: i64) -> i64 { a + b }",
         None),
    ]
    for src, golden_hex in nfn_cases:
        srcb = src.encode()
        srcc = ctypes.create_string_buffer(srcb, len(srcb))
        strbuf = ctypes.create_string_buffer(256)
        offs = (ctypes.c_int64 * 32)()
        ccell = (ctypes.c_int64 * 1)()
        es = lib.selftest_mic3_module_nfn(
            ctypes.cast(srcc, ctypes.c_void_p).value, len(srcb),
            ctypes.cast(strbuf, ctypes.c_void_p).value,
            ctypes.cast(offs, ctypes.c_void_p).value,
            ctypes.cast(ccell, ctypes.c_void_p).value)
        got = read_string_handle(read_i64_at(es, 0))
        golden = bytes.fromhex(golden_hex) if golden_hex else None
        live = _live_oracle(src)
        if golden is not None and live is not None and live != golden:
            raise SystemExit(
                f"FAIL: nfn golden stale vs live\n  golden {golden.hex()}"
                f"\n  live   {live.hex()}")
        want = live if live is not None else golden
        if want is None:
            raise SystemExit("FAIL: no oracle for nfn case")
        failures += got != want
        total += 1
        ok = "OK (N-fn AST-driven, byte-exact vs --emit-mic3)" \
            if got == want else f"FAIL want {want.hex()}"
        print(f"  mod_nfn [{len(got)}B] = {got.hex()}  {ok}")

    # --- Strategy A inc. 6: ARBITRARY expr-tree / if BODIES in the N-fn path ------
    #   Each function body may now be an arbitrary param/const/binop expression tree
    #   OR a single `if cond { then } else { else }` value (single-expression
    #   branches), not just the four narrow shapes. The N-fn module invariants are
    #   preserved (next_id = N, instr_count = 3*N, union first-seen string table,
    #   per fn k: FN_DEF{body} || ConstI64(%k,0) || Output(%k)); only each fn's
    #   FN_DEF body now routes to the proven 4f tree emitter / 4j OP_IF emitter.
    #   Every case is cross-checked vs a hard-coded golden AND a LIVE `--emit-mic3`
    #   regeneration (golden-staleness guard). Additive: the canary mic@1 path is
    #   untouched, so the fixed point stays byte-identical.
    nfn_body_cases = [
        # 2 fns: arbitrary tree `a*b + c` (3 params) + tree with a REPEATED param
        # `(x+1)*(x-1)` (x emitted as ONE PARAM, body_len de-duplicated).
        ("pub fn f(a: i64, b: i64, c: i64) -> i64 { a * b + c }  "
         "pub fn g(x: i64) -> i64 { (x + 1) * (x - 1) }",
         "4d4943330206016601610162016301670178020006150003010002010302"
         "010400051800010018010201180203020403020001040400030201000013"
         "001504010500010500061800050001010204020000010103020404010003"
         "04050202040101001301000000"),
        # 2 fns: if-expression body (single-expr branches, empty merges/bindings)
        # + a plain arith fn (the if's dst threads after both synth consts).
        ("pub fn sel(a: i64, b: i64) -> i64 { if a { b } else { a } }  "
         "pub fn add(a: i64, b: i64) -> i64 { a + b }",
         "4d49433302040373656c0161016203616464020006150002010002010104"
         "000318000100180102011c00000101020001010103000004000001000013"
         "001503020100020101020003180001001801020104020000010101001301"
         "000000"),
        # 3-fn MIX — one arbitrary tree, one if, one zero-arg const: the qualitative
        # leap (a heterogeneous module assembled fn-by-fn from three body kinds).
        ("pub fn tree(a: i64, b: i64) -> i64 { a * b + a }  "
         "pub fn br(a: i64, b: i64) -> i64 { if a { b } else { a } }  "
         "pub fn k() -> i64 { 7 }",
         "4d4943330205047472656501610162026272016b03000915000201000201010300"
         "041800010018010201040202000104030002000100001300150302010002010104"
         "000318000100180102011c0000010102000101010300000400000101001301150400"
         "0100000101000e0102001302000000"),
    ]
    for src, golden_hex in nfn_body_cases:
        srcb = src.encode()
        srcc = ctypes.create_string_buffer(srcb, len(srcb))
        strbuf = ctypes.create_string_buffer(512)
        offs = (ctypes.c_int64 * 64)()
        ccell = (ctypes.c_int64 * 1)()
        es = lib.selftest_mic3_module_nfn(
            ctypes.cast(srcc, ctypes.c_void_p).value, len(srcb),
            ctypes.cast(strbuf, ctypes.c_void_p).value,
            ctypes.cast(offs, ctypes.c_void_p).value,
            ctypes.cast(ccell, ctypes.c_void_p).value)
        got = read_string_handle(read_i64_at(es, 0))
        golden = bytes.fromhex(golden_hex)
        live = _live_oracle(src)
        if live is not None and live != golden:
            raise SystemExit(
                f"FAIL: nfn-body golden stale vs live\n  golden {golden.hex()}"
                f"\n  live   {live.hex()}")
        want = live if live is not None else golden
        failures += got != want
        total += 1
        ok = "OK (tree/if body, byte-exact vs --emit-mic3)" \
            if got == want else f"FAIL want {want.hex()}"
        print(f"  mod_body [{len(got)}B] = {got.hex()}  {ok}")

    # --- Strategy A inc. 7: FIELD-ACCESS (struct_defs registry + field-idx load) -
    #   `struct S { a, b }  fn get_a(s){s.a}  fn get_b(s){s.b}` — the first pure-MIND
    #   emit of (1) a NON-EMPTY std-surface struct_defs registry and (2) the
    #   field-access lowering. A field READ resolves via struct_defs[type] to a
    #   field INDEX, then lowers to `__mind_load_i64(base [+ idx*8])`:
    #     idx 0 -> PARAM %0 ; CALL %1=load([%0])             (direct, no offset)
    #     idx 1 -> PARAM %0 ; CONST %1=8 ; BINOP %2=add(%0,%1) ; CALL %3=load([%2])
    #   The struct decl AND each fn each emit a module-level ConstI64/Output scaffold
    #   (flat next_id = 3). strtab[7] first-seen: get_a, s, __mind_load_i64, get_b,
    #   then the registry strings S, a, b. Cross-checked vs a hard-coded golden AND a
    #   LIVE `--emit-mic3` regeneration (golden-staleness guard). Additive: the canary
    #   mic@1 path (emit_expr_env ast_field -> const.i64 0) is untouched, so the fixed
    #   point stays byte-identical.
    field_src = (
        "struct S { a: i64, b: i64 }\n"
        "pub fn get_a(s: S) -> i64 { s.a }\n"
        "pub fn get_b(s: S) -> i64 { s.b }\n"
    )
    field_golden_hex = (
        "4d4943330207056765745f6101730f5f5f6d696e645f6c6f61645f693634056765"
        "745f62015301610162030008010000130015000101000101000218000100160102"
        "010001010013011503010100010300041800010001011004020000011603020102"
        "010200130201040205060000"
    )
    field_strs = [b"get_a", b"s", b"__mind_load_i64", b"get_b", b"S", b"a", b"b"]
    field_sbuf = b"".join(field_strs)
    field_soff = [0]
    for s in field_strs:
        field_soff.append(field_soff[-1] + len(s))
    field_sbuf_c = ctypes.create_string_buffer(field_sbuf, len(field_sbuf))
    field_soff_c = (ctypes.c_int64 * len(field_soff))(*field_soff)
    lib.selftest_mic3_module_field.restype = ctypes.c_void_p
    lib.selftest_mic3_module_field.argtypes = [ctypes.c_int64] * 4
    # load_idx = string index of "__mind_load_i64" (=2); foff = field-b byte offset
    # (idx 1 * 8 = 8) — threaded so the emit is a genuine lowering, not a blob.
    es = lib.selftest_mic3_module_field(
        ctypes.cast(field_sbuf_c, ctypes.c_void_p).value,
        ctypes.cast(field_soff_c, ctypes.c_void_p).value,
        2, 8)
    got = read_string_handle(read_i64_at(es, 0))
    golden = bytes.fromhex(field_golden_hex)
    live = _live_oracle(field_src)
    if live is not None and live != golden:
        raise SystemExit(
            f"FAIL: field golden stale vs live\n  golden {golden.hex()}"
            f"\n  live   {live.hex()}")
    want = live if live is not None else golden
    failures += got != want
    total += 1
    ok = "OK (struct_defs + field load, byte-exact vs --emit-mic3, 111B)" \
        if got == want else f"FAIL want {want.hex()}"
    print(f"  field    [{len(got)}B] = {got.hex()}  {ok}")

    if failures:
        raise SystemExit(f"FAIL: {failures}/{total} mic@3 primitive mismatches")
    print(f"  PASS — {total}/{total} byte-exact vs reference "
          f"(uleb128 + header + string-table + instr emit + multi-fn module)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
