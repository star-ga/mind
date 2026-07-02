#!/usr/bin/env python3
# Reference-vector driver for std/http2_frame.mind (pure-MIND HTTP/2 framing,
# RFC 9113 §3.4 / §4.1 / §6 — HTTP-provable-stack Phase 2.2).
#
# Ground truth: the `hyperframe` library (the reference h2 frame codec used by
# python-hyper/h2).  Every expected byte string is produced by hyperframe (or,
# for the connection preface which hyperframe does not model, hardcoded from
# RFC 9113 §3.4) and asserted for internal consistency against the hand-derived
# RFC 9113 §4.1 bit layout BEFORE any MIND output is compared.
#
# Cases:
#   (a) SETTINGS payload parse == the pairs hyperframe encoded (§6.5)
#   (b) frame-header write == hyperframe's serialized 9 header bytes for
#       several (length, type, flags, stream_id) incl. a large 24-bit length
#       and the max 31-bit stream id; R bit zeroed on send (§4.1)
#   (c) preface write == the exact 24 magic bytes; check accepts them and
#       rejects a corrupted preface (§3.4)
#   (d) round-trips parse(write(x)) == x for header, SETTINGS, WINDOW_UPDATE,
#       GOAWAY, RST_STREAM, PING — with hyperframe re-parsing MIND's bytes as
#       an independent cross-check (§6.4, §6.5, §6.7, §6.8, §6.9)
#   (e) fail-closed: a frame header whose declared 24-bit Length exceeds the
#       available buffer is rejected (-1); short/misaligned §6 payloads are
#       rejected (-1)
#
# Usage: python3 http2_frame_driver.py <http2_frame.so>

import ctypes
import struct
import sys

from hyperframe.frame import (
    DataFrame,
    Frame,
    GoAwayFrame,
    HeadersFrame,
    PingFrame,
    RstStreamFrame,
    SettingsFrame,
    WindowUpdateFrame,
)

L = ctypes.CDLL(sys.argv[1])

FNS = [
    ("http2_frame_parse_header", 2),
    ("http2_frame_write_header", 5),
    ("http2_frame_parse", 3),
    ("http2_preface_write", 1),
    ("http2_preface_check", 1),
    ("http2_settings_parse", 4),
    ("http2_settings_write", 3),
    ("http2_window_update_parse", 3),
    ("http2_window_update_write", 3),
    ("http2_rst_stream_parse", 3),
    ("http2_rst_stream_write", 3),
    ("http2_ping_parse", 3),
    ("http2_ping_write", 3),
    ("http2_goaway_parse", 5),
    ("http2_goaway_write", 5),
]
for name, nargs in FNS:
    fn = getattr(L, name)
    fn.restype = ctypes.c_int64
    fn.argtypes = [ctypes.c_int64] * nargs
for name in ("data", "headers", "priority", "rst_stream", "settings",
             "push_promise", "ping", "goaway", "window_update", "continuation"):
    fn = getattr(L, f"http2_type_{name}")
    fn.restype = ctypes.c_int64
    fn.argtypes = []

results = []


def buf(data: bytes):
    return ctypes.create_string_buffer(bytes(data), max(1, len(data)))


def out(n: int):
    return ctypes.create_string_buffer(max(1, n))


def addr(b):
    return ctypes.cast(b, ctypes.c_void_p).value


def record(name, got: bytes, exp: bytes):
    ok = got == exp
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    print(f"        got={got.hex()}")
    print(f"        exp={exp.hex()}")


def record_bool(name, cond, detail=""):
    results.append(bool(cond))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}  {detail}")


def fields4(b):
    """Read the 4 i64 header fields [length, type, flags, stream_id]."""
    return struct.unpack("<4q", b.raw[:32])


def rfc9113_header(length, ftype, flags, sid):
    """Hand-derived RFC 9113 §4.1 layout: Length(24 BE)||Type(8)||Flags(8)||
    R(1)+StreamId(31 BE), R sent as zero."""
    return (struct.pack(">I", length & 0xFFFFFF)[1:]
            + bytes([ftype & 0xFF, flags & 0xFF])
            + struct.pack(">I", sid & 0x7FFFFFFF))


print("=" * 72)
print("(a) SETTINGS parse — hyperframe-serialized frame (RFC 9113 §6.5)")
print("=" * 72)
# HEADER_TABLE_SIZE=4096, ENABLE_PUSH=0, INITIAL_WINDOW_SIZE=2^31-1 (max),
# MAX_FRAME_SIZE=16384.
sset = {0x1: 4096, 0x2: 0, 0x4: 0x7FFFFFFF, 0x5: 16384}
sf = SettingsFrame(stream_id=0)
sf.settings = dict(sset)
sdata = sf.serialize()
# Internal consistency: hyperframe bytes == hand-derived RFC layout.
exp_payload = b"".join(struct.pack(">HI", k, v) for k, v in sset.items())
assert sdata == rfc9113_header(len(exp_payload), 0x4, 0, 0) + exp_payload, \
    "hyperframe disagrees with hand-derived RFC 9113 layout"
print(f"reference (hyperframe): {sdata.hex()}")
sb = buf(sdata)
fb = out(32)
consumed = L.http2_frame_parse(addr(sb), len(sdata), addr(fb))
record_bool("SETTINGS: http2_frame_parse consumed == 9+payload",
            consumed == len(sdata), f"consumed={consumed}")
flen, ftyp, ffl, fsid = fields4(fb)
record_bool("SETTINGS: header fields == (len(payload), 0x4, 0, 0)",
            (flen, ftyp, ffl, fsid) == (len(exp_payload), 0x4, 0, 0),
            f"got={(flen, ftyp, ffl, fsid)}")
pairs = out(16 * len(sset))
n = L.http2_settings_parse(addr(sb) + 9, flen, addr(pairs), len(sset))
record_bool("SETTINGS: entry count", n == len(sset), f"n={n}")
got_pairs = [struct.unpack("<2q", pairs.raw[i * 16:i * 16 + 16]) for i in range(max(n, 0))]
exp_pairs = list(sset.items())
record_bool("SETTINGS: parsed pairs == pairs hyperframe encoded",
            got_pairs == exp_pairs, f"got={got_pairs} exp={exp_pairs}")

print("=" * 72)
print("(b) frame-header write == hyperframe header bytes (RFC 9113 §4.1)")
print("=" * 72)
# Reference frames built by hyperframe; expected header = first 9 serialized
# bytes.  Includes a large 24-bit length (0xABCDEF) and the max 31-bit sid.
d_big = DataFrame(stream_id=0x7FFFFFFF)
d_big.data = b"x" * 0xABCDEF
d_big.flags.add("END_STREAM")           # flags 0x1
h_med = HeadersFrame(stream_id=12345)
h_med.data = b"y" * 100
h_med.flags.add("END_STREAM")           # 0x1
h_med.flags.add("END_HEADERS")          # 0x4 -> flags 0x5
p_ack = PingFrame(stream_id=0)
p_ack.opaque_data = b"\x00" * 8
p_ack.flags.add("ACK")                  # flags 0x1
hdr_cases = [
    ("SETTINGS len=24 t=4 f=0 sid=0", (24, 0x4, 0x0, 0), sdata[:9]
     if len(exp_payload) == 24 else None),
    ("DATA large len=0xABCDEF t=0 f=1 sid=0x7fffffff (31-bit max)",
     (0xABCDEF, 0x0, 0x1, 0x7FFFFFFF), d_big.serialize()[:9]),
    ("HEADERS len=100 t=1 f=0x5 sid=12345", (100, 0x1, 0x5, 12345),
     h_med.serialize()[:9]),
    ("PING ACK len=8 t=6 f=1 sid=0", (8, 0x6, 0x1, 0), p_ack.serialize()[:9]),
]
for nm, (ln, ty, fl, sid), ref in hdr_cases:
    assert ref is not None and len(ref) == 9
    # Internal consistency: hyperframe header == hand-derived RFC layout.
    assert ref == rfc9113_header(ln, ty, fl, sid), \
        f"hyperframe disagrees with hand RFC layout for {nm}"
    print(f"reference (hyperframe): {ref.hex()}")
    ob = out(9)
    rc = L.http2_frame_write_header(ln, ty, fl, sid, addr(ob))
    record_bool(f"hdr write rc==9 [{nm}]", rc == 9, f"rc={rc}")
    record(f"hdr write bytes [{nm}]", ob.raw[:9], ref)
# R bit MUST be sent as zero (§4.1): sid with bit 31 set is masked on write.
ob = out(9)
L.http2_frame_write_header(8, 0x6, 0, 0xFFFFFFFF, addr(ob))
record("hdr write: R bit zeroed on send (sid 0xffffffff -> 0x7fffffff)",
       ob.raw[:9], rfc9113_header(8, 0x6, 0, 0x7FFFFFFF))
# R bit MUST be ignored on receipt (§4.1): parse header with R bit set.
rbuf = buf(bytes.fromhex("000008") + bytes([0x6, 0x0]) + bytes.fromhex("ffffffff"))
fb = out(32)
L.http2_frame_parse_header(addr(rbuf), addr(fb))
record_bool("hdr parse: R bit ignored on receipt (sid -> 0x7fffffff)",
            fields4(fb) == (8, 0x6, 0x0, 0x7FFFFFFF), f"got={fields4(fb)}")

print("=" * 72)
print("(c) client connection preface — 24-byte magic (RFC 9113 §3.4)")
print("=" * 72)
PREFACE = b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n"
assert len(PREFACE) == 24
print(f"reference (RFC 9113 §3.4): {PREFACE.hex()}")
pb = out(24)
rc = L.http2_preface_write(addr(pb))
record_bool("preface write rc==24", rc == 24, f"rc={rc}")
record("preface write bytes", pb.raw[:24], PREFACE)
good = buf(PREFACE)
record_bool("preface check accepts the magic",
            L.http2_preface_check(addr(good)) == 0)
corrupt = bytearray(PREFACE)
corrupt[10] ^= 0x01  # '/' -> '.'
bad = buf(bytes(corrupt))
record_bool("preface check rejects a corrupted preface",
            L.http2_preface_check(addr(bad)) == 1)

print("=" * 72)
print("(d) round-trips parse(write(x)) == x  (+ hyperframe re-parse of MIND)")
print("=" * 72)


def hf_parse(data: bytes):
    f, ln = Frame.parse_frame_header(memoryview(data[:9]))
    f.parse_body(memoryview(data[9:9 + ln]))
    return f


# Header round-trip.
for ln, ty, fl, sid in [(0, 0x1, 0x4, 0x7FFFFFFF), (0xFFFFFF, 0x0, 0xFF, 1),
                        (6, 0x4, 0x0, 0)]:
    hb = out(9)
    L.http2_frame_write_header(ln, ty, fl, sid, addr(hb))
    fb = out(32)
    L.http2_frame_parse_header(addr(hb), addr(fb))
    record_bool(f"header round-trip ({ln:#x},{ty:#x},{fl:#x},{sid:#x})",
                fields4(fb) == (ln, ty, fl, sid), f"got={fields4(fb)}")

# SETTINGS round-trip: write full frame -> frame_parse -> settings_parse.
rt_pairs = [(0x3, 100), (0x4, 65535), (0x6, 0xFFFFFFFF)]
pin = buf(b"".join(struct.pack("<2q", k, v) for k, v in rt_pairs))
sout = out(9 + 6 * len(rt_pairs))
total = L.http2_settings_write(addr(pin), len(rt_pairs), addr(sout))
record_bool("SETTINGS write total == 9+count*6", total == 9 + 6 * len(rt_pairs),
            f"total={total}")
fb = out(32)
consumed = L.http2_frame_parse(addr(sout), total, addr(fb))
pairs2 = out(16 * len(rt_pairs))
n2 = L.http2_settings_parse(addr(sout) + 9, fields4(fb)[0], addr(pairs2), len(rt_pairs))
got2 = [struct.unpack("<2q", pairs2.raw[i * 16:i * 16 + 16]) for i in range(max(n2, 0))]
record_bool("SETTINGS round-trip pairs", got2 == rt_pairs, f"got={got2}")
hf = hf_parse(sout.raw[:total])
record_bool("SETTINGS: hyperframe re-parses MIND bytes",
            isinstance(hf, SettingsFrame) and hf.settings == dict(rt_pairs)
            and hf.stream_id == 0, f"hf={hf!r}")

# WINDOW_UPDATE round-trip (§6.9).
wout = out(13)
total = L.http2_window_update_write(77, 0x7FFFFFFF, addr(wout))
record_bool("WINDOW_UPDATE write total == 13", total == 13, f"total={total}")
fb = out(32)
L.http2_frame_parse(addr(wout), 13, addr(fb))
inc = out(8)
rc = L.http2_window_update_parse(addr(wout) + 9, fields4(fb)[0], addr(inc))
record_bool("WINDOW_UPDATE round-trip (max 31-bit increment)",
            rc == 0 and fields4(fb)[:4] == (4, 0x8, 0, 77)
            and struct.unpack("<q", inc.raw[:8])[0] == 0x7FFFFFFF,
            f"rc={rc} fields={fields4(fb)}")
hf = hf_parse(wout.raw[:13])
record_bool("WINDOW_UPDATE: hyperframe re-parses MIND bytes",
            isinstance(hf, WindowUpdateFrame) and hf.stream_id == 77
            and hf.window_increment == 0x7FFFFFFF, f"hf={hf!r}")

# GOAWAY round-trip (§6.8): last-stream-id || error code || debug data.
dbg = b"graceful shutdown"
gout = out(17 + len(dbg))
dbgb = buf(dbg)
total = L.http2_goaway_write(0x7FFFFFFE, 0xB, addr(dbgb), len(dbg), addr(gout))
record_bool("GOAWAY write total == 17+debug_len", total == 17 + len(dbg),
            f"total={total}")
fb = out(32)
L.http2_frame_parse(addr(gout), total, addr(fb))
gfields = out(16)
gdbg = out(len(dbg))
dlen = L.http2_goaway_parse(addr(gout) + 9, fields4(fb)[0], addr(gfields), addr(gdbg), len(dbg))
lsid, ecode = struct.unpack("<2q", gfields.raw[:16])
record_bool("GOAWAY round-trip fields",
            dlen == len(dbg) and (lsid, ecode) == (0x7FFFFFFE, 0xB)
            and gdbg.raw[:dlen] == dbg,
            f"dlen={dlen} lsid={lsid:#x} err={ecode:#x}")
hf = hf_parse(gout.raw[:total])
record_bool("GOAWAY: hyperframe re-parses MIND bytes",
            isinstance(hf, GoAwayFrame) and hf.last_stream_id == 0x7FFFFFFE
            and hf.error_code == 0xB and hf.additional_data == dbg, f"hf={hf!r}")

# RST_STREAM round-trip (§6.4).
rout = out(13)
total = L.http2_rst_stream_write(9, 0x8, addr(rout))  # CANCEL
fb = out(32)
L.http2_frame_parse(addr(rout), 13, addr(fb))
ec = out(8)
rc = L.http2_rst_stream_parse(addr(rout) + 9, fields4(fb)[0], addr(ec))
record_bool("RST_STREAM round-trip",
            total == 13 and rc == 0 and fields4(fb)[:4] == (4, 0x3, 0, 9)
            and struct.unpack("<q", ec.raw[:8])[0] == 0x8,
            f"total={total} rc={rc} fields={fields4(fb)}")
hf = hf_parse(rout.raw[:13])
record_bool("RST_STREAM: hyperframe re-parses MIND bytes",
            isinstance(hf, RstStreamFrame) and hf.stream_id == 9
            and hf.error_code == 0x8, f"hf={hf!r}")

# PING round-trip (§6.7): 8-byte opaque, ACK flag.
opaque = bytes.fromhex("0102030405060708")
pob = buf(opaque)
pout = out(17)
total = L.http2_ping_write(0x1, addr(pob), addr(pout))
fb = out(32)
L.http2_frame_parse(addr(pout), 17, addr(fb))
op2 = out(8)
rc = L.http2_ping_parse(addr(pout) + 9, fields4(fb)[0], addr(op2))
record_bool("PING round-trip (ACK, opaque preserved)",
            total == 17 and rc == 0 and fields4(fb)[:4] == (8, 0x6, 0x1, 0)
            and op2.raw[:8] == opaque,
            f"total={total} rc={rc} fields={fields4(fb)}")
hf = hf_parse(pout.raw[:17])
record_bool("PING: hyperframe re-parses MIND bytes",
            isinstance(hf, PingFrame) and hf.opaque_data == opaque
            and "ACK" in hf.flags, f"hf={hf!r}")

print("=" * 72)
print("(e) fail-closed: declared length > buffer / malformed payloads")
print("=" * 72)
# Header claims 100-byte payload; only 20 bytes exist in the buffer.
over = rfc9113_header(100, 0x0, 0, 1) + b"\x00" * 11
assert len(over) == 20 and 9 + 100 > len(over)
ovb = buf(over)
fb = out(32)
ctypes.memset(addr(fb), 0x5A, 32)  # sentinel: fields must stay untouched
rc = L.http2_frame_parse(addr(ovb), len(over), addr(fb))
record_bool("over-length frame rejected (declared 100 > 11 available)",
            rc == -1, f"rc={rc}")
record_bool("over-length frame: no fields written (sentinel intact)",
            fb.raw[:32] == b"\x5a" * 32)
short = buf(b"\x00" * 8)
record_bool("truncated header (8 bytes) rejected",
            L.http2_frame_parse(addr(short), 8, addr(out(32))) == -1)
# Exact-fit frame is ACCEPTED (boundary of the fail-closed check).
exact = rfc9113_header(4, 0x8, 0, 1) + struct.pack(">I", 5)
exb = buf(exact)
record_bool("exact-fit frame accepted (9+4 == 13)",
            L.http2_frame_parse(addr(exb), 13, addr(out(32))) == 13)
# Off-by-ONE over-length: declared = available+1 (mutation-kill: a fail-open
# bound like `length > buf_len + 9` passes the 100-vs-11 case above but not this).
ob1 = buf(rfc9113_header(12, 0x0, 0, 1) + b"\x00" * 11)  # buf 20, need 21
record_bool("over-length by exactly 1 rejected (declared 12 > 11 available)",
            L.http2_frame_parse(addr(ob1), 20, addr(out(32))) == -1)
# §6 payload-size rules (FRAME_SIZE_ERROR -> -1).
z = buf(b"\x00" * 16)
record_bool("SETTINGS payload len 7 (not %6) rejected",
            L.http2_settings_parse(addr(z), 7, addr(out(32)), 2) == -1)
record_bool("WINDOW_UPDATE payload len 3 rejected",
            L.http2_window_update_parse(addr(z), 3, addr(out(8))) == -1)
record_bool("RST_STREAM payload len 5 rejected",
            L.http2_rst_stream_parse(addr(z), 5, addr(out(8))) == -1)
record_bool("PING payload len 7 rejected",
            L.http2_ping_parse(addr(z), 7, addr(out(8))) == -1)
gsent = out(16)
ctypes.memset(addr(gsent), 0x5A, 16)  # sentinel: reject must write no fields
record_bool("GOAWAY payload len 7 (< fixed 8) rejected",
            L.http2_goaway_parse(addr(z), 7, addr(gsent), addr(out(1)), 1) == -1)

# out_cap fail-closed (the hardening): a valid payload that would exceed the
# caller-allocated output must be REJECTED (-1), not written out of bounds.
record_bool("SETTINGS: count > out_pairs_cap rejected",
            L.http2_settings_parse(addr(out(12)), 12, addr(out(16)), 1) == -1)  # 2 pairs, cap 1
record_bool("SETTINGS: count <= out_pairs_cap accepted",
            L.http2_settings_parse(addr(out(12)), 12, addr(out(32)), 2) == 2)   # 2 pairs, cap 2
record_bool("GOAWAY: debug_len > debug_out_cap rejected",
            L.http2_goaway_parse(addr(out(16)), 16, addr(out(16)), addr(out(4)), 4) == -1)  # 8 dbg, cap 4
record_bool("GOAWAY: debug_len <= debug_out_cap accepted",
            L.http2_goaway_parse(addr(out(16)), 16, addr(out(16)), addr(out(8)), 8) == 8)   # 8 dbg, cap 8
record_bool("GOAWAY reject: no fields written (sentinel intact)",
            gsent.raw[:16] == b"\x5a" * 16)
# WINDOW_UPDATE R bit MUST be ignored on receipt (§6.9): payload ffffffff.
wrb = buf(b"\xff\xff\xff\xff")
winc = out(8)
rc = L.http2_window_update_parse(addr(wrb), 4, addr(winc))
record_bool("WINDOW_UPDATE: R bit masked on receipt (ffffffff -> 0x7fffffff)",
            rc == 0 and struct.unpack("<q", winc.raw[:8])[0] == 0x7FFFFFFF,
            f"rc={rc} inc={struct.unpack('<q', winc.raw[:8])[0]:#x}")

print("=" * 72)
total = len(results)
passed = sum(1 for r in results if r)
print(f"SUMMARY: {passed}/{total} checks PASSED")
print("=" * 72)
sys.exit(0 if passed == total else 1)
