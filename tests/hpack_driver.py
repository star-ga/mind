#!/usr/bin/env python3
# Official-vector driver for std/hpack.mind (pure-MIND HPACK decoding,
# RFC 7541).
#
# Exercises every pub fn entry point in the compiled .so against the
# published RFC 7541 test material:
#   - hpack_int_decode  : RFC 7541 Appendix C.1.1-C.1.3 (prefix integers)
#   - hpack_huff_decode : Appendix B strings from C.4/C.6 + invalid-padding
#                         fail-closed cases
#   - hpack_decode      : Appendix C.2.1-C.2.4 (literal field examples),
#                         C.3.1-C.3.3 (request sequence, no Huffman),
#                         C.4.1-C.4.3 (request sequence, Huffman),
#                         C.5.1-C.5.3 (responses, table size 256, eviction),
#                         C.6.1-C.6.3 (responses, Huffman, eviction)
#     The C.3/C.4/C.5/C.6 sequences reuse ONE dynamic-table state buffer per
#     sequence, so later blocks reference (and evict) entries inserted by
#     earlier blocks — the persistent-state contract of hpack_decode.
#
# Each hardcoded wire vector is ALSO cross-checked against the Python `hpack`
# package (an independent trusted reference) when it is importable, so the
# hex cannot be a transcription error.  Prints PASS/FAIL per example with
# the verbatim decoded (name, value) list.
#
# Usage: python3 hpack_driver.py <hpack.so>

import ctypes
import sys

so_path = sys.argv[1]
L = ctypes.CDLL(so_path)

for name, nargs in (("hpack_decode", 5), ("hpack_dyn_init", 2),
                    ("hpack_int_decode", 5), ("hpack_huff_decode", 4)):
    fn = getattr(L, name)
    fn.restype = ctypes.c_int64
    fn.argtypes = [ctypes.c_int64] * nargs

results = []


def buf(data: bytes):
    """Mutable ctypes buffer holding data (>=1 byte so the pointer is valid)."""
    return ctypes.create_string_buffer(bytes(data), max(1, len(data)))


def out(n: int):
    return ctypes.create_string_buffer(max(1, n))


def addr(b):
    return ctypes.cast(b, ctypes.c_void_p).value


def record_bool(name, cond, detail=""):
    results.append(bool(cond))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}  {detail}")


# ---------------------------------------------------------------------------
# RFC 7541 Appendix C vectors (wire hex + published header lists).
# ---------------------------------------------------------------------------
C2 = [
    ("C.2.1", "400a637573746f6d2d6b65790d637573746f6d2d686561646572",
     [("custom-key", "custom-header")]),
    ("C.2.2", "040c2f73616d706c652f70617468", [(":path", "/sample/path")]),
    ("C.2.3", "100870617373776f726406736563726574", [("password", "secret")]),
    ("C.2.4", "82", [(":method", "GET")]),
]
REQ1 = [(":method", "GET"), (":scheme", "http"), (":path", "/"),
        (":authority", "www.example.com")]
REQ2 = REQ1 + [("cache-control", "no-cache")]
REQ3 = [(":method", "GET"), (":scheme", "https"), (":path", "/index.html"),
        (":authority", "www.example.com"), ("custom-key", "custom-value")]
C3 = [("C.3.1", "828684410f7777772e6578616d706c652e636f6d", REQ1),
      ("C.3.2", "828684be58086e6f2d6361636865", REQ2),
      ("C.3.3", "828785bf400a637573746f6d2d6b65790c637573746f6d2d76616c7565",
       REQ3)]
C4 = [("C.4.1", "828684418cf1e3c2e5f23a6ba0ab90f4ff", REQ1),
      ("C.4.2", "828684be5886a8eb10649cbf", REQ2),
      ("C.4.3", "828785bf408825a849e95ba97d7f8925a849e95bb8e8b4bf", REQ3)]
RESP1 = [(":status", "302"), ("cache-control", "private"),
         ("date", "Mon, 21 Oct 2013 20:13:21 GMT"),
         ("location", "https://www.example.com")]
RESP2 = [(":status", "307"), ("cache-control", "private"),
         ("date", "Mon, 21 Oct 2013 20:13:21 GMT"),
         ("location", "https://www.example.com")]
RESP3 = [(":status", "200"), ("cache-control", "private"),
         ("date", "Mon, 21 Oct 2013 20:13:22 GMT"),
         ("location", "https://www.example.com"),
         ("content-encoding", "gzip"),
         ("set-cookie",
          "foo=ASDJKHQKBZXOQWEOPIUAXQWEOIU; max-age=3600; version=1")]
C5 = [("C.5.1",
       "4803333032580770726976617465611d4d6f6e2c203231204f637420323031"
       "332032303a31333a323120474d546e1768747470733a2f2f7777772e657861"
       "6d706c652e636f6d", RESP1),
      ("C.5.2", "4803333037c1c0bf", RESP2),
      ("C.5.3",
       "88c1611d4d6f6e2c203231204f637420323031332032303a31333a32322047"
       "4d54c05a04677a69707738666f6f3d4153444a4b48514b425a584f5157454f"
       "50495541585157454f49553b206d61782d6167653d333630303b2076657273"
       "696f6e3d31", RESP3)]
C6 = [("C.6.1",
       "488264025885aec3771a4b6196d07abe941054d444a8200595040b8166e082"
       "a62d1bff6e919d29ad171863c78f0b97c8e9ae82ae43d3", RESP1),
      ("C.6.2", "4883640effc1c0bf", RESP2),
      ("C.6.3",
       "88c16196d07abe941054d444a8200595040b8166e084a62d1bffc05a839bd9"
       "ab77ad94e7821dd7f2e6c7b335dfdfcd5b3960d5af27087f3672c1ab270fb5"
       "291f9587316065c003ed4ee5b1063d5007", RESP3)]

# ---------------------------------------------------------------------------
# Independent reference: cross-check every hardcoded vector against the
# Python `hpack` package BEFORE comparing MIND output.
# ---------------------------------------------------------------------------
try:
    import hpack as pyhpack

    for name, hx, exp in C2:
        got = [(n, v) for n, v in pyhpack.Decoder().decode(bytes.fromhex(hx))]
        assert got == exp, f"python-hpack disagrees with RFC 7541 {name}: {got}"
    for seq, size in ((C3, None), (C4, None), (C5, 256), (C6, 256)):
        d = pyhpack.Decoder()
        if size is not None:
            d.header_table_size = size
        for name, hx, exp in seq:
            got = [(n, v) for n, v in d.decode(bytes.fromhex(hx))]
            assert got == exp, \
                f"python-hpack disagrees with RFC 7541 {name}: {got}"
    print("reference: python-hpack cross-check of all Appendix C vectors OK")
except ImportError:
    print("reference: python-hpack not installed — RFC 7541 Appendix C "
          "hardcoded values are the authority for this run")

# ---------------------------------------------------------------------------
# hpack_int_decode — RFC 7541 Appendix C.1 prefix-integer examples.
# ---------------------------------------------------------------------------
print("=" * 72)
print("hpack_int_decode — RFC 7541 C.1.1-C.1.3")
print("=" * 72)


def int_case(name, hx, prefix_bits, exp_value, exp_pos):
    data = bytes.fromhex(hx)
    b = buf(data)
    o2 = out(16)
    rc = L.hpack_int_decode(addr(b), len(data), 0, prefix_bits, addr(o2))
    value = int.from_bytes(o2.raw[0:8], "little")
    pos = int.from_bytes(o2.raw[8:16], "little")
    ok = rc == 0 and value == exp_value and pos == exp_pos
    record_bool(name, ok, f"rc={rc} value={value} pos={pos} "
                          f"(exp value={exp_value} pos={exp_pos})")


int_case("C.1.1 encode 10 in a 5-bit prefix", "0a", 5, 10, 1)
int_case("C.1.2 encode 1337 in a 5-bit prefix", "1f9a0a", 5, 1337, 3)
int_case("C.1.3 encode 42 at an octet boundary (8-bit)", "2a", 8, 42, 1)
# Truncated continuation must fail closed.
b = buf(bytes.fromhex("1f"))
o2 = out(16)
rc = L.hpack_int_decode(addr(b), 1, 0, 5, addr(o2))
record_bool("C.1 truncated continuation rejected", rc != 0, f"rc={rc}")

# ---------------------------------------------------------------------------
# hpack_huff_decode — Appendix B strings + padding fail-closed.
# ---------------------------------------------------------------------------
print("=" * 72)
print("hpack_huff_decode — RFC 7541 Appendix B")
print("=" * 72)


def huff_case(name, hx, exp: bytes):
    data = bytes.fromhex(hx)
    b, o = buf(data), out(256)
    n = L.hpack_huff_decode(addr(b), len(data), addr(o), 256)
    got = o.raw[:n] if n >= 0 else b"<error>"
    ok = n == len(exp) and got == exp
    record_bool(name, ok, f"n={n} got={got!r} exp={exp!r}")


huff_case("Huffman 'www.example.com' (C.4.1)", "f1e3c2e5f23a6ba0ab90f4ff",
          b"www.example.com")
huff_case("Huffman 'no-cache' (C.4.2)", "a8eb10649cbf", b"no-cache")
huff_case("Huffman '302' (C.6.1)", "6402", b"302")
huff_case("Huffman 'gzip' (C.6.3)", "9bd9ab", b"gzip")
# '0' is code 00000 (5 bits); trailing 000 padding is NOT the EOS prefix
# (must be all ones) -> decode error per RFC 7541 s5.2.
b, o = buf(bytes.fromhex("00")), out(16)
n = L.hpack_huff_decode(addr(b), 1, addr(o), 16)
record_bool("Huffman invalid all-zero padding rejected", n < 0, f"n={n}")
# 8+ bits of padding (a full 0xff byte after a symbol) is also an error.
b, o = buf(bytes.fromhex("07ffffff")), out(16)  # '0' then 27 one-bits
n = L.hpack_huff_decode(addr(b), 4, addr(o), 16)
record_bool("Huffman >7-bit padding rejected", n < 0, f"n={n}")

# ---------------------------------------------------------------------------
# hpack_decode — full header blocks, RFC 7541 Appendix C.2-C.6.
# ---------------------------------------------------------------------------
STATE_HDR = 40  # dynamic-table state header bytes (see std/hpack.mind)


def new_state(max_size):
    st = out(STATE_HDR + max_size)
    L.hpack_dyn_init(addr(st), max_size)
    return st


def parse_out(raw, npairs):
    """Parse npairs (name_len u16 LE, name, value_len u16 LE, value) records."""
    pairs, p = [], 0
    for _ in range(npairs):
        nl = int.from_bytes(raw[p:p + 2], "little")
        name = raw[p + 2:p + 2 + nl]
        p += 2 + nl
        vl = int.from_bytes(raw[p:p + 2], "little")
        value = raw[p + 2:p + 2 + vl]
        p += 2 + vl
        pairs.append((name.decode(), value.decode()))
    return pairs


def decode_case(name, state, hx, exp):
    data = bytes.fromhex(hx)
    b, o = buf(data), out(8192)
    n = L.hpack_decode(addr(b), len(data), addr(state), addr(o), 8192)
    if n != len(exp):
        record_bool(name, False, f"pair count {n} != {len(exp)}")
        return
    got = parse_out(o.raw, n)
    record_bool(name, got == exp, f"\n        got={got}\n        exp={exp}")


print("=" * 72)
print("hpack_decode — C.2 literal header field examples (fresh table each)")
print("=" * 72)
for name, hx, exp in C2:
    decode_case(name, new_state(4096), hx, exp)

for label, seq, size in (("C.3 request sequence, no Huffman", C3, 4096),
                         ("C.4 request sequence, Huffman", C4, 4096),
                         ("C.5 response sequence, eviction (max 256)", C5, 256),
                         ("C.6 response sequence, Huffman + eviction "
                          "(max 256)", C6, 256)):
    print("=" * 72)
    print(f"hpack_decode — {label}: ONE persistent dynamic table")
    print("=" * 72)
    st = new_state(size)
    for name, hx, exp in seq:
        decode_case(name, st, hx, exp)

# Fail-closed: an indexed reference to a never-inserted dynamic entry.
print("=" * 72)
print("hpack_decode — fail-closed checks")
print("=" * 72)
st = new_state(4096)
b, o = buf(bytes.fromhex("be")), out(64)  # index 62: dynamic table is empty
n = L.hpack_decode(addr(b), 1, addr(st), addr(o), 64)
record_bool("reference to empty dynamic table rejected", n < 0, f"n={n}")
st = new_state(4096)
b, o = buf(bytes.fromhex("80")), out(64)  # index 0 is reserved (s6.1)
n = L.hpack_decode(addr(b), 1, addr(st), addr(o), 64)
record_bool("indexed field with index 0 rejected", n < 0, f"n={n}")

# Fail-closed: a s6.3 dynamic-table size update above the init ceiling is a
# decoding error (RFC 7541 s6.3/s4.2) — prevents the arena heap overflow.
# "3f45" = 001 size-update, 5-bit-prefix integer = 31 + 69 = 100.
st = new_state(64)
b, o = buf(bytes.fromhex("3f45")), out(64)  # size update -> 100 > cap 64
n = L.hpack_decode(addr(b), 2, addr(st), addr(o), 64)
record_bool("size update above init ceiling rejected", n < 0, f"n={n}")
# The same update below the ceiling is still accepted (no field -> 0 pairs).
st = new_state(4096)
b, o = buf(bytes.fromhex("3f45")), out(64)  # size update -> 100 <= cap 4096
n = L.hpack_decode(addr(b), 2, addr(st), addr(o), 64)
record_bool("size update within init ceiling accepted", n == 0, f"n={n}")

print("=" * 72)
total = len(results)
passed = sum(1 for r in results if r)
print(f"SUMMARY: {passed}/{total} checks PASSED")
print("=" * 72)
sys.exit(0 if passed == total else 1)
