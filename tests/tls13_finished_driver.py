#!/usr/bin/env python3
# Official-vector driver for std/tls13_finished.mind (pure-MIND TLS 1.3
# Finished-message MAC + transcript hash, RFC 8446 §4.4.1 / §4.4.4).
#
# Exercises every pub fn entry point against the RFC 8448 §3 "Simple 1-RTT
# Handshake" published constants:
#   - server_handshake_traffic_secret (printed in RFC 8448 §3),
#   - the server finished_key ("expanded" in the "{server} calculate finished"
#     block: 008d3b66...),
#   - the server Finished verify_data (the 32 bytes after the 14 00 00 20
#     handshake header of the Finished message inside the published 657-octet
#     server flight payload — tail ...9572cb7fffee5454b78f0718).
#
# GROUND TRUTH IS TWO INDEPENDENT REFERENCES, cross-checked against each other
# BEFORE either is compared to the MIND .so:
#   (a) the RFC 8448 §3 published hex constants (hardcoded below: ClientHello,
#       ServerHello, the 657-octet server flight, shts, finished_key), and
#   (b) a from-scratch Python composition built ONLY on hashlib/hmac:
#       HKDF-Expand-Label per RFC 8446 §7.1 + HMAC-SHA256 per RFC 2104.
# The driver first PROVES composition (b) reproduces every RFC 8448 (a)
# constant (hello-transcript hash, finished_key, verify_data — so the ground
# truth is real, not circular); only then are the constants compared to the
# MIND .so output.  PASS/FAIL per case, verbatim hex compared.
#
# Build the .so first (combine deps in dependency order, stripping imports —
# sha256 MUST precede hkdf MUST precede tls13_keyschedule MUST precede
# tls13_finished):
#   cat std/sha256.mind                                    > /tmp/tls13f.mind
#   grep -v '^import std.sha256;' std/hkdf.mind           >> /tmp/tls13f.mind
#   grep -vE '^import std\.(sha256|hkdf);' std/tls13_keyschedule.mind \
#                                                         >> /tmp/tls13f.mind
#   grep -vE '^import std\.' std/tls13_finished.mind      >> /tmp/tls13f.mind
#   mindc /tmp/tls13f.mind --emit-shared /tmp/tls13f.so   # needs mlir-build
#
# Usage: python3 tls13_finished_driver.py <tls13f.so>

import ctypes
import hashlib
import hmac as _pyhmac
import sys

so_path = sys.argv[1]
T = ctypes.CDLL(so_path)

# ctypes signatures (all args are i64 addresses/lengths; return i64).
_SIGS = {
    "tls13_transcript_hash": 3,
    "tls13_finished_verify_data": 3,
    "tls13_finished_check": 3,
    "tls13_finished_from_secret": 3,
    "tls13_finished_key": 2,  # from std/tls13_keyschedule.mind (case a)
}
for _name, _n in _SIGS.items():
    fn = getattr(T, _name)
    fn.restype = ctypes.c_int64
    fn.argtypes = [ctypes.c_int64] * _n

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


def record_int(name, got: int, exp: int):
    ok = got == exp
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    print(f"        got={got}")
    print(f"        exp={exp}")


# ---------------------------------------------------------------------------
# RFC 8448 §3 "Simple 1-RTT Handshake" published constants (reference (a)).
# ---------------------------------------------------------------------------

# "{client} construct a ClientHello handshake message: ClientHello (196 octets)"
CLIENT_HELLO = bytes.fromhex(
    "010000c00303cb34ecb1e78163ba1c38c6dacb196a6dffa21a8d9912ec18a2ef"
    "6283024dece7000006130113031302010000910000000b000900000673657276"
    "6572ff01000100000a00140012001d0017001800190100010101020103010400"
    "230000003300260024001d002099381de560e4bd43d23d8e435a7dbafeb3c06e"
    "51c13cae4d5413691e529aaf2c002b0003020304000d0020001e040305030603"
    "020308040805080604010501060102010402050206020202002d00020101001c"
    "00024001")
assert len(CLIENT_HELLO) == 196

# "{server} construct a ServerHello handshake message: ServerHello (90 octets)"
SERVER_HELLO = bytes.fromhex(
    "020000560303a6af06a4121860dc5e6e60249cd34c95930c8ac5cb1434dac155"
    "772ed3e2692800130100002e00330024001d0020c9828876112095fe66762bdb"
    "f7c672e156d6cc253b833df1dd69b1b04e751f0f002b00020304")
assert len(SERVER_HELLO) == 90

# "{server} send handshake record: payload (657 octets)" — the concatenated
# EncryptedExtensions || Certificate || CertificateVerify || Finished
# handshake messages (same constant as tests/tls13_record_driver.py).
SERVER_FLIGHT = bytes.fromhex(
    "080000240022000a00140012001d00170018001901000101010201030104001c"
    "00024001000000000b0001b9000001b50001b0308201ac30820115a003020102"
    "020102300d06092a864886f70d01010b0500300e310c300a0603550403130372"
    "7361301e170d3136303733303031323335395a170d3236303733303031323335"
    "395a300e310c300a0603550403130372736130819f300d06092a864886f70d01"
    "0101050003818d0030818902818100b4bb498f8279303d980836399b36c6988c"
    "0c68de55e1bdb826d3901a2461eafd2de49a91d015abbc9a95137ace6c1af19e"
    "aa6af98c7ced43120998e187a80ee0ccb0524b1b018c3e0b63264d449a6d38e2"
    "2a5fda430846748030530ef0461c8ca9d9efbfae8ea6d1d03e2bd193eff0ab9a"
    "8002c47428a6d35a8d88d79f7f1e3f0203010001a31a301830090603551d1304"
    "023000300b0603551d0f0404030205a0300d06092a864886f70d01010b050003"
    "81810085aad2a0e5b9276b908c65f73a7267170618a54c5f8a7b337d2df7a594"
    "365417f2eae8f8a58c8f8172f9319cf36b7fd6c55b80f21a03015156726096fd"
    "335e5e67f2dbf102702e608ccae6bec1fc63a42a99be5c3eb7107c3c54e9b9eb"
    "2bd5203b1c3b84e0a8b2f759409ba3eac9d91d402dcc0cc8f8961229ac9187b4"
    "2b4de100000f000084080400805a747c5d88fa9bd2e55ab085a61015b7211f82"
    "4cd484145ab3ff52f1fda8477b0b7abc90db78e2d33a5c141a078653fa6bef78"
    "0c5ea248eeaaa785c4f394cab6d30bbe8d4859ee511f602957b15411ac027671"
    "459e46445c9ea58c181e818e95b8c3fb0bf3278409d3be152a3da5043e063dda"
    "65cdf5aea20d53dfacd42f74f3140000209b9b141d906337fbd2cbdce71df4de"
    "da4ab42c309572cb7fffee5454b78f0718")
assert len(SERVER_FLIGHT) == 657

# The trailing 36 bytes are the Finished message: 14 00 00 20 || verify_data.
FINISHED_MSG = SERVER_FLIGHT[-36:]
assert FINISHED_MSG[:4] == bytes.fromhex("14000020")
RFC_VERIFY_DATA = FINISHED_MSG[4:]  # ...9572cb7fffee5454b78f0718

# RFC 8448 §3 printed secrets/keys.
SHTS = bytes.fromhex(  # server_handshake_traffic_secret
    "b67b7d690cc16c4e75e54213cb2d37b4e9c912bcded9105d42befd59d391ad38")
RFC_FINISHED_KEY = bytes.fromhex(  # "{server} calculate finished" expanded
    "008d3b66f816ea559f96b537e885c31fc068bf492c652f01f288a1d8cdc19fc8")
HELLO_HASH = bytes.fromhex(  # Transcript-Hash(ClientHello..ServerHello),
    # printed in RFC 8448 §3 (the c/s hs traffic Derive-Secret hash input)
    "860c06edc07858ee8e78f0e7428c58edd6b43f2ca3e6e95f02ed063cf0e1cad8")

# Transcript up to CertificateVerify (the server Finished's MAC input,
# RFC 8446 §4.4.4): CH || SH || EncryptedExtensions..CertificateVerify.
TRANSCRIPT_TO_CV = CLIENT_HELLO + SERVER_HELLO + SERVER_FLIGHT[:-36]

# ---------------------------------------------------------------------------
# Reference composition (b): HKDF-Expand-Label + HMAC from scratch, built ONLY
# on hashlib/hmac (RFC 5869 + RFC 8446 §7.1 + RFC 2104).
# ---------------------------------------------------------------------------
def ref_hkdf_expand(prk: bytes, info: bytes, length: int) -> bytes:
    t, okm, i = b"", b"", 1
    while len(okm) < length:
        t = _pyhmac.new(prk, t + info + bytes([i]), hashlib.sha256).digest()
        okm += t
        i += 1
    return okm[:length]


def ref_expand_label(secret: bytes, label: bytes, ctx: bytes, length: int) -> bytes:
    full = b"tls13 " + label
    hl = length.to_bytes(2, "big") + bytes([len(full)]) + full + bytes([len(ctx)]) + ctx
    return ref_hkdf_expand(secret, hl, length)


# ---------------------------------------------------------------------------
# CROSS-CHECK 1: prove composition (b) reproduces every RFC 8448 (a) constant.
# If any assert fires the ground truth is wrong and we stop before touching MIND.
# ---------------------------------------------------------------------------
print("=" * 72)
print("Cross-check: Python composition (b) reproduces RFC 8448 §3 constants (a)")
print("=" * 72)

b_hello_hash = hashlib.sha256(CLIENT_HELLO + SERVER_HELLO).digest()
assert b_hello_hash == HELLO_HASH, (
    f"sha256(CH||SH) {b_hello_hash.hex()} != RFC hello hash {HELLO_HASH.hex()}")
print(f"[ OK ] composition(b) == RFC8448  hello_transcript_hash "
      f"{HELLO_HASH.hex()}")

b_thash_cv = hashlib.sha256(TRANSCRIPT_TO_CV).digest()

b_finished_key = ref_expand_label(SHTS, b"finished", b"", 32)
assert b_finished_key == RFC_FINISHED_KEY, (
    f"ExpandLabel(shts,'finished','',32) {b_finished_key.hex()} != "
    f"RFC finished_key {RFC_FINISHED_KEY.hex()}")
print(f"[ OK ] composition(b) == RFC8448  server_finished_key   "
      f"{RFC_FINISHED_KEY.hex()}")

b_verify_data = _pyhmac.new(b_finished_key, b_thash_cv, hashlib.sha256).digest()
assert b_verify_data == RFC_VERIFY_DATA, (
    f"HMAC(finished_key, thash) {b_verify_data.hex()} != "
    f"RFC verify_data {RFC_VERIFY_DATA.hex()}")
print(f"[ OK ] composition(b) == RFC8448  server_verify_data    "
      f"{RFC_VERIFY_DATA.hex()}")
print(f"       (transcript hash CH..CertificateVerify = {b_thash_cv.hex()})")
print("Ground truth confirmed: (b) reproduces (a) — finished_key AND the")
print("published server Finished verify_data derive from the RFC transcript.\n")

# ---------------------------------------------------------------------------
# Now compare the MIND .so output to the (doubly-confirmed) constants.
# ---------------------------------------------------------------------------
print("=" * 72)
print("MIND std/tls13_finished.mind vs RFC 8448 §3 (RFC 8446 §4.4.1/§4.4.4)")
print("=" * 72)

# Transcript-Hash sanity: MIND sha256 over CH||SH == RFC hello hash, and over
# CH..CertificateVerify == the validated thash.
msgs = buf(CLIENT_HELLO + SERVER_HELLO)
ob = out(32)
T.tls13_transcript_hash(addr(msgs), len(CLIENT_HELLO + SERVER_HELLO), addr(ob))
record("tls13_transcript_hash(CH||SH) == RFC hello hash", ob.raw[:32], HELLO_HASH)

msgs = buf(TRANSCRIPT_TO_CV)
th = out(32)
T.tls13_transcript_hash(addr(msgs), len(TRANSCRIPT_TO_CV), addr(th))
record("tls13_transcript_hash(CH..CertificateVerify)", th.raw[:32], b_thash_cv)
thash_b = buf(th.raw[:32])

# (a) finished_key: tls13_finished_key(shts) == RFC 008d3b66...
shts_b = buf(SHTS)
fkb = out(32)
T.tls13_finished_key(addr(shts_b), addr(fkb))
record("(a) finished_key = ExpandLabel(shts,'finished','',32)",
       fkb.raw[:32], RFC_FINISHED_KEY)
fk_b = buf(fkb.raw[:32])

# (b) verify_data = HMAC(finished_key, transcript_hash) == RFC server Finished.
vdb = out(32)
T.tls13_finished_verify_data(addr(fk_b), addr(thash_b), addr(vdb))
record("(b) verify_data = HMAC(finished_key, thash) == RFC server Finished",
       vdb.raw[:32], RFC_VERIFY_DATA)

# (c) check: accepts the correct verify_data (1), REJECTS a bit-flip (0).
good_b = buf(RFC_VERIFY_DATA)
rc = T.tls13_finished_check(addr(fk_b), addr(thash_b), addr(good_b))
record_int("(c1) tls13_finished_check(correct verify_data) -> 1", rc, 1)

flipped = bytearray(RFC_VERIFY_DATA)
flipped[0] ^= 0x01  # single bit flip in byte 0
bad_b = buf(bytes(flipped))
rc = T.tls13_finished_check(addr(fk_b), addr(thash_b), addr(bad_b))
record_int("(c2) tls13_finished_check(bit-flipped byte 0) -> 0 (fail-closed)",
           rc, 0)

flipped = bytearray(RFC_VERIFY_DATA)
flipped[31] ^= 0x80  # single bit flip in the LAST byte
bad_b = buf(bytes(flipped))
rc = T.tls13_finished_check(addr(fk_b), addr(thash_b), addr(bad_b))
record_int("(c3) tls13_finished_check(bit-flipped byte 31) -> 0 (fail-closed)",
           rc, 0)

# (d) from_secret: shts -> verify_data in one shot, matches (b).
vdb = out(32)
T.tls13_finished_from_secret(addr(shts_b), addr(thash_b), addr(vdb))
record("(d) tls13_finished_from_secret(shts, thash) == RFC server Finished",
       vdb.raw[:32], RFC_VERIFY_DATA)

print("=" * 72)
total = len(results)
passed = sum(1 for r in results if r)
print(f"SUMMARY: {passed}/{total} checks PASSED")
print("=" * 72)
sys.exit(0 if passed == total else 1)
