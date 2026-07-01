#!/usr/bin/env python3
# Official-vector driver for std/tls13_record.mind (pure-MIND TLS 1.3 record
# layer, RFC 8446 §5.1-5.3).
#
# Exercises tls13_record_seal / tls13_record_open / tls13_record_nonce against
# the RFC 8448 §3 "Simple 1-RTT Handshake" published server handshake flight:
# server_handshake_traffic_secret -> key/iv (RFC 8448 §3), the 657-octet
# plaintext handshake payload (EncryptedExtensions..Finished, inner content
# type 0x16), and the resulting 679-octet on-the-wire encrypted record at
# sequence number 0 — all spelled out in hex in RFC 8448 §3.
#
# GROUND TRUTH IS TWO INDEPENDENT REFERENCES, cross-checked against each other
# BEFORE either is compared to the MIND .so:
#   (a) the RFC 8448 §3 published record bytes (hardcoded below), and
#   (b) pyca cryptography AESGCM sealing the same payload with the RFC 8446
#       §5.2/§5.3 nonce+aad construction built here from scratch.
# The driver first PROVES composition (b) reproduces (a) byte-for-byte (so the
# ground truth is real, not circular); only then is the MIND .so compared.
#
# Cases:
#   (a) seal(payload, seq=0) == RFC 8448 complete record, byte-for-byte
#   (b) open(RFC record, seq=0) == payload, inner content type 0x16
#   (c) REJECT: tampered ciphertext byte AND tampered tag byte both fail closed
#   (d) seq=1 nonce differs from seq=0 nonce (and only in the low 8 bytes)
#
# Build the .so first (combine deps in dependency order, stripping imports —
# sha256 -> hkdf -> tls13_keyschedule -> aes_gcm -> tls13_record):
#   cat std/sha256.mind                                   > /tmp/tls13rec_combined.mind
#   grep -v '^import std.sha256;' std/hkdf.mind          >> /tmp/tls13rec_combined.mind
#   grep -vE '^import std\.(sha256|hkdf);' std/tls13_keyschedule.mind \
#                                                        >> /tmp/tls13rec_combined.mind
#   cat std/aes_gcm.mind                                 >> /tmp/tls13rec_combined.mind
#   grep -vE '^import std\.' std/tls13_record.mind       >> /tmp/tls13rec_combined.mind
#   mindc /tmp/tls13rec_combined.mind --emit-shared /tmp/tls13rec.so  # needs mlir-build
#
# Usage: python3 tls13_record_driver.py <tls13rec.so>

import ctypes
import sys

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

so_path = sys.argv[1]
T = ctypes.CDLL(so_path)

# ctypes signatures (all args are i64 addresses/lengths/values; return i64).
_SIGS = {
    "tls13_record_nonce": 3,
    "tls13_record_seal": 7,
    "tls13_record_open": 6,
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


def record(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        print(detail)


# ---------------------------------------------------------------------------
# RFC 8448 §3 published constants (reference (a)).
# Key/iv are the server_handshake_traffic_secret write key/iv (RFC 8448 §3,
# already KAT-verified in tests/tls13_keyschedule_driver.py).
# ---------------------------------------------------------------------------
KEY = bytes.fromhex("3fce516009c21727d0f2e4e86ee403bc")
IV = bytes.fromhex("5d313eb2671276ee13000b30")

# "{server} send handshake record: payload (657 octets)" — the concatenated
# EncryptedExtensions..Finished handshake messages (inner content type 0x16).
PAYLOAD = bytes.fromhex(
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
    "da4ab42c309572cb7fffee5454b78f0718"
)
assert len(PAYLOAD) == 657

# "{server} send handshake record: complete record (679 octets)" — the wire
# TLSCiphertext (5-byte header + 658-byte ciphertext + 16-byte tag) at seq 0.
RECORD = bytes.fromhex(
    "17030302a2d1ff334a56f5bff6594a07cc87b580233f500f45e489e7f33af35e"
    "df7869fcf40aa40aa2b8ea73f848a7ca07612ef9f945cb960b4068905123ea78"
    "b111b429ba9191cd05d2a389280f526134aadc7fc78c4b729df828b5ecf7b13b"
    "d9aefb0e57f271585b8ea9bb355c7c79020716cfb9b1183ef3ab20e37d57a6b9"
    "d7477609aee6e122a4cf51427325250c7d0e509289444c9b3a648f1d71035d2e"
    "d65b0e3cdd0cbae8bf2d0b227812cbb360987255cc744110c453baa4fcd61092"
    "8d809810e4b7ed1a8fd991f06aa6248204797e36a6a73b70a2559c09ead68694"
    "5ba246ab66e5edd8044b4c6de3fcf2a89441ac66272fd8fb330ef8190579b368"
    "4596c960bd596eea520a56a8d650f563aad27409960dca63d3e688611ea5e22f"
    "4415cf9538d51a200c27034272968a264ed6540c84838d89f72c24461aad6d26"
    "f59ecaba9acbbb317b66d902f4f292a36ac1b639c637ce343117b65962224531"
    "7b49eeda0c6258f100d7d961ffb138647e92ea330faeea6dfa31c7a84dc3bd7e"
    "1b7a6c7178af36879018e3f252107f243d243dc7339d5684c8b0378bf30244da"
    "8c87c843f5e56eb4c5e8280a2b48052cf93b16499a66db7cca71e4599426f7d4"
    "61e66f99882bd89fc50800becca62d6c74116dbd2972fda1fa80f85df881edbe"
    "5a37668936b335583b599186dc5c6918a396fa48a181d6b6fa4f9d62d513afbb"
    "992f2b992f67f8afe67f76913fa388cb5630c8ca01e0c65d11c66a1e2ac4c859"
    "77b7c7a6999bbf10dc35ae69f5515614636c0b9b68c19ed2e31c0b3b66763038"
    "ebba42f3b38edc0399f3a9f23faa63978c317fc9fa66a73f60f0504de93b5b84"
    "5e275592c12335ee340bbc4fddd502784016e4b3be7ef04dda49f4b440a30cb5"
    "d2af939828fd4ae3794e44f94df5a631ede42c1719bfdabf0253fe5175be898e"
    "750edc53370d2b"
)
assert len(RECORD) == 679

INNER_TYPE = 0x16  # handshake


# ---------------------------------------------------------------------------
# Reference composition (b): RFC 8446 §5.2/§5.3 seal built from scratch on
# pyca AESGCM — nonce = iv XOR left-padded seq_be8, aad = 5 header bytes.
# ---------------------------------------------------------------------------
def ref_nonce(iv: bytes, seq: int) -> bytes:
    return bytes(a ^ b for a, b in zip(iv, seq.to_bytes(12, "big")))


def ref_seal(key: bytes, iv: bytes, seq: int, content: bytes, itype: int) -> bytes:
    inner = content + bytes([itype])
    aad = b"\x17\x03\x03" + (len(inner) + 16).to_bytes(2, "big")
    return aad + AESGCM(key).encrypt(ref_nonce(iv, seq), inner, aad)


# ---------------------------------------------------------------------------
# CROSS-CHECK 1: prove composition (b) reproduces the RFC 8448 (a) record.
# If this assert fires the ground truth is wrong and we stop before touching MIND.
# ---------------------------------------------------------------------------
print("=" * 72)
print("Cross-check: pyca AESGCM composition (b) reproduces RFC 8448 §3 record (a)")
print("=" * 72)
b_record = ref_seal(KEY, IV, 0, PAYLOAD, INNER_TYPE)
assert b_record == RECORD, (
    f"pyca composition does NOT reproduce RFC 8448 record:\n"
    f"  got={b_record.hex()}\n  exp={RECORD.hex()}"
)
print(f"[ OK ] pyca seal(key,iv,seq=0,payload||0x16) == RFC 8448 complete record")
print(f"       record[:16]={RECORD[:16].hex()}  record[-16:]={RECORD[-16:].hex()}")
b_open = AESGCM(KEY).decrypt(ref_nonce(IV, 0), RECORD[5:], RECORD[:5])
assert b_open == PAYLOAD + bytes([INNER_TYPE])
print(f"[ OK ] pyca open(RFC record) == payload || 0x16")
print("Ground truth confirmed: (b) reproduces (a) byte-for-byte.\n")

# ---------------------------------------------------------------------------
# Now compare the MIND .so output to the (doubly-confirmed) constants.
# ---------------------------------------------------------------------------
print("=" * 72)
print("MIND std/tls13_record.mind vs RFC 8448 §3")
print("=" * 72)

key_b = buf(KEY)
iv_b = buf(IV)

# (a) seal(payload, seq=0) == RFC 8448 complete record, byte-for-byte.
pt_b = buf(PAYLOAD)
rec_out = out(len(RECORD) + 8)
ret = T.tls13_record_seal(
    addr(key_b), addr(iv_b), 0, addr(pt_b), len(PAYLOAD), INNER_TYPE, addr(rec_out)
)
got = rec_out.raw[: len(RECORD)]
record(
    "(a) seal(RFC payload, seq=0) == RFC 8448 complete record (679 B)",
    ret == len(RECORD) and got == RECORD,
    f"        ret={ret} (exp {len(RECORD)})\n"
    f"        got[:16]={got[:16].hex()}  got[-16:]={got[-16:].hex()}\n"
    f"        exp[:16]={RECORD[:16].hex()}  exp[-16:]={RECORD[-16:].hex()}",
)

# (b) open(RFC record, seq=0) == payload, inner content type 0x16.
rec_b = buf(RECORD)
ct_out = out(len(RECORD))
ret = T.tls13_record_open(
    addr(key_b), addr(iv_b), 0, addr(rec_b), len(RECORD), addr(ct_out)
)
got_type = ret & 0xFF if ret >= 0 else None
got_len = ret >> 8 if ret >= 0 else None
got_pt = ct_out.raw[: len(PAYLOAD)]
record(
    "(b) open(RFC record, seq=0) == RFC payload, inner type 0x16",
    ret >= 0 and got_type == INNER_TYPE and got_len == len(PAYLOAD) and got_pt == PAYLOAD,
    f"        ret={ret} -> type={got_type} len={got_len} "
    f"(exp type=0x16 len={len(PAYLOAD)})\n"
    f"        got[:16]={got_pt[:16].hex()}  got[-16:]={got_pt[-16:].hex()}\n"
    f"        exp[:16]={PAYLOAD[:16].hex()}  exp[-16:]={PAYLOAD[-16:].hex()}",
)

# (c) REJECT path — fail closed, never the plaintext.
# c1: flip a ciphertext byte (first byte after the header).
tampered_ct = bytearray(RECORD)
tampered_ct[5] ^= 0x01
t_b = buf(bytes(tampered_ct))
ct_out = out(len(RECORD))
ret = T.tls13_record_open(
    addr(key_b), addr(iv_b), 0, addr(t_b), len(RECORD), addr(ct_out)
)
leaked = ct_out.raw[: len(PAYLOAD)] == PAYLOAD
record(
    "(c1) open(tampered ciphertext byte) REJECTED (fail closed)",
    ret < 0 and not leaked,
    f"        ret={ret} (exp <0)  plaintext_leaked={leaked}",
)

# c2: flip a tag byte (last byte of the record).
tampered_tag = bytearray(RECORD)
tampered_tag[-1] ^= 0x01
t_b = buf(bytes(tampered_tag))
ct_out = out(len(RECORD))
ret = T.tls13_record_open(
    addr(key_b), addr(iv_b), 0, addr(t_b), len(RECORD), addr(ct_out)
)
leaked = ct_out.raw[: len(PAYLOAD)] == PAYLOAD
record(
    "(c2) open(tampered tag byte) REJECTED (fail closed)",
    ret < 0 and not leaked,
    f"        ret={ret} (exp <0)  plaintext_leaked={leaked}",
)

# (d) seq=1 nonce differs from seq=0 nonce; both match the RFC 8446 §5.3
# construction (validated ref_nonce); only the low 8 bytes may change.
n0 = out(12)
n1 = out(12)
T.tls13_record_nonce(addr(iv_b), 0, addr(n0))
T.tls13_record_nonce(addr(iv_b), 1, addr(n1))
mind_n0, mind_n1 = n0.raw[:12], n1.raw[:12]
ok = (
    mind_n0 == ref_nonce(IV, 0)
    and mind_n1 == ref_nonce(IV, 1)
    and mind_n0 != mind_n1
    and mind_n0[:4] == mind_n1[:4]
)
record(
    "(d) nonce(seq=1) != nonce(seq=0); both match RFC 8446 §5.3 (iv XOR seq_be8)",
    ok,
    f"        nonce(seq=0)={mind_n0.hex()} (exp {ref_nonce(IV, 0).hex()})\n"
    f"        nonce(seq=1)={mind_n1.hex()} (exp {ref_nonce(IV, 1).hex()})",
)

print("=" * 72)
total = len(results)
passed = sum(1 for r in results if r)
print(f"SUMMARY: {passed}/{total} checks PASSED")
print("=" * 72)
sys.exit(0 if passed == total else 1)
