#!/usr/bin/env python3
# Official-vector driver for std/tls13_handshake.mind (pure-MIND TLS 1.3
# handshake CRYPTO ORCHESTRATION: RFC 8446 §7.1 key schedule composition,
# §4.4.3 CertificateVerify, §4.4.4 Finished), replay-verified end to end
# against RFC 8448 §3 "Simple 1-RTT Handshake".
#
# The driver replays everything the CLIENT computes in the recorded handshake:
#   (a) x25519(client_priv, server_pub) == the RFC ECDHE shared secret
#   (b) tls13_hs_derive_handshake_secrets == RFC c/s hs traffic secrets +
#       record key/iv (3fce5160... / 5d313eb2...)
#   (c) tls13_hs_verify_cert_verify == 1 with the RFC server cert RSA key +
#       the RFC CertificateVerify signature (rsa_pss_rsae_sha256) — and == 0
#       for a bit-flipped signature / mangled signed content (fail-closed)
#   (d) tls13_hs_verify_server_finished == 1 for the RFC server Finished,
#       == 0 for a bit-flipped one (fail-closed)
#   (e) tls13_hs_derive_app_secrets == RFC master secret + c/s application
#       traffic secrets (+ key/iv vs the validated composition)
#   (f) tls13_hs_client_finished == the RFC client Finished verify_data
#
# GROUND TRUTH IS INDEPENDENT REFERENCES, cross-checked against each other
# BEFORE anything is compared to the MIND .so:
#   (1) the RFC 8448 §3 published hex constants (hardcoded below: ClientHello,
#       ServerHello, the 657-octet server flight — which CONTAINS the DER
#       certificate, the CertificateVerify signature and the server Finished —
#       the x25519 keys, the DHE shared secret, every traffic secret, the
#       client Finished verify_data), and
#   (2) from-scratch Python compositions built ONLY on hashlib/hmac/pow:
#       x25519 (RFC 7748 §5 Montgomery ladder), HKDF-Expand-Label /
#       Derive-Secret (RFC 8446 §7.1), RSASSA-PSS-VERIFY (RFC 8017
#       §8.1.2/§9.1.2), Finished MAC (RFC 8446 §4.4.4).
# The driver first PROVES composition (2) reproduces every RFC 8448 (1)
# constant — including verifying the RFC CertificateVerify signature over the
# exact §4.4.3 signed content with the RSA key extracted from the RFC's own
# DER certificate — so the ground truth is real, not circular; only then are
# the constants compared to the MIND .so output.  PASS/FAIL per case,
# verbatim hex compared.
#
# Build the .so first (combine deps in dependency order, stripping imports):
#   cat std/sha256.mind                                     > /tmp/tls13hs.mind
#   grep -v '^import std.sha256;' std/hkdf.mind            >> /tmp/tls13hs.mind
#   grep -v '^import std.sha256;' std/x509.mind            >> /tmp/tls13hs.mind
#   grep -vE '^import std\.(sha256|hkdf);' std/tls13_keyschedule.mind \
#                                                          >> /tmp/tls13hs.mind
#   grep -vE '^import std\.' std/aes_gcm.mind              >> /tmp/tls13hs.mind
#   grep -vE '^import std\.' std/tls13_record.mind         >> /tmp/tls13hs.mind
#   grep -vE '^import std\.' std/tls13_finished.mind       >> /tmp/tls13hs.mind
#   grep -vE '^import std\.' std/rsa_pss.mind              >> /tmp/tls13hs.mind
#   grep -vE '^import std\.' std/x25519.mind               >> /tmp/tls13hs.mind
#   grep -vE '^import std\.' std/tls13_handshake.mind      >> /tmp/tls13hs.mind
#   mindc /tmp/tls13hs.mind --emit-shared /tmp/tls13hs.so   # needs mlir-build
#
# Usage: python3 tls13_handshake_driver.py <tls13hs.so>

import ctypes
import hashlib
import hmac as _pyhmac
import sys

so_path = sys.argv[1]
T = ctypes.CDLL(so_path)

# ctypes signatures (all args are i64 addresses/lengths; return i64).
_SIGS = {
    "x25519": 3,
    "tls13_hs_derive_handshake_secrets": 3,
    "tls13_hs_derive_app_secrets": 3,
    "tls13_hs_cert_verify_content": 2,
    "tls13_hs_rsa_pss_verify": 8,
    "tls13_hs_verify_cert_verify": 6,
    "tls13_hs_verify_server_finished": 3,
    "tls13_hs_client_finished": 3,
    "tls13_peer_auth_supported": 0,
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
# RFC 8448 §3 "Simple 1-RTT Handshake" published constants (reference (1)).
# ---------------------------------------------------------------------------

# "{client} create an ephemeral x25519 key pair" — private key.
CLIENT_X25519_PRIV = bytes.fromhex(
    "49af42ba7f7994852d713ef2784bcbcaa7911de26adc5642cb634540e7ea5005")
# The client public key is IN the ClientHello key_share below (99381de5...).

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
CLIENT_X25519_PUB = CLIENT_HELLO[0x6D:0x6D + 32]  # key_share x25519 entry
assert CLIENT_X25519_PUB == bytes.fromhex(
    "99381de560e4bd43d23d8e435a7dbafeb3c06e51c13cae4d5413691e529aaf2c")

# "{server} construct a ServerHello handshake message: ServerHello (90 octets)"
SERVER_HELLO = bytes.fromhex(
    "020000560303a6af06a4121860dc5e6e60249cd34c95930c8ac5cb1434dac155"
    "772ed3e2692800130100002e00330024001d0020c9828876112095fe66762bdb"
    "f7c672e156d6cc253b833df1dd69b1b04e751f0f002b00020304")
assert len(SERVER_HELLO) == 90
SERVER_X25519_PUB = SERVER_HELLO[-38:-6]  # key_share x25519 entry
assert SERVER_X25519_PUB == bytes.fromhex(
    "c9828876112095fe66762bdbf7c672e156d6cc253b833df1dd69b1b04e751f0f")

# "IKM (32 octets)" of the Handshake-Secret extract = the x25519 shared secret.
DHE = bytes.fromhex("8bd4054fb55b9d63fdfbacf9f04b9f0d35e6d63f537563efd46272900f89492d")

# "{server} send handshake record: payload (657 octets)" — the concatenated
# EncryptedExtensions || Certificate || CertificateVerify || Finished
# handshake messages (same constant as tests/tls13_record_driver.py /
# tests/tls13_finished_driver.py).
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

# Split the flight into the four handshake messages (4-byte header + body).
EE_MSG = SERVER_FLIGHT[:40]           # 08 00 00 24 || 0x24 bytes
CERT_MSG = SERVER_FLIGHT[40:485]      # 0b 00 01 b9 || 0x1b9 bytes
CV_MSG = SERVER_FLIGHT[485:621]       # 0f 00 00 84 || 0x84 bytes
SF_MSG = SERVER_FLIGHT[621:657]       # 14 00 00 20 || verify_data
assert EE_MSG[:4] == bytes.fromhex("08000024")
assert CERT_MSG[:4] == bytes.fromhex("0b0001b9")
assert CV_MSG[:4] == bytes.fromhex("0f000084")
assert SF_MSG[:4] == bytes.fromhex("14000020")
RFC_SERVER_VERIFY_DATA = SF_MSG[4:]   # ...9572cb7fffee5454b78f0718

# CertificateVerify: algorithm 0x0804 = rsa_pss_rsae_sha256 (RFC 8446 §4.2.3),
# 2-byte signature length 0x0080 = 128 (RSA-1024!), then the signature.
assert CV_MSG[4:6] == bytes.fromhex("0804"), "not rsa_pss_rsae_sha256"
assert CV_MSG[6:8] == bytes.fromhex("0080")
CV_SIG = CV_MSG[8:136]
assert len(CV_SIG) == 128

# The server cert's RSA public key, extracted from the DER certificate inside
# the RFC's own Certificate message (modulus INTEGER: 02 81 81 00 || 128
# bytes; exponent INTEGER: 02 03 01 00 01 = 65537).
_midx = CERT_MSG.find(b"\x02\x81\x81\x00")
assert _midx > 0, "RSA modulus INTEGER not found in RFC certificate"
RSA_N = CERT_MSG[_midx + 4:_midx + 4 + 128]
assert len(RSA_N) == 128 and (RSA_N[0] & 0x80) != 0  # 1024-bit modulus
assert CERT_MSG[_midx + 132:_midx + 137] == b"\x02\x03\x01\x00\x01"
RSA_E = 65537

# RFC 8448 §3 printed secrets/keys (same constants as the committed
# tests/tls13_keyschedule_driver.py, which validates them all).
HELLO_HASH = bytes.fromhex(  # Transcript-Hash(ClientHello..ServerHello)
    "860c06edc07858ee8e78f0e7428c58edd6b43f2ca3e6e95f02ed063cf0e1cad8")
# Transcript-Hash(ClientHello..server Finished) — the RFC 8448 §3 "{server}
# derive secret 'tls13 c ap traffic'" printed hash input.  NOTE: this is the
# value the RFC actually prints (verbatim from rfc8448.txt); the FIN_HASH /
# cats / sats constants in tests/tls13_keyschedule_driver.py
# (...570726b0 / 51a49156... / 446d47f1...) appear NOWHERE in RFC 8448 —
# they are a mutually-consistent but foreign vector set (pre-existing
# transcription defect there; its composition cross-check is circular for
# those three entries because it derives cats/sats FROM its FIN_HASH).
FIN_HASH = bytes.fromhex(
    "9608102a0f1ccc6db6250b7b7e417b1a000eaada3daae4777a7686c9ff83df13")
RFC = {
    "handshake_secret": "1dc826e93606aa6fdc0aadc12f741b01046aa6b99f691ed221a9f0ca043fbeac",
    "chts": "b3eddb126e067f35a780b3abf45e2d8f3b1a950738f52e9600746a0e27a55a21",
    "shts": "b67b7d690cc16c4e75e54213cb2d37b4e9c912bcded9105d42befd59d391ad38",
    "master_secret": "18df06843d13a08bf2a449844c5f8a478001bc4d4c627984d5a41da8d0402919",
    # "{server} derive secret 'tls13 c ap traffic' / 's ap traffic'" expanded:
    "cats": "9e40646ce79a7f9dc05af8889bce6552875afa0b06df0087f792ebb7c17504a5",
    "sats": "a11af9f05531f856ad47116b45a950328204b4f44bfb6b3a4b4f1f3fcb631643",
    "s_hs_key": "3fce516009c21727d0f2e4e86ee403bc",
    "s_hs_iv": "5d313eb2671276ee13000b30",
    "c_hs_key": "dbfaa693d1762c5b666af5d950258d01",
    "c_hs_iv": "5bd3c71b836e0b76bb73265f",
    # "{server} derive write traffic keys for application data" (PRK = sats):
    "s_ap_key": "9f02283b6c9c07efc26bb9f2ac92e356",
    "s_ap_iv": "cf782b88dd83549aadf1e984",
}
RFC = {k: bytes.fromhex(v) for k, v in RFC.items()}

# "{client} calculate finished" — the client Finished verify_data (RFC 8448 §3
# prints the client's Finished handshake message 14 00 00 20 || this value).
RFC_CLIENT_VERIFY_DATA = bytes.fromhex(
    "a8ec436d677634ae525ac1fcebe11a039ec17694fac6e98527b642f2edd5ce61")

# Transcript hashes at each stage (Transcript-Hash = SHA-256 of the message
# concatenation, RFC 8446 §4.4.1).  Anchored to the RFC-published hashes at
# both ends: CH..SH == HELLO_HASH and CH..SF == FIN_HASH.
TH_CH_SH = hashlib.sha256(CLIENT_HELLO + SERVER_HELLO).digest()
TH_TO_CERT = hashlib.sha256(CLIENT_HELLO + SERVER_HELLO + EE_MSG + CERT_MSG).digest()
TH_TO_CV = hashlib.sha256(
    CLIENT_HELLO + SERVER_HELLO + EE_MSG + CERT_MSG + CV_MSG).digest()
TH_TO_SF = hashlib.sha256(
    CLIENT_HELLO + SERVER_HELLO + EE_MSG + CERT_MSG + CV_MSG + SF_MSG).digest()
assert TH_CH_SH == HELLO_HASH, "transcript split broken: CH..SH hash mismatch"
assert TH_TO_SF == FIN_HASH, "transcript split broken: CH..SF hash mismatch"

# ---------------------------------------------------------------------------
# Reference composition (2): everything from scratch on hashlib/hmac/pow.
# ---------------------------------------------------------------------------

# x25519 (RFC 7748 §5) — Montgomery ladder over GF(2^255 - 19).
_P25519 = 2**255 - 19


def ref_x25519(k: bytes, u: bytes) -> bytes:
    kb = bytearray(k)
    kb[0] &= 248
    kb[31] &= 127
    kb[31] |= 64
    kn = int.from_bytes(bytes(kb), "little")
    x1 = int.from_bytes(u, "little") & ((1 << 255) - 1)
    x2, z2, x3, z3 = 1, 0, x1, 1
    swap = 0
    for t in reversed(range(255)):
        kt = (kn >> t) & 1
        swap ^= kt
        if swap:
            x2, x3 = x3, x2
            z2, z3 = z3, z2
        swap = kt
        A = (x2 + z2) % _P25519
        AA = A * A % _P25519
        B = (x2 - z2) % _P25519
        BB = B * B % _P25519
        E = (AA - BB) % _P25519
        C = (x3 + z3) % _P25519
        D = (x3 - z3) % _P25519
        DA = D * A % _P25519
        CB = C * B % _P25519
        x3 = (DA + CB) % _P25519
        x3 = x3 * x3 % _P25519
        z3 = (DA - CB) % _P25519
        z3 = z3 * z3 % _P25519
        z3 = z3 * x1 % _P25519
        x2 = AA * BB % _P25519
        z2 = E * (AA + 121665 * E % _P25519) % _P25519
    if swap:
        x2, x3 = x3, x2
        z2, z3 = z3, z2
    return (x2 * pow(z2, _P25519 - 2, _P25519) % _P25519).to_bytes(32, "little")


# HKDF / key schedule (RFC 5869 + RFC 8446 §7.1).
def ref_hkdf_extract(salt: bytes, ikm: bytes) -> bytes:
    if not salt:
        salt = b"\x00" * 32
    return _pyhmac.new(salt, ikm, hashlib.sha256).digest()


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


def ref_derive_secret(secret: bytes, label: bytes, thash: bytes) -> bytes:
    return ref_expand_label(secret, label, thash, 32)


# RSASSA-PSS-VERIFY (RFC 8017 §8.1.2 / §9.1.2), SHA-256 / MGF1-SHA256.
def ref_mgf1(seed: bytes, mask_len: int) -> bytes:
    m = b""
    c = 0
    while len(m) < mask_len:
        m += hashlib.sha256(seed + c.to_bytes(4, "big")).digest()
        c += 1
    return m[:mask_len]


def ref_pss_verify(n: bytes, e: int, sig: bytes, msg: bytes, salt_len: int = 32) -> bool:
    k = len(n)
    if len(sig) != k:
        return False
    nn = int.from_bytes(n, "big")
    s = int.from_bytes(sig, "big")
    if s >= nn:
        return False
    em = pow(s, e, nn).to_bytes(k, "big")
    em_len = k                      # emBits = 8k - 1 (top bit of n set)
    db_len = em_len - 33
    if em[-1] != 0xBC or (em[0] & 0x80) != 0:
        return False
    h = em[db_len:db_len + 32]
    db = bytes(a ^ b for a, b in zip(em[:db_len], ref_mgf1(h, db_len)))
    db = bytes([db[0] & 0x7F]) + db[1:]
    one_pos = db_len - salt_len - 1
    if any(db[:one_pos]) or db[one_pos] != 0x01:
        return False
    salt = db[db_len - salt_len:]
    mp = b"\x00" * 8 + hashlib.sha256(msg).digest() + salt
    return hashlib.sha256(mp).digest() == h


# TLS 1.3 §4.4.3 CertificateVerify signed content (server variant).
CV_CONTEXT = b"TLS 1.3, server CertificateVerify"
assert len(CV_CONTEXT) == 33


def ref_cv_content(thash: bytes) -> bytes:
    return b"\x20" * 64 + CV_CONTEXT + b"\x00" + thash


# ---------------------------------------------------------------------------
# CROSS-CHECK 1: prove composition (2) reproduces every RFC 8448 (1) constant.
# If any assert fires the ground truth is wrong and we stop before touching MIND.
# ---------------------------------------------------------------------------
print("=" * 72)
print("Cross-check: Python composition (2) reproduces RFC 8448 §3 constants (1)")
print("=" * 72)

# x25519: priv -> pub (basepoint 9) and priv x server_pub -> the RFC DHE.
b_client_pub = ref_x25519(CLIENT_X25519_PRIV, b"\x09" + b"\x00" * 31)
assert b_client_pub == CLIENT_X25519_PUB, (
    f"x25519 pubkey {b_client_pub.hex()} != RFC ClientHello key_share")
print(f"[ OK ] composition(2) == RFC8448  x25519 client pubkey  {b_client_pub.hex()}")
b_dhe = ref_x25519(CLIENT_X25519_PRIV, SERVER_X25519_PUB)
assert b_dhe == DHE, f"x25519 shared {b_dhe.hex()} != RFC DHE {DHE.hex()}"
print(f"[ OK ] composition(2) == RFC8448  ECDHE shared secret   {b_dhe.hex()}")

# Key schedule: Early -> Handshake -> traffic secrets -> Master -> app secrets.
EMPTY_HASH = hashlib.sha256(b"").digest()
b_early = ref_hkdf_extract(b"", b"\x00" * 32)
b_hs = ref_hkdf_extract(ref_derive_secret(b_early, b"derived", EMPTY_HASH), DHE)
b_chts = ref_derive_secret(b_hs, b"c hs traffic", TH_CH_SH)
b_shts = ref_derive_secret(b_hs, b"s hs traffic", TH_CH_SH)
b_master = ref_hkdf_extract(
    ref_derive_secret(b_hs, b"derived", EMPTY_HASH), b"\x00" * 32)
b_cats = ref_derive_secret(b_master, b"c ap traffic", TH_TO_SF)
b_sats = ref_derive_secret(b_master, b"s ap traffic", TH_TO_SF)
for k, v in [("handshake_secret", b_hs), ("chts", b_chts), ("shts", b_shts),
             ("master_secret", b_master), ("cats", b_cats), ("sats", b_sats),
             ("s_hs_key", ref_expand_label(b_shts, b"key", b"", 16)),
             ("s_hs_iv", ref_expand_label(b_shts, b"iv", b"", 12)),
             ("c_hs_key", ref_expand_label(b_chts, b"key", b"", 16)),
             ("c_hs_iv", ref_expand_label(b_chts, b"iv", b"", 12))]:
    assert v == RFC[k], f"composition (2) != RFC 8448 {k}: {v.hex()} != {RFC[k].hex()}"
    print(f"[ OK ] composition(2) == RFC8448  {k:18s} {RFC[k].hex()}")

# App-phase record key/iv.  The server direction is RFC-printed ("{server}
# derive write traffic keys for application data"); the client direction is
# validated through the composition, proven RFC-faithful above.
b_s_ap_key = ref_expand_label(b_sats, b"key", b"", 16)
b_s_ap_iv = ref_expand_label(b_sats, b"iv", b"", 12)
assert b_s_ap_key == RFC["s_ap_key"] and b_s_ap_iv == RFC["s_ap_iv"]
print(f"[ OK ] composition(2) == RFC8448  s_ap_key/iv        "
      f"{RFC['s_ap_key'].hex()} / {RFC['s_ap_iv'].hex()}")
b_c_ap_key = ref_expand_label(b_cats, b"key", b"", 16)
b_c_ap_iv = ref_expand_label(b_cats, b"iv", b"", 12)

# CertificateVerify: verify the RFC's own signature over the exact §4.4.3
# content with the RSA key extracted from the RFC's own DER certificate.
b_cv_content = ref_cv_content(TH_TO_CERT)
assert ref_pss_verify(RSA_N, RSA_E, CV_SIG, b_cv_content), (
    "RFC 8448 CertificateVerify signature FAILED to verify over the §4.4.3 "
    "content — extraction or construction is wrong")
print("[ OK ] composition(2): RFC CertificateVerify sig VERIFIES over the")
print(f"       exact §4.4.3 content (thash CH..Cert = {TH_TO_CERT.hex()})")
# ...and fails closed on a mangled context prefix (64 x 0x20 -> 63 + 0x21).
assert not ref_pss_verify(
    RSA_N, RSA_E, CV_SIG, b"\x21" + b_cv_content[1:]), "mangled context verified!?"
print("[ OK ] composition(2): mangled context prefix REJECTED (fail-closed)")

# Server Finished: HMAC(ExpandLabel(shts,'finished','',32), thash CH..CV).
b_s_finkey = ref_expand_label(b_shts, b"finished", b"", 32)
b_s_vd = _pyhmac.new(b_s_finkey, TH_TO_CV, hashlib.sha256).digest()
assert b_s_vd == RFC_SERVER_VERIFY_DATA, (
    f"server verify_data {b_s_vd.hex()} != RFC {RFC_SERVER_VERIFY_DATA.hex()}")
print(f"[ OK ] composition(2) == RFC8448  server_verify_data   {b_s_vd.hex()}")

# Client Finished: HMAC(ExpandLabel(chts,'finished','',32), thash CH..SF).
b_c_finkey = ref_expand_label(b_chts, b"finished", b"", 32)
b_c_vd = _pyhmac.new(b_c_finkey, TH_TO_SF, hashlib.sha256).digest()
assert b_c_vd == RFC_CLIENT_VERIFY_DATA, (
    f"client verify_data {b_c_vd.hex()} != RFC {RFC_CLIENT_VERIFY_DATA.hex()}")
print(f"[ OK ] composition(2) == RFC8448  client_verify_data   {b_c_vd.hex()}")
print("Ground truth confirmed: composition (2) reproduces every RFC 8448 §3")
print("constant end to end (ECDHE, key schedule, CertificateVerify, Finished).\n")

# ---------------------------------------------------------------------------
# Now replay the handshake through the MIND .so and compare to the
# (doubly-confirmed) constants.
# ---------------------------------------------------------------------------
print("=" * 72)
print("MIND std/tls13_handshake.mind vs RFC 8448 §3 (full client-side replay)")
print("=" * 72)

# (a) ECDHE: x25519(client_priv, server_pub) == RFC shared secret.
priv_b = buf(CLIENT_X25519_PRIV)
spub_b = buf(SERVER_X25519_PUB)
dhe_out = out(32)
T.x25519(addr(priv_b), addr(spub_b), addr(dhe_out))
record("(a) x25519(client_priv, server_pub) == RFC ECDHE shared secret",
       dhe_out.raw[:32], DHE)

# (b) Handshake-phase secrets from the shared secret + Transcript(CH..SH).
hh_b = buf(TH_CH_SH)
hs_out = out(152)
T.tls13_hs_derive_handshake_secrets(addr(dhe_out), addr(hh_b), addr(hs_out))
record("(b) Handshake Secret (out+0)", hs_out.raw[0:32], RFC["handshake_secret"])
record("(b) client_handshake_traffic_secret (out+32)", hs_out.raw[32:64], RFC["chts"])
record("(b) server_handshake_traffic_secret (out+64)", hs_out.raw[64:96], RFC["shts"])
record("(b) client hs record key (out+96)", hs_out.raw[96:112], RFC["c_hs_key"])
record("(b) client hs record iv  (out+112)", hs_out.raw[112:124], RFC["c_hs_iv"])
record("(b) server hs record key (out+124) == RFC 3fce5160...",
       hs_out.raw[124:140], RFC["s_hs_key"])
record("(b) server hs record iv  (out+140) == RFC 5d313eb2...",
       hs_out.raw[140:152], RFC["s_hs_iv"])

# (c) Server CertificateVerify (RFC 8446 §4.4.3) — server authentication.
thc_b = buf(TH_TO_CERT)
cvc = out(130)
n130 = T.tls13_hs_cert_verify_content(addr(thc_b), addr(cvc))
record_int("(c0) tls13_hs_cert_verify_content returns 130", n130, 130)
record("(c0) §4.4.3 signed content byte-exact (64x20 || ctx || 00 || thash)",
       cvc.raw[:130], b_cv_content)

n_b = buf(RSA_N)
sig_b = buf(CV_SIG)
rc = T.tls13_hs_verify_cert_verify(
    addr(n_b), 128, RSA_E, addr(sig_b), 128, addr(thc_b))
record_int("(c1) tls13_hs_verify_cert_verify(RFC cert key, RFC sig) -> 1", rc, 1)

flipped_sig = bytearray(CV_SIG)
flipped_sig[0] ^= 0x01
bad_sig_b = buf(bytes(flipped_sig))
rc = T.tls13_hs_verify_cert_verify(
    addr(n_b), 128, RSA_E, addr(bad_sig_b), 128, addr(thc_b))
record_int("(c2) bit-flipped signature -> 0 (fail-closed)", rc, 0)

flipped_th = bytearray(TH_TO_CERT)
flipped_th[0] ^= 0x01
bad_th_b = buf(bytes(flipped_th))
rc = T.tls13_hs_verify_cert_verify(
    addr(n_b), 128, RSA_E, addr(sig_b), 128, addr(bad_th_b))
record_int("(c3) mangled signed content (flipped transcript hash) -> 0", rc, 0)

# Direct PSS entry point: a mangled context prefix must also fail (proves the
# 64-space || context-string construction is load-bearing, not decorative).
mangled = bytearray(b_cv_content)
mangled[0] = 0x21  # first pad byte 0x20 -> 0x21
mang_b = buf(bytes(mangled))
rc = T.tls13_hs_rsa_pss_verify(
    addr(n_b), 128, RSA_E, addr(sig_b), 128, addr(mang_b), 130, 32)
record_int("(c4) tls13_hs_rsa_pss_verify(mangled context prefix) -> 0", rc, 0)
good_b = buf(b_cv_content)
rc = T.tls13_hs_rsa_pss_verify(
    addr(n_b), 128, RSA_E, addr(sig_b), 128, addr(good_b), 130, 32)
record_int("(c5) tls13_hs_rsa_pss_verify(exact content) -> 1", rc, 1)

# (d) Server Finished (RFC 8446 §4.4.4) — fail-closed verification.
shts_b = buf(hs_out.raw[64:96])
thcv_b = buf(TH_TO_CV)
vd_b = buf(RFC_SERVER_VERIFY_DATA)
rc = T.tls13_hs_verify_server_finished(addr(shts_b), addr(thcv_b), addr(vd_b))
record_int("(d1) tls13_hs_verify_server_finished(RFC verify_data) -> 1", rc, 1)

flipped_vd = bytearray(RFC_SERVER_VERIFY_DATA)
flipped_vd[17] ^= 0x40
bad_vd_b = buf(bytes(flipped_vd))
rc = T.tls13_hs_verify_server_finished(addr(shts_b), addr(thcv_b), addr(bad_vd_b))
record_int("(d2) bit-flipped server Finished -> 0 (fail-closed)", rc, 0)

# (e) Application-phase secrets from the Handshake Secret + Transcript(CH..SF).
hs_secret_b = buf(hs_out.raw[0:32])
thsf_b = buf(TH_TO_SF)
ap_out = out(152)
T.tls13_hs_derive_app_secrets(addr(hs_secret_b), addr(thsf_b), addr(ap_out))
record("(e) Master Secret (out+0)", ap_out.raw[0:32], RFC["master_secret"])
record("(e) client_application_traffic_secret_0 (out+32)",
       ap_out.raw[32:64], RFC["cats"])
record("(e) server_application_traffic_secret_0 (out+64)",
       ap_out.raw[64:96], RFC["sats"])
record("(e) client ap record key (out+96)  [vs validated (2)]",
       ap_out.raw[96:112], b_c_ap_key)
record("(e) client ap record iv  (out+112) [vs validated (2)]",
       ap_out.raw[112:124], b_c_ap_iv)
record("(e) server ap record key (out+124) [vs validated (2)]",
       ap_out.raw[124:140], b_s_ap_key)
record("(e) server ap record iv  (out+140) [vs validated (2)]",
       ap_out.raw[140:152], b_s_ap_iv)

# (f) The client's own Finished verify_data over Transcript(CH..SF).
chts_b = buf(hs_out.raw[32:64])
cvd = out(32)
T.tls13_hs_client_finished(addr(chts_b), addr(thsf_b), addr(cvd))
record("(f) tls13_hs_client_finished == RFC client Finished verify_data",
       cvd.raw[:32], RFC_CLIENT_VERIFY_DATA)

# Capability scope (honesty marker): std/tls13 does NOT authenticate a peer.
# tls13_hs_verify_cert_verify checks signature math only (no chain/validity/
# hostname). This must stay 0 until real peer auth lands, or the driver is RED.
record_int("scope: tls13_peer_auth_supported()==0 (no chain/validity/hostname)",
           T.tls13_peer_auth_supported(), 0)

print("=" * 72)
total = len(results)
passed = sum(1 for r in results if r)
print(f"SUMMARY: {passed}/{total} checks PASSED")
print("=" * 72)
sys.exit(0 if passed == total else 1)
