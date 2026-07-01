#!/usr/bin/env python3
# Official-vector driver for std/tls13_keyschedule.mind (pure-MIND TLS 1.3 key
# schedule, RFC 8446 §7.1).
#
# Exercises every pub fn entry point against the RFC 8448 §3 "Simple 1-RTT
# Handshake" published constants (early_secret, derived, handshake_secret,
# c/s hs traffic secrets, master secret, c/s ap traffic secrets, and the
# per-secret record-protection key/iv values — ALL spelled out in hex in
# RFC 8448 §3, used verbatim as known-answer vectors), plus HKDF-Expand-Label /
# Derive-Secret unit checks and the Finished-MAC key derivation.
#
# GROUND TRUTH IS TWO INDEPENDENT REFERENCES, cross-checked against each other
# BEFORE either is compared to the MIND .so:
#   (a) the RFC 8448 §3 published hex constants (hardcoded below), and
#   (b) a from-scratch Python composition of HKDF-Expand-Label / Derive-Secret
#       built only on hashlib HMAC-SHA256.
# The driver first PROVES composition (b) reproduces every RFC 8448 (a) constant
# (so the ground truth is real, not circular); only then are the constants
# compared to the MIND .so output.  PASS/FAIL per case, verbatim hex compared.
#
# Build the .so first (combine deps in dependency order, stripping imports —
# exactly as hkdf+sha256 are combined; sha256 MUST precede hkdf MUST precede
# tls13_keyschedule):
#   cat std/sha256.mind                              > /tmp/tls13_combined.mind
#   grep -v '^import std.sha256;' std/hkdf.mind     >> /tmp/tls13_combined.mind
#   grep -vE '^import std\.(sha256|hkdf);' std/tls13_keyschedule.mind \
#                                                   >> /tmp/tls13_combined.mind
#   mindc /tmp/tls13_combined.mind --emit-shared /tmp/tls13.so   # needs mlir-build
#
# Usage: python3 tls13_keyschedule_driver.py <tls13.so>

import ctypes
import hashlib
import hmac as _pyhmac
import sys

so_path = sys.argv[1]
T = ctypes.CDLL(so_path)

# ctypes signatures (all args are i64 addresses/lengths; return i64).
_SIGS = {
    "tls13_hkdf_expand_label": 8,
    "tls13_derive_secret_hashed": 6,
    "tls13_derive_secret": 7,
    "tls13_early_secret": 1,
    "tls13_handshake_secret": 2,
    "tls13_client_handshake_traffic_secret": 3,
    "tls13_server_handshake_traffic_secret": 3,
    "tls13_master_secret": 2,
    "tls13_client_application_traffic_secret": 3,
    "tls13_server_application_traffic_secret": 3,
    "tls13_derive_key": 2,
    "tls13_derive_iv": 2,
    "tls13_finished_key": 2,
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


# ---------------------------------------------------------------------------
# Reference composition (b): HKDF-Expand-Label / Derive-Secret from scratch,
# built ONLY on hashlib HMAC-SHA256 (RFC 5869 + RFC 8446 §7.1).
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# RFC 8448 §3 "Simple 1-RTT Handshake" published constants (reference (a)).
# ---------------------------------------------------------------------------
DHE = bytes.fromhex("8bd4054fb55b9d63fdfbacf9f04b9f0d35e6d63f537563efd46272900f89492d")
HELLO_HASH = bytes.fromhex(  # Transcript-Hash(ClientHello..ServerHello)
    "860c06edc07858ee8e78f0e7428c58edd6b43f2ca3e6e95f02ed063cf0e1cad8")
FIN_HASH = bytes.fromhex(  # Transcript-Hash(ClientHello..server Finished)
    "9608102a0f1ccc6db6250b7b7e417b1a570726b0eb16be7576e785e75f2e7fac")

RFC = {
    "early_secret": "33ad0a1c607ec03b09e6cd9893680ce210adf300aa1f2660e1b22e10f170f92a",
    "derived_early": "6f2615a108c702c5678f54fc9dbab69716c076189c48250cebeac3576c3611ba",
    "handshake_secret": "1dc826e93606aa6fdc0aadc12f741b01046aa6b99f691ed221a9f0ca043fbeac",
    "chts": "b3eddb126e067f35a780b3abf45e2d8f3b1a950738f52e9600746a0e27a55a21",
    "shts": "b67b7d690cc16c4e75e54213cb2d37b4e9c912bcded9105d42befd59d391ad38",
    "master_secret": "18df06843d13a08bf2a449844c5f8a478001bc4d4c627984d5a41da8d0402919",
    "cats": "51a49156290c60a6ed39effbe1768d977b9719719cab8df4ef518c538138fef6",
    "sats": "446d47f14fbdf924b34e1dd19bb5c2aa2a8082e1adb5580dbc43fe279830f1c6",
    # Record-protection key/iv, RFC 8448 §3 write keys/ivs.
    "s_hs_key": "3fce516009c21727d0f2e4e86ee403bc",
    "s_hs_iv": "5d313eb2671276ee13000b30",
    "c_hs_key": "dbfaa693d1762c5b666af5d950258d01",
    "c_hs_iv": "5bd3c71b836e0b76bb73265f",
}
RFC = {k: bytes.fromhex(v) for k, v in RFC.items()}

EMPTY_HASH = hashlib.sha256(b"").digest()  # Transcript-Hash("")

# ---------------------------------------------------------------------------
# CROSS-CHECK 1: prove composition (b) reproduces every RFC 8448 (a) constant.
# If any assert fires the ground truth is wrong and we stop before touching MIND.
# ---------------------------------------------------------------------------
print("=" * 72)
print("Cross-check: Python composition (b) reproduces RFC 8448 §3 constants (a)")
print("=" * 72)
b_early = ref_hkdf_extract(b"", b"\x00" * 32)
b_derived_early = ref_derive_secret(b_early, b"derived", EMPTY_HASH)
b_hs = ref_hkdf_extract(b_derived_early, DHE)
b_chts = ref_derive_secret(b_hs, b"c hs traffic", HELLO_HASH)
b_shts = ref_derive_secret(b_hs, b"s hs traffic", HELLO_HASH)
b_derived_hs = ref_derive_secret(b_hs, b"derived", EMPTY_HASH)
b_master = ref_hkdf_extract(b_derived_hs, b"\x00" * 32)
b_cats = ref_derive_secret(b_master, b"c ap traffic", FIN_HASH)
b_sats = ref_derive_secret(b_master, b"s ap traffic", FIN_HASH)
b_s_hs_key = ref_expand_label(b_shts, b"key", b"", 16)
b_s_hs_iv = ref_expand_label(b_shts, b"iv", b"", 12)
b_c_hs_key = ref_expand_label(b_chts, b"key", b"", 16)
b_c_hs_iv = ref_expand_label(b_chts, b"iv", b"", 12)

_xcheck = {
    "early_secret": b_early, "derived_early": b_derived_early,
    "handshake_secret": b_hs, "chts": b_chts, "shts": b_shts,
    "master_secret": b_master, "cats": b_cats, "sats": b_sats,
    "s_hs_key": b_s_hs_key, "s_hs_iv": b_s_hs_iv,
    "c_hs_key": b_c_hs_key, "c_hs_iv": b_c_hs_iv,
}
for k, v in _xcheck.items():
    ok = v == RFC[k]
    assert ok, f"composition (b) does NOT reproduce RFC 8448 {k}: {v.hex()} != {RFC[k].hex()}"
    print(f"[ OK ] composition(b) == RFC8448  {k:18s} {RFC[k].hex()}")
print("Ground truth confirmed: (b) reproduces (a) for all 12 published constants.\n")

# ---------------------------------------------------------------------------
# Now compare the MIND .so output to the (doubly-confirmed) constants.
# ---------------------------------------------------------------------------
print("=" * 72)
print("MIND std/tls13_keyschedule.mind vs RFC 8448 §3")
print("=" * 72)

dhe_b = buf(DHE)
hello_b = buf(HELLO_HASH)
fin_b = buf(FIN_HASH)

# HKDF-Expand-Label / Derive-Secret unit checks (against composition (b),
# itself RFC-validated above).
ob = out(32)
sec_b = buf(b_shts)
lbl_b = buf(b"key")
T.tls13_hkdf_expand_label(addr(sec_b), 32, addr(lbl_b), 3, addr(sec_b), 0, 16, addr(ob))
record("HKDF-Expand-Label(shts,'key','',16)", ob.raw[:16], RFC["s_hs_key"])

hs_b = buf(b_hs)
ob = out(32)
lbl_b = buf(b"c hs traffic")
T.tls13_derive_secret_hashed(addr(hs_b), 32, addr(lbl_b), 12, addr(hello_b), addr(ob))
record("Derive-Secret(hs,'c hs traffic',hello) [hashed form]", ob.raw[:32], RFC["chts"])

# Full Derive-Secret form (hashes empty messages -> SHA256('') internally):
early_b = buf(b_early)
ob = out(32)
lbl_b = buf(b"derived")
dummy = out(1)
T.tls13_derive_secret(addr(early_b), 32, addr(lbl_b), 7, addr(dummy), 0, addr(ob))
record("Derive-Secret(early,'derived','') [full form, SHA256(\"\")]",
       ob.raw[:32], RFC["derived_early"])

# Full key-schedule chain via the high-level pub fns.
ob = out(32)
T.tls13_early_secret(addr(ob))
record("Early Secret", ob.raw[:32], RFC["early_secret"])

hs_out = out(32)
T.tls13_handshake_secret(addr(dhe_b), addr(hs_out))
record("Handshake Secret", hs_out.raw[:32], RFC["handshake_secret"])
hs_b = buf(hs_out.raw[:32])

ob = out(32)
T.tls13_client_handshake_traffic_secret(addr(hs_b), addr(hello_b), addr(ob))
record("client_handshake_traffic_secret", ob.raw[:32], RFC["chts"])
chts_b = buf(ob.raw[:32])

ob = out(32)
T.tls13_server_handshake_traffic_secret(addr(hs_b), addr(hello_b), addr(ob))
record("server_handshake_traffic_secret", ob.raw[:32], RFC["shts"])
shts_b = buf(ob.raw[:32])

ms_out = out(32)
T.tls13_master_secret(addr(hs_b), addr(ms_out))
record("Master Secret", ms_out.raw[:32], RFC["master_secret"])
master_b = buf(ms_out.raw[:32])

ob = out(32)
T.tls13_client_application_traffic_secret(addr(master_b), addr(fin_b), addr(ob))
record("client_application_traffic_secret_0", ob.raw[:32], RFC["cats"])

ob = out(32)
T.tls13_server_application_traffic_secret(addr(master_b), addr(fin_b), addr(ob))
record("server_application_traffic_secret_0", ob.raw[:32], RFC["sats"])

# Record-protection key/iv from the handshake traffic secrets.
kb = out(16)
T.tls13_derive_key(addr(shts_b), addr(kb))
record("server_handshake key = ExpandLabel(shts,'key','',16)", kb.raw[:16], RFC["s_hs_key"])
ivb = out(12)
T.tls13_derive_iv(addr(shts_b), addr(ivb))
record("server_handshake iv  = ExpandLabel(shts,'iv','',12)", ivb.raw[:12], RFC["s_hs_iv"])
kb = out(16)
T.tls13_derive_key(addr(chts_b), addr(kb))
record("client_handshake key = ExpandLabel(chts,'key','',16)", kb.raw[:16], RFC["c_hs_key"])
ivb = out(12)
T.tls13_derive_iv(addr(chts_b), addr(ivb))
record("client_handshake iv  = ExpandLabel(chts,'iv','',12)", ivb.raw[:12], RFC["c_hs_iv"])

# Finished-MAC key (RFC 8446 §4.4.4).  RFC 8448 §3 does not print finished_key as
# a standalone hex line, so it is checked against composition (b) — which is
# proven byte-identical to every published RFC 8448 constant above, making it a
# validated (non-circular) reference for this corollary derivation.
b_s_finkey = ref_expand_label(b_shts, b"finished", b"", 32)
fkb = out(32)
T.tls13_finished_key(addr(shts_b), addr(fkb))
record("finished_key(shts) = ExpandLabel(shts,'finished','',32) [vs validated (b)]",
       fkb.raw[:32], b_s_finkey)

print("=" * 72)
total = len(results)
passed = sum(1 for r in results if r)
print(f"SUMMARY: {passed}/{total} checks PASSED")
print("=" * 72)
sys.exit(0 if passed == total else 1)
