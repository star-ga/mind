#!/usr/bin/env python3
# Ground-truth driver for std/ecdsa_p256.mind (pure-MIND ECDSA P-256/SHA-256
# signature verification, FIPS 186-5 §6.4.2 / SEC1 §4.1.4 — the
# ecdsa_secp256r1_sha256 scheme TLS 1.3 uses for CertificateVerify with ECDSA
# certificates, RFC 8446 §4.4.3).
#
# Exercises the compiled .so against pyca/cryptography as the independent
# ground truth (every expected value is produced by pyca BEFORE the MIND
# result is computed — non-circular):
#   - p256_scalar_base_mult : d*G affine coordinates vs pyca
#                             ec.derive_private_key(d).public_numbers() for the
#                             RFC 6979 A.2.5 key and a fresh random key.
#   - ecdsa_p256_verify     : a fresh SECP256R1 key signs a message with
#                             ec.ECDSA(SHA256); (r, s) extracted via
#                             decode_dss_signature, Q via public_numbers().
#                             MIND must ACCEPT that signature, and REJECT a
#                             bit-flipped r, a bit-flipped s, a different
#                             message, and a wrong public key — each rejection
#                             cross-checked against pyca's own verdict.
#   - fail-closed extras    : r=0, s=0, r=n, s=n (out of [1, n-1]),
#                             Q not on the curve.
#   - RFC 6979 A.2.5 KAT    : the P-256/SHA-256 deterministic-ECDSA known-
#                             answer vector for message "sample", cross-checked
#                             against pyca before MIND sees it.
#
# Build the .so first (std/ecdsa_p256.mind composes std/sha256.mind, so
# combine the two sources with the import line stripped, exactly as
# tests/rsa_pss_driver.py combines sha256+x509+rsa_pss):
#   cat std/sha256.mind                    >  /tmp/ecdsa_p256_combined.mind
#   grep -v '^import ' std/ecdsa_p256.mind >> /tmp/ecdsa_p256_combined.mind
#   mindc /tmp/ecdsa_p256_combined.mind --emit-shared /tmp/ecdsa_p256.so
#
# Usage: python3 ecdsa_p256_driver.py <ecdsa_p256.so>

import ctypes
import sys

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import (
    decode_dss_signature,
    encode_dss_signature,
)
from cryptography.exceptions import InvalidSignature

so_path = sys.argv[1]
E = ctypes.CDLL(so_path)
E.ecdsa_p256_verify.restype = ctypes.c_int64
E.ecdsa_p256_verify.argtypes = [ctypes.c_int64] * 6
E.p256_scalar_base_mult.restype = ctypes.c_int64
E.p256_scalar_base_mult.argtypes = [ctypes.c_int64] * 3

# NIST P-256 group order (SEC2 secp256r1).
N = 0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551

results = []


def buf(data: bytes):
    return ctypes.create_string_buffer(bytes(data), max(1, len(data)))


def out(n: int):
    return ctypes.create_string_buffer(max(1, n))


def addr(b):
    return ctypes.cast(b, ctypes.c_void_p).value


def i2b32(x: int) -> bytes:
    return x.to_bytes(32, "big")


def record_bool(name, cond, detail=""):
    results.append(bool(cond))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}  {detail}")


def mind_verify(qx: int, qy: int, r: int, s: int, msg: bytes) -> int:
    qxb, qyb = buf(i2b32(qx)), buf(i2b32(qy))
    rb, sb, mb = buf(i2b32(r)), buf(i2b32(s)), buf(msg)
    rc = E.ecdsa_p256_verify(addr(qxb), addr(qyb), addr(rb), addr(sb),
                             addr(mb), len(msg))
    assert rc in (0, 1), f"non-boolean rc={rc}"
    return rc


def mind_base_mult(scalar: int):
    sb, xb, yb = buf(i2b32(scalar)), out(32), out(32)
    rc = E.p256_scalar_base_mult(addr(sb), addr(xb), addr(yb))
    if rc == 1:
        return None  # point at infinity
    assert rc == 0, f"p256_scalar_base_mult rc={rc}"
    return int.from_bytes(xb.raw[:32], "big"), int.from_bytes(yb.raw[:32], "big")


def pyca_verdict(pub, r: int, s: int, msg: bytes) -> bool:
    try:
        pub.verify(encode_dss_signature(r, s), msg, ec.ECDSA(hashes.SHA256()))
        return True
    except InvalidSignature:
        return False


# ---------------------------------------------------------------------------
# p256_scalar_base_mult — group law KAT vs pyca (d*G == public_numbers).
# ---------------------------------------------------------------------------
print("=" * 72)
print("p256_scalar_base_mult — d*G vs pyca ec.derive_private_key public point")
print("=" * 72)

# RFC 6979 A.2.5 private key (public constants; also anchors the KAT below).
D_6979 = 0xC9AFA9D845BA75166B5C215767B1D6934E50C3DB36E89B127B8A622B120F6721
pn = ec.derive_private_key(D_6979, ec.SECP256R1()).public_key().public_numbers()
print(f"        pyca: d*G = ({i2b32(pn.x).hex()}, {i2b32(pn.y).hex()})")
got = mind_base_mult(D_6979)
record_bool("base_mult: RFC 6979 A.2.5 key d*G matches pyca",
            got == (pn.x, pn.y),
            f"got=({got[0]:064x},{got[1]:064x})" if got else "got=infinity")

# A fresh random key (pyca computes the public point FIRST).
rand_key = ec.generate_private_key(ec.SECP256R1())
d_rand = rand_key.private_numbers().private_value
pn2 = rand_key.public_key().public_numbers()
print(f"        pyca: rand d*G = ({i2b32(pn2.x).hex()[:16]}.., {i2b32(pn2.y).hex()[:16]}..)")
got = mind_base_mult(d_rand)
record_bool("base_mult: fresh random key d*G matches pyca",
            got == (pn2.x, pn2.y))

# ---------------------------------------------------------------------------
# Ground truth: pyca signs with a fresh SECP256R1 key + ECDSA(SHA256).
# ---------------------------------------------------------------------------
print("=" * 72)
print("pyca ground truth — SECP256R1 sign(msg, ECDSA(SHA256))")
print("=" * 72)

key = ec.generate_private_key(ec.SECP256R1())
pub = key.public_key()
qn = pub.public_numbers()
msg = b"MIND provable stack: TLS 1.3 CertificateVerify uses ecdsa_secp256r1_sha256"
sig_der = key.sign(msg, ec.ECDSA(hashes.SHA256()))
sig_r, sig_s = decode_dss_signature(sig_der)
print(f"        Qx = {i2b32(qn.x).hex()}")
print(f"        Qy = {i2b32(qn.y).hex()}")
print(f"        msg = {msg!r}")
print(f"        r = {i2b32(sig_r).hex()}")
print(f"        s = {i2b32(sig_s).hex()}")

# pyca verifies its OWN signature (independent accept ground truth).
assert pyca_verdict(pub, sig_r, sig_s, msg), \
    "pyca failed to verify its own ECDSA signature (setup bug)"
print("        pyca verify(valid sig): ACCEPT")

# ---------------------------------------------------------------------------
# (a) MIND accepts the pyca-valid signature.
# ---------------------------------------------------------------------------
print("=" * 72)
print("ecdsa_p256_verify — accept + reject paths")
print("=" * 72)
rc = mind_verify(qn.x, qn.y, sig_r, sig_s, msg)
record_bool("accept: pyca-valid signature accepted (rc==1)", rc == 1,
            f"rc={rc} pyca=ACCEPT")

# ---------------------------------------------------------------------------
# (b) Bit-flipped r and bit-flipped s must be rejected (pyca first).
# ---------------------------------------------------------------------------
bad_r = sig_r ^ 1
assert 1 <= bad_r < N and not pyca_verdict(pub, bad_r, sig_s, msg), \
    "pyca accepted a bit-flipped r (setup bug)"
rc = mind_verify(qn.x, qn.y, bad_r, sig_s, msg)
record_bool("reject: bit-flipped r rejected (rc==0)", rc == 0,
            f"rc={rc} pyca=REJECT")

bad_s = sig_s ^ (1 << 100)
assert 1 <= bad_s < N and not pyca_verdict(pub, sig_r, bad_s, msg), \
    "pyca accepted a bit-flipped s (setup bug)"
rc = mind_verify(qn.x, qn.y, sig_r, bad_s, msg)
record_bool("reject: bit-flipped s rejected (rc==0)", rc == 0,
            f"rc={rc} pyca=REJECT")

# ---------------------------------------------------------------------------
# (c) Signature over a DIFFERENT message must be rejected.
# ---------------------------------------------------------------------------
other_msg = b"a completely different message the key never signed"
assert not pyca_verdict(pub, sig_r, sig_s, other_msg)
rc = mind_verify(qn.x, qn.y, sig_r, sig_s, other_msg)
record_bool("reject: wrong message rejected (rc==0)", rc == 0,
            f"rc={rc} pyca=REJECT")

# ---------------------------------------------------------------------------
# (d) Verification against a WRONG public key must be rejected.
# ---------------------------------------------------------------------------
other_pub = ec.generate_private_key(ec.SECP256R1()).public_key()
on = other_pub.public_numbers()
assert not pyca_verdict(other_pub, sig_r, sig_s, msg)
rc = mind_verify(on.x, on.y, sig_r, sig_s, msg)
record_bool("reject: wrong public key rejected (rc==0)", rc == 0,
            f"rc={rc} pyca=REJECT")

# ---------------------------------------------------------------------------
# (e) Out-of-range r/s: reject r=0, s=0, r=n, s=n (SEC1 §4.1.4 step 1 —
#     r, s must be in [1, n-1]; structural, no pyca encoding possible for 0).
# ---------------------------------------------------------------------------
print("=" * 72)
print("fail-closed extras — r/s out of [1, n-1], off-curve public key")
print("=" * 72)
for name, r_v, s_v in [("r=0", 0, sig_s), ("s=0", sig_r, 0),
                       ("r=n", N, sig_s), ("s=n", sig_r, N)]:
    rc = mind_verify(qn.x, qn.y, r_v, s_v, msg)
    record_bool(f"reject: {name} (out of [1, n-1]) rejected (rc==0)", rc == 0,
                f"rc={rc}")

# Public key not on the curve (Qy+1): partial public-key validation
# (SEC1 §3.2.2.1) must fail closed.
rc = mind_verify(qn.x, qn.y + 1, sig_r, sig_s, msg)
record_bool("reject: off-curve public key (Qy+1) rejected (rc==0)", rc == 0,
            f"rc={rc}")

# ---------------------------------------------------------------------------
# (f) RFC 6979 A.2.5 known-answer vector — P-256, SHA-256, message "sample"
#     (deterministic ECDSA, so (r, s) are fixed constants).  Cross-checked
#     against pyca BEFORE MIND sees it.
# ---------------------------------------------------------------------------
print("=" * 72)
print('RFC 6979 A.2.5 KAT — P-256 / SHA-256, message "sample"')
print("=" * 72)
KAT_QX = 0x60FED4BA255A9D31C961EB74C6356D68C049B8923B61FA6CE669622E60F29FB6
KAT_QY = 0x7903FE1008B8BC99A41AE9E95628BC64F2F1B20C2D7E9F5177A3C294D4462299
KAT_R = 0xEFD48B2AACB6A8FD1140DD9CD45E81D69D2C877B56AAF991C34D0EA84EAF3716
KAT_S = 0xF7CB1C942D657C41D436C7A1B6E29F65F3E900DBB9AFF4064DC4AB2F843ACDA8
KAT_MSG = b"sample"

kat_pub = ec.EllipticCurvePublicNumbers(KAT_QX, KAT_QY,
                                        ec.SECP256R1()).public_key()
assert pyca_verdict(kat_pub, KAT_R, KAT_S, KAT_MSG), \
    "pyca rejected the RFC 6979 A.2.5 vector (transcription bug)"
print("        pyca verify(RFC 6979 A.2.5 vector): ACCEPT")
rc = mind_verify(KAT_QX, KAT_QY, KAT_R, KAT_S, KAT_MSG)
record_bool("accept: RFC 6979 A.2.5 P-256/SHA-256 KAT accepted (rc==1)",
            rc == 1, f"rc={rc} pyca=ACCEPT")

print("=" * 72)
total = len(results)
passed = sum(1 for r in results if r)
print(f"SUMMARY: {passed}/{total} checks PASSED")
print("=" * 72)
sys.exit(0 if passed == total else 1)
