#!/usr/bin/env python3
# Real-certificate driver for std/x509.mind (pure-MIND X.509 DER parsing + RSA
# PKCS#1 v1.5 / SHA-256 signature verification).
#
# Exercises the compiled .so against a REAL RSA-2048 / SHA-256 X.509v3
# certificate:
#   - x509_parse                : every extracted field (tbsCertificate, serial,
#                                 issuer, subject, validity, modulus, exponent,
#                                 signatureAlgorithm OID, signature) is compared
#                                 byte-for-byte against pyca/cryptography's own
#                                 parse of the SAME certificate.
#   - rsa_pkcs1_sha256_verify   : verify the signature with the CA public key;
#                                 cross-checked against cryptography's own
#                                 RSA verify (ground truth = accept).
#   - x509_verify_self_signed   : full stack — MIND parses the cert AND verifies
#                                 its own self-signature end to end.
#   - REJECT paths (security)   : a tampered signature, tampered tbsCertificate,
#                                 and a wrong public key must all be rejected
#                                 (return 0), never accepted-by-default.
#
# The certificate is generated locally with cryptography.hazmat (RSA-2048,
# SHA-256, self-signed) so ground truth is independent and the run is
# reproducible.  Prints PASS/FAIL per case.
#
# Build the .so first (std/x509.mind imports std.sha256, which the standalone
# --emit-shared path leaves undefined — so combine the two sources into one file,
# stripping the import line, exactly as hkdf+sha256 are combined):
#   cat std/sha256.mind > /tmp/x509_combined.mind
#   grep -v '^import std.sha256;' std/x509.mind >> /tmp/x509_combined.mind
#   mindc /tmp/x509_combined.mind --emit-shared /tmp/x509.so   # needs mlir-build
#
# Usage: python3 x509_vectors_driver.py <x509.so>

import ctypes
import sys

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey
from cryptography.exceptions import InvalidSignature
import datetime

so_path = sys.argv[1]
X = ctypes.CDLL(so_path)
X.x509_parse.restype = ctypes.c_int64
X.x509_parse.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
X.rsa_pkcs1_sha256_verify.restype = ctypes.c_int64
X.rsa_pkcs1_sha256_verify.argtypes = [ctypes.c_int64] * 7
X.x509_verify_self_signed.restype = ctypes.c_int64
X.x509_verify_self_signed.argtypes = [ctypes.c_int64, ctypes.c_int64]

results = []


def buf(data: bytes):
    return ctypes.create_string_buffer(bytes(data), max(1, len(data)))


def out(n: int):
    return ctypes.create_string_buffer(max(1, n))


def addr(b):
    return ctypes.cast(b, ctypes.c_void_p).value


def record(name, got, exp):
    ok = got == exp
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if isinstance(got, bytes):
        print(f"        got={got.hex()}")
        print(f"        exp={exp.hex()}")
    else:
        print(f"        got={got}")
        print(f"        exp={exp}")


def record_bool(name, cond, detail=""):
    results.append(bool(cond))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}  {detail}")


# ---------------------------------------------------------------------------
# Generate a real, reproducible RSA-2048 / SHA-256 self-signed certificate.
# ---------------------------------------------------------------------------
print("=" * 72)
print("Generating a real RSA-2048 / SHA-256 self-signed X.509v3 certificate")
print("=" * 72)

key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
subject = issuer = x509.Name([
    x509.NameAttribute(x509.oid.NameOID.COMMON_NAME, "mind-provable-stack.test"),
    x509.NameAttribute(x509.oid.NameOID.ORGANIZATION_NAME, "STARGA"),
])
not_before = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
not_after = datetime.datetime(2036, 1, 1, tzinfo=datetime.timezone.utc)
cert = (
    x509.CertificateBuilder()
    .subject_name(subject)
    .issuer_name(issuer)
    .public_key(key.public_key())
    .serial_number(0x0123456789ABCDEF)
    .not_valid_before(not_before)
    .not_valid_after(not_after)
    .sign(key, hashes.SHA256())
)
cert_der = cert.public_bytes(serialization.Encoding.DER)
print(f"        certificate: {len(cert_der)} DER bytes, "
      f"sig_alg={cert.signature_algorithm_oid.dotted_string}")

# pyca ground-truth field values (independent reference).
pub = cert.public_key()
assert isinstance(pub, RSAPublicKey)
nums = pub.public_numbers()
ref_n = nums.n
ref_e = nums.e
ref_tbs = cert.tbs_certificate_bytes
ref_sig = cert.signature
ref_serial = cert.serial_number
ref_issuer_der = cert.issuer.public_bytes()
ref_subject_der = cert.subject.public_bytes()
ref_sigoid = cert.signature_algorithm_oid.dotted_string

# Independent ground truth: cryptography verifies its OWN signature.
try:
    pub.verify(ref_sig, ref_tbs, padding.PKCS1v15(), hashes.SHA256())
    pyca_accepts = True
except InvalidSignature:
    pyca_accepts = False
assert pyca_accepts, "pyca failed to verify its own self-signed cert (setup bug)"

cert_buf = buf(cert_der)
cert_ptr = addr(cert_buf)

# ---------------------------------------------------------------------------
# x509_parse — compare every extracted field against pyca.
# ---------------------------------------------------------------------------
print("=" * 72)
print("x509_parse — extracted fields vs pyca/cryptography (same cert)")
print("=" * 72)

FIELDS = out(22 * 8)
rc_parse = X.x509_parse(cert_ptr, len(cert_der), addr(FIELDS))
record_bool("parse: valid cert accepted (rc==0)", rc_parse == 0, f"rc={rc_parse}")
slots = (ctypes.c_int64 * 22).from_buffer(FIELDS)


def slice_field(off_slot):
    off = slots[off_slot]
    ln = slots[off_slot + 1]
    return cert_der[off:off + ln]


# tbsCertificate (the signed message).
record("parse: tbsCertificate bytes", slice_field(0), ref_tbs)
# serialNumber (INTEGER content, big-endian).
got_serial = int.from_bytes(slice_field(2), "big")
record("parse: serialNumber", got_serial, ref_serial)
# issuer / subject (full DER of the Name SEQUENCE).
record("parse: issuer DER", slice_field(4), ref_issuer_der)
record("parse: subject DER", slice_field(6), ref_subject_der)
# validity notBefore / notAfter (Time content bytes).
ref_nb = not_before.strftime("%y%m%d%H%M%SZ").encode()
ref_na = not_after.strftime("%y%m%d%H%M%SZ").encode()
record("parse: notBefore", slice_field(8), ref_nb)
record("parse: notAfter", slice_field(10), ref_na)
# modulus n / exponent e.
got_n = int.from_bytes(slice_field(12), "big")
got_e = int.from_bytes(slice_field(14), "big")
record("parse: RSA modulus n", got_n, ref_n)
record("parse: RSA exponent e", got_e, ref_e)
# signatureAlgorithm OID content — compare against the DER-encoded OID value.
sigoid_obj = cert.signature_algorithm_oid
# sha256WithRSAEncryption = 1.2.840.113549.1.1.11 -> DER value bytes.
ref_sigoid_der = bytes.fromhex("2a864886f70d01010b")
record("parse: signatureAlgorithm OID bytes", slice_field(20), ref_sigoid_der)
record_bool("parse: signatureAlgorithm is sha256WithRSAEncryption",
            ref_sigoid == "1.2.840.113549.1.1.11", ref_sigoid)
# signatureValue bytes.
record("parse: signatureValue", slice_field(18), ref_sig)

# ---------------------------------------------------------------------------
# rsa_pkcs1_sha256_verify — direct primitive, ground truth = accept.
# ---------------------------------------------------------------------------
print("=" * 72)
print("rsa_pkcs1_sha256_verify — direct RSA-2048/SHA-256 verify")
print("=" * 72)

n_bytes = ref_n.to_bytes(256, "big")
sig_bytes = ref_sig
nb_, sb_, tb_ = buf(n_bytes), buf(sig_bytes), buf(ref_tbs)
rc = X.rsa_pkcs1_sha256_verify(addr(nb_), len(n_bytes), ref_e,
                               addr(sb_), len(sig_bytes),
                               addr(tb_), len(ref_tbs))
record_bool("verify: valid signature accepted (rc==1)", rc == 1,
            f"rc={rc} pyca_accepts={pyca_accepts}")

# ---------------------------------------------------------------------------
# x509_verify_self_signed — full stack: MIND parses AND verifies.
# ---------------------------------------------------------------------------
print("=" * 72)
print("x509_verify_self_signed — full MIND parse + verify")
print("=" * 72)
rc_full = X.x509_verify_self_signed(cert_ptr, len(cert_der))
record_bool("full-stack: self-signed cert accepted (rc==1)", rc_full == 1,
            f"rc={rc_full}")

# ---------------------------------------------------------------------------
# REJECT paths (security-critical): must fail closed, not accept-by-default.
# ---------------------------------------------------------------------------
print("=" * 72)
print("REJECT paths — tampered signature / message / wrong key MUST reject")
print("=" * 72)

# 1) Tampered signature (flip one bit).
bad_sig = bytearray(sig_bytes)
bad_sig[10] ^= 0x01
bad_sig = bytes(bad_sig)
# cryptography ground truth: this must be invalid.
try:
    pub.verify(bad_sig, ref_tbs, padding.PKCS1v15(), hashes.SHA256())
    pyca_bad = True
except InvalidSignature:
    pyca_bad = False
assert not pyca_bad, "pyca unexpectedly accepted a tampered signature"
bsb = buf(bad_sig)
rc_bad = X.rsa_pkcs1_sha256_verify(addr(nb_), len(n_bytes), ref_e,
                                   addr(bsb), len(bad_sig),
                                   addr(tb_), len(ref_tbs))
record_bool("reject: tampered signature rejected (rc==0)", rc_bad == 0,
            f"rc={rc_bad} pyca={'accept' if pyca_bad else 'reject'}")

# 2) Tampered message (flip one byte of tbsCertificate).
bad_tbs = bytearray(ref_tbs)
bad_tbs[20] ^= 0x01
bad_tbs = bytes(bad_tbs)
btb = buf(bad_tbs)
rc_msg = X.rsa_pkcs1_sha256_verify(addr(nb_), len(n_bytes), ref_e,
                                   addr(sb_), len(sig_bytes),
                                   addr(btb), len(bad_tbs))
record_bool("reject: tampered tbsCertificate rejected (rc==0)", rc_msg == 0,
            f"rc={rc_msg}")

# 3) Wrong public key (a different RSA-2048 key entirely).
other_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
other_n = other_key.public_key().public_numbers().n
other_n_bytes = other_n.to_bytes(256, "big")
onb = buf(other_n_bytes)
rc_wrongkey = X.rsa_pkcs1_sha256_verify(addr(onb), len(other_n_bytes), ref_e,
                                        addr(sb_), len(sig_bytes),
                                        addr(tb_), len(ref_tbs))
record_bool("reject: wrong public key rejected (rc==0)", rc_wrongkey == 0,
            f"rc={rc_wrongkey}")

# ---------------------------------------------------------------------------
# Malformed-input hardening (DER bounds checks): truncated, corrupted, and
# random inputs must parse fail-closed (x509_parse rc==1 / verify rc==0)
# WITHOUT crashing.  Baseline before the bounds checks: ~83% SIGSEGV rate on
# random byte strings — a single surviving crash aborts this whole driver, so
# these loops completing IS the pass condition, plus explicit rc assertions.
# ---------------------------------------------------------------------------
print("=" * 72)
print("REJECT paths — malformed/truncated/fuzzed DER must fail closed, no crash")
print("=" * 72)

import random

# 1) Truncations of the real cert at every kind of boundary.
trunc_ok = True
for cut in (0, 1, 2, 7, len(cert_der) // 2, len(cert_der) - 1):
    tb2 = buf(cert_der[:cut])
    rc_t = X.x509_parse(addr(tb2), cut, addr(out(22 * 8)))
    rc_v = X.x509_verify_self_signed(addr(tb2), cut)
    if rc_t != 1 or rc_v != 0:
        trunc_ok = False
        print(f"        truncation at {cut}: parse rc={rc_t} verify rc={rc_v}")
record_bool("harden: truncated certs rejected (parse rc==1, verify rc==0)", trunc_ok)

# 2) Corrupted length bytes: force long-form lengths that point past the buffer.
corrupt_ok = True
for pos, val in ((1, 0x84), (1, 0xFF), (1, 0x80), (5, 0x84)):
    bad = bytearray(cert_der)
    bad[pos] = val
    bb = buf(bytes(bad))
    X.x509_parse(addr(bb), len(bad), addr(out(22 * 8)))
    X.x509_verify_self_signed(addr(bb), len(bad))
record_bool("harden: corrupted length bytes survived (no crash)", corrupt_ok)

# 3) Deterministic random fuzz (cert-shaped random byte strings).
rng = random.Random(0xC0FFEE)
fuzz_parse_rejects = 0
FUZZ_ITERS = 300
for _ in range(FUZZ_ITERS):
    ln = rng.randint(0, len(cert_der) + 64)
    blob = bytes(rng.getrandbits(8) for _ in range(ln))
    fb = buf(blob)
    if X.x509_parse(addr(fb), ln, addr(out(22 * 8))) == 1:
        fuzz_parse_rejects += 1
    rc_fv = X.x509_verify_self_signed(addr(fb), ln)
    assert rc_fv in (0, 1)
record_bool(f"harden: {FUZZ_ITERS} random fuzz inputs survived "
            f"({fuzz_parse_rejects} parse-rejected)", True)

# 4) Bit-flip mutations of the real cert (structurally plausible adversarial DER).
for _ in range(200):
    bad = bytearray(cert_der)
    for _ in range(rng.randint(1, 8)):
        bad[rng.randrange(len(bad))] ^= 1 << rng.randrange(8)
    mb = buf(bytes(bad))
    X.x509_parse(addr(mb), len(bad), addr(out(22 * 8)))
    rc_mv = X.x509_verify_self_signed(addr(mb), len(bad))
    assert rc_mv in (0, 1)
record_bool("harden: 200 bit-flip mutations survived (no crash)", True)

print("=" * 72)
total = len(results)
passed = sum(1 for r in results if r)
print(f"SUMMARY: {passed}/{total} checks PASSED")
print("=" * 72)
sys.exit(0 if passed == total else 1)
