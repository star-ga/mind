#!/usr/bin/env python3
# Ground-truth driver for std/rsa_pss.mind (pure-MIND RSASSA-PSS-VERIFY,
# RFC 8017 §8.1.2 / §9.1.2 with SHA-256 + MGF1 — the rsa_pss_rsae_sha256
# scheme TLS 1.3 uses for CertificateVerify, RFC 8446 §4.4.3).
#
# Exercises the compiled .so against pyca/cryptography as the independent
# ground truth (every expected value is produced by pyca / stdlib hashlib
# BEFORE the MIND result is computed — non-circular):
#   - mgf1_sha256          : compared byte-for-byte against an independent
#                            pure-hashlib MGF1 (RFC 8017 §B.2.1) for several
#                            seed/length cases.
#   - rsa_pss_verify_sha256: a fresh RSA-2048 key signs a message with
#                            padding.PSS(MGF1(SHA256), salt_length=32) +
#                            SHA256; MIND must ACCEPT that signature, and
#                            must REJECT a bit-flipped signature, a signature
#                            checked against a different message, and a
#                            signature checked against a wrong public key —
#                            each rejection cross-checked against pyca's own
#                            verdict on the same tampered input.
#   - fail-closed extras   : wrong salt_len, sig >= n, bad lengths.
#
# Build the .so first (std/rsa_pss.mind composes std/sha256.mind +
# std/x509.mind's bignum, so combine the three sources with import lines
# stripped, exactly as tests/x509_vectors_driver.py combines sha256+x509):
#   cat std/sha256.mind > /tmp/rsa_pss_combined.mind
#   grep -v '^import std.sha256;' std/x509.mind >> /tmp/rsa_pss_combined.mind
#   grep -v '^import '            std/rsa_pss.mind >> /tmp/rsa_pss_combined.mind
#   mindc /tmp/rsa_pss_combined.mind --emit-shared /tmp/rsa_pss.so
#
# Usage: python3 rsa_pss_driver.py <rsa_pss.so>

import ctypes
import hashlib
import sys

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.exceptions import InvalidSignature

so_path = sys.argv[1]
P = ctypes.CDLL(so_path)
P.mgf1_sha256.restype = ctypes.c_int64
P.mgf1_sha256.argtypes = [ctypes.c_int64] * 4
P.rsa_pss_verify_sha256.restype = ctypes.c_int64
P.rsa_pss_verify_sha256.argtypes = [ctypes.c_int64] * 8

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
# Independent MGF1 reference (RFC 8017 §B.2.1) — pure stdlib hashlib, no pyca:
#   T = "" ; for counter = 0 .. ceil(maskLen/hLen)-1:
#       T = T || Hash(seed || I2OSP(counter, 4))
#   return leftmost maskLen octets of T.
# ---------------------------------------------------------------------------
def mgf1_ref(seed: bytes, mask_len: int) -> bytes:
    t = b""
    counter = 0
    while len(t) < mask_len:
        t += hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
        counter += 1
    return t[:mask_len]


def mind_mgf1(seed: bytes, mask_len: int) -> bytes:
    sb = buf(seed)
    mb = out(mask_len)
    rc = P.mgf1_sha256(addr(sb), len(seed), addr(mb), mask_len)
    assert rc == 0, f"mgf1_sha256 rc={rc}"
    return mb.raw[:mask_len]


print("=" * 72)
print("mgf1_sha256 — vs independent pure-hashlib MGF1 (RFC 8017 §B.2.1)")
print("=" * 72)

MGF1_CASES = [
    (b"", 32),                       # empty seed, one digest exactly
    (b"foo", 60),                    # partial second block
    (bytes(range(32)), 223),         # the RSA-2048 dbMask geometry (7.0 blocks)
    (b"\x00" * 8, 1),                # single output byte
    (b"mind-rsa-pss-seed", 100),     # multi-block, non-multiple of 32
]
for seed, mlen in MGF1_CASES:
    exp = mgf1_ref(seed, mlen)       # reference computed FIRST
    print(f"        ref  MGF1(seed={seed.hex() or '(empty)'}, {mlen}) = {exp.hex()}")
    got = mind_mgf1(seed, mlen)
    record(f"mgf1: seed_len={len(seed)} mask_len={mlen}", got, exp)

# ---------------------------------------------------------------------------
# Ground truth: pyca signs with RSA-2048 / PSS(MGF1(SHA256), salt=32) / SHA256
# — the exact rsa_pss_rsae_sha256 parameters (RFC 8446 §4.2.3: salt = hLen).
# ---------------------------------------------------------------------------
print("=" * 72)
print("pyca ground truth — RSA-2048 PSS(MGF1(SHA256), salt_length=32) + SHA256")
print("=" * 72)

key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
pub = key.public_key()
nums = pub.public_numbers()
ref_n, ref_e = nums.n, nums.e
assert ref_n.bit_length() == 2048 and ref_e == 65537

PSS32 = padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=32)
msg = b"MIND provable stack: TLS 1.3 CertificateVerify uses rsa_pss_rsae_sha256"
sig = key.sign(msg, PSS32, hashes.SHA256())
print(f"        n (2048-bit) = {ref_n.to_bytes(256, 'big')[:16].hex()}...")
print(f"        e = {ref_e}")
print(f"        msg = {msg!r}")
print(f"        sig = {sig.hex()}")

# pyca verifies its OWN signature (independent accept ground truth).
try:
    pub.verify(sig, msg, PSS32, hashes.SHA256())
    pyca_valid = True
except InvalidSignature:
    pyca_valid = False
assert pyca_valid, "pyca failed to verify its own PSS signature (setup bug)"
print("        pyca verify(valid sig): ACCEPT")

n_bytes = ref_n.to_bytes(256, "big")
nb = buf(n_bytes)
sb = buf(sig)
mb = buf(msg)


def mind_verify(n_b, sig_b, msg_b, salt_len=32, e=None):
    rc = P.rsa_pss_verify_sha256(addr(n_b), 256, e if e is not None else ref_e,
                                 addr(sig_b), len(sig_b),
                                 addr(msg_b), len(msg_b), salt_len)
    assert rc in (0, 1), f"non-boolean rc={rc}"
    return rc


# ---------------------------------------------------------------------------
# (a) MIND accepts the pyca-valid signature.
# ---------------------------------------------------------------------------
print("=" * 72)
print("rsa_pss_verify_sha256 — accept + reject paths")
print("=" * 72)
rc = mind_verify(nb, sb, mb)
record_bool("accept: pyca-valid PSS signature accepted (rc==1)", rc == 1,
            f"rc={rc} pyca=ACCEPT")

# ---------------------------------------------------------------------------
# (b) Bit-flipped signature must be rejected (pyca cross-checked first).
# ---------------------------------------------------------------------------
for pos in (0, 100, 255):
    bad_sig = bytearray(sig)
    bad_sig[pos] ^= 0x01
    bad_sig = bytes(bad_sig)
    try:
        pub.verify(bad_sig, msg, PSS32, hashes.SHA256())
        pyca_bad = True
    except InvalidSignature:
        pyca_bad = False
    assert not pyca_bad, "pyca accepted a bit-flipped signature (setup bug)"
    rc = mind_verify(nb, buf(bad_sig), mb)
    record_bool(f"reject: bit-flipped signature (byte {pos}) rejected (rc==0)",
                rc == 0, f"rc={rc} pyca=REJECT")

# ---------------------------------------------------------------------------
# (c) Signature over a DIFFERENT message must be rejected.
# ---------------------------------------------------------------------------
other_msg = b"a completely different message the key never signed"
try:
    pub.verify(sig, other_msg, PSS32, hashes.SHA256())
    pyca_bad = True
except InvalidSignature:
    pyca_bad = False
assert not pyca_bad
rc = mind_verify(nb, sb, buf(other_msg))
record_bool("reject: wrong message rejected (rc==0)", rc == 0,
            f"rc={rc} pyca=REJECT")

# ---------------------------------------------------------------------------
# (d) Verification against a WRONG public key must be rejected.
# ---------------------------------------------------------------------------
other_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
other_pub = other_key.public_key()
other_n = other_pub.public_numbers().n
try:
    other_pub.verify(sig, msg, PSS32, hashes.SHA256())
    pyca_bad = True
except InvalidSignature:
    pyca_bad = False
assert not pyca_bad
onb = buf(other_n.to_bytes(256, "big"))
rc = mind_verify(onb, sb, mb)
record_bool("reject: wrong public key rejected (rc==0)", rc == 0,
            f"rc={rc} pyca=REJECT")

# ---------------------------------------------------------------------------
# Fail-closed extras (structural EMSA-PSS / RSAVP1 checks, RFC 8017).
# ---------------------------------------------------------------------------
print("=" * 72)
print("fail-closed extras — salt-length mismatch, s >= n, bad geometry")
print("=" * 72)

# Wrong expected salt length (sig used salt=32; verifying with 20 must fail —
# pyca agrees when told to expect exactly 20).
PSS20 = padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=20)
try:
    pub.verify(sig, msg, PSS20, hashes.SHA256())
    pyca_bad = True
except InvalidSignature:
    pyca_bad = False
assert not pyca_bad
rc = mind_verify(nb, sb, mb, salt_len=20)
record_bool("reject: salt_len mismatch (expect 20, sig has 32) (rc==0)",
            rc == 0, f"rc={rc} pyca=REJECT")

# Signature representative out of range: s = n (RSAVP1 §5.2.2 requires s < n).
rc = mind_verify(nb, buf(n_bytes), mb)
record_bool("reject: s == n (representative out of range) (rc==0)", rc == 0,
            f"rc={rc}")

# s = n - 1 is in range but decrypts to garbage — must still reject cleanly.
rc = mind_verify(nb, buf((ref_n - 1).to_bytes(256, "big")), mb)
record_bool("reject: s == n-1 (garbage EM) (rc==0)", rc == 0, f"rc={rc}")

# Bad signature length (255 bytes) must be rejected up front (§8.1.2 step 1).
rc = P.rsa_pss_verify_sha256(addr(nb), 256, ref_e, addr(sb), 255,
                             addr(mb), len(msg), 32)
record_bool("reject: sig_len != 256 rejected (rc==0)", rc == 0, f"rc={rc}")

# Bad salt_len bounds (negative / > 222) must be rejected up front (§9.1.2 step 3).
rc0 = mind_verify(nb, sb, mb, salt_len=-1)
rc1 = mind_verify(nb, sb, mb, salt_len=223)
record_bool("reject: salt_len out of bounds (-1, 223) rejected (rc==0)",
            rc0 == 0 and rc1 == 0, f"rc={rc0},{rc1}")

print("=" * 72)
total = len(results)
passed = sum(1 for r in results if r)
print(f"SUMMARY: {passed}/{total} checks PASSED")
print("=" * 72)
sys.exit(0 if passed == total else 1)
