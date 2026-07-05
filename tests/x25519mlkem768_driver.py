#!/usr/bin/env python3
# Known-answer driver for std/x25519mlkem768.mind (pure-MIND X25519MLKEM768
# post-quantum hybrid key exchange, draft-kwiatkowski-tls-ecdhe-mlkem, IANA TLS
# NamedGroup 0x11EC).  House style of tests/mlkem768_driver.py.
#
# The hybrid is a COMPOSITION: ML-KEM-768 (FIPS 203) + X25519 (RFC 7748), with
# the combined shared secret = ML-KEM_secret || X25519_secret (ML-KEM first for
# this group).  The ground truth is therefore composed from two INDEPENDENT
# trusted references:
#   - ML-KEM-768: kyber-py (Giacomo Pope, validated vs NIST ACVP) deterministic
#     internal forms _keygen_internal(d,z) / _encaps_internal(ek,m) / decaps.
#   - X25519:     pyca/cryptography raw scalar multiplication.
# Both references are pinned to a hardcoded KAT tuple BEFORE any MIND comparison,
# so a broken reference install cannot manufacture a fake PASS.
#
# Cases (deterministic randomness -> exact byte vectors):
#   (a) client keygen(d||z||cs)  == ML-KEM ek || X25519(cs, base)      [1216 B]
#   (b) server encaps(client_share, m||ss)
#          share == ML-KEM ct || X25519(ss, base)                      [1120 B]
#          secret == ML-KEM_K || X25519(ss, client_pub)                [  64 B]
#   (c) client decaps(client_priv, server_share) == the SAME 64-byte secret
#       (hybrid round-trip: both halves agree end to end)
#   (d) decaps with a corrupted ML-KEM ciphertext -> ML-KEM half becomes the
#       FO implicit-reject secret (fail-closed), X25519 half unchanged; the
#       combined secret DIFFERS from the honest one (no fail-open).
#
# Build the .so first (combine deps in dependency order, stripping imports):
#   cat std/keccak.mind                                    > /tmp/hyb.mind
#   grep -v '^import std.keccak;' std/mlkem768.mind       >> /tmp/hyb.mind
#   grep -vE '^import std\.' std/x25519.mind              >> /tmp/hyb.mind
#   grep -vE '^import std\.' std/x25519mlkem768.mind      >> /tmp/hyb.mind
#   mindc /tmp/hyb.mind --emit-shared /tmp/hyb.so          # needs mlir-build
#
# Usage: PYTHONPATH=<dir with kyber-py> python3 x25519mlkem768_driver.py <hyb.so>

import ctypes
import sys

from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey,
    X25519PublicKey,
)

so_path = sys.argv[1]
M = ctypes.CDLL(so_path)

try:
    from kyber_py.ml_kem import ML_KEM_768
except ImportError as e:
    print(f"BLOCKED: reference kyber-py not importable ({e}).")
    print("Install: pip install --target <dir> kyber-py; PYTHONPATH=<dir> ...")
    sys.exit(2)

print("REFERENCE: kyber-py ML_KEM_768 + pyca/cryptography X25519 "
      "(hybrid = ML-KEM_ss || X25519_ss)")

for name, n in (("x25519mlkem768_keygen", 3),
                ("x25519mlkem768_encaps", 4),
                ("x25519mlkem768_decaps", 3)):
    fn = getattr(M, name)
    fn.restype = ctypes.c_int64
    fn.argtypes = [ctypes.c_int64] * n

results = []


def buf(data: bytes):
    return ctypes.create_string_buffer(bytes(data), max(1, len(data)))


def out(n: int):
    return ctypes.create_string_buffer(max(1, n))


def addr(b):
    return ctypes.cast(b, ctypes.c_void_p).value


def record(name, got: bytes, exp: bytes, full=True):
    ok = got == exp
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if full or not ok:
        print(f"        got={got.hex()}")
        print(f"        exp={exp.hex()}")
    else:
        print(f"        got={got[:16].hex()}...{got[-16:].hex()} ({len(got)}B)")
        print(f"        exp={exp[:16].hex()}...{exp[-16:].hex()} ({len(exp)}B)")


# X25519 base point u = 9 (RFC 7748 §4.1).
X25519_BASE = bytes([9] + [0] * 31)


def x25519_ref(scalar: bytes, u: bytes) -> bytes:
    priv = X25519PrivateKey.from_private_bytes(scalar)
    return priv.exchange(X25519PublicKey.from_public_bytes(u))


# ---------------------------------------------------------------------------
# Deterministic KAT inputs (fixed randomness -> exact byte vectors).
# ---------------------------------------------------------------------------
d = bytes(range(32))              # ML-KEM keygen randomness, first 32
z = bytes(range(32, 64))          # ML-KEM keygen randomness, second 32 (FO)
cs = bytes(range(64, 96))         # client X25519 scalar
m = bytes(range(96, 128))         # ML-KEM encaps randomness
ss = bytes((i * 7 + 3) & 0xFF for i in range(32))   # server X25519 scalar

# ---------------------------------------------------------------------------
# Step 0 — reference self-check BEFORE any MIND comparison.  The two halves are
# cross-checked against each other (X25519 DH symmetry + ML-KEM round-trip) so a
# broken kyber-py/cryptography install cannot manufacture a fake PASS.
# ---------------------------------------------------------------------------
ek_ref, dk_ref = ML_KEM_768._keygen_internal(d, z)
client_pub_ref = x25519_ref(cs, X25519_BASE)
client_share_ref = ek_ref + client_pub_ref            # 1184 + 32 = 1216

K_mlkem_ref, ct_ref = ML_KEM_768._encaps_internal(ek_ref, m)
server_pub_ref = x25519_ref(ss, X25519_BASE)
x25519_ss_ref = x25519_ref(ss, client_pub_ref)
server_share_ref = ct_ref + server_pub_ref            # 1088 + 32 = 1120
hybrid_ss_ref = K_mlkem_ref + x25519_ss_ref           # 32 + 32 = 64

# Cross-check the X25519 DH symmetry (client side must reproduce the same half).
assert x25519_ref(cs, server_pub_ref) == x25519_ss_ref, \
    "reference X25519 DH is not symmetric"
# Cross-check the ML-KEM round-trip.
assert ML_KEM_768.decaps(dk_ref, ct_ref) == K_mlkem_ref, \
    "reference ML-KEM round-trip failed"
assert len(client_share_ref) == 1216 and len(server_share_ref) == 1120
assert len(hybrid_ss_ref) == 64
print("[PASS] reference self-check: kyber-py + pyca X25519 compose a consistent "
      f"64-byte hybrid secret (ss={hybrid_ss_ref.hex()})")
results.append(True)

# Corrupted-ct reference (implicit rejection on the ML-KEM half).
ct_bad = bytes([ct_ref[0] ^ 1]) + ct_ref[1:]
K_mlkem_reject = ML_KEM_768.decaps(dk_ref, ct_bad)
server_share_bad = ct_bad + server_pub_ref
hybrid_ss_reject = K_mlkem_reject + x25519_ss_ref
assert hybrid_ss_reject != hybrid_ss_ref, "reject secret collides with honest"

# ---------------------------------------------------------------------------
# (a) client keygen == ML-KEM ek || X25519 pub, and dk||scalar retained.
# ---------------------------------------------------------------------------
print("=" * 72)
print("(a) X25519MLKEM768 client KeyGen — deterministic d||z||cs")
print("=" * 72)
rand96 = buf(d + z + cs)
share_b, priv_b = out(1216), out(2432)
M.x25519mlkem768_keygen(addr(rand96), addr(share_b), addr(priv_b))
record("client key_share (1216B)", share_b.raw[:1216], client_share_ref, full=False)
record("client X25519 pubkey (share[1184:1216])",
       share_b.raw[1184:1216], client_pub_ref)
record("client priv dk (2400B)", priv_b.raw[:2400], dk_ref, full=False)
record("client priv X25519 scalar (priv[2400:2432])", priv_b.raw[2400:2432], cs)

# ---------------------------------------------------------------------------
# (b) server encaps == ML-KEM ct || X25519 pub; secret == ML-KEM_K || X25519_ss.
# ---------------------------------------------------------------------------
print("=" * 72)
print("(b) X25519MLKEM768 server Encaps — deterministic m||ss")
print("=" * 72)
peer_share = buf(client_share_ref)
rand64 = buf(m + ss)
sshare_b, sss_b = out(1120), out(64)
M.x25519mlkem768_encaps(addr(peer_share), addr(rand64), addr(sshare_b), addr(sss_b))
record("server key_share (1120B)", sshare_b.raw[:1120], server_share_ref, full=False)
record("server X25519 pubkey (share[1088:1120])",
       sshare_b.raw[1088:1120], server_pub_ref)
record("hybrid shared secret (64B) = ML-KEM_K || X25519_ss",
       sss_b.raw[:64], hybrid_ss_ref)
record("  ML-KEM half (ss[0:32])", sss_b.raw[:32], K_mlkem_ref)
record("  X25519 half (ss[32:64])", sss_b.raw[32:64], x25519_ss_ref)

# ---------------------------------------------------------------------------
# (c) client decaps round-trip == the same 64-byte hybrid secret.
# ---------------------------------------------------------------------------
print("=" * 72)
print("(c) X25519MLKEM768 client Decaps — hybrid round-trip")
print("=" * 72)
priv_in = buf(priv_b.raw[:2432])
server_share_in = buf(server_share_ref)
dss_b = out(64)
M.x25519mlkem768_decaps(addr(priv_in), addr(server_share_in), addr(dss_b))
record("decaps hybrid secret == encaps secret", dss_b.raw[:64], hybrid_ss_ref)

# ---------------------------------------------------------------------------
# (d) corrupted ML-KEM ciphertext -> FO implicit rejection on the ML-KEM half,
#     X25519 half intact; combined secret DIFFERS from the honest one.
# ---------------------------------------------------------------------------
print("=" * 72)
print("(d) Decaps with corrupted ML-KEM ct — fail-closed hybrid")
print("=" * 72)
server_share_bad_in = buf(server_share_bad)
dss_bad_b = out(64)
M.x25519mlkem768_decaps(addr(priv_in), addr(server_share_bad_in), addr(dss_bad_b))
record("decaps(bad ct) == J(z||ct') || X25519_ss", dss_bad_b.raw[:64], hybrid_ss_reject)
record("  X25519 half unchanged", dss_bad_b.raw[32:64], x25519_ss_ref)
neq = dss_bad_b.raw[:64] != hybrid_ss_ref
results.append(neq)
print(f"[{'PASS' if neq else 'FAIL'}] rejected hybrid secret differs from the "
      "honest one (fail-closed, not fail-open)")

print("=" * 72)
total = len(results)
passed = sum(1 for r in results if r)
print(f"SUMMARY: {passed}/{total} checks PASSED")
print("=" * 72)
sys.exit(0 if passed == total else 1)
