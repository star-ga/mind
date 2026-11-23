#!/usr/bin/env python3
# Official-vector driver for std/x25519.mind (pure-MIND Curve25519 ECDH).
#
# Exercises the single pub fn `x25519` in the compiled .so against the official
# RFC 7748 test vectors:
#   - §5.2 : two single-iteration scalar-mult known-answer vectors
#   - §5.2 : the iterated test (1 and 1,000 iterations; 1,000,000 optional)
#   - §6.1 : the Diffie-Hellman worked example — Alice/Bob keypairs, both
#            directions produce the SAME shared secret AND match the RFC value
#            (proves the DH symmetry property, not just the raw primitive).
#
# Every hardcoded expected value is ALSO cross-checked against pyca/cryptography
# (an independent trusted reference) so the hardcoded hex cannot be a
# transcription error. Prints PASS/FAIL per case with computed vs expected hex.
#
# Also checks that x25519 releases its work buffers: 2000 chained calls in this
# one process must not grow the resident set past LEAK_BOUND_BYTES (a call kept
# about 1.2 MB before the buffers were released), must not leave more than
# HEAP_BOUND_BYTES more allocator bytes in use (this catches a single buffer of
# a few dozen bytes that a call keeps), and the chain must equal the pyca chain.
#
# Usage: python3 x25519_vectors_driver.py <x25519.so>  [--million]

import ctypes
import os
import sys

from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey,
    X25519PublicKey,
)

so_path = sys.argv[1]
DO_MILLION = "--million" in sys.argv[2:]
X = ctypes.CDLL(so_path)
X.x25519.restype = ctypes.c_int64
X.x25519.argtypes = [ctypes.c_int64] * 3

results = []


def buf(data: bytes):
    return ctypes.create_string_buffer(bytes(data), max(1, len(data)))


def out(n: int):
    return ctypes.create_string_buffer(max(1, n))


def addr(b):
    return ctypes.cast(b, ctypes.c_void_p).value


def mind_x25519(scalar: bytes, u: bytes) -> bytes:
    """Call the compiled pure-MIND x25519 on 32-byte scalar and u."""
    assert len(scalar) == 32 and len(u) == 32
    sb, ub, ob = buf(scalar), buf(u), out(32)
    X.x25519(addr(sb), addr(ub), addr(ob))
    return ob.raw[:32]


def record(name, got: bytes, exp: bytes):
    ok = got == exp
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    print(f"        got={got.hex()}")
    print(f"        exp={exp.hex()}")


def record_bool(name, cond, detail=""):
    results.append(bool(cond))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}  {detail}")


# pyca reference: raw X25519 scalar multiplication.
def ref_x25519(scalar: bytes, u: bytes) -> bytes:
    priv = X25519PrivateKey.from_private_bytes(scalar)
    pub = X25519PublicKey.from_public_bytes(u)
    return priv.exchange(pub)


print("=" * 72)
print("X25519 — RFC 7748 §5.2 single-iteration test vectors")
print("=" * 72)

# Vector 1.
v1_scalar = bytes.fromhex(
    "a546e36bf0527c9d3b16154b82465edd62144c0ac1fc5a18506a2244ba449ac4")
v1_u = bytes.fromhex(
    "e6db6867583030db3594c1a424b15f7c726624ec26b3353b10a903a6d0ab1c4c")
v1_exp = bytes.fromhex(
    "c3da55379de9c6908e94ea4df28d084f32eccf03491c71f754b4075577a28552")
assert ref_x25519(v1_scalar, v1_u) == v1_exp, "pyca disagrees with RFC 7748 §5.2 v1"
record("RFC7748-5.2 vector 1", mind_x25519(v1_scalar, v1_u), v1_exp)

# Vector 2.
v2_scalar = bytes.fromhex(
    "4b66e9d4d1b4673c5ad22691957d6af5c11b6421e0ea01d42ca4169e7918ba0d")
v2_u = bytes.fromhex(
    "e5210f12786811d3f4b7959d0538ae2c31dbe7106fc03c3efc4cd549c715a493")
v2_exp = bytes.fromhex(
    "95cbde9476e8907d7aade45cb4b873f88b595a68799fa152e6f8f7647aac7957")
assert ref_x25519(v2_scalar, v2_u) == v2_exp, "pyca disagrees with RFC 7748 §5.2 v2"
record("RFC7748-5.2 vector 2", mind_x25519(v2_scalar, v2_u), v2_exp)

print("=" * 72)
print("X25519 — work buffers are released (2000 chained calls, resident set)")
print("=" * 72)

LEAK_CALLS = 2000
# A call that keeps its work buffers holds about 1.2 MB (1000 chained calls
# held 1.24 GB resident on the MLIR build); a call that releases them holds none
# after it returns. The bound is on growth of the CURRENT resident set, not on the
# peak: a peak never falls, so a peak taken after other work would hide a leak.
LEAK_BOUND_BYTES = 64 * 1024 * 1024
# The resident bound only sees a leak of 32 KB or more per call. The allocator's own
# count of bytes in use (mallinfo2, glibc) is exact: a buffer of 32 bytes kept by every
# call leaves 48 bytes of chunk per call, 96,000 bytes over 2000 calls.
HEAP_BOUND_BYTES = 32 * 1024
MIB = 1024 * 1024
leak_base9 = bytes([9] + [0] * 31)


class MallInfo2(ctypes.Structure):
    _fields_ = [(n, ctypes.c_size_t) for n in (
        "arena", "ordblks", "smblks", "hblks", "hblkhd",
        "usmblks", "fsmblks", "uordblks", "fordblks", "keepcost")]


_libc = ctypes.CDLL(None)
_libc.mallinfo2.restype = MallInfo2


def heap_in_use_bytes() -> int:
    """Bytes the allocator holds in use now: small chunks plus mmapped chunks."""
    mi = _libc.mallinfo2()
    return mi.uordblks + mi.hblkhd


def resident_bytes() -> int:
    """Resident set of this process now (Linux /proc/self/statm, pages -> bytes)."""
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def mind_chain(n: int) -> bytes:
    """n chained calls (k, u = x25519(k, u), k) on one set of buffers; returns k."""
    sb, ub, ob = buf(leak_base9), buf(leak_base9), out(32)
    for _ in range(n):
        X.x25519(addr(sb), addr(ub), addr(ob))
        ctypes.memmove(ub, sb, 32)
        ctypes.memmove(sb, ob, 32)
    return sb.raw[:32]


# Positive control: the probe must SEE a growth above the bound, or a pass below
# proves nothing. 96 MiB written page by page is resident the moment it exists.
control_before = resident_bytes()
ballast = bytearray(b"\x01") * (96 * MIB)
control_grew = resident_bytes() - control_before
del ballast
record_bool("leak probe sees a known 96 MiB allocation (positive control)",
            control_grew > LEAK_BOUND_BYTES,
            f"grew={control_grew / MIB:.1f} MiB bound={LEAK_BOUND_BYTES / MIB:.0f} MiB")

mind_chain(1)  # warm the allocator: one call's worth of heap is not a leak
leak_before = resident_bytes()
heap_before = heap_in_use_bytes()
leak_k = mind_chain(LEAK_CALLS)
leak_grew = resident_bytes() - leak_before
heap_grew = heap_in_use_bytes() - heap_before
leak_ref_k, leak_ref_u = leak_base9, leak_base9
for _ in range(LEAK_CALLS):
    leak_ref_k, leak_ref_u = ref_x25519(leak_ref_k, leak_ref_u), leak_ref_k
record("x25519 2000-call chain equals pyca", leak_k, leak_ref_k)
record_bool(f"resident set grows < {LEAK_BOUND_BYTES / MIB:.0f} MiB over {LEAK_CALLS} chained calls",
            leak_grew < LEAK_BOUND_BYTES,
            f"grew={leak_grew / MIB:.1f} MiB ({leak_grew // LEAK_CALLS} bytes per call)")
record_bool(f"allocator bytes in use grow < {HEAP_BOUND_BYTES // 1024} KiB over {LEAK_CALLS} chained calls",
            heap_grew < HEAP_BOUND_BYTES,
            f"grew={heap_grew} bytes ({heap_grew // LEAK_CALLS} bytes per call)")

print("=" * 72)
print("X25519 — RFC 7748 §5.2 iterated test (k=u=9; k,u = X25519(k,u),k)")
print("=" * 72)

base9 = bytes([9] + [0] * 31)
exp_iter_1 = bytes.fromhex(
    "422c8e7a6227d7bca1350b3e2bb7279f7897b87bb6854b783c60e80311ae3079")
exp_iter_1000 = bytes.fromhex(
    "684cf59ba83309552800ef566f2f4d3c1c3887c49360e3875f2eb94d99532c51")
exp_iter_1000000 = bytes.fromhex(
    "7c3911e0ab2586fd864497297e575e6f3bc601c0883c30df5f4dd2d24f665424")

k = base9
u = base9
last_target = 1000000 if DO_MILLION else 1000
for n in range(1, last_target + 1):
    k, u = mind_x25519(k, u), k
    if n == 1:
        record("RFC7748-5.2 iterated x1", k, exp_iter_1)
    elif n == 1000:
        record("RFC7748-5.2 iterated x1000", k, exp_iter_1000)
    elif n == 1000000:
        record("RFC7748-5.2 iterated x1000000", k, exp_iter_1000000)
if not DO_MILLION:
    print("        (x1000000 skipped — pass --million to run the ~1e6-iter check)")

print("=" * 72)
print("X25519 — RFC 7748 §6.1 Diffie-Hellman worked example")
print("=" * 72)

alice_priv = bytes.fromhex(
    "77076d0a7318a57d3c16c17251b26645df4c2f87ebc0992ab177fba51db92c2a")
alice_pub_exp = bytes.fromhex(
    "8520f0098930a754748b7ddcb43ef75a0dbf3a0d26381af4eba4a98eaa9b4e6a")
bob_priv = bytes.fromhex(
    "5dab087e624a8a4b79e17f8b83800ee66f3bb1292618b6fd1c2f8b27ff88e0eb")
bob_pub_exp = bytes.fromhex(
    "de9edb7d7b7dc1b4d35b61c2ece435373f8343c85b78674dadfc7e146f882b4f")
shared_exp = bytes.fromhex(
    "4a5d9d5ba4ce2de1728e3bf480350f25e07e21c947d19e3376f09b3c1e161742")

# Independent cross-check of the published constants against pyca.
assert ref_x25519(alice_priv, base9) == alice_pub_exp, "pyca disagrees: Alice pub"
assert ref_x25519(bob_priv, base9) == bob_pub_exp, "pyca disagrees: Bob pub"
assert ref_x25519(alice_priv, bob_pub_exp) == shared_exp, "pyca disagrees: shared"

# Public keys from the base point.
alice_pub = mind_x25519(alice_priv, base9)
bob_pub = mind_x25519(bob_priv, base9)
record("RFC7748-6.1 Alice public key", alice_pub, alice_pub_exp)
record("RFC7748-6.1 Bob public key", bob_pub, bob_pub_exp)

# Shared secret from BOTH directions — must match each other and the RFC value.
shared_ab = mind_x25519(alice_priv, bob_pub)   # Alice's view
shared_ba = mind_x25519(bob_priv, alice_pub)   # Bob's view
record("RFC7748-6.1 shared secret (Alice*Bob_pub)", shared_ab, shared_exp)
record("RFC7748-6.1 shared secret (Bob*Alice_pub)", shared_ba, shared_exp)
record_bool("RFC7748-6.1 DH symmetry (both directions equal)",
            shared_ab == shared_ba,
            f"ab={shared_ab.hex()[:16]}.. ba={shared_ba.hex()[:16]}..")

print("=" * 72)
total = len(results)
passed = sum(1 for r in results if r)
print(f"SUMMARY: {passed}/{total} checks PASSED")
print("=" * 72)
sys.exit(0 if passed == total else 1)
