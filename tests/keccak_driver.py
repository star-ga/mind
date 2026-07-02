#!/usr/bin/env python3
# Official-vector driver for std/keccak.mind (pure-MIND FIPS 202).
#
# Exercises every pub fn entry point in the compiled .so against the
# well-known NIST FIPS 202 known-answer vectors:
#   - keccak_rotl64  : rotation KATs (the sign-extension footgun sentinel)
#   - SHA3-256       : '', 'abc', 200 bytes of 0xa3 (multi-block, > rate 136)
#   - SHA3-512       : '', 200 bytes of 0xa3 (multi-block, > rate 72)
#   - SHAKE128       : ('' , 32), (200*0xa3, 32), and a 500-byte XOF output
#                      (> rate 168, exercises multi-squeeze)
#   - SHAKE256       : ('' , 32), (200*0xa3, 32), and a 500-byte XOF output
#
# Each hardcoded published digest is ALSO cross-checked against Python
# hashlib (an independent trusted reference) BEFORE the MIND output is
# compared, so a transcription error in the hex cannot produce a fake PASS.
# Prints PASS/FAIL per case with computed vs expected bytes in hex.
#
# Usage: python3 keccak_driver.py <keccak.so>

import ctypes
import hashlib
import sys

keccak_so = sys.argv[1]
K = ctypes.CDLL(keccak_so)

for name, nargs in (("keccak_rotl64", 2), ("keccak_sha3_256", 3),
                    ("keccak_sha3_512", 3), ("keccak_shake128", 4),
                    ("keccak_shake256", 4)):
    fn = getattr(K, name)
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


def record(name, got: bytes, exp: bytes):
    ok = got == exp
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    print(f"        got={got.hex()}")
    print(f"        exp={exp.hex()}")


def record_bool(name, cond, detail=""):
    results.append(bool(cond))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}  {detail}")


print("=" * 72)
print("keccak_rotl64 — 64-bit rotate-left KATs (arithmetic-shift sentinel)")
print("=" * 72)
M64 = (1 << 64) - 1


def rotl_case(x, n, exp):
    got = K.keccak_rotl64(ctypes.c_int64(x - (1 << 64) if x >> 63 else x), n) & M64
    record_bool(f"rotl64(0x{x:016x}, {n}) == 0x{exp:016x}", got == exp,
                f"got=0x{got:016x}")


rotl_case(1, 1, 2)
rotl_case(0x8000000000000000, 1, 1)          # sign bit must NOT smear
rotl_case(0x8000000000000001, 1, 3)
rotl_case(0xFFFFFFFFFFFFFFFF, 17, 0xFFFFFFFFFFFFFFFF)
rotl_case(0x0123456789ABCDEF, 0, 0x0123456789ABCDEF)
rotl_case(0x0123456789ABCDEF, 63, ((0x0123456789ABCDEF << 63) |
                                   (0x0123456789ABCDEF >> 1)) & M64)

MULTI = b"\xa3" * 200  # > every rate (72/136/168) → exercises multi-absorb

print("=" * 72)
print("SHA3-256 — FIPS 202 known-answer vectors (rate 136, ds 0x06)")
print("=" * 72)


def sha3_256_case(nm, msg, exp_hex):
    exp = bytes.fromhex(exp_hex)
    # Independent reference FIRST (non-circular).
    assert hashlib.sha3_256(msg).digest() == exp, f"hashlib disagrees: {nm}"
    ib, ob = buf(msg), out(32)
    K.keccak_sha3_256(addr(ib), len(msg), addr(ob))
    record(nm, ob.raw[:32], exp)


sha3_256_case("SHA3-256('')", b"",
              "a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a")
sha3_256_case("SHA3-256('abc')", b"abc",
              "3a985da74fe225b2045c172d6bd390bd855f086e3e9d525b46bfe24511431532")
sha3_256_case("SHA3-256(200*0xa3) multi-block", MULTI,
              "79f38adec5c20307a98ef76e8324afbfd46cfd81b22e3973c65fa1bd9de31787")

print("=" * 72)
print("SHA3-512 — FIPS 202 known-answer vectors (rate 72, ds 0x06)")
print("=" * 72)


def sha3_512_case(nm, msg, exp_hex):
    exp = bytes.fromhex(exp_hex)
    assert hashlib.sha3_512(msg).digest() == exp, f"hashlib disagrees: {nm}"
    ib, ob = buf(msg), out(64)
    K.keccak_sha3_512(addr(ib), len(msg), addr(ob))
    record(nm, ob.raw[:64], exp)


sha3_512_case("SHA3-512('')", b"",
              "a69f73cca23a9ac5c8b567dc185a756e97c982164fe25859e0d1dcc1475c80a6"
              "15b2123af1f5f94c11e3e9402c3ac558f500199d95b6d3e301758586281dcd26")
sha3_512_case("SHA3-512(200*0xa3) multi-block", MULTI,
              "e76dfad22084a8b1467fcf2ffa58361bec7628edf5f3fdc0e4805dc48caeeca8"
              "1b7c13c30adf52a3659584739a2df46be589c51ca1a4a8416df6545a1ce8ba00")

print("=" * 72)
print("SHAKE128 — FIPS 202 XOF (rate 168, ds 0x1F)")
print("=" * 72)


def shake_case(nm, mind_fn, ref_ctor, msg, outlen, exp_hex=None):
    ref = ref_ctor(msg).digest(outlen)
    if exp_hex is not None:
        assert ref == bytes.fromhex(exp_hex), f"hashlib disagrees: {nm}"
    ib, ob = buf(msg), out(outlen)
    mind_fn(addr(ib), len(msg), addr(ob), outlen)
    record(nm, ob.raw[:outlen], ref)


shake_case("SHAKE128('', 32)", K.keccak_shake128, hashlib.shake_128, b"", 32,
           "7f9c2ba4e88f827d616045507605853ed73b8093f6efbc88eb1a6eacfa66ef26")
shake_case("SHAKE128(200*0xa3, 32) multi-block", K.keccak_shake128,
           hashlib.shake_128, MULTI, 32,
           "131ab8d2b594946b9c81333f9bb6e0ce75c3b93104fa3469d3917457385da037")
shake_case("SHAKE128(200*0xa3, 500) multi-squeeze XOF", K.keccak_shake128,
           hashlib.shake_128, MULTI, 500)

print("=" * 72)
print("SHAKE256 — FIPS 202 XOF (rate 136, ds 0x1F)")
print("=" * 72)
shake_case("SHAKE256('', 32)", K.keccak_shake256, hashlib.shake_256, b"", 32,
           "46b9dd2b0ba88d13233b3feb743eeb243fcd52ea62b81b82b50c27646ed5762f")
shake_case("SHAKE256(200*0xa3, 32) multi-block", K.keccak_shake256,
           hashlib.shake_256, MULTI, 32,
           "cd8a920ed141aa0407a22d59288652e9d9f1a7ee0c1e7c1ca699424da84a904d")
shake_case("SHAKE256(200*0xa3, 500) multi-squeeze XOF", K.keccak_shake256,
           hashlib.shake_256, MULTI, 500)

print("=" * 72)
total = len(results)
passed = sum(1 for r in results if r)
print(f"SUMMARY: {passed}/{total} checks PASSED")
print("=" * 72)
sys.exit(0 if passed == total else 1)
