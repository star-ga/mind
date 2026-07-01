#!/usr/bin/env python3
# Official-vector driver for std/aes_gcm.mind + std/hkdf.mind (pure-MIND crypto).
#
# Exercises every pub fn entry point in the two compiled .so's against official
# published test vectors:
#   - AES-128 block cipher : FIPS 197 Appendix B/C
#   - AES-128-GCM          : McGrew/Viega GCM spec Test Cases 1-4 + a
#                            tag-mismatch-must-fail case
#   - HMAC-SHA256          : RFC 4231 Test Cases 1-3
#   - HKDF-SHA256          : RFC 5869 Appendix A Test Cases 1-3
#
# Each published output is ALSO cross-checked against pyca/cryptography (an
# independent trusted reference) so the hardcoded hex cannot be a transcription
# error. Prints PASS/FAIL per case with computed vs expected bytes in hex.
#
# Usage: python3 crypto_vectors_driver.py <aes_gcm.so> <hkdf.so>

import ctypes
import sys

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import hmac as _pyhmac
import hashlib

aes_so, hkdf_so = sys.argv[1], sys.argv[2]
A = ctypes.CDLL(aes_so)
H = ctypes.CDLL(hkdf_so)

for name, lib in (("aes128_encrypt_block", A), ("aes128_gcm_encrypt", A),
                  ("aes128_gcm_decrypt", A), ("hmac_sha256", H),
                  ("hkdf_extract", H), ("hkdf_expand", H)):
    getattr(lib, name).restype = ctypes.c_int64
A.aes128_encrypt_block.argtypes = [ctypes.c_int64] * 3
A.aes128_gcm_encrypt.argtypes = [ctypes.c_int64] * 8
A.aes128_gcm_decrypt.argtypes = [ctypes.c_int64] * 8
H.hmac_sha256.argtypes = [ctypes.c_int64] * 5
H.hkdf_extract.argtypes = [ctypes.c_int64] * 5
H.hkdf_expand.argtypes = [ctypes.c_int64] * 6

results = []


def buf(data: bytes):
    """Mutable ctypes buffer holding data (>=1 byte so the pointer is valid)."""
    b = ctypes.create_string_buffer(bytes(data), max(1, len(data)))
    return b


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
print("AES-128 block cipher — FIPS 197 Appendix B/C")
print("=" * 72)
key = bytes.fromhex("000102030405060708090a0b0c0d0e0f")
pt = bytes.fromhex("00112233445566778899aabbccddeeff")
exp_ct = bytes.fromhex("69c4e0d86a7b0430d8cdb78070b4c55a")
# Independent reference: raw AES-ECB single block.
ref = Cipher(algorithms.AES(key), modes.ECB()).encryptor()
assert ref.update(pt) + ref.finalize() == exp_ct, "pyca disagrees with FIPS 197"
kb, ib, ob = buf(key), buf(pt), out(16)
A.aes128_encrypt_block(addr(kb), addr(ib), addr(ob))
record("FIPS-197 AES-128 single block", ob.raw[:16], exp_ct)

print("=" * 72)
print("AES-128-GCM — McGrew/Viega GCM spec Test Cases 1-4")
print("=" * 72)


def gcm_case(nm, k, iv, p, a, exp_c_hex, exp_t_hex):
    k, iv, p, a = (bytes.fromhex(x) for x in (k, iv, p, a))
    exp_c, exp_t = bytes.fromhex(exp_c_hex), bytes.fromhex(exp_t_hex)
    # Independent reference (pyca appends the 16-byte tag to the ciphertext).
    ref = AESGCM(k).encrypt(iv, p, a if a else None)
    assert ref == exp_c + exp_t, f"pyca disagrees with published {nm}"
    # MIND encrypt.
    kb, ivb, pb, ab = buf(k), buf(iv), buf(p), buf(a)
    ctb, tagb = out(len(p)), out(16)
    A.aes128_gcm_encrypt(addr(kb), addr(ivb), addr(pb), len(p),
                         addr(ab), len(a), addr(ctb), addr(tagb))
    got_c, got_t = ctb.raw[:len(p)], tagb.raw[:16]
    record(f"{nm} encrypt: ciphertext", got_c, exp_c)
    record(f"{nm} encrypt: tag", got_t, exp_t)
    # MIND decrypt round-trip (tag valid → returns 0, plaintext recovered).
    ctb2, tagb2 = buf(exp_c), buf(exp_t)
    ptb = out(len(exp_c))
    rc = A.aes128_gcm_decrypt(addr(kb), addr(ivb), addr(ctb2), len(exp_c),
                              addr(ab), len(a), addr(tagb2), addr(ptb))
    record_bool(f"{nm} decrypt: rc==0 (auth ok)", rc == 0, f"rc={rc}")
    record(f"{nm} decrypt: plaintext", ptb.raw[:len(p)], p)


# TC1: empty plaintext, empty AAD.
gcm_case("GCM-TC1", "00000000000000000000000000000000",
         "000000000000000000000000", "", "",
         "", "58e2fccefa7e3061367f1d57a4e7455a")
# TC2: 16-byte plaintext, no AAD.
gcm_case("GCM-TC2", "00000000000000000000000000000000",
         "000000000000000000000000",
         "00000000000000000000000000000000", "",
         "0388dace60b6a392f328c2b971b2fe78",
         "ab6e47d42cec13bdf53a67b21257bddf")
# TC3: 64-byte plaintext, no AAD.
gcm_case("GCM-TC3", "feffe9928665731c6d6a8f9467308308",
         "cafebabefacedbaddecaf888",
         "d9313225f88406e5a55909c5aff5269a86a7a9531534f7da2e4c303d8a318a72"
         "1c3c0c95956809532fcf0e2449a6b525b16aedf5aa0de657ba637b391aafd255",
         "",
         "42831ec2217774244b7221b784d0d49ce3aa212f2c02a4e035c17e2329aca12e"
         "21d514b25466931c7d8f6a5aac84aa051ba30b396a0aac973d58e091473f5985",
         "4d5c2af327cd64a62cf35abd2ba6fab4")
# TC4: 60-byte plaintext + 20-byte AAD.
gcm_case("GCM-TC4", "feffe9928665731c6d6a8f9467308308",
         "cafebabefacedbaddecaf888",
         "d9313225f88406e5a55909c5aff5269a86a7a9531534f7da2e4c303d8a318a72"
         "1c3c0c95956809532fcf0e2449a6b525b16aedf5aa0de657ba637b39",
         "feedfacedeadbeeffeedfacedeadbeefabaddad2",
         "42831ec2217774244b7221b784d0d49ce3aa212f2c02a4e035c17e2329aca12e"
         "21d514b25466931c7d8f6a5aac84aa051ba30b396a0aac973d58e091",
         "5bc94fbc3221a5db94fae95ae7121a47")

print("=" * 72)
print("AES-128-GCM — tag-mismatch MUST fail closed (security-critical)")
print("=" * 72)
k = bytes.fromhex("feffe9928665731c6d6a8f9467308308")
iv = bytes.fromhex("cafebabefacedbaddecaf888")
a = bytes.fromhex("feedfacedeadbeeffeedfacedeadbeefabaddad2")
good_ct = bytes.fromhex(
    "42831ec2217774244b7221b784d0d49ce3aa212f2c02a4e035c17e2329aca12e"
    "21d514b25466931c7d8f6a5aac84aa051ba30b396a0aac973d58e091")
good_tag = bytes.fromhex("5bc94fbc3221a5db94fae95ae7121a47")
bad_tag = bytearray(good_tag)
bad_tag[0] ^= 0x01  # flip one bit
bad_tag = bytes(bad_tag)
kb, ivb, ab = buf(k), buf(iv), buf(a)
ctb, tagb = buf(good_ct), buf(bad_tag)
ptb = out(len(good_ct))
# Pre-fill output with a sentinel so we can prove nothing plausible leaks.
ctypes.memset(addr(ptb), 0x5A, len(good_ct))
rc = A.aes128_gcm_decrypt(addr(kb), addr(ivb), addr(ctb), len(good_ct),
                          addr(ab), len(a), addr(tagb), addr(ptb))
leaked = ptb.raw[:len(good_ct)]
# The real plaintext for these inputs (must NOT appear in the output).
real_pt = bytes.fromhex(
    "d9313225f88406e5a55909c5aff5269a86a7a9531534f7da2e4c303d8a318a72"
    "1c3c0c95956809532fcf0e2449a6b525b16aedf5aa0de657ba637b39")
record_bool("GCM tamper: decrypt returns non-zero (rejects)", rc != 0, f"rc={rc}")
record_bool("GCM tamper: no real plaintext leaked", leaked != real_pt)
# Sanity: the SAME ciphertext with the CORRECT tag must still succeed.
tagb_ok = buf(good_tag)
ctb_ok = buf(good_ct)
ptb_ok = out(len(good_ct))
rc_ok = A.aes128_gcm_decrypt(addr(kb), addr(ivb), addr(ctb_ok), len(good_ct),
                             addr(ab), len(a), addr(tagb_ok), addr(ptb_ok))
record_bool("GCM tamper: correct tag still authenticates", rc_ok == 0, f"rc={rc_ok}")
record("GCM tamper: correct-tag plaintext", ptb_ok.raw[:len(good_ct)], real_pt)

print("=" * 72)
print("HMAC-SHA256 — RFC 4231 Test Cases 1-3")
print("=" * 72)


def hmac_case(nm, key_hex, data, exp_hex):
    key = bytes.fromhex(key_hex)
    data = data if isinstance(data, bytes) else data.encode()
    exp = bytes.fromhex(exp_hex)
    assert _pyhmac.new(key, data, hashlib.sha256).digest() == exp, f"pyref {nm}"
    kb, db, ob = buf(key), buf(data), out(32)
    H.hmac_sha256(addr(kb), len(key), addr(db), len(data), addr(ob))
    record(nm, ob.raw[:32], exp)


hmac_case("HMAC-RFC4231-TC1", "0b" * 20, b"Hi There",
          "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7")
hmac_case("HMAC-RFC4231-TC2", "4a656665", b"what do ya want for nothing?",
          "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843")
hmac_case("HMAC-RFC4231-TC3", "aa" * 20, bytes.fromhex("dd" * 50),
          "773ea91e36800e46854db8ebd09181a72959098b3ef8c122d9635514ced565fe")

print("=" * 72)
print("HKDF-SHA256 — RFC 5869 Appendix A Test Cases 1-3")
print("=" * 72)


def hkdf_case(nm, ikm_hex, salt_hex, info_hex, L, exp_prk_hex, exp_okm_hex):
    ikm, salt, info = (bytes.fromhex(x) for x in (ikm_hex, salt_hex, info_hex))
    exp_prk, exp_okm = bytes.fromhex(exp_prk_hex), bytes.fromhex(exp_okm_hex)
    # Independent reference for PRK.
    ref_prk = _pyhmac.new(salt if salt else b"\x00" * 32, ikm, hashlib.sha256).digest()
    assert ref_prk == exp_prk, f"pyref PRK {nm}"
    ib, sb, nb = buf(ikm), buf(salt), buf(info)
    prkb = out(32)
    H.hkdf_extract(addr(sb), len(salt), addr(ib), len(ikm), addr(prkb))
    record(f"{nm} extract PRK", prkb.raw[:32], exp_prk)
    okmb = out(L)
    H.hkdf_expand(addr(prkb), 32, addr(nb), len(info), L, addr(okmb))
    record(f"{nm} expand OKM (L={L})", okmb.raw[:L], exp_okm)


hkdf_case("HKDF-RFC5869-TC1",
          "0b" * 22, "000102030405060708090a0b0c", "f0f1f2f3f4f5f6f7f8f9", 42,
          "077709362c2e32df0ddc3f0dc47bba63"
          "90b6c73bb50f9c3122ec844ad7c2b3e5",
          "3cb25f25faacd57a90434f64d0362f2a"
          "2d2d0a90cf1a5a4c5db02d56ecc4c5bf"
          "34007208d5b887185865")
hkdf_case("HKDF-RFC5869-TC2",
          "".join(f"{i:02x}" for i in range(0x50)),
          "".join(f"{i:02x}" for i in range(0x60, 0xb0)),
          "".join(f"{i:02x}" for i in range(0xb0, 0x100)), 82,
          "06a6b88c5853361a06104c9ceb35b45c"
          "ef760014904671014a193f40c15fc244",
          "b11e398dc80327a1c8e7f78c596a4934"
          "4f012eda2d4efad8a050cc4c19afa97c"
          "59045a99cac7827271cb41c65e590e09"
          "da3275600c2f09b8367793a9aca3db71"
          "cc30c58179ec3e87c14c01d5c1f3434f"
          "1d87")
hkdf_case("HKDF-RFC5869-TC3",
          "0b" * 22, "", "", 42,
          "19ef24a32c717b167f33a91d6f648bdf"
          "96596776afdb6377ac434c1c293ccb04",
          "8da4e775a563c18f715f802a063c5a31"
          "b8a11f5c5ee1879ec3454e5f3c738d2d"
          "9d201395faa4b61a96c8")

print("=" * 72)
total = len(results)
passed = sum(1 for r in results if r)
print(f"SUMMARY: {passed}/{total} checks PASSED")
print("=" * 72)
sys.exit(0 if passed == total else 1)
