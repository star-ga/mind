# Copyright 2025 STARGA Inc.
# Licensed under the Apache License, Version 2.0.
# Part of the MIND project (Machine Intelligence Native Design).

"""x509 blocker-case drivers (b1-b6) and the path / malformed fixture
generators. They are modes of tests/x509_vectors_driver.py and run through it:

    python3 tests/x509_vectors_driver.py <b1..b6|gen-path|gen-malformed> [args...]

Kept in this sibling module so the driver stays under the 800-line
module_size_ratchet ceiling.
"""

import ctypes
import sys

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey
from cryptography.exceptions import InvalidSignature
import datetime

_X509_DIR = __import__("pathlib").Path(__file__).resolve().parent / "x509"


def _x509_mode_b1(argv):
    """Test for Blocker B1: RSA verification must accept valid 3072- and 4096-bit keys (and reject invalid ones)."""
    import ctypes
    import pathlib
    import struct
    import sys

    SO = argv[0]
    FX = _X509_DIR / "testdata" / "adv"
    TD = _X509_DIR / "testdata"
    NOW = int((TD / "now.txt").read_text().strip())

    X = ctypes.CDLL(SO)
    X.x509_verify_server_chain.restype = ctypes.c_int64
    X.x509_verify_server_chain.argtypes = [ctypes.c_int64] * 4

    keep = []

    def der(path):
        return path.read_bytes()

    def blob_raw(ders):
        n = len(ders)
        hdr = [n]
        off = 8 + 16 * n
        for d in ders:
            hdr += [off, len(d)]
            off += len(d)
        buf = ctypes.create_string_buffer(off + 8)
        ctypes.memmove(buf, struct.pack("<%dq" % len(hdr), *hdr), len(hdr) * 8)
        base = ctypes.cast(buf, ctypes.c_void_p).value
        p = 8 + 16 * n
        for d in ders:
            ctypes.memmove(base + p, d, len(d))
            p += len(d)
        keep.append(buf)
        return base

    def cstr(s):
        b = ctypes.create_string_buffer(s.encode("utf-8") + b"\0")
        keep.append(b)
        return ctypes.cast(b, ctypes.c_void_p).value

    def verify(chain_ders, anchor_ders, host, now=NOW):
        return X.x509_verify_server_chain(blob_raw(chain_ders), blob_raw(anchor_ders), cstr(host), now)

    cases = [
        ("R01 RSA-4096 root -> RSA-2048 inter -> EC leaf",
         [der(FX / "rsa4096_leaf.der"), der(FX / "rsa4096_inter2048.der")],
         [der(FX / "rsa4096_root.der")],
         "adv.example.com", 0),
        ("R03 RSA-3072 root -> EC leaf",
         [der(FX / "rsa3072_leaf.der")],
         [der(FX / "rsa3072_root.der")],
         "adv.example.com", 0),
        ("R06 RSA-PSS 4096 root -> PSS 2048 inter -> leaf",
         [der(FX / "pss4096_leaf.der"), der(FX / "pss4096_inter2048.der")],
         [der(FX / "pss4096_root.der")],
         "adv.example.com", 0),
    ]

    all_ok = True
    for name, chain, anchors, host, want in cases:
        got = verify(chain, anchors, host)
        ok = (got == want)
        if not ok:
            all_ok = False
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: got {got}, want {want}")

    # Also test rejection of invalid RSA sizes (e.g. 1024-bit, 500-byte)
    X.rsa_pkcs1_sha256_verify.restype = ctypes.c_int64
    X.rsa_pkcs1_sha256_verify.argtypes = [ctypes.c_int64] * 7
    X.rsa_pss_verify_sha256.restype = ctypes.c_int64
    X.rsa_pss_verify_sha256.argtypes = [ctypes.c_int64] * 8

    dummy_buf = (ctypes.c_char * 512)(*([128] + [0]*511))
    dummy_sig = (ctypes.c_char * 512)(*([1]*512))
    dummy_msg = (ctypes.c_char * 32)(*([2]*32))

    for bad_len in [128, 500, 1024]:
        rc_pkcs = X.rsa_pkcs1_sha256_verify(ctypes.addressof(dummy_buf), bad_len, 65537,
                                            ctypes.addressof(dummy_sig), bad_len,
                                            ctypes.addressof(dummy_msg), 32)
        rc_pss = X.rsa_pss_verify_sha256(ctypes.addressof(dummy_buf), bad_len, 65537,
                                         ctypes.addressof(dummy_sig), bad_len,
                                         ctypes.addressof(dummy_msg), 32, 32)
        ok_pkcs = (rc_pkcs == 0)
        ok_pss = (rc_pss == 0)
        if not ok_pkcs or not ok_pss:
            all_ok = False
        print(f"[{'PASS' if ok_pkcs else 'FAIL'}] reject invalid {bad_len}-byte PKCS1: got {rc_pkcs}, want 0")
        print(f"[{'PASS' if ok_pss else 'FAIL'}] reject invalid {bad_len}-byte PSS: got {rc_pss}, want 0")

    sys.exit(0 if all_ok else 1)


def _x509_mode_b2(argv):
    """Test for Blocker B2: anchor's own signatureAlgorithm must not poison verification."""
    import ctypes
    import pathlib
    import struct
    import sys

    SO = argv[0]
    FX = _X509_DIR / "testdata" / "adv"
    TD = _X509_DIR / "testdata"
    NOW = int((TD / "now.txt").read_text().strip())

    X = ctypes.CDLL(SO)
    X.x509_verify_server_chain.restype = ctypes.c_int64
    X.x509_verify_server_chain.argtypes = [ctypes.c_int64] * 4
    X.x509_load_anchors.restype = ctypes.c_int64
    X.x509_load_anchors.argtypes = [ctypes.c_int64, ctypes.c_int64]

    keep = []

    def der(path):
        return path.read_bytes()

    def blob_raw(ders):
        n = len(ders)
        hdr = [n]
        off = 8 + 16 * n
        for d in ders:
            hdr += [off, len(d)]
            off += len(d)
        buf = ctypes.create_string_buffer(off + 8)
        ctypes.memmove(buf, struct.pack("<%dq" % len(hdr), *hdr), len(hdr) * 8)
        base = ctypes.cast(buf, ctypes.c_void_p).value
        p = 8 + 16 * n
        for d in ders:
            ctypes.memmove(base + p, d, len(d))
            p += len(d)
        keep.append(buf)
        return base

    def cstr(s):
        b = ctypes.create_string_buffer(s.encode("utf-8") + b"\0")
        keep.append(b)
        return ctypes.cast(b, ctypes.c_void_p).value

    def load_file(path):
        data = path.read_bytes()
        buf = ctypes.create_string_buffer(data, max(len(data), 1))
        keep.append(buf)
        return X.x509_load_anchors(ctypes.cast(buf, ctypes.c_void_p).value, len(data))

    all_ok = True

    # T01: SHA-1 self-signed anchor, leaf signed sha256WithRSA
    got_t01 = X.x509_verify_server_chain(blob_raw([der(FX / "sha1root_sha256_leaf.der")]),
                                         blob_raw([der(FX / "sha1_selfsigned_root.der")]),
                                         cstr("adv.example.com"), NOW)
    ok_t01 = (got_t01 == 0)
    if not ok_t01: all_ok = False
    print(f"[{'PASS' if ok_t01 else 'FAIL'}] T01 SHA-1 self-signed anchor, leaf signed sha256WithRSA: got {got_t01}, want 0")

    # T02: DER bundle [sha1_root, ec_root] verifying ec_leaf chain
    b_der = load_file(FX / "bundle_sha1_plus_ec.der")
    got_t02 = X.x509_verify_server_chain(blob_raw([der(TD / "ec_leaf.der"), der(TD / "ec_inter.der")]),
                                         b_der, cstr("www.example.com"), NOW)
    ok_t02 = (got_t02 == 0)
    if not ok_t02: all_ok = False
    print(f"[{'PASS' if ok_t02 else 'FAIL'}] T02 DER bundle with SHA-1 anchor: got {got_t02}, want 0")

    # T03: PEM bundle [ec_root, sha1_root] verifying ec_leaf chain
    b_pem = load_file(FX / "bundle_ec_plus_sha1.pem")
    got_t03 = X.x509_verify_server_chain(blob_raw([der(TD / "ec_leaf.der"), der(TD / "ec_inter.der")]),
                                         b_pem, cstr("www.example.com"), NOW)
    ok_t03 = (got_t03 == 0)
    if not ok_t03: all_ok = False
    print(f"[{'PASS' if ok_t03 else 'FAIL'}] T03 PEM bundle with SHA-1 anchor: got {got_t03}, want 0")

    # T04b: Unusual / malformed anchor alongside valid ec_root in explicit blob
    corrupted_anchor = b"\x30\x05\x02\x01\x00\x00\x00"  # invalid cert DER
    got_t04b = X.x509_verify_server_chain(blob_raw([der(TD / "ec_leaf.der"), der(TD / "ec_inter.der")]),
                                          blob_raw([corrupted_anchor, der(TD / "ec_root.der")]),
                                          cstr("www.example.com"), NOW)
    ok_t04b = (got_t04b == 0)
    if not ok_t04b: all_ok = False
    print(f"[{'PASS' if ok_t04b else 'FAIL'}] T04b Unusual anchor does not poison valid anchor: got {got_t04b}, want 0")

    sys.exit(0 if all_ok else 1)


def _x509_mode_b3(argv):
    """Test for Blocker B3: cert -> anchor signature check must not be skipped (kills M11)."""
    import ctypes
    import pathlib
    import struct
    import sys

    SO = argv[0]
    FX = _X509_DIR / "testdata" / "adv"
    TD = _X509_DIR / "testdata"
    NOW = int((TD / "now.txt").read_text().strip())

    X = ctypes.CDLL(SO)
    X.x509_verify_server_chain.restype = ctypes.c_int64
    X.x509_verify_server_chain.argtypes = [ctypes.c_int64] * 4

    keep = []

    def der(path):
        return path.read_bytes()

    def blob_raw(ders):
        n = len(ders)
        hdr = [n]
        off = 8 + 16 * n
        for d in ders:
            hdr += [off, len(d)]
            off += len(d)
        buf = ctypes.create_string_buffer(off + 8)
        ctypes.memmove(buf, struct.pack("<%dq" % len(hdr), *hdr), len(hdr) * 8)
        base = ctypes.cast(buf, ctypes.c_void_p).value
        p = 8 + 16 * n
        for d in ders:
            ctypes.memmove(base + p, d, len(d))
            p += len(d)
        keep.append(buf)
        return base

    def cstr(s):
        b = ctypes.create_string_buffer(s.encode("utf-8") + b"\0")
        keep.append(b)
        return ctypes.cast(b, ctypes.c_void_p).value

    # Case: xs_inter has issuer "xs root", xs_rootA has subject "xs root" but wrong key.
    # Signature verification between xs_inter and xs_rootA MUST FAIL with -3 BAD_SIGNATURE.
    # If cert -> anchor signature verification is skipped (mutant M11), this returns 0.
    chain = blob_raw([der(FX / "xs_leaf.der"), der(FX / "xs_inter.der")])
    anchor_wrong_key = blob_raw([der(FX / "xs_rootA.der")])

    res = X.x509_verify_server_chain(chain, anchor_wrong_key, cstr("adv.example.com"), NOW)
    print(f"T08 xs_inter against wrong-key anchor A: got {res}, want -3")
    if res != -3:
        print("[FAIL] cert -> anchor signature verification not enforced!")
        sys.exit(1)

    print("[PASS] cert -> anchor signature verification correctly rejected wrong key with -3")
    sys.exit(0)


def _x509_mode_b4(argv):
    """Test for Blocker B4: leaf expiry must be isolated (kills mutant M03)."""
    import ctypes
    import pathlib
    import struct
    import sys

    SO = argv[0]
    TD = _X509_DIR / "testdata"
    NOW = int((TD / "now.txt").read_text().strip())

    X = ctypes.CDLL(SO)
    X.x509_verify_server_chain.restype = ctypes.c_int64
    X.x509_verify_server_chain.argtypes = [ctypes.c_int64] * 4

    keep = []

    def der(path):
        return path.read_bytes()

    def blob_raw(ders):
        n = len(ders)
        hdr = [n]
        off = 8 + 16 * n
        for d in ders:
            hdr += [off, len(d)]
            off += len(d)
        buf = ctypes.create_string_buffer(off + 8)
        ctypes.memmove(buf, struct.pack("<%dq" % len(hdr), *hdr), len(hdr) * 8)
        base = ctypes.cast(buf, ctypes.c_void_p).value
        p = 8 + 16 * n
        for d in ders:
            ctypes.memmove(base + p, d, len(d))
            p += len(d)
        keep.append(buf)
        return base

    def cstr(s):
        b = ctypes.create_string_buffer(s.encode("utf-8") + b"\0")
        keep.append(b)
        return ctypes.cast(b, ctypes.c_void_p).value

    # Case: path_leaf_expired is valid until NOW + 5 days.
    # path_ec_inter and path_ec_root are valid until NOW + 1000 days.
    # At now = NOW + 6 days, ONLY the leaf is expired.
    # The intermediate and root certificates are still valid.
    # x509_verify_server_chain MUST return -6 (EXPIRED).
    # If the leaf expiration check is skipped/neutered (mutant M03), this returns 0.
    chain = blob_raw([der(TD / "path_leaf_expired.der"), der(TD / "path_ec_inter.der")])
    anchor = blob_raw([der(TD / "path_ec_root.der")])

    res = X.x509_verify_server_chain(chain, anchor, cstr("path.example.com"), NOW + 6 * 86400)
    print(f"leaf_expired test: got {res}, want -6")
    if res != -6:
        print(f"[FAIL] leaf expiry check not isolated! got {res}, expected -6")
        sys.exit(1)

    print("[PASS] leaf expiry check correctly isolated and returned -6")
    sys.exit(0)


def _x509_mode_b5(argv):
    """Test for Blocker X1: mid-label / non-leftmost wildcards must be refused.

    Kills mutant M01_wildcard_anywhere (std/x509_name.mind: `if star != 0` weakened).
    RFC 6125 §6.4.3 as required by R-01 Need 2: a wildcard is legal only as the
    complete left-most label ``*.``.  Patterns whose star sits anywhere else —
    including the verifier-discovered shapes ``a.*.com`` / ``x.example.*`` whose
    second byte is ``.`` — must return 0 (no match) from x509_dns_match.

    Also pins three further negatives of our own: bare ``*``, double-star
    ``*.*.com``, partial ``f*o.example.com``, and an IDN/punycode left label
    ``xn--*.example.com`` / ``a.xn--*.com``; and wildcard suffixes with an empty
    or root label (``*.``, ``*..example.com``, ``*.example.com.``).
    """
    import ctypes
    import sys

    SO = argv[0]
    X = ctypes.CDLL(SO)
    X.x509_dns_match.restype = ctypes.c_int64
    X.x509_dns_match.argtypes = [ctypes.c_int64] * 4

    _keep = []


    def cstr(s: str):
        b = ctypes.create_string_buffer(s.encode("utf-8"))
        _keep.append(b)
        return ctypes.cast(b, ctypes.c_void_p).value


    def dns_match(pat: str, host: str) -> int:
        return X.x509_dns_match(cstr(pat), len(pat), cstr(host), len(host))


    # (name, pattern, host, want)  want: 1 = match, 0 = no match
    CASES = [
        # --- existing legal leftmost wildcard (must stay green) ---
        ("wild_ok", "*.example.com", "www.example.com", 1),
        ("wild_ok_two", "*.example.com", "a.example.com", 1),
        ("wild_bare_label", "*.example.com", "example.com", 0),
        ("wild_two_labels", "*.example.com", "a.b.example.com", 0),
        ("wild_case", "*.Example.COM", "www.EXAMPLE.com", 1),
        # --- X1 kill-M01: star not leftmost AND pat[1]=='.' (verifier shapes) ---
        ("m01_mid_star_dot", "a.*.com", "z.*.com", 0),
        ("m01_trailing_star", "x.example.*", "b.example.*", 0),
        ("m01_mid_star_dot_host_real", "a.*.com", "z.foo.com", 0),
        ("m01_trailing_star_host_real", "x.example.*", "b.example.org", 0),
        # --- further wildcard negatives (policy floor) ---
        ("wild_star_alone", "*", "anything.example.com", 0),
        ("wild_star_alone_empty_host", "*", "", 0),
        ("wild_double_star", "*.*.com", "a.b.com", 0),
        ("wild_double_star_mid", "a.*.*.com", "z.b.c.com", 0),
        ("wild_partial", "f*o.example.com", "foo.example.com", 0),
        ("wild_partial_mid", "f*o.example.com", "fXo.example.com", 0),
        ("wild_inner_star", "foo*.example.com", "foobar.example.com", 0),
        ("wild_idn_star_left", "xn--*.example.com", "xn--foo.example.com", 0),
        ("wild_idn_mid_star_dot", "a.xn--*.com", "z.xn--*.com", 0),
        ("wild_empty_left", "*.example.com", ".example.com", 0),
        # --- the labels after "*." must be non-empty (RFC 6125 §6.4.3 DNS-ID) ---
        ("wild_root_only", "*.", "com.", 0),
        ("wild_empty_suffix_label", "*..example.com", "a..example.com", 0),
        ("wild_trailing_dot_suffix", "*.example.com.", "a.example.com.", 0),
        ("wild_ok_after_label_check", "*.a.example.com", "x.a.example.com", 1),
    ]

    failed = 0
    for name, pat, host, want in CASES:
        got = dns_match(pat, host)
        ok = got == want
        if not ok:
            failed += 1
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: pat={pat!r} host={host!r} got {got}, want {want}")

    print("=" * 60)
    if failed:
        print(f"{failed}/{len(CASES)} FAILED — wildcard policy not enforced (M01 would survive)")
        sys.exit(1)
    print(f"ALL {len(CASES)} WILDCARD CASES PASS")
    sys.exit(0)


def _x509_mode_b6(argv):
    """Test for Blocker X2: trust-anchor validity policy is intentional and pinned.

    Policy (RFC 5280 §6.1.1(d) trust-anchor semantics, matching Chromium):
    anchors are trusted public-key bindings; their self-signature and validity
    period are not evaluated; every non-anchor certificate in the path is checked
    for signature, validity, CA/pathLen, key usage and name constraints.

    Two halves, both pinned here:
      (1) leaf under an EXPIRED anchor            -> 0  (documented, not a bug)
      (2) leaf under a valid anchor via an EXPIRED
          intermediate                           -> -6 (must still fail closed)

    Disabling the is_anchor bypass in x509_parent_policy makes (1) go red.
    Disabling intermediate validity checks makes (2) go red.
    """
    import ctypes
    import pathlib
    import struct
    import sys

    SO = argv[0]
    TD = _X509_DIR / "testdata"
    NOW = int((TD / "now.txt").read_text().strip())

    X = ctypes.CDLL(SO)
    X.x509_verify_server_chain.restype = ctypes.c_int64
    X.x509_verify_server_chain.argtypes = [ctypes.c_int64] * 4

    keep = []


    def der(path):
        return path.read_bytes()


    def blob_raw(ders):
        n = len(ders)
        hdr = [n]
        off = 8 + 16 * n
        for d in ders:
            hdr += [off, len(d)]
            off += len(d)
        buf = ctypes.create_string_buffer(off + 8)
        ctypes.memmove(buf, struct.pack("<%dq" % len(hdr), *hdr), len(hdr) * 8)
        base = ctypes.cast(buf, ctypes.c_void_p).value
        p = 8 + 16 * n
        for d in ders:
            ctypes.memmove(base + p, d, len(d))
            p += len(d)
        keep.append(buf)
        return base


    def cstr(s):
        b = ctypes.create_string_buffer(s.encode("utf-8") + b"\0")
        keep.append(b)
        return ctypes.cast(b, ctypes.c_void_p).value


    failed = 0


    def check(name, got, want):
        global failed
        ok = got == want
        if not ok:
            failed += 1
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: got {got}, want {want}")


    # (1) EXPIRED anchor is still a trusted key binding -> 0
    # path_anchor_expired: notAfter = NOW - 1 day (expired)
    # path_exp_inter, path_exp_leaf: valid dates
    chain = blob_raw([der(TD / "path_exp_leaf.der"), der(TD / "path_exp_inter.der")])
    anchor = blob_raw([der(TD / "path_anchor_expired.der")])
    res = X.x509_verify_server_chain(chain, anchor, cstr("path.example.com"), NOW)
    check("expired_anchor_still_trusted", res, 0)

    # (2) EXPIRED intermediate under a valid anchor -> -6 (fail closed)
    # path_expired is an expired intermediate (same subject/key as path_ec_inter)
    # signed by the valid path_ec_root.
    chain2 = blob_raw([der(TD / "path_ec_leaf.der"), der(TD / "path_expired.der")])
    anchor2 = blob_raw([der(TD / "path_ec_root.der")])
    res2 = X.x509_verify_server_chain(chain2, anchor2, cstr("path.example.com"), NOW)
    check("expired_intermediate_rejected", res2, -6)

    # Control: the same leaf via the VALID intermediate under the valid root -> 0
    chain3 = blob_raw([der(TD / "path_ec_leaf.der"), der(TD / "path_ec_inter.der")])
    res3 = X.x509_verify_server_chain(chain3, anchor2, cstr("path.example.com"), NOW)
    check("valid_intermediate_accepted", res3, 0)

    # (3) An anchor must itself parse under the certificate signature-algorithm
    # rules; one that does not (RSASSA-PSS with SHA-384 parameters on its own
    # self-signature) is skipped, so the chain finds no anchor -> -7. The control
    # anchor (same shape, sha384WithRSAEncryption self-signature) verifies -> 0.
    ADV = TD / "adv"
    ADV_NOW = 1792140749
    chain4 = blob_raw([der(ADV / "a_leaf_pss384_root.der"), der(ADV / "a_inter_pss384_root.der")])
    anchor4 = blob_raw([der(ADV / "a_root_pss384.der")])
    res4 = X.x509_verify_server_chain(chain4, anchor4, cstr("anc.example.com"), ADV_NOW)
    check("unsupported_anchor_sigparams_skipped", res4, -7)
    chain5 = blob_raw([der(ADV / "a_leaf_rsa384_root.der"), der(ADV / "a_inter_rsa384_root.der")])
    anchor5 = blob_raw([der(ADV / "a_root_rsa384.der")])
    res5 = X.x509_verify_server_chain(chain5, anchor5, cstr("anc.example.com"), ADV_NOW)
    check("supported_anchor_sigalg_accepted", res5, 0)

    print("=" * 60)
    if failed:
        print(f"{failed}/5 FAILED — trust-anchor validity policy not pinned")
        sys.exit(1)
    print("ALL 5 ANCHOR-VALIDITY POLICY CASES PASS")
    sys.exit(0)


def _x509_mode_gen_path(argv):
    """Public regression certificates. Keys exist only in process memory, never on disk.
    Run after gen_fixtures.sh; fixed validity relative to its committed now.txt.
    """
    import datetime as dt
    import ipaddress
    from pathlib import Path
    from cryptography import x509 as x
    from cryptography.hazmat.primitives import hashes, serialization as ser
    from cryptography.hazmat.primitives.asymmetric import ec, rsa, padding
    from cryptography.x509.oid import NameOID, ExtendedKeyUsageOID as E
    TD = _X509_DIR / 'testdata'
    NOW = dt.datetime.fromtimestamp(int((TD / 'now.txt').read_text()), dt.timezone.utc)
    D = dt.timedelta(days=1)
    serial = 9000

    def key(alg):
        return rsa.generate_private_key(public_exponent=65537, key_size=2048) if alg in ('rsa', 'pss') else ec.generate_private_key(ec.SECP384R1() if alg == 'p384' else ec.SECP256R1())

    def cert(file, subject, k, issuer, ik, alg='ec', ca=False, pathlen=None, before=-10, after=1000, ku=True, eku=None, names=None, nc=None):
        global serial
        serial += 1
        name = x.Name([x.NameAttribute(NameOID.COMMON_NAME, subject)])
        b = x.CertificateBuilder().subject_name(name).issuer_name(issuer.subject if issuer else name).public_key(k.public_key()).serial_number(serial).not_valid_before(NOW + before*D).not_valid_after(NOW + after*D)
        b = b.add_extension(x.BasicConstraints(ca, pathlen), True)
        b = b.add_extension(x.KeyUsage(not ca, False, False, False, False, ca and ku, False, None, None), True)
        if eku is not None:
            b = b.add_extension(x.ExtendedKeyUsage(eku), True)
        if names:
            b = b.add_extension(x.SubjectAlternativeName(names), False)
        if nc:
            b = b.add_extension(nc, True)
        opts = {'rsa_padding': padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=32)} if alg == 'pss' else {}
        c = b.sign(ik, hashes.SHA384() if alg == 'p384' else hashes.SHA256(), **opts)
        (TD / (file + '.der')).write_bytes(c.public_bytes(ser.Encoding.DER))
        return c

    roots = []
    for alg in ('ec', 'rsa', 'pss', 'p384'):
        rk, ik, lk = key(alg), key(alg), key(alg)
        prefix = 'path_' + alg
        root = cert(prefix+'_root', prefix+' root', rk, None, rk, alg, ca=True, pathlen=3)
        inter = cert(prefix+'_inter', prefix+' inter', ik, root, rk, alg, ca=True, pathlen=0)
        cert(prefix+'_leaf', prefix+' leaf', lk, inter, ik, alg, eku=[E.SERVER_AUTH], names=[x.DNSName('path.example.com')])
        roots.append(root)
        if alg == 'ec':
            cert('path_leaf_expired', prefix+' leaf', lk, inter, ik, alg, before=-10, after=5, eku=[E.SERVER_AUTH], names=[x.DNSName('path.example.com')])
            # Same subject/key, differing policies: alternative path search must backtrack.
            for suffix, opts in [('expired', {'before':-20,'after':-1}), ('future', {'before':1}), ('badku', {'ku':False}), ('badeku', {'eku':[E.CLIENT_AUTH]})]:
                cert('path_'+suffix, prefix+' inter', ik, root, rk, ca=True, **opts)
            other = key('ec')
            cert('path_wrongkey', prefix+' inter', other, root, rk, ca=True)
            cert('path_root_len0', prefix+' root', rk, None, rk, ca=True, pathlen=0)
            rk_wrong = key('ec')
            cert('path_anchor_wrongkey', prefix+' root', rk_wrong, None, rk_wrong, ca=True, pathlen=3)
            nc = x.NameConstraints([x.DNSName('.example.com'), x.IPAddress(ipaddress.ip_network('10.0.0.0/8'))], [x.DNSName('blocked.example.com'), x.IPAddress(ipaddress.ip_network('10.1.0.0/16'))])
            ci = cert('path_nc_inter', prefix+' inter', ik, root, rk, ca=True, nc=nc)
            for suffix, names in [('outside', [x.DNSName('outside.test')]), ('ip_good', [x.IPAddress(ipaddress.ip_address('10.2.0.1'))]), ('ip_bad', [x.IPAddress(ipaddress.ip_address('10.1.0.1'))]), ('ip_outside', [x.IPAddress(ipaddress.ip_address('192.0.2.1'))]), ('ipv6', [x.IPAddress(ipaddress.ip_address('2001:db8::1'))])]:
                cert('path_'+suffix, suffix, lk, ci, ik, eku=[E.SERVER_AUTH], names=names)
            # X1: non-leftmost / partial wildcards (RFC 6125 §6.4.3). These SANs must
            # never match; chain-level twins of test_b5_wildcard_negatives.py. Kills M01.
            for suffix, san, host in [
                ('m01_mid_star_dot', 'a.*.com', 'z.*.com'),
                ('m01_trailing_star', 'x.example.*', 'b.example.*'),
                ('wild_star_alone', '*', 'anything.example.com'),
                ('wild_double_star', '*.*.com', 'a.b.com'),
                ('wild_partial', 'f*o.example.com', 'foo.example.com'),
                ('wild_idn_star', 'xn--*.example.com', 'xn--foo.example.com'),
                ('wild_idn_mid_star_dot', 'a.xn--*.com', 'z.xn--*.com'),
            ]:
                cert('path_'+suffix, suffix, lk, inter, ik, eku=[E.SERVER_AUTH], names=[x.DNSName(san)])
            # X2: expired trust anchor (validity NOT evaluated on anchors) and a
            # valid-dated intermediate under it, so both halves of the policy pin.
            rkx, ikx, lkx = key('ec'), key('ec'), key('ec')
            exp_anchor = cert('path_anchor_expired', 'path expired anchor', rkx, None, rkx, ca=True, pathlen=3, before=-20, after=-1)
            exp_inter = cert('path_exp_inter', 'path exp inter', ikx, exp_anchor, rkx, ca=True, pathlen=0)
            cert('path_exp_leaf', 'path exp leaf', lkx, exp_inter, ikx, eku=[E.SERVER_AUTH], names=[x.DNSName('path.example.com')])
    # Combine all four algorithm roots with original roots for full bundle matrix.
    original = [x.load_der_x509_certificate((TD / (n+'_root.der')).read_bytes()) for n in ('ec','rsa','pss','p384')]
    (TD/'bundle_anchors.der').write_bytes(b''.join(c.public_bytes(ser.Encoding.DER) for c in original+roots))
    (TD/'bundle_anchors.pem').write_bytes(b''.join(c.public_bytes(ser.Encoding.PEM) for c in original+roots))
    print('generated path fixtures; no private keys written')


def _x509_mode_gen_malformed(argv):
    """Derive malformed public certificates without any private-key material."""
    from pathlib import Path
    TD = _X509_DIR / 'testdata'

    def tlv(tag, body):
        n = len(body)
        size = bytes([n]) if n < 128 else bytes([128 + (n.bit_length()+7)//8]) + n.to_bytes((n.bit_length()+7)//8, 'big')
        return bytes([tag]) + size + body

    def parts(data):
        out = []
        p = 0
        while p < len(data):
            start = p
            tag, size = data[p:p+2]
            p += 2
            if size > 127:
                width = size & 127
                size = int.from_bytes(data[p:p+width], 'big')
                p += width
            out.append((tag, data[p:p+size], data[start:p+size]))
            p += size
        assert p == len(data)
        return out

    outer = parts(parts((TD/'path_pss_leaf.der').read_bytes())[0][1])
    tbs = parts(outer[0][1])
    alg = parts(outer[1][1])
    params = parts(alg[1][1])
    for name, ps in [('pss_default_hash', [p for p in params if p[0] != 160]), ('pss_default_mgf', [p for p in params if p[0] != 161]), ('pss_bad_trailer', params + [(163, b'', tlv(163, tlv(2, b'\x02')))])]:
        new_alg = tlv(48, alg[0][2] + tlv(48, b''.join(p[2] for p in ps)))
        # Both AlgorithmIdentifiers must agree, to test parameter policy itself.
        body = b''.join(new_alg if i == 2 else p[2] for i,p in enumerate(tbs))
        (TD/(name+'.der')).write_bytes(tlv(48, tlv(48, body) + new_alg + outer[2][2]))
    print('derived three malformed PSS parameter fixtures')

    ec_raw = (TD/'ec_leaf.der').read_bytes()
    ec_parts = parts(parts(ec_raw)[0][1])
    (TD/'der_trailing.der').write_bytes(ec_raw + b'junk')
    (TD/'der_sig_bits.der').write_bytes(tlv(48, ec_parts[0][2] + ec_parts[1][2] + tlv(3, b'\x01' + ec_parts[2][1][1:])))

    # Strip a necessary positive sign octet: same magnitude, illegal negative INTEGER.
    for fixture in ('ec_leaf', 'ec_leaf_wild'):
        ep = parts(parts((TD/(fixture+'.der')).read_bytes())[0][1])
        sig = parts(parts(ep[2][1][1:])[0][1])
        for idx, integer in enumerate(sig):
            if integer[1][0] == 0:
                bad = b''.join(tlv(2, v[1][1:]) if i == idx else v[2] for i,v in enumerate(sig))
                (TD/'der_negative_ecdsa.der').write_bytes(tlv(48, ep[0][2] + ep[1][2] + tlv(3, b'\x00' + tlv(48, bad))))
                (TD/'der_negative_ecdsa_host.txt').write_text('www.example.com' if fixture == 'ec_leaf' else 'www.wild.example.com')
                break
        else:
            continue
        break
    else:
        raise RuntimeError('regenerate EC fixtures to obtain a sign-octet case')


MODES = {
    "b1": _x509_mode_b1,
    "b2": _x509_mode_b2,
    "b3": _x509_mode_b3,
    "b4": _x509_mode_b4,
    "b5": _x509_mode_b5,
    "b6": _x509_mode_b6,
    "gen-path": _x509_mode_gen_path,
    "gen-malformed": _x509_mode_gen_malformed,
}
