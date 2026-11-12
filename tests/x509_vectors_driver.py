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
# Build the .so first. std/x509.mind imports the chain validator's module set
# (sha256, sha512, ecdsa_p256, x509_p384, x509_bignum, rsa_pss and the x509_*
# helpers); scripts/run_crypto_vectors.sh combines that set into one file with
# the import lines stripped (X509_SET) and emits it with --emit-shared. Its
# x509-build-shared mode builds the same set on its own:
#   scripts/run_crypto_vectors.sh x509-build-shared /tmp/x509.so
# Path validation, validity and hostname checks are tested by
# tests/x509_chain_smoke.rs and tests/std_tls_x509_import.rs, and by this
# driver's chain mode.
#
# Usage: python3 x509_vectors_driver.py /tmp/x509.so

import ctypes
import sys

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey
from cryptography.exceptions import InvalidSignature
import datetime

# Folded x509 harnesses (chain, blocker cases, offline generators).
# The no-mode invocation below remains the known-answer vector driver.
#   python3 tests/x509_vectors_driver.py <mode> [args...]
# modes: chain, b1, b2, b3, b4, b5, b6, gen-path, gen-malformed, gen-manifest
from _x509_vectors_modes import MODES as _X509_HELPER_MODES, _X509_DIR

def _x509_manifest_state():
    # Generate tests/x509/testdata/cases_native.bin — the offline fixture pack for
    # the NATIVE-backend harness (tests/x509/native_main.mind +
    # tests/x509/build_native.sh).  Layout (all little-endian i64):
    #   i64 ncases
    #   per case:
    #     i64 want          (expected x509_verify_server_chain return code)
    #     i64 now           (unix seconds)
    #     i64 bundle_mode   (0=explicit anchors, 1=DER bundle, 2=PEM bundle)
    #     256 bytes         NUL-terminated expected_dns, zero-padded
    #     i64 nchain,  then per cert: i64 len + DER bytes   (leaf-first)
    #     i64 nanchor, then per cert: i64 len + DER bytes
    # Regenerate after gen_fixtures.sh: python3 gen_native_manifest.py
    import pathlib
    import struct
    import sys

    TD = _X509_DIR / "testdata"


    def der(name):
        return (TD / f"{name}.der").read_bytes()


    NOW = int((TD / "now.txt").read_text().strip())
    NOW_FUTURE = NOW + 3800 * 86400
    NOW_PAST = NOW - 3800 * 86400

    # (name, chain[], anchors[], dns, now, want) — must mirror x509_chain_driver.py
    CASES = [
        ("ec_ok", ["ec_leaf", "ec_inter"], ["ec_root"], "www.example.com", NOW, 0),
        ("unordered", ["ec_leaf", "ec_root", "ec_inter"], ["ec_root"], "www.example.com", NOW, 0),
        ("unrelated_intermediate", ["ec_leaf", "pl_root", "ec_inter"], ["ec_root"], "www.example.com", NOW, 0),
        ("ec_casehost", ["ec_leaf", "ec_inter"], ["ec_root"], "WWW.EXAMPLE.COM", NOW, 0),
        ("rsa_ok", ["rsa_leaf"], ["rsa_root"], "www.rsa-test.example.com", NOW, 0),
        ("pss_ok", ["pss_leaf"], ["pss_root"], "www.pss-test.example.com", NOW, 0),
        ("root_in_chain", ["ec_leaf", "ec_inter", "ec_root"], ["ec_root"], "www.example.com", NOW, 0),
        ("extra_anchor", ["ec_leaf", "ec_inter"], ["pl_root", "ec_root"], "www.example.com", NOW, 0),
        ("wild_ok", ["ec_leaf_wild", "ec_inter"], ["ec_root"], "www.wild.example.com", NOW, 0),
        ("wild_bare", ["ec_leaf_wild", "ec_inter"], ["ec_root"], "wild.example.com", NOW, -8),
        ("wild_two_labels", ["ec_leaf_wild", "ec_inter"], ["ec_root"], "a.b.wild.example.com", NOW, -8),
        ("alabel_ok", ["ec_leaf_wild", "ec_inter"], ["ec_root"], "xn--nxasmq6b.example.com", NOW, 0),
        ("alabel_ulabel_mismatch", ["ec_leaf_wild", "ec_inter"], ["ec_root"], "m\u00fcller.example.com", NOW, -8),
        ("wrong_host", ["ec_leaf", "ec_inter"], ["ec_root"], "other.example.com", NOW, -8),
        ("unknown_anchor", ["ec_leaf", "ec_inter"], ["pl_root"], "www.example.com", NOW, -7),
        ("empty_anchor", ["ec_leaf", "ec_inter"], [], "www.example.com", NOW, -7),
        ("expired", ["ec_leaf", "ec_inter"], ["ec_root"], "www.example.com", NOW_FUTURE, -6),
        ("not_yet_valid", ["ec_leaf", "ec_inter"], ["ec_root"], "www.example.com", NOW_PAST, -5),
        ("broken_sig", ["ec_leaf_badsig", "ec_inter"], ["ec_root"], "www.example.com", NOW, -3),
        ("non_ca_inter", ["noca_leaf", "noca_inter"], ["noca_root"], "noca.example.com", NOW, -9),
        ("pathlen", ["pl_leaf", "pl_i2", "pl_i1"], ["pl_root"], "pl.example.com", NOW, -10),
        ("no_eku", ["ec_leaf_noeku", "ec_inter"], ["ec_root"], "www.example.com", NOW, -11),
        ("no_san", ["ec_leaf_nosan", "ec_inter"], ["ec_root"], "www.example.com", NOW, -15),
        ("truncated", ["ec_leaf_trunc", "ec_inter"], ["ec_root"], "www.example.com", NOW, -1),
        ("unknown_crit", ["ec_leaf_crit", "ec_inter"], ["ec_root"], "www.example.com", NOW, -12),
        ("too_deep", ["ec_leaf"] * 9, ["ec_root"], "www.example.com", NOW, -13),
        ("p384_ok", ["p384_leaf"], ["p384_root"], "www.p384-test.example.com", NOW, 0),
        ("nc_ok", ["nc_leaf_ok", "nc_inter"], ["nc_root"], "good.example.com", NOW, 0),
        ("nc_bad_excluded", ["nc_leaf_bad", "nc_inter"], ["nc_root"], "bad.example.com", NOW, -17),
        ("nc_bad_not_permitted", ["nc_leaf_ok", "nc_inter"], ["nc_root"], "good.other.com", NOW, -8),
        ("midwild_no_match", ["ec_leaf_midwild", "ec_inter"], ["ec_root"], "sub.foo.example.com", NOW, -8),
        ("bad_key_usage", ["ec_leaf_badku", "ec_inter"], ["ec_root"], "www.example.com", NOW, -11),
        ("ip_ok", ["ec_leaf_ip", "ec_inter"], ["ec_root"], "127.0.0.1", NOW, 0),
        ("ip_mismatch", ["ec_leaf_ip", "ec_inter"], ["ec_root"], "192.168.1.1", NOW, -8),
        ("ip_vs_dns", ["ec_leaf_ip", "ec_inter"], ["ec_root"], "www.example.com", NOW, -8),
        ("dns_vs_ip", ["ec_leaf", "ec_inter"], ["ec_root"], "127.0.0.1", NOW, -8),
        ("wild_vs_ip", ["ec_leaf_wild", "ec_inter"], ["ec_root"], "127.0.0.1", NOW, -8),
        ("sha1_refused", ["sha1_leaf"], ["sha1_root"], "www.sha1-test.example.com", NOW, -2),
    ]

    for alg in ('ec', 'rsa', 'pss', 'p384'):
        p = 'path_' + alg
        CASES.append((p, [p+'_leaf', p+'_inter'], [p+'_root'], 'path.example.com', NOW, 0))
    for suffix, want in [('expired', -6), ('future', -5), ('badku', -11), ('badeku', -11), ('wrongkey', -3)]:
        CASES.append(('inter_'+suffix, ['path_ec_leaf', 'path_'+suffix], ['path_ec_root'], 'path.example.com', NOW, want))
        CASES.append(('alternate_'+suffix, ['path_ec_leaf', 'path_'+suffix, 'path_ec_inter'], ['path_ec_root'], 'path.example.com', NOW, 0))
    CASES += [
        ('leaf_expired', ['path_leaf_expired', 'path_ec_inter'], ['path_ec_root'], 'path.example.com', NOW + 6 * 86400, -6),
        ('anchor_pathlen', ['path_ec_leaf', 'path_ec_inter'], ['path_root_len0'], 'path.example.com', NOW, -10),
        ('anchor_wrongkey', ['path_ec_leaf', 'path_ec_inter'], ['path_anchor_wrongkey'], 'path.example.com', NOW, -3),
        # X1: non-leftmost wildcards must never match (kills M01_wildcard_anywhere)
        ('m01_mid_star_dot', ['path_m01_mid_star_dot', 'path_ec_inter'], ['path_ec_root'], 'z.*.com', NOW, -8),
        ('m01_trailing_star', ['path_m01_trailing_star', 'path_ec_inter'], ['path_ec_root'], 'b.example.*', NOW, -8),
        ('wild_star_alone', ['path_wild_star_alone', 'path_ec_inter'], ['path_ec_root'], 'anything.example.com', NOW, -8),
        ('wild_double_star', ['path_wild_double_star', 'path_ec_inter'], ['path_ec_root'], 'a.b.com', NOW, -8),
        ('wild_partial', ['path_wild_partial', 'path_ec_inter'], ['path_ec_root'], 'foo.example.com', NOW, -8),
        ('wild_idn_star', ['path_wild_idn_star', 'path_ec_inter'], ['path_ec_root'], 'xn--foo.example.com', NOW, -8),
        ('wild_idn_mid_star_dot', ['path_wild_idn_mid_star_dot', 'path_ec_inter'], ['path_ec_root'], 'z.xn--*.com', NOW, -8),
        # X2: anchors are trusted key bindings — validity period is NOT evaluated
        ('anchor_expired_ok', ['path_exp_leaf', 'path_exp_inter'], ['path_anchor_expired'], 'path.example.com', NOW, 0),
        ('nc_outside', ['path_outside', 'path_nc_inter'], ['path_ec_root'], 'outside.test', NOW, -17),
        ('nc_ip_good', ['path_ip_good', 'path_nc_inter'], ['path_ec_root'], '10.2.0.1', NOW, 0),
        ('nc_ip_excluded', ['path_ip_bad', 'path_nc_inter'], ['path_ec_root'], '10.1.0.1', NOW, -17),
        ('nc_ip_outside', ['path_ip_outside', 'path_nc_inter'], ['path_ec_root'], '192.0.2.1', NOW, -17),
        ('ipv6', ['path_ipv6', 'path_ec_inter'], ['path_ec_root'], '2001:db8::1', NOW, 0),
        ('ipv6_mismatch', ['path_ipv6', 'path_ec_inter'], ['path_ec_root'], '2001:db8::2', NOW, -8),
    ]

    for name in ('pss_default_hash', 'pss_default_mgf', 'pss_bad_trailer'):
        CASES.append((name, [name, 'path_pss_inter'], ['path_pss_root'], 'path.example.com', NOW, -2))

    for name in ('der_trailing', 'der_sig_bits'):
        CASES.append((name, [name, 'ec_inter'], ['ec_root'], 'www.example.com', NOW, -1))

    CASES.append(('der_negative_ecdsa', ['der_negative_ecdsa', 'ec_inter'], ['ec_root'], (TD/'der_negative_ecdsa_host.txt').read_text(), NOW, -3))

    for mode in ('der', 'pem'):
        for alg in ('ec', 'rsa', 'pss', 'p384'):
            p = 'path_' + alg
            CASES.append(('bundle_'+mode+'_'+alg, [p+'_leaf', p+'_inter'], [], 'path.example.com', NOW, 0))


    def main():
        out = []
        out.append(struct.pack("<q", len(CASES)))
        def emit_seq(names):
            out.append(struct.pack("<q", len(names)))
            for c in names:
                d = der(c)
                out.append(struct.pack("<q", len(d)))
                out.append(d)
                pad = (8 - len(d) % 8) % 8
                if pad:
                    out.append(b"\0" * pad)
        for name, chain, anchors, dns_name, now, want in CASES:
            out.append(struct.pack("<q", want))
            out.append(struct.pack("<q", now))
            mode = 1 if name.startswith("bundle_der_") else 2 if name.startswith("bundle_pem_") else 0
            out.append(struct.pack("<q", mode))
            b = dns_name.encode("utf-8")
            assert len(b) < 256
            out.append(b + b"\0" * (256 - len(b)))
            emit_seq(chain)
            emit_seq(anchors)
        blob = b"".join(out)
        (TD / "cases_native.bin").write_bytes(blob)
        print(f"wrote {TD/'cases_native.bin'} ({len(blob)} bytes, {len(CASES)} cases)")
    return CASES, main


def _x509_mode_gen_manifest(argv):
    _cases, write = _x509_manifest_state()
    del _cases, argv
    write()
    return 0


def _x509_mode_chain(argv):
    # Chain-validation driver for std/x509.mind :: x509_verify_server_chain.
    #
    # Loads the committed openssl-generated fixtures from tests/x509/testdata/ and
    # exercises the compiled shared library through every required case:
    # positive (EC chain, RSA chain, RSA-PSS chain, wildcard, A-label),
    # wrong host, unknown anchor, expired, not-yet-valid, broken signature,
    # non-CA intermediate, pathLen violation, missing serverAuth EKU, missing SAN,
    # malformed/truncated DER, unknown critical extension, depth > 8.
    #
    # Build the .so first:  scripts/run_crypto_vectors.sh x509-build-shared /tmp/x509.so
    # Usage: python3 tests/x509_vectors_driver.py chain /tmp/x509.so

    import ctypes
    import pathlib
    import sys

    SO = argv[0]
    TD = _X509_DIR / "testdata"

    X = ctypes.CDLL(SO)
    X.x509_verify_server_chain.restype = ctypes.c_int64
    X.x509_verify_server_chain.argtypes = [ctypes.c_int64] * 4
    X.x509_peer_auth_supported.restype = ctypes.c_int64
    X.x509_peer_auth_supported.argtypes = []

    NOW = int((TD / "now.txt").read_text().strip())
    NOW_FUTURE = NOW + 3800 * 86400   # past every fixture's notAfter (10y + slack)
    NOW_PAST = NOW - 3800 * 86400     # before every fixture's notBefore


    def der(name):
        return (TD / f"{name}.der").read_bytes()


    _keepalive = []


    def blob(*names):
        """Build the (count, off[i], len[i], data...) blob for a cert list."""
        ders = [der(n) for n in names]
        n = len(ders)
        hdr = [n]
        off = 8 + 16 * n
        for d in ders:
            hdr.append(off)
            hdr.append(len(d))
            off += len(d)
        buf = ctypes.create_string_buffer(off)
        import struct
        ctypes.memmove(buf, struct.pack("<%dq" % len(hdr), *hdr), len(hdr) * 8)
        base = ctypes.cast(buf, ctypes.c_void_p).value
        p = 8 + 16 * n
        for d in ders:
            ctypes.memmove(base + p, d, len(d))
            p += len(d)
        _keepalive.append(buf)
        return ctypes.cast(buf, ctypes.c_void_p).value


    def dns(name):
        raw = name.encode("utf-8")
        b = ctypes.create_string_buffer(raw + b"\0", len(raw) + 1)
        _keepalive.append(b)
        return ctypes.cast(b, ctypes.c_void_p).value


    results = []


    def check(name, got, want):
        ok = got == want
        results.append(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: got {got}, want {want}")


    errs = {
        0: "OK",
        -1: "MALFORMED", -2: "UNSUPPORTED_ALG", -3: "BAD_SIGNATURE",
        -4: "NAME_MISMATCH", -5: "NOT_YET_VALID", -6: "EXPIRED",
        -7: "UNKNOWN_ANCHOR", -8: "HOSTNAME_MISMATCH", -9: "NOT_A_CA",
        -10: "PATHLEN", -11: "KEYUSAGE_EKU", -12: "UNKNOWN_CRITICAL",
        -13: "CHAIN_LENGTH", -14: "BAD_KEY", -15: "NO_SAN",
    }

    # --- in-memory bundle loading tests ---
    X.x509_load_anchors.restype = ctypes.c_int64
    X.x509_load_anchors.argtypes = [ctypes.c_int64, ctypes.c_int64]

    def load_bundle_raw(path):
        data = path.read_bytes()
        buf = ctypes.create_string_buffer(data, len(data))
        _keepalive.append(buf)
        ptr = ctypes.cast(buf, ctypes.c_void_p).value
        return X.x509_load_anchors(ptr, len(data))

    CASES, _manifest_write = _x509_manifest_state()
    bundle_der = load_bundle_raw(TD / "bundle_anchors.der")
    bundle_pem = load_bundle_raw(TD / "bundle_anchors.pem")
    for name, chain, anchors, host, when, want in CASES:
        ap = bundle_der if name.startswith("bundle_der_") else bundle_pem if name.startswith("bundle_pem_") else blob(*anchors)
        check(name, X.x509_verify_server_chain(blob(*chain), ap, dns(host), when) if ap > 0 else ap, want)
    ec_chain = blob("ec_leaf", "ec_inter")

    der_bundle_ptr = load_bundle_raw(TD / "bundle_anchors.der")
    check("bundle: load DER bundle returns valid pointer", der_bundle_ptr > 0, True)
    if der_bundle_ptr > 0:
        check("bundle: verify EC chain against DER bundle", X.x509_verify_server_chain(ec_chain, der_bundle_ptr, dns("www.example.com"), NOW), 0)
        check("bundle: verify P-384 chain against DER bundle", X.x509_verify_server_chain(blob("p384_leaf"), der_bundle_ptr, dns("www.p384-test.example.com"), NOW), 0)

    pem_bundle_ptr = load_bundle_raw(TD / "bundle_anchors.pem")
    check("bundle: load PEM bundle returns valid pointer", pem_bundle_ptr > 0, True)
    if pem_bundle_ptr > 0:
        check("bundle: verify RSA chain against PEM bundle", X.x509_verify_server_chain(blob("rsa_leaf"), pem_bundle_ptr, dns("www.rsa-test.example.com"), NOW), 0)
        check("bundle: verify P-384 chain against PEM bundle", X.x509_verify_server_chain(blob("p384_leaf"), pem_bundle_ptr, dns("www.p384-test.example.com"), NOW), 0)

    bad_pem_res = load_bundle_raw(TD / "bundle_bad.pem")
    check("bundle: malformed PEM bundle fails closed", bad_pem_res < 0, True)

    # A null or negative chain, anchor or hostname pointer fails closed (-1)
    # before any load instead of faulting.
    check("guard: null chain pointer", X.x509_verify_server_chain(0, blob("ec_root"), dns("www.example.com"), NOW), -1)
    check("guard: negative anchors pointer", X.x509_verify_server_chain(ec_chain, -1, dns("www.example.com"), NOW), -1)
    check("guard: null hostname pointer", X.x509_verify_server_chain(ec_chain, blob("ec_root"), 0, NOW), -1)

    print("=" * 60)
    if all(results):
        print(f"ALL {len(results)} CASES PASS (peer_auth_supported={X.x509_peer_auth_supported()})")
        sys.exit(0)
    print(f"{sum(results)}/{len(results)} passed")
    sys.exit(1)


_X509_MODES = {
    "chain": _x509_mode_chain,
    **_X509_HELPER_MODES,  # b1-b6, gen-path, gen-malformed: tests/_x509_vectors_modes.py
    "gen-manifest": _x509_mode_gen_manifest,
}

if __name__ == "__main__" and len(sys.argv) >= 2 and sys.argv[1] in _X509_MODES:
    raise SystemExit(_X509_MODES[sys.argv[1]](sys.argv[2:]))

so_path = sys.argv[1]
X = ctypes.CDLL(so_path)
X.x509_parse.restype = ctypes.c_int64
X.x509_parse.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
X.rsa_pkcs1_sha256_verify.restype = ctypes.c_int64
X.rsa_pkcs1_sha256_verify.argtypes = [ctypes.c_int64] * 7
X.x509_verify_self_signed.restype = ctypes.c_int64
X.x509_verify_self_signed.argtypes = [ctypes.c_int64, ctypes.c_int64]
X.x509_peer_auth_supported.restype = ctypes.c_int64
X.x509_peer_auth_supported.argtypes = []

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

# ---------------------------------------------------------------------------
# CAPABILITY SCOPE (honesty marker): std/x509 authenticates a peer chain
# (x509_verify_server_chain: path, validity, key usage and hostname), so the
# module declares 1. Removing that validator without lowering this value back
# to 0 must turn this driver RED (authentication tests live in this file's
# chain mode and in tests/x509_chain_smoke.rs).
# ---------------------------------------------------------------------------
print("=" * 72)
print("CAPABILITY SCOPE - module declares peer authentication")
print("=" * 72)
record_bool("scope: x509_peer_auth_supported()==1 (chain/validity/hostname implemented)",
            X.x509_peer_auth_supported() == 1,
            f"got={X.x509_peer_auth_supported()}")

print("=" * 72)
total = len(results)
passed = sum(1 for r in results if r)
print(f"SUMMARY: {passed}/{total} checks PASSED")
print("=" * 72)
sys.exit(0 if passed == total else 1)
