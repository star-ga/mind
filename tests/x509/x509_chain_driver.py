#!/usr/bin/env python3
# Chain-validation driver for std/x509.mind :: x509_verify_server_chain.
#
# Loads the committed openssl-generated fixtures from tests/x509/testdata/ and
# exercises the compiled shared library through every required case:
# positive (EC chain, RSA chain, RSA-PSS chain, wildcard, A-label),
# wrong host, unknown anchor, expired, not-yet-valid, broken signature,
# non-CA intermediate, pathLen violation, missing serverAuth EKU, missing SAN,
# malformed/truncated DER, unknown critical extension, depth > 8.
#
# Build the .so first:  tests/x509/build_shared.sh /tmp/x509.so
# Usage: python3 x509_chain_driver.py /tmp/x509.so

import ctypes
import pathlib
import sys

SO = sys.argv[1]
TD = pathlib.Path(__file__).parent / "testdata"

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

ec_chain = blob("ec_leaf", "ec_inter")
ec_anchor = blob("ec_root")

# --- positives ---
check("positive: EC chain", X.x509_verify_server_chain(ec_chain, ec_anchor, dns("www.example.com"), NOW), 0)
check("positive: hostname case-insensitive", X.x509_verify_server_chain(ec_chain, ec_anchor, dns("WWW.EXAMPLE.COM"), NOW), 0)
check("positive: RSA pkcs1 chain", X.x509_verify_server_chain(blob("rsa_leaf"), blob("rsa_root"), dns("www.rsa-test.example.com"), NOW), 0)
check("positive: RSA-PSS chain", X.x509_verify_server_chain(blob("pss_leaf"), blob("pss_root"), dns("www.pss-test.example.com"), NOW), 0)
check("positive: root included in chain", X.x509_verify_server_chain(blob("ec_leaf", "ec_inter", "ec_root"), ec_anchor, dns("www.example.com"), NOW), 0)
check("positive: extra unrelated anchor tolerated", X.x509_verify_server_chain(ec_chain, blob("pl_root", "ec_root"), dns("www.example.com"), NOW), 0)

# wildcard
wild_chain = blob("ec_leaf_wild", "ec_inter")
check("positive: wildcard *.wild.example.com matches www.wild.example.com", X.x509_verify_server_chain(wild_chain, ec_anchor, dns("www.wild.example.com"), NOW), 0)
check("negative: wildcard does not match bare domain", X.x509_verify_server_chain(wild_chain, ec_anchor, dns("wild.example.com"), NOW), -8)
check("negative: wildcard does not cross two labels", X.x509_verify_server_chain(wild_chain, ec_anchor, dns("a.b.wild.example.com"), NOW), -8)
check("positive: A-label exact match", X.x509_verify_server_chain(wild_chain, ec_anchor, dns("xn--nxasmq6b.example.com"), NOW), 0)
check("negative: A-label is not U-label", X.x509_verify_server_chain(wild_chain, ec_anchor, dns("mÃ¼ller.example.com"), NOW), -8)

# --- negatives on the EC chain ---
check("negative: wrong host", X.x509_verify_server_chain(ec_chain, ec_anchor, dns("other.example.com"), NOW), -8)
check("negative: unknown anchor", X.x509_verify_server_chain(ec_chain, blob("pl_root"), dns("www.example.com"), NOW), -7)
check("negative: empty anchor set", X.x509_verify_server_chain(ec_chain, blob(), dns("www.example.com"), NOW), -7)
check("negative: expired", X.x509_verify_server_chain(ec_chain, ec_anchor, dns("www.example.com"), NOW_FUTURE), -6)
check("negative: not yet valid", X.x509_verify_server_chain(ec_chain, ec_anchor, dns("www.example.com"), NOW_PAST), -5)
check("negative: broken signature", X.x509_verify_server_chain(blob("ec_leaf_badsig", "ec_inter"), ec_anchor, dns("www.example.com"), NOW), -3)
check("negative: non-CA intermediate", X.x509_verify_server_chain(blob("noca_leaf", "noca_inter"), blob("noca_root"), dns("noca.example.com"), NOW), -9)
check("negative: pathLen violation", X.x509_verify_server_chain(blob("pl_leaf", "pl_i2", "pl_i1"), blob("pl_root"), dns("pl.example.com"), NOW), -10)
check("negative: missing serverAuth EKU", X.x509_verify_server_chain(blob("ec_leaf_noeku", "ec_inter"), ec_anchor, dns("www.example.com"), NOW), -11)
check("negative: missing SAN (CN fallback not used)", X.x509_verify_server_chain(blob("ec_leaf_nosan", "ec_inter"), ec_anchor, dns("www.example.com"), NOW), -15)
check("negative: truncated DER", X.x509_verify_server_chain(blob("ec_leaf_trunc", "ec_inter"), ec_anchor, dns("www.example.com"), NOW), -1)
check("negative: unknown critical extension", X.x509_verify_server_chain(blob("ec_leaf_crit", "ec_inter"), ec_anchor, dns("www.example.com"), NOW), -12)
check("negative: chain depth > 8", X.x509_verify_server_chain(blob(*(["ec_leaf"] * 9)), ec_anchor, dns("www.example.com"), NOW), -13)

print("=" * 60)
if all(results):
    print(f"ALL {len(results)} CASES PASS (peer_auth_supported={X.x509_peer_auth_supported()})")
    sys.exit(0)
print(f"{sum(results)}/{len(results)} passed")
sys.exit(1)
