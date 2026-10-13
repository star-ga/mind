#!/usr/bin/env python3
# Generate tests/x509/testdata/cases_native.bin — the offline fixture pack for
# the NATIVE-backend harness (tests/x509/native_main.mind +
# tests/x509/build_native.sh).  Layout (all little-endian i64):
#   i64 ncases
#   per case:
#     i64 want          (expected x509_verify_server_chain return code)
#     i64 now           (unix seconds)
#     256 bytes         NUL-terminated expected_dns, zero-padded
#     i64 nchain,  then per cert: i64 len + DER bytes   (leaf-first)
#     i64 nanchor, then per cert: i64 len + DER bytes
# Regenerate after gen_fixtures.sh: python3 gen_native_manifest.py
import pathlib
import struct
import sys

TD = pathlib.Path(__file__).parent / "testdata"


def der(name):
    return (TD / f"{name}.der").read_bytes()


NOW = int((TD / "now.txt").read_text().strip())
NOW_FUTURE = NOW + 3800 * 86400
NOW_PAST = NOW - 3800 * 86400

# (name, chain[], anchors[], dns, now, want) — must mirror x509_chain_driver.py
CASES = [
    ("ec_ok", ["ec_leaf", "ec_inter"], ["ec_root"], "www.example.com", NOW, 0),
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
]


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
        b = dns_name.encode("utf-8")
        assert len(b) < 256
        out.append(b + b"\0" * (256 - len(b)))
        emit_seq(chain)
        emit_seq(anchors)
    blob = b"".join(out)
    (TD / "cases_native.bin").write_bytes(blob)
    print(f"wrote {TD/'cases_native.bin'} ({len(blob)} bytes, {len(CASES)} cases)")


if __name__ == "__main__":
    main()
