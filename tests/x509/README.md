# X.509 chain and trust-anchor tests

All committed certificates are public test certificates. Most were generated
offline with openssl under temporary keys that were never committed; the `kp_*`
fixtures are derived from fixed inputs (below). No network or system trust
store is used, and every case runs at the fixed clock in `testdata/now.txt`.

### The `kp_*` key-policy fixtures (deterministic)

`kp_root`, `kp_leaf`, `kp_leaf_badkey` and `kp_rsa3072_leaf` come from fixed
inputs, so regenerating them gives the same bytes. Each RSA prime is the first
probable prime `p >= c` with `p mod 65537 != 1`, stepping `c` by 2, where `c`
is the first `bits/2` bits of the stream SHA-256(L || be32(0)) ||
SHA-256(L || be32(1)) || ... for the ASCII label L =
`mind-x509-kp/<key>/<p|q>`, with the top two bits and the low bit set;
primality is Miller-Rabin with the first 64 primes as bases; e = 65537. The
keys are `root` and `leaf` (RSA-2048) and `leaf3072` (RSA-3072). Every
certificate is X.509 v3, sha256WithRSAEncryption (PKCS#1 v1.5, so the
signature is deterministic) by `root`, valid 2026-01-01T00:00:00Z to
2036-01-01T00:00:00Z. The extensions below come in the order listed, then
subjectKeyIdentifier and authorityKeyIdentifier (keyIdentifier only), both the
SHA-1 of the subjectPublicKey bits (RFC 5280 method 1):

- `kp_root`: CN `MindTest Key Policy Root`, self-issued, serial
  0x6b70000000000001, basicConstraints cA (critical), keyUsage keyCertSign and
  cRLSign (critical).
- `kp_leaf` (serial 0x6b70000000000002, CN `kp leaf`, key `leaf`) and
  `kp_rsa3072_leaf` (serial 0x6b70000000000003, CN `kp rsa3072 leaf`, key
  `leaf3072`): basicConstraints not cA (critical), keyUsage digitalSignature
  and keyEncipherment (critical), EKU serverAuth (critical), SAN dNSName
  `www.kp-test.example.com`.
- `kp_leaf_badkey`: the tbsCertificate of `kp_leaf` with its RSAPublicKey
  re-encoded as SEQUENCE { n, e, INTEGER 1 } (every enclosing length
  re-encoded, nothing else changed), signed again by `root`.

The chain verifier refuses `kp_leaf_badkey` with -14 while `kp_leaf` under
the same root verifies (0); the TLS client refuses `kp_rsa3072_leaf` at
Certificate with E_OVF while its chain verifies (0). SHA-256 of the DER:

| file | bytes | SHA-256 |
|------|-------|---------|
| `kp_root.der` | 815 | `0788adfce95e19e33667687e505b56d4f81b10b4a4b6274a68294ebe653efc59` |
| `kp_leaf.der` | 857 | `e72e6424289644b0e3f2d4b47796d517a5eced53b45eca76fe08fbb0e65fa45d` |
| `kp_leaf_badkey.der` | 860 | `50788c77ed839e40ff1f32bf98717236cb4fa81ab002e7f95e9a8786bfde5d01` |
| `kp_rsa3072_leaf.der` | 993 | `9de35e42fb7d10b4b1c82e11b7043c5571ddb2a4644d33ed6b011a4f9d0dde73` |

## API and supported policy

`x509_verify_server_chain(chain, anchors, expected_dns, now_unix)` (in
`std/x509_path.mind`; `import std.x509;` loads it) returns zero on success and
a negative verdict on failure. Both blobs contain an i64 count, then count
pairs of i64 offset/length, followed by DER bytes. The first chain
certificate is the leaf; the remaining certificates are unordered candidates.
The caller must provide valid blob storage and offsets: this pointer-only ABI
has no total buffer-length argument. The DNS/IP argument is NUL-terminated.

Path search is bounded to eight supplied certificates and 32 anchors, retries
alternative issuers, and enforces signatures, validity, CA/pathLen, KU, EKU,
and SAN identity. A leaf must carry an EKU that lists serverAuth (-11), an IP
literal host matches only an iPAddress SAN (-8), and a certificate whose
tbsCertificate signature AlgorithmIdentifier differs from its outer
signatureAlgorithm does not parse (-1). Supported signatures: RSA PKCS#1
SHA-256, RSA-PSS with explicit SHA-256/MGF1-SHA-256, ECDSA P-256
SHA-256/SHA-384, ECDSA P-384 SHA-384. SHA-1 is refused, including implicit PSS
hash/MGF defaults. Name constraints support DNS subtrees and IPv4/IPv6
address/mask subtrees; other constraint name forms and nonzero subtree bounds
fail closed. Issuer/subject matching uses exact DER names.
Revocation and certificate-policy processing are outside this implementation.

Trust-anchor policy (RFC 5280 §6.1.1(d)): anchors are trusted public-key
bindings. Their self-signature is never verified and their validity period is
not evaluated, so an unrecognised signature algorithm on an anchor is
tolerated. Each anchor must still parse as a certificate: malformed DER, or
RSASSA-PSS parameters on its own signature other than SHA-256, make the parse
fail and that anchor is skipped (a chain that needs it fails -7; pinned in
`x509_trust_anchor_policy` in `tests/x509_chain_smoke.rs`, with a control anchor that verifies). An anchor
that issues a path link must be a CA (basicConstraints cA, else -9), carry
keyCertSign if it has keyUsage (else -11), list serverAuth if it has an EKU
(else -11), and its pathLen (-10) and name constraints (-17) apply to the path
below it. Every non-anchor certificate in the path is checked for signature,
validity, CA/pathLen, key usage and name constraints. A leaf under an expired
anchor therefore verifies (return 0); a leaf under a valid anchor via an
expired intermediate fails closed with -6. A chain, anchor or hostname pointer
<= 0 is refused with -1. A DNS wildcard is legal only as the complete
left-most label `*.` followed by non-empty labels (`*.`, `*..example.com` and
`*.example.com.` never match).

`x509_load_anchors(bytes, length)` decodes an in-memory concatenated DER or PEM
bundle to an owned blob (positive address), or returns a negative error. It
checks encoding/envelope structure; the chain verifier parses anchor certificates
and applies policy. The caller owns the returned blob. No implicit trust store.
The loader accepts at most 32 anchors. The crypto/parser implementation uses
heap scratch allocations and is not yet qualified for long-running connection
loads; these tests establish functional correctness, not an allocation bound.

### Known fail-closed PEM limits

The PEM loader is strict: anything that is not a clean run of
`-----BEGIN CERTIFICATE-----` … `-----END CERTIFICATE-----` blocks is refused
with -1. Two shapes that appear in real-world bundles are therefore rejected
rather than skipped:

1. **Bundle comment lines** — e.g. Mozilla `cacert.pem` `##` header comments
   outside the PEM blocks. Refused with -1.
2. **`openssl x509 -text` dumps** — human-readable certificate text between or
   before PEM blocks. Refused with -1.

These are deliberate fail-closed policies (a parser that silently skips unknown
bytes is a substitution hazard). Strip comments/text before loading, or feed
DER. `testdata/bundle_bad.pem` is the malformed-PEM rejection case.

## Gates

```sh
cargo test --release --no-default-features \
  --features 'mlir-build std-surface cross-module-imports' \
  --test x509_chain_smoke --test std_tls_x509_import
```

`tests/x509_chain_smoke.rs` compiles the validator's module set with
`mindc --emit-shared` and replays `testdata/cases_native.bin` (every case of the
matrix: positive EC / RSA-PKCS1 / RSA-PSS / P-384 chains through intermediates,
DER and PEM anchor bundles, and each refusal class), then the regression cases
for the fixes carried with the validator: RSA-3072/4096 and RSA-PSS 4096
issuers, an anchor's own signature algorithm not poisoning verification, the
last-certificate-to-anchor signature check, leaf expiry and leaf not-yet-valid
on their own, the wildcard rules, and the trust-anchor policy above. They read `testdata/` and
`testdata/adv/` (RSA-3072/4096 chains, a SHA-1 self-signed root, mixed DER/PEM
anchor bundles, a cross-signed anchor pair, and RSASSA-PSS/SHA-384 versus
sha384WithRSAEncryption self-signed anchors).

`tests/x509_pss_salt.rs` covers the RSA-PSS salt-length bound (RFC 8017 §9.1.2,
`n_len - 34` for SHA-256) with `adv/pss3072_long_*` and `adv/pss4096_long_*`:
a root and a leaf per key size, openssl-generated under temporary keys that
were not kept, the leaf signed with the largest salt the root key allows (350
and 478). `pss3072_long_leaf_salt351` is `pss3072_long_leaf` with saltLength 351
patched into both AlgorithmIdentifier copies (the signature no longer matches,
so a refusal as a bad signature (-3) would also be a failure of the bound check,
which must answer -2 first).

`tests/std_tls_x509_import.rs` runs the verifier through `import std.x509;`
under `mindc test` and in a `mindc build` executable, each with a mutated copy
that must fail. Both engines also run the short DNS names, the IPv4 and IPv6
literal parsers and the DER and PEM anchor loader. Three rules have their own
fixtures under `eku_root`: `eku_client_leaf` (EKU clientAuth only, -11),
`dns_ip_leaf` (a dNSName `127.0.0.1`, -8 for host `127.0.0.1`) and the valid
control `eku_server_leaf`; the test derives an `ec_leaf` whose outer
signatureAlgorithm says SHA-384 while its tbsCertificate says SHA-256 (-1). `scripts/run_crypto_vectors.sh` builds the same module set for
the field-level `tests/x509_vectors_driver.py`.

## `--backend native`

The native backend compiles only the std modules of its frozen seed, and these
modules are not seed rows, so `mindc build --backend native` refuses
`import std.x509;` at the import. The `mindc build` (MLIR) executable is what
the tests above run.
