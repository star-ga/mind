# X.509 chain and trust-anchor tests

All committed certificates are public test certificates, generated offline with
openssl under temporary keys that were never committed. No network or system
trust store is used, and every case runs at the fixed clock in `testdata/now.txt`.

## API and supported policy

`x509_verify_server_chain(chain, anchors, expected_dns, now_unix)` returns zero
on success and a negative verdict on failure. Both blobs contain an i64 count,
then count pairs of i64 offset/length, followed by DER bytes. The first chain
certificate is the leaf; the remaining certificates are unordered candidates.
The caller must provide valid blob storage and offsets: this pointer-only ABI
has no total buffer-length argument. The DNS/IP argument is NUL-terminated.

Path search is bounded to eight supplied certificates and 32 anchors, retries
alternative issuers, and enforces signatures, validity, CA/pathLen, KU, EKU,
and SAN identity. Supported signatures: RSA PKCS#1 SHA-256, RSA-PSS with explicit
SHA-256/MGF1-SHA-256, ECDSA P-256 SHA-256/SHA-384, ECDSA P-384 SHA-384. SHA-1 is
refused, including implicit PSS hash/MGF defaults. Name constraints support DNS
subtrees and IPv4/IPv6 address/mask subtrees; other constraint name forms and
nonzero subtree bounds fail closed. Issuer/subject matching uses exact DER names.
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
last-certificate-to-anchor signature check, leaf expiry on its own, the
wildcard rules, and the trust-anchor policy above. They read `testdata/` and
`testdata/adv/` (RSA-3072/4096 chains, a SHA-1 self-signed root, mixed DER/PEM
anchor bundles, a cross-signed anchor pair, and RSASSA-PSS/SHA-384 versus
sha384WithRSAEncryption self-signed anchors).

`tests/std_tls_x509_import.rs` runs the verifier through `import std.x509;`
under `mindc test` and in a `mindc build` executable, each with a mutated copy
that must fail. `scripts/run_crypto_vectors.sh` builds the same module set for
the field-level `tests/x509_vectors_driver.py`.

## `--backend native`

The native backend compiles only the std modules of its frozen seed, and these
modules are not seed rows, so `mindc build --backend native` refuses
`import std.x509;` at the import. The `mindc build` (MLIR) executable is what
the tests above run.
