# std.https loopback fixtures (TEST-ONLY)

Everything here is test-only and public by design. NEVER use these keys or
anchors for real traffic or trust.

Used by `tests/std_https_loopback.rs`, which builds `server.mind` and
`client.mind` with `mindc build` (the harness prepends the std imports and
writes the DER certificates, the server key and the bundle paths as MIND
stores) and runs them on 127.0.0.1, and by `tests/std_https_openssl.rs`, which
builds `request.mind` (one request per run: spec on stdin, result on stdout)
and runs it against `openssl s_server` presenting the same certificates with
the keys below.

| File | What |
|------|------|
| `root.der` | RSA-2048 root `CN=MIND Test HTTPS Root`, CA, keyCertSign/cRLSign |
| `inter.der` | RSA-2048 intermediate, CA pathlen 0, signed by the root |
| `leaf_good.der` | RSA-2048 leaf, SAN `localhost`, serverAuth, 2026-01-01..2046-01-01 |
| `leaf_expired.der` | the same key, validity 2020-01-01..2021-01-01 |
| `leaf_wronghost.der` | the same key, SAN `wrong.invalid` |
| `leaf_ec.der` | a P-256 leaf for `localhost` (refused: not RSA-2048) |
| `leaf_rsa3072.der` | an RSA-3072 leaf for `localhost` (refused: not RSA-2048) |
| `server_key_test_only.txt` | modulus `n` and private exponent `d` of the RSA-2048 leaf key, hex |
| `leaf_key_test_only.cnf` | the same RSA-2048 leaf key with its CRT values, as `openssl asn1parse -genconf` input (PKCS#1 DER) |
| `leaf_rsa3072_key_test_only.cnf` | the RSA-3072 key of `leaf_rsa3072.der`, in the same form |
| `request.mind` | the one-request client of `tests/std_https_openssl.rs` |
| `ca_bundle.pem` | 40 P-256 filler roots, then the test root, with `#` comment and blank lines |
| `ca_other.pem` | an unrelated RSA-2048 root (the unknown-anchor case) |
| `ca_bad.pem` | a malformed PEM block (refused) |

Every certificate signature is sha256WithRSAEncryption (PKCS#1 v1.5).

## Regeneration (OpenSSL 3.5)

```
NB="-not_before 20260101000000Z"; NA="-not_after 20460101000000Z"
# ca.ext:    basicConstraints=critical,CA:TRUE / keyUsage=critical,keyCertSign,cRLSign / subjectKeyIdentifier=hash
# inter.ext: as ca.ext with pathlen:0, plus authorityKeyIdentifier=keyid
# leaf.ext:  basicConstraints=critical,CA:FALSE / keyUsage=critical,digitalSignature,keyEncipherment
#            extendedKeyUsage=serverAuth / subjectAltName=DNS:localhost (DNS:wrong.invalid for the wrong-host leaf)
openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:2048 -out root.key
openssl req -new -key root.key -subj "/CN=MIND Test HTTPS Root" | \
  openssl x509 -req -key root.key -sha256 $NB $NA -set_serial 1 -extfile ca.ext -out root.pem
openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:2048 -out inter.key
openssl req -new -key inter.key -subj "/CN=MIND Test HTTPS Intermediate" | \
  openssl x509 -req -CA root.pem -CAkey root.key -sha256 $NB $NA -set_serial 2 -extfile inter.ext -out inter.pem
openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:2048 -out leaf.key
openssl req -new -key leaf.key -subj "/CN=localhost" | \
  openssl x509 -req -CA inter.pem -CAkey inter.key -sha256 $NB $NA -set_serial 10 -extfile leaf.ext -out leaf_good.pem
# leaf_expired: -not_before 20200101000000Z -not_after 20210101000000Z, serial 11
# leaf_wronghost: wrong-host leaf.ext, serial 12
# leaf_ec / leaf_rsa3072: an EC P-256 / RSA-3072 key, leaf.ext, serials 13 / 14
# other root: as root.pem with CN=MIND Test Unrelated Root, serial 3
# filler roots: 40 self-signed P-256 CAs, CN=MIND Test Filler Root <i>, serials 101..140
openssl x509 -in <cert>.pem -outform DER -out <cert>.der
openssl pkey -in leaf.key -noout -text   # modulus and privateExponent -> server_key_test_only.txt
# the same fields with the primes, exponents and coefficient -> leaf_key_test_only.cnf
# (and the RSA-3072 key -> leaf_rsa3072_key_test_only.cnf); check with
openssl asn1parse -genconf leaf_key_test_only.cnf -noout -out k.der && openssl pkey -inform DER -in k.der -check -noout
```
The intermediate's and the root's private keys are not kept.
