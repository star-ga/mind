# TLS 1.3 client recorded flights

The pure-MIND tests complete eight deterministic TLS 1.3 handshakes: AES-128-GCM
and ChaCha20-Poly1305, X25519 and X25519MLKEM768, each with and without HRR.
They verify CONNECTED, exact client Finished bytes, one application record in
each direction, and version/suite/group/ALPN/certificate/verdict evidence.
Each positive flight has expired-time, wrong-name and unrelated-anchor cases,
each paired with the same flight under correct configuration. Sixteen additional
flights flip one CV signature bit or one Finished bit before valid AEAD sealing.
All assertions and byte comparisons are in tests/tls13_client_cases.mind and
its runner. Failure returns a nonzero MIND result; no application data is released.

## Chain verification (real verifier)

Production always calls x509_verify_server_chain at Certificate, and both
suites compile the unmodified std/ sources: the recorded flights are
authenticated by the real std/x509 path validator against the caller's anchor
set. The ABI is the x509 one: four i64 arguments, an i64 count plus (relative
offset, length) blob for the chain and for the anchors, a NUL-terminated
hostname. The client keeps its public u32be serialized anchor input and DER
evidence layout and converts to that ABI at the mandatory call. Verdicts:
expired -6, name mismatch -8, unknown anchor -7 (not yet valid -5).
There is no runtime selection flag, callback, global or peer-controlled
verifier selector, and nothing under std/ references a test verifier.

tests/tls13_client_shipped.mind replays all eight positive recordings with an
anchor set that does not contain the fixture root. It requires verdict -7,
ERROR/E_CHAIN, no application data, and latched refusal on
feed/pump/tick/write/close.

`test_x509_verify.inc.mind` is the narrow fixture verifier that stood in for
the real one before the client and the path validator joined (pinned anchor
DER, RSA PKCS1/SHA256 signatures, validity, exact SAN). No build splices it in;
it is kept, uncalled, as a reference for the fixture PKI.

## Provenance and regeneration

Client entropy byte i is i mod 256; server entropy byte i is (101+i) mod 256.
Time is 1700000000.
The fixture PKI uses deterministic seed 20261014, RSA-2048, SAN `rsa`, serverAuth
EKU, and validity 2020-01-01 through 2030-01-01. The unrelated root is separate.
`fixture_private.mind` contains a PUBLIC, TEST-ONLY private RSA key. NEVER use it
for real traffic or trust. No standalone private-key JSON is committed.

The containers, `fixture_keys.mind`, `fixture_private.mind` and `cvsigs.mind`
were produced offline: the PKI from the fixed seed, the CertificateVerify
transcripts by the pure-MIND generator (`generate.mind`), and deterministic
RSA-PSS signatures pinned in `cvsigs.mind`; MIND computes key agreement,
traffic secrets, AEAD records and Finished. The test gate computes no expected
value: tests/std_tls_x509_import.rs embeds the frozen bytes verbatim as stores,
builds the program with `mindc build` and propagates the MIND result as the
exit code. The handshake driver's `make-fixture-keys` mode is the offline
provenance tool that emits the PKI as MIND constants, and its `regenerate`
mode the offline signing/freezing tool (the MIND generator provides the CV
transcripts); neither is imported or run by any test gate.

SHA256SUMS pins all 24 containers plus key/signature constants. The generator
shares primitives with the client, so this is regression evidence, not an
independent cryptographic oracle. The RFC 8448 §3 transcript and key-schedule
known answers in tests/std_tls_x509_import.rs, and the crypto-vector drivers,
supply published vectors for those primitives.

Each container starts with twelve little-endian i64 fields: suite, group,
flags, CH1/CH2/HRR/SH lengths, encrypted-flight length, server application-record
length, expected client Finished-record length, expected client application-record
length, tamper mode (0 normal, 1 CV, 2 Finished). The eight byte strings follow
in that order. Certificate is fragmented even across its handshake header.

```sh
cargo test --release --no-default-features \
  --features 'mlir-build std-surface cross-module-imports' --test std_tls_x509_import
export MINDC=${CARGO_TARGET_DIR:-target}/release/mindc
nice -n 15 timeout 600 python3 tests/tls13_handshake_driver.py client
nice -n 15 timeout 600 python3 tests/tls13_handshake_driver.py client --native
sha256sum -c tests/fixtures/tls13_client/SHA256SUMS
# Development-only, intentional fixture replacement:
PYTHONDONTWRITEBYTECODE=1 nice -n 15 timeout 120 python3 tests/tls13_handshake_driver.py make-fixture-keys
PYTHONDONTWRITEBYTECODE=1 nice -n 15 timeout 600 python3 tests/tls13_handshake_driver.py regenerate
```

`tests/tls13_handshake_driver.py client` (run by `scripts/run_crypto_vectors.sh`)
is the same suite through `--emit-shared`: it concatenates the client's module
set, the fixture PKI, the cases, the runner and the shipped suite with the
frozen containers embedded as stores, and propagates the MIND result of
`tls13ct_run_all` and `tls13ct_run_shipped`; it computes no expected value.
`--shipped` runs only the shipped suite. `--native` builds the same program
with `mindc build --backend native` and never falls back to MLIR.

The recorded flights run in the `mindc build` executable only: the tree-walking
evaluator behind `mindc test` would take hours for 24 handshakes. `mindc build
--backend native` refuses `import std.tls13_client;` at the import, because the
client modules are not rows of the native seed. The `--native` driver mode is
refused with exit 3 and writes no artifact. `native_alloc_probe.mind`
reproduces the frozen-profile allocation edge (closure.unresolved_edge); the
frozen-profile fence also excludes built-in calls, division and shifts.
`native_hkdf_loop_probe.mind` reduces a second blocker from hkdf_expand: an
intrinsic-free value-if inside a loop reaches the frozen emitter and is
refused with exit 3, while the identical source executes as zero under MLIR.
Admission alone cannot close that emitter gap; it needs a reseed. Native
acceptance is unqualified until the compiler supports this program.

## Consumer API

Caller owns sockets, time, tick budget, entropy and immutable anchor storage.
Initialize tls13c_state_size bytes with tls13c_init; supply anchors as u32be count
followed by repeated u32be DER length and DER. Set ALPN in INIT using a wire list
of u8 length plus protocol bytes. Build/seal CH1, drain read_out, then feed/pump.
Flag bit0 enables hybrid; bit3 defers key shares for HRR. On status 10 build/seal
CH2 with original entropy. Only one retry is accepted. DNS names send SNI; IP
literals do not. No socket or clock is opened internally.

Evidence accessors expose version, suite, group, selected ALPN, DER chain, and
verdict (1 unchecked, 0 accept, negative refusal). Evidence alone is not proof
of authentication. Limits: 8192-byte handshake messages, 16384-byte transcript,
1024-byte ALPN offers, 254-byte cookie, 253-byte hostname, a 2048-byte
application receive buffer (a larger record, or several records that together
exceed it in one pump, fails the connection with an overflow error), and at most
32 trust anchors (the std/x509 verifier refuses a larger anchor set with -13);
CV supports RSA-PSS/SHA256 with an RSA-2048 leaf key only. Other CV
algorithms and keys fail closed.

Known limitations (declared, not fixed here; each fails closed unless marked):
- A post-handshake NewSessionTicket or KeyUpdate is fatal (the connection
  errors), so most live TLS 1.3 servers end the session after the handshake.
- Application data and close_notify processed in the same pump: the data is
  dropped and the connection reports a clean close (silent truncation, does
  not fail closed). Pump after every feed until this is fixed.
- Only RSA-PSS CV with an RSA-2048 leaf key is implemented. Its key is taken
  from the leaf only when the SPKI decodes strictly as rsaEncryption
  (`x509_rsa_spki`); an ECDSA or other leaf key and an RSA-1024 key fail
  closed at CertificateVerify (E_CV), a malformed rsaEncryption key at
  Certificate (E_CHAIN), and an RSA-3072 or larger key at Certificate (E_OVF).
  The RFC 8448 §3 server key is RSA-1024: `tls13ct_cv` checks its published
  CertificateVerify signature with the RSA-PSS verify primitive
  (`tls13_hs_rsa_pss_verify`) and checks that the client refuses it (E_CV).
- Hostname bytes are not validated (no NUL/LDH check); SNI sends dns_len bytes
  while the verifier stops at the first NUL. Callers must pass a clean name.
- write_app with a full output buffer fails the connection (no backpressure).
- Ephemeral keys stay in state while connected and are zeroed on close/fail;
  the anchor buffer is re-read at Certificate and must stay immutable; CCS
  records before Finished are bounded only by the tick budget.
