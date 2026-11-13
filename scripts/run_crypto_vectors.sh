#!/usr/bin/env bash
# Build every pure-MIND crypto/TLS std module to a shared object and run its
# official-vector (RFC KAT) driver under tests/*_driver.py. Each driver's exit
# code gates this script; a single wrong known-answer test turns it red.
#
# This is the runner behind the `crypto_vectors` CI job (see
# .github/workflows/crypto-vectors.yml). It intentionally has NO silent-skip
# path: if `mindc` cannot emit a shared object (missing MLIR toolchain) the
# build step errors and the script aborts, so the gate can never pass
# vacuously.
#
# Usage:
#   MINDC=./target/release/mindc scripts/run_crypto_vectors.sh
#
# Env:
#   MINDC   path to a mindc built with the `mlir-build` feature (required —
#           the standalone `--emit-shared` path lowers MIND IR to MLIR text and
#           shells out to mlir-opt/mlir-translate/clang).
#   PYREFS  optional extra dir prepended to PYTHONPATH (for hpack / kyber-py
#           reference packages installed with `pip install --target`).
set -euo pipefail

# Offline x509 shared-library emit, native harness, and fixture regeneration.
x509_build_shared() {
# Build std/x509.mind (+ its std crypto dependencies) as a shared library via
# mindc --emit-shared (MLIR route).  Cross-module calls between std modules are
# resolved by concatenation (the std imports inside bundled mode only cover the
# seeded/bundled subset listed in examples/mindc_mind/testdata/stdlib_manifest.txt),
# exactly as tests/rsa_pss_driver.py and tests/ecdsa_p256_driver.py document.
#
#   Usage: scripts/run_crypto_vectors.sh x509-build-shared <output.so> [mindc]
set -euo pipefail
OUT="${1:?usage: scripts/run_crypto_vectors.sh x509-build-shared <out.so> [mindc]}"
export MINDC_BUILD_TOOL_TIMEOUT_SECS="${MINDC_BUILD_TOOL_TIMEOUT_SECS:-900}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MINDC="${2:-${CARGO_TARGET_DIR:-$ROOT/target}/release/mindc}"
COMBINED="${OUT%.so}.combined.mind"
{
    grep -v '^import ' "$ROOT/std/sha256.mind"
    grep -v '^import ' "$ROOT/std/sha512.mind"
    grep -v '^import ' "$ROOT/std/ecdsa_p256.mind"
    grep -v '^import ' "$ROOT/std/x509_p384.mind"
    grep -v '^import ' "$ROOT/std/x509_bignum.mind"
    grep -v '^import ' "$ROOT/std/rsa_pss.mind"
    grep -v '^import ' "$ROOT/std/x509_bundle.mind"
    grep -v '^import ' "$ROOT/std/x509_der.mind"
    grep -v '^import ' "$ROOT/std/x509_ext.mind"
    grep -v '^import ' "$ROOT/std/x509_name.mind"
    grep -v '^import ' "$ROOT/std/x509_sig.mind"
    grep -v '^import ' "$ROOT/std/x509_path.mind"
    grep -v '^import ' "$ROOT/std/x509.mind"
} > "$COMBINED"
"$MINDC" "$COMBINED" --emit-shared "$OUT"
}
x509_build_native() {
# Build and run the NATIVE-backend harness for std/x509.mind chain validation.
#
# The pure-MIND native harness embeds the fixture pack (tests/x509/testdata/cases_native.bin, whose
# layout is documented in the gen-manifest mode) is embedded as a hex string
# literal in the generated main. SHA-256 comes from the native seed blob;
# the remaining crypto and X.509 sources are concatenated. --backend native
# is mandatory: selecting a CPU target alone still uses MLIR.
#
#   Usage: scripts/run_crypto_vectors.sh x509-build-native [mindc]
set -euo pipefail
export MINDC_BUILD_TOOL_TIMEOUT_SECS="${MINDC_BUILD_TOOL_TIMEOUT_SECS:-900}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HERE="$ROOT/tests/x509"
MINDC="${1:-${CARGO_TARGET_DIR:-$ROOT/target}/release/mindc}"
WORK="$ROOT/target/x509-native"
mkdir -p "$WORK/src"

{
    echo "import std.sha256;"
    grep -v '^import ' "$ROOT/std/sha512.mind"
    grep -v '^import ' "$ROOT/std/ecdsa_p256.mind"
    grep -v '^import ' "$ROOT/std/x509_p384.mind"
    grep -v '^import ' "$ROOT/std/x509_bignum.mind"
    grep -v '^import ' "$ROOT/std/rsa_pss.mind"
    grep -v '^import ' "$ROOT/std/x509_bundle.mind"
    grep -v '^import ' "$ROOT/std/x509_der.mind"
    grep -v '^import ' "$ROOT/std/x509_ext.mind"
    grep -v '^import ' "$ROOT/std/x509_name.mind"
    grep -v '^import ' "$ROOT/std/x509_sig.mind"
    grep -v '^import ' "$ROOT/std/x509_path.mind"
    grep -v '^import ' "$ROOT/std/x509.mind"
    cat "$HERE/native_head.mind"
    python3 - "$HERE/testdata/cases_native.bin" "$HERE/testdata/bundle_anchors.der" "$HERE/testdata/bundle_anchors.pem" "$HERE/testdata/bundle_bad.pem" <<'PY'
import pathlib, sys
print('let cases_hex: String = "' + pathlib.Path(sys.argv[1]).read_bytes().hex() + '";')
print('let bundle_der_hex: String = "' + pathlib.Path(sys.argv[2]).read_bytes().hex() + '";')
print('let bundle_pem_hex: String = "' + pathlib.Path(sys.argv[3]).read_bytes().hex() + '";')
print('let bundle_bad_hex: String = "' + pathlib.Path(sys.argv[4]).read_bytes().hex() + '";')
PY
    cat "$HERE/native_body.mind"
} > "$WORK/src/main.mind"

cat > "$WORK/Mind.toml" <<'TOML'
[package]
name = "x509native"
version = "0.1.0"

[build]
entry = "src/main.mind"

[targets.cpu]
backend = "cpu"
sources = ["src/main.mind"]
TOML
( cd "$WORK" && "$MINDC" build --backend native --no-cache )
BIN="$WORK/target/debug/x509native"
"$BIN"
rc=$?
echo "native harness exit=$rc"
exit $rc
}
x509_gen_fixtures() {
# Regenerate the committed X.509 DER fixtures for std/x509.mind chain-validation
# tests.  Purely local: openssl CLI + python3 only, no network.
#
# Layout produced under tests/x509/testdata/ (all certs valid 10 years from
# generation; tests pin now_unix from testdata/now.txt):
#
#   EC primary chain (ECDSA P-256 / SHA-256):
#     ec_root.der  (self-signed CA, pathlen:2)
#     ec_inter.der (CA, pathlen:0, signed by ec_root)
#     ec_leaf.der  (SAN www.example.com, EKU serverAuth, signed by ec_inter)
#     ec_leaf_badsig.der   (leaf with one signature byte flipped)
#     ec_leaf_trunc.der    (leaf truncated to 100 bytes -> malformed)
#     ec_leaf_noeku.der    (leaf without EKU)
#     ec_leaf_nosan.der    (leaf with CN only, no SAN)
#     ec_leaf_crit.der     (leaf with unknown CRITICAL extension 1.2.3.4)
#     ec_leaf_wild.der     (SAN *.example.com + A-label xn--nxasmq6b.example.com)
#     ec_leaf_midwild.der  (SAN sub.*.example.com - wildcard not in leftmost label)
#     ec_leaf_badku.der    (leaf with keyUsage keyAgreement, missing digitalSignature)
#     ec_leaf_ip.der       (SAN IP:127.0.0.1)
#   Path-length chain:
#     pl_root.der (CA pathlen:2), pl_i1.der (CA pathlen:0), pl_i2.der (CA),
#     pl_leaf.der
#   Non-CA intermediate chain: noca_root.der, noca_inter.der (CA:FALSE),
#     noca_leaf.der
#   RSA chain (sha256WithRSAEncryption): rsa_root.der, rsa_leaf.der
#   RSA-PSS chain (RSASSA-PSS sha256, saltlen 32): pss_root.der, pss_leaf.der
#   EC P-384 chain: p384_root.der, p384_leaf.der
#   Name Constraints chain: nc_root.der, nc_inter.der, nc_leaf_ok.der, nc_leaf_bad.der
#   SHA-1 chain: sha1_root.der, sha1_leaf.der (refused)
#   In-memory bundles: bundle_anchors.pem, bundle_anchors.der, bundle_bad.pem
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/tests/x509"
TD=testdata
rm -rf "$TD"
mkdir -p "$TD"

# Private keys must NEVER be committed; generate in an ephemeral dir outside repo
TMP_DIR="$(mktemp -d /tmp/x509_gen_XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

DAYS=3650

ec_key() { openssl ecparam -genkey -name P-256 -out "$TMP_DIR/$1.key"; }
csr()    { openssl req -new -key "$TMP_DIR/$1.key" -subj "$2" -out "$TMP_DIR/$1.csr"; }
sign()   { # name cacert cakey extfile
    openssl x509 -req -in "$TMP_DIR/$1.csr" -CA "$TMP_DIR/$2.pem" -CAkey "$TMP_DIR/$3.key" \
        -set_serial "$(openssl rand -hex 8 | tr a-f A-F | python3 -c 'import sys;print(int(sys.stdin.read().strip(),16))')" \
        -sha256 -days "$DAYS" -extfile "$TMP_DIR/$4.ext" -out "$TMP_DIR/$1.pem"
}
der()    { openssl x509 -in "$TMP_DIR/$1.pem" -outform DER -out "$TD/$1.der"; }

# --- EC root (self-signed, pathlen:2) ---
ec_key ec_root
openssl req -x509 -new -key "$TMP_DIR/ec_root.key" -sha256 -days $DAYS \
    -subj "/CN=MindTest EC Root CA" \
    -addext "basicConstraints=critical,CA:TRUE,pathlen:2" \
    -addext "keyUsage=critical,keyCertSign,cRLSign" \
    -out "$TMP_DIR/ec_root.pem"
der ec_root

# --- EC intermediate (pathlen:0) ---
ec_key ec_inter
csr ec_inter "/CN=MindTest EC Intermediate CA"
cat > "$TMP_DIR/ec_inter.ext" <<'EOF'
basicConstraints=critical,CA:TRUE,pathlen:0
keyUsage=critical,keyCertSign,cRLSign
EOF
sign ec_inter ec_root ec_root ec_inter
der ec_inter

leaf_by() { # name parent ext-lines...
    local name="$1" parent="$2"
    shift 2
    ec_key "$name"
    csr "$name" "/CN=${name}"
    {
        echo "basicConstraints=critical,CA:FALSE"
        echo "keyUsage=critical,digitalSignature"
        for l in "$@"; do echo "$l"; done
    } > "$TMP_DIR/${name}.ext"
    sign "$name" "$parent" "$parent" "$name"
    der "$name"
}

leaf_by ec_leaf ec_inter \
    "extendedKeyUsage=critical,serverAuth" \
    "subjectAltName=DNS:www.example.com"

leaf_by ec_leaf_noeku ec_inter \
    "subjectAltName=DNS:www.example.com"

leaf_by ec_leaf_nosan ec_inter \
    "extendedKeyUsage=critical,serverAuth"

leaf_by ec_leaf_crit ec_inter \
    "extendedKeyUsage=critical,serverAuth" \
    "subjectAltName=DNS:www.example.com" \
    "1.2.3.4=critical,DER:05:00"

leaf_by ec_leaf_wild ec_inter \
    "extendedKeyUsage=critical,serverAuth" \
    "subjectAltName=DNS:*.wild.example.com,DNS:xn--nxasmq6b.example.com"

leaf_by ec_leaf_midwild ec_inter \
    "extendedKeyUsage=critical,serverAuth" \
    "subjectAltName=DNS:sub.*.example.com"

# ec_leaf_badku: keyUsage lacks digitalSignature
ec_key ec_leaf_badku
csr ec_leaf_badku "/CN=ec_leaf_badku"
cat > "$TMP_DIR/ec_leaf_badku.ext" <<'EOF'
basicConstraints=critical,CA:FALSE
keyUsage=critical,keyAgreement
extendedKeyUsage=critical,serverAuth
subjectAltName=DNS:www.example.com
EOF
sign ec_leaf_badku ec_inter ec_inter ec_leaf_badku
der ec_leaf_badku

# ec_leaf_ip: SAN iPAddress = 127.0.0.1
leaf_by ec_leaf_ip ec_inter \
    "extendedKeyUsage=critical,serverAuth" \
    "subjectAltName=IP:127.0.0.1"

# --- Path-length violation chain: root(pathlen:2) -> i1(pathlen:0) -> i2 -> leaf
ec_key pl_root
openssl req -x509 -new -key "$TMP_DIR/pl_root.key" -sha256 -days $DAYS \
    -subj "/CN=MindTest PL Root CA" \
    -addext "basicConstraints=critical,CA:TRUE,pathlen:2" \
    -addext "keyUsage=critical,keyCertSign,cRLSign" \
    -out "$TMP_DIR/pl_root.pem"
der pl_root

ec_key pl_i1
csr pl_i1 "/CN=MindTest PL Intermediate 1"
cat > "$TMP_DIR/pl_i1.ext" <<'EOF'
basicConstraints=critical,CA:TRUE,pathlen:0
keyUsage=critical,keyCertSign,cRLSign
EOF
sign pl_i1 pl_root pl_root pl_i1
der pl_i1

ec_key pl_i2
csr pl_i2 "/CN=MindTest PL Intermediate 2"
cat > "$TMP_DIR/pl_i2.ext" <<'EOF'
basicConstraints=critical,CA:TRUE
keyUsage=critical,keyCertSign,cRLSign
EOF
sign pl_i2 pl_i1 pl_i1 pl_i2
der pl_i2

leaf_by pl_leaf pl_i2 \
    "extendedKeyUsage=critical,serverAuth" \
    "subjectAltName=DNS:pl.example.com"

# --- Non-CA intermediate chain ---
ec_key noca_root
openssl req -x509 -new -key "$TMP_DIR/noca_root.key" -sha256 -days $DAYS \
    -subj "/CN=MindTest NOCA Root CA" \
    -addext "basicConstraints=critical,CA:TRUE" \
    -addext "keyUsage=critical,keyCertSign,cRLSign" \
    -out "$TMP_DIR/noca_root.pem"
der noca_root

ec_key noca_inter
csr noca_inter "/CN=MindTest NOCA Intermediate"
cat > "$TMP_DIR/noca_inter.ext" <<'EOF'
basicConstraints=critical,CA:FALSE
keyUsage=critical,digitalSignature
EOF
sign noca_inter noca_root noca_root noca_inter
der noca_inter

leaf_by noca_leaf noca_inter \
    "extendedKeyUsage=critical,serverAuth" \
    "subjectAltName=DNS:noca.example.com"

# --- RSA PKCS#1 v1.5 chain ---
openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:2048 -out "$TMP_DIR/rsa_root.key"
openssl req -x509 -new -key "$TMP_DIR/rsa_root.key" -sha256 -days $DAYS \
    -subj "/CN=MindTest RSA Root CA" \
    -addext "basicConstraints=critical,CA:TRUE" \
    -addext "keyUsage=critical,keyCertSign,cRLSign" \
    -out "$TMP_DIR/rsa_root.pem"
der rsa_root

openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:2048 -out "$TMP_DIR/rsa_leaf.key"
csr rsa_leaf "/CN=rsa leaf"
cat > "$TMP_DIR/rsa_leaf.ext" <<'EOF'
basicConstraints=critical,CA:FALSE
keyUsage=critical,digitalSignature,keyEncipherment
extendedKeyUsage=critical,serverAuth
subjectAltName=DNS:www.rsa-test.example.com
EOF
sign rsa_leaf rsa_root rsa_root rsa_leaf
der rsa_leaf

# --- RSA-PSS chain (root self-signed PSS, leaf signed PSS saltlen=digest) ---
openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:2048 -out "$TMP_DIR/pss_root.key"
openssl req -x509 -new -key "$TMP_DIR/pss_root.key" -sha256 -days $DAYS \
    -subj "/CN=MindTest PSS Root CA" \
    -addext "basicConstraints=critical,CA:TRUE" \
    -addext "keyUsage=critical,keyCertSign,cRLSign" \
    -sigopt rsa_padding_mode:pss -sigopt rsa_pss_saltlen:digest \
    -out "$TMP_DIR/pss_root.pem"
der pss_root

openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:2048 -out "$TMP_DIR/pss_leaf.key"
csr pss_leaf "/CN=pss leaf"
cat > "$TMP_DIR/pss_leaf.ext" <<'EOF'
basicConstraints=critical,CA:FALSE
keyUsage=critical,digitalSignature,keyEncipherment
extendedKeyUsage=critical,serverAuth
subjectAltName=DNS:www.pss-test.example.com
EOF
openssl x509 -req -in "$TMP_DIR/pss_leaf.csr" -CA "$TMP_DIR/pss_root.pem" -CAkey "$TMP_DIR/pss_root.key" \
    -set_serial 424242 -sha256 -days $DAYS \
    -sigopt rsa_padding_mode:pss -sigopt rsa_pss_saltlen:digest \
    -extfile "$TMP_DIR/pss_leaf.ext" -out "$TMP_DIR/pss_leaf.pem"
der pss_leaf

# --- EC P-384 chain ---
openssl ecparam -genkey -name secp384r1 -out "$TMP_DIR/p384_root.key"
openssl req -x509 -new -key "$TMP_DIR/p384_root.key" -sha384 -days $DAYS \
    -subj "/CN=MindTest P384 Root CA" \
    -addext "basicConstraints=critical,CA:TRUE" \
    -addext "keyUsage=critical,keyCertSign,cRLSign" \
    -out "$TMP_DIR/p384_root.pem"
der p384_root

openssl ecparam -genkey -name secp384r1 -out "$TMP_DIR/p384_leaf.key"
openssl req -new -key "$TMP_DIR/p384_leaf.key" -subj "/CN=p384 leaf" -out "$TMP_DIR/p384_leaf.csr"
cat > "$TMP_DIR/p384_leaf.ext" <<'EOF'
basicConstraints=critical,CA:FALSE
keyUsage=critical,digitalSignature
extendedKeyUsage=critical,serverAuth
subjectAltName=DNS:www.p384-test.example.com
EOF
openssl x509 -req -in "$TMP_DIR/p384_leaf.csr" -CA "$TMP_DIR/p384_root.pem" -CAkey "$TMP_DIR/p384_root.key" \
    -set_serial 384384 -sha384 -days $DAYS \
    -extfile "$TMP_DIR/p384_leaf.ext" -out "$TMP_DIR/p384_leaf.pem"
der p384_leaf

# --- Name Constraints chain ---
ec_key nc_root
openssl req -x509 -new -key "$TMP_DIR/nc_root.key" -sha256 -days $DAYS \
    -subj "/CN=MindTest NC Root CA" \
    -addext "basicConstraints=critical,CA:TRUE" \
    -addext "keyUsage=critical,keyCertSign,cRLSign" \
    -out "$TMP_DIR/nc_root.pem"
der nc_root

ec_key nc_inter
csr nc_inter "/CN=MindTest NC Intermediate CA"
cat > "$TMP_DIR/nc_inter.ext" <<'EOF'
basicConstraints=critical,CA:TRUE
keyUsage=critical,keyCertSign,cRLSign
nameConstraints=critical,permitted;DNS:.example.com,excluded;DNS:bad.example.com
EOF
sign nc_inter nc_root nc_root nc_inter
der nc_inter

leaf_by nc_leaf_ok nc_inter \
    "extendedKeyUsage=critical,serverAuth" \
    "subjectAltName=DNS:good.example.com"

leaf_by nc_leaf_bad nc_inter \
    "extendedKeyUsage=critical,serverAuth" \
    "subjectAltName=DNS:bad.example.com"

# --- SHA-1 chain (refused) ---
openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:2048 -out "$TMP_DIR/sha1_root.key"
openssl req -x509 -new -key "$TMP_DIR/sha1_root.key" -sha1 -days $DAYS \
    -subj "/CN=MindTest SHA1 Root CA" \
    -addext "basicConstraints=critical,CA:TRUE" \
    -addext "keyUsage=critical,keyCertSign,cRLSign" \
    -out "$TMP_DIR/sha1_root.pem"
der sha1_root

openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:2048 -out "$TMP_DIR/sha1_leaf.key"
openssl req -new -key "$TMP_DIR/sha1_leaf.key" -subj "/CN=sha1 leaf" -out "$TMP_DIR/sha1_leaf.csr"
cat > "$TMP_DIR/sha1_leaf.ext" <<'EOF'
basicConstraints=critical,CA:FALSE
keyUsage=critical,digitalSignature
extendedKeyUsage=critical,serverAuth
subjectAltName=DNS:www.sha1-test.example.com
EOF
openssl x509 -req -in "$TMP_DIR/sha1_leaf.csr" -CA "$TMP_DIR/sha1_root.pem" -CAkey "$TMP_DIR/sha1_root.key" \
    -set_serial 111111 -sha1 -days $DAYS \
    -extfile "$TMP_DIR/sha1_leaf.ext" -out "$TMP_DIR/sha1_leaf.pem"
der sha1_leaf

# --- Bundles for in-memory anchor loader ---
cat "$TMP_DIR/ec_root.pem" "$TMP_DIR/rsa_root.pem" "$TMP_DIR/p384_root.pem" > "$TD/bundle_anchors.pem"
cat "$TD/ec_root.der" "$TD/rsa_root.der" "$TD/p384_root.der" > "$TD/bundle_anchors.der"
cat "$TMP_DIR/ec_root.pem" > "$TD/bundle_bad.pem"
echo "--- CORRUPTED BASE64 LINE ---" >> "$TD/bundle_bad.pem"
echo "!!NOT_BASE64_AT_ALL!!" >> "$TD/bundle_bad.pem"

# --- derived fixtures: broken signature + truncation ---
python3 - <<'PY'
import pathlib
td = pathlib.Path("testdata")
leaf = (td / "ec_leaf.der").read_bytes()
b = bytearray(leaf)
b[-1] ^= 0x55  # flip bits in the last byte (inside signatureValue)
(td / "ec_leaf_badsig.der").write_bytes(bytes(b))
(td / "ec_leaf_trunc.der").write_bytes(leaf[:100])

# now.txt: a timestamp comfortably inside every fixture's validity window
import datetime, cryptography.x509 as cX
c = cX.load_der_x509_certificate(leaf)
nb = c.not_valid_before_utc
now = int((nb + datetime.timedelta(days=2)).timestamp())
(td / "now.txt").write_text(str(now) + "\n")
PY

# sanity: the good chains must verify with openssl itself
openssl verify -CAfile "$TMP_DIR/ec_root.pem" -untrusted "$TMP_DIR/ec_inter.pem" -verify_hostname www.example.com -purpose sslserver "$TMP_DIR/ec_leaf.pem" >/dev/null
openssl verify -CAfile "$TMP_DIR/rsa_root.pem" "$TMP_DIR/rsa_leaf.pem" >/dev/null
openssl verify -CAfile "$TMP_DIR/pss_root.pem" "$TMP_DIR/pss_leaf.pem" >/dev/null
openssl verify -CAfile "$TMP_DIR/p384_root.pem" "$TMP_DIR/p384_leaf.pem" >/dev/null

echo "fixtures written to $TD"
ls -la "$TD"

python3 "$ROOT/tests/x509_vectors_driver.py" gen-path
python3 "$ROOT/tests/x509_vectors_driver.py" gen-malformed
python3 "$ROOT/tests/x509_vectors_driver.py" gen-manifest
}

# Folded x509 transports. No argument keeps the crypto-vector gate.
case "${1:-}" in
  x509-build-shared)
    shift
    x509_build_shared "$@"
    exit $?
    ;;
  x509-build-native)
    shift
    x509_build_native "$@"
    exit $?
    ;;
  x509-gen-fixtures)
    shift
    x509_gen_fixtures "$@"
    exit $?
    ;;
esac


MINDC="${MINDC:-./target/release/mindc}"
STD="std"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

if [ -n "${PYREFS:-}" ]; then
  export PYTHONPATH="${PYREFS}:${PYTHONPATH:-}"
fi

if [ ! -x "$MINDC" ]; then
  echo "FATAL: mindc not found or not executable at: $MINDC" >&2
  exit 1
fi

# emit <out.so> <src.mind>... — concatenate sources (already import-stripped by
# the caller) into one module and emit a shared object. Fail-closed: a mindc
# emit error aborts the whole run.
emit() {
  local out="$1"; shift
  cat "$@" > "$WORK/combined.mind"
  echo ">>> mindc --emit-shared $out  (<= $*)"
  "$MINDC" "$WORK/combined.mind" --emit-shared "$out"
}

# strip <file> <import-regex> — echo a std source with matching import lines
# removed, into the work dir; echoes the produced path.
strip() {
  local src="$1" re="$2" base
  base="$(basename "$src").$RANDOM.stripped"
  grep -vE "$re" "$src" > "$WORK/$base"
  echo "$WORK/$base"
}

RESULTS=()
RAN_DRIVERS=()
run_driver() {
  local name="$1"; shift
  echo "======================================================================"
  echo "DRIVER: $name"
  echo "======================================================================"
  local log="$WORK/$name.out" rc=0
  python3 "$@" > "$log" 2>&1 || rc=$?
  cat "$log"
  RAN_DRIVERS+=("$(basename "$1")")
  # A driver's exit code says its process ended well, not that it checked
  # anything. Every driver ends with `SUMMARY: <passed>/<total> checks PASSED`;
  # a driver whose vector table is empty prints `0/0 checks PASSED` and exits 0,
  # which the old aggregation recorded as a PASS.
  #
  # CAPTURE WITHOUT ABORTING. The previous form was
  #     checks="$(grep -oE '...' "$log" | tail -1 | sed -E '...')"
  # and under `set -euo pipefail` a driver that prints NO summary makes grep exit
  # 1, pipefail propagates it, the assignment fails and the script dies right
  # here -- before reaching the branch that names the refusal. The one case the
  # check exists for produced an opaque exit instead of a diagnostic. The `|| true`
  # below is confined to this capture; strict mode stays on everywhere else.
  #
  # The pattern requires the literal `PASSED`, so a `... checks FAILED` summary is
  # not silently accepted as a count.
  local summaries="" n_sum=0 checks=0 passed=0 count_status=""
  # Count all summary records before validating their content. Otherwise a
  # malformed or FAILED record followed by a good one would disappear here.
  summaries="$(grep -E '^SUMMARY:' "$log" || true)"
  if [ -n "$summaries" ]; then
    n_sum="$(printf '%s\n' "$summaries" | grep -c . || true)"
  fi

  if [ "$rc" -ne 0 ]; then
    RESULTS+=("FAIL  $name — driver process exited $rc; see $log")
  elif [ "$n_sum" -eq 0 ]; then
    # Covers absent AND malformed: anything not matching the exact contract is
    # not a summary. A driver that prints only "PASS" reports no count and is
    # refused here rather than credited.
    RESULTS+=("FAIL  $name — no well-formed 'SUMMARY: <passed>/<total> checks PASSED' line; the driver never reported how many known-answer checks it ran")
  elif [ "$n_sum" -gt 1 ]; then
    # Ambiguous: refuse rather than pick one. Taking the last would let a driver
    # append a clean summary after a bad one and be credited for it.
    RESULTS+=("FAIL  $name — $n_sum SUMMARY lines; ambiguous, refusing to choose between them")
  elif [[ ! "$summaries" =~ ^SUMMARY:\ ([0-9]+)/([0-9]+)\ checks\ PASSED$ ]]; then
    RESULTS+=("FAIL  $name — malformed SUMMARY; expected exactly 'SUMMARY: <passed>/<total> checks PASSED'")
  else
    passed="${BASH_REMATCH[1]}"
    checks="${BASH_REMATCH[2]}"
    # Python already executes every driver. Its integer comparison cannot
    # overflow like shell test(1), whose error status in an if/elif can fall
    # through to PASS for an oversized counter. Conversion failures refuse.
    count_status="$(python3 - "$passed" "$checks" <<'PY'
import sys
try:
    passed, checks = map(int, sys.argv[1:])
except ValueError:
    print("invalid")
else:
    print("empty" if checks < 1 else "partial" if passed != checks else "pass")
PY
    )"
    if [ "$count_status" = empty ]; then
      RESULTS+=("FAIL  $name — 0 checks; a driver with an empty vector table asserts nothing")
    elif [ "$count_status" = partial ]; then
      RESULTS+=("FAIL  $name — only $passed of $checks checks passed")
    elif [ "$count_status" = pass ]; then
      RESULTS+=("PASS  $name ($checks checks)")
    else
      RESULTS+=("FAIL  $name — invalid SUMMARY counters")
    fi
  fi
}

# ---- single-module builds (no imports) ------------------------------------
emit "$WORK/aes_gcm.so"      "$STD/aes_gcm.mind"
emit "$WORK/keccak.so"       "$STD/keccak.mind"
emit "$WORK/hpack.so"        "$STD/hpack.mind"
emit "$WORK/http2_frame.so"  "$STD/http2_frame.mind"
emit "$WORK/x25519.so"       "$STD/x25519.mind"

# ---- hkdf = sha256 + hkdf(-sha256) ----------------------------------------
emit "$WORK/hkdf.so" "$STD/sha256.mind" "$(strip "$STD/hkdf.mind" '^import std\.sha256;')"

# ---- x509 = sha256 + the chain validator's module set (-imports) ----------
# std/x509.mind imports sha512, ecdsa_p256, x509_p384, rsa_pss and the x509_*
# helpers (x509_bignum holds the RSA bignum that rsa_pss and x509_sig share;
# rsa_pss in turn imports it); the same import-stripped set as
# x509-build-shared above. rsa_pss is part of the set, so rsa_pss.so and
# tls13_hs.so reuse X509_SET below.
X509_SET=()
for m in sha512 ecdsa_p256 x509_p384 x509_bignum rsa_pss x509_bundle x509_der \
         x509_ext x509_name x509_sig x509_path x509; do
  X509_SET+=("$(strip "$STD/$m.mind" '^(pub )?import ')")
done
emit "$WORK/x509.so" "$STD/sha256.mind" "${X509_SET[@]}"

# ---- ecdsa_p256 = sha256 + sha512 + ecdsa_p256(-imports) ------------------
emit "$WORK/ecdsa_p256.so" "$STD/sha256.mind" "$STD/sha512.mind" "$(strip "$STD/ecdsa_p256.mind" '^import ')"

# ---- mlkem768 = keccak + mlkem768(-keccak) --------------------------------
emit "$WORK/mlkem768.so" "$STD/keccak.mind" "$(strip "$STD/mlkem768.mind" '^import std\.keccak;')"

# ---- x25519mlkem768 = keccak + mlkem768(-keccak) + x25519 + hybrid(-imports) --
# The PQC hybrid (draft-kwiatkowski-tls-ecdhe-mlkem, NamedGroup 0x11EC). Its
# known-answer driver has been complete in tests/ all along and was never
# invoked: 14 drivers on disk, 13 run, and EXPECTED_DRIVERS was hand-set to 13.
emit "$WORK/x25519mlkem768.so" "$STD/keccak.mind" \
  "$(strip "$STD/mlkem768.mind" '^import std\.keccak;')" \
  "$STD/x25519.mind" \
  "$(strip "$STD/x25519mlkem768.mind" '^import ')"

# ---- rsa_pss = sha256 + X509_SET (rsa_pss included) ------------------------
emit "$WORK/rsa_pss.so" "$STD/sha256.mind" "${X509_SET[@]}"

# ---- tls13_keyschedule = sha256 + hkdf(-sha256) + keyschedule(-sha256,-hkdf)
emit "$WORK/tls13_ks.so" "$STD/sha256.mind" \
  "$(strip "$STD/hkdf.mind" '^import std\.sha256;')" \
  "$(strip "$STD/tls13_keyschedule.mind" '^import std\.(sha256|hkdf);')"

# ---- tls13_record = sha256 + hkdf(-sha256) + keyschedule(-sha256,-hkdf)
#                     + aes_gcm + tls13_record(-std) -------------------------
emit "$WORK/tls13_rec.so" "$STD/sha256.mind" \
  "$(strip "$STD/hkdf.mind" '^import std\.sha256;')" \
  "$(strip "$STD/tls13_keyschedule.mind" '^import std\.(sha256|hkdf);')" \
  "$STD/aes_gcm.mind" \
  "$(strip "$STD/tls13_record.mind" '^import std\.')"

# ---- tls13_finished = sha256 + hkdf(-sha256) + keyschedule(-sha256,-hkdf)
#                       + tls13_finished(-std) ------------------------------
emit "$WORK/tls13_fin.so" "$STD/sha256.mind" \
  "$(strip "$STD/hkdf.mind" '^import std\.sha256;')" \
  "$(strip "$STD/tls13_keyschedule.mind" '^import std\.(sha256|hkdf);')" \
  "$(strip "$STD/tls13_finished.mind" '^import std\.')"

# ---- tls13_handshake = sha256 + hkdf(-sha256) + X509_SET (rsa_pss included)
#      + keyschedule(-sha256,-hkdf) + aes_gcm(-std) + tls13_record(-std)
#      + tls13_finished(-std) + x25519(-std) + tls13_handshake(-std) -------
emit "$WORK/tls13_hs.so" "$STD/sha256.mind" \
  "$(strip "$STD/hkdf.mind" '^import std\.sha256;')" \
  "${X509_SET[@]}" \
  "$(strip "$STD/tls13_keyschedule.mind" '^import std\.(sha256|hkdf);')" \
  "$(strip "$STD/aes_gcm.mind" '^import std\.')" \
  "$(strip "$STD/tls13_record.mind" '^import std\.')" \
  "$(strip "$STD/tls13_finished.mind" '^import std\.')" \
  "$(strip "$STD/x25519.mind" '^import std\.')" \
  "$(strip "$STD/tls13_handshake.mind" '^(pub )?import std\.')"

# ---- run every driver, exit code gates the run ----------------------------
run_driver keccak            tests/keccak_driver.py            "$WORK/keccak.so"
run_driver hpack             tests/hpack_driver.py             "$WORK/hpack.so"
run_driver http2_frame       tests/http2_frame_driver.py       "$WORK/http2_frame.so"
run_driver x25519_vectors    tests/x25519_vectors_driver.py    "$WORK/x25519.so"
run_driver crypto_vectors    tests/crypto_vectors_driver.py    "$WORK/aes_gcm.so" "$WORK/hkdf.so"
run_driver x509_vectors      tests/x509_vectors_driver.py      "$WORK/x509.so"
run_driver ecdsa_p256        tests/ecdsa_p256_driver.py        "$WORK/ecdsa_p256.so"
run_driver mlkem768          tests/mlkem768_driver.py          "$WORK/mlkem768.so"
run_driver x25519mlkem768    tests/x25519mlkem768_driver.py    "$WORK/x25519mlkem768.so"
run_driver rsa_pss           tests/rsa_pss_driver.py           "$WORK/rsa_pss.so"
run_driver tls13_keyschedule tests/tls13_keyschedule_driver.py "$WORK/tls13_ks.so"
run_driver tls13_record      tests/tls13_record_driver.py      "$WORK/tls13_rec.so"
run_driver tls13_finished    tests/tls13_finished_driver.py    "$WORK/tls13_fin.so"
run_driver tls13_handshake   tests/tls13_handshake_driver.py   "$WORK/tls13_hs.so"
# The client driver composes its own module set and builds with $MINDC.
MINDC="$MINDC" run_driver tls13_client tests/tls13_handshake_driver.py client

echo "======================================================================"
echo "CRYPTO-VECTOR SUMMARY"
echo "======================================================================"

# Count floor, DERIVED FROM THE TREE.
#
# The floor used to be `EXPECTED_DRIVERS=13`, hand-maintained, with a comment
# saying to raise it when a driver is added. Nobody did, and the number was equal
# to how many drivers the script HAPPENED to invoke -- so it could never catch the
# gap it existed to catch. Measured: 14 `tests/*_driver.py` on disk, 13 invoked.
# The unrun one was x25519mlkem768, a complete known-answer driver for the PQC
# hybrid key exchange, silently absent from every green run.
#
# A count set to the length of the list it is checking is not a check. The
# expectation now comes from the filesystem, and every driver present must
# actually have run -- named, so the failure says which.
DRIVERS_ON_DISK=()
while IFS= read -r f; do
  DRIVERS_ON_DISK+=("$(basename "$f")")
done < <(ls tests/*_driver.py 2>/dev/null | sort)
EXPECTED_DRIVERS="${#DRIVERS_ON_DISK[@]}"

if [ "$EXPECTED_DRIVERS" -lt 1 ]; then
  echo "RESULT: RED — found NO tests/*_driver.py on disk. Either the corpus is gone"
  echo "        or this script is running from the wrong directory; a floor derived"
  echo "        from an empty search would make every run trivially green."
  exit 1
fi

missing=()
for d in "${DRIVERS_ON_DISK[@]}"; do
  seen=0
  for r in "${RAN_DRIVERS[@]}"; do
    [ "$r" = "$d" ] && { seen=1; break; }
  done
  [ "$seen" -eq 0 ] && missing+=("$d")
done
if [ "${#missing[@]}" -gt 0 ]; then
  echo "RESULT: RED — ${#missing[@]} known-answer driver(s) exist on disk but were never run:"
  for m in "${missing[@]}"; do echo "        $m"; done
  echo "        A vector gate that skips a driver is green about vectors it never checked."
  exit 1
fi

if [ "${#RESULTS[@]}" -lt "$EXPECTED_DRIVERS" ]; then
  echo "RESULT: RED — only ${#RESULTS[@]} crypto driver(s) ran, expected $EXPECTED_DRIVERS"
  echo "        (derived from tests/*_driver.py). A vector gate that ran no vectors"
  echo "        asserts nothing."
  exit 1
fi

fail=0
for r in "${RESULTS[@]}"; do
  echo "  $r"
  [[ "$r" == FAIL* ]] && fail=1
done
if [ "$fail" -ne 0 ]; then
  echo "RESULT: RED — at least one crypto driver failed"
  exit 1
fi
echo "RESULT: GREEN — all ${#RESULTS[@]} crypto drivers passed"
