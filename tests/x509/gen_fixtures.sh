#!/usr/bin/env bash
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
#   Path-length chain:
#     pl_root.der (CA pathlen:2), pl_i1.der (CA pathlen:0), pl_i2.der (CA),
#     pl_leaf.der
#   Non-CA intermediate chain: noca_root.der, noca_inter.der (CA:FALSE),
#     noca_leaf.der
#   RSA chain (sha256WithRSAEncryption): rsa_root.der, rsa_leaf.der
#   RSA-PSS chain (RSASSA-PSS sha256, saltlen 32): pss_root.der, pss_leaf.der
set -euo pipefail
cd "$(dirname "$0")"
TD=testdata
rm -rf "$TD" tmp
mkdir -p "$TD" tmp

DAYS=3650

ec_key() { openssl ecparam -genkey -name P-256 -out "tmp/$1.key"; }
csr()    { openssl req -new -key "tmp/$1.key" -subj "$2" -out "tmp/$1.csr"; }
sign()   { # name cacert cakey extfile days
    openssl x509 -req -in "tmp/$1.csr" -CA "tmp/$2.pem" -CAkey "tmp/$3.key" \
        -set_serial "$(openssl rand -hex 8 | tr a-f A-F | python3 -c 'import sys;print(int(sys.stdin.read().strip(),16))')" \
        -sha256 -days "$DAYS" -extfile "tmp/$4.ext" -out "tmp/$1.pem"
}
der()    { openssl x509 -in "tmp/$1.pem" -outform DER -out "$TD/$1.der"; }

# --- EC root (self-signed, pathlen:2) ---
ec_key ec_root
openssl req -x509 -new -key tmp/ec_root.key -sha256 -days $DAYS \
    -subj "/CN=MindTest EC Root CA" \
    -addext "basicConstraints=critical,CA:TRUE,pathlen:2" \
    -addext "keyUsage=critical,keyCertSign,cRLSign" \
    -out tmp/ec_root.pem
der ec_root

# --- EC intermediate (pathlen:0) ---
ec_key ec_inter
csr ec_inter "/CN=MindTest EC Intermediate CA"
cat > tmp/ec_inter.ext <<'EOF'
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
    } > "tmp/${name}.ext"
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

# --- Path-length violation chain: root(pathlen:2) -> i1(pathlen:0) -> i2 -> leaf
ec_key pl_root
openssl req -x509 -new -key tmp/pl_root.key -sha256 -days $DAYS \
    -subj "/CN=MindTest PL Root CA" \
    -addext "basicConstraints=critical,CA:TRUE,pathlen:2" \
    -addext "keyUsage=critical,keyCertSign,cRLSign" \
    -out tmp/pl_root.pem
der pl_root

ec_key pl_i1
csr pl_i1 "/CN=MindTest PL Intermediate 1"
cat > tmp/pl_i1.ext <<'EOF'
basicConstraints=critical,CA:TRUE,pathlen:0
keyUsage=critical,keyCertSign,cRLSign
EOF
sign pl_i1 pl_root pl_root pl_i1
der pl_i1

ec_key pl_i2
csr pl_i2 "/CN=MindTest PL Intermediate 2"
cat > tmp/pl_i2.ext <<'EOF'
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
openssl req -x509 -new -key tmp/noca_root.key -sha256 -days $DAYS \
    -subj "/CN=MindTest NOCA Root CA" \
    -addext "basicConstraints=critical,CA:TRUE" \
    -addext "keyUsage=critical,keyCertSign,cRLSign" \
    -out tmp/noca_root.pem
der noca_root

ec_key noca_inter
csr noca_inter "/CN=MindTest NOCA Intermediate"
cat > tmp/noca_inter.ext <<'EOF'
basicConstraints=critical,CA:FALSE
keyUsage=critical,digitalSignature
EOF
sign noca_inter noca_root noca_root noca_inter
der noca_inter

leaf_by noca_leaf noca_inter \
    "extendedKeyUsage=critical,serverAuth" \
    "subjectAltName=DNS:noca.example.com"

# --- RSA PKCS#1 v1.5 chain ---
openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:2048 -out tmp/rsa_root.key
openssl req -x509 -new -key tmp/rsa_root.key -sha256 -days $DAYS \
    -subj "/CN=MindTest RSA Root CA" \
    -addext "basicConstraints=critical,CA:TRUE" \
    -addext "keyUsage=critical,keyCertSign,cRLSign" \
    -out tmp/rsa_root.pem
der rsa_root

openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:2048 -out tmp/rsa_leaf.key
csr rsa_leaf "/CN=rsa leaf"
cat > tmp/rsa_leaf.ext <<'EOF'
basicConstraints=critical,CA:FALSE
keyUsage=critical,digitalSignature,keyEncipherment
extendedKeyUsage=critical,serverAuth
subjectAltName=DNS:www.rsa-test.example.com
EOF
sign rsa_leaf rsa_root rsa_root rsa_leaf
der rsa_leaf

# --- RSA-PSS chain (root self-signed PSS, leaf signed PSS saltlen=digest) ---
openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:2048 -out tmp/pss_root.key
openssl req -x509 -new -key tmp/pss_root.key -sha256 -days $DAYS \
    -subj "/CN=MindTest PSS Root CA" \
    -addext "basicConstraints=critical,CA:TRUE" \
    -addext "keyUsage=critical,keyCertSign,cRLSign" \
    -sigopt rsa_padding_mode:pss -sigopt rsa_pss_saltlen:digest \
    -out tmp/pss_root.pem
der pss_root

openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:2048 -out tmp/pss_leaf.key
csr pss_leaf "/CN=pss leaf"
cat > tmp/pss_leaf.ext <<'EOF'
basicConstraints=critical,CA:FALSE
keyUsage=critical,digitalSignature,keyEncipherment
extendedKeyUsage=critical,serverAuth
subjectAltName=DNS:www.pss-test.example.com
EOF
openssl x509 -req -in tmp/pss_leaf.csr -CA tmp/pss_root.pem -CAkey tmp/pss_root.key \
    -set_serial 424242 -sha256 -days $DAYS \
    -sigopt rsa_padding_mode:pss -sigopt rsa_pss_saltlen:digest \
    -extfile tmp/pss_leaf.ext -out tmp/pss_leaf.pem
der pss_leaf

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
import datetime, cryptography.x509 as cX, cryptography.hazmat.primitives.serialization as ser
c = cX.load_der_x509_certificate(leaf)
nb = c.not_valid_before_utc
now = int((nb + datetime.timedelta(days=2)).timestamp())
(td / "now.txt").write_text(str(now) + "\n")
PY

# sanity: the good chains must verify with openssl itself
openssl verify -CAfile tmp/ec_root.pem -untrusted tmp/ec_inter.pem -verify_hostname www.example.com -purpose sslserver tmp/ec_leaf.pem >/dev/null
openssl verify -CAfile tmp/rsa_root.pem tmp/rsa_leaf.pem >/dev/null
openssl verify -CAfile tmp/pss_root.pem tmp/pss_leaf.pem >/dev/null

rm -f testdata/*.key 2>/dev/null || true
echo "fixtures written to $TD"
ls -la "$TD"
