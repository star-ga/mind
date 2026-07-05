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
run_driver() {
  local name="$1"; shift
  echo "======================================================================"
  echo "DRIVER: $name"
  echo "======================================================================"
  if python3 "$@"; then
    RESULTS+=("PASS  $name")
  else
    RESULTS+=("FAIL  $name")
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

# ---- x509 = sha256 + x509(-sha256) ----------------------------------------
emit "$WORK/x509.so" "$STD/sha256.mind" "$(strip "$STD/x509.mind" '^import std\.sha256;')"

# ---- ecdsa_p256 = sha256 + ecdsa_p256(-imports) ---------------------------
emit "$WORK/ecdsa_p256.so" "$STD/sha256.mind" "$(strip "$STD/ecdsa_p256.mind" '^import ')"

# ---- mlkem768 = keccak + mlkem768(-keccak) --------------------------------
emit "$WORK/mlkem768.so" "$STD/keccak.mind" "$(strip "$STD/mlkem768.mind" '^import std\.keccak;')"

# ---- rsa_pss = sha256 + x509(-sha256) + rsa_pss(-imports) ------------------
emit "$WORK/rsa_pss.so" "$STD/sha256.mind" \
  "$(strip "$STD/x509.mind" '^import std\.sha256;')" \
  "$(strip "$STD/rsa_pss.mind" '^import ')"

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

# ---- tls13_handshake = sha256 + hkdf(-sha256) + x509(-sha256)
#      + keyschedule(-sha256,-hkdf) + aes_gcm(-std) + tls13_record(-std)
#      + tls13_finished(-std) + rsa_pss(-std) + x25519(-std)
#      + tls13_handshake(-std) --------------------------------------------
emit "$WORK/tls13_hs.so" "$STD/sha256.mind" \
  "$(strip "$STD/hkdf.mind" '^import std\.sha256;')" \
  "$(strip "$STD/x509.mind" '^import std\.sha256;')" \
  "$(strip "$STD/tls13_keyschedule.mind" '^import std\.(sha256|hkdf);')" \
  "$(strip "$STD/aes_gcm.mind" '^import std\.')" \
  "$(strip "$STD/tls13_record.mind" '^import std\.')" \
  "$(strip "$STD/tls13_finished.mind" '^import std\.')" \
  "$(strip "$STD/rsa_pss.mind" '^import std\.')" \
  "$(strip "$STD/x25519.mind" '^import std\.')" \
  "$(strip "$STD/tls13_handshake.mind" '^import std\.')"

# ---- run every driver, exit code gates the run ----------------------------
run_driver keccak            tests/keccak_driver.py            "$WORK/keccak.so"
run_driver hpack             tests/hpack_driver.py             "$WORK/hpack.so"
run_driver http2_frame       tests/http2_frame_driver.py       "$WORK/http2_frame.so"
run_driver x25519_vectors    tests/x25519_vectors_driver.py    "$WORK/x25519.so"
run_driver crypto_vectors    tests/crypto_vectors_driver.py    "$WORK/aes_gcm.so" "$WORK/hkdf.so"
run_driver x509_vectors      tests/x509_vectors_driver.py      "$WORK/x509.so"
run_driver ecdsa_p256        tests/ecdsa_p256_driver.py        "$WORK/ecdsa_p256.so"
run_driver mlkem768          tests/mlkem768_driver.py          "$WORK/mlkem768.so"
run_driver rsa_pss           tests/rsa_pss_driver.py           "$WORK/rsa_pss.so"
run_driver tls13_keyschedule tests/tls13_keyschedule_driver.py "$WORK/tls13_ks.so"
run_driver tls13_record      tests/tls13_record_driver.py      "$WORK/tls13_rec.so"
run_driver tls13_finished    tests/tls13_finished_driver.py    "$WORK/tls13_fin.so"
run_driver tls13_handshake   tests/tls13_handshake_driver.py   "$WORK/tls13_hs.so"

echo "======================================================================"
echo "CRYPTO-VECTOR SUMMARY"
echo "======================================================================"
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
