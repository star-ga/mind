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
# helpers (rsa_pss in turn imports x509_bignum). rsa_pss is part of the set,
# so rsa_pss.so and tls13_hs.so reuse X509_SET below.
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
