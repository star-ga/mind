#!/usr/bin/env bash
# fast_keystone.sh — fast LOCAL front-end keystone gate for the pure-MIND self-host
# driver (examples/mindc_mind/main.mind).
#
# Proves, in ~1-2 min, that the front-end emit is byte-identical and the cdylib build
# is deterministic — WITHOUT the ~25-minute `cargo test --release phase_g_keystone_bootstrap`
# release-test recompile. This is the per-increment gate U1 runs before committing a
# self-host change; CI still runs the full cargo keystone + cross_substrate.
#
# Checks:
#   1. deterministic build  — Mind.toml-driven cdylib == direct-path cdylib (byte-identical)
#   2. self_host_body_smoke — real-body emitter
#   3. mic3_flip_smoke      — whole-module mic@3 self-host FLIP byte-identical (nfn==--emit-mic3)
#   4. mic3_primitives_smoke— mic@3 codec primitives
#   5. self_host_mlir_smoke — per-construct MLIR emit. NOTE: post-pivot to the native-ELF
#                             self-host backend, MLIR text is a spec-NON-NORMATIVE surface,
#                             so this is a DEBUG AID, not the normative gate. The normative
#                             artifact gate (native-ELF byte-identity + program output-hash
#                             + mic@3 trace_hash) is added separately.
#
# Usage:  examples/mindc_mind/fast_keystone.sh                  # uses target/release/mindc
#         MINDC_REBUILD=1 examples/mindc_mind/fast_keystone.sh  # rebuild mindc first
#         MINDC_BIN=/path/to/mindc examples/mindc_mind/fast_keystone.sh
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT" || exit 2
MINDC="${MINDC_BIN:-$ROOT/target/release/mindc}"
ENTRY="examples/mindc_mind/main.mind"
DIR_SO="/tmp/fk_direct_$$.so"; MT_SO="/tmp/fk_mt_$$.so"
t0=$(date +%s); pass=0; fail=0
chk() { local n="$1"; shift
  if "$@" >/tmp/fk_step.log 2>&1; then echo "  PASS  $n"; pass=$((pass+1))
  else echo "  FAIL  $n"; tail -4 /tmp/fk_step.log | sed 's/^/        /'; fail=$((fail+1)); fi; }

if [ "${MINDC_REBUILD:-0}" = "1" ]; then
  echo "[rebuilding mindc release binary]"
  cargo build --release --no-default-features \
    --features "mlir-build std-surface cross-module-imports" --bin mindc \
    || { echo "ERROR: mindc build failed"; exit 2; }
fi
[ -x "$MINDC" ] || { echo "ERROR: mindc not found at $MINDC (try MINDC_REBUILD=1)"; exit 2; }

echo "== fast front-end keystone =="
# 1. deterministic build: Mind.toml-driven vs direct must be byte-identical
if ! "$MINDC" build "$ENTRY" --release --emit=cdylib --out="$DIR_SO" >/tmp/fk_step.log 2>&1; then
  echo "  FAIL  direct cdylib build"; tail -6 /tmp/fk_step.log | sed 's/^/        /'; exit 1; fi
if ! "$MINDC" build --release --emit=cdylib --out="$MT_SO" >/tmp/fk_step.log 2>&1; then
  echo "  FAIL  Mind.toml cdylib build"; tail -6 /tmp/fk_step.log | sed 's/^/        /'; exit 1; fi
if cmp -s "$DIR_SO" "$MT_SO"; then
  echo "  PASS  byte-identical build ($(stat -c%s "$MT_SO") B, sha=$(sha256sum "$MT_SO" | cut -c1-16))"; pass=$((pass+1))
else
  echo "  FAIL  build NOT byte-identical (Mind.toml-driven != direct-path)"; fail=$((fail+1)); fi

# 2-5. front-end + construct smokes against the built .so
export MINDC_SO="$MT_SO" MINDC="$MINDC" MINDC_BIN="$MINDC"
chk "body_smoke (real-body emit)"          python3 examples/mindc_mind/self_host_body_smoke.py
chk "mic3_flip (whole-module FLIP)"        python3 examples/mindc_mind/mic3_flip_smoke.py
chk "mic3_primitives (codec)"              python3 examples/mindc_mind/mic3_primitives_smoke.py
chk "mlir_smoke (constructs; debug aid)"   python3 examples/mindc_mind/self_host_mlir_smoke.py
chk "mod_operator (% remainder, both paths)" python3 examples/mindc_mind/mod_operator_smoke.py

rm -f "$DIR_SO" "$MT_SO"
echo "== $pass passed, $fail failed in $(($(date +%s)-t0))s =="
[ "$fail" -eq 0 ]
