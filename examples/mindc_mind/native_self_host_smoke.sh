#!/usr/bin/env bash
# native_self_host_smoke.sh — the NATIVE BOOTSTRAP FIXED POINT gate (#14).
#
# Proves the native-ELF-compiled MIND compiler reproduces the canonical binary IR
# of its OWN self-host source, with ZERO LLVM/MLIR/clang/libc in the compile path:
#
#   1. build the `mind-native` backend bin (--features native-backend)
#   2. driver = examples/mindc_mind/main.mind + the appended native self-host
#      driver fn main (self_host_native_driver_main.frag.mind): read stdin ->
#      selftest_mic3_module_nfn -> print mic@3 bytes
#   3. compile the driver to a native static ELF via mind-native (no LLVM)
#   4. run it with main.mind on stdin  -> native mic@3
#   5. oracle = `mindc main.mind --emit-mic3`  -> canonical mic@3
#   6. assert byte-identical (the fixed point) + the native ELF is byte-reproducible
#
# Usage:  examples/mindc_mind/native_self_host_smoke.sh   (from repo root)
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT" || exit 2
TGT="${CARGO_TARGET_DIR:-/tmp/mind-native-target}"
MAIN="examples/mindc_mind/main.mind"
FRAG="examples/mindc_mind/self_host_native_driver_main.frag.mind"
DRV="/tmp/nshs_driver_$$.mind"; ELF="/tmp/nshs_driver_$$.elf"
NAT="/tmp/nshs_native_$$.bin"; ORC="/tmp/nshs_oracle_$$.bin"
t0=$(date +%s); fail=0
[ -f "$FRAG" ] || { echo "ERROR: $FRAG missing"; exit 2; }

echo "== native self-host fixed-point smoke =="
echo "[build mind-native + the release mindc oracle]"
CARGO_TARGET_DIR="$TGT" cargo build --release --features native-backend --bin mind-native >/tmp/nshs_build.log 2>&1 \
  || { echo "  FAIL  mind-native build"; tail -5 /tmp/nshs_build.log; exit 1; }
NATIVE="$TGT/release/mind-native"; [ -x "$NATIVE" ] || NATIVE="$TGT/debug/mind-native"
[ -x "./target/release/mindc" ] || cargo build --release --no-default-features \
  --features "mlir-build std-surface cross-module-imports" --bin mindc >/tmp/nshs_build.log 2>&1

# driver = canonical main.mind + the appended self-host driver fn main
cat "$MAIN" "$FRAG" > "$DRV"

echo "[compile driver -> native ELF (zero LLVM) x2 for byte-reproducibility]"
"$NATIVE" "$DRV" "$ELF" >/tmp/nshs_c.log 2>&1   || { echo "  FAIL  native compile"; tail -5 /tmp/nshs_c.log; exit 1; }
"$NATIVE" "$DRV" "${ELF}.2" >/tmp/nshs_c.log 2>&1 || { echo "  FAIL  native compile (2nd)"; exit 1; }
if cmp -s "$ELF" "${ELF}.2"; then echo "  PASS  native ELF byte-reproducible ($(stat -c%s "$ELF") B)"; else echo "  FAIL  native ELF NOT byte-reproducible"; fail=1; fi

echo "[run native ELF on main.mind (stdin) + canonical oracle]"
"$ELF" < "$MAIN" > "$NAT" 2>/dev/null || { echo "  FAIL  native ELF run"; exit 1; }
./target/release/mindc "$MAIN" --emit-mic3 "$ORC" >/tmp/nshs_o.log 2>&1 || { echo "  FAIL  oracle emit"; tail -3 /tmp/nshs_o.log; exit 1; }

if cmp -s "$NAT" "$ORC"; then
  echo "  PASS  FIXED POINT: native nfn(main.mind) == mindc --emit-mic3 ($(stat -c%s "$NAT") B, sha=$(sha256sum "$NAT" | cut -c1-16))"
else
  echo "  FAIL  native mic@3 != oracle ($(stat -c%s "$NAT") vs $(stat -c%s "$ORC") B)"; fail=1
fi

rm -f "$DRV" "$ELF" "${ELF}.2" "$NAT" "$ORC"
echo "== $([ $fail -eq 0 ] && echo PASS || echo FAIL) in $(($(date +%s)-t0))s =="
[ "$fail" -eq 0 ]
