#!/usr/bin/env bash
# Build std/x509.mind (+ its std crypto dependencies) as a shared library via
# mindc --emit-shared (MLIR route).  Cross-module calls between std modules are
# resolved by concatenation (the std imports inside bundled mode only cover the
# seeded/bundled subset listed in examples/mindc_mind/testdata/stdlib_manifest.txt),
# exactly as tests/rsa_pss_driver.py and tests/ecdsa_p256_driver.py document.
#
#   Usage: tests/x509/build_shared.sh <output.so> [mindc]
set -euo pipefail
OUT="${1:?usage: build_shared.sh <out.so> [mindc]}"
MINDC="${2:-/home/n/mind-targets/std-x509/release/mindc}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
COMBINED="${OUT%.so}.combined.mind"
{
    grep -v '^import ' "$ROOT/std/sha256.mind"
    grep -v '^import ' "$ROOT/std/sha512.mind"
    grep -v '^import ' "$ROOT/std/ecdsa_p256.mind"
    grep -v '^import ' "$ROOT/std/x509.mind"
    grep -v '^import ' "$ROOT/std/rsa_pss.mind"
} > "$COMBINED"
"$MINDC" "$COMBINED" --emit-shared "$OUT"
