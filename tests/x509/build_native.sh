#!/usr/bin/env bash
# Build and run the NATIVE-backend harness for std/x509.mind chain validation.
#
# The public cpu native backend has no runtime-IO intrinsics (__mind_open is a
# stub there), so the fixture pack (tests/x509/testdata/cases_native.bin, whose
# layout is documented in gen_native_manifest.py) is embedded as a hex string
# literal in the generated main.  The same concatenated std sources as
# build_shared.sh are prepended (std imports resolve only within the bundled
# subset for single-file compiles).
#
#   Usage: tests/x509/build_native.sh [mindc]
set -euo pipefail
MINDC="${1:-/home/n/mind-targets/std-x509/release/mindc}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
WORK="$ROOT/target/x509-native"
mkdir -p "$WORK/src"

{
    grep -v '^import ' "$ROOT/std/sha256.mind"
    grep -v '^import ' "$ROOT/std/sha512.mind"
    grep -v '^import ' "$ROOT/std/ecdsa_p256.mind"
    grep -v '^import ' "$ROOT/std/x509.mind"
    grep -v '^import ' "$ROOT/std/rsa_pss.mind"
    cat "$HERE/native_head.mind"
    python3 - "$HERE/testdata/cases_native.bin" <<'PY'
import pathlib, sys
data = pathlib.Path(sys.argv[1]).read_bytes()
print('let cases_hex: String = "' + data.hex() + '";')
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
( cd "$WORK" && "$MINDC" build --no-cache )
BIN="$WORK/target/debug/x509native"
"$BIN"
rc=$?
echo "native harness exit=$rc"
exit $rc
