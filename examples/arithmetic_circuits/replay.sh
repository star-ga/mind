#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")" && pwd)"
mindc="${MINDC:-}"
[[ -n "$mindc" && -x "$mindc" ]] || { echo "CIRCUIT=BLOCKED_MINDC" >&2; exit 2; }
for tool in timeout cmp mktemp head sha256sum; do command -v "$tool" >/dev/null || exit 2; done
tmp="$(mktemp -d)"
trap 'rm -rf -- "$tmp"' EXIT
"$mindc" --version
sha256sum "$mindc" "$root/permanent_2x2.mind"
timeout 120 "$mindc" check "$root/permanent_2x2.mind"
timeout 120 "$mindc" build "$root/permanent_2x2.mind" --backend native --target cpu --emit binary --out "$tmp/circuit.elf"
printf '\177ELF' > "$tmp/magic"
head -c 4 "$tmp/circuit.elf" > "$tmp/actualmagic"
cmp "$tmp/magic" "$tmp/actualmagic"
printf 'EXAMPLE=ARITHMETIC_CIRCUIT\nCASES=3\nNEGATIVE=5\nTHEOREM_AUTHORITY=NONE\n' > "$tmp/expected"
rc=0
timeout 30 "$tmp/circuit.elf" > "$tmp/out" 2> "$tmp/err" || rc=$?
[[ "$rc" -eq 0 && ! -s "$tmp/err" ]] && cmp -s "$tmp/out" "$tmp/expected" || { echo "CIRCUIT=FAIL rc=$rc" >&2; exit 1; }
echo "CIRCUIT=PASS_FINITE_ONLY"
