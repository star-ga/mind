#!/usr/bin/env bash
# Builds and runs the seven bounded examples of this directory with the native
# backend. The arithmetic and every check are in the MIND programs; this script
# only builds each one, runs it, and compares its exit status and output.
set -euo pipefail
root="$(cd "$(dirname "$0")" && pwd)"
compiler="${MINDC:-}"
[[ -n "$compiler" && -x "$compiler" ]] || { echo "EXAMPLES=NO_MINDC (set MINDC to a mindc binary)" >&2; exit 2; }
for tool in timeout cmp head mktemp sha256sum; do
  command -v "$tool" >/dev/null || { echo "EXAMPLES=MISSING_TOOL_$tool" >&2; exit 2; }
done
tmp="$(mktemp -d)"
trap 'rm -rf -- "$tmp"' EXIT
printf '\177ELF' > "$tmp/elf_magic"
"$compiler" --version > "$tmp/version"
read -r version < "$tmp/version"
echo "MINDC_VERSION=$version"
sha256sum "$compiler" > "$tmp/sha"
read -r digest _ < "$tmp/sha"
echo "MINDC_SHA256=$digest"
count=0
# implication_truth_table writes nothing; each other program writes its EXAMPLE= and SCOPE= lines.
for name in implication_truth_table prime_counting divisor_functions mobius_finite harmonic_exact farey_finite goldbach_bounded; do
  source="$root/$name.mind"
  [[ -f "$source" ]] || { echo "EXAMPLES=MISSING_SOURCE_$name" >&2; exit 2; }
  timeout 180 "$compiler" check "$source" > "$tmp/check.out" 2> "$tmp/check.err" || {
    echo "EXAMPLES=CHECK_ERROR_$name" >&2; cat "$tmp/check.out" "$tmp/check.err" >&2; exit 1;
  }
  # --emit binary: the Mind.toml at the repository root selects a library.
  timeout 180 "$compiler" build "$source" --backend native --target cpu --emit binary \
    --out "$tmp/$name.elf" > "$tmp/build.out" 2> "$tmp/build.err" || {
    echo "EXAMPLES=BUILD_ERROR_$name" >&2; cat "$tmp/build.out" "$tmp/build.err" >&2; exit 1;
  }
  [[ -s "$tmp/$name.elf" ]] || { echo "EXAMPLES=NO_BINARY_$name" >&2; exit 1; }
  head -c 4 "$tmp/$name.elf" > "$tmp/magic"
  cmp -s "$tmp/magic" "$tmp/elf_magic" || { echo "EXAMPLES=NOT_ELF_$name" >&2; exit 1; }
  if [[ "$name" == implication_truth_table ]]; then
    : > "$tmp/expected"
  else
    printf 'EXAMPLE=%s\nSCOPE=FINITE_EXAMPLE\n' "${name^^}" > "$tmp/expected"
  fi
  rc=0
  timeout 30 "$tmp/$name.elf" > "$tmp/out" 2> "$tmp/err" || rc=$?
  if [[ "$rc" -ne 0 || -s "$tmp/err" ]] || ! cmp -s "$tmp/out" "$tmp/expected"; then
    echo "EXAMPLES=FAIL_$name EXIT=$rc" >&2; exit 1
  fi
  echo "EXAMPLE=$name PASS"
  count=$((count + 1))
done
[[ "$count" -eq 7 ]] || { echo "EXAMPLES=INCOMPLETE" >&2; exit 1; }
echo "EXAMPLES_PASSED=7"
