#!/usr/bin/env bash
# Author: STARGA Inc. Source assembly only; executable assertions are MIND.
set -euo pipefail
umask 077
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
subject=${1:?path to the reviewed std/sha256.mind}
profile=${2:?real, observed, raw, or raw-observed}
out=${3:?fresh output directory}
[[ ! -e "$out" && ! -L "$out" ]] || exit 64
case "$profile" in
  real|raw) observed=0 ;;
  observed|raw-observed) observed=1 ;;
  *) exit 64 ;;
esac
mkdir -p "$out"
# Namespace only the subject functions to avoid binding the compiler's bundled
# std.sha256 accidentally. No constants, operators or control flow change.
sed -E 's/\b(mask32|rotr32|shr32|ch|maj|bsig0|bsig1|ssig0|ssig1|k_word|h_init|w_get|w_set|compress_block|sha256|hash)\b/st_subject_\1/g' "$subject" > "$out/subject.mind"
if [[ "$observed" == 1 ]]; then
  sed -E \
    -e 's/^pub fn st_subject_sha256\(in_addr: i64, in_len: i64, out_addr: i64\)/pub fn st_subject_sha256(in_addr: i64, in_len: i64, out_addr: i64, audit: i64)/' \
    -e 's/^pub fn st_subject_hash\(input: bytes\)/pub fn st_subject_hash(input: bytes, audit: i64)/' \
    -e 's/st_subject_sha256\(scratch, len, out\)/st_subject_sha256(scratch, len, out, audit)/g' \
    -e 's/__mind_alloc\(/st_allocate(audit, /g' \
    -e 's/__mind_free\(/st_release(audit, /g' \
    "$out/subject.mind" > "$out/instrumented.mind"
  mv "$out/instrumented.mind" "$out/subject.mind"
fi
cat "$out/subject.mind" > "$out/prelude.mind"
printf '\n' >> "$out/prelude.mind"
cat "$here/observer.mind" >> "$out/prelude.mind"
printf '\n' >> "$out/prelude.mind"
cat "$here/vectors.mind" >> "$out/prelude.mind"
printf '\n' >> "$out/prelude.mind"
printf 'fn st_observed()->i64 { %s }\n' "$observed" >> "$out/prelude.mind"
if [[ "$observed" == 1 ]]; then
  printf '%s\n' 'fn st_raw(s:i64,p:i64,n:i64,o:i64)->i64 { st_subject_sha256(p,n,o,s) }' 'fn st_wrapped(s:i64,b:bytes)->bytes[32] { st_subject_hash(b,s) }' >> "$out/prelude.mind"
else
  printf '%s\n' 'fn st_raw(s:i64,p:i64,n:i64,o:i64)->i64 { st_subject_sha256(p,n,o) }' 'fn st_wrapped(s:i64,b:bytes)->bytes[32] { st_subject_hash(b) }' >> "$out/prelude.mind"
fi
awk '/^pub fn main\(\)->i64 \{/ {exit} {print}' "$here/driver.mind" >> "$out/prelude.mind"
case "$profile" in
  real|observed) awk 'found || /^pub fn main\(\)->i64 \{/ {found=1; print}' "$here/driver.mind" > "$out/entry.mind" ;;
  raw) cat "$here/raw_driver.mind" > "$out/entry.mind" ;;
  raw-observed) cat "$here/raw_observed_driver.mind" > "$out/entry.mind" ;;
esac
cat "$out/prelude.mind" "$out/entry.mind" > "$out/main.mind"
printf '%s\n' '[package]' 'name="sha256-lifetime-diagnostic"' 'version="0.0.0"' 'authors=["STARGA Inc."]' '[build]' 'entry="main.mind"' > "$out/Mind.toml"
sha256sum "$subject" "$here/observer.mind" "$here/vectors.mind" "$here/driver.mind" "$out/main.mind" > "$out/SOURCES.sha256"
