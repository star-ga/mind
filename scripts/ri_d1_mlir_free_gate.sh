#!/usr/bin/env bash
# RI-D1 gate assertion #1 — MLIR is un-linkable on the native path (fail-closed
# BY CONSTRUCTION, not by runtime observation).
#
# Builds mindc with the MLIR toolchain feature (mlir-build) COMPILED OUT, then
# proves `mindc build --backend native` still produces a correct, running ELF
# while spawning ZERO external toolchain (mlir-opt/mlir-translate/clang/ld).
# Because the MLIR pipeline code (src/eval/mlir_build.rs, cfg(feature="mlir-build"))
# is not in the binary at all, a silent MLIR fallback cannot exist — the compiler
# would fail to LINK if the native path secretly depended on it. This is the
# strongest leg of the RI dependency-cut for rows #4/#5/#6 (MLIR_OPT / MLIR_TRANSLATE
# / CLANG): not merely "0 toolchain execve at runtime" but "MLIR physically absent."
#
# Exit: 0 pass; 1 a check failed; 2 BLOCKED (missing cargo / stage1.elf / strace).
# Optional overrides for isolated validation: RI_D1_TARGET_DIR selects the Cargo
# target directory and RI_D1_CARGO_JOBS selects the Cargo parallelism.
set -u
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE" || exit 2
STAGE1="examples/mindc_mind/testdata/selfhost_loop/stage1.elf"
TARGET_DIR="${RI_D1_TARGET_DIR:-target/mlir-free}"
MINDC="$TARGET_DIR/release/mindc"
JOBS="${RI_D1_CARGO_JOBS:-2}"
TD="$(mktemp -d -t ri_d1_mlir_free.XXXXXX)"
trap 'rm -rf "$TD"' EXIT
BUILD_LOG="$TD/mindc-build.log"

command -v cargo >/dev/null || { echo "BLOCKED: cargo missing"; exit 2; }
command -v strace >/dev/null || { echo "BLOCKED: strace missing"; exit 2; }
[ -f "$STAGE1" ] || { echo "BLOCKED: stage1.elf missing at $STAGE1"; exit 2; }

echo "== building mindc with mlir-build OFF (MLIR physically compiled out) =="
# Dedicated target dir so this never clobbers the default (mlir-on) build.
CARGO_TARGET_DIR="$TARGET_DIR" \
  cargo build --release --locked --bin mindc \
  --no-default-features --features "std-surface cross-module-imports" \
  -j "$JOBS" \
  >"$BUILD_LOG" 2>&1
if [ $? -ne 0 ] || [ ! -x "$MINDC" ]; then
  echo "FAIL: mindc did NOT build with mlir-build off — an un-gated MLIR reference"
  echo "      remains on the native path. Offending errors:"
  grep -E "^error" "$BUILD_LOG" | head -15 || true
  exit 1
fi
echo "  ok: mindc built MLIR-free ($(stat -c%s "$MINDC") bytes)"

# WSP-10: inspect the binary that the build above actually produced.  The
# feature list is only an input request; a default-feature or dependency change
# could still put MLIR into this artifact.  These symbol/marker checks are
# evidence about this binary, not a formal proof of all possible MLIR code.
command -v nm >/dev/null 2>&1 || {
  echo "BLOCKED: nm missing — cannot inspect the just-built artifact"
  exit 2
}
command -v strings >/dev/null 2>&1 || {
  echo "BLOCKED: strings missing — cannot inspect the just-built artifact"
  exit 2
}

NM_OUT="$TD/mindc.nm"
NM_ERR="$TD/mindc.nm.err"
nm_rc=0
nm -C "$MINDC" >"$NM_OUT" 2>"$NM_ERR" || nm_rc=$?
if [ "$nm_rc" -ne 0 ]; then
  echo "FAIL: nm could not inspect the just-built artifact $MINDC (rc=$nm_rc)"
  sed -n '1,20p' "$NM_ERR"
  exit 1
fi
total_syms="$(awk 'NF { count++ } END { print count + 0 }' "$NM_OUT")"
if [ "${total_syms:-0}" -lt 1 ]; then
  echo "FAIL: nm returned zero symbols for $MINDC; an empty inspection is not"
  echo "      evidence that MLIR is absent"
  exit 1
fi

STRINGS_OUT="$TD/mindc.strings"
STRINGS_ERR="$TD/mindc.strings.err"
strings_rc=0
strings -a "$MINDC" >"$STRINGS_OUT" 2>"$STRINGS_ERR" || strings_rc=$?
if [ "$strings_rc" -ne 0 ]; then
  echo "FAIL: strings could not inspect the just-built artifact $MINDC (rc=$strings_rc)"
  sed -n '1,20p' "$STRINGS_ERR"
  exit 1
fi
string_lines="$(awk 'NF { count++ } END { print count + 0 }' "$STRINGS_OUT")"
if [ "${string_lines:-0}" -lt 1 ]; then
  echo "FAIL: strings returned no content for $MINDC; an empty inspection is not"
  echo "      evidence that MLIR is absent"
  exit 1
fi

# Do not use grep pipelines here: grep's no-match status is expected for a
# clean artifact and must not mask, or turn into, the inspection status above.
# The native signature adapter is named ``extern_type_to_mlir_*`` for historical
# compatibility and is present in both feature profiles; it is not an MLIR
# pipeline symbol.  Exclude that known adapter while counting the implementation
# symbols whose presence distinguishes the MLIR-enabled artifact.
implementation_mlir_syms="$(awk 'tolower($0) ~ /mlir/ && tolower($0) !~ /extern_type_to_mlir/ { count++ } END { print count + 0 }' "$NM_OUT")"
mlir_marks="$(awk '/mlir_build/ { count++ } END { print count + 0 }' "$STRINGS_OUT")"
if [ "${implementation_mlir_syms:-0}" -ne 0 ] || [ "${mlir_marks:-0}" -ne 0 ]; then
  echo "FAIL: just-built artifact carries MLIR — $implementation_mlir_syms implementation MLIR symbol(s),"
  echo "      $mlir_marks mlir_build marker(s), $(stat -c%s "$MINDC") bytes"
  echo "      (build flags alone cannot establish MLIR absence)"
  exit 1
fi
echo "  ok: artifact inspected — zero implementation MLIR symbols of $total_syms total, 0 mlir_build markers"

fails=0

# Vacuity floor for the strace evidence. Both legs below conclude "ZERO
# toolchain was spawned" from an EMPTY grep over the strace log -- but an empty
# log is equally consistent with tracing never having happened at all (a seccomp
# or container quirk, a clobbered redirect, an strace that logged elsewhere).
# Absence of evidence was being read as evidence of absence, which is the whole
# assertion of this gate. A traced run ALWAYS records at least mindc's own
# execve, so a log with none is a broken observation, not a clean result.
assert_traced() {  # $1 = strace log, $2 = leg name
  n="$(grep -c 'execve("' "$1" 2>/dev/null)"
  if [ "${n:-0}" -lt 1 ]; then
    echo "FAIL: strace recorded ZERO execve on the $2 leg ($1) — tracing did not"
    echo "      happen, so 'no toolchain spawned' is an EMPTY observation rather"
    echo "      than evidence. Expected at least mindc's own execve."
    return 1
  fi
  echo "  ok: $2 leg traced ($n execve record(s) observed)"
  return 0
}

# A scalar in-profile program: 7 + 35 -> exit 42.
printf 'fn add(a:i64,b:i64)->i64{return a+b;}\nfn main()->i64{return add(7,35);}\n' > "$TD/p.mind"
env MINDC_STD_DIR="$HERE/std" MINDC_NATIVE_ELF="$HERE/$STAGE1" \
  strace -f -e trace=execve -qq "$MINDC" build --backend native "$TD/p.mind" --out "$TD/p.elf" \
  >/dev/null 2>"$TD/strace"
rc=$?

# Parse ONLY execve("<path>") tokens (never mindc's own "zero MLIR/LLVM/clang"
# status prose). Assert the toolchain set is empty.
tool="$(grep -oE 'execve\("[^"]+"' "$TD/strace" | sed 's/execve("//' \
        | grep -oiE 'mlir-opt|mlir-translate|clang|/ld$|ld\.lld|lld|/cc$' | sort -u | tr '\n' ',')"

assert_traced "$TD/strace" "in-profile" || fails=$((fails+1))

if [ "$rc" -ne 0 ] || [ ! -s "$TD/p.elf" ]; then
  echo "FAIL: MLIR-free native build rc=$rc, artifact=$([ -s "$TD/p.elf" ] && echo yes || echo no)"
  fails=$((fails+1))
else
  chmod +x "$TD/p.elf"; "$TD/p.elf"; got=$?
  if [ "$got" -ne 42 ]; then
    echo "FAIL: emitted ELF exit=$got, expected 42"; fails=$((fails+1))
  elif [ -n "$tool" ]; then
    echo "FAIL: native build spawned toolchain [$tool] — not MLIR-free"; fails=$((fails+1))
  else
    echo "  ok: --backend native -> running ELF (exit 42), ZERO toolchain execve"
  fi
fi

# Out-of-profile (tensor) must still fail-closed on the MLIR-free binary — never
# a silent MLIR fallback (there is no MLIR to fall back to).
printf 'fn main()->i64{let t=zeros([4]); return 0;}\n' > "$TD/t.mind"
env MINDC_STD_DIR="$HERE/std" MINDC_NATIVE_ELF="$HERE/$STAGE1" \
  strace -f -e trace=execve -qq "$MINDC" build --backend native "$TD/t.mind" --out "$TD/t.elf" \
  >/dev/null 2>"$TD/strace2"
trc=$?
ttool="$(grep -oE 'execve\("[^"]+"' "$TD/strace2" | sed 's/execve("//' \
         | grep -oiE 'mlir-opt|mlir-translate|clang' | sort -u | tr '\n' ',')"
assert_traced "$TD/strace2" "out-of-profile" || fails=$((fails+1))

if [ "$trc" -eq 0 ] || [ -s "$TD/t.elf" ] || [ -n "$ttool" ]; then
  echo "FAIL: out-of-profile tensor build rc=$trc artifact=$([ -s "$TD/t.elf" ] && echo yes || echo no) toolchain=[$ttool] — must fail-closed"
  fails=$((fails+1))
else
  echo "  ok: out-of-profile tensor -> fail-closed (rc=$trc), no artifact, no toolchain"
fi

if [ "$fails" -ne 0 ]; then
  echo "FAIL  RI-D1 MLIR-free gate ($fails check(s) failed)"
  exit 1
fi
echo "PASS  RI-D1: MLIR is un-linkable (mlir-build compiled out) yet --backend native"
echo "      builds + runs correct ELFs toolchain-free — no MLIR fallback can exist."
exit 0
