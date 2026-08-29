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
set -u
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE" || exit 2
STAGE1="examples/mindc_mind/testdata/selfhost_loop/stage1.elf"
MINDC="target/mlir-free/release/mindc"

command -v cargo >/dev/null || { echo "BLOCKED: cargo missing"; exit 2; }
command -v strace >/dev/null || { echo "BLOCKED: strace missing"; exit 2; }
[ -f "$STAGE1" ] || { echo "BLOCKED: stage1.elf missing at $STAGE1"; exit 2; }

echo "== building mindc with mlir-build OFF (MLIR physically compiled out) =="
# Dedicated target dir so this never clobbers the default (mlir-on) build.
CARGO_TARGET_DIR=target/mlir-free \
  cargo build --release --bin mindc \
  --no-default-features --features "std-surface cross-module-imports" \
  >/tmp/ri_d1_mlir_free_build.log 2>&1
if [ $? -ne 0 ] || [ ! -x "$MINDC" ]; then
  echo "FAIL: mindc did NOT build with mlir-build off — an un-gated MLIR reference"
  echo "      remains on the native path. Offending errors:"
  grep -E "^error" /tmp/ri_d1_mlir_free_build.log | head -15
  exit 1
fi
echo "  ok: mindc built MLIR-free ($(stat -c%s "$MINDC") bytes)"

fails=0
TD="$(mktemp -d)"
trap 'rm -rf "$TD"' EXIT

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
