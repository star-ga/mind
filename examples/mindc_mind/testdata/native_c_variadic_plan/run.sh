#!/usr/bin/env bash
# Copyright 2026 STARGA Inc. Transport only; MIND performs every assertion.
set -euo pipefail
export LC_ALL=C OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for variable in ${!LD_@}; do unset "$variable"; done
unset GLIBC_TUNABLES MINDC_NATIVE_ELF MINDC_SO MIND_RUNTIME_PATH MINDC_STD_DIR MINDC_SO_NOBUILD
out=$1
seed=$2
seed_hash=$3
mindc=$4
mindc_hash=$5
cd "$out"
export MIND_STDLIB_PATH="$out/std"
: "${MLIR_OPT:?Set exact mlir-opt path}"
: "${MLIR_TRANSLATE:?Set exact mlir-translate path}"
: "${CLANG:?Set exact clang path}"
export CC="$CLANG"
test ! -e run
sha256sum --check PREEXECUTION.sha256 > preexecution-check.stdout
printf '%s  %s\n' "$seed_hash" "$seed" | sha256sum --check > seed-check.stdout
printf '%s  %s\n' "$mindc_hash" "$mindc" | sha256sum --check > mindc-check.stdout
mkdir run
cp "$seed" run/compiler.elf
chmod 500 run/compiler.elf
printf '1\n' > run/pass.stdout
printf '0\n' > run/fail.stdout
: > run/empty
ulimit -v 2097152
ulimit -c 0
build() {
    local variant=$1
    local dest="$out/run/$variant"
    mkdir "$dest"
    set +e
    (ulimit -t 180; timeout 240s "$out/run/compiler.elf" < "$out/$variant/input.packet") > "$dest/native.elf" 2> "$dest/native-build.stderr"
    local rc=$?
    set -e
    printf '%s\n' "$rc" > "$dest/native-build.exit"
    test "$rc" = 0
    cmp "$dest/native-build.stderr" "$out/run/empty"
    chmod 500 "$dest/native.elf"
    mkdir "$dest/mlir"
    cp "$out/$variant/main.mind" "$out/$variant/Mind.toml" "$dest/mlir/"
    set +e
    (cd "$dest/mlir"; ulimit -t 180; timeout 240s "$mindc" build --backend mlir --target cpu --emit binary --release --no-cache --out "$dest/mlir.elf") > "$dest/mlir-build.stdout" 2> "$dest/mlir-build.stderr"
    rc=$?
    set -e
    printf '%s\n' "$rc" > "$dest/mlir-build.exit"
    test "$rc" = 0
    test -x "$dest/mlir.elf"
    sha256sum "$dest/native.elf" "$dest/mlir.elf" > "$dest/EXECUTABLES.sha256"
}
run_case() {
    local variant=$1 case_id=$2 expected_exit=$3
    local backend
    for backend in native mlir; do
        local stem="$out/run/$variant/$case_id.$backend"
        set +e
        (ulimit -t 10; timeout 12s "$out/run/$variant/$backend.elf" < "$out/fixtures/$case_id.input" 3> "$stem.receipt") > "$stem.stdout" 2> "$stem.stderr"
        local rc=$?
        set -e
        printf '%s\n' "$rc" > "$stem.exit"
        test "$rc" = "$expected_exit"
        cmp "$stem.stderr" "$out/run/empty"
        if test "$expected_exit" = 0; then
            cmp "$stem.stdout" "$out/run/pass.stdout"
            cmp "$stem.receipt" "$out/fixtures/$case_id.expected"
        else
            # Only the MIND assertion's normal exit1/0LF detects a mutation.
            cmp "$stem.stdout" "$out/run/fail.stdout"
        fi
    done
    cmp "$out/run/$variant/$case_id.native.receipt" "$out/run/$variant/$case_id.mlir.receipt"
    printf '%s\t%s\t%s\n' "$variant" "$case_id" "$expected_exit" >> "$out/run/OBSERVED.tsv"
}
build baseline
while IFS=$'\t' read -r case_id variant expected_exit label; do
    if test "$variant" = baseline; then run_case "$variant" "$case_id" "$expected_exit"; fi
done < fixtures/CASES.tsv
while IFS=$'\t' read -r variant case_id expected_exit; do
    build "$variant"
    run_case "$variant" "$case_id" "$expected_exit"
done < fixtures/MUTATIONS.tsv
sha256sum --check PREEXECUTION.sha256 > postexecution-check.stdout
printf '%s  %s\n' "$seed_hash" "$seed" | sha256sum --check > seed-postcheck.stdout
printf '%s  %s\n' "$mindc_hash" "$mindc" | sha256sum --check > mindc-postcheck.stdout
printf 'MIND variadic planner: 65 controls and 6 normal mutation detections per backend\n'
