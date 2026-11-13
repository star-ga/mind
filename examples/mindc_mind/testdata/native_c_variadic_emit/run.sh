#!/usr/bin/env bash
# Copyright 2026 STARGA Inc. Transport only; MIND performs the assertions.
set -euo pipefail
export LC_ALL=C OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for variable in ${!LD_@}; do unset "$variable"; done
unset GLIBC_TUNABLES MINDC_NATIVE_ELF MINDC_SO MIND_RUNTIME_PATH MINDC_STD_DIR MINDC_SO_NOBUILD
out=$1 seed=$2 seed_hash=$3 mindc=$4 mindc_hash=$5
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
    local variant=$1 dest="$out/run/$1"
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
construct() {
    local variant=$1 case_id=$2 side=$3 stem="$out/run/$1/$2.$3"
    set +e
    (ulimit -t 10; timeout 12s "$out/run/$variant/$side.elf" < "$out/fixtures/$case_id.input" 3> "$stem.artifact") > "$stem.stdout" 2> "$stem.stderr"
    local rc=$?
    set -e
    printf '%s\n' "$rc" > "$stem.exit"
    test "$rc" = 0
    cmp "$stem.stderr" "$out/run/empty"
    cmp "$stem.stdout" "$out/run/pass.stdout"
}
call_case() {
    local variant=$1 case_id=$2 expected=$3 side verifier
    for side in native mlir; do
        construct "$variant" "$case_id" "$side"
        local stem="$out/run/$variant/$case_id.$side"
        set +e
        (ulimit -t 10; timeout 12s "$out/run/capture" < "$stem.artifact") > "$stem.raw" 2> "$stem.capture.stderr"
        local rc=$?
        set -e
        printf '%s\n' "$rc" > "$stem.capture.exit"
        test "$rc" = 0
        cmp "$stem.capture.stderr" "$out/run/empty"
        cat "$out/fixtures/$case_id.verify" "$stem.raw" > "$stem.verification-input"
        for verifier in native mlir; do
            local verdict="$stem.$verifier-verifier"
            set +e
            (ulimit -t 10; timeout 12s "$out/run/verifier/$verifier.elf" < "$stem.verification-input" 3> "$verdict.semantic") > "$verdict.stdout" 2> "$verdict.stderr"
            rc=$?
            set -e
            printf '%s\n' "$rc" > "$verdict.exit"
            test "$rc" = "$expected"
            cmp "$verdict.stderr" "$out/run/empty"
            if test "$expected" = 0; then
                cmp "$verdict.stdout" "$out/run/pass.stdout"
                cmp "$verdict.semantic" "$out/fixtures/$case_id.semantic"
            else
                # Only normal MIND assertion exit1/0LF counts as detection.
                cmp "$verdict.stdout" "$out/run/fail.stdout"
            fi
        done
        cmp "$stem.native-verifier.semantic" "$stem.mlir-verifier.semantic"
    done
    cmp "$out/run/$variant/$case_id.native.artifact" "$out/run/$variant/$case_id.mlir.artifact"
    cmp "$out/run/$variant/$case_id.native.native-verifier.semantic" "$out/run/$variant/$case_id.mlir.native-verifier.semantic"
    printf '%s\t%s\t%s\n' "$variant" "$case_id" "$expected" >> "$out/run/OBSERVED.tsv"
}
set +e
(ulimit -t 180; timeout 240s "$CLANG" -std=c11 -O2 -Wall -Wextra -Werror fixtures/capture.c fixtures/capture.S -o run/capture) > run/capture-build.stdout 2> run/capture-build.stderr
rc=$?
set -e
printf '%s\n' "$rc" > run/capture-build.exit
test "$rc" = 0
sha256sum run/capture > run/CAPTURE.sha256
build verifier
build baseline
while IFS=$'\t' read -r case_id variant mode label; do
    if test "$variant" != baseline; then continue; fi
    if test "$mode" = call; then
        call_case "$variant" "$case_id" 0
    else
        construct "$variant" "$case_id" native
        construct "$variant" "$case_id" mlir
        cmp "run/$variant/$case_id.native.artifact" "run/$variant/$case_id.mlir.artifact"
        printf '%s\t%s\t0\n' "$variant" "$case_id" >> run/OBSERVED.tsv
    fi
done < fixtures/CASES.tsv
while IFS=$'\t' read -r variant case_id expected; do
    build "$variant"
    if test "$variant" = allocation-zero; then
        construct "$variant" "$case_id" native
        construct "$variant" "$case_id" mlir
        cmp "run/$variant/$case_id.native.artifact" "run/$variant/$case_id.mlir.artifact"
        printf '%s\t%s\t0\n' "$variant" "$case_id" >> run/OBSERVED.tsv
    else
        call_case "$variant" "$case_id" "$expected"
    fi
done < fixtures/MUTATIONS.tsv
sha256sum --check PREEXECUTION.sha256 > postexecution-check.stdout
printf '%s  %s\n' "$seed_hash" "$seed" | sha256sum --check > seed-postcheck.stdout
printf '%s  %s\n' "$mindc_hash" "$mindc" | sha256sum --check > mindc-postcheck.stdout
printf 'MIND variadic emitter: 55 controls, 8 normal mutation detections; both builder/checker backends\n'
