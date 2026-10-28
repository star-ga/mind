#!/usr/bin/env bash
# Author: STARGA Inc. Build/transport only; executable assertions are MIND.
set -euo pipefail
umask 077
repo=$(cd -- "$(dirname -- "$0")/.." && pwd -P)
compiler=${MINDC:?set MINDC to an absolute compiler executable}
result=${1:?supply a new absolute evidence directory}
case "$compiler:$result" in /*:/*) ;; *) exit 64 ;; esac
test ! -e "$result"
mkdir -p "$result/fixture" "$result/expected"
printf M > "$result/fixture/existing"
printf '0\n' > "$result/expected/stdout"
: > "$result/expected/stderr"
printf '0\n' > "$result/expected/exit"
sha256sum "$compiler" "$repo/tests/open_error_kat.mind" > "$result/inputs.sha256"
"$compiler" --version > "$result/compiler.version"
prlimit --pid $$ > "$result/process.limits"
test "$(prlimit --pid $$ --as --noheadings --output SOFT | tr -d ' ')" = 2147483648
for profile in debug release; do
    project="$result/$profile"
    mkdir "$project"
    cp "$repo/tests/open_error_kat.mind" "$project/main.mind"
    cat > "$project/Mind.toml" <<'TOML'
[package]
name = "open_error_regression"
version = "0.0.1"
[build]
entry = "main.mind"
[targets.cpu]
backend = "cpu"
sources = ["main.mind"]
TOML
    (
        cd "$project"
        printf '%s\n' "mindc build --backend mlir --target cpu --emit binary --optimize $profile --no-cache --out program" > build.command
        set +e
        timeout --kill-after=5s 180s "$compiler" build --backend mlir --target cpu --emit binary --optimize "$profile" --no-cache --out program > build.stdout 2> build.stderr
        code=$?
        set -e
        printf '%s\n' "$code" > build.exit
        test "$code" = 0
        sha256sum program > program.sha256
        test ! -e "$result/fixture/missing"
        set +e
        timeout --kill-after=5s 30s ./program "$result/fixture/existing" "$result/fixture/missing" "$result/fixture/existing/child" > run.stdout 2> run.stderr
        code=$?
        set -e
        printf '%s\n' "$code" > run.exit
        for stream in stdout stderr exit; do cmp "$result/expected/$stream" "run.$stream"; done
    )
done
sha256sum -c "$result/inputs.sha256" > "$result/inputs.after.checked"
printf 'open-error debug/release expected outcomes observed\n'
