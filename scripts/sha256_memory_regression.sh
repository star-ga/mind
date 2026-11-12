#!/usr/bin/env bash
# Author: STARGA Inc. Build/transport harness; all executable checks are MIND.
set -euo pipefail
umask 077
repo=$(cd -- "$(dirname -- "$0")/.." && pwd -P)
compiler=${MINDC:?set MINDC to the tested compiler executable}
result=${1:?supply a new absolute evidence directory}
case "$compiler:$result" in /*:/*) ;; *) exit 64 ;; esac
test ! -e "$result"
mkdir -p "$result"
compiler=$(realpath -- "$compiler")
sha256sum "$compiler" > "$result/compiler.sha256"
"$compiler" --version > "$result/compiler.version"
cg=$(awk -F: '$1=="0" {print $3}' /proc/self/cgroup)
for key in memory.max memory.swap.max cpu.max cpuset.cpus.effective; do
    cat "/sys/fs/cgroup$cg/$key" > "$result/$key"
done
test "$(cat "$result/memory.max")" = 2147483648
test "$(cat "$result/memory.swap.max")" = 0
test "$(cat "$result/cpu.max")" = '100000 100000'
case "$(cat "$result/cpuset.cpus.effective")" in ''|*[!0-9]*) exit 64 ;; esac
prlimit --pid $$ > "$result/process.limits"
test "$(prlimit --pid $$ --as --noheadings --output SOFT | tr -d ' ')" = 2147483648
cat "/sys/fs/cgroup$cg/memory.events" > "$result/memory.events.before"
finish() {
    code=$?
    trap - EXIT
    for key in memory.events memory.peak memory.swap.current; do
        cat "/sys/fs/cgroup$cg/$key" > "$result/$key"
    done
    if ! awk 'NR==FNR {b[$1]=$2;next} ($1=="oom" || $1=="oom_kill" || $1=="oom_group_kill") && $2>b[$1] {bad=1} END {exit bad}' "$result/memory.events.before" "$result/memory.events"; then code=1; fi
    printf '%s\n' "$code" > "$result/runner.exit"
    exit "$code"
}
trap finish EXIT
# Exactly one literal replacement. A changed source seam refuses preparation.
replace_one() {
    awk -v old="$2" -v new="$3" '
      {p=index($0,old); if(p) {count++; $0=substr($0,1,p-1) new substr($0,p+length(old))} print}
      END {if(count!=1) exit 65}' "$1" > "$1.next"
    mv -- "$1.next" "$1"
}
mkdir "$result/projects" "$result/expected"
for id in KAT RPT F01 F02 F03 F04 M01 M02 L01 L02 L03 L04 L05 L06; do
    project="$result/projects/$id"
    mkdir "$project"
    cp "$repo/std/sha256.mind" "$project/module.mind"
    expected=0; line=0; arg=''; testfile=sha256_memory_kat.mind
    case "$id" in
        RPT|M01) testfile=sha256_repeated_kat.mind ;;
        F*) testfile=sha256_allocation_kat.mind; arg=${id#F0} ;;
        L*) testfile=sha256_legacy_kat.mind; case "$id" in L01|L03|L05) arg=1 ;; *) arg=2 ;; esac ;;
    esac
    cp "$repo/tests/$testfile" "$project/check.mind"
    case "$id" in
        F01|F04|L03|L04|L05|L06) replace_one "$project/module.mind" 'let work: i64 = __mind_alloc(704);' 'let work: i64 = 0;' ;;
        F02) replace_one "$project/module.mind" 'let scratch: i64 = __mind_alloc(scratch_len);' 'let scratch: i64 = 0;' ;;
        F03) replace_one "$project/module.mind" 'let out: i64 = __mind_alloc(32);' 'let out: i64 = 0;' ;;
        M01) replace_one "$project/module.mind" '__mind_free(work);' 'let retained_work: i64 = work;'; expected=1; line=1 ;;
        M02) replace_one "$project/module.mind" 'return 1779033703;' 'return 1779033702;'; expected=1; line=110 ;;
    esac
    case "$id" in
        L01|L02)
            # Isolate the named test-only function; do not edit other returns.
            awk '/^fn sl_expected_trap\(/{inside=1} {if(inside && /return 1;/){sub(/return 1;/,"return 0;");n++} print} inside && /^}/{inside=0} END{if(n!=1)exit 65}' "$project/check.mind" > "$project/check.next"
            mv "$project/check.next" "$project/check.mind" ;;
        L05|L06) replace_one "$project/module.mind" '__mind_assert_fail(0)' '0'; expected=1; line=1 ;;
    esac
    cat "$project/module.mind" > "$project/main.mind"
    case "$id" in
        L*) sed '/^import std.sha256;$/d' "$repo/std/io_canon.mind" >> "$project/main.mind" ;;
    esac
    cat "$project/check.mind" >> "$project/main.mind"
    cat > "$project/Mind.toml" <<'TOML'
[package]
name = "sha256_memory_regression"
version = "0.0.1"
[build]
entry = "main.mind"
[targets.cpu]
backend = "cpu"
sources = ["main.mind"]
TOML
    printf '%s\n' "$expected" > "$result/expected/$id.exit"
    printf '%s\n' "$line" > "$result/expected/$id.stdout"
    : > "$result/expected/$id.stderr"
    printf '%s\n' "$arg" > "$result/expected/$id.args"
done
# Freeze all sources and expected bytes before compiling or executing a case.
find "$result/projects" "$result/expected" -type f -print0 | sort -z | xargs -0 sha256sum > "$result/cases.sha256"
sha256sum "$result/cases.sha256" > "$result/cases.manifest.sha256"
for id in KAT RPT F01 F02 F03 F04 M01 M02 L01 L02 L03 L04 L05 L06; do
    project="$result/projects/$id"
    (
        cd "$project"
        printf '%s\n' 'mindc build --backend mlir --target cpu --emit binary --optimize debug --no-cache --out program' > build.command
        set +e
        timeout --kill-after=5s 180s "$compiler" build --backend mlir --target cpu --emit binary --optimize debug --no-cache --out program > build.stdout 2> build.stderr
        built=$?
        set -e
        printf '%s\n' "$built" > build.exit
        test "$built" = 0
        sha256sum program > program.sha256
        args=(); read -r -a args < "$result/expected/$id.args" || true
        runner=(); case "$id" in RPT|M01) runner=(prlimit --as=33554432 --) ;; esac
        set +e
        timeout --kill-after=5s 60s "${runner[@]}" ./program "${args[@]}" > run.stdout 2> run.stderr
        observed=$?
        set -e
        printf '%s\n' "$observed" > run.exit
        cmp "$result/expected/$id.exit" run.exit
        cmp "$result/expected/$id.stdout" run.stdout
        cmp "$result/expected/$id.stderr" run.stderr
    )
    printf '%s expected outcome observed\n' "$id"
done
sha256sum -c "$result/cases.sha256" > "$result/cases.after.checked"
