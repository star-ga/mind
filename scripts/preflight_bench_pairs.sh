#!/usr/bin/env bash
# Pair HEAD with the merge-base after the absolute bench floor misses.
# The floor file is not changed. The 3% pair rule is not changed.
set -uo pipefail
cd "$(dirname "$0")/.."

if [ "$#" -ne 3 ]; then
  echo "usage: preflight_bench_pairs.sh <bench_tmp> <head_target> <floor_file>" >&2
  exit 4
fi

bench_tmp="$1"
head_td="$2"
floor_file="$3"
base_wt=""
base_td="${bench_tmp}/base-target"
pairs="${bench_tmp}/pairs.tsv"

cleanup() {
  if [ -n "${base_wt:-}" ] && [ -d "${base_wt:-}" ]; then
    git worktree remove --force "$base_wt" || true
  fi
}
trap cleanup EXIT

# A detached gate checkout has neither main ref. Same fixed order as
# resolve_hygiene_base in preflight.sh: local main, then the remote-tracking
# main, then MIND_HYGIENE_BASE. A candidate counts only when it names a commit
# that shares history with HEAD. A miss is named. It is never a pass.
base_sha=""
base_tries=""
try_base() { # <label> <ref>
  local mb=""
  if ! git rev-parse --verify -q "${2}^{commit}" >/dev/null; then
    base_tries="${base_tries}; ${1} not a commit"
  elif ! mb=$(git merge-base HEAD "$2" 2>/dev/null) || [ -z "$mb" ]; then
    base_tries="${base_tries}; ${1} shares no history with HEAD"
  else
    base_sha="$mb"
  fi
}
try_base refs/heads/main refs/heads/main
[ -n "$base_sha" ] || try_base refs/remotes/origin/main refs/remotes/origin/main
if [ -z "$base_sha" ]; then
  if [ -z "${MIND_HYGIENE_BASE:-}" ]; then
    base_tries="${base_tries}; MIND_HYGIENE_BASE unset"
  else
    try_base "MIND_HYGIENE_BASE=${MIND_HYGIENE_BASE}" "$MIND_HYGIENE_BASE"
  fi
fi
if [ -z "$base_sha" ]; then
  echo "FAIL: no merge-base; the paired bench was not evaluated. Tried: ${base_tries#; }"
  echo "Fetch refs/heads/main or refs/remotes/origin/main, or set MIND_HYGIENE_BASE to a commit."
  echo "bench logs retained in ${bench_tmp}"
  exit 4
fi

echo "LOAD_START $(cat /proc/loadavg)"
echo "bench pairs base=${base_sha} head=$(git rev-parse HEAD)"
base_wt="${bench_tmp}/base-src"
if ! git worktree add --detach "$base_wt" "$base_sha"; then
  echo "FAIL: could not check out the merge-base; the paired bench was not evaluated"
  echo "bench logs retained in ${bench_tmp}"
  exit 4
fi

# Build both bench binaries before any timing. Equal bytes run equal code,
# so equal sha256 is the verdict. Different bytes go to the timed pairs.
build_exe() {
  local rc=0
  CARGO_TARGET_DIR="$1" CARGO_NET_OFFLINE=true CARGO_TERM_COLOR=never \
    cargo bench --manifest-path "$2" --offline --locked --no-run \
    --bench compiler --no-default-features --message-format=json \
    > "$3" 2>"$3.err" || rc=$?
  if [ "$rc" -ne 0 ]; then
    tail -15 "$3.err" >&2
    return "$rc"
  fi
  python3 tools/bench_pairs.py bench-exe "$3"
}
base_exe=""
head_exe=""
if ! base_exe=$(build_exe "$base_td" "${base_wt}/Cargo.toml" "${bench_tmp}/base-build.json") \
  || ! head_exe=$(build_exe "$head_td" Cargo.toml "${bench_tmp}/head-build.json"); then
  echo "FAIL: could not build the base and head bench binaries; the paired bench was not evaluated"
  echo "bench logs retained in ${bench_tmp}"
  exit 4
fi
same_rc=0
python3 tools/bench_pairs.py same-binary "$base_exe" "$head_exe" || same_rc=$?
if [ "$same_rc" -eq 0 ]; then
  echo "bench logs retained in ${bench_tmp}"
  exit 0
fi
if [ "$same_rc" -ne 1 ]; then
  echo "FAIL: bench binary compare could not decide (exit ${same_rc}); not a pass"
  echo "bench logs retained in ${bench_tmp}"
  exit 4
fi

pick_rc=0
pick_out=$(python3 tools/bench_pairs.py pick-core) || pick_rc=$?
printf '%s\n' "$pick_out"
if [ "$pick_rc" -ne 0 ]; then
  echo "FAIL: no core for the paired bench; the regression gate was not evaluated"
  echo "bench logs retained in ${bench_tmp}"
  exit 4
fi

core=$(printf '%s\n' "$pick_out" | awk '$1=="PICK" {print $2}')
idle=$(printf '%s\n' "$pick_out" | awk '$1=="IDLE" {print $2}')
sib=$(printf '%s\n' "$pick_out" | awk '$1=="SIB" {print $2}')
sib_idle=$(printf '%s\n' "$pick_out" | awk '$1=="SIB_IDLE" {print $2}')
if [ -z "${core:-}" ] || [ -z "${sib:-}" ]; then
  echo "FAIL: no core for the paired bench; the regression gate was not evaluated"
  echo "bench logs retained in ${bench_tmp}"
  exit 4
fi
echo "PAIR_CORE ${core} IDLE ${idle:-} SIB ${sib} SIB_IDLE ${sib_idle:-}"

pin=(taskset -c "$core")
pin_mode="taskset"
if taskset -c "$core" chrt -f 2 true >/dev/null 2>&1; then
  pin=(taskset -c "$core" chrt -f 2)
  pin_mode="taskset+fifo"
fi
echo "PIN ${pin_mode} core=${core} sib=${sib}"

printf '# protocol=pipeline\n# core=%s\n# sib=%s\n# idle=%s\n# sib_idle=%s\nrow\ti\torder\tt_base_ns\tt_cand_ns\n' \
  "$core" "$sib" "${idle:-}" "${sib_idle:-}" > "$pairs"

run_side() {
  local rc=0
  CARGO_TARGET_DIR="$1" CARGO_NET_OFFLINE=true CARGO_TERM_COLOR=never \
    "${pin[@]}" cargo bench --manifest-path "$2" --offline --locked \
    --bench compiler --no-default-features -- \
    --warm-up-time 3 --measurement-time 8 --output-format bencher \
    compiler_pipeline/parse_typecheck_ir > "$3" 2>"$3.err" || rc=$?
  return "$rc"
}

i=0
while [ "$i" -lt 9 ]; do
  if [ $((i % 2)) -eq 0 ]; then
    order=BC
  else
    order=CB
  fi
  echo "PAIR i=${i} order=${order} LOAD $(cat /proc/loadavg) CORE ${core} SIB ${sib}"
  rc=0
  if [ "$order" = "BC" ]; then
    run_side "$base_td" "${base_wt}/Cargo.toml" "${bench_tmp}/p${i}-base.out" || rc=$?
    if [ "$rc" -eq 0 ]; then
      run_side "$head_td" Cargo.toml "${bench_tmp}/p${i}-cand.out" || rc=$?
    fi
  else
    run_side "$head_td" Cargo.toml "${bench_tmp}/p${i}-cand.out" || rc=$?
    if [ "$rc" -eq 0 ]; then
      run_side "$base_td" "${base_wt}/Cargo.toml" "${bench_tmp}/p${i}-base.out" || rc=$?
    fi
  fi
  if [ "$rc" -ne 0 ]; then
    tail -15 "${bench_tmp}/p${i}-base.err" "${bench_tmp}/p${i}-cand.err" 2>/dev/null || true
    echo "FAIL: paired bench command failed (exit ${rc}) at pair ${i}"
    echo "bench logs retained in ${bench_tmp}"
    exit 4
  fi
  row_rc=0
  python3 tools/bench_pairs.py append-rows "$pairs" "$i" "$order" \
    "${bench_tmp}/p${i}-base.out" "${bench_tmp}/p${i}-cand.out" || row_rc=$?
  if [ "$row_rc" -ne 0 ]; then
    echo "FAIL: paired bench output did not contain the three pipeline rows"
    echo "bench logs retained in ${bench_tmp}"
    exit 4
  fi
  i=$((i + 1))
done

echo "LOAD_END $(cat /proc/loadavg)"
echo "PAIRS_TSV"
cat "$pairs"
pair_rc=0
python3 tools/bench_gate.py --pairs "$pairs" || pair_rc=$?
echo "GATE pairs rc=${pair_rc}"
if [ "$pair_rc" -eq 1 ]; then
  echo "FAIL: bench REGRESSION: head is slower than the merge-base on the paired run"
  echo "bench logs retained in ${bench_tmp}"
  exit 1
fi
if [ "$pair_rc" -eq 2 ]; then
  load=$(awk '{print $1}' /proc/loadavg)
  python3 tools/bench_pairs.py noisy-line "$load"
  echo "bench logs retained in ${bench_tmp}"
  exit 2
fi
if [ "$pair_rc" -ne 0 ]; then
  echo "FAIL: paired bench could not decide (exit ${pair_rc}); not a pass"
  echo "bench logs retained in ${bench_tmp}"
  exit "$pair_rc"
fi
host_rc=0
python3 tools/bench_pairs.py host-floor "$pairs" "$floor_file" || host_rc=$?
if [ "$host_rc" -ne 0 ]; then
  echo "FAIL: paired gate passed but the base is not above the floor; not a slow-host pass"
  echo "bench logs retained in ${bench_tmp}"
  exit "$host_rc"
fi
echo "bench logs retained in ${bench_tmp}"
exit 0
