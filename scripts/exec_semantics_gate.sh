#!/usr/bin/env bash
# exec_semantics_gate.sh — run the test tiers CI could not reach, and PROVE they ran.
#
# Runs the feature combinations whose cfg-gated tests ordinary CI misses.
# Each tier checks positive aggregate and critical-harness execution counts,
# named failures, captured cargo exit status, and capability-skip markers.
# Raw logs remain intact; ANSI decoding is confined to the analysis copy.
# Doctrine and measurement history: docs/gates/exec-semantics-tiers.md.
# Defeat replays: scripts/test_exec_semantics_gate.py.
#
# Usage: [exec|lowering|pkg] | --print-count | --from-log <log> <tier>
set -uo pipefail
cd "$(dirname "$0")/.."

run_tier_cargo() {
  local tier=$1 features=$2 require_toolchain=$3
  local -a tier_env=()
  # Eligible VNNI hosts must execute the manifest-bound native workload.
  # Missing opt-in is a verification failure, so every exec caller sets it.
  if [ "$tier" = exec ]; then
    tier_env+=(MIND_INTDOT_VNNI_VERIFY=1)
  fi
  if [ "$require_toolchain" = 1 ]; then
    tier_env+=(MIND_BENCH_REQUIRE=1)
  fi
  env "${tier_env[@]}" cargo test --no-default-features --features "$features" \
    --no-fail-fast
}

# ---------------------------------------------------------------------------
# TIER DEFINITIONS. Floors protect both aggregate execution and critical harnesses.
# Re-measure with --print-count at every landing; do not infer counts from previous
# runs. The floors and unchanged capability checks are explained in
# docs/gates/exec-semantics-tiers.md.
#
#   tier      features                                   harnesses  executed  floor
#   exec      mlir-build std-surface cross-module-imports      340      2170   2152
#   lowering  std-surface,mlir-lowering                        335      1891   1876
#   pkg       pkg                                              334      1474   1462
#
# Re-measure AT THE TIP of a landing, never mid-wave, and read the numbers off a
# real run's own `harnesses` / `tests executed` lines rather than doing arithmetic
# on a previous measurement. `-- --show-output` is REJECTED as a way to see skip
# markers: it inflates these very floors. Both rules were paid for — the
# measurement history and the demonstrations are in
# docs/gates/exec-semantics-tiers.md §1-2.
# ---------------------------------------------------------------------------
TIERS=(exec lowering pkg)

# `exec` — the executable-semantics tier. 110 mlir-build-gated files, including the
# alias-miscompile gate, the array-OOB bounds-trap gate and the array bounds/dtype
# gate. The C-ABI export feature also executes the manifest-cache regressions;
# without it that entire test target is erased rather than exercised.
FEATURES_exec="mlir-build std-surface cross-module-imports ffi-c-user"
FLOOR_TESTS_exec=2152
FLOOR_HARNESSES_exec=336
# MIND_BENCH_REQUIRE=1 turns "MLIR toolchain missing -> skip" into a hard failure, so
# this tier cannot pass vacuously on a runner where mlir-opt/clang never installed.
# Correct ONLY here: this is the tier that actually enables mlir-build.
REQUIRE_TOOLCHAIN_exec=1

# `lowering` — 8 files / ~53 tests carry
# `#![cfg(all(feature = "std-surface", feature = "mlir-lowering"))]`. ci.yml runs each
# of those features ALONE (the 'Test (gated ...)' and 'Run gated tests ...' steps of
# the build_test job) and never together, so the whole group was erased in both runs.
FEATURES_lowering="std-surface,mlir-lowering"
FLOOR_TESTS_lowering=1876
FLOOR_HARNESSES_lowering=331
# NOT set here. These tiers deliberately build WITHOUT mlir-build, so a target that
# needs a cdylib emit (phase_g_keystone_bootstrap) correctly reports
#   error[build]: cdylib emit requires the 'mlir-build' feature
# and skips. Under MIND_BENCH_REQUIRE=1 that correct skip becomes a hard failure and
# the tier reds for a reason that is not a defect — measured while building this gate.
# The keystone is gated for real by the `exec` tier and by preflight's own 7/7 step.
REQUIRE_TOOLCHAIN_lowering=0

# `pkg` — tests/package_basic.rs + tests/package_traversal.rs are `#![cfg(feature =
# "pkg")]`. ci.yml's feature-compile matrix runs `cargo check --features pkg` but never
# `cargo test`, so neither had ever executed.
FEATURES_pkg="pkg"
FLOOR_TESTS_pkg=1462
FLOOR_HARNESSES_pkg=330
REQUIRE_TOOLCHAIN_pkg=0

# ---------------------------------------------------------------------------
# QUARANTINE — targets RED for a reason NAMED here.
#
# A RATCHET, not an excuse. Any failure in a target NOT listed fails the gate, and a
# listed target that starts PASSING also fails the gate (with an instruction to delete
# its line). The list may only ever shrink.
#
# Every entry is a genuine, separately-scoped defect that this tier was HIDING — they
# are the payload of the finding, not collateral from it.
# Each entry needs its own fix; none is closed by this gate.
#
# An entry is the name the triage reports: an integration target's bare name, or one
# of the `@`-prefixed pseudo-targets (`@lib`, `@doc`, `@bin:<name>`, `@bench:<name>`)
# that stand for the harness kinds cargo does not name with `--test`. See
# rerun_selector() below.
#
# CRITICAL_<tier> — per-harness minimums for the gates each tier NAMES as the reason
# it exists. An AGGREGATE floor cannot protect a SPECIFIC test: re-erasing exactly the
# two files this gate was written to defend once produced ALL TIERS OK / exit 0.
# Format: "<target> <min_executed>", checked INDEPENDENTLY of the floors.
#
# PER-TIER, not global; the count compared is tests EXECUTED, not passed; and a row is
# added by the SAME change that adds the gate it protects — a new gate whose only
# protection is the aggregate floor is a gate anyone may delete. Every minimum below
# was measured IN ITS OWN TIER, never assumed from another.
# The full argument and the per-row measurements: docs/gates/exec-semantics-tiers.md §5.
CRITICAL_exec=(
  "determinism_veto_control 2"       # the determinism-by-default veto positive control
  "alias_miscompile_run 1"    # the alias-miscompile regression gate
  "array_oob_trap_run 1"      # ARRAY_OOB_CONTRACT=DETERMINISTIC_BOUNDS_TRAP
  "fail_closed_capability_skip 20"  # measured 22; the capability-skip helper contract
  "fail_closed_capability_skip_env 4"  # measured 6; the MIND_BENCH_REQUIRE READER itself
  "fail_closed_capability_skip_stub_exec 4"  # its end-to-end leg, spawned for real (unix)
  "capability_refusal_cause_scan 4"  # measured 5; every capability refusal in src/ mints its cause
  "fail_open_skip_site_ratchet 12"  # measured 14; the fail-open skip prohibition
  "closed_declaration_wiring 4"     # the ci.yml CLOSED note rests on a live gate
  "harness_portability 2"           # no test file may red a matrix row it cannot run on
  "mindc_artifact_name 7"           # measured 7 under this feature set; the artifact NAME's one owner
  "harness_scratch_isolation 3"     # scratch isolation for every wedge gate's artifacts
  "module_size_ratchet 5"           # the 800-line ceiling and its budget table
  "mindc_cache_build_inputs 3"      # explicit entry, manifest exports, native sources
  "const_table_build_determinism 1" # 8 ASLR-on builds of a const-table .so -> 1 hash
)
# The two largest members of the std-surface+mlir-lowering group that no CI run
# enabled BOTH features for; 26 of the group's 57 tests live in these two files.
CRITICAL_lowering=(
  "determinism_veto_control 2"       # the determinism-by-default veto positive control
  "extern_c_phase_a 1"
  "extern_c_phase_b 1"
  "fail_closed_capability_skip 20"
  "fail_closed_capability_skip_env 4"  # measured 6; the MIND_BENCH_REQUIRE READER itself
  "fail_closed_capability_skip_stub_exec 4"
  "capability_refusal_cause_scan 4"
  "fail_open_skip_site_ratchet 12"
  "closed_declaration_wiring 4"
  "harness_portability 2"
  "harness_scratch_isolation 3"     # scratch isolation for every wedge gate's artifacts
  "module_size_ratchet 5"           # the 800-line ceiling and its budget table
)
# The entire reason the `pkg` tier exists: ci.yml only ever `cargo check`ed pkg.
CRITICAL_pkg=(
  "package_basic 1"
  "package_traversal 1"
  "fail_closed_capability_skip 20"
  "fail_closed_capability_skip_env 4"  # measured 6; the MIND_BENCH_REQUIRE READER itself
  "fail_closed_capability_skip_stub_exec 4"
  "capability_refusal_cause_scan 4"
  "fail_open_skip_site_ratchet 12"
  "closed_declaration_wiring 4"
  "harness_portability 2"
  "harness_scratch_isolation 3"     # scratch isolation for every wedge gate's artifacts
  "module_size_ratchet 5"           # the 800-line ceiling and its budget table
)

QUARANTINE_exec=(
)
QUARANTINE_lowering=()
QUARANTINE_pkg=()

# ENV_TOLERATED — may fail LOCALLY for a documented environmental reason and pass in
# CI's fresh checkout (or the reverse). Neither outcome fails the gate. Kept distinct
# from QUARANTINE so a real regression is never hidden behind a local-only excuse.
#
#   g2_differential_mlir      — dlopens the gitignored, possibly stale in-tree
#       libmindc_mind.so. scripts/preflight.sh already excludes it for this reason.
#   mindfuzz_cross_substrate  — needs the MLIR toolchain; soft-skips in ci.yml's
#       build_test job. Also already excluded by scripts/preflight.sh.
ENV_TOLERATED_exec=(g2_differential_mlir)
# `lowering` and `pkg` are EMPTY on purpose. Both g2_differential_mlir and
# mindfuzz_cross_substrate now carry `required-features = ["mlir-build"]` in
# Cargo.toml, so cargo does not build them in tiers that cannot run them, and an
# entry naming a target a tier never builds is dead configuration — the TOLERANCE
# SHRINK-RATCHET below reds a tier on exactly that shape. Structural absence beats
# runtime tolerance; the retirement history is in
# docs/gates/exec-semantics-tiers.md §4.
ENV_TOLERATED_lowering=()
ENV_TOLERATED_pkg=()

# ---------------------------------------------------------------------------
print_only=0
# --from-log: analyse a log this script already wrote (it keeps them at a stable
# path precisely so the "which test, and why" question can be answered after the
# fact) WITHOUT running cargo. It is analysis, never evidence: the pass banner is
# deliberately withheld in this mode, so a green CI line can never be produced by
# replaying a saved log.
from_log=""
want=()
while [ $# -gt 0 ]; do
  case "$1" in
    --print-count) print_only=1 ;;
    --from-log)
      shift
      from_log="${1:-}"
      [ -n "$from_log" ] || { echo "--from-log needs a log path" >&2; exit 2; } ;;
    exec|lowering|pkg) want+=("$1") ;;
    *) echo "usage: $0 [--print-count] [--from-log <log> <tier>] [exec|lowering|pkg ...]" >&2; exit 2 ;;
  esac
  shift
done
if [ -n "$from_log" ]; then
  [ -r "$from_log" ] || { echo "--from-log: cannot read $from_log" >&2; exit 2; }
  # The tier selects the floors, quarantine and ENV_TOLERATED list to judge the
  # log against, so it cannot be inferred from the file.
  [ ${#want[@]} -eq 1 ] || { echo "--from-log needs exactly one tier" >&2; exit 2; }
fi
[ ${#want[@]} -eq 0 ] && want=("${TIERS[@]}")

# cargo names four harness KINDS in its rerun hint, and only one of them is a
# `--test <name>` integration target. The other three (`--lib`, `--doc`,
# `--bin <n>`, `--bench <n>`) are given PSEUDO-TARGET names below so they can be
# triaged, reported and quarantined exactly like a real target. The `@` prefix
# cannot collide with a cargo target name (those are file stems: [A-Za-z0-9_-]),
# so `@lib` can never be confused with an integration test called `lib`.
#
#   @lib            src/lib.rs unit tests        rerun: --lib
#   @doc            documentation tests          rerun: --doc
#   @bin:<name>     a binary's unit tests        rerun: --bin <name>
#   @bench:<name>   a bench target's tests       rerun: --bench <name>
rerun_selector() {
  case "$1" in
    @lib)     echo "--lib" ;;
    @doc)     echo "--doc" ;;
    @bin:*)   echo "--bin ${1#@bin:}" ;;
    @bin)     echo "--bins" ;;
    @bench:*) echo "--bench ${1#@bench:}" ;;
    *)        echo "--test $1" ;;
  esac
}

# Cargo honours CARGO_TERM_COLOR=always even when stdout/stderr are redirected,
# so the durable raw log can contain ANSI control sequences in front of tokens
# this gate anchors at column zero (`Running`, `test result:` and `error:`).  Parse
# a decoded copy and leave the raw log byte-for-byte intact for CI diagnosis.
#
# CSI covers Cargo/rustc's SGR colour output. OSC and the remaining single-byte
# ANSI escape forms are handled too, while a truncated/unknown escape fails
# closed rather than silently changing the evidence the gate can see.
normalise_cargo_log() {
  python3 - "$1" "$2" <<'PY'
import re
import sys
from pathlib import Path

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
data = source.read_bytes()

# OSC: ESC ] ... BEL, or ESC ] ... ESC \
data = re.sub(rb"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)", b"", data)
# CSI: ESC [ parameter bytes, intermediate bytes, final byte.
data = re.sub(rb"\x1b\[[0-?]*[ -/]*[@-~]", b"", data)
# Remaining complete two-byte Fe escapes. `[` and `]` here would be an
# incomplete CSI/OSC sequence and therefore stay behind for the refusal below.
data = re.sub(rb"\x1b[@-Z\\^_]", b"", data)

if b"\x1b" in data:
    print(
        f"FAIL: unsupported or truncated ANSI escape in raw cargo log: {source}",
        file=sys.stderr,
    )
    raise SystemExit(1)

destination.write_bytes(data)
PY
}

# Shared marker classification stays beside the tier counts it protects.
# Doctrine and historical measurements: docs/gates/exec-semantics-tiers.md.

# True when $1 is one of the remaining arguments.
in_list() { local n=$1; shift; local e; for e in "$@"; do [ "$e" = "$n" ] && return 0; done; return 1; }

# consume_skip_markers <log> <tier> <require_toolchain> [env-tolerated-target...]
#
# `ran=0` is a failure to EXECUTE — a DIFFERENT question from "did it pass", and
# the two need separate verdicts because the fixes are unrelated. Fail-closed by
# default: a ran=0 from any target not named in ENV_TOLERATED_<tier> and not
# carrying a `class=` the gate helper declares legal here reds the tier, so the
# next gate to print a marker is enforced without anyone wiring it up.
#
# Sets four arrays in the CALLER's scope, deliberately un-`local`: `zero_fatal`,
# `zero_ok`, `zero_expected` and `env_skipped`. The tier loop reads them BY NAME
# — the failure triage grants environmental tolerance ONLY on `env_skipped` —
# so they are the interface, not an implementation detail to be renamed.
#
# Doctrine, and the measurement behind every rule here (why the marker had to
# leave libtest's capture to be seen at all, why attribution resets at every
# compilation-unit boundary, and which `class=` is fatal where):
# docs/gates/exec-semantics-tiers.md §3.
consume_skip_markers() {
  local log=$1 tier=$2 require_toolchain=$3
  shift 3
  local tolerated=("$@")
  local ret=0
  local -a markers
  local m mtarget mname mran mclass z
  mapfile -t markers < <(
    awk '
      # Clear attribution at EVERY compilation-unit boundary, not just at
      # `Running tests/*.rs`: cargo also emits `Doc-tests`, `Running benches/...`
      # and `Running unittests src/lib.rs`, and without this reset a marker from
      # one of those inherits the last tests/<t>.rs identity — which can silently
      # downgrade a genuine ran=0 from fatal to tolerated. Fails closed to
      # <unattributed> instead.
      /^[[:space:]]*(Running|Doc-tests) / { t = "" }
      match($0, /Running tests\/[A-Za-z0-9_]+\.rs/) {
        t = substr($0, RSTART + 14, RLENGTH - 17); next
      }
      /SDLC-GATE [A-Za-z0-9_.-]+ ran=[0-9]+/ {
        name = ""; ran = ""; cls = ""
        for (i = 1; i < NF; i++) if ($i == "SDLC-GATE") { name = $(i + 1); break }
        for (i = 1; i <= NF; i++) if ($i ~ /^ran=[0-9]+$/) { ran = substr($i, 5); break }
        # The absence CLASS the producer stamped (tests/common/gate.rs::Absent).
        # A producer that stamps none is `unclassified` and stays FAIL-CLOSED
        # below — a marker may never buy tolerance by omitting a field.
        for (i = 1; i <= NF; i++) if ($i ~ /^class=[a-z]+$/) { cls = substr($i, 7); break }
        if (name != "" && ran != "")
          printf "%s %s %s %s\n", (t == "" ? "<unattributed>" : t), name, ran,
                                   (cls == "" ? "unclassified" : cls)
      }' "$log" | sort -u
  )
  zero_fatal=()
  zero_ok=()
  # A ran=0 whose CLASS the gate helper already declared legal here. Counted and
  # printed on every run, never fatal -- and never a SECOND rule, which is the
  # whole reason the class travels in the marker:
  #   class=optional   gate.rs::is_fail_closed answers NO for Absent::OptionalInput
  #       in EVERY tier. A tier that failed on it would be a second, contradictory
  #       owner of one rule; what was missing was VISIBILITY, never fatality.
  #   class=toolchain in a REQUIRE_TOOLCHAIN_<tier>=0 tier
  #       REQUIRE_TOOLCHAIN_<tier> IS this tier's answer to "may a toolchain
  #       absence skip here", and gate.rs applies the identical rule per-process
  #       through MIND_BENCH_REQUIRE. In a =1 tier the helper PANICS instead of
  #       marking, so a toolchain marker arriving there stays FATAL below.
  # Neither arm is named in ENV_TOLERATED, deliberately: that list ALSO excuses a
  # target's FAILURES, and only one of those two permissions is being granted.
  zero_expected=()
  # Targets that ACTUALLY reported an environmental skip. This is the evidence
  # tolerance is granted on below -- not the mere fact of being listed.
  env_skipped=()
  for m in ${markers[@]+"${markers[@]}"}; do
    read -r mtarget mname mran mclass <<<"$m"
    [ "$mran" = 0 ] || continue
    if in_list "$mtarget" ${tolerated[@]+"${tolerated[@]}"}; then
      zero_ok+=("$mname (tests/$mtarget.rs)")
      env_skipped+=("$mtarget")
    elif [ "$mclass" = optional ] \
      || { [ "$mclass" = toolchain ] && [ "$require_toolchain" != 1 ]; }; then
      zero_expected+=("$mname (tests/$mtarget.rs)")
    else
      zero_fatal+=("$mname (tests/$mtarget.rs)")
    fi
  done
  if [ ${#zero_fatal[@]} -gt 0 ]; then
    echo
    echo "FAIL[$tier]: ${#zero_fatal[@]} gate(s) reported ran=0 — they did not EXECUTE."
    for z in "${zero_fatal[@]}"; do echo "  - $z"; done
    echo "  A failure to EXECUTE, not a failure to pass: these asserted NOTHING, so the"
    echo "  tier's green says nothing about what they defend. Each is one of two shapes:"
    echo "    * class=toolchain here, where REQUIRE_TOOLCHAIN_$tier=$require_toolchain demands a real"
    echo "      backend — install the toolchain; a skip asserts NOTHING. (The gate helper"
    echo "      PANICS on this, so a marker means a producer that never read the rule.)"
    echo "    * no class= field at all — a producer outside tests/common/gate.rs. Stamp its"
    echo "      absence class, or — if the skip is genuinely environmental — name the"
    echo "      target in ENV_TOLERATED_$tier with the reason, where it stays counted."
    ret=1
  fi
  if [ ${#zero_ok[@]} -gt 0 ]; then
    echo "gates that did NOT run : ${#zero_ok[@]}  (env-tolerated ran=0 — a skip, never a pass)"
    for z in "${zero_ok[@]}"; do echo "  - $z"; done
  fi
  if [ ${#zero_expected[@]} -gt 0 ]; then
    echo "gates that did NOT run : ${#zero_expected[@]}  (class the helper declares legal here — a skip, never a pass)"
    for z in "${zero_expected[@]}"; do echo "  - $z"; done
  fi
  return $ret
}

# check_dead_tolerance <log> <tier> <harnesses> <floor_h> [env-tolerated-target...]
#
# The missing half of the quarantine ratchet. A tolerated name absent from this
# tier's own `Running tests/<t>.rs` set is tolerance for something the tier never
# ran: it cannot excuse anything today and silently pre-approves a ran=0 the day
# the target becomes buildable. Judged against the tier's OWN log, never a
# hand-copied expectation. See docs/gates/exec-semantics-tiers.md §4.
check_dead_tolerance() {
  local log=$1 tier=$2 harnesses=$3 floor_h=$4
  shift 4
  local tolerated=("$@")
  local ret=0
  local -a tier_targets dead_tolerance
  local tname d
  mapfile -t tier_targets < <(
    grep -oE 'Running tests/[A-Za-z0-9_]+\.rs' "$log" | sed -e 's|Running tests/||' -e 's|\.rs$||' | sort -u
  )
  dead_tolerance=()
  for tname in ${tolerated[@]+"${tolerated[@]}"}; do
    in_list "$tname" ${tier_targets[@]+"${tier_targets[@]}"} || dead_tolerance+=("$tname")
  done
  if [ ${#dead_tolerance[@]} -gt 0 ] && [ "$harnesses" -ge "$floor_h" ]; then
    echo
    echo "FAIL[$tier]: ENV_TOLERATED_$tier names ${#dead_tolerance[@]} target(s) this tier never built."
    for d in "${dead_tolerance[@]}"; do echo "  - $d"; done
    echo "  Tolerance for a target that does not run here is dead configuration: it"
    echo "  cannot excuse anything today and silently pre-approves a ran=0 the day the"
    echo "  target becomes buildable. Remove the entry (structural absence beats runtime"
    echo "  tolerance), or restore the target to this tier."
    ret=1
  fi
  return $ret
}

overall=0
for tier in "${want[@]}"; do
  eval "features=\$FEATURES_$tier"
  eval "floor_t=\$FLOOR_TESTS_$tier"
  eval "floor_h=\$FLOOR_HARNESSES_$tier"
  eval "require_toolchain=\$REQUIRE_TOOLCHAIN_$tier"
  eval "quarantine=(\"\${QUARANTINE_$tier[@]}\")"
  eval "tolerated=(\"\${ENV_TOLERATED_$tier[@]}\")"
  eval "critical=(\${CRITICAL_$tier[@]+\"\${CRITICAL_$tier[@]}\"})"

  echo
  echo "=============================================================="
  echo "== tier '$tier':  cargo test --no-default-features --features \"$features\""
  echo "==   MIND_BENCH_REQUIRE=$require_toolchain (1 = a missing MLIR toolchain is a FAILURE, not a skip)"
  echo "=============================================================="
  # Keep the raw log at a STABLE path, not a mktemp that is deleted on the way out:
  # when this gate fails, the next question is always "which test, and why", and a
  # 40-line tail cannot answer it. CI uploads this path even on failure, and the
  # file is left in place for the developer.
  raw_log="${MIND_TIER_LOG_DIR:-${TMPDIR:-/tmp}}/mind-tier-$tier.log"
  mkdir -p "$(dirname "$raw_log")"

  if [ -n "$from_log" ]; then
    raw_log="$from_log"
    # cargo's exit status is RECORDED IN THE LOG (see the marker written below), so
    # a replayed log is judged on the same evidence as the run that produced it.
    # Before that, this mode read "n/a" and the status assert below could not run at
    # all — the analysis silently answered a weaker question than the live gate.
    # A log written before the marker existed reports `unknown`: that leg of the
    # verdict is then honestly UNAVAILABLE rather than silently assumed to be 0.
    :
  else
    run_tier_cargo "$tier" "$features" "$require_toolchain" >"$raw_log" 2>&1
    cargo_status=$?
    echo "MIND_TIER_CARGO_EXIT=$cargo_status" >>"$raw_log"
  fi

  analysis_log=$(mktemp "${TMPDIR:-/tmp}/mind-tier-analysis.XXXXXX") || {
    echo "FAIL[$tier]: could not allocate an ANSI-normalised analysis log." >&2
    exit 1
  }
  trap 'rm -f "${analysis_log:-}"' EXIT
  if ! normalise_cargo_log "$raw_log" "$analysis_log"; then
    echo "FAIL[$tier]: could not decode the raw cargo log; refusing to grade partial evidence." >&2
    echo "full raw log: $raw_log" >&2
    exit 1
  fi

  if [ -n "$from_log" ]; then
    cargo_status=$(sed -n 's/^MIND_TIER_CARGO_EXIT=\([0-9]\{1,\}\)$/\1/p' "$analysis_log" | tail -1)
    cargo_status="${cargo_status:-unknown}"
    echo "ANALYSIS-ONLY: re-reading $raw_log; NO tests were run by this invocation."
    echo "               recorded cargo exit: $cargo_status"
  fi

  # --- POSITIVE-COUNT ASSERT (the anti-silent-zeroing core) ----------------
  harnesses=$(grep -c '^test result:' "$analysis_log")
  read -r passed failed ignored <<<"$(
    grep '^test result:' "$analysis_log" | awk '{p+=$4; f+=$6; i+=$8} END {print p+0, f+0, i+0}'
  )"
  executed=$((passed + failed + ignored))

  echo "harnesses      : $harnesses  (floor $floor_h)"
  echo "tests executed : $executed  (floor $floor_t)  [passed=$passed failed=$failed ignored=$ignored]"

  if [ "$print_only" = 1 ]; then
    rm -f "$analysis_log"
    analysis_log=""
    trap - EXIT
    continue
  fi

  rc=0
  # A test file that does not COMPILE also yields zero harnesses, and it is a very
  # different bug from a cfg-erased one — distinguish them, or the gate sends the
  # reader hunting a feature flag when the real problem is a broken test source.
  # (Hit for real while building this gate: an in-flight test file calling a
  # non-existent `TempDir::join` failed the whole build with harnesses=0.)
  mapfile -t build_errs < <(
    grep -E '^error(\[[A-Z0-9]+\])?: ' "$analysis_log" | grep -v '^error: test failed' | sort -u | head -5
  )
  # --- CRITICAL HARNESS MINIMUMS (independent of the aggregate floors) ------
  # Parsed per-target from the `Running ...` / `test result:` pairing, so erasing
  # one file is visible even when the tier total still clears its floor. Runs
  # BEFORE the floor chain and sets `overall` on its own, so the two verdicts are
  # independent rather than mutually exclusive.
  crit_bad=()
  for spec in ${critical[@]+"${critical[@]}"}; do
    ctarget="${spec%% *}"; cmin="${spec##* }"
    # Anchor on the exact `Running tests/<target>.rs` line cargo prints, not a
    # substring: `Running .*foo-` also matches a sibling target named
    # `bar_foo`, which would let an erased gate borrow another file's count.
    cran=$(awk -v t="$ctarget" '
      index($0, "Running tests/" t ".rs") { seen=1; next }
      seen && /^test result:/ {
        n=0
        for (i=1;i<=NF;i++)
          if ($i=="passed;" || $i=="failed;" || $i=="ignored;") n += $(i-1)
        print n; exit
      }' "$analysis_log")
    cran=${cran:-0}
    if [ "$cran" -lt "$cmin" ]; then
      crit_bad+=("$ctarget executed $cran test(s) (need >=$cmin)")
    fi
  done
  if [ ${#crit_bad[@]} -gt 0 ]; then
    overall=1
    echo "FAIL[$tier]: a CRITICAL harness did not run its tests."
    for b in "${crit_bad[@]}"; do echo "  $b"; done
    echo "  These are named in this script's header as the reason it exists. They are"
    echo "  checked per-harness precisely because an aggregate floor cannot protect a"
    echo "  specific test: an erased file still prints 'ok. 0 passed' and the tier"
    echo "  total can absorb it inside its slack."
  fi

  if [ ${#build_errs[@]} -gt 0 ] && [ "$harnesses" -lt "$floor_h" ]; then
    echo
    echo "FAIL[$tier]: the tier did not BUILD — this is a compile error, not a cfg problem."
    for e in "${build_errs[@]}"; do echo "  $e"; done
    echo "  full raw log: $raw_log"
    rc=1
  elif [ "$executed" -lt "$floor_t" ] || [ "$harnesses" -lt "$floor_h" ]; then
    echo
    echo "FAIL[$tier]: the tier did not prove it ran."
    echo "  executed=$executed (need >=$floor_t), harnesses=$harnesses (need >=$floor_h)"
    echo "  A COLLAPSE here means test files were cfg'd OUT by a missing feature, NOT"
    echo "  that tests were deleted — cargo prints 'ok. 0 passed' and exits 0 for an"
    echo "  erased file. Check that --features \"$features\" is intact."
    rc=1
  fi

  # --- SKIP-MARKER CONSUMER: ran=0 is a failure to EXECUTE -----------------
  # Sets `zero_fatal`, `zero_ok`, `zero_expected` and `env_skipped` in this
  # scope: the failure triage below reads `env_skipped` as the evidence
  # environmental tolerance is granted on, and the ok banner reads the other two
  # so a green tier can never hide that it is green ABOUT LESS than it looks.
  consume_skip_markers "$analysis_log" "$tier" "$require_toolchain" \
    ${tolerated[@]+"${tolerated[@]}"} || rc=1

  # --- TOLERANCE SHRINK-RATCHET (the missing half of the quarantine ratchet) --
  check_dead_tolerance "$analysis_log" "$tier" "$harnesses" "$floor_h" \
    ${tolerated[@]+"${tolerated[@]}"} || rc=1

  # --- FAILURE TRIAGE against the quarantine ratchet -----------------------
  # EVERY harness kind, not just `--test`. This sed used to match one shape:
  #   error: test failed, to rerun pass `--test <name>`
  # cargo prints `--lib` for src/lib.rs unit tests, `--doc` for doctests,
  # `--bin <n>` for a binary's tests and `--bench <n>` for a bench target — none of
  # which produced an entry here. `unexpected` is built only from this list, so a
  # failing lib unit test, doctest or bin test was attributed to NOTHING and the
  # tier printed `ok[...]`. Measured on a synthetic tier log carrying a lib harness
  # with `3 failed`: `failed=3` printed, `0 failing` printed, exit 0.
  # Non---test kinds arrive as the `@`-prefixed pseudo-targets rerun_selector maps
  # back to a runnable command; they are quarantinable like any other target.
  mapfile -t failing < <(
    sed -n \
      -e 's/^error: test failed, to rerun pass `--test \([A-Za-z0-9_-]*\)`.*/\1/p' \
      -e 's/^error: test failed, to rerun pass `--bin \([A-Za-z0-9_-]*\)`.*/@bin:\1/p' \
      -e 's/^error: test failed, to rerun pass `--bench \([A-Za-z0-9_-]*\)`.*/@bench:\1/p' \
      -e 's/^error: test failed, to rerun pass `--lib`.*/@lib/p' \
      -e 's/^error: test failed, to rerun pass `--doc`.*/@doc/p' \
      "$analysis_log" | sort -u
  )

  # A CRASH is never tolerable, whatever the target's environmental status.
  #
  # Environmental tolerance exists for a real reason: g2_differential_mlir dlopens a
  # gitignored, possibly-stale in-tree .so, so it can fail LOCALLY for reasons that say
  # nothing about the code. But that reason covers a MISSING or STALE oracle — it does
  # not cover the compiler DYING on a fixture. Those are different events and only one
  # of them is environmental.
  #
  # Measured 2026-08-29: with g2 failing, a replay still printed "ALL TIERS OK /
  # GATE EXIT=0", because the tolerated check below `continue`s before `unexpected` is
  # built. The harness had just been fixed to DETECT a crash (MIND_CRASH); this line is
  # what stopped that detection from ever GATING anything.
  #
  # The harness prints an unmistakable marker when it classifies a crash, so tolerance
  # is withdrawn for exactly that case and left intact for the environmental one.
  crashed=()
  # The per-fixture marker is `MIND_CRASH <path>.mind  [reason]`. Anchor on that exact
  # shape: the SUMMARY line also carries the token (`... / 0 MIND_CRASH / ...`), and the
  # looser `MIND_CRASH [^0]` matched the `/` immediately after it -- so EVERY run that
  # printed a summary was reported as a crash, a pure divergence with `0 MIND_CRASH`
  # included. Right exit code, wrong cause, and it would send someone hunting a segfault
  # that never happened.
  if grep -qE "GATE FAILED: the pure-MIND compiler CRASHED|MIND_CRASH [A-Za-z0-9_/.-]+[.]mind" "$analysis_log" 2>/dev/null; then
    mapfile -t crashed < <(grep -oE "MIND_CRASH [A-Za-z0-9_/.-]+[.]mind" "$analysis_log" | awk '{print $2}' | sort -u)
    echo
    echo "FAIL[$tier]: the pure-MIND compiler CRASHED. Tolerance does NOT apply —"
    echo "  an unsupported construct returns a null handle; these terminated abnormally."
    for c in ${crashed[@]+"${crashed[@]}"}; do echo "  - $c"; done
    rc=1
  fi

  unexpected=()
  for t in "${failing[@]}"; do
    in_list "$t" ${quarantine[@]+"${quarantine[@]}"} && continue
    # Tolerance is granted ONLY for the outcome it exists for: the target could not
    # RUN here. The evidence is the target's own `ran=0` marker, collected into
    # `env_skipped` above.
    #
    # This was previously written the other way round: tolerance applied by DEFAULT
    # and was withdrawn for one named bad outcome (a crash). That is a denylist, and
    # it fails open on every outcome nobody thought to name. The one that mattered is
    # a byte-DIVERGENCE between the pure-MIND and Rust compilers: it prints no crash
    # marker, so `g2_differential_mlir` could report 99 diverging fixtures while this
    # tier printed `ok[exec]` and exited 0 -- the differential gate, whose entire
    # purpose is catching that disagreement, could not fail the build it gates.
    #
    # A target that RAN and produced findings is a real failure, whatever list it is
    # on. Environmental tolerance covers a missing oracle, never a wrong answer.
    if in_list "$t" ${tolerated[@]+"${tolerated[@]}"}; then
      if in_list "$t" ${env_skipped[@]+"${env_skipped[@]}"}; then
        continue
      fi
      unexpected+=("$t")
      continue
    fi
    unexpected+=("$t")
  done
  if [ ${#unexpected[@]} -gt 0 ]; then
    echo
    echo "FAIL[$tier]: ${#unexpected[@]} target(s) failed that are NOT quarantined:"
    for t in "${unexpected[@]}"; do
      echo "  - $t   (rerun: cargo test --no-default-features --features \"$features\" $(rerun_selector "$t"))"
    done
    rc=1
  fi

  # A quarantined target that now PASSES must leave the list, or the quarantine
  # quietly grows into a permanent excuse.
  fixed=()
  for t in ${quarantine[@]+"${quarantine[@]}"}; do
    in_list "$t" ${failing[@]+"${failing[@]}"} || fixed+=("$t")
  done
  if [ ${#fixed[@]} -gt 0 ]; then
    echo
    echo "FAIL[$tier]: ${#fixed[@]} quarantined target(s) now PASS — delete them from"
    echo "      QUARANTINE_$tier in scripts/exec_semantics_gate.sh (the list may only shrink):"
    for t in "${fixed[@]}"; do echo "  - $t"; done
    rc=1
  fi

  # --- AGGREGATE `failed` COUNT: asserted, not decorated -------------------
  # `failed` is summed from every `test result:` line and printed on the summary
  # line above — and until now it was compared to nothing, so a tier could print
  # `failed=3` and `ok[...]` in the same breath. It is asserted here rather than
  # left to the triage above because the two rest on DIFFERENT evidence: the
  # triage greps one cargo message whose wording cargo owns, this counts the
  # harness's own report. A cargo that reworded (or dropped) the rerun hint would
  # take the triage with it; the count survives.
  #
  # Attribution is per-harness so the shrink-only quarantine keeps working: a
  # failure inside a QUARANTINE_<tier> target, or inside a target that reported an
  # env-tolerated ran=0, is ACCOUNTED FOR. Everything else is not, and reds the
  # tier — which is the whole point.
  mapfile -t failed_blocks < <(
    awk '
      # Attribute at every compilation-unit boundary, mirroring the marker
      # consumer above so one target can never inherit another one'"'"'s identity.
      /^[[:space:]]*(Running|Doc-tests) / {
        key = "<unattributed>"
        if (match($0, /Running tests\/[A-Za-z0-9_-]+\.rs/))
          key = substr($0, RSTART + 14, RLENGTH - 17)
        else if ($0 ~ /Running unittests src\/lib\.rs/)
          key = "@lib"
        else if (match($0, /Running unittests src\/bin\/[A-Za-z0-9_-]+\.rs/))
          key = "@bin:" substr($0, RSTART + 26, RLENGTH - 29)
        else if ($0 ~ /Running unittests src\/main\.rs/) {
          # The bin NAME is not on this line; recover it from the deps binary
          # (`.../deps/mindc-9f8a…`) so this key matches the `@bin:<name>` the
          # rerun hint yields for the same failure.
          key = "@bin"
          if (match($0, /deps\/[A-Za-z0-9_-]+-[0-9a-f]+\)/)) {
            b = substr($0, RSTART + 5, RLENGTH - 6)
            sub(/-[0-9a-f]+$/, "", b)
            if (b != "") key = "@bin:" b
          }
        }
        else if (match($0, /Running benches\/[A-Za-z0-9_-]+\.rs/))
          key = "@bench:" substr($0, RSTART + 16, RLENGTH - 19)
        else if ($0 ~ /^[[:space:]]*Doc-tests /)
          key = "@doc"
        next
      }
      /^test result:/ {
        f = 0
        for (i = 1; i <= NF; i++) if ($i == "failed;") f = $(i - 1) + 0
        if (f > 0) print (key == "" ? "<unattributed>" : key) " " f
        key = ""
      }' "$analysis_log"
  )
  unaccounted=()
  unaccounted_n=0
  for fb in ${failed_blocks[@]+"${failed_blocks[@]}"}; do
    fkey="${fb%% *}"; fn="${fb##* }"
    in_list "$fkey" ${quarantine[@]+"${quarantine[@]}"} && continue
    in_list "$fkey" ${env_skipped[@]+"${env_skipped[@]}"} && continue
    unaccounted+=("$fkey reported $fn failing test(s)")
    unaccounted_n=$((unaccounted_n + fn))
  done
  if [ "$unaccounted_n" -gt 0 ]; then
    echo
    echo "FAIL[$tier]: $unaccounted_n test(s) FAILED in target(s) nothing accounts for."
    for u in "${unaccounted[@]}"; do echo "  - $u"; done
    echo "  The tier's own summary line prints this count; it is now asserted. A tier"
    echo "  cannot be green while a test it ran is red."
    rc=1
  fi

  # --- CARGO'S OWN EXIT STATUS: asserted, not printed ----------------------
  # It was captured and used exactly once, in an echo inside the failure branch —
  # so cargo could report failure while this gate reported success. A non-zero
  # exit is only acceptable when the log NAMES the target(s) responsible and the
  # ratchet above has already judged them; a non-zero exit with nothing to
  # attribute it to (a link error, a harness that aborted before printing its
  # result, a future cargo whose wording moved) fails closed.
  # `unknown` = a pre-marker log replayed through --from-log: unavailable, and
  # said so above, rather than assumed green.
  if [ "$cargo_status" != 0 ] && [ "$cargo_status" != unknown ] && [ ${#failing[@]} -eq 0 ] \
     && [ "$unaccounted_n" -eq 0 ]; then
    echo
    echo "FAIL[$tier]: cargo exited $cargo_status but no failing target could be named."
    echo "  Nothing in the log attributes it: no 'error: test failed, to rerun pass ...'"
    echo "  hint and no harness reporting failed>0. Something failed OUTSIDE the test"
    echo "  results (a link/build failure, an aborted harness, or a cargo message this"
    echo "  gate does not parse). Fail-closed: read the log, then teach the triage."
    echo "  full raw log: $raw_log"
    rc=1
  fi

  # `crit_bad` is checked here too: it sets `overall` on its own (the two verdicts
  # are independent), but printing "ok[$tier]" alongside "GATE FAILED" would be the
  # same false-green shape this whole script exists to remove.
  if [ "$rc" = 0 ] && [ ${#crit_bad[@]} -eq 0 ]; then
    echo "ok[$tier]: $executed tests executed across $harnesses harnesses; ${#failing[@]} failing"
    echo "          target(s), all accounted for (quarantine ${#quarantine[@]}, env-tolerated ${#tolerated[@]})."
    # An env-tolerated ran=0 does not red the tier, but it must never vanish into
    # the ok line: the tier is green ABOUT LESS than it looks.
    zero_seen=$(( ${#zero_ok[@]} + ${#zero_expected[@]} ))
    if [ "$zero_seen" -gt 0 ]; then
      echo "          NOTE: $zero_seen gate(s) above reported ran=0 and asserted NOTHING."
    fi
  else
    echo
    echo "cargo exit was $cargo_status; failing targets: ${failing[*]:-none}"
    echo "full raw log: $raw_log"
    for t in ${unexpected[@]+"${unexpected[@]}"}; do
      echo "--- tier '$tier' :: $t   (rerun: $(rerun_selector "$t")) ---"
      case "$t" in
        # A pseudo-target has no `Running tests/<t>.rs` block to anchor on: its
        # harness header is `Running unittests src/...` or `Doc-tests <crate>`.
        # Anchor on the harness's own failure list instead, or the reader gets an
        # empty excerpt for exactly the failures this gate was just taught to see.
        @*)
          awk '/^failures:$/ {on=1} on {print} on && /^test result:/ {exit}' "$analysis_log" | head -40 ;;
        *)
          awk -v t="tests/$t.rs" '$0 ~ ("Running " t) {on=1} on {print} on && /^test result:/ {exit}' "$analysis_log" \
            | grep -Ev '^test .* \.\.\. ok$' | head -40 ;;
      esac
    done
    overall=1
  fi
  rm -f "$analysis_log"
  analysis_log=""
  trap - EXIT
done

echo
if [ "$print_only" = 1 ]; then
  echo "--print-count: re-derive the floors from the numbers above; this mode never fails"
  exit 0
fi
if [ -n "$from_log" ]; then
  # No pass banner here, ever. --from-log ran no tests, so nothing it prints may
  # be greppable as proof that the tier is green.
  echo "ANALYSIS-ONLY (--from-log): judged a saved log; NO tests were run. rc=$overall"
  exit $overall
fi
[ "$overall" = 0 ] && echo "ALL TIERS OK" || echo "GATE FAILED — see FAIL[...] above"
exit $overall
