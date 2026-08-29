#!/usr/bin/env bash
# exec_semantics_gate.sh — run the test tiers CI could not reach, and PROVE they ran.
#
# WHY THIS EXISTS
# ---------------
# Measured at f2a2d87d: 123 integration-test files / 262 test functions were reachable
# by NO `cargo test` invocation in .github/workflows/ci.yml or scripts/preflight.sh.
# Every broad `cargo test` in CI runs `--no-default-features` with at most
# {std-surface, cross-module-imports, autodiff, mlir-lowering, cpu-buffers}; the only
# runs that enable `mlir-build` are `--test <single-target>` SELECTORS, and exactly ONE
# of the 110 mlir-build-gated files (cross_substrate_identity) is named by one.
#
# A file-level `#![cfg(all(unix, feature = "mlir-build", ...))]` that is unsatisfied
# ERASES the file — and the harness still prints `ok. 0 passed` and exits 0:
#
#   $ cargo test --no-default-features --features std-surface,cross-module-imports \
#       --test alias_miscompile_run --test array_oob_trap_run
#   running 0 tests
#   test result: ok. 0 passed; 0 failed; ...      <- the alias-miscompile gate
#   running 0 tests
#   test result: ok. 0 passed; 0 failed; ...      <- the array-OOB bounds-TRAP gate
#   CARGO EXIT=0
#
# So the alias-miscompile gate and the ARRAY_OOB_CONTRACT=DETERMINISTIC_BOUNDS_TRAP
# gate were EMPTY HARNESSES in every CI run. A lowering regression that made `a[i]`
# skip its bounds check, or reintroduced the alias miscompile, was caught by NOTHING.
#
# GATE-EVIDENCE RULE (the doctrine scripts/preflight.sh already states in its header):
# never accept `exit 0` as proof a gate ran. Every tier below asserts a POSITIVE test
# count against a floor, so a future feature-gating accident cannot silently zero it.
#
# The floor counts tests EXECUTED (passed + failed + ignored). That is deterministic
# from the source tree given the feature set — unlike a pass count it does not move
# with the environment — and it collapses to 0 under exactly the accident being
# guarded against.
#
# Usage:
#   scripts/exec_semantics_gate.sh                 # all tiers
#   scripts/exec_semantics_gate.sh exec            # one tier: exec | lowering | pkg
#   scripts/exec_semantics_gate.sh --print-count   # report counts, never fail
set -uo pipefail
cd "$(dirname "$0")/.."

# ---------------------------------------------------------------------------
# TIER DEFINITIONS.  measured@f2a2d87d + the E2621 method-receiver fix.
# Floors sit a hair under the measured value: a feature-gating accident erases whole
# FILES (dozens to hundreds of tests at once), so a ~0.7% margin costs the gate
# nothing while keeping a one-test environmental difference from redding main.
# Re-derive after intentionally adding or removing tests: --print-count.
#
#   tier      features                                   harnesses  executed  floor
#   exec      mlir-build std-surface cross-module-imports      317      1913   1900
#   lowering  std-surface,mlir-lowering                        318      1663   1650
#   pkg       pkg                                              317      1265   1255
# ---------------------------------------------------------------------------
TIERS=(exec lowering pkg)

# `exec` — the executable-semantics tier. 110 mlir-build-gated files, including the
# alias-miscompile gate, the array-OOB bounds-trap gate and the array bounds/dtype
# gate. Dropping ANY of the three features silently erases most of it.
FEATURES_exec="mlir-build std-surface cross-module-imports"
FLOOR_TESTS_exec=1900
FLOOR_HARNESSES_exec=310
# MIND_BENCH_REQUIRE=1 turns "MLIR toolchain missing -> skip" into a hard failure, so
# this tier cannot pass vacuously on a runner where mlir-opt/clang never installed.
# Correct ONLY here: this is the tier that actually enables mlir-build.
REQUIRE_TOOLCHAIN_exec=1

# `lowering` — 8 files / ~53 tests carry
# `#![cfg(all(feature = "std-surface", feature = "mlir-lowering"))]`. ci.yml runs each
# of those features ALONE (the 'Test (gated ...)' and 'Run gated tests ...' steps of
# the build_test job) and never together, so the whole group was erased in both runs.
FEATURES_lowering="std-surface,mlir-lowering"
FLOOR_TESTS_lowering=1650
FLOOR_HARNESSES_lowering=310
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
FLOOR_TESTS_pkg=1255
FLOOR_HARNESSES_pkg=310
REQUIRE_TOOLCHAIN_pkg=0

# ---------------------------------------------------------------------------
# QUARANTINE — targets RED at f2a2d87d for a reason NAMED here.
#
# A RATCHET, not an excuse. Any failure in a target NOT listed fails the gate, and a
# listed target that starts PASSING also fails the gate (with an instruction to delete
# its line). The list may only ever shrink.
#
# Every entry is a genuine, separately-scoped defect that this tier was HIDING — they
# are the payload of the finding, not collateral from it.
# deferred: each needs its own fix; none is closed by this gate.
#
#   array_load_bounds_and_dtype  — `oob_index_is_deterministic_clamp` asserts the CLAMP
#       semantics (at(-1)==first, at(100)==last) that commit 80cb1f73 deliberately
#       replaced with a deterministic bounds TRAP. docs/ARRAY_SEMANTICS.md:193 pins
#       CANONICAL_DECISION = ARRAY_OOB_CONTRACT=DETERMINISTIC_BOUNDS_TRAP ("Remove the
#       MLIR clamp"). The test predates that decision (2026-08-12 vs 2026-08-15) and was
#       never updated because it never ran. Upgrade path: reassert against the trap
#       (exit 77), the way tests/array_oob_trap_run.rs already does.
#   f64_abi_negative_control     — asserts the EXACT MLIR-lowering diagnostic "expects
#       different type than prior uses: 'f64' vs 'i64'". The type checker now rejects
#       the same program EARLIER and more precisely with `E2027: no implicit int<->float
#       conversion (RFC 0011)`. The control's real contract ("the negative is rejected AT
#       the f64 call-argument ABI boundary, and not as a parse or a link error") still
#       holds. Upgrade path: accept either diagnostic, keeping the not-parse / not-link
#       non-vacuity assertions that make this control non-vacuous.
#   tensor_param_fail_loud_run   — same class: `tensor_param_emit_shared_fails_loud`
#       pins a superseded diagnostic string; the rejection itself still happens
#       (`lower::non_i64_return`).
#   std_surface_intrinsics       — `each_intrinsic_lowers_to_func_call_with_private_decl`
#       expects `func.call @__mind_load_i64(%`; the intrinsic now lowers inline to
#       `llvm.inttoptr` + `llvm.load`. Stale shape expectation, not a miscompile.
#   std_surface_net_fs_process   — 4 `mlir_functional::fs_*` tests. std.fs's file-I/O
#       intrinsics are NOT registered in STD_SURFACE_INTRINSICS, so `mindc --emit-shared`
#       cannot lower them:
#         warning[E2024]: `__mind_open` is not registered in mindc's Rust/MLIR intrinsic
#         table — mindc build's default backends cannot emit this call
#       A real std-surface completeness gap, not a test defect.
#   std_surface_cdylib_link      — the emitted cdylib carries an undefined
#       `_Exit@GLIBC_2.2.5`, which the link gate rejects. `_Exit` is how the
#       deterministic bounds trap exits; the gate's allowed-undefined set predates it.
#   std_surface_io_canon         — `undefined symbol: sha256` when dlopening
#       libio_canon.so; the runtime-support object does not provide it.
#   std_surface_self_emit_shared — `vec_module_emits_self_contained_shared`: the vec
#       module's emitted .so is not self-contained.
# CRITICAL_<tier> — per-harness minimums for the gates each tier NAMES as the reason
# it exists. An AGGREGATE floor cannot protect a SPECIFIC test: measured slack was 14
# tests against FLOOR_TESTS_exec, and an erased test file still prints
# `test result: ok. 0 passed` so the HARNESS count does not move either. Re-erasing
# exactly the two files this gate was written to defend therefore produced
# ALL TIERS OK / exit 0 — the gate could not protect the thing it names.
# Format: "<target> <min_executed>", checked INDEPENDENTLY of the floors.
#
# PER-TIER, not global. A global list is wrong twice over: a target is only expected
# to run in the tier whose features satisfy its `#![cfg(...)]`, so checking the
# mlir-build gates in the `lowering`/`pkg` tiers would red those tiers for a target
# they correctly do not build — and it would leave `lowering` and `pkg` with NO
# per-harness minimum at all, which is precisely the hole this check exists to close
# (pkg's 2 files / 2 tests are invisible inside an aggregate floor of 1255).
#
# The count compared is tests EXECUTED (passed + failed + ignored), not passed: this
# check answers "did the substance run", and a target that ran and FAILED is caught
# by the quarantine triage below with a message that names the real problem. Counting
# `passed` here would report a failing gate as "did not run".
CRITICAL_exec=(
  "alias_miscompile_run 1"    # the alias-miscompile regression gate
  "array_oob_trap_run 1"      # ARRAY_OOB_CONTRACT=DETERMINISTIC_BOUNDS_TRAP
)
# The two largest members of the std-surface+mlir-lowering group that no CI run
# enabled BOTH features for; 26 of the group's 57 tests live in these two files.
CRITICAL_lowering=(
  "extern_c_phase_a 1"
  "extern_c_phase_b 1"
)
# The entire reason the `pkg` tier exists: ci.yml only ever `cargo check`ed pkg.
CRITICAL_pkg=(
  "package_basic 1"
  "package_traversal 1"
)

QUARANTINE_exec=(
  array_load_bounds_and_dtype
  f64_abi_negative_control
  tensor_param_fail_loud_run
  std_surface_net_fs_process
  std_surface_cdylib_link
  std_surface_io_canon
  std_surface_self_emit_shared
)
QUARANTINE_lowering=(std_surface_intrinsics)
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
ENV_TOLERATED_lowering=(mindfuzz_cross_substrate)
ENV_TOLERATED_pkg=(mindfuzz_cross_substrate)

# ---------------------------------------------------------------------------
print_only=0
want=()
for a in "$@"; do
  case "$a" in
    --print-count) print_only=1 ;;
    exec|lowering|pkg) want+=("$a") ;;
    *) echo "usage: $0 [--print-count] [exec|lowering|pkg ...]" >&2; exit 2 ;;
  esac
done
[ ${#want[@]} -eq 0 ] && want=("${TIERS[@]}")

in_list() { local n=$1; shift; local e; for e in "$@"; do [ "$e" = "$n" ] && return 0; done; return 1; }

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
  # Keep the log at a STABLE path, not a mktemp that is deleted on the way out:
  # when this gate fails, the next question is always "which test, and why", and a
  # 40-line tail of a deleted file cannot answer it. CI uploads nothing, so the path
  # is printed on failure and the file is left in place for the developer.
  log="${MIND_TIER_LOG_DIR:-${TMPDIR:-/tmp}}/mind-tier-$tier.log"
  mkdir -p "$(dirname "$log")"

  if [ "$require_toolchain" = 1 ]; then
    MIND_BENCH_REQUIRE=1 cargo test --no-default-features --features "$features" \
      --no-fail-fast >"$log" 2>&1
  else
    cargo test --no-default-features --features "$features" \
      --no-fail-fast >"$log" 2>&1
  fi
  cargo_status=$?

  # --- POSITIVE-COUNT ASSERT (the anti-silent-zeroing core) ----------------
  harnesses=$(grep -c '^test result:' "$log")
  read -r passed failed ignored <<<"$(
    grep '^test result:' "$log" | awk '{p+=$4; f+=$6; i+=$8} END {print p+0, f+0, i+0}'
  )"
  executed=$((passed + failed + ignored))

  echo "harnesses      : $harnesses  (floor $floor_h)"
  echo "tests executed : $executed  (floor $floor_t)  [passed=$passed failed=$failed ignored=$ignored]"

  if [ "$print_only" = 1 ]; then continue; fi

  rc=0
  # A test file that does not COMPILE also yields zero harnesses, and it is a very
  # different bug from a cfg-erased one — distinguish them, or the gate sends the
  # reader hunting a feature flag when the real problem is a broken test source.
  # (Hit for real while building this gate: an in-flight test file calling a
  # non-existent `TempDir::join` failed the whole build with harnesses=0.)
  mapfile -t build_errs < <(
    grep -E '^error(\[[A-Z0-9]+\])?: ' "$log" | grep -v '^error: test failed' | sort -u | head -5
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
      }' "$log")
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
    echo "  full log: $log"
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

  # --- FAILURE TRIAGE against the quarantine ratchet -----------------------
  mapfile -t failing < <(
    sed -n 's/^error: test failed, to rerun pass `--test \([A-Za-z0-9_]*\)`.*/\1/p' "$log" | sort -u
  )

  unexpected=()
  for t in "${failing[@]}"; do
    in_list "$t" ${quarantine[@]+"${quarantine[@]}"} && continue
    in_list "$t" ${tolerated[@]+"${tolerated[@]}"} && continue
    unexpected+=("$t")
  done
  if [ ${#unexpected[@]} -gt 0 ]; then
    echo
    echo "FAIL[$tier]: ${#unexpected[@]} target(s) failed that are NOT quarantined:"
    for t in "${unexpected[@]}"; do
      echo "  - $t   (rerun: cargo test --no-default-features --features \"$features\" --test $t)"
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

  # `crit_bad` is checked here too: it sets `overall` on its own (the two verdicts
  # are independent), but printing "ok[$tier]" alongside "GATE FAILED" would be the
  # same false-green shape this whole script exists to remove.
  if [ "$rc" = 0 ] && [ ${#crit_bad[@]} -eq 0 ]; then
    echo "ok[$tier]: $executed tests executed across $harnesses harnesses; ${#failing[@]} failing"
    echo "          target(s), all accounted for (quarantine ${#quarantine[@]}, env-tolerated ${#tolerated[@]})."
  else
    echo
    echo "cargo exit was $cargo_status; failing targets: ${failing[*]:-none}"
    echo "full log: $log"
    for t in ${unexpected[@]+"${unexpected[@]}"}; do
      echo "--- tier '$tier' :: $t ---"
      awk -v t="tests/$t.rs" '$0 ~ ("Running " t) {on=1} on {print} on && /^test result:/ {exit}' "$log" \
        | grep -Ev '^test .* \.\.\. ok$' | head -40
    done
    overall=1
  fi
done

echo
if [ "$print_only" = 1 ]; then
  echo "--print-count: re-derive the floors from the numbers above; this mode never fails"
  exit 0
fi
[ "$overall" = 0 ] && echo "ALL TIERS OK" || echo "GATE FAILED — see FAIL[...] above"
exit $overall
