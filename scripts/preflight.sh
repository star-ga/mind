#!/usr/bin/env bash
# preflight.sh — local CI-parity gate. Run before pushing to avoid red CI.
#
# Mirrors the jobs that actually gate main: rustfmt (ci.yml Format Check),
# build+test, mindc check over std/+examples/ (ci.yml mindcraft_check, which
# also flags `mindc fmt` drift), and — with --full — the keystone byte-identity
# gate and the frozen-frontend bench gate (bench-gate.yml, --no-default-features).
#
#   scripts/preflight.sh          # FAST (~seconds-min): fmt + build + mindc check
#                                 #   + the two wiring contracts (cfg-gate + smokes)
#   scripts/preflight.sh --full   # + keystone 7/7 + cross-substrate determinism
#                                 #   + executable-semantics tier + self-host LOOP
#                                 #   + mic@3 FLIP + bench gate
#
# The executable-semantics tier is the long pole (~317 harnesses, ~1900 tests, a full
# mlir-build compile). Set MIND_PREFLIGHT_SKIP_EXEC_SEMANTICS=1 to opt out explicitly;
# it then prints "SKIPPED ... this gate did NOT run" rather than passing silently.
#
# GATE-EVIDENCE RULE: never accept `exit 0` as proof a gate ran. Every cargo-test
# gate below asserts a POSITIVE test count. A correct `--test <target>` selector is
# NOT sufficient — a test file that is `#![cfg]`d out by a missing feature compiles
# to nothing and cargo still reports `ok. 0 passed` with exit 0 (measured
# 2026-08-27 on cross_substrate_identity with `--features mlir-build` alone, which
# leaves the file's required `cross-module-imports` off).
#
# Exits non-zero if any gate fails; prints exactly what to run to fix it.
#
# AGGREGATE, NEVER FAIL-FAST. Every gate runs; failures accumulate in `fail` via
# `bad` and the script exits non-zero at the END with all of them reported. Four
# gates used to `exit 1` inline instead, and the cost was measured: the RI-D1
# readiness ratchet sits at line ~314 of ~376 and is RED on any branch lacking the
# RI-D1 work (it is a ratchet for capability that lives on
# feat/native-float-narrow-codegen). Its inline exit turned the remaining ~60% of
# this script into dead code -- so the self-host LOOP gate and the criterion BENCH
# gate never ran, and a genuinely broken bootstrap seed (stage1 != frozen seed,
# main.mind changed in 6a59af63 with no --reseed) rode the branch invisibly.
#
# A known-red early gate must never be able to hide an unknown red later one. If a
# gate has a REAL dependency on an earlier one, skip its dependents and SAY they
# were skipped -- never exit silently.
set -uo pipefail
cd "$(dirname "$0")/.."

fail=0
# Every CI-parity build below goes here instead of ./target. Those builds use
# different feature sets than a normal dev build, and writing them into the shared
# target dir silently replaces the working mindc — which caused two false diagnoses
# (a phantom "1294 check errors" and a phantom "stale golden") before it was found.
PF_TARGET="${PF_TARGET:-target-preflight}"
step() { printf '\n\033[1m== %s ==\033[0m\n' "$1"; }
bad()  { printf '\033[31mFAIL:\033[0m %s\n' "$1"; fail=1; }

step "rustfmt  [ci.yml Format Check]"
if cargo fmt --check >/dev/null 2>&1; then echo "ok"; else
  bad "rustfmt drift — run: cargo fmt"
  cargo fmt --check 2>&1 | grep "Diff in" | head
fi

step "build (default features = std-surface)  [ci.yml Build & Test]"
if cargo build --quiet 2>/tmp/preflight-build.err; then echo "ok"; else
  bad "build broken"; tail -15 /tmp/preflight-build.err
fi

step "mindc check std/ examples/  [ci.yml mindcraft_check — error-severity incl. fmt::drift]"
# ISOLATED TARGET DIR (2026-08-27). This step must build mindc the way CI does —
# --no-default-features, i.e. WITHOUT mlir-build — but it used to write that binary
# into the shared ./target, silently REPLACING the developer's working mindc. Two
# false diagnoses in one session came from exactly that:
#   * `mindc check std examples` then reports ~1294 E2003/E2007 "no std surface"
#     errors where the CI-featured binary reports 0 — same tree, opposite verdict;
#   * `mindc build --backend mlir` starts emitting a ~16KB LAUNCHER STUB that prints
#     "[mind-runtime] Parsed ... evaluated ..." and EXITS 0 regardless of the
#     program's return value, instead of a real ~37KB compiled binary — so every
#     exit-code-based value check silently reads 0.
# Detection if you ever suspect it: `strings target/release/mindc | grep -c mlir-opt`
# is 4 on a full build and 2 on a --no-default-features one.
# Building into target-preflight/ keeps this gate CI-faithful without corrupting the
# tree the rest of the session measures against.
if CARGO_TARGET_DIR="$PF_TARGET" cargo build --release --no-default-features --features "std-surface cross-module-imports" --bin mindc --quiet 2>/dev/null; then
  errs=$("$PF_TARGET/release/mindc" check std/ examples/ 2>&1 | grep -E ': error:' || true)
  if [ -z "$errs" ]; then echo "ok (warnings allowed)"; else
    bad "mindc check error-severity diagnostics (fmt::drift → mindc fmt <file>; tuple-return → #[repr(C)] struct):"
    printf '%s\n' "$errs" | head
  fi
else
  bad "could not build mindc"
fi

step "no-features compile parity  [ci.yml 'Check compiles (no features)' + gated feature-combo steps]"
# The keystone/build/mindc gates above ALL use std-surface, so a change that
# compiles there but not under --no-default-features (a mis-placed #[cfg], a
# std-surface-only variant referenced from a non-std-surface path) rides straight
# onto red main. Mirror every feature combo ci.yml compiles.
nf_ok=1
for feats in "" "std-surface" "cross-module-imports" "std-surface,cross-module-imports" \
             "autodiff" "mlir-lowering" "cpu-buffers" "cpu-exec" "pkg"; do
  if [ -z "$feats" ]; then flag=(--no-default-features); label="--no-default-features";
  else flag=(--no-default-features --features "$feats"); label="--no-default-features --features $feats"; fi
  if ! cargo check "${flag[@]}" --quiet 2>/tmp/preflight-nf.err; then
    bad "cargo check $label BROKEN — this is the exact class that slips past the std-surface gates:"
    grep -E '^error' /tmp/preflight-nf.err | head -4
    nf_ok=0
  fi
done
[ "$nf_ok" = 1 ] && echo "ok (all CI feature combos compile)"

step "cfg-gate wiring contract  [ci.yml executable_semantics_tier first step]"
# In the FAST path on purpose: pure text analysis, runs in well under a second, and
# it is the cheapest possible catch for the most expensive failure shape in this
# repo. A new `tests/*.rs` carrying a crate-level
# `#![cfg(all(unix, feature = "..."))]` whose feature combo no CI run enables does
# NOT fail visibly — cargo builds it as an EMPTY HARNESS, prints `ok. 0 passed` and
# exits 0. Measured 2026-08-28 before this lint existed: 69 such files / 161 #[test]
# fns, including tests/alias_miscompile_run.rs and tests/array_oob_trap_run.rs, the
# dedicated regression gates for a real alias miscompile and for the deterministic
# array-OOB trap. Both PASS when their features are enabled; neither had ever run.
# The lint derives the required set from the test sources and the enabling set from
# ci.yml + Cargo.toml's feature graph + the tier definitions in
# scripts/exec_semantics_gate.sh, so there is no second hand-maintained list to drift.
if [ -f scripts/cfg_gate_wiring_lint.py ]; then
  if cw_out=$(python3 scripts/run_gate.py scripts/cfg_gate_wiring_lint.py 2>&1); then
    printf '%s\n' "$cw_out" | tail -1
  else
    bad "cfg-gate wiring contract FAILED — a feature-gated test file runs NOWHERE in CI:"
    printf '%s\n' "$cw_out" | head -12
  fi
else
  bad "scripts/cfg_gate_wiring_lint.py MISSING — CI runs it; preflight cannot verify it"
fi

step "tier gate fails CLOSED on a failing test  [ci.yml executable_semantics_tier]"
# The lint above proves a test FILE is reachable by some tier. This proves the tier
# script GRADES what it runs: its triage matched only cargo's `--test <name>` rerun
# hint, so a failing lib unit test, doctest or bin test (`--lib` / `--doc` /
# `--bin <n>`) was attributed to nothing, cargo's exit status was printed and never
# compared, and the aggregate `failed` count was printed and never asserted.
# Synthetic tier logs replayed through --from-log, asserting the exit code — pure
# text analysis, sub-second, so it belongs in the FAST path beside the lint above.
if [ -f scripts/test_exec_semantics_gate.py ]; then
  if tg_out=$(python3 scripts/run_gate.py scripts/test_exec_semantics_gate.py 2>&1); then
    printf '%s\n' "$tg_out" | grep -E '^ran=' | sed 's/^/  /'
  else
    bad "tier-gate mutation proof FAILED — exec_semantics_gate.sh no longer reds on a failing test:"
    printf '%s\n' "$tg_out" | grep -E '^\[FAIL\]|^ran=' | head -12
  fi
else
  bad "scripts/test_exec_semantics_gate.py MISSING — CI runs it; preflight cannot verify it"
fi

step "comment-enumeration contract  [ci.yml enumeration_drift_lint.py]"
# The tc_let shape, generalised: a comment that COUNTS or NAMES a set, beside code
# that owns the real set, with nothing comparing the two. `parse_let` documented
# ty=0 as "the exact no-annotation value every let consumer already handles" and
# named the handlers; tc_let was not among them and read ast_span_lo(0). Two
# enumerations that had already rotted the same way are pinned here.
if [ -f scripts/enumeration_drift_lint.py ]; then
  if ed_out=$(python3 scripts/run_gate.py scripts/enumeration_drift_lint.py 2>&1); then
    printf '%s\n' "$ed_out" | tail -1
  else
    bad "comment-enumeration contract FAILED — a comment disagrees with its code:"
    printf '%s\n' "$ed_out" | head -12
  fi
else
  bad "scripts/enumeration_drift_lint.py MISSING — CI runs it; preflight cannot verify it"
fi

step "smoke-corpus wiring contract  [ci.yml mindcraft_self_host first step]"
# Same class, other corpus: examples/mindc_mind/*.py is the ONLY regression gate for
# constructs main.mind does not self-use, and its wiring used to be two hand-copied
# lists plus a prose claim that they were identical (they were not: 9 gates ran only
# in fast_keystone.sh and one ran in neither runner). SMOKE_WIRING.tsv is the checked
# contract; the lint recomputes the real wiring and fails on drift in either direction.
if [ -f examples/mindc_mind/smoke_wiring_lint.py ]; then
  if sw_out=$(python3 scripts/run_gate.py examples/mindc_mind/smoke_wiring_lint.py 2>&1); then
    printf '%s\n' "$sw_out" | tail -1
  else
    bad "smoke-corpus wiring contract FAILED — a smoke is unclassified or its wiring drifted:"
    printf '%s\n' "$sw_out" | head -12
  fi
else
  bad "examples/mindc_mind/smoke_wiring_lint.py MISSING — CI runs it; preflight cannot verify it"
fi

step "std manifest contract  [examples/mindc_mind/testdata/stdlib_manifest.txt]"
# Same class again, third corpus: WHICH std/*.mind modules each consumer links used
# to be 17 hand-copied literal lists (the native bridge in src/bin/mindc.rs + 16
# copies across examples/mindc_mind/*.py) plus a fourth, DIFFERENT list
# (STDLIB_MIND_SOURCES, src/project/stdlib.rs) for general `use std.<m>` resolution
# — with nothing asserting any of them agreed. stdlib_manifest.txt is the checked
# contract; the lint recomputes each consumer's real membership and fails on drift
# in BOTH directions, and pins the seed blob's sha256 so a reseed event (which
# changes the compiled bytes of every native build) cannot land as a quiet edit.
if [ -f examples/mindc_mind/stdlib_manifest_lint.py ]; then
  if sm_out=$(python3 scripts/run_gate.py examples/mindc_mind/stdlib_manifest_lint.py 2>&1); then
    printf '%s\n' "$sm_out" | tail -1
  else
    bad "std manifest contract FAILED — a std module list drifted, or the seed blob changed:"
    printf '%s\n' "$sm_out" | head -12
  fi
else
  bad "examples/mindc_mind/stdlib_manifest_lint.py MISSING — CI runs it; preflight cannot verify it"
fi

step "no-AI-attribution FILE gate  [docs-claims.yml — no model/tool credit in TRACKED FILES]"
# The sibling of the message gate below, and it was MISSING here. .githooks/pre-commit
# runs it, but a hook is opt-in (core.hooksPath must be set) and preflight is what
# this repo tells you to run before pushing — so the whole-tree file gate was
# enforced by CI and by a hook a developer may not have installed, and by nothing
# in between. It also carries the ARTIFACT-scoped bare-name rule over ANATOMY.md,
# which is the one leak class that has actually shipped: a generated index named a
# model as this compiler's owner while every locally-runnable gate said PASS.
# Text-only (`git grep`), seconds, so it belongs in the FAST path.
# Run under scripts/run_gate.py: a whole-tree grep is exactly the shape that can
# exit 0 having scanned nothing, and the runner refuses a PASS with no asserted
# count rather than trusting this script to notice.
if [ ! -f scripts/check_no_ai_attribution.sh ]; then
  bad "scripts/check_no_ai_attribution.sh MISSING — CI runs it; preflight cannot verify it"
else
  if na_out=$(python3 scripts/run_gate.py scripts/check_no_ai_attribution.sh 2>&1); then
    printf '%s\n' "$na_out" | tail -1
  else
    bad "no-AI-attribution file gate FAILED — a tracked file credits a model or a tool:"
    printf '%s\n' "$na_out" | grep -E '^(::error::|ANATOMY\.md:|[^ ]+:[0-9]+:)' | head -12
  fi
fi

step "commit-message hygiene main..HEAD  [docs-claims.yml — no model/tool credit in HISTORY]"
# Text-only (git log + grep), seconds, and in the FAST path for the same reason
# the cfg-gate lint is: it is the cheapest catch for the one failure class that
# CANNOT be fixed after the fact. A file can be corrected by the next commit; a
# MESSAGE on a public repo cannot be corrected without rewriting every descendant
# commit. Until this gate existed nothing checked messages at all — no commit-msg
# hook, no CI step reading `git log` — so the credit the file gate rejects inside
# a comment rode into permanent history in the message beside it.
# Shares ONE pattern definition with the file gate (scripts/ai_attribution_patterns.sh),
# so the two cannot drift. GATE-EVIDENCE: the script prints the commit COUNT it
# scanned and reports an empty range as empty, never as a silent pass.
if [ ! -f scripts/check_commit_messages.sh ]; then
  bad "scripts/check_commit_messages.sh MISSING — CI runs it; preflight cannot verify it"
elif ! git rev-parse --verify -q main >/dev/null; then
  bad "no local 'main' ref — cannot derive the unpushed range; fetch it, do not skip (a skipped message gate is how a credit reaches public history)"
else
  if cm_out=$(bash scripts/check_commit_messages.sh main..HEAD 2>&1); then
    printf '%s\n' "$cm_out" | grep -E '^commit-message gate:' | tail -1
  else
    bad "commit-message gate FAILED — a commit message credits a model or a tool. Fix by AMENDING/REBASING the messages BEFORE pushing:"
    printf '%s\n' "$cm_out" | grep -E '^(::error::|    msg:)' | head -12
  fi
fi

# --- git-level SDLC gates -------------------------------------------------
# These run UNCONDITIONALLY: they are OUTSIDE the `--full` branch below, so every
# preflight executes them.
#
# They were previously nested inside the mic@3 smoke's SUCCESS branch (merge
# collateral, 5b0d0097), and then inside `--full` while a comment claimed they were
# unconditional -- a gate whose execution depends on an unrelated red gate, or on a
# flag, is not a gate, and a COMMENT saying otherwise is worse than silence because
# it stops anyone from checking. examples/mindc_mind/smoke_wiring_lint.py now
# locates this branch and fails on an unconditional-execution claim written inside
# it, so the arrangement below is checked rather than asserted.
#
# They are text-only (git + python3 stdlib), need no toolchain and take seconds, so
# "unconditional" costs nothing. The first two exist because of a real incident: a
# merge deleted a security ENFORCEMENT line while its enum variant, Display arm,
# error mapping AND its test all survived, so nothing failed to compile and the test
# kept passing over a rule that no longer existed. The three gate-runner checks
# below are text-only for the same reason and stood under the same unconditional
# claim, so they are hoisted WITH them: leaving them inside `--full` would keep the
# claim false for three of five gates while repairing it for two.
#
# Every call goes through scripts/run_gate.py. The runner is what turns "exit 0"
# into "exited 0 AND published asserted=<N> AND announced no SKIP"; a direct call
# here would be the second way to run a gate -- the way that cannot see a vacuous
# pass -- in the very file that checks for it.
#
# --min-asserted 0 on lost_by_merge only: that gate is conditional BY DESIGN -- on a
# non-merge HEAD there is no merge to audit and it correctly reports `ran=0`. That is
# "not applicable", not "checked nothing", and it is the ONLY exemption from the
# runner's N>=1 rule in this file; every other gate here must publish a non-zero
# assertion count.
python3 scripts/run_gate.py --min-asserted 0 scripts/sdlc/lost_by_merge.py HEAD || bad "lost-by-merge gate FAILED"
python3 scripts/run_gate.py scripts/sdlc/enforcement_bijection.py || bad "enforcement/test pairing FAILED"

# Gate-runner wiring: a workflow may reach a gate ONLY through scripts/run_gate.py.
# The runner is what refuses "exit 0 with nothing asserted"; a step invoking the
# same gate directly is a second way to run it — the way that cannot see a vacuous
# pass. Measured before this lint: 11 direct workflow invocations, two of them
# publishing asserted=0 while exiting 0.
python3 scripts/run_gate.py scripts/gate_runner_wiring_lint.py \
  || bad "gate-runner wiring FAILED — a workflow reaches a gate without the runner"

# The wiring lint's own SCOPE. It started at repository scripts only, so a step
# whose gate IS `cargo test` was neither routed nor required to be declared — a
# whole class of gate outside the mechanism with nothing recording that fact.
# This pins the widened scope and pins the deferral: a cargo exemption must name
# the upgrade path that ends it, and deleting the record turns the gate red. It is
# the SAME module's --self-test entry point: a gate and the proof that it bites are
# one artifact, so the proof cannot be deleted while the gate keeps riding green.
python3 scripts/run_gate.py scripts/gate_runner_wiring_lint.py --self-test \
  || bad "gate-runner cargo scope FAILED — a cargo gate is uncovered and undeclared"

# The contract behind every OTHER gate in this file: scripts/gate_assert.py must
# derive `asserted=<N>` from evidence (evaluated asserts, reported verdicts), never
# from an integer a gate printed. Measured before this test existed: a smoke whose
# case loop was emptied still reported the length of its case list, and a gate
# writing `asserted=99` to stderr outranked the shim's verdict. Text-only, seconds.
python3 scripts/run_gate.py tests/gate_assert_count_contract_test.py \
  || bad "gate-assert count contract FAILED"

# The smoke-wiring lint's own mutation proof: fixture repos in which a class=gate
# row reaches no workflow, and in which preflight claims UNCONDITIONAL execution
# from inside `--full`, must BOTH red the real lint. Two of its five cases are
# controls that must stay green, so a lint stubbed either way is caught.
python3 scripts/run_gate.py scripts/test_smoke_wiring_lint.py \
  || bad "smoke-wiring lint self-test FAILED — the CI-coverage rule does not bite"

if [ "${1:-}" = "--full" ]; then
  step "CI build-test cargo feature matrix parity  [ci.yml build_test]"
  # Derive the matrix from the build_test job so this local leg cannot silently
  # drift when CI adds or removes a cargo-test feature set. The reader is
  # dependency-free and returns one row for each cargo test invocation,
  # preserving exact argv, selectors, timeout bounds, and the lockfile branch.
  # It also executes the typed rows directly: there is no shell reparse, eval,
  # process-substitution decoder, or status-masking pipeline in this path.
  matrix=$(python3 scripts/workflow_scan.py run_cargo_test_matrix 2>&1)
  matrix_rc=$?
  printf '%s\n' "$matrix"
  if [ "$matrix_rc" -ne 0 ]; then
    bad "workflow cargo-test matrix execution FAILED (rc=$matrix_rc)"
  else
    echo "ok (CI build_test cargo matrix execution)"
  fi
  # Neither historical exclusion matches a selected build_test target.
  # The shrink-only set is now empty; a new exception needs a new contract,
  # not a name added to a failure-filtering regular expression.
  PREFLIGHT_TEST_EXCLUSIONS=()
  for excluded in "${PREFLIGHT_TEST_EXCLUSIONS[@]}"; do
    bad "stale exclusion '$excluded' — delete this line"
  done

  step "keystone byte-identity 7/7  [ci.yml + cross-substrate — the wedge invariant]"
  ks_build_out=$(MIND_BENCH_REQUIRE=1 cargo test --release --no-default-features \
       --features "mlir-build std-surface cross-module-imports" \
       --test phase_g_keystone_bootstrap --no-run 2>&1); ks_build_rc=$?
  if [ "$ks_build_rc" -ne 0 ]; then
    bad "BUILD_INCOMPLETE: keystone build exited $ks_build_rc; determinism was not evaluated"
    printf '%s\n' "$ks_build_out" | tail -20
  else
    ks_out=$(MIND_BENCH_REQUIRE=1 cargo test --release --no-default-features \
         --features "mlir-build std-surface cross-module-imports" \
         --test phase_g_keystone_bootstrap -- --test-threads=1 2>&1); ks_rc=$?
    ks_n=$(printf '%s\n' "$ks_out" | sed -n 's/^test result: ok\. \([0-9]*\) passed.*/\1/p' | tail -1)
    if [ "$ks_rc" -eq 0 ] && [ "${ks_n:-0}" -ge 7 ] 2>/dev/null; then
      echo "ok ($ks_n/7 byte-identical)"
    else
      bad "keystone NOT 7/7 (passed='${ks_n:-none}' rc=$ks_rc); completed build, runtime proof failed"
    fi
  fi

  step "cross-substrate determinism 24/24  [ci.yml cross_substrate_identity — THE wedge invariant]"
  # Was ABSENT from preflight until 2026-08-27 while preflight still printed
  # "safe to push". Features must match ci.yml:286 EXACTLY — tests/cross_substrate_identity.rs
  # carries a file-level #![cfg(all(feature="mlir-build", feature="std-surface",
  # feature="cross-module-imports"))]; drop any one and the whole file vanishes and
  # cargo reports `ok. 0 passed` with exit 0. Hence the POSITIVE-count assert.
  xs_out=$(MIND_BENCH_REQUIRE=1 MIND_INTDOT_VNNI_VERIFY=1 cargo test --no-default-features \
             --features "mlir-build std-surface cross-module-imports" \
             --test cross_substrate_identity -- --nocapture 2>&1 || true)
  xs_n=$(printf '%s\n' "$xs_out" | sed -n 's/^test result: ok\. \([0-9]*\) passed.*/\1/p' | tail -1)
  if [ -n "$xs_n" ] && [ "$xs_n" -ge 24 ]; then
    echo "ok ($xs_n/24+ reproducibility gates byte-identical)"
  else
    bad "cross-substrate determinism gate did NOT prove itself (passed='${xs_n:-none}', need >=24) — 0 tests means the file was cfg'd out, NOT a pass; do NOT push"
  fi

  step "executable-semantics tier  [ci.yml executable_semantics_tier — the tier that ran NOWHERE]"
  # Added 2026-08-28. Measured at f2a2d87d: 123 integration-test files / 262 test
  # functions were reachable by NO cargo-test invocation in ci.yml OR in this script.
  # The 8 broad `cargo test` runs never enable mlir-build; the 4 that do are all
  # `--test <target>` selectors, and exactly ONE of the 110 mlir-build-gated files is
  # named by one. So the alias-miscompile gate and the array-OOB bounds-TRAP gate both
  # compiled to EMPTY harnesses and reported `ok. 0 passed` with exit 0 in every run.
  # The gate body is shared with the CI job (scripts/exec_semantics_gate.sh) so the two
  # cannot drift; it asserts a POSITIVE test count and triages failures against a
  # shrink-only quarantine list. Skipped by MIND_PREFLIGHT_SKIP_EXEC_SEMANTICS=1 when
  # you need the fast path — that is a documented OPT-OUT, never a silent pass.
  if [ -n "${MIND_PREFLIGHT_SKIP_EXEC_SEMANTICS:-}" ]; then
    echo "SKIPPED by MIND_PREFLIGHT_SKIP_EXEC_SEMANTICS — this gate did NOT run"
  elif [ -x scripts/exec_semantics_gate.sh ]; then
    if es_out=$(scripts/exec_semantics_gate.sh 2>&1); then
      printf '%s\n' "$es_out" | grep -E '^(harnesses|tests executed|ok:)' | sed 's/^/  /'
    else
      bad "executable-semantics tier FAILED (a NON-quarantined target broke, or the tier collapsed to 0):"
      printf '%s\n' "$es_out" | grep -E '^(FAIL|  - |harnesses|tests executed)' | head -12
    fi
  else
    bad "scripts/exec_semantics_gate.sh MISSING — CI runs it; preflight cannot verify it"
  fi

  step "gate-vacuity sweep  [ci.yml gate_vacuity — the positive control for every gate]"
  # THE POSITIVE CONTROL. Runs the whole smoke corpus with the toolchain handles
  # pointed at paths that do not exist and requires EVERY compiler-dependent gate
  # to be REJECTED. Measured before scripts/run_gate.py existed: 108 of the 146
  # corpus gates still exited 0 with the compiler binary absent — i.e. two thirds
  # of this repo's regression corpus could report success having compiled nothing.
  # A gate that cannot fail is not a gate, and a fix proven through one is an
  # unproven fix. This sweep fails when it finds NOTHING wrong with the negative
  # case, so it cannot itself rot into a vacuous pass.
  if vs_out=$(python3 scripts/run_gate.py --vacuity-sweep 2>&1); then
    printf '%s\n' "$vs_out" | grep -E '^(ran=|PASS )' | sed 's/^/  /'
  else
    bad "gate-vacuity sweep FAILED — a gate reports success with no compiler present:"
    printf '%s\n' "$vs_out" | grep -E '^(VACUOUS|ran=|FAIL)' | head -12
  fi

  step "whole-module mic@3 FLIP  [examples/mindc_mind/mic3_flip_smoke.py]"
  # Banked lesson (reference_mic3_flip_required_local_gate_2026_08_06): REQUIRED for ANY
  # lower.rs / emit / mic@3 change. Keystone cargo-test + oracle-parity do NOT cover the
  # whole-module FLIP — this is the gate that reverted #287-F2 (#223 -> #224).
  if [ -f examples/mindc_mind/mic3_flip_smoke.py ]; then
    if fl_out=$(python3 scripts/run_gate.py examples/mindc_mind/mic3_flip_smoke.py 2>&1); then
      echo "ok (whole-module FLIP byte-identical)"
    else
      bad "mic@3 whole-module FLIP gate FAILED — this is the #287-F2 revert class:"; printf '%s\n' "$fl_out" | tail -5
    fi
  else
    bad "mic3_flip_smoke.py MISSING — the FLIP gate cannot run; do NOT push"
  fi

  step "mic@3 primitives / golden-vs-live oracle  [ci.yml mindcraft_self_host — runs it, preflight did not]"
  # Added 2026-08-27 after CI's KEYSTONE job failed on a gate preflight never ran.
  # This smoke cross-checks three things: the hardcoded goldens in the .py, a LIVE
  # regeneration via `mindc --emit-mic3` (the Rust oracle), and the self-host .so's
  # output. The golden-vs-live half is Rust-only — it does not touch main.mind — so a
  # failure here is emitter/golden staleness, NOT a self-host regression. Known-red as
  # of this commit: task #316 (goldens predate the #318 lower.rs merge fix). It is
  # reported, never silently skipped; see the banked rule "never print KEYSTONE=PASS
  # while #316 is red".
# --- artifact-dependent SDLC gates (--full only, and WHY) -----------------
# These two are NOT in the fast path above, and the reason is a prerequisite, not
# an oversight: dtk parity needs a built self-host .so and the RI-D1 ratchet needs
# target/release/mindc (the fast path builds mindc into $PF_TARGET, deliberately,
# so it cannot clobber the developer's binary) plus strace. Running them in the
# fast path would report BLOCKED, and a gate that reports "not measured" on every
# fast preflight teaches you to ignore it.
#
# DTK parity now ALSO runs in ci.yml's KEYSTONE job against the .so that job
# builds, so its coverage no longer depends on a developer remembering `--full`.
# The RI-D1 ratchet is not yet in CI. It is GREEN (the three over-rejected
# programs are tracked as a pinned DEFERRED list instead of failing the gate), but
# it needs the frozen stage1.elf, strace and a native-capable mindc on the runner,
# which no CI job provides today. examples/mindc_mind/SMOKE_WIRING.tsv carries that
# exemption with its own `deferred:` marker, which smoke_wiring_lint.py requires
# before a class=gate row is allowed to reach no CI at all.

# DTK register-allocator cross-implementation parity. The pure-MIND planner SHIPS
# inside the frozen stage1.elf, so a divergence between it and the Rust reference is
# a silent wrong-register miscompile. This was the only gate checking that, and it
# was executed by nothing at all.
#
# MINDC_SO is deliberately NOT set here any more. It used to default to
# `examples/mindc_mind/libmindc_mind.so` "so a missing .so FAILS rather than
# skipping" -- but that in-tree artifact is a gitignored leftover that `cargo build`
# never regenerates, and handing it through MINDC_SO laundered months-old bytes as a
# promised real oracle: this gate printed ALL PASS on the exact .so the self-host
# loop gate refuses. Both properties now hold WITHOUT the default, because
# `_selfhost_so.resolve_so()` builds a fresh cdylib when MINDC_SO is unset and
# REFUSES (non-zero) any oracle it cannot prove fresh -- so a missing or stale .so
# still fails rather than skipping, and a stale one can no longer pass.
MIND_DTK_SKIP_RUST_REGEN=1 python3 scripts/run_gate.py examples/mindc_mind/testdata/dtk_plan_parity_smoke.py \
  || bad "DTK regalloc parity FAILED"

# Stale-oracle provenance contract for the ~47 self-host smokes (pure stdlib, no
# build). The freshness qualification used to live in ONE importer and the MINDC_SO
# route set no marker at all, so a stale oracle could certify byte-identity in every
# sibling gate. This asserts resolve_so() still refuses what it cannot prove fresh.
# --min-asserted 16: sixteen named legs, no loop -- an unpinned floor of 1 would be
# cleared by any single one of them, so the floor is the leg count (same pin as the
# ci.yml step; scripts/gate_runner_wiring_lint.py reads both call sites).
python3 scripts/run_gate.py --min-asserted 16 \
  examples/mindc_mind/selfhost_so_provenance_smoke.py \
  || bad "self-host stale-oracle guard FAILED"

# RI-D1 readiness ratchet (#313): native-backend readiness for the frozen profile.
# Was green at 9d3d5d41 and is RED now (the allowlist/corpus bijection gap above);
# it stays here, failing loudly, because a ratchet that is quietly removed while it
# is red is how a readiness claim survives the loss of its evidence.
python3 scripts/run_gate.py examples/mindc_mind/ri_d1_frozen_profile_gate.py || bad "RI-D1 readiness gate FAILED"

  if [ -f examples/mindc_mind/mic3_primitives_smoke.py ]; then
    # No MINDC_SO default (see the DTK step above): unset, the resolver builds a
    # fresh oracle and refuses a non-fresh one, which is both fail-closed and
    # stale-proof. The old default named CI's /tmp path, which locally does not
    # exist at all.
    if mp_out=$(python3 scripts/run_gate.py examples/mindc_mind/mic3_primitives_smoke.py 2>&1); then

      echo "ok (mic@3 primitives byte-exact vs the live oracle)"
    else
      bad "mic@3 primitives smoke FAILED (stale golden vs live oracle — see #316):"
      printf '%s\n' "$mp_out" | grep -E "FAIL|golden|live" | head -4
    fi
  else
    bad "mic3_primitives_smoke.py MISSING — CI runs it; preflight cannot verify it"
  fi

  step "self-host LOOP gate  [ci.yml keystone job — catches main.mind/std drift the frozen seed wasn't re-blessed for]"
  # The keystone 7/7 above is self-consistent (both sides rebuild from the CURRENT
  # main.mind), so it PASSES even when a main.mind/std edit left the checked-in frozen
  # stage0 seed stale. This gate runs the FROZEN pure-MIND ELF on the CURRENT source
  # (PRIMARY mode — no MINDC_SO needed) and asserts it still reproduces the seed: the
  # exact drift that reddened main after #10 added main.mind helpers with no --reseed.
  # Fix on FAIL:  python3 examples/mindc_mind/self_host_loop_smoke.py --advance
  # with MINDC_SO UNSET (the resolver then emits a fresh oracle itself, and refuses
  # a stale one -- never point MINDC_SO at the in-tree libmindc_mind.so: that would
  # corroborate the new seed with whatever compiler that leftover artifact came
  # from), then commit the advanced testdata/selfhost_loop/{stage1.elf,MANIFEST.txt}.
  # --advance is the ordinary answer to source drift: the EXISTING frozen pure-MIND
  # stage0 compiles the new source, the result must be a fixed point, and a fresh
  # Rust oracle must agree -- so the pure-MIND seed chain stays unbroken. --reseed
  # re-mints the seed from RUST output and is legacy: reach for it only when the old
  # seed cannot compile the new source at all, and say so in the change.
  # ONE site, deliberately. preflight used to run this smoke TWICE — a duplicate
  # "self-host LOOP byte-identity" step ran it earlier. That cost a second run of a
  # minutes-long gate, kept a second copy of the hardcoded "--reseed in the SAME
  # change" advice 028fcbf4 removed from this one, and had the two sites DISAGREE
  # about exit 2. A blocked required gate now fails the aggregate, consistently
  # with CI. The absent-file guard is the deleted site's contribution.
  if [ ! -f examples/mindc_mind/self_host_loop_smoke.py ]; then
    bad "self_host_loop_smoke.py MISSING — the loop gate cannot run; do NOT push"
  fi
  # --min-asserted 2: BOTH legs (PRIMARY reproduction + the Rust drift ORACLE)
  # or it is not this gate. A run that cannot build the `.so` reports 1 and is
  # refused here rather than printing a skip note and grading green.
  loop_rc=0; python3 scripts/run_gate.py --min-asserted 2 examples/mindc_mind/self_host_loop_smoke.py >/tmp/preflight-loop.out 2>&1 || loop_rc=$?
  if [ "$loop_rc" = 0 ]; then echo "ok (frozen seed reproduces current source)"
  elif [ "$loop_rc" = 2 ]; then
  # exit 2 = BLOCKED, "could not evaluate". The smoke uses it for more than one
  # cause (absent frozen fixture here; a refused fallback oracle under --reseed),
  # so print ITS reason rather than asserting one this script cannot know.
    bad "self-host loop gate BLOCKED — a required gate was not evaluated; do NOT push"
    grep -E "FAIL|BLOCKED" /tmp/preflight-loop.out | tail -2
  else
    # Do NOT prescribe --reseed here. This branch fires for EVERY non-zero exit, but
    # only one of the causes is drift; another is "no fresh oracle could be built"
    # (a default-feature mindc cannot emit a cdylib, so the smoke falls back to a
    # possibly months-old in-tree .so). Re-blessing the frozen bootstrap from a stale
    # oracle freezes the WRONG compiler -- and an operator following this line is
    # exactly who would do it. The smoke already prints the correct, cause-specific
    # advice; surface THAT rather than overriding it with a guess.
    bad "self-host loop gate FAILED — see the smoke's own verdict below and follow the advice it prints (/tmp/preflight-loop.out)"
    grep -E "FAIL|BLOCKED|WARN\[_selfhost_so\]" /tmp/preflight-loop.out | tail -4
  fi

  step "bench gate (frozen low-level frontend)  [bench-gate.yml, --no-default-features]"
  # Pin the same committed floor as bench-gate.yml. Selecting by mtime made a
  # later historical correctness file silently change the local gate.
  base=.bench-baseline-2026-06-01-correctness.txt
  if [ -f "$base" ] && [ -f tools/bench_gate.py ]; then
    # ISOLATED TARGET DIR, same reason as the mindc-check step above — and this one is
    # nastier, because `cargo bench` builds in the RELEASE profile and this invocation
    # passes --no-default-features with NO feature flags at all. Run in the shared
    # ./target it rewrote target/release/mindc WITHOUT std-surface, as preflight's very
    # LAST step — so the toolchain was left broken at the exact moment preflight printed
    # "safe to push". The symptom is a loud fail-close on the next MLIR build:
    #   lower_expr: no IR lowering for `Let` in value position — refusing to emit a
    #   const-0 placeholder (that would be a silent miscompile)
    # which reads like a compiler regression and is really just a feature-stripped binary.
    # The gate is unaffected: bench_gate.py compares against a committed
    # .bench-baseline-*.txt file, not criterion's own on-disk history.
    bench_tmp=""
    if bench_tmp=$(mktemp -d "${TMPDIR:-/tmp}/mind-preflight-bench.XXXXXX"); then
      bench_out="$bench_tmp/bench.out"
      bench_err="$bench_tmp/bench.err"
      bench_target="$bench_tmp/target"
      bench_rc=0
      CARGO_TARGET_DIR="$bench_target" cargo bench --bench compiler --no-default-features -- \
        --warm-up-time 3 --measurement-time 8 --output-format bencher \
        > "$bench_out" 2>"$bench_err" || bench_rc=$?
      if [ "$bench_rc" -ne 0 ]; then
        bad "compiler bench command failed (exit $bench_rc); the regression gate was not evaluated"
        tail -15 "$bench_err"
        echo "bench logs retained in $bench_tmp"
      elif python3 tools/bench_gate.py --baseline "$base" --current "$bench_out" --require-pipeline --threshold 0.10; then
        echo "ok (regression <= +10% vs $base; speedups always pass)"
        rm -rf "$bench_tmp"
      else
        bad "frozen-frontend bench regression >+10% vs $base — STOP & decide: revert, or re-bless baseline if a dramatic win elsewhere justifies it"
        echo "bench logs retained in $bench_tmp"
      fi
    else
      bad "could not create a private temporary directory for compiler bench output"
    fi
  else
    bad "bench gate prerequisites missing: expected $base and tools/bench_gate.py"
  fi
fi

echo
if [ "$fail" = 0 ]; then
  printf '\033[32m✓ preflight PASS — safe to push\033[0m\n'
else
  printf '\033[31m✗ preflight FAIL — fix the above before pushing\033[0m\n'
fi
exit "$fail"
