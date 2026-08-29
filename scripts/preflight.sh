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
  if cw_out=$(python3 scripts/cfg_gate_wiring_lint.py 2>&1); then
    printf '%s\n' "$cw_out" | tail -1
  else
    bad "cfg-gate wiring contract FAILED — a feature-gated test file runs NOWHERE in CI:"
    printf '%s\n' "$cw_out" | head -12
  fi
else
  bad "scripts/cfg_gate_wiring_lint.py MISSING — CI runs it; preflight cannot verify it"
fi

step "smoke-corpus wiring contract  [ci.yml mindcraft_self_host first step]"
# Same class, other corpus: examples/mindc_mind/*.py is the ONLY regression gate for
# constructs main.mind does not self-use, and its wiring used to be two hand-copied
# lists plus a prose claim that they were identical (they were not: 9 gates ran only
# in fast_keystone.sh and one ran in neither runner). SMOKE_WIRING.tsv is the checked
# contract; the lint recomputes the real wiring and fails on drift in either direction.
if [ -f examples/mindc_mind/smoke_wiring_lint.py ]; then
  if sw_out=$(python3 examples/mindc_mind/smoke_wiring_lint.py 2>&1); then
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
  if sm_out=$(python3 examples/mindc_mind/stdlib_manifest_lint.py 2>&1); then
    printf '%s\n' "$sm_out" | tail -1
  else
    bad "std manifest contract FAILED — a std module list drifted, or the seed blob changed:"
    printf '%s\n' "$sm_out" | head -12
  fi
else
  bad "examples/mindc_mind/stdlib_manifest_lint.py MISSING — CI runs it; preflight cannot verify it"
fi

if [ "${1:-}" = "--full" ]; then
  step "no-features test parity  [ci.yml Build & Test 'Test' steps — the fail-close regression class]"
  # A test that feeds lower_to_ir free operands / hits a gated path lowered to a
  # silent const-0 pre-#9 now panics loud — and only shows under the non-std-surface
  # test steps the keystone never runs. The `error: test failed` grep is abort-aware
  # (a stack-overflow SIGABRT prints NO `test result: FAILED`, only the cargo line).
  # Two accepted CI-ABSENT local-only failures are excluded: mindfuzz_cross_substrate
  # (needs the MLIR toolchain; soft-skips in CI's build_test job) and g2_differential_mlir
  # (dlopens the gitignored, stale in-tree libmindc_mind.so; CI's fresh checkout rebuilds).
  nft_out=$(cargo test --no-default-features --features std-surface,cross-module-imports \
              --no-fail-fast 2>&1 | grep -E "error: test failed|has overflowed its stack" | \
              grep -viE "mindfuzz_cross_substrate|g2_differential_mlir" || true)
  if [ -z "$nft_out" ]; then echo "ok (no test failures beyond the accepted CI-absent mindfuzz + g2)"; else
    bad "cargo test --no-default-features --features std-surface,cross-module-imports FAILS beyond mindfuzz:"; printf '%s\n' "$nft_out" | head
  fi

  step "keystone byte-identity 7/7  [ci.yml + cross-substrate — the wedge invariant]"
  ks_out=$(MIND_BENCH_REQUIRE=1 cargo test --release \
       --features "mlir-build std-surface cross-module-imports" \
       --test phase_g_keystone_bootstrap -- --test-threads=1 2>&1 || true)
  ks_n=$(printf '%s\n' "$ks_out" | sed -n 's/^test result: ok\. \([0-9]*\) passed.*/\1/p' | tail -1)
  if [ "${ks_n:-0}" -ge 7 ] 2>/dev/null; then
    echo "ok ($ks_n/7 byte-identical)"
  else
    bad "keystone NOT 7/7 (passed='${ks_n:-none}') — 0 passed means the target was cfg'd out or SKIPped, which is NOT a pass; a cross-substrate byte-identity regression; do NOT push"
  fi

  step "cross-substrate determinism 24/24  [ci.yml cross_substrate_identity — THE wedge invariant]"
  # Was ABSENT from preflight until 2026-08-27 while preflight still printed
  # "safe to push". Features must match ci.yml:286 EXACTLY — tests/cross_substrate_identity.rs
  # carries a file-level #![cfg(all(feature="mlir-build", feature="std-surface",
  # feature="cross-module-imports"))]; drop any one and the whole file vanishes and
  # cargo reports `ok. 0 passed` with exit 0. Hence the POSITIVE-count assert.
  xs_out=$(MIND_BENCH_REQUIRE=1 cargo test --no-default-features \
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

  step "self-host LOOP byte-identity  [examples/mindc_mind/self_host_loop_smoke.py]"
  # Banked lesson (reference_preflight_missing_loop_gate_reseed): a main.mind/std edit
  # can pass the local keystone 7/7 and still RED CI on a stale frozen seed. The loop
  # gate is the one that catches it. Any main.mind/std edit needs --reseed in the SAME change.
  if [ -f examples/mindc_mind/self_host_loop_smoke.py ]; then
    if lp_out=$(python3 examples/mindc_mind/self_host_loop_smoke.py 2>&1); then
      echo "ok (stage1==stage2==stage3 byte-identical)"
    else
      bad "self-host LOOP gate FAILED (stale frozen seed? reseed in the SAME change):"; printf '%s\n' "$lp_out" | tail -5
    fi
  else
    bad "self_host_loop_smoke.py MISSING — the loop gate cannot run; do NOT push"
  fi

  step "whole-module mic@3 FLIP  [examples/mindc_mind/mic3_flip_smoke.py]"
  # Banked lesson (reference_mic3_flip_required_local_gate_2026_08_06): REQUIRED for ANY
  # lower.rs / emit / mic@3 change. Keystone cargo-test + oracle-parity do NOT cover the
  # whole-module FLIP — this is the gate that reverted #287-F2 (#223 -> #224).
  if [ -f examples/mindc_mind/mic3_flip_smoke.py ]; then
    if fl_out=$(python3 examples/mindc_mind/mic3_flip_smoke.py 2>&1); then
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
  if [ -f examples/mindc_mind/mic3_primitives_smoke.py ]; then
    if mp_out=$(MINDC_SO="${MINDC_SO:-/tmp/libmindc_mind_self_host.so}" \
                python3 examples/mindc_mind/mic3_primitives_smoke.py 2>&1); then

# SDLC gates (git-level, no build). Both exist because of a real incident: a merge
# deleted a security ENFORCEMENT line while its enum variant, Display arm, error
# mapping AND its test all survived, so nothing failed to compile and the test kept
# passing over a rule that no longer existed.
python3 scripts/sdlc/lost_by_merge.py HEAD || { echo "preflight: lost-by-merge gate FAILED"; exit 1; }

# DTK register-allocator cross-implementation parity. The pure-MIND planner SHIPS
# inside the frozen stage1.elf, so a divergence between it and the Rust reference is
# a silent wrong-register miscompile. This was the only gate checking that, and it
# was executed by nothing at all. MINDC_SO is set explicitly so a missing .so FAILS
# rather than skipping.
MINDC_SO="${MINDC_SO:-$(ls examples/mindc_mind/libmindc_mind.so 2>/dev/null || echo /tmp/libmindc_mind_self_host.so)}" \
MIND_DTK_SKIP_RUST_REGEN=1 python3 examples/mindc_mind/testdata/dtk_plan_parity_smoke.py \
  || { echo "preflight: DTK regalloc parity FAILED"; exit 1; }
python3 scripts/sdlc/enforcement_bijection.py || { echo "preflight: enforcement/test pairing FAILED"; exit 1; }

# RI-D1 readiness ratchet (#313): native-backend readiness for the frozen profile.
# Verified green at 9d3d5d41; a regression here must block a push, not surface at flip time.
python3 examples/mindc_mind/ri_d1_frozen_profile_gate.py || { echo "preflight: RI-D1 readiness gate FAILED"; exit 1; }
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
  # Fix on FAIL:  MINDC_SO=<built .so> python3 examples/mindc_mind/self_host_loop_smoke.py --reseed
  # then commit the re-blessed testdata/selfhost_loop/{stage1.elf,MANIFEST.txt}.
  loop_rc=0; python3 examples/mindc_mind/self_host_loop_smoke.py >/tmp/preflight-loop.out 2>&1 || loop_rc=$?
  if [ "$loop_rc" = 0 ]; then echo "ok (frozen seed reproduces current source)"
  elif [ "$loop_rc" = 2 ]; then echo "skip (frozen bootstrap fixture missing — BLOCKED)"
  else bad "self-host loop drift — main.mind/std changed but the frozen seed was NOT re-blessed; --reseed in this change (see /tmp/preflight-loop.out)"; tail -3 /tmp/preflight-loop.out; fi

  step "bench gate (frozen low-level frontend)  [bench-gate.yml, --no-default-features]"
  base=$(ls -t .bench-baseline-*correctness*.txt 2>/dev/null | head -1)
  if [ -n "$base" ] && [ -f tools/bench_gate.py ]; then
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
    CARGO_TARGET_DIR="$PF_TARGET" cargo bench --bench compiler --no-default-features -- \
      --warm-up-time 3 --measurement-time 8 --output-format bencher > /tmp/preflight-bench.out 2>/dev/null
    if python3 tools/bench_gate.py --baseline "$base" --current /tmp/preflight-bench.out --threshold 0.10; then
      echo "ok (regression <= +10% vs $base; speedups always pass)"
    else
      bad "frozen-frontend bench regression >+10% vs $base — STOP & decide: revert, or re-bless baseline if a dramatic win elsewhere justifies it"
    fi
  else
    echo "skip (no correctness baseline / bench_gate.py)"
  fi
fi

echo
if [ "$fail" = 0 ]; then
  printf '\033[32m✓ preflight PASS — safe to push\033[0m\n'
else
  printf '\033[31m✗ preflight FAIL — fix the above before pushing\033[0m\n'
fi
exit "$fail"
