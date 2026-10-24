# Executable-semantics tier gate — doctrine and measurement history

Reference for `scripts/exec_semantics_gate.sh`. The **rules** live in the script,
where they execute; this file carries the **reasoning and the measurements** behind
them, so the gate stays one readable file instead of paying for its own history in
lines.

Nothing here is optional background. Every paragraph records a defect that was
measured on this tree, and each one names the check that now prevents it. When a
rule in the script looks arbitrary, its justification is here.

---

## 0. The finding this gate closes

Measured at `f2a2d87d`: **123 integration-test files / 262 test functions were
reachable by NO `cargo test` invocation** in `.github/workflows/ci.yml` or
`scripts/preflight.sh`.

Every broad `cargo test` in CI runs `--no-default-features` with at most
`{std-surface, cross-module-imports, autodiff, mlir-lowering, cpu-buffers}`. The only
runs that enable `mlir-build` are `--test <single-target>` SELECTORS, and exactly ONE
of the 110 `mlir-build`-gated files (`cross_substrate_identity`) is named by one.

A file-level `#![cfg(all(unix, feature = "mlir-build", ...))]` that is unsatisfied
ERASES the file — and the harness still prints `ok. 0 passed` and exits 0:

```
$ cargo test --no-default-features --features std-surface,cross-module-imports \
    --test alias_miscompile_run --test array_oob_trap_run
running 0 tests
test result: ok. 0 passed; 0 failed; ...      <- the alias-miscompile gate
running 0 tests
test result: ok. 0 passed; 0 failed; ...      <- the array-OOB bounds-TRAP gate
CARGO EXIT=0
```

So the alias-miscompile gate and the `ARRAY_OOB_CONTRACT=DETERMINISTIC_BOUNDS_TRAP`
gate were EMPTY HARNESSES in every CI run. A lowering regression that made `a[i]` skip
its bounds check, or that reintroduced the alias miscompile, was caught by NOTHING.

The three independent readings the tier verdict rests on are listed in the script
header. Each was added because the previous set could be defeated alone; the defeat
that motivated readings (2) and (3) — a tier log carrying a lib harness with
`3 failed` printing `failed=3` and then `ok[exec]: ... 0 failing`, exit 0 — is owned
and replayed by `scripts/test_exec_semantics_gate.py`, which asserts the EXIT CODE of
a synthetic log per defeat rather than any prose.

---

## 1. The floors are a ratchet, not a historical note

Each tier asserts a positive count of tests EXECUTED (passed + failed + ignored)
against a floor. Executed is deterministic from the source tree given the feature
set — unlike a pass count it does not move with the environment — and it collapses
to 0 under exactly the accident being guarded against (a file-level `#![cfg(...)]`
that is unsatisfied ERASES the file while the harness still prints `ok. 0 passed`
and exits 0).

A floor left at its first-measured value accrues slack, and slack is deletable test
coverage. At a floor of 1900 the `exec` tier carried 202 tests of headroom — enough
to delete EVERY test in the largest gate file in the tree and still print
`ok[exec]`. Re-measure at landing, every landing, with `--print-count`.

Floors sit a hair under the measured value (~0.8%): a feature-gating accident erases
whole FILES, dozens to hundreds of tests at once, so that margin costs the gate
nothing while keeping a one-test environmental difference from redding `main`.

### Measurement history

| when | exec (harnesses/executed) | lowering | pkg |
|------|---------------------------|----------|-----|
| measured@f2a2d87d | 317 / 1913 | 318 / 1663 | 317 / 1265 |
| mid-wave | 332 / 2102 | 327 / 1830 | 326 / 1415 |
| current (tip of the marker-visibility landing) | 340 / 2170 | 335 / 1891 | 334 / 1474 |

The current row was RE-MEASURED by running all three tiers — one full
`scripts/exec_semantics_gate.sh` run, the numbers read off its own `harnesses` /
`tests executed` lines, not arithmetic on a previous measurement. That landing adds
no test and deletes none: it is the same run twice over, once with the marker
discarded by the harness's stdout capture and once with it written to the process
stdout handle, and BOTH read 340 / 2170, 335 / 1891, 334 / 1474.

**Re-measure at the TIP of a landing, never mid-wave.** Floors pinned at 332 / 2102
mid-wave and then left alone while four further commits added test files sat 4
harnesses and 31 tests below the tree — and that slack was exactly the coverage the
wave itself had just added, so the ratchet was carrying its own new gates as
deletable. Demonstrated on the tier's own log: erasing five whole harnesses
(`bare_variant_ambiguity_run`, `bitwise_no_panic_any_feature`,
`bytes_fixed_into_vec_run`, `conv2d_grad`, `cross_module`) left 331 / 2109, and the
floors 328 / 2085 printed no complaint.

---

## 2. `-- --show-output` is REJECTED as a way to see skip markers

It splices every passing test's captured stdout into the tier log, and some tests
here print a subprocess `mindc test` summary starting with `test result:`. Measured:
it added 5 phantom harnesses and 9 phantom executed tests (one of them a phantom
FAILURE) to both `lowering` and `pkg`, corrupting the floors above and the aggregate
`failed` assert. On the identical tree it read 345 / 2179, 340 / 1900, 339 / 1483.

The accepted mechanism is the producer side: `tests/common/gate.rs` writes the marker
to the PROCESS stdout handle, which the test harness does not shim.

---

## 3. The skip-marker consumer: `ran=0` is a failure to EXECUTE

A gate that cannot run prints an honest `SDLC-GATE <name> ran=<n> fail=<k>` marker
instead of asserting nothing in silence (`tests/g2_differential_mlir.rs` prints one
when the pure-MIND oracle `.so` is unobtainable). Until the consumer existed NOTHING
read those lines — `git grep` found producers and no consumer — so "compared every
fixture and found no divergence" and "compared nothing" produced the same green tier.
A marker nobody reads is documentation, not a gate.

This answers a DIFFERENT question from the failure triage. That one asks *did it
pass*; this one asks *did it run at all*, and the two need separate verdicts because
the fixes are unrelated: a failure to pass is a code defect, a failure to execute is
a missing input or an erased harness.

Environmental skips stay possible through the list that ALREADY documents them
per-tier (`ENV_TOLERATED_<tier>`), rather than a second list that would drift out of
step with it. The default is fail-closed: a `ran=0` marker from any target NOT named
there fails the tier, so the next gate to print a marker is enforced without anyone
remembering to wire it up.

### The consumer read nothing until the marker survived capture

The test harness discards a PASSING test's stdout, and a capability skip PASSES, so
every marker written with the ordinary print macro was thrown away before the
consumer's `awk` ever saw the log. Measured on `tests/std_mlir_bindings_smoke.rs`:
`ok. 4 passed` and ZERO `SDLC-GATE` lines, with the skip having happened. Every
`gate::skipped_optional` was an invisible PASS in every tier.

### Attribution resets at EVERY compilation-unit boundary

Not just at `Running tests/*.rs`. Cargo also emits `Doc-tests <crate>`,
`Running benches/<x>.rs` and `Running unittests src/lib.rs`; without the reset a
marker printed from one of those inherits the identity of whichever `tests/<t>.rs`
block appeared last. Measured: a marker inside a `Doc-tests` block was reported as
coming from `tests/filler_299.rs`, which printed nothing. Two wrong outcomes — the
gate names the wrong file, and if that borrowed carrier happens to be env-tolerated,
a genuine `ran=0` is silently downgraded from fatal to tolerated. That is the exact
false-green class the consumer exists to remove, so it fails closed to
`<unattributed>` instead.

### Which `ran=0` is fatal — decided from the marker's `class=` field

Never from prose, and never from a hand-copied list of names. A producer that stamps
no class is `unclassified` and stays FAIL-CLOSED: a marker may never buy tolerance by
omitting a field.

* **`class=optional`** — `tests/common/gate.rs::is_fail_closed` answers NO for
  `Absent::OptionalInput` in EVERY tier. `MIND_BENCH_REQUIRE` says "use a real
  backend", not "install everything", and the exec tier's own
  `tests/fail_closed_capability_skip_env.rs` pins that an opt-in corpus or opt-in
  silicon survives enforcement. A tier that failed on it would be a second,
  contradictory owner of one rule. What was MISSING was never fatality, it was
  VISIBILITY: the marker never reached the tier log at all, so `skipped_optional`
  graded as a silent PASS.
* **`class=toolchain` in a `REQUIRE_TOOLCHAIN_<tier>=0` tier** —
  `REQUIRE_TOOLCHAIN_<tier>` IS that tier's answer to "may a toolchain absence skip
  here", and `gate.rs` applies the identical rule per-process through
  `MIND_BENCH_REQUIRE`. `lowering` and `pkg` answer 0 because they build WITHOUT
  `mlir-build` on purpose, so `phase_g_keystone_bootstrap` and `mindc_cache_phase_f`
  cannot run there BY CONSTRUCTION. `exec` answers 1, where the helper PANICS instead
  of marking — so a toolchain marker arriving there anyway came from a producer that
  did not consult the rule, and stays FATAL.

Neither arm is named in `ENV_TOLERATED`, deliberately: that list ALSO excuses a
target's FAILURES, so putting the cross-substrate canary gate on it to permit its
deferred VNNI rung would have made a real canary divergence unaccounted-for. Two
different permissions, and only one of them is being granted.

### The four caller-scope arrays

`consume_skip_markers` prints its own FAIL block, returns non-zero on failure, and
sets four arrays the rest of the tier loop reads by name:

| array | meaning |
|-------|---------|
| `zero_fatal` | `ran=0` that fails the tier |
| `zero_ok` | `ran=0` excused by `ENV_TOLERATED_<tier>` |
| `zero_expected` | `ran=0` whose `class=` the gate helper declares legal here |
| `env_skipped` | the TARGETS behind `zero_ok` — the evidence the failure triage grants environmental tolerance on, so a tolerated target that RAN and failed is still a real failure |

---

## 4. The tolerance shrink-ratchet

A quarantined target that starts PASSING must leave its list, or the quarantine
quietly grows into a permanent excuse. `ENV_TOLERATED` had no such rule, so an entry
naming a target the tier does not even BUILD produced no signal at all — the
`mindfuzz` entries sat dead until a human noticed them.

Dead tolerance is not harmless: it is pre-granted permission for a target to report
`ran=0` the day it becomes buildable there, with nobody deciding that.

The test is the tier's OWN log, not a hand-copied expectation: cargo prints
`Running tests/<t>.rs` for every target it built, so a tolerated name absent from that
set is tolerance for something the tier never ran.

### How the `lowering` / `pkg` entries were retired

`g2_differential_mlir` and `mindfuzz_cross_substrate` were once listed for all three
tiers. Neither had a `required-features` entry, so cargo built them in every tier —
but both need a cdylib emit that only the `exec` feature set can produce, so in the
other two they skipped and emitted `ran=0`, which the consumer treats as fatal.
Without those entries the consumer would have RED two tiers that are green today.

Both now carry `required-features = ["mlir-build"]` in `Cargo.toml`, so cargo does
not build them in tiers that cannot run them. **Structural absence beats runtime
tolerance.** The `lowering` and `pkg` entries were therefore removed: they tolerated
a `ran=0` that can no longer be emitted there, and the shrink-ratchet now reds a tier
on exactly that shape. `g2_differential_mlir` stays listed for `exec`, where it IS
built and can fail for the documented environmental reason — and there it is also
exempt from crash tolerance, because a `MIND_CRASH` escapes tolerance and fails the
tier.

---

## 5. `CRITICAL_<tier>` — per-harness minimums

An AGGREGATE floor cannot protect a SPECIFIC test. Measured slack was 14 tests
against `FLOOR_TESTS_exec`, and an erased test file still prints
`test result: ok. 0 passed`, so the HARNESS count does not move either. Re-erasing
exactly the two files this gate was written to defend therefore produced
`ALL TIERS OK` and exit 0 — the gate could not protect the thing it names.

The list is **per-tier, not global**. A global list is wrong twice over: a target is
only expected to run in the tier whose features satisfy its `#![cfg(...)]`, so
checking the `mlir-build` gates in `lowering`/`pkg` would red those tiers for a target
they correctly do not build — and it would leave `lowering` and `pkg` with NO
per-harness minimum at all, which is precisely the hole the check exists to close
(`pkg`'s 2 files / 2 tests are invisible inside a four-figure aggregate floor).

The count compared is tests EXECUTED, not passed: the check answers "did the substance
run", and a target that ran and FAILED is caught by the quarantine triage with a
message that names the real problem. Counting `passed` would report a failing gate as
"did not run".

**A row is added by the SAME change that adds the gate it protects.** A new gate whose
only protection is the aggregate floor is a gate anyone may delete.

Measurement notes for the current rows:

* The fail-closed skip gates carry no `required-features`, so they build and run in
  ALL THREE tiers — measured per tier (21, 4, 6 and 2 executed in `exec`, `lowering`
  and `pkg` alike), never assumed from one of them.
* The META-gates reach the same argument. `harness_scratch_isolation` guards the
  scratch ISOLATION the wedge gates' artifacts depend on (two test processes on one
  box writing one fixed temp path is a flake that reads as a compiler regression), and
  `module_size_ratchet` guards the 800-line ceiling and its budget table. Both walk
  the tree from disk and carry no `required-features`, so they run in all three tiers
  — measured 3 and 5 executed in each, never assumed from one.
* `mindc_artifact_name` is listed for `exec` only: its 4 resolver tests build
  everywhere, but the 3 end-to-end tests that drive `mindc build` are
  `#[cfg(feature = "mlir-build")]` and only this tier's feature set has it, so 7 is an
  `exec` number and 4 is what the others would see.
* `std_process_wait_surface` holds the `std/process.mind` checked-wait functions
  (`exit_error`, `exit_core`, `exit_stopped`, `wait_flags`, `child_exec_errno`,
  `child_spawn_errno`). They were split out of `std_surface_net_fs_process`, which
  is at its line budget, so the new file would otherwise be protected only by the
  aggregate floor. It is `#![cfg(feature = "std-surface")]`; measured 1 executed in
  `exec`, and listed there only.
* `fail_closed_capability_skip_stub_exec` is the ONE with a file-level cfg,
  `#![cfg(unix)]`, because it spawns a POSIX shell stub and `ci.yml`'s `build_test`
  matrix also runs `windows-latest`. Every tier here runs on a unix host, so the row
  is a real minimum in all three; on Windows cargo does not build the target at all
  and there is nothing to count. `harness_portability` is what keeps that gate at FILE
  scope instead of per item.

---

## 6. Why this doctrine is a document and not a comment block

The gate is one file on purpose. It was split into a script plus a sourced library
when it crossed the 800-line house ceiling, and the split bought nothing the tree
wanted: it added a second shell surface to keep coherent (`declare -F` guards,
deliberately un-`local` interface arrays), and it moved the tracked harness file count
UP, against the ratchet in `scripts/check_gate_wiring.py` that exists to stop exactly
that drift.

The lines that pushed the file over the ceiling were predominantly THIS prose — the
measurement history, the marker-capture finding, the `class=` rationale — not logic.
Prose that has outgrown a comment block belongs in a document; the checks belong in
the gate, together, where one reader can see the whole verdict. Both ratchets are
satisfied by moving the words rather than the code.
