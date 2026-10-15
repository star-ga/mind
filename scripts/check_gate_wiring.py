#!/usr/bin/env python3
"""Wiring lint: a whole-tree gate's CI trigger must cover every path it scans.

scripts/check_no_ai_attribution.sh and scripts/check_json_not_evidence.sh both
`git grep` the ENTIRE tracked tree -- their verdict does not depend on which
files a push changed. A `paths:` filter on the workflow that runs them is
therefore a category error: it cannot make the verdict more correct, it can only
suppress the run. That is exactly what had happened -- the trigger listed
'**.md' while the attribution gate scans .md/.rs/.py/.mind/.sh/.toml/.rst/.txt,
so a named-model attribution landing in any source file never started the
workflow, and the json-not-evidence gate (which scans ONLY *.mind) could never
fire on the file class capable of breaking it.

This lint fails if that drift is reintroduced. It checks six things:
  1. every gate STEP the workflow must run is present, as code and not as prose,
     inside the job the release manifest names;
  2. no push/pull_request paths filter excludes any path class they scan;
  3. every row of .github/required-ci-jobs.tsv resolves to a real job, with a
     matching check-run name prefix and a non-empty step list, in the workflow
     it names -- and the workflow carrying these gates HAS a row, so a release
     cannot be cut from a commit where they never ran;
  4. the tracked hooks directory runs them too (local defence-in-depth);
  5. that directory carries every hook a maintainer needs, because
     core.hooksPath selects ONE directory and silently drops the rest;
  6. the tracked harness surface (.py/.sh) stays at or below its recorded
     ceiling, so "one more small script" per wave has to be decided rather
     than merely accumulate.

Check 3 is the reason this file grew a manifest reader. docs-claims.yml owns the
gates that keep a named-model credit out of a PUBLIC repo -- in file contents AND
in commit messages, which no later commit can correct -- yet it had no row in the
release manifest, which mirrored ci.yml alone. verify_ci_green.py therefore only
asked whether docs-claims was RED for the released commit, and a workflow that
never ran is not red. Deleting the commit-message step, or the whole workflow,
weakened the release contract with every gate still green.

Dependency-free (no PyYAML): the `on:` block is parsed by indentation, and the
scan scope is read out of the scripts themselves so the two can never disagree
silently. Fails closed -- an unparseable input is an error, not a pass.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# The repo tracks no Python build products and .gitignore has no __pycache__
# rule; importing a sibling module would create one on every local run.
sys.dont_write_bytecode = True

from workflow_scan import (  # noqa: E402
    blocks,
    cargo_test_execution_argv,
    cargo_test_matrix,
    job_display_name,
    name_prefix,
    workflow_jobs,
)

WORKFLOW_DIR = ".github/workflows"
GATE_WORKFLOW = "docs-claims.yml"
GATE_JOB = "docs-claims"
WORKFLOW = f"{WORKFLOW_DIR}/{GATE_WORKFLOW}"
MANIFEST = ".github/required-ci-jobs.tsv"
# An unqualified manifest row means this workflow (see the manifest header).
DEFAULT_WORKFLOW = "ci.yml"
HOOK = ".githooks/pre-commit"
GATES = (
    "scripts/check_no_ai_attribution.sh",
    "scripts/check_json_not_evidence.sh",
)
# Events whose trigger must cover the whole tree. workflow_dispatch is manual.
GATED_EVENTS = ("push", "pull_request")

# --------------------------------------------------------------------------
# HARNESS SURFACE RATCHET
#
# This repository's product surface is MIND and Rust; every tracked .py/.sh is
# HARNESS -- a CI gate, a driver, or a mutation proof for one. Harness is
# legitimate and load-bearing, and it is also the one file class that grows
# without anybody deciding to grow it: each wave that closes a finding tends to
# add "one more small script", and the total is visible only to whoever thinks
# to count it. A prose baseline recorded in a note is not a baseline -- nothing
# reads it, and the next wave inherits a number it never sees.
#
# So the baseline lives here as an executable CEILING. Raising it is a
# deliberate edit to this line, in the same commit as the files that need the
# room, with the reason in the commit message; that is exactly the friction a
# ratchet is for. Folding a self-test into the gate it proves (the accepted
# pattern: `<gate>.py --self-test`) LOWERS the count, and lowering the ceiling
# to match is what makes the ratchet bite the next time.
# 247 -> 249. Two mutation proofs that could not be folded joined the harness in
# this integration: scripts/test_exec_semantics_gate.py (nine replayed tier logs
# proving the executable-semantics gate attributes a failing target instead of
# exiting 0) and scripts/test_smoke_wiring_lint.py (five fixture repos proving the
# smoke-wiring lint reds a class=gate row that reaches no workflow, and reds an
# "UNCONDITIONALLY" claim written from inside preflight's `--full` branch). Both
# drive their gate through its REAL CLI as a subprocess, so neither can be folded
# behind a `--self-test` flag without importing the gate it is meant to mutate.
# 249 -> 250 for examples/mindc_mind/selfhost_so_provenance_smoke.py, which is not a
# mutation proof at all but a GATE in its own right: it is the only thing asserting
# that _selfhost_so.resolve_so() still refuses an oracle it cannot prove fresh, and
# that no sibling smoke opts out of that contract unnamed.
# 250 -> 251 -> 250: the exec_semantics_markers.sh extraction was reversed.
# Historical rationale moved to docs/gates/exec-semantics-tiers.md; the shell
# functions remain together in exec_semantics_gate.sh, below its source ceiling.
# 252 -> 257 adds five advance/LOOP modules under the same source ceilings.
# 257 -> 258 adds self_host_alias_controls_smoke.py for alias/scope/join controls.
# 258 -> 259 adds the required, non-vacuous two-feature pqc_hybrid_ci.sh gate.
# 259 -> 260 adds the required pure-MIND v04 prefix mirror gate driver.
# The signing publication also adds pqc_hybrid_cli_ci.sh, the required three-OS
# gate for actual-CLI signing controls. It pins the exact test names, requires
# each to pass once, and requires the final tally, so cfg-out tests cannot report
# a false zero-pass success. It gates the shipped binary separately from the
# library gate. Both tracked controls are retained in the ratchet.
# 263 -> 285: compiler-delivery integration census, 42cadc3c -> b4ab203d.
# +12 independently wired gates in examples/mindc_mind: intrinsic_id_table_lint,
# self_host_bounded_profile_smoke, and self_host_native_{alloc_bounded,cf,
# div_trap,f64_intrinsics,f64_lit,main_args,misc_census,process,shadow_scope,sys}_smoke.
# Their CI runners are checked by SMOKE_WIRING.tsv; no new gate is exempted.
# +5 imported helpers: _cf_{corpus,giant,oracle}, _shadow_scope_lang, _w8_native.
# These retain the independent model/capacity/oracle and shared-driver boundaries.
# +1 allocator-byte regeneration utility: testdata/native_alloc_bounded/rt/gen_mind.py.
# +5 std_process_wait acceptance files: _harness.py, accept.py, cases_spawn.py,
# cases_status.py, run_cases.sh. They preserve the process-isolated native fault
# checks and module-size split; the runner is manual, not a claim of CI execution.
# -1 obsolete tools/cargo-deny-sanitize.sh. Total: 263 + 23 - 1 = 285.
# Keep counting ALL tracked .py/.sh, including helpers/tools/manual drivers; do
# not hide this growth by changing the pathspec or classifying files out of it.
# 285 -> 287 adds native_dynamic_writer_check.py and
# native_cabi_registry_smoke.py. These standalone diagnostics require explicit
# compiler/native/oracle artifacts. Their production integration and CI runner
# dependencies remain open and are recorded in SMOKE_WIRING.
# 287 -> 288 adds self_host_source_note_smoke.py: native PT_NOTE bounds/identity
# refusals and exact HOME-lowered string bytes/source digest, wired in ci.yml.
# 288 -> 289 adds the standalone native_dynamic_layout_check.py transport.
# Its explicit native/MLIR prerequisites and manual execution remain visible.
# 289 -> 290 adds native_dynamic_image_check.py, a standalone checked-layout
# writer extension diagnostic with explicit native/MLIR artifacts and no
# implicit CI execution claim. Existing writer and planner gates stay intact.
HARNESS_CEILING = 290  # Exact reviewed integration inventory; no spare allowance.
# Read as a git PATHSPEC against the index: the tree on disk carries untracked
# scratch files whose count is nobody's contract, and a working-tree glob would
# make this gate's verdict depend on what happens to be lying around.
HARNESS_PATHSPEC = ("*.py", "*.sh")


def tracked_harness_files(root: Path) -> list[str]:
    """Every TRACKED .py/.sh path, newline-split off `git ls-files`."""
    out = subprocess.run(
        ["git", "ls-files", "--", *HARNESS_PATHSPEC],
        capture_output=True, text=True, cwd=root, check=True,
    ).stdout
    return [ln for ln in out.splitlines() if ln.strip()]


def check_harness_ratchet(root: Path, failures: list[str]) -> None:
    """The harness file count must not exceed the recorded ceiling."""
    files = tracked_harness_files(root)
    n = len(files)
    if n == 0:
        # An empty result is a broken pathspec, not a repository with no
        # harness -- and `0 <= ceiling` is the shape of a vacuous pass.
        failures.append(
            "harness ratchet: `git ls-files` matched no .py/.sh at all. That is "
            "a broken pathspec, not a clean tree; a count of zero can never "
            "breach a ceiling, so this check must fail closed on it."
        )
        return
    if n > HARNESS_CEILING:
        failures.append(
            f"harness ratchet: {n} tracked .py/.sh files exceeds the recorded "
            f"ceiling of {HARNESS_CEILING} by {n - HARNESS_CEILING}. A gate's "
            "mutation proof belongs INSIDE the gate it proves "
            "(`python3 <gate>.py --self-test`), not in a new file. Fold it, or "
            "raise HARNESS_CEILING in scripts/check_gate_wiring.py in the same "
            "commit and say in the message what the new files buy."
        )
        return
    print(f"[PASS] harness ratchet: {n} tracked .py/.sh <= ceiling {HARNESS_CEILING}")

# (workflow file, job id) -> scripts that job MUST invoke as a step.
# Checked against the job body with comment lines removed: a workflow that
# merely mentions a gate in prose is not a workflow that runs it, and that is
# precisely how a deleted step would otherwise keep passing a substring test.
REQUIRED_STEPS: dict[tuple[str, str], tuple[str, ...]] = {
    (GATE_WORKFLOW, GATE_JOB): (
        "scripts/check_gate_wiring.py",
        "scripts/test_gate_wiring.py",
        "scripts/check_claims.py",
        # The mutation proofs for check_claims.py's three derived comparisons
        # (cost, printed counts, and the floor ratchet), in one file with one
        # step. The gate above only
        # reports that the docs currently agree with the tree; these prove the
        # comparisons BITE. A silently deleted mutation proof leaves a check
        # that can stop comparing and still ride green.
        "tests/check_claims_gate_tests.py",
        "scripts/check_no_ai_attribution.sh",
        "scripts/check_json_not_evidence.sh",
        # The commit-message gate. A file can be corrected by the next commit;
        # a message cannot be corrected without rewriting every descendant, so
        # its step is the one whose deletion costs the most and shows the least.
        "scripts/check_commit_messages.sh",
        "scripts/check_release_gating.py",
    ),
}

# core.hooksPath points at exactly ONE directory, and git reports nothing when a
# hook is simply absent -- so a hooks directory missing an entry looks identical
# to one that ran clean. Every hook the documented install promises must exist
# here, and each must reach the script that holds the rules.
REQUIRED_HOOKS: dict[str, tuple[str, ...]] = {
    ".githooks/pre-commit": GATES + ("scripts/anatomy-hook.sh",),
    ".githooks/commit-msg": ("scripts/commit-msg-hook.sh",),
    ".githooks/post-commit": ("hooks.local",),
    ".githooks/post-merge": ("hooks.local",),
}


def repo_root() -> Path:
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True, text=True, check=True,
    )
    return Path(out.stdout.strip())


def scanned_globs(script: Path) -> list[str]:
    """Extension globs the script passes to `git grep` as a pathspec.

    Read off the real `git grep` command line, not the whole file: these scripts
    carry prose comments containing apostrophes ("json's", "grep's") that would
    desynchronise naive quote pairing. Comment lines are dropped first, then
    line continuations are joined so a multi-line pathspec is seen whole.
    Exclusions are git pathspec magic (':!...') and never match the '*.ext'
    shape, so only positive entries are returned.
    """
    code = [
        ln for ln in script.read_text(encoding="utf-8").splitlines()
        if not ln.lstrip().startswith("#")
    ]
    logical = "\n".join(code).replace("\\\n", " ")
    # A gate may hold its pathspec in a `PATHSPEC=( ... )` array so that the
    # `git ls-files` corpus-floor count and the `git grep` scan read the SAME
    # list (one source, no drift). Flatten any such array onto one synthetic
    # `git grep` line so this parser keeps seeing the real scan scope; without
    # it the array form reads as "no scope" and fails this lint closed.
    for body in re.findall(r"PATHSPEC=\((.*?)\)", logical, re.DOTALL):
        logical += "\ngit grep " + " ".join(body.split())
    globs: set[str] = set()
    for line in logical.splitlines():
        if "git grep" not in line:
            continue
        for tok in re.findall(r"'([^']*)'", line):
            if re.fullmatch(r"\*\.[A-Za-z0-9]+", tok):
                globs.add(tok)
    return sorted(globs)


def parse_on_block(workflow_text: str) -> dict[str, dict[str, list[str]]]:
    """Return {event: {'paths': [...], 'paths-ignore': [...]}} from the `on:` block."""
    lines = workflow_text.splitlines()
    start = next((i for i, ln in enumerate(lines) if re.match(r"^on:\s*$", ln)), None)
    if start is None:
        sys.exit(f"check_gate_wiring: FAIL - no top-level `on:` block in {WORKFLOW}")

    end = len(lines)
    for i in range(start + 1, len(lines)):
        ln = lines[i]
        if ln.strip() and not ln.startswith((" ", "\t")) and not ln.lstrip().startswith("#"):
            end = i
            break
    block = lines[start + 1:end]

    events: dict[str, dict[str, list[str]]] = {}
    event = None
    key = None
    for ln in block:
        if not ln.strip() or ln.lstrip().startswith("#"):
            continue
        indent = len(ln) - len(ln.lstrip())
        stripped = ln.strip()
        if indent == 2 and stripped.endswith(":"):
            event = stripped[:-1]
            events.setdefault(event, {})
            key = None
        elif indent == 4 and event and stripped.rstrip(":") in ("paths", "paths-ignore"):
            key = stripped.rstrip(":")
            events[event].setdefault(key, [])
        elif indent == 4 and event:
            key = None  # some other sub-key, e.g. `branches:`
        elif indent >= 6 and stripped.startswith("- ") and event and key:
            events[event][key].append(stripped[2:].strip().strip("'\""))
    return events


def gh_glob_to_regex(pat: str) -> re.Pattern[str]:
    """GitHub Actions filter-pattern semantics: `**` spans `/`, `*` and `?` do not."""
    out, i = [], 0
    while i < len(pat):
        if pat.startswith("**", i):
            out.append(".*")
            i += 2
        elif pat[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pat[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(pat[i]))
            i += 1
    return re.compile("^" + "".join(out) + "$")


def representatives(root: Path, glob: str) -> list[str]:
    """Real tracked files matching an extension glob, at root depth and nested."""
    ext = glob[1:]  # '*.rs' -> '.rs'
    out = subprocess.run(
        ["git", "ls-files", "--", glob],
        capture_output=True, text=True, cwd=root, check=True,
    )
    files = [f for f in out.stdout.splitlines() if "node_modules" not in f]
    if not files:
        # No tracked file of this type YET. Probe synthetically anyway, or a glob
        # the gates scan but the tree does not currently contain (e.g. *.rst)
        # would be checked vacuously and could regress unnoticed.
        return [f"probe{ext}", f"dir/probe{ext}"]
    nested = next((f for f in files if "/" in f), None)
    flat = next((f for f in files if "/" not in f), None)
    picked = [f for f in (flat, nested) if f]
    return picked or [files[0]]


def triggers(path: str, filt: dict[str, list[str]]) -> bool:
    """Would a push/PR changing exactly `path` start this workflow?"""
    if "paths" in filt:
        return any(gh_glob_to_regex(p).match(path) for p in filt["paths"])
    if "paths-ignore" in filt:
        return not any(gh_glob_to_regex(p).match(path) for p in filt["paths-ignore"])
    return True  # no filter -> always runs



def code(text: str) -> str:
    """`text` with comment lines dropped.

    Every check below asks what a workflow or a hook DOES. A YAML comment that
    names a gate is documentation, not an invocation, and a substring test that
    cannot tell them apart lets a step be deleted while a comment above it keeps
    the lint green -- the exact shape this file exists to refuse.
    """
    return "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))


def steps_of(job_body: str) -> list[str]:
    """The `- ...` entries of a job's `steps:` list, as raw text blocks."""
    body = blocks(job_body, 4).get("steps", "")
    out: list[str] = []
    for line in body.splitlines():
        if line.startswith("      - "):
            out.append(line)
        elif out:
            out[-1] += "\n" + line
    return out


def parse_manifest(root: Path, failures: list[str]) -> list[tuple[str, str, str]]:
    """Rows as (workflow file, job id, check-run name prefix).

    An unqualified job id means DEFAULT_WORKFLOW, so rows written before the
    qualifier existed keep their meaning byte-for-byte.
    """
    path = root / MANIFEST
    if not path.is_file():
        failures.append(f"missing {MANIFEST} - the release-required set is unreadable")
        return []
    rows: list[tuple[str, str, str]] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        parts = raw.split("\t")
        if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
            failures.append(f"{MANIFEST}:{lineno}: expected '<job id>TAB<name prefix>', got {raw!r}")
            continue
        jid, prefix = parts[0].strip(), parts[1]
        workflow, sep, tail = jid.partition(":")
        if not sep:
            workflow, tail = DEFAULT_WORKFLOW, jid
        rows.append((workflow, tail, prefix))
    if not rows:
        failures.append(f"{MANIFEST} declares no required jobs - the release gate would be vacuous")
    return rows


def check_manifest(root: Path, failures: list[str]) -> None:
    """Every required row must resolve to a real, non-empty job."""
    rows = parse_manifest(root, failures)

    # The whole point of the finding: the workflow that owns the attribution
    # gates must itself be in the release-required set. Without a row, a release
    # is verified only against "docs-claims is not RED", which a workflow that
    # never ran satisfies trivially.
    if rows and not any(wf == GATE_WORKFLOW for wf, _, _ in rows):
        failures.append(
            f"{MANIFEST} has no row for {GATE_WORKFLOW} - the public-artifact "
            "hygiene gates (file contents AND commit messages) sit OUTSIDE the "
            "release-required set, so a release can be cut from a commit where "
            "they never ran"
        )

    for workflow, jid, prefix in rows:
        wf_path = root / WORKFLOW_DIR / workflow
        if not wf_path.is_file():
            failures.append(f"{MANIFEST} names workflow {workflow!r}, which does not exist")
            continue
        jobs = workflow_jobs(wf_path)
        if jid not in jobs:
            failures.append(
                f"{MANIFEST} requires job {jid!r} of {workflow}, which has no such job "
                f"(it declares {sorted(jobs) or '(none)'}) - the release verifier would "
                "look for a check-run that is never produced"
            )
            continue
        body = jobs[jid]
        declared = name_prefix(job_display_name(body) or jid)
        if declared != prefix:
            failures.append(
                f"{workflow} job {jid!r}: manifest prefix {prefix!r} != workflow name "
                f"prefix {declared!r}"
            )
        if not steps_of(body):
            failures.append(
                f"{workflow} job {jid!r} declares no steps - a required gate that runs "
                "nothing is a green check-run asserting nothing"
            )


def shell_logical_lines(text: str) -> list[str]:
    """Join backslash continuations so one shell command is one string.

    preflight writes its keystone invocation across three physical lines.  A
    lint that reads physical lines sees ``--features`` and ``--test`` on
    different lines and cannot tell which command they belong to, so it falls
    back to searching the whole file -- which comments and unrelated commands
    can satisfy.
    """
    joined: list[str] = []
    buf = ""
    for raw in text.splitlines():
        stripped = raw.rstrip()
        if stripped.endswith("\\"):
            buf += stripped[:-1].rstrip() + " "
            continue
        buf += stripped
        joined.append(buf)
        buf = ""
    if buf:
        joined.append(buf)
    return joined


def preflight_keystone_commands(text: str) -> list[str]:
    """Every EXECUTABLE keystone cargo-test command in preflight.

    Comment lines are dropped: a gate named only in prose is not a gate.  This
    returns commands rather than literals so a caller compares what preflight
    actually runs, not the first ``--features`` string that appears anywhere in
    the file.
    """
    out: list[str] = []
    for line in shell_logical_lines(text):
        body = line.strip()
        if body.startswith("#"):
            continue
        if "cargo test" in body and "phase_g_keystone_bootstrap" in body:
            out.append(body)
    return out


def command_features(command: str) -> set[str]:
    """The feature set one shell command declares, empty when it declares none.

    Fails closed on a shape it does not understand: an unquoted or variable
    ``--features`` yields the empty set, which then mismatches CI and reports,
    rather than silently passing an unparsed command.
    """
    for pattern in (r'--features\s+"([^"]*)"', r"--features\s+'([^']*)'",
                    r"--features\s+([A-Za-z0-9_,-]+)"):
        found = re.search(pattern, command)
        if found:
            return set(found.group(1).replace(",", " ").split())
    return set()


def preflight_test_exclusions(text: str) -> list[str] | None:
    """The declared shrink-only exclusion set, derived and never hardcoded.

    ``None`` means preflight declares no exclusion array at all.  An empty list
    means it declares one and it is empty, which is the only direction this
    contract allows the set to move without a new ruling.
    """
    found = re.search(r"PREFLIGHT_TEST_EXCLUSIONS=\(([^)]*)\)", text)
    if not found:
        return None
    return [word.strip().strip('"').strip("'")
            for word in found.group(1).split() if word.strip()]


def check_preflight_feature_parity(root: Path, failures: list[str]) -> None:
    """Require preflight's cargo-test loop to mirror CI's build_test matrix."""
    ci_path = root / WORKFLOW_DIR / "ci.yml"
    preflight_path = root / "scripts/preflight.sh"
    if not ci_path.is_file() or not preflight_path.is_file():
        failures.append("WSP-06(a): ci.yml or scripts/preflight.sh is missing")
        return
    jobs = workflow_jobs(ci_path)
    text = preflight_path.read_text(encoding="utf-8")
    rows = cargo_test_matrix(jobs.get("build_test", ""))
    if len(rows) < 10:
        failures.append(
            f"WSP-06(a): ci.yml build_test exposes only {len(rows)} cargo-test "
            "invocations; expected at least 10 (matrix derivation is vacuous)"
        )
    ci_features = {row["features"] for row in rows}
    # The keystone is a separate CI job and must have the exact feature set
    # preflight runs. Compare parsed invocations, both directions, rather than
    # searching prose (comments and unrelated test commands are not evidence).
    keystone_rows = []
    for body in jobs.values():
        keystone_rows.extend(
            row for row in cargo_test_matrix(body)
            if "phase_g_keystone_bootstrap" in row.get("selector", "")
        )
    expected_keystone = {"mlir-build", "std-surface", "cross-module-imports"}
    if len(keystone_rows) != 1:
        failures.append(
            f"WSP-06(a): expected exactly one parsed keystone invocation, found {len(keystone_rows)}"
        )
    else:
        actual_keystone = set(keystone_rows[0]["features"].replace(",", " ").split())
        if actual_keystone != expected_keystone:
            failures.append(
                "WSP-06(a): keystone CI feature set differs from required exact set: "
                f"{sorted(actual_keystone)} vs {sorted(expected_keystone)}"
            )
        # Compare the ACTUAL preflight keystone commands against CI, in both
        # directions. Scanning for the first --features literal that mentions
        # mlir-build accepts a comment, a different gate, or a stale line.
        for command in preflight_keystone_commands(text):
            preflight_keystone = command_features(command)
            if preflight_keystone != actual_keystone:
                failures.append(
                    "WSP-06(a): preflight keystone feature set does not equal CI keystone: "
                    f"{sorted(preflight_keystone)} vs {sorted(actual_keystone)}"
                )
                break
    required = {"", "std-surface", "cross-module-imports",
                "std-surface,cross-module-imports", "autodiff",
                "mlir-lowering", "cpu-buffers", "ffi-c"}
    missing_ci = sorted(required - ci_features)
    if missing_ci:
        failures.append(
            "WSP-06(a): ci.yml build_test is missing expected cargo-test "
            f"feature rows: {missing_ci}"
        )
    text = preflight_path.read_text(encoding="utf-8")
    if "scripts/workflow_scan.py run_cargo_test_matrix" not in text:
        failures.append(
            "WSP-06(a): preflight has no workflow_scan run_cargo_test_matrix call; "
            "CI feature rows have no derived preflight leg"
        )
    if "matrix_rc=$?" not in text or "workflow cargo-test matrix execution FAILED" not in text:
        failures.append(
            "WSP-06(a): preflight does not preserve the typed matrix executor's failure status"
        )
    if any(token in text for token in ("base64 -d", "mapfile -t row_fields", "eval ")):
        failures.append("WSP-06(a): preflight reparses typed matrix argv through a shell decoder")
    for row in rows:
        original = list(row.get("argv", []))
        if not original:
            failures.append("WSP-06(a): scanner accepted a cargo-test row with empty argv")
            continue
        unlocked = cargo_test_execution_argv(row, False)
        locked = cargo_test_execution_argv(row, True)
        for label, argv in (("unlocked", unlocked), ("locked", locked)):
            separator = argv.index("--") if "--" in argv else len(argv)
            cargo_args = argv[:separator]
            if "--no-fail-fast" not in cargo_args:
                failures.append(f"WSP-06(a): {label} replay lacks Cargo --no-fail-fast before --")
            if ("--locked" in cargo_args) != (label == "locked"):
                failures.append(f"WSP-06(a): {label} replay violates Cargo.lock branch parity")
        separator = original.index("--") if "--" in original else len(original)
        policy = ([] if "--no-fail-fast" in original[:separator] else ["--no-fail-fast"]) + ["--locked"]
        if locked != original[:separator] + policy + original[separator:]:
            failures.append("WSP-06(a): typed replay does not preserve the scanner's exact argv")
    if not any(set(row.get("lock_modes", [])) == {"locked", "unlocked"} for row in rows):
        failures.append("WSP-06(a): CI Cargo.lock locked/unlocked branch was not normalized")
    # Assert what the keystone commands DO, not the exact byte string they were
    # once written as. The previous form required a `BUILD_INCOMPLETE=1` env
    # assignment that no longer exists; a contract test that pins a removed
    # implementation detail forces bad code back in to placate itself.
    keystone_cmds = preflight_keystone_commands(text)
    if len(keystone_cmds) != 2:
        failures.append(
            "WSP-06(a): expected exactly two executable keystone commands in preflight "
            f"(one --no-run build phase, one run phase), found {len(keystone_cmds)}"
        )
    if any("--no-default-features" not in cmd for cmd in keystone_cmds):
        failures.append(
            "WSP-06(a): keystone preflight invocation lacks --no-default-features"
        )
    scanner_text = (root / "scripts/workflow_scan.py").read_text(encoding="utf-8")
    if 'command = ["cargo", "test", *argv]' not in scanner_text or "subprocess.run(" not in scanner_text:
        failures.append(
            "WSP-06(a): typed matrix executor does not execute cargo test (fake success is not evidence)"
        )
    if "passed < 1" not in scanner_text or "failed != 0" not in scanner_text:
        failures.append(
            "WSP-06(a): matrix executor accepts a cargo test row without positive result evidence"
        )
    if "cargo_test_execution_argv(row, locked)" not in scanner_text:
        failures.append(
            "WSP-06(a): matrix executor bypasses the typed argv replay path"
        )
    if "timeout=timeout" not in scanner_text or "except subprocess.TimeoutExpired" not in scanner_text:
        failures.append(
            "WSP-06(a): matrix executor does not preserve per-row CI timeout protection"
        )
    # The split build phase is a STRUCTURE (one --no-run command, one run
    # command) plus a label the build-failure branch emits. It is not an
    # environment variable, and requiring the variable let a build interruption
    # be reported as a determinism regression.
    build_phase = [cmd for cmd in keystone_cmds if "--no-run" in cmd]
    run_phase = [cmd for cmd in keystone_cmds if "--no-run" not in cmd]
    if len(build_phase) != 1 or len(run_phase) != 1:
        failures.append(
            "WSP-06(a): keystone lacks the required split build phase "
            f"(--no-run commands={len(build_phase)}, run commands={len(run_phase)})"
        )
    if "BUILD_INCOMPLETE" not in text:
        failures.append(
            "WSP-06(a): keystone build failure emits no BUILD_INCOMPLETE label; an "
            "interrupted build must not be reported as a determinism result"
        )
    if "byte-identity regression" in text:
        failures.append(
            "WSP-06(a): preflight still claims a byte-identity regression on a path "
            "that may never have evaluated determinism; label the cause it proved"
        )
    if "ks_build_rc" not in text or "ks_rc" not in text:
        failures.append(
            "WSP-06(a): keystone build/run results are not independently fail-closed"
        )
    # Derive the exemption set from preflight instead of naming it here. The old
    # form hardcoded 'mindfuzz_cross_substrate' and so demanded that CI keep a
    # target the exemption existed to skip -- the lint outlived the thing it
    # guarded and then failed on its own removal.
    ci_text = ci_path.read_text(encoding="utf-8")
    declared = preflight_test_exclusions(text)
    if declared is None:
        failures.append(
            "WSP-06(a): preflight declares no PREFLIGHT_TEST_EXCLUSIONS array; the "
            "shrink-only exemption set must be explicit even when it is empty"
        )
    else:
        for excluded in declared:
            if not re.search(
                rf"cargo\s+test[^#]*--test\s+{re.escape(excluded)}(?:\s|$)", ci_text
            ):
                failures.append(
                    f"WSP-06(a): exclusion '{excluded}' has no matching executable CI "
                    "cargo-test target; delete this stale exemption"
                )
    # An exemption applied outside the declared array is an undeclared one.
    for stray in re.findall(r"for\s+excluded\s+in\s+([A-Za-z0-9_][^;\n]*)", text):
        if '"${PREFLIGHT_TEST_EXCLUSIONS[@]}"' not in stray:
            failures.append(
                "WSP-06(a): preflight filters failures through an exemption name that "
                f"is not in the declared array: {stray.strip()!r}"
            )


def check_required_steps(root: Path, failures: list[str]) -> None:
    """Each named gate must be invoked by a step of its job, as code."""
    for (workflow, jid), scripts in REQUIRED_STEPS.items():
        wf_path = root / WORKFLOW_DIR / workflow
        if not wf_path.is_file():
            failures.append(f"missing {WORKFLOW_DIR}/{workflow}")
            continue
        jobs = workflow_jobs(wf_path)
        if jid not in jobs:
            failures.append(f"{workflow} has no job {jid!r} to carry its gate steps")
            continue
        bodies = [code(s) for s in steps_of(jobs[jid])]
        for script in scripts:
            if not any(script in s for s in bodies):
                failures.append(
                    f"{workflow} job {jid!r} no longer runs {script} in any step "
                    "(a comment naming it does not count)"
                )


def index_mode(root: Path, rel: str) -> str | None:
    """The file mode git has RECORDED for `rel`, or None when it is untracked."""
    out = subprocess.run(
        ["git", "ls-files", "-s", "--", rel],
        capture_output=True, text=True, cwd=root, check=True,
    ).stdout.split()
    return out[0] if out else None


def require_executable(root: Path, rel: str, failures: list[str]) -> None:
    """git execs a hook; without the executable bit it is skipped in silence.

    Asserted against the INDEX, not this disk: `core.fileMode=false` (set in
    some clones, including the one this was written in) hides a local chmod from
    git entirely, so a file can be executable here and land as 100644 for
    everyone else.
    """
    mode = index_mode(root, rel)
    if mode is not None and not mode.endswith("755"):
        failures.append(
            f"{rel} is mode {mode} in the index, not 100755 - git will not "
            f"execute it (fix: git update-index --chmod=+x {rel})"
        )


def check_hooks(root: Path, failures: list[str]) -> None:
    """One tracked hooks directory that runs everything the install promises."""
    for hook, needles in REQUIRED_HOOKS.items():
        path = root / hook
        if not path.is_file():
            failures.append(
                f"missing {hook} - `git config core.hooksPath .githooks` selects ONE "
                "directory, so a hook absent from it is silently not run"
            )
            continue
        require_executable(root, hook, failures)
        text = code(path.read_text(encoding="utf-8"))
        for needle in needles:
            if needle not in text:
                failures.append(f"{hook} does not reach {needle}")
            # The payload a hook chains, read OUT of the requirement above
            # rather than hand-listed here: a second list is the drift this
            # lint exists to catch. A wrapper invokes it through `bash`, so its
            # bit is irrelevant THERE -- but an older clone symlinks the payload
            # straight into the hooks directory (that install was documented
            # until this lint replaced it, and is still live in clones made
            # before), git resolves the symlink, and skips a target without the
            # executable bit. 100644 on a payload is a hook that looks
            # installed and runs nothing.
            elif needle.startswith("scripts/"):
                require_executable(root, needle, failures)


def main() -> int:
    root = repo_root()
    wf_path = root / WORKFLOW
    if not wf_path.is_file():
        sys.exit(f"check_gate_wiring: FAIL - missing {WORKFLOW}")
    wf_text = wf_path.read_text(encoding="utf-8")
    failures: list[str] = []

    # (1) every gate STEP must still be there, as code rather than as prose.
    check_required_steps(root, failures)

    # (2) trigger coverage must be a superset of what the gates scan.
    events = parse_on_block(wf_text)
    for event in GATED_EVENTS:
        if event not in events:
            failures.append(f"{WORKFLOW}: `on.{event}` is absent - the gate cannot run for it")
            continue
        filt = events[event]
        for gate in GATES:
            gate_path = root / gate
            if not gate_path.is_file():
                failures.append(f"missing gate script {gate}")
                continue
            globs = scanned_globs(gate_path)
            if not globs:
                failures.append(f"{gate}: could not read its scan scope (fail-closed)")
                continue
            for glob in globs:
                for rep in representatives(root, glob):
                    if not triggers(rep, filt):
                        failures.append(
                            f"on.{event} does not trigger for '{rep}' "
                            f"(scanned by {gate} via {glob}) "
                            f"-- filter={filt or '{}'}"
                        )

    # (3) preflight's cargo-test feature matrix must mirror build_test in both
    # directions; this is structural and does not execute a compiler build.
    check_preflight_feature_parity(root, failures)

    # (4) every release-required row resolves to a real, non-empty job.
    check_manifest(root, failures)

    # (4) local defence-in-depth: one tracked hooks directory runs them too.
    check_hooks(root, failures)

    # (5) the harness surface may not grow past its recorded ceiling.
    check_harness_ratchet(root, failures)

    if failures:
        print("::error::gate wiring is broken - a whole-tree gate is not reachable:")
        for f in failures:
            print(f"  - {f}")
        print("")
        print("These gates `git grep` the ENTIRE tree, so their verdict does not depend")
        print("on which files changed. A `paths:` filter on their workflow can only")
        print("suppress the run, never improve it -- remove the filter rather than")
        print("widening it, or the trigger list and the scan scope drift apart again.")
        return 1

    print("gate-wiring lint: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
