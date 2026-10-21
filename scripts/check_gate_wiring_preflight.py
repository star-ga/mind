#!/usr/bin/env python3
"""Preflight cargo-test parity checks extracted from check_gate_wiring.

The wiring lint calls `check_preflight_feature_parity`. This module holds
that check and the shell-command readers it needs. It is harness, counted
by the harness ratchet, and it is not a gate of its own.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.dont_write_bytecode = True

from workflow_scan import (  # noqa: E402
    cargo_test_execution_argv,
    cargo_test_matrix,
    workflow_jobs,
)

WORKFLOW_DIR = ".github/workflows"


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
