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

This lint fails if that drift is reintroduced. It checks three things:
  1. the workflow still invokes both gate scripts;
  2. no push/pull_request paths filter excludes any path class they scan;
  3. the tracked pre-commit hook runs them too (local defence-in-depth).

Dependency-free (no PyYAML): the `on:` block is parsed by indentation, and the
scan scope is read out of the scripts themselves so the two can never disagree
silently. Fails closed -- an unparseable input is an error, not a pass.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

WORKFLOW = ".github/workflows/docs-claims.yml"
HOOK = ".githooks/pre-commit"
GATES = (
    "scripts/check_no_ai_attribution.sh",
    "scripts/check_json_not_evidence.sh",
)
# Events whose trigger must cover the whole tree. workflow_dispatch is manual.
GATED_EVENTS = ("push", "pull_request")


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


def main() -> int:
    root = repo_root()
    wf_path = root / WORKFLOW
    if not wf_path.is_file():
        sys.exit(f"check_gate_wiring: FAIL - missing {WORKFLOW}")
    wf_text = wf_path.read_text(encoding="utf-8")
    failures: list[str] = []

    # (1) the workflow must still invoke both gates.
    for gate in GATES:
        if gate not in wf_text:
            failures.append(f"{WORKFLOW} no longer runs {gate}")

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

    # (3) local defence-in-depth: the tracked hook runs them too.
    hook_path = root / HOOK
    if not hook_path.is_file():
        failures.append(f"missing {HOOK}")
    else:
        hook_text = hook_path.read_text(encoding="utf-8")
        for gate in GATES:
            if gate not in hook_text:
                failures.append(f"{HOOK} does not run {gate}")

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
