#!/usr/bin/env python3
"""A test must never outlive the rule it asserts.

The incident (2026-08-29): a merge deleted the line that ENFORCED reserved-namespace
rejection. The enum variant, the Display arm, the error mapping and the TEST all
survived. Nothing failed to compile, and the test kept passing — because it exercised
a code path that no longer rejected anything. A green test asserting a rule nothing
enforces is worse than no test: it is a false statement that renews itself on every
CI run.

`lost_by_merge.py` catches the merge-shaped case. This catches the general case, in
either direction, by making the link between a rule and its test GREPPABLE:

    // enforced-by: SIG-ND-01     on the line that actually enforces the rule
    // enforces:    SIG-ND-01     on the test that must fail if that line goes

Then the invariant is a bijection: every id claimed by a test has at least one
enforcement site, and every enforcement site has at least one test. Either half
missing is a defect with a different meaning:

  * test with no enforcement  -> the rule is GONE and the suite is lying (the incident)
  * enforcement with no test  -> the rule can be deleted tomorrow and nothing notices

Exit 0 = every id resolves both ways. Exit 1 = a broken pairing, or zero markers in
a tree that declares required paths (a lint asserting nothing is not a passing lint).
"""
from __future__ import annotations
import pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
ENFORCED_BY = re.compile(r"enforced-by:\s*([A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+)")
ENFORCES = re.compile(r"enforces:\s*([A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+)")
# `sdlc` is excluded because THIS FILE documents the marker syntax in its own
# docstring; scanning it counts the examples as real rules and reports a pairing
# that does not exist. A lint that matches its own documentation is measuring
# itself, not the tree.
SKIP_DIRS = {".git", "target", "node_modules", "__pycache__", ".venv", "dist", "build", "sdlc"}
EXTS = {".rs", ".py", ".ts", ".js", ".go", ".mind", ".sh"}


def scan() -> tuple[dict, dict]:
    enforced: dict[str, list[str]] = {}
    asserts: dict[str, list[str]] = {}
    for p in ROOT.rglob("*"):
        if not p.is_file() or p.suffix not in EXTS:
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "enforce" not in txt:            # cheap pre-filter
            continue
        rel = str(p.relative_to(ROOT))
        for m in ENFORCED_BY.finditer(txt):
            enforced.setdefault(m.group(1), []).append(rel)
        for m in ENFORCES.finditer(txt):
            asserts.setdefault(m.group(1), []).append(rel)
    return enforced, asserts


def main() -> int:
    enforced, asserts = scan()
    problems: list[str] = []

    for rule, where in sorted(asserts.items()):
        if rule not in enforced:
            problems.append(
                f"{rule}: a test claims to assert it ({', '.join(sorted(set(where)))}) but NO "
                f"`enforced-by: {rule}` site exists. Either the enforcement was deleted and the "
                f"test is now green over nothing, or the marker was never added."
            )
    for rule, where in sorted(enforced.items()):
        if rule not in asserts:
            problems.append(
                f"{rule}: enforced at {', '.join(sorted(set(where)))} but NO test carries "
                f"`enforces: {rule}`. Nothing would fail if that line were deleted."
            )

    total = len(set(enforced) | set(asserts))
    print(f"SDLC-GATE enforcement-bijection ran={total} fail={len(problems)}")
    if problems:
        print("\nFAIL: rule/test pairing is broken.\n", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1
    if total == 0:
        print(
            "  no `enforced-by:` / `enforces:` markers found yet — this lint asserts nothing "
            "until rules are marked. Adding markers is how it starts protecting them."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
