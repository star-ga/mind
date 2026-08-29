#!/usr/bin/env python3
"""Fail a merge that silently DROPPED a line both parents agreed on.

The incident this exists for (2026-08-29, commit 9d3d5d41): a merge deleted the
`return Err(ParseMapError::ReservedKey(..))` line that ENFORCED reserved-namespace
rejection, while the enum variant, its Display arm, the error mapping AND its test
all survived. Nothing failed to compile. The only symptom was an unrelated
dead-code warning about a now-unconstructed variant — pure luck. A test asserting
a rule that nothing enforces is worse than no test: it reports green forever.

A bad conflict resolution has exactly one mechanical signature: a line present in
BOTH merge parents but absent from the result. Neither side removed it, so no
diff against either parent shows an intentional deletion — it exists only in the
three-way relationship.

    lost = (lines(P1) INTERSECT lines(P2)) MINUS lines(MERGE)      [multisets]

Multisets, so a line that merely MOVED is a non-event (same count both sides).
A `lost` line that looks like enforcement is a hard FAIL with no trailer override:
a merge is never the right place to delete enforcement. Do it in a real commit
carrying `Removes-Enforcement: <RULE-ID> — <reason>`, where it is reviewable.

Exit 0 = nothing enforcement-shaped was lost. Exit 1 = something was.
Prints `SDLC-GATE lost-by-merge ran=<n> fail=<k>` — `ran=0` on a required check
is itself a failure, so the count is always emitted.
"""
from __future__ import annotations
import collections, re, subprocess, sys

# Lines that CARRY a rule. Deliberately broad: a false positive costs one trailer,
# a false negative ships a rule nobody enforces.
ENFORCEMENT_RE = re.compile(
    r"return\s+Err\(|bail!|ensure!|panic!\(|unreachable!|assert(_eq|_ne)?!\("
    r"|=>\s*(return\s+)?Err|^\s*raise\s|^\s*assert\s|return\s+False|sys\.exit\("
    r"|throw\s|reject\(|kernel\.clamp|enforced-by:"
)

def sh(*args: str) -> str:
    # Binary blobs (frozen ELF seeds, fixtures) live in these trees; decode
    # defensively rather than crashing the gate on them. They are filtered out
    # by is_text() before any line accounting happens.
    r = subprocess.run(args, capture_output=True)
    return r.stdout.decode("utf-8", errors="replace")


def is_text(rev: str, path: str) -> bool:
    """A NUL byte means binary — line-based accounting is meaningless there."""
    raw = subprocess.run(["git", "show", f"{rev}:{path}"], capture_output=True).stdout
    return b"\x00" not in raw[:8192] and bool(raw)

def lines_of(rev: str, path: str) -> collections.Counter:
    blob = sh("git", "show", f"{rev}:{path}")
    return collections.Counter(l.strip() for l in blob.splitlines() if l.strip())

def main() -> int:
    rev = sys.argv[1] if len(sys.argv) > 1 else "HEAD"
    parents = sh("git", "rev-list", "--parents", "-n", "1", rev).split()
    if len(parents) < 3:
        print(f"SDLC-GATE lost-by-merge ran=0 fail=0  ({rev[:8]} is not a merge; nothing to check)")
        return 0
    _, p1, p2 = parents[0], parents[1], parents[2]

    files = [f for f in sh("git", "diff", "--name-only", f"{p1}...{rev}").splitlines() if f]
    files += [f for f in sh("git", "diff", "--name-only", f"{p2}...{rev}").splitlines() if f]
    files = sorted(set(files))

    checked, findings, warned = 0, [], []
    for path in files:
        if not (is_text(p1, path) and is_text(p2, path)):
            continue                      # binary fixture / seed: skip
        l1, l2 = lines_of(p1, path), lines_of(p2, path)
        if not l1 or not l2:
            continue                      # added on one side only: not a lost-line case
        lm = lines_of(rev, path)
        checked += 1
        lost = (l1 & l2) - lm             # Counter & = min per key; - = multiset difference
        for line in lost:
            (findings if ENFORCEMENT_RE.search(line) else warned).append((path, line))

    for path, line in warned[:20]:
        print(f"  note: {path}: dropped a line both parents had: {line[:100]}")
    if warned:
        print(f"  ({len(warned)} non-enforcement line(s) dropped by the merge — review, not blocking)")

    print(f"SDLC-GATE lost-by-merge ran={checked} fail={len(findings)}")
    if findings:
        print(f"\nFAIL: this merge DROPPED {len(findings)} enforcement line(s) present in BOTH parents.\n",
              file=sys.stderr)
        for path, line in findings:
            print(f"  {path}\n      {line[:160]}", file=sys.stderr)
        print("\nNeither parent deleted these, so no two-way diff shows it. If a rule genuinely\n"
              "should go, delete it in a real commit with `Removes-Enforcement: <RULE-ID> — <reason>`\n"
              "so it is reviewable — never as a side effect of a conflict resolution.", file=sys.stderr)
        return 1
    if checked == 0:
        print("FAIL: examined 0 files — the check asserted nothing.", file=sys.stderr)
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
