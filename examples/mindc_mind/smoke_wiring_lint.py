#!/usr/bin/env python3
"""smoke_wiring_lint.py — machine-checked contract for WHERE each
examples/mindc_mind smoke actually runs.

WHY THIS EXISTS
---------------
The smoke corpus in this directory is the only regression gate for a large
family of constructs `main.mind` does not itself use (refs, enums, Option/Result
payloads, field stores, closures/fn-values, the float/SSE legs of the native-ELF
backend, ...).  For those, "runs in CI" is not a nicety — a regression that is
not executed by CI lands GREEN.

That contract used to be asserted in a *prose comment* on both sides
(.github/workflows/ci.yml said "fast_keystone.sh runs the SAME set ... wire it
into BOTH"; fast_keystone.sh said "wired here so CI protects each landed rung").
Prose cannot be executed, so both statements silently drifted: nine capability
gates ran ONLY in fast_keystone.sh and dozens more ran in neither runner.

This lint replaces the prose with a checked artifact.  SMOKE_WIRING.tsv declares,
for every `*.py` in this directory, which runners are expected to execute it and
why.  The lint recomputes the ACTUAL wiring by parsing the runner scripts and
fails on any divergence, in either direction:

  * a file on disk with no manifest row            -> FAIL (new smoke, unclassified)
  * a manifest row with no file on disk            -> FAIL (stale row)
  * declared runners != actual runners             -> FAIL (drift)
  * an unwired row with no stated reason           -> FAIL (silent gap)

So adding a smoke and forgetting to wire it is now a build error that forces an
explicit, reviewed decision instead of an invisible hole.

RUNNERS PARSED (the complete set that executes smokes in this repo)
  ci        .github/workflows/*.yml
  keystone  examples/mindc_mind/fast_keystone.sh
  preflight scripts/preflight.sh

Usage:
  python3 examples/mindc_mind/smoke_wiring_lint.py            # check (exit 1 on drift)
  python3 examples/mindc_mind/smoke_wiring_lint.py --print    # print the actual wiring table
  python3 examples/mindc_mind/smoke_wiring_lint.py --regen    # rewrite the runners= column from reality
                                                              # (class/note are preserved; review the diff)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SMOKE_DIR = ROOT / "examples" / "mindc_mind"
MANIFEST = SMOKE_DIR / "SMOKE_WIRING.tsv"

# runner label -> files whose text is scanned for smoke invocations
RUNNERS: dict[str, list[Path]] = {
    "ci": sorted((ROOT / ".github" / "workflows").glob("*.yml")),
    "keystone": [SMOKE_DIR / "fast_keystone.sh"],
    "preflight": [ROOT / "scripts" / "preflight.sh"],
}

# Every smoke must be classified as exactly one of these.
#   gate   a standalone regression gate; SHOULD run somewhere (unwired => must justify)
#   helper an importable module / shared fixture, never invoked directly
#   tool   an analysis or measurement utility, not a pass/fail gate
#   wip    landed-but-unfinished; unwired on purpose, with the completion condition
VALID_CLASSES = {"gate", "helper", "tool", "wip"}

NAME_RE = r"[A-Za-z0-9_]+"
# Matches a runner reference with an OPTIONAL nested directory segment. The
# on-disk scan below counts gate scripts one level deep (see the note there), so
# this must too — widening one side and not the other is precisely the
# "one rule, two sites, only one updated" defect this repo keeps finding, and it
# would report permanent DRIFT for a gate that IS correctly wired.
PATH_RE = re.compile(rf"examples/mindc_mind/(?:[A-Za-z0-9_]+/)?({NAME_RE})\.py")
# `for s in a b c ; do ... examples/mindc_mind/$s.py ... done`
LOOP_RE = re.compile(
    rf"for\s+({NAME_RE})\s+in\s+(.*?);\s*do(.*?)\bdone\b", re.DOTALL
)


def _strip_noise(text: str) -> str:
    """Drop lines that MENTION a smoke without EXECUTING it.

    Shell/YAML comments and YAML `- name:` step titles routinely quote a smoke
    filename for documentation; counting those as execution is exactly the kind
    of prose-equals-reality error this lint exists to catch.
    """
    keep = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("#") or s.startswith("- name:") or s.startswith("name:"):
            continue
        keep.append(line)
    return "\n".join(keep)


def executed_smokes(paths: list[Path]) -> set[str]:
    found: set[str] = set()
    for p in paths:
        if not p.is_file():
            continue
        text = _strip_noise(p.read_text(encoding="utf-8"))
        found.update(PATH_RE.findall(text))
        for var, words, body in LOOP_RE.findall(text):
            if f"examples/mindc_mind/${var}.py" not in body and \
               f"examples/mindc_mind/${{{var}}}.py" not in body:
                continue
            for w in words.replace("\\", " ").split():
                if re.fullmatch(NAME_RE, w):
                    found.add(w)
    return found


def actual_wiring() -> dict[str, set[str]]:
    # Gate-shaped scripts NESTED below this directory count too. A `*.py` glob on
    # SMOKE_DIR alone had a blind spot exactly one level deep: the ONLY
    # cross-implementation gate for the DTK register allocator lives at
    # testdata/dtk_plan_parity_smoke.py, was referenced by no runner, and was
    # invisible to THIS lint — the check whose entire job is finding unwired gates
    # could not see it. A meta-gate with a scan blind spot is the failure mode it
    # exists to prevent, one directory deeper.
    on_disk = {p.stem for p in SMOKE_DIR.glob("*.py")}
    for sub in SMOKE_DIR.rglob("*.py"):
        if sub.parent == SMOKE_DIR:
            continue
        if "__pycache__" in sub.parts:
            continue
        if sub.stem.endswith(("_smoke", "_gate", "_lint")):
            on_disk.add(sub.stem)
    wiring: dict[str, set[str]] = {n: set() for n in on_disk}
    for label, paths in RUNNERS.items():
        for name in executed_smokes(paths):
            if name in wiring:
                wiring[name].add(label)
    return wiring


def parse_manifest() -> tuple[dict[str, tuple[set[str], str, str]], list[str]]:
    """-> ({name: (runners, class, note)}, raw_lines)"""
    rows: dict[str, tuple[set[str], str, str]] = {}
    raw = MANIFEST.read_text(encoding="utf-8").splitlines()
    for lineno, line in enumerate(raw, 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 4:
            raise SystemExit(
                f"{MANIFEST}:{lineno}: expected 4 tab-separated columns "
                f"(smoke, runners, class, note), got {len(parts)}"
            )
        name, runners, cls, note = (p.strip() for p in parts)
        if name in rows:
            raise SystemExit(f"{MANIFEST}:{lineno}: duplicate row for {name!r}")
        decl = set() if runners == "none" else {r for r in runners.split(",") if r}
        rows[name] = (decl, cls, note)
    return rows, raw


def fmt(runners: set[str]) -> str:
    order = ["ci", "keystone", "preflight"]
    return ",".join(r for r in order if r in runners) or "none"


def main() -> int:
    argv = sys.argv[1:]
    actual = actual_wiring()

    if "--print" in argv:
        width = max(len(n) for n in actual)
        for name in sorted(actual):
            print(f"{name:<{width}}  {fmt(actual[name])}")
        print(f"\n{len(actual)} smokes; "
              f"ci={sum('ci' in v for v in actual.values())} "
              f"keystone={sum('keystone' in v for v in actual.values())} "
              f"unwired={sum(not v for v in actual.values())}")
        return 0

    # Vacuity guard. A PASS must never be reachable by finding NOTHING: a moved
    # workflows directory, a broken glob or a renamed smoke dir would otherwise
    # make "0 smokes, 0 problems" read as green — the exact false-green shape
    # this lint exists to prevent.
    if not actual:
        print("smoke_wiring_lint: FAIL — no smokes found under "
              f"{SMOKE_DIR}; the corpus path is wrong, not empty.")
        return 1
    if not any("ci" in v for v in actual.values()):
        print("smoke_wiring_lint: FAIL — no smoke is executed by any workflow in "
              f"{ROOT / '.github' / 'workflows'}; either CI wiring was deleted or "
              "the workflow parser stopped matching. Refusing to pass vacuously.")
        return 1

    rows, raw = parse_manifest()

    if "--regen" in argv:
        out = []
        for line in raw:
            if not line.strip() or line.lstrip().startswith("#"):
                out.append(line)
                continue
            name, _, cls, note = (p.strip() for p in line.split("\t"))
            out.append("\t".join([name, fmt(actual.get(name, set())), cls, note]))
        MANIFEST.write_text("\n".join(out) + "\n", encoding="utf-8")
        print(f"regenerated runners= column in {MANIFEST}")
        return 0

    errors: list[str] = []

    for name in sorted(set(actual) - set(rows)):
        errors.append(
            f"UNCLASSIFIED: examples/mindc_mind/{name}.py exists but has no row in "
            f"SMOKE_WIRING.tsv (actual runners: {fmt(actual[name])}). Add a row "
            f"declaring where it runs and why."
        )
    for name in sorted(set(rows) - set(actual)):
        # A `wip` row is allowed to name a file that is not committed yet: it
        # asserts NO coverage, it only records that an unfinished smoke exists in
        # someone's working tree. Failing on it would red a fresh checkout for a
        # file the checkout correctly does not have. Every other class must match
        # disk exactly — a `gate` row with no file is a deleted gate and stays a
        # hard error.
        if rows[name][1] == "wip":
            print(f"  note: wip row {name!r} names a file not present in this "
                  f"checkout (uncommitted work in progress) — not an error.")
            continue
        errors.append(
            f"STALE: SMOKE_WIRING.tsv lists {name!r} but "
            f"examples/mindc_mind/{name}.py does not exist. Remove the row."
        )
    for name in sorted(set(rows) & set(actual)):
        decl, cls, note = rows[name]
        if cls not in VALID_CLASSES:
            errors.append(
                f"BAD CLASS: {name}: {cls!r} not in {sorted(VALID_CLASSES)}"
            )
        if decl != actual[name]:
            errors.append(
                f"DRIFT: {name}: manifest says runners={fmt(decl)} but the runner "
                f"scripts actually execute it in {fmt(actual[name])}. Either wire it "
                f"where the manifest claims, or update the manifest."
            )
        if not actual[name] and not note:
            errors.append(
                f"SILENT GAP: {name} runs in NO runner and gives no reason. State why "
                f"in the note column (or wire it)."
            )

    if errors:
        print("smoke_wiring_lint: FAIL")
        for e in errors:
            print(f"  - {e}")
        print(f"\n{len(errors)} problem(s). "
              f"`python3 examples/mindc_mind/smoke_wiring_lint.py --print` shows the "
              f"actual wiring; `--regen` rewrites the runners= column from reality.")
        return 1

    n_ci = sum("ci" in v for v in actual.values())
    n_ks = sum("keystone" in v for v in actual.values())
    n_un = sum(not v for v in actual.values())
    print(
        f"smoke_wiring_lint: PASS — {len(actual)} smokes classified; "
        f"ci={n_ci} keystone={n_ks} unwired={n_un} (all unwired rows justified)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
