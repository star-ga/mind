#!/usr/bin/env python3
"""H2b exclusion positive controls (Ruling Q1.3(b), Schedule R7).

For every path listed in testdata/h2b_exclusion_pins.json this gate asserts:

  1. the file's bytes match the committed external digest (pin site is the
     catalogue itself — a digest pinned OUTSIDE any gate run);
  2. `git ls-files --error-unmatch` still lists the path (never silently
     untracked by an exclusion);
  3. the path still appears in ANATOMY.md (anatomy.sh walks
     `git ls-files --cached`, so an ignored-but-tracked file stays listed);
  4. every .gitignore reason line for the path cites this pin site and names
     the sealed module it imports (R7 caller rows).

Red mutants (each must make this gate FAIL):
  --mutant-pin     flip one hex digit in a stored digest
  --mutant-track   pretend `git ls-files` lost the path
  --mutant-anatomy drop the path from the ANATOMY index
  --mutant-reason  strip the pin-site citation from the reason line
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
PINS = ROOT / "examples/mindc_mind/testdata/h2b_exclusion_pins.json"
PIN_SITE = "examples/mindc_mind/testdata/h2b_exclusion_pins.json"


def anatomy_index() -> set[str]:
    section = ""
    indexed: set[str] = set()
    for line in (ROOT / "ANATOMY.md").read_text().splitlines():
        m = re.match(r"### `(.+)/`", line)
        if m:
            section = m.group(1) + "/"
        m = re.match(r"- `([^`]+)`", line)
        if m:
            indexed.add(section + m.group(1))
    return indexed


def gitignore_reasons() -> dict[str, str]:
    """Full comment block immediately above each /examples/*.mind entry."""
    lines = (ROOT / ".gitignore").read_text().splitlines()
    reasons: dict[str, str] = {}
    for i, line in enumerate(lines):
        if line.startswith("/examples/") and line.endswith(".mind"):
            block: list[str] = []
            j = i - 1
            while j >= 0 and lines[j].lstrip().startswith("#"):
                block.append(lines[j])
                j -= 1
            reasons[line[1:]] = "\n".join(reversed(block))
    return reasons


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mutant-pin", action="store_true")
    ap.add_argument("--mutant-track", action="store_true")
    ap.add_argument("--mutant-anatomy", action="store_true")
    ap.add_argument("--mutant-reason", action="store_true")
    args = ap.parse_args()

    doc = json.loads(PINS.read_text())
    entries = doc["entries"]
    reasons = gitignore_reasons()
    indexed = anatomy_index()
    asserted = 0
    failures: list[str] = []

    for entry in entries:
        path = entry["path"]
        want = entry["sha256"]
        if args.mutant_pin and path == entries[0]["path"]:
            want = ("0" if want[0] != "0" else "1") + want[1:]
        blob = (ROOT / path).read_bytes()
        got = hashlib.sha256(blob).hexdigest()
        if got != want:
            failures.append(f"PIN_DRIFT {path} want={want} got={got}")
        else:
            asserted += 1

        tracked = True
        if args.mutant_track and path == entries[0]["path"]:
            tracked = False
        else:
            r = subprocess.run(
                ["git", "ls-files", "--error-unmatch", path],
                cwd=ROOT,
                capture_output=True,
            )
            tracked = r.returncode == 0
        if not tracked:
            failures.append(f"UNTRACKED {path}")
        else:
            asserted += 1

        in_anatomy = path in indexed
        if args.mutant_anatomy and path == entries[0]["path"]:
            in_anatomy = False
        if not in_anatomy:
            failures.append(f"ANATOMY_MISSING {path}")
        else:
            asserted += 1

        reason = reasons.get(path, "")
        if args.mutant_reason and path == entries[0]["path"]:
            reason = reason.replace(PIN_SITE, "testdata/nowhere.json")
        if PIN_SITE not in reason:
            failures.append(f"REASON_NO_PIN_SITE {path} reason={reason!r}")
        else:
            asserted += 1
        if entry["kind"] == "caller-exclusion" and "imports " not in reason:
            failures.append(f"REASON_NO_SEALED_IMPORT {path}")
        else:
            asserted += 1

    print(f"h2b_exclusion_controls: entries={len(entries)} asserted={asserted}")
    for f in failures:
        print(f"FAIL {f}")
    if failures:
        print("h2b_exclusion_controls: FAIL")
        return 1
    print("h2b_exclusion_controls: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
