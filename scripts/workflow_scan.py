#!/usr/bin/env python3
"""workflow_scan.py — dependency-free structural reader for the GitHub Actions
workflow YAML in this repo.

WHY NOT PyYAML
--------------
The lints that consume this run in two places with different Pythons: the
`actions/setup-python` interpreter used by `.github/workflows/docs-claims.yml`
(a clean 3.12 with no third-party packages) and whatever `python3` a maintainer
has locally.  A lint that is the gate for a supply-chain property must not be
able to fail *open* because an import was missing, so this module parses the
subset of YAML our workflows actually use with the stdlib only:

  * block mappings with consistent 2-space indentation
  * block sequences (`- item`)
  * inline flow sequences (`[a, b]`)
  * block scalars (`|`, `>`) kept verbatim as opaque text

That is deliberately not a YAML parser.  It answers structural questions
("which jobs exist", "what does this job declare in `needs:`", "what raw text
sits inside this job") and nothing else; anything subtler belongs in a real
parser and a real dependency.
"""

from __future__ import annotations

import re
from pathlib import Path

_KEY_RE = re.compile(r"^(?P<indent> *)(?P<key>[A-Za-z0-9_.\-]+):(?P<rest>.*)$")


def _significant(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and not stripped.startswith("#")


def blocks(text: str, indent: int) -> dict[str, str]:
    """Map every `key:` at exactly `indent` spaces to the raw text beneath it.

    The value excludes the `key:` header line itself but includes any inline
    remainder on that line as the first entry, so `needs: build` and

        needs:
          - build

    both come back as text containing "build".  Order is source order.
    """
    lines = text.splitlines()
    starts: list[tuple[int, str, str]] = []
    for i, line in enumerate(lines):
        if not _significant(line):
            continue
        m = _KEY_RE.match(line)
        if m and len(m.group("indent")) == indent:
            starts.append((i, m.group("key"), m.group("rest").strip()))

    out: dict[str, str] = {}
    for n, (i, key, rest) in enumerate(starts):
        end = len(lines)
        for j in range(i + 1, len(lines)):
            line = lines[j]
            if not _significant(line):
                continue
            leading = len(line) - len(line.lstrip(" "))
            if leading <= indent:
                end = j
                break
        body = "\n".join(lines[i + 1 : end])
        out[key] = (rest + "\n" + body) if rest else body
    return out


def scalar_list(body: str) -> list[str]:
    """Read a `needs:`-shaped value: scalar, `[a, b]` flow list, or `- a` block."""
    body = body.strip()
    if not body:
        return []
    if body.startswith("["):
        inner = body[1 : body.index("]")] if "]" in body else body[1:]
        return [p.strip().strip("'\"") for p in inner.split(",") if p.strip()]
    items = [
        ln.strip()[1:].strip().strip("'\"")
        for ln in body.splitlines()
        if ln.strip().startswith("- ")
    ]
    if items:
        return items
    first = body.splitlines()[0].strip().strip("'\"")
    return [first] if first else []


def workflow_jobs(path: Path) -> dict[str, str]:
    """job id -> raw body text, for one workflow file."""
    text = path.read_text(encoding="utf-8")
    top = blocks(text, 0)
    if "jobs" not in top:
        return {}
    return blocks(top["jobs"], 2)


def job_display_name(job_body: str) -> str | None:
    """The literal `name:` a job declares, or None when it relies on the id."""
    fields = blocks(job_body, 4)
    name = fields.get("name")
    if name is None:
        return None
    return name.splitlines()[0].strip().strip("'\"")


def name_prefix(display_name: str) -> str:
    """The stable prefix of a job's check-run name.

    GitHub renders a matrix job as `<name> (<matrix values>)`, and a `name:`
    that interpolates `${{ matrix.* }}` renders with those values substituted.
    Neither expansion is knowable from the file, so the checked contract is the
    part before the first interpolation — everything to its left is literal and
    stable across matrix edits.
    """
    cut = display_name.find("${{")
    return display_name if cut < 0 else display_name[:cut]
