#!/usr/bin/env python3
"""Docs-claim CI gate — fail if any public surface drifts from config/capabilities.toml.

Conservative regression guard. External review (2026-06) found public claims that
contradicted each other (IR versions, autodiff status, runtime boundary, tool counts);
each was fixed, and this gate stops them from silently drifting back:

  1. forbidden_phrases — specific contradictions that must never reappear on a surface.
  2. canonical IR — at least one version doc must state the canonical mic@1 / mic@3 pair.
  3. [counts] (OPTIONAL) — numbers in the docs re-derived from the real tree, FLAGGED
     on drift. Floor + tolerance keep the false-positive rate low; a missing source
     path SKIPS the entry instead of crashing.

Run from the mind repo root (CI + pre-commit). Exit non-zero on drift.
Low false-positive by design: it only flags exact known-bad phrases and counts that
breach a declared floor/tolerance — never fuzzy matches.

-----------------------------------------------------------------------------------
REUSABLE CROSS-REPO GATE (mindlang.dev + mind-mem)
-----------------------------------------------------------------------------------
This script is repo-agnostic: it reads whatever `config/capabilities.toml` sits next
to it and checks the surfaces that manifest declares. To run the SAME gate against the
SAME shared manifest from a sibling repo (e.g. so mindlang.dev / mind-mem cannot claim
a tool/test/IR-version count that contradicts the canonical mind manifest):

  # Option A — vendor the manifest. Copy config/capabilities.toml into the sibling
  # repo and point SURFACE_GLOBS (below) at that repo's docs, then run:
  #     python3 scripts/check_claims.py

  # Option B — check a sibling tree against the canonical mind manifest in place:
  #     CHECK_CLAIMS_ROOT=/path/to/mindlang.dev \
  #     CHECK_CLAIMS_CAPS=/path/to/mind/config/capabilities.toml \
  #         python3 /path/to/mind/scripts/check_claims.py
  #   CHECK_CLAIMS_ROOT  — tree whose surfaces + [counts] sources are checked.
  #   CHECK_CLAIMS_CAPS  — manifest to check against (defaults to the sibling
  #                        config/capabilities.toml). Use the mind manifest to make
  #                        it the single cross-surface source of truth.
  #   CHECK_CLAIMS_SURFACES — optional ':'-separated glob override for that repo's
  #                        doc layout (e.g. "src/content/**/*.md:public/**/*.html").

  # A sibling's CI step is then just:
  #     - run: python3 ../mind/scripts/check_claims.py
  #       env: { CHECK_CLAIMS_ROOT: ., CHECK_CLAIMS_CAPS: ../mind/config/capabilities.toml }

Counts whose source paths don't exist in the sibling tree are skipped silently, so the
shared manifest can carry mind-specific [counts] without breaking other repos.

  # CHECK_CLAIMS_MODE — "full" (default) runs forbidden-phrase + canonical-IR +
  #   [counts]. "phrases" runs ONLY forbidden-phrase + canonical-IR and skips the
  #   [counts] gate entirely. Use "phrases" from a sibling repo whose tree shape
  #   differs from mind's (e.g. a Python/TS repo with a tests/ dir that holds no
  #   .rs files): there, a mind-specific [counts] glob can resolve to an empty list
  #   rather than "unreachable", producing a false floor breach. The forbidden-phrase
  #   + counts cross-repo value is exactly the regression guard the siblings need.
"""
from __future__ import annotations

import os
import re
import sys
import tomllib
from pathlib import Path

# Repo root + manifest are overridable so a sibling repo can run this exact script
# against the shared manifest (see the cross-repo block above).
_SCRIPT_ROOT = Path(__file__).resolve().parent.parent
ROOT = Path(os.environ.get("CHECK_CLAIMS_ROOT", _SCRIPT_ROOT)).resolve()
_CAPS_PATH = Path(
    os.environ.get("CHECK_CLAIMS_CAPS", ROOT / "config" / "capabilities.toml")
).resolve()
CAPS = tomllib.loads(_CAPS_PATH.read_text())

# Surfaces this repo is responsible for (globs relative to ROOT). Override per-repo
# via CHECK_CLAIMS_SURFACES (':'-separated) for a different doc layout.
_DEFAULT_SURFACES = ["README.md", "STATUS.md", "docs/**/*.md"]
SURFACE_GLOBS = (
    os.environ["CHECK_CLAIMS_SURFACES"].split(":")
    if os.environ.get("CHECK_CLAIMS_SURFACES")
    else _DEFAULT_SURFACES
)

# "full" (default) = forbidden + canonical + counts. "phrases" = forbidden + canonical
# only (skip the mind-specific [counts] gate). Any unrecognised value is treated as
# "full" so a typo can never silently weaken the gate.
MODE = os.environ.get("CHECK_CLAIMS_MODE", "full").strip().lower()
_PHRASES_ONLY = MODE == "phrases"


def surfaces() -> list[Path]:
    out: list[Path] = []
    for g in SURFACE_GLOBS:
        out += [p for p in ROOT.glob(g) if p.is_file()]
    return sorted(set(out))


def check_forbidden(files: list[Path]) -> list[tuple[Path, int, str, str]]:
    forb = CAPS.get("forbidden_phrases", {})
    phrases = [(cat, ph) for cat, lst in forb.items() for ph in lst]
    violations: list[tuple[Path, int, str, str]] = []
    for f in files:
        text = f.read_text(encoding="utf-8", errors="replace")
        low = text.lower()
        for cat, ph in phrases:
            idx = low.find(ph.lower())
            if idx != -1:
                line = text.count("\n", 0, idx) + 1
                violations.append((f, line, cat, ph))
    return violations


def check_canonical(files: list[Path]) -> list[str]:
    ct, cb = CAPS["ir"]["canonical_text"], CAPS["ir"]["canonical_binary"]
    joined = "\n".join(f.read_text(encoding="utf-8", errors="replace") for f in files)
    return [v for v in (ct, cb) if v not in joined]


# --------------------------------------------------------------------------------
# Auto-derived counts. Each derivation is best-effort: a missing/unreadable source
# returns None (the entry is SKIPPED, never crashes), and any unexpected error is
# swallowed into a skip so the gate can never fail a clean checkout on a count bug.
# --------------------------------------------------------------------------------

def _files_for(spec: dict, default_pattern: str = "*") -> list[Path] | None:
    """Resolve a spec's source files.

    Two forms (a spec uses one):
      - `globs = ["src/**/*.rs", "tests/**/*.rs"]`  — ROOT-relative globs.
      - `path = "tests"` (+ optional `pattern`)     — recursive under one dir.
    Returns None when the source can't be reached (entry is then skipped).
    """
    globs = spec.get("globs")
    if globs:
        out: list[Path] = []
        for g in globs:
            out += [p for p in ROOT.glob(g) if p.is_file()]
        # None signals "unreachable" only when no glob root exists at all.
        if not out and not any((ROOT / g.split("/")[0]).exists() for g in globs):
            return None
        return sorted(set(out))

    path = spec.get("path")
    if path is None:
        return None
    base = ROOT / path
    if not base.is_dir():
        return None
    try:
        return sorted(p for p in base.rglob(spec.get("pattern", default_pattern)) if p.is_file())
    except OSError:
        return None


def _derive_glob_count(spec: dict) -> int | None:
    files = _files_for(spec)
    return None if files is None else len(files)


def _derive_regex_count(spec: dict) -> int | None:
    files = _files_for(spec)
    if not files:  # None (unreachable) or [] (no matching files) → can't verify
        return None
    try:
        rx = re.compile(spec["regex"])
    except (re.error, KeyError):
        return None
    total = 0
    for f in files:
        try:
            total += len(rx.findall(f.read_text(encoding="utf-8", errors="replace")))
        except OSError:
            continue
    return total


# A pytest-discovered test: top-level or method `def test_*`.
_PY_TEST_RX = re.compile(r"^\s*def\s+test_\w*\s*\(", re.MULTILINE)


def _derive_py_test_count(spec: dict) -> int | None:
    files = _files_for({**spec, "pattern": "test_*.py"})
    if not files:
        return None
    total = 0
    for f in files:
        try:
            total += len(_PY_TEST_RX.findall(f.read_text(encoding="utf-8", errors="replace")))
        except OSError:
            continue
    return total


_DERIVERS = {
    "glob_count": _derive_glob_count,
    "regex_count": _derive_regex_count,
    "py_test_count": _derive_py_test_count,
}


def check_counts() -> tuple[list[str], list[str]]:
    """Return (drift_messages, info_messages). Drift fails the gate; info is advisory."""
    drift: list[str] = []
    info: list[str] = []
    counts = CAPS.get("counts", {})
    for name, spec in counts.items():
        kind = spec.get("kind")
        deriver = _DERIVERS.get(kind)
        if deriver is None:
            info.append(f"counts[{name}]: unknown kind {kind!r} — skipped")
            continue
        try:
            actual = deriver(spec)
        except Exception:  # noqa: BLE001 — a count bug must never crash the gate
            actual = None
        if actual is None:
            info.append(f"counts[{name}]: source unavailable — skipped")
            continue

        declared = int(spec["declared"])
        mode = spec.get("mode", "exact")
        if mode == "floor":
            if declared > actual:
                drift.append(
                    f"DRIFT [counts/{name}] floor breached: docs claim {declared}+ "
                    f"but tree has {actual} ({spec.get('surface', '?')})"
                )
            else:
                info.append(f"counts[{name}]: OK floor {declared}+ <= {actual}")
        else:  # exact
            tol = int(spec.get("tolerance", 0))
            if abs(declared - actual) > tol:
                drift.append(
                    f"DRIFT [counts/{name}] exact mismatch: docs claim {declared} "
                    f"(±{tol}) but tree has {actual} ({spec.get('surface', '?')})"
                )
            else:
                info.append(f"counts[{name}]: OK exact {declared} ~= {actual} (±{tol})")
    return drift, info


def main() -> int:
    files = surfaces()
    if not files:
        print(f"check_claims: no surfaces found under {ROOT} (run from the repo root)")
        return 2

    ok = True
    for f, line, cat, ph in check_forbidden(files):
        print(f"DRIFT [{cat}] {f.relative_to(ROOT)}:{line}: forbidden phrase reappeared: {ph!r}")
        ok = False

    missing = check_canonical(files)
    if missing:
        print(f"DRIFT [ir]: canonical IR version(s) absent from docs: {missing}")
        ok = False

    if _PHRASES_ONLY:
        print("check_claims: MODE=phrases — forbidden-phrase + canonical-IR only ([counts] skipped)")
    else:
        drift, info = check_counts()
        for line in info:
            print(line)
        for line in drift:
            print(line)
            ok = False

    if ok:
        print(f"check_claims: OK — {len(files)} surfaces consistent with {_CAPS_PATH.name}")
        return 0
    print("check_claims: FAILED — docs drifted from the capability manifest (config/capabilities.toml)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
