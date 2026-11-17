#!/usr/bin/env python3
"""Docs-claim CI gate — fail if any public surface drifts from config/capabilities.toml.

Conservative regression guard. External review (2026-06) found public claims that
contradicted each other (IR versions, autodiff status, runtime boundary, tool counts);
each was fixed, and this gate stops them from silently drifting back:

  1. forbidden_phrases — specific contradictions that must never reappear on a surface.
  2. canonical IR — at least one version doc must state the canonical mic@1 / mic@3 pair.
  3. [counts] (OPTIONAL) — numbers in the docs re-derived from the real tree, FLAGGED
     on drift. Floor + tolerance keep the false-positive rate low; a missing source
     path SKIPS the entry instead of crashing. TWO legs: the manifest's `declared`
     number against the tree, AND — when the entry declares `surface_regex` — the
     number PRINTED ON THE SURFACE against the tree. The second leg exists because
     the first one cannot see the docs at all: README typed "~1,390 tests across 174
     test files" while the tree held 325 files and 2,217 `#[test]`s, and a manifest
     floor of 156 kept reporting `[PASS] 156+ <= 325` throughout. A floor is a
     RATCHET: it also declares `floor_tolerance`, the maximum lag it may sit below
     the tree, so a floor that stops ratcheting fails instead of passing.
  4. cost claim — the published "MIC saves $N/year" figure RE-DERIVED from the price
     input in config/token_pricing.toml and the tokenizer-measured counts in the
     benchmark output, then required verbatim on every declared surface. It is the
     one check that fails CLOSED on a missing input: a dollar figure whose inputs
     cannot be read is exactly the drift this gate exists to stop.

  5. installer pipelines — a shown `NAME=x A | sh` hands NAME to A, never to the
     shell that runs the script, and `irm URL | iex` passes no script parameter.
     Every surface, and the usage comments of scripts/install.*, must write a MIND_*
     variable on the shell stage and show -Version only in the script block form. A
     page that exists to show the install command (docs/install.md) must show one.

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

import json
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
        hits: list[tuple[Path, int, str, str]] = []
        for cat, ph in phrases:
            idx = low.find(ph.lower())
            if idx != -1:
                line = text.count("\n", 0, idx) + 1
                hits.append((f, line, cat, ph))
        # One verdict line per surface ACTUALLY read. scripts/gate_assert.py
        # derives this gate's assertion count from these lines, so a surface
        # glob that resolved to nothing now reports zero instead of a total
        # this function typed out of `len(files)`.
        print(f"[{'FAIL' if hits else 'PASS'}] surface {f.relative_to(ROOT)}")
        violations.extend(hits)
    return violations


def check_canonical(files: list[Path]) -> list[str]:
    ct, cb = CAPS["ir"]["canonical_text"], CAPS["ir"]["canonical_binary"]
    joined = "\n".join(f.read_text(encoding="utf-8", errors="replace") for f in files)
    return [v for v in (ct, cb) if v not in joined]


# --------------------------------------------------------------------------------
# Installer pipelines. In `NAME=x A | B` the variable reaches A only. A page that
# wrote the version pin before the download tool gave the pin to that tool, the
# installer never saw it, and the command installed the latest release. A MIND_*
# assignment on a stage that feeds a shell is therefore always wrong, whatever the
# first command is.
# --------------------------------------------------------------------------------

_SEQ_RX = re.compile(r"\|\||&&|;")
# A shell stage: optional sudo/env wrapper and assignments, then sh/bash/zsh/dash/ksh.
_SINK_RX = re.compile(
    r"\s*(?:(?:env|sudo)\s+(?:-\S+\s+)*)?(?:[A-Za-z_]\w*=\S*\s+)*"
    r"(?:\S*/)?(?:ba|z|da|k)?sh(?=[\s`.,;:)]|$)"
)
# Assignments that start a command: at the line start, after a backtick (prose that
# quotes a command) or after a prompt mark. `export X=1; cmd` never matches.
_ENV_RUN_RX = re.compile(
    r"(?:^\s*|`\s*|(?:^|\s)[$>#]\s+)(?:(?:env|sudo)\s+)?((?:[A-Za-z_]\w*=\S*\s+)+)\S"
)
# PowerShell: `irm URL | iex` runs the text with no arguments, so a script parameter
# written on irm or iex never reaches install.ps1 (the script block form does).
_PS_SINK_RX = re.compile(r"\s*(?:iex|invoke-expression)(?=[\s`.,;:)]|$)", re.IGNORECASE)
_PS_ARG_RX = re.compile(r"\s(-(?:Version|InstallDir|DryRun))\b", re.IGNORECASE)
_INSTALL_PAGE = "docs/install.md"


def _logical_lines(text: str):
    """(first line number, text) with `\\` and trailing-`|` continuations joined."""
    buf: list[str] = []
    start = 0
    for n, raw in enumerate(text.splitlines(), 1):
        if not buf:
            start = n
        body = raw.rstrip()
        cont = body.endswith("\\") or (body.endswith("|") and not body.lstrip().startswith("|"))
        buf.append(body[:-1] if body.endswith("\\") else body)
        if not cont:
            yield start, " ".join(buf)
            buf = []
    if buf:
        yield start, " ".join(buf)


def installer_pipelines(text: str):
    """Yield (line, arguments given to a stage before the shell) per pipe-to-shell.

    The names are MIND_* variables for a shell sink and -Version/-InstallDir/-DryRun
    for a PowerShell sink (iex runs the piped text with no arguments).
    """
    for line, logical in _logical_lines(text):
        # A table cell shows a pipe as `\|`: split the row into cells, restore the pipe.
        rows = ([c.replace("\\|", "|") for c in re.split(r"(?<!\\)\|", logical)]
                if logical.lstrip().startswith("|") else [logical])
        for row in rows:
            for cmd in _SEQ_RX.split(row):
                stages = cmd.split("|")
                sink = next((i for i, st in enumerate(stages)
                             if i and (_SINK_RX.match(st) or _PS_SINK_RX.match(st))), None)
                if sink is None:
                    continue
                wrong: list[str] = []
                for st in stages[:sink]:
                    m = _ENV_RUN_RX.search(st)
                    if m:
                        wrong += [n for n in re.findall(r"([A-Za-z_]\w*)=", m.group(1))
                                  if n.startswith("MIND_")]
                if _PS_SINK_RX.match(stages[sink]):
                    wrong += [a for st in stages[:sink + 1] for a in _PS_ARG_RX.findall(" " + st)]
                yield line, wrong


def check_installer_env(files: list[Path]) -> list[tuple[Path, int, list[str]]]:
    """Every surface plus the usage comments of scripts/install.*; one verdict per file."""
    scripts = [ROOT / "scripts" / n for n in ("install.sh", "install.ps1")]
    sources = sorted(set(files) | {p for p in scripts if p.is_file()})
    bad: list[tuple[Path, int, list[str]]] = []
    for f in sources:
        seen = 0
        hits: list[tuple[Path, int, list[str]]] = []
        for line, wrong in installer_pipelines(f.read_text(encoding="utf-8", errors="replace")):
            seen += 1
            if wrong:
                hits.append((f, line, wrong))
        if f == ROOT / _INSTALL_PAGE and not seen:
            hits.append((f, 1, []))  # the page that shows the command shows none
        if seen or hits:
            print(f"[{'FAIL' if hits else 'PASS'}] installer env {f.relative_to(ROOT)} "
                  f"({seen} pipe-to-shell command(s))")
        bad.extend(hits)
    return bad


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


# --------------------------------------------------------------------------------
# The SURFACE leg: the number a reader actually sees, compared to the tree.
#
# The manifest leg above compares `declared` (a number in config/) against the
# tree. Nothing compared the number PRINTED IN THE DOCS, so the two could drift
# apart indefinitely while the gate reported PASS — measured: README.md said
# "~1,390 tests across 174 test files", tests/**/*.rs held 325 files and
# src/+tests/ held 2,217 `#[test]`s, and `[PASS] counts[rust_test_files]: floor
# 156+ <= 325` printed on every run. The manifest's own note quoted README text
# ("... across 156 test files") that had not existed for months.
#
# An entry opts in with `surface_regex` (exactly one capture group, matched
# against the whole `surface` file). Semantics:
#   surface_mode = "floor" — the surface is phrased "N+": typed must never
#       EXCEED the derived count, and must not lag it by more than
#       `surface_tolerance`. The lag bound is what stops a floor from going
#       stale, which is the whole defect above.
#   surface_mode = "exact" — |typed - derived| <= `surface_tolerance`.
#
# Fail-CLOSED, deliberately: a declared surface that is missing, a regex that
# matches nothing, and a capture that is not a number are all DRIFT, never a
# skip. A pattern that matches nothing asserts nothing, and that is exactly how
# this check would quietly stop checking after an innocuous re-wording.
#
# deferred: [counts.stdlib_modules] has NO surface leg. Its surface text
# ("13 stdlib modules (vec/string/map/io/...)" in STATUS.md) is a HISTORICAL
# v0.7.0 release marker naming the 13 modules that shipped in that release, not
# a live count of std/ (42 files today) — raising it to the derived figure would
# make the release history wrong, and lowering the derived count is not an
# option. Upgrade path: split the claim into a historical milestone sentence and
# a live "std/ ships N+ modules" sentence, then give the live one a
# surface_regex. Until then check_counts prints an explicit NOTE for every entry
# whose surface number is not parsed, so the gap is visible on every run rather
# than inferred from an absent field.
# --------------------------------------------------------------------------------


def _grouped_int(raw: str) -> int | None:
    """`"1,390"` -> 1390. None when the capture is not a plain number."""
    txt = raw.strip().replace(",", "").replace("\u202f", "").replace(" ", "")
    return int(txt) if txt.isdigit() else None


def _check_surface_number(name: str, spec: dict, derived: int) -> tuple[list[str], list[str]]:
    """Compare every occurrence of the surface's printed number to `derived`."""
    rx_src = spec.get("surface_regex")
    surface = spec.get("surface")
    if not rx_src:
        return [], [
            f"counts[{name}]: NOTE surface number on {surface} is NOT parsed "
            f"(no surface_regex) — manifest-vs-tree comparison only"
        ]
    if not surface:
        return [f"DRIFT [counts/{name}] surface_regex declared without a `surface`"], []
    path = ROOT / surface
    if not path.is_file():
        return [f"DRIFT [counts/{name}] declared surface {surface} does not exist"], []
    try:
        rx = re.compile(rx_src)
    except re.error as err:
        return [f"DRIFT [counts/{name}] surface_regex is invalid: {err}"], []
    if rx.groups != 1:
        return [
            f"DRIFT [counts/{name}] surface_regex must have exactly one capture "
            f"group (the number); it has {rx.groups}"
        ], []
    text = path.read_text(encoding="utf-8", errors="replace")
    found = rx.findall(text)
    if not found:
        return [
            f"DRIFT [counts/{name}] {surface} no longer prints the claim this entry "
            f"checks: surface_regex {rx_src!r} matched nothing. Point it at the live "
            f"wording or delete the entry — a pattern that matches nothing asserts "
            f"nothing."
        ], []

    mode = spec.get("surface_mode", "exact")
    tol = int(spec.get("surface_tolerance", 0))
    drift: list[str] = []
    info: list[str] = []
    for occ in found:
        typed = _grouped_int(occ)
        if typed is None:
            drift.append(
                f"DRIFT [counts/{name}] {surface}: captured {occ!r}, which is not a number"
            )
            continue
        if mode == "floor":
            if typed > derived:
                drift.append(
                    f"DRIFT [counts/{name}] {surface} claims {typed}+ but the tree has "
                    f"{derived} — the surface OVER-CLAIMS"
                )
            elif derived - typed > tol:
                drift.append(
                    f"DRIFT [counts/{name}] {surface} prints {typed}+ but the tree has "
                    f"{derived} — stale by {derived - typed} (bound {tol}); raise the "
                    f"surface to the derived figure"
                )
            else:
                info.append(
                    f"[PASS] counts[{name}]/surface: {surface} prints {typed}+ "
                    f"<= {derived} (lag {derived - typed} <= {tol})"
                )
        elif abs(typed - derived) > tol:
            drift.append(
                f"DRIFT [counts/{name}] {surface} prints {typed} but the tree has "
                f"{derived} (±{tol})"
            )
        else:
            info.append(
                f"[PASS] counts[{name}]/surface: {surface} prints {typed} "
                f"~= {derived} (±{tol})"
            )
    return drift, info


# --------------------------------------------------------------------------------
# FLOORS ARE RATCHETS. A `mode = "floor"` entry says the tree may never fall
# BELOW `declared`; it says nothing about the tree rising above it, so a floor
# set once and never re-raised turns its own slack into deletable corpus while
# printing a pass. Measured here: counts[stdlib_modules] declared 13 against 42
# files in std/, so 29 stdlib modules (~70%) could have been deleted under
# `[PASS] floor 13+ <= 42`; counts[rust_test_files] held 156 against 325.
#
# So a floor also declares `floor_tolerance`: the maximum lag it may carry below
# the tree. Exceed it and the gate fails with the value to raise `declared` to —
# i.e. every time a floor is exceeded by more than its bound, it gets raised.
# Required, not optional: an unbounded floor is DRIFT, because it is exactly the
# defect above wearing a passing verdict. The bound governs only how promptly a
# floor must be re-raised; the protection against deletion is `declared` itself,
# which is why `declared` belongs at the current tree value and not below it.
# --------------------------------------------------------------------------------


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
            lag_bound = spec.get("floor_tolerance")
            if declared > actual:
                drift.append(
                    f"DRIFT [counts/{name}] floor breached: docs claim {declared}+ "
                    f"but tree has {actual} ({spec.get('surface', '?')})"
                )
            elif lag_bound is None:
                drift.append(
                    f"DRIFT [counts/{name}] mode=\"floor\" declares no "
                    f"`floor_tolerance`: an unbounded floor stops ratcheting the "
                    f"moment the tree grows past it. Declare the maximum lag this "
                    f"floor may carry below the tree (currently {actual})."
                )
            elif actual - declared > int(lag_bound):
                drift.append(
                    f"DRIFT [counts/{name}] floor is stale: it declares {declared}+ "
                    f"but the tree has {actual} — a lag of {actual - declared} past "
                    f"the ratchet bound {lag_bound}. At {declared} this floor "
                    f"ratchets nothing: {actual - declared} could be deleted and the "
                    f"gate would still pass. Raise `declared` to {actual}."
                )
            else:
                info.append(
                    f"[PASS] counts[{name}]: floor {declared}+ <= {actual} "
                    f"(lag {actual - declared} <= {lag_bound})"
                )
        else:  # exact
            tol = int(spec.get("tolerance", 0))
            if abs(declared - actual) > tol:
                drift.append(
                    f"DRIFT [counts/{name}] exact mismatch: docs claim {declared} "
                    f"(±{tol}) but tree has {actual} ({spec.get('surface', '?')})"
                )
            else:
                info.append(f"[PASS] counts[{name}]: exact {declared} ~= {actual} (±{tol})")

        # Second leg: the number printed on the surface, not just the one in
        # config/. A manifest floor stays green exactly while the docs go stale.
        s_drift, s_info = _check_surface_number(name, spec, actual)
        drift.extend(s_drift)
        info.extend(s_info)
    return drift, info


# --------------------------------------------------------------------------------
# Published cost claim. The dollar figure on a public surface must be the ARITHMETIC
# RESULT of two committed inputs — the price in config/token_pricing.toml and the
# tokenizer-measured counts in the benchmark's machine-readable output — never a
# hand-typed number.
#
# The defect this exists to stop: README published "$6,780/year per million IR
# operations" while the methodology it cited yielded $396 from the same reference
# IR. The gap back-solved to a $0.030/1K price stated in no file, and nothing
# compared the headline against the benchmark, so it drifted unchallenged.
#
# Fail-closed by construction: every input this check needs is either present and
# checkable, or the check FAILS. The one silent path is a sibling repo with no
# config/token_pricing.toml at all (it publishes no cost claim, so there is
# nothing to verify).
#
# Drift from this check is tagged `[cost-claim]`, deliberately distinct from the
# `[cost_headline]` forbidden-phrase category in capabilities.toml. They catch the
# same historical headline by two independent mechanisms, and an overlapping tag
# let the gate test's "failed for the right reason" assertion pass on the phrase
# hit alone — i.e. stay green with this arithmetic check deleted.
# --------------------------------------------------------------------------------

_PRICING_PATH = ROOT / "config" / "token_pricing.toml"

# Any "$N/year" figure on a declared surface. Every match must equal the derived
# saving; a second, stale figure elsewhere in the file is drift too.
_ANNUAL_FIGURE_RX = re.compile(r"\$([\d,]+(?:\.\d+)?)\s*/\s*year")


def _table_label(line: str) -> str | None:
    """Normalised first cell of a markdown table row, or None if not a row.

    "| **`mic@1`** (canonical text) | 119 | ... |" -> "mic@1". Emphasis, backticks
    and a parenthetical gloss are display sugar; the label underneath is what the
    pricing config maps to a benchmark measurement.
    """
    stripped = line.strip()
    if not stripped.startswith("|"):
        return None
    cells = stripped.split("|")
    if len(cells) < 3:
        return None
    cell = cells[1].strip().strip("*").strip().strip("`").strip()
    return cell.split(" (")[0].strip()


def _number_in(line: str, value: str) -> bool:
    """True if `value` appears as a standalone number, comma-grouped or not."""
    plain = value.replace(",", "")
    grouped = f"{int(plain):,}" if plain.isdigit() else value
    alts = "|".join(re.escape(v) for v in dict.fromkeys((value, plain, grouped)))
    return re.search(rf"(?<![\d.]){alts}(?![\d.])", line) is not None


def _check_cost_tables(text: str, rel: str, pricing: dict, cost: dict) -> list[str]:
    """Published cost TABLE rows must carry the derived numbers, not hand-typed ones.

    Scope, deliberately narrow: a markdown table row whose first cell is a mapped
    format label AND which carries a dollar amount. Token-only rows are left to the
    [counts] gate; ASCII charts are not table rows. Without this the headline
    sentence was derived while the table under it could still drift by hand.
    """
    labels = pricing["claim"].get("table_labels", {})
    if not labels:
        return []
    tokens = cost.get("per_format_tokens", {})
    annual = cost.get("per_format_annual_usd", {})
    drift: list[str] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if "$" not in line:
            continue
        label = _table_label(line)
        meas_label = labels.get(label) if label else None
        if meas_label is None:
            continue
        if meas_label not in tokens or meas_label not in annual:
            drift.append(
                f"DRIFT [cost-claim] {rel}:{lineno}: row {label!r} maps to "
                f"{meas_label!r}, which the benchmark output does not measure"
            )
            continue
        want_tokens = str(tokens[meas_label])
        want_annual = f"{annual[meas_label]:,.0f}"
        if not _number_in(line, want_tokens):
            drift.append(
                f"DRIFT [cost-claim] {rel}:{lineno}: row {label!r} does not carry the "
                f"measured token count {want_tokens} ({pricing['measurement']['tokenizer']})"
            )
        if not _number_in(line, want_annual):
            drift.append(
                f"DRIFT [cost-claim] {rel}:{lineno}: row {label!r} does not carry the "
                f"derived annual cost ${want_annual}"
            )
    return drift


def _derive_cost_claim(pricing: dict) -> tuple[str, float, dict]:
    """Recompute the annual saving and render the canonical claim sentence.

    Raises ValueError with a human-readable reason on any unusable input; the
    caller turns that into drift (never into a skip).
    """
    meas = pricing["measurement"]
    results_path = ROOT / meas["results_json"]
    if not results_path.is_file():
        raise ValueError(
            f"benchmark output {meas['results_json']} is missing — "
            f"re-run: {meas.get('refresh_command', '(no refresh_command declared)')}"
        )
    results = json.loads(results_path.read_text(encoding="utf-8"))
    rows = {r.get("label"): r for r in results.get("measurements", [])}
    field = meas["token_field"]
    counts: dict[str, int] = {}
    for role in ("baseline_label", "candidate_label"):
        label = meas[role]
        row = rows.get(label)
        if row is None:
            raise ValueError(f"benchmark output has no measurement labelled {label!r}")
        value = row.get(field)
        if not isinstance(value, int):
            raise ValueError(
                f"measurement {label!r} has no {field!r} count "
                f"(tokenizer {meas.get('tokenizer')!r} did not run) — a chars/4 "
                f"estimate must never back a price"
            )
        counts[role] = value

    saved_tokens = counts["baseline_label"] - counts["candidate_label"]
    if saved_tokens <= 0:
        raise ValueError(
            f"{meas['candidate_label']} is not cheaper than {meas['baseline_label']} "
            f"({counts['candidate_label']} vs {counts['baseline_label']} tokens)"
        )
    price = float(pricing["pricing"]["input_usd_per_1k_tokens"])
    volume = int(pricing["workload"]["ir_operations_per_year"])
    saving = saved_tokens * volume / 1000.0 * price

    # A cost block written by the benchmark is a convenience, not an authority:
    # it must agree with this independent recomputation.
    declared = results.get("cost_model", {}).get("annual_savings_usd")
    if declared is not None and abs(float(declared) - saving) > 0.005:
        raise ValueError(
            f"benchmark output states annual_savings_usd={declared} but its own "
            f"token counts and config/token_pricing.toml give {saving:.2f}"
        )

    sentence = pricing["claim"]["template"].format(
        price_per_1k=f"{price:g}",
        price_as_of=pricing["pricing"]["as_of"],
        annual_savings=f"{saving:,.0f}",
        volume_human=pricing["workload"]["volume_human"],
    )
    per_format = results.get("cost_model", {})
    return sentence, saving, per_format


def check_cost_claim() -> tuple[list[str], list[str]]:
    """Return (drift_messages, info_messages) for the published cost figure."""
    if not _PRICING_PATH.is_file():
        return [], ["cost: no config/token_pricing.toml — no cost claim to verify"]
    try:
        pricing = tomllib.loads(_PRICING_PATH.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, OSError) as err:
        return [f"DRIFT [cost-claim] config/token_pricing.toml is unreadable: {err}"], []

    try:
        sentence, saving, cost = _derive_cost_claim(pricing)
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as err:
        return [f"DRIFT [cost-claim] cannot derive the published figure: {err}"], []

    drift: list[str] = []
    info = [f"cost: derived ${saving:,.2f}/year — {sentence!r}"]
    expected_figure = f"{saving:,.0f}"
    for rel in pricing["claim"].get("surfaces", []):
        path = ROOT / rel
        if not path.is_file():
            drift.append(f"DRIFT [cost-claim] declared surface {rel} does not exist")
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if sentence not in text:
            drift.append(
                f"DRIFT [cost-claim] {rel}: does not carry the derived claim.\n"
                f"           expected verbatim: {sentence}\n"
                f"           refresh with: {pricing['measurement'].get('refresh_command', '?')}"
            )
        drift.extend(_check_cost_tables(text, rel, pricing, cost))
        for found in _ANNUAL_FIGURE_RX.findall(text):
            if found != expected_figure:
                drift.append(
                    f"DRIFT [cost-claim] {rel}: stale annual figure ${found}/year "
                    f"(the benchmark and config derive ${expected_figure}/year)"
                )
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

    for f, line, names in check_installer_env(files):
        where = f"{f.relative_to(ROOT)}:{line}"
        if not names:
            print(f"DRIFT [installer] {where}: this page shows no installer command "
                  f"('... | sh'); the check has nothing to examine")
        elif names[0].startswith("-"):
            print(f"DRIFT [installer] {where}: {', '.join(names)} goes to irm or iex, which pass "
                  f"no argument to the script. Use '& ([scriptblock]::Create((irm URL))) {names[0]} ...'")
        else:
            print(f"DRIFT [installer] {where}: {', '.join(names)} goes to the download tool, "
                  f"not to the installer. Write it on the shell: '... | {names[0]}=... sh'")
        ok = False

    # The cost claim runs in BOTH modes: it is a claim check, not a tree-shape
    # count, so a sibling repo that publishes the figure is held to it too.
    cost_drift, cost_info = check_cost_claim()
    for line in cost_info:
        print(line)
    for line in cost_drift:
        print(line)
        ok = False

    counts_checked = 0
    if _PHRASES_ONLY:
        print("check_claims: MODE=phrases — forbidden-phrase + canonical-IR only ([counts] skipped)")
    else:
        drift, info = check_counts()
        counts_checked = len(drift) + len(info)
        for line in info:
            print(line)
        for line in drift:
            print(line)
            ok = False

    # The evidence this gate publishes is the per-surface and per-count verdict
    # lines above — scripts/gate_assert.py counts THOSE. A printed total was
    # worse than nothing here: it was computed from `len(files)`, so a surface
    # glob that resolved to a corpus this gate never looked at still reported a
    # full count. This summary is advisory and carries no verdict token.
    print(f"check_claims: surfaces={len(files)} counts={counts_checked}")

    if ok:
        print(f"check_claims: OK — {len(files)} surfaces consistent with {_CAPS_PATH.name}")
        return 0
    print("check_claims: FAILED — docs drifted from the capability manifest (config/capabilities.toml)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
