#!/usr/bin/env python3
"""stdlib_manifest_lint.py — machine-checked contract for WHICH std/*.mind
modules each consumer links.

enforces: STDLIB-MANIFEST

This lint IS the test for the STDLIB-MANIFEST rule: delete the manifest read in
`native_bridge.rs::frozen_std_seed_modules` (or let any consumer's membership drift) and
check 3 / check 6 below go red. The `enforced-by:` sites are the readers.

WHY THIS EXISTS
---------------
"Which std modules go into the native backend's std blob" was asserted by
SEVENTEEN hand-copied literal lists — `STD_MODULES` in
`src/build/native_bridge.rs::run_native_backend_bridge` plus sixteen `_STDLIB_MODULES` /
`STD_MODULES` copies across `examples/mindc_mind/*.py` — with nothing executable
asserting they agreed. The bridge's own comment admitted it:

    "This list is the twin of self_host_standalone_driver_smoke.py::_STDLIB_MODULES"
    "deferred: unify both readers on one committed manifest"

A fourth, DIFFERENT list — `STDLIB_MIND_SOURCES` in `src/project/stdlib.rs`, 23
entries — governs general `use std.<m>` resolution and had drifted from the seed
set by two modules (`http`, `sha512`) with no check and no record of whether that
was deliberate.

Two hand-copied lists that must agree, with nothing asserting they do, is the most
common defect class in this tree. `testdata/stdlib_manifest.txt` is the one
committed contract; this lint recomputes every consumer's REAL membership from its
source and fails on drift, IN BOTH DIRECTIONS, with positive counts.

WHAT IS CHECKED
---------------
  1. manifest <-> std/*.mind on disk, BOTH directions (the manifest is exhaustive,
     like `profile_frozen_admits` — a new std module is a hard failure until
     someone records a deliberate in/out decision for each consumer).
  2. the seed blob composed FROM the manifest hashes to the golden recorded in the
     manifest header — a positive pin on the seed SET and its ORDER, so a reseed
     event cannot land as a quiet one-line edit.
  3. the Rust bridge reads the manifest and carries no literal list of its own.
  4. `native_bridge.rs::FROZEN_STD_SEED_COUNT` == `_stdlib_manifest.FROZEN_STD_SEED_COUNT`
     == the manifest's actual seed count.
  5. `STDLIB_MIND_SOURCES` <-> the manifest's `bundled` column, BOTH directions.
  6. every REMAINING hand-copied `_STDLIB_MODULES` / `STD_MODULES` literal in
     `examples/mindc_mind/*.py` equals the manifest seed list exactly.

Check 6 is the bridge to full migration: those fourteen smokes still carry a
literal, but the literal can no longer drift silently.

  deferred: migrate the remaining literal lists onto `_stdlib_manifest.seed_modules()`
  the way self_host_standalone_driver_smoke.py now does — upgrade path: replace each
  `_STDLIB_MODULES = [...]` with `import _stdlib_manifest` +
  `_STDLIB_MODULES = _stdlib_manifest.seed_modules()`, one smoke per change, each
  verified by re-running that smoke. Not done in one sweep because several of these
  smokes are byte-identity gates that must be individually re-run, and an unrunnable
  bulk edit is exactly the kind of change this lint exists to prevent. Until then
  check 6 holds the invariant.
"""

import ast
import pathlib
import re
import sys

_HERE = pathlib.Path(__file__).parent.resolve()
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_HERE))

import _stdlib_manifest  # noqa: E402

_BRIDGE_RS = _REPO / "src" / "build" / "native_bridge.rs"
_STDLIB_RS = _REPO / "src" / "project" / "stdlib.rs"
_STD_DIR = _REPO / "std"

# Smokes that legitimately consume the manifest rather than a literal.
_MIGRATED = {"_stdlib_manifest.py", "stdlib_manifest_lint.py"}

_LIST_RE = re.compile(r"^_?STD(?:LIB)?_MODULES\s*=\s*(\[[^\]]*\])", re.M)


def _both_ways(name_a, set_a, name_b, set_b, errors):
    """Report BOTH directions of a set mismatch, never just the one that fired."""
    for m in sorted(set_a - set_b):
        errors.append(f"{name_a} has '{m}' but {name_b} does not")
    for m in sorted(set_b - set_a):
        errors.append(f"{name_b} has '{m}' but {name_a} does not")


def main(argv):
    if "--print-seed-sha256" in argv:
        print(_stdlib_manifest.actual_seed_sha256())
        return 0

    errors = []

    manifest_all = _stdlib_manifest.all_modules()
    manifest_seed = _stdlib_manifest.seed_modules()
    manifest_bundled = _stdlib_manifest.bundled_modules()

    if len(set(manifest_all)) != len(manifest_all):
        dupes = sorted({m for m in manifest_all if manifest_all.count(m) > 1})
        errors.append(f"manifest lists these modules more than once: {dupes}")

    # ---- 1. manifest <-> disk, BOTH directions -------------------------------
    disk = sorted(p.stem for p in _STD_DIR.glob("*.mind"))
    if not disk:
        errors.append(f"no std/*.mind found under {_STD_DIR} — refusing a vacuous pass")
    _both_ways("std/ on disk", set(disk), "the manifest", set(manifest_all), errors)

    # ---- 2. seed blob golden (pins the SET and the ORDER) --------------------
    declared = _stdlib_manifest.declared_seed_sha256()
    if declared is None:
        errors.append(
            "manifest header has no `# SEED_BLOB_SHA256 <64 hex>` line — the seed "
            "order is then unpinned and a reseed could land silently"
        )
    else:
        actual = _stdlib_manifest.actual_seed_sha256()
        if actual != declared:
            errors.append(
                f"SEED BLOB CHANGED: manifest declares {declared}, std/ + manifest "
                f"now compose {actual}. If this is intentional it is a WHOLE-CORPUS "
                f"RESEED EVENT (re-mint testdata/selfhost_loop/stage1.elf), not a "
                f"routine edit."
            )

    # ---- 3. the Rust bridge reads the manifest, keeps no literal ------------
    bridge_src = _BRIDGE_RS.read_text(encoding="utf-8")
    if "include_str!(\"../../examples/mindc_mind/testdata/stdlib_manifest.txt\")" not in bridge_src:
        errors.append(
            f"{_BRIDGE_RS.relative_to(_REPO)} no longer include_str!s the manifest — "
            f"the bridge has gone back to a hand-copied list"
        )
    if re.search(r"const\s+STD_MODULES\s*:\s*\[&str", bridge_src):
        errors.append(
            f"{_BRIDGE_RS.relative_to(_REPO)} reintroduced a literal `const STD_MODULES` "
            f"array — the manifest must be the only source of truth"
        )

    # ---- 4. frozen seed count agrees on both sides --------------------------
    m = re.search(r"const\s+FROZEN_STD_SEED_COUNT\s*:\s*usize\s*=\s*(\d+)", bridge_src)
    if not m:
        errors.append(
            f"{_BRIDGE_RS.relative_to(_REPO)} has no FROZEN_STD_SEED_COUNT tripwire"
        )
    else:
        rust_count = int(m.group(1))
        if rust_count != len(manifest_seed):
            errors.append(
                f"native_bridge.rs FROZEN_STD_SEED_COUNT={rust_count} but the manifest declares "
                f"{len(manifest_seed)} seed modules"
            )
        if rust_count != _stdlib_manifest.FROZEN_STD_SEED_COUNT:
            errors.append(
                f"native_bridge.rs FROZEN_STD_SEED_COUNT={rust_count} but "
                f"_stdlib_manifest.FROZEN_STD_SEED_COUNT="
                f"{_stdlib_manifest.FROZEN_STD_SEED_COUNT}"
            )

    # ---- 5. STDLIB_MIND_SOURCES <-> the `bundled` column, BOTH directions ---
    stdlib_src = _STDLIB_RS.read_text(encoding="utf-8")
    bundled_actual = re.findall(r'\(\s*"std\.([a-z0-9_]+)",\s*include_str!', stdlib_src)
    if not bundled_actual:
        errors.append(
            f"could not extract STDLIB_MIND_SOURCES from "
            f"{_STDLIB_RS.relative_to(_REPO)} — the lint would pass vacuously"
        )
    _both_ways(
        "STDLIB_MIND_SOURCES (src/project/stdlib.rs)", set(bundled_actual),
        "the manifest `bundled` column", set(manifest_bundled), errors,
    )

    # ---- 6. every remaining hand-copied literal equals the manifest seed ----
    n_literals = 0
    for py in sorted(_HERE.glob("*.py")):
        if py.name in _MIGRATED:
            continue
        found = _LIST_RE.findall(py.read_text(encoding="utf-8"))
        for raw in found:
            n_literals += 1
            try:
                lst = ast.literal_eval(raw)
            except (ValueError, SyntaxError) as exc:
                errors.append(f"{py.name}: unparseable module list ({exc})")
                continue
            if list(lst) != manifest_seed:
                errors.append(
                    f"{py.name}: hand-copied std list DRIFTED from the manifest "
                    f"(has {len(lst)} entries; first difference: "
                    f"{_first_diff(lst, manifest_seed)})"
                )

    if errors:
        print("stdlib_manifest_lint: FAIL")
        for e in errors:
            print(f"  - {e}")
        print(
            f"\n{len(errors)} problem(s). The contract is "
            f"examples/mindc_mind/testdata/stdlib_manifest.txt; every std/*.mind must "
            f"appear there exactly once with a deliberate in/out decision per consumer."
        )
        return 1

    print(
        f"stdlib_manifest_lint: PASS — {len(manifest_all)} std modules, manifest<->disk "
        f"exact both ways; seed={len(manifest_seed)} (blob sha256 pinned) "
        f"bundled={len(bundled_actual)} unreachable={len(manifest_all) - len(manifest_bundled)}; "
        f"{n_literals} remaining hand-copied list(s) verified against the manifest"
    )
    return 0


def _first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return f"index {i}: {x!r} vs manifest {y!r}"
    return f"length {len(a)} vs manifest {len(b)}"


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
