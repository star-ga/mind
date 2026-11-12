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
  7. every count the manifest's comments state (`SEED_COUNT`, `BUNDLED_COUNT`,
     `TOTAL_COUNT`, and the `<n> modules are not `seed`` sentence) appears
     EXACTLY ONCE, in its canonical form, and equals the real row count. A
     three-way merge keeps both sides' count lines; this check refuses that,
     and refuses its own hostile variants of the real header on every run.

`--self-test` is this lint's own mutation proof (python3 scripts/run_gate.py
examples/mindc_mind/stdlib_manifest_lint.py --self-test). It builds a mini repo from
the lint's REAL file set, damages a copy of the REAL manifest (duplicated header,
wrong count), runs the lint's real entry point on it as a subprocess and requires a
non-zero exit with the count-header message. The in-process probes above call
`count_header_errors` directly, so deleting the one line in `main()` that feeds the
real manifest to it left them green; only a run through the entry point sees that.

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
import shutil
import subprocess
import sys
import tempfile

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
MAIN_MIND_LINE_PIN = 68061  # only down: new code goes in a part; re-pin DOWN with every extraction
_SEED_PRIVATE_PREFIXES = {"arena": ("arena_",), "async": ("async_", "sched_", "snd_"), "blas": ("blas_",), "cli": ("cli_",),
    "fs": ("fs_",), "io": ("io_",), "io_canon": ("canon_",), "iouring": ("io_", "iouring_", "sys_io_uring_"), "map": ("map_",),
    "json": ("jv_", "jvsb_", "jvstr_", "jvpr_"), "net": ("net_",), "process": ("proc_",), "reactor": ("reactor_",),
    "regex": ("rx_",), "ring": ("ring_",), "sha256": ("sha_",), "string": ("string_",), "time": ("time_",),
    "toml": ("toml_", "tv_"), "tui": ("tui_",), "vec": ("vec_",)}


def _both_ways(name_a, set_a, name_b, set_b, errors):
    """Report BOTH directions of a set mismatch, never just the one that fired."""
    for m in sorted(set_a - set_b):
        errors.append(f"{name_a} has '{m}' but {name_b} does not")
    for m in sorted(set_b - set_a):
        errors.append(f"{name_b} has '{m}' but {name_a} does not")


# ---- 7. header counts: one canonical line each, equal to the rows ------------
# Merge 701e90fd kept both sides' count lines (55/55 AND 45/45, and a stale
# "34 modules are not `seed`") and nothing parsed them. The canonical forms:
_HEADER_COUNT_RE = re.compile(
    r"^#\s*(SEED_COUNT|BUNDLED_COUNT|TOTAL_COUNT)\s+(0|[1-9][0-9]*)\s*$"
)
_NON_SEED_RE = re.compile(r"^#\s*(0|[1-9][0-9]*) modules are not `seed`:")
_NON_SEED = "non-seed"
# Any comment line that LOOKS like a count statement — any case, `_`/`-`/space
# separators, a misspelt name, or a leading "<n> modules" — must be one of the
# canonical forms above, so no spelling of a count escapes the check.
_COUNT_LIKE_RE = re.compile(
    r"^#\s*(?:[a-z]+(?:[_-][a-z]+)*[_-]count\b"
    r"|(?:seed|bundled|total|non[\s_-]*seed)\s+count\b"
    r"|[0-9]+\s+modules?\b)",
    re.I,
)


def count_header_errors(text, total, seed, bundled):
    """Every count stated in the manifest's comment lines, judged against the rows.

    The whole file is read, not a header window: a count line a merge left after
    the rows is still a count line. Empty result = each count appears exactly
    once, in canonical form, with the real value.
    """
    want = {
        "SEED_COUNT": seed,
        "BUNDLED_COUNT": bundled,
        "TOTAL_COUNT": total,
        _NON_SEED: total - seed,
    }
    seen = {name: [] for name in want}
    errors = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.startswith("#"):
            continue
        m = _HEADER_COUNT_RE.match(line)
        if m:
            seen[m.group(1)].append((lineno, int(m.group(2))))
            continue
        m = _NON_SEED_RE.match(line)
        if m:
            seen[_NON_SEED].append((lineno, int(m.group(1))))
            continue
        if _COUNT_LIKE_RE.match(line):
            errors.append(
                f"manifest line {lineno} states a count in a form the lint cannot "
                f"check: {line!r} (write `# SEED_COUNT <n>`, `# BUNDLED_COUNT <n>`, "
                f"`# TOTAL_COUNT <n>` or `# <n> modules are not `seed`:`)"
            )
    for name, real in want.items():
        hits = seen[name]
        if not hits:
            errors.append(f"manifest header has no {name} count line (rows give {real})")
        elif len(hits) > 1:
            lines = ", ".join(str(lineno) for lineno, _ in hits)
            errors.append(
                f"manifest states {name} {len(hits)} times (lines {lines}) — exactly "
                f"one count line per count"
            )
        for lineno, n in hits:
            if n != real:
                errors.append(
                    f"manifest line {lineno}: {name} says {n} but the rows give {real}"
                )
    return errors


def count_header_probes(text, total, seed, bundled):
    """Hostile variants of the REAL manifest text; the check must refuse each one.

    Built from the real header so the probes track its format: a probe that no
    longer applies leaves the text unchanged and is then reported as accepted.
    """
    t, s, b, ns = total, seed, bundled, total - seed
    tot, sd, bd = f"# TOTAL_COUNT {t}", f"# SEED_COUNT {s}", f"# BUNDLED_COUNT {b}"
    nsd = f"# {ns} modules are not `seed`:"
    body = text.rstrip("\n")
    return {
        "duplicated, same value": text.replace(tot, f"{tot}\n{tot}", 1),
        "duplicated, merge shape": text.replace(bd, f"{bd}\n# BUNDLED_COUNT {b - 12}", 1),
        "stale value": text.replace(sd, f"# SEED_COUNT {s + 1}", 1),
        "missing line": text.replace(tot + "\n", "", 1),
        "stale non-seed": text.replace(nsd, f"# {ns - 2} modules are not `seed`:", 1),
        "duplicated non-seed": text.replace(nsd, f"{nsd} (copy)\n{nsd}", 1),
        "missing non-seed": text.replace(nsd, "# Some modules are not `seed`:", 1),
        "lower-case spelling": text.replace(tot, f"{tot}\n# total_count {t + 1}", 1),
        "dash spelling": text.replace(tot, f"{tot}\n# TOTAL-COUNT {t + 1}", 1),
        "space spelling": text.replace(tot, f"{tot}\n# Total count {t + 1}", 1),
        "misspelt name": text.replace(tot, f"{tot}\n# BUNDELD_COUNT {b}", 1),
        "trailing junk": text.replace(sd, f"{sd}x", 1),
        "second number": text.replace(sd, f"{sd} {s + 1}", 1),
        "leading zero": text.replace(sd, f"# SEED_COUNT 0{s}", 1),
        "after the rows": f"{body}\n# BUNDLED_COUNT {b - 12}\n",
        "uncheckable module count": text.replace(tot, f"{tot}\n# {t} modules are bundled", 1),
    }


def _run_on_manifest(manifest_text):
    """Run a COPY of this lint, as a subprocess, in a mini repo holding `manifest_text`."""
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        mind = root / "examples" / "mindc_mind"
        (mind / "testdata").mkdir(parents=True)
        for name in ("stdlib_manifest_lint.py", "_stdlib_manifest.py", "main.mind"):  # main.mind: the line ratchet
            shutil.copyfile(_HERE / name, mind / name)
        (mind / "testdata" / "stdlib_manifest.txt").write_text(
            manifest_text, encoding="utf-8")
        for src in (_BRIDGE_RS, _STDLIB_RS):
            dst = root / src.relative_to(_REPO)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
        (root / "std").mkdir()
        for src in sorted(_STD_DIR.glob("*.mind")):
            shutil.copyfile(src, root / "std" / src.name)
        p = subprocess.run(
            [sys.executable, str(mind / "stdlib_manifest_lint.py")],
            capture_output=True, text=True, cwd=root,
        )
        return p.returncode, p.stdout + p.stderr


def selftest():
    """The lint's real entry point must refuse a damaged copy of the real manifest.

    The first case is an UNDAMAGED copy that must pass, so the mini repo is proven
    one the lint can pass in and each red case is red because of its damage; every
    damage is proven to change the text (a replace that matched nothing is an
    error, not a pass).
    """
    real = _stdlib_manifest.MANIFEST_PATH.read_text(encoding="utf-8")

    def header(name):
        hits = re.findall(rf"^# {name} (\d+)$", real, re.M)
        assert len(hits) == 1, f"real manifest must state {name} exactly once: {hits}"
        return int(hits[0]), f"# {name} {hits[0]}"

    def damaged(old, new):
        assert old in real, f"damage anchor not in the real manifest: {old!r}"
        text = real.replace(old, new, 1)
        assert text != real, f"damage did not change the manifest: {old!r}"
        return text

    (seed, seed_line), (bundled, bundled_line), (_, total_line) = (
        header("SEED_COUNT"), header("BUNDLED_COUNT"), header("TOTAL_COUNT"))
    cases = [
        ("control: real manifest copy -> lint passes", real, 0,
         ["stdlib_manifest_lint: PASS"]),
        # THE SURVIVING MUTANT: a stale second header, reached only via the real read.
        ("duplicated BUNDLED_COUNT header (stale value) -> lint fails",
         damaged(bundled_line, f"{bundled_line}\n# BUNDLED_COUNT {bundled - 12}"), 1,
         ["stdlib_manifest_lint: FAIL", "manifest states BUNDLED_COUNT 2 times"]),
        ("duplicated TOTAL_COUNT header (same value) -> lint fails",
         damaged(total_line, f"{total_line}\n{total_line}"), 1,
         ["stdlib_manifest_lint: FAIL", "manifest states TOTAL_COUNT 2 times"]),
        ("wrong SEED_COUNT value -> lint fails",
         damaged(seed_line, f"# SEED_COUNT {seed + 1}"), 1,
         ["stdlib_manifest_lint: FAIL",
          f"SEED_COUNT says {seed + 1} but the rows give {seed}"]),
    ]
    failures = 0
    for label, text, want_rc, want_in_output in cases:
        rc, out = _run_on_manifest(text)
        missing = [s for s in want_in_output if s not in out]
        ok = (rc == 0 if want_rc == 0 else rc != 0) and not missing \
            and "Traceback" not in out
        # ONE VERDICT LINE PER CASE (scripts/gate_assert.py counts reported verdicts).
        print(f"[{'PASS' if ok else 'FAIL'}] {label} (expected "
              f"{'exit 0' if want_rc == 0 else 'non-zero exit'}, got {rc})")
        if not ok:
            failures += 1
            print(f"    missing={missing}")
            for line in out.strip().splitlines()[:12]:
                print(f"    {line}")
    # The SANCTIONED count-marker shape read by scripts/gate_assert.py.
    print(f"\nSDLC-GATE stdlib_manifest_lint_selftest ran={len(cases)} fail={failures}")
    return 1 if failures else 0


def main(argv):
    if "--self-test" in argv or "--selftest" in argv:
        return selftest()
    if "--print-seed-sha256" in argv:
        print(_stdlib_manifest.actual_seed_sha256())
        return 0

    errors = []
    # ---- 7. main.mind line ratchet (only down); `--main-mind-pin` runs it alone ----
    main_lines = (_HERE / "main.mind").read_bytes().count(b"\n")
    if main_lines > MAIN_MIND_LINE_PIN:
        errors.append(f"main.mind has {main_lines} lines > MAIN_MIND_LINE_PIN {MAIN_MIND_LINE_PIN}: "
                      "put new compiler code in a COMPILER_COMPONENTS part")
    if "--main-mind-pin" in argv:
        print(errors[0] if errors else f"main.mind pin PASS: {main_lines} <= MAIN_MIND_LINE_PIN {MAIN_MIND_LINE_PIN}")
        return 1 if errors else 0

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

    # ---- 8. seed private fns carry a _SEED_PRIVATE_PREFIXES prefix (the flat image reserves every seed name)
    seen = {}  # private name -> the seed module that defines it
    for mod in manifest_seed:
        prefixes = _SEED_PRIVATE_PREFIXES.get(mod, ())
        for name in re.findall(r"^[ \t]*fn (\w+)", (_STD_DIR / f"{mod}.mind").read_text(encoding="utf-8"), re.M):
            if not name.startswith(prefixes):
                errors.append(f"std/{mod}.mind private fn `{name}` lacks a module prefix {prefixes}")
            if seen.setdefault(name, mod) != mod: errors.append(f"private fn `{name}` in std/{seen[name]}.mind and std/{mod}.mind")
    if not seen: errors.append("no private seed fn found: the prefix check would pass vacuously")

    # ---- 7. header counts: exactly one canonical line each, equal to rows ---
    counts = (len(manifest_all), len(manifest_seed), len(manifest_bundled))
    manifest_text = _stdlib_manifest.MANIFEST_PATH.read_text(encoding="utf-8")
    errors.extend(count_header_errors(manifest_text, *counts))
    probes = count_header_probes(manifest_text, *counts)
    for name, variant in probes.items():
        if not count_header_errors(variant, *counts):
            errors.append(
                f"header-count check accepted its hostile '{name}' variant of the "
                f"manifest — a guard in count_header_errors has been lost"
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
        f"{n_literals} remaining hand-copied list(s) verified against the manifest; "
        f"{len(seen)} seed private fn(s) carry their module prefix, no name in two modules; "
        f"header counts SEED/BUNDLED/TOTAL/non-seed stated once each and exact, "
        f"{len(probes)} hostile header variants refused"
    )
    return 0


def _first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return f"index {i}: {x!r} vs manifest {y!r}"
    return f"length {len(a)} vs manifest {len(b)}"


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
