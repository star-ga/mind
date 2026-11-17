#!/usr/bin/env python3
"""Gate tests for scripts/check_claims.py — its two DERIVED numbers must bite.

check_claims.py compares public surfaces against the tree. Two of its legs
compute a number rather than matching a string, and a computed comparison is
the kind that can stop comparing and still exit 0. This file is the mutation
proof for both: each case breaks one input and requires the gate to go red.

All four suites live in ONE file on purpose. The product surface of this repository
is MIND and Rust; every tracked .py/.sh is harness, and harness that grows by a
file per gate is the surface nobody decided to grow. A gate's proof belongs
beside the gate's other proof, and scripts/check_gate_wiring.py holds the whole
harness surface to a ceiling that this shape is what keeps affordable.

SUITE 1 — THE COST CLAIM MUST BE DERIVED, NOT TYPED
---------------------------------------------------
README.md published "MIC saves $6,780/year per million IR operations vs JSON"
while the methodology it cited (benchmarks/BENCHMARK_RESULTS.md) yielded $396
from the same reference IR: a 17.1x gap that back-solved to a $0.030/1K token
price stated in NO file in the tree. Nothing in CI compared the two, so the
headline could drift from its own benchmark indefinitely.

The fix is arithmetic, not prose: check_claims.py recomputes the annual saving
from (a) the committed price input in config/token_pricing.toml and (b) the
tokenizer-measured token counts in the benchmark's machine-readable output, then
requires the resulting sentence to appear verbatim on every declared surface.
Suite 1 mutates each input in turn and requires the gate to go red.

SUITE 2 — A COUNT PRINTED ON A SURFACE MUST EQUAL THE DERIVED COUNT
-------------------------------------------------------------------
`[counts]` in config/capabilities.toml compared the MANIFEST's `declared` number
against the tree, and nothing ever read the number a reader actually sees.
README.md typed "~1,390 tests across 174 test files" while the tree held 325
files under tests/ and 2,217 `#[test]` functions; the manifest declared a FLOOR
of 156, so `156 <= 325` reported `[PASS]`, the gate exited 0, and the surface
number was free to drift by 151 files unchallenged. The manifest's own note
quoted README text ("... across 156 test files") that no longer existed — a
floor comparison stays green precisely while the surface and the tree drift
apart.

The fix is a second leg in check_claims.py: an entry may declare
`surface_regex`, and the number that regex captures ON THE SURFACE is compared
to the DERIVED count. Suite 2 mutates the surface number, the surface sentence
and the surface file in turn and requires the gate to go red for each, and it
asserts the LIVE manifest actually wires that leg to the README claims (a new
check kind nothing uses is not a gate).

SUITE 3 — A FLOOR MUST KEEP RATCHETING
---------------------------------------
`mode = "floor"` bounded the tree from below and nothing from above, so a floor
set once drifted arbitrarily far under the tree while printing `[PASS]`:
counts[stdlib_modules] declared 13 against 42 files in std/, leaving ~70% of the
standard library deletable with this gate green. Suite 3 requires a floor to
declare the maximum lag it may carry, reddens the gate when the lag is exceeded,
and reddens it for a floor entry that declares no bound at all.

SUITE 4 — A DOCUMENTED INSTALLER PIPELINE MUST HAND ITS VARIABLE TO THE INSTALLER
---------------------------------------------------------------------------------
docs/install.md showed `MIND_VERSION=v0.10.3 curl -sSL <url> | sh`. In
`VAR=x A | B` the variable reaches A only, so the installer never saw the pin and
installed the latest release, while every other check on that page stayed green.
check_claims.py now refuses a `MIND_*` assignment on any stage before a shell in
a pipeline, on every surface and in the usage comments of scripts/install.*.
Suite 4 runs the wrong form in every shape a page can show it, and the right
form next to it. A green case must also prove the check examined a command: a
pass over zero commands is not a pass.

Run:  python3 tests/check_claims_gate_tests.py
Exit: 0 = all cases pass, 1 = a case failed (prints the offending case).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CHECK = REPO / "scripts" / "check_claims.py"
PRICING = REPO / "config" / "token_pricing.toml"
CAPS = REPO / "config" / "capabilities.toml"

# A minimal surface tree: check_claims also runs the forbidden-phrase and
# canonical-IR checks, so the synthetic README must satisfy those to isolate
# the cost assertion. MODE=phrases skips the [counts] gate, which derives from
# a real source tree this fixture deliberately does not have.
COST_README_TEMPLATE = """# Fixture surface

Canonical IR: mic@1 text, mic@3 binary.

| Format | Tokens | Annual (1M IRs) |
|--------|--------|-----------------|
{table}

**{claim}.**
"""


def cost_table(pricing: dict, results: dict) -> str:
    """The published cost TABLE, rendered from the benchmark output.

    The headline sentence is only half the surface: the table under it carries
    the same numbers and used to be hand-typed too.
    """
    cost = results["cost_model"]
    lines = []
    for display, label in pricing["claim"].get("table_labels", {}).items():
        lines.append(
            f"| {display} | {cost['per_format_tokens'][label]} "
            f"| ${cost['per_format_annual_usd'][label]:,.0f} |"
        )
    return "\n".join(lines)


def cost_run_gate(root: Path) -> tuple[int, str]:
    env = dict(os.environ)
    env["CHECK_CLAIMS_ROOT"] = str(root)
    env["CHECK_CLAIMS_CAPS"] = str(REPO / "config" / "capabilities.toml")
    env["CHECK_CLAIMS_SURFACES"] = "README.md"
    env["CHECK_CLAIMS_MODE"] = "phrases"
    proc = subprocess.run(
        [sys.executable, str(CHECK)],
        env=env, capture_output=True, text=True, check=False,
    )
    return proc.returncode, proc.stdout + proc.stderr


def build_fixture(tmp: Path, *, claim: str, pricing_text: str, results: dict,
                  table: str | None = None) -> None:
    (tmp / "config").mkdir(parents=True, exist_ok=True)
    (tmp / "benchmarks").mkdir(parents=True, exist_ok=True)
    (tmp / "config" / "token_pricing.toml").write_text(pricing_text, encoding="utf-8")
    (tmp / "benchmarks" / "mic_map_benchmark_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    pricing = tomllib.loads(pricing_text)
    if table is None:
        table = cost_table(pricing, results)
    body = COST_README_TEMPLATE.format(claim=claim, table=table)
    # Every surface the config declares must carry the claim, so the fixture
    # mirrors the real repo's surface list rather than assuming README alone.
    surfaces = pricing["claim"].get("surfaces", ["README.md"])
    for rel in surfaces:
        path = tmp / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    (tmp / "README.md").write_text(body, encoding="utf-8")


def expected_claim(pricing: dict, results: dict) -> str:
    """Independent re-derivation of the sentence, written from the config spec.

    Deliberately NOT imported from check_claims.py: a test that reuses the
    implementation it checks proves only self-consistency.
    """
    m = pricing["measurement"]
    field = m["token_field"]
    rows = {r["label"]: r for r in results["measurements"]}
    base = rows[m["baseline_label"]][field]
    cand = rows[m["candidate_label"]][field]
    price = pricing["pricing"]["input_usd_per_1k_tokens"]
    volume = pricing["workload"]["ir_operations_per_year"]
    saving = (base - cand) * volume / 1000.0 * price
    return pricing["claim"]["template"].format(
        price_per_1k=f"{price:g}",
        price_as_of=pricing["pricing"]["as_of"],
        annual_savings=f"{saving:,.0f}",
        volume_human=pricing["workload"]["volume_human"],
    )


def cost_main() -> int:
    if not PRICING.is_file():
        print(f"FAIL setup: {PRICING} does not exist — the price input must be committed")
        return 1
    pricing_text = PRICING.read_text(encoding="utf-8")
    pricing = tomllib.loads(pricing_text)
    results_path = REPO / pricing["measurement"]["results_json"]
    if not results_path.is_file():
        print(f"FAIL setup: {results_path} does not exist — benchmark output must be committed")
        return 1
    results = json.loads(results_path.read_text(encoding="utf-8"))
    good = expected_claim(pricing, results)

    failures: list[str] = []

    def case(name: str, *, claim: str, results_mut: dict, pricing_mut: str, want_ok: bool,
             table: str | None = None) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            build_fixture(tmp, claim=claim, pricing_text=pricing_mut,
                          results=results_mut, table=table)
            rc, out = cost_run_gate(tmp)
            ok = rc == 0
            if ok != want_ok:
                failures.append(
                    f"{name}: expected {'PASS' if want_ok else 'FAIL'}, got rc={rc}\n{out}"
                )
            elif not want_ok and "DRIFT [cost-claim]" not in out:
                # The tag must be the ARITHMETIC check's, not the forbidden-phrase
                # category's. capabilities.toml also lists the historical $6,780
                # headline, so an overlapping tag would let this assertion pass on
                # the phrase hit alone and the derived-figure check could be deleted
                # without a single case going red.
                failures.append(
                    f"{name}: failed for the wrong reason (no DRIFT [cost-claim])\n{out}"
                )
            # The shared verdict vocabulary (scripts/gate_assert.py counts a
            # PASS/FAIL line as one reported assertion). Printed `ok `/`BAD`
            # before, which counts as nothing: this file ran nine mutation
            # cases and published `asserted=0`, so an emptied case list would
            # still have exited 0 under the gate runner.
            print(f"  [{'PASS' if ok == want_ok else 'FAIL'}] {name} (rc={rc})")

    # 1. Agreement: the derived sentence on the surface passes.
    case("agreement", claim=good, results_mut=results, pricing_mut=pricing_text, want_ok=True)

    # 2. The historical defect: a hand-typed headline the benchmark never produced.
    case(
        "stale_headline_6780",
        claim="MIC saves $6,780/year per million IR operations vs JSON",
        results_mut=results, pricing_mut=pricing_text, want_ok=False,
    )

    # 3. Benchmark output moves, README does not.
    moved = json.loads(json.dumps(results))
    for row in moved["measurements"]:
        if row["label"] == pricing["measurement"]["candidate_label"]:
            row[pricing["measurement"]["token_field"]] += 40
    case("benchmark_moved", claim=good, results_mut=moved, pricing_mut=pricing_text, want_ok=False)

    # 4. Price input moves, README does not.
    repriced = pricing_text.replace(
        f"input_usd_per_1k_tokens = {pricing['pricing']['input_usd_per_1k_tokens']}",
        "input_usd_per_1k_tokens = 0.03",
    )
    if repriced == pricing_text:
        failures.append("repricing mutation did not apply — check the config key spelling")
    case("price_moved", claim=good, results_mut=results, pricing_mut=repriced, want_ok=False)

    # 5. Fail closed when the tokenizer view is missing: a chars/4 estimate must
    #    never silently back a dollar figure.
    untokenized = json.loads(json.dumps(results))
    for row in untokenized["measurements"]:
        row[pricing["measurement"]["token_field"]] = None
    case("tokenizer_missing", claim=good, results_mut=untokenized, pricing_mut=pricing_text, want_ok=False)

    # 6. A hand-edited cost block in the benchmark output cannot override the
    #    gate's own arithmetic.
    forged = json.loads(json.dumps(results))
    forged.setdefault("cost_model", {})["annual_savings_usd"] = 6780.0
    case("forged_cost_block", claim=good, results_mut=forged, pricing_mut=pricing_text, want_ok=False)

    # 7. The table under the headline is derived too: a row hand-edited back to the
    #    old chars/4 estimate must go red even though the sentence still agrees.
    stale_table = "\n".join(
        # 67 tokens / $117 are the pre-fix chars/4 numbers for this row.
        "| TOON | 67 | $117 |" if line.startswith("| TOON ") else line
        for line in cost_table(pricing, results).splitlines()
    )
    case("stale_table_row", claim=good, results_mut=results, pricing_mut=pricing_text,
         want_ok=False, table=stale_table)

    # 8. The ONE silent path, asserted so it stays narrow: a sibling repo with no
    #    config/token_pricing.toml publishes no cost claim, so there is nothing to
    #    verify and the gate must not invent a failure. Every other missing input
    #    above fails closed.
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        build_fixture(tmp, claim=good, pricing_text=pricing_text, results=results)
        (tmp / "config" / "token_pricing.toml").unlink()
        rc, out = cost_run_gate(tmp)
        if rc != 0 or "no cost claim to verify" not in out:
            failures.append(f"sibling_without_config: expected a clean skip, got rc={rc}\n{out}")
        print(f"  [{'PASS' if rc == 0 else 'FAIL'}] sibling_without_config (rc={rc})")

    # 9. The real repo tree must satisfy the gate end to end.
    rc = subprocess.run([sys.executable, str(CHECK)], cwd=REPO,
                        capture_output=True, text=True, check=False)
    if rc.returncode != 0:
        failures.append(f"repo_tree: check_claims.py failed on the real tree\n{rc.stdout}{rc.stderr}")
    print(f"  [{'PASS' if rc.returncode == 0 else 'FAIL'}] repo_tree (rc={rc.returncode})")

    if failures:
        print("\ncheck_claims cost gate: FAILED")
        for f in failures:
            print("-" * 70)
            print(f)
        return 1
    print("\ncheck_claims cost gate: OK — the published figure is derived, not typed")
    return 0

# ==========================================================================
# SUITE 2 — a count PRINTED on a public surface must equal the derived count
# ==========================================================================

FIXTURE_FILES = 12

MANIFEST_TEMPLATE = """
[ir]
canonical_text   = "mic@1"
canonical_binary = "mic@3"

[counts.fixture_files]
declared          = {declared}
mode              = "floor"
{floor_line}kind              = "glob_count"
globs             = ["tests/**/*.rs"]
surface           = "README.md"
surface_regex     = '([\\d,]+)\\+ test files'
surface_mode      = "{surface_mode}"
surface_tolerance = {surface_tolerance}
note              = "fixture"
"""

SURFACE_README_TEMPLATE = """# Fixture surface

Canonical IR: mic@1 text, mic@3 binary.

{sentence}
"""


def write_tree(root: Path, sentence: str, n_files: int = FIXTURE_FILES) -> None:
    (root / "tests").mkdir(parents=True, exist_ok=True)
    for i in range(n_files):
        (root / "tests" / f"t{i:03d}.rs").write_text(
            "#[test]\nfn t() {}\n", encoding="utf-8"
        )
    (root / "README.md").write_text(
        SURFACE_README_TEMPLATE.format(sentence=sentence), encoding="utf-8"
    )


def surface_run_gate(root: Path, caps: Path, surfaces: str = "README.md") -> tuple[int, str]:
    env = dict(os.environ)
    env["CHECK_CLAIMS_ROOT"] = str(root)
    env["CHECK_CLAIMS_CAPS"] = str(caps)
    env["CHECK_CLAIMS_SURFACES"] = surfaces
    proc = subprocess.run(
        [sys.executable, str(CHECK)],
        env=env, capture_output=True, text=True, check=False,
    )
    return proc.returncode, proc.stdout + proc.stderr


def surface_case(
    label: str,
    *,
    sentence: str,
    want_red: bool,
    surface_mode: str = "floor",
    surface_tolerance: int = 0,
    n_files: int = FIXTURE_FILES,
    drop_surface: bool = False,
) -> bool:
    """One fixture run. Returns True when the gate behaved as required."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        write_tree(root, sentence, n_files)
        surfaces = "README.md"
        if drop_surface:
            (root / "README.md").unlink()
            # The forbidden-phrase leg needs SOME surface, or the gate exits 2
            # ("no surfaces found") for an unrelated reason and proves nothing.
            (root / "OTHER.md").write_text(
                "Canonical IR: mic@1 text, mic@3 binary.\n", encoding="utf-8"
            )
            surfaces = "OTHER.md"
        caps = root / "caps.toml"
        caps.write_text(
            MANIFEST_TEMPLATE.format(
                declared=3, floor_line="floor_tolerance   = 99\n",
                surface_mode=surface_mode, surface_tolerance=surface_tolerance
            ),
            encoding="utf-8",
        )
        rc, out = surface_run_gate(root, caps, surfaces)

    red = rc != 0
    right_reason = "DRIFT [counts/fixture_files]" in out
    ok = red == want_red and (right_reason if want_red else True)
    print(f"[{'PASS' if ok else 'FAIL'}] {label}: rc={rc} want_red={want_red}")
    if not ok:
        print(out)
    return ok


def live_wiring() -> bool:
    """The LIVE manifest must wire the surface leg to the README test counts.

    Without this, every case above could pass against a fixture while the real
    README claim stayed unparsed — the exact hole the fixture cases describe.
    """
    caps = tomllib.loads(CAPS.read_text(encoding="utf-8"))
    counts = caps.get("counts", {})
    ok = True
    for name in ("rust_test_files", "rust_test_fns"):
        spec = counts.get(name, {})
        wired = bool(spec.get("surface_regex"))
        print(f"[{'PASS' if wired else 'FAIL'}] live manifest: counts[{name}] "
              f"parses its surface number")
        ok &= wired
        if not wired:
            continue
        # The regex must actually match the declared surface TODAY: a pattern
        # that matches nothing is a check that asserts nothing.
        surface = REPO / spec["surface"]
        hits = re.findall(spec["surface_regex"], surface.read_text(encoding="utf-8"))
        print(f"[{'PASS' if hits else 'FAIL'}] live manifest: counts[{name}] "
              f"regex matches {spec['surface']} (hits={len(hits)})")
        ok &= bool(hits)
    return bool(ok)


def surface_main() -> int:
    n = FIXTURE_FILES
    results = [
        # Agreement: the surface prints exactly what the tree holds.
        surface_case("floor: surface number equals derived",
             sentence=f"A suite across {n}+ test files.", want_red=False),
        # The defect: the surface under-claims far beyond the staleness bound.
        surface_case("floor: stale surface number is DRIFT",
             sentence=f"A suite across {n - 5}+ test files.", want_red=True),
        # A surface may lag by at most the declared bound.
        surface_case("floor: lag within tolerance passes",
             sentence=f"A suite across {n - 2}+ test files.",
             surface_tolerance=3, want_red=False),
        surface_case("floor: lag beyond tolerance is DRIFT",
             sentence=f"A suite across {n - 5}+ test files.",
             surface_tolerance=3, want_red=True),
        # Over-claiming is drift in floor mode whatever the tolerance.
        surface_case("floor: over-claim is DRIFT",
             sentence=f"A suite across {n + 1}+ test files.",
             surface_tolerance=99, want_red=True),
        # exact mode is two-sided.
        surface_case("exact: equal passes",
             sentence=f"A suite across {n}+ test files.",
             surface_mode="exact", want_red=False),
        surface_case("exact: off by one is DRIFT",
             sentence=f"A suite across {n + 1}+ test files.",
             surface_mode="exact", want_red=True),
        # Vacuity: a surface that no longer prints the claim must fail CLOSED,
        # never silently stop being checked.
        surface_case("sentence removed from surface is DRIFT",
             sentence="A suite of tests.", want_red=True),
        surface_case("declared surface missing is DRIFT",
             sentence=f"A suite across {n}+ test files.",
             drop_surface=True, want_red=True),
        # Every printed occurrence is checked, not just the first.
        surface_case("second stale occurrence is DRIFT",
             sentence=(f"A suite across {n}+ test files.\n\n"
                       f"Elsewhere: {n - 5}+ test files."),
             want_red=True),
        live_wiring(),
    ]
    failed = results.count(False)
    # Deliberately NOT a `ran=<n> fail=<k>` marker: scripts/gate_assert.py reads
    # that shape only from the sdlc/* family, and examples/mindc_mind/
    # smoke_wiring_lint.py refuses a hand-made one precisely because a typed
    # total is a number a gate claims rather than one it earned. Every case
    # above prints its own [PASS]/[FAIL] verdict line, which is the evidence
    # the shim counts; this summary is advisory.
    print(f"check_claims surface-count gate: cases={len(results)} failed={failed}")
    if failed:
        print("FAILED — the surface-count comparison is not load-bearing")
        return 1
    print("OK — every mutation of a printed count reddens the gate")
    return 0


# ==========================================================================
# SUITE 3 — a floor that lags the tree has stopped being a ratchet
# ==========================================================================
#
# `mode = "floor"` only forbids the tree from falling BELOW `declared`. Nothing
# bounded how far the tree could rise above it, so a floor set once went stale
# silently and the slack it accumulated became deletable corpus: measured on
# this repo, counts[stdlib_modules] declared 13 against 42 files in std/, so
# 29 stdlib modules (~70% of std/) could be deleted with the gate printing
# `[PASS] floor 13+ <= 42` the whole way down. The same shape held the test
# corpus at floor 156 against 325 files.
#
# The fix is a REQUIRED lag bound: a floor entry declares `floor_tolerance`,
# the maximum distance it may sit below the tree, and exceeding it is DRIFT
# that names the value to raise `declared` to. Fail-closed: a floor entry with
# NO bound is itself DRIFT, because an unbounded floor is the defect above
# wearing a passing verdict.


def floor_case(
    label: str,
    *,
    declared: int,
    want_red: bool,
    floor_tolerance: int | None = 0,
    expect: str = "",
    n_files: int = FIXTURE_FILES,
) -> bool:
    """One manifest-floor fixture run. True when the gate behaved as required."""
    floor_line = "" if floor_tolerance is None else f"floor_tolerance   = {floor_tolerance}\n"
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        # The surface prints the derived count exactly, so the SURFACE leg is
        # silent and every red below is attributable to the manifest leg.
        write_tree(root, f"A suite across {n_files}+ test files.", n_files)
        caps = root / "caps.toml"
        caps.write_text(
            MANIFEST_TEMPLATE.format(
                declared=declared, floor_line=floor_line,
                surface_mode="floor", surface_tolerance=99,
            ),
            encoding="utf-8",
        )
        rc, out = surface_run_gate(root, caps)

    red = rc != 0
    ok = red == want_red and (expect in out if want_red else True)
    print(f"[{'PASS' if ok else 'FAIL'}] {label}: rc={rc} want_red={want_red}")
    if not ok:
        print(out)
    return ok


def live_floors() -> bool:
    """Every floor in the LIVE manifest must carry its ratchet bound.

    Without this the fixture cases could all pass while the real manifest kept
    an unbounded floor — the exact entry that let ~70% of std/ become
    deletable. The lag itself is checked by running the gate on the real tree
    (SUITE 1, case `repo_tree`); this asserts the bound EXISTS to be checked.
    """
    caps = tomllib.loads(CAPS.read_text(encoding="utf-8"))
    ok = True
    floors = [n for n, spec in caps.get("counts", {}).items()
              if spec.get("mode") == "floor"]
    for name in floors:
        spec = caps["counts"][name]
        bounded = "floor_tolerance" in spec
        print(f"[{'PASS' if bounded else 'FAIL'}] live manifest: counts[{name}] "
              f"declares its ratchet bound")
        ok &= bounded
    print(f"[{'PASS' if floors else 'FAIL'}] live manifest: floor entries found "
          f"({len(floors)})")
    return bool(ok and floors)


def floor_main() -> int:
    n = FIXTURE_FILES
    results = [
        # The ratchet still bites downward: the tree may not fall below.
        floor_case("floor breached by the tree is DRIFT",
                   declared=n + 1, want_red=True, floor_tolerance=0,
                   expect="floor breached"),
        # Exactly current, no slack: the healthy state.
        floor_case("floor equal to the tree passes",
                   declared=n, want_red=False, floor_tolerance=0),
        # The defect: a floor far below the tree used to print [PASS].
        floor_case("floor stale beyond its bound is DRIFT",
                   declared=3, want_red=True, floor_tolerance=0,
                   expect="ratchets nothing"),
        # A declared amount of slack is allowed, and no more.
        floor_case("floor lag within its bound passes",
                   declared=n - 2, want_red=False, floor_tolerance=3),
        floor_case("floor lag beyond its bound is DRIFT",
                   declared=n - 5, want_red=True, floor_tolerance=3,
                   expect="ratchets nothing"),
        # Fail-closed: an unbounded floor asserts nothing about growth.
        floor_case("floor without a bound is DRIFT",
                   declared=n, want_red=True, floor_tolerance=None,
                   expect="floor_tolerance"),
        live_floors(),
    ]
    failed = results.count(False)
    print(f"check_claims floor-ratchet gate: cases={len(results)} failed={failed}")
    if failed:
        print("FAILED — a stale floor is still reported as a pass")
        return 1
    print("OK — a floor that stops ratcheting reddens the gate")
    return 0


# ==========================================================================
# SUITE 4 — a documented installer pipeline hands its variable to the installer
# ==========================================================================

INSTALL_URL = "https://example.invalid/install.sh"
INSTALL_CAPS = '[ir]\ncanonical_text   = "mic@1"\ncanonical_binary = "mic@3"\n'
INSTALL_README = "Canonical IR: mic@1 text, mic@3 binary.\n"


def doc(*lines: str) -> str:
    """A page that shows each line as a command in one fenced block."""
    return "# Install\n\n```sh\n" + "\n".join(lines) + "\n```\n"


def install_case(
    label: str,
    files: dict[str, str],
    *,
    red_in: str | None = None,
    vacuous: bool = False,
    examined: tuple[str, int] | None = None,
) -> bool:
    """One fixture tree. A red case names the file the DRIFT line must point at,
    and the line must be the wrong-side finding: the "page shows no command" line
    would otherwise hide a missed command on a one-command page (`vacuous` marks
    the case that wants that line). A green case that names `examined` must show
    the check counted that many pipe-to-shell commands on that file."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        files = {"README.md": "", **files}
        for rel, text in files.items():
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text + (INSTALL_README if rel == "README.md" else ""),
                            encoding="utf-8")
        caps = root / "caps.toml"
        caps.write_text(INSTALL_CAPS, encoding="utf-8")
        env = dict(os.environ, CHECK_CLAIMS_ROOT=str(root), CHECK_CLAIMS_CAPS=str(caps),
                   CHECK_CLAIMS_SURFACES="README.md:docs/**/*.md", CHECK_CLAIMS_MODE="phrases")
        proc = subprocess.run([sys.executable, str(CHECK)], env=env,
                              capture_output=True, text=True, check=False)
    out = proc.stdout + proc.stderr
    if red_in is not None:
        drift = [ln for ln in out.splitlines() if ln.startswith(f"DRIFT [installer] {red_in}:")]
        empty = [ln for ln in drift if "shows no installer command" in ln]
        ok = proc.returncode != 0 and bool(empty if vacuous else set(drift) - set(empty))
    else:
        ok = proc.returncode == 0 and (
            examined is None
            or f"installer env {examined[0]} ({examined[1]} pipe-to-shell" in out)
    print(f"[{'PASS' if ok else 'FAIL'}] {label}: rc={proc.returncode} red_in={red_in}")
    if not ok:
        print(out)
    return ok


def live_install() -> bool:
    """The LIVE page must show the pin on the installer, and the gate must count it.

    Fixture cases alone would stay green if the real page lost its pin example
    or the surface glob stopped reaching it.
    """
    text = (REPO / "docs" / "install.md").read_text(encoding="utf-8")
    pinned = re.search(r"\|\s*MIND_VERSION=v\d+\.\d+\.\d+\s+sh\b", text) is not None
    proc = subprocess.run([sys.executable, str(CHECK)], cwd=REPO,
                          capture_output=True, text=True, check=False)
    counted = re.search(r"\[PASS\] installer env docs/install\.md \([1-9]\d* pipe-to-shell",
                        proc.stdout) is not None
    for what, ok in (("docs/install.md pins on the installer", pinned),
                     ("the gate counts its installer commands", counted),
                     ("the gate is green on the real tree", proc.returncode == 0)):
        print(f"[{'PASS' if ok else 'FAIL'}] live: {what}")
    return pinned and counted and proc.returncode == 0


def install_main() -> int:
    good = f"curl -sSL {INSTALL_URL} | MIND_VERSION=v0.10.3 sh"
    bad = f"MIND_VERSION=v0.10.3 curl -sSL {INSTALL_URL} | sh"
    page = "docs/install.md"
    results = [
        # The healthy page, and the controls that must not turn red.
        install_case("pin on sh passes", {page: doc(good)}, examined=(page, 1)),
        install_case("dry run on sh passes",
                     {page: doc(f"curl -sSL {INSTALL_URL} | MIND_DRY_RUN=1 sh")},
                     examined=(page, 1)),
        install_case("export before curl reaches sh and passes",
                     {page: doc(f"export MIND_VERSION=v0.10.3; curl -sSL {INSTALL_URL} | sh")},
                     examined=(page, 1)),
        install_case("a variable for curl itself passes",
                     {page: doc(f"HTTPS_PROXY=http://p:3128 curl -sSL {INSTALL_URL} | sh")},
                     examined=(page, 1)),
        install_case("a sibling repo without the page passes", {}),
        # The defect, in every shape a page can show it.
        install_case("pin before curl is DRIFT", {page: doc(bad)}, red_in=page),
        install_case("dry run before curl is DRIFT",
                     {page: doc(f"MIND_DRY_RUN=1 curl -sSL {INSTALL_URL} | sh")}, red_in=page),
        install_case("any MIND_ variable is DRIFT, not only the pin",
                     {page: doc(f"MIND_INSTALL_DIR=/opt/m curl -sSL {INSTALL_URL} | sh")},
                     red_in=page),
        install_case("wget is DRIFT",
                     {page: doc(f"MIND_VERSION=v0.10.3 wget -qO- {INSTALL_URL} | sh")},
                     red_in=page),
        install_case("env prefix is DRIFT",
                     {page: doc(f"env MIND_VERSION=v0.10.3 curl -sSL {INSTALL_URL} | sh")},
                     red_in=page),
        install_case("bash and sudo sinks are DRIFT",
                     {page: doc(f"MIND_VERSION=v0.10.3 curl -sSL {INSTALL_URL} | sudo -E bash -s --")},
                     red_in=page),
        install_case("a non-curl first stage is DRIFT",
                     {page: doc("MIND_VERSION=v0.10.3 cat scripts/install.sh | sh")},
                     red_in=page),
        install_case("shell prompt prefix is DRIFT", {page: doc("$ " + bad)}, red_in=page),
        install_case("inline code in prose is DRIFT",
                     {page: f"# Install\n\nRun `{bad}` to pin.\n"}, red_in=page),
        install_case("backslash continuation is DRIFT",
                     {page: doc(f"MIND_VERSION=v0.10.3 curl -sSL \\\n  {INSTALL_URL} | sh")},
                     red_in=page),
        install_case("trailing pipe continuation is DRIFT",
                     {page: doc(f"MIND_VERSION=v0.10.3 curl -sSL {INSTALL_URL} |\n  sh")},
                     red_in=page),
        install_case("a table cell shows the pipe escaped and is DRIFT",
                     {page: doc(good) + f"\n| Linux | `{bad.replace('|', chr(92) + '|')}` |\n"},
                     red_in=page),
        install_case("a table cell with the pin on sh passes",
                     {page: doc(good) + f"\n| Linux | `{good.replace('|', chr(92) + '|')}` |\n| Shell | sh |\n"},
                     examined=(page, 2)),
        install_case("irm piped to iex passes",
                     {page: doc(good, f"irm {INSTALL_URL} | iex")}, examined=(page, 2)),
        install_case("a script parameter on iex is DRIFT",
                     {page: doc(good, f"irm {INSTALL_URL} | iex -Version v0.10.3")}, red_in=page),
        install_case("a script parameter on irm is DRIFT",
                     {page: doc(good, f"irm {INSTALL_URL} -Version v0.10.3 | Invoke-Expression")},
                     red_in=page),
        # First-match masking: a good command earlier must not hide a bad one.
        install_case("a good command first does not hide a bad one",
                     {page: doc(good, "", bad)}, red_in=page),
        # Sibling sites: every surface the gate reads carries its own proof.
        install_case("README.md surface is DRIFT",
                     {page: doc(good), "README.md": doc(bad)}, red_in="README.md"),
        install_case("a second docs page is DRIFT",
                     {page: doc(good), "docs/other.md": doc(bad)}, red_in="docs/other.md"),
        install_case("usage comment of scripts/install.sh is DRIFT",
                     {page: doc(good), "scripts/install.sh": f"#!/bin/sh\n#   {bad}\n"},
                     red_in="scripts/install.sh"),
        install_case("usage comment of scripts/install.ps1 is DRIFT",
                     {page: doc(good), "scripts/install.ps1": f"# Usage:\n#   {bad}\n"},
                     red_in="scripts/install.ps1"),
        # Vacuity: the page that exists to show the command shows none.
        install_case("docs/install.md without an installer command is DRIFT",
                     {page: "# Install\n\nDownload a release.\n"}, red_in=page, vacuous=True),
        live_install(),
    ]
    failed = results.count(False)
    print(f"check_claims installer-env gate: cases={len(results)} failed={failed}")
    if failed:
        print("FAILED — a variable on the download tool is still reported as a pass")
        return 1
    print("OK — an installer variable on the wrong side of the pipe reddens the gate")
    return 0


def main() -> int:
    """All four suites, every verdict. A failure in either fails the gate.

    Run unconditionally rather than short-circuiting: a merged file that stopped
    after the first red suite would report LESS evidence than the two separate
    files it replaced, which is a worse gate wearing a smaller file count.
    """
    print("=== check_claims: the cost claim must be derived ===")
    cost_rc = cost_main()
    print("\n=== check_claims: a printed count must equal the derived count ===")
    surface_rc = surface_main()
    print("\n=== check_claims: a floor must keep ratcheting ===")
    floor_rc = floor_main()
    print("\n=== check_claims: an installer variable must reach the installer ===")
    install_rc = install_main()
    if cost_rc or surface_rc or floor_rc or install_rc:
        print(f"\ncheck_claims gate tests: FAILED "
              f"(cost={'FAIL' if cost_rc else 'PASS'}, "
              f"surface={'FAIL' if surface_rc else 'PASS'}, "
              f"floor={'FAIL' if floor_rc else 'PASS'}, "
              f"install={'FAIL' if install_rc else 'PASS'})")
        return 1
    print("\ncheck_claims gate tests: OK — all four comparisons bite")
    return 0


if __name__ == "__main__":
    sys.exit(main())
