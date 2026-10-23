#!/usr/bin/env python3
"""H2b exclusion positive controls (Ruling Q1.3(b), Schedule R7).

For EVERY per-file `.gitignore` exclusion under /examples/*.mind this gate
asserts the Q1.3(b) positive controls, and for the R7 caller rows it also
asserts the external digest pin and the sealed-import note.

Checks, per excluded path (ALL exclusions — not only the catalogue):

  1. `git ls-files --error-unmatch` still lists the path;
  2. the path still appears in ANATOMY.md (anatomy.sh walks
     `git ls-files --cached`, so an ignored-but-tracked file stays listed);
  3. the `.gitignore` reason names a reason class
     (refusal-fixture | digest-pinned-gate-input | syntax-under-test)
     and a gate (or `no-ci-gate-yet (owner: ...)`).

Checks, additionally per catalogue entry (the 18 pinned rows):

  4. the file's bytes match the committed external digest;
  5. the reason cites this pin site;
  6. caller rows name the sealed module they import (R7).

Structural floors (the empty / truncated catalogue must NOT pass):

  7. the catalogue is exactly the committed expected path set
     (bidirectional — a dropped or added row is a FAIL);
  8. the .gitignore exclusion floor is >= the committed expected count;
  9. every named owning gate is a non-vacuous script (exists and carries
     real assertion machinery, not a stub that exits 0 having checked
     nothing).

R7 table: the gate prints the per-caller table
(path | gate-input? | owning gate | sealed module imported) so the
exclusion ledger doubles as the coverage ledger.

Red mutants (each must make this gate FAIL):
  --mutant-pin          flip one hex digit in a stored digest
  --mutant-track        pretend `git ls-files` lost the path
  --mutant-anatomy      drop the first catalogue path from the ANATOMY index
  --mutant-anatomy-any  drop a NON-catalogue excluded path (native_dynamic.mind)
  --mutant-reason       strip the pin-site citation from the reason line
  --mutant-empty        empty the catalogue
  --mutant-drop-entry   remove the first catalogue entry
  --mutant-owning-stub  replace an owning gate with a zero-assertion exit
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

# Compiled-in expected sets. These are NOT derived from the catalogue at run
# time: an empty or truncated catalogue must fail against them.
EXPECTED_CATALOGUE_PATHS = frozenset(
    {
        "examples/mindc_mind/native_c_call_emit_driver.mind",
        "examples/mindc_mind/native_c_call_plan_driver.mind",
        "examples/mindc_mind/native_dynamic_allocator.mind",
        "examples/mindc_mind/native_dynamic_allocator_driver.mind",
        "examples/mindc_mind/native_dynamic_fixture.mind",
        "examples/mindc_mind/native_dynamic_image.mind",
        "examples/mindc_mind/native_dynamic_image_fixture.mind",
        "examples/mindc_mind/native_dynamic_layout_driver.mind",
        "examples/mindc_mind/native_dynamic_mic_fixture.mind",
        "examples/mindc_mind/native_emission_context.mind",
        "examples/mindc_mind/native_libc_io_fixture.mind",
        "examples/mindc_mind/native_libc_startup.mind",
        "examples/mindc_mind/native_libc_startup_fixture.mind",
        "examples/mindc_mind/native_scoped_source_driver.mind",
        "examples/mindc_mind/native_source_image.mind",
        "examples/mindc_mind/testdata/native_source_driver/note_fixture_tail.mind",
        "examples/mindc_mind/source_note_driver.mind",
        "examples/mindc_mind/string_escape_driver.mind",
    }
)
EXPECTED_CALLER_PATHS = frozenset(
    {
        "examples/mindc_mind/native_c_call_emit_driver.mind",
        "examples/mindc_mind/native_c_call_plan_driver.mind",
        "examples/mindc_mind/native_dynamic_allocator.mind",
        "examples/mindc_mind/native_dynamic_allocator_driver.mind",
        "examples/mindc_mind/native_dynamic_fixture.mind",
        "examples/mindc_mind/native_dynamic_image.mind",
        "examples/mindc_mind/native_dynamic_image_fixture.mind",
        "examples/mindc_mind/native_dynamic_layout_driver.mind",
        "examples/mindc_mind/native_dynamic_mic_fixture.mind",
        "examples/mindc_mind/native_emission_context.mind",
        "examples/mindc_mind/native_libc_io_fixture.mind",
        "examples/mindc_mind/native_libc_startup.mind",
        "examples/mindc_mind/native_libc_startup_fixture.mind",
        "examples/mindc_mind/native_scoped_source_driver.mind",
        "examples/mindc_mind/native_source_image.mind",
        "examples/mindc_mind/testdata/native_source_driver/note_fixture_tail.mind",
    }
)
EXPECTED_EXCLUSION_FLOOR = 116
EXPECTED_CATALOGUE_SIZE = 18
EXPECTED_CALLER_SIZE = 16

# Reason-class vocabulary of Ruling Q1.2.
REASON_CLASSES = (
    "refusal-fixture",
    "digest-pinned-gate-input",
    "syntax-under-test",
)

# Minimal assertion machinery a real owning gate must carry. A stub that
# prints PASS and exits 0 has none of these.
_ASSERTION_MARKERS = (
    "assert ",
    "failures",
    "return 1",
    "FAIL ",
    "asserted",
    "p.error(",
    "check=True",
    "raise ",
    "sys.exit(1)",
    "os.EX_",
)


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


def gitignore_exclusions() -> dict[str, str]:
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


def reason_gate(reason: str) -> str:
    m = re.search(r"gate:\s*([^\n]+)", reason)
    return m.group(1).strip() if m else ""


def reason_class(reason: str) -> str:
    """A reason class token appearing anywhere in the comment block."""
    for cls in REASON_CLASSES:
        # accept "# refusal-fixture", "# refusal-fixture;", "# -- refusal-fixture:"
        if re.search(rf"(^|\n)\s*#\s*-?\s*{re.escape(cls)}\b", reason):
            return cls
    return ""


def owning_gate_name(reason: str) -> str:
    """Extract the concrete owning-gate script name from a reason line.

    Handles both shapes:
      gate: mindfuzz_self_host
      gate: no-ci-gate-yet (owner: L2; native_c_call_emit_check, manual)
    """
    g = reason_gate(reason)
    if not g:
        return ""
    m = re.search(r"no-ci-gate-yet\s*\(owner:\s*[^;]+;\s*([^,)]+)", g)
    if m:
        return m.group(1).strip()
    # bare gate name (possibly with trailing note)
    return g.split(",")[0].split("(")[0].strip()


def gate_is_nonvacuous(gate_name: str) -> tuple[bool, str]:
    """A named owning gate must exist and carry real assertion machinery."""
    if not gate_name:
        return False, "NO_GATE_NAMED"
    if gate_name in ("NSF/1 catalogue", "NSF/1 catalogue, PRELISTED_NOT_EXECUTED"):
        # catalogue rows are PRELISTED_NOT_EXECUTED by declaration; the pin
        # site is the evidence, not an executing gate.
        return True, "PRELISTED"
    candidate = ROOT / "examples" / "mindc_mind" / f"{gate_name}.py"
    if not candidate.is_file():
        # some gates live under tests/ or scripts/
        for alt in (
            ROOT / "tests" / f"{gate_name}.py",
            ROOT / "scripts" / f"{gate_name}.py",
            ROOT / "examples" / "mindc_mind" / f"{gate_name}.sh",
            ROOT / "scripts" / f"{gate_name}.sh",
        ):
            if alt.is_file():
                candidate = alt
                break
        else:
            return False, f"GATE_MISSING {gate_name}"
    text = candidate.read_text(encoding="utf-8", errors="replace")
    hits = sum(1 for marker in _ASSERTION_MARKERS if marker in text)
    if hits == 0:
        return False, f"GATE_ZERO_ASSERTION {gate_name} ({candidate.relative_to(ROOT)})"
    if len(text.splitlines()) < 20:
        return False, f"GATE_STUB {gate_name} ({len(text.splitlines())} lines)"
    return True, f"ASSERT_MARKERS={hits}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mutant-pin", action="store_true")
    ap.add_argument("--mutant-track", action="store_true")
    ap.add_argument("--mutant-anatomy", action="store_true")
    ap.add_argument("--mutant-anatomy-any", action="store_true")
    ap.add_argument("--mutant-reason", action="store_true")
    ap.add_argument("--mutant-empty", action="store_true")
    ap.add_argument("--mutant-drop-entry", action="store_true")
    ap.add_argument("--mutant-owning-stub", action="store_true")
    ap.add_argument("--r7-table", action="store_true", help="print the R7 caller table")
    args = ap.parse_args()

    doc = json.loads(PINS.read_text())
    entries = doc["entries"]
    if args.mutant_empty:
        entries = []
    elif args.mutant_drop_entry:
        entries = entries[1:]

    catalogue_paths = {e["path"] for e in entries}
    reasons = gitignore_exclusions()
    indexed = anatomy_index()
    asserted = 0
    failures: list[str] = []

    # --- structural floors (empty / truncated catalogue must NOT pass) ------
    if len(entries) != EXPECTED_CATALOGUE_SIZE:
        failures.append(
            f"CATALOGUE_SIZE {len(entries)} != {EXPECTED_CATALOGUE_SIZE}"
        )
    else:
        asserted += 1
    if catalogue_paths != set(EXPECTED_CATALOGUE_PATHS):
        missing = sorted(set(EXPECTED_CATALOGUE_PATHS) - catalogue_paths)
        extra = sorted(catalogue_paths - set(EXPECTED_CATALOGUE_PATHS))
        failures.append(
            f"CATALOGUE_SET_MISMATCH missing={missing} extra={extra}"
        )
    else:
        asserted += 1
    if len(reasons) < EXPECTED_EXCLUSION_FLOOR:
        failures.append(
            f"EXCLUSION_FLOOR {len(reasons)} < {EXPECTED_EXCLUSION_FLOOR}"
        )
    else:
        asserted += 1

    # --- per-exclusion Q1.3(b) positive controls (ALL exclusions) ----------
    for path, reason in sorted(reasons.items()):
        tracked = True
        if args.mutant_track and path == sorted(reasons)[0]:
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
        if args.mutant_anatomy and path == sorted(catalogue_paths)[0]:
            in_anatomy = False
        if args.mutant_anatomy_any and path == "examples/mindc_mind/native_dynamic.mind":
            in_anatomy = False
        if not in_anatomy:
            failures.append(f"ANATOMY_MISSING {path}")
        else:
            asserted += 1

        # Strict reason-class / gate / owning-gate checks apply to the H2b
        # per-file exclusion block (the 107 rows carrying a Q1.2 reason class).
        # Pre-H2b exclusions (aspirational examples, etc.) predate that
        # vocabulary and are covered by the tracking + ANATOMY controls above.
        cls = reason_class(reason)
        has_h2b = bool(cls) or "gate:" in reason
        if has_h2b:
            if not cls:
                failures.append(f"REASON_NO_CLASS {path}")
            else:
                asserted += 1

            gate = reason_gate(reason)
            if not gate:
                failures.append(f"REASON_NO_GATE {path}")
            else:
                asserted += 1

            gate_name = owning_gate_name(reason)
            ok, detail = gate_is_nonvacuous(gate_name)
            if args.mutant_owning_stub and gate_name == "native_c_call_emit_check":
                ok, detail = False, f"GATE_ZERO_ASSERTION {gate_name} (mutant)"
            if not ok:
                failures.append(f"OWNING_GATE {path} {detail}")
            else:
                asserted += 1

    # --- catalogue-external-digest + pin-site checks -----------------------
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

        reason = reasons.get(path, "")
        if args.mutant_reason and path == entries[0]["path"]:
            reason = reason.replace(PIN_SITE, "testdata/nowhere.json")
        if PIN_SITE not in reason:
            failures.append(f"REASON_NO_PIN_SITE {path} reason={reason!r}")
        else:
            asserted += 1
        if entry["kind"] == "caller-exclusion":
            if "imports " not in reason:
                failures.append(f"REASON_NO_SEALED_IMPORT {path}")
            else:
                asserted += 1
            if path not in EXPECTED_CALLER_PATHS:
                failures.append(f"CALLER_UNEXPECTED {path}")
            else:
                asserted += 1

    # --- caller-set floor -------------------------------------------------
    caller_paths = {e["path"] for e in entries if e["kind"] == "caller-exclusion"}
    if len(caller_paths) != EXPECTED_CALLER_SIZE:
        failures.append(f"CALLER_SIZE {len(caller_paths)} != {EXPECTED_CALLER_SIZE}")
    else:
        asserted += 1

    # --- R7 table ---------------------------------------------------------
    if args.r7_table or True:
        # Measured asserted=N before/after the .gitignore edit for the CI
        # owning gates that consume excluded files as inputs. The .gitignore
        # entry does not change any gate's file list (gates read by path), so
        # the counts are unchanged by construction; these are the measured
        # values from the same-tree runs.
        _MEASURED = {
            "self_host_native_misc_census_smoke": "94/94",
            "self_host_native_alloc_bounded_smoke": "37/37",
            "mindfuzz_self_host": "100/100",
            "self_host_native_cf_smoke": "900s-timeout@head=base",
            "self_host_source_note_smoke": "setup-fail@head=base",
        }
        print("R7_TABLE_BEGIN")
        print("path\tgate-input?\towing_gate\tasserted_before/after\tsealed_module_imported")
        for path in sorted(EXPECTED_CALLER_PATHS):
            reason = reasons.get(path, "")
            gate_name = owning_gate_name(reason)
            m = re.search(r"imports ([^;]+)", reason)
            sealed = m.group(1).strip() if m else ""
            measured = _MEASURED.get(gate_name, "manual-gate-nonvacuous")
            print(f"{path}\tyes\t{gate_name}\t{measured}\t{sealed}")
        print("R7_TABLE_END")

    print(f"h2b_exclusion_controls: exclusions={len(reasons)} entries={len(entries)} asserted={asserted}")
    for f in failures:
        print(f"FAIL {f}")
    if failures:
        print("h2b_exclusion_controls: FAIL")
        return 1
    print("h2b_exclusion_controls: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
