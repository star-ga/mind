#!/usr/bin/env python3
"""Regression test: examples/mindc_mind/stdlib_manifest_lint.py must actually
CHECK the real manifest's count headers, not merely carry a function that could.

The hole this pins shut: check 7 (`count_header_errors`) was wired into `main()` by
one line, `errors.extend(count_header_errors(manifest_text, *counts))`. Deleting
that line left the lint green, because the lint's own hostile-variant probes call
`count_header_errors` directly and the real manifest was then simply never judged.
A reviewer reproduced it: delete the line, add a second `# BUNDLED_COUNT` header to
the manifest, and the gate still printed PASS.

So this test does NOT call `count_header_errors`. It builds a miniature repo that
holds the lint's REAL file set (the lint, its manifest reader, a copy of the real
manifest, the two Rust sources and std/*.mind it reads), damages the manifest copy,
and runs the lint's real entry point as a subprocess -- the same file, argv and exit
code CI judges. It then asserts a non-zero exit AND the count-header message, so a
lint that fails for some other reason (a crash, a missing file) does not pass here.

Non-vacuity: the first case is an UNDAMAGED copy that must exit 0, so the fixture is
proven a tree the lint can pass in, and every damaged case is proven to differ from
the real text (a replace that matched nothing is an error, not a pass).

Run: ``python3 scripts/test_stdlib_manifest_lint.py`` (no third-party deps).
"""
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MIND_DIR = REPO / "examples" / "mindc_mind"
LINT_FILES = ("stdlib_manifest_lint.py", "_stdlib_manifest.py")
MANIFEST = MIND_DIR / "testdata" / "stdlib_manifest.txt"
RUST_SOURCES = ("src/build/native_bridge.rs", "src/project/stdlib.rs")

REAL_TEXT = MANIFEST.read_text(encoding="utf-8")


def header_count(name: str) -> int:
    """The value of the single `# <name> <n>` header line in the REAL manifest."""
    hits = re.findall(rf"^# {name} (\d+)$", REAL_TEXT, re.M)
    assert len(hits) == 1, f"real manifest must state {name} exactly once, got {hits}"
    return int(hits[0])


def run_lint(manifest_text: str) -> tuple[int, str]:
    """Run the lint's real entry point against a fixture repo holding this manifest."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        mind = root / "examples" / "mindc_mind"
        (mind / "testdata").mkdir(parents=True)
        for name in LINT_FILES:
            shutil.copyfile(MIND_DIR / name, mind / name)
        (mind / "testdata" / "stdlib_manifest.txt").write_text(
            manifest_text, encoding="utf-8")
        for rel in RUST_SOURCES:
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO / rel, root / rel)
        (root / "std").mkdir()
        for src in sorted((REPO / "std").glob("*.mind")):
            shutil.copyfile(src, root / "std" / src.name)
        p = subprocess.run(
            [sys.executable, str(mind / "stdlib_manifest_lint.py")],
            capture_output=True, text=True, cwd=root,
        )
        return p.returncode, p.stdout + p.stderr


def damaged(old: str, new: str) -> str:
    """The real manifest with `old` replaced by `new`; refuses a no-op edit."""
    assert old in REAL_TEXT, f"damage anchor not in the real manifest: {old!r}"
    text = REAL_TEXT.replace(old, new, 1)
    assert text != REAL_TEXT, f"damage did not change the manifest: {old!r}"
    return text


def main() -> int:
    seed, bundled, total = (header_count(n) for n in
                            ("SEED_COUNT", "BUNDLED_COUNT", "TOTAL_COUNT"))
    seed_line, bundled_line, total_line = (
        f"# SEED_COUNT {seed}", f"# BUNDLED_COUNT {bundled}", f"# TOTAL_COUNT {total}")

    results = []

    def case(label, text, want_rc, want_in_output):
        rc, out = run_lint(text)
        ok = (rc == 0) if want_rc == 0 else (rc != 0)
        missing = [s for s in want_in_output if s not in out]
        crashed = "Traceback" in out
        ok = ok and not missing and not crashed
        # ONE VERDICT LINE PER CASE (scripts/gate_assert.py counts reported verdicts).
        print(f"[{'PASS' if ok else 'FAIL'}] {label} (expected "
              f"{'exit 0' if want_rc == 0 else 'non-zero exit'}, got {rc})")
        results.append((label, ok, "" if ok else
                        f"missing={missing} crashed={crashed}\n{out}"))

    # Control: the undamaged copy passes, so every red case below is red because
    # of its damage and not because the fixture repo is not one the lint can run in.
    case("control: real manifest copy -> lint passes", REAL_TEXT, 0,
         ["stdlib_manifest_lint: PASS"])

    # THE SURVIVING MUTANT: a second BUNDLED_COUNT header (the merge shape, a stale
    # value from the other side), reached only through the real manifest read.
    case("duplicated BUNDLED_COUNT header (stale value) -> lint fails",
         damaged(bundled_line, f"{bundled_line}\n# BUNDLED_COUNT {bundled - 12}"),
         1, ["stdlib_manifest_lint: FAIL",
             "manifest states BUNDLED_COUNT 2 times"])

    # Same value twice is still a duplicate: a merge that keeps both identical lines.
    case("duplicated TOTAL_COUNT header (same value) -> lint fails",
         damaged(total_line, f"{total_line}\n{total_line}"),
         1, ["stdlib_manifest_lint: FAIL", "manifest states TOTAL_COUNT 2 times"])

    # One header, wrong value: the count must equal the real row count.
    case("wrong SEED_COUNT value -> lint fails",
         damaged(seed_line, f"# SEED_COUNT {seed + 1}"),
         1, ["stdlib_manifest_lint: FAIL",
             f"SEED_COUNT says {seed + 1} but the rows give {seed}"])

    failures = [r for r in results if not r[1]]
    for label, _, detail in failures:
        print(f"  {label}")
        for line in detail.strip().splitlines()[:12]:
            print(f"    {line}")
    # The SANCTIONED count-marker shape read by scripts/gate_assert.py.
    print(f"\nSDLC-GATE stdlib_manifest_lint_selftest ran={len(results)} "
          f"fail={len(failures)}")
    if failures:
        return 1
    print("test_stdlib_manifest_lint: PASS — the lint judges the real manifest's "
          "count headers (duplicate and wrong-value both refused)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
