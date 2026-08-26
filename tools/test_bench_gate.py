#!/usr/bin/env python3
"""Regression test for the bench-gate fail-closed contract (tools/bench_gate.py).

Guards the vacuous-PASS hole: an empty / truncated / partial ``bench.out``, or a
missing baseline file, must exit NONZERO (code 4 = malformed gate input) — never
fall through to ``bench gate: PASS``. A uniformly-noisy run still returns its
inconclusive code, not a false green.

Run: ``python3 tools/test_bench_gate.py`` (no third-party deps).
"""
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
GATE = HERE / "bench_gate.py"

GOOD_CURRENT = (
    "test compiler_pipeline/parse_typecheck_ir/small_matmul ... bench:  3000 ns/iter (+/- 50)\n"
    "test compiler_pipeline/parse_typecheck_ir/medium_mlp ... bench:  6100 ns/iter (+/- 80)\n"
    "test compiler_pipeline/parse_typecheck_ir/large_network ... bench:  16800 ns/iter (+/- 120)\n"
)
GOOD_BASELINE = "small_matmul:   3.00 µs\nmedium_mlp:   6.13 µs\nlarge_network:   16.82 µs\n"


def run(baseline: Path, current: Path) -> int:
    return subprocess.run(
        [sys.executable, str(GATE), "--baseline", str(baseline), "--current", str(current)],
        capture_output=True,
        text=True,
    ).returncode


def main() -> int:
    failures: list[str] = []
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        base = d / "baseline.txt"
        base.write_text(GOOD_BASELINE)
        good = d / "good.out"
        good.write_text(GOOD_CURRENT)
        empty = d / "empty.out"
        empty.write_text("")
        partial = d / "partial.out"
        partial.write_text("\n".join(GOOD_CURRENT.splitlines()[:2]) + "\n")

        cases = [
            ("empty current -> exit 4 (the vacuous-PASS hole)", run(base, empty), 4),
            ("partial current 2/3 -> exit 4 (no silent skip)", run(base, partial), 4),
            ("missing baseline -> exit 4 (no default substitution)", run(d / "nope.txt", good), 4),
            ("valid full run -> exit 0 (normal PASS preserved)", run(base, good), 0),
        ]
        for label, got, want in cases:
            ok = got == want
            print(f"[{'PASS' if ok else 'FAIL'}] {label} (got {got}, want {want})")
            if not ok:
                failures.append(label)

    if failures:
        print(f"\n{len(failures)} bench-gate contract test(s) FAILED")
        return 1
    print("\nbench-gate fail-closed contract: all cases OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
