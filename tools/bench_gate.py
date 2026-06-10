#!/usr/bin/env python3
# Copyright 2025 STARGA Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
"""
Bench gate for mindc 0.2.5+.

Reads a frozen baseline (e.g. .bench-baseline-2026-04-28-pratt.txt) and
the current criterion bench output (`cargo bench --bench compiler --
--output-format bencher`) and fails the job if any of the canonical
pipeline benches regress beyond the configured threshold.

This is the CI half of the discipline pinned in docs/versioning.md:
mindc's parser/typecheck/IR pipeline is part of the runtime hot path
for mind-runtime, so we treat regressions on it the same as a public
ABI break.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Canonical pipeline benches the gate watches. Names match
# `benches/compiler.rs`.
WATCHED = ("small_matmul", "medium_mlp", "large_network")

# Pre-Pratt baseline values (microseconds). Hard-coded so the gate stays
# usable even when the baseline file is missing or unreadable.
DEFAULT_BASELINE_US = {
    "small_matmul": 3.00,
    "medium_mlp": 6.13,
    "large_network": 16.82,
}

# Default criterion output uses two-line entries:
#   Benchmarking compiler_pipeline/parse_typecheck_ir/small_matmul: Analyzing
#                           time:   [2.9686 µs 2.9883 µs 3.0673 µs]
BENCHMARKING_LINE = re.compile(
    r"Benchmarking\s+compiler_pipeline/parse_typecheck_ir/(?P<name>\S+?):"
)
TIME_LINE = re.compile(
    r"time:\s*\[\s*([0-9.]+)\s*(?P<unit>[µu]?s|ms|ns)\s+"
    r"(?P<mid>[0-9.]+)\s*[µu]?s|ms|ns"
)
# Robust: match the middle "mid" value with whatever unit precedes it.
TIME_PATTERN = re.compile(
    r"time:\s*\[\s*([0-9.]+)\s*([µu]?s|ms|ns)\s+"
    r"([0-9.]+)\s*([µu]?s|ms|ns)\s+"
    r"([0-9.]+)\s*([µu]?s|ms|ns)\s*\]"
)


def _to_us(value: float, unit: str) -> float:
    if unit == "ns":
        return value / 1000.0
    if unit in ("µs", "us"):
        return value
    if unit == "ms":
        return value * 1000.0
    return value


def parse_baseline(path: Path) -> dict[str, float]:
    """Read a baseline file. Falls back to DEFAULT_BASELINE_US."""
    if not path.exists():
        return dict(DEFAULT_BASELINE_US)
    out: dict[str, float] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip().lstrip("-").strip()
        for name in WATCHED:
            if line.startswith(f"{name}:"):
                # Lines like "small_matmul:   3.00 µs   ..."
                m = re.search(r"([0-9]+\.[0-9]+)\s*(?:µs|us|microseconds)", line)
                if m:
                    out[name] = float(m.group(1))
    for name in WATCHED:
        out.setdefault(name, DEFAULT_BASELINE_US[name])
    return out


# Bencher-style one-line output (criterion --output-format bencher):
#   test compiler_pipeline/parse_typecheck_ir/small_matmul ... bench:  2773 ns/iter (+/- 76)
BENCHER_LINE = re.compile(
    r"test\s+compiler_pipeline/parse_typecheck_ir/(?P<name>\S+)\s+\.\.\.\s+"
    r"bench:\s+(?P<value>[0-9.]+)\s+(?P<unit>ns|us|µs|ms)/iter"
)


def parse_current(path: Path) -> dict[str, float]:
    """Read criterion bench output into microseconds.

    Handles both criterion's default format (`Benchmarking ... time: [..]`)
    and `--output-format bencher` (`test ... bench: N ns/iter (+/- ..)`).
    The CI workflow uses bencher format.

    For default format, uses the mean (middle) of `[low mid high]`.
    For bencher format, the single value is the median.
    """
    text = path.read_text()
    out: dict[str, float] = {}

    # Pass 1: bencher format (one-liner; preferred — matches CI invocation).
    for raw in text.splitlines():
        m = BENCHER_LINE.search(raw)
        if m and m.group("name") not in out:
            value = float(m.group("value"))
            out[m.group("name")] = _to_us(value, m.group("unit"))

    # Pass 2: default criterion format (two-line "Benchmarking ... time:").
    # Skipped if pass 1 already populated the bench (bencher numbers are
    # canonical for the gate).
    pending: str | None = None
    for raw in text.splitlines():
        m = BENCHMARKING_LINE.search(raw)
        if m:
            pending = m.group("name")
            continue
        if pending is not None:
            t = TIME_PATTERN.search(raw)
            if t and pending not in out:
                mid_value = float(t.group(3))
                mid_unit = t.group(4)
                out[pending] = _to_us(mid_value, mid_unit)
            pending = None
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="mindc bench gate")
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--current", type=Path, required=True)
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.10,
        help=(
            "max allowed mean REGRESSION as a fraction; 0.10 = +10%%. "
            "One-sided: a speedup (negative delta) never fails, no upper "
            "bound. Exceeding it is a stop-and-decide trigger, not a block."
        ),
    )
    args = ap.parse_args()

    baseline = parse_baseline(args.baseline)
    current = parse_current(args.current)

    rows: list[tuple[str, float, float, float, bool]] = []
    failed = False
    for name in WATCHED:
        b = baseline.get(name)
        c = current.get(name)
        if b is None or c is None:
            print(f"::warning::missing bench for {name} (baseline={b}, current={c})")
            continue
        # One-sided gate: delta < 0 is a speedup (always OK, no upper bound);
        # only a regression strictly beyond +threshold trips it. A trip is a
        # stop-and-decide signal (revert, or re-bless the baseline with a
        # documented rationale when a dramatic win elsewhere justifies the
        # slowdown), not a permanent block.
        delta = (c - b) / b
        ok = delta <= args.threshold
        failed = failed or not ok
        rows.append((name, b, c, delta, ok))

    print()
    print(f"{'bench':<18} {'baseline':>12} {'current':>12} {'delta':>10}   gate")
    for name, b, c, delta, ok in rows:
        marker = "OK" if ok else "FAIL"
        print(
            f"{name:<18} {b:>10.3f} µs {c:>10.3f} µs {delta:>+9.2%}   {marker}"
        )
    print()

    if failed:
        print(
            f"::error::pipeline regression exceeded +{args.threshold:.0%} "
            f"(one-sided gate; a speedup never fails). STOP and decide if it "
            f"is worth it: revert, or re-bless the baseline with a documented "
            f"rationale when a dramatic win elsewhere justifies the slowdown."
        )
        return 1
    print("bench gate: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
