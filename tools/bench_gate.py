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
# The "(+/- N)" is criterion's measured spread — load-bearing here: a run on
# a busy box produces a huge spread, and a delta computed from a noisy median
# is statistically meaningless, so the gate must NOT trip on it.
BENCHER_LINE = re.compile(
    r"test\s+compiler_pipeline/parse_typecheck_ir/(?P<name>\S+)\s+\.\.\.\s+"
    r"bench:\s+(?P<value>[0-9.]+)\s+(?P<unit>ns|us|µs|ms)/iter"
    r"(?:\s*\(\+/-\s*(?P<variance>[0-9.]+)\s*(?P<vunit>ns|us|µs|ms)?\))?"
)


def parse_current(path: Path) -> dict[str, tuple[float, float | None]]:
    """Read criterion bench output into ``{name: (microseconds, rel_variance)}``.

    ``rel_variance`` is the bencher ``(+/- N)`` spread divided by the median
    (``None`` when unavailable, e.g. the default two-line format). It lets the
    gate ignore load-contaminated runs instead of failing on noise.

    Handles both criterion's default format (`Benchmarking ... time: [..]`)
    and `--output-format bencher` (`test ... bench: N ns/iter (+/- ..)`).
    The CI workflow uses bencher format.
    """
    text = path.read_text()
    out: dict[str, tuple[float, float | None]] = {}

    # Pass 1: bencher format (one-liner; preferred — matches CI invocation).
    for raw in text.splitlines():
        m = BENCHER_LINE.search(raw)
        if m and m.group("name") not in out:
            us = _to_us(float(m.group("value")), m.group("unit"))
            rel_var: float | None = None
            if m.group("variance") is not None and us > 0:
                vunit = m.group("vunit") or m.group("unit")
                var_us = _to_us(float(m.group("variance")), vunit)
                rel_var = var_us / us
            out[m.group("name")] = (us, rel_var)

    # Pass 2: default criterion format (two-line "Benchmarking ... time:").
    # The [low mid high] spread gives a variance proxy: (high-low)/mid.
    pending: str | None = None
    for raw in text.splitlines():
        m = BENCHMARKING_LINE.search(raw)
        if m:
            pending = m.group("name")
            continue
        if pending is not None:
            t = TIME_PATTERN.search(raw)
            if t and pending not in out:
                low = _to_us(float(t.group(1)), t.group(2))
                mid = _to_us(float(t.group(3)), t.group(4))
                high = _to_us(float(t.group(5)), t.group(6))
                rel_var = ((high - low) / mid) if mid > 0 else None
                out[pending] = (mid, rel_var)
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
    ap.add_argument(
        "--max-rel-variance",
        type=float,
        default=0.12,
        help=(
            "max trusted spread; a bench whose criterion (+/- N) spread "
            "exceeds this fraction of its median is INCONCLUSIVE (the box was "
            "loaded), not a regression — the gate refuses to fail on it and "
            "asks for an idle re-run. 0.12 = 12%%."
        ),
    )
    args = ap.parse_args()

    baseline = parse_baseline(args.baseline)
    current = parse_current(args.current)

    # (name, baseline_us, current_us, delta, rel_var, verdict)
    #   verdict ∈ {"OK", "FAIL", "NOISY"}
    rows: list[tuple[str, float, float, float, float | None, str]] = []
    failed = False
    trusted = 0
    for name in WATCHED:
        b = baseline.get(name)
        cv = current.get(name)
        if b is None or cv is None:
            print(f"::warning::missing bench for {name} (baseline={b}, current={cv})")
            continue
        c, rel_var = cv
        delta = (c - b) / b
        # A regression only counts when the measurement is TRUSTWORTHY. A run
        # on a busy box has a large (+/- N) spread; a delta from a noisy median
        # is meaningless, so flag it NOISY (inconclusive) rather than FAIL.
        if rel_var is not None and rel_var > args.max_rel_variance:
            verdict = "NOISY"
        else:
            trusted += 1
            # One-sided gate: delta < 0 is a speedup (always OK); only a
            # regression strictly beyond +threshold on a TRUSTED measurement
            # trips it (stop-and-decide, not a permanent block).
            ok = delta <= args.threshold
            verdict = "OK" if ok else "FAIL"
            failed = failed or not ok
        rows.append((name, b, c, delta, rel_var, verdict))

    print()
    print(f"{'bench':<18} {'baseline':>12} {'current':>12} {'delta':>10} {'spread':>8}   gate")
    for name, b, c, delta, rel_var, verdict in rows:
        sp = f"{rel_var:.1%}" if rel_var is not None else "  n/a"
        print(
            f"{name:<18} {b:>10.3f} µs {c:>10.3f} µs {delta:>+9.2%} {sp:>8}   {verdict}"
        )
    print()

    noisy = [r[0] for r in rows if r[5] == "NOISY"]
    if failed:
        print(
            f"::error::pipeline regression exceeded +{args.threshold:.0%} on a "
            f"low-variance (trustworthy) measurement. STOP and decide: revert, "
            f"or re-bless the baseline with a documented rationale when a "
            f"dramatic win elsewhere justifies the slowdown."
        )
        return 1
    if trusted == 0 and noisy:
        # Every watched bench was too noisy to trust — the box was loaded.
        # Do NOT pass-or-fail on garbage; signal a re-run is needed.
        print(
            f"::warning::all benches inconclusive (spread > "
            f"{args.max_rel_variance:.0%}): the box was loaded during the run. "
            f"Re-run on an idle host (taskset-pinned) before trusting deltas."
        )
        return 2
    if noisy:
        print(
            f"::warning::ignored {len(noisy)} noisy bench(es) {noisy} "
            f"(spread > {args.max_rel_variance:.0%}); gated on the "
            f"{trusted} trustworthy one(s)."
        )
    print("bench gate: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
