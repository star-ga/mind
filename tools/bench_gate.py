#!/usr/bin/env python3
# Copyright 2025 STARGA Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
"""
Bench gate for mindc — two-baseline ratcheting perf-regression gate.

Reads the current criterion bench output (`cargo bench --bench <b> --
--output-format bencher`) and gates it against TWO references:

  * CHAMPION (--champion): the last-best committed numbers — a monotonic
    "only-ever-faster" ratchet. This is what catches SLOW DRIFT: ten commits
    each +3% pass any per-commit or stale-floor check yet lose 30% cumulatively;
    only a best-ever ratchet remembers the real target.
  * FLOOR (--floor / --baseline): an absolute correctness-milestone backstop
    (e.g. .bench-baseline-2026-06-01-correctness.txt). Never advances; the
    safety net if the champion chain is ever mis-blessed.

A bench FAILS if, on a TRUSTWORTHY (low-variance) measurement, current exceeds
either champion×(1+threshold) or floor×(1+threshold). One-sided: a speedup never
fails. Between champion×(1-deadband) and champion×(1+threshold) it PASSES with no
re-bless (the dead band absorbs criterion's 2-10% runner variance so noise cannot
tighten the ratchet to its own minimum). Below champion×(1-deadband) with every
other bench at parity, the bench is flagged a RE-BLESS CANDIDATE — the gate never
re-blesses automatically; a human/CI confirmation run commits the new champion.

Bench selection is generic: the gate watches whatever bench names appear in the
champion (or floor) reference, so adding `simple_benchmarks` (scalar_math et al.)
to the gate is just adding them to the champion file — no code change here.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Legacy hard-coded fallback (compiler_pipeline only) — used when NEITHER a
# champion nor a floor file is readable, so the gate stays usable bare.
DEFAULT_BASELINE_US = {
    "small_matmul": 3.00,
    "medium_mlp": 6.13,
    "large_network": 16.82,
}

# Any criterion bench id: "<group>/<sub>/<name>" or "<group>/<name>". We key on
# the LAST path component (the fixture name), which is unique across our benches
# (small_matmul, scalar_math, tensor_ops, ...). Matches both the two-line
# "Benchmarking <id>:" and the one-line bencher "test <id> ... bench:" formats.
_ID = r"(?P<id>[A-Za-z0-9_]+(?:/[A-Za-z0-9_]+)*)"
BENCHMARKING_LINE = re.compile(rf"Benchmarking\s+{_ID}\s*:")
BENCHER_LINE = re.compile(
    rf"test\s+{_ID}\s+\.\.\.\s+"
    r"bench:\s+(?P<value>[0-9.]+)\s+(?P<unit>ns|us|µs|ms)/iter"
    r"(?:\s*\(\+/-\s*(?P<variance>[0-9.]+)\s*(?P<vunit>ns|us|µs|ms)?\))?"
)
TIME_PATTERN = re.compile(
    r"time:\s*\[\s*([0-9.]+)\s*([µu]?s|ms|ns)\s+"
    r"([0-9.]+)\s*([µu]?s|ms|ns)\s+"
    r"([0-9.]+)\s*([µu]?s|ms|ns)\s*\]"
)
# A reference-file line: "<name>   2.98 µs   ..." (baseline/champion files).
REF_LINE = re.compile(
    r"^\s*-?\s*(?P<name>[A-Za-z0-9_]+)\s*[:=]?\s+"
    r"(?P<value>[0-9]+\.[0-9]+)\s*(?:µs|us|microseconds)"
)


def _to_us(value: float, unit: str) -> float:
    if unit == "ns":
        return value / 1000.0
    if unit in ("µs", "us"):
        return value
    if unit == "ms":
        return value * 1000.0
    return value


def _leaf(bench_id: str) -> str:
    return bench_id.rsplit("/", 1)[-1]


def parse_reference(path: Path | None, prefix: str | None = None) -> dict[str, float]:
    """Read a champion/floor reference file into ``{full_bench_id: microseconds}``.

    Accepts both the prose milestone-baseline format (``small_matmul: 2.98 µs``)
    and a bencher dump. Bencher lines already carry the full id. Prose short
    names are prefixed with ``prefix`` (e.g. ``compiler_pipeline/parse_typecheck_ir``)
    so the legacy compiler-pipeline floor keys on full ids and can NEVER
    leaf-collide onto a same-named fixture in another bench family (e.g.
    simple_benchmarks' slower ``small_matmul``). Returns ``{}`` if unreadable.
    """
    if path is None or not path.exists():
        return {}
    out: dict[str, float] = {}
    text = path.read_text()
    # Bencher-format champion dumps key on the FULL bench id (group/sub/name),
    # so distinct benches that share a leaf fixture name — e.g.
    # compiler_pipeline/.../small_matmul vs compile_small/.../small_matmul — do
    # NOT collide. Prose milestone lines ("small_matmul: 2.98 µs") key on the
    # short leaf name; a full-id lookup falls back to the leaf (see main()).
    for raw in text.splitlines():
        m = BENCHER_LINE.search(raw)
        if m:
            out.setdefault(m.group("id"), _to_us(float(m.group("value")), m.group("unit")))
    for raw in text.splitlines():
        m = REF_LINE.search(raw)
        if m:
            key = f"{prefix}/{m.group('name')}" if prefix else m.group("name")
            out.setdefault(key, float(m.group("value")))
    return out


def parse_current(path: Path) -> dict[str, tuple[float, float | None]]:
    """Read criterion bench output into ``{name: (microseconds, rel_variance)}``.

    ``rel_variance`` is the bencher ``(+/- N)`` spread over the median (``None``
    when unavailable). Keyed on the leaf fixture name so it lines up with the
    reference files regardless of the benchmark group prefix.
    """
    text = path.read_text()
    out: dict[str, tuple[float, float | None]] = {}

    # Pass 1: bencher one-liner (matches the CI invocation). Key on the FULL
    # bench id so leaf-name-colliding fixtures across benches stay distinct.
    for raw in text.splitlines():
        m = BENCHER_LINE.search(raw)
        if m:
            name = m.group("id")
            if name in out:
                continue
            us = _to_us(float(m.group("value")), m.group("unit"))
            rel_var: float | None = None
            if m.group("variance") is not None and us > 0:
                vunit = m.group("vunit") or m.group("unit")
                rel_var = _to_us(float(m.group("variance")), vunit) / us
            out[name] = (us, rel_var)

    # Pass 2: default two-line "Benchmarking <id>: / time: [low mid high]".
    pending: str | None = None
    for raw in text.splitlines():
        m = BENCHMARKING_LINE.search(raw)
        if m:
            pending = m.group("id")
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
    ap = argparse.ArgumentParser(description="mindc two-baseline bench gate")
    ap.add_argument(
        "--champion",
        type=Path,
        default=None,
        help="ratcheting best-ever reference (the primary gate). Optional.",
    )
    ap.add_argument(
        "--floor",
        "--baseline",
        dest="floor",
        type=Path,
        default=None,
        help="absolute correctness-milestone floor (backstop). --baseline is a "
        "back-compat alias.",
    )
    ap.add_argument(
        "--floor-prefix",
        default="compiler_pipeline/parse_typecheck_ir",
        help="bench-id prefix applied to the floor file's short fixture names "
        "(the legacy floor was measured for compiler_pipeline). Set '' if the "
        "floor file already uses full bench ids.",
    )
    ap.add_argument("--current", type=Path, required=True)
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.10,
        help="max allowed REGRESSION over a reference; 0.10 = +10%%. One-sided.",
    )
    ap.add_argument(
        "--deadband",
        type=float,
        default=0.05,
        help="a win must beat champion by more than this to be a re-bless "
        "candidate; 0.05 = 5%%. Absorbs runner noise so the ratchet can't "
        "self-tighten.",
    )
    ap.add_argument(
        "--max-rel-variance",
        type=float,
        default=0.12,
        help="a bench whose criterion (+/- N) spread exceeds this fraction of "
        "its median is INCONCLUSIVE (loaded box), not a regression.",
    )
    args = ap.parse_args()

    champion = parse_reference(args.champion)
    # The legacy correctness-milestone floor carries short fixture names that
    # were measured for the compiler_pipeline bench; key them onto that full id
    # so they only gate those benches (never a same-named fixture elsewhere).
    floor = parse_reference(args.floor, prefix=args.floor_prefix)
    if not champion and not floor:
        floor = dict(DEFAULT_BASELINE_US)
    # Gate whatever names the references declare (champion preferred).
    watched = sorted(champion.keys() or floor.keys())
    current = parse_current(args.current)

    rows: list[tuple[str, float | None, float | None, float, float, float | None, str]] = []
    failed = False
    trusted = 0
    rebless_candidates: list[str] = []
    for name in watched:
        ch = champion.get(name)
        # Floor keys on full ids too (prose short names were prefixed at parse),
        # so a bench only gets a floor when one was actually measured for IT.
        fl = floor.get(name)
        ref = ch if ch is not None else fl  # champion is the primary reference
        cv = current.get(name)
        if ref is None or cv is None:
            print(f"::warning::missing bench for {name} (champion={ch}, floor={fl}, current={cv})")
            continue
        c, rel_var = cv
        d_ch = (c - ch) / ch if ch is not None else None
        d_fl = (c - fl) / fl if fl is not None else None
        delta = (c - ref) / ref
        if rel_var is not None and rel_var > args.max_rel_variance:
            verdict = "NOISY"
        else:
            trusted += 1
            # FAIL if it regresses beyond threshold over EITHER reference.
            over_ch = d_ch is not None and d_ch > args.threshold
            over_fl = d_fl is not None and d_fl > args.threshold
            if over_ch or over_fl:
                verdict = "FAIL"
                failed = True
            elif ch is not None and d_ch < -args.deadband:
                verdict = "WIN?"  # re-bless candidate (needs confirmation run)
                rebless_candidates.append(name)
            else:
                verdict = "OK"
        rows.append((name, ch, fl, c, delta, rel_var, verdict))

    print()
    hdr = f"{'bench':<16} {'champion':>10} {'floor':>10} {'current':>10} {'Δchamp':>9} {'spread':>7}  gate"
    print(hdr)
    for name, ch, fl, c, delta, rel_var, verdict in rows:
        chs = f"{ch:.3f}" if ch is not None else "   —"
        fls = f"{fl:.3f}" if fl is not None else "   —"
        sp = f"{rel_var:.1%}" if rel_var is not None else " n/a"
        print(f"{name:<16} {chs:>10} {fls:>10} {c:>9.3f}µ {delta:>+8.2%} {sp:>7}  {verdict}")
    print()

    noisy = [r[0] for r in rows if r[6] == "NOISY"]
    if failed:
        print(
            "::error::pipeline regression exceeded "
            f"+{args.threshold:.0%} over the champion or floor on a trustworthy "
            "measurement. STOP and decide: revert, or (if a dramatic win "
            "elsewhere justifies it) re-bless the champion with a documented "
            "rationale + a confirmation run."
        )
        return 1
    if trusted == 0 and noisy:
        print(
            f"::warning::all benches inconclusive (spread > {args.max_rel_variance:.0%}): "
            "loaded box. Re-run on an idle host before trusting deltas."
        )
        return 2
    if noisy:
        print(
            f"::warning::ignored {len(noisy)} noisy bench(es) {noisy} "
            f"(spread > {args.max_rel_variance:.0%}); gated on the {trusted} trustworthy one(s)."
        )
    if rebless_candidates:
        print(
            f"::notice::re-bless candidate(s) {rebless_candidates}: beat champion by "
            f">{args.deadband:.0%}. NOT auto-applied — confirm on a second run, then "
            "commit the champion file from the CI bench.out (whole-file, CI-runner units)."
        )
    print("bench gate: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
