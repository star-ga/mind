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
champion and floor references, so adding `simple_benchmarks` (scalar_math et al.)
to the gate is just adding them to the champion file — no code change here.

The gate also requires a minimum number of trustworthy frontier fixtures. A
partial result with noisy rows is inconclusive and cannot pass as a complete
frontier assertion.

Ordering assertions (--require-faster A,B, repeatable) compare two benches of the
SAME current run: the pair passes only if current(A) < current(B), strictly. This
is how a same-box, same-session head-to-head (e.g. a MIND end-to-end compile
against an external toolchain timed in the same criterion run) is asserted
without a stored cross-machine number. One-sided like everything else here: A
being faster by any margin passes. A pair whose A or B spread exceeds
--max-rel-variance is INCONCLUSIVE (exit 2), never a pass; a pair naming a bench
the run did not produce is malformed input (exit 4). Every asserted pair adds one
to `asserted=`.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

# FAIL-CLOSED input policy: NO hardcoded baseline fallback. If neither a
# champion nor a floor reference is readable, the gate exits non-zero rather
# than substituting a baked-in number — a gate that invents its own baseline
# reports PASS on no evidence.
WATCHED = ("small_matmul", "medium_mlp", "large_network")
# The compiler pipeline's canonical frontier.  Generic benchmark groups remain
# supported by default; pipeline callers opt into this inventory contract with
# ``--require-pipeline``.
PIPELINE_BENCH_IDS = frozenset(
    {
        "compiler_pipeline/parse_typecheck_ir/small_matmul",
        "compiler_pipeline/parse_typecheck_ir/medium_mlp",
        "compiler_pipeline/parse_typecheck_ir/large_network",
    }
)
# Any criterion bench id: "<group>/<sub>/<name>" or "<group>/<name>". We key on
# the full path so same-named fixtures in different groups stay distinct.
# Matches both the two-line
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
#
# The trailing (?!...) is load-bearing, not cosmetic. Baseline files carry TWO
# line shapes that both start "<name>  <float> µs":
#
#   prose comparison:  "small_matmul:   2.80 µs -> 2.98 µs   (+6.4%)"   <- OLD then NEW
#   frozen reference:  "small_matmul:   2.98 µs   (3-run: ...)"          <- the real number
#
# Without the guard this pattern matched the comparison line FIRST (it appears
# earlier in the file), and `out.setdefault` then locked in its LEFT-hand value
# -- the SUPERSEDED pre-milestone number -- while the frozen block below was
# silently discarded as a duplicate key. Measured on
# .bench-baseline-2026-06-01-correctness.txt: the gate enforced 2.80/6.55/17.10
# (the 2026-05-18 numbers the file exists to replace) instead of the documented
# 2.98/6.93/18.43. That errs STRICT -- roughly a 2-4% effective threshold
# instead of the advertised 10% -- so it never produced a false green, but the
# workflow, this baseline file and docs/versioning.md all described a gate that
# was not the one running. A perf gate whose enforced number nobody can read off
# its own reference file is not auditable.
#
# So: reject any line whose value is followed by an arrow (a TRANSITION, i.e. a
# before->after narration), and accept only a standalone measurement.
_ARROW = r"(?:->|\u2192|=>)"
# The name class accepts `/` so a reference may be written as a FULL bench id in
# the human-readable form, not only as a bencher dump. Before this, a champion
# file written the obvious way --
#     compiler_pipeline/parse_typecheck_ir/small_matmul:  2.508 µs
# -- parsed to NOTHING, while the same file with short names parsed to short
# keys that no current measurement (which always carries full ids) could match.
# Either way the gate reported "watched bench produced no measurement", which
# reads as a broken bench rather than an unreadable reference. The champion was
# expressible only as a raw bencher dump, and nothing said so at the point of use.
REF_LINE = re.compile(
    r"^\s*-?\s*(?P<name>[A-Za-z0-9_/]+)\s*[:=]?\s+"
    r"(?P<value>[0-9]+\.[0-9]+)\s*(?:µs|us|microseconds)"
    rf"(?!\s*{_ARROW})"
)


def _to_us(value: float, unit: str) -> float:
    if unit == "ns":
        return value / 1000.0
    if unit in ("µs", "us"):
        return value
    if unit == "ms":
        return value * 1000.0
    return value


def _require_measurement(value: float, label: str, *, allow_zero: bool = False) -> float:
    """Reject forged or malformed timing values before they reach the gate."""
    if not math.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        raise ValueError(f"{label} must be finite and {'non-negative' if allow_zero else 'positive'}")
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
    # short leaf name and receive the explicit prefix above.
    for raw in text.splitlines():
        m = BENCHER_LINE.search(raw)
        if m:
            value = _to_us(float(m.group("value")), m.group("unit"))
            _require_measurement(value, f"reference {m.group('id')}")
            out.setdefault(m.group("id"), value)
    for raw in text.splitlines():
        m = REF_LINE.search(raw)
        if m:
            value = float(m.group("value"))
            _require_measurement(value, f"reference {m.group('name')}")
            key = f"{prefix}/{m.group('name')}" if prefix else m.group("name")
            out.setdefault(key, value)
    return out


def parse_pair(raw: str) -> tuple[str, str]:
    """Parse one ``--require-faster`` value ``"A,B"`` into two full bench ids."""
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"--require-faster expects 'A,B' (two bench ids), got {raw!r}")
    a, b = parts
    if a == b:
        raise ValueError(f"--require-faster compares a bench with itself: {raw!r}")
    return a, b


def parse_current(path: Path) -> dict[str, tuple[float, float | None]]:
    """Read criterion bench output into ``{name: (microseconds, rel_variance)}``.

    ``rel_variance`` is the bencher ``(+/- N)`` spread over the median (``None``
    when unavailable). Keyed on the full benchmark id to prevent collisions
    between fixtures in different benchmark groups.
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
            _require_measurement(us, f"current {name}")
            rel_var: float | None = None
            if m.group("variance") is not None and us > 0:
                vunit = m.group("vunit") or m.group("unit")
                spread = _to_us(float(m.group("variance")), vunit)
                _require_measurement(spread, f"current {name} spread", allow_zero=True)
                rel_var = spread / us
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
                _require_measurement(low, f"current {pending} low")
                _require_measurement(mid, f"current {pending} median")
                _require_measurement(high, f"current {pending} high")
                if not low <= mid <= high:
                    raise ValueError(
                        f"current {pending} interval must satisfy low <= median <= high"
                    )
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
    ap.add_argument(
        "--min-trusted",
        type=int,
        default=3,
        help="minimum trustworthy frontier fixtures required; default 3",
    )
    ap.add_argument(
        "--require-pipeline",
        action="store_true",
        help="require all three canonical compiler_pipeline/parse_typecheck_ir "
        "fixtures in every supplied reference and current run",
    )
    ap.add_argument(
        "--require-faster",
        action="append",
        default=[],
        metavar="A,B",
        help="assert current(A) < current(B) for two benches of this run "
        "(repeatable; one-sided; noisy pair -> inconclusive exit 2).",
    )
    args = ap.parse_args()

    try:
        champion = parse_reference(args.champion)
        # The legacy correctness-milestone floor carries short fixture names that
        # were measured for the compiler_pipeline bench; key them onto that full id
        # so they only gate those benches (never a same-named fixture elsewhere).
        floor = parse_reference(args.floor, prefix=args.floor_prefix)
        current = parse_current(args.current)
        pairs = [parse_pair(raw) for raw in args.require_faster]
    except (OSError, ValueError) as exc:
        print("asserted=0")
        print(f"::error::bench gate input is malformed or unreadable: {exc}")
        return 4
    if args.require_pipeline:
        # A supplied but missing/empty reference is still a failed pipeline
        # contract.  Do this check separately for champion and floor so a
        # valid-looking companion reference cannot mask a partial one.
        for label, supplied, values in (
            ("champion", args.champion is not None, champion),
            ("floor", args.floor is not None, floor),
        ):
            if supplied:
                missing = sorted(PIPELINE_BENCH_IDS - values.keys())
                if missing:
                    raise_value = (
                        f"--require-pipeline {label} reference is missing canonical "
                        f"fixture(s): {', '.join(missing)}"
                    )
                    print("asserted=0")
                    print(f"::error::{raise_value}")
                    return 4
        missing = sorted(PIPELINE_BENCH_IDS - current.keys())
        if missing:
            print("asserted=0")
            print(
                "::error::--require-pipeline current run is missing canonical "
                f"fixture(s): {', '.join(missing)}"
            )
            return 4
    if not champion and not floor:
        # FAIL-CLOSED: neither reference readable. Previously this substituted a
        # hardcoded DEFAULT_BASELINE_US, which meant a missing/renamed/corrupt
        # reference file still produced a green gate measured against a baked-in
        # number nobody re-derived. A perf gate with no reference has nothing to
        # compare against and must say so.
        print("asserted=0")
        print("::error::bench gate has NO reference: neither --champion nor --floor "
              "was readable. Refusing to invent a baseline.")
        return 4
    # Gate every name declared by either reference.  A champion is the primary
    # value when both exist, but it must not hide a floor-only backstop.
    watched = sorted(set(champion) | set(floor))

    rows: list[tuple[str, float | None, float | None, float, float, float | None, str]] = []
    missing: list[str] = []
    failed = False
    trusted = 0
    required_noisy: list[str] = []
    rebless_candidates: list[str] = []
    for name in watched:
        ch = champion.get(name)
        # Floor keys on full ids too (prose short names were prefixed at parse),
        # so a bench only gets a floor when one was actually measured for IT.
        fl = floor.get(name)
        ref = ch if ch is not None else fl  # champion is the primary reference
        cv = current.get(name)
        if ref is None or cv is None:
            # A missing MEASUREMENT is not a warning, it is an absent gate.
            #
            # This fails CLOSED on a missing reference (above) but used to fail OPEN
            # here: every watched name `continue`d, rows stayed empty, trusted==0 with
            # no noisy entries skipped the inconclusive guard, and the run printed
            # "bench gate: PASS" exit 0 having compared NOTHING. Measured 2026-08-29 —
            # an empty --current produced three warnings and a green gate.
            #
            # Both consumers can produce that empty file: preflight.sh redirects the
            # bench output and never tests its exit code, and bench-gate.yml pipes
            # `cargo bench | tee` without pipefail. So a bench that failed to RUN read
            # as a bench that found no regression.
            missing.append(f"{name} (champion={ch}, floor={fl}, current={cv})")
            print(f"::error::missing bench for {name} (champion={ch}, floor={fl}, current={cv})")
            continue
        c, rel_var = cv
        if args.require_pipeline and name in PIPELINE_BENCH_IDS:
            if rel_var is not None and rel_var > args.max_rel_variance:
                required_noisy.append(name)
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

    # Ordering assertions: both sides come from THIS run (same box, same
    # session), so no stored reference is involved and no cross-machine number
    # can leak into the verdict.
    pair_rows: list[tuple[str, str, float, float, str]] = []
    pair_failed: list[str] = []
    pair_missing: list[str] = []
    pair_noisy: list[str] = []
    pairs_asserted = 0
    for a, b in pairs:
        ca = current.get(a)
        cb = current.get(b)
        if ca is None or cb is None:
            pair_missing.append(f"{a} < {b} (current {a}={ca}, {b}={cb})")
            print(f"::error::--require-faster {a},{b}: the run produced no measurement for "
                  f"{a if ca is None else b}")
            continue
        (va, sa), (vb, sb) = ca, cb
        if any(sp is not None and sp > args.max_rel_variance for sp in (sa, sb)):
            verdict = "NOISY"
            pair_noisy.append(f"{a},{b}")
        else:
            pairs_asserted += 1
            if va < vb:
                verdict = "OK"
            else:
                verdict = "FAIL"
                pair_failed.append(f"{a} ({va:.3f}µs) is not faster than {b} ({vb:.3f}µs)")
        pair_rows.append((a, b, va, vb, verdict))

    print()
    hdr = f"{'bench':<16} {'champion':>10} {'floor':>10} {'current':>10} {'Δchamp':>9} {'spread':>7}  gate"
    print(hdr)
    for name, ch, fl, c, delta, rel_var, verdict in rows:
        chs = f"{ch:.3f}" if ch is not None else "   —"
        fls = f"{fl:.3f}" if fl is not None else "   —"
        sp = f"{rel_var:.1%}" if rel_var is not None else " n/a"
        print(f"{name:<16} {chs:>10} {fls:>10} {c:>9.3f}µ {delta:>+8.2%} {sp:>7}  {verdict}")
    print()
    if pair_rows:
        print(f"{'require-faster A < B':<48} {'A':>12} {'B':>12} {'A/B':>7}  gate")
        for a, b, va, vb, verdict in pair_rows:
            print(f"{a + ' < ' + b:<48} {va:>11.3f}µ {vb:>11.3f}µ {va / vb:>7.3f}  {verdict}")
        print()

    noisy = [r[0] for r in rows if r[6] == "NOISY"]
    # Machine-readable count consumed by the shared gate runner. Only complete,
    # low-variance fixture comparisons and low-variance ordering pairs
    # contribute to the assertion count.
    print(f"asserted={trusted + pairs_asserted}")
    if required_noisy:
        print(
            "::error::--require-pipeline canonical fixture(s) are noisy: "
            f"{required_noisy} (spread exceeds {args.max_rel_variance:.0%}); "
            "the pipeline frontier is not asserted even when other benches are trusted."
        )
        return 2
    if pair_failed:
        print("::error::--require-faster ordering violated on a trustworthy measurement:")
        for line in pair_failed:
            print(f"  - {line}")
    if failed:
        print(
            "::error::pipeline regression exceeded "
            f"+{args.threshold:.0%} over the champion or floor on a trustworthy "
            "measurement. STOP and decide: revert, or (if a dramatic win "
            "elsewhere justifies it) re-bless the champion with a documented "
            "rationale + a confirmation run."
        )
        return 1
    if pair_failed:
        return 1
    if pair_missing:
        print(
            f"::error::bench gate FAILED: {len(pair_missing)} --require-faster pair(s) name a "
            "bench the run did not produce; an ordering nobody measured is not asserted."
        )
        for m in pair_missing:
            print(f"  - {m}")
        return 4
    if missing:
        print(
            f"::error::bench gate FAILED: {len(missing)} watched bench(es) produced no "
            "measurement. A gate that compared nothing has not shown the absence of a "
            "regression — it has shown the absence of a bench."
        )
        for m in missing:
            print(f"  - {m}")
        return 4
    if pair_noisy:
        print(
            f"::error::--require-faster pair(s) {pair_noisy} are INCONCLUSIVE (spread > "
            f"{args.max_rel_variance:.0%}); the ordering is not asserted. Re-run on an idle host."
        )
        return 2
    if trusted < args.min_trusted:
        print(
            f"::error::bench gate FAILED: only {trusted} trustworthy frontier "
            f"fixture(s) asserted; minimum is {args.min_trusted}. "
            f"Noisy fixtures exceed the {args.max_rel_variance:.0%} trust bar."
        )
        return 2
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
