#!/usr/bin/env python3
# Copyright 2025 STARGA Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
"""Pair support for the preflight bench leg.

The 3% fail bound and the 12% width bound stay in bench_gate.py.
This module does not change them. A pipeline file uses those bounds
on each of the three fixtures. Nine pairs is the minimum.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

IDLE_QUIET = 98.0


def parse_cpu_list(text: str) -> list[int]:
    """Expand ``0-3,36`` into cpu ids. A bad token is skipped."""
    out: list[int] = []
    for part in text.replace("\n", " ").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            if left.strip().isdigit() and right.strip().isdigit():
                a, b = int(left), int(right)
                if a <= b:
                    out.extend(range(a, b + 1))
            continue
        if part.isdigit():
            out.append(int(part))
    return out


def choose_core(
    idle: dict[int, float],
    siblings: dict[int, list[int]],
) -> tuple[int, float, int, float, bool] | None:
    """Pick ``(cpu, idle, sibling, sibling_idle, both_quiet)``.

    A core qualifies as quiet only when it and every SMT sibling are at
    or above 98 percent idle. A sibling list that names only this cpu
    is quiet when this cpu is quiet. A quiet pair beats every busy pair.
    When no quiet pair exists, the pair with the highest minimum idle
    is returned and ``both_quiet`` is false. The caller still measures.
    """
    if not idle:
        return None
    ranked: list[tuple[bool, float, float, int, int, int, float]] = []
    for cpu, pct in idle.items():
        others = [s for s in siblings.get(cpu, [cpu]) if s != cpu and s in idle]
        if others:
            sib = min(others, key=lambda s: (idle[s], s))
            sib_pct = idle[sib]
            worst = min([pct, *[idle[s] for s in others]])
            quiet = pct >= IDLE_QUIET and all(idle[s] >= IDLE_QUIET for s in others)
        else:
            sib, sib_pct, worst = cpu, pct, pct
            quiet = pct >= IDLE_QUIET
        ranked.append((quiet, worst, pct, -cpu, cpu, sib, sib_pct))
    ranked.sort(reverse=True)
    quiet, _worst, pct, _neg, cpu, sib, sib_pct = ranked[0]
    return cpu, pct, sib, sib_pct, bool(quiet)


def _sample_ticks() -> dict[int, tuple[int, int]]:
    rows: dict[int, tuple[int, int]] = {}
    for line in Path("/proc/stat").read_text().splitlines():
        if not line.startswith("cpu") or not line[3:4].isdigit():
            continue
        parts = line.split()
        nums = [int(x) for x in parts[1:9]]
        rows[int(parts[0][3:])] = (nums[3] + nums[4], sum(nums))
    return rows


def _allowed_cpus() -> set[int]:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("Cpus_allowed_list:"):
            return set(parse_cpu_list(line.split(":", 1)[1]))
    return set()


def _siblings(cpu: int) -> list[int]:
    path = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
    try:
        found = parse_cpu_list(path.read_text())
    except OSError:
        return [cpu]
    return found or [cpu]


def pick_core_live() -> tuple[int, int]:
    """Print the pick. Exit 0 when a cpu was chosen, else 3."""
    before = _sample_ticks()
    time.sleep(1)
    after = _sample_ticks()
    allow = _allowed_cpus()
    idle: dict[int, float] = {}
    for cpu, (idle_a, total_a) in before.items():
        if allow and cpu not in allow:
            continue
        idle_b, total_b = after.get(cpu, (idle_a, total_a))
        total = total_b - total_a
        idle[cpu] = (100.0 * (idle_b - idle_a) / total) if total else 0.0
    siblings = {cpu: _siblings(cpu) for cpu in idle}
    chosen = choose_core(idle, siblings)
    top = sorted(idle.items(), key=lambda item: (-item[1], item[0]))[:8]
    print("TOP " + " ".join(f"{cpu}:{pct:.1f}" for cpu, pct in top))
    if chosen is None:
        print("PICK")
        return 3
    cpu, pct, sib, sib_pct, quiet = chosen
    print(f"PICK {cpu}")
    print(f"IDLE {pct:.1f}")
    print(f"SIB {sib}")
    print(f"SIB_IDLE {sib_pct:.1f}")
    print(f"QUIET {1 if quiet else 0}")
    return 0


def noisy_line(load: str) -> str:
    """The host could not decide. This is not a pass."""
    return f"NOISY host load={load}; no verdict; re-run on a quiet host"


def _pipeline_ids() -> tuple[str, ...]:
    from bench_gate import PIPELINE_BENCH_IDS

    return tuple(sorted(PIPELINE_BENCH_IDS))


def append_pair_rows(pairs: Path, index: int, order: str, base_out: Path, cand_out: Path) -> int:
    """Append the three pipeline rows. Exit 4 when a row is missing."""
    from bench_gate import BENCHER_LINE, _to_us

    def grab(path: Path) -> dict[str, int]:
        found: dict[str, int] = {}
        for line in path.read_text(errors="replace").splitlines():
            match = BENCHER_LINE.search(line)
            if not match or match.group("id") in found:
                continue
            us = _to_us(float(match.group("value")), match.group("unit"))
            found[match.group("id")] = int(round(us * 1000.0))
        return found

    base, cand = grab(base_out), grab(cand_out)
    want = _pipeline_ids()
    if any(name not in base or name not in cand or base[name] <= 0 or cand[name] <= 0 for name in want):
        return 4
    with pairs.open("a", encoding="utf-8") as handle:
        for name in want:
            handle.write(f"{name}\t{index}\t{order}\t{base[name]}\t{cand[name]}\n")
            if name.endswith("small_matmul"):
                print(
                    f"PAIR_ROW i={index} order={order} small_matmul "
                    f"base_ns={base[name]} head_ns={cand[name]}"
                )
    return 0


def grade_pipeline_pairs(rows: dict[str, list[tuple[int, str, int, int]]]) -> int:
    """Per fixture, the bench_gate pair bounds. Nine pairs bound the median."""
    import math

    from bench_gate import PAIR_FAIL_BOUND, PAIR_NOISY_WIDTH, median_bounds

    minimum, ids = 9, _pipeline_ids()
    errors: list[str] = []
    extra = sorted(set(rows) - set(ids))
    if extra:
        errors.append("unknown row " + ", ".join(extra))
    for row in ids:
        got = rows.get(row) or []
        if len(got) < minimum:
            errors.append(f"pipeline: {row} has {len(got)} pairs, minimum {minimum}")
            continue
        if sorted(i for i, *_ in got) != list(range(len(got))):
            errors.append(f"pipeline: {row} indices are not 0..{len(got) - 1}")
        bad_i = [i for i, order, *_ in got if order != ("BC" if i % 2 == 0 else "CB")]
        if bad_i:
            errors.append(f"pipeline: {row} breaks ABBA at {bad_i[:5]}")
    if errors:
        print("asserted=0")
        print("::error::pipeline paired input does not follow the protocol; nothing is decided:")
        print("\n".join(f"  - {err}" for err in errors))
        return 4
    failed = noisy = False
    for row in ids:
        ds = [math.log(tc / tb) for _, _, tb, tc in rows[row]]
        med, lo, hi = median_bounds(ds)
        verdict = "FAIL" if lo > PAIR_FAIL_BOUND else "NOISY" if hi - lo > PAIR_NOISY_WIDTH else "OK"
        failed, noisy = failed or verdict == "FAIL", noisy or verdict == "NOISY"
        print(
            f"{row} N={len(ds)} R={math.exp(med):.4f} "
            f"L95={math.exp(lo):.4f} U95={math.exp(hi):.4f} {verdict}"
        )
    print(f"asserted={0 if failed or noisy else len(ids)}")
    if failed or noisy:
        print("::error::pipeline paired gate: " + ("FAIL" if failed else "INCONCLUSIVE"))
        return 1 if failed else 2
    print("pipeline paired gate: PASS")
    return 0


def report_host_floor(pairs: Path, floor_file: Path) -> int:
    """Print one slow-host line per fixture. Exit 3 when the base is not above the floor."""
    import statistics

    from bench_gate import parse_pairs, parse_reference

    _meta, rows = parse_pairs(pairs)
    floor = parse_reference(floor_file, prefix="compiler_pipeline/parse_typecheck_ir")
    for row in _pipeline_ids():
        base_us = statistics.median(tb for _i, _o, tb, _c in rows[row]) / 1000.0
        head_us = statistics.median(tc for _i, _o, _b, tc in rows[row]) / 1000.0
        level = floor[row]
        print(row)
        if not base_us > level:
            print(f"BASE_UNDER_FLOOR base={base_us:.3f} floor={level:.3f}")
            return 3
        print(
            "HOST-BELOW-FLOOR "
            f"base={base_us:.3f} head={head_us:.3f} floor={level:.3f}; "
            "absolute floor is judged by CI"
        )
    return 0


def main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else ""
    if cmd == "pick-core":
        return pick_core_live()
    if cmd == "noisy-line" and len(argv) == 3:
        print(noisy_line(argv[2]))
        return 0
    if cmd == "append-rows" and len(argv) == 7:
        return append_pair_rows(Path(argv[2]), int(argv[3]), argv[4], Path(argv[5]), Path(argv[6]))
    if cmd == "host-floor" and len(argv) == 4:
        return report_host_floor(Path(argv[2]), Path(argv[3]))
    if cmd == "grade" and len(argv) == 3:
        from bench_gate import parse_pairs

        _meta, rows = parse_pairs(Path(argv[2]))
        return grade_pipeline_pairs(rows)
    print(
        "usage: bench_pairs.py pick-core | noisy-line LOAD | "
        "append-rows PAIRS I ORDER BASE CAND | host-floor PAIRS FLOOR | grade PAIRS",
        file=sys.stderr,
    )
    return 4


if __name__ == "__main__":
    sys.exit(main(sys.argv))
