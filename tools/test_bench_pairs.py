#!/usr/bin/env python3
"""Tests for the pipeline pair module. No third-party deps.

Run: ``python3 tools/test_bench_pairs.py``
"""
from __future__ import annotations

import math
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import bench_pairs as bp  # noqa: E402

IDS = (
    "compiler_pipeline/parse_typecheck_ir/large_network",
    "compiler_pipeline/parse_typecheck_ir/medium_mlp",
    "compiler_pipeline/parse_typecheck_ir/small_matmul",
)
BASE_NS = 4_000_000


def report(failures: list[str], label: str, ok: bool, detail: str = "") -> None:
    print(("[PASS] " if ok else "[FAIL] ") + label)
    if not ok and detail:
        print(detail[:800])
    if not ok:
        failures.append(label)


def pipeline_text(ratios: list[float], *, order=None, drop: str = "", extra: str = "") -> str:
    lines = ["# protocol=pipeline", "row\ti\torder\tt_base_ns\tt_cand_ns"]
    for row in IDS:
        if row == drop:
            continue
        for i, ratio in enumerate(ratios):
            token = order(i) if order else ("BC" if i % 2 == 0 else "CB")
            lines.append(f"{row}\t{i}\t{token}\t{BASE_NS}\t{round(BASE_NS * ratio)}")
    if extra:
        lines.append(extra)
    return "\n".join(lines) + "\n"


def grade(text: str) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "pairs.tsv"
        path.write_text(text)
        return subprocess.run(
            [sys.executable, str(HERE / "bench_gate.py"), "--pairs", str(path)],
            capture_output=True,
            text=True,
        )


def main() -> int:
    failures: list[str] = []
    gate_lines = len((HERE / "bench_gate.py").read_text().splitlines())
    preflight_lines = len((ROOT / "scripts" / "preflight.sh").read_text().splitlines())
    pair_py = len((HERE / "bench_pairs.py").read_text().splitlines())
    pair_sh = len((ROOT / "scripts" / "preflight_bench_pairs.sh").read_text().splitlines())
    report(failures, f"bench_gate.py stays within base+10 lines ({gate_lines})", gate_lines <= 769)
    report(failures, f"preflight.sh stays at or under 800 lines ({preflight_lines})", preflight_lines <= 800)
    report(failures, f"bench_pairs.py stays at or under 400 lines ({pair_py})", pair_py <= 400)
    report(failures, f"preflight_bench_pairs.sh stays at or under 400 lines ({pair_sh})", pair_sh <= 400)

    shell = (ROOT / "scripts" / "preflight_bench_pairs.sh").read_text()
    preflight = (ROOT / "scripts" / "preflight.sh").read_text()
    report(failures, "preflight calls the pair script", "scripts/preflight_bench_pairs.sh" in preflight)
    report(failures, "the pair script asks for a core and names a noisy host",
           "pick-core" in shell and "noisy-line" in shell and "exit 2" in shell)
    report(failures, "the pair module reads the SMT sibling list",
           "thread_siblings_list" in (HERE / "bench_pairs.py").read_text())
    report(failures, "the fail bound is still 3 percent",
           math.isclose(math.log(1.03), __import__("bench_gate").PAIR_FAIL_BOUND))
    report(failures, "the noisy width is still 12 percent",
           math.isclose(math.log(1.12), __import__("bench_gate").PAIR_NOISY_WIDTH))

    same = grade(pipeline_text([1.0] * 9))
    slow = grade(pipeline_text([1.10] * 9))
    faster = grade(pipeline_text([0.90] * 9))
    inside = grade(pipeline_text([1.02] * 9))
    fifteen = grade(pipeline_text([1.15] * 9))
    wide = [1.0, 1.0, 1.01, 1.02, 1.03, 1.04, 1.25, 1.26, 1.27]
    noisy = grade(pipeline_text(wide))
    proven = grade(pipeline_text([1.50, 1.50, 1.55, 1.60, 1.70, 1.80, 2.00, 2.10, 2.20]))
    short = grade(pipeline_text([1.0] * 8))
    broken = grade(pipeline_text([1.0] * 9, order=lambda _i: "BC"))
    missing = grade(pipeline_text([1.0] * 9, drop=IDS[0]))
    bare = grade(pipeline_text([1.0] * 9).replace("# protocol=pipeline\n", ""))
    report(failures, "pipeline: equal pairs exit 0", same.returncode == 0, same.stdout + same.stderr)
    report(failures, "pipeline: +10 percent exits 1", slow.returncode == 1, slow.stdout)
    report(failures, "pipeline: 10 percent faster exits 0", faster.returncode == 0, faster.stdout)
    report(failures, "pipeline: +2 percent stays inside the bound", inside.returncode == 0, inside.stdout)
    report(failures, "pipeline: +15 percent exits 1", fifteen.returncode == 1, fifteen.stdout)
    report(failures, "pipeline: wide interval exits 2", noisy.returncode == 2, noisy.stdout)
    report(failures, "pipeline: wide but proven slower exits 1", proven.returncode == 1, proven.stdout)
    report(failures, "pipeline: eight pairs exit 4", short.returncode == 4, short.stdout)
    report(failures, "pipeline: broken ABBA exits 4", broken.returncode == 4, broken.stdout)
    report(failures, "pipeline: a missing row exits 4", missing.returncode == 4, missing.stdout)
    report(failures, "pipeline rows without the protocol header exit 4", bare.returncode == 4, bare.stdout)
    report(failures, "a pass line is printed", "pipeline paired gate: PASS" in same.stdout, same.stdout)
    report(failures, "a noisy pair says inconclusive", "INCONCLUSIVE" in noisy.stdout, noisy.stdout)

    line = bp.noisy_line("23.64")
    cli = subprocess.run(
        [sys.executable, str(HERE / "bench_pairs.py"), "noisy-line", "23.64"],
        capture_output=True, text=True,
    )
    want = "NOISY host load=23.64; no verdict; re-run on a quiet host"
    report(failures, "noisy line matches the required text", line == want and cli.stdout.strip() == want, cli.stdout)

    quiet = bp.choose_core(
        {0: 99.0, 36: 99.0, 1: 100.0, 37: 10.0},
        {0: [0, 36], 36: [0, 36], 1: [1, 37], 37: [1, 37]},
    )
    report(failures, "a quiet SMT pair beats an idle core with a busy sibling",
           quiet is not None and quiet[0] == 0 and quiet[2] == 36 and quiet[4] is True, str(quiet))
    busy = bp.choose_core({1: 100.0, 37: 40.0}, {1: [1, 37], 37: [1, 37]})
    report(failures, "a busy sibling is still a pick and is not marked quiet",
           busy is not None and busy[0] == 1 and busy[4] is False, str(busy))
    alone = bp.choose_core({4: 99.0}, {4: [4]})
    report(failures, "a cpu with no sibling can be quiet",
           alone == (4, 99.0, 4, 99.0, True), str(alone))
    report(failures, "an empty idle map picks nothing", bp.choose_core({}, {}) is None)
    report(failures, "cpu list 0-1,36 expands", bp.parse_cpu_list("0-1,36") == [0, 1, 36])

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        floor = d / "floor.txt"
        floor.write_text("small_matmul:   2.98 µs\nmedium_mlp:   6.93 µs\nlarge_network:   18.43 µs\n")
        pairs = d / "pairs.tsv"
        pairs.write_text(pipeline_text([1.05] * 9))
        host = subprocess.run(
            [sys.executable, str(HERE / "bench_pairs.py"), "host-floor", str(pairs), str(floor)],
            capture_output=True, text=True,
        )
        report(failures, "base above the floor prints the slow-host line",
               host.returncode == 0 and host.stdout.count("HOST-BELOW-FLOOR") == 3
               and "absolute floor is judged by CI" in host.stdout, host.stdout + host.stderr)
        low = d / "low.txt"
        low.write_text("small_matmul:   9000.00 µs\nmedium_mlp:   6.93 µs\nlarge_network:   18.43 µs\n")
        under = subprocess.run(
            [sys.executable, str(HERE / "bench_pairs.py"), "host-floor", str(pairs), str(low)],
            capture_output=True, text=True,
        )
        report(failures, "base under the floor is not a slow-host pass",
               under.returncode == 3 and "BASE_UNDER_FLOOR" in under.stdout, under.stdout)

        out_b = d / "b.out"
        out_c = d / "c.out"
        out_b.write_text("".join(
            f"test {row} ... bench:  4000 ns/iter (+/- 10)\n" for row in IDS
        ))
        out_c.write_text("".join(
            f"test {row} ... bench:  4200 ns/iter (+/- 10)\n" for row in IDS
        ))
        dest = d / "rows.tsv"
        dest.write_text("# protocol=pipeline\nrow\ti\torder\tt_base_ns\tt_cand_ns\n")
        added = subprocess.run(
            [sys.executable, str(HERE / "bench_pairs.py"), "append-rows",
             str(dest), "0", "BC", str(out_b), str(out_c)],
            capture_output=True, text=True,
        )
        text = dest.read_text()
        report(failures, "append-rows writes three pipeline rows",
               added.returncode == 0 and text.count("\t0\tBC\t") == 3
               and "small_matmul base_ns=4000 head_ns=4200" in added.stdout,
               added.stdout + added.stderr + text)

    if failures:
        print(f"\n{len(failures)} bench-pair test(s) FAILED")
        return 1
    print("\nbench-pair contract: all cases OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
