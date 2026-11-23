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

``--self-test`` proves the pair contract: the grade bounds, the ABBA order rule, the core
pick, the identical-binary rule and the base the pair script resolves in a temporary
repository. CI runs it through scripts/run_gate.py.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

IDLE_QUIET = 98.0
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


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


def bench_executable(messages: Path, name: str = "compiler") -> str:
    """The path of bench target ``name`` in cargo JSON messages, or ''."""
    import json

    found = ""
    for line in messages.read_text(errors="replace").splitlines():
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        target = msg.get("target") if isinstance(msg, dict) else None
        if (
            msg.get("reason") == "compiler-artifact"
            and isinstance(target, dict)
            and target.get("name") == name
            and "bench" in (target.get("kind") or [])
            and msg.get("executable")
        ):
            found = str(msg["executable"])
    return found


def same_binary(base_exe: Path, head_exe: Path) -> int:
    """Exit 0 when both bench binaries hash equal, 1 when they differ, 4 when one is absent."""
    import hashlib

    digests: list[str] = []
    for path in (base_exe, head_exe):
        if not path.is_file():
            print(f"FAIL: bench binary missing: {path}")
            return 4
        digests.append(hashlib.sha256(path.read_bytes()).hexdigest())
    base_sha, head_sha = digests
    if base_sha == head_sha:
        print(f"IDENTICAL base={base_sha} head={head_sha}; no timing needed")
        return 0
    print(f"DIFFERENT base={base_sha} head={head_sha}; timing the pairs")
    return 1


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


# ---- self-test: every check below runs only under --self-test ----

_ST_IDS = (
    "compiler_pipeline/parse_typecheck_ir/large_network",
    "compiler_pipeline/parse_typecheck_ir/medium_mlp",
    "compiler_pipeline/parse_typecheck_ir/small_matmul",
)
_ST_BASE_NS = 4_000_000


def _st_report(failures: list[str], label: str, ok: bool, detail: str = "") -> None:
    print(("[PASS] " if ok else "[FAIL] ") + label)
    if not ok and detail:
        print(detail[:800])
    if not ok:
        failures.append(label)


def _st_pipeline_text(ratios: list[float], *, order=None, drop: str = "", extra: str = "") -> str:
    lines = ["# protocol=pipeline", "row\ti\torder\tt_base_ns\tt_cand_ns"]
    for row in _ST_IDS:
        if row == drop:
            continue
        for i, ratio in enumerate(ratios):
            token = order(i) if order else ("BC" if i % 2 == 0 else "CB")
            lines.append(f"{row}\t{i}\t{token}\t{_ST_BASE_NS}\t{round(_ST_BASE_NS * ratio)}")
    if extra:
        lines.append(extra)
    return "\n".join(lines) + "\n"


def _st_grade(text: str) -> subprocess.CompletedProcess[str]:
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "pairs.tsv"
        path.write_text(text)
        return subprocess.run(
            [sys.executable, str(HERE / "bench_gate.py"), "--pairs", str(path)],
            capture_output=True,
            text=True,
        )


def _st_pair_env(hint: str | None, stub: Path, log: Path, exe: Path | None = None) -> dict[str, str]:
    """A clean environment: no outer git state, no outer base hint, a stub cargo first on PATH.

    HOME stays as it is: a git installed under HOME needs it to start.
    """
    import os

    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_") and k != "MIND_HYGIENE_BASE"}
    env.update(
        GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1",
        GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@example.invalid",
        GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@example.invalid",
        PAIR_STUB_LOG=str(log), PATH=f"{stub}{os.pathsep}{env.get('PATH', '')}",
    )
    if hint is not None:
        env["MIND_HYGIENE_BASE"] = hint
    if exe is not None:
        env["PAIR_STUB_EXE"] = str(exe)
    return env


def _st_check_pair_base(failures: list[str]) -> None:
    """Run the real pair script in a temporary repository and read which base it resolves.

    The stub cargo fails at once, so a resolved base shows as the "bench pairs base=" line plus
    one stub call, and a miss shows as the named failure with no stub call. With PAIR_STUB_EXE
    set the stub builds one fixed bench binary for both sides, so the identical-binary rule runs.
    """
    import re
    import shutil
    import subprocess
    import tempfile

    script = ROOT / "scripts" / "preflight_bench_pairs.sh"
    with tempfile.TemporaryDirectory() as td:
        top = Path(td)
        repo, stub, log = top / "repo", top / "stub", top / "cargo.log"
        for d in (repo / "scripts", stub):
            d.mkdir(parents=True)
        shutil.copy(script, repo / "scripts" / script.name)
        (repo / "tools").mkdir()
        shutil.copy(HERE / "bench_pairs.py", repo / "tools" / "bench_pairs.py")
        exe = top / "bench.bin"
        exe.write_bytes(b"\x7fELF one bench binary for both sides")
        (stub / "cargo").write_text(
            '#!/bin/sh\necho called >> "$PAIR_STUB_LOG"\n'
            'if [ -n "${PAIR_STUB_EXE:-}" ]; then\n'
            '  printf \'{"reason":"compiler-artifact","target":{"name":"compiler","kind":["bench"]},'
            '"executable":"%s"}\\n\' "$PAIR_STUB_EXE"\n  exit 0\nfi\n'
            'echo "stub cargo: no build here" >&2\nexit 97\n'
        )
        (stub / "cargo").chmod(0o755)
        env = _st_pair_env(None, stub, log)

        def git(*args: str, check: bool = True) -> str:
            done = subprocess.run(["git", "-c", "commit.gpgsign=false", *args], cwd=repo, env=env,
                                  capture_output=True, text=True, check=check)
            return done.stdout.strip()

        git("init", "-q")
        git("symbolic-ref", "HEAD", "refs/heads/work")
        shas = []
        for n in range(3):
            (repo / "f.txt").write_text(f"{n}\n")
            git("add", "-A")
            git("commit", "-q", "-m", f"c{n}")
            shas.append(git("rev-parse", "HEAD"))
        c1, c2, c3 = shas
        git("checkout", "-q", "--detach", c1)
        (repo / "f.txt").write_text("side\n")
        git("add", "-A")
        git("commit", "-q", "-m", "side")
        side = git("rev-parse", "HEAD")
        git("checkout", "-q", "--detach", c3)
        # Every case above uses a candidate that is an ancestor of HEAD, where the merge-base is the
        # candidate itself. The side commit forks from c1, so the merge-base (c1) differs from it.
        _st_report(failures, "pair base: the side commit diverged from HEAD",
               side not in (c1, c2, c3) and git("merge-base", "HEAD", side) == c1)
        orphan = git("commit-tree", git("hash-object", "-w", "-t", "tree", "/dev/null"), "-m", "unrelated")
        ahead = git("commit-tree", f"{c3}^{{tree}}", "-p", c3, "-m", "ahead")
        _st_report(failures, "pair base: the ahead commit is a descendant of HEAD",
               ahead != c3 and git("merge-base", "HEAD", ahead) == c3)

        def run_pair(main: str | None, origin: str | None, hint: str | None, use_exe: bool = False):
            for name, sha in (("refs/heads/main", main), ("refs/remotes/origin/main", origin)):
                if sha:
                    git("update-ref", name, sha)
                elif git("rev-parse", "-q", "--verify", name, check=False):
                    git("update-ref", "-d", name)
            log.unlink(missing_ok=True)
            bench = top / "bench"
            shutil.rmtree(bench, ignore_errors=True)
            bench.mkdir()
            done = subprocess.run(
                ["bash", str(repo / "scripts" / script.name), str(bench), str(top / "head-target"),
                 str(top / "floor.txt")],
                cwd=top, env=_st_pair_env(hint, stub, log, exe if use_exe else None),
                capture_output=True, text=True, timeout=60,
            )
            out = done.stdout + done.stderr
            found = re.search(r"^bench pairs base=([0-9a-f]{40}) head=", out, re.M)
            return done.returncode, out, (found.group(1) if found else None), log.exists()

        miss = "FAIL: no merge-base"
        empty = "pair range is empty: HEAD is on every base candidate"
        cases = [
            ("no main, no origin, no hint: exit 4, every miss named", None, None, None, None,
             ["refs/heads/main not a commit", "refs/remotes/origin/main not a commit",
              "MIND_HYGIENE_BASE unset"]),
            ("no main, hint names a commit: the hint resolves", None, None, c1, c1, []),
            ("no main, hint is not a commit: exit 4, hint named", None, None, "no-such-ref", None,
             ["MIND_HYGIENE_BASE=no-such-ref not a commit"]),
            ("origin/main alone resolves", None, c2, None, c2, []),
            ("origin/main wins over the hint", None, c2, c1, c2, []),
            ("local main wins over origin/main", c1, c2, None, c1, []),
            ("local main wins over origin/main and the hint", c1, c2, c3, c1, []),
            ("a main with no shared history is a named miss", orphan, None, None, None,
             ["refs/heads/main shares no history with HEAD"]),
            ("a main with no shared history falls through to the hint", orphan, None, c1, c1, []),
            ("diverged local main: the base is the merge-base, not main", side, None, None, c1, []),
            ("diverged origin/main: the base is the merge-base, not origin/main", None, side, None, c1, []),
            ("diverged hint: the base is the merge-base, not the hint", None, None, side, c1, []),
            ("diverged local main wins over a nearer origin/main", side, c2, None, c1, []),
        ]
        ran = 0
        for label, main, origin, hint, want, names in cases:
            rc, out, base, stubbed = run_pair(main, origin, hint)
            ran += 1
            if want is None:
                ok = rc == 4 and base is None and not stubbed and miss in out and all(n in out for n in names)
            else:
                ok = rc == 4 and base == want and stubbed and miss not in out and empty not in out
            _st_report(failures, f"pair base: {label}", ok, f"rc={rc} base={base} stub={stubbed}\n{out}")
        _st_report(failures, f"pair base: all {len(cases)} cases ran", ran == len(cases) == 13)

        # A candidate whose merge-base with HEAD is HEAD itself gives no comparison. It is skipped and
        # the next candidate is tried. When every resolved candidate was skipped the range is empty.
        # HEAD is c3 here. Each row says which candidate position holds HEAD or a descendant of it.
        self_cases = [
            ("HEAD is local main, origin/main is the parent: the base is the parent",
             c3, c2, None, c2, False),
            ("HEAD is local main, origin/main is the parent, the hint is older: origin/main wins",
             c3, c2, c1, c2, False),
            ("HEAD is local main and origin/main: the hint is the base",
             c3, c3, c1, c1, False),
            ("HEAD is origin/main and there is no main: the hint is the base",
             None, c3, c1, c1, False),
            ("local main is ahead of HEAD, origin/main is the parent: the base is the parent",
             ahead, c2, None, c2, False),
            ("HEAD is local main and origin/main: the range is empty, the base is HEAD",
             c3, c3, None, c3, True),
            ("HEAD is local main alone: the range is empty",
             c3, None, None, c3, True),
            ("HEAD is the hint alone: the range is empty",
             None, None, c3, c3, True),
            ("HEAD is local main, the hint is not a commit: the range is empty",
             c3, None, "no-such-ref", c3, True),
            ("local main is ahead of HEAD alone: the range is empty",
             ahead, None, None, c3, True),
            ("an unrelated main and HEAD as origin/main: the range is empty",
             orphan, c3, None, c3, True),
        ]
        for label, main, origin, hint, want, is_empty in self_cases:
            rc, out, base, stubbed = run_pair(main, origin, hint)
            ok = (rc == 4 and base == want and stubbed and miss not in out
                  and out.count(empty) == (1 if is_empty else 0))
            _st_report(failures, f"pair self base: {label}", ok, f"rc={rc} base={base} stub={stubbed}\n{out}")
        _st_report(failures, f"pair self base: all {len(self_cases)} cases ran", len(self_cases) == 11)

        # An empty range is decided by the identical-binary rule, never by the range itself.
        rc, out, base, stubbed = run_pair(c3, c3, None, use_exe=True)
        _st_report(failures, "pair self base: an empty range with one binary passes on the identical-binary rule",
               rc == 0 and base == c3 and out.count(empty) == 1 and "IDENTICAL" in out, f"rc={rc}\n{out}")
        rc, out, base, stubbed = run_pair(None, None, None, use_exe=True)
        _st_report(failures, "pair self base: no candidate at all is still exit 4 even with a building cargo",
               rc == 4 and base is None and not stubbed and miss in out and empty not in out, f"rc={rc}\n{out}")
        rc, out, base, stubbed = run_pair(orphan, None, None, use_exe=True)
        _st_report(failures, "pair self base: a lone unrelated main is still exit 4 even with a building cargo",
               rc == 4 and base is None and not stubbed and miss in out and empty not in out, f"rc={rc}\n{out}")


def _code_strings(fn) -> list[str]:
    """The string constants of the code of `fn`, for a self-test row that reads what a fn names."""
    return [c for c in fn.__code__.co_consts if isinstance(c, str)]


def self_test() -> int:
    """Prove the pair contract. Exit 0 when every case holds, else 1.

    Run: ``python3 tools/bench_pairs.py --self-test`` (no third-party deps).
    """
    import hashlib
    import math
    import subprocess
    import tempfile

    from bench_gate import PAIR_FAIL_BOUND, PAIR_NOISY_WIDTH

    failures: list[str] = []
    gate_lines = len((HERE / "bench_gate.py").read_text().splitlines())
    preflight_lines = len((ROOT / "scripts" / "preflight.sh").read_text().splitlines())
    pair_py = len((HERE / "bench_pairs.py").read_text().splitlines())
    pair_sh = len((ROOT / "scripts" / "preflight_bench_pairs.sh").read_text().splitlines())
    _st_report(failures, f"bench_gate.py stays within base+10 lines ({gate_lines})", gate_lines <= 769)
    _st_report(failures, f"preflight.sh stays at or under 800 lines ({preflight_lines})", preflight_lines <= 800)
    _st_report(failures, f"bench_pairs.py stays at or under 800 lines ({pair_py})", pair_py <= 800)
    _st_report(failures, f"preflight_bench_pairs.sh stays at or under 400 lines ({pair_sh})", pair_sh <= 400)

    shell = (ROOT / "scripts" / "preflight_bench_pairs.sh").read_text()
    preflight = (ROOT / "scripts" / "preflight.sh").read_text()
    _st_report(failures, "preflight calls the pair script", "scripts/preflight_bench_pairs.sh" in preflight)
    _st_report(failures, "the pair script asks for a core and names a noisy host",
           "pick-core" in shell and "noisy-line" in shell and "exit 2" in shell)
    _st_report(failures, "the pair module reads the SMT sibling list",
           any("thread_siblings_list" in c for c in _code_strings(_siblings)))
    _st_report(failures, "the fail bound is still 3 percent",
           math.isclose(math.log(1.03), PAIR_FAIL_BOUND))
    _st_report(failures, "the noisy width is still 12 percent",
           math.isclose(math.log(1.12), PAIR_NOISY_WIDTH))

    same = _st_grade(_st_pipeline_text([1.0] * 9))
    slow = _st_grade(_st_pipeline_text([1.10] * 9))
    faster = _st_grade(_st_pipeline_text([0.90] * 9))
    inside = _st_grade(_st_pipeline_text([1.02] * 9))
    fifteen = _st_grade(_st_pipeline_text([1.15] * 9))
    wide = [1.0, 1.0, 1.01, 1.02, 1.03, 1.04, 1.25, 1.26, 1.27]
    noisy = _st_grade(_st_pipeline_text(wide))
    proven = _st_grade(_st_pipeline_text([1.50, 1.50, 1.55, 1.60, 1.70, 1.80, 2.00, 2.10, 2.20]))
    short = _st_grade(_st_pipeline_text([1.0] * 8))
    broken = _st_grade(_st_pipeline_text([1.0] * 9, order=lambda _i: "BC"))
    missing = _st_grade(_st_pipeline_text([1.0] * 9, drop=_ST_IDS[0]))
    bare = _st_grade(_st_pipeline_text([1.0] * 9).replace("# protocol=pipeline\n", ""))
    _st_report(failures, "pipeline: equal pairs exit 0", same.returncode == 0, same.stdout + same.stderr)
    _st_report(failures, "pipeline: +10 percent exits 1", slow.returncode == 1, slow.stdout)
    _st_report(failures, "pipeline: 10 percent faster exits 0", faster.returncode == 0, faster.stdout)
    _st_report(failures, "pipeline: +2 percent stays inside the bound", inside.returncode == 0, inside.stdout)
    _st_report(failures, "pipeline: +15 percent exits 1", fifteen.returncode == 1, fifteen.stdout)
    _st_report(failures, "pipeline: wide interval exits 2", noisy.returncode == 2, noisy.stdout)
    _st_report(failures, "pipeline: wide but proven slower exits 1", proven.returncode == 1, proven.stdout)
    _st_report(failures, "pipeline: eight pairs exit 4", short.returncode == 4, short.stdout)
    _st_report(failures, "pipeline: broken ABBA exits 4", broken.returncode == 4, broken.stdout)
    _st_report(failures, "pipeline: a missing row exits 4", missing.returncode == 4, missing.stdout)
    _st_report(failures, "pipeline rows without the protocol header exit 4", bare.returncode == 4, bare.stdout)
    _st_report(failures, "a pass line is printed", "pipeline paired gate: PASS" in same.stdout, same.stdout)
    _st_report(failures, "a noisy pair says inconclusive", "INCONCLUSIVE" in noisy.stdout, noisy.stdout)

    line = noisy_line("23.64")
    cli = subprocess.run(
        [sys.executable, str(HERE / "bench_pairs.py"), "noisy-line", "23.64"],
        capture_output=True, text=True,
    )
    want = "NOISY host load=23.64; no verdict; re-run on a quiet host"
    _st_report(failures, "noisy line matches the required text", line == want and cli.stdout.strip() == want, cli.stdout)

    quiet = choose_core(
        {0: 99.0, 36: 99.0, 1: 100.0, 37: 10.0},
        {0: [0, 36], 36: [0, 36], 1: [1, 37], 37: [1, 37]},
    )
    _st_report(failures, "a quiet SMT pair beats an idle core with a busy sibling",
           quiet is not None and quiet[0] == 0 and quiet[2] == 36 and quiet[4] is True, str(quiet))
    busy = choose_core({1: 100.0, 37: 40.0}, {1: [1, 37], 37: [1, 37]})
    _st_report(failures, "a busy sibling is still a pick and is not marked quiet",
           busy is not None and busy[0] == 1 and busy[4] is False, str(busy))
    alone = choose_core({4: 99.0}, {4: [4]})
    _st_report(failures, "a cpu with no sibling can be quiet",
           alone == (4, 99.0, 4, 99.0, True), str(alone))
    _st_report(failures, "an empty idle map picks nothing", choose_core({}, {}) is None)
    _st_report(failures, "cpu list 0-1,36 expands", parse_cpu_list("0-1,36") == [0, 1, 36])

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        floor = d / "floor.txt"
        floor.write_text("small_matmul:   2.98 µs\nmedium_mlp:   6.93 µs\nlarge_network:   18.43 µs\n")
        pairs = d / "pairs.tsv"
        pairs.write_text(_st_pipeline_text([1.05] * 9))
        host = subprocess.run(
            [sys.executable, str(HERE / "bench_pairs.py"), "host-floor", str(pairs), str(floor)],
            capture_output=True, text=True,
        )
        _st_report(failures, "base above the floor prints the slow-host line",
               host.returncode == 0 and host.stdout.count("HOST-BELOW-FLOOR") == 3
               and "absolute floor is judged by CI" in host.stdout, host.stdout + host.stderr)
        low = d / "low.txt"
        low.write_text("small_matmul:   9000.00 µs\nmedium_mlp:   6.93 µs\nlarge_network:   18.43 µs\n")
        under = subprocess.run(
            [sys.executable, str(HERE / "bench_pairs.py"), "host-floor", str(pairs), str(low)],
            capture_output=True, text=True,
        )
        _st_report(failures, "base under the floor is not a slow-host pass",
               under.returncode == 3 and "BASE_UNDER_FLOOR" in under.stdout, under.stdout)

        out_b = d / "b.out"
        out_c = d / "c.out"
        out_b.write_text("".join(
            f"test {row} ... bench:  4000 ns/iter (+/- 10)\n" for row in _ST_IDS
        ))
        out_c.write_text("".join(
            f"test {row} ... bench:  4200 ns/iter (+/- 10)\n" for row in _ST_IDS
        ))
        dest = d / "rows.tsv"
        dest.write_text("# protocol=pipeline\nrow\ti\torder\tt_base_ns\tt_cand_ns\n")
        added = subprocess.run(
            [sys.executable, str(HERE / "bench_pairs.py"), "append-rows",
             str(dest), "0", "BC", str(out_b), str(out_c)],
            capture_output=True, text=True,
        )
        text = dest.read_text()
        _st_report(failures, "append-rows writes three pipeline rows",
               added.returncode == 0 and text.count("\t0\tBC\t") == 3
               and "small_matmul base_ns=4000 head_ns=4200" in added.stdout,
               added.stdout + added.stderr + text)

        exe_a, exe_b, exe_c = d / "a.bin", d / "b.bin", d / "c.bin"
        exe_a.write_bytes(b"\x7fELF same bench bytes")
        exe_b.write_bytes(b"\x7fELF same bench bytes")
        exe_c.write_bytes(b"\x7fELF other bench bytes")
        sha_a = hashlib.sha256(exe_a.read_bytes()).hexdigest()
        sha_c = hashlib.sha256(exe_c.read_bytes()).hexdigest()

        def same(base: Path, head: Path) -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                [sys.executable, str(HERE / "bench_pairs.py"), "same-binary", str(base), str(head)],
                capture_output=True, text=True,
            )

        equal, differ, absent = same(exe_a, exe_b), same(exe_a, exe_c), same(exe_a, d / "none.bin")
        want_equal = f"IDENTICAL base={sha_a} head={sha_a}; no timing needed"
        _st_report(failures, "identical bench binaries pass without timing",
               equal.returncode == 0 and equal.stdout.strip() == want_equal, equal.stdout)
        _st_report(failures, "different bench binaries go to the timed pairs",
               differ.returncode == 1 and "IDENTICAL" not in differ.stdout
               and f"base={sha_a} head={sha_c}" in differ.stdout, differ.stdout)
        _st_report(failures, "a missing bench binary is not a pass",
               absent.returncode == 4 and "IDENTICAL" not in absent.stdout, absent.stdout)

        msgs = d / "build.json"
        msgs.write_text(
            "not json\n"
            '{"reason":"compiler-artifact","target":{"name":"mind","kind":["lib"]},"executable":null}\n'
            '{"reason":"compiler-artifact","target":{"name":"compiler","kind":["bench"]},'
            f'"executable":"{exe_a}"}}\n'
            '{"reason":"build-finished","success":true}\n'
        )
        found = subprocess.run(
            [sys.executable, str(HERE / "bench_pairs.py"), "bench-exe", str(msgs)],
            capture_output=True, text=True,
        )
        _st_report(failures, "bench-exe reads the compiler bench path from cargo messages",
               found.returncode == 0 and found.stdout.strip() == str(exe_a), found.stdout)
        msgs.write_text('{"reason":"build-finished","success":true}\n')
        none = subprocess.run(
            [sys.executable, str(HERE / "bench_pairs.py"), "bench-exe", str(msgs)],
            capture_output=True, text=True,
        )
        _st_report(failures, "bench-exe without a bench artifact is not a pass",
               none.returncode == 4 and none.stdout.strip() == "", none.stdout)

    shell_same = shell.index("same-binary")
    _st_report(failures, "the binary compare runs before any timed pair",
           shell_same < shell.index("pick-core") and "--no-run" in shell
           and "exit 0" in shell[shell_same:shell.index("pick-core")])

    def pair_guard_exits(text: str) -> set[int]:
        lines = text.splitlines()
        call_at = next(
            (i for i, line in enumerate(lines) if "scripts/preflight_bench_pairs.sh" in line),
            None,
        )
        if call_at is None:
            return set()
        guard = ""
        for j in range(call_at, -1, -1):
            stripped = lines[j].strip()
            if stripped.startswith("elif ") or stripped.startswith("if "):
                guard = stripped
                break
        found: set[int] = set()
        if "-eq 1" in guard:
            found.add(1)
        if "-eq 2" in guard:
            found.add(2)
        return found

    pair_at = preflight.index("scripts/preflight_bench_pairs.sh")
    hard_at = preflight.index("could not assert a result")
    _st_report(
        failures,
        "bench exit 1 and exit 2 both reach the pair script",
        pair_guard_exits(preflight) == {1, 2},
        str(pair_guard_exits(preflight)),
    )
    _st_report(
        failures,
        "a bench exit other than 1 or 2 stays a hard fail after the pair call",
        hard_at > pair_at,
    )

    _st_check_pair_base(failures)

    if failures:
        print(f"\n{len(failures)} bench-pair test(s) FAILED")
        return 1
    print("\nbench-pair contract: all cases OK")
    return 0


def main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else ""
    if cmd == "--self-test" and len(argv) == 2:
        return self_test()
    if cmd == "pick-core":
        return pick_core_live()
    if cmd == "noisy-line" and len(argv) == 3:
        print(noisy_line(argv[2]))
        return 0
    if cmd == "append-rows" and len(argv) == 7:
        return append_pair_rows(Path(argv[2]), int(argv[3]), argv[4], Path(argv[5]), Path(argv[6]))
    if cmd == "host-floor" and len(argv) == 4:
        return report_host_floor(Path(argv[2]), Path(argv[3]))
    if cmd == "bench-exe" and len(argv) == 3:
        exe = bench_executable(Path(argv[2]))
        print(exe)
        return 0 if exe else 4
    if cmd == "same-binary" and len(argv) == 4:
        return same_binary(Path(argv[2]), Path(argv[3]))
    if cmd == "grade" and len(argv) == 3:
        from bench_gate import parse_pairs

        _meta, rows = parse_pairs(Path(argv[2]))
        return grade_pipeline_pairs(rows)
    print(
        "usage: bench_pairs.py pick-core | noisy-line LOAD | "
        "append-rows PAIRS I ORDER BASE CAND | host-floor PAIRS FLOOR | grade PAIRS | "
        "bench-exe MESSAGES | same-binary BASE_EXE HEAD_EXE | --self-test",
        file=sys.stderr,
    )
    return 4


if __name__ == "__main__":
    sys.exit(main(sys.argv))
