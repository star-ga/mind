#!/usr/bin/env python3
"""Tests for the pipeline pair module. No third-party deps.

Run: ``python3 tools/test_bench_pairs.py``
"""
from __future__ import annotations

import math
import os
import re
import shutil
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


def pair_env(hint: str | None, stub: Path, log: Path) -> dict[str, str]:
    """A clean environment: no outer git state, no outer base hint, a stub cargo first on PATH.

    HOME stays as it is: a git installed under HOME needs it to start.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_") and k != "MIND_HYGIENE_BASE"}
    env.update(
        GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1",
        GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@example.invalid",
        GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@example.invalid",
        PAIR_STUB_LOG=str(log), PATH=f"{stub}{os.pathsep}{env.get('PATH', '')}",
    )
    if hint is not None:
        env["MIND_HYGIENE_BASE"] = hint
    return env


def check_pair_base(failures: list[str]) -> None:
    """Run the real pair script in a temporary repository and read which base it resolves.

    The stub cargo fails at once, so a resolved base shows as the "bench pairs base=" line plus
    one stub call, and a miss shows as the named failure with no stub call.
    """
    script = ROOT / "scripts" / "preflight_bench_pairs.sh"
    with tempfile.TemporaryDirectory() as td:
        top = Path(td)
        repo, stub, log = top / "repo", top / "stub", top / "cargo.log"
        for d in (repo / "scripts", stub):
            d.mkdir(parents=True)
        shutil.copy(script, repo / "scripts" / script.name)
        (stub / "cargo").write_text(
            '#!/bin/sh\necho called >> "$PAIR_STUB_LOG"\necho "stub cargo: no build here" >&2\nexit 97\n'
        )
        (stub / "cargo").chmod(0o755)
        env = pair_env(None, stub, log)

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
        git("checkout", "-q", "--detach")
        orphan = git("commit-tree", git("hash-object", "-w", "-t", "tree", "/dev/null"), "-m", "unrelated")

        def run_pair(main: str | None, origin: str | None, hint: str | None):
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
                cwd=top, env=pair_env(hint, stub, log), capture_output=True, text=True, timeout=60,
            )
            out = done.stdout + done.stderr
            found = re.search(r"^bench pairs base=([0-9a-f]{40}) head=", out, re.M)
            return done.returncode, out, (found.group(1) if found else None), log.exists()

        miss = "FAIL: no merge-base"
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
        ]
        ran = 0
        for label, main, origin, hint, want, names in cases:
            rc, out, base, stubbed = run_pair(main, origin, hint)
            ran += 1
            if want is None:
                ok = rc == 4 and base is None and not stubbed and miss in out and all(n in out for n in names)
            else:
                ok = rc == 4 and base == want and stubbed and miss not in out
            report(failures, f"pair base: {label}", ok, f"rc={rc} base={base} stub={stubbed}\n{out}")
        report(failures, f"pair base: all {len(cases)} cases ran", ran == len(cases) == 9)


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

        exe_a, exe_b, exe_c = d / "a.bin", d / "b.bin", d / "c.bin"
        exe_a.write_bytes(b"\x7fELF same bench bytes")
        exe_b.write_bytes(b"\x7fELF same bench bytes")
        exe_c.write_bytes(b"\x7fELF other bench bytes")
        sha_a = __import__("hashlib").sha256(exe_a.read_bytes()).hexdigest()
        sha_c = __import__("hashlib").sha256(exe_c.read_bytes()).hexdigest()

        def same(base: Path, head: Path) -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                [sys.executable, str(HERE / "bench_pairs.py"), "same-binary", str(base), str(head)],
                capture_output=True, text=True,
            )

        equal, differ, absent = same(exe_a, exe_b), same(exe_a, exe_c), same(exe_a, d / "none.bin")
        want_equal = f"IDENTICAL base={sha_a} head={sha_a}; no timing needed"
        report(failures, "identical bench binaries pass without timing",
               equal.returncode == 0 and equal.stdout.strip() == want_equal, equal.stdout)
        report(failures, "different bench binaries go to the timed pairs",
               differ.returncode == 1 and "IDENTICAL" not in differ.stdout
               and f"base={sha_a} head={sha_c}" in differ.stdout, differ.stdout)
        report(failures, "a missing bench binary is not a pass",
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
        report(failures, "bench-exe reads the compiler bench path from cargo messages",
               found.returncode == 0 and found.stdout.strip() == str(exe_a), found.stdout)
        msgs.write_text('{"reason":"build-finished","success":true}\n')
        none = subprocess.run(
            [sys.executable, str(HERE / "bench_pairs.py"), "bench-exe", str(msgs)],
            capture_output=True, text=True,
        )
        report(failures, "bench-exe without a bench artifact is not a pass",
               none.returncode == 4 and none.stdout.strip() == "", none.stdout)

    shell_same = shell.index("same-binary")
    report(failures, "the binary compare runs before any timed pair",
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
    report(
        failures,
        "bench exit 1 and exit 2 both reach the pair script",
        pair_guard_exits(preflight) == {1, 2},
        str(pair_guard_exits(preflight)),
    )
    report(
        failures,
        "a bench exit other than 1 or 2 stays a hard fail after the pair call",
        hard_at > pair_at,
    )

    check_pair_base(failures)

    if failures:
        print(f"\n{len(failures)} bench-pair test(s) FAILED")
        return 1
    print("\nbench-pair contract: all cases OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
