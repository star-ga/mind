#!/usr/bin/env python3
"""Regression test for the bench-gate fail-closed contract (tools/bench_gate.py).

Guards the vacuous-PASS hole: an empty / truncated / partial ``bench.out``, or a
missing baseline file, must exit NONZERO (code 4 = malformed gate input) — never
fall through to ``bench gate: PASS``. A uniformly-noisy run still returns its
inconclusive code, not a false green.

Run: ``python3 tools/test_bench_gate.py`` (no third-party deps).
"""
import os
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
    return run_capture(baseline, current).returncode


def run_capture(baseline: Path, current: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(GATE), "--baseline", str(baseline), "--current", str(current)],
        capture_output=True,
        text=True,
    )


def run_args(*argv: str) -> int:
    return subprocess.run(
        [sys.executable, str(GATE), *argv], capture_output=True, text=True
    ).returncode


def bencher(small_ns: int, medium_ns: int, large_ns: int, spread: int = 50) -> str:
    """A bencher-format current run with a spread small enough to be trusted."""
    return (
        f"test compiler_pipeline/parse_typecheck_ir/small_matmul ... bench:  {small_ns} ns/iter (+/- {spread})\n"
        f"test compiler_pipeline/parse_typecheck_ir/medium_mlp ... bench:  {medium_ns} ns/iter (+/- {spread})\n"
        f"test compiler_pipeline/parse_typecheck_ir/large_network ... bench:  {large_ns} ns/iter (+/- {spread})\n"
    )


def main() -> int:
    failures: list[str] = []

    preflight = HERE.parent / "scripts" / "preflight.sh"
    preflight_text = preflight.read_text(encoding="utf-8")
    wiring_cases = [
        ("preflight pins the committed correctness floor",
         "base=.bench-baseline-2026-06-01-correctness.txt" in preflight_text),
        ("preflight records the compiler bench exit status",
         "bench_rc=0" in preflight_text and "|| bench_rc=$?" in preflight_text),
        ("preflight rejects a failed compiler bench",
         'if [ "$bench_rc" -ne 0 ]; then' in preflight_text),
        ("preflight uses a fresh private bench-output directory",
         'mktemp -d "${TMPDIR:-/tmp}/mind-preflight-bench.XXXXXX"' in preflight_text
         and 'bench_out="$bench_tmp/bench.out"' in preflight_text
         and 'bench_target="$bench_tmp/target"' in preflight_text
         and 'CARGO_TARGET_DIR="$bench_target"' in preflight_text
         and "/tmp/preflight-bench.out" not in preflight_text),
        ("preflight rejects missing gate prerequisites",
         'bad "bench gate prerequisites missing:' in preflight_text),
        ("preflight does not select a floor by mtime",
         "ls -t .bench-baseline-*correctness" not in preflight_text),
        ("preflight requires the canonical pipeline inventory",
         "--require-pipeline" in preflight_text),
        ("workflow requires the canonical pipeline inventory",
         "--require-pipeline" in (HERE.parent / ".github" / "workflows" / "bench-gate.yml").read_text()),
        ("bench runner requires the canonical pipeline inventory",
         "--require-pipeline" in (HERE / "run_bench_gate.sh").read_text()),
    ]
    for label, ok in wiring_cases:
        print(f"[{'PASS' if ok else 'FAIL'}] {label}")
        if not ok:
            failures.append(label)

    # Execute only the extracted bench block with a fake cargo that fails. This
    # proves the shell records the command failure and never grades stale output.
    preflight_start = preflight_text.index(
        '  step "bench gate (frozen low-level frontend)'
    )
    preflight_end = preflight_text.index("\nfi\n\necho", preflight_start)
    preflight_block = preflight_text[preflight_start:preflight_end]
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / ".bench-baseline-2026-06-01-correctness.txt").write_text(GOOD_BASELINE)
        (d / "tools").mkdir()
        (d / "tools" / "bench_gate.py").write_text(
            "from pathlib import Path\n"
            "import os\n"
            "Path(os.environ['GATE_MARKER']).write_text('called')\n"
        )
        fake_bin = d / "bin"
        fake_bin.mkdir()
        fake_cargo = fake_bin / "cargo"
        fake_cargo.write_text(
            "#!/bin/sh\n"
            "if [ \"$PREFLIGHT_FAKE_CARGO_RC\" -ne 0 ]; then\n"
            "  echo cargo-failed >&2\n"
            "  exit \"$PREFLIGHT_FAKE_CARGO_RC\"\n"
            "fi\n"
            "exit 0\n"
        )
        fake_cargo.chmod(0o755)
        shell = (
            "set -uo pipefail\nfail=0\nPF_TARGET=target-preflight\n"
            "step() { :; }\nbad() { echo BAD: $*; fail=1; }\n"
            + preflight_block
            + '\nexit "$fail"\n'
        )
        marker = d / "gate-called"
        env = dict(os.environ, PATH=str(fake_bin) + os.pathsep + os.environ["PATH"],
                   GATE_MARKER=str(marker))
        failed_bench = subprocess.run(
            ["bash", "-c", shell], cwd=d,
            env=dict(env, PREFLIGHT_FAKE_CARGO_RC="17"),
            capture_output=True, text=True,
        )
        ok = (failed_bench.returncode != 0
              and "compiler bench command failed" in failed_bench.stdout
              and "bench logs retained in " in failed_bench.stdout
              and not marker.exists())
        print(f"[{'PASS' if ok else 'FAIL'}] failed compiler bench is rejected before comparator")
        if not ok:
            failures.append(f"failed compiler bench shell control: {failed_bench.stdout + failed_bench.stderr}")

        succeeded_bench = subprocess.run(
            ["bash", "-c", shell], cwd=d,
            env=dict(env, PREFLIGHT_FAKE_CARGO_RC="0"),
            capture_output=True, text=True,
        )
        ok = succeeded_bench.returncode == 0 and marker.exists()
        print(f"[{'PASS' if ok else 'FAIL'}] successful compiler bench invokes comparator")
        if not ok:
            failures.append(f"successful compiler bench shell control: {succeeded_bench.stdout + succeeded_bench.stderr}")

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
        one_noisy = d / "one-noisy.out"
        one_noisy.write_text(
            "test compiler_pipeline/parse_typecheck_ir/small_matmul ... bench: 3000 ns/iter (+/- 50)\n"
            "test compiler_pipeline/parse_typecheck_ir/medium_mlp ... bench: 6100 ns/iter (+/- 80)\n"
            "test compiler_pipeline/parse_typecheck_ir/large_network ... bench: 14689 ns/iter (+/- 2157)\n"
        )
        zero_current = d / "zero-current.out"
        zero_current.write_text(bencher(0, 0, 0))
        huge = "9" * 400 + ".0"
        nonfinite_current = d / "nonfinite-current.out"
        nonfinite_current.write_text(
            f"test compiler_pipeline/parse_typecheck_ir/small_matmul ... bench: {huge} ns/iter\n"
            + GOOD_CURRENT.split("\n", 1)[1]
        )
        zero_baseline = d / "zero-baseline.txt"
        zero_baseline.write_text("small_matmul: 0.00 µs\nmedium_mlp: 6.13 µs\nlarge_network: 16.82 µs\n")
        nonfinite_baseline = d / "nonfinite-baseline.txt"
        nonfinite_baseline.write_text(
            f"small_matmul: {huge} µs\nmedium_mlp: 6.13 µs\nlarge_network: 16.82 µs\n"
        )
        reversed_current = d / "reversed-current.out"
        reversed_current.write_text(
            "Benchmarking compiler_pipeline/parse_typecheck_ir/small_matmul:\n"
            "                        time:   [3.0 µs 2.0 µs 1.0 µs]\n"
            + "Benchmarking compiler_pipeline/parse_typecheck_ir/medium_mlp:\n"
            "                        time:   [6.0 µs 6.1 µs 6.2 µs]\n"
            "Benchmarking compiler_pipeline/parse_typecheck_ir/large_network:\n"
            "                        time:   [16.0 µs 16.1 µs 16.2 µs]\n"
        )
        out_of_order_current = d / "out-of-order-current.out"
        out_of_order_current.write_text(
            "Benchmarking compiler_pipeline/parse_typecheck_ir/small_matmul:\n"
            "                        time:   [2.0 µs 3.0 µs 2.5 µs]\n"
            + "Benchmarking compiler_pipeline/parse_typecheck_ir/medium_mlp:\n"
            "                        time:   [6.0 µs 6.1 µs 6.2 µs]\n"
            "Benchmarking compiler_pipeline/parse_typecheck_ir/large_network:\n"
            "                        time:   [16.0 µs 16.1 µs 16.2 µs]\n"
        )

        cases = [
            ("empty current -> exit 4 (the vacuous-PASS hole)", run(base, empty), 4),
            ("partial current 2/3 -> exit 4 (no silent skip)", run(base, partial), 4),
            ("missing baseline -> exit 4 (no default substitution)", run(d / "nope.txt", good), 4),
            ("valid full run -> exit 0 (normal PASS preserved)", run(base, good), 0),
            ("one of three frontier fixtures noisy -> nonzero (no false green)", run(base, one_noisy), 2),
            ("zero current measurement -> exit 4 (no forged speedup)", run(base, zero_current), 4),
            ("non-finite current measurement -> exit 4", run(base, nonfinite_current), 4),
            ("reversed Criterion interval -> exit 4", run(base, reversed_current), 4),
            ("out-of-order Criterion interval -> exit 4", run(base, out_of_order_current), 4),
            ("zero reference measurement -> exit 4", run(zero_baseline, good), 4),
            ("non-finite reference measurement -> exit 4", run(nonfinite_baseline, good), 4),
        ]
        for label, output, want in (
            ("full run reports asserted=3", run_capture(base, good).stdout, "asserted=3"),
            ("noisy fixture run reports asserted=2", run_capture(base, one_noisy).stdout, "asserted=2"),
        ):
            ok = want in output
            print(f"[{'PASS' if ok else 'FAIL'}] {label}")
            if not ok:
                failures.append(label)
        # --- THE ENFORCEMENT PATH ------------------------------------------
        # Every case above exercises a REFUSAL to measure (exit 4) or the happy
        # path. Not one drove the gate to its actual FAIL verdict, so the only
        # branch that can stop a regression from landing had no test at all.
        # A gate whose enforcement is untested is a gate nobody has watched fail.
        regressed = d / "regressed.out"
        regressed.write_text(bencher(3600, 7400, 20200))   # ~+20% on every bench
        within = d / "within.out"
        within.write_text(bencher(3120, 6380, 17480))      # ~+4%, under the 10% default
        faster = d / "faster.out"
        faster.write_text(bencher(2700, 5500, 15100))      # a win: one-sided, must pass

        # Generic benchmark groups remain supported unless a caller explicitly
        # requests the compiler-pipeline inventory contract.
        generic_base = d / "generic-baseline.txt"
        generic_base.write_text(
            "test simple_benchmarks/scalar_math ... bench: 3000 ns/iter\n"
            "test simple_benchmarks/tensor_ops ... bench: 6100 ns/iter\n"
            "test simple_benchmarks/io_roundtrip ... bench: 16800 ns/iter\n"
        )
        generic_current = d / "generic-current.out"
        generic_current.write_text(
            "test simple_benchmarks/scalar_math ... bench: 3000 ns/iter (+/- 50)\n"
            "test simple_benchmarks/tensor_ops ... bench: 6100 ns/iter (+/- 80)\n"
            "test simple_benchmarks/io_roundtrip ... bench: 16800 ns/iter (+/- 120)\n"
        )
        cases += [
            ("generic benchmark group passes without pipeline flag",
             run_args("--baseline", str(generic_base), "--current", str(generic_current)), 0),
        ]
        # ALL three above the variance threshold: a uniformly loaded box.
        all_noisy = d / "all_noisy.out"
        all_noisy.write_text(
            "test compiler_pipeline/parse_typecheck_ir/small_matmul ... bench:  3600 ns/iter (+/- 900)\n"
            "test compiler_pipeline/parse_typecheck_ir/medium_mlp ... bench:  7400 ns/iter (+/- 1850)\n"
            "test compiler_pipeline/parse_typecheck_ir/large_network ... bench:  20200 ns/iter (+/- 5050)\n"
        )
        # MIXED: two benches too noisy to trust, one clean and regressed. The
        # first draft of this test assumed a single spread made all three noisy;
        # it did not (900/20200 is 4.5%), the gate gated on the one trustworthy
        # bench, and the test caught its own wrong expectation rather than the
        # gate's behaviour. Both shapes are asserted now, because "ignored the
        # noise, enforced on what it could trust" is the behaviour that matters
        # on a shared machine.
        mixed_noisy = d / "mixed_noisy.out"
        mixed_noisy.write_text(bencher(3600, 7400, 20200, spread=900))

        cases += [
            ("+20% regression -> exit 1 (THE enforcement path)",
             run_args("--baseline", str(base), "--current", str(regressed)), 1),
            ("+4% under threshold -> exit 0",
             run_args("--baseline", str(base), "--current", str(within)), 0),
            ("faster than baseline -> exit 0 (gate is one-sided)",
             run_args("--baseline", str(base), "--current", str(faster)), 0),
            ("+20% with EVERY bench over the spread limit -> exit 2 (loaded box is "
             "INCONCLUSIVE, not a regression)",
             run_args("--baseline", str(base), "--current", str(all_noisy)), 2),
            ("+20% with 2 noisy and 1 trustworthy -> exit 1 (enforce on what can be trusted)",
             run_args("--baseline", str(base), "--current", str(mixed_noisy)), 1),
        ]

        # --- THE CHAMPION RATCHET ------------------------------------------
        # The champion is the whole point of the tool -- it catches slow drift
        # that never trips the floor: ten commits at +3% each pass any
        # stale-floor check and lose 30% together. `--champion` appeared ZERO
        # times across .github/workflows, so the ratchet the tool advertises had
        # never run, and nothing here tested it either.
        #
        # This case is exactly that shape: current EQUALS the floor (so the floor
        # is satisfied and would pass) while sitting ~20% above the champion.
        champ = d / "champion.txt"
        champ.write_text(
            "compiler_pipeline/parse_typecheck_ir/small_matmul:   2.508 µs\n"
            "compiler_pipeline/parse_typecheck_ir/medium_mlp:     5.222 µs\n"
            "compiler_pipeline/parse_typecheck_ir/large_network: 12.231 µs\n"
        )
        champbeat = d / "champbeat.out"
        champbeat.write_text(bencher(2400, 5000, 11800))

        import importlib.util as _ilu2
        _s2 = _ilu2.spec_from_file_location("bench_gate_ref", GATE)
        _bg2 = _ilu2.module_from_spec(_s2)
        _s2.loader.exec_module(_bg2)
        champ_keys = len(_bg2.parse_reference(champ))

        ratchet = ["--champion", str(champ), "--floor", str(base),
                   "--floor-prefix", "compiler_pipeline/parse_typecheck_ir"]
        cases += [
            ("champion in full-id prose parses to 3 keys (it parsed to 0 before)",
             champ_keys, 3),
            ("at the floor but +20% over champion -> exit 1 (the ratchet, not the floor)",
             run_args(*ratchet, "--current", str(good)), 1),
            ("beats the champion -> exit 0 (the ratchet is one-sided too)",
             run_args(*ratchet, "--current", str(champbeat)), 0),
        ]

        # The opt-in pipeline contract rejects a wrong-group or partial
        # reference, even when its rows would otherwise meet min-trusted.
        wrong_group = d / "wrong-group.txt"
        wrong_group.write_text(
            "test simple_benchmarks/scalar_math ... bench: 3000 ns/iter\n"
            "test simple_benchmarks/tensor_ops ... bench: 6100 ns/iter\n"
            "test simple_benchmarks/io_roundtrip ... bench: 16800 ns/iter\n"
        )
        partial_champion = d / "partial-champion.txt"
        partial_champion.write_text(
            "test compiler_pipeline/parse_typecheck_ir/small_matmul ... bench: 3000 ns/iter\n"
            "test compiler_pipeline/parse_typecheck_ir/medium_mlp ... bench: 6100 ns/iter\n"
        )
        cases += [
            ("pipeline flag rejects wrong-group reference",
             run_args("--baseline", str(wrong_group), "--current", str(good),
                       "--require-pipeline"), 4),
            ("pipeline flag rejects partial champion reference",
             run_args("--champion", str(partial_champion), "--current", str(good),
                       "--require-pipeline"), 4),
        ]

        # Three clean non-pipeline benches must not make a noisy canonical
        # frontier pass when the pipeline contract is requested.
        pipeline_extra_ref = d / "pipeline-extra-ref.txt"
        pipeline_extra_ref.write_text(
            "test compiler_pipeline/parse_typecheck_ir/small_matmul ... bench: 3000 ns/iter\n"
            "test compiler_pipeline/parse_typecheck_ir/medium_mlp ... bench: 6100 ns/iter\n"
            "test compiler_pipeline/parse_typecheck_ir/large_network ... bench: 16800 ns/iter\n"
            "test simple_benchmarks/scalar_math ... bench: 3000 ns/iter\n"
            "test simple_benchmarks/tensor_ops ... bench: 6100 ns/iter\n"
            "test simple_benchmarks/io_roundtrip ... bench: 16800 ns/iter\n"
        )
        canonical_noisy_extra_clean = d / "canonical-noisy-extra-clean.out"
        canonical_noisy_extra_clean.write_text(
            "test compiler_pipeline/parse_typecheck_ir/small_matmul ... bench: 3000 ns/iter (+/- 900)\n"
            "test compiler_pipeline/parse_typecheck_ir/medium_mlp ... bench: 6100 ns/iter (+/- 900)\n"
            "test compiler_pipeline/parse_typecheck_ir/large_network ... bench: 16800 ns/iter (+/- 900)\n"
            "test simple_benchmarks/scalar_math ... bench: 3000 ns/iter (+/- 50)\n"
            "test simple_benchmarks/tensor_ops ... bench: 6100 ns/iter (+/- 50)\n"
            "test simple_benchmarks/io_roundtrip ... bench: 16800 ns/iter (+/- 50)\n"
        )
        cases += [
            ("pipeline flag rejects noisy canonical rows despite 3 clean extras",
             run_args("--baseline", str(pipeline_extra_ref),
                       "--current", str(canonical_noisy_extra_clean),
                       "--require-pipeline"), 2),
        ]

        # Both references contribute watched rows.  A floor-only regression and
        # a champion-only regression must each reach the comparator.
        union_champion = d / "union-champion.txt"
        union_champion.write_text(
            "test compiler_pipeline/parse_typecheck_ir/small_matmul ... bench: 3000 ns/iter\n"
            "test compiler_pipeline/parse_typecheck_ir/medium_mlp ... bench: 6100 ns/iter\n"
            "test compiler_pipeline/parse_typecheck_ir/large_network ... bench: 16800 ns/iter\n"
            "test extra_group/champion_only ... bench: 3000 ns/iter\n"
        )
        union_floor = d / "union-floor.txt"
        union_floor.write_text(
            "test compiler_pipeline/parse_typecheck_ir/small_matmul ... bench: 3000 ns/iter\n"
            "test compiler_pipeline/parse_typecheck_ir/medium_mlp ... bench: 6100 ns/iter\n"
            "test compiler_pipeline/parse_typecheck_ir/large_network ... bench: 16800 ns/iter\n"
            "test extra_group/floor_only ... bench: 3000 ns/iter\n"
        )
        union_floor_regressed = d / "union-floor-regressed.out"
        union_floor_regressed.write_text(
            GOOD_CURRENT
            + "test extra_group/champion_only ... bench: 3000 ns/iter (+/- 50)\n"
            + "test extra_group/floor_only ... bench: 4000 ns/iter (+/- 50)\n"
        )
        union_champion_regressed = d / "union-champion-regressed.out"
        union_champion_regressed.write_text(
            GOOD_CURRENT
            + "test extra_group/champion_only ... bench: 4000 ns/iter (+/- 50)\n"
            + "test extra_group/floor_only ... bench: 3000 ns/iter (+/- 50)\n"
        )
        union_args = ("--champion", str(union_champion), "--floor", str(union_floor),
                      "--require-pipeline")
        cases += [
            ("union watches floor-only backstop",
             run_args(*union_args, "--current", str(union_floor_regressed)), 1),
            ("union watches champion-only backstop",
             run_args(*union_args, "--current", str(union_champion_regressed)), 1),
        ]
        # --- transition-line parsing (the superseded-number bug) ------------
        # A real baseline file narrates the re-baseline in prose BEFORE stating
        # the frozen numbers:
        #
        #   small_matmul:   2.80 µs -> 2.98 µs   (+6.4%)   <- OLD -> NEW prose
        #   ...
        #   small_matmul:   2.98 µs   (3-run: ...)          <- the frozen number
        #
        # REF_LINE used to match the prose line first and setdefault() locked in
        # its LEFT-hand (superseded) value, so the gate silently enforced the
        # numbers the file exists to REPLACE. Assert the frozen block wins.
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location(
            "bench_gate", Path(__file__).resolve().parent / "bench_gate.py"
        )
        _bg = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_bg)

        rebaselined = d / "rebaselined.txt"
        rebaselined.write_text(
            "Net effect vs the prior baseline:\n"
            "\n"
            "  small_matmul:   2.80 µs \u2192 2.98 µs   (+6.4%)\n"
            "  medium_mlp:     6.55 µs \u2192 6.93 µs   (+5.8%)\n"
            "  large_network: 17.10 µs \u2192 18.43 µs  (+7.8%)\n"
            "\n"
            "Frozen reference measurement:\n"
            "\n"
            "  small_matmul:   2.98 µs   (3-run: 2.93 / 2.98 / 3.16)\n"
            "  medium_mlp:     6.93 µs   (3-run: 6.80 / 6.93 / 7.28)\n"
            "  large_network:  18.43 µs  (3-run: 17.95 / 18.43 / 19.52)\n"
        )
        parsed = _bg.parse_reference(rebaselined)
        want_frozen = {"small_matmul": 2.98, "medium_mlp": 6.93, "large_network": 18.43}
        for name, want_v in want_frozen.items():
            got_v = parsed.get(name)
            ok = got_v == want_v
            label = f"frozen block wins over prose transition line ({name})"
            print(f"[{'PASS' if ok else 'FAIL'}] {label} (got {got_v}, want {want_v})")
            if not ok:
                failures.append(label)

        # And the same check against the REAL committed baseline the CI gate
        # uses, so a future edit to that file cannot reintroduce the bug.
        real = Path(__file__).resolve().parent.parent / ".bench-baseline-2026-06-01-correctness.txt"
        if real.exists():
            rp = _bg.parse_reference(real, prefix="compiler_pipeline/parse_typecheck_ir")
            for name, want_v in want_frozen.items():
                key = f"compiler_pipeline/parse_typecheck_ir/{name}"
                got_v = rp.get(key)
                ok = got_v == want_v
                label = f"committed baseline parses to frozen value ({name})"
                print(f"[{'PASS' if ok else 'FAIL'}] {label} (got {got_v}, want {want_v})")
                if not ok:
                    failures.append(label)

        # --- ORDERING ASSERTIONS (--require-faster A,B) ---------------------
        # Same-run head-to-head: current(A) < current(B), strictly, one-sided,
        # with the same >12% spread inconclusive band as every other row.
        e2e_rows = (
            "test e2e_native/add ... bench: 60000000 ns/iter (+/- 900000)\n"
            "test e2e_reference/rustc_cdylib/add ... bench: 90000000 ns/iter (+/- 900000)\n"
            "test e2e_reference/clang_c_to_so/add ... bench: 100000000 ns/iter (+/- 900000)\n"
        )
        rf_faster = d / "rf-faster.out"
        rf_faster.write_text(GOOD_CURRENT + e2e_rows)
        rf_slower = d / "rf-slower.out"
        rf_slower.write_text(GOOD_CURRENT + e2e_rows.replace(
            "e2e_native/add ... bench: 60000000", "e2e_native/add ... bench: 520000000"))
        rf_equal = d / "rf-equal.out"
        rf_equal.write_text(GOOD_CURRENT + e2e_rows.replace(
            "e2e_native/add ... bench: 60000000", "e2e_native/add ... bench: 90000000"))
        rf_noisy = d / "rf-noisy.out"
        # 20 ms spread on a 90 ms median = 22% > the 12% default band.
        rf_noisy.write_text(GOOD_CURRENT + e2e_rows.replace(
            "rustc_cdylib/add ... bench: 90000000 ns/iter (+/- 900000)",
            "rustc_cdylib/add ... bench: 90000000 ns/iter (+/- 20000000)"))
        pair_rustc = ("--require-faster", "e2e_native/add,e2e_reference/rustc_cdylib/add")
        pair_clang = ("--require-faster", "e2e_native/add,e2e_reference/clang_c_to_so/add")
        rf_base = ("--baseline", str(base))

        def rf_run(current: Path, *extra: str) -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                [sys.executable, str(GATE), *rf_base, "--current", str(current), *extra],
                capture_output=True, text=True,
            )

        rf_pass = rf_run(rf_faster, *pair_rustc, *pair_clang)
        rf_fail = rf_run(rf_slower, *pair_rustc, *pair_clang)
        cases += [
            ("require-faster A<B on both pairs -> exit 0 (positive control)",
             rf_pass.returncode, 0),
            ("require-faster A>B -> exit 1 (the ordering enforcement path)",
             rf_fail.returncode, 1),
            ("require-faster A==B -> exit 1 (strict: a tie is not faster)",
             rf_run(rf_equal, *pair_rustc).returncode, 1),
            ("require-faster pair over the spread band -> exit 2 (inconclusive, not a pass)",
             rf_run(rf_noisy, *pair_rustc).returncode, 2),
            ("require-faster naming an unmeasured bench -> exit 4 (fail closed)",
             rf_run(good, *pair_rustc).returncode, 4),
            ("require-faster malformed single id -> exit 4",
             rf_run(rf_faster, "--require-faster", "e2e_native/add").returncode, 4),
            ("require-faster bench against itself -> exit 4",
             rf_run(rf_faster, "--require-faster", "e2e_native/add,e2e_native/add").returncode, 4),
            ("without --require-faster the slower pair is recorded, not asserted -> exit 0",
             rf_run(rf_slower).returncode, 0),
        ]
        for label, output, want in (
            ("asserted counts 3 fixtures + 2 ordering pairs", rf_pass.stdout, "asserted=5"),
            ("a violated pair still counts as asserted", rf_fail.stdout, "asserted=5"),
            ("noisy pair is not counted as asserted", rf_run(rf_noisy, *pair_rustc).stdout,
             "asserted=3"),
            ("violated pair is named in the verdict", rf_fail.stdout,
             "e2e_native/add (520000.000µs) is not faster than e2e_reference/rustc_cdylib/add"),
        ):
            ok = want in output
            print(f"[{'PASS' if ok else 'FAIL'}] {label}")
            if not ok:
                failures.append(f"{label}: {output}")

        # --- THE COMMITTED END-TO-END FLOOR --------------------------------
        # Full-id floor read with --floor-prefix ''. Every row must parse, and
        # every row must be millisecond-class: an e2e floor that parsed to
        # microseconds would be a unit slip that makes the gate unfailable.
        e2e_floors = sorted((HERE.parent).glob(".bench-baseline-*-e2e.txt"))
        ok = len(e2e_floors) >= 1
        print(f"[{'PASS' if ok else 'FAIL'}] an end-to-end floor file is committed")
        if not ok:
            failures.append("no .bench-baseline-*-e2e.txt floor committed")
        for floor_path in e2e_floors:
            fp = _bg.parse_reference(floor_path, prefix="")
            want_ids = {
                *(f"e2e_compile_shared/{c}/{s}" for c in ("warm", "cold")
                  for s in ("small", "medium", "large")),
                *(f"e2e_native/{s}" for s in ("const_return", "add", "local_call", "while_sum")),
            }
            missing_ids = sorted(want_ids - fp.keys())
            ok = not missing_ids
            print(f"[{'PASS' if ok else 'FAIL'}] {floor_path.name} carries every gated e2e id")
            if not ok:
                failures.append(f"{floor_path.name} missing {missing_ids}")
            bad_units = sorted(k for k, v in fp.items() if not 10_000.0 <= v <= 60_000_000.0)
            ok = not bad_units and bool(fp)
            print(f"[{'PASS' if ok else 'FAIL'}] {floor_path.name} rows are millisecond-class")
            if not ok:
                failures.append(f"{floor_path.name} rows outside 10 ms..60 s: {bad_units}")
            stray = sorted(k for k in fp if not k.startswith(("e2e_compile_shared/", "e2e_native/")))
            ok = not stray
            print(f"[{'PASS' if ok else 'FAIL'}] {floor_path.name} floors no reference toolchain")
            if not ok:
                failures.append(f"{floor_path.name} floors external toolchain rows: {stray}")

        wf_text = (HERE.parent / ".github" / "workflows" / "bench-gate.yml").read_text()
        e2e_job = wf_text[wf_text.find("\n  e2e-gate:"):] if "\n  e2e-gate:" in wf_text else ""
        for label, ok in (
            ("workflow has an e2e-gate job", bool(e2e_job)),
            ("e2e-gate runs the e2e_ benches under pipefail",
             "set -eo pipefail" in e2e_job and "cargo bench --bench compiler" in e2e_job
             and "-- e2e_" in e2e_job),
            ("e2e-gate grades against the committed e2e floor with full ids",
             any(f.name in e2e_job for f in e2e_floors) and "--floor-prefix ''" in e2e_job),
            ("e2e-gate does not touch the frontend floor",
             ".bench-baseline-2026-06-01-correctness.txt" not in e2e_job),
        ):
            print(f"[{'PASS' if ok else 'FAIL'}] {label}")
            if not ok:
                failures.append(label)

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
