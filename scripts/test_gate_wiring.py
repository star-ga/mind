#!/usr/bin/env python3
"""Regression test for scripts/check_gate_wiring.py.

Guards the hole that made the attribution gates advisory-only: the gates that
keep a named-model credit out of a PUBLIC repo (file contents AND commit
messages) all live in .github/workflows/docs-claims.yml, and that workflow had
no row in .github/required-ci-jobs.tsv.  The manifest mirrors ci.yml, so the
release verifier only ever asked "is docs-claims not RED" -- and a workflow that
never ran is not red.  Deleting the commit-message step, or the manifest row
itself, therefore weakened the release contract while every gate stayed green.

Each case below MUTATES a scratch copy of the wiring and asserts the lint goes
RED.  A mutation that does not turn the lint red is a lint that is not gating
that property, so the assertion is on the exit code, never on the prose.

Run: ``python3 scripts/test_gate_wiring.py`` (no third-party deps).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINT = ROOT / "scripts" / "check_gate_wiring.py"


def lint_constant(name: str) -> int:
    """Read an integer constant OFF the lint source.

    Not re-typed here and not imported: a second copy of the number is the drift
    this repo keeps paying for, and importing the module would compile it into a
    __pycache__ the repo does not track.
    """
    # Permit an accurate inline rationale while still requiring the executable
    # assignment to remain a top-level decimal literal.  The gate's ceiling
    # carries such a rationale; rejecting it here makes the mutation proof
    # vacuous before any wiring case can run.
    m = re.search(rf"^{name}\s*=\s*(\d+)(?:\s*#.*)?$",
                  LINT.read_text(encoding="utf-8"), re.MULTILINE)
    if not m:
        raise SystemExit(f"test_gate_wiring: {name} not found in {LINT.name} "
                         "-- the constant moved and this test would go vacuous")
    return int(m.group(1))


HARNESS_CEILING = lint_constant("HARNESS_CEILING")

# Everything the lint reads. Copied into a scratch git repo so a mutation can be
# applied without touching the working tree.
COPIED = (
    ".github/required-ci-jobs.tsv",
    ".github/workflows/docs-claims.yml",
    ".github/workflows/ci.yml",
    ".githooks/pre-commit",
    ".githooks/commit-msg",
    ".githooks/post-commit",
    ".githooks/post-merge",
    "scripts/check_no_ai_attribution.sh",
    "scripts/check_json_not_evidence.sh",
    "scripts/check_commit_messages.sh",
    "scripts/commit-msg-hook.sh",
    "scripts/anatomy-hook.sh",
    "scripts/check_gate_wiring.py",
    "scripts/check_gate_wiring_preflight.py",
    "scripts/preflight.sh",
    "scripts/workflow_scan.py",
)


def scratch(dst: Path) -> None:
    """A minimal git repo holding just the wiring the lint inspects."""
    for rel in COPIED:
        src = ROOT / rel
        if not src.is_file():
            raise SystemExit(f"test_gate_wiring: missing input {rel} -- cannot build a scratch tree")
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out)
    subprocess.run(["git", "init", "-q"], cwd=dst, check=True)
    # `git ls-files` reads the INDEX, so staging is enough; no commit, no hooks.
    subprocess.run(["git", "add", "-A"], cwd=dst, check=True,
                   capture_output=True, text=True)


def run_lint(cwd: Path) -> tuple[int, str]:
    lint = cwd / "scripts/check_gate_wiring.py"
    if not lint.is_file():
        lint = LINT
    proc = subprocess.run(
        [sys.executable, str(lint)],
        cwd=cwd, capture_output=True, text=True,
    )
    return proc.returncode, (proc.stdout + proc.stderr)


def drop_step(text: str, marker: str) -> str:
    """Delete the `- name:` step block whose body contains `marker`.

    Steps are the 6-space `- name:` entries of a job; the block runs to the next
    one (or to the end of the step list).
    """
    lines = text.splitlines(keepends=True)
    starts = [i for i, ln in enumerate(lines) if ln.startswith("      - name:")]
    for n, i in enumerate(starts):
        end = starts[n + 1] if n + 1 < len(starts) else len(lines)
        if marker in "".join(lines[i:end]):
            return "".join(lines[:i] + lines[end:])
    raise SystemExit(f"test_gate_wiring: no step containing {marker!r} -- mutation is vacuous")


def drop_manifest_row(text: str, needle: str) -> str:
    kept = [ln for ln in text.splitlines(keepends=True) if needle not in ln]
    if len(kept) == len(text.splitlines(keepends=True)):
        raise SystemExit(f"test_gate_wiring: no manifest row matching {needle!r} -- mutation is vacuous")
    return "".join(kept)


def main() -> int:
    failures: list[str] = []

    def case(label: str, mutate, expect_nonzero: bool,
             expect_in: str | None = None) -> None:
        """One mutation, one verdict.

        `expect_in` pins the REASON as well as the exit code. A mutation that
        breaks the scratch tree in some other way also exits non-zero, so a case
        asserting the code alone can pass while the property it names is not
        gated at all -- the shape this repo already paid for elsewhere.
        """
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            scratch(d)
            if mutate is not None:
                mutate(d)
            rc, out = run_lint(d)
            ok = (rc != 0) if expect_nonzero else (rc == 0)
            if ok and expect_in is not None and expect_in not in out:
                ok = False
                out += f"\n(expected the failure to name: {expect_in!r})"
            verdict = "ok" if ok else "FAIL"
            print(f"  [{verdict}] {label}: exit={rc}")
            if not ok:
                failures.append(f"{label}: exit={rc}\n{out.rstrip()}")

    print("check_gate_wiring self-test")

    # Positive control. Without it, every mutation below could be passing
    # because the scratch tree is broken for an unrelated reason.
    case("unmutated scratch copy -> PASS", None, expect_nonzero=False)

    # Exercise the split keystone build/run control with bounded fakes.
    for mode in ("keystone_ok", "build_incomplete"):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            scratch(d)
            text = (d / "scripts/preflight.sh").read_text()
            start = text.index('  step "CI build-test cargo feature matrix parity')
            end = text.index('  step "cross-substrate determinism', start)
            segment = text[start:end]
            fake = d / "bin"
            fake.mkdir()
            cargo = fake / "cargo"
            cargo.write_text(
                f"#!{sys.executable}\n"
                "import os, sys\n"
                "mode = os.environ['PREFLIGHT_TEST_MODE']\n"
                "args = sys.argv[1:]\n"
                "if '--no-run' in args:\n"
                "    sys.exit(130 if mode == 'build_incomplete' else 0)\n"
                "print('test result: ok. 7 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out')\n"
            )
            cargo.chmod(0o755)
            (d / "scripts/workflow_scan.py").write_text(
                "print('cargo test matrix: rows=10 failed_rows=0')\n", encoding="utf-8"
            )
            env = dict(os.environ, PATH=str(fake) + os.pathsep + os.environ["PATH"],
                       PREFLIGHT_TEST_MODE=mode)
            script = ('set -uo pipefail\nfailed=0\nstep() { :; }\n'
                      'bad() { printf "%s\\n" "$*"; failed=1; }\n'
                      + segment + '\nexit "$failed"\n')
            result = subprocess.run(["bash", "-c", script], cwd=d, env=env,
                                    capture_output=True, text=True, timeout=30)
            output = result.stdout + result.stderr
            if mode == "keystone_ok":
                ok = result.returncode == 0 and "7/7 byte-identical" in output
            else:
                ok = (result.returncode != 0 and "BUILD_INCOMPLETE" in output
                      and "cross-substrate byte-identity regression" not in output
                      and "keystone NOT 7/7" not in output)
            print(f"  [{'ok' if ok else 'FAIL'}] executed preflight {mode}: exit={result.returncode}")
            if not ok:
                failures.append(f"executed preflight {mode}: {output}")

    # Run the real scanner entry point against argv-recording fake cargo.  This
    # pins the 10 semantic CI rows, positive-count contract, dynamic lock option,
    # and the exact pre-`--` location of injected Cargo options.
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        scratch(d)
        scanner = d / "scripts/workflow_scan.py"
        fake = d / "bin"
        fake.mkdir()
        cargo = fake / "cargo"
        cargo.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys, time\n"
            "args = sys.argv[1:]\n"
            "with open(os.environ['PREFLIGHT_ARGV_LOG'], 'a') as f: f.write(json.dumps(args) + '\\n')\n"
            "if os.environ.get('PREFLIGHT_TIMEOUT_PROBE') == '1' and 'timeout-probe' in args: time.sleep(2)\n"
            "n = 0 if os.environ.get('PREFLIGHT_EMPTY') == '1' else 1\n"
            "print(f'test result: ok. {n} passed; 0 failed; 0 ignored; 0 measured; 0 filtered out')\n"
            "if n: print('test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out')\n",
            encoding="utf-8")
        cargo.chmod(0o755)
        log = d / "argv.log"
        env = dict(os.environ, PATH=str(fake) + os.pathsep + os.environ["PATH"],
                   PREFLIGHT_ARGV_LOG=str(log))
        exact = subprocess.run([sys.executable, str(scanner), "run_cargo_test_matrix"],
                               cwd=d, env=env, capture_output=True, text=True, timeout=15)
        recorded = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
        bare = ["test", "--verbose", "--no-default-features", "--no-fail-fast"]
        ok = (exact.returncode == 0 and len(recorded) == 10 and bare in recorded
              and any("ffi::sys" in argv for argv in recorded)
              and any("ffi_header" in argv for argv in recorded)
              and all("--locked" not in argv for argv in recorded))
        print(f"  [{'ok' if ok else 'FAIL'}] typed executor preserves 10 unlocked CI argv: exit={exact.returncode}")
        if not ok:
            failures.append(f"typed argv probe: exit={exact.returncode}, recorded={recorded}, output={exact.stdout + exact.stderr}")

        log.write_text("", encoding="utf-8")
        empty_env = dict(env, PREFLIGHT_EMPTY="1")
        empty = subprocess.run([sys.executable, str(scanner), "run_cargo_test_matrix"],
                               cwd=d, env=empty_env, capture_output=True, text=True, timeout=15)
        empty_argv = log.read_text(encoding="utf-8").splitlines()
        ok = empty.returncode != 0 and len(empty_argv) == 10 and "passed=0" in empty.stdout
        print(f"  [{'ok' if ok else 'FAIL'}] zero-test matrix rows fail closed: exit={empty.returncode}")
        if not ok:
            failures.append(f"zero-test probe: exit={empty.returncode}, rows={len(empty_argv)}, output={empty.stdout + empty.stderr}")

        (d / "Cargo.lock").touch()
        log.write_text("", encoding="utf-8")
        locked = subprocess.run([sys.executable, str(scanner), "run_cargo_test_matrix"],
                                cwd=d, env=env, capture_output=True, text=True, timeout=15)
        locked_argv = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
        ok = (locked.returncode == 0 and len(locked_argv) == 10
              and all("--locked" in argv[:argv.index("--") if "--" in argv else len(argv)]
                      for argv in locked_argv))
        print(f"  [{'ok' if ok else 'FAIL'}] typed executor preserves locked CI argv: exit={locked.returncode}")
        if not ok:
            failures.append(f"locked argv probe: exit={locked.returncode}, recorded={locked_argv}, output={locked.stdout + locked.stderr}")

        # A one-second workflow timeout must stop a sleeping row, report it,
        # continue through the remaining rows, and return non-zero.
        commands = ['timeout 1 cargo test --no-default-features timeout-probe -- "case  one"']
        commands.extend(f"cargo test --no-default-features case-{n}" for n in range(1, 10))
        body = "\n".join(f"          {command}" for command in commands)
        (d / ".github/workflows/ci.yml").write_text(
            "name: fixture\njobs:\n  build_test:\n    steps:\n      - name: test\n        run: |\n" + body + "\n",
            encoding="utf-8",
        )
        (d / "Cargo.lock").unlink()
        log.write_text("", encoding="utf-8")
        timeout_env = dict(env, PREFLIGHT_TIMEOUT_PROBE="1")
        timed = subprocess.run([sys.executable, str(scanner), "run_cargo_test_matrix"],
                               cwd=d, env=timeout_env, capture_output=True, text=True, timeout=15)
        timed_argv = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
        first = timed_argv[0] if timed_argv else []
        separator = first.index("--") if "--" in first else -1
        ok = (timed.returncode != 0 and len(timed_argv) == 10
              and "timed out after 1s" in (timed.stdout + timed.stderr)
              and separator > 0 and "--no-fail-fast" in first[:separator]
              and first[separator + 1:] == ["case  one"])
        print(f"  [{'ok' if ok else 'FAIL'}] typed executor timeout fails closed: exit={timed.returncode}")
        if not ok:
            failures.append(f"timeout probe: exit={timed.returncode}, argv={timed_argv}, output={timed.stdout + timed.stderr}")

        # Parse errors fail before cargo can certify an empty or partial matrix.
        (d / ".github/workflows/ci.yml").write_text(
            "name: fixture\njobs:\n  build_test:\n    steps:\n      - run: |\n          cargo test 'unterminated\n",
            encoding="utf-8",
        )
        log.write_text("", encoding="utf-8")
        malformed = subprocess.run([sys.executable, str(scanner), "run_cargo_test_matrix"],
                                   cwd=d, env=env, capture_output=True, text=True, timeout=15)
        ok = malformed.returncode != 0 and not log.read_text(encoding="utf-8")
        print(f"  [{'ok' if ok else 'FAIL'}] malformed matrix fails before execution: exit={malformed.returncode}")
        if not ok:
            failures.append(f"malformed matrix probe: exit={malformed.returncode}, output={malformed.stdout + malformed.stderr}")

        (d / ".github/workflows/ci.yml").write_text(
            "name: fixture\njobs:\n  build_test:\n    steps:\n      - run: |\n"
            "          cargo test\n" +
            "\n".join(f"          cargo test --no-default-features case-{n}" for n in range(1, 10)) + "\n",
            encoding="utf-8",
        )
        log.write_text("", encoding="utf-8")
        empty_argv_row = subprocess.run(
            [sys.executable, str(scanner), "run_cargo_test_matrix"], cwd=d, env=env,
            capture_output=True, text=True, timeout=15,
        )
        ok = empty_argv_row.returncode != 0 and not log.read_text(encoding="utf-8")
        print(f"  [{'ok' if ok else 'FAIL'}] empty parsed argv fails before execution: exit={empty_argv_row.returncode}")
        if not ok:
            failures.append(f"empty argv probe: exit={empty_argv_row.returncode}, output={empty_argv_row.stdout + empty_argv_row.stderr}")

    # WSP-06(a): the local cargo-test leg must be mechanically derived from the
    # CI build_test rows. Removing the derivation must be diagnosed by name.
    def remove_matrix_derivation(d: Path) -> None:
        p = d / "scripts/preflight.sh"
        text = p.read_text(encoding="utf-8").replace(
            "scripts/workflow_scan.py run_cargo_test_matrix",
            "scripts/workflow_scan.py missing_matrix", 1)
        p.write_text(text, encoding="utf-8")
    case("preflight matrix derivation removed -> RED", remove_matrix_derivation,
         expect_nonzero=True, expect_in="cargo_test_matrix call")

    def add_unconditional_feature(d: Path) -> None:
        p = d / "scripts/workflow_scan.py"
        text = p.read_text(encoding="utf-8").replace(
            "inserted: list[str] = []",
            'inserted: list[str] = ["--features", "std-surface"]', 1)
        p.write_text(text, encoding="utf-8")
    case("bare matrix row forced to std-surface -> RED", add_unconditional_feature,
         expect_nonzero=True, expect_in="typed replay does not preserve")

    def mutate(p: Path, old: str, new: str, label: str) -> None:
        """Replace once, and FAIL LOUDLY when the anchor is gone.

        A mutation whose anchor no longer exists silently mutates nothing, the
        gate stays green, and the control reports itself as proving something.
        Two controls here were vacuous exactly that way after the keystone
        `BUILD_INCOMPLETE=1` assignment was removed.
        """
        text = p.read_text(encoding="utf-8")
        if old not in text:
            raise SystemExit(f"test_gate_wiring: mutation anchor absent for {label}: {old!r}")
        p.write_text(text.replace(old, new, 1), encoding="utf-8")

    def remove_keystone_bare_flag(d: Path) -> None:
        mutate(d / "scripts/preflight.sh",
               "MIND_BENCH_REQUIRE=1 cargo test --release --no-default-features \\",
               "MIND_BENCH_REQUIRE=1 cargo test --release \\",
               "keystone bare-feature parity")
    case("keystone bare-feature parity removed -> RED", remove_keystone_bare_flag,
         expect_nonzero=True, expect_in="keystone preflight invocation lacks --no-default-features")

    def remove_selector_replay(d: Path) -> None:
        mutate(d / "scripts/workflow_scan.py",
               "return argv[:separator] + inserted + argv[separator:]",
               "return argv[:1] + inserted",
               "typed selector replay")
    case("CI target selector replay removed -> RED", remove_selector_replay,
         expect_nonzero=True, expect_in="typed replay does not preserve")

    def replace_matrix_with_fake_success(d: Path) -> None:
        mutate(d / "scripts/workflow_scan.py",
               'command = ["cargo", "test", *argv]',
               'command = ["true"]',
               "matrix cargo execution")
    case("matrix cargo execution replaced by fake success -> RED",
         replace_matrix_with_fake_success, expect_nonzero=True,
         expect_in="does not execute cargo test")

    def remove_row_timeout(d: Path) -> None:
        mutate(d / "scripts/workflow_scan.py",
               "capture_output=True, text=True, timeout=timeout,",
               "capture_output=True, text=True, timeout=None,",
               "matrix row timeout")
    case("CI row timeout protection removed -> RED", remove_row_timeout,
         expect_nonzero=True, expect_in="timeout protection")

    def remove_keystone_build_split(d: Path) -> None:
        # Collapse the two-phase keystone into one command by deleting the
        # --no-run build leg's distinguishing flag.
        mutate(d / "scripts/preflight.sh",
               "--test phase_g_keystone_bootstrap --no-run 2>&1); ks_build_rc=$?",
               "--test phase_g_keystone_bootstrap 2>&1); ks_build_rc=$?",
               "keystone build split")
    case("keystone build split removed -> RED", remove_keystone_build_split,
         expect_nonzero=True, expect_in="split build phase")

    def drop_build_incomplete_label(d: Path) -> None:
        mutate(d / "scripts/preflight.sh",
               "BUILD_INCOMPLETE: keystone build exited",
               "keystone build exited",
               "BUILD_INCOMPLETE label")
    case("BUILD_INCOMPLETE label removed -> RED", drop_build_incomplete_label,
         expect_nonzero=True, expect_in="BUILD_INCOMPLETE")

    def reintroduce_byte_identity_claim(d: Path) -> None:
        # An interrupted build must never be reported as a determinism result.
        mutate(d / "scripts/preflight.sh",
               "BUILD_INCOMPLETE: keystone build exited $ks_build_rc; determinism was not evaluated",
               "a cross-substrate byte-identity regression; do NOT push",
               "byte-identity mislabel")
    case("build failure relabelled a byte-identity regression -> RED",
         reintroduce_byte_identity_claim, expect_nonzero=True,
         expect_in="byte-identity regression")

    def add_stale_exclusion(d: Path) -> None:
        # A name added to the shrink-only set with no matching CI target.
        mutate(d / "scripts/preflight.sh",
               "PREFLIGHT_TEST_EXCLUSIONS=()",
               'PREFLIGHT_TEST_EXCLUSIONS=("no_such_ci_target")',
               "stale exclusion")
    case("exclusion with no matching CI target -> RED", add_stale_exclusion,
         expect_nonzero=True, expect_in="stale exemption")

    def undeclared_exclusion_filter(d: Path) -> None:
        # An exemption applied outside the declared array is undeclared.
        mutate(d / "scripts/preflight.sh",
               'for excluded in "${PREFLIGHT_TEST_EXCLUSIONS[@]}"; do',
               "for excluded in mindfuzz_cross_substrate; do",
               "undeclared exclusion filter")
    case("exemption filtered outside the declared array -> RED",
         undeclared_exclusion_filter, expect_nonzero=True,
         expect_in="not in the declared array")

    def remove_ci_keystone_feature(d: Path) -> None:
        p = d / ".github/workflows/ci.yml"
        text = p.read_text(encoding="utf-8")
        phase = text.index("phase_g_keystone_bootstrap")
        pos = text.rfind('"mlir-build std-surface cross-module-imports"', 0, phase)
        if pos < 0:
            raise SystemExit("test_gate_wiring: keystone feature marker absent")
        marker = '"mlir-build std-surface cross-module-imports"'
        text = text[:pos] + '"std-surface cross-module-imports"' + text[pos + len(marker):]
        p.write_text(text, encoding="utf-8")
        subprocess.run(["git", "add", ".github/workflows/ci.yml"], cwd=d, check=True,
                       capture_output=True)
    case("CI keystone feature set drifted -> RED", remove_ci_keystone_feature,
         expect_nonzero=True, expect_in="keystone CI feature set differs")

    # THE mutation named by the finding: remove the commit-message gate step.
    # A commit message is the one artifact a later commit cannot correct, and
    # until this lint existed nothing noticed the step's absence.
    def del_msg_step(d: Path) -> None:
        p = d / ".github/workflows/docs-claims.yml"
        p.write_text(drop_step(p.read_text(encoding="utf-8"),
                               "scripts/check_commit_messages.sh"), encoding="utf-8")
    case("commit-message gate step deleted -> RED", del_msg_step, expect_nonzero=True)

    def del_file_gate_step(d: Path) -> None:
        p = d / ".github/workflows/docs-claims.yml"
        p.write_text(drop_step(p.read_text(encoding="utf-8"),
                               "scripts/check_no_ai_attribution.sh"), encoding="utf-8")
    case("file attribution gate step deleted -> RED", del_file_gate_step, expect_nonzero=True)

    # The finding itself: the workflow carrying those gates must be in the
    # release-required manifest, or a release can be cut from a commit where it
    # never ran.
    def del_manifest_row(d: Path) -> None:
        p = d / ".github/required-ci-jobs.tsv"
        p.write_text(drop_manifest_row(p.read_text(encoding="utf-8"), "docs-claims"),
                     encoding="utf-8")
    case("docs-claims row removed from the manifest -> RED", del_manifest_row,
         expect_nonzero=True)

    # A row whose job does not exist is a gate the release verifier will look
    # for and never find.
    def rename_job(d: Path) -> None:
        p = d / ".github/workflows/docs-claims.yml"
        p.write_text(p.read_text(encoding="utf-8").replace("\n  docs-claims:\n",
                                                           "\n  docs-claims-renamed:\n"),
                     encoding="utf-8")
    case("manifest job id absent from its workflow -> RED", rename_job, expect_nonzero=True)

    # Prose is not a gate: mentioning the script in a comment must not satisfy
    # the step requirement.
    def comment_out_step(d: Path) -> None:
        p = d / ".github/workflows/docs-claims.yml"
        txt = drop_step(p.read_text(encoding="utf-8"), "scripts/check_json_not_evidence.sh")
        txt = txt.replace("jobs:\n", "jobs:\n  # runs scripts/check_json_not_evidence.sh\n", 1)
        p.write_text(txt, encoding="utf-8")
    case("gate named only in a comment -> RED", comment_out_step, expect_nonzero=True)

    # Local defence-in-depth: the reconciled hooks directory must carry the
    # commit-msg hook, or the documented install silently drops it.
    def del_commit_msg_hook(d: Path) -> None:
        (d / ".githooks/commit-msg").unlink()
        subprocess.run(["git", "add", "-A"], cwd=d, check=True, capture_output=True)
    case(".githooks/commit-msg removed -> RED", del_commit_msg_hook, expect_nonzero=True)

    # git execs a hook; without the executable bit it is skipped in silence.
    # `core.fileMode=false` hides a local chmod from git entirely, so the bit is
    # asserted against the INDEX.
    def unset_exec_bit(d: Path) -> None:
        subprocess.run(["git", "update-index", "--chmod=-x", ".githooks/commit-msg"],
                       cwd=d, check=True, capture_output=True)
    case("hook not executable in the index -> RED", unset_exec_bit, expect_nonzero=True)

    # A hook may be a thin wrapper, but an older clone symlinks the PAYLOAD
    # script straight into the hooks directory. git resolves the symlink and
    # skips a target that is not executable, so the payload's index mode is as
    # load-bearing as the wrapper's: 100644 there is a hook that looks installed
    # and never runs.
    def unset_payload_exec_bit(d: Path) -> None:
        subprocess.run(["git", "update-index", "--chmod=-x", "scripts/commit-msg-hook.sh"],
                       cwd=d, check=True, capture_output=True)
    case("hook payload script not executable in the index -> RED",
         unset_payload_exec_bit, expect_nonzero=True)

    def del_posthooks(d: Path) -> None:
        (d / ".githooks/post-commit").unlink()
        subprocess.run(["git", "add", "-A"], cwd=d, check=True, capture_output=True)
    case(".githooks/post-commit removed -> RED", del_posthooks, expect_nonzero=True)

    # The harness ratchet. Every tracked .py/.sh here is harness -- a gate, a
    # driver, or a mutation proof for one -- and it is the file class that grows
    # without anyone deciding to grow it. The ceiling is only a ratchet if going
    # over it is RED, and only a ratchet if an empty scan is RED too.
    def overflow_harness(d: Path) -> None:
        room = d / "scratch_harness"
        room.mkdir()
        for i in range(HARNESS_CEILING + 1):
            (room / f"h{i:04d}.py").write_text("# harness\n", encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=d, check=True, capture_output=True)
    case(f"tracked .py/.sh over the ceiling ({HARNESS_CEILING}) -> RED",
         overflow_harness, expect_nonzero=True,
         expect_in="exceeds the recorded ceiling")

    # `0 <= ceiling` is the shape of a vacuous pass: a broken pathspec would
    # report a clean tree forever. The scratch keeps failing for its OTHER
    # missing inputs here, which is why this case pins the REASON.
    def erase_harness(d: Path) -> None:
        for f in list(d.rglob("*.py")) + list(d.rglob("*.sh")):
            f.unlink()
        subprocess.run(["git", "add", "-A"], cwd=d, check=True, capture_output=True)
    case("no tracked .py/.sh at all -> RED (not a vacuous pass)",
         erase_harness, expect_nonzero=True,
         expect_in="matched no .py/.sh at all")

    # And the real tree must pass, from the real repo root.
    rc, out = run_lint(ROOT)
    print(f"  [{'ok' if rc == 0 else 'FAIL'}] real tree -> PASS: exit={rc}")
    if rc != 0:
        failures.append(f"real tree: exit={rc}\n{out.rstrip()}")

    if failures:
        print("\ncheck_gate_wiring self-test: FAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\ncheck_gate_wiring self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
