#!/usr/bin/env python3
"""Regression test for the CI-coverage and branch-awareness contracts of
examples/mindc_mind/smoke_wiring_lint.py.

Two holes this pins shut, both of which let a GREEN lint certify an UNEXECUTED
gate:

  1. CI COVERAGE. The manifest legitimises `preflight` and `keystone` as
     runners, but no workflow invokes scripts/preflight.sh or
     examples/mindc_mind/fast_keystone.sh — so a `class=gate` row wired only to
     those two runs in nobody's CI. The lint checked that the manifest matched
     reality; it never asked whether reality contained CI. A gate may still opt
     out, but only by carrying a `deferred:` marker in its note that names the
     upgrade path.

  2. BRANCH AWARENESS. scripts/preflight.sh carried a comment asserting four
     gates "run UNCONDITIONALLY", while the gates sat inside the `--full`
     branch. The lint parsed preflight.sh as flat text, so it happily confirmed
     the runner column the false claim implied. The lint now knows where the
     `--full` branch starts and rejects an unconditional-execution claim made
     from inside it.

Hermetic: every case builds a miniature fixture tree (workflows + preflight +
smoke files + manifest) and runs the REAL lint against it, so this exercises the
shipped exit codes rather than a re-implementation.

Run: ``python3 scripts/test_smoke_wiring_lint.py`` (no third-party deps).

SECOND MODE, ``--anatomy``: the ANATOMY.md self-test (atomic generation, the
refresh-on-main-only hook rule, the push-to-main freshness scope; architecture
ruling 17). It lives here, behind a flag, because the harness ratchet in
scripts/check_gate_wiring.py caps the tracked .py/.sh count and every other
script with room is a different gate. The docs-claims workflow runs it as
``run_gate.py scripts/test_smoke_wiring_lint.py --anatomy``.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO
LINT = REPO / "examples" / "mindc_mind" / "smoke_wiring_lint.py"
# The lint imports the verdict vocabulary from scripts/gate_assert.py rather than
# re-spelling it, and refuses to run at all without the verdict-shape residual
# list. A fixture repo holding only the lint is therefore not a repo the lint can
# run in -- it died on `ModuleNotFoundError: gate_assert` and every case reported
# the exit code of a crash instead of the exit code of a verdict. Both are copied
# in ALONGSIDE the lint, from the real tree, so the fixture keeps exercising the
# SHIPPED module set and cannot drift into testing a reduced one.
SHIM = REPO / "scripts" / "gate_assert.py"
# An EMPTY residual is legal; a MISSING one is not. The real tree's list names
# real gates, and copying it in would make every fixture case fail on "residual
# entry names nothing" -- an assertion about the fixture, never about the lint.
# `smoke_wiring_lint` is the ONE name carried over from the real list, because the
# fixture copies that exact file in as one of its gates and the real tree already
# declares it residual (it reports four whole-run verdicts, not one per case). The
# fixture's own smokes are empty files and are on no residual.
RESIDUAL_FIXTURE = (
    "# fixture corpus: only the copied-in lint is on the verdict-shape residual\n"
    "smoke_wiring_lint\n"
)

# A workflow that executes the two smokes named here, so the lint's vacuity
# guard ("no smoke is executed by any workflow") is satisfied in every case and
# each case tests only the property it names.
CI_YML = """\
jobs:
  gates:
    steps:
      - name: run the corpus
        run: |
          python3 examples/mindc_mind/smoke_wiring_lint.py
          python3 examples/mindc_mind/alpha_smoke.py
          python3 examples/mindc_mind/beta_smoke.py
"""

# beta runs in BOTH runners; gamma only in preflight (and only under --full);
# delta is invoked by nobody. That spread is what lets each case below isolate
# one property.
PREFLIGHT_CLAIM_OUTSIDE = """\
#!/usr/bin/env bash
# --- git-level SDLC gates ---
# These run UNCONDITIONALLY and BEFORE the mic@3 smoke.
python3 examples/mindc_mind/beta_smoke.py

if [ "${1:-}" = "--full" ]; then
  python3 examples/mindc_mind/gamma_smoke.py
fi
"""

PREFLIGHT_CLAIM_INSIDE = """\
#!/usr/bin/env bash
python3 examples/mindc_mind/beta_smoke.py

if [ "${1:-}" = "--full" ]; then
  # --- git-level SDLC gates ---
  # These run UNCONDITIONALLY and BEFORE the mic@3 smoke.
  python3 examples/mindc_mind/gamma_smoke.py
fi
"""

SMOKES = ("alpha_smoke", "beta_smoke", "gamma_smoke", "delta_smoke")


def build(tmp: Path, rows: list[tuple[str, str, str, str]], preflight: str) -> Path:
    """Materialise a fixture repo and return the path of the lint inside it."""
    smoke_dir = tmp / "examples" / "mindc_mind"
    smoke_dir.mkdir(parents=True)
    (tmp / ".github" / "workflows").mkdir(parents=True)
    (tmp / "scripts").mkdir(parents=True)
    (tmp / ".github" / "workflows" / "ci.yml").write_text(CI_YML, encoding="utf-8")
    (tmp / "scripts" / "preflight.sh").write_text(preflight, encoding="utf-8")
    for name in SMOKES:
        (smoke_dir / f"{name}.py").write_text("", encoding="utf-8")
    (smoke_dir / "SMOKE_WIRING.tsv").write_text(
        "# fixture manifest\n" + "".join("\t".join(r) + "\n" for r in rows),
        encoding="utf-8",
    )
    shutil.copyfile(LINT, smoke_dir / "smoke_wiring_lint.py")
    shutil.copyfile(SHIM, tmp / "scripts" / "gate_assert.py")
    (smoke_dir / "VERDICT_SHAPE_RESIDUAL.txt").write_text(
        RESIDUAL_FIXTURE, encoding="utf-8")
    return smoke_dir / "smoke_wiring_lint.py"


def run(rows: list[tuple[str, str, str, str]], preflight: str) -> tuple[int, str]:
    with tempfile.TemporaryDirectory() as td:
        lint = build(Path(td), rows, preflight)
        p = subprocess.run(
            [sys.executable, str(lint)], capture_output=True, text=True
        )
        return p.returncode, p.stdout + p.stderr


# Rows whose wiring already matches the fixture and whose classes are all
# legitimate; each case overrides exactly one of gamma/delta.
FIXED = [
    ("smoke_wiring_lint", "ci", "gate", ""),
    ("alpha_smoke", "ci", "gate", ""),
    ("beta_smoke", "ci,preflight", "gate", ""),
]
GAMMA_OK = ("gamma_smoke", "preflight", "tool", "Analysis only; prints numbers, no verdict.")
DELTA_OK = ("delta_smoke", "none", "helper", "Imported by the smokes; not a gate.")


def main() -> int:
    results: list[tuple[str, int, int, str]] = []

    def case(label, want, gamma=GAMMA_OK, delta=DELTA_OK, preflight=PREFLIGHT_CLAIM_OUTSIDE):
        rc, out = run(FIXED + [gamma, delta], preflight)
        ok = rc == want
        # ONE VERDICT LINE PER CASE, printed here rather than recapped at the end.
        # scripts/gate_assert.py derives `asserted=<N>` from reported verdicts, so
        # a single summary line publishes the same count whether this list holds
        # five cases or none -- emptying the case list would leave the count, and
        # the exit code, exactly as they are now.
        print(f"[{'PASS' if ok else 'FAIL'}] {label} (expected exit {want}, "
              f"got {rc})")
        results.append((label, want, rc, "" if ok else out))

    # Control: nothing wrong -> the lint still passes (the new rules did not
    # simply red everything).
    case("control: legitimate corpus -> exit 0", 0)

    # THE MUTATION the audit prescribes: a class=gate row wired to preflight
    # only. Executed by no workflow, so a regression in it lands CI-green.
    case("gate wired to preflight only -> exit 1 (runs in no CI)", 1,
         gamma=("gamma_smoke", "preflight", "gate",
                "Runs in preflight; nobody said why it is not in CI."))

    # Same hole one step further: a gate wired nowhere at all, with a note that
    # explains the gate but claims no deferral.
    case("gate wired nowhere, note without a deferral -> exit 1", 1,
         delta=("delta_smoke", "none", "gate",
                "Important cross-implementation gate; currently unwired."))

    # The exemption is real but must be EXPLICIT: a `deferred:` marker naming
    # the upgrade path keeps a red/unrunnable gate honestly recorded instead of
    # forcing it into CI or out of the tree.
    case("gate off CI WITH a deferred: marker -> exit 0 (justified)", 0,
         delta=("delta_smoke", "none", "gate",
                "RED at this commit. deferred: wire into CI once the divergence is fixed."))

    # Branch awareness: an "runs UNCONDITIONALLY" claim written INSIDE the
    # `--full` branch is false and must red the lint.
    case("preflight claims UNCONDITIONALLY from inside --full -> exit 1", 1,
         preflight=PREFLIGHT_CLAIM_INSIDE)

    failures = [r for r in results if r[3]]
    for label, want, rc, out in failures:
        for line in out.strip().splitlines()[:12]:
            print(f"    {line}")
    # The SANCTIONED count-marker shape. scripts/gate_assert.py reads a count only
    # as `SDLC-GATE <name> ran=<n> fail=<k>`; a bare `ran=/fail=` is a number the
    # gate typed rather than one it earned, and examples/mindc_mind/
    # smoke_wiring_lint.py refuses it for exactly that reason.
    print(f"\nSDLC-GATE smoke_wiring_lint_selftest ran={len(results)} "
          f"fail={len(failures)}")
    if failures:
        return 1
    print("test_smoke_wiring_lint: PASS — CI-coverage rule and preflight "
          "branch awareness both enforced")
    return 0


# === --anatomy mode: ANATOMY.md self-test ====================================
# Each case MUTATES a scratch copy and asserts the generator or hook goes RED (or
# leaves the file alone); a pass on the real tree proves nothing about that.
ANATOMY = ROOT / "scripts" / "anatomy.sh"
ANATOMY_HOOK = ROOT / "scripts" / "anatomy-hook.sh"

failures: list[str] = []
checked = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    """One asserted property. Prints a verdict line either way."""
    global checked
    checked += 1
    if cond:
        print(f"[PASS] {name}")
    else:
        print(f"[FAIL] {name}{': ' + detail if detail else ''}")
        failures.append(name)


def git(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, check=True,
                          capture_output=True, text=True)


# --- ANATOMY.md is regenerated atomically ------------------------------------
# A pre-commit regeneration that died or was interrupted under load committed a
# TRUNCATED ANATOMY.md (3921 -> 967 lines) twice: the generator redirected its
# output straight onto the tracked file, so the file was cut to zero the moment
# it started and the hook staged whatever was left. Each case below makes the
# generator fail on purpose and asserts the committed index is byte-identical.

CRASH_ON = "zz_last.md"  # a root file: the crash lands after the header, mid-body
IDENT = ("-c", "user.name=STARGA Inc", "-c", "user.email=noreply@star.ga")


def anatomy_env(**extra: str) -> dict[str, str]:
    """Hermetic env: no inherited GIT_* state (this may run inside a hook) and
    the identity guard stubbed, so a sibling checkout's guard cannot decide the
    verdict of a case that is about ANATOMY.md. MIND_ANATOMY_REFRESH is dropped
    too: a landing run exports it, and the hook's branch rule is under test."""
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("GIT_") and k != "MIND_ANATOMY_REFRESH"}
    env.update(extra)
    return env


def anatomy_scratch(repo: Path, bindir: Path) -> bytes:
    """A scratch repo with the real generator and hook plus a good, committed
    index. Returns that index's bytes."""
    bindir.mkdir(parents=True, exist_ok=True)
    (repo / "scripts").mkdir(parents=True)
    for src in (ANATOMY, ANATOMY_HOOK):
        shutil.copy2(src, repo / "scripts" / src.name)
        (repo / "scripts" / src.name).chmod(0o755)
    for name in ("a.md", "b.md", "c.md", CRASH_ON):
        (repo / name).write_text(f"# {name}\n")
    git("init", "-q", "-b", "main", cwd=repo)  # the hook refreshes on main
    git("add", "-A", cwd=repo)
    r = subprocess.run(["bash", "scripts/anatomy.sh", ".", "--output", "ANATOMY.md"],
                       cwd=repo, capture_output=True, text=True, env=anatomy_env())
    check("scratch: the generator builds a first index", r.returncode == 0,
          f"rc={r.returncode} stderr={r.stderr.strip()[:300]!r}")
    git("add", "ANATOMY.md", cwd=repo)
    git(*IDENT, "commit", "-q", "--no-verify", "-m", "chore: seed", cwd=repo)
    guard = bindir / "sdlc-ok"
    guard.write_text("#!/bin/sh\nexit 0\n")
    guard.chmod(0o755)
    return (repo / "ANATOMY.md").read_bytes()


def crash_env(bindir: Path, action: str) -> dict[str, str]:
    """PATH with a `basename` that fails (or SIGKILLs the generator, or deletes a
    file the body has not reached yet) when asked about CRASH_ON. The generator
    calls basename per file while writing the body, under `set -e`, so this is a
    crash after the header was written -- the exact shape of the interrupted run.
    `fired` is the positive control: it proves the crash happened and the cases
    below are not passing vacuously."""
    real = shutil.which("basename")
    shim = bindir / "basename"
    shim.write_text(
        '#!/bin/sh\n'
        'for a in "$@"; do\n'
        f'  if [ "$a" = "{CRASH_ON}" ]; then\n'
        '    : > "$SHIM_FIRED"\n'
        '    if [ "$SHIM_ACTION" = kill ]; then kill -KILL "$(cat "$SHIM_PID")"; fi\n'
        '    if [ "$SHIM_ACTION" = vanish ]; then rm -f scripts/anatomy-hook.sh; continue; fi\n'
        '    exit 1\n'
        '  fi\n'
        'done\n'
        f'exec "{real}" "$@"\n')
    shim.chmod(0o755)
    return anatomy_env(PATH=f"{bindir}{os.pathsep}{os.environ['PATH']}",
                       MIND_SDLC=str(bindir / "sdlc-ok"), SHIM_ACTION=action,
                       SHIM_FIRED=str(bindir / "fired"), SHIM_PID=str(bindir / "pid"))


def hook_env(bindir: Path) -> dict[str, str]:
    """No crash shim: the hook and generator run as they do on a clean commit."""
    return anatomy_env(MIND_SDLC=str(bindir / "sdlc-ok"))


def run_hook(repo: Path, env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "scripts/anatomy-hook.sh"], cwd=repo,
                          capture_output=True, text=True, env=env)


def check_index(repo: Path, target: str) -> subprocess.CompletedProcess:
    """`anatomy.sh --check`: a file path, or `-` to read the index from stdin."""
    return subprocess.run(["bash", str(ANATOMY), "--check", target], cwd=repo,
                          capture_output=True, text=True, env=anatomy_env())


def case_anatomy_crashed_generator_leaves_the_index_untouched() -> None:
    with tempfile.TemporaryDirectory() as td:
        repo, bindir = Path(td) / "repo", Path(td) / "bin"
        good = anatomy_scratch(repo, bindir)
        (repo / "a.md").write_text("# a, changed: a fresh index would differ\n")
        env = crash_env(bindir, "exit")
        r = subprocess.run(["bash", "scripts/anatomy.sh", ".", "--output", "ANATOMY.md"],
                           cwd=repo, capture_output=True, text=True, env=env)
        check("a generator that dies mid-body fails loudly (positive control: it did die)",
              (bindir / "fired").exists() and r.returncode != 0,
              f"fired={(bindir / 'fired').exists()} rc={r.returncode}")
        check("a generator that dies mid-body leaves ANATOMY.md byte-identical",
              (repo / "ANATOMY.md").read_bytes() == good,
              "the tracked index was cut or rewritten by a run that never finished")
        check("a generator that dies mid-body leaves no temp file behind",
              not list(repo.glob("ANATOMY.md.tmp.*")),
              f"left: {[p.name for p in repo.glob('ANATOMY.md.tmp.*')]}")
        # Control for the two checks above: a run that DOES finish replaces the
        # file, so "unchanged" cannot mean "the writer is simply disconnected".
        ok = subprocess.run(["bash", "scripts/anatomy.sh", ".", "--output", "ANATOMY.md"],
                            cwd=repo, capture_output=True, text=True, env=anatomy_env())
        now = (repo / "ANATOMY.md").read_bytes()
        check("a finished run does replace the index", ok.returncode == 0 and now != good
              and b"a, changed" in now, f"rc={ok.returncode}")


def case_anatomy_killed_generator_leaves_the_index_untouched() -> None:
    """SIGKILL cannot be trapped: no cleanup runs, so the target must never have
    been the file being written."""
    with tempfile.TemporaryDirectory() as td:
        repo, bindir = Path(td) / "repo", Path(td) / "bin"
        good = anatomy_scratch(repo, bindir)
        (repo / "a.md").write_text("# a, changed: a fresh index would differ\n")
        env = crash_env(bindir, "kill")
        # `exec` keeps the shell's PID, so the shim can address the generator.
        r = subprocess.run(
            ["bash", "-c", 'echo $$ > "$SHIM_PID"; exec bash scripts/anatomy.sh . --output ANATOMY.md'],
            cwd=repo, capture_output=True, text=True, env=env)
        check("the generator was SIGKILLed mid-body (positive control)",
              (bindir / "fired").exists() and r.returncode == -9,
              f"fired={(bindir / 'fired').exists()} rc={r.returncode}")
        check("a SIGKILLed generator leaves ANATOMY.md byte-identical",
              (repo / "ANATOMY.md").read_bytes() == good,
              "the tracked index was cut by a run that was killed")


def case_anatomy_refuses_an_index_that_disagrees_with_itself() -> None:
    """A run that finishes can still be wrong: a file that disappears between the
    count and the listing leaves a header that promises more entries than the
    body has. The generator checks its own output before the rename."""
    with tempfile.TemporaryDirectory() as td:
        repo, bindir = Path(td) / "repo", Path(td) / "bin"
        good = anatomy_scratch(repo, bindir)
        env = crash_env(bindir, "vanish")
        r = subprocess.run(["bash", "scripts/anatomy.sh", ".", "--output", "ANATOMY.md"],
                           cwd=repo, capture_output=True, text=True, env=env)
        check("a file vanishing mid-run is observed (positive control)",
              (bindir / "fired").exists() and not (repo / "scripts" / "anatomy-hook.sh").exists())
        check("an index that disagrees with itself is refused, with the reason",
              r.returncode != 0 and "header says" in r.stderr,
              f"rc={r.returncode} stderr={r.stderr.strip()[:300]!r}")
        check("a refused index leaves ANATOMY.md byte-identical",
              (repo / "ANATOMY.md").read_bytes() == good)


def case_anatomy_sweeps_dead_temp_files_only() -> None:
    """SIGKILL leaves its temp file behind, and an untracked stray in the repo
    root is one `git add -A` from being committed. A later run removes temp
    files whose writer is gone, and must never touch a live concurrent run's."""
    with tempfile.TemporaryDirectory() as td:
        repo, bindir = Path(td) / "repo", Path(td) / "bin"
        anatomy_scratch(repo, bindir)
        gone = subprocess.Popen(["true"])
        gone.wait()
        dead = repo / f"ANATOMY.md.tmp.{gone.pid}"
        live = repo / f"ANATOMY.md.tmp.{os.getpid()}"  # this process is alive
        dead.write_text("partial")
        live.write_text("partial")
        r = subprocess.run(["bash", "scripts/anatomy.sh", ".", "--output", "ANATOMY.md"],
                           cwd=repo, capture_output=True, text=True, env=anatomy_env())
        check("sweep: the run itself succeeds", r.returncode == 0, r.stderr.strip()[:200])
        check("sweep: a temp file whose writer is dead is removed", not dead.exists())
        check("sweep: a temp file whose writer is alive is kept", live.exists())


def case_anatomy_check_rejects_a_damaged_index() -> None:
    with tempfile.TemporaryDirectory() as td:
        repo, bindir = Path(td) / "repo", Path(td) / "bin"
        good = anatomy_scratch(repo, bindir).decode()
        lines = good.splitlines(keepends=True)
        entries = [i for i, ln in enumerate(lines) if ln.startswith("- `")]
        n = re.search(r"^\*\*Files:\*\* (\d+)", good, re.M).group(1)
        damaged = {
            "cut mid-body (the shape that shipped)": "".join(lines[:entries[len(entries) // 2]]),
            "whose header count disagrees with the body": good.replace(
                f"**Files:** {n}", f"**Files:** {int(n) + 1}", 1),
            "with the closing line missing": "".join(lines[:-1]),
            "that is empty": "",
        }
        (repo / "good.md").write_text(good, encoding="utf-8")
        r = check_index(repo, "good.md")
        check("--check accepts a whole index (positive control)", r.returncode == 0,
              f"rc={r.returncode} stderr={r.stderr.strip()[:200]!r}")
        for label, text in damaged.items():
            (repo / "bad.md").write_text(text, encoding="utf-8")
            r = check_index(repo, "bad.md")
            check(f"--check rejects an index {label}", r.returncode != 0,
                  "a damaged index passed the integrity check")
        r = subprocess.run(["bash", str(ANATOMY), "--check", "-"], cwd=repo, input=good,
                           capture_output=True, text=True, env=anatomy_env())
        check("--check - reads the index from stdin", r.returncode == 0, r.stderr.strip()[:200])
        r = subprocess.run(["bash", str(ANATOMY), "--check", "-"], cwd=repo,
                           input=damaged["cut mid-body (the shape that shipped)"],
                           capture_output=True, text=True, env=anatomy_env())
        check("--check - rejects a damaged index on stdin", r.returncode != 0)


def case_precommit_hook_never_stages_a_bad_anatomy() -> None:
    """The hook is the last gate before the commit. Each leg below fed it a
    different way of producing a bad index and asserts nothing bad is staged."""
    def staged(repo: Path) -> bytes:
        return subprocess.run(["git", "show", ":ANATOMY.md"], cwd=repo, check=True,
                              capture_output=True).stdout

    # (a) the clean path still works.
    with tempfile.TemporaryDirectory() as td:
        repo, bindir = Path(td) / "repo", Path(td) / "bin"
        anatomy_scratch(repo, bindir)
        (repo / "d.md").write_text("# d\n")
        git("add", "d.md", cwd=repo)
        r = run_hook(repo, hook_env(bindir))
        check("hook, clean commit: exits 0 and stages the refreshed index",
              r.returncode == 0 and b"`d.md`" in staged(repo),
              f"rc={r.returncode} stderr={r.stderr.strip()[:300]!r}")
        check("hook, clean commit: the staged index passes --check",
              subprocess.run(["bash", str(ANATOMY), "--check", "-"], cwd=repo,
                             input=staged(repo).decode(), capture_output=True,
                             text=True, env=anatomy_env()).returncode == 0)

    # (b) the generator dies: commit blocked, index and worktree file untouched.
    with tempfile.TemporaryDirectory() as td:
        repo, bindir = Path(td) / "repo", Path(td) / "bin"
        good = anatomy_scratch(repo, bindir)
        (repo / "d.md").write_text("# d\n")
        git("add", "d.md", cwd=repo)
        r = run_hook(repo, crash_env(bindir, "exit"))
        # Positive control for this leg: the clean leg above reached the same
        # code with the same staged file, so the crash is what blocks here.
        check("hook, generator dies: commit blocked, shim fired",
              (bindir / "fired").exists() and r.returncode != 0,
              f"fired={(bindir / 'fired').exists()} rc={r.returncode}")
        check("hook, generator dies: ANATOMY.md in the worktree is untouched",
              (repo / "ANATOMY.md").read_bytes() == good)
        check("hook, generator dies: ANATOMY.md in the index is untouched",
              staged(repo) == good)
        check("hook, generator dies: the failure is explained",
              "ANATOMY.md" in r.stderr, f"stderr={r.stderr.strip()[:300]!r}")

    # (c) a generator that EXITS 0 with a truncated file (an old or foreign
    # build): the hook must verify the result itself, not trust the status.
    with tempfile.TemporaryDirectory() as td:
        repo, bindir = Path(td) / "repo", Path(td) / "bin"
        good = anatomy_scratch(repo, bindir)
        lines = good.decode().splitlines(keepends=True)
        cut = next(i for i, ln in enumerate(lines) if ln.startswith("- `")) + 2
        (repo / "truncated.part").write_text("".join(lines[:cut]), encoding="utf-8")
        real = repo / "scripts" / "anatomy.real.sh"
        shutil.copy2(ANATOMY, real)
        stub = repo / "scripts" / "anatomy.sh"
        stub.write_text('#!/usr/bin/env bash\n'
                        'if [ "$1" = --check ]; then exec bash "$(dirname "$0")/anatomy.real.sh" "$@"; fi\n'
                        'cp truncated.part ANATOMY.md\n')
        stub.chmod(0o755)
        (repo / "d.md").write_text("# d\n")
        git("add", "d.md", cwd=repo)
        r = run_hook(repo, hook_env(bindir))
        check("hook, generator exits 0 with a truncated file: commit blocked",
              r.returncode != 0, f"rc={r.returncode} stderr={r.stderr.strip()[:300]!r}")
        check("hook, generator exits 0 with a truncated file: nothing bad staged",
              staged(repo) == good)

    # (d) only ANATOMY.md is staged, so nothing is regenerated: what is staged
    # must still be whole.
    with tempfile.TemporaryDirectory() as td:
        repo, bindir = Path(td) / "repo", Path(td) / "bin"
        good = anatomy_scratch(repo, bindir)
        env = hook_env(bindir)
        lines = good.decode().splitlines(keepends=True)
        (repo / "ANATOMY.md").write_text("".join(lines[:len(lines) // 2]), encoding="utf-8")
        git("add", "ANATOMY.md", cwd=repo)
        r = run_hook(repo, env)
        check("hook, only a truncated ANATOMY.md staged: commit blocked",
              r.returncode != 0, f"rc={r.returncode} stderr={r.stderr.strip()[:300]!r}")
        (repo / "ANATOMY.md").write_text(good.decode().replace("UTC", "UTC ", 1), encoding="utf-8")
        git("add", "ANATOMY.md", cwd=repo)
        r = run_hook(repo, env)
        check("hook, only a whole ANATOMY.md staged: commit allowed (positive control)",
              r.returncode == 0, f"rc={r.returncode} stderr={r.stderr.strip()[:300]!r}")


def case_precommit_hook_refreshes_anatomy_on_main_only() -> None:
    """Architecture ruling 17: ANATOMY.md is regenerated on main only.

    A generated file every landing changes made every open PR conflict with
    main, and GitHub starts no pull_request workflow on a conflicting PR. So the
    hook refreshes it on branch main, or when MIND_ANATOMY_REFRESH=1 is set (the
    landing worktree is a detached HEAD), and leaves it alone everywhere else.
    The main leg is the positive control for every other leg: same scratch repo,
    same staged file, the refresh happens there and only there. The generator
    crash shim is a second control: on a PR branch it must never fire, because
    the generator must not even start (case (b) above shows it fires on main)."""
    def staged_names(repo: Path) -> list[str]:
        return git("diff", "--cached", "--name-only", cwd=repo).stdout.split()

    def leg(label: str, *, branch: str | None, refresh: str | None,
            want_refresh: bool, crash: bool = False) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo, bindir = Path(td) / "repo", Path(td) / "bin"
            good = anatomy_scratch(repo, bindir)
            if branch is None:
                git("checkout", "-q", "--detach", cwd=repo)
            elif branch != "main":
                git("checkout", "-q", "-b", branch, cwd=repo)
            (repo / "d.md").write_text("# d\n")
            git("add", "d.md", cwd=repo)
            env = crash_env(bindir, "exit") if crash else hook_env(bindir)
            if refresh is not None:
                env["MIND_ANATOMY_REFRESH"] = refresh
            r = run_hook(repo, env)
            now = (repo / "ANATOMY.md").read_bytes()
            if want_refresh:
                check(f"hook, {label}: ANATOMY.md is refreshed and staged",
                      r.returncode == 0 and b"`d.md`" in now
                      and "ANATOMY.md" in staged_names(repo),
                      f"rc={r.returncode} staged={staged_names(repo)} "
                      f"stderr={r.stderr.strip()[:200]!r}")
            else:
                check(f"hook, {label}: ANATOMY.md is left untouched and not staged",
                      r.returncode == 0 and now == good
                      and "ANATOMY.md" not in staged_names(repo)
                      and "d.md" in staged_names(repo),
                      f"rc={r.returncode} changed={now != good} "
                      f"staged={staged_names(repo)} stderr={r.stderr.strip()[:200]!r}")
            if crash:
                check(f"hook, {label}: the generator never started",
                      (bindir / "fired").exists() == want_refresh,
                      f"fired={(bindir / 'fired').exists()}")

    leg("on main (control)", branch="main", refresh=None, want_refresh=True)
    leg("on a PR branch", branch="feature/x", refresh=None, want_refresh=False)
    leg("on a PR branch, generator would crash", branch="feature/x", refresh=None,
        want_refresh=False, crash=True)
    leg("on a branch named like main", branch="mainline", refresh=None, want_refresh=False)
    leg("on a PR branch, MIND_ANATOMY_REFRESH=0", branch="feature/x", refresh="0",
        want_refresh=False)
    leg("on a PR branch, MIND_ANATOMY_REFRESH=1", branch="feature/x", refresh="1",
        want_refresh=True)
    leg("detached HEAD", branch=None, refresh=None, want_refresh=False)
    leg("detached HEAD, MIND_ANATOMY_REFRESH=1 (landing worktree)", branch=None,
        refresh="1", want_refresh=True)
    leg("on main, MIND_ANATOMY_REFRESH=1", branch="main", refresh="1", want_refresh=True)

    # A staged ANATOMY.md on a PR branch (a merge that took main's copy) is not
    # regenerated, but it must still be whole.
    for whole, label in ((False, "truncated"), (True, "whole")):
        with tempfile.TemporaryDirectory() as td:
            repo, bindir = Path(td) / "repo", Path(td) / "bin"
            good = anatomy_scratch(repo, bindir)
            git("checkout", "-q", "-b", "feature/x", cwd=repo)
            lines = good.decode().splitlines(keepends=True)
            body = good.decode() if whole else "".join(lines[:len(lines) // 2])
            (repo / "ANATOMY.md").write_text(body, encoding="utf-8")
            (repo / "d.md").write_text("# d\n")
            git("add", "ANATOMY.md", "d.md", cwd=repo)
            r = run_hook(repo, hook_env(bindir))
            check(f"hook, PR branch, {label} ANATOMY.md staged with another file: "
                  f"{'allowed and left as staged' if whole else 'commit blocked'}",
                  (r.returncode == 0 and (repo / "ANATOMY.md").read_text() == body)
                  if whole else r.returncode != 0,
                  f"rc={r.returncode} stderr={r.stderr.strip()[:200]!r}")


def freshness_enforced(env: "dict[str, str] | os._Environ[str]" = os.environ) -> bool:
    """True only on a push to main (architecture ruling 17).

    ANATOMY.md is regenerated on main only, so a PR branch carries whatever copy
    it forked from and is stale by design; comparing it with the PR's tracked
    tree would fail every PR that adds a file. Freshness is enforced where the
    file is published. The two variables are the ones GitHub Actions sets; a run
    outside CI (neither set) does not enforce it either."""
    return (env.get("GITHUB_EVENT_NAME") == "push"
            and env.get("GITHUB_REF") == "refs/heads/main")


def case_freshness_scope_is_push_to_main_only() -> None:
    """The selector itself, with a positive and negative leg each, so the
    main-only assertions in the next case cannot be switched off (or on) by a
    drifted condition without this going red."""
    for env, want, label in (
        ({"GITHUB_EVENT_NAME": "push", "GITHUB_REF": "refs/heads/main"}, True,
         "push to main"),
        ({"GITHUB_EVENT_NAME": "pull_request", "GITHUB_REF": "refs/pull/372/merge"},
         False, "pull_request"),
        ({"GITHUB_EVENT_NAME": "pull_request", "GITHUB_REF": "refs/heads/main"},
         False, "pull_request carrying a main ref"),
        ({"GITHUB_EVENT_NAME": "push", "GITHUB_REF": "refs/heads/fix/x"}, False,
         "push to another branch"),
        ({"GITHUB_EVENT_NAME": "push", "GITHUB_REF": "refs/tags/v1"}, False,
         "push of a tag"),
        ({"GITHUB_EVENT_NAME": "push"}, False, "push with no ref"),
        ({}, False, "a run outside CI"),
    ):
        check(f"ANATOMY.md freshness is {'enforced' if want else 'not enforced'}: {label}",
              freshness_enforced(env) is want)


def case_committed_anatomy_matches_the_tracked_tree() -> None:
    """The committed index must describe THIS tree: every path it lists is
    tracked, and tracked top-level files are not missing from it.

    MAIN-ONLY (see freshness_enforced): every assertion below -- the index lists
    something, every listed file is tracked, and the two tracked files the stale
    index once dropped are listed -- runs on a push to main and nowhere else. The
    no-attribution scan of the same file (case_committed_anatomy_names_no_model)
    and every scope case run on every event."""
    if not freshness_enforced():
        print("[SCOPE] ANATOMY.md freshness is enforced on push to main only "
              f"(event={os.environ.get('GITHUB_EVENT_NAME', '')!r} "
              f"ref={os.environ.get('GITHUB_REF', '')!r}); not asserted in this run")
        return
    doc = ROOT / "ANATOMY.md"
    tracked = set(subprocess.run(["git", "ls-files"], cwd=ROOT, check=True,
                                 capture_output=True, text=True).stdout.split())
    basenames = {Path(p).name for p in tracked}
    listed = re.findall(r"^- `([^`]+)`", doc.read_text(), re.M)
    check("ANATOMY.md lists something", bool(listed), "no entries parsed")
    unknown = sorted({n for n in listed if n not in basenames and n not in tracked})
    check("every file ANATOMY.md lists is tracked", not unknown,
          f"untracked entries: {unknown[:5]}")
    # Spot-check the other direction with files the stale index dropped.
    for rel in ("config/token_pricing.toml", ".githooks/post-merge"):
        if rel in tracked:
            check(f"ANATOMY.md lists tracked {rel}", Path(rel).name in set(listed),
                  "tracked file missing from the generated index")

def anatomy_main() -> int:
    for case in (
        case_anatomy_crashed_generator_leaves_the_index_untouched,
        case_anatomy_killed_generator_leaves_the_index_untouched,
        case_anatomy_refuses_an_index_that_disagrees_with_itself,
        case_anatomy_sweeps_dead_temp_files_only,
        case_anatomy_check_rejects_a_damaged_index,
        case_precommit_hook_never_stages_a_bad_anatomy,
        case_precommit_hook_refreshes_anatomy_on_main_only,
        case_freshness_scope_is_push_to_main_only,
        case_committed_anatomy_matches_the_tracked_tree,
    ):
        case()
    # No `asserted=` line on purpose: scripts/run_gate.py owns that contract line.
    if failures:
        print(f"::error::anatomy self-test: {len(failures)} FAILED: {failures}")
        return 1
    print(f"anatomy self-test: PASS ({checked} properties)")
    return 0


if __name__ == "__main__":
    raise SystemExit(anatomy_main() if "--anatomy" in sys.argv[1:] else main())
