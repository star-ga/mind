#!/usr/bin/env python3
"""Self-test for the no-AI-attribution gate and for what feeds it.

WHY THIS EXISTS
---------------
A public-repo doc shipped a line naming a model as this compiler's owner, and
every gate stayed green.  Two independent holes had to line up, and each one is
invisible from inside the other:

1. ``scripts/anatomy.sh`` enumerated ``git ls-files --cached --others
   --exclude-standard`` -- i.e. TRACKED **and UNTRACKED** files.  Regenerating
   the index in a checkout that happened to hold a private, uncommitted note
   transcribed that note's title into a COMMITTED, public document.  Nothing
   about the generator's own output looks wrong; the leak is in its input set.
2. ``scripts/check_no_ai_attribution.sh`` excluded ``ANATOMY.md`` from its
   pathspec, so the one file in the tree whose contents are copied from
   arbitrary other files was also the one file the attribution gate never read.

A third defect lives in the same generator: it wrote ANATOMY.md in place, so a
generator that died mid-write (an interrupted or concurrent pre-commit hook under
load) left a TRUNCATED index that the hook then staged and committed -- twice.
The ``case_anatomy_*`` and ``case_precommit_hook_*`` cases pin the repair: the
index is built beside the target and renamed into place only when whole.

Neither hole is testable by running the gate on the real tree and watching it
pass: a pass proves the tree is clean today, never that the gate would object
if it were not.  So each case below MUTATES a scratch copy and asserts the
gate/generator goes RED (or drops the file).  A mutation that does not change
the verdict is a property nothing is gating.

Scope values are READ OUT of the scripts under test (the scan floor, the
pathspec) rather than hand-copied here -- a hand-copied scope is the next
drift waiting to happen, which is the same defect class this file exists for.

Run: ``python3 scripts/test_no_ai_attribution.py`` (no third-party deps).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "scripts" / "check_no_ai_attribution.sh"
PATTERNS = ROOT / "scripts" / "ai_attribution_patterns.sh"
ANATOMY = ROOT / "scripts" / "anatomy.sh"
ANATOMY_HOOK = ROOT / "scripts" / "anatomy-hook.sh"

def rulebook(var: str) -> str:
    """Read one variable out of scripts/ai_attribution_patterns.sh.

    Sourced, never hand-copied. Two reasons, both load-bearing:
      * a hand-copied vocabulary is the drift this repo keeps paying for -- the
        self-test would keep asserting against a rule the gate no longer has;
      * this file is TRACKED and lives inside the gate's own scan scope, so a
        literal vendor name written here would make the gate red on its own
        test. Excluding this file instead would add a third never-scanned path,
        which is precisely the hole being closed. Building every vendor token at
        runtime keeps the source clean and the coverage honest.
    """
    r = subprocess.run(["bash", "-c", f'. "{PATTERNS}"; printf "%s" "${var}"'],
                       capture_output=True, text=True, check=True)
    val = r.stdout.strip()
    if not val:
        raise SystemExit(f"test_no_ai_attribution: {PATTERNS.name} defined no {var}")
    return val


# A credit shape the gate's own rulebook must already match, used as the payload
# for the in-scope cases: the first vendor token adjacent to the first credit
# noun, which is the canonical forbidden shape. These cases therefore test SCOPE
# -- "is this file read at all" -- and never pattern strength.
_NAME = rulebook("AI_NAMES").split("|")[0]
_WORD = rulebook("CREDIT_WORDS").split("|")[0]
CREDIT = f"- `notes.md` (~10 tok) - {_NAME} {_WORD} of the lowering pass"

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


def read_scan_floor() -> int:
    """MIN_SCANNED, read from the gate rather than duplicated here."""
    m = re.search(r"^MIN_SCANNED=(\d+)", GATE.read_text(), re.M)
    if not m:
        raise SystemExit(
            "test_no_ai_attribution: no MIN_SCANNED= in the gate -- the vacuity "
            "floor moved or was deleted; this self-test cannot build a scratch "
            "tree that clears it."
        )
    return int(m.group(1))


def scratch_gate_repo(dst: Path) -> None:
    """A git repo holding the gate, its rulebook, and enough filler to clear the
    gate's own vacuity floor (which otherwise fails every case for the wrong
    reason)."""
    (dst / "scripts").mkdir(parents=True, exist_ok=True)
    for src in (GATE, PATTERNS):
        shutil.copy2(src, dst / "scripts" / src.name)
    filler = dst / "docs"
    filler.mkdir(exist_ok=True)
    for i in range(read_scan_floor() + 1):
        (filler / f"f{i:04d}.md").write_text(f"# filler {i}\n")
    git("init", "-q", cwd=dst)
    git("add", "-A", cwd=dst)


def run_gate(repo: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "scripts/check_no_ai_attribution.sh"],
                          cwd=repo, capture_output=True, text=True)


def case_generated_doc_is_in_scope() -> None:
    """ANATOMY.md is generated FROM other files, so it is the likeliest carrier
    of a credit nobody wrote by hand. Excluding it from the pathspec made it the
    one place a credit could sit in a public artifact and pass."""
    with tempfile.TemporaryDirectory() as td:
        repo = Path(td)
        scratch_gate_repo(repo)
        (repo / "ANATOMY.md").write_text(f"# Repository Anatomy\n\n{CREDIT}\n")
        git("add", "ANATOMY.md", cwd=repo)
        r = run_gate(repo)
        check("credit in generated ANATOMY.md turns the gate RED",
              r.returncode != 0,
              f"gate exited 0; stdout={r.stdout.strip()[:200]!r}")


def case_control_clean_tree_passes() -> None:
    """Positive control for the case above: without the payload the same scratch
    tree must PASS, so a RED verdict there is attributable to the credit and not
    to a scratch repo the gate simply refuses."""
    with tempfile.TemporaryDirectory() as td:
        repo = Path(td)
        scratch_gate_repo(repo)
        (repo / "ANATOMY.md").write_text("# Repository Anatomy\n\n- `notes.md` (~10 tok)\n")
        git("add", "ANATOMY.md", cwd=repo)
        r = run_gate(repo)
        check("clean generated ANATOMY.md still PASSES",
              r.returncode == 0,
              f"gate exited {r.returncode}; stderr={r.stderr.strip()[:300]!r}")


def case_anatomy_indexes_only_tracked_files() -> None:
    """The generator's INPUT set is the root cause: an uncommitted private note
    in the checkout must not be transcribable into a committed public index."""
    with tempfile.TemporaryDirectory() as td:
        repo = Path(td)
        (repo / "tracked_module.md").write_text("# Tracked Module\n")
        git("init", "-q", cwd=repo)
        git("add", "tracked_module.md", cwd=repo)
        # Untracked, not ignored -- exactly the private-handoff-note shape.
        (repo / "PRIVATE-HANDOFF.md").write_text("# Handoff for a named model (owner)\n")
        r = subprocess.run(["bash", str(ANATOMY), ".", "--output", "ANATOMY.md"],
                           cwd=repo, capture_output=True, text=True)
        out = (repo / "ANATOMY.md").read_text() if (repo / "ANATOMY.md").is_file() else ""
        check("anatomy.sh ran on the scratch repo", r.returncode == 0,
              f"rc={r.returncode} stderr={r.stderr.strip()[:300]!r}")
        check("anatomy.sh omits UNTRACKED files from the index",
              "PRIVATE-HANDOFF" not in out,
              "an uncommitted note was transcribed into the generated doc")
        # Positive control: prove the generator indexes anything at all, so the
        # assertion above cannot pass because the output was empty.
        check("anatomy.sh still indexes TRACKED files",
              "tracked_module.md" in out,
              "generator produced no index; the omission check would be vacuous")


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
    verdict of a case that is about ANATOMY.md."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
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
    git("init", "-q", cwd=repo)
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


def case_bare_vendor_name_in_generated_index_is_red() -> None:
    """THE SHAPE THAT ACTUALLY SHIPPED, run through the PRIMARY gate.

    The line read "Handoff for <vendor> (compiler owner)". "owner" is not credit
    vocabulary, so neither PATTERN nor PATTERN_CREDIT sees it -- the gate the
    pre-commit hook and preflight both run printed PASS over it. The rule that
    catches it used to live here, in a CI-only self-test, i.e. at a layer no
    developer runs; it now lives in the gate, and this case asserts that by
    driving the gate rather than by re-implementing its grep."""
    with tempfile.TemporaryDirectory() as td:
        repo = Path(td)
        scratch_gate_repo(repo)
        (repo / "ANATOMY.md").write_text(
            f"# Repository Anatomy\n\n- `handoff.md` (~10 tok) - Handoff for {_NAME} (compiler owner)\n")
        git("add", "ANATOMY.md", cwd=repo)
        r = run_gate(repo)
        check("bare vendor name in generated ANATOMY.md turns the PRIMARY gate RED",
              r.returncode != 0,
              f"gate exited 0; stdout={r.stdout.strip()[:200]!r}")


def case_bare_vendor_name_outside_the_artifact_still_passes() -> None:
    """Negative control for the rule's SCOPE, and the reason it may not be
    tree-wide: naming a supported client as an integration target is allowed
    policy and appears legitimately in README.md and scripts/anatomy.sh. The
    same bare token that is fatal inside the generated index must stay legal in
    ordinary prose, or the fix above would have flagged real documentation."""
    with tempfile.TemporaryDirectory() as td:
        repo = Path(td)
        scratch_gate_repo(repo)
        (repo / "ANATOMY.md").write_text("# Repository Anatomy\n\n- `notes.md` (~10 tok)\n")
        (repo / "README.md").write_text(
            f"# Readme\n\nSupported clients: {_NAME}, and other MCP hosts.\n")
        git("add", "ANATOMY.md", "README.md", cwd=repo)
        r = run_gate(repo)
        check("bare vendor name OUTSIDE the generated index still PASSES",
              r.returncode == 0,
              f"gate exited {r.returncode}; stdout={r.stdout.strip()[:200]!r}")


def case_missing_artifact_fails_closed() -> None:
    """A rule with nothing to scan asserted nothing. An untracked ANATOMY.md
    must be a RED gate, never a quiet pass over an evaporated scope -- the same
    vacuity floor the tree-wide scan already applies to itself."""
    with tempfile.TemporaryDirectory() as td:
        repo = Path(td)
        scratch_gate_repo(repo)  # deliberately no ANATOMY.md
        r = run_gate(repo)
        check("gate fails CLOSED when ANATOMY.md is not tracked",
              r.returncode != 0,
              f"gate exited 0 with no artifact to scan; stdout={r.stdout.strip()[:200]!r}")


def case_committed_anatomy_names_no_model() -> None:
    """The committed artifact, judged BY THE GATE rather than by a second copy
    of its rule.

    This case used to re-implement the bare-name grep in Python. That made the
    self-test the only enforcer, and a second definition of the rule besides:
    two hand-copied vocabularies are the drift this repo keeps paying for. The
    real ANATOMY.md is now copied into a scratch tree and handed to the gate, so
    the verdict is attributable to THIS file alone (not to an unrelated hit
    elsewhere in the tree) while the rule itself has exactly one definition."""
    doc = ROOT / "ANATOMY.md"
    if not doc.is_file():
        check("committed ANATOMY.md exists", False, "ANATOMY.md missing")
        return
    with tempfile.TemporaryDirectory() as td:
        repo = Path(td)
        scratch_gate_repo(repo)
        shutil.copy2(doc, repo / "ANATOMY.md")
        git("add", "ANATOMY.md", cwd=repo)
        r = run_gate(repo)
        check("committed ANATOMY.md names no model (judged by the gate)",
              r.returncode == 0,
              f"gate exited {r.returncode}; stdout={r.stdout.strip()[-400:]!r}")


def case_committed_anatomy_matches_the_tracked_tree() -> None:
    """The committed index must describe THIS tree: every path it lists is
    tracked, and tracked top-level files are not missing from it."""
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


def case_author_identity_guard_rejects_trailing_dot_and_accepts_exact() -> None:
    """The author identity guard must require the exact string
    `STARGA Inc <noreply@star.ga>` and reject `STARGA Inc.` (trailing period)
    or personal addresses in both commit-range checks and local hooks."""
    import os

    regex_str = rulebook("EXPECTED_AUTHOR_IDENTITY_REGEX")
    check("EXPECTED_AUTHOR_IDENTITY_REGEX pins exact STARGA Inc <noreply@star.ga>",
          regex_str == r"^STARGA Inc <noreply@star\.ga>$",
          f"got {regex_str!r}")
    pat = re.compile(regex_str)
    check("author regex accepts exact 'STARGA Inc <noreply@star.ga>'",
          bool(pat.match("STARGA Inc <noreply@star.ga>")))
    check("author regex rejects trailing-period 'STARGA Inc. <noreply@star.ga>'",
          not pat.match("STARGA Inc. <noreply@star.ga>"))
    check("author regex rejects personal email",
          not pat.match("STARGA Inc <user@example.com>"))

    ccm = ROOT / "scripts" / "check_commit_messages.sh"
    cmh = ROOT / "scripts" / "commit-msg-hook.sh"
    with tempfile.TemporaryDirectory() as td:
        repo = Path(td)
        (repo / "scripts").mkdir(parents=True, exist_ok=True)
        for src in (PATTERNS, ccm, cmh):
            shutil.copy2(src, repo / "scripts" / src.name)
        git("init", "-q", cwd=repo)
        git("config", "user.name", "STARGA Inc", cwd=repo)
        git("config", "user.email", "noreply@star.ga", cwd=repo)
        (repo / "a.txt").write_text("base\n")
        git("add", "-A", cwd=repo)
        git("commit", "-q", "-m", "chore: base commit", cwd=repo)
        base_sha = git("rev-parse", "HEAD", cwd=repo).stdout.strip()

        # 1. Exact identity -> PASS
        (repo / "a.txt").write_text("exact\n")
        git("add", "a.txt", cwd=repo)
        git("commit", "-q", "-m", "fix: exact author commit", cwd=repo)
        r_ok = subprocess.run(["bash", "scripts/check_commit_messages.sh", f"{base_sha}..HEAD"],
                              cwd=repo, capture_output=True, text=True)
        check("check_commit_messages.sh accepts exact 'STARGA Inc <noreply@star.ga>'",
              r_ok.returncode == 0,
              f"rc={r_ok.returncode} out={r_ok.stdout.strip()[:200]!r}")
        good_sha = git("rev-parse", "HEAD", cwd=repo).stdout.strip()

        # 2. Trailing period 'STARGA Inc.' -> RED
        (repo / "a.txt").write_text("trailing-dot\n")
        git("add", "a.txt", cwd=repo)
        git("-c", "user.name=STARGA Inc.", "-c", "user.email=noreply@star.ga",
            "commit", "-q", "-m", "fix: trailing dot author", cwd=repo)
        r_dot = subprocess.run(["bash", "scripts/check_commit_messages.sh", f"{good_sha}..HEAD"],
                               cwd=repo, capture_output=True, text=True)
        check("check_commit_messages.sh rejects 'STARGA Inc. <noreply@star.ga>' -> RED",
              r_dot.returncode != 0 and "identity:" in r_dot.stdout,
              f"rc={r_dot.returncode} out={r_dot.stdout.strip()[:200]!r}")

        # 3. Local commit-msg hook rejects trailing period and accepts exact
        msg_file = repo / "COMMIT_EDITMSG"
        msg_file.write_text("fix: check local commit-msg hook\n")
        env_bad = os.environ.copy()
        env_bad["GIT_AUTHOR_NAME"] = "STARGA Inc."
        env_bad["GIT_AUTHOR_EMAIL"] = "noreply@star.ga"
        env_bad["GIT_COMMITTER_NAME"] = "STARGA Inc."
        env_bad["GIT_COMMITTER_EMAIL"] = "noreply@star.ga"
        env_bad.pop("STARGA_ALLOW_AUTHOR", None)
        r_hook_bad = subprocess.run(["bash", "scripts/commit-msg-hook.sh", str(msg_file)],
                                    cwd=repo, env=env_bad, capture_output=True, text=True)
        check("commit-msg-hook.sh rejects 'STARGA Inc. <noreply@star.ga>' -> RED",
              r_hook_bad.returncode != 0,
              f"rc={r_hook_bad.returncode} err={r_hook_bad.stderr.strip()[:200]!r}")

        env_good = os.environ.copy()
        env_good["GIT_AUTHOR_NAME"] = "STARGA Inc"
        env_good["GIT_AUTHOR_EMAIL"] = "noreply@star.ga"
        env_good["GIT_COMMITTER_NAME"] = "STARGA Inc"
        env_good["GIT_COMMITTER_EMAIL"] = "noreply@star.ga"
        env_good.pop("STARGA_ALLOW_AUTHOR", None)
        r_hook_good = subprocess.run(["bash", "scripts/commit-msg-hook.sh", str(msg_file)],
                                     cwd=repo, env=env_good, capture_output=True, text=True)
        check("commit-msg-hook.sh accepts exact 'STARGA Inc <noreply@star.ga>' -> PASS",
              r_hook_good.returncode == 0,
              f"rc={r_hook_good.returncode} err={r_hook_good.stderr.strip()[:200]!r}")


def _identity_guard_repo(td: Path):
    """Scratch repo holding the commit-message guard and its rulebook, with a
    base commit, a feature-branch commit and an extra base-branch commit (so a
    --no-ff merge lands a 2-parent commit inside the judged range).
    Returns (repo, base_sha, feature_sha)."""
    repo = td
    (repo / "scripts").mkdir(parents=True, exist_ok=True)
    for src in (PATTERNS, ROOT / "scripts" / "check_commit_messages.sh"):
        shutil.copy2(src, repo / "scripts" / src.name)
    git("init", "-q", cwd=repo)
    git("config", "user.name", "STARGA Inc", cwd=repo)
    git("config", "user.email", "noreply@star.ga", cwd=repo)
    (repo / "a.txt").write_text("base\n")
    git("add", "-A", cwd=repo)
    git("commit", "-q", "-m", "chore: base", cwd=repo)
    base_sha = git("rev-parse", "HEAD", cwd=repo).stdout.strip()
    git("checkout", "-q", "-b", "feature", cwd=repo)
    (repo / "a.txt").write_text("feature\n")
    git("add", "a.txt", cwd=repo)
    git("commit", "-q", "-m", "fix: feature work", cwd=repo)
    feature_sha = git("rev-parse", "HEAD", cwd=repo).stdout.strip()
    git("checkout", "-q", "-", cwd=repo)
    (repo / "b.txt").write_text("other\n")
    git("add", "b.txt", cwd=repo)
    git("commit", "-q", "-m", "chore: other", cwd=repo)
    return repo, base_sha, feature_sha


def _merge_with_identity(repo: Path, feature_sha: str, msg: str, *,
                         author, committer):
    """Create a --no-ff merge commit with explicit author/committer identities."""
    import os
    env = os.environ.copy()
    env["GIT_AUTHOR_NAME"], env["GIT_AUTHOR_EMAIL"] = author
    env["GIT_COMMITTER_NAME"], env["GIT_COMMITTER_EMAIL"] = committer
    return subprocess.run(["git", "merge", "--no-ff", "-m", msg, feature_sha],
                          cwd=repo, env=env, capture_output=True, text=True)


def _run_ccm(repo: Path, rng: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "scripts/check_commit_messages.sh", rng],
                          cwd=repo, capture_output=True, text=True)


def case_identity_guard_rejects_foreign_author_merge_in_range() -> None:
    """(a) A real merge commit with a foreign author INSIDE the judged range
    must fail the identity guard. A blanket skip of merge commits (2+ parents)
    destroyed exactly this property: a foreign-identity merge in the range then
    passed unchecked."""
    with tempfile.TemporaryDirectory() as td:
        repo, base_sha, feature_sha = _identity_guard_repo(Path(td))
        r_merge = _merge_with_identity(
            repo, feature_sha, "Merge feature into base",
            author=("CI Runner", "ci@example.com"),
            committer=("CI Runner", "ci@example.com"))
        check("foreign-author merge commit created",
              r_merge.returncode == 0, f"rc={r_merge.returncode}")
        merge_sha = git("rev-parse", "HEAD", cwd=repo).stdout.strip()
        n_parents = len(git("rev-list", "--no-walk", "--parents", merge_sha,
                            cwd=repo).stdout.split()) - 1
        check("foreign-author merge commit has 2 parents", n_parents == 2,
              f"got {n_parents} parents")
        r = _run_ccm(repo, f"{base_sha}..HEAD")
        check("check_commit_messages.sh rejects a foreign-author merge commit in range -> RED",
              r.returncode != 0 and "identity:" in r.stdout,
              f"rc={r.returncode} out={r.stdout.strip()[:200]!r}")


def case_identity_guard_accepts_house_identity_merge_in_range() -> None:
    """(b) A merge commit inside the range passes when its author is the exact
    house identity. The committer may also be the hosting-platform identity
    where the rulebook already allows it (EXPECTED_COMMITTER_IDENTITY_REGEX)."""
    with tempfile.TemporaryDirectory() as td:
        repo, base_sha, feature_sha = _identity_guard_repo(Path(td))
        r_merge = _merge_with_identity(
            repo, feature_sha, "Merge feature into base",
            author=("STARGA Inc", "noreply@star.ga"),
            committer=("STARGA Inc", "noreply@star.ga"))
        check("house-identity merge commit created",
              r_merge.returncode == 0, f"rc={r_merge.returncode}")
        r = _run_ccm(repo, f"{base_sha}..HEAD")
        check("check_commit_messages.sh accepts a house-identity merge commit in range -> PASS",
              r.returncode == 0,
              f"rc={r.returncode} out={r.stdout.strip()[:200]!r}")
        git("checkout", "-q", "-b", "feature2", cwd=repo)
        (repo / "a.txt").write_text("feature2\n")
        git("add", "a.txt", cwd=repo)
        git("commit", "-q", "-m", "fix: feature2 work", cwd=repo)
        feature2_sha = git("rev-parse", "HEAD", cwd=repo).stdout.strip()
        git("checkout", "-q", "-", cwd=repo)
        r_merge2 = _merge_with_identity(
            repo, feature2_sha, "Merge feature2 into base",
            author=("STARGA Inc", "noreply@star.ga"),
            committer=("GitHub", "noreply@github.com"))
        check("house-author/platform-committer merge commit created",
              r_merge2.returncode == 0, f"rc={r_merge2.returncode}")
        r2 = _run_ccm(repo, f"{base_sha}..HEAD")
        check("check_commit_messages.sh accepts house author with platform committer -> PASS",
              r2.returncode == 0,
              f"rc={r2.returncode} out={r2.stdout.strip()[:200]!r}")


def case_identity_guard_keeps_synthetic_ci_merge_out_of_scope() -> None:
    """(c) The synthetic integration merge commit CI creates for a
    pull_request (platform identity) is out of scope because the judged range
    is base..head of the PR branch, which does not contain it. That is a
    property of WHERE the range ends, not a blanket merge skip: case (a) shows
    a foreign-author merge inside the range is still rejected, and the contrast
    below shows the synthetic merge would be rejected too if the range reached
    it."""
    with tempfile.TemporaryDirectory() as td:
        repo, base_sha, feature_sha = _identity_guard_repo(Path(td))
        r_merge = _merge_with_identity(
            repo, feature_sha, "Merge pull request",
            author=("GitHub", "noreply@github.com"),
            committer=("GitHub", "noreply@github.com"))
        check("synthetic integration merge commit created",
              r_merge.returncode == 0, f"rc={r_merge.returncode}")
        merge_sha = git("rev-parse", "HEAD", cwd=repo).stdout.strip()
        n_parents = len(git("rev-list", "--no-walk", "--parents", merge_sha,
                            cwd=repo).stdout.split()) - 1
        check("synthetic integration merge commit has 2 parents", n_parents == 2,
              f"got {n_parents} parents")
        in_range = git("rev-list", f"{base_sha}..{feature_sha}",
                       cwd=repo).stdout.split()
        check("synthetic merge commit is not in the base..head range",
              merge_sha not in in_range,
              f"merge {merge_sha[:10]} found among {len(in_range)} commits")
        r = _run_ccm(repo, f"{base_sha}..{feature_sha}")
        check("base..head judges only the PR's own commits -> PASS",
              r.returncode == 0,
              f"rc={r.returncode} out={r.stdout.strip()[:200]!r}")
        r_old = _run_ccm(repo, f"{base_sha}..{merge_sha}")
        check("a range that INCLUDES the synthetic merge rejects its identity -> RED",
              r_old.returncode != 0 and "identity:" in r_old.stdout,
              f"rc={r_old.returncode} out={r_old.stdout.strip()[:200]!r}")
        wf = (ROOT / ".github" / "workflows" / "docs-claims.yml").read_text()
        check("docs-claims builds the pull_request range from the event base/head shas",
              "github.event.pull_request.base.sha" in wf
              and "github.event.pull_request.head.sha" in wf,
              "workflow no longer references the pull_request base/head shas")


def case_identity_anchor_exempts_history_unless_forced() -> None:
    """The historical identity anchor: a commit at or before it predates exact
    author identity enforcement and is exempt from that rule, and `--identity`
    lifts the exemption. A scratch repo never contains the anchor commit, so
    both arms run against this checkout's own history (the CI job clones it in
    full). A checkout without the anchor fails here rather than silently
    checking a different scope."""
    ccm = ROOT / "scripts" / "check_commit_messages.sh"
    m = re.search(r'^IDENTITY_ANCHOR="([0-9a-f]{40})"', ccm.read_text(), re.M)
    check("check_commit_messages.sh pins a 40-hex IDENTITY_ANCHOR", m is not None)
    if m is None:
        return
    anchor = m.group(1)
    present = subprocess.run(["git", "cat-file", "-e", f"{anchor}^{{commit}}"],
                             cwd=ROOT, capture_output=True).returncode == 0
    check("the identity anchor is in this checkout", present,
          "shallow clone: fetch full history (actions/checkout fetch-depth: 0)")
    if not present:
        return
    ident = git("log", "-1", "--format=%an <%ae>", anchor, cwd=ROOT).stdout.strip()
    house = re.compile(rulebook("EXPECTED_AUTHOR_IDENTITY_REGEX"))
    check("the anchor commit predates the house identity (the case the exemption is for)",
          not house.match(ident))
    rng = f"{anchor}~1..{anchor}"
    r_skip = subprocess.run(["bash", str(ccm), rng], cwd=ROOT,
                            capture_output=True, text=True)
    check("a commit at the anchor is exempt from the identity rule -> PASS",
          r_skip.returncode == 0 and "scanned 1 commits" in r_skip.stdout,
          f"rc={r_skip.returncode} out={r_skip.stdout.strip()[:200]!r}")
    r_forced = subprocess.run(["bash", str(ccm), "--identity", rng], cwd=ROOT,
                              capture_output=True, text=True)
    check("--identity checks the same commit's identity -> RED",
          r_forced.returncode != 0 and "identity:" in r_forced.stdout,
          f"rc={r_forced.returncode} out={r_forced.stdout.strip()[:200]!r}")


def main() -> int:
    for case in (
        case_generated_doc_is_in_scope,
        case_control_clean_tree_passes,
        case_anatomy_indexes_only_tracked_files,
        case_anatomy_crashed_generator_leaves_the_index_untouched,
        case_anatomy_killed_generator_leaves_the_index_untouched,
        case_anatomy_refuses_an_index_that_disagrees_with_itself,
        case_anatomy_sweeps_dead_temp_files_only,
        case_anatomy_check_rejects_a_damaged_index,
        case_precommit_hook_never_stages_a_bad_anatomy,
        case_bare_vendor_name_in_generated_index_is_red,
        case_bare_vendor_name_outside_the_artifact_still_passes,
        case_missing_artifact_fails_closed,
        case_committed_anatomy_names_no_model,
        case_committed_anatomy_matches_the_tracked_tree,
        case_author_identity_guard_rejects_trailing_dot_and_accepts_exact,
        case_identity_guard_rejects_foreign_author_merge_in_range,
        case_identity_guard_accepts_house_identity_merge_in_range,
        case_identity_guard_keeps_synthetic_ci_merge_out_of_scope,
        case_identity_anchor_exempts_history_unless_forced,
    ):
        case()
    # No `asserted=` line here on purpose: scripts/run_gate.py owns that
    # contract line and refuses a gate that publishes its own. The [PASS]/[FAIL]
    # verdict line each check() emits is what the shim counts.
    if failures:
        print(f"::error::no-ai-attribution self-test: {len(failures)} FAILED: {failures}")
        return 1
    print(f"no-ai-attribution self-test: PASS ({checked} properties)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
