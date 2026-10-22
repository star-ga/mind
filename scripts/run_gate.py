#!/usr/bin/env python3
"""run_gate.py — the shared gate runner that makes a VACUOUS PASS structurally fail.

THE DEFECT THIS CLOSES
----------------------
Every gate in this repo is invoked as `python3 <gate>` / `bash <gate>` and judged
by its exit code alone. An exit code answers "did anything that ran fail" — it
cannot answer "did anything run". Measured on this tree, with the compiler
binary absent:

  * examples/mindc_mind/rh_f64_aggregate_canary_smoke.py prints
    `SKIP  mindc not built` and returns 0 — a REQUIRED CI step, green, having
    compiled nothing and compared nothing.
  * a name-filtered test selector that matches zero tests reports success.
  * a corpus checker whose on-disk scan and manifest disagree still exits 0.

A fix proven through a gate like that is an unproven fix. Every gate must
therefore publish HOW MUCH it checked, and the runner — not the gate author's
memory — must refuse the answer "nothing".

THE CONTRACT (enforced here, defined in scripts/gate_assert.py)
---------------------------------------------------------------
A gate PASSES only when ALL of these hold:

  1. it exits 0;
  2. its output carries a final machine-readable `asserted=<N>` line;
  3. N >= 1 (or >= --min-asserted);
  4. no output line ANNOUNCES A SKIP — `gate_assert.is_skip_line` (one shared
     definition, applied to stdout and stderr alike) matches both a `SKIP` /
     `SKIPPED` line and a leg a multi-leg gate reports as `SKIPPED` mid-line.
     An announced skip is an unasserted invariant, and it must be a red gate or
     an explicit, recorded deferral, never a silent green.

Python gates get (2) for free: they are executed through `gate_assert.py`, which
counts evaluated `assert` statements and printed verdict lines and emits the
line unconditionally — including on an early `return 0` skip, where the count is
0 and the gate goes red. Non-Python gates have no shim, so the RUNNER applies
the same verdict-line rule to their stdout itself. A gate may NEVER print the
contract line itself — for a Python gate the shim treats that as forgery and
forces the count to zero, and for every other gate the runner does, so the rule
has one strength and not two.

USAGE
  python3 scripts/run_gate.py <gate> [args...]     # run one gate under contract
  python3 scripts/run_gate.py --sweep-all          # every gate in the corpus
  python3 scripts/run_gate.py --vacuity-sweep      # THE PROOF, see below

THE VACUITY SWEEP — TWO NEGATIVE CASES, NOT ONE
-----------------------------------------------
`--vacuity-sweep` is the positive control for this entire mechanism: a gate that
can still report success with no compiler present was never testing the compiler.
Being a positive control, it fails when it finds NOTHING wrong with the negative
case — so it cannot itself rot into a vacuous pass.

There are TWO ways to take the compiler away, and only running one of them was a
hole big enough to drive the original defect through:

  1. ENV-POINTED — MINDC / MINDC_BIN / MINDC_SO / MINDC_NATIVE_ELF are SET to
     paths that do not exist. This bites a gate that reads a handle.
  2. UNSET-HANDLE — those variables are REMOVED and every PATH entry holding a
     `mindc` executable is scrubbed, in a tree with no build artifacts. This is
     the case leg 1 cannot see: a gate that defaulted to the BARE NAME `"mindc"`
     never reads the handle at all, so pointing the handle at nothing changes
     nothing and the gate quietly runs a `mindc` off PATH — on a developer box,
     a symlink into a DIFFERENT checkout. Measured on a pristine copy of this
     tree with no `target/` and every handle unset: 21 compiler-dependent gates
     still printed `ALL PASS / asserted=1`, exercising another tree's binary.

Both legs must reject before the sweep passes. Each leg's expected-RED set is
derived from the handles THAT LEG removes (see TOOLCHAIN_HANDLES /
BINARY_HANDLES) rather than hand-copied, so the two scopes cannot drift apart;
every gate in the corpus is required RED by at least leg 1.

Each vacuity leg passes the trailing arguments the workflow actually uses,
read from that same scan. Runner flags are not gate arguments. Disagreeing
invocations are all run, and the gate is vacuous if any argv meets the
contract. A dynamic `$s.py` loop is not a concrete path. Each gate runs in
its own session, killed on return and on timeout, so stage2.elf cannot
outlive the step.

Leg 2 needs a tree with no `target/release/mindc`, which the runner cannot
arrange in place. On CI that is free — the sweep is wired before the cargo build.
Locally it runs against a scratch copy of the WORKING TREE built from
`git ls-files --cached --others --exclude-standard` (so uncommitted gate fixes
are under test while gitignored build artifacts are left behind). If neither can
be arranged the leg REFUSES with exit 1; it is never reported as passed.

The corpus scope is READ OUT of examples/mindc_mind/SMOKE_WIRING.tsv (the rows
already classified `gate`) plus concrete Python gates invoked through this
runner in a workflow. The latter is parsed with the same workflow extraction
owner used by the wiring lint, so options and shell continuations cannot hide a
load-bearing gate. A concrete missing workflow path is an error; dynamic loop
paths remain governed by the manifest. Both sources are derived rather than
hand-copied into this file, because a
second hand-maintained list of the same set is the drift this repo keeps paying
for. Which of those gates the sweep expects to go RED is likewise read out of
each gate's own SOURCE — a gate is compiler-dependent iff it names a compiler
handle (MINDC / MINDC_BIN / MINDC_SO / the self-host `.so` resolver / the
release binary path). A static lint like smoke_wiring_lint.py depends on no
compiler and is CORRECT to pass without one; excluding it via a hand-written
allowlist would be a second scope to drift out of sync with the first.

MEASURED BEFORE THIS RUNNER EXISTED (the whole justification): with the compiler
binary absent, 108 of the 146 corpus gates still exited 0.
"""

from __future__ import annotations

import argparse
import ast
import os
import pathlib
import shlex
import signal
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gate_runner_wiring_lint import scan as scan_workflow  # noqa: E402
from gate_assert import (ASSERTED_PREFIX, FORGED_RE, VERDICT_RE,  # noqa: E402
                         is_skip_line)

ROOT = pathlib.Path(__file__).resolve().parents[1]
SMOKE_DIR = ROOT / "examples" / "mindc_mind"
MANIFEST = SMOKE_DIR / "SMOKE_WIRING.tsv"
SHIM = ROOT / "scripts" / "gate_assert.py"

DEFAULT_TIMEOUT = int(os.environ.get("MIND_GATE_TIMEOUT", "900"))
SWEEP_TIMEOUT = int(os.environ.get("MIND_GATE_SWEEP_TIMEOUT", "180"))

# The handles the sweep actually removes. A gate whose CODE names one of them
# must not be able to report success once they are gone. Matched against the
# gate's code with docstrings and comments stripped, because prose routinely
# names `mindc` in a gate that never invokes it ("Pure Python 3, stdlib only.
# No mindc build.") and a raw substring match misclassifies exactly the static
# lints it must leave alone. `'mindc'` is matched WITH its quotes so the
# directory name `examples/mindc_mind` is not mistaken for the binary.
#
# The complement is not a loophole: a gate that runs only the COMMITTED frozen
# stage1.elf (self_host_arena_growth_smoke, self_host_continue_smoke) genuinely
# does not depend on the built toolchain and is correct to pass without it, and
# so are the three pure-text lints. Naming them in a literal list here instead
# would be a second scope to drift.
TOOLCHAIN_HANDLES = ("MINDC_SO", "MINDC_BIN", "MINDC_NATIVE_ELF",
                     "'MINDC'", "'mindc'", "_selfhost_so", "resolve_so")

# The narrower set for the UNSET-HANDLE leg. Each leg's expected-RED set is
# derived from the handles THAT LEG actually removes — never hand-copied, so the
# two cannot drift apart. The unset leg removes the compiler BINARY (env handles
# plus every PATH entry holding a `mindc`); it does NOT remove the committed
# frozen artifacts, and `resolve_so` with MINDC_SO unset is documented to fall
# back to the in-tree `.so` rather than require a compiler. So a gate whose only
# handle is `resolve_so` does not need a binary and is correctly exempt HERE,
# while the env-pointed leg — which sets MINDC_SO to a path that is not there,
# making a missing `.so` a broken wiring — still requires it to go red.
# Measured: self_host_loop_smoke.py's PRIMARY leg runs the frozen stage1.elf and
# asserts stage1==stage2==stage3 with zero compiler in the chain; calling that a
# vacuous pass would be wrong.
BINARY_HANDLES = ("MINDC_BIN", "MINDC_NATIVE_ELF", "'MINDC'", "'mindc'",
                  "resolve_mindc")


class Result:
    def __init__(self, gate: str, rc: int, asserted: int | None,
                 skipped: bool, reason: str, output: str) -> None:
        self.gate = gate
        self.rc = rc
        self.asserted = asserted
        self.skipped = skipped
        self.reason = reason
        self.output = output

    @property
    def ok(self) -> bool:
        return self.reason == ""


def _parse_asserted(text: str) -> int | None:
    """Last `asserted=<N>` line wins; None when the gate never published one."""
    n: int | None = None
    for line in text.splitlines():
        s = line.strip()
        if s.startswith(ASSERTED_PREFIX):
            try:
                n = int(s[len(ASSERTED_PREFIX):].strip())
            except ValueError:
                return None
    return n


def _has_skip(text: str) -> bool:
    """Both streams, one shared rule (gate_assert.is_skip_line)."""
    return any(is_skip_line(line) for line in text.splitlines())


def run_one(gate: str, args: list[str], *, min_asserted: int = 1,
            timeout: int = DEFAULT_TIMEOUT, env: dict[str, str] | None = None,
            echo: bool = True, root: pathlib.Path | None = None) -> Result:
    # `root` lets a sweep run the SAME corpus against a scratch copy of this
    # tree (see _scratch_tree): the unset-handle leg needs a tree with no
    # target/release/mindc in it, which the runner cannot arrange in place.
    base = root or ROOT
    shim = base / "scripts" / "gate_assert.py"
    p = pathlib.Path(gate)
    if not p.is_absolute():
        p = (base / gate) if (base / gate).exists() else p
    if not p.exists():
        return Result(gate, 127, None, False,
                      f"gate not found: {gate}", "")

    if p.suffix == ".py":
        cmd = [sys.executable, str(shim), str(p), *args]
    elif p.suffix == ".sh":
        cmd = ["bash", str(p), *args]
    else:
        cmd = [str(p), *args]

    proc = subprocess.Popen(
        cmd, cwd=str(base), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=env, start_new_session=True)
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        reap_keystone_group(proc)
        try:
            stdout, stderr = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            stdout, stderr = "", ""
        out = (stdout or "") + (stderr or "")
        return Result(gate, 124, None, False, f"timed out after {timeout}s", out)
    reap_keystone_group(proc)
    stdout = stdout or ""
    stderr = stderr or ""
    out = stdout + stderr
    rc = proc.returncode if proc.returncode is not None else 1

    if echo:
        sys.stdout.write(out)
        sys.stdout.flush()

    # STDOUT ONLY. The shim prints the contract line on stdout after restoring
    # the real stream; reading the stderr half too let a gate's own
    # `asserted=99` on stderr land AFTER it in the concatenation and win.
    forged = False
    if p.suffix == ".py":
        n = _parse_asserted(stdout)
    else:
        # A non-Python gate is NOT executed through the shim, so nothing
        # publishes the contract line on its behalf and nothing was stopping it
        # from publishing its own: reading `asserted=<N>` out of ANY gate's
        # output made `echo "asserted=42"; exit 0` grade PASS having checked
        # nothing — the exact forgery the shim refuses for Python gates. The
        # count is therefore derived HERE, from the same verdict-line rule, over
        # the runner-owned stream (stdout — both routed `.sh` gates report their
        # verdict there), and a printed contract line on EITHER stream is
        # forgery. One rule for both kinds of gate, not a weaker second one.
        forged = any(FORGED_RE.match(line) for line in out.splitlines())
        n = 0 if forged else sum(1 for line in stdout.splitlines()
                                 if VERDICT_RE.search(line))

    skipped = _has_skip(out)
    reason = ""
    if rc != 0:
        reason = f"exit {rc}"
    elif forged:
        reason = (f"the gate published its own {ASSERTED_PREFIX}line — "
                  "the contract line is the runner's verdict, not the gate's")
    elif n is None:
        reason = f"no `{ASSERTED_PREFIX}<N>` line in output (gate published no count)"
    elif n < min_asserted:
        # Two distinct facts, named distinctly: 0 is a gate that checked NOTHING,
        # while 0 < n < floor is a MULTI-LEG gate that ran fewer legs than this
        # call site requires (the self-host LOOP gate reporting its PRIMARY leg
        # with the drift ORACLE dropped is exactly this). Calling the second one
        # "checked nothing" sends the reader hunting the wrong defect.
        reason = (f"asserted={n} (< {min_asserted}) — the gate checked nothing"
                  if n == 0 else
                  f"asserted={n} (< {min_asserted}) — the gate reported fewer "
                  f"asserted legs than this call site requires")
    elif skipped:
        reason = "output contains a `SKIP` line — an unasserted invariant"
    return Result(gate, rc, n, skipped, reason, out)


def workflow_invoked_gates() -> list[str]:
    """Return concrete Python gates invoked through this runner by a workflow.

    Use the same run-block, continuation, shell-segment, and argument-position
    parser as ``gate_runner_wiring_lint.py``.  This recognizes options between
    ``run_gate.py`` and the path, including across continued lines.  Dynamic
    loop paths are already represented by the manifest; a concrete missing path
    is an error rather than a silently shrunk corpus.
    """
    found: list[str] = []
    missing: list[str] = []
    workflow_dir = ROOT / ".github" / "workflows"
    if not workflow_dir.is_dir():
        return found
    for workflow in sorted(workflow_dir.glob("*.yml")) + sorted(
        workflow_dir.glob("*.yaml")
    ):
        try:
            invocations, _ = scan_workflow(workflow)
        except OSError as exc:
            raise RuntimeError(f"cannot scan workflow {workflow}: {exc}") from exc
        for invocation in invocations:
            if not invocation.routed or not invocation.path.endswith(".py"):
                continue
            relative = invocation.path
            # `$s.py` / `${gate}.py` are shell templates. Their concrete members
            # are governed by SMOKE_WIRING.tsv; there is no single path to add.
            if "$" in relative or "{" in relative or "}" in relative:
                continue
            candidate = ROOT / relative
            if not candidate.is_file():
                missing.append(f"{workflow.name}:{relative}")
            elif relative not in found:
                found.append(relative)
    if missing:
        raise RuntimeError(
            "workflow invokes missing concrete gate(s): " + ", ".join(sorted(missing))
        )
    return found


def corpus() -> list[str]:
    """The manifest's gates plus every gate a workflow invokes."""
    gates: list[str] = []
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = [c.strip() for c in line.split("\t")]
        if len(parts) < 3 or parts[2] != "gate":
            continue
        name = parts[0]
        cand = SMOKE_DIR / f"{name}.py"
        if not cand.exists():
            nested = [q for q in SMOKE_DIR.rglob(f"{name}.py")
                      if "__pycache__" not in q.parts]
            if not nested:
                continue
            cand = nested[0]
        gates.append(str(cand.relative_to(ROOT)))
    for extra in workflow_invoked_gates():
        if extra not in gates:
            gates.append(extra)
    return gates


def _code_without_prose(src: str) -> str:
    """The gate's executable code with docstrings (and, via unparse, comments) gone."""
    import ast

    tree = ast.parse(src)
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) \
                and isinstance(body[0].value, ast.Constant) \
                and isinstance(body[0].value.value, str):
            body.pop(0)
    return ast.unparse(tree)


def compiler_dependent(rel: str,
                       handles: tuple[str, ...] = TOOLCHAIN_HANDLES) -> bool:
    """True iff this gate consults one of `handles` — the ones a leg removes."""
    try:
        src = (ROOT / rel).read_text(encoding="utf-8")
    except OSError:
        return False
    try:
        code = _code_without_prose(src)
    except SyntaxError:
        return True  # unparseable: assume it runs something, never assume it does not
    return any(m in code for m in handles)


# The env-var handles the ENV-POINTED leg sets, in the one place that sets them.
# `supplies_own_handles()` reads this same tuple rather than re-typing the names,
# because a leg whose action and whose expected-RED derivation name two different
# sets is the drift this runner exists to refuse.
ENV_POINTED_HANDLES = ("MINDC", "MINDC_BIN", "MINDC_SO", "MINDC_NATIVE_ELF",
                       "MINDC_SO_NOBUILD")


def _absent_toolchain_env() -> dict[str, str]:
    """Every compiler handle the corpus consults, pointed at a path that is not there."""
    env = dict(os.environ)
    missing = str(ROOT / ".gate-vacuity-sweep" / "absent")
    for var in ENV_POINTED_HANDLES:
        env[var] = missing
    env["MINDC_SO_NOBUILD"] = "1"
    return env


def supplies_own_handles(rel: str) -> bool:
    """True iff this gate WRITES every env handle the env-pointed leg would set.

    The leg's whole action is putting those variables in the environment. A gate
    that ASSIGNS them itself -- `os.environ["MINDC_SO"] = <fabricated path>` --
    never reads what the leg put there, so removing the compiler that way proves
    nothing about it, in either direction. Requiring it RED would be requiring a
    gate to fail for a reason it cannot observe.

    That is the same derivation the rest of this file uses (an expected-RED set
    read out of what the leg actually removes), not a second hand-written
    allowlist: the answer comes from the GATE'S OWN SOURCE, and it is narrow.
    Measured over the whole corpus at the commit that added this: exactly ONE
    gate qualifies -- selfhost_so_provenance_smoke.py, which fabricates stale,
    fresh, sidecar-stamped and absent oracles in a temp tree in order to assert
    that resolve_so() REFUSES the ones it cannot prove fresh. It is a static
    contract test of the resolver, in the same class as the pure lints, and it
    is correct to pass with no compiler anywhere.

    A gate that merely READS a handle is untouched by this: it is still required
    RED. The exemption is printed by name on every sweep, so growth in this class
    is visible rather than silent.
    """
    try:
        src = (ROOT / rel).read_text(encoding="utf-8")
        code = _code_without_prose(src)
        tree = ast.parse(code)
    except (OSError, SyntaxError):
        return False  # unreadable/unparseable: never assume immunity

    def _env_key(node: ast.AST) -> str | None:
        """`os.environ[<literal>]` -> the literal, else None."""
        if not isinstance(node, ast.Subscript):
            return None
        v = node.value
        if not (isinstance(v, ast.Attribute) and v.attr == "environ"):
            return None
        k = node.slice
        return k.value if isinstance(k, ast.Constant) and isinstance(k.value, str) else None

    written: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                key = _env_key(tgt)
                if key:
                    written.add(key)
        elif isinstance(node, ast.Call):
            fn = node.func
            # `os.environ.pop("MINDC_SO", None)` -- clearing the ambient value is
            # supplying it just as much as overwriting it is.
            if isinstance(fn, ast.Attribute) and fn.attr in ("pop", "setdefault") \
                    and isinstance(fn.value, ast.Attribute) and fn.value.attr == "environ" \
                    and node.args and isinstance(node.args[0], ast.Constant) \
                    and isinstance(node.args[0].value, str):
                written.add(node.args[0].value)

    named = {h for h in ENV_POINTED_HANDLES if f"'{h}'" in code or f'"{h}"' in code}
    return bool(named) and named <= written


def _unset_toolchain_env() -> tuple[dict[str, str], str]:
    """Every compiler handle REMOVED, and every PATH entry holding a `mindc` gone.

    The env-pointed leg (`_absent_toolchain_env`) sets each handle to a path that
    does not exist. That is one of two negative cases, and it is the WEAKER one:
    a gate whose default is a BARE NAME never reads the handle it was given, so
    pointing the handle at nothing changes nothing and the gate quietly runs a
    `mindc` off PATH — another checkout's binary. Measured on a pristine
    `git archive HEAD` copy with no `target/` at all: 21 compiler-dependent gates
    still reported PASS. This leg is the case that catches them.
    """
    env = dict(os.environ)
    for var in ("MINDC", "MINDC_BIN", "MINDC_SO", "MINDC_NATIVE_ELF"):
        env.pop(var, None)
    env["MINDC_SO_NOBUILD"] = "1"
    kept, dropped = [], []
    for entry in env.get("PATH", "").split(os.pathsep):
        if not entry:
            continue
        cand = pathlib.Path(entry) / "mindc"
        (dropped if os.access(cand, os.X_OK) else kept).append(entry)
    env["PATH"] = os.pathsep.join(kept)
    note = (f"PATH scrubbed of {len(dropped)} entry/entries holding a `mindc`: "
            f"{', '.join(dropped) or '(none)'}")
    return env, note


def _scratch_tree() -> tuple[pathlib.Path | None, str]:
    """A copy of this WORKING TREE with no build artifacts, for the unset leg.

    Built from `git ls-files --cached --others --exclude-standard`, i.e. tracked
    plus untracked-but-not-ignored files at their working-tree content — so an
    uncommitted gate fix is under test, while `/target/` (gitignored) and every
    stray `.so` are left behind. Returns (None, reason) when git cannot produce
    it; the caller must then REFUSE, never report the leg as passed.
    """
    import shutil
    import tempfile

    if shutil.which("git") is None:
        return None, "git is not on PATH"
    try:
        listing = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            cwd=str(ROOT), capture_output=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as e:
        return None, f"git ls-files failed: {e}"
    if listing.returncode != 0:
        return None, f"git ls-files exit {listing.returncode}: {listing.stderr[-300:]!r}"
    dest = pathlib.Path(tempfile.mkdtemp(prefix="mind-vacuity-unset-"))
    tar_c = subprocess.run(["tar", "--null", "-T", "-", "-cf", "-"],
                           cwd=str(ROOT), input=listing.stdout,
                           capture_output=True)
    if tar_c.returncode != 0:
        return None, f"tar pack exit {tar_c.returncode}: {tar_c.stderr[-300:]!r}"
    tar_x = subprocess.run(["tar", "-xf", "-", "-C", str(dest)],
                           input=tar_c.stdout, capture_output=True)
    if tar_x.returncode != 0:
        return None, f"tar unpack exit {tar_x.returncode}: {tar_x.stderr[-300:]!r}"
    return dest, str(dest)


def _unset_handle_leg(gates: list[str], jobs: int, timeout: int) -> int:
    """The second negative case: handles UNSET and no `mindc` reachable at all.

    Runs in ROOT when ROOT/target/release/mindc is absent (the CI case — this
    sweep is wired before the cargo build), otherwise in a scratch copy of the
    working tree, which by construction has no build artifacts. If neither can
    be arranged the leg REFUSES with exit 1: a leg that could not create its own
    negative case has proven nothing and must never be reported as passed.
    """
    import concurrent.futures as cf
    import shutil

    exempt = [g for g in gates if not compiler_dependent(g, BINARY_HANDLES)]
    gates = [g for g in gates if compiler_dependent(g, BINARY_HANDLES)]
    print(f"run_gate: {len(exempt)} of the toolchain-dependent gates name no "
          f"compiler BINARY handle and are exempt from the unset-handle leg "
          f"(the env-pointed leg still requires them red): "
          f"{', '.join(pathlib.Path(g).name for g in sorted(exempt)) or '(none)'}")
    if not gates:
        print("run_gate: FAIL — no binary-dependent gate found; the handle "
              "detection is wrong, not the corpus. Refusing to pass vacuously.")
        return 1

    env, path_note = _unset_toolchain_env()
    if shutil.which("mindc", path=env["PATH"]) is not None:
        print("run_gate: FAIL — a `mindc` is STILL reachable after scrubbing PATH "
              f"({shutil.which('mindc', path=env['PATH'])}). The unset-handle leg "
              "cannot create its negative case; refusing to report it as passed.")
        return 1

    scratch: pathlib.Path | None = None
    if not (ROOT / "target" / "release" / "mindc").exists():
        root = ROOT
        where = f"{ROOT} (no target/release/mindc present)"
    else:
        scratch, why = _scratch_tree()
        if scratch is None:
            print("run_gate: FAIL — this tree HAS target/release/mindc and the "
                  f"runner cannot remove it, so the unset-handle leg needs a "
                  f"scratch copy, which could not be made: {why}. Refusing to "
                  "report the leg as passed.")
            return 1
        root = scratch
        where = f"{scratch} (scratch copy of the working tree, no build artifacts)"
        if (root / "target" / "release" / "mindc").exists():
            print("run_gate: FAIL — the scratch copy contains a mindc binary; "
                  "the negative case was not created. Refusing to pass.")
            shutil.rmtree(scratch, ignore_errors=True)
            return 1

    print(f"run_gate: unset-handle leg over {len(gates)} gates in {where}")
    print(f"run_gate: {path_note}")
    try:
        with cf.ThreadPoolExecutor(max_workers=jobs) as ex:
            results = list(ex.map(
                lambda g: _run_wired(g, timeout, env, root), gates))
    finally:
        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=True)

    leaked = [r for r in results if r.ok]
    for r in leaked:
        print(f"VACUOUS  {r.gate}: passed with NO compiler reachable and every "
              f"handle UNSET (exit {r.rc}, asserted={r.asserted}) — it resolved "
              f"some other tree's binary, not the one under test")
    print(f"unset-handle leg: gates={len(results)} vacuous={len(leaked)} "
          f"rejected={len(results) - len(leaked)}")
    if leaked:
        print(f"FAIL  {len(leaked)} gate(s) pass with no compiler reachable.")
        return 1
    print("PASS  every gate failed closed with the handles unset and no `mindc` "
          "on PATH.")
    return 0


def reap_keystone_group(proc: subprocess.Popen) -> None:
    """Kill the gate session, including a grandchild the direct child left behind."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except OSError:
        if proc.poll() is None:
            try:
                proc.kill()
            except OSError:
                pass
    if proc.poll() is not None:
        return
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
            proc.wait(timeout=15)
        except (OSError, subprocess.TimeoutExpired):
            pass


_RUNNER_VALUED = {"--min-asserted", "--timeout", "--jobs"}
_RUNNER_BARE = {"--sweep-all", "--vacuity-sweep", "--list"}
_CI_ARGVS: dict[str, list[list[str]]] | None = None


def _gate_tail(tokens: list[str]) -> list[str]:
    """Drop runner flags that a workflow placed after the gate path."""
    out: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in _RUNNER_VALUED and i + 1 < len(tokens):
            i += 2
            continue
        if tok in _RUNNER_BARE or tok in {">", ">>", "<"}:
            i += 2 if tok in {">", ">>", "<"} and i + 1 < len(tokens) else 1
            continue
        out.append(tok)
        i += 1
    return out


def _ci_argv_table() -> dict[str, list[list[str]]]:
    """Concrete gate path -> distinct trailing arg lists, workflow order."""
    global _CI_ARGVS
    if _CI_ARGVS is not None:
        return _CI_ARGVS
    table: dict[str, list[list[str]]] = {}
    workflow_dir = ROOT / ".github" / "workflows"
    if workflow_dir.is_dir():
        names = sorted(workflow_dir.glob("*.yml")) + sorted(workflow_dir.glob("*.yaml"))
        for workflow in names:
            invocations, _exempt = scan_workflow(workflow)
            for inv in invocations:
                if not inv.routed or "$" in inv.path or "{" in inv.path or "}" in inv.path:
                    continue
                tail = inv.segment[inv.offset + len(inv.path):]
                args = _gate_tail(shlex.split(tail, posix=True))
                bucket = table.setdefault(inv.path, [])
                if args not in bucket:
                    bucket.append(args)
    _CI_ARGVS = table
    return table


def _ci_argvs(rel: str) -> list[list[str]]:
    found = _ci_argv_table().get(rel)
    return found if found else [[]]


def _run_wired(gate: str, timeout: int, env: dict[str, str] | None,
               root: pathlib.Path | None = None) -> Result:
    """Run every distinct CI argv. Vacuous if any of them meets the contract."""
    last: Result | None = None
    for args in _ci_argvs(gate):
        last = run_one(gate, args, timeout=timeout, env=env, echo=False, root=root)
        if last.ok:
            return last
    assert last is not None
    return last


def sweep(expect_all_fail: bool, jobs: int, timeout: int) -> int:
    import concurrent.futures as cf

    gates = corpus()
    if not gates:
        print("run_gate: FAIL — the gate corpus is EMPTY; the manifest scope is "
              "wrong, not the corpus. Refusing to pass vacuously.")
        return 1
    env = None
    mode = "sweep-all"
    if expect_all_fail:
        env = _absent_toolchain_env()
        mode = "vacuity-sweep (toolchain absent)"
        static = [g for g in gates if not compiler_dependent(g)]
        gates = [g for g in gates if compiler_dependent(g)]
        print(f"run_gate: {len(static)} corpus gate(s) consult no compiler handle "
              f"and are correctly exempt from the absent-toolchain sweep: "
              f"{', '.join(pathlib.Path(g).name for g in sorted(static)) or '(none)'}")
        # SECOND exemption class, derived the same way and printed the same way:
        # a gate that SETS the handles this leg sets cannot observe this leg.
        own = [g for g in gates if supplies_own_handles(g)]
        gates = [g for g in gates if g not in own]
        print(f"run_gate: {len(own)} corpus gate(s) supply their own toolchain "
              f"handles and are structurally immune to the env-pointed leg: "
              f"{', '.join(pathlib.Path(g).name for g in sorted(own)) or '(none)'}")
        if not gates:
            print("run_gate: FAIL — no compiler-dependent gate found in the corpus; "
                  "the handle detection is wrong, not the corpus. "
                  "Refusing to pass vacuously.")
            return 1
    print(f"run_gate: {mode} over {len(gates)} gates from {MANIFEST.name}")

    def _one(g: str) -> Result:
        if expect_all_fail:
            return _run_wired(g, timeout, env)
        return run_one(g, [], timeout=timeout, env=env, echo=False)

    with cf.ThreadPoolExecutor(max_workers=jobs) as ex:
        results = list(ex.map(_one, gates))

    if expect_all_fail:
        leaked = [r for r in results if r.ok]
        for r in leaked:
            print(f"VACUOUS  {r.gate}: passed with NO compiler present "
                  f"(exit {r.rc}, asserted={r.asserted}) — it asserts nothing about "
                  f"the compiler")
        print(f"env-pointed leg: gates={len(results)} vacuous={len(leaked)} "
              f"rejected={len(results) - len(leaked)}")
        if leaked:
            print(f"FAIL  {len(leaked)} gate(s) pass without a compiler.")
            return 1
        print("PASS  every gate in the corpus failed closed with the toolchain "
              "handles pointed at absent paths.")
        # SECOND negative case. Pointing a handle at nothing only bites a gate
        # that READS the handle; a gate defaulting to a bare `mindc` name silently
        # runs whatever is on PATH. Both legs must be red for the sweep to pass.
        rc2 = _unset_handle_leg(gates, jobs, timeout)
        if rc2 != 0:
            return rc2
        print("PASS  vacuity sweep: BOTH negative cases (handles absent, handles "
              "unset + no `mindc` on PATH) reject every compiler-dependent gate.")
        return 0

    bad = [r for r in results if not r.ok]
    for r in bad:
        print(f"FAIL  {r.gate}: {r.reason}")
    print(f"SDLC-GATE gate-corpus ran={len(results)} fail={len(bad)}")
    if bad:
        return 1
    print("PASS  every gate in the corpus reported a non-zero assertion count.")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--sweep-all", action="store_true",
                    help="run every gate in the corpus under the contract")
    ap.add_argument("--vacuity-sweep", action="store_true",
                    help="THE PROOF: run the corpus with the toolchain absent; "
                         "pass only if EVERY gate is rejected")
    ap.add_argument("--min-asserted", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=0,
                    help="per-gate timeout (default: 900s single, 180s in a sweep)")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--list", action="store_true", help="print the gate corpus")
    ap.add_argument("gate", nargs="?", help="gate script to run")
    ap.add_argument("args", nargs=argparse.REMAINDER)
    ns = ap.parse_args(argv)

    if ns.list:
        for g in corpus():
            print(g)
        return 0
    if ns.vacuity_sweep:
        return sweep(True, ns.jobs, ns.timeout or SWEEP_TIMEOUT)
    if ns.sweep_all:
        return sweep(False, ns.jobs, ns.timeout or SWEEP_TIMEOUT)
    if not ns.gate:
        ap.print_usage(sys.stderr)
        return 2

    r = run_one(ns.gate, ns.args, min_asserted=ns.min_asserted,
                timeout=ns.timeout or DEFAULT_TIMEOUT)
    if r.ok:
        print(f"run_gate: PASS {ns.gate} asserted={r.asserted}")
        return 0
    print(f"run_gate: FAIL {ns.gate}: {r.reason}")
    return 1 if r.rc == 0 else r.rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
