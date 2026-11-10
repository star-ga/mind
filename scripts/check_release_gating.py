#!/usr/bin/env python3
"""check_release_gating.py — machine-checked supply-chain contract for the
Release workflow.

WHY THIS EXISTS
---------------
`.github/workflows/release.yml` publishes checksum-bearing binaries for five
platforms under the project's own name.  Whatever it will build and upload is,
for every downstream user, "what MIND released".  That makes the set of
conditions under which it is *allowed to run* a security property, not a
convenience.

Before this lint, the workflow had two holes:

  1. The publishing job gated on `needs: build` alone, and the build job's whole
     step list was Checkout / Install Rust / Cache / Install cross / Build /
     Package / Upload.  No test, no keystone, no cross-substrate identity.  CI
     lives in a *different* workflow (`ci.yml`) that has no `tags:` trigger, so
     on the tag-push path no CI run is even started for that ref — the release
     could be cut from a commit whose CI was red, skipped by `paths-ignore`, or
     never run at all.
  2. `workflow_dispatch` took a free-text `version` string and then created the
     tag itself (`git tag -a v$version && git push origin v$version`) with the
     workflow-level `contents: write` token.  A dispatch on any branch therefore
     minted a brand-new tag at that branch's tip and published a non-draft
     release from it, with no requirement that the commit had ever been
     reviewed, merged, or tested.

Neither hole is visible by reading a green Actions page, and prose in a comment
cannot be executed.  This lint states both properties as assertions that fail
the build, in the same spirit as examples/mindc_mind/smoke_wiring_lint.py.

WHAT IS CHECKED
---------------
  R1  release.yml never creates or pushes a git tag.
  R2  Every job that publishes a GitHub Release transitively `needs:` the gate
      job, and the gate job actually invokes scripts/verify_ci_green.py.
  R3  A tag push is the only trigger: no workflow_dispatch (which used to take
      a free-text version and mint its tag, and in any form runs in the cache
      scope of the branch it is dispatched on), and no branch push.
  R4  `contents: write` is not granted workflow-wide; only the publishing job
      may hold it (the builders and the gate run read-only).
  R5  .github/required-ci-jobs.tsv and every workflow it names agree in both
      directions, so a deleted or renamed gate cannot silently drop out of what
      a release is verified against. A row may qualify its job id as
      `<workflow file>:<job id>`; an unqualified row means ci.yml, which is why
      every pre-existing row keeps its meaning unchanged.
  R6  scripts/verify_ci_green.py exists and reads that same manifest.
  R7  Cargo.lock is committed (tracked, not ignored) and every cargo/cross call
      in a `run:` block of a workflow (.yml or .yaml) or a local composite action
      passes `--locked` (or `--frozen`) before any `--`, unless its subcommand is
      in NON_RESOLVING. That list is the exception, so a plugin subcommand
      (`cargo deny`, which an earlier list of resolving subcommands left out) is
      held to the rule by default. Before this rule the lockfile was gitignored
      and release.yml built unlocked, so the shipped fips204 / fips205 were
      whatever 0.4.x was newest on tag day, not what CI tested. One command may be
      exempted by a `# lockfile-exempt: <reason>` line directly above it; an
      exemption above a locked command is stale and fails.
  R8  ci.yml's release_profile job is a faithful pre-tag copy of release.yml's
      Linux legs: the same runner/target pairs (x86_64 AND aarch64, each on a
      runner of its own architecture, so nothing is cross-built unexecuted); a
      `cargo build` equal to release.yml's argument for argument, inline env
      prefix included (only the order of the feature names may differ), which
      carries `--release --locked --bin mindc --target`; a signing-smoke `cargo
      test --test evidence_release_smoke` with release.yml's feature selection
      and `--release --locked --target`; and both run the shared post-build smoke
      action on the packaged archive. Step/job `env:` blocks and the toolchain
      action are not compared.

A second mode resolves dependency graphs and so needs cargo:
  --keystone-pqc-free  every feature selection (`--no-default-features`,
      `--all-features`, every `--features`/`-F`) the keystone
      (mindcraft_self_host) and byte-identity (cross_substrate_identity) jobs
      build with resolves NEITHER fips204 NOR fips205 under `cargo tree --locked
      --target all -e normal,build,dev`; positive control: release.yml's
      selection, read the same way, resolves BOTH.

Usage:
  python3 scripts/check_release_gating.py           # check (exit 1 on violation)
  python3 scripts/check_release_gating.py --print   # show what it derived
  python3 scripts/check_release_gating.py --keystone-pqc-free
"""

from __future__ import annotations

import fnmatch
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
# The repo tracks no Python build products and .gitignore has no __pycache__
# rule; importing a sibling module would create one on every local run.
sys.dont_write_bytecode = True

from gate_runner_wiring_lint import (  # noqa: E402
    join_continuations,
    run_block_lines,
    split_segments,
    strip_shell_comment,
)
from workflow_scan import (  # noqa: E402
    blocks,
    job_display_name,
    name_prefix,
    scalar_list,
    workflow_jobs,
)

ROOT = Path(__file__).resolve().parents[1]
RELEASE_YML = ROOT / ".github" / "workflows" / "release.yml"
WORKFLOW_DIR = ROOT / ".github" / "workflows"
CI_YML = WORKFLOW_DIR / "ci.yml"
# A manifest row without a `<workflow>:` qualifier names a job of this workflow.
DEFAULT_WORKFLOW = "ci.yml"
MANIFEST = ROOT / ".github" / "required-ci-jobs.tsv"
VERIFIER = ROOT / "scripts" / "verify_ci_green.py"
LOCKFILE = "Cargo.lock"
ACTIONS_DIR = ROOT / ".github" / "actions"
# A cargo (or cross) call: optional `+toolchain` / global flags, then the subcommand.
CARGO_RE = re.compile(
    r"(?<![A-Za-z0-9_./-])(?:cargo|cross)((?:\s+[+-]\S+)*)\s+([A-Za-z][A-Za-z0-9_-]*)"
    r"(?![A-Za-z0-9_-])"
)
# Subcommands that never resolve the dependency graph. Every other subcommand,
# plugins included, must be locked: a list of the RESOLVING ones missed `deny`.
NON_RESOLVING = frozenset({"fmt", "help", "version"})
LOCK_FLAG_RE = re.compile(r"(?<![A-Za-z0-9_-])--(?:locked|frozen)(?![A-Za-z0-9_-])")
LOCK_EXEMPT_RE = re.compile(r"^#\s*lockfile-exempt:\s*(\S.{19,})$")
RELEASE_PROFILE_JOB = "release_profile"
SMOKE_ACTION = "./.github/actions/release-artifact-smoke"
KEYSTONE_JOBS = ("mindcraft_self_host", "cross_substrate_identity")
PQC_CRATES = ("fips204", "fips205")
# `${{ matrix.target }}` is folded to `${{matrix.target}}` so it stays one token.
GHA_EXPR_RE = re.compile(r"\$\{\{(.*?)\}\}")
SELECTION_FLAGS = ("--no-default-features", "--all-features")
# What every release-profile cargo call must carry; a build also BUILD_BIN.
RELEASE_ARGS = (("--release",), ("--locked",), ("--target", "${{matrix.target}}"))
# One string, split: run_gate.py's vacuity sweep reads a bare quoted binary name
# in a gate's code as a compiler handle, and this gate never runs the compiler.
BUILD_BIN = tuple("--bin mindc".split())
SIGNING_SMOKE = ("--test", "evidence_release_smoke")

# Actions/commands that create a GitHub Release. A job using any of these is a
# publishing job and carries the full weight of the gate.
PUBLISH_MARKERS = (
    "softprops/action-gh-release",
    "ncipollo/release-action",
    "gh release create",
)

# Any of these appearing in a `run:` block means the workflow mints refs itself.
TAG_MINTING_RE = re.compile(
    r"git\s+tag\b|git\s+push\s+\S+\s+(?:refs/tags/|v?\$|['\"]?v[0-9])"
    r"|api\s+\S*/git/refs\b.*-X\s*POST|-X\s*POST\s+\S*/git/refs\b",
)


# The rules this gate owns, and the one-line summary each prints when it has
# ACTUALLY been evaluated. The summary used to be six unconditional `print`
# calls at the end of main(): a hand-typed list that reported all six rules as
# checked whether or not their code still existed, and published no assertion
# count at all (`asserted=0` under scripts/run_gate.py — a gate that, by the
# shared contract, checked nothing). Each rule now records that it ran, and a
# rule that records nothing is a failure, so deleting a rule's body can no
# longer leave its reassuring line behind.
RULES: dict[str, str] = {
    "R1": "release.yml mints no tags",
    "R2": "publish + build gated on the CI-green verifier",
    "R3": "tag push is the only trigger (no dispatch, no branch push)",
    "R4": "contents:write scoped to the publishing job",
    "R5": "required-ci-jobs.tsv == every workflow it names (both directions)",
    "R6": "verify_ci_green.py present and reading that manifest",
    "R7": "Cargo.lock committed; every resolving workflow/action cargo call is --locked",
    "R8": "release_profile runs release.yml's Linux build verbatim + the signing smoke",
}


def _fail(errors: list[str], rule: str, msg: str) -> None:
    errors.append(f"[{rule}] {msg}")


def _ran(ran: set[str], rule: str) -> None:
    """Record that `rule`'s body was reached and evaluated."""
    ran.add(rule)


def code(body: str) -> str:
    """The job body with comment lines dropped.

    Every rule below asks what a job DOES, and a comment saying
    "see scripts/verify_ci_green.py" is not a job that runs it. Matching on the
    raw text let a job keep its gate status by mentioning the verifier in prose
    while the `run:` line had been replaced — caught by a mutation test, so the
    stripping is applied everywhere a rule reads a body.
    """
    return "\n".join(
        ln for ln in body.splitlines() if not ln.strip().startswith("#")
    )


def parse_manifest() -> list[tuple[str, str, str]]:
    """Rows as (workflow file, job id, check-run name prefix).

    Not every release-gating job lives in ci.yml: the public-artifact hygiene
    gates run in docs-claims.yml, on that workflow's own unfiltered trigger. A
    row may therefore qualify its job id as `<workflow file>:<job id>`. An
    unqualified id still means DEFAULT_WORKFLOW, so no existing row changes.
    """
    rows: list[tuple[str, str, str]] = []
    for lineno, raw in enumerate(
        MANIFEST.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw.rstrip("\n")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
            raise SystemExit(
                f"{MANIFEST}:{lineno}: expected '<job id>\\t<name prefix>', got: {line!r}"
            )
        jid = parts[0].strip()
        workflow, sep, tail = jid.partition(":")
        if not sep:
            workflow, tail = DEFAULT_WORKFLOW, jid
        rows.append((workflow, tail, parts[1]))
    return rows


def check_release_workflow(errors: list[str], ran: set[str]) -> None:
    text = RELEASE_YML.read_text(encoding="utf-8")
    top = blocks(text, 0)
    jobs = workflow_jobs(RELEASE_YML)

    _ran(ran, "R1")
    # R1 — no self-minted tags anywhere in the workflow. Comment lines are
    # excluded: the file documents the human tag-push procedure in prose, and a
    # lint that cannot tell prose from a `run:` step would force the docs out.
    for jid, body in jobs.items():
        offending = [
            ln.strip() for ln in code(body).splitlines() if TAG_MINTING_RE.search(ln)
        ]
        if offending:
            _fail(
                errors,
                "R1",
                f"job '{jid}' creates or pushes a git ref: {offending}. "
                "A release must be cut from a tag that already exists on reviewed, "
                "CI-green history — the workflow must never mint the ref it publishes.",
            )

    _ran(ran, "R2")
    # R2 — publishing jobs must transitively depend on the gate job.
    gate_jobs = [jid for jid, body in jobs.items() if "verify_ci_green.py" in code(body)]
    for jid in gate_jobs:
        if "--required-jobs" in code(jobs[jid]):
            _fail(
                errors,
                "R2",
                f"gate job '{jid}' passes --required-jobs to verify_ci_green.py. That "
                "flag is a local-replay escape hatch; the shipped gate must read the "
                "checked-in .github/required-ci-jobs.tsv so the required set is the "
                "reviewed one.",
            )
    if not gate_jobs:
        _fail(
            errors,
            "R2",
            "no job in release.yml runs scripts/verify_ci_green.py. Publication "
            "must be gated on a green CI run for the exact commit being released.",
        )

    needs: dict[str, list[str]] = {
        jid: scalar_list(blocks(body, 4).get("needs", "")) for jid, body in jobs.items()
    }

    def transitive(jid: str, seen: set[str] | None = None) -> set[str]:
        seen = seen if seen is not None else set()
        for dep in needs.get(jid, []):
            if dep not in seen:
                seen.add(dep)
                transitive(dep, seen)
        return seen

    publishing = [
        jid
        for jid, body in jobs.items()
        if any(marker in code(body) for marker in PUBLISH_MARKERS)
    ]
    if not publishing:
        _fail(
            errors,
            "R2",
            "no publishing job found in release.yml — the lint cannot confirm what "
            "it is meant to gate. Update PUBLISH_MARKERS if the release action changed.",
        )
    for jid in publishing:
        deps = transitive(jid)
        if not any(g in deps or g == jid for g in gate_jobs):
            _fail(
                errors,
                "R2",
                f"publishing job '{jid}' does not depend on the CI-green gate "
                f"{gate_jobs or '(missing)'}; its dependency closure is "
                f"{sorted(deps) or '(none)'}.",
            )
    # Builders must be gated too: a build that starts before the gate wastes five
    # runners on an unverified commit and produces artifacts that look official.
    for jid, body in jobs.items():
        if jid in gate_jobs or jid in publishing:
            continue
        if "cargo build" in code(body) or "cross build" in code(body):
            deps = transitive(jid)
            if not any(g in deps for g in gate_jobs):
                _fail(
                    errors,
                    "R2",
                    f"build job '{jid}' runs before the CI-green gate "
                    f"{gate_jobs}; add it to `needs:`.",
                )

    _ran(ran, "R3")
    # R3 — a tag push is the only trigger. Any other event either mints the ref
    # it releases (the old free-text `version` dispatch) or runs in the cache
    # scope of a branch: a dispatch from main executes the tagged commit with a
    # runtime token that can write main's cache, which every later main run
    # restores. A tag-push run is scoped to its tag alone.
    triggers = blocks(top.get("on", ""), 2)
    if not triggers:
        _fail(errors, "R3", "release.yml declares no `on:` trigger to check.")
    for event in sorted(set(triggers) - {"push"}):
        _fail(
            errors,
            "R3",
            f"release.yml is triggered by `{event}`; a release must run only on a "
            "tag push, in the tag's own cache scope. Re-run a tag-push run instead "
            "of dispatching one.",
        )
    if "push" in triggers:
        push = blocks(triggers["push"], 4)
        if set(push) != {"tags"}:
            _fail(
                errors,
                "R3",
                f"release.yml's push trigger must filter on `tags:` alone; found "
                f"{sorted(push) or '(no filter)'}.",
            )

    _ran(ran, "R4")
    # R4 — no workflow-wide write token.
    wf_perms = top.get("permissions", "")
    if re.search(r"contents:\s*write", wf_perms):
        _fail(
            errors,
            "R4",
            "workflow-level `permissions:` grants `contents: write`, so every job "
            "(including the five build runners) holds a token that can push refs. "
            "Scope write to the publishing job only.",
        )
    for jid in publishing:
        job_perms = blocks(jobs[jid], 4).get("permissions", "")
        if not re.search(r"contents:\s*write", job_perms):
            _fail(
                errors,
                "R4",
                f"publishing job '{jid}' does not declare `permissions: contents: write`; "
                "it cannot create the release with a read-only workflow token.",
            )


def check_manifest_matches_workflows(
    errors: list[str], show: bool, ran: set[str]
) -> None:
    rows = parse_manifest()
    _ran(ran, "R5")

    # One map per workflow the manifest names. Both directions are enforced for
    # EVERY such workflow, not just ci.yml: once a workflow is part of what a
    # release is verified against, a new job inside it that nobody listed is the
    # same silent gap as a new ci.yml job with no row.
    by_workflow: dict[str, dict[str, str]] = {}
    for workflow, _, _ in rows:
        if workflow in by_workflow:
            continue
        path = WORKFLOW_DIR / workflow
        if not path.is_file():
            _fail(
                errors,
                "R5",
                f"{MANIFEST.relative_to(ROOT)} names workflow {workflow!r}, which does "
                "not exist under .github/workflows/.",
            )
            by_workflow[workflow] = {}
            continue
        by_workflow[workflow] = {
            jid: name_prefix(job_display_name(body) or jid)
            for jid, body in workflow_jobs(path).items()
        }

    if show:
        for workflow, jobs in by_workflow.items():
            print(f"{workflow} jobs -> required check-run name prefix")
            for jid, prefix in jobs.items():
                print(f"  {jid:34s} {prefix!r}")
        print(f"\n{MANIFEST.relative_to(ROOT)} rows")
        for workflow, jid, prefix in rows:
            print(f"  {workflow}:{jid:34s} {prefix!r}")

    declared: dict[str, dict[str, str]] = {}
    for workflow, jid, prefix in rows:
        declared.setdefault(workflow, {})[jid] = prefix

    for workflow, jobs in by_workflow.items():
        for jid in jobs:
            if jid not in declared.get(workflow, {}):
                _fail(
                    errors,
                    "R5",
                    f"{workflow} job '{jid}' has no row in "
                    f"{MANIFEST.relative_to(ROOT)} — a new gate that no release is "
                    "verified against. Add it (or justify the gap).",
                )
        for jid, prefix in declared.get(workflow, {}).items():
            if jid not in jobs:
                _fail(
                    errors,
                    "R5",
                    f"{MANIFEST.relative_to(ROOT)} requires job '{jid}', which no longer "
                    f"exists in {workflow}. A gate was deleted or renamed; re-decide "
                    "explicitly.",
                )
            elif jobs[jid] != prefix:
                _fail(
                    errors,
                    "R5",
                    f"job '{jid}': manifest prefix {prefix!r} != {workflow} name prefix "
                    f"{jobs[jid]!r}. The gate would look for a check-run name that "
                    "is never produced.",
                )

    _ran(ran, "R6")
    # R6 — the runtime verifier exists and consumes this same manifest.
    if not VERIFIER.exists():
        _fail(errors, "R6", f"{VERIFIER.relative_to(ROOT)} is missing.")
    else:
        vtext = VERIFIER.read_text(encoding="utf-8")
        if "required-ci-jobs.tsv" not in vtext:
            _fail(
                errors,
                "R6",
                f"{VERIFIER.relative_to(ROOT)} does not reference "
                "required-ci-jobs.tsv; the static manifest and the runtime gate "
                "would be two unrelated sources of truth.",
            )


def unlocked_cargo_commands(path: Path) -> tuple[list[str], list[str]]:
    """(unlocked resolving commands, stale exemptions) in one workflow's run blocks."""
    unlocked: list[str] = []
    stale: list[str] = []
    exempt_at: int | None = None
    for lineno, cmd in join_continuations(run_block_lines(path.read_text(encoding="utf-8"))):
        if cmd.startswith("#"):
            if LOCK_EXEMPT_RE.match(cmd):
                exempt_at = lineno
            continue
        hits = []
        for seg in split_segments(strip_shell_comment(cmd)):
            for m in CARGO_RE.finditer(seg):
                # Only flags before a bare `--` are cargo's; after it they are libtest's.
                own = re.split(r"\s--(?:\s|$)", seg[m.start():], maxsplit=1)[0]
                if m.group(2) not in NON_RESOLVING and not LOCK_FLAG_RE.search(own):
                    hits.append(f"{path.relative_to(ROOT)}:{lineno}: {seg.strip()}")
                    break
        if exempt_at is not None and not hits:
            stale.append(f"{path.relative_to(ROOT)}:{exempt_at}")
        if exempt_at is None:
            unlocked.extend(hits)
        exempt_at = None
    return unlocked, stale


def lockfile_ignored(gitignore: Path) -> list[str]:
    """Non-negated .gitignore patterns that match the root Cargo.lock."""
    if not gitignore.is_file():
        return []
    hits = []
    for raw in gitignore.read_text(encoding="utf-8").splitlines():
        pat = raw.strip()
        if not pat or pat.startswith(("#", "!")):
            continue
        bare = pat.removeprefix("**/").removeprefix("/").rstrip("/")
        if fnmatch.fnmatchcase(LOCKFILE, bare):
            hits.append(pat)
    return hits


def check_lockfile(errors: list[str], ran: set[str]) -> None:
    _ran(ran, "R7")
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", LOCKFILE],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if tracked.returncode != 0 or not (ROOT / LOCKFILE).is_file():
        _fail(errors, "R7", f"{LOCKFILE} is not a tracked file; release and CI builds "
              "would each resolve the newest semver-compatible dependencies.")
    for pat in lockfile_ignored(ROOT / ".gitignore"):
        _fail(errors, "R7", f".gitignore pattern {pat!r} ignores {LOCKFILE}.")
    # A local composite action's `run:` steps execute inside the workflow using it.
    workflows = sorted(p for ext in ("yml", "yaml") for p in (
        *WORKFLOW_DIR.glob(f"*.{ext}"), *ACTIONS_DIR.glob(f"*/action.{ext}")))
    if not workflows:
        _fail(errors, "R7", "no workflow found to scan — a broken scan, not a clean tree.")
    for path in workflows:
        unlocked, stale = unlocked_cargo_commands(path)
        for hit in unlocked:
            _fail(errors, "R7", f"cargo command resolves without --locked: {hit}")
        for at in stale:
            _fail(errors, "R7", f"stale `# lockfile-exempt:` at {at}: the next command is "
                  "already locked or resolves nothing; remove the exemption.")


class CargoCall(NamedTuple):
    sub: str  # the subcommand
    args: list[str]  # cargo's own arguments: everything before a bare `--`
    prefix: tuple[str, ...]  # what precedes `cargo` in its segment (env assignments)


def cargo_calls(cmd: str) -> list[CargoCall]:
    """Every cargo/cross call in one shell command. A call shlex cannot split
    stops the gate: an unread command cannot be judged equal to another."""
    calls: list[CargoCall] = []
    for seg in split_segments(strip_shell_comment(cmd)):
        seg = GHA_EXPR_RE.sub(lambda e: "${{" + "".join(e.group(1).split()) + "}}", seg)
        for m in CARGO_RE.finditer(seg):
            try:
                args = shlex.split(m.group(1)) + shlex.split(seg[m.end():])
                prefix = tuple(shlex.split(seg[:m.start()]))
            except ValueError as exc:
                raise SystemExit(f"cannot split cargo call {seg.strip()!r}: {exc}") from exc
            cut = args.index("--") if "--" in args else len(args)
            calls.append(CargoCall(m.group(2), args[:cut], prefix))
    return calls


def job_calls(job_body: str, verb: str | None = None) -> list[CargoCall]:
    """The job's cargo `verb` calls (every resolving call when None)."""
    rows = join_continuations(run_block_lines(job_body))
    return [call for _, c in rows for call in cargo_calls(c)
            if call.sub == verb or (verb is None and call.sub not in NON_RESOLVING)]


def selection(args: list[str]) -> tuple[bool, bool, str]:
    """(--no-default-features, --all-features, sorted feature names) of one call."""
    names: list[str] = []
    for i, tok in enumerate(args):
        if tok in ("--features", "-F") and i + 1 < len(args):
            names += re.split(r"[ ,]+", args[i + 1])
        elif tok.startswith("--features="):
            names += re.split(r"[ ,]+", tok.partition("=")[2])
        elif tok.startswith("-F") and len(tok) > 2:
            names += re.split(r"[ ,]+", tok[2:])
    return (SELECTION_FLAGS[0] in args, SELECTION_FLAGS[1] in args,
            " ".join(sorted({n for n in names if n})))


def canonical(call: CargoCall) -> tuple:
    """Comparison key of one call: its env prefix, every other argument in order,
    and its feature selection (so only the order of feature names is free)."""
    rest: list[str] = []
    value_next = False
    for tok in call.args:
        if value_next:
            value_next = False
        elif tok in ("--features", "-F"):
            value_next = True
        elif not tok.startswith(("--features=", "-F")) and tok not in SELECTION_FLAGS:
            rest.append(tok)
    return call.prefix, tuple(rest), selection(call.args)


def has(args: list[str], need: tuple[str, ...]) -> bool:
    return any(tuple(args[i:i + len(need)]) == need for i in range(len(args)))


def describe(sel: tuple[bool, bool, str]) -> str:
    flags = [f for f, on in zip(SELECTION_FLAGS, sel) if on]
    return " ".join(flags + [f"--features {sel[2]!r}"])


def matrix_rows(job_body: str) -> list[dict[str, str]]:
    """Scalar `- key: value` items of the job's `matrix: include:` list."""
    lines = job_body.splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.strip() == "include:"), None)
    if start is None:
        return []
    base = len(lines[start]) - len(lines[start].lstrip())
    rows: list[dict[str, str]] = []
    for ln in lines[start + 1:]:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        if len(ln) - len(ln.lstrip()) <= base:
            break
        m = re.match(r"^(-\s+)?([A-Za-z0-9_-]+):\s*(.*?)\s*$", s)
        if not m:
            continue
        if m.group(1) or not rows:
            rows.append({})
        rows[-1][m.group(2)] = m.group(3).strip("\"'")
    return rows


def native_pair(runner: str, target: str) -> bool:
    return target.startswith("aarch64-") == runner.endswith("-arm")


def release_build() -> tuple[str, tuple | None]:
    """release.yml's build job body and the comparison key of its ONE cargo build."""
    for body in workflow_jobs(RELEASE_YML).values():
        keys = {canonical(call) for call in job_calls(body, "build")}
        if keys:
            return body, keys.pop() if len(keys) == 1 else None
    return "", None


def check_release_profile(errors: list[str], ran: set[str]) -> None:
    _ran(ran, "R8")
    rel_build, rel_key = release_build()
    if rel_key is None or not rel_key[2][2]:
        _fail(errors, "R8", "release.yml must run ONE cargo build with a --features set; "
              f"found {[c.args for c in job_calls(rel_build, 'build')]}.")
    rel_linux = {(r.get("os", ""), r.get("target", "")) for r in matrix_rows(rel_build)
                 if r.get("os", "").startswith("ubuntu")}
    ci = workflow_jobs(CI_YML).get(RELEASE_PROFILE_JOB)
    if ci is None:
        _fail(errors, "R8", f"ci.yml has no `{RELEASE_PROFILE_JOB}` job: the release "
              "feature set first compiles, and aarch64 first runs, on the tag push.")
        return
    ci_pairs = {(r.get("runner", ""), r.get("target", "")) for r in matrix_rows(ci)}
    if ci_pairs != rel_linux:
        _fail(errors, "R8", f"release_profile runs {sorted(ci_pairs)}; release.yml's Linux "
              f"legs are {sorted(rel_linux)}.")
    archs = {t.split("-", 1)[0] for _, t in ci_pairs}
    if not {"x86_64", "aarch64"} <= archs:
        _fail(errors, "R8", f"release_profile must cover x86_64 AND aarch64, has {sorted(archs)}.")
    for runner, target in sorted(ci_pairs | rel_linux):
        if not native_pair(runner, target):
            _fail(errors, "R8", f"{target} on {runner} is not native: that binary "
                  "cannot be executed by the smoke before it ships.")
    builds = job_calls(ci, "build")
    if not builds:
        _fail(errors, "R8", "release_profile runs no `cargo build`.")
    def shown(call: CargoCall) -> str:
        return " ".join((*call.prefix, "cargo", call.sub, *call.args))

    labelled = [("release.yml", c) for c in job_calls(rel_build, "build")]
    for label, call in labelled + [("release_profile", c) for c in builds]:
        missing = [" ".join(n) for n in (*RELEASE_ARGS, BUILD_BIN) if not has(call.args, n)]
        if missing:
            _fail(errors, "R8", f"{label} `{shown(call)}` lacks {missing}.")
    for call in builds:
        if canonical(call) != rel_key:
            _fail(errors, "R8", f"release_profile `{shown(call)}` is not release.yml's "
                  f"`{shown(labelled[0][1]) if labelled else '(none)'}`: every argument "
                  "and env prefix must match; only the feature names may be reordered.")
    tests = [c.args for c in job_calls(ci, "test")]
    if not any(has(args, SIGNING_SMOKE) for args in tests):
        _fail(errors, "R8", f"release_profile runs no `cargo test {' '.join(SIGNING_SMOKE)}`: "
              "the packaged binary is never signed, verified or refused a stripped leg.")
    for args in tests:
        missing = [" ".join(n) for n in RELEASE_ARGS if not has(args, n)]
        if missing or rel_key is None or selection(args) != rel_key[2]:
            _fail(errors, "R8", f"release_profile `cargo test {' '.join(args)}` must carry "
                  f"{[' '.join(n) for n in RELEASE_ARGS]} and release.yml's selection "
                  f"({describe(rel_key[2]) if rel_key else 'underivable'}); lacks {missing}.")
    for label, body in (("release.yml build", rel_build), ("ci.yml release_profile", ci)):
        if f"uses: {SMOKE_ACTION}" not in code(body):
            _fail(errors, "R8", f"{label} does not run {SMOKE_ACTION} on its archive.")


def tree_packages(sel: tuple[bool, bool, str]) -> set[str] | None:
    cmd = ["cargo", "tree", "--locked", "--target", "all", "-e", "normal,build,dev",
           "--prefix", "none", "--format", "{p}"]
    cmd += [flag for flag, on in zip(SELECTION_FLAGS, sel) if on]
    cmd += ["--features", sel[2]] if sel[2] else []
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
    print(f"$ {' '.join(cmd)}  -> rc={proc.returncode}")
    if proc.returncode != 0:
        print(proc.stderr.strip())
        return None
    return {ln.split()[0] for ln in proc.stdout.splitlines() if ln.strip()}


def keystone_pqc_free() -> int:
    ok = True

    def verdict(cond: bool, label: str) -> None:
        nonlocal ok
        print(f"[{'PASS' if cond else 'FAIL'}] {label}")
        ok = ok and cond

    jobs = workflow_jobs(CI_YML)
    sets: set[tuple[bool, bool, str]] = set()
    for jid in KEYSTONE_JOBS:
        calls = job_calls(jobs.get(jid, ""))
        verdict(bool(calls), f"{jid}: feature sets derived from its cargo commands")
        sets |= {selection(call.args) for call in calls}
    key = release_build()[1]
    release = key[2] if key else None
    verdict(release is not None and bool(release[2]),
            f"release.yml feature selection derived: {describe(release) if release else None}")
    shipped = tree_packages(release) if release else None
    for crate in PQC_CRATES:
        verdict(shipped is not None and crate in shipped,
                f"positive control: the release selection resolves {crate}")
    for sel in sorted(sets):
        pkgs = tree_packages(sel)
        verdict(pkgs is not None and "mind" in pkgs, f"keystone set {describe(sel)} resolves")
        for crate in PQC_CRATES:
            verdict(pkgs is not None and crate not in pkgs,
                    f"keystone set {describe(sel)} does not compile {crate}")
    return 0 if ok else 1


def main(argv: list[str]) -> int:
    if "--keystone-pqc-free" in argv:
        return keystone_pqc_free()
    show = "--print" in argv
    errors: list[str] = []
    ran: set[str] = set()
    check_release_workflow(errors, ran)
    check_manifest_matches_workflows(errors, show, ran)
    check_lockfile(errors, ran)
    check_release_profile(errors, ran)

    # A rule whose body was never reached asserted nothing, and a gate that
    # asserts nothing must not report a contract as upheld.
    for rule in RULES:
        if rule not in ran:
            _fail(
                errors,
                rule,
                "this rule's body was never evaluated — it was deleted, renamed "
                "or short-circuited. A contract nobody checked is not a contract "
                "that holds.",
            )

    if errors:
        print("release gating contract: FAIL", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        print(
            f"\n{len(errors)} violation(s). See the module docstring in "
            "scripts/check_release_gating.py for what each rule protects.",
            file=sys.stderr,
        )
        return 1

    print("release gating contract: OK")
    # One verdict line per rule that actually ran — the evidence the shared gate
    # runner counts, rather than a fixed list printed regardless.
    for rule, summary in RULES.items():
        print(f"  [PASS] {rule} {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
