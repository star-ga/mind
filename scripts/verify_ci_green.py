#!/usr/bin/env python3
"""verify_ci_green.py — refuse to release a commit that CI has not proven green.

WHAT THIS ANSWERS
-----------------
"Is this exact commit one that the full gate suite actually passed?"

The Release workflow builds five platform binaries and publishes them under the
project's name.  `ci.yml` — which owns every real gate (Build & Test, the
cross-substrate bit-identity legs, the self-host keystone, the executable
miscompile gate) — is a *separate* workflow with no `tags:` trigger, so nothing
about a tag push causes those gates to run.  Without this check, the release
job's `needs:` could only ever prove that the release job's own builders
compiled; it could not prove anything about correctness or determinism.

So the gate is stated over the commit, not over the workflow graph: find the CI
run for this exact SHA and require it to have completed successfully, require
every job the manifest names to be present and successful inside it, and refuse
when there is no such run at all.

FAIL-CLOSED, INCLUDING ON ABSENCE
---------------------------------
`ci.yml` carries `paths-ignore`, so some commits on `main` legitimately have no
CI run.  "No run" is therefore a real and reachable state, and it is treated as
a BLOCK, never as a pass: a commit CI never examined is exactly the commit a
release must not be cut from.  The escape hatch is to run CI on that ref
(workflow_dispatch) and wait for green — not to skip the check.

WHAT COUNTS AS A QUALIFYING RUN
-------------------------------
  * `head_sha` equals the resolved release commit;
  * the run is this workflow file, in this repository (`head_repository`), so a
    fork cannot supply the evidence for our release;
  * `event` is `push` or `workflow_dispatch`, NOT `pull_request`. A
    `pull_request` run tests the *merge* of the head into the base branch, which
    is a different tree than the commit itself once the base has moved — green
    there does not mean this commit's own tree is green;
  * `status` is `completed` and `conclusion` is `success`. In flight is not
    green.

Additionally, every OTHER first-party workflow (bench gate, crypto vectors,
docs claims, cargo-deny, mindcraft) that has a *completed* run for this SHA must
not have failed.  Those workflows are path-filtered, so requiring them
unconditionally would block legitimate releases; requiring that they are not RED
when they did run costs nothing and closes "release from a SHA whose bench gate
is red".

Usage:
  python3 scripts/verify_ci_green.py --repo star-ga/mind --ref v0.12.0
  python3 scripts/verify_ci_green.py --repo star-ga/mind --ref <40-hex sha>

Exit 0 = release may proceed.  Exit 1 = blocked, with the reason.
Requires the `gh` CLI (preinstalled on GitHub-hosted runners) authenticated via
GH_TOKEN/GITHUB_TOKEN; no third-party Python packages.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / ".github" / "required-ci-jobs.tsv"

SHA_RE = re.compile(r"^[0-9a-f]{40}$")
# A ref name is pasted into an API path, so it is validated here as well as by
# the workflow's stricter vX.Y.Z check — this script is also run by hand, and a
# ref containing '..' or a query separator could pivot the request elsewhere.
REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
ACCEPTED_EVENTS = ("push", "workflow_dispatch")
# Conclusions that mean "this workflow ran and did not pass".
RED = ("failure", "cancelled", "timed_out", "action_required", "startup_failure")


class Blocked(Exception):
    """A release-blocking condition, phrased so the log says what to do next."""


def gh_api(path: str) -> object:
    proc = subprocess.run(
        ["gh", "api", "-H", "Accept: application/vnd.github+json", path],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if proc.returncode != 0:
        raise Blocked(
            f"GitHub API call failed for {path!r}: "
            f"{proc.stderr.strip() or proc.stdout.strip()}"
        )
    return json.loads(proc.stdout)


def required_jobs(manifest: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for raw in manifest.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        jid, _, prefix = raw.partition("\t")
        rows.append((jid.strip(), prefix))
    if not rows:
        raise Blocked(
            f"{manifest} declares no required jobs. An empty manifest would make "
            "this gate vacuous, so it is refused."
        )
    return rows


def resolve_ref(repo: str, ref: str) -> tuple[str, str]:
    """(commit sha, tag name or '') for a tag name or a raw 40-hex sha."""
    if SHA_RE.match(ref):
        return ref, ""
    tag = ref[len("refs/tags/") :] if ref.startswith("refs/tags/") else ref
    if not REF_RE.match(tag) or ".." in tag:
        raise Blocked(f"ref {ref!r} is not a well-formed tag name.")
    try:
        obj = gh_api(f"repos/{repo}/git/ref/tags/{tag}")
    except Blocked as exc:
        raise Blocked(
            f"tag {tag!r} does not exist in {repo}. This workflow never creates "
            "tags: push the tag at a reviewed, CI-green commit first, then release "
            f"it.\n  underlying: {exc}"
        ) from exc
    node = obj["object"]  # type: ignore[index]
    if node["type"] == "tag":  # annotated tag -> dereference to the commit
        node = gh_api(f"repos/{repo}/git/tags/{node['sha']}")["object"]  # type: ignore[index]
    if node["type"] != "commit":
        raise Blocked(f"tag {tag!r} does not point at a commit (got {node['type']!r}).")
    return node["sha"], tag


def ci_runs(repo: str, workflow: str, sha: str) -> list[dict]:
    data = gh_api(
        f"repos/{repo}/actions/workflows/{workflow}/runs?head_sha={sha}&per_page=100"
    )
    return list(data["workflow_runs"])  # type: ignore[index]


def pick_qualifying_run(runs: list[dict], repo: str, workflow: str, sha: str) -> dict:
    if not runs:
        raise Blocked(
            f"no {workflow} run exists for commit {sha}. CI has never examined this "
            "tree (ci.yml has paths-ignore, and no tags: trigger, so a tag push starts "
            "no CI run at all). Dispatch CI on this ref, wait for green, then release."
        )
    rejected: list[str] = []
    qualifying: list[dict] = []
    for run in runs:
        head_repo = (run.get("head_repository") or {}).get("full_name")
        why = None
        if head_repo != repo:
            why = f"head_repository={head_repo!r} (not this repo)"
        elif run.get("event") not in ACCEPTED_EVENTS:
            why = (
                f"event={run.get('event')!r} — a pull_request run tests the merge "
                "commit, not this tree"
            )
        elif run.get("status") != "completed":
            why = f"status={run.get('status')!r} (still in flight)"
        elif run.get("conclusion") != "success":
            why = f"conclusion={run.get('conclusion')!r}"
        if why:
            rejected.append(f"run {run.get('id')}: {why}")
        else:
            qualifying.append(run)
    if not qualifying:
        detail = "\n    ".join(rejected)
        raise Blocked(
            f"{workflow} has run(s) for {sha} but none qualifies as green:\n"
            f"    {detail}"
        )
    return max(qualifying, key=lambda r: r["id"])


def check_required_jobs(repo: str, run: dict, rows: list[tuple[str, str]]) -> None:
    data = gh_api(f"repos/{repo}/actions/runs/{run['id']}/jobs?per_page=100")
    jobs = list(data["jobs"])  # type: ignore[index]
    total = data.get("total_count", len(jobs))  # type: ignore[union-attr]
    if total > len(jobs):
        # One page holds 100 jobs and ci.yml declares far fewer, so this is
        # unreachable today — but a truncated page would silently under-report
        # which gates ran, so it fails closed rather than guessing.
        raise Blocked(
            f"run {run['id']} reports {total} jobs but only {len(jobs)} were "
            "fetched; the required-gate check would be reading a truncated list."
        )
    passed = [j["name"] for j in jobs if j.get("conclusion") == "success"]
    missing: list[str] = []
    lines: list[str] = []
    for jid, prefix in rows:
        hits = [n for n in passed if n == prefix or n.startswith(prefix)]
        if hits:
            lines.append(f"  OK      {jid:34s} <- {', '.join(sorted(hits))}")
        else:
            missing.append(f"{jid} (expected a successful job named {prefix!r}*)")
            lines.append(f"  MISSING {jid:34s} <- no successful job matching {prefix!r}")
    print(f"required CI jobs inside run {run['id']} ({len(jobs)} jobs total):")
    print("\n".join(lines))
    if missing:
        raise Blocked(
            "the CI run is green overall but does not contain every required gate: "
            + "; ".join(missing)
            + ". Either a gate was renamed (update .github/required-ci-jobs.tsv in "
            "the same change) or it did not run for this commit."
        )


def check_no_red_sibling_workflows(repo: str, sha: str, workflow: str) -> None:
    data = gh_api(f"repos/{repo}/actions/runs?head_sha={sha}&per_page=100")
    runs = list(data["workflow_runs"])  # type: ignore[index]
    total = data.get("total_count", len(runs))  # type: ignore[union-attr]
    if total > len(runs):
        # Unlike the green-run search, truncation here would fail OPEN: an
        # unseen page could hold the red run. So it is refused outright.
        raise Blocked(
            f"{total} workflow runs exist for {sha} but only {len(runs)} were "
            "fetched; a red sibling workflow could be on an unread page."
        )
    reds: list[str] = []
    for run in runs:
        path = run.get("path") or ""
        # Only first-party workflow files. `dynamic/...` runs (code scanning,
        # dependency graph, third-party review bots) are not our gates and are
        # not evidence about this tree.
        if not path.startswith(".github/workflows/"):
            continue
        if path.endswith(f"/{workflow}"):
            continue
        if run.get("status") == "completed" and run.get("conclusion") in RED:
            reds.append(f"{path} (run {run['id']}: {run['conclusion']})")
    if reds:
        raise Blocked(
            "another first-party workflow is RED on this commit: "
            + "; ".join(reds)
            + ". These are path-filtered so they are not required to have run, but a "
            "failing one is a stop."
        )
    print("sibling first-party workflows: none red on this commit")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", required=True, help="owner/name")
    ap.add_argument("--ref", required=True, help="tag name or 40-hex commit sha")
    ap.add_argument("--workflow", default="ci.yml", help="gating workflow file name")
    # Local-analysis escape hatch ONLY: lets a maintainer replay this gate against a
    # historical commit whose ci.yml had a different job set. The Release workflow
    # must never pass it — scripts/check_release_gating.py (R6) fails the build if
    # release.yml does, so the shipped gate always reads the checked-in manifest.
    ap.add_argument(
        "--required-jobs",
        default=str(MANIFEST),
        help="alternate manifest path (local replay only; forbidden in release.yml)",
    )
    args = ap.parse_args(argv)

    try:
        rows = required_jobs(Path(args.required_jobs))
        sha, tag = resolve_ref(args.repo, args.ref)
        label = f"{tag} -> {sha}" if tag else sha
        print(f"release candidate: {label}")
        run = pick_qualifying_run(
            ci_runs(args.repo, args.workflow, sha), args.repo, args.workflow, sha
        )
        print(
            f"green {args.workflow} run: {run['id']} "
            f"(event={run['event']}, branch={run.get('head_branch')!r}) "
            f"{run.get('html_url', '')}"
        )
        check_required_jobs(args.repo, run, rows)
        check_no_red_sibling_workflows(args.repo, sha, args.workflow)
    except Blocked as exc:
        # Flush first: the progress lines above and this verdict must interleave
        # in the order they happened when a CI log merges the two streams.
        sys.stdout.flush()
        print(f"\nRELEASE BLOCKED: {exc}", file=sys.stderr)
        sys.stderr.flush()
        return 1

    print(f"\nRELEASE ALLOWED: {label} is CI-green on every required gate.")
    if tag:
        print(f"sha={sha}")
        print(f"tag={tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
