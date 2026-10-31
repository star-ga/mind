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


def main() -> int:
    for case in (
        case_generated_doc_is_in_scope,
        case_control_clean_tree_passes,
        case_anatomy_indexes_only_tracked_files,
        case_bare_vendor_name_in_generated_index_is_red,
        case_bare_vendor_name_outside_the_artifact_still_passes,
        case_missing_artifact_fails_closed,
        case_committed_anatomy_names_no_model,
        case_committed_anatomy_matches_the_tracked_tree,
        case_author_identity_guard_rejects_trailing_dot_and_accepts_exact,
        case_identity_guard_rejects_foreign_author_merge_in_range,
        case_identity_guard_accepts_house_identity_merge_in_range,
        case_identity_guard_keeps_synthetic_ci_merge_out_of_scope,
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
