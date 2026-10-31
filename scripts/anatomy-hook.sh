#!/usr/bin/env bash

# STARGA author guard (chained first: a wrong-identity commit must never be created).
#
# Resolved, not hardcoded. This is a tracked file in a PUBLIC repository, so an
# absolute path out of one maintainer's home directory does not belong in it --
# and, worse, an unconditional `bash <missing path> || exit 1` made every commit
# on any other clone fail with a confusing 127. That is what would have happened
# to everyone who followed the documented `core.hooksPath .githooks` install
# once this hook joined that directory.
#
# Order: an explicit MIND_SDLC override, then PATH, then a sibling checkout of
# the guard beside this repo, then one beside $HOME. When none resolves the
# guard is not installed on this machine and cannot assert anything, so the hook
# says so once and continues -- the remaining gates below still run.
_sdlc_bin() {
  if [ -n "${MIND_SDLC:-}" ] && [ -x "${MIND_SDLC}" ]; then printf '%s' "$MIND_SDLC"; return 0; fi
  if command -v sdlc >/dev/null 2>&1; then command -v sdlc; return 0; fi
  local root; root="$(git rev-parse --show-toplevel 2>/dev/null || echo .)"
  local cand
  for cand in "$root/../mind-sdlc/bin/sdlc" "${HOME:-/nonexistent}/mind-sdlc/bin/sdlc"; do
    if [ -x "$cand" ]; then printf '%s' "$cand"; return 0; fi
  done
  return 1
}
_root_for_patterns="$(git rev-parse --show-toplevel 2>/dev/null || echo .)"
if [ -f "$_root_for_patterns/scripts/ai_attribution_patterns.sh" ]; then
  # shellcheck source=scripts/ai_attribution_patterns.sh
  . "$_root_for_patterns/scripts/ai_attribution_patterns.sh"
fi
EXPECTED_AUTHOR_IDENTITY_REGEX="${EXPECTED_AUTHOR_IDENTITY_REGEX:-^STARGA Inc <noreply@star\.ga>$}"
if [ "${STARGA_ALLOW_AUTHOR:-0}" != "1" ]; then
  _author_raw="$(git var GIT_AUTHOR_IDENT 2>/dev/null || true)"
  _author_ident="${_author_raw%%>*}>"
  if [ -z "$_author_raw" ] || ! printf '%s\n' "$_author_ident" | grep -qE "$EXPECTED_AUTHOR_IDENTITY_REGEX"; then
    echo "author-guard: BLOCKED — author identity is '$_author_ident', expected exact 'STARGA Inc <noreply@star.ga>'." >&2
    exit 1
  fi
  _committer_raw="$(git var GIT_COMMITTER_IDENT 2>/dev/null || true)"
  _committer_ident="${_committer_raw%%>*}>"
  if [ -z "$_committer_raw" ] || ! printf '%s\n' "$_committer_ident" | grep -qE "$EXPECTED_AUTHOR_IDENTITY_REGEX"; then
    echo "author-guard: BLOCKED — committer identity is '$_committer_ident', expected exact 'STARGA Inc <noreply@star.ga>'." >&2
    exit 1
  fi
fi
if _guard="$(_sdlc_bin)"; then
  bash "$_guard" precommit "$(git rev-parse --show-toplevel)" || exit 1
else
  echo "pre-commit: identity guard not installed (set MIND_SDLC to enable it); skipping." >&2
fi

# anatomy-hook.sh — the identity/format/attribution/ANATOMY leg of the
# pre-commit hook. Not installed directly any more: .githooks/pre-commit chains
# it, and `git config core.hooksPath .githooks` (CONTRIBUTING.md § Setup) is the
# one install. Chaining rather than copying keeps a single definition of these
# legs — a maintainer who still has .git/hooks/pre-commit symlinked at this file
# runs exactly the same code.
#
# If ANATOMY.md is stale after staged changes, regenerates and stages it.
# Author: STARGA Inc <noreply@star.ga>

set -euo pipefail

# --- rustfmt gate (2026-08-21) ---------------------------------------------
# Block fmt drift BEFORE it reaches CI's `Format Check` job. Fmt drift reded
# main twice in one session; the keystone/build gates a small commit runs do
# NOT check formatting, so a hand-written file that looks clean fails CI. Runs
# only when a Rust file is staged; `cargo fmt --check` is a fast parse+compare
# (no build) in the commit's own worktree (PWD, where git invokes the hook).
# Match the CI `Format Check` gate EXACTLY: `cargo fmt --check` (which respects
# the crate edition + rustfmt.toml). A per-file `rustfmt --edition X --check`
# does NOT load rustfmt.toml or the crate's real edition, so it FALSE-POSITIVES
# on pre-existing lines the CI gate accepts (observed 2026-08-21: an edition-2021
# per-file check flagged an edition-2024 crate + reformatted an unrelated comment,
# blocking a `cargo fmt --check`-clean commit). Correctness > the theoretical
# "undeclared file" gap. Only run when Rust is staged; parse+compare, no build.
if command -v cargo >/dev/null 2>&1 \
   && git diff --cached --name-only --diff-filter=ACMR | grep -q '\.rs$'; then
  if ! cargo fmt --check >/dev/null 2>&1; then
    echo "pre-commit: rustfmt drift (cargo fmt --check fails the CI Format Check)." >&2
    echo "  run 'cargo fmt', then re-stage." >&2
    exit 1
  fi
fi

# --- no-AI-attribution gate (2026-08-21) -----------------------------------
# The "No AI-attribution" CI gate (Docs Claims job) reds main on any named-model
# attribution in a tracked file. It silently reded main for a WHOLE session — a
# model name in a source comment — because the gate ran only in CI, not here.
# Run the authoritative CI script at COMMIT time so a leak fails locally first.
# Runs only where the script exists (this repo); it is a fast git-grep.
if [ -f scripts/check_no_ai_attribution.sh ]; then
  if ! bash scripts/check_no_ai_attribution.sh >/dev/null 2>&1; then
    echo "pre-commit: named-model AI-attribution in a tracked file (STARGA policy)." >&2
    echo "  run 'bash scripts/check_no_ai_attribution.sh' to see it; replace with" >&2
    echo "  'recent research' / 'cross-model review' / a neutral 'audit finding'." >&2
    exit 1
  fi
fi

# The repository root, asked of git rather than derived from this file's path.
# The `../..` walk below is correct only when this script is INVOKED as
# .git/hooks/pre-commit (dirname .git/hooks -> the root); invoked by its own
# name from .githooks/pre-commit it resolves to the repo's PARENT directory,
# where anatomy.sh does not exist and the `[ -x "$SCRIPT" ] || exit 0` guard
# below turns the whole ANATOMY refresh into a silent no-op. A hook always runs
# inside the worktree, so git is the authority; the path walk stays as the
# fallback for a non-git invocation.
ROOT_DIR="$(git rev-parse --show-toplevel 2>/dev/null)" \
  || ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." 2>/dev/null && pwd)"

# Fail CLOSED if the root we resolved is not the worktree git is committing in.
#
# This is not hypothetical. A maintainer clone had .git/hooks/pre-commit
# symlinked at an OLDER copy of this script, one whose ROOT_DIR walked
# `dirname $BASH_SOURCE/../..` BEFORE consulting git. Committing from a `git
# worktree` therefore resolved ROOT_DIR to the MAIN clone, regenerated THAT
# checkout's ANATOMY.md there -- from that checkout's files, including its
# untracked private notes -- and then ran `git add ANATOMY.md`, which honours
# the GIT_INDEX_FILE git exported for the worktree's commit. The other
# checkout's index was staged into this one's commit, and six commits landed
# carrying a file index that described a different tree. Nothing printed a
# warning: every gate had already run, and the substitution happened after them.
#
# A hook cannot make a stale INSTALLED copy of itself correct, but it can refuse
# to write when the two roots disagree, so the next mis-install is a blocked
# commit with a diagnosis instead of a silently wrong public document.
GIT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [ -n "$GIT_ROOT" ] && [ "$ROOT_DIR" != "$GIT_ROOT" ]; then
  echo "pre-commit: ANATOMY refresh resolved a DIFFERENT tree than the commit's." >&2
  echo "  resolved:  $ROOT_DIR" >&2
  echo "  committing: $GIT_ROOT" >&2
  echo "  Regenerating there would stage another checkout's file index into this" >&2
  echo "  commit. This is a stale installed hook: re-install the tracked one with" >&2
  echo "  'git config core.hooksPath .githooks' (CONTRIBUTING.md § Setup) and drop" >&2
  echo "  any .git/hooks/pre-commit symlink." >&2
  exit 1
fi

ANATOMY="$ROOT_DIR/ANATOMY.md"
SCRIPT="$ROOT_DIR/scripts/anatomy.sh"

# Only run if anatomy.sh exists in this repo
[ -x "$SCRIPT" ] || exit 0

# Check if any tracked source files are staged (not just ANATOMY.md itself)
staged_files=$(git diff --cached --name-only --diff-filter=ACMR | grep -v '^ANATOMY.md$' || true)
[ -z "$staged_files" ] && exit 0

# Regenerate
cd "$ROOT_DIR"
"$SCRIPT" . --output ANATOMY.md 2>/dev/null

# Stage the updated ANATOMY.md if it changed
if ! git diff --quiet -- ANATOMY.md 2>/dev/null; then
  git add ANATOMY.md
fi

exit 0
