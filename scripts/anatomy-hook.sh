#!/usr/bin/env bash

# STARGA author guard (chained first: a wrong-identity commit must never be created).
bash "/home/n/mind-sdlc/bin/sdlc" precommit "$(git rev-parse --show-toplevel)" || exit 1

# anatomy-hook.sh — Git pre-commit hook to refresh ANATOMY.md
# Install: ln -sf ../../scripts/anatomy-hook.sh .git/hooks/pre-commit
#   or:   cp scripts/anatomy-hook.sh .git/hooks/pre-commit
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

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." 2>/dev/null && pwd)" \
  || ROOT_DIR="$(git rev-parse --show-toplevel)"

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
