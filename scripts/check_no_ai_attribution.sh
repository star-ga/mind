#!/usr/bin/env bash
# Public-artifact hygiene gate: no AI tool/model named as having worked on MIND.
#
# STARGA policy: public artifacts (README, docs, benchmarks, PR/commit text)
# never attribute work to an AI assistant. Naming a supported MCP *client*
# (e.g. "Claude Code", "Gemini CLI", "Cursor") as an integration target is fine;
# bare review/authorship attributions like "Copilot", "ChatGPT", or
# "N-LLM consensus" tables are not.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"

# Forbidden patterns in tracked Markdown (excluding third-party node_modules and
# the changelog's historical entries). Add patterns conservatively to avoid
# false positives on legitimate technical prose.
PATTERN='copilot|chatgpt|[0-9]+-llm consensus|claude/[a-z-]+-[A-Za-z0-9]{4,}'

hits=$(git grep -inE "$PATTERN" -- '*.md' ':!node_modules' ':!**/node_modules' ':!ANATOMY.md' 2>/dev/null || true)
if [ -n "$hits" ]; then
  echo "::error::AI-attribution found in public docs (forbidden by STARGA policy):"
  echo "$hits"
  echo ""
  echo "Replace named-AI attributions with 'recent research' / 'cross-model review',"
  echo "and remove AI-tool review artifacts. (MCP client names are fine.)"
  exit 1
fi
echo "no-ai-attribution gate: PASS"
