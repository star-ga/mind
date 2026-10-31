# shellcheck shell=bash
# ai_attribution_patterns.sh — the SINGLE definition of "a model or tool credit".
#
# Sourced, never executed: shell variables only, no side effects, no output, no
# cd, no exit. Three consumers must agree on ONE definition or the policy has a
# hole the size of whichever consumer was forgotten:
#   * scripts/check_no_ai_attribution.sh  — tracked FILE contents (CI + hook)
#   * scripts/check_commit_messages.sh    — commit MESSAGES over a rev-range (CI)
#   * scripts/commit-msg-hook.sh          — the message being written, locally
# The file gate existed alone for months; commit messages were checked by nothing
# at all (no commit-msg hook, no CI step read `git log`), so the same credit that
# is forbidden in a comment could enter history permanently in the message beside
# it. Two hand-copied regex lists would have re-opened that gap on the first edit
# to either one, which is why this file exists rather than a second copy.
#
# STARGA policy: naming a supported MCP / CLI *client* as an integration target,
# or an LLM backend a tool shells out to at runtime, is fine. Bare review or
# authorship attributions are not.

# Attribution-shaped patterns. The first token below (= an internal model
# codename) is never a legitimate integration target in these repos, so it is
# flagged bare (word-bounded, to spare "affable"/"ineffable"). Other vendors are
# flagged only when adjacent to an authorship/review verb, so legitimate words
# ("grok" the verb, "opus", an integration line naming a CLI) do not
# false-positive.
PATTERN='\bfable\b|copilot|chatgpt|[0-9]+[- ]llm consensus|claude/[a-z-]+-[A-Za-z0-9]{4,}|\b(deepseek|mistral|grok|gemini|gpt|opus|sonnet|haiku|kimi|qwen|nemotron|glm|moonshot|zhipu|anthropic|openai)[- ]?(audit|panel|review|finding|consensus|converged|driven|flagged|authored)\b'

# Second pattern: PROVENANCE-shaped credits, which the verb-adjacency rule above
# structurally cannot see. That rule requires the vendor token to sit IMMEDIATELY
# before an authorship verb ([- ]? separator), so a credit written as a noun
# phrase -- "<vendor> PR #216 Finding 2", "(<vendor> corr-audit #3)", "reviewed
# by <vendor>-5" -- read as clean. Eight such credits were live in .rs/.py source
# when this rule was added. It therefore:
#   * matches in BOTH orders (vendor -> credit-word and credit-word -> vendor),
#   * tolerates a SHORT gap (up to two <=4-letter filler words, e.g. "by"), which
#     is what catches the parenthesised "(<vendor> Finding 2)" shape and the
#     trailing "reviewed by <vendor>" shape with one rule,
#   * adds the review-NOUN vocabulary the first pattern lacks: finding(s), pr,
#     corr-audit, sweep, scan, reviewed, verified, found.
# The gap is deliberately short and admits no long word: that is precisely what
# keeps the ALLOWED integration lines clean -- the coding-agent client list in
# scripts/anatomy.sh, the shelled-out CLI backend in tools/mindfuzz, the
# "<vendor>-style token count" heuristics in benchmarks/, the "<vendor> announced
# the Model Hardware Standard" third-party-fact lines in docs/, and the plugin
# install section in README.md. Widening the gap starts flagging those; measure
# against them (git grep -inE the AI_NAMES list alone) before touching it.
AI_NAMES='codex|claude|deepseek|mistral|grok|gemini|gpt|opus|sonnet|haiku|kimi|qwen|nemotron|glm|moonshot|zhipu|anthropic|openai|llama|fable|copilot|chatgpt'
# The credit vocabulary covers the NOUNS a credit is written with, not just the
# verbs. It was extended when the message gate was added, because a message
# reads differently from a comment: "per the <tool> analysis", "took the <tool>
# suggestion", "<tool> feedback" are the natural shapes there and NONE of them
# matched the original list. Measured before landing the widening: 8/8 of the
# eight real credits that were live in source still caught, and 0 false
# positives across every line in the file gate's scan scope that mentions any
# AI_NAMES token at all (24 lines: integration targets, runtime backends,
# third-party facts, plugin install docs). That measurement is exhaustive for
# this pattern, not a sample: a match REQUIRES an AI_NAMES token on the line, so
# a line without one cannot be a false positive. Re-run it before any further
# widening --
#   git grep -inE "\b(${AI_NAMES})\b" -- <the gate's PATHSPEC>
# then grep that output with the candidate pattern; anything it flags is a line
# you are about to break.
CREDIT_WORDS='corr-audit|audit|analysis|analyses|panel|review|reviews|reviewer|reviewed|verified|found|finding|findings|sweep|scan|consensus|converged|driven|flagged|authored|suggested|suggestion|recommended|recommendation|feedback|ruling|assisted|assistance|proposed|wrote|rewrote|generated|pr'
CREDIT_GAP='([^[:alnum:]]{1,3}[a-z]{1,4}){0,2}[^[:alnum:]]{0,3}'
PATTERN_CREDIT="\b(${AI_NAMES})\b${CREDIT_GAP}\b(${CREDIT_WORDS})\b|\b(${CREDIT_WORDS})\b${CREDIT_GAP}\b(${AI_NAMES})\b"

# Trailer-shaped credits. These are MESSAGE-only shapes: a co-authorship trailer
# credits a second author whatever name follows it (this project is single-author
# by policy, so the trailer is forbidden regardless of the name, which is why
# this rule does not consult AI_NAMES at all), and a "Generated with/by" line
# credits the tool that produced the commit. Both are the default output of
# assistant tooling, so they are the most likely way the class enters history.
PATTERN_COAUTHOR='^[[:space:]]*co[- ]authored[- ]by[[:space:]]*:'

# "Generated with/by <tool>" — the standard assistant footer. Deliberately NOT
# every "generated by" line: commit messages in this repo legitimately say
# "testdata generated by scripts/x.py", and a rule that blocks those would be
# routed around rather than obeyed. It fires when the line names a tool the way
# the footer does — a URL or a markdown link — or names any AI_NAMES token in any
# position on the line (which the noun-phrase rule above cannot see, because
# "with"/"by" plus a vendor name is not a credit-word pairing).
PATTERN_GENERATED="generated[[:space:]]+(with|by)[[:space:]].*(https?://|\\[[^]]*\\]\\(|\\b(${AI_NAMES})\\b)"

# Exact author identity required on all new commits in this repository.
# Rejects trailing-period variants ("STARGA Inc.") and personal addresses.
EXPECTED_AUTHOR_IDENTITY='STARGA Inc <noreply@star.ga>'
EXPECTED_AUTHOR_IDENTITY_REGEX='^STARGA Inc <noreply@star\.ga>$'
EXPECTED_COMMITTER_IDENTITY_REGEX='^(STARGA Inc <noreply@star\.ga>|GitHub <noreply@github\.com>)$'
