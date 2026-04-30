# Copyright 2025-2026 STARGA Inc.
# Licensed under the Apache License, Version 2.0 (the "License").
# Part of the MIND project (Machine Intelligence Native Design).

"""AI-assisted proof generation hook.

When the typechecker / verifier returns UNSAT (a constraint cannot be
discharged automatically), the bridge collects the relevant context and
emits a structured prompt that an external LLM service can consume.
This module *only* builds the prompt — it never calls a model — so the
core compiler stays sandbox-free.

Workflow

1. Compiler hits UNSAT on some invariant (e.g., shape constraint in
   a transposed matmul produced by the PyTorch bridge).
2. Caller constructs an :class:`UnsatContext` describing the failure.
3. Caller calls :func:`build_unsat_prompt` to render a self-contained
   prompt; pipes the prompt to its LLM of choice.
4. The LLM returns a candidate ``.mind`` patch; caller re-runs mindc.

The prompt format is stable and versioned so STARGA-internal automation
(e.g. ``mind-fleet`` agents) can parse the output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class UnsatContext:
    """Single UNSAT failure surfaced by the compiler."""

    constraint: str
    location: str
    expected: str
    actual: str
    notes: List[str] = field(default_factory=list)


PROMPT_VERSION = "1"


def build_unsat_prompt(ctx: UnsatContext) -> str:
    """Render an LLM prompt that asks for a `.mind` patch resolving the failure.

    Output is deterministic — the same `UnsatContext` always produces
    the same prompt — so prompts are cacheable.
    """
    lines = [
        "# MIND UNSAT Resolution Request",
        f"prompt-version: {PROMPT_VERSION}",
        "",
        "## Constraint that failed",
        f"  {ctx.constraint}",
        "",
        "## Source location",
        f"  {ctx.location}",
        "",
        "## Expected vs actual",
        f"  expected: {ctx.expected}",
        f"  actual:   {ctx.actual}",
    ]
    if ctx.notes:
        lines.append("")
        lines.append("## Compiler notes")
        for n in ctx.notes:
            lines.append(f"  - {n}")
    lines.extend(
        [
            "",
            "## Task",
            "Produce a minimal `.mind` patch that satisfies the constraint",
            "without weakening the surrounding invariants. Reply with the patch",
            "in fenced ```mind``` blocks. Do not narrate; do not add commentary",
            "outside the code blocks.",
        ]
    )
    return "\n".join(lines) + "\n"
