"""MIND-Fuzz mutation engine.

Implements the "mutate code using an LLM" half of arXiv:2501.00655 (Listing 2):
build a prompt from (current program, one mutation instruction) and ask an
off-the-shelf LLM to rewrite the program. The instruction is chosen by a COUNTER
(iteration % N), never at random, so a run replays byte-for-byte.

LLM call: `env -u ANTHROPIC_API_KEY claude -p "<prompt>"` (the local Claude CLI).
gemini --yolo is flaky in non-trusted dirs, so we use claude. If the LLM step
fails, times out, or returns something that does not look like a MIND program, we
FALL BACK to a deterministic template mutator so the LOOP is always demonstrable
(the fallback is recorded in the step log).
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

# Listing-2 prompt template, adapted: MIND is the target language and we demand a
# raw program back (no markdown), which keeps the compile step from choking on
# ``` fences.
PROMPT_TEMPLATE = (
    "You are a MIND-language source-to-source mutator. MIND is a Rust-like, "
    "deterministic systems language: functions are `pub fn name(p: i64) -> i64 "
    "{{ ... }}`, statements are `let x: i64 = ...;` and `return ...;`, it has "
    "if/else and `while` and bounded loops, integer types i16/i32/i64, and "
    "intrinsics named `__mind_blas_*`. MIND has NO pointers, NO unions, NO "
    "C-style structs.\n\n"
    "Apply EXACTLY this mutation to the program below, changing as little else "
    "as possible: {instruction}.\n\n"
    "Output ONLY the complete mutated MIND program. No markdown fences, no "
    "explanation, no commentary.\n\n"
    "Program:\n{code}\n"
)

# A line looks like MIND if the whole blob contains at least one function def and
# a return; this is a cheap "did the LLM give us a program" gate, NOT a compiler.
_FN_RE = re.compile(r"\bfn\s+\w+\s*\(")


def looks_like_mind(text: str) -> bool:
    return bool(text) and bool(_FN_RE.search(text))


def strip_fences(text: str) -> str:
    """Remove a single leading/trailing markdown code fence if the LLM added one."""
    t = text.strip()
    if t.startswith("```"):
        # drop the first line (``` or ```mind) and a trailing ``` line
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines)
    return t.strip() + "\n"


@dataclass
class MutationResult:
    code: str
    instruction: str
    tag: str  # "preserve" or "semantics"
    engine: str  # "llm" or "template-fallback"


def parse_instruction(raw: str) -> tuple[str, str]:
    """Split a mutations.txt line into (tag, instruction_text)."""
    raw = raw.strip()
    m = re.match(r"\[(preserve|semantics)\]\s*(.*)", raw)
    if m:
        return m.group(1), m.group(2).strip()
    # untagged lines default to semantics (most conservative for the ref oracle)
    return "semantics", raw


def load_instructions(path: Path) -> list[str]:
    out: list[str] = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s)
    return out


def llm_mutate(code: str, instruction: str, timeout_s: int = 120) -> str | None:
    """Call the local claude CLI. Returns cleaned text or None on any failure."""
    prompt = PROMPT_TEMPLATE.format(instruction=instruction, code=code)
    try:
        proc = subprocess.run(
            ["env", "-u", "ANTHROPIC_API_KEY", "claude", "-p", prompt],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    if proc.returncode != 0:
        return None
    out = strip_fences(proc.stdout)
    return out if looks_like_mind(out) else None


def template_mutate(code: str, instruction: str) -> str:
    """Deterministic, LLM-free fallback so the loop always advances.

    Applies a tiny, syntactically-safe transform keyed off the instruction text.
    These are intentionally simple; the point is to keep the fuzz loop moving and
    demonstrable when the LLM is unavailable, not to be clever.
    """
    instr = instruction.lower()

    # "change an integer literal": bump the first standalone integer literal.
    if "integer literal" in instr:

        def bump(m: re.Match) -> str:
            return str(int(m.group(0)) + 1)

        # bump the first `= <int>;` literal we find
        new, count = re.subn(r"(?<==\s)(\d+)(?=\s*;)", bump, code, count=1)
        if count:
            return new

    # "redundant let" / "constant-folded temporary" / "dead conditional":
    # inject a harmless dead statement after the first `{`.
    insert = None
    if "redundant let" in instr or "constant-folded" in instr:
        insert = "    let _mf_dead: i64 = 2 + 3;\n"
    elif "dead conditional" in instr:
        insert = "    if false { let _mf_z: i64 = 0; }\n"
    elif "bounded counting loop" in instr or "counting loop" in instr:
        insert = "    let mut _mf_acc: i64 = 0;\n"

    if insert is not None:
        idx = code.find("{")
        if idx != -1:
            nl = code.find("\n", idx)
            if nl != -1:
                return code[: nl + 1] + insert + code[nl + 1 :]

    # default: append a trailing comment marker (always compiles, no-op).
    return code.rstrip("\n") + "\n// _mf_touched\n"


def mutate(code: str, counter: int, instructions: list[str], use_llm: bool = True) -> MutationResult:
    """Pick instruction by COUNTER, mutate via LLM with template fallback."""
    raw = instructions[counter % len(instructions)]
    tag, instruction = parse_instruction(raw)

    if use_llm:
        out = llm_mutate(code, instruction)
        if out is not None:
            return MutationResult(out, instruction, tag, "llm")

    return MutationResult(
        template_mutate(code, instruction), instruction, tag, "template-fallback"
    )
