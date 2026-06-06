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


def _first_scalar_fn_arg(code: str) -> str:
    """Return the parameter name of the first `fn name(<arg>: i64 ...)` so the
    template transforms can fold a *live* value (the argument) into the result.
    Falls back to "a" (the seed convention) if none is found."""
    m = re.search(r"\bfn\s+\w+\s*\(\s*(\w+)\s*:", code)
    return m.group(1) if m else "a"


def _inject_after_first_brace(code: str, stmt: str) -> str | None:
    """Insert `stmt` (already newline-terminated) right after the first `{`."""
    idx = code.find("{")
    if idx == -1:
        return None
    nl = code.find("\n", idx)
    if nl == -1:
        return None
    return code[: nl + 1] + stmt + code[nl + 1 :]


def _rewrite_first_return(code: str, build: "callable") -> str | None:
    """Rewrite the FIRST `return <expr>;` via build(expr) -> new_expr. Keeps the
    transform local and value-correct for the round-trip / fold rows. Returns
    None if no return is found."""
    m = re.search(r"return\s+(.+?);", code, flags=re.DOTALL)
    if not m:
        return None
    new_expr = build(m.group(1).strip())
    return code[: m.start()] + f"return {new_expr};" + code[m.end():]


# A deterministic cursor of edge values the template mutator cycles through so
# successive `[semantics]` edge-value steps don't all pick the same constant.
_EDGE_VALUES = (
    "2147483647", "-2147483648", "9223372036854775807",
    "-9223372036854775808", "0", "-1", "127", "128", "255", "256",
    "32767", "32768",
)


def template_mutate(code: str, instruction: str, counter: int = 0) -> str:
    """Deterministic, LLM-free fallback so the loop always advances.

    Applies a small, syntactically-safe, VALID-.mind transform keyed off the
    instruction text. The aggressive tier-2 rows below fold an edge value, a
    round-trip `as` cast, an overflowing op, a shift, a nested if, a loop, or an
    SSA-id chain into a *live* value so the determinism / verify / mic@3 /
    reference oracles actually have something to bite on (the original fallback
    mostly appended an inert comment, which reached no new ground).
    """
    instr = instruction.lower()
    arg = _first_scalar_fn_arg(code)

    # ---- tier 2 aggressive rows (checked first; most edits target a return) --

    # round-trip cast on a live value: must PRESERVE the value, so the reference
    # oracle can assert it. (x as i32) as i64 is identity for in-range x; we only
    # apply it on the returned expression which the preserve tag guards.
    if "round-trip cast" in instr or "identity round-trip" in instr:
        out = _rewrite_first_return(code, lambda e: f"(({e}) as i32) as i64")
        if out is not None:
            return out

    # narrowing casts that MAY change the value (semantics tag): i32 / i16 / i8.
    if "narrow a live i64 value to i32" in instr:
        out = _rewrite_first_return(code, lambda e: f"(({e}) as i32) as i64")
        if out is not None:
            return out
    if "narrow a live i64 value to i16" in instr:
        out = _rewrite_first_return(code, lambda e: f"(({e}) as i16) as i64")
        if out is not None:
            return out
    if "unsigned type" in instr:
        out = _rewrite_first_return(code, lambda e: f"(({e}) as u32) as i64")
        if out is not None:
            return out
    if "i32-typed local" in instr or "arithmetic in i32" in instr:
        stmt = f"    let _mf_w: i32 = ({arg} as i32) + 7;\n"
        out = _inject_after_first_brace(code, stmt)
        if out is not None:
            return _rewrite_first_return(out, lambda e: f"({e}) + (_mf_w as i64)") or out

    # overflow / large-constant arithmetic.
    if "multiply a live value by a large constant" in instr:
        out = _rewrite_first_return(code, lambda e: f"({e}) * 2147483647")
        if out is not None:
            return out
    if "add a large constant" in instr:
        out = _rewrite_first_return(code, lambda e: f"({e}) + 9223372036854775807")
        if out is not None:
            return out
    if "shift a live value left" in instr:
        out = _rewrite_first_return(code, lambda e: f"(({e}) << 33)")
        if out is not None:
            return out
    if "shift a live value right" in instr:
        out = _rewrite_first_return(code, lambda e: f"(({e}) >> 63)")
        if out is not None:
            return out
    if "divide a live value by a small constant" in instr:
        out = _rewrite_first_return(code, lambda e: f"(({e}) / 3) + (({arg}) % 3)")
        if out is not None:
            return out

    # edge-value literal substitution feeding the result.
    if "i32 boundary value" in instr or "i64 boundary value" in instr or \
       "bytewise edge values" in instr:
        # counter-keyed so the run replays byte-for-byte (no hidden cursor).
        ev = _EDGE_VALUES[counter % len(_EDGE_VALUES)]
        out = _rewrite_first_return(code, lambda e: f"({e}) ^ ({ev})")
        if out is not None:
            return out
    if "all-ones bit pattern" in instr or "let e: i64 = -1" in instr:
        stmt = "    let _mf_e: i64 = -1;\n"
        out = _inject_after_first_brace(code, stmt)
        if out is not None:
            return _rewrite_first_return(out, lambda e: f"({e}) ^ _mf_e") or out

    # control-flow depth.
    if "three levels of nested" in instr:
        stmt = (
            f"    if {arg} > 0 {{ if {arg} > 1 {{ if {arg} > 2 {{ "
            f"let _mf_d: i64 = 1; }} }} }}\n"
        )
        out = _inject_after_first_brace(code, stmt)
        if out is not None:
            return out
    if "never executes (zero iterations)" in instr:
        stmt = (
            "    let mut _mf_i: i64 = 0;\n"
            "    while _mf_i < 0 { _mf_i = _mf_i + 1; }\n"
            "    while _mf_i < 1 { _mf_i = _mf_i + 1; }\n"
        )
        out = _inject_after_first_brace(code, stmt)
        if out is not None:
            return out
    if "large constant bound" in instr:
        stmt = (
            "    let mut _mf_j: i64 = 0;\n    let mut _mf_s: i64 = 0;\n"
            "    while _mf_j < 1000 { _mf_s = _mf_s + _mf_j; _mf_j = _mf_j + 1; }\n"
        )
        out = _inject_after_first_brace(code, stmt)
        if out is not None:
            return _rewrite_first_return(out, lambda e: f"({e}) + (_mf_s & 1)") or out
    if "early `return` inside a loop" in instr or "early return inside a loop" in instr:
        stmt = (
            "    let mut _mf_k: i64 = 0;\n"
            f"    while _mf_k < 100 {{ if _mf_k == {arg} {{ return _mf_k; }} "
            "_mf_k = _mf_k + 1; }\n"
        )
        out = _inject_after_first_brace(code, stmt)
        if out is not None:
            return out

    # shadowing + SSA-id pressure.
    if "shadow an existing local" in instr:
        stmt = "    let _mf_sh: i64 = 1;\n    let _mf_sh: i64 = _mf_sh + 1;\n"
        out = _inject_after_first_brace(code, stmt)
        if out is not None:
            return _rewrite_first_return(out, lambda e: f"({e}) + (_mf_sh - 2)") or out
    if "chain of eight dependent locals" in instr:
        chain = f"    let _mf_c0: i64 = {arg};\n"
        for i in range(1, 8):
            chain += f"    let _mf_c{i}: i64 = _mf_c{i-1} + 1;\n"
        out = _inject_after_first_brace(code, chain)
        if out is not None:
            return _rewrite_first_return(out, lambda e: f"({e}) + (_mf_c7 - _mf_c0 - 7)") or out
    if "long single-expression chain" in instr:
        out = _rewrite_first_return(
            code,
            lambda e: f"((({e}) + 3) * 2 - 1) & 1048575 | ({arg} ^ 7)",
        )
        if out is not None:
            return out

    # mixed-width arithmetic.
    if "an i32 local and an i64 local" in instr or "i32 local and an i64" in instr:
        stmt = f"    let _mf_x32: i32 = ({arg} as i32) + 1;\n    let _mf_x64: i64 = {arg} + 2;\n"
        out = _inject_after_first_brace(code, stmt)
        if out is not None:
            return _rewrite_first_return(out, lambda e: f"({e}) + (_mf_x32 as i64) + (_mf_x64 - {arg} - 2)") or out
    if "i16 local, an i32 local and the i64" in instr:
        stmt = (
            f"    let _mf_a16: i16 = ({arg} as i16);\n"
            f"    let _mf_a32: i32 = ({arg} as i32);\n"
        )
        out = _inject_after_first_brace(code, stmt)
        if out is not None:
            return _rewrite_first_return(
                out, lambda e: f"({e}) + ((_mf_a16 as i64) - (_mf_a32 as i64))"
            ) or out

    # ---- tier 1 semantics-light (original behaviour) -------------------------

    # "change an integer literal": bump the first standalone integer literal.
    if "integer literal" in instr:

        def bump(m: re.Match) -> str:
            return str(int(m.group(0)) + 1)

        new, count = re.subn(r"(?<==\s)(\d+)(?=\s*;)", bump, code, count=1)
        if count:
            return new

    # "redundant let" / "constant-folded temporary" / "dead conditional":
    insert = None
    if "redundant let" in instr or "constant-folded" in instr:
        insert = "    let _mf_dead: i64 = 2 + 3;\n"
    elif "dead conditional" in instr:
        insert = "    if false { let _mf_z: i64 = 0; }\n"
    elif "bounded counting loop" in instr or "counting loop" in instr:
        insert = (
            "    let mut _mf_acc: i64 = 0;\n    let mut _mf_ci: i64 = 0;\n"
            "    while _mf_ci < 4 { _mf_acc = _mf_acc + _mf_ci; _mf_ci = _mf_ci + 1; }\n"
        )

    if insert is not None:
        out = _inject_after_first_brace(code, insert)
        if out is not None:
            return out

    # "widen an i16 local to i32": inject a widened-local round-trip into result.
    if "widen an i16 local" in instr or "i16 local to i32" in instr:
        stmt = f"    let _mf_wide: i64 = ({arg} as i32) as i64;\n"
        out = _inject_after_first_brace(code, stmt)
        if out is not None:
            return _rewrite_first_return(out, lambda e: f"({e}) + (_mf_wide - {arg})") or out

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
        template_mutate(code, instruction, counter), instruction, tag, "template-fallback"
    )
