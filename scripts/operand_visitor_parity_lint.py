#!/usr/bin/env python3
"""Fail closed when the two IR operand visitors drift apart.

`crate::opt::ir_canonical::for_each_operand` (immutable) and
`crate::opt::native_opt::for_each_operand_mut` (mutable) are, in the words of the
latter's own doc comment, "two views of one audited list". `remap_operands` uses
the MUTABLE one to rename a value id at every USE site; the immutable one is used
to read operands. If the mutable mirror is missing a variant, that variant's
operands are silently not renamed — CSE/GVN or const-`If` pruning then leaves an
instruction reading a stale SSA id. That is a SILENT MISCOMPILE in the optimizer,
not a crash.

Measured 2026-08-28: merging `origin/main` (which added `Instr::ArrayStore`) into a
branch carrying the optimizer left the mutable mirror at 42 variants against the
immutable visitor's 43. Only an unrelated `non-exhaustive patterns` compile error
surfaced it, and only because the match had no catch-all — a single `_ => {}` in
either visitor would have hidden the drift permanently.

This lint removes the luck: it compares the two variant sets directly and fails on
any asymmetry in either direction. It also refuses a wildcard arm in either
visitor, because a wildcard makes the compiler's exhaustiveness check — the only
other thing standing between a new variant and a silent miscompile — inoperative.

Exit 0 = the two visitors cover exactly the same variants and neither has a
wildcard. Exit 1 = drift, a wildcard, or a visitor that could not be parsed.
"""
from __future__ import annotations
import pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
IMMUT = (ROOT / "src/opt/ir_canonical.rs", "fn for_each_operand")
MUTBL = (ROOT / "src/opt/native_opt.rs", "fn for_each_operand_mut")

VARIANT_RE = re.compile(r"\bI(?:nstr)?::([A-Z]\w+)")
# `_ =>` or `_ if ... =>` at arm position; not `_` inside a binding pattern.
WILDCARD_RE = re.compile(r"^\s*_\s*(?:if[^=]*)?=>", re.M)


def body(path: pathlib.Path, fn: str) -> str:
    """The brace-balanced body of `fn` in `path`. Raises if absent or unbalanced."""
    if not path.exists():
        raise SystemExit(f"FAIL: {path} does not exist — visitor parity cannot be checked.")
    src = path.read_text(encoding="utf-8", errors="replace")
    i = src.find(fn)
    if i < 0:
        raise SystemExit(
            f"FAIL: `{fn}` not found in {path.name}. It was renamed or deleted; this lint "
            f"must be updated deliberately, not silently skipped."
        )
    j = src.index("{", i)
    depth = 0
    for k in range(j, len(src)):
        if src[k] == "{":
            depth += 1
        elif src[k] == "}":
            depth -= 1
            if depth == 0:
                return src[j : k + 1]
    raise SystemExit(f"FAIL: unbalanced braces in `{fn}` ({path.name}).")


def main() -> int:
    b_imm = body(*IMMUT)
    b_mut = body(*MUTBL)
    v_imm = set(VARIANT_RE.findall(b_imm))
    v_mut = set(VARIANT_RE.findall(b_mut))

    problems: list[str] = []

    missing = sorted(v_imm - v_mut)
    extra = sorted(v_mut - v_imm)
    if missing:
        problems.append(
            "variants the MUTABLE mirror does not cover (their operands would NOT be "
            f"renamed by remap_operands -> silent miscompile): {missing}"
        )
    if extra:
        problems.append(
            "variants only the MUTABLE visitor covers (the immutable reader is then "
            f"blind to those operands): {extra}"
        )

    for (path, fn), b in ((IMMUT, b_imm), (MUTBL, b_mut)):
        if WILDCARD_RE.search(b):
            problems.append(
                f"`{fn}` ({path.name}) contains a wildcard `_ =>` arm. That disables the "
                "compiler's exhaustiveness check, which is the only thing that catches a "
                "newly-added Instr variant here. Enumerate variants explicitly."
            )

    if not v_imm or not v_mut:
        problems.append(
            f"parsed {len(v_imm)} immutable / {len(v_mut)} mutable variants — a visitor "
            "that yields zero variants means this lint is asserting nothing."
        )

    if problems:
        print("FAIL: IR operand-visitor parity is broken.\n", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        print(
            "\nBoth visitors are two views of ONE audited operand list. Fix by mirroring "
            "the arm exactly — visiting the same operands, and never the instruction's "
            "`dst`, which is a definition rather than an operand.",
            file=sys.stderr,
        )
        return 1

    print(f"PASS: both IR operand visitors cover the same {len(v_imm)} Instr variants, "
          f"neither uses a wildcard arm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
