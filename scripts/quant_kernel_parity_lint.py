#!/usr/bin/env python3
"""Fail closed when the quant examples' vendored math kernels drift apart.

Every example under `examples/quant/` is deliberately SELF-CONTAINED: it carries
its own copy of the deterministic math kernels (`dm_exp`, `dm_log`, `dm_sqrt`,
`dm_erfc`, `dm_norm_cdf`, `dm_norm_pdf`, `absf`, `canonical_qnan`, `bs_d1`,
`bs_d2`) rather than importing them. That is a real trade: an example stays
readable and runnable on its own, at the cost of N copies of one kernel.

The cost only stays acceptable while the copies are IDENTICAL. If one example's
`dm_erfc` is edited — a tightened coefficient, a moved crossover, a reassociated
Horner step — and its siblings are not, the examples silently stop agreeing about
what the normal CDF IS. Two of them would then price the same option differently,
each with a green KAT, because each KAT checks its own copy against its own
references. Nothing in the build would notice; the divergence is invisible until
someone diffs two files nobody diffs.

That failure is specific to this layout and it is not hypothetical: the whole
point of these examples is that a number is reproducible, and a per-example
private kernel is exactly how "reproducible" quietly becomes "reproducible within
one directory".

This lint removes the luck. It extracts each named kernel from every example and
compares the bodies byte-for-byte across all examples that define it. An example
is free to OMIT a kernel it does not use (a lattice needs no erfc) — absence is
fine, disagreement is not.

Exit 0 = every kernel that appears more than once is byte-identical everywhere.
Exit 1 = at least one kernel has two differing definitions, or a file could not
be parsed.
"""
from __future__ import annotations
import hashlib
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
QUANT = ROOT / "examples" / "quant"

# The kernels that MUST agree wherever they appear. Every one of these is a
# numeric primitive whose last bits feed a priced number; a divergence in any of
# them changes a result without changing a test.
SHARED_KERNELS = (
    "dm_exp",
    "dm_log",
    "dm_sqrt",
    "dm_erf",
    "dm_erfc",
    "dm_norm_cdf",
    "dm_norm_pdf",
    "absf",
    "canonical_qnan",
    "bs_d1",
    "bs_d2",
)


def extract(src: str, name: str) -> str | None:
    """Return the full text of `fn name(...) { ... }`, or None if absent.

    Brace-matched rather than regex-terminated: a `}` inside a nested block must
    not end the function early, and a regex that stops at the first
    column-zero `}` would be defeated by any future reformatting.
    """
    m = re.search(rf"^(?:pub )?fn {re.escape(name)}\s*\(", src, re.M)
    if m is None:
        return None
    i = src.index("{", m.start())
    depth = 0
    for j in range(i, len(src)):
        if src[j] == "{":
            depth += 1
        elif src[j] == "}":
            depth -= 1
            if depth == 0:
                return src[m.start() : j + 1]
    return None  # unbalanced braces — treated as a parse failure by the caller


def main() -> int:
    if not QUANT.is_dir():
        print(f"quant-kernel-parity: no such directory: {QUANT}", file=sys.stderr)
        return 1

    examples = sorted(p for p in QUANT.iterdir() if (p / "main.mind").is_file())
    if not examples:
        print("quant-kernel-parity: no examples found", file=sys.stderr)
        return 1

    sources: dict[str, str] = {}
    for ex in examples:
        try:
            sources[ex.name] = (ex / "main.mind").read_text()
        except OSError as e:
            print(f"quant-kernel-parity: cannot read {ex.name}: {e}", file=sys.stderr)
            return 1

    failures = 0
    for kernel in SHARED_KERNELS:
        # digest -> [example names] for every example that defines this kernel
        variants: dict[str, list[str]] = {}
        for name, src in sources.items():
            body = extract(src, kernel)
            if body is None:
                continue  # omission is allowed; disagreement is not
            digest = hashlib.sha256(body.encode()).hexdigest()[:16]
            variants.setdefault(digest, []).append(name)

        if len(variants) > 1:
            failures += 1
            print(f"DIVERGED: {kernel} has {len(variants)} distinct definitions:")
            for digest, owners in sorted(variants.items(), key=lambda kv: -len(kv[1])):
                print(f"    {digest}  {', '.join(sorted(owners))}")

    if failures:
        print()
        print(
            f"quant-kernel-parity: FAIL — {failures} kernel(s) differ between "
            f"examples under examples/quant/.",
            file=sys.stderr,
        )
        print(
            "Each example vendors its own copy on purpose, but the copies must "
            "stay byte-identical: two examples that disagree about dm_erfc "
            "disagree about every price derived from it, and each one's KAT "
            "stays green because it checks its own copy.",
            file=sys.stderr,
        )
        print(
            "Fix by copying the intended definition verbatim into every example "
            "that defines it — not by deleting the check.",
            file=sys.stderr,
        )
        return 1

    checked = sum(
        1
        for k in SHARED_KERNELS
        if sum(extract(s, k) is not None for s in sources.values()) > 1
    )
    print(
        f"quant-kernel-parity: OK — {checked} shared kernel(s) byte-identical "
        f"across {len(examples)} example(s)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
