#!/usr/bin/env python3
"""Fail closed when a feature-gated integration test is wired to no CI selector.

Root cause this closes (measured 2026-08-28): a crate-level
`#![cfg(all(unix, feature = "mlir-build", ...))]` with no `[[test]]
required-features` stanza does NOT make cargo skip the file visibly — cargo
builds it as an EMPTY HARNESS and prints `ok. 0 passed`, exit 0. At the time
this lint was written, 56 such files held 101 `#[test]` functions and NOT ONE
was named by any `--test` selector in ci.yml, including the two dedicated
regression gates for the alias miscompile and the deterministic OOB trap.

The defect class is "a gate's scope is a hand-maintained list, and nothing
asserts the substance it names actually executed". So this lint does not
maintain a second list: it DERIVES the required set from the source files
themselves, and fails if any member is unreachable from CI.

Exit 0 = every feature-gated test file is named by a selector on a CI line that
actually enables the features its cfg demands. Exit 1 = at least one is erased.
"""
from __future__ import annotations
import re, sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
TESTS = ROOT / "tests"
CI = ROOT / ".github" / "workflows" / "ci.yml"

# A file may be exempt ONLY with a written reason on the same line. The reason
# is required so an exemption is an argument, not a silent deletion.
EXEMPT = {
    # name: reason
}

# NOT line-anchored: a crate-level attribute is routinely wrapped across lines
# (`#![cfg(all(\n  unix,\n  feature = "mlir-build",\n))]`). A single-line pattern
# silently fails to match those, so the file reads as UNCONDITIONAL and is skipped
# — a false green inside the tool whose job is finding false greens. Measured:
# 49 of tests/*.rs take the wrapped form. Balance the parens instead.
CFG_START_RE = re.compile(r'^#!\[cfg\(', re.M)
FEAT_RE = re.compile(r'feature\s*=\s*"([a-z0-9\-_]+)"')


def required_features(src: str) -> set[str] | None:
    """Features a crate-level cfg demands, or None if the file is unconditional."""
    m = CFG_START_RE.search(src)
    if not m:
        return None
    i = src.index('(', m.start())
    depth = 0
    for j in range(i, len(src)):
        if src[j] == '(':
            depth += 1
        elif src[j] == ')':
            depth -= 1
            if depth == 0:
                feats = set(FEAT_RE.findall(src[i + 1:j]))
                return feats or None
    raise ValueError('unbalanced crate-level #![cfg(...)] — cannot determine required features')


def ci_lines(text: str) -> list[str]:
    """Logical `cargo test` invocations, YAML line-continuations joined."""
    out, cur = [], ""
    for raw in text.splitlines():
        line = raw.strip()
        if cur:
            cur += " " + line.rstrip("\\").strip()
            if not raw.rstrip().endswith("\\"):
                out.append(cur); cur = ""
            continue
        if "cargo test" in line or "cargo nextest" in line:
            if raw.rstrip().endswith("\\"):
                cur = line.rstrip("\\").strip()
            else:
                out.append(line)
    if cur:
        out.append(cur)
    return out


def main() -> int:
    if not CI.exists():
        print(f"FAIL: {CI} missing", file=sys.stderr); return 1
    ci_text = CI.read_text(encoding="utf-8")
    invocations = ci_lines(ci_text)

    # For each invocation: which tests it selects, and which features it enables.
    wired: dict[str, list[set[str]]] = {}
    broad: list[set[str]] = []   # invocations with NO --test selector run EVERY target
    for inv in invocations:
        sel = re.findall(r'--test\s+([a-z0-9_]+)', inv)
        feats: set[str] = set()
        if "--all-features" in inv:
            feats = {"*"}
        else:
            for fm in re.findall(r'--features[= ]\s*"([^"]*)"', inv) + \
                      re.findall(r"--features[= ]\s*'([^']*)'", inv) + \
                      re.findall(r'--features[= ]([a-z0-9\-_,]+)', inv):
                feats |= set(re.split(r'[ ,]+', fm.strip()))
            if "--no-default-features" not in inv:
                feats.add("std-surface")  # crate default
        # `--doc`, `--bin`, `--lib` restrict away from integration tests; those are
        # not broad runners even without a --test selector.
        restricted = any(f in inv for f in ("--doc", "--lib", "--bins", "--bin "))
        if sel:
            for s in sel:
                wired.setdefault(s, []).append(feats)
        elif not restricted:
            broad.append(feats)

    erased, ok = [], 0
    for path in sorted(TESTS.glob("*.rs")):
        name = path.stem
        feats = required_features(path.read_text(encoding="utf-8", errors="replace"))
        if feats is None:
            continue  # unconditional: a broad `cargo test` runs it
        if name in EXEMPT:
            continue
        runs = False
        for enabled in wired.get(name, []) + broad:
            if "*" in enabled or feats <= enabled:
                runs = True; break
        if runs:
            ok += 1
        else:
            n_tests = len(re.findall(r'^\s*#\[test\]', path.read_text(encoding="utf-8", errors="replace"), re.M))
            erased.append((name, sorted(feats), n_tests))

    if erased:
        total = sum(e[2] for e in erased)
        print(f"FAIL: {len(erased)} feature-gated test file(s) are named by NO CI selector "
              f"that enables their features — {total} #[test] fn(s) compile to an empty "
              f"harness and report `ok. 0 passed`, exit 0.\n", file=sys.stderr)
        for name, feats, n in erased:
            print(f"  tests/{name}.rs  needs [{' '.join(feats)}]  ({n} #[test])", file=sys.stderr)
        print("\nFix: name it on a ci.yml `cargo test` line that enables those features "
              "(and assert a positive per-target count), or add it to EXEMPT with a written "
              "reason.", file=sys.stderr)
        return 1

    print(f"PASS: every feature-gated integration test ({ok}) is reachable from a CI "
          f"selector that enables its features")
    return 0


if __name__ == "__main__":
    sys.exit(main())
