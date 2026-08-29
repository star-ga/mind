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
# The one runner in this repo that executes a BROAD `cargo test` (no --test
# selector) with mlir-build enabled, and the only thing that actually rescues the
# mlir-build-gated corpus. Its tiers count as CI wiring ONLY for the tiers ci.yml
# really invokes, and only because each tier asserts a POSITIVE executed count
# plus per-harness minimums — a bare selector, or an exit-0, is not evidence.
GATE = ROOT / "scripts" / "exec_semantics_gate.sh"
CARGO = ROOT / "Cargo.toml"

# A file may be exempt ONLY with a written reason on the same line. The reason
# is required so an exemption is an argument, not a silent deletion.
EXEMPT = {
    "mlir_gpu": "needs a CUDA/ROCm device; GitHub-hosted runners have no GPU. The "
                "GPU byte-identity gate lives in the private runtime repo, on real "
                "hardware. Genuinely environment-gated, not unwired by accident.",
    "mlir_jit": "needs an MLIR ExecutionEngine JIT runtime that the apt "
                "mlir-20-tools package does not ship (only the mlir-opt / "
                "mlir-translate binaries). Genuinely environment-gated.",
}

# NOT a line-anchored regex. A crate-level attribute is routinely wrapped:
#     #![cfg(all(
#         unix,
#         feature = "mlir-build",
#     ))]
# A single-line `^#!\[cfg\((.+)\)\]` silently fails to match those, so the file
# reads as UNCONDITIONAL and is skipped — a false green inside the very tool whose
# job is to find false greens. Measured when this was fixed: 49 of the tests/*.rs
# files take the wrapped form. Balance the parens instead of pattern-matching them.
CFG_START_RE = re.compile(r'^#!\[cfg\(', re.M)
FEAT_RE = re.compile(r'feature\s*=\s*"([a-z0-9\-_]+)"')
# `FEATURES_<tier>="..."` in the gate script — its own definition, read rather than
# re-declared here, so the tier feature sets have exactly ONE home in the tree.
TIER_FEAT_RE = re.compile(r'^FEATURES_([a-z0-9_]+)="([^"]*)"', re.M)
GATE_CALL_RE = re.compile(r'scripts/exec_semantics_gate\.sh([^\n|;&]*)')
CARGO_FEAT_RE = re.compile(r'^([a-z0-9\-_]+)\s*=\s*\[([^\]]*)\]', re.M)
QUOTED_RE = re.compile(r'"([^"]+)"')


def feature_deps() -> dict[str, set[str]]:
    """`[features]` edges from Cargo.toml, so an ENABLED set can be closed over.

    Without this, `--features mlir-build` looks like it does not satisfy a file
    gated on `mlir-subprocess`, even though cargo enables it transitively
    (`mlir-build = ["mlir-subprocess"]`). That is a FALSE erasure report, and the
    worst kind: it would send someone to "fix" wiring that is already correct,
    and it teaches the reader to distrust the lint.
    """
    if not CARGO.exists():
        return {}
    text = CARGO.read_text(encoding="utf-8")
    start = text.find("\n[features]")
    if start < 0:
        return {}
    body = text[start + 1:]
    nxt = body.find("\n[", 1)
    if nxt > 0:
        body = body[:nxt]
    deps: dict[str, set[str]] = {}
    for name, items in CARGO_FEAT_RE.findall(body):
        # `dep:x` / `x?/y` / `x/y` are optional-dependency syntax, not features
        # this repo's test cfgs ever name; keeping them is harmless but noisy.
        deps[name] = {i for i in QUOTED_RE.findall(items) if "/" not in i and ":" not in i}
    return deps


def close_over(feats: set[str], deps: dict[str, set[str]]) -> set[str]:
    """Transitive feature closure of an enabled set (cargo's own semantics)."""
    seen, stack = set(feats), list(feats)
    while stack:
        cur = stack.pop()
        for nxt in deps.get(cur, ()):  # noqa: B007
            if nxt not in seen:
                seen.add(nxt); stack.append(nxt)
    return seen


def parse_cfg(expr: str):
    """Parse a cfg predicate into a tree: ('all'|'any'|'not', [children]) or ('feat', name).

    A flat "collect every `feature = \"x\"` I can see" model is WRONG: it reads
    `any(feature = "mlir-lowering", feature = "mlir-build")` as requiring BOTH,
    and then reports a file as unwired when a runner that enables either one
    already runs it. That is a false positive from the tool whose entire job is
    to distinguish real erasure from apparent erasure, so the predicate is
    evaluated properly instead.
    """
    expr = expr.strip()
    m = re.match(r'^(all|any|not)\s*\((.*)\)$', expr, re.S)
    if m:
        op, inner = m.group(1), m.group(2)
        parts, depth, cur = [], 0, ''
        for ch in inner:
            if ch == ',' and depth == 0:
                parts.append(cur); cur = ''
                continue
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            cur += ch
        if cur.strip():
            parts.append(cur)
        return (op, [parse_cfg(x) for x in parts if x.strip()])
    fm = re.match(r'^feature\s*=\s*"([a-z0-9\-_]+)"$', expr)
    if fm:
        return ('feat', fm.group(1))
    # `unix`, `target_os = "..."`, etc. Not a feature; always true for our purposes
    # (CI runs on unix), and deliberately NOT treated as a requirement.
    return ('true', None)


def cfg_satisfied(node, enabled: set[str]) -> bool:
    """Is this cfg predicate satisfied by `enabled`? `*` means all-features."""
    kind, val = node
    if kind == 'true':
        return True
    if kind == 'feat':
        return '*' in enabled or val in enabled
    if kind == 'all':
        return all(cfg_satisfied(c, enabled) for c in val)
    if kind == 'any':
        return any(cfg_satisfied(c, enabled) for c in val)
    if kind == 'not':
        return not all(cfg_satisfied(c, enabled) for c in val)
    return True


def cfg_features(node) -> set[str]:
    """Every feature NAME mentioned, for reporting only — never for satisfaction."""
    kind, val = node
    if kind == 'feat':
        return {val}
    if kind in ('all', 'any', 'not'):
        out = set()
        for c in val:
            out |= cfg_features(c)
        return out
    return set()


def required_features(src: str):
    """The crate-level cfg predicate TREE, or None if the file is unconditional."""
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
                node = parse_cfg(src[i + 1:j])
                return node if cfg_features(node) else None
    raise ValueError('unbalanced crate-level #![cfg(...)] — cannot determine required features')


def code_only(text: str) -> str:
    """Drop whole-line comments.

    ci.yml documents this gate in prose that names the script; counting a comment
    as an invocation is the same prose-equals-reality error the lint exists to
    catch, and here it would be worse than useless — a mention with no tier
    argument would credit EVERY tier.
    """
    return "\n".join(l for l in text.splitlines() if not l.strip().startswith("#"))


def tier_wiring(ci_text: str) -> list[set[str]]:
    """Feature sets of exec_semantics_gate.sh tiers that ci.yml actually runs.

    Returns one set per invoked tier; each behaves like a broad `cargo test`
    because that is literally what the script runs for that tier.
    """
    if not GATE.exists():
        return []
    tiers = {t: set(re.split(r'[ ,]+', f.strip()) ) - {""}
             for t, f in TIER_FEAT_RE.findall(GATE.read_text(encoding="utf-8"))}
    calls = GATE_CALL_RE.findall(code_only(ci_text))
    if not calls:
        return []
    if not tiers:
        raise SystemExit(
            f"FAIL: {CI.name} invokes {GATE.name} but no FEATURES_<tier>=\"...\" "
            f"assignment could be read from it. The lint would then credit nothing "
            f"and report every mlir-build test as erased; refusing to guess."
        )
    out: list[set[str]] = []
    for args in calls:
        # `--print-count` reports numbers and NEVER fails, so it proves nothing and
        # must not be credited as wiring. Without this it would be read as a bare
        # invocation and credit every tier at once.
        if "--print-count" in args:
            continue
        named = [w for w in args.split() if w in tiers]
        for t in (named or list(tiers)):
            out.append(set(tiers[t]))
    return out


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
    tier_feats = tier_wiring(ci_text)

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
    broad.extend(tier_feats)

    # Expand every enabled set over the Cargo feature graph BEFORE comparing.
    deps = feature_deps()
    wired = {k: [close_over(f, deps) for f in v] for k, v in wired.items()}
    broad = [close_over(f, deps) for f in broad]

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
            if cfg_satisfied(feats, enabled):
                runs = True; break
        if runs:
            ok += 1
        else:
            n_tests = len(re.findall(r'^\s*#\[test\]', path.read_text(encoding="utf-8", errors="replace"), re.M))
            erased.append((name, sorted(cfg_features(feats)), n_tests))

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
