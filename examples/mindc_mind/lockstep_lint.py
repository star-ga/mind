#!/usr/bin/env python3
"""lockstep_lint.py -- native-ELF walker lockstep linter for the pure-MIND self-host compiler.

WHAT THIS CATCHES (the #1 self-host bug class, caught up front instead of by the
byte-identity gate after the fact -- cf. commits c27a766, 0b5f489, the #229 fix):

  The pure-MIND native-ELF backend in examples/mindc_mind/main.mind carries SEVERAL
  parallel "walker" functions that recurse over the AST dispatching on `ast_kind`:
  the emit walkers (nb_expr / nb_stmt), the count walkers (nb_count_expr /
  nb_count_stmt -- the frame-slot sizer that MUST agree with emit), and the scan
  walkers (nb_ccount_*, nb_argv_*, nb_at_scan_*, nb_reach_*, nb_dtk_scan_*).

  INVARIANT: within a family (expr walkers, stmt walkers), every ast_kind that HAS
  CHILDREN must be recursed into by EVERY walker. A walker that fails to recurse into
  a child-bearing node -- a missing arm, or an arm that returns without descending --
  SKIPS that node's whole subtree. For the count-vs-emit pair that is a frame-slot
  DESYNC: count under-sizes the frame, emit writes past it => silent miscompile.

PRECISION (this is the whole game -- a lint that cries wolf on good code is worse
than none, and one that can't catch a planted bug is a null gate):

  * The unit of the invariant is a RECURSING arm, not a mere `if k == ast_X()` arm.
    A LEAF kind (int_lit, float_lit, ident, break, continue) has no children to skip;
    a missing leaf arm is harmless and is NOT flagged. Likewise addr_of / deref are
    treated as leaves by this backend (their operand is always an ident), so their
    arms return without descending and are correctly NOT counted as child-bearing.
  * "child-bearing" is decided by MAJORITY VOTE across a family's walkers: a kind is
    required iff at least half the family's walkers recurse into it. On a known-good
    main.mind all walkers agree, so the required set is exactly the true child-bearing
    set and the matrix is in perfect lockstep (zero flags). A single deleted/duplicated
    arm then shows up as the lone dissenter.
  * A "recursing arm" is one whose body calls a DESCENT function -- one of the walkers,
    or any helper that (transitively) calls a walker (nb_call_args, nb_struct_fields,
    nb_if_stmt, nb_emit_while, nb_count_stmts, ...). This is computed as a fixpoint over
    the call graph, so struct_lit/array_lit/call arms that descend via a helper are
    recognised, while int_lit's incidental `ast_child0(node)` value-read is not.

SCOPE: the native-ELF frame-slot walker families, discovered empirically as the
dispatch-walkers whose names match ^nb_.*_(expr|stmt)$ (the stable native-backend
convention) that actually recurse. The `--prefix` knob generalises the tool; the
member *set* is never hardcoded -- add an `nb_foo_expr` walker and it is picked up.

USAGE:
  python3 lockstep_lint.py                 # lint ./main.mind, print matrix, exit 1 on gaps
  python3 lockstep_lint.py --file X.mind   # lint a copy (used by the planted-bug self-test)
  python3 lockstep_lint.py --all-walkers   # also list dispatch-walkers outside scope (dtype/etc)

Exit code 0 = every family in lockstep; 1 = at least one desync-risk gap flagged.
Pure Python 3, stdlib only. No mindc build.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

# --- regexes -----------------------------------------------------------------
FN_RE = re.compile(r"^(?:pub )?fn ([A-Za-z0-9_]+)\(")
# an ast_kind CONSTANT def: `pub fn ast_X() -> i64 { N }` with an integer-literal body.
AST_DEF_RE = re.compile(r"^(?:pub )?fn (ast_[a-z0-9_]+)\(\) -> i64 \{")
INT_BODY_RE = re.compile(r"^\s*(-?\d+)\s*$")
# the dispatch variable is whatever is bound to ast_kind(node).
DISPATCH_VAR_RE = re.compile(r"let ([A-Za-z_][A-Za-z0-9_]*)\s*:\s*i64\s*=\s*ast_kind\(node\)")
# a call site: NAME( . used to build the call graph.
CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\(")

MIN_ARMS = 4  # a "dispatch walker" needs at least this many k==ast_X() arms.


def load(path):
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read().split("\n")


def function_table(lines):
    """Return [(name, start_idx, end_idx)] using next-fn-start as the robust boundary.

    MIND has no nested fns, so the line before the next `fn` def ends the body. This is
    immune to braces embedded in strings/comments (brace-tracking is not)."""
    starts = [(i, m.group(1)) for i, l in enumerate(lines) for m in [FN_RE.match(l)] if m]
    out = []
    for idx, (s, name) in enumerate(starts):
        e = starts[idx + 1][0] - 1 if idx + 1 < len(starts) else len(lines) - 1
        out.append((name, s, e))
    return out


def ast_kind_constants(lines, fns):
    """name->value for every `ast_X() -> i64 { N }` whose body is a bare integer literal.

    Excludes allocators like ast_alloc() (body is __mind_alloc(64), not a literal)."""
    kinds = {}
    for name, s, e in fns:
        m = AST_DEF_RE.match(lines[s])
        if not m:
            continue
        # body is the next non-blank, non-comment line before the closing brace.
        for j in range(s + 1, min(e, s + 8) + 1):
            t = lines[j].strip()
            if not t or t.startswith("//"):
                continue
            bm = INT_BODY_RE.match(lines[j])
            if bm:
                kinds[m.group(1)[4:]] = int(bm.group(1))  # strip "ast_"
            break
    return kinds


def strip_code(line):
    """Blank out // comments and "..." string contents so brace-counting is reliable
    (both comments and string literals in this file contain stray { } braces)."""
    out = []
    i, in_str, esc = 0, False, False
    while i < len(line):
        c = line[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            i += 1
            continue
        if c == '"':
            in_str = True
            i += 1
            continue
        if c == "/" and i + 1 < len(line) and line[i + 1] == "/":
            break  # rest of line is a comment
        out.append(c)
        i += 1
    return "".join(out)


def arm_regions(lines, s, e, var):
    """Yield (kind, arm_start_idx, arm_end_idx) for each TOP-LEVEL `if <var> == ast_X()` arm,
    bounding each arm by BRACE MATCHING on comment/string-stripped lines.

    Brace matching (not next-header delimiting) is required so the LAST arm does not
    absorb the function's fall-through default tail (`... nb_count_expr(node)`), which
    would falsely mark a leaf arm like `continue` as recursing. Only the dispatch
    variable counts -- `ast_kind(init) == ast_ident()` nested guards are excluded
    because their left operand is `ast_kind(init)`, not <var>."""
    hdr = re.compile(r"^\s*if\s+" + re.escape(var) + r"\s*==\s*ast_([a-z0-9_]+)\(\)\s*\{")
    for i in range(s, e + 1):
        m = hdr.match(lines[i])
        if not m:
            continue
        depth = 0
        arm_end = i
        for j in range(i, e + 1):
            code = strip_code(lines[j])
            depth += code.count("{") - code.count("}")
            if depth <= 0:
                arm_end = j
                break
        yield m.group(1), i, arm_end


def calls_in(lines, a, b):
    """Set of function names called anywhere in lines[a..b]."""
    return set(CALL_RE.findall("\n".join(lines[a:b + 1])))


def discover_walkers(lines, fns, ast_kinds):
    """Every dispatch-walker: a fn that binds k=ast_kind(node) and has >=MIN_ARMS arms.

    Returns dict name -> {span:(s,e), var, arms:{kind:(as,ae)}}."""
    walkers = {}
    for name, s, e in fns:
        body = "\n".join(lines[s:e + 1])
        vm = DISPATCH_VAR_RE.search(body)
        if not vm:
            continue
        var = vm.group(1)
        arms = {k: (a, b) for k, a, b in arm_regions(lines, s, e, var) if k in ast_kinds}
        if len(arms) >= MIN_ARMS:
            walkers[name] = {"span": (s, e), "var": var, "arms": arms}
    return walkers


def descent_closure(lines, fns, walker_names):
    """Fixpoint set of 'descent functions': the walkers plus any fn that (transitively)
    calls a descent function. An arm that calls one of these descends into the AST."""
    fn_index = {name: (s, e) for name, s, e in fns}
    callees = {name: calls_in(lines, s, e) for name, s, e in fns}
    descent = set(walker_names)
    changed = True
    while changed:
        changed = False
        for name, cs in callees.items():
            if name in descent:
                continue
            if cs & descent:
                descent.add(name)
                changed = True
    return descent


def is_gate_walker(name):
    """A fail-closed WHITELIST eligibility gate (not a frame-slot sizer): any
    child-bearing kind it does not recurse into makes the whole fn ineligible
    (empty plan => stack scheme => byte-identical), so a "missing" arm is safe by
    design and must not be flagged as a lockstep desync. See DTK slice 1 (#254)."""
    return "dtk_scan" in name


def recurse_kinds(lines, walker, descent):
    """Kinds whose arm body calls a descent function (i.e. descends into children)."""
    out = set()
    for kind, (a, b) in walker["arms"].items():
        # drop the guard header line so `if k == ast_X()` itself is not miscounted.
        if calls_in(lines, a + 1, b) & descent:
            out.add(kind)
    return out


def analyse(path, prefix, show_all):
    lines = load(path)
    fns = function_table(lines)
    ast_kinds = ast_kind_constants(lines, fns)
    walkers = discover_walkers(lines, fns, ast_kinds)
    descent = descent_closure(lines, fns, set(walkers))

    # per-walker recurse set
    for name, w in walkers.items():
        w["recurse"] = recurse_kinds(lines, w, descent)
        w["line"] = w["span"][0] + 1

    # scope: native-ELF frame-slot families, by the stable _expr / _stmt convention.
    # the optional `(.*_)?` also admits the reference emit walkers nb_expr / nb_stmt,
    # which carry no separator before the suffix but are the family's ground truth.
    fam_re = re.compile(r"^" + re.escape(prefix) + r"(?:.*_)?(expr|stmt)$")
    families = {"expr": [], "stmt": []}
    scoped = set()
    for name in sorted(walkers, key=lambda n: walkers[n]["line"]):
        m = fam_re.match(name)
        if m and walkers[name]["recurse"]:
            families[m.group(1)].append(name)
            scoped.add(name)

    report = Report(path, ast_kinds, walkers, families, scoped, show_all)
    return report


class Report:
    def __init__(self, path, ast_kinds, walkers, families, scoped, show_all):
        self.path = path
        self.ast_kinds = ast_kinds
        self.walkers = walkers
        self.families = families
        self.scoped = scoped
        self.show_all = show_all
        self.flags = []  # (family, kind, [missing walkers])

    def emit(self):
        w = self.walkers
        print("=" * 78)
        print(f"lockstep_lint  --  {self.path}")
        print("=" * 78)
        print(f"ast_kind constants: {len(self.ast_kinds)} "
              f"(values {min(self.ast_kinds.values())}..{max(self.ast_kinds.values())})")
        print(f"dispatch-walkers discovered: {len(w)}   "
              f"native-ELF frame-slot walkers in scope: {len(self.scoped)}")
        print()

        for fam in ("expr", "stmt"):
            self._emit_family(fam)

        if self.show_all:
            self._emit_out_of_scope()

        print("=" * 78)
        if self.flags:
            print(f"RESULT: FAIL -- {len(self.flags)} lockstep gap(s) (desync risk).")
            for fam, kind, miss in self.flags:
                miss_s = ", ".join(f"{m} (L{w[m]['line']})" for m in miss)
                print(f"  [{fam}] child-bearing kind ast_{kind} "
                      f"({self.ast_kinds.get(kind,'?')}) recursed by the family but "
                      f"NOT by: {miss_s}")
        else:
            print("RESULT: PASS -- every family walker in lockstep on all child-bearing kinds.")
        print("=" * 78)
        return 1 if self.flags else 0

    def _emit_family(self, fam):
        members = self.families[fam]
        w = self.walkers
        print("-" * 78)
        print(f"FAMILY: {fam}  ({len(members)} walkers)")
        for m in members:
            print(f"    {m:20s} L{w[m]['line']:<6d} "
                  f"arms={len(w[m]['arms']):2d} recurses={len(w[m]['recurse']):2d}")
        if not members:
            print("    (none discovered)")
            print()
            return

        n = len(members)
        need = (n + 1) // 2  # majority: >= ceil(n/2)
        recset = {m: w[m]["recurse"] for m in members}
        # required child-bearing kinds: recursed by a majority of the family.
        allk = set().union(*recset.values()) if recset else set()
        required = sorted(
            (k for k in allk if sum(k in recset[m] for m in members) >= need),
            key=lambda k: self.ast_kinds.get(k, 999),
        )

        # coverage matrix (rows = required child-bearing kinds, cols = walkers)
        print()
        hdr = "    " + f"{'ast_kind':16s} " + " ".join(f"{self._abbr(m):>10s}" for m in members)
        print(hdr)
        print("    " + "-" * (len(hdr) - 4))
        for k in required:
            cells = []
            miss = []
            for m in members:
                if k in recset[m]:
                    cells.append(f"{'R':>10s}")
                elif is_gate_walker(m):
                    # Fail-closed WHITELIST gate (nb_dtk_scan_*, #254): an unlisted
                    # child-bearing kind makes the fn INELIGIBLE (the plan is empty
                    # => stack scheme => byte-identical), so NOT recursing is the
                    # safe behaviour, not a silent desync. Shown but not flagged.
                    cells.append(f"{'gate':>10s}")
                else:
                    cells.append(f"{'--MISS--':>10s}")
                    miss.append(m)
            print("    " + f"{k:16s} " + " ".join(cells))
            if miss:
                self.flags.append((fam, k, miss))
        print()
        # leaf kinds (handled by some walker but below majority-recurse): informational.
        leafk = sorted(
            (k for k in set().union(*[set(w[m]["arms"]) for m in members])
             if k not in required),
            key=lambda k: self.ast_kinds.get(k, 999),
        )
        if leafk:
            print(f"    leaf / non-child-bearing arms (not part of the invariant): "
                  f"{', '.join(leafk)}")
        print(f"    required child-bearing kinds (majority-recursed): "
              f"{', '.join(required) if required else '(none)'}")
        print()

    def _emit_out_of_scope(self):
        w = self.walkers
        others = sorted((n for n in w if n not in self.scoped),
                        key=lambda n: -len(w[n]["recurse"]))
        print("-" * 78)
        print("OTHER dispatch-walkers (out of native-ELF frame-slot scope, not linted):")
        for n in others:
            print(f"    {n:26s} L{w[n]['line']:<6d} "
                  f"arms={len(w[n]['arms']):2d} recurses={len(w[n]['recurse']):2d}  "
                  f"[{','.join(sorted(w[n]['recurse']))}]")
        print()

    @staticmethod
    def _abbr(name):
        # compact column headers: nb_count_expr -> count, nb_at_scan_expr -> at_scan
        base = name
        for suf in ("_expr", "_stmt"):
            if base.endswith(suf):
                base = base[: -len(suf)]
        if base.startswith("nb_"):
            base = base[3:]
        return base or name


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--file", default=os.path.join(here, "main.mind"),
                    help="the .mind file to lint (default: ./main.mind)")
    ap.add_argument("--prefix", default="nb_",
                    help="native-backend walker name prefix (default: nb_)")
    ap.add_argument("--all-walkers", action="store_true",
                    help="also list dispatch-walkers outside the native-ELF scope")
    args = ap.parse_args(argv)

    if not os.path.exists(args.file):
        print(f"error: {args.file} not found", file=sys.stderr)
        return 2
    report = analyse(args.file, args.prefix, args.all_walkers)
    return report.emit()


if __name__ == "__main__":
    raise SystemExit(main())
