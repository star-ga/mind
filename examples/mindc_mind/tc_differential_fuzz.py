#!/usr/bin/env python3
"""tcdiff — differential fuzzer for the self-host source-position tc-rule ports.

VERIFICATION_APPARATUS.md §2.1 Stage 1 [BUILD-NOW]: the FIRST-LINE detector for
under-fire, over-fire, and fail-open divergences between the pure-MIND
`selftest_tc_*` rule ports (examples/mindc_mind/main.mind) and live
`mindc check`, swept over a GENERATED template-bank × filler-matrix grid
(position-context × token-class) instead of a hand-authored CASES list.

Rules wired — shared export signature
`selftest_tc_<rule>(src, src_len, pos, std_src, std_len) -> i64`
(1 = fires · 0 = in-domain no-fire · -3 = fail-closed decline):

    E2002 unknown_ident       E2003 unknown_call
    E2009 undeclared_assign   E2012 fn_value_call

Differential invariant (uniform across rules; the 0-vs--3 internal distinction
never enters the comparison — that is WHY fail-open is caught):

    codes_all = every type_check code live `mindc check` reports in the file
    E1001 in codes_all          -> unparseable: discarded (counted, never scored)
    port_fires = (port(src, pos) == 1)
    live_fires = (rule's E-code present at the hole's exact line:col)
    DIVERGENCE iff port_fires != live_fires
        port silent (0/-3) + live fires -> under-fire / fail-open   (§1.1/§1.3)
        port fires + live silent        -> over-fire / fail-closed-reject (§1.2)

Oracle independence: there is NO Python leg-2 in this harness — live `mindc
check` is the sole, authoritative oracle, so the gate structurally cannot
collude with the port's tokeniser/position classifier (§1.4 null-gate class).

Generator axes:
  position-contexts (templates): let-RHS value-use, return value, callee,
    assign-target, annotation-type, struct-field-value, if/while/match
    cond-tail, match-guard, method-receiver, call-arg, binop-operand,
    index-base, nested value-use at depth 1-3.
  token-classes (fillers): bound-local, fn-value local, module decl fn, fresh
    undefined idents (regenerated per run), every parser stmt-keyword that
    lexes tk_ident-like, the folded non-ident keywords
    (import/use/pub/else/fn/let/if), bool literals, int literal, every
    BARE_BUILTINS name (extracted from resolve.rs each run), sampled std
    exports, qualified `Col::Red`/`Foo::Bar`, `__mind_`/`tensor.` prefixes.

CLI (matches mind-self-host.md step 2t):

    MINDC_SO=/path/port.so MINDC_BIN=./target/release/mindc \\
    python3 examples/mindc_mind/tc_differential_fuzz.py --rule E2002 --ci

    --rule E2002|E2003|E2009|E2012|all   rule(s) to diff (required)
    --ci             fixed seed; runs the planted-bug mutation gate FIRST
                     (Rule 3b: a harness that cannot kill a planted mutant is
                     a null oracle), then the live position sentinels, then
                     the sweep + template coverage floor; exit nonzero on ANY
                     divergence.
    --budget N       cap on generated cases (default 1500 = the full grid)
    --seed N         RNG seed (default 0xC1DF under --ci, else random)
    --mutate [MODE]  plant a deterministic bug in the harness's model of the
                     port to prove the gate catches it: bool-overfire
                     (default; mimics the historical E2002 r2 over-fire on
                     true/false), suppress (mimics the E2003 folded-keyword
                     fail-open), pos-off (position mapping one byte off).
                     Divergences are EXPECTED: exit 1 proves detection;
                     exit 3 means the planted bug was NOT caught (null gate).
    --shrink / --no-shrink   delta-debug each divergence to a minimal
                     parseable fixture (default: on)
    --so PATH        prebuilt port .so (equivalent to MINDC_SO)
    --main-mind PATH main.mind to build when no prebuilt .so is given

Exit codes: 0 clean · 1 divergence(s) or coverage-floor RED · 2 infra
failure · 3 mutation gate failed (planted bug not caught).

Env: MINDC_SO (prebuilt port .so, skips the build) · MINDC_BIN (mindc binary,
default `mindc`; used for the .so build AND as the live oracle).

Plumbing (build_so, the 5×i64 ctypes wrapper, BARE_BUILTINS / stmt-keyword /
bundled-std extraction, collect_decl_names, line/col math) is reused from
self_host_tc_unknown_ident_smoke.py, where it is verified against live.
"""
import argparse
import ctypes
import os
import random
import re
import subprocess
import sys
import tempfile
from collections import namedtuple

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
MAIN_MIND_DEFAULT = os.path.join(HERE, "main.mind")
RESOLVE_RS = os.path.join(REPO, "src", "type_checker", "resolve.rs")
PARSER_RS = os.path.join(REPO, "src", "parser", "mod.rs")
STDLIB_RS = os.path.join(REPO, "src", "project", "stdlib.rs")
STD_DIR = os.path.join(REPO, "std")

MARKER = "@@HOLE@@"
PARSE_ERROR_CODE = "E1001"
CI_SEED = 0xC1DF

RULES = {
    "E2002": "selftest_tc_unknown_ident",
    "E2003": "selftest_tc_unknown_call",
    "E2009": "selftest_tc_undeclared_assign",
    "E2012": "selftest_tc_fn_value_call",
}

# Keywords the lexer folds to a non-tk_ident token kind (Rule 3c) — the
# E2003 historical fail-open lived exactly here.
FOLDED_KW = {"import", "use", "pub", "else", "fn", "let", "if"}

# Every type_check diagnostic with a position (E1001 = parse error).
DIAG_ALL_RE = re.compile(r":(\d+):(\d+): error: .*\[type_check::(E\d+)\]")


def fail_infra(msg):
    print(f"INFRA FAIL: {msg}")
    sys.exit(2)


# ── table extraction (drift fails loud; reused from the E2002 smoke) ────────
def bare_builtins():
    with open(RESOLVE_RS) as f:
        rs = f.read()
    m = re.search(r"const BARE_BUILTINS: &\[&str\] = &\[(.*?)\];", rs, re.S)
    if not m:
        fail_infra("BARE_BUILTINS table not found in resolve.rs")
    names = re.findall(r'"([A-Za-z0-9_]+)"', m.group(1))
    if len(names) < 30:
        fail_infra(f"extracted only {len(names)} BARE_BUILTINS — drifted")
    return set(names)


def stmt_keywords():
    with open(PARSER_RS) as f:
        rs = f.read()
    m = re.search(
        r"fn stmt_keyword\(w: &\[u8\]\) -> Option<StmtKw> \{(.*?)\n\}", rs, re.S
    )
    if not m:
        fail_infra("stmt_keyword recogniser not found in parser/mod.rs")
    kws = set(re.findall(r'b"([a-z]+)"', m.group(1)))
    if len(kws) < 20:
        fail_infra(f"extracted only {len(kws)} stmt keywords — drifted")
    return kws


def bundled_modules():
    with open(STDLIB_RS) as f:
        rs = f.read()
    mods = re.findall(r'\("std\.([a-z0-9_]+)",\s*include_str!', rs)
    if len(mods) < 20:
        fail_infra(f"extracted only {len(mods)} bundled modules — drifted")
    return mods


# ── std decl-name mirror (only used to PICK std-export fillers; verdicts on
#    them come from live, never from this model) ─────────────────────────────
TOKEN_RE = re.compile(r'"[^"\n]*"|\'[^\'\n]*\'|[A-Za-z_]\w*|\S')


def is_word(t):
    return re.fullmatch(r"[A-Za-z_]\w*", t) is not None


def d_tokenize(src):
    return TOKEN_RE.findall(re.sub(r"//[^\n]*", "", src))


def collect_decl_names(src):
    toks = d_tokenize(src)
    n = len(toks)
    names = set()

    def skip_brace(j):
        d = 1
        while j < n and d > 0:
            if toks[j] == "{":
                d += 1
            elif toks[j] == "}":
                d -= 1
            j += 1
        return j

    i = 0
    while i < n:
        t = toks[i]
        if t == "fn" and i + 1 < n and is_word(toks[i + 1]):
            names.add(toks[i + 1])
            i += 2
            continue
        if t in ("struct", "const", "type") and i + 1 < n and is_word(toks[i + 1]):
            names.add(toks[i + 1])
            i += 2
            continue
        if t == "enum" and i + 2 < n and is_word(toks[i + 1]) and toks[i + 2] == "{":
            names.add(toks[i + 1])
            i = skip_brace(i + 3)
            continue
        if t == "{":
            i = skip_brace(i + 1)
            continue
        i += 1
    return names


def std_concat_and_names():
    concat = b""
    names = set()
    for m in bundled_modules():
        path = os.path.join(STD_DIR, m + ".mind")
        with open(path, "rb") as f:
            s = f.read()
        if not s.endswith(b"\n"):
            s += b"\n"
        concat += s
        names |= collect_decl_names(s.decode())
    return concat, names


# ── template bank: the position-context axis ───────────────────────────────
# Common preamble is diagnostic-clean under live `mindc check` (probed):
# `helper` = module decl fn, `bnd` = bound local, `fv` = fn-value local,
# `Pt` = struct for field-value templates, `Col` = enum for qualified names.
PRELUDE_SRC = (
    "fn helper(a: i64) -> i64 { return a }\n"
    "struct Pt { x: i64 }\n"
    "enum Col { Red, Green }\n"
)

TEMPLATES = [
    ("let_rhs", ["let z = @@HOLE@@"]),
    ("ret_val", ["return @@HOLE@@"]),
    ("callee", ["let z = @@HOLE@@(1)"]),
    ("assign_tgt", ["@@HOLE@@ = 2"]),
    ("annot_type", ["let t: @@HOLE@@ = 1"]),
    ("sfield_val", ["let q = Pt { x: @@HOLE@@ }"]),
    ("if_cond", ["if @@HOLE@@ {", "    let d = 1", "}"]),
    ("while_cond", ["while @@HOLE@@ {", "    break", "}"]),
    ("match_scrut", ["match @@HOLE@@ {", "    _ => 1,", "}"]),
    ("match_guard", ["match bnd {", "    _ if @@HOLE@@ => 1,", "    _ => 2,", "}"]),
    # Match-arm BODY value position (`=> <atom>`). `=>` lexes as tk_eq+tk_gt in
    # the self-host lexer, so a folded keyword here was invisible to the
    # value-atom classifier (codex Finding 2, 2026-07-24): live parses the arm
    # body as Lit(Ident) and fires E2002 on use/pub/else/import/fn/let.
    ("match_arm_val", ["match bnd {", "    _ => @@HOLE@@,", "}"]),
    # Match-arm BODY assign l-value (`=> x = 5`). A BARE arm body (l-value
    # directly after `=>`) — live attributes the E2009 diagnostic to the arm-
    # PATTERN column (`_`), NOT the l-value token, so the port must DECLINE at
    # the l-value here (codex Finding 2, 2026-07-24). Distinct from a BLOCK arm
    # `_ => { x = 5 }`, where live DOES fire E2009 at the l-value and the port
    # keeps firing — the two are separated by the l-value's prev token (`=>`
    # vs `{`).
    ("match_arm_assign", ["match bnd {", "    _ => @@HOLE@@ = 5,", "}"]),
    ("method_recv", ["let r = @@HOLE@@.foo(2)"]),
    ("call_arg", ["let a = helper(@@HOLE@@)"]),
    ("binop_rhs", ["let b = 1 + @@HOLE@@"]),
    ("index_base", ["let ix = @@HOLE@@[0]"]),
]


def nested_templates(rng):
    """Value-use hole under 1-3 levels of if/while nesting (scope-frame depth)."""
    out = []
    for depth in (1, 2, 3):
        kinds = [rng.choice(("if", "while")) for _ in range(depth)]
        lines = []
        for lvl, kind in enumerate(kinds):
            pad = "    " * lvl
            hdr = "if 1 > 0 {" if kind == "if" else "while 0 > 1 {"
            lines.append(pad + hdr)
        lines.append("    " * depth + "let nz = @@HOLE@@")
        for lvl in range(depth - 1, -1, -1):
            lines.append("    " * lvl + "}")
        out.append((f"nested_d{depth}", lines))
    return out


def deep_binding_cases(rng):
    """Deep-scope stress: programs with 251..N flat local bindings then a use
    of an EARLY or LATE one — the scope-table cap class (codex Finding 1, the
    old fixed 250-entry buffer false-fired E2002 on valid code past 250 locals).

    A BOUND name must resolve clean (live no-fire) at every depth: this is the
    false-fire regression guard — before the fix the port fired at N>=251 while
    live stayed clean, which the sweep flags as a divergence. An UNDEFINED name
    (kept well under the 4096 fail-closed cap) must fire in BOTH port and live,
    proving the deep regime is a non-null gate rather than a blanket suppressor.
    """
    out = []
    for n in (251, 260, 300, 512, 1024):
        binds = "".join(f"    let b{i:04d} = 0\n" for i in range(n))
        for which, target in (("early", "b0000"), ("late", f"b{n - 1:04d}")):
            src = (PRELUDE_SRC + "fn main() -> i64 {\n" + binds
                   + f"    let z = {target}\n    return 0\n}}\n")
            pos = src.index(f"let z = {target}") + len("let z = ")
            out.append(Case(f"deep_bind_{which}_n{n}", "bound_local", target, src, pos))
    # UNDEFINED-at-depth — MUST fire E2002 in both port and live at every
    # depth, PAST the old premature 4096 fail-closed sentinel (4097/8192): the
    # sentinel tripped at cnt>=4096 while the (n+16)-entry buffer is nowhere
    # near full, and tc_sf_lookup read the negative sentinel as BOUND, SILENTLY
    # SUPPRESSING a genuine unknown ident past 4096 bindings (a fail-OPEN; live
    # still fires). These deep-undefined cases are the regression guard for
    # that class — before the sentinel-removal fix, port=0 while live=E2002 at
    # n>=4097, which the sweep flags as a divergence.
    for n in (251, 300, 1000, 4097, 8192):
        binds = "".join(f"    let b{i:05d} = 0\n" for i in range(n))
        und = f"und_{rng.randrange(10 ** 6):06d}"
        src = (PRELUDE_SRC + "fn main() -> i64 {\n" + binds
               + f"    let z = {und}\n    return 0\n}}\n")
        pos = src.index(f"let z = {und}") + len("let z = ")
        out.append(Case(f"deep_bind_undef_n{n}", "undefined", und, src, pos))
    return out


def assemble(body_lines):
    """Full marked source for one template instance (exactly one MARKER)."""
    src = PRELUDE_SRC + "fn main() -> i64 {\n"
    src += "    let bnd = 1\n    let fv = helper\n"
    for ln in body_lines:
        src += "    " + ln + "\n"
    src += "    return 0\n}\n"
    if src.count(MARKER) != 1:
        fail_infra("template must contain exactly one hole marker")
    return src


# ── filler matrix: the token-class axis ────────────────────────────────────
def build_fillers(rng, bare, kws, std_names):
    fillers = [
        ("bound_local", "bnd"),
        ("fn_value_local", "fv"),
        ("decl_fn", "helper"),
    ]
    # Fresh undefined names, regenerated every run so they cannot drift into
    # accidental resolvability.
    for i in range(2):
        fillers.append(("undefined", f"und_{rng.randrange(10 ** 6):06d}_{i}"))
    kw_all = kws | {"else", "mut", "in", "as"}
    for kw in sorted(kw_all - FOLDED_KW):
        fillers.append(("kw_identish", kw))
    for kw in sorted(FOLDED_KW):
        fillers.append(("kw_folded", kw))
    fillers += [("bool_lit", "true"), ("bool_lit", "false"), ("int_lit", "42")]
    for b in sorted(bare):
        fillers.append(("builtin", b))
    for s in sorted(rng.sample(sorted(std_names), min(3, len(std_names)))):
        fillers.append(("std_export", s))
    fillers += [
        ("qualified", "Col::Red"),
        ("qualified", "Foo::Bar"),
        ("mind_prefix", "__mind_alloc"),
        ("tensor_prefix", "tensor.add"),
    ]
    return fillers


Case = namedtuple("Case", "template fclass filler src pos")


def gen_cases(rng, fillers, templates, budget):
    cases = []
    for tname, body in templates:
        marked = assemble(body)
        pos = marked.index(MARKER)
        for fclass, tok in fillers:
            cases.append(Case(tname, fclass, tok, marked.replace(MARKER, tok, 1), pos))
    rng.shuffle(cases)
    return cases[:budget]


# ── live oracle: one `mindc check` per unique source, cached ───────────────
class LiveOracle:
    def __init__(self, mindc, workdir):
        self.mindc = mindc
        self.workdir = workdir
        self.cache = {}
        self.runs = 0

    def check(self, src):
        """-> (codes_all: set, at: {(line, col): set(codes)})"""
        hit = self.cache.get(src)
        if hit is not None:
            return hit
        path = os.path.join(self.workdir, f"case_{self.runs}.mind")
        self.runs += 1
        with open(path, "w") as f:
            f.write(src)
        r = subprocess.run(
            [self.mindc, "check", path], capture_output=True, text=True
        )
        codes_all, at = set(), {}
        for m in DIAG_ALL_RE.finditer(r.stdout + r.stderr):
            code = m.group(3)
            codes_all.add(code)
            at.setdefault((int(m.group(1)), int(m.group(2))), set()).add(code)
        self.cache[src] = (codes_all, at)
        return codes_all, at


def line_col(src, pos):
    line = src.count("\n", 0, pos) + 1
    col = pos - (src.rfind("\n", 0, pos) + 1) + 1
    return line, col


# ── the port: ctypes over the self-host .so ────────────────────────────────
def build_so(args):
    so = args.so or os.environ.get("MINDC_SO")
    if so:
        return so, False
    mindc = os.environ.get("MINDC_BIN", "mindc")
    out = tempfile.NamedTemporaryFile(suffix=".so", delete=False).name
    cmd = [mindc, args.main_mind, "--emit-shared", out]
    print("BUILD:", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-4000:])
        print(r.stderr[-4000:])
        fail_infra(f"port .so build failed rc={r.returncode}")
    return out, True


class Port:
    def __init__(self, so_path, std_concat, rules):
        st = os.stat(so_path)
        print(f"SO: {so_path} ({st.st_size} bytes)")
        if st.st_size < 4096:
            fail_infra(".so too small (stub?)")
        self.lib = ctypes.CDLL(so_path)
        self.fns = {}
        # Only the REQUESTED rules' exports are required — a rule whose port
        # has not landed yet (e.g. E2003 while its port is in flight) must not
        # block fuzzing the landed ones.
        for code in rules:
            sym = RULES[code]
            try:
                fn = getattr(self.lib, sym)
            except AttributeError:
                fail_infra(f"export {sym} missing from {so_path} — is the "
                           f"{code} port landed in this main.mind?")
            fn.argtypes = [ctypes.c_int64] * 5
            fn.restype = ctypes.c_int64
            self.fns[code] = fn
        self.sb = ctypes.create_string_buffer(std_concat, len(std_concat))
        self.sp = ctypes.cast(self.sb, ctypes.c_void_p).value
        self.slen = len(std_concat)

    def call(self, rule, src, pos):
        data = src.encode()
        b = ctypes.create_string_buffer(data, len(data))
        return self.fns[rule](
            ctypes.cast(b, ctypes.c_void_p).value, len(data), pos, self.sp, self.slen
        )


def modeled_verdict(port, rule, fclass, src, pos, mutation):
    """The harness's model of the port verdict, with the planted-bug hooks.

    Mutations live HERE (the comparison model), not in main.mind — each
    deterministically mimics a real historical port-bug class so the gate can
    prove it would catch that class (Rule 3b non-null requirement).
    """
    if mutation == "suppress":
        return -3  # E2003-class fail-open: decline everywhere
    if mutation == "bool-overfire" and fclass == "bool_lit":
        return 1  # E2002 r2-class over-fire on true/false
    if mutation == "pos-off":
        return port.call(rule, src, pos + 1)
    return port.call(rule, src, pos)


# ── divergence evaluation ──────────────────────────────────────────────────
Divergence = namedtuple("Divergence", "case rule raw live_at kind")


def eval_case(case, rule, port, oracle, mutation):
    """-> ('unparseable', None) | ('agree', None) | ('divergent', Divergence)"""
    codes_all, at = oracle.check(case.src)
    if PARSE_ERROR_CODE in codes_all:
        return "unparseable", None
    lc = line_col(case.src, case.pos)
    live_at = at.get(lc, set())
    live_fires = rule in live_at
    raw = modeled_verdict(port, rule, case.fclass, case.src, case.pos, mutation)
    port_fires = raw == 1
    if port_fires == live_fires:
        return "agree", None
    if live_fires:
        kind = "fail-open decline" if raw == -3 else "under-fire"
    else:
        kind = "over-fire / fail-closed-reject"
    return "divergent", Divergence(case, rule, raw, sorted(live_at), kind)


# ── shrinker: delta-debug to a minimal parseable divergent fixture ─────────
def _still_divergent(marked, div, port, oracle, mutation):
    if marked.count(MARKER) != 1:
        return None
    pos = marked.index(MARKER)
    src = marked.replace(MARKER, div.case.filler, 1)
    cand = div.case._replace(src=src, pos=pos)
    state, _ = eval_case(cand, div.rule, port, oracle, mutation)
    return cand if state == "divergent" else None


def shrink(div, port, oracle, mutation):
    case = div.case
    marked = case.src[: case.pos] + MARKER + case.src[case.pos + len(case.filler):]
    best = case
    changed = True
    while changed:
        changed = False
        lines = marked.split("\n")
        # pass 1: drop any single line that does not hold the hole
        for i in range(len(lines)):
            if MARKER in lines[i] or not lines[i].strip():
                continue
            cand_marked = "\n".join(lines[:i] + lines[i + 1:])
            cand = _still_divergent(cand_marked, div, port, oracle, mutation)
            if cand is not None:
                marked, best, changed = cand_marked, cand, True
                break
        if changed:
            continue
        # pass 2: unwrap one nesting level (drop a `... {` header + its `}`)
        for i, ln in enumerate(lines):
            s = ln.strip()
            if not s.endswith("{") or MARKER in ln or s.startswith("fn "):
                continue
            depth = 0
            close = None
            for j in range(i, len(lines)):
                depth += lines[j].count("{") - lines[j].count("}")
                if depth == 0 and j > i:
                    close = j
                    break
            if close is None or lines[close].strip() != "}":
                continue
            kept = [lines[k] for k in range(len(lines)) if k not in (i, close)]
            cand_marked = "\n".join(kept)
            cand = _still_divergent(cand_marked, div, port, oracle, mutation)
            if cand is not None:
                marked, best, changed = cand_marked, cand, True
                break
    return div._replace(case=best)


def print_divergence(div, shrunk=False):
    c = div.case
    line, col = line_col(c.src, c.pos)
    tag = "SHRUNK FIXTURE" if shrunk else "DIVERGENCE"
    print(f"\n{tag}: rule={div.rule} kind={div.kind}")
    print(
        f"  template={c.template} filler_class={c.fclass} filler={c.filler!r} "
        f"pos={c.pos} ({line}:{col}) port_raw={div.raw} live_at={div.live_at}"
    )
    print("  " + "-" * 60)
    for ln in c.src.rstrip("\n").split("\n"):
        print("  | " + ln)
    print("  " + "-" * 60)


# ── live position sentinels (prove line:col matching is intact) ────────────
SENTINEL_E2002 = "fn main() -> i64 {\n    return nope\n}\n"
SENTINEL_E2003 = "fn main() -> i64 {\n    return foo(1)\n}\n"


def sentinels(oracle):
    s1 = SENTINEL_E2002
    _, at1 = oracle.check(s1)
    if "E2002" not in at1.get(line_col(s1, s1.index("nope")), set()):
        fail_infra("live sentinel — E2002 not at the undefined value use")
    s2 = SENTINEL_E2003
    _, at2 = oracle.check(s2)
    if "E2003" not in at2.get(line_col(s2, s2.index("foo")), set()):
        fail_infra("live sentinel — E2003 not at the undefined callee")
    print("live sentinels: E2002 value-use + E2003 callee (2/2)")


# ── planted-bug mutation gate (Rule 3b: runs FIRST in --ci) ────────────────
# Mini-grid with a live-firing parseable case for every rule:
#   E2002 undefined×let_rhs · E2003 undefined×callee ·
#   E2009 undefined×assign_tgt · E2012 fv×callee
GATE_TEMPLATES = [t for t in TEMPLATES if t[0] in ("let_rhs", "callee", "assign_tgt")]


def mutation_gate(rule, port, oracle, rng):
    fillers = [
        ("undefined", f"und_{rng.randrange(10 ** 6):06d}_g"),
        ("fn_value_local", "fv"),
        ("bound_local", "bnd"),
        ("bool_lit", "true"),
    ]
    cases = gen_cases(rng, fillers, GATE_TEMPLATES, budget=10 ** 6)
    caught = None
    firing = 0
    for case in cases:
        codes_all, at = oracle.check(case.src)
        if PARSE_ERROR_CODE in codes_all:
            continue
        if rule in at.get(line_col(case.src, case.pos), set()):
            firing += 1
            state, div = eval_case(case, rule, port, oracle, "suppress")
            if state == "divergent" and caught is None:
                caught = div
    if firing == 0:
        fail_infra(
            f"mutation gate: no live-firing {rule} case in the gate grid — "
            "the planted bug cannot be exercised"
        )
    if caught is None:
        print(
            f"MUTATION GATE FAILED: planted suppress-mutant survived on "
            f"{firing} live-firing {rule} cases — this harness is a null gate"
        )
        sys.exit(3)
    shrunk = shrink(caught, port, oracle, "suppress")
    ok_kind = shrunk.kind in ("fail-open decline", "under-fire")
    if not ok_kind:
        print("MUTATION GATE FAILED: shrunk class != planted fail-open class")
        sys.exit(3)
    print(
        f"mutation gate: planted fail-open mutant CAUGHT on {rule} "
        f"({firing} live-firing gate cases; shrunk to "
        f"{len(shrunk.case.src.splitlines())} lines, kind={shrunk.kind})"
    )


# ── sweep ──────────────────────────────────────────────────────────────────
def sweep(rules, cases, port, oracle, mutation, do_shrink, max_shrink=5):
    stats = {"generated": len(cases), "unparseable": 0, "scored": 0}
    coverage = {}  # (template, fclass) -> [generated, parseable]
    divergences = []
    for case in cases:
        cell = coverage.setdefault((case.template, case.fclass), [0, 0])
        cell[0] += 1
        codes_all, _ = oracle.check(case.src)
        if PARSE_ERROR_CODE in codes_all:
            stats["unparseable"] += 1
            continue
        cell[1] += 1
        for rule in rules:
            state, div = eval_case(case, rule, port, oracle, mutation)
            stats["scored"] += 1
            if state == "divergent":
                divergences.append(div)
    for div in divergences[:max_shrink] if do_shrink else []:
        print_divergence(div)
        print("  shrinking...")
        print_divergence(shrink(div, port, oracle, mutation), shrunk=True)
    for div in divergences[max_shrink:] if do_shrink else divergences:
        print_divergence(div)
    return stats, coverage, divergences


def coverage_report(coverage, templates, enforce_floor):
    print("\ncoverage (template: parseable/generated; per-cell holes feed §2.2):")
    by_template = {}
    for (tname, _fclass), (gen, ok) in sorted(coverage.items()):
        agg = by_template.setdefault(tname, [0, 0, 0])
        agg[0] += gen
        agg[1] += ok
        if ok == 0:
            agg[2] += 1
    red = []
    for tname, _body in templates:
        gen, ok, holes = by_template.get(tname, [0, 0, 0])
        mark = "ok " if ok > 0 else "RED"
        print(f"  {mark} {tname:12s} {ok:4d}/{gen:<4d} parseable, {holes} empty cells")
        if ok == 0 and gen > 0:
            red.append(tname)
    if red and enforce_floor:
        print(f"COVERAGE FLOOR RED: templates at 100% parse-discard: {red}")
        return False
    return True


# ── main ───────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="tcdiff — port-vs-live differential fuzzer "
        "(VERIFICATION_APPARATUS.md §2.1)"
    )
    ap.add_argument("--rule", required=True, choices=[*RULES, "all"])
    ap.add_argument("--ci", action="store_true")
    ap.add_argument("--budget", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument(
        "--mutate",
        nargs="?",
        const="bool-overfire",
        choices=["bool-overfire", "suppress", "pos-off"],
        default=None,
    )
    shr = ap.add_mutually_exclusive_group()
    shr.add_argument("--shrink", dest="shrink", action="store_true", default=True)
    shr.add_argument("--no-shrink", dest="shrink", action="store_false")
    ap.add_argument("--so", default=None)
    ap.add_argument("--main-mind", default=MAIN_MIND_DEFAULT)
    args = ap.parse_args()

    seed = args.seed if args.seed is not None else (
        CI_SEED if args.ci else random.randrange(2 ** 32)
    )
    rng = random.Random(seed)
    rules = list(RULES) if args.rule == "all" else [args.rule]
    mutation = args.mutate
    print(
        f"tcdiff: rules={rules} seed={seed:#x} budget={args.budget} "
        f"ci={args.ci} mutate={mutation}"
    )
    if mutation:
        print(
            f"MUTATE MODE ({mutation}): planted model bug ACTIVE — divergences "
            "are EXPECTED; catching them is the proof the gate works"
        )

    so, built = build_so(args)
    std_concat, std_names = std_concat_and_names()
    port = Port(so, std_concat, rules)
    bare = bare_builtins()
    kws = stmt_keywords()
    print(
        f"tables: {len(bare)} bare builtins, {len(kws)} parser stmt-keywords, "
        f"{len(std_names)} std exports ({len(std_concat)} bytes std concat)"
    )

    fillers = build_fillers(rng, bare, kws, std_names)
    templates = TEMPLATES + nested_templates(rng)
    exit_code = 0
    with tempfile.TemporaryDirectory() as workdir:
        oracle = LiveOracle(os.environ.get("MINDC_BIN", "mindc"), workdir)
        sentinels(oracle)
        if args.ci and not mutation:
            for rule in rules:
                mutation_gate(rule, port, oracle, random.Random(seed ^ 0xA5))
        cases = gen_cases(rng, fillers, templates, args.budget)
        # Deep-binding stress is always appended (never budget-trimmed) so the
        # scope-cap class is exercised on every --ci run.
        cases = cases + deep_binding_cases(rng)
        stats, coverage, divergences = sweep(
            rules, cases, port, oracle, mutation, args.shrink
        )
        floor_ok = coverage_report(coverage, templates, enforce_floor=args.ci)
        parseable = stats["generated"] - stats["unparseable"]
        rate = 100.0 * parseable / max(1, stats["generated"])
        print(
            f"\ntcdiff sweep: rules={rules} generated={stats['generated']} "
            f"parseable={parseable} ({rate:.1f}%) unparseable="
            f"{stats['unparseable']} scored={stats['scored']} "
            f"divergences={len(divergences)}"
        )
        if divergences:
            print(
                "RESULT: DIVERGENT — the pure-MIND port disagrees with live "
                "mindc on parseable code"
                + (" (planted mutant caught — the gate works)" if mutation else "")
            )
            exit_code = 1
        elif mutation:
            print(
                "MUTATION GATE FAILED: planted bug produced ZERO divergences — "
                "this harness is a null gate"
            )
            exit_code = 3
        elif not floor_ok:
            exit_code = 1
        else:
            print("RESULT: 0 divergences — port agrees with live mindc "
                  "on every parseable generated case")
    # deferred: nightly batch mode (N unique-named single-fn modules concatenated
    # behind one `mindc check` to amortize subprocess cost 10-50x, §2.1) — not
    # needed at the current ~40ms/case scale; upgrade path: batch gen_cases
    # output into one file with renamed fns + per-case position offsets.
    if built:
        try:
            os.unlink(so)
        except OSError:
            pass
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
