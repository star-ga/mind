#!/usr/bin/env python3
"""CPU-as-oracle smoke for the pure-MIND B1 E2002 rule — unknown identifier.

Ports resolve.rs `ident_resolvable` (UNKNOWN_IDENT_CODE, E2002): a bare
identifier in VALUE/USE position that resolves to NOTHING. The exact Rust
condition (resolve.rs:551-570; the walk arm resolve.rs:744-771): E2002 fires
at a walked `Node::Lit(Literal::Ident(name))` iff

    NOT ( scopes.contains(name)          [D2 — live local binding]
       || syms.name_resolvable(name)     — module decls + prelude [D1]
                                           OR bundled-std exports  [D3]
       || name == "bytes"
       || name.contains("::")            — qualified path, never bare-undef
       || cm_symbol_exported(name) )     — vacuous in the single-file model

CRUCIAL grounded difference from E2003/E2012 (live-verified below): the
BARE_BUILTINS set, "__mind_"/"tensor." prefixes, "gen_deref", and the E2024
intrinsic table are CALL-only exclusions — NOT in ident_resolvable. A bare
`sqrt` or `__mind_alloc` in value position DOES fire E2002.

The pure-MIND twin `selftest_tc_unknown_ident(src, src_len, pos, std_src,
std_len)` takes the fixture source, the byte offset of the queried VALUE-USE
ident, and the CONCATENATED bundled-std source (the D3 caller-supplied
model). Returns 1 (E2002 fires) / 0 (it does not) / -3 (fail-closed
sentinel: not an ident / not a classifiable value-use position — callee,
binding/decl name, annotation, member name, assign target, struct-lit head).

Three-way agreement, machine-checked (no hand-authored verdict table), with
LIVE AUTHORITATIVE ON EVERY CASE and leg 2 fully independent of the port's
position classifier:
  1. MIND core over (source, pos, std concat).
  2. The RESOLUTION UNION recomputed independently in Python — the D2 frame
     semantics + the D1/D3 decl-set mirrors + prelude + "bytes" — with ZERO
     position/shape guards (it never replicates tc_ui_shape's heuristics;
     it presumes the fixture's declared value-use position and computes only
     resolvability). BARE_BUILTINS re-extracted from resolve.rs every run.
  3. The LIVE `mindc check` oracle: E2002 present at exactly the query
     line:col <=> verdict 1, checked for EVERY case including declines.

Per-case modes (position-classification claims are validated ONLY by live):
  classify — MIND must answer 0/1 and got == leg2 == live.
  never    — structurally-never-fires positions (qualified `::` head, folded
             tensor/string static heads): got == 0 AND live == 0 (leg 2 does
             not apply — the ident is not a walked value use).
  decline  — out-of-domain positions (callee, binders, annotation types,
             member names, struct-lit heads, assign targets): got == -3 AND
             live == 0. Declining where live FIRES is the fail-open bug this
             mode exists to catch — such a case fails RED.

Env: MINDC_SO (prebuilt .so, skips the build) or MINDC_BIN (default mindc).
Template: self_host_tc_fn_value_call_smoke.py (D4).
"""
import ctypes
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN_MIND = os.path.join(HERE, "main.mind")
REPO = os.path.dirname(os.path.dirname(HERE))
RESOLVE_RS = os.path.join(REPO, "src", "type_checker", "resolve.rs")
STDLIB_RS = os.path.join(REPO, "src", "project", "stdlib.rs")
STD_DIR = os.path.join(REPO, "std")

PRELUDE = {"Result", "Option", "Ok", "Err", "Some", "None"}

# ── fixtures ────────────────────────────────────────────────────────────────
S_UNDEF = """\
fn main() -> i64 {
    return nope
}
"""

S_LOCAL = """\
fn main() -> i64 {
    let x = 1
    return x
}
"""

S_PARAM = """\
fn use_it(g: i64) -> i64 {
    return g
}
fn main() -> i64 {
    return use_it(3)
}
"""

S_DECL = """\
fn add1(x: i64) -> i64 { return x + 1 }
fn main() -> i64 {
    let g = add1
    return 0
}
"""

S_STD = """\
fn main() -> i64 {
    let v = vec_new
    return 0
}
"""

S_BUILTIN = """\
fn main() -> i64 {
    let y = sqrt
    return 0
}
"""

S_MIND_PREFIX = """\
fn main() -> i64 {
    let p = __mind_alloc
    return 0
}
"""

S_BYTES = """\
fn main() -> i64 {
    let x = bytes
    return 0
}
"""

S_PRELUDE_USE = """\
fn main() -> i64 {
    let x = None
    return 0
}
"""

S_BEFORE = """\
fn main() -> i64 {
    let y = w + 1
    let w = 2
    return y
}
"""

S_SHADOW = """\
fn add1(x: i64) -> i64 { return x + 1 }
fn main() -> i64 {
    let sqrt = 1
    return sqrt
}
"""

S_SIBLING = """\
fn main() -> i64 {
    if 1 > 0 {
        let w = 5
    }
    return w
}
"""

S_NESTED_OK = """\
fn main() -> i64 {
    let a = 1
    if 1 > 0 {
        if 2 > 1 {
            return a
        }
    }
    return 0
}
"""

S_FIELD_UNDEF = """\
fn main() -> i64 {
    let z = undef.f
    return 0
}
"""

S_FIELD_BOUND = """\
fn main() -> i64 {
    let s = 1
    let z = s.f
    return 0
}
"""

S_QUALIFIED = """\
enum Col { Red, Green }
fn main() -> i64 {
    let z = Col::Red
    return 0
}
"""

S_CALLEE = """\
fn main() -> i64 {
    return foo(1)
}
"""

S_LET_NAME = """\
fn main() -> i64 {
    let x = 1
    return 0
}
"""

S_ANNOT = """\
fn main() -> i64 {
    let x: i64 = 1
    return x
}
"""

# ── the 6 previously-divergent shapes (cross-family review 2026-07-23) ─────
S_SL_FIELD = """\
struct P { x: i64 }
fn main() -> i64 {
    let q = P { x: undef }
    return 0
}
"""

S_SL_FIELD_OK = """\
struct P { x: i64 }
fn main() -> i64 {
    let a = 1
    let q = P { x: a }
    return 0
}
"""

S_IF_COND = """\
fn main() -> i64 {
    if undef {
        return 1
    }
    return 0
}
"""

S_IF_COND_OK = """\
fn main() -> i64 {
    let c = 1
    if c {
        return 1
    }
    return 0
}
"""

S_WHILE_COND = """\
fn main() -> i64 {
    while undef {
        return 1
    }
    return 0
}
"""

S_MATCH_SCRUT = """\
fn main() -> i64 {
    match undef {
        _ => 1,
    }
    return 0
}
"""

S_GUARD = """\
fn main() -> i64 {
    let x = 1
    match x {
        _ if undef => 1,
        _ => 2,
    }
    return 0
}
"""

S_GUARD_OK = """\
fn main() -> i64 {
    let x = 1
    let g = 1
    match x {
        _ if g => 1,
        _ => 2,
    }
    return 0
}
"""

S_RECV = """\
fn main() -> i64 {
    let z = undef.foo(2)
    return 0
}
"""

S_RECV_BOUND = """\
fn main() -> i64 {
    let s = 1
    let z = s.foo()
    return 0
}
"""

S_MODHEAD = """\
module mymod {
    pub fn addx(a: i64) -> i64 { return a }
}
fn main() -> i64 {
    let z = mymod.addx(1)
    return 0
}
"""

# ── varied value-use positions ─────────────────────────────────────────────
S_ARG = """\
fn f(a: i64) -> i64 { return a }
fn main() -> i64 {
    return f(undef)
}
"""

S_PAREN = """\
fn main() -> i64 {
    return (undef)
}
"""

S_AS = """\
fn main() -> i64 {
    let z = undef as i64
    return 0
}
"""

S_CHAIN = """\
fn main() -> i64 {
    let z = undef.a.b
    return 0
}
"""

S_BINOP = """\
fn main() -> i64 {
    let z = 1 + undef
    return 0
}
"""

# ── structurally-never / out-of-domain fixtures ────────────────────────────
S_TENSOR_HEAD = """\
fn main() -> i64 {
    let z = tensor.zeros(4)
    return 0
}
"""

S_STRING_HEAD = """\
fn main() -> i64 {
    let z = string.from_utf8_bytes(1)
    return 0
}
"""

S_METHOD_NAME = """\
fn main() -> i64 {
    let s = 1
    let z = s.foo(2)
    return 0
}
"""

S_SL_HEAD = """\
struct P { x: i64 }
fn main() -> i64 {
    let z = Undef { x: 1 }
    return 0
}
"""

S_BINDER = """\
fn main() -> i64 {
    let x = 1
    match x {
        n => n,
    }
    return 0
}
"""

S_ASSIGN = """\
fn main() -> i64 {
    let counter = 1
    conter = counter + 1
    return 0
}
"""

S_PAT_FIELD = """\
struct P { x: i64 }
fn main() -> i64 {
    let x = 1
    let p = P { x: x }
    match p {
        P { x: v } => v,
    }
    return 0
}
"""

# ── bool literals + reserved keywords (re-blind-review 2026-07-23) ─────────
S_BOOL_LET = """\
fn main() -> i64 {
    let b = true
    return 0
}
"""

S_BOOL_IF = """\
fn main() -> i64 {
    if true {
        return 1
    }
    return 0
}
"""

S_BOOL_FIELD = """\
struct P { b: i64 }
fn main() -> i64 {
    let q = P { b: true }
    return 0
}
"""

S_BOOL_ARG = """\
fn f(a: i64) -> i64 { return a }
fn main() -> i64 {
    return f(true)
}
"""

S_BOOL_RET = """\
fn main() -> i64 {
    return true
}
"""

S_BOOL_WHILE = """\
fn main() -> i64 {
    while false {
        return 1
    }
    return 0
}
"""

S_KW_LOOP_EXPR = """\
fn main() -> i64 {
    let z = loop
    return 0
}
"""

S_KW_BREAK = """\
fn main() -> i64 {
    let mut i = 0
    while i < 3 {
        i = i + 1
        break
    }
    return i
}
"""

S_KW_CONTINUE = """\
fn main() -> i64 {
    let mut i = 0
    while i < 3 {
        i = i + 1
        continue
    }
    return i
}
"""

S_KW_FOR = """\
fn main() -> i64 {
    let mut s = 0
    for i in 0..3 {
        s = s + i
    }
    return s
}
"""

# (src, needle, occurrence, mode, note) — pos is the start byte of the
# `occurrence`-th match of `needle`, which begins at the queried ident.
# Modes: "classify" (got == leg2-resolution == live), "never" (got == 0 ==
# live), "decline" (got == -3 AND live == 0). Verdicts are NEVER
# hand-authored — leg 2 and live decide.
CASES = [
    ("classify", S_UNDEF, "nope", 0, "genuinely-undefined ident -> E2002"),
    ("classify", S_LOCAL, "x\n", 0, "local let used -> no (D2)"),
    ("classify", S_PARAM, "g\n", 0, "param used -> no (D2)"),
    ("classify", S_DECL, "add1\n", 0, "module fn as value -> no (D1)"),
    ("classify", S_STD, "vec_new", 0, "std export as value -> no (D3)"),
    ("classify", S_BUILTIN, "sqrt", 0, "bare builtin as VALUE -> E2002 (call-only excl)"),
    ("classify", S_MIND_PREFIX, "__mind_alloc", 0, "__mind_ as VALUE -> E2002 (call-only excl)"),
    ("classify", S_BYTES, "bytes", 0, "bytes builtin value type -> no"),
    ("classify", S_PRELUDE_USE, "None", 0, "prelude variant used -> no (D1 prelude)"),
    ("classify", S_BEFORE, "w + 1", 0, "use before let -> E2002 (not yet bound)"),
    ("classify", S_SHADOW, "sqrt\n", 0, "local shadows builtin name -> no (D2)"),
    ("classify", S_SIBLING, "w\n}", 0, "binding dead in exited block -> E2002"),
    ("classify", S_NESTED_OK, "a\n", 0, "outer let, doubly-nested use -> no (D2)"),
    ("classify", S_FIELD_UNDEF, "undef.f", 0, "field-access head undefined -> E2002"),
    ("classify", S_FIELD_BOUND, "s.f", 0, "field-access head bound -> no (D2)"),
    # the 6 previously-divergent shapes (review 2026-07-23) — must classify:
    ("classify", S_SL_FIELD, "undef }", 0, "struct-lit FIELD VALUE -> E2002 (div 1)"),
    ("classify", S_SL_FIELD_OK, "a }", 0, "struct-lit field value bound -> no (div 1)"),
    ("classify", S_IF_COND, "undef {", 0, "if-cond tail -> E2002 (div 2)"),
    ("classify", S_IF_COND_OK, "c {", 0, "if-cond tail bound -> no (div 2)"),
    ("classify", S_WHILE_COND, "undef {", 0, "while-cond tail -> E2002 (div 3)"),
    ("classify", S_MATCH_SCRUT, "undef {", 0, "match scrutinee tail -> E2002 (div 4)"),
    ("classify", S_RECV, "undef.foo", 0, "method receiver -> E2002 (div 5)"),
    ("classify", S_RECV_BOUND, "s.foo", 0, "method receiver bound -> no (div 5)"),
    ("classify", S_MODHEAD, "mymod.addx(1", 0, "module-decl head IS walked -> E2002 (div 5)"),
    ("classify", S_GUARD, "undef =>", 0, "match-arm guard tail -> E2002 (div 6)"),
    ("classify", S_GUARD_OK, "g =>", 0, "match-arm guard tail bound -> no (div 6)"),
    # varied value-use positions:
    ("classify", S_ARG, "undef)", 0, "call ARG -> E2002 (args walked)"),
    ("classify", S_PAREN, "undef)", 0, "parenthesised use -> E2002"),
    ("classify", S_AS, "undef as", 0, "cast operand -> E2002"),
    ("classify", S_CHAIN, "undef.a.b", 0, "field chain head -> E2002"),
    ("classify", S_BINOP, "undef\n", 0, "binop operand -> E2002"),
    # structurally-never positions (got == 0 == live; not walked value uses):
    ("never", S_QUALIFIED, "Col::Red\n", 0, "qualified head -> never (:: exclusion)"),
    ("never", S_TENSOR_HEAD, "tensor.zeros", 0, "tensor.* folded head -> never"),
    ("never", S_STRING_HEAD, "string.from", 0, "string static-type head -> never"),
    # out-of-domain positions (got == -3 AND live == 0):
    ("decline", S_CALLEE, "foo(1", 0, "callee position (E2003 domain)"),
    ("decline", S_LET_NAME, "x = 1", 0, "let-binding name"),
    ("decline", S_ANNOT, "i64 = 1", 0, "annotation type name"),
    ("decline", S_METHOD_NAME, "foo(2", 0, "method NAME after dot"),
    ("decline", S_SL_HEAD, "Undef {", 0, "struct-lit HEAD (never walked)"),
    ("decline", S_BINDER, "n =>", 0, "bare match-pattern binder"),
    ("decline", S_ASSIGN, "conter", 0, "assign target (E2009 domain)"),
    ("decline", S_PAT_FIELD, "v }", 0, "struct-PATTERN field sub-binder"),
    # bool literals — Literal::Int at parse time, never Lit(Ident); leg 2
    # recognises them independently (Rule 3a), so classify-mode proves all
    # three legs agree on 0 at every value position:
    ("classify", S_BOOL_LET, "true", 0, "bool literal in let RHS -> never E2002"),
    ("classify", S_BOOL_IF, "true", 0, "bool literal as if-cond -> never E2002"),
    ("classify", S_BOOL_FIELD, "true", 0, "bool literal struct-field value -> never"),
    ("classify", S_BOOL_ARG, "true", 0, "bool literal call arg -> never"),
    ("classify", S_BOOL_RET, "true", 0, "bool literal return value -> never"),
    ("classify", S_BOOL_WHILE, "false", 0, "bool literal while-cond -> never"),
    # reserved keywords at STATEMENT occurrence — consumed as keywords, never
    # a Lit(Ident): decline, and live must show no E2002 at the keyword pos:
    ("decline", S_UNDEF, "return", 0, "`return` keyword occurrence"),
    ("decline", S_WHILE_COND, "while", 0, "`while` keyword occurrence"),
    ("decline", S_MATCH_SCRUT, "match", 0, "`match` keyword (lexical gate)"),
    ("decline", S_KW_FOR, "for", 0, "`for` keyword occurrence"),
    ("decline", S_KW_FOR, "in 0", 0, "`in` keyword occurrence"),
    ("decline", S_KW_BREAK, "break", 0, "`break` keyword occurrence"),
    ("decline", S_KW_CONTINUE, "continue", 0, "`continue` keyword occurrence"),
    ("decline", S_AS, "as i64", 0, "`as` keyword occurrence"),
    # keyword-SPELLED ident in EXPRESSION position — the parser DOES produce
    # a Lit(Ident) there and live fires E2002 (probe-verified): the positional
    # gate must fall through and classify, not blanket-decline the spelling:
    ("classify", S_KW_LOOP_EXPR, "loop", 0, "`loop` in expr position -> E2002 fires"),
]

# ── table extraction (drift fails loud) ─────────────────────────────────────
def bare_builtins():
    with open(RESOLVE_RS) as f:
        rs = f.read()
    m = re.search(r"const BARE_BUILTINS: &\[&str\] = &\[(.*?)\];", rs, re.S)
    if not m:
        print("FAIL: BARE_BUILTINS table not found in resolve.rs")
        sys.exit(1)
    names = re.findall(r'"([A-Za-z0-9_]+)"', m.group(1))
    if len(names) < 30:
        print(f"FAIL: extracted only {len(names)} BARE_BUILTINS — drifted")
        sys.exit(1)
    return set(names)


def ident_resolvable_source_guard():
    """Drift tripwire: assert ident_resolvable still has EXACTLY the arms this
    port mirrors (scopes / name_resolvable / bytes / :: / cm export) and does
    NOT grow a builtin/prefix arm without this smoke noticing."""
    with open(RESOLVE_RS) as f:
        rs = f.read()
    m = re.search(
        r"fn ident_resolvable\(&self, name: &str\) -> bool \{(.*?)\n    \}",
        rs,
        re.S,
    )
    if not m:
        print("FAIL: ident_resolvable not found in resolve.rs")
        sys.exit(1)
    body = m.group(1)
    for arm in (
        "self.scopes.contains(name)",
        "self.syms.name_resolvable(name)",
        '"bytes"',
        'contains("::")',
        "cm_symbol_exported_res(name)",
    ):
        if arm not in body:
            print(f"FAIL: ident_resolvable arm {arm!r} missing — rule drifted")
            sys.exit(1)
    for absent in ("BARE_BUILTINS", "__mind_", "tensor.", "gen_deref"):
        if absent in body:
            print(f"FAIL: ident_resolvable grew a {absent!r} arm — rule drifted")
            sys.exit(1)


def bundled_modules():
    with open(STDLIB_RS) as f:
        rs = f.read()
    mods = re.findall(r'\("std\.([a-z0-9_]+)",\s*include_str!', rs)
    if len(mods) < 20:
        print(f"FAIL: extracted only {len(mods)} bundled modules — drifted")
        sys.exit(1)
    return mods


# The parser's closed statement-keyword recogniser (`stmt_keyword`,
# src/parser/mod.rs) — extracted every run so a keyword ADDED to the parser
# without a matching arm in main.mind's tc_ui_stmt_kw_word fails LOUD here.
# The MIND-side gate must cover the extracted set minus the words this
# lexer emits as DISTINCT token kinds (fn/let/use/pub/if — plus `else`,
# kw-kinded but not statement-leading), minus match/region (gated lexically
# as expression-position hijackers), plus the operator words mut/in/as.
PARSER_RS = os.path.join(REPO, "src", "parser", "mod.rs")
DISTINCT_KINDS = {"fn", "let", "use", "pub", "if"}
LEXICAL_GATE = {"match", "region"}
MIND_STMT_KW_GATE = {
    "return", "break", "continue", "while", "for", "loop", "struct",
    "enum", "const", "type", "module", "import", "export", "extern",
    "assert", "print", "invariant", "mut", "in", "as",
}


def stmt_keywords():
    with open(PARSER_RS) as f:
        rs = f.read()
    m = re.search(
        r"fn stmt_keyword\(w: &\[u8\]\) -> Option<StmtKw> \{(.*?)\n\}", rs, re.S
    )
    if not m:
        print("FAIL: stmt_keyword recogniser not found in parser/mod.rs")
        sys.exit(1)
    kws = set(re.findall(r'b"([a-z]+)"', m.group(1)))
    if len(kws) < 20:
        print(f"FAIL: extracted only {len(kws)} stmt keywords — drifted")
        sys.exit(1)
    expected_gate = (kws - DISTINCT_KINDS - LEXICAL_GATE) | {"mut", "in", "as"}
    if expected_gate != MIND_STMT_KW_GATE:
        print("FAIL: parser stmt-keyword table drifted from the MIND gate:")
        print(f"  missing from MIND gate: {sorted(expected_gate - MIND_STMT_KW_GATE)}")
        print(f"  stale in MIND gate:     {sorted(MIND_STMT_KW_GATE - expected_gate)}")
        sys.exit(1)
    return kws


# ── leg 2a: D1/D3 decl-set mirror (identical to the D1/D3/D4 smokes) ───────
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
        if t == "let":
            j = i + 1
            if j + 1 < n and toks[j] == "mut" and is_word(toks[j + 1]):
                j += 1
            if j < n and is_word(toks[j]):
                names.add(toks[j])
                i = j + 1
                continue
            i += 1
            continue
        if t in ("struct", "const", "type") and i + 1 < n and is_word(toks[i + 1]):
            names.add(toks[i + 1])
            i += 2
            continue
        if t == "enum" and i + 2 < n and is_word(toks[i + 1]) and toks[i + 2] == "{":
            ename = toks[i + 1]
            names.add(ename)
            j, d, p, expect = i + 3, 1, 0, True
            while j < n and d > 0:
                c = toks[j]
                if c == "{":
                    d += 1
                elif c == "}":
                    d -= 1
                elif c == "(":
                    p += 1
                elif c == ")":
                    p -= 1
                elif d == 1 and p == 0:
                    if c == ",":
                        expect = True
                    elif expect and is_word(c):
                        names.add(c)
                        names.add(f"{ename}::{c}")
                        expect = False
                j += 1
            i = j
            continue
        if t == "extern":
            if i + 2 < n and toks[i + 1] == "const" and is_word(toks[i + 2]):
                names.add(toks[i + 2])
                i += 3
                continue
            j = i + 1
            while j < n and toks[j] != "{":
                j += 1
            i = j + 1
            continue
        if t == "module" and i + 1 < n and is_word(toks[i + 1]):
            j = i + 2
            while j + 1 < n and toks[j] == "." and is_word(toks[j + 1]):
                j += 2
            if j < n and toks[j] == "{":
                j += 1
            i = j
            continue
        if t == "{":
            i = skip_brace(i + 1)
            continue
        i += 1
    return names


# ── leg 2b: D2 frame-semantics mirror (identical to the D2/D4 smokes) ──────
KW = {"fn", "let", "use", "pub", "if", "else"}
TWO = ("=>", "==", "<=", ">=", "!=", "<<", ">>", "&&", "||", "->")
OPEN_OP = {"+", "-", "*", "/", "%", "<", ">", "<=", ">=", "==", "!=", "=",
           "&", "|", "^", "<<", ">>", "&&", "||", ".", "!", ",", ":", "->"}
BINONLY = {"+", "/", "%", "<", ">", "<=", ">=", "==", "!=", "|", "^", "<<",
           ">>", "&&", "||", ".", ",", ":", "->"}


def tokenize(src):
    toks = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c in " \t\r\n":
            i += 1
            continue
        if src.startswith("//", i):
            j = src.find("\n", i)
            i = n if j < 0 else j
            continue
        if c in "\"'":
            j = i + 1
            while j < n and src[j] != c:
                j += 1
            toks.append(("str", i, j + 1))
            i = j + 1
            continue
        m = re.match(r"[A-Za-z_]\w*", src[i:])
        if m:
            t = m.group(0)
            toks.append((t if t in KW else "ident", i, i + len(t)))
            i += len(t)
            continue
        m = re.match(r"\d+", src[i:])
        if m:
            toks.append(("int", i, i + len(m.group(0))))
            i += len(m.group(0))
            continue
        two = src[i : i + 2]
        if two in TWO:
            toks.append((two, i, i + 2))
            i += 2
            continue
        toks.append((c, i, i + 1))
        i += 1
    return toks


def scope_verdict(src, pos):
    toks = tokenize(src)
    n = len(toks)
    p = None
    for ix, (k, lo, _hi) in enumerate(toks):
        if lo == pos and k == "ident":
            p = ix
            break
    if p is None:
        return -3
    qname = src[toks[p][1] : toks[p][2]]
    binds = []

    def text(ix):
        return src[toks[ix][1] : toks[ix][2]]

    def kind(ix):
        return toks[ix][0]

    def has_nl(a, b):
        return "\n" in src[a:b]

    def skip_brace(i, d):
        while i < n and d > 0:
            if kind(i) == "{":
                d += 1
            elif kind(i) == "}":
                d -= 1
            i += 1
        return i

    def find_kind(i, want):
        while i < n and kind(i) != want:
            i += 1
        return i

    def find_rparen(i, d):
        while i < n:
            if kind(i) == "(":
                d += 1
            elif kind(i) == ")":
                d -= 1
                if d == 0:
                    return i
            i += 1
        return n

    def find_lbrace0(i, pd):
        while i < n:
            k = kind(i)
            if k == "{" and pd == 0:
                return i
            if k == "(":
                pd += 1
            elif k == ")":
                pd -= 1
            i += 1
        return n

    def frag_end(i, start, stop_comma):
        pd = 0
        while i < n:
            k = kind(i)
            if pd == 0:
                if k == ";" or k == "}":
                    return i
                if k == "," and stop_comma:
                    return i
                if (
                    i > start
                    and has_nl(toks[i - 1][2], toks[i][1])
                    and toks[i - 1][0] not in OPEN_OP
                ):
                    return i
            if k in "([":
                pd += 1
            elif k in ")]":
                pd -= 1
            elif k == "{":
                i = skip_brace(i + 1, 1)
                continue
            i += 1
        return n

    def bind(cnt, ix):
        del binds[cnt:]
        binds.append(text(ix))
        return cnt + 1

    def lookup(cnt):
        return 1 if qname in binds[:cnt] else 0

    def bind_range(cnt, a, b):
        for ix in range(a, b):
            if kind(ix) == "ident" and text(ix) != "_":
                cnt = bind(cnt, ix)
        return cnt

    def bind_pat(cnt, a, g):
        while a < g:
            if kind(a) == "ident" and text(a) != "_":
                if a + 1 < g and kind(a + 1) == "{":
                    a += 1
                    continue
                if a + 1 < g and kind(a + 1) == "(":
                    a += 1
                    continue
                if a + 1 < g and kind(a + 1) == ":":
                    if a + 2 < g and kind(a + 2) == ":":
                        a += 4
                        continue
                    a += 2
                    continue
                cnt = bind(cnt, a)
            a += 1
        return cnt

    def find_fat(i, d):
        while i < n:
            k = kind(i)
            if d == 0 and k == "=>":
                return i
            if k in "([{":
                d += 1
            elif k in ")]}":
                d -= 1
            i += 1
        return n

    def walk(limit, i, cnt):
        while i < limit:
            if i == p:
                return -2 if lookup(cnt) else -1
            k = kind(i)
            if k == "}":
                return i + 1
            if k == "{":
                r = walk(limit, i + 1, cnt)
                if r < 0:
                    return r
                i = r
                continue
            if k == "let":
                s = i + 1
                if (
                    s + 1 < n
                    and kind(s) == "ident"
                    and text(s) == "mut"
                    and kind(s + 1) == "ident"
                ):
                    s += 1
                if kind(s) == "(":
                    close = find_rparen(s + 1, 1)
                    a, b, after = s + 1, close, close + 1
                else:
                    a, b, after = s, s + 1, s + 1
                eq = find_kind(after, "=")
                r0 = eq + 1
                end = frag_end(r0, r0, 0)
                rw = walk(end, r0, cnt)
                if rw < 0:
                    return rw
                if end < n and kind(end) in BINONLY:
                    i = end
                    continue
                cnt = bind_range(cnt, a, b)
                i = end + 1 if end < n and kind(end) == ";" else end
                continue
            if k == "ident" and text(i) == "for":
                lb = find_lbrace0(i + 3, 0)
                hw = walk(lb, i + 3, cnt)
                if hw < 0:
                    return hw
                c2 = bind(cnt, i + 1)
                r = walk(limit, lb + 1, c2)
                if r < 0:
                    return r
                i = r
                continue
            if k == "ident" and text(i) == "match":
                lb = find_lbrace0(i + 1, 0)
                sw = walk(lb, i + 1, cnt)
                if sw < 0:
                    return sw
                a = lb + 1
                while a < n and kind(a) != "}":
                    fa = find_fat(a, 0)
                    g = a
                    while g < fa and kind(g) != "if":
                        g += 1
                    c2 = bind_pat(cnt, a, g)
                    if g < fa:
                        gw = walk(fa, g + 1, c2)
                        if gw < 0:
                            return gw
                    b0 = fa + 1
                    if b0 < n and kind(b0) == "{":
                        r = walk(limit, b0 + 1, c2)
                        if r < 0:
                            return r
                        a = r
                    else:
                        end = frag_end(b0, b0, 1)
                        r = walk(end, b0, c2)
                        if r < 0:
                            return r
                        a = end
                    if a < n and kind(a) == ",":
                        a += 1
                i = a + 1
                continue
            i += 1
        return i

    i = 0
    while i < n:
        if kind(i) == "fn":
            lp = find_kind(i + 1, "(")
            rp = find_rparen(lp + 1, 1)
            lb = find_lbrace0(rp + 1, 0)
            bend = skip_brace(lb + 1, 1)
            if lb < p < bend:
                cnt = 0
                a = lp + 1
                d = 0
                while a < rp:
                    k = kind(a)
                    if k == "(":
                        d += 1
                    elif k == ")":
                        d -= 1
                    elif d == 0 and k == "ident" and a + 1 < rp and kind(a + 1) == ":":
                        cnt = bind(cnt, a)
                        a += 2
                        continue
                    a += 1
                r = walk(n, lb + 1, cnt)
                return {-2: 1, -1: 0}.get(r, -3)
            i = bend
            continue
        i += 1
    return -3


# ── leg 2: the RESOLUTION UNION, recomputed independently ──────────────────
# ZERO position/shape guards here — this leg NEVER replicates tc_ui_shape's
# classifier. It presumes the fixture's queried position is a value use (the
# "classify"-mode contract, validated against LIVE) and recomputes only
# resolve.rs's ident_resolvable union: D2 scope frames + D1 decls/prelude +
# D3 std exports + "bytes". If the port's classifier wrongly declines or the
# resolution union drifts, leg 1 vs leg 2 vs leg 3 disagree and the smoke
# goes RED.
def resolution_verdict(src, pos, std_set):
    name_m = re.match(r"[A-Za-z_]\w*", src[pos:])
    if not name_m:
        return -99
    name = name_m.group(0)
    # INDEPENDENT bool-literal knowledge (Rule 3a — leg 2 must not share the
    # port's blind spot): the parser lowers bare `true`/`false` in value
    # position to Literal::Int (src/parser/mod.rs:4108-4112) — never a
    # Lit(Ident) — so E2002 structurally cannot fire on them.
    if name in ("true", "false"):
        return 0
    sv = scope_verdict(src, pos)
    if sv == 1:
        return 0
    if sv != 0:
        return -99  # the D2 mirror could not reach the position — fail loud
    if name in PRELUDE:
        return 0
    if name in collect_decl_names(src):
        return 0
    if name in std_set:
        return 0
    if name == "bytes":
        return 0
    return 1


# ── leg 3: the live mindc oracle (position-matched E2002 presence) ─────────
DIAG_RE = re.compile(
    r":(\d+):(\d+): error: .*\[type_check::(E2002|E2003|E2009|E2012)\]"
)


def live_verdict(mindc, src, pos, workdir, idx):
    line = src.count("\n", 0, pos) + 1
    col = pos - (src.rfind("\n", 0, pos) + 1) + 1
    path = os.path.join(workdir, f"case_{idx}.mind")
    with open(path, "w") as f:
        f.write(src)
    r = subprocess.run([mindc, "check", path], capture_output=True, text=True)
    for m in DIAG_RE.finditer(r.stdout + r.stderr):
        if int(m.group(1)) == line and int(m.group(2)) == col:
            return 1 if m.group(3) == "E2002" else 0
    return 0


def live_code_at(mindc, src, pos, workdir, idx):
    line = src.count("\n", 0, pos) + 1
    col = pos - (src.rfind("\n", 0, pos) + 1) + 1
    path = os.path.join(workdir, f"case_{idx}.mind")
    with open(path, "w") as f:
        f.write(src)
    r = subprocess.run([mindc, "check", path], capture_output=True, text=True)
    for m in DIAG_RE.finditer(r.stdout + r.stderr):
        if int(m.group(1)) == line and int(m.group(2)) == col:
            return m.group(3)
    return None


def pos_of(src, needle, occ):
    hits = [m.start() for m in re.finditer(re.escape(needle), src)]
    return hits[occ]


def build_so():
    so = os.environ.get("MINDC_SO")
    if so:
        return so, False
    mindc = os.environ.get("MINDC_BIN", "mindc")
    out = tempfile.NamedTemporaryFile(suffix=".so", delete=False).name
    cmd = [mindc, MAIN_MIND, "--emit-shared", out]
    print("BUILD:", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("BUILD FAILED rc=", r.returncode)
        print(r.stdout[-4000:])
        print(r.stderr[-4000:])
        sys.exit(1)
    return out, True


def main():
    so, built = build_so()
    st = os.stat(so)
    print(f"SO: {so} ({st.st_size} bytes)")
    if st.st_size < 4096:
        print("FAIL: .so too small (stub?)")
        sys.exit(1)
    lib = ctypes.CDLL(so)
    fn = lib.selftest_tc_unknown_ident
    fn.argtypes = [ctypes.c_int64] * 5
    fn.restype = ctypes.c_int64

    mindc = os.environ.get("MINDC_BIN", "mindc")

    ident_resolvable_source_guard()
    bare = bare_builtins()
    kws = stmt_keywords()
    mods = bundled_modules()
    concat = b""
    std_set = set()
    for m in mods:
        path = os.path.join(STD_DIR, m + ".mind")
        with open(path, "rb") as f:
            s = f.read()
        if not s.endswith(b"\n"):
            s += b"\n"
        concat += s
        std_set |= collect_decl_names(s.decode())
    print(f"tables: ident_resolvable arms verified, {len(bare)} bare "
          f"builtins, {len(kws)} parser stmt-keywords (gate in sync), "
          f"{len(mods)} std modules ({len(concat)} bytes, "
          f"{len(std_set)} exports)")

    sb = ctypes.create_string_buffer(concat, len(concat))
    sp = ctypes.cast(sb, ctypes.c_void_p).value

    def mind_verdict(src, pos):
        b = ctypes.create_string_buffer(src.encode(), len(src.encode()))
        return fn(ctypes.cast(b, ctypes.c_void_p).value,
                  len(src.encode()), pos, sp, len(concat))

    total = fails = positives = negatives = live_checked = 0
    with tempfile.TemporaryDirectory() as workdir:
        # Live sentinels: both directions must reach resolve with position
        # matching intact — an E2002 hit, and a callee that reports E2003
        # (NOT E2002) at the same-column shape.
        if live_verdict(mindc, S_UNDEF, pos_of(S_UNDEF, "nope", 0), workdir,
                        "s_in") != 1:
            print("FAIL: live sentinel — E2002 not reported at the "
                  "undefined value use")
            sys.exit(1)
        if live_code_at(mindc, S_CALLEE, pos_of(S_CALLEE, "foo(1", 0),
                        workdir, "s_out") != "E2003":
            print("FAIL: live sentinel — E2003 not reported at the "
                  "undefined-callee call site")
            sys.exit(1)
        print("live sentinels: E2002 value-use + E2003 callee (2/2)")

        for idx, (mode, src, needle, occ, note) in enumerate(CASES):
            pos = pos_of(src, needle, occ)
            got = mind_verdict(src, pos)
            live = live_verdict(mindc, src, pos, workdir, idx)
            live_checked += 1
            if mode == "classify":
                exp = resolution_verdict(src, pos, std_set)
                ok = got == exp == live
                exp_s = str(exp)
            elif mode == "never":
                exp = 0
                ok = got == 0 and live == 0
                exp_s = "n0"
            else:  # decline — allowed ONLY where live does not fire
                exp = -3
                ok = got == -3 and live == 0
                exp_s = "d-3"
            total += 1
            if mode == "classify" and exp == 1:
                positives += 1
            elif mode == "classify" and exp == 0:
                negatives += 1
            if not ok:
                fails += 1
            mark = "ok " if ok else "DIFF"
            print(f"  {mark} got={got} rule={exp_s} live={live} pos={pos} "
                  f"[{mode}] {note}")

        # FULL BARE_BUILTINS sweep: every extracted builtin used BARE in
        # value position. Grounded expectation: BARE_BUILTINS is a CALL-only
        # exclusion, so each fires E2002 UNLESS it happens to be a std-export
        # or decl name — the recomputed resolution union (leg 2) decides per
        # name and all three legs must agree. This is the tripwire that
        # proves the ident/call exclusion-set asymmetry stays real.
        sweep_fails = 0
        for bidx, name in enumerate(sorted(bare)):
            src = (f"fn main() -> i64 {{\n    let z = {name}\n"
                   f"    return 0\n}}\n")
            pos = pos_of(src, f"{name}\n", 0)
            exp = resolution_verdict(src, pos, std_set)
            got = mind_verdict(src, pos)
            live = live_verdict(mindc, src, pos, workdir, f"bb_{bidx}")
            live_checked += 1
            total += 1
            if exp == 1:
                positives += 1
            elif exp == 0:
                negatives += 1
            if not (got == exp == live):
                sweep_fails += 1
                fails += 1
                print(f"  DIFF got={got} rule={exp} live={live} "
                      f"bare builtin value-use {name!r}")
        print(f"  ok  bare-builtin value-use sweep: {len(bare)} names, "
              f"{sweep_fails} diffs")

    print(f"unknown_ident: cases={total} positives={positives} "
          f"negatives={negatives} live_checked={live_checked} fails={fails}")
    if positives < 12 or negatives < 10 or live_checked < 40:
        print("FAIL: vacuous corpus")
        sys.exit(1)
    if fails:
        print("FAIL: pure-MIND E2002 rule diverges from the Rust rule")
        sys.exit(1)
    print("ALL PASS")
    if built:
        try:
            os.unlink(so)
        except OSError:
            pass


if __name__ == "__main__":
    main()
