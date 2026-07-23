#!/usr/bin/env python3
"""CPU-as-oracle smoke for the pure-MIND B1/D4 CAPSTONE — the FULL E2012 rule.

Ports resolve.rs `is_fn_value_call` (FN_VALUE_CALL_CODE, E2012): a CALL whose
callee resolves ONLY as a local lexical binding (a `let`-bound value / a
param — a function value) and to NOTHING emittable. The exact Rust condition
(resolve.rs:655-678), assembled from the landed scope-frame primitives:

    scopes.contains(name)                                 [D2 positive trigger]
    && !( syms.name_resolvable(name)   — module decls + prelude [D1]
                                         OR bundled-std exports  [D3]
       || name.contains("::")
       || name == "bytes"
       || name.startswith("__mind_")
       || name.startswith("tensor.")   — covered by dot-adjacency (an ident
                                         token can never contain a dot)
       || name == "gen_deref"
       || name in BARE_BUILTINS        — the 34-name empirically-swept set
       || std_surface_intrinsic_arity(name).is_some()   — the E2024 table
       || cm_lookup_fn(name).is_some() — vacuous in the single-file model )

The pure-MIND twin `selftest_tc_fn_value_call(src, src_len, pos, std_src,
std_len)` takes the fixture source, the byte offset of the queried call-site
callee ident, and the CONCATENATED bundled-std source (the D3 caller-supplied
model). Returns 1 (E2012 fires) / 0 (it does not) / -3 (fail-closed sentinel:
not an ident / not a bare-call callee / unclassifiable position).

Three-way agreement, machine-checked (no hand-authored verdict table):
  1. MIND core over (source, pos, std concat).
  2. The exact Rust rule recomputed in Python: the D2 frame semantics + the
     D1/D3 decl-set mirrors + every literal exclusion arm, with BARE_BUILTINS
     re-extracted from resolve.rs and STD_SURFACE_INTRINSICS re-extracted
     from type_checker/mod.rs on every run — table drift fails LOUD here.
  3. The LIVE `mindc check` oracle: E2012 present at exactly the query
     line:col <=> verdict 1. Fail-closed sentinels prove both live
     directions (an E2012 hit and a position-matched E2003 miss) actually
     reach resolve.

Env: MINDC_SO (prebuilt .so, skips the build) or MINDC_BIN (default mindc).
Template: self_host_tc_scope_frame_smoke.py + self_host_tc_std_export_smoke.py.
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
TC_MOD_RS = os.path.join(REPO, "src", "type_checker", "mod.rs")
STDLIB_RS = os.path.join(REPO, "src", "project", "stdlib.rs")
STD_DIR = os.path.join(REPO, "std")

PRELUDE = {"Result", "Option", "Ok", "Err", "Some", "None"}

# ── fixtures ────────────────────────────────────────────────────────────────
S_POS = """\
fn add1(x: i64) -> i64 { return x + 1 }
fn main() -> i64 {
    let f = add1
    return f(41)
}
"""

S_PARAM = """\
fn use_it(g: i64) -> i64 {
    return g(1)
}
fn main() -> i64 {
    return use_it(3)
}
"""

S_NESTED = """\
fn add1(x: i64) -> i64 { return x + 1 }
fn main() -> i64 {
    let f = add1
    if 1 > 0 {
        if 2 > 1 {
            return f(41)
        }
    }
    return 0
}
"""

S_BLOCK_LET = """\
fn add1(x: i64) -> i64 { return x + 1 }
fn main() -> i64 {
    if 1 > 0 {
        let f = add1
        return f(41)
    }
    return 0
}
"""

S_REBIND = """\
fn main() -> i64 {
    let f = 1
    let f = f(2)
    return f
}
"""

S_TUPLE_LET = """\
fn main() -> i64 {
    let (f, h) = (1, 2)
    return f(3)
}
"""

S_DECL_FN = """\
fn add1(x: i64) -> i64 { return x + 1 }
fn main() -> i64 {
    return add1(5)
}
"""

S_SHADOW_DECL = """\
fn add1(x: i64) -> i64 { return x + 1 }
fn main() -> i64 {
    let add1 = 1
    return add1(5)
}
"""

S_STD_PLAIN = """\
fn main() -> i64 {
    let v = vec_new()
    return 0
}
"""

S_STD_SHADOW = """\
fn main() -> i64 {
    let vec_new = 1
    let v = vec_new()
    return 0
}
"""

S_UNDEF = """\
fn main() -> i64 {
    return foo(1)
}
"""

S_BEFORE = """\
fn add1(x: i64) -> i64 { return x + 1 }
fn main() -> i64 {
    let y = f(41)
    let f = add1
    return y
}
"""

S_SIBLING = """\
fn add1(x: i64) -> i64 { return x + 1 }
fn main() -> i64 {
    if 1 > 0 {
        let f = add1
    }
    return f(41)
}
"""

S_MIND_SHADOW = """\
fn main() -> i64 {
    let __mind_alloc = 1
    let p = __mind_alloc(8)
    return 0
}
"""

S_MIND_UNREG = """\
fn main() -> i64 {
    let __mind_zzz = 1
    let p = __mind_zzz(1)
    return 0
}
"""

S_BYTE_SHADOW = """\
fn main() -> i64 {
    let byte = 1
    let z = byte(2)
    return 0
}
"""

S_BYTES_SHADOW = """\
fn main() -> i64 {
    let bytes = 1
    let z = bytes(2)
    return 0
}
"""

S_GENDEREF_SHADOW = """\
fn main() -> i64 {
    let gen_deref = 1
    let z = gen_deref(3)
    return 0
}
"""

S_PRELUDE_SHADOW = """\
fn main() -> i64 {
    let Some = 1
    let z = Some(2)
    return 0
}
"""

S_QUALIFIED = """\
enum Col { Red(i64), Green }
fn main() -> i64 {
    let Col = 1
    let z = Col::Red(1)
    return 0
}
"""

S_DOT_HEAD = """\
fn main() -> i64 {
    let s = 1
    let z = s.foo(2)
    return 0
}
"""

S_NOT_A_CALL = """\
fn main() -> i64 {
    let x = 1
    return x + 1
}
"""

# (src, needle, occurrence, live_ok, note) — pos is the start byte of the
# `occurrence`-th match of `needle`, which begins at the queried ident.
# Expected verdicts are NEVER hand-authored: they come from the recomputed
# Rust rule (leg 2) and must equal legs 1 and 3.
CASES = [
    (S_POS, "f(41", 0, True, "let-bound value called -> E2012"),
    (S_PARAM, "g(1", 0, True, "param value called -> E2012"),
    (S_NESTED, "f(41", 0, True, "outer let, doubly-nested call -> E2012"),
    (S_BLOCK_LET, "f(41", 0, True, "let-in-block call -> E2012"),
    (S_REBIND, "f(2", 0, True, "prior binding live in rebind RHS -> E2012"),
    (S_TUPLE_LET, "f(3", 0, True, "tuple-let value called -> E2012"),
    (S_DECL_FN, "add1(5", 0, True, "module fn called -> no (not a local)"),
    (S_SHADOW_DECL, "add1(5", 0, True, "local shadows decl fn -> no (D1 excl)"),
    (S_STD_PLAIN, "vec_new()", 0, True, "std export called -> no (not a local)"),
    (S_STD_SHADOW, "vec_new()", 0, True, "local shadows std fn -> no (D3 excl)"),
    (S_UNDEF, "foo(1", 0, True, "undefined callee -> no (E2003, not E2012)"),
    (S_BEFORE, "f(41", 0, True, "use before let -> no (not yet bound)"),
    (S_SIBLING, "f(41", 0, True, "binding dead in sibling block -> no"),
    (S_MIND_SHADOW, "__mind_alloc(8", 0, True, "shadowed intrinsic -> no (__mind_ prefix)"),
    (S_MIND_UNREG, "__mind_zzz(1", 0, True, "shadowed unregistered __mind_ -> no (prefix)"),
    (S_BYTE_SHADOW, "byte(2", 0, True, "shadowed 'byte' -> no (E2024 table entry)"),
    (S_BYTES_SHADOW, "bytes(2", 0, True, "shadowed 'bytes' -> no (value type)"),
    (S_GENDEREF_SHADOW, "gen_deref(3", 0, True, "shadowed gen_deref -> no (builtin)"),
    (S_PRELUDE_SHADOW, "Some(2", 0, True, "shadowed prelude Some -> no (name_resolvable)"),
    (S_QUALIFIED, "Col::Red(1", 0, True, "qualified head -> no (:: exclusion)"),
    (S_DOT_HEAD, "s.foo(2", 0, True, "dot head -> no (method/path head)"),
    (S_NOT_A_CALL, "x + 1", 0, False, "non-call ident -> -3 sentinel"),
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


def intrinsics_table():
    with open(TC_MOD_RS) as f:
        rs = f.read()
    m = re.search(
        r"STD_SURFACE_INTRINSICS[^=]*=\s*&?\[(.*?)\];", rs, re.S
    )
    if not m:
        print("FAIL: STD_SURFACE_INTRINSICS table not found in mod.rs")
        sys.exit(1)
    entries = re.findall(r'\("([A-Za-z0-9_.]+)",\s*(\d+)\)', m.group(1))
    if len(entries) < 30:
        print(f"FAIL: extracted only {len(entries)} intrinsics — drifted")
        sys.exit(1)
    return {n for n, _a in entries}


def bundled_modules():
    with open(STDLIB_RS) as f:
        rs = f.read()
    mods = re.findall(r'\("std\.([a-z0-9_]+)",\s*include_str!', rs)
    if len(mods) < 20:
        print(f"FAIL: extracted only {len(mods)} bundled modules — drifted")
        sys.exit(1)
    return mods


# ── leg 2a: D1/D3 decl-set mirror (identical to the D1/D3 smokes) ──────────
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


# ── leg 2b: D2 frame-semantics mirror (identical to the D2 smoke) ──────────
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


# ── leg 2: the FULL recomputed rule ────────────────────────────────────────
def full_verdict(src, pos, std_set, bare, intrin):
    toks = tokenize(src)
    n = len(toks)
    p = None
    for ix, (k, lo, _hi) in enumerate(toks):
        if lo == pos and k == "ident":
            p = ix
            break
    if p is None or p + 1 >= n:
        return -3
    name = src[toks[p][1] : toks[p][2]]
    nk = toks[p + 1][0]
    if nk == ":":
        if p + 2 < n and toks[p + 2][0] == ":":
            return 0  # qualified `Head::...` — resolve.rs "::" exclusion
        return -3
    if nk == ".":
        return 0  # method/path head — E2012 structurally never fires
    if nk != "(":
        return -3
    sv = scope_verdict(src, pos)
    if sv == 0:
        return 0
    if sv != 1:
        return -3
    if name in PRELUDE:
        return 0
    if name in collect_decl_names(src):
        return 0
    if name in std_set:
        return 0
    if name in ("bytes", "gen_deref"):
        return 0
    if name.startswith("__mind_"):
        return 0
    if name in bare:
        return 0
    if name in intrin:
        return 0
    return 1


# ── leg 3: the live mindc oracle (position-matched E2012 presence) ─────────
DIAG_RE = re.compile(r":(\d+):(\d+): error: .*\[type_check::(E2012|E2003|E2002)\]")


def live_verdict(mindc, src, pos, workdir, idx):
    line = src.count("\n", 0, pos) + 1
    col = pos - (src.rfind("\n", 0, pos) + 1) + 1
    path = os.path.join(workdir, f"case_{idx}.mind")
    with open(path, "w") as f:
        f.write(src)
    r = subprocess.run([mindc, "check", path], capture_output=True, text=True)
    for m in DIAG_RE.finditer(r.stdout + r.stderr):
        if int(m.group(1)) == line and int(m.group(2)) == col:
            return 1 if m.group(3) == "E2012" else 0
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
    fn = lib.selftest_tc_fn_value_call
    fn.argtypes = [ctypes.c_int64] * 5
    fn.restype = ctypes.c_int64

    mindc = os.environ.get("MINDC_BIN", "mindc")

    bare = bare_builtins()
    intrin = intrinsics_table()
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
    print(f"tables: {len(bare)} bare builtins, {len(intrin)} intrinsics, "
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
        # Fail-closed sentinels: both live directions must reach resolve
        # with position matching intact.
        if live_verdict(mindc, S_POS, pos_of(S_POS, "f(41", 0), workdir,
                        "s_in") != 1:
            print("FAIL: live sentinel — E2012 not reported at the "
                  "function-value call site")
            sys.exit(1)
        if live_code_at(mindc, S_UNDEF, pos_of(S_UNDEF, "foo(1", 0), workdir,
                        "s_out") != "E2003":
            print("FAIL: live sentinel — E2003 not reported at the "
                  "undefined-callee call site")
            sys.exit(1)
        print("live sentinels: E2012 fn-value + E2003 undefined (2/2)")

        for idx, (src, needle, occ, live_ok, note) in enumerate(CASES):
            pos = pos_of(src, needle, occ)
            exp = full_verdict(src, pos, std_set, bare, intrin)
            got = mind_verdict(src, pos)
            if live_ok:
                live = live_verdict(mindc, src, pos, workdir, idx)
                live_checked += 1
                ok = got == exp == live
                live_s = str(live)
            else:
                live = None
                ok = got == exp
                live_s = "-"
            total += 1
            if exp == 1:
                positives += 1
            elif exp == 0:
                negatives += 1
            if not ok:
                fails += 1
            mark = "ok " if ok else "DIFF"
            print(f"  {mark} got={got} rule={exp} live={live_s} pos={pos} {note}")

        # FULL BARE_BUILTINS sweep: every extracted builtin, shadowed by a
        # local let and then called — the exclusion must hold for the whole
        # table (this is what verifies the 34 baked packed-word constants).
        sweep_fails = 0
        for bidx, name in enumerate(sorted(bare)):
            src = (f"fn main() -> i64 {{\n    let {name} = 1\n"
                   f"    let z = {name}(1)\n    return 0\n}}\n")
            pos = pos_of(src, f"{name}(1", 0)
            exp = full_verdict(src, pos, std_set, bare, intrin)
            got = mind_verdict(src, pos)
            live = live_verdict(mindc, src, pos, workdir, f"bb_{bidx}")
            live_checked += 1
            total += 1
            negatives += 1
            if not (got == exp == live == 0):
                sweep_fails += 1
                fails += 1
                print(f"  DIFF got={got} rule={exp} live={live} "
                      f"shadowed builtin {name!r}")
        print(f"  ok  bare-builtin shadow sweep: {len(bare)} names, "
              f"{sweep_fails} diffs")

    print(f"fn_value_call: cases={total} positives={positives} "
          f"negatives={negatives} live_checked={live_checked} fails={fails}")
    if positives < 5 or negatives < 12 or live_checked < 20:
        print("FAIL: vacuous corpus")
        sys.exit(1)
    if fails:
        print("FAIL: pure-MIND E2012 rule diverges from the Rust rule")
        sys.exit(1)
    print("ALL PASS")
    if built:
        try:
            os.unlink(so)
        except OSError:
            pass


if __name__ == "__main__":
    main()
