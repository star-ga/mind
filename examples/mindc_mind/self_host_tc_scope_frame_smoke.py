#!/usr/bin/env python3
"""CPU-as-oracle smoke for the pure-MIND B1/D2 NESTED SCOPE-FRAME WALK.

Ports the resolve.rs `Scopes` frame stack + the `Resolver::walk`/`walk_block`
binding registration (the positive-trigger half of E2012, reusable for
E2002/E2009): at a call-site ident position inside a fn body, is that name a
LIVE LOCAL binding under nested-frame scoping with sequential visibility?
Semantics mirrored: frame 0 = fn params; `{` pushes / `}` pops a frame; `Let`
and `LetTuple` bind AFTER their RHS walks (a name is NOT in scope in its own
initializer, a use BEFORE its `let` never sees it); `For` binds its loop var
in a fresh frame AFTER the header range walks in the outer scope; `Match`
arms each push a frame, bind the pattern (bare idents and payload/tuple
sub-patterns; `_`, literals and variant heads bind nothing), and the guard
walks WITH the binds live; a binding in an exited sibling block is dead.

The pure-MIND twin `selftest_tc_scope_frame(src, src_len, pos)` takes the
FULL fixture source plus the BYTE OFFSET of the queried call-site ident and
returns 1 (live local binding) / 0 (not) — the frame walk itself, not just
the final bit, is MIND-computed via the real self-host lexer.

Three-way agreement, machine-checked (no hand-authored verdict table):
  1. MIND core over (source, pos).
  2. The exact Rust frame semantics recomputed in Python: a token-level
     mirror of Scopes push/pop/bind + the walk order above.
  3. The LIVE `mindc check` oracle where applicable: for a callee that is
     neither a module decl nor a std export, resolve reports
       E2012 at exactly the query line:col  <=>  scopes.contains == true
       E2003/E2002 at the query line:col    <=>  scopes.contains == false
     (is_fn_value_call fires iff the callee resolves ONLY as a local
     binding). Fail-closed sentinels prove both directions of the live leg
     actually reach resolve with position matching.

Env: MINDC_SO (prebuilt .so, skips the build) or MINDC_BIN (default mindc).
Template: self_host_tc_decl_names_smoke.py.
"""
import ctypes
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN_MIND = os.path.join(HERE, "main.mind")

# ── fixtures ────────────────────────────────────────────────────────────────
S_POS = """\
fn add1(x: i64) -> i64 { return x + 1 }
fn main() -> i64 {
    let f = add1
    return f(41)
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

S_PARAM = """\
fn use_it(g: i64) -> i64 {
    return g(1)
}
fn main() -> i64 {
    return use_it(3)
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

S_OWN_INIT = """\
fn main() -> i64 {
    let f = f(41)
    return f
}
"""

S_REBIND = """\
fn main() -> i64 {
    let f = 1
    let f = f(2)
    return f
}
"""

S_SHADOW = """\
fn main() -> i64 {
    let g = 1
    if 1 > 0 {
        let g = 2
        let z = g(3)
    }
    return g(4)
}
"""

S_FOR_BODY = """\
fn main() -> i64 {
    for g in 0..3 {
        let z = g(1)
    }
    return 0
}
"""

S_FOR_AFTER = """\
fn main() -> i64 {
    for g in 0..3 {
        let z = 1
    }
    return g(1)
}
"""

S_FOR_HDR_OUTER = """\
fn main() -> i64 {
    let h = 1
    for i in 0..h(3) {
        let z = 1
    }
    return 0
}
"""

S_FOR_HDR_SELF = """\
fn main() -> i64 {
    for i in 0..i(3) {
        let z = 1
    }
    return 0
}
"""

S_MATCH_BIND = """\
fn main() -> i64 {
    let v = 1
    let w = match v {
        g => g(1),
    }
    return w
}
"""

S_MATCH_SIBLING_ARM = """\
fn main() -> i64 {
    let v = 1
    let w = match v {
        g => 1,
        _ => g(2),
    }
    return w
}
"""

S_MATCH_GUARD = """\
fn main() -> i64 {
    let v = 1
    let w = match v {
        g if g(0) > 0 => 1,
        _ => 0,
    }
    return w
}
"""

S_TUPLE_LET = """\
fn main() -> i64 {
    let (f, h) = (1, 2)
    return f(3)
}
"""

S_TUPLE_OWN = """\
fn main() -> i64 {
    let (f, h) = (f(1), 2)
    return f
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

S_WHILE = """\
fn add1(x: i64) -> i64 { return x + 1 }
fn main() -> i64 {
    let mut c = 0
    let f = add1
    while c < 1 {
        c = c + 1
        let z = f(1)
    }
    return 0
}
"""

S_ELSE = """\
fn add1(x: i64) -> i64 { return x + 1 }
fn main() -> i64 {
    if 1 > 0 {
        let f = add1
        let a = 1
    } else {
        return f(41)
    }
    return 0
}
"""

S_SEMI = """\
fn main() -> i64 {
    let a = 1;
    let f = a;
    return f(2);
}
"""

S_STRUCT_PAT_HEAD = """\
struct Point { x: i64 }
fn main() -> i64 {
    let p = Point { x: 1 }
    let w = match p {
        Point { x: px } => Point(9),
        _ => 0,
    }
    return w
}
"""

S_STRUCT_PAT_BIND = """\
struct Point { x: i64 }
fn main() -> i64 {
    let p = Point { x: 1 }
    let w = match p {
        Point { x: px } => px(9),
        _ => 0,
    }
    return w
}
"""

S_NL_TRAIL_OP = """\
fn main() -> i64 {
    let f = 1 +
        f(2)
    return f
}
"""

S_NL_LEAD_OP = """\
fn main() -> i64 {
    let f = 1
        + f(2)
    return f
}
"""

# (src, callee, occurrence-index of `callee(` incl. decl sites, live-eligible)
# Expected verdicts are NEVER hand-authored: they come from the recomputed
# Rust frame semantics (leg 2) and must equal legs 1 and 3.
CASES = [
    (S_POS, "f", 0, True, "let then call"),
    (S_BEFORE, "f", 0, True, "use before let"),
    (S_SIBLING, "f", 0, True, "exited sibling block"),
    (S_PARAM, "g", 0, True, "param frame 0"),
    (S_BLOCK_LET, "f", 0, True, "let-in-block same block"),
    (S_NESTED, "f", 0, True, "outer let, doubly-nested use"),
    (S_OWN_INIT, "f", 0, True, "name in its own initializer"),
    (S_REBIND, "f", 0, True, "rebind: prior binding live in new RHS"),
    (S_SHADOW, "g", 0, True, "inner shadow, use in inner block"),
    (S_SHADOW, "g", 1, True, "outer binding live after shadow block"),
    (S_FOR_BODY, "g", 0, True, "for-var inside body"),
    (S_FOR_AFTER, "g", 0, True, "for-var after loop"),
    (S_FOR_HDR_OUTER, "h", 0, True, "outer let in for header"),
    (S_FOR_HDR_SELF, "i", 0, True, "for-var in its own header"),
    (S_MATCH_BIND, "g", 0, True, "match pattern bind in arm body"),
    (S_MATCH_SIBLING_ARM, "g", 0, True, "pattern bind dead in later arm"),
    (S_MATCH_GUARD, "g", 0, True, "pattern bind live in guard"),
    (S_TUPLE_LET, "f", 0, True, "tuple-let name"),
    (S_TUPLE_OWN, "f", 0, True, "tuple-let name in own RHS"),
    (S_DECL_FN, "add1", 1, False, "decl fn is NOT a local binding"),
    (S_SHADOW_DECL, "add1", 1, False, "local shadowing a decl fn (scopes half only)"),
    (S_WHILE, "f", 0, True, "outer let inside while body"),
    (S_ELSE, "f", 0, True, "then-branch binding dead in else"),
    (S_SEMI, "f", 0, True, "semicolon-terminated statements"),
    (S_STRUCT_PAT_HEAD, "Point", 0, False, "struct-pattern HEAD binds nothing"),
    (S_STRUCT_PAT_BIND, "px", 0, True, "struct-pattern field sub-pattern binds"),
    (S_NL_TRAIL_OP, "f", 0, True, "trailing-op newline-continued own initializer"),
    (S_NL_LEAD_OP, "f", 0, True, "leading-op newline-continued own initializer"),
]

# ── leg 2: the Rust frame semantics recomputed over tokens ─────────────────
KW = {"fn", "let", "use", "pub", "if", "else"}
TWO = ("=>", "==", "<=", ">=", "!=", "<<", ">>", "&&", "||", "->")
# Operators that cannot END an expression: a depth-0 line break AFTER one of
# these continues the fragment (the Pratt greedy extent).
OPEN_OP = {"+", "-", "*", "/", "%", "<", ">", "<=", ">=", "==", "!=", "=",
           "&", "|", "^", "<<", ">>", "&&", "||", ".", "!", ",", ":", "->"}
# Binary-ONLY operators that cannot START a statement: a fragment ending at a
# line break directly BEFORE one of these split a newline-continued
# initializer -> the let binding is withheld (fail-close, never false
# in-scope). Excludes prefix-capable `-` `*` `&` `!`.
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
    """Recomputed Rust rule: Scopes push/pop/bind over the token stream."""
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
    binds = []  # the frame stack, flattened: (cnt-prefix == live bindings)

    def text(ix):
        return src[toks[ix][1] : toks[ix][2]]

    def kind(ix):
        return toks[ix][0]

    def has_nl(a, b):
        return "\n" in src[a:b]

    def skip_brace(i, d):  # i just after '{'
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
                    a += 1  # struct-pattern HEAD (type name) binds nothing
                    continue
                if a + 1 < g and kind(a + 1) == "(":
                    a += 1  # variant head binds nothing
                    continue
                if a + 1 < g and kind(a + 1) == ":":
                    if a + 2 < g and kind(a + 2) == ":":
                        a += 4  # Enum::Variant
                        continue
                    a += 2  # struct-pattern field label
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
        """-2 in scope / -1 not / else resumed index."""
        while i < limit:
            if i == p:
                return -2 if lookup(cnt) else -1
            k = kind(i)
            if k == "}":
                return i + 1
            if k == "{":
                r = walk(limit, i + 1, cnt)  # fresh frame: resume with cnt
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
                rw = walk(end, r0, cnt)  # RHS: names NOT yet live
                if rw < 0:
                    return rw
                if end < n and kind(end) in BINONLY:
                    i = end  # fail-close: split continuation, withhold binding
                    continue
                cnt = bind_range(cnt, a, b)
                i = end + 1 if end < n and kind(end) == ";" else end
                continue
            if k == "ident" and text(i) == "for":
                lb = find_lbrace0(i + 3, 0)
                hw = walk(lb, i + 3, cnt)  # header in OUTER scope
                if hw < 0:
                    return hw
                c2 = bind(cnt, i + 1)  # fresh frame: loop var
                r = walk(limit, lb + 1, c2)
                if r < 0:
                    return r
                i = r  # pop: resume with cnt
                continue
            if k == "ident" and text(i) == "match":
                lb = find_lbrace0(i + 1, 0)
                sw = walk(lb, i + 1, cnt)  # scrutinee in outer scope
                if sw < 0:
                    return sw
                a = lb + 1
                while a < n and kind(a) != "}":
                    fa = find_fat(a, 0)
                    g = a
                    while g < fa and kind(g) != "if":
                        g += 1
                    c2 = bind_pat(cnt, a, g)  # fresh frame per arm
                    if g < fa:
                        gw = walk(fa, g + 1, c2)  # guard WITH binds live
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
                i = a + 1  # past the match '}'
                continue
            i += 1
        return i

    # top level: locate the fn whose body holds p; params are frame 0
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


# ── leg 3: the live mindc oracle (position-matched E2012/E2003/E2002) ──────
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
    return None


def pos_of(src, name, occ):
    hits = [m.start() for m in re.finditer(rf"\b{re.escape(name)}\s*\(", src)]
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
    fn = lib.selftest_tc_scope_frame
    fn.argtypes = [ctypes.c_int64] * 3
    fn.restype = ctypes.c_int64

    mindc = os.environ.get("MINDC_BIN", "mindc")

    total = fails = positives = negatives = live_checked = 0
    with tempfile.TemporaryDirectory() as workdir:
        # Fail-closed sentinels: both live directions must reach resolve with
        # position matching intact.
        if live_verdict(mindc, S_POS, pos_of(S_POS, "f", 0), workdir, "s_in") != 1:
            print("FAIL: live sentinel — E2012 not reported at the in-scope "
                  "call site (live leg not reaching resolve)")
            sys.exit(1)
        if live_verdict(mindc, S_BEFORE, pos_of(S_BEFORE, "f", 0), workdir,
                        "s_out") != 0:
            print("FAIL: live sentinel — E2003 not reported at the "
                  "use-before-let call site")
            sys.exit(1)
        print("live sentinels: E2012 in-scope + E2003 use-before-let (2/2)")

        for idx, (src, name, occ, live_ok, note) in enumerate(CASES):
            pos = pos_of(src, name, occ)
            exp = scope_verdict(src, pos)
            sb = ctypes.create_string_buffer(src.encode(), len(src.encode()))
            got = fn(
                ctypes.cast(sb, ctypes.c_void_p).value, len(src.encode()), pos
            )
            if live_ok:
                live = live_verdict(mindc, src, pos, workdir, idx)
                live_checked += 1
                ok = got == exp == live
                live_s = "?" if live is None else str(live)
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

    print(
        f"scope_frame: cases={total} positives={positives} "
        f"negatives={negatives} live_checked={live_checked} fails={fails}"
    )
    if positives < 8 or negatives < 8 or live_checked < 10:
        print("FAIL: vacuous corpus")
        sys.exit(1)
    if fails:
        print("FAIL: pure-MIND scope-frame walk diverges from the Rust rule")
        sys.exit(1)
    print("ALL PASS")
    if built:
        try:
            os.unlink(so)
        except OSError:
            pass


if __name__ == "__main__":
    main()
