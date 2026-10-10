"""Control-flow corpus for self_host_native_cf_smoke.py (helper module, no verdict).

Generates deterministic random MIND programs whose functions nest if / else /
else-if / while / break / continue / return and value-if to several levels, with
`&&` / `||` / `!` in conditions and value-ifs inside conditions and expressions,
in functions with and without parameters. Every program is a well-typed i64
program that terminates by construction:

  * every `while` is guarded by its own counter `cN < K` (K <= 5), and the
    counter's increment is the FIRST statement of the body, so neither
    `continue` nor `break` can skip it;
  * a helper may only call helpers with a LOWER index (no recursion);
  * divisors are literal 1..7 and shift counts literal 0..5, so no division
    trap or undefined shift exists in the corpus.

Side effects are observable: `out(v)` writes one letter per call to stdout, and
`main`'s value is the exit status. The module also carries an independent
interpreter (`run_program`) of the same AST with MIND's i64 semantics
(wrapping + - * << , truncating / %, arithmetic >>), so the expected exit code
and stdout of every program come from a third implementation that shares no
code with either compiler.
"""

import random

MASK = (1 << 64) - 1

OUT_FN = (
    "fn out(v: i64) -> i64 {\n"
    "    let p: i64 = __mind_alloc(8);\n"
    "    __mind_store_i8(p, 65 + (v % 26 + 26) % 26);\n"
    "    __mind_write(1, p, 1, 0 - 1);\n"
    "    v\n"
    "}\n"
)


def wrap(v: int) -> int:
    v &= MASK
    return v - (1 << 64) if v >> 63 else v


def tdiv(a: int, b: int) -> int:
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b > 0) else -q


def tmod(a: int, b: int) -> int:
    return a - b * tdiv(a, b)


# ---------------------------------------------------------------- AST shapes
# expr:  ("lit", n) ("var", name) ("bin", op, a, b) ("call", fname, [args])
#        ("vif", cond, [stmts], expr, [stmts], expr)
# cond:  ("cmp", op, a, b) ("and", c, c) ("or", c, c) ("not", c) ("vifc", vif_expr, op, expr)
# stmt:  ("let", name, expr) ("set", name, expr) ("out", expr)
#        ("if", cond, [stmts], else)  else = None | [stmts] | ("elif", if_stmt)
#        ("while", counter, k, cond, [stmts]) ("break",) ("continue",) ("ret", expr)


class Gen:
    def __init__(self, seed: int, profile: str = "mix"):
        self.r = random.Random(seed)
        self.profile = profile
        self.n = 0
        self.muts = []
        self.in_loop = False

    def fresh(self, p: str) -> str:
        self.n += 1
        return f"{p}{self.n}"

    # ---- expressions
    def expr(self, scope, d, fns):
        r = self.r
        roll = r.random()
        if d <= 0 or roll < 0.30:
            if scope and r.random() < 0.7:
                return ("var", r.choice(scope))
            return ("lit", r.randint(0, 20))
        if roll < 0.62:
            op = r.choice(["+", "-", "*", "+", "-", "/", "%", "<<", ">>", "|"])
            a = self.expr(scope, d - 1, fns)
            if op in ("/", "%"):
                return ("bin", op, a, ("lit", r.randint(1, 7)))
            if op in ("<<", ">>"):
                return ("bin", op, a, ("lit", r.randint(0, 5)))
            return ("bin", op, a, self.expr(scope, d - 1, fns))
        if roll < 0.78 and fns:
            f, arity = r.choice(fns)
            return ("call", f, [self.expr(scope, d - 1, fns) for _ in range(arity)])
        if roll < 0.86:
            return ("call", "out", [self.expr(scope, d - 1, fns)])
        # value-if, optionally with statements in the branches
        c = self.cond(scope, d - 1, fns)
        ts, es = [], []
        sc_t, sc_e = list(scope), list(scope)
        if r.random() < 0.35:
            t = self.fresh("t")
            ts.append(("let", t, self.expr(scope, d - 1, fns)))
            sc_t.append(t)
        if r.random() < 0.25:
            ts.append(("out", self.expr(sc_t, d - 1, fns)))
        if self.muts and r.random() < 0.2:
            # an assignment to an enclosing variable inside a value-if branch
            ts.append(("set", r.choice(self.muts), self.expr(sc_t, d - 1, fns)))
        if r.random() < 0.3:
            e = self.fresh("e")
            es.append(("let", e, self.expr(scope, d - 1, fns)))
            sc_e.append(e)
        if r.random() < 0.12:
            # one branch leaves the value-if: return, or break/continue in a loop
            kinds = ["ret"] + (["break", "continue"] if self.in_loop else [])
            k = r.choice(kinds)
            pay = self.expr(scope, d - 1, fns) if k == "ret" else None
            return ("vifd", c, k, pay, es, self.expr(sc_e, d - 1, fns))
        return ("vif", c, ts, self.expr(sc_t, d - 1, fns), es, self.expr(sc_e, d - 1, fns))

    def cond(self, scope, d, fns):
        r = self.r
        roll = r.random()
        if d <= 0 or roll < 0.45:
            return ("cmp", r.choice(["<", "<=", ">", ">=", "==", "!="]),
                    self.expr(scope, max(d - 1, 0), fns), self.expr(scope, max(d - 1, 0), fns))
        if roll < 0.66:
            return ("and", self.cond(scope, d - 1, fns), self.cond(scope, d - 1, fns))
        if roll < 0.86:
            return ("or", self.cond(scope, d - 1, fns), self.cond(scope, d - 1, fns))
        if roll < 0.92:
            return ("not", self.cond(scope, d - 1, fns))
        v = self.expr(scope, d - 1, fns)
        if v[0] != "vif":
            c = self.cond(scope, d - 1, fns)
            v = ("vif", c, [], self.expr(scope, d - 1, fns), [], self.expr(scope, d - 1, fns))
        return ("vifc", v, r.choice(["<", ">", "==", "!="]), self.expr(scope, d - 1, fns))

    # ---- statements
    def block(self, scope, muts, d, fns, in_loop, n):
        out = []
        scope = list(scope)
        muts = list(muts)
        for _ in range(n):
            st = self.stmt(scope, muts, d, fns, in_loop)
            out.append(st)
            if st[0] == "let":
                scope.append(st[1])
                muts.append(st[1])
            if st[0] == "seq":
                out.pop()
                out.extend(st[1])
                for s in st[1]:
                    if s[0] == "let":
                        scope.append(s[1])
        return out

    def stmt(self, scope, muts, d, fns, in_loop):
        r = self.r
        self.muts = list(muts)
        self.in_loop = in_loop
        w = {"let": 3, "set": 4, "out": 2, "if": 4, "while": 3, "ret": 1, "exit": 2}
        if d <= 0:
            w["if"] = 0
            w["while"] = 0
        if not in_loop:
            w["exit"] = 0
        if not muts:
            w["set"] = 0
        kinds = [k for k in w for _ in range(w[k])]
        k = r.choice(kinds)
        ed = 2
        if k == "let":
            return ("let", self.fresh("v"), self.expr(scope, ed, fns))
        if k == "set":
            return ("set", r.choice(muts), self.expr(scope, ed, fns))
        if k == "out":
            return ("out", self.expr(scope, ed, fns))
        if k == "ret":
            return ("ret", self.expr(scope, ed, fns))
        if k == "exit":
            return (r.choice(["break", "continue"]),)
        if k == "if":
            return self.if_stmt(scope, muts, d, fns, in_loop)
        # while with its own counter declared just before it
        c = self.fresh("c")
        kk = r.randint(1, 5)
        cond = ("cmp", "<", ("var", c), ("lit", kk))
        if r.random() < 0.6:
            # no break/continue inside a loop's own condition (it would name
            # the enclosing loop, a corner the corpus does not pin)
            self.in_loop = False
            extra = self.cond(scope + [c], 2, fns)
            cond = ("and", cond, extra) if r.random() < 0.6 else ("and", extra, cond)
        body = [("set", c, ("bin", "+", ("var", c), ("lit", 1)))]
        body += self.block(scope + [c], muts, d - 1, fns, True, r.randint(1, 4))
        return ("seq", [("let", c, ("lit", 0)), ("while", c, kk, cond, body)])

    def if_stmt(self, scope, muts, d, fns, in_loop):
        r = self.r
        self.muts, self.in_loop = list(muts), in_loop
        c = self.cond(scope, 2, fns)
        then = self.block(scope, muts, d - 1, fns, in_loop, r.randint(1, 3))
        roll = r.random()
        if roll < 0.35:
            els = None
        elif roll < 0.55:
            els = ("elif", self.if_stmt(scope, muts, d - 1, fns, in_loop))
        else:
            els = self.block(scope, muts, d - 1, fns, in_loop, r.randint(1, 3))
        return ("if", c, then, els)

    def program(self, n_helpers=None):
        r = self.r
        self.shadow = {}
        nh = r.randint(0, 3) if n_helpers is None else n_helpers
        fns, sigs = [], []
        for i in range(nh):
            name = f"h{i}"
            arity = r.randint(0, 3)
            params = [f"p{j}" for j in range(arity)]
            body = self.block(params, params, r.randint(1, 3), list(sigs), False, r.randint(1, 4))
            self.muts, self.in_loop = list(params), False
            body.append(("ret", self.expr(params, 2, sigs)))
            fns.append((name, params, body))
            self.shadow[name] = r.random() < 0.5
            sigs.append((name, arity))
        body = self.block([], [], r.randint(2, 4), list(sigs), False, r.randint(2, 5))
        self.muts, self.in_loop = [], False
        body.append(("ret", self.expr([], 1, sigs)))
        fns.append(("main", [], body))
        return fns, dict(self.shadow)


# ---------------------------------------------------------------- printing
def pe(e) -> str:
    k = e[0]
    if k == "lit":
        return str(e[1])
    if k == "var":
        return e[1]
    if k == "bin":
        return f"({pe(e[2])} {e[1]} {pe(e[3])})"
    if k == "call":
        return f"{e[1]}({', '.join(pe(a) for a in e[2])})"
    if k == "vif":
        return f"(if {pc(e[1])} {{ {ps_inline(e[2])}{pe(e[3])} }} else {{ {ps_inline(e[4])}{pe(e[5])} }})"
    if k == "vifd":
        leave = f"return {pe(e[3])};" if e[2] == "ret" else f"{e[2]};"
        return f"(if {pc(e[1])} {{ {leave} }} else {{ {ps_inline(e[4])}{pe(e[5])} }})"
    raise ValueError(k)


def pc(c) -> str:
    k = c[0]
    if k == "cmp":
        return f"{pe(c[2])} {c[1]} {pe(c[3])}"
    if k == "and":
        return f"({pc(c[1])}) && ({pc(c[2])})"
    if k == "or":
        return f"({pc(c[1])}) || ({pc(c[2])})"
    if k == "not":
        return f"!({pc(c[1])})"
    if k == "vifc":
        return f"{pe(c[1])} {c[2]} {pe(c[3])}"
    raise ValueError(k)


def ps_inline(stmts) -> str:
    return "".join(ps(s, 0).strip() + " " for s in stmts)


def ps(s, ind: int) -> str:
    sp = "    " * ind
    k = s[0]
    if k == "let":
        return f"{sp}let mut {s[1]}: i64 = {pe(s[2])};\n"
    if k == "set":
        return f"{sp}{s[1]} = {pe(s[2])};\n"
    if k == "out":
        return f"{sp}out({pe(s[1])});\n"
    if k == "ret":
        return f"{sp}return {pe(s[1])};\n"
    if k in ("break", "continue"):
        return f"{sp}{k};\n"
    if k == "while":
        body = "".join(ps(x, ind + 1) for x in s[4])
        return f"{sp}while {pc(s[3])} {{\n{body}{sp}}}\n"
    if k == "if":
        return sp + pif(s, ind) + "\n"
    raise ValueError(k)


def pif(s, ind: int) -> str:
    sp = "    " * ind
    then = "".join(ps(x, ind + 1) for x in s[2])
    txt = f"if {pc(s[1])} {{\n{then}{sp}}}"
    els = s[3]
    if els is None:
        return txt
    if els[0] == "elif":
        return txt + " else " + pif(els[1], ind)
    body = "".join(ps(x, ind + 1) for x in els)
    return txt + f" else {{\n{body}{sp}}}"


def render(prog) -> str:
    fns, shadow = prog
    out = [OUT_FN]
    for name, params, body in fns:
        sig = ", ".join(f"{p}: i64" for p in params)
        out.append(f"fn {name}({sig}) -> i64 {{\n")
        # half the helpers assign their parameters directly, the other half
        # first rebind them as mutable locals (both are valid MIND)
        if shadow.get(name):
            for p in params:
                out.append(f"    let mut {p}: i64 = {p};\n")
        out.extend(ps(s, 1) for s in body)
        out.append("}\n")
    return "".join(out)


# ---------------------------------------------------------------- interpreter
class Ret(Exception):
    def __init__(self, v):
        self.v = v


class Brk(Exception):
    pass


class Cont(Exception):
    pass


class Budget(Exception):
    pass


class Interp:
    def __init__(self, fns, budget=2_000_000):
        self.fns = {f[0]: f for f in fns}
        self.out = []
        self.budget = budget

    def tick(self):
        self.budget -= 1
        if self.budget < 0:
            raise Budget()

    def call(self, name, args):
        self.tick()
        if name == "out":
            v = args[0]
            t = tmod(v, 26)
            self.out.append(65 + tmod(t + 26, 26))
            return v
        _, params, body = self.fns[name]
        env = [dict(zip(params, args))]
        try:
            self.block(body, env)
        except Ret as r:
            return r.v
        raise RuntimeError("fell off the end")

    def look(self, env, n):
        for fr in reversed(env):
            if n in fr:
                return fr
        raise KeyError(n)

    def ev(self, e, env):
        self.tick()
        k = e[0]
        if k == "lit":
            return e[1]
        if k == "var":
            return self.look(env, e[1])[e[1]]
        if k == "bin":
            a = self.ev(e[2], env)
            b = self.ev(e[3], env)
            op = e[1]
            if op == "+":
                return wrap(a + b)
            if op == "-":
                return wrap(a - b)
            if op == "*":
                return wrap(a * b)
            if op == "/":
                return wrap(tdiv(a, b))
            if op == "%":
                return wrap(tmod(a, b))
            if op == "<<":
                return wrap(a << b)
            if op == ">>":
                return a >> b
            if op == "|":
                return wrap((a & MASK) | (b & MASK))
            raise ValueError(op)
        if k == "call":
            args = [self.ev(a, env) for a in e[2]]
            return self.call(e[1], args)
        if k == "vif":
            if self.cond(e[1], env):
                env.append({})
                try:
                    self.block(e[2], env, scoped=False)
                    return self.ev(e[3], env)
                finally:
                    env.pop()
            env.append({})
            try:
                self.block(e[4], env, scoped=False)
                return self.ev(e[5], env)
            finally:
                env.pop()
        if k == "vifd":
            if self.cond(e[1], env):
                if e[2] == "ret":
                    raise Ret(self.ev(e[3], env))
                raise Brk() if e[2] == "break" else Cont()
            env.append({})
            try:
                self.block(e[4], env, scoped=False)
                return self.ev(e[5], env)
            finally:
                env.pop()
        raise ValueError(k)

    def cond(self, c, env):
        k = c[0]
        if k == "cmp":
            a = self.ev(c[2], env)
            b = self.ev(c[3], env)
            return {"<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b,
                    "==": a == b, "!=": a != b}[c[1]]
        if k == "and":
            return self.cond(c[1], env) and self.cond(c[2], env)
        if k == "or":
            return self.cond(c[1], env) or self.cond(c[2], env)
        if k == "not":
            return not self.cond(c[1], env)
        if k == "vifc":
            a = self.ev(c[1], env)
            b = self.ev(c[3], env)
            return {"<": a < b, ">": a > b, "==": a == b, "!=": a != b}[c[2]]
        raise ValueError(k)

    def block(self, stmts, env, scoped=True):
        if scoped:
            env.append({})
        try:
            for s in stmts:
                self.stmt(s, env)
        finally:
            if scoped:
                env.pop()

    def stmt(self, s, env):
        self.tick()
        k = s[0]
        if k == "let":
            env[-1][s[1]] = self.ev(s[2], env)
        elif k == "set":
            v = self.ev(s[2], env)
            self.look(env, s[1])[s[1]] = v
        elif k == "out":
            self.call("out", [self.ev(s[1], env)])
        elif k == "ret":
            raise Ret(self.ev(s[1], env))
        elif k == "break":
            raise Brk()
        elif k == "continue":
            raise Cont()
        elif k == "if":
            self.if_(s, env)
        elif k == "while":
            while self.cond(s[3], env):
                try:
                    self.block(s[4], env)
                except Brk:
                    break
                except Cont:
                    continue
        else:
            raise ValueError(k)

    def if_(self, s, env):
        if self.cond(s[1], env):
            self.block(s[2], env)
            return
        els = s[3]
        if els is None:
            return
        if els[0] == "elif":
            self.if_(els[1], env)
            return
        self.block(els, env)


def run_program(prog):
    """(exit_code, stdout_bytes) of `main`, or None when the step budget is hit."""
    it = Interp(prog[0])
    try:
        v = it.call("main", [])
    except Budget:
        return None
    return v & 255, bytes(it.out)


def _expr_facts(e, reads, vsets):
    """Collect variable reads and the targets assigned inside value-if branches."""
    k = e[0]
    if k == "var":
        reads.add(e[1])
    elif k == "bin":
        _expr_facts(e[2], reads, vsets)
        _expr_facts(e[3], reads, vsets)
    elif k == "call":
        for a in e[2]:
            _expr_facts(a, reads, vsets)
    elif k in ("vif", "vifd"):
        _cond_facts(e[1], reads, vsets)
        branches = [(e[2], e[3]), (e[4], e[5])] if k == "vif" else [(e[4], e[5])]
        if k == "vifd" and e[3] is not None:
            _expr_facts(e[3], reads, vsets)
        for stmts, tail in branches:
            for st in stmts:
                if st[0] == "set":
                    vsets.add(st[1])
                _expr_facts(st[2] if st[0] in ("let", "set") else st[1], reads, vsets)
            _expr_facts(tail, reads, vsets)


def _cond_facts(c, reads, vsets):
    k = c[0]
    if k == "cmp":
        _expr_facts(c[2], reads, vsets)
        _expr_facts(c[3], reads, vsets)
    elif k in ("and", "or"):
        _cond_facts(c[1], reads, vsets)
        _cond_facts(c[2], reads, vsets)
    elif k == "not":
        _cond_facts(c[1], reads, vsets)
    elif k == "vifc":
        _expr_facts(c[1], reads, vsets)
        _expr_facts(c[3], reads, vsets)


def _stmts_branch_assign(stmts) -> bool:
    for st in stmts:
        reads, vsets = set(), set()
        k = st[0]
        if k in ("let", "set"):
            _expr_facts(st[2], reads, vsets)
        elif k in ("out", "ret"):
            _expr_facts(st[1], reads, vsets)
        elif k == "if":
            _cond_facts(st[1], reads, vsets)
            if _stmts_branch_assign(st[2]):
                return True
            els = st[3]
            while els is not None and els[0] == "elif":
                _cond_facts(els[1][1], reads, vsets)
                if _stmts_branch_assign(els[1][2]):
                    return True
                els = els[1][3]
            if els is not None and _stmts_branch_assign(els):
                return True
        elif k == "while":
            _cond_facts(st[3], reads, vsets)
            if _stmts_branch_assign(st[4]):
                return True
        if vsets:
            return True
    return False


def has_branch_assign(prog) -> bool:
    """True when a value-if branch assigns an enclosing variable anywhere.

    The MLIR backend loses such assignments in several positions, all pinned by
    fixtures: a value-if inside a discarded call statement
    (mlir_lost_branch_assign.mind, first seen in corpus program cf_3_168), in a
    `while` condition (mlir_lost_cond_assign.mind, cf_4_55), and in a loop body
    where the store must be loop-carried (mlir_lost_loop_assign.mind, cf_11_36).
    MIND keeps every such store: the interpreter and the native backend agree.
    The smoke therefore gates these programs on native == Python and reports
    the MLIR result separately; every other program must agree three ways."""
    return any(_stmts_branch_assign(body) for _, _, body in prog[0])


def corpus(seed: int, count: int):
    """`count` (name, source, expected, branch_assign) tuples; expected =
    (exit, stdout); branch_assign = has_branch_assign(program)."""
    out = []
    i = 0
    while len(out) < count:
        g = Gen(seed * 1000003 + i)
        prog = g.program()
        i += 1
        exp = run_program(prog)
        if exp is None or len(exp[1]) > 4000:
            continue
        out.append((f"cf_{seed}_{i - 1}", render(prog), exp, has_branch_assign(prog)))
    return out
