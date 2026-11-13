"""Mini-language of the nested-binding scoping smoke (self_host_native_shadow_scope_smoke.py).

Programs are nested tuples (let / set / if / while / break / continue / return over
i64 expressions). This module holds what the smoke needs besides the compilers:

  * a lexically scoped interpreter (the oracle: each block, branch and loop body
    gets its own frame, so a nested `let` never touches an outer binding);
  * the MIND source printer;
  * the seeded generator (shadowing mixed with while/if/else/break/continue/return
    at block depths 1-4, a helper fn whose parameters are shadowed);
  * the alpha-renamer that builds a program's shadow-free twin.

Imported by the smoke; running it directly does nothing.
"""
from __future__ import annotations

import random

MASK = (1 << 64) - 1


def wrap(v: int) -> int:
    v &= MASK
    return v - (1 << 64) if v >> 63 else v


# ---------------------------------------------------------------- program model
class Ret(Exception):
    def __init__(self, v):
        self.v = v


class Brk(Exception):
    pass


class Cont(Exception):
    pass


class Env:
    def __init__(self, trace=None):
        self.frames = [{}]
        # frame count at the start of each enclosing loop body (innermost last)
        self.loop_bases = []
        # trace["exit_shadow"]: a break/continue ran while a name declared inside
        # the loop body also had a binding outside it (the MLIR defect class)
        self.trace = trace if trace is not None else {}

    def push(self):
        self.frames.append({})

    def pop(self):
        self.frames.pop()

    def let(self, n, v, mut):
        self.frames[-1][n] = [v, mut]

    def find(self, n):
        for fr in reversed(self.frames):
            if n in fr:
                return fr[n]
        raise KeyError(n)

    def visible(self):
        seen = {}
        for fr in self.frames:
            for n, b in fr.items():
                seen[n] = b
        return seen


def ev(e, env, fns):
    k = e[0]
    if k == "c":
        return e[1]
    if k == "v":
        return env.find(e[1])[0]
    if k == "bin":
        a, b = ev(e[2], env, fns), ev(e[3], env, fns)
        return wrap({"+": a + b, "-": a - b, "*": a * b}[e[1]])
    if k == "cmp":
        a, b = ev(e[2], env, fns), ev(e[3], env, fns)
        return int({"<": a < b, ">": a > b, "==": a == b, "!=": a != b,
                    "<=": a <= b, ">=": a >= b}[e[1]])
    if k == "vif":
        return ev(e[2], env, fns) if ev(e[1], env, fns) else ev(e[3], env, fns)
    if k == "call":
        return call(fns, e[1], [ev(a, env, fns) for a in e[2]], env.trace)
    raise ValueError(k)


def run_block(stmts, env, fns, new_scope=True):
    if new_scope:
        env.push()
    try:
        for s in stmts:
            run_stmt(s, env, fns)
    finally:
        if new_scope:
            env.pop()


def run_stmt(s, env, fns):
    k = s[0]
    if k == "let":
        env.let(s[1], ev(s[3], env, fns), s[2])
    elif k == "set":
        b = env.find(s[1])
        assert b[1], f"assign to immutable {s[1]}"
        b[0] = ev(s[2], env, fns)
    elif k == "if":
        if ev(s[1], env, fns):
            run_block(s[2], env, fns)
        elif s[3] is not None:
            run_block(s[3], env, fns)
    elif k == "while":
        env.loop_bases.append(len(env.frames))
        try:
            while ev(s[1], env, fns):
                try:
                    run_block(s[2], env, fns)
                except Brk:
                    break
                except Cont:
                    continue
        finally:
            env.loop_bases.pop()
    elif k in ("break", "continue"):
        base = env.loop_bases[-1]
        inside = {n for fr in env.frames[base:] for n in fr}
        outside = {n for fr in env.frames[:base] for n in fr}
        if inside & outside:
            env.trace["exit_shadow"] = True
        raise Brk() if k == "break" else Cont()
    elif k == "return":
        raise Ret(ev(s[1], env, fns))
    else:
        raise ValueError(k)


def call(fns, name, args, trace=None):
    params, body = fns[name]
    env = Env(trace)
    for p, a in zip(params, args):
        env.let(p, a, False)
    try:
        run_block(body, env, fns, new_scope=False)
    except Ret as r:
        return r.v
    raise RuntimeError("fn fell off the end")


def model_exit(fns, trace=None) -> int:
    return call(fns, "main", [], trace) & 255


# ---------------------------------------------------------------- printer
def pe(e):
    k = e[0]
    if k == "c":
        return str(e[1]) if e[1] >= 0 else f"(0 - {-e[1]})"
    if k == "v":
        return e[1]
    if k in ("bin", "cmp"):
        return f"({pe(e[2])} {e[1]} {pe(e[3])})" if k == "bin" else f"{pe(e[2])} {e[1]} {pe(e[3])}"
    if k == "vif":
        return f"if {pe(e[1])} {{ {pe(e[2])} }} else {{ {pe(e[3])} }}"
    if k == "call":
        return f"{e[1]}({', '.join(pe(a) for a in e[2])})"
    raise ValueError(k)


def ps(s, ind):
    p = "    " * ind
    k = s[0]
    if k == "let":
        return [f"{p}let {'mut ' if s[2] else ''}{s[1]}: i64 = {pe(s[3])};"]
    if k == "set":
        return [f"{p}{s[1]} = {pe(s[2])};"]
    if k == "if":
        out = [f"{p}if {pe(s[1])} {{"] + [ln for t in s[2] for ln in ps(t, ind + 1)]
        if s[3] is not None:
            out += [f"{p}}} else {{"] + [ln for t in s[3] for ln in ps(t, ind + 1)]
        return out + [f"{p}}}"]
    if k == "while":
        return [f"{p}while {pe(s[1])} {{"] + [ln for t in s[2] for ln in ps(t, ind + 1)] + [f"{p}}}"]
    if k in ("break", "continue"):
        return [f"{p}{k};"]
    if k == "return":
        return [f"{p}return {pe(s[1])};"]
    raise ValueError(k)


def render(fns) -> str:
    out = []
    for name, (params, body) in fns.items():
        ret = " -> i64"
        sig = ", ".join(f"{q}: i64" for q in params)
        out.append(f"fn {name}({sig}){ret} {{")
        for s in body:
            out += ps(s, 1)
        out.append("}")
        out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------- generator
POOL = ["x", "y", "z"]


class Gen:
    def __init__(self, rng: random.Random, mode: str):
        self.r = rng
        self.mode = mode
        self.k = 0

    def expr(self, scope, d=0):
        r = self.r
        names = list(scope)
        c = r.random()
        if names and c < 0.45:
            return ("v", r.choice(names))
        if d < 2 and c < 0.85:
            op = r.choice(["+", "+", "-", "*"])
            b = ("c", r.choice([2, 3])) if op == "*" else self.expr(scope, d + 1)
            return ("bin", op, self.expr(scope, d + 1), b)
        return ("c", r.randint(-9, 40))

    def cond(self, scope):
        return ("cmp", self.r.choice(["<", ">", "==", "!=", "<=", ">="]),
                self.expr(scope, 1), ("c", self.r.randint(-5, 30)))

    def block(self, scope, depth, maxd, in_loop, nstmt, restricted=False):
        """scope: dict name->mut (visible). Returns stmts; mutates a COPY.

        restricted (lean mode, an `if` branch inside a loop): only assignments and
        break/continue -- the loop shapes the native control-flow subset accepts."""
        r = self.r
        lean = self.mode in ("lean", "exit")
        scope = dict(scope)
        out = []
        if restricted:
            for _ in range(nstmt):
                muts = [n for n, m in scope.items() if m]
                if muts and r.random() < 0.8:
                    out.append(("set", r.choice(muts), self.expr(scope)))
                else:
                    out.append(("break",) if r.random() < 0.5 else ("continue",))
            return out
        for _ in range(nstmt):
            c = r.random()
            if c < 0.34:
                n = r.choice(POOL)
                mut = r.random() < 0.7
                if r.random() < 0.15 and scope and not (lean and in_loop):
                    e = ("vif", self.cond(scope), self.expr(scope), self.expr(scope))
                else:
                    e = self.expr(scope)
                out.append(("let", n, mut, e))
                scope[n] = mut
            elif c < 0.58:
                muts = [n for n, m in scope.items() if m]
                if muts:
                    n = r.choice(muts)
                    out.append(("set", n, self.expr(scope)))
            elif c < 0.74 and depth < maxd:
                rs = lean and in_loop
                th = self.block(scope, depth + 1, maxd, in_loop, r.randint(1, 4), rs)
                el = (self.block(scope, depth + 1, maxd, in_loop, r.randint(1, 3), rs)
                      if r.random() < (0.2 if lean else 0.45) else None)
                out.append(("if", self.cond(scope), th, el))
            elif c < 0.88 and depth < maxd:
                self.k += 1
                kv = f"k{self.k}"
                out.append(("let", kv, True, ("c", 0)))
                body_scope = dict(scope)
                body_scope.pop(kv, None)
                pre, post = [], []
                muts = [n for n, m in body_scope.items() if m and not n.startswith("k")]
                if self.mode == "exit" and muts:
                    # write a carried outer name, shadow it, then leave the
                    # iteration early while the shadow is in scope
                    n = r.choice(muts)
                    shadow_mut = r.random() < 0.6
                    pre = [("set", n, self.expr(body_scope)),
                           ("let", n, shadow_mut, self.expr(body_scope))]
                    body_scope[n] = shadow_mut
                    post = [("if", self.cond(body_scope),
                             [("break",)] if r.random() < 0.5 else [("continue",)], None)]
                inner = self.block(body_scope, depth + 1, maxd, True, r.randint(1, 4))
                body = [("set", kv, ("bin", "+", ("v", kv), ("c", 1)))] + pre + inner + post
                out.append(("while", ("cmp", "<", ("v", kv), ("c", r.randint(1, 4))), body))
            elif in_loop and c < 0.93:
                out.append(("if", self.cond(scope),
                            [("break",)] if r.random() < 0.5 else [("continue",)], None))
            elif self.mode != "noret" and c < 0.96 and depth > 0 and not (lean and in_loop):
                out.append(("if", self.cond(scope), [("return", self.mix(scope))], None))
        return out

    def mix(self, scope):
        e = ("c", 1)
        for i, n in enumerate(sorted(scope)):
            if n.startswith("k"):
                continue
            e = ("bin", "+", ("bin", "*", e, ("c", 3)), ("v", n))
        return e

    def program(self):
        r = self.r
        fns = {}
        maxd = r.randint(1, 4)
        helper = r.random() < 0.4
        if helper:
            params = ["x", "y"]
            scope = {"x": False, "y": False}
            body = self.block(scope, 0, maxd, False, r.randint(2, 5))
            vis = dict(scope)
            for s in body:
                if s[0] == "let":
                    vis[s[1]] = s[2]
            body.append(("return", self.mix(vis)))
            fns["h"] = (params, body)
        scope = {"x": True, "y": True}
        body = [("let", "x", True, ("c", r.randint(0, 20))),
                ("let", "y", True, ("c", r.randint(0, 20)))]
        body += self.block(scope, 0, maxd, False, r.randint(2, 6))
        vis = dict(scope)
        for s in body:
            if s[0] == "let":
                vis[s[1]] = s[2]
        tail = self.mix(vis)
        if helper:
            tail = ("bin", "+", tail, ("call", "h", [("v", "x"), ("v", "y")]))
        body.append(("return", tail))
        fns["main"] = ([], body)
        return fns


def has_nested_shadow(fns) -> bool:
    def walk(stmts, outer, depth):
        here = set()
        for s in stmts:
            if s[0] == "let":
                if depth > 0 and s[1] in outer and not s[1].startswith("k"):
                    return True
                here.add(s[1])
            vis = outer | here
            if s[0] == "if":
                if walk(s[2], vis, depth + 1) or (s[3] and walk(s[3], vis, depth + 1)):
                    return True
            if s[0] == "while" and walk(s[2], vis, depth + 1):
                return True
        return False
    return any(walk(b, set(p), 0) for p, b in fns.values())


def alpha_rename(fns):
    """The same program with every nested shadowing `let` (and the reads and
    writes that resolve to it) renamed to a fresh name: identical semantics, no
    shadowing. A native refusal of the original is fail-closed only when this
    twin is refused too (the control-flow subset, not the scoping, refused it)."""
    fresh = [0]

    def rn_e(e, env):
        k = e[0]
        if k == "c":
            return e
        if k == "v":
            return ("v", env.get(e[1], e[1]))
        if k in ("bin", "cmp"):
            return (k, e[1], rn_e(e[2], env), rn_e(e[3], env))
        if k == "vif":
            return (k, rn_e(e[1], env), rn_e(e[2], env), rn_e(e[3], env))
        if k == "call":
            return (k, e[1], [rn_e(a, env) for a in e[2]])
        raise ValueError(k)

    def rn_b(stmts, env, outer, local):
        env, local, out = dict(env), dict(local), []
        for s in stmts:
            k = s[0]
            if k == "let":
                init = rn_e(s[3], env)
                n = s[1]
                if n in local:
                    alias = local[n]  # same-block re-declaration keeps its storage name
                elif n in outer:
                    fresh[0] += 1
                    alias = f"{n}_s{fresh[0]}"
                else:
                    alias = n
                local[n] = alias
                env[n] = alias
                out.append(("let", alias, s[2], init))
            elif k == "set":
                out.append(("set", env.get(s[1], s[1]), rn_e(s[2], env)))
            elif k == "if":
                vis = outer | set(local)
                out.append(("if", rn_e(s[1], env), rn_b(s[2], env, vis, {}),
                            None if s[3] is None else rn_b(s[3], env, vis, {})))
            elif k == "while":
                out.append(("while", rn_e(s[1], env),
                            rn_b(s[2], env, outer | set(local), {})))
            elif k == "return":
                out.append(("return", rn_e(s[1], env)))
            else:
                out.append(s)
        return out

    return {name: (params, rn_b(body, {}, set(), {q: q for q in params}))
            for name, (params, body) in fns.items()}
