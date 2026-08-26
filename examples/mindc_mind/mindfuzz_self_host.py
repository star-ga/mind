#!/usr/bin/env python3
"""
mindfuzz_self_host.py — Csmith-lineage DIFFERENTIAL fuzzer retiring the
"twin emitters must agree" gap class for the MIND self-host mic@3 note
(Rust-independence #14 follow-on; the in-tree sibling of
tests/mindfuzz_cross_substrate.rs, aimed at a DIFFERENT emitter pair).

What it proves
--------------
The fixed-fixture smokes (self_host_native_elf_smoke.py, the *_mic3_smoke.py
family) prove the pure-MIND self-host emitter agrees with the Rust oracle on a
HANDFUL of committed programs. This fuzzer proves the agreement over the
CONSTRUCT SPACE: it deterministically generates random valid MIND programs from
the currently-supported scalar-i64 statement subset, compiles each TWO ways,
and asserts the mic@3 trace-hash NOTES are byte-identical:

  SELF-HOST note — libmindc_mind.so (pure-MIND front-end, main.mind): seed the
    bundled stdlib AHEAD of the program (`combined = std_blob ++ prog`,
    `user_lo = len(std_blob)` — the _seeded_buffer() contract of the native-ELF
    smoke), call `selftest_native_elf_u(combined, len, user_lo)`, and read the
    self-computed 32-byte PT_NOTE off the emitted ELF (`got[-32:]`; the last 52
    bytes are the 12-byte nhdr + "MIND" name + hash). A len-0 result is the
    entry's fail-closed refusal (nb_trace_hash hit a shape it cannot emit —
    NEVER a wrong note).

  RUST reference — the A9b test-time derivation (no frozen note): the note is
    `mini_sha256(emit_mic3(seeded+pruned combined IR))`, and the committed
    `tests/_ref_mic3_dump.rs::ref_ir` reproduces EXACTLY that seeding +
    call-graph prune (`mindc --emit-mic3` cannot substitute — it emits the
    standard-pipeline IR with a different id namespace). We run that helper
    with MIND_REF_OUT_DIR pointed at a temp dir holding the fuzz program AS
    `main.mind` and parse the `REF main: ... note=<hex>` line — the SAME
    mechanism derive_rust_ref_notes() uses in the smoke, parameterized to an
    arbitrary user source. Both notes reflow together on any benign std/*.mind
    edit (the seed lives in BOTH oracles), so this gate cannot re-stale.

Because SHA-256 is collision-resistant, one wrong mic@3 byte flips the note —
note byte-identity across the generated space is exactly the "twin emitters
agree" assertion. On any mismatch the fuzzer additionally byte-diffs the
UNSEEDED mic@3 of the program alone (`selftest_nb_mic3(prog, 0)` vs
`mindc --emit-mic3 prog`) to localize the divergence, then delta-debounces the
program to a minimal reproducer.

Generator subset (seeded, deterministic; every program terminates)
------------------------------------------------------------------
  fn main() -> i64 + optional helper fns, i64-only;
  let / let mut with i64 init (const or binop of in-scope vars);
  assign to in-scope mut vars (top-level, in-if, in-loop);
  statement-if without else and if/else (branches assign outer mut vars or
    bind branch-local lets);
  while loops with bounded literal counters — nested 2- and 3-deep, MULTIPLE
    carried vars whose first-sight write order deliberately differs from
    declaration order, break, guarded continue (increment placed FIRST in the
    body so a continue can never skip it — termination by construction);
  SAME-SPELLING SHADOWING (a body-local `let x` over an outer `x`; loop
    counters are never shadowed, preserving the termination argument);
  + - * / % << >> | with divisors forced odd (`d | 1`) and shift amounts
    bounded to 0..5, and all six comparisons as loop/if conditions.

Semantics of disagreement / refusal
-----------------------------------
  * EITHER oracle fail-closes a program (self-host len-0 ELF / Rust parse or
    lower rejection)  -> SKIPPED (logged + counted), NOT a failure: the
    fuzzer targets the intersection both accept. A skip histogram is printed.
  * both accept, notes differ -> MINIMIZE (delta-debounce over the AST), print
    the minimal reproducer + note hex + first-diverging mic@3 bytes, stage it
    under mindfuzz_self_host_staged/, exit 1.
  * both accept, notes agree  -> counted toward K.
  * K == 0 (nothing both accept) -> FAIL exit 1: a fuzzer that compared
    nothing proves nothing — never a silent green.
  * .so unbuildable / cargo absent -> BLOCKED exit 2 (honest, no fake pass).

Batching (speed): accepted programs are grouped (--batch, default 8) into ONE
combined source (entries renamed fzN + one synthetic main calling fz0 — every
fuzz fn stays a prune ROOT on both sides, exactly like the solo seam), so one
cargo derivation covers the group. A batch that matches byte-for-byte passes
all members (any single wrong byte in any member flips the batch SHA-256); a
batch that fails closed or mismatches falls back to PER-PROGRAM derivations,
which are the ground truth either way. `--batch 1` disables batching.

Run:
  python3 examples/mindc_mind/mindfuzz_self_host.py --count 50
  python3 examples/mindc_mind/mindfuzz_self_host.py --count 1000 --seed 7 --verbose
  MINDC_SO=<.so> MINDC_BIN=<mindc> python3 examples/mindc_mind/mindfuzz_self_host.py
"""

import argparse
import ctypes
import hashlib
import os
import pathlib
import random
import re
import shutil
import subprocess
import sys
import tempfile

_HERE = pathlib.Path(__file__).parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_HERE))
from _selfhost_so import resolve_so  # noqa: E402

# Same 21 modules, SAME order as the smoke's _seeded_buffer() / Rust KEEP list
# (std.http + std.sha512 excluded exactly as the native-ELF oracle bundle does).
_STDLIB_MODULES = [
    "arena", "async", "blas", "cli", "fs", "io", "io_canon", "iouring",
    "json", "map", "net", "process", "reactor", "regex", "ring", "sha256",
    "string", "time", "toml", "tui", "vec",
]

_RUST_REF_TEST = "_ref_mic3_dump"
_RUST_REF_FEATURES = "std-surface cross-module-imports"

DEFAULT_COUNT = 200
DEFAULT_SEED = 1234
DEFAULT_BATCH = 8
_STMT_BUDGET = 40  # hard cap on total generated statements per program

_ARITH = ["+", "-", "*", "/", "%", "<<", ">>", "|"]
_ARITH_W = [30, 20, 15, 10, 10, 5, 5, 5]
_CMPS = ["<", "<=", ">", ">=", "==", "!="]


def std_blob() -> bytes:
    """The seeded stdlib prefix, byte-identical to the smoke's _seeded_buffer():
    newline-joined so tokens never merge across module boundaries."""
    std_dir = _REPO / "std"
    parts = [(std_dir / f"{m}.mind").read_bytes() for m in _STDLIB_MODULES]
    return b"\n".join(parts) + b"\n"


# ---------------------------------------------------------------------------
# AST: tuples all the way down — one source of truth for generation, emission,
# and AST-level minimization.
#   Expr  = ("const", v) | ("var", name) | ("bin", op, l, r)
#   Cond  = ("cmp", op, l, r)
#   Stmt  = ("let", mut, name, expr) | ("assign", name, expr)
#         | ("if", cond, then_stmts, else_stmts_or_None)
#         | ("while", name_counter, cond, body_stmts)
#         | ("break",) | ("continue",)
#   Fn    = ("fn", name, params, stmts, ret_expr)
#   Prog  = list[Fn]  (helpers first, entry `main` last)
# ---------------------------------------------------------------------------


class Scope:
    """Proper block-scope stack (innermost binding wins) mirroring what BOTH
    parsers do with braced blocks, plus the set of protected loop-counter
    names the generator must never shadow or assign. A loop counter is declared
    as a real `let mut cN: i64 = 0;` statement AHEAD of its while (the text
    `while cN < LIT` needs it in scope at the condition), stays visible after
    the loop (valid MIND), but only its own increment may write it."""

    def __init__(self):
        self.frames = [[]]       # stack of frames; each = list[(name, mut)]
        self.counters = set()    # loop counters: shadow/assign-forbidden

    def push_frame(self):
        self.frames.append([])

    def pop_frame(self):
        self.frames.pop()

    def push(self, name, mut):
        self.frames[-1].append((name, mut))

    def _bindings(self):
        for fr in self.frames:
            yield from fr

    def names(self):
        return [n for n, _ in self._bindings()]

    def _innermost(self):
        d = {}
        for n, m in self._bindings():
            d[n] = m
        return d

    def mut_names(self):
        # assignable = innermost binding is mut (a non-mut shadow hides the
        # outer mut binding, exactly as the parsers resolve it)
        return [n for n, m in self._innermost().items()
                if m and n not in self.counters]

    def shadowable(self):
        return [n for n in self._innermost() if n not in self.counters]

    def all_mut_decl_order(self):
        # declaration order, kept iff the INNERMOST binding is mut (carried-var
        # shuffle candidates)
        inner = self._innermost()
        seen, out = set(), []
        for n, m in self._bindings():
            if n not in seen:
                seen.add(n)
                if m and n not in self.counters and inner.get(n):
                    out.append(n)
        return out


class Gen:
    """Seeded generator over the supported subset. Every draw comes from
    self.r — random.Random(seed) — so a (seed, count) pair is fully
    reproducible (no wall-clock, no OS entropy)."""

    def __init__(self, seed: int):
        self.r = random.Random(seed)
        self.helper_seq = 0
        self.var_seq = 0
        self.ctr_seq = 0

    # -- expressions ---------------------------------------------------------

    def expr(self, sc: Scope, depth: int):
        r = self.r
        if depth <= 0 or r.randrange(100) < 45:
            return self.leaf(sc)
        op = r.choices(_ARITH, weights=_ARITH_W, k=1)[0]
        lhs = self.expr(sc, depth - 1)
        if op in ("/", "%"):
            # provably non-zero divisor: (d | 1) is always odd (MIN/-1 overflow
            # unreachable with these magnitudes)
            rhs = ("bin", "|", self.expr(sc, depth - 1), ("const", 1))
        elif op in ("<<", ">>"):
            rhs = ("const", r.randrange(6))  # always inside [0, 63]
        else:
            rhs = self.expr(sc, depth - 1)
        return ("bin", op, lhs, rhs)

    def leaf(self, sc: Scope):
        names = sc.names()
        r = self.r
        if names and r.randrange(100) < 60:
            return ("var", r.choice(names))
        return ("const", r.randrange(10))

    def cond(self, sc: Scope):
        return ("cmp", self.r.choice(_CMPS), self.expr(sc, 2), self.expr(sc, 2))

    # -- statements ----------------------------------------------------------

    def fresh_var(self):
        self.var_seq += 1
        return f"v{self.var_seq}"

    def fresh_counter(self):
        self.ctr_seq += 1
        return f"c{self.ctr_seq}"

    def gen_let(self, sc: Scope, depth=2):
        r = self.r
        mut = r.randrange(100) < 60
        shadowable = sc.shadowable()
        if shadowable and r.randrange(100) < 12:
            name = r.choice(shadowable)  # SAME-SPELLING SHADOWING
        else:
            name = self.fresh_var()
        init = self.expr(sc, depth)  # BEFORE the push: no self-referential lets
        sc.push(name, mut)
        return ("let", mut, name, init)

    def gen_assign(self, sc: Scope):
        targets = sc.mut_names()
        if not targets:
            return self.gen_let(sc)
        return ("assign", self.r.choice(targets), self.expr(sc, 3))

    def gen_if(self, sc: Scope, budget: int, loop_depth: int, in_loop: bool):
        """Statement-if, weighted by the EMPIRICAL support matrix of the nb
        mic@3 emitter (probed 2026-08-24 on the current tree):

          * if WITHOUT else: assign / let / mixed / nested-if branches all
            emit, in-loop and top-level.
          * if/else: the two branches must be KIND-HOMOGENEOUS — let+let emits
            everywhere; assign+assign emits IN-LOOP only; MIXED branch kinds
            (one writes outer vars, the other binds locals) fail closed.

        The fail-closed shapes are still generated at low frequency (PROBES):
        they are skipped honestly today and automatically start being
        differentially compared the day the emitter ports them."""
        r = self.r
        c = self.cond(sc)
        have_mut = bool(sc.mut_names())
        with_else = r.randrange(100) < 35
        if in_loop:
            # in-loop, ONLY pure-assign branches emit (with or without else);
            # any branch-local let (even in a bare if) fails closed — kept as a
            # 5% probe
            style = ("let" if (r.randrange(100) < 5 or not have_mut)
                     else "assign")
        elif not with_else:
            style = "mixed"  # every branch shape is proven for top-level bare if
        else:
            # top-level if/else: let+let is proven; 8% assign+assign probe
            # (known fail-closed)
            style = ("assign-probe" if (have_mut and r.randrange(100) < 8)
                     else "let")

        def branch():
            stmts = []
            sc.push_frame()  # branch-local lets stay branch-local (both parsers)
            n = 1 + r.randrange(3)
            for _ in range(n):
                if budget[0] <= 0:
                    break
                budget[0] -= 1
                roll = r.randrange(100)
                if style in ("assign", "assign-probe"):
                    stmts.append(self.gen_assign(sc) if sc.mut_names()
                                 else self.gen_let(sc))
                elif style == "let":
                    stmts.append(self.gen_let(sc))
                elif roll < 20 and budget[0] >= 2:
                    stmts.append(self.gen_if(sc, budget, loop_depth, False))
                elif sc.mut_names() and roll < 65:
                    stmts.append(self.gen_assign(sc))
                else:
                    stmts.append(self.gen_let(sc))
            if not stmts:
                budget[0] -= 1
                stmts.append(self.gen_assign(sc) if (style in ("assign",
                             "assign-probe") and sc.mut_names())
                             else self.gen_let(sc))
            sc.pop_frame()
            return stmts

        then_b = branch()
        else_b = branch() if with_else else None
        return ("if", c, then_b, else_b)

    def gen_while(self, sc: Scope, budget: int, loop_depth: int) -> list:
        """Bounded while. Returns the STATEMENT LIST `let mut cN: i64 = 0;` +
        `while cN < LIT { ... }` — the counter is a REAL declaration ahead of
        the loop (the condition text references it) and stays in scope after
        the loop (valid MIND), protected from shadows and foreign assigns.
        The increment goes FIRST when the body may `continue` (a continue jumps
        to the condition, skipping everything after it — placement is what
        keeps termination by construction), LAST otherwise (the proven
        SRC_LOOP_CONTROL shape)."""
        r = self.r
        ctr = self.fresh_counter()
        sc.push(ctr, True)
        sc.counters.add(ctr)
        bound = 1 + r.randrange(9)
        budget[0] -= 2  # the counter `let` + the `while` node itself
        out = [("let", True, ctr, ("const", 0))]

        body = []
        sc.push_frame()
        wants_continue = r.randrange(100) < 25
        if wants_continue:
            budget[0] -= 1
            body.append(("assign", ctr, ("bin", "+", ("var", ctr), ("const", 1))))

        # carried vars: assign >=2 outer mut vars, sometimes in reverse
        # declaration order, sometimes reading a var AFTER it was written this
        # iteration (first-sight vs declaration-order stress)
        carried = sc.all_mut_decl_order()
        if len(carried) >= 2 and r.randrange(100) < 70:
            k = min(len(carried), 2 + r.randrange(2))
            pick = r.sample(carried, k)
            if r.randrange(100) < 50:
                pick = list(reversed(pick))
            for nm in pick:
                if budget[0] <= 0:
                    break
                budget[0] -= 1
                body.append(("assign", nm, self.expr(sc, 2)))

        # generic body statements (nested whiles 2-3 deep, ifs, break/continue)
        extra = 1 + r.randrange(4)
        for _ in range(extra):
            if budget[0] <= 0:
                break
            roll = r.randrange(100)
            if loop_depth < 3 and roll < 25 and budget[0] >= 6:
                body.extend(self.gen_while(sc, budget, loop_depth + 1))
            elif roll < 55:
                budget[0] -= 1
                body.append(self.gen_if(sc, budget, loop_depth + 1, True))
            elif roll < 65 and wants_continue:
                budget[0] -= 1
                guard = ("cmp", r.choice(_CMPS), ("var", ctr),
                         ("const", r.randrange(bound + 1)))
                body.append(("if", guard, [("continue",)], None))
            elif roll < 75:
                budget[0] -= 1
                guard = ("cmp", r.choice(_CMPS), ("var", ctr),
                         ("const", r.randrange(bound + 1)))
                body.append(("if", guard, [("break",)], None))
            elif sc.mut_names():
                budget[0] -= 1
                body.append(self.gen_assign(sc))
            else:
                budget[0] -= 1
                body.append(self.gen_let(sc))

        if not wants_continue:
            budget[0] -= 1
            body.append(("assign", ctr, ("bin", "+", ("var", ctr), ("const", 1))))
        sc.pop_frame()

        cond = ("cmp", "<", ("var", ctr), ("const", bound))
        out.append(("while", ctr, cond, body))
        return out

    # -- functions / programs ------------------------------------------------

    def gen_fn(self, name: str, params: list, budget: int):
        # helpers get a FRESH scope holding only their params, like real MIND fns
        sc = Scope()
        for p in params:
            sc.push(p, False)
        stmts = []
        n = 2 + self.r.randrange(5)
        for _ in range(n):
            if budget[0] <= 0:
                break
            roll = self.r.randrange(100)
            if roll < 45:
                budget[0] -= 1
                stmts.append(self.gen_let(sc))
            elif sc.mut_names() and roll < 65:
                budget[0] -= 1
                stmts.append(self.gen_assign(sc))
            elif roll < 85 and budget[0] >= 6:
                stmts.extend(self.gen_while(sc, budget, 1))
            else:
                budget[0] -= 1
                stmts.append(self.gen_if(sc, budget, 1, False))
        if not stmts:
            budget[0] -= 1
            stmts.append(self.gen_let(sc))
        ret = self.expr(sc, 2)
        return ("fn", name, params, stmts, ret), sc

    def _program_once(self) -> list:
        """One program: 0-2 helpers + fn main. Helpers are only ever called
        from main's `return` (the fixture-proven call position)."""
        r = self.r
        budget = [_STMT_BUDGET]
        fns = []
        n_helpers = r.choices([0, 1, 2], weights=[55, 30, 15], k=1)[0]
        for _ in range(n_helpers):
            self.helper_seq += 1
            params = [f"x{self.helper_seq}"]
            if r.randrange(100) < 35:
                params.append(f"y{self.helper_seq}")
            fn, _sc = self.gen_fn(f"fzh{self.helper_seq}", params, budget)
            fns.append(fn)

        main_sc = Scope()
        stmts = []
        n = 4 + r.randrange(9)
        for _ in range(n):
            if budget[0] <= 0:
                break
            roll = r.randrange(100)
            if roll < 35:
                budget[0] -= 1
                stmts.append(self.gen_let(main_sc))
            elif main_sc.mut_names() and roll < 50:
                budget[0] -= 1
                stmts.append(self.gen_assign(main_sc))
            elif roll < 80 and budget[0] >= 6:
                stmts.extend(self.gen_while(main_sc, budget, 1))
            else:
                budget[0] -= 1
                stmts.append(self.gen_if(main_sc, budget, 1, False))
        if not stmts:
            budget[0] -= 1
            stmts.append(self.gen_let(main_sc))

        if fns and r.randrange(100) < 40:
            callee = fns[r.randrange(len(fns))]
            args = [("const", r.randrange(9)) for _ in callee[2]]
            ret = ("call", callee[1], args)
        else:
            ret = self.expr(main_sc, 2)
        fns.append(("fn", "main", [], stmts, ret))
        return fns

    def program(self) -> list:
        """Draw one program, redrawing (deterministically — same RNG stream) if
        the statement accounting overshoots the ~40-stmt cap."""
        for _attempt in range(5):
            fns = self._program_once()
            if count_stmts(fns) <= _STMT_BUDGET:
                return fns
        return fns


# ---------------------------------------------------------------------------
# Emitter — AST -> canonical .mind source (one stmt per line, fully-parenthesized
# binops; parseable by BOTH the Rust parser and the pure-MIND self-host lexer).
# ---------------------------------------------------------------------------


def emit_expr(e) -> str:
    k = e[0]
    if k == "const":
        return str(e[1])
    if k == "var":
        return e[1]
    if k == "bin":
        return f"({emit_expr(e[2])} {e[1]} {emit_expr(e[3])})"
    if k == "call":
        return f"{e[1]}({', '.join(emit_expr(a) for a in e[2])})"
    raise ValueError(f"bad expr {e!r}")


def emit_cond(c) -> str:
    return f"{emit_expr(c[2])} {c[1]} {emit_expr(c[3])}"


def emit_stmts(stmts, ind: int, out: list):
    pad = "    " * ind
    for s in stmts:
        k = s[0]
        if k == "let":
            mut = "mut " if s[1] else ""
            out.append(f"{pad}let {mut}{s[2]}: i64 = {emit_expr(s[3])};")
        elif k == "assign":
            out.append(f"{pad}{s[1]} = {emit_expr(s[2])};")
        elif k == "if":
            if s[3] is None:
                out.append(f"{pad}if {emit_cond(s[1])} {{")
                emit_stmts(s[2], ind + 1, out)
                out.append(f"{pad}}}")
            else:
                out.append(f"{pad}if {emit_cond(s[1])} {{")
                emit_stmts(s[2], ind + 1, out)
                out.append(f"{pad}}} else {{")
                emit_stmts(s[3], ind + 1, out)
                out.append(f"{pad}}}")
        elif k == "while":
            out.append(f"{pad}while {emit_cond(s[2])} {{")
            emit_stmts(s[3], ind + 1, out)
            out.append(f"{pad}}}")
        elif k == "break":
            out.append(f"{pad}break;")
        elif k == "continue":
            out.append(f"{pad}continue;")
        else:
            raise ValueError(f"bad stmt {s!r}")


def emit_fn(fn, entry_name: str | None = None) -> list:
    _, name, params, stmts, ret = fn
    nm = entry_name if entry_name is not None else name
    out = []
    ps = ", ".join(f"{p}: i64" for p in params)
    out.append(f"fn {nm}({ps}) -> i64 {{")
    emit_stmts(stmts, 1, out)
    out.append(f"    return {emit_expr(ret)};")
    out.append("}")
    return out


def emit_program(fns: list, entry_name: str = "main") -> str:
    """Emit the whole program; the fn literally named `main` is re-emitted as
    `entry_name` (batching renames each member's entry to fzN)."""
    lines = ["// mindfuzz_self_host generated program - deterministic seed."]
    for fn in fns:
        nm = entry_name if fn[1] == "main" else fn[1]
        lines.extend(emit_fn(fn, nm))
    return "\n".join(lines) + "\n"


def count_stmts(fns: list) -> int:
    def walk(stmts) -> int:
        n = 0
        for s in stmts:
            n += 1
            if s[0] == "if":
                n += walk(s[2]) + (walk(s[3]) if s[3] else 0)
            elif s[0] == "while":
                n += walk(s[3])
        return n
    return sum(walk(fn[3]) for fn in fns)


# ---------------------------------------------------------------------------
# Self-host oracle (pure-MIND front-end in libmindc_mind.so)
# ---------------------------------------------------------------------------

_RD = lambda a, o=0: ctypes.cast(a + o, ctypes.POINTER(ctypes.c_int64))[0]


def _es_bytes(es) -> bytes:
    """Read the returned EmitState's String buffer (addr/len at +0/+8) — the
    `rd = lambda...; sh = rd(es,0); got = string_at(rd(sh,0), rd(sh,8))` idiom
    of self_host_native_elf_smoke.py line ~559."""
    if not es:
        return b""
    sh = _RD(es, 0)
    if not sh:
        return b""
    n = _RD(sh, 8)
    return ctypes.string_at(_RD(sh, 0), n) if n > 0 else b""


class SelfHost:
    def __init__(self, so_path: pathlib.Path):
        self.lib = ctypes.CDLL(str(so_path))
        for nm, na in (("selftest_native_elf_u", 3), ("selftest_nb_mic3", 3)):
            fn = getattr(self.lib, nm)
            fn.restype = ctypes.c_int64
            fn.argtypes = [ctypes.c_int64] * na

    def _call(self, fn, src: bytes, ulo: int) -> bytes:
        sb = ctypes.create_string_buffer(src, len(src))
        return _es_bytes(fn(ctypes.cast(sb, ctypes.c_void_p).value, len(src), ulo))

    def seeded_elf(self, combined: bytes, ulo: int) -> bytes:
        return self._call(self.lib.selftest_native_elf_u, combined, ulo)

    def seeded_mic3(self, combined: bytes, ulo: int) -> bytes:
        return self._call(self.lib.selftest_nb_mic3, combined, ulo)


def note_of_elf(elf: bytes) -> bytes | None:
    """Validate the ELF + note structure (magic, MIND note name) and return the
    self-computed 32-byte PT_NOTE, or None if the image is structurally wrong."""
    if len(elf) < 64 or elf[:4] != b"\x7fELF":
        return None
    if len(elf) < 52 or elf[-40:-36] != b"MIND":
        return None
    return elf[-32:]


# ---------------------------------------------------------------------------
# Rust oracle — A9b test-time derivation of the seeded+pruned note via the
# committed tests/_ref_mic3_dump.rs, parameterized by MIND_REF_OUT_DIR so the
# fuzz program stands in as `main.mind` (no existing file edited, no new Rust
# code). One cargo invocation per derivation (~2 s warm).
# ---------------------------------------------------------------------------

_REF_NOTE_RE = re.compile(r"REF main:.*note=([0-9a-f]{64})")


class RustRef:
    def __init__(self, repo: pathlib.Path):
        self.repo = repo
        self.cmd = [
            "cargo", "test", "--features", _RUST_REF_FEATURES,
            "--test", _RUST_REF_TEST, "dump_ref", "--", "--ignored", "--nocapture",
        ]
        self.mindc = pathlib.Path(
            os.environ.get("MINDC_BIN", str(repo / "target" / "release" / "mindc"))
        )

    def seeded_note(self, prog_src: str, tmp: pathlib.Path) -> bytes | None:
        """Derive the Rust emit_mic3 reference note for prog_src, or None if
        Rust rejects the program (parse/lower failure panics dump_ref) or the
        toolchain misbehaves."""
        d = tmp / "ref"
        d.mkdir(exist_ok=True)
        (d / "main.mind").write_text(prog_src)
        env = dict(os.environ, MIND_REF_OUT_DIR=str(d), MIND_MIC3_BYTES_DIR=str(d))
        try:
            proc = subprocess.run(
                self.cmd, cwd=str(self.repo), capture_output=True, text=True,
                env=env, timeout=600,
            )
        except (subprocess.TimeoutExpired, OSError):
            return None
        if proc.returncode != 0:
            return None
        m = _REF_NOTE_RE.search(proc.stdout + proc.stderr)
        if not m:
            return None
        note = bytes.fromhex(m.group(1))
        # Cross-check the parsed line against the written fixture (loud failure
        # over a silent parse of the wrong artifact).
        f = d / "_ref_main.note"
        if f.exists():
            try:
                disk = bytes.fromhex(f.read_text().strip())
                if disk != note:
                    return None
            except ValueError:
                return None
        return note

    def unseeded_mic3(self, prog_src: str, tmp: pathlib.Path) -> bytes:
        """Diagnostic leg: Rust mic@3 of the program ALONE (mindc --emit-mic3).
        b'' when mindc is absent or rejects the program."""
        if not self.mindc.exists():
            return b""
        p = tmp / "u.mind"
        o = tmp / "u.mic3"
        p.write_text(prog_src)
        try:
            subprocess.run(
                [str(self.mindc), "--emit-mic3", str(o), str(p)],
                capture_output=True, timeout=120,
            )
        except (subprocess.TimeoutExpired, OSError):
            return b""
        return o.read_bytes() if o.exists() else b""


# ---------------------------------------------------------------------------
# Per-program verdict machinery
# ---------------------------------------------------------------------------

# status strings
MATCH = "match"
MISMATCH = "mismatch"
SKIP_SELFHOST = "skip-selfhost-closed"
SKIP_RUST = "skip-rust-rejected"
INVALID = "invalid"


def first_diverge(a: bytes, b: bytes) -> str:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            lo, hi = max(0, i - 8), min(n, i + 16)
            return (
                f"first diverge @ byte {i} (0x{i:x}): "
                f"selfhost={a[i]:#04x} rust={b[i]:#04x}\n"
                f"  selfhost[{lo}:{hi}]: {a[lo:hi].hex()}\n"
                f"  rust    [{lo}:{hi}]: {b[lo:hi].hex()}"
            )
    return f"length mismatch: selfhost={len(a)}B rust={len(b)}B (common prefix identical)"


class Harness:
    def __init__(self, sh: SelfHost, rust: RustRef, std: bytes, tmp: pathlib.Path,
                 verbose: bool = False):
        self.sh = sh
        self.rust = rust
        self.std = std
        self.tmp = tmp
        self.verbose = verbose

    def check_source(self, prog_src: str) -> tuple:
        """Run BOTH oracles on one source. Returns (status, sh_note, rust_note).
        status in {MATCH, MISMATCH, SKIP_SELFHOST, SKIP_RUST}."""
        combined = self.std + prog_src.encode()
        ulo = len(self.std)
        elf = self.sh.seeded_elf(combined, ulo)
        sh_note = note_of_elf(elf)
        if sh_note is None:
            return SKIP_SELFHOST, None, None  # len-0 (or malformed) = fail-closed
        rust_note = self.rust.seeded_note(prog_src, self.tmp)
        if rust_note is None:
            return SKIP_RUST, sh_note, None
        return (MATCH if sh_note == rust_note else MISMATCH), sh_note, rust_note

    def diagnose(self, prog_src: str, lines: list):
        """Append mismatch diagnostics: note hex + the UNSEEDED mic@3 byte diff
        (selftest_nb_mic3(prog,0) vs mindc --emit-mic3 prog) localizing the
        divergence to the emitter's handling of the program body itself."""
        lines.append("notes (32B PT_NOTE = sha256 of seeded+pruned mic@3):")
        combined = self.std + prog_src.encode()
        elf = self.sh.seeded_elf(combined, len(self.std))
        sh_note = note_of_elf(elf) or b""
        rust_note = self.rust.seeded_note(prog_src, self.tmp) or b""
        lines.append(f"  selfhost: {sh_note.hex()}")
        lines.append(f"  rust-ref: {rust_note.hex()}")
        if sh_note and rust_note:
            di = next((i for i in range(32) if sh_note[i] != rust_note[i]), -1)
            lines.append(f"  first differing note byte: index {di}")
        sh_mic = self.sh.seeded_mic3(combined, len(self.std))
        lines.append(f"selfhost seeded mic@3: {len(sh_mic)}B "
                     f"(sha256 {hashlib.sha256(sh_mic).hexdigest()[:32]}...)")
        sh_u = self.sh.seeded_mic3(prog_src.encode(), 0)
        ru_u = self.rust.unseeded_mic3(prog_src, self.tmp)
        if sh_u and ru_u:
            if sh_u == ru_u:
                lines.append(
                    "unseeded program-alone mic@3: BYTE-IDENTICAL "
                    f"({len(sh_u)}B) — divergence lives in the SEEDING/prune/"
                    "placeholder machinery (next_id / strtab / extern counts), "
                    "not in the fn-body emitter"
                )
            else:
                lines.append("unseeded program-alone mic@3 DIVERGES — the twin "
                             f"emitters disagree on the program body itself:\n  "
                             + first_diverge(sh_u, ru_u))
        elif not sh_u:
            lines.append("unseeded selftest_nb_mic3 fail-closed (0B) on the "
                         "program alone — body-shape gap in the nb mic@3 emitter")
        elif not ru_u:
            lines.append("mindc --emit-mic3 unavailable/rejected the unseeded "
                         "program — body diff unavailable")


# ---------------------------------------------------------------------------
# Delta-debounce minimizer — AST-level mutations; a reduction is kept only
# while BOTH oracles still accept AND still mismatch (fail-closed reductions
# are rejected, so the minimal reproducer stays inside the accepted intersection).
# ---------------------------------------------------------------------------


def _map_exprs(stmts, f):
    out = []
    for s in stmts:
        k = s[0]
        if k == "let":
            out.append(("let", s[1], s[2], f(s[3])))
        elif k == "assign":
            out.append(("assign", s[1], f(s[2])))
        elif k == "if":
            out.append(("if", s[1], _map_exprs(s[2], f),
                        _map_exprs(s[3], f) if s[3] else None))
        elif k == "while":
            out.append(("while", s[1], s[2], _map_exprs(s[3], f)))
        else:
            out.append(s)
    return out


def _mutate_candidates(fns: list):
    """Deterministic candidate-reduction stream (coarse -> fine)."""
    # 1) drop whole helpers (main may then not call them — calls into nowhere
    #    fail-closed both oracles, which the acceptance check rejects: safe)
    for i, fn in enumerate(fns):
        if fn[1] != "main":
            yield fns[:i] + fns[i + 1:]
    # 2) drop one statement at a time (reverse order, every fn)
    for fi, fn in enumerate(fns):
        _, name, params, stmts, ret = fn
        for si in range(len(stmts) - 1, -1, -1):
            new = fns[:fi] + [("fn", name, params, stmts[:si] + stmts[si + 1:], ret)] + fns[fi + 1:]
            yield new
    # 3) drop else branches
    def strip_else(stmts):
        out = []
        for s in stmts:
            if s[0] == "if" and s[3] is not None:
                out.append(("if", s[1], s[2], None))
            elif s[0] == "if":
                out.append(("if", s[1], strip_else(s[2]), None))
            elif s[0] == "while":
                out.append(("while", s[1], s[2], strip_else(s[3])))
            else:
                out.append(s)
        return out
    for fi, fn in enumerate(fns):
        _, name, params, stmts, ret = fn
        ns = strip_else(stmts)
        if ns != stmts:
            yield fns[:fi] + [("fn", name, params, ns, ret)] + fns[fi + 1:]
    # 4) replace every expr with const 0 (per fn)
    for fi, fn in enumerate(fns):
        _, name, params, stmts, ret = fn
        ns = _map_exprs(stmts, lambda e: ("const", 0))
        yield fns[:fi] + [("fn", name, params, ns, ("const", 0))] + fns[fi + 1:]


def minimize(harness: Harness, fns: list, max_trials: int = 120) -> tuple:
    """Delta-debounce: keep any reduction that still mismatches. Returns
    (minimal_fns, trials_used)."""
    current = fns
    trials = 0
    progress = True
    while progress and trials < max_trials:
        progress = False
        for cand in _mutate_candidates(current):
            if trials >= max_trials:
                break
            trials += 1
            src = emit_program(cand)
            status, _sh, _ru = harness.check_source(src)
            if status == MISMATCH:
                current = cand
                progress = True
                break  # restart the candidate stream on the smaller program
    return current, trials


# ---------------------------------------------------------------------------
# Batching — one cargo derivation per group of accepted programs.
# ---------------------------------------------------------------------------


def batch_source(members: list, ids: list) -> str:
    """Concatenate member programs with entries renamed fz<id>, then a synthetic
    `fn main` calling the first entry. Every fuzz fn starts at/after user_lo,
    so ALL of them are prune roots on BOTH sides (self-host nb_mark_roots; Rust
    ref_ir collects every user FnDef) — the batch note covers every member body,
    and one wrong byte in any member flips it."""
    parts = ["// mindfuzz_self_host BATCH - members: "
             + " ".join(str(i) for i in ids)]
    for fns, i in zip(members, ids):
        parts.append(emit_program(fns, entry_name=f"fz{i}").rstrip("\n"))
    parts.append(f"fn main() -> i64 {{\n    return fz{ids[0]}();\n}}")
    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Differential fuzzer: pure-MIND self-host mic@3 note vs the "
                    "Rust emit_mic3 reference note over generated programs.")
    ap.add_argument("--count", type=int, default=DEFAULT_COUNT,
                    help=f"number of programs to generate (default {DEFAULT_COUNT}; "
                         f"$MINDFUZZ_SELFHOST_ITERS overrides the default)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help=f"PRNG seed — runs are fully reproducible (default {DEFAULT_SEED})")
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH,
                    help=f"accepted programs per Rust derivation (default {DEFAULT_BATCH}; "
                         "1 = per-program, the strictest and slowest mode)")
    ap.add_argument("--verbose", action="store_true",
                    help="one status line per program")
    args = ap.parse_args()

    count = args.count
    if count == DEFAULT_COUNT and os.environ.get("MINDFUZZ_SELFHOST_ITERS"):
        try:
            count = max(1, int(os.environ["MINDFUZZ_SELFHOST_ITERS"]))
        except ValueError:
            pass
    if count < 1:
        print("FAIL  --count must be >= 1")
        return 1
    batch = max(1, args.batch)

    # --- oracle setup (fail LOUD / BLOCKED, never a fake pass) ---------------
    so = resolve_so()  # MINDC_SO verbatim; else FRESH mindc-built .so; else legacy
    if not so.exists():
        print(f"BLOCKED: self-host .so unavailable ({so}) — cannot build/load "
              f"libmindc_mind.so (set MINDC_SO or build target/release/mindc)")
        return 2
    try:
        sh = SelfHost(so)
    except OSError as e:
        print(f"BLOCKED: dlopen({so}) failed: {e}")
        return 2
    if shutil.which("cargo") is None:
        print("BLOCKED: cargo not on PATH — the Rust emit_mic3 reference note "
              "cannot be derived (no frozen-note fallback in this fuzzer)")
        return 2

    std = std_blob()
    gen = Gen(args.seed)

    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        rust = RustRef(_REPO)
        harness = Harness(sh, rust, std, tmp, args.verbose)

        # Mechanism sanity: the fixed `add` fixture must derive + match before
        # any fuzz verdict is trusted (catches a wholesale dump_ref breakage).
        sanity = ("fn add(a: i64, b: i64) -> i64 {\n    return a + b;\n}\n"
                  "fn main() -> i64 {\n    return add(2, 3);\n}\n")
        st, _shn, _run = harness.check_source(sanity)
        if st != MATCH:
            print(f"BLOCKED: sanity fixture `add` did not MATCH across the twin "
                  f"emitters (status={st}) — the oracle mechanism itself is broken; "
                  "refusing to fuzz on top of it")
            return 2

        print(f"[mindfuzz_self_host] seed={args.seed} count={count} batch={batch} "
              f"so={so.name} std_blob={len(std)}B")

        # --- PHASE 1: generate + self-host acceptance (cheap, in-process) ----
        programs = []          # (id, fns, src, sh_note)
        skipped_sh = 0
        for i in range(count):
            fns = gen.program()
            src = emit_program(fns)
            combined = std + src.encode()
            elf = sh.seeded_elf(combined, len(std))
            sh_note = note_of_elf(elf)
            if sh_note is None:
                skipped_sh += 1
                if args.verbose:
                    print(f"  [{i:4}] SKIP  self-host fail-closed (len-0 / malformed)")
                continue
            programs.append((i, fns, src, sh_note))
            if args.verbose:
                print(f"  [{i:4}] ACC   self-host emitted ({len(elf)}B ELF, "
                      f"note {sh_note.hex()[:12]}...)")

        accepted = len(programs)
        print(f"[phase 1] {accepted}/{count} accepted by the self-host emitter "
              f"({skipped_sh} fail-closed)")

        # --- PHASE 2: Rust-reference derivation (batched fast path + solo
        #     fallback ground truth) -----------------------------------------
        matched = 0
        skipped_rust = 0
        failing = []           # (id, fns, src, sh_note, rust_note)
        solo = batch == 1
        groups = [programs[g:g + batch] for g in range(0, len(programs), batch)] \
            if not solo else [[p] for p in programs]

        for gi, group in enumerate(groups):
            if len(group) == 1:
                (i, fns, src, sh_note) = group[0]
                rust_note = rust.seeded_note(src, tmp)
                if rust_note is None:
                    skipped_rust += 1
                    if args.verbose:
                        print(f"  [{i:4}] SKIP  rust rejected (dump_ref failed/panicked)")
                    continue
                if rust_note == sh_note:
                    matched += 1
                    if args.verbose:
                        print(f"  [{i:4}] MATCH note {sh_note.hex()[:12]}...")
                else:
                    failing.append((i, fns, src, sh_note, rust_note))
                    print(f"  [{i:4}] MISMATCH selfhost={sh_note.hex()[:12]}... "
                          f"rust={rust_note.hex()[:12]}...")
                continue

            ids = [p[0] for p in group]
            bsrc = batch_source([p[1] for p in group], ids)
            combined = std + bsrc.encode()
            belf = sh.seeded_elf(combined, len(std))
            bnote_sh = note_of_elf(belf)
            bnote_ru = rust.seeded_note(bsrc, tmp) if bnote_sh is not None else None
            if bnote_sh is not None and bnote_ru is not None and bnote_sh == bnote_ru:
                matched += len(group)
                if args.verbose:
                    print(f"  [batch {gi}] MATCH {len(group)} programs "
                          f"(note {bnote_sh.hex()[:12]}...)")
                continue

            # Fast path failed (batch closed / rust rejected / batch mismatch):
            # per-program ground truth.
            why = ("batch fail-closed" if bnote_sh is None else
                   "rust rejected batch" if bnote_ru is None else "batch note mismatch")
            if args.verbose:
                print(f"  [batch {gi}] {why} -> per-program fallback")
            for (i, fns, src, sh_note) in group:
                rust_note = rust.seeded_note(src, tmp)
                if rust_note is None:
                    skipped_rust += 1
                    if args.verbose:
                        print(f"  [{i:4}] SKIP  rust rejected")
                    continue
                if rust_note == sh_note:
                    matched += 1
                    if args.verbose:
                        print(f"  [{i:4}] MATCH (fallback) note {sh_note.hex()[:12]}...")
                else:
                    failing.append((i, fns, src, sh_note, rust_note))
                    print(f"  [{i:4}] MISMATCH selfhost={sh_note.hex()[:12]}... "
                          f"rust={rust_note.hex()[:12]}...")

        skipped = skipped_sh + skipped_rust

        # --- verdict ----------------------------------------------------------
        if failing:
            (i, fns, src, sh_note, rust_note) = failing[0]
            print(f"\nFAIL  program [{i}] diverged — minimizing (delta-debounce)...")
            minimal, trials = minimize(harness, fns)
            msrc = emit_program(minimal)
            mstat, msh, mru = harness.check_source(msrc)
            print(f"\n===== MINIMAL REPRODUCER (program [{i}], {trials} minimize trials, "
                  f"{count_stmts(minimal)} stmts, status={mstat}) =====")
            print(msrc)
            print("===== DIAGNOSTICS =====")
            diag = []
            harness.diagnose(msrc, diag)
            print("\n".join(diag))
            staged = _HERE / "mindfuzz_self_host_staged"
            staged.mkdir(exist_ok=True)
            (staged / f"prog{i:04}.mind").write_text(src)
            (staged / f"prog{i:04}.minimal.mind").write_text(msrc)
            (staged / f"prog{i:04}.notes.txt").write_text(
                f"seed={args.seed} program=[{i}]\n"
                f"selfhost_note={msh.hex() if msh else sh_note.hex()}\n"
                f"rust_ref_note={mru.hex() if mru else rust_note.hex()}\n"
                + "\n".join(diag) + "\n")
            print(f"\nFAIL  {len(failing)}/{matched + len(failing)} compared programs "
                  f"diverged (staged under {staged.name}/prog{i:04}.*) — REAL twin-emitter "
                  "mic@3 divergence over the generated space")
            return 1

        if matched == 0:
            print(f"\nFAIL  0 of {count} generated programs were accepted by BOTH "
                  f"oracles ({skipped_sh} self-host fail-closed, {skipped_rust} "
                  "rust-rejected) — the generator produced nothing comparable; "
                  "refusing a silent green")
            return 1

        print(f"\nPASS  mindfuzz_self_host: {count} generated, {matched} "
              f"accepted-and-matched (mic@3 notes byte-identical across the twin "
              f"emitters), {skipped} skipped ({skipped_sh} self-host fail-closed / "
              f"{skipped_rust} rust-rejected)  [seed={args.seed}]")
        return 0


if __name__ == "__main__":
    sys.exit(main())
