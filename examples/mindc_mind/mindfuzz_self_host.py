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

RUNTIME leg (a matching note can never hide a wrong value)
---------------------------------------------------------
The PT_NOTE is the self-host's mic@3 emitter; the ELF CODE next to it comes from
a DIFFERENT pure-MIND emitter (the nb_* native-ELF backend). A note can agree
with Rust while the machine code computes something else (measured: a loop-body
`let mut x` shadowing an outer `x` exited 3 where Rust/MLIR exits 5, note
identical). So every COMPARED program (and every regression fixture the
self-host emits) is also EXECUTED both ways: the self-host ELF, and the
`mindc build --backend mlir --emit=binary` binary of the same source. Exit codes
must be equal; a mismatch, a crash or timeout on either side, or an MLIR build
failure on a program both emitters accepted is a FAIL.

Shared-gap blindness (documented limit, not a gap in this gate)
---------------------------------------------------------------
Both legs are measured AGAINST the Rust lowering (src/eval/lower.rs): the note
leg compares to Rust's emit_mic3, and the runtime leg's reference binary is
built from that same Rust IR via MLIR. A defect that lives in the Rust lowering
itself and that the self-host faithfully mirrors produces equal notes AND equal
exit codes, so this fuzzer cannot see it. That is by design — the oracle
comparison IS the gate: "correct" here means "agrees with the bootstrap
reference". Rust-side semantic bugs are the job of the Rust-side suites
(tree-eval vs MLIR cross-checks, tests/mindfuzz_cross_substrate.rs).

Semantics of disagreement / refusal
-----------------------------------
  * the SELF-HOST emitter fail-closes a program (EMPTY ELF)  -> refused
    (logged + counted), NOT a failure: a refusal is the emitter's documented
    fail-closed contract (never a wrong note). The count is printed. Only an
    EMPTY result is a refusal: a NON-EMPTY image that is not a well-formed
    MIND-noted ELF is a malformed artifact and FAILS the run.
  * the self-host EMITS a note but the Rust oracle rejects the program
    (dump_ref parse/lower failure)  -> FAIL exit 1. The self-host accepted a
    program the reference rejects — that is a divergence too, and a gate that
    let it through as a "skip" could measure nothing (it did, for a month,
    while the dump_ref oracle was broken).
  * both accept, notes differ -> MINIMIZE (delta-debounce over the AST), print
    the minimal reproducer + note hex + first-diverging mic@3 bytes, stage it
    under mindfuzz_self_host_staged/, exit 1.
  * both accept, notes agree  -> counted toward K.
  * K == 0 (nothing both accept) -> FAIL exit 1: a fuzzer that compared
    nothing proves nothing — never a silent green.
  * .so unbuildable / cargo absent / the `add` sanity fixture does not match
    -> FAIL exit 1, labelled BLOCKED (the oracle mechanism is unavailable; a
    gate that cannot compare anything is a failed gate, never a skipped one).

Gate use (CI): the seed and count are pinned on the command line
(`--seed 1234 --count 200`); an explicit `--count` always wins over the
$MINDFUZZ_SELFHOST_ITERS default override, so the gate is deterministic.
Every compared batch / program prints one PASS or MISMATCH verdict line.

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
import os
import pathlib
import random
import shutil
import sys
import tempfile

_HERE = pathlib.Path(__file__).parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_HERE))
from _mindfuzz_twin_emit import (  # noqa: E402
    MATCH,
    MISMATCH,
    SKIP_SELFHOST,
    Harness,
    MalformedElf,
    RuntimeCheck,
    RustRef,
    SelfHost,
    batch_source,
    emit_program,
    minimize,
    note_of_elf,
)
from _selfhost_so import resolve_so  # noqa: E402

# Same 21 modules, SAME order as the smoke's _seeded_buffer() / Rust KEEP list
# (std.http + std.sha512 excluded exactly as the native-ELF oracle bundle does).
_STDLIB_MODULES = [
    "arena", "async", "blas", "cli", "fs", "io", "io_canon", "iouring",
    "json", "map", "net", "process", "reactor", "regex", "ring", "sha256",
    "string", "time", "toml", "tui", "vec",
]

# Committed regression corpus: shapes that once diverged between the twin
# emitters. Each must MATCH on EVERY run, independent of --seed/--count, so a
# seed advance can never silently drop a fixed divergence class.
_REGRESS_DIR = _HERE / "testdata" / "mindfuzz_regress"
# Reproducers of divergence classes the self-host currently answers by REFUSING
# (fail closed). Each must be MATCH or a self-host refusal — never a different
# note, never a self-host accept the Rust oracle rejects — and runtime-equal
# whenever emitted. Move a fixture to _REGRESS_DIR once the emitter ports it.
_FAILCLOSED_DIR = _HERE / "testdata" / "mindfuzz_regress_failclosed"

DEFAULT_COUNT = 200
DEFAULT_SEED = 1234
DEFAULT_BATCH = 8
# Floor on programs actually COMPARED (accepted by both emitters) for the pinned
# (DEFAULT_SEED, DEFAULT_COUNT) corpus. A self-host that fail-closes everything is
# "0 divergences" too; this floor makes that a FAIL instead of a green. Set just
# under the measured count (refusals are legitimate, a collapse is not); raise it
# when the emitter accepts more, never lower it to absorb a regression.
PINNED_MIN_COMPARED = 80
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


def params_of(sc: "Scope") -> list:
    """The function's parameter names (the outermost frame's immutable entries)."""
    return [n for n, m in sc.frames[0] if not m] if sc.frames else []


class Gen:
    """Seeded generator over the supported subset. Every draw comes from
    self.r — random.Random(seed) — so a (seed, count) pair is fully
    reproducible (no wall-clock, no OS entropy)."""

    def __init__(self, seed: int):
        self.r = random.Random(seed)
        # SECOND stream for the round-3 probe shapes (a while inside an if-branch,
        # a body-local shadow-then-assign in a loop). Drawing those decisions and
        # their contents from r2 leaves the main stream — and so every program
        # without a probe — exactly as before, keeping the pinned corpus's compared
        # floor meaningful while the probes still land at a fixed rate.
        self.r2 = random.Random(seed * 7919 + 17)
        # THIRD stream for the straight-line register-plan probe helper (see
        # gen_dtk_fn): same isolation contract as r2.
        self.r3 = random.Random(seed * 104729 + 31)
        self._ops = None  # restricted (ops, weights) while generating a DTK probe
        self.helper_seq = 0
        self.var_seq = 0
        self.ctr_seq = 0

    def _on_r2(self, f):
        """Run a generator step on the probe stream r2 (main stream untouched)."""
        saved = self.r
        self.r = self.r2
        try:
            return f()
        finally:
            self.r = saved

    # -- expressions ---------------------------------------------------------

    def expr(self, sc: Scope, depth: int):
        r = self.r
        if depth <= 0 or r.randrange(100) < 45:
            return self.leaf(sc)
        ops, wts = self._ops if self._ops else (_ARITH, _ARITH_W)
        op = r.choices(ops, weights=wts, k=1)[0]
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
                if style == "mixed" and budget[0] >= 6 and self.r2.randrange(100) < 8:
                    # A WHILE inside an if-branch (statement-level, any style):
                    # the loop writes outer vars first seen inside it, reads
                    # earlier loops' exit ids, and nests under else-branches —
                    # the classes behind the e7/h1/v1/wit/r20 reproducers.
                    budget[0] += 1  # gen_while charges its own 2
                    stmts.extend(self._on_r2(lambda: self.gen_while(sc, budget, loop_depth + 1)))
                    continue
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

        # BODY-LOCAL SHADOW then assign (r2 probe): `let mut p = E; p = E2;` over
        # an outer name — preferring a PARAMETER (the pcand / s3_412 class: the
        # assign is a candidate for the loop's alias-break yet writes the
        # per-iteration shadow, never the outer binding; the native ELF must leave
        # the outer binding untouched).
        if sc.shadowable() and budget[0] >= 3 and self.r2.randrange(100) < 10:
            budget[0] -= 2
            r2 = self.r2
            outer = [n for n in sc.shadowable() if n in params_of(sc)]
            name = r2.choice(outer if (outer and r2.randrange(100) < 70)
                             else sc.shadowable())
            init = self._on_r2(lambda: self.expr(sc, 2))
            sc.push(name, True)
            body.append(("let", True, name, init))
            body.append(("assign", name, self._on_r2(lambda: self.expr(sc, 2))))

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

    def gen_dtk_fn(self, name: str, params: list):
        """Straight-line helper (lets / assigns / tail return, register-plan-safe
        ops only) — the ONLY shape the native emitter's DTK callee-saved register
        plan accepts, so it is what exercises rbx/r12-homed values. Assign RHS is
        a bare ident half the time (`w = y`): that shape mints a copy slot the
        plan must not reg-home (round-2 blocker: helper `w = y; return w` exited
        0 vs Rust 2 with byte-identical notes). Drawn wholly from r3.

        THREE modes (r3-drawn), because ONE bare-ident assign anywhere makes the
        whole helper plan-ineligible — round 2 always appended `tgt = src`, so
        every probe silently fell back to the stack scheme and the register plan
        was never exercised (round-3 review found the plan bug this way):
          0 (40%) no bare-ident RHS at all: the plan is ELIGIBLE, values are
            register-homed (the only mode that exercises rbx/r12 homes).
          1 (30%) bare-ident LETs of reassigned sources, no bare-ident assign:
            `let v = m; m = <expr>; .. v` (round-3 blocker: helper `let q = p;
            let e = q; q = 5; return e` exited 0, Rust 3 — the LET copy slot was
            read from a register-homed source's stale stack home).
          2 (30%) bare-ident assigns `w = y` (round-2 blocker shape)."""
        def body():
            mode = self.r.choices([0, 1, 2], weights=[40, 30, 30], k=1)[0]

            def nv(e):  # never a bare ident (modes 0/1): `y` -> `(y + 0)`
                return ("bin", "+", e, ("const", 0)) if e[0] == "var" else e

            sc = Scope()
            for p in params:
                sc.push(p, False)
            saved = self._ops
            self._ops = (["+", "-", "*", "|"], [30, 20, 15, 5])
            try:
                # a guaranteed mut local, so an assign always has a target
                acc = self.fresh_var()
                stmts = [("let", True, acc, nv(self.expr(sc, 2)))]
                sc.push(acc, True)
                for _ in range(1 + self.r.randrange(4)):
                    muts = sc.mut_names()  # a non-mut shadow can hide every mut
                    if muts and self.r.randrange(100) < 55:
                        tgt = self.r.choice(muts)
                        if mode == 2 and self.r.randrange(100) < 60:
                            stmts.append(("assign", tgt, ("var", self.r.choice(sc.names()))))
                        else:
                            stmts.append(("assign", tgt, nv(self.expr(sc, 2))))
                    elif mode == 1 and muts and self.r.randrange(100) < 50:
                        v = self.fresh_var()
                        stmts.append(("let", self.r.randrange(2) == 0, v, ("var", self.r.choice(muts))))
                        sc.push(v, stmts[-1][1])
                    else:
                        v = self.fresh_var()
                        mut = self.r.randrange(100) < 60
                        stmts.append(("let", mut, v, nv(self.expr(sc, 2))))
                        sc.push(v, mut)
                if not sc.mut_names():
                    acc = self.fresh_var()
                    stmts.append(("let", True, acc, nv(self.expr(sc, 2))))
                    sc.push(acc, True)
                tgt = self.r.choice(sc.mut_names())
                if mode == 1:
                    # the full trigger: copy a mut source, then overwrite it
                    v = self.fresh_var()
                    stmts.append(("let", self.r.randrange(2) == 0, v, ("var", tgt)))
                    sc.push(v, stmts[-1][1])
                    stmts.append(("assign", tgt, nv(self.expr(sc, 2))))
                    ret = ("var", v) if self.r.randrange(100) < 60 else ("bin", "+", ("var", v), ("var", tgt))
                elif mode == 2:
                    stmts.append(("assign", tgt, ("var", self.r.choice(sc.names()))))
                    ret = ("var", tgt) if self.r.randrange(100) < 60 else self.expr(sc, 2)
                else:
                    stmts.append(("assign", tgt, nv(self.expr(sc, 2))))
                    ret = ("var", tgt) if self.r.randrange(100) < 60 else self.expr(sc, 2)
            finally:
                self._ops = saved
            return ("fn", name, params, stmts, ret)
        saved = self.r
        self.r = self.r3
        try:
            return body()
        finally:
            self.r = saved

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
            # 1-5 params. Wide helpers matter: a param is vid 0..n-1 inside the
            # body, so >=3 params is what reaches id-collision classes between
            # param vids and body-local / loop alias-break copy vids (the fixed
            # n_4_3 / t3_ctrl_noread / q12_chain_collide3 regressions).
            roll = r.randrange(100)
            n_par = 1 if roll < 40 else 2 if roll < 60 else 3 + r.randrange(3)
            params = [f"{pfx}{self.helper_seq}" for pfx in "xyzwu"[:n_par]]
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
        # Register-plan probe (r3 stream; main stream untouched): 30% of programs
        # whose return is not already a call return a straight-line helper.
        if ret[0] != "call" and self.r3.randrange(100) < 30:
            self.helper_seq += 1
            n_par = 1 + self.r3.randrange(3)
            params = [f"{pfx}{self.helper_seq}" for pfx in "xyz"[:n_par]]
            probe = self.gen_dtk_fn(f"fzd{self.helper_seq}", params)
            fns.append(probe)
            ret = ("call", probe[1], [("const", self.r3.randrange(9)) for _ in params])
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
# Driver
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Differential fuzzer: pure-MIND self-host mic@3 note vs the "
                    "Rust emit_mic3 reference note over generated programs.")
    ap.add_argument("--count", type=int, default=None,
                    help=f"number of programs to generate (default {DEFAULT_COUNT}; "
                         f"$MINDFUZZ_SELFHOST_ITERS overrides the DEFAULT only — an "
                         f"explicit --count always wins)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help=f"PRNG seed — runs are fully reproducible (default {DEFAULT_SEED})")
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH,
                    help=f"accepted programs per Rust derivation (default {DEFAULT_BATCH}; "
                         "1 = per-program, the strictest and slowest mode)")
    ap.add_argument("--verbose", action="store_true",
                    help="one status line per program")
    ap.add_argument("--min-compared", type=int, default=None,
                    help=f"FAIL unless at least this many generated programs were "
                         f"compared (default {PINNED_MIN_COMPARED} for the pinned "
                         f"seed={DEFAULT_SEED}/count={DEFAULT_COUNT} corpus, else 1)")
    args = ap.parse_args()

    count = args.count
    if count is None:
        count = DEFAULT_COUNT
        if os.environ.get("MINDFUZZ_SELFHOST_ITERS"):
            try:
                count = max(1, int(os.environ["MINDFUZZ_SELFHOST_ITERS"]))
            except ValueError:
                print("FAIL  $MINDFUZZ_SELFHOST_ITERS is not an integer")
                return 1
    if count < 1:
        print("FAIL  --count must be >= 1")
        return 1
    batch = max(1, args.batch)
    min_compared = args.min_compared
    if min_compared is None:
        pinned = args.seed == DEFAULT_SEED and count == DEFAULT_COUNT
        min_compared = PINNED_MIN_COMPARED if pinned else 1
    min_compared = max(1, min_compared)

    # --- oracle setup (fail LOUD / BLOCKED, never a fake pass) ---------------
    so = resolve_so()  # MINDC_SO verbatim; else FRESH mindc-built .so; else legacy
    if not so.exists():
        print(f"FAIL  BLOCKED: self-host .so unavailable ({so}) — cannot build/load "
              f"libmindc_mind.so (set MINDC_SO or build target/release/mindc)")
        return 1
    try:
        sh = SelfHost(so)
    except OSError as e:
        print(f"FAIL  BLOCKED: dlopen({so}) failed: {e}")
        return 1
    if shutil.which("cargo") is None:
        print("FAIL  BLOCKED: cargo not on PATH — the Rust emit_mic3 reference note "
              "cannot be derived (no frozen-note fallback in this fuzzer)")
        return 1

    std = std_blob()
    gen = Gen(args.seed)

    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        rust = RustRef(_REPO)
        harness = Harness(sh, rust, std, tmp, args.verbose)
        if not rust.mindc.exists():
            print(f"FAIL  BLOCKED: {rust.mindc} absent — the runtime leg (Rust/MLIR "
                  "binary exit code) cannot run; a note-only comparison can hide a "
                  "wrong value")
            return 1
        rt = RuntimeCheck(rust.mindc, tmp)

        # Mechanism sanity: the fixed `add` fixture must derive + match before
        # any fuzz verdict is trusted (catches a wholesale dump_ref breakage).
        sanity = ("fn add(a: i64, b: i64) -> i64 {\n    return a + b;\n}\n"
                  "fn main() -> i64 {\n    return add(2, 3);\n}\n")
        st, _shn, _run = harness.check_source(sanity)
        if st != MATCH:
            print(f"FAIL  BLOCKED: sanity fixture `add` did not MATCH across the twin "
                  f"emitters (status={st}) — the oracle mechanism itself is broken; "
                  "refusing to fuzz on top of it")
            return 1

        print(f"[mindfuzz_self_host] seed={args.seed} count={count} batch={batch} "
              f"so={so.name} std_blob={len(std)}B")

        # --- PHASE 0b (below): the fail-closed corpus — reproducers the
        #     self-host may REFUSE (fail closed) but must never emit a
        #     different note for; runtime-checked whenever emitted.
        # --- PHASE 0: the committed regression corpus (every fixture must be
        #     accepted by BOTH emitters with byte-identical notes; a refusal or a
        #     Rust rejection is a failure here, not a skip — these programs are
        #     known-good shapes, so anything but MATCH is a regression). --------
        fixtures = sorted(_REGRESS_DIR.glob("*.mind"))
        if not fixtures:
            print(f"FAIL  BLOCKED: no regression fixtures under {_REGRESS_DIR} — "
                  "the corpus gate would compare nothing")
            return 1
        for fx in fixtures:
            st, shn, rn = harness.check_source(fx.read_text())
            if st != MATCH:
                lines = [f"FAIL  [regress] {fx.name}: status={st} "
                         f"selfhost={shn.hex() if shn else '-'} "
                         f"rust={rn.hex() if rn else '-'}"]
                if st == MISMATCH:
                    harness.diagnose(fx.read_text(), lines)
                print("\n".join(lines))
                return 1
            why = rt.check(fx.read_text(), harness.seeded_elf(fx.read_text()))
            if why:
                print(f"FAIL  [regress] {fx.name}: {why}")
                return 1
            print(f"  PASS  [regress] {fx.name} note {shn.hex()[:12]}... identical, "
                  "runtime exit equal")

        fc_fixtures = sorted(_FAILCLOSED_DIR.glob("*.mind"))
        if not fc_fixtures:
            print(f"FAIL  BLOCKED: no fail-closed fixtures under {_FAILCLOSED_DIR}")
            return 1
        for fx in fc_fixtures:
            st, shn, rn = harness.check_source(fx.read_text())
            if st == SKIP_SELFHOST:
                print(f"  PASS  [failclosed] {fx.name} refused (fail-closed)")
                continue
            if st != MATCH:
                lines = [f"FAIL  [failclosed] {fx.name}: status={st} "
                         f"selfhost={shn.hex() if shn else '-'} "
                         f"rust={rn.hex() if rn else '-'}"]
                if st == MISMATCH:
                    harness.diagnose(fx.read_text(), lines)
                print("\n".join(lines))
                return 1
            why = rt.check(fx.read_text(), harness.seeded_elf(fx.read_text()))
            if why:
                print(f"FAIL  [failclosed] {fx.name}: {why}")
                return 1
            print(f"  PASS  [failclosed] {fx.name} note {shn.hex()[:12]}... identical, "
                  "runtime exit equal")

        # --- PHASE 1: generate + self-host acceptance (cheap, in-process) ----
        programs = []          # (id, fns, src, sh_note)
        elfs = {}              # id -> the exact self-host image whose note was compared
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
                    print(f"  [{i:4}] refused: self-host fail-closed (len-0 / malformed)")
                continue
            programs.append((i, fns, src, sh_note))
            elfs[i] = elf
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
        rt_fail = []           # (id, src, why) — runtime-leg failures
        solo = batch == 1
        groups = [programs[g:g + batch] for g in range(0, len(programs), batch)] \
            if not solo else [[p] for p in programs]

        for gi, group in enumerate(groups):
            if len(group) == 1:
                (i, fns, src, sh_note) = group[0]
                rust_note = rust.seeded_note(src, tmp)
                if rust_note is None:
                    skipped_rust += 1
                    print(f"  [{i:4}] REJECTED-BY-RUST (dump_ref failed/panicked) "
                          f"after the self-host emitted a note")
                    continue
                if rust_note == sh_note:
                    matched += 1
                    print(f"  PASS  [{i:4}] note {sh_note.hex()[:12]}... identical")
                    why = rt.check(src, elfs[i])
                    if why:
                        rt_fail.append((i, src, why))
                        print(f"  [{i:4}] {why}")
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
                print(f"  PASS  [batch {gi}] {len(group)} programs "
                      f"(ids {ids[0]}..{ids[-1]}) note {bnote_sh.hex()[:12]}... identical")
                for (i, _f, src, _n) in group:
                    why = rt.check(src, elfs[i])
                    if why:
                        rt_fail.append((i, src, why))
                        print(f"  [{i:4}] {why}")
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
                    print(f"  [{i:4}] REJECTED-BY-RUST (dump_ref failed/panicked) "
                          f"after the self-host emitted a note")
                    continue
                if rust_note == sh_note:
                    matched += 1
                    print(f"  PASS  [{i:4}] (fallback) note {sh_note.hex()[:12]}... identical")
                    why = rt.check(src, elfs[i])
                    if why:
                        rt_fail.append((i, src, why))
                        print(f"  [{i:4}] {why}")
                else:
                    failing.append((i, fns, src, sh_note, rust_note))
                    print(f"  [{i:4}] MISMATCH selfhost={sh_note.hex()[:12]}... "
                          f"rust={rust_note.hex()[:12]}...")


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

        if rt_fail:
            (i, src, why) = rt_fail[0]
            staged = _HERE / "mindfuzz_self_host_staged"
            staged.mkdir(exist_ok=True)
            (staged / f"prog{i:04}.runtime.mind").write_text(src)
            print(f"\n===== RUNTIME DIVERGENCE (program [{i}]) =====\n{src}\n{why}")
            print(f"\nFAIL  {len(rt_fail)}/{matched} compared programs ran to a different "
                  f"exit code than the Rust/MLIR binary although the notes matched "
                  f"(staged {staged.name}/prog{i:04}.runtime.mind)  [seed={args.seed}]")
            return 1

        if skipped_rust:
            print(f"\nFAIL  {skipped_rust} program(s) the self-host EMITTED a note for were "
                  f"rejected by the Rust dump_ref oracle — an accept/reject divergence "
                  f"(or a broken oracle); never counted as a skip  [seed={args.seed}]")
            return 1

        if matched < min_compared:
            print(f"\nFAIL  only {matched} of {count} generated programs were compared "
                  f"(floor {min_compared}; {skipped_sh} self-host fail-closed, "
                  f"{skipped_rust} rust-rejected) — a collapse in what the twin "
                  f"emitters both accept is a regression, not a pass  [seed={args.seed}]")
            return 1

        print(f"\nmindfuzz_self_host: 0 divergences — {count} generated, {matched} "
              f"compared with mic@3 notes byte-identical across the twin emitters "
              f"and {rt.runs} runtime exit codes equal (self-host ELF vs Rust/MLIR), "
              f"{skipped_sh} refused by the self-host (fail-closed), 0 rust-rejected, "
              f"compared floor {min_compared}  [seed={args.seed} count={count}]")
        return 0


def main_guarded() -> int:
    try:
        return main()
    except MalformedElf as e:
        print(f"FAIL  malformed self-host artifact (not a refusal): {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main_guarded())
